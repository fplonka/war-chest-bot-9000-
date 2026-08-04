"""ReBeL training for War Chest.

Two phases inside one wall-clock budget:

1. **Warm start** (`--warm-frac` of the budget). Both players are a stochastic
   one-ply greedy bot on a public-information evaluation; value targets blend
   that evaluation (squashed into (-1, 1)) with the realised game outcome.
   ReBeL never plays a policy directly — every move comes out of CFR using the
   value network at the leaves — so the value network is the natural place to
   inject a starting behaviour. Without it CFR searches on noise and no game
   ever ends inside the horizon. The network at the end of this phase is the
   *initial checkpoint*.

2. **ReBeL** (the rest). Self-play where every decision solves a depth-limited
   CFR subgame over public belief states; the targets are the CFR root values,
   one per config in each player's belief support.

Everything except the gradient step runs in Rust across all cores; Python ships
weights down and pulls tensors back once per epoch.

A training row is a public state plus, for each player, the whole belief: the
exact configs in support, their probabilities, and the value the solve gave
each. The config lists are ragged, so they live in a flat arena and a batch is
assembled by gathering spans -- see `Buffer`.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest
import mirror

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
# Weight slots in the Rust workers: 0 live, 1 initial checkpoint, 2 champion.
CHAMP_SLOT = 2


class Mlp(nn.Module):
    """The value network: `v(PBS, config) -> scalar`.

    Two towers. The config tower embeds one player's exact private state; the
    PBS tower embeds the public state and, through a belief-weighted sum of the
    *same* config embeddings, the belief. The value is their inner product.

        z(c) = relu(phi(c) Wc + bc)                 config embedding   [dg]
        g(c) = z(c) Wg + bg                         readout embedding  [r + 1]
        e_p  = sum_c beta_p(c) z(c)                 belief             [dg]
        h    = relu(LN(relu(LN(x W0 + b0)) W1 + b1 + [e_0; e_1] Wb))
        u    = h Wu + bu                            PBS readout        [r]
        v(c) = <u, g(c)[:r]> + g(c)[r]

    This is `csrc/liars_dice`'s shape with its two fixed-width private-state
    dimensions replaced by learned functions of the private state, because War
    Chest's private states do not fit in a fixed-width table. Set `g` to a
    one-hot lookup and the two are the same network.

    `rank` is the one dimension that has to be chosen rather than inherited: the
    reference gets `rank = hidden` for free because its readout is a lookup,
    while here every config costs a `rank`-long dot product. A config is
    sixteen numbers, so 64 is not a binding constraint on what the value can
    depend on, and it is 6x less per-config work than the hidden width.

    LayerNorm on every hidden layer, as the reference does (`use_layer_norm:
    true`): the raw features include unbounded-ish coin counts and the
    bootstrapped targets shift scale over training, so normalising between the
    affine and the activation is what keeps the hidden distribution stable as
    the target distribution moves.
    """

    def __init__(self, hidden, dg=64, rank=64):
        super().__init__()
        self.dims = [PUBFEAT, hidden, CFEAT, dg, rank]
        self.w0 = nn.Linear(PUBFEAT, hidden)
        self.w1 = nn.Linear(hidden, hidden)
        # The belief's connection into the hidden layer. No bias: it is added to
        # a layer that already has one.
        self.wb = nn.Linear(2 * dg, hidden, bias=False)
        self.wc = nn.Linear(CFEAT, dg)
        self.wg = nn.Linear(dg, rank + 1)
        self.wu = nn.Linear(hidden, rank)
        self.ln0 = nn.LayerNorm(hidden)
        self.ln1 = nn.LayerNorm(hidden)
        # Start near zero so the first bootstrapped targets are not dominated by
        # random leaf values.
        nn.init.zeros_(self.wg.bias)
        nn.init.normal_(self.wg.weight, std=1e-3)

    def forward(self, xpub, phi, inv, w, seg, nseg):
        """Values for every config in a ragged batch.

        `xpub` is `[B, PUBFEAT]`. The configs of every row and player are
        concatenated into one list of length `N`; `w[i]` is config `i`'s belief
        probability and `seg[i] = 2 * row + player` says where it belongs.

        The config tower runs over *distinct* configs only: `phi` is `[U, CFEAT]`
        and `inv` maps each of the `N` entries to its row in it. A batch of 1024
        positions carries ~50k configs drawn from a couple of thousand distinct
        ones, and the readout embedding is `dg x (hidden + 1)` — by far the
        widest matmul here if it runs per entry. The Rust solver deduplicates
        for the same reason.
        """
        z = F.relu(self.wc(phi))
        g = self.wg(z)
        # The belief: a weighted sum of config embeddings, per (row, player).
        e = torch.zeros(nseg, z.shape[1], dtype=z.dtype, device=z.device)
        e.index_add_(0, seg, z[inv] * w.unsqueeze(1))
        h = F.relu(self.ln0(self.w0(xpub)))
        h = F.relu(self.ln1(self.w1(h) + self.wb(e.reshape(xpub.shape[0], -1))))
        u = self.wu(h)
        rk = u.shape[1]
        gc = g[inv]
        return (u[seg // 2] * gc[:, :rk]).sum(-1) + gc[:, rk]

    def push(self, slot):
        """Ship weights to the Rust workers (row-major `[in, out]` per matrix).

        The order here is `Mlp::from_flat`'s and nothing else knows it, so the
        two cannot drift apart without `test_parity.py` failing.
        """
        w = np.concatenate([l.weight.detach().cpu().t().contiguous().numpy().ravel()
                            for l in (self.w0, self.w1, self.wb, self.wc, self.wg, self.wu)])
        b = np.concatenate([l.bias.detach().cpu().numpy().ravel()
                            for l in (self.w0, self.w1, self.wc, self.wg, self.wu)])
        ln = np.concatenate([t.detach().cpu().numpy().ravel()
                             for n in (self.ln0, self.ln1) for t in (n.weight, n.bias)])
        warchest.set_weights(self.dims, np.ascontiguousarray(w, np.float32),
                             np.ascontiguousarray(b, np.float32),
                             np.ascontiguousarray(ln, np.float32), slot)


class Buffer:
    """Fixed-capacity FIFO over rows whose config lists are ragged.

    Two rings advance together: one over rows, one over the config arena the
    rows point into. A row's configs sit at an *absolute* arena offset, so a row
    is live exactly while both rings still hold it -- and because both are
    written in order, the rows the arena has evicted are always the oldest ones,
    which is a single monotone pointer rather than a validity test per row.

    Bootstrapped targets are averaged over whatever history the buffer holds, so
    its length is a real algorithmic knob and not just a memory setting -- the
    reference implementation runs a 2M buffer. Counts are stored as the `uint8`
    they are and everything else as float16, which is what makes that
    affordable: a row costs `PUBFEAT * 2` bytes plus 20 per config.

    Preallocated and written with wraparound rather than grown by
    concatenation. The concatenate form rebuilt the whole buffer every epoch:
    at an 800k cap that copies ~2.6 GB per epoch and transiently holds two
    copies of it, which is most of a 16 GB machine. `np.zeros` maps zero pages
    lazily, so reserving the full capacity up front costs nothing until it is
    actually filled.
    """

    def __init__(self, cap, ccap):
        self.cap, self.ccap = cap, ccap
        self.x = np.zeros((cap, PUBFEAT), np.float16)
        self.cstart = np.zeros(cap, np.int64)   # absolute arena offset
        self.clen = np.zeros((cap, 2), np.int32)
        self.cc = np.zeros((ccap, CCOUNTS), np.uint8)
        self.cp = np.zeros(ccap, np.uint8)
        self.cw = np.zeros(ccap, np.float16)
        self.cy = np.zeros(ccap, np.float16)
        self.rows = 0   # rows ever written
        self.cfgs = 0   # configs ever written
        self.lo = 0     # oldest row whose configs are still in the arena

    def add(self, x, cc, cw, cy, coff):
        n = len(x)
        lens = np.diff(coff).reshape(n, 2)
        cp = np.repeat(np.tile([0, 1], n).astype(np.uint8), lens.ravel())
        starts = self.cfgs + coff[:-1:2]
        for i in range(0, n, 4096):
            j = min(i + 4096, n)
            sl = np.arange(i, j) + self.rows
            self.x[sl % self.cap] = x[i:j]
            self.cstart[sl % self.cap] = starts[i:j]
            self.clen[sl % self.cap] = lens[i:j]
        m = len(cw)
        sl = (np.arange(m) + self.cfgs) % self.ccap
        self.cc[sl], self.cp[sl], self.cw[sl], self.cy[sl] = cc, cp, cw, cy
        self.rows += n
        self.cfgs += m
        # Advance past every row the arena no longer holds in full.
        floor = self.cfgs - self.ccap
        self.lo = max(self.lo, self.rows - self.cap)
        while self.lo < self.rows and self.cstart[self.lo % self.cap] < floor:
            self.lo += 1

    def clear(self):
        self.lo = self.rows

    def __len__(self):
        return self.rows - self.lo

    def gather(self, ids):
        """Assemble a batch from absolute row ids."""
        s = ids % self.cap
        lens = self.clen[s].sum(1).astype(np.int64)
        total = int(lens.sum())
        # Arena indices of every config of every chosen row, flattened.
        base = np.repeat(self.cstart[s], lens)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        at = (base + within) % self.ccap
        seg = 2 * np.repeat(np.arange(len(ids), dtype=np.int64), lens) + self.cp[at]
        return (self.x[s], self.cc[at], self.cp[at], self.cw[at].astype(np.float32),
                self.cy[at].astype(np.float32), seg)

    def sample(self, batch, rng):
        return self.gather(rng.integers(self.lo, self.rows, size=batch))

    def ordered(self):
        """Live rows oldest-first.

        Age order is what makes an honest held-out split possible: rows from
        one epoch come from the same games and are heavily correlated, so a
        random split leaks. Splitting by recency does not.
        """
        return self.gather(np.arange(self.lo, self.rows))


def make_batch(parts, rng, device, augment):
    """Numpy batch -> tensors, with the 180-degree mirror on a random half.

    Rotating the board and swapping the seats is an exact symmetry, so every
    row is usable twice. Measured offline on a frozen dump: held-out loss
    0.008446 -> 0.008161 and the train/test gap shrinks 38%, because the binding
    constraint on this network is distinct positions, not parameters. Applied
    per batch rather than stored, so the buffer does not double.

    On the config side the swap is one bit: a config carries the seat it belongs
    to as a feature, and `seg` is `2 * row + seat`.
    """
    x, cc, cp, cw, cy, seg = parts
    x = x.astype(np.float32)
    if augment:
        which = rng.random(len(x)) < 0.5
        x[which] = mirror.mirror_x(x[which])
        flip = which[seg // 2]
        cp = np.where(flip, 1 - cp, cp)
        seg = np.where(flip, seg ^ 1, seg)
    # Distinct configs only. Every count fits in four bits, so the whole vector
    # packs into one integer and the dedup is a sort rather than a row compare.
    packed = cp.astype(np.uint64)
    for k in range(CCOUNTS):
        packed = (packed << np.uint64(4)) | cc[:, k].astype(np.uint64)
    uniq, inv = np.unique(packed, return_inverse=True)
    first = np.zeros(len(uniq), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    phi = np.concatenate([cc[first].astype(np.float32) / CNORM,
                          cp[first, None].astype(np.float32)], 1)
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    return (t(x), t(phi), t(inv, torch.long), t(cw), t(seg, torch.long), t(cy), 2 * len(x))


def value_loss(net, xpub, phi, inv, w, seg, y, nseg):
    # Belief-weighted Huber over every config in the support. Weighting by the
    # belief is what makes the loss match the distribution CFR queries: a config
    # the belief gives 1% to is worth 1% of the gradient.
    v = net(xpub, phi, inv, w, seg, nseg)
    per = F.smooth_l1_loss(v, y, reduction="none", beta=0.5)
    return (per * w).sum() / w.sum().clamp(min=1e-6)


def train_steps(net, opt, buf, steps, batch, rng, device, augment=True):
    if len(buf) < batch:
        return float("nan")
    tot = 0.0
    for _ in range(steps):
        parts = make_batch(buf.sample(batch, rng), rng, device, augment)
        loss = value_loss(net, *parts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        tot += loss.detach().item()
    return tot / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--warm-frac", type=float, default=0.2)
    # The warm phase only has to initialise the network, and its data is
    # *discarded* at the switch. As a fraction of a long budget it is pure
    # waste: 0.2 of a nine-hour run is 1.8 hours of throwaway work. Prefer an
    # absolute length on any run longer than about half an hour.
    ap.add_argument("--warm-minutes", type=float, default=-1.0,
                    help="absolute warm-start length in minutes; overrides --warm-frac")
    ap.add_argument("--hidden", type=int, default=384)
    # Width of a config embedding, and so of one player's belief block. This is
    # the rank of the value function's dependence on the private state, and it
    # is also what the belief is summarised into -- the one place where a fixed
    # width is a real approximation, since a belief is a distribution over a
    # config space too large to enumerate.
    ap.add_argument("--dg", type=int, default=64)
    # Rank of the value readout's inner product -- see `Mlp`.
    ap.add_argument("--rank", type=int, default=64)
    # Arena size per row of replay capacity. Self-play carries ~24 configs a
    # decision; whichever of the two rings fills first sets the real window.
    ap.add_argument("--cfgs-per-row", type=int, default=48)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Step-decay the learning rate at fixed fractions of the ReBeL phase (the
    # reference repo halves Adam lr every 400 epochs, twice; wall-clock
    # fractions are more robust here since epoch throughput varies).
    ap.add_argument("--lr-decay-frac", default="0.33,0.67",
                    help="fractions of the ReBeL phase at which to halve the lr (comma-separated)")
    # ReBeL-phase value targets are pure CFR bootstrap. Blending some of the
    # realised game outcome in (MuZero-style n-step / TD(lambda)) can speed
    # learning; 0 recovers plain ReBeL.
    ap.add_argument("--mc-mix", type=float, default=0.0)
    # Extra fixed-reference diagnostics logged alongside the champion gate.
    # They do not select anything -- both saturate -- but they are comparable
    # across runs in a way the champion score is not. Each costs a full match,
    # so "none" is reasonable on a long run.
    ap.add_argument("--gate-vs", default="greedy",
                    choices=["none", "greedy", "init", "both"])
    ap.add_argument("--warm-games", type=int, default=96)
    ap.add_argument("--rebel-games", type=int, default=48)
    ap.add_argument("--train-gen-ratio", type=float, default=4.0)
    # depth 1 puts *zero* opponent decision nodes in the subgame, which reduces
    # CFR to 1-ply value iteration over the network. 2 is the reference's
    # setting for liar's dice and the minimum that is actually ReBeL.
    ap.add_argument("--depth", type=int, default=2)
    # CFR iterations per subgame. The old default of 8 was justified on
    # micro-endgames, which converge almost immediately and badly understate the
    # error on the ~540-node subgames self-play actually solves. Measured on
    # real mid-game positions against a converged T=512 reference
    # (`examples/solvererr.rs`), mean |error| in the root value is:
    #
    #     T=8  0.0098   (8% of the spread of the values themselves)
    #     T=16 0.0036   (3%)
    #     T=32 0.0016   (1.3%)
    #
    # This is *bias*, not noise -- the same position gives the same wrong number
    # every time -- so the network fits it happily and converges to the fixed
    # point of the under-solved operator. No training loss curve can show it.
    # Against that, T=16 costs 36% of the target rate and T=32 costs 63%, while
    # the data-scaling curve says a 36% data cut is worth roughly 3.5% of
    # held-out loss. Bias that compounds through bootstrapping is worth more
    # than that, and T=32 buys little more than T=16 for twice the cost.
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--explore", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--eval-mix", type=float, default=0.5)
    # Horizon payoff per marker of differential. Each side has 6 markers, so the
    # differential reaches +-5 and this must stay far below a real win (+-1) or
    # stalling out the clock becomes a competing win condition: at 0.15 a
    # five-marker lead banked 0.75 risk-free, which is what collapsed the first
    # run. 0.04 caps the shaped payoff at +-0.20.
    ap.add_argument("--cap-value", type=float, default=0.04)
    # Fraction of the ReBeL phase over which the horizon payoff decays to zero.
    # It reaches zero early so the tail of training -- and the checkpoint we
    # ship -- is fitted to the real game.
    ap.add_argument("--anneal-frac", type=float, default=0.4)
    # Gating is pure overhead against training time, so it runs rarely rather
    # than with many games: at 120 games the standard error is ~0.046, so a
    # peak 2 sigma above trend is probably noise and selecting on it biases the
    # reported score upward. Fewer, larger gates cost the same and select
    # better. `final_vs_init` is the headline number for exactly this reason --
    # it is not the quantity the checkpoint was selected on.
    ap.add_argument("--gate-every", type=float, default=1200.0)
    ap.add_argument("--gate-games", type=int, default=300)
    # Promotion threshold against the reigning champion, AlphaGo Zero's gating
    # rule. At 300 paired games the standard error is ~0.029, so 0.55 is about
    # 1.7 sigma: high enough that noise alone rarely promotes, low enough that
    # real progress is not held back.
    ap.add_argument("--promote", type=float, default=0.55)
    # Dump the replay buffer at the end of the run, oldest row first. Targets
    # here are a deterministic function of the input, so a frozen dump supports
    # noise-free offline comparisons of network architectures -- which is the
    # only way to resolve effects smaller than the +-0.05 that a short training
    # run wanders by on its own.
    ap.add_argument("--dump-buffer", default="",
                    help="path for an .npz dump of the replay buffer")
    # Replay capacity, and a genuine algorithmic knob rather than a memory
    # setting. The held-out error of this network falls monotonically with the
    # number of distinct positions it trains on, with no sign of saturating:
    # 40k -> 0.0122, 80k -> 0.0103, 160k -> 0.0086, 284k -> 0.0082. A nine-hour
    # run generates over ten million rows, so capacity decides how many of them
    # survive to be trained on. At 2544 bytes a row, 2M costs 4.7 GiB of arrays
    # and peaks at 5.1 GiB alongside PyTorch -- comfortable on a 16 GiB machine,
    # where 3M (7.5 GiB) would leave little headroom for the workers.
    ap.add_argument("--cap", type=int, default=2_000_000)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--random-draft", action="store_true")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable the 180-degree mirror augmentation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 8)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    value = Mlp(args.hidden, args.dg, args.rank).to(dev)
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    # Step-decay plan: halve the lr at each listed fraction of the ReBeL phase.
    lr_decays = sorted(float(x) for x in args.lr_decay_frac.split(",") if x.strip())
    next_decay = 0
    value.push(0)
    # Buffer capacity is the knob the data-scaling curve points at, so every
    # byte per row is a row we cannot hold. Public features are float16; counts
    # are the uint8 they already are; probabilities and targets live in [-1, 1]
    # where float16 resolves to ~0.001, a fiftieth of the network's own error.
    buf = Buffer(args.cap, args.cap * args.cfgs_per_row)

    total = args.minutes * 60.0
    warm = total * args.warm_frac if args.warm_minutes < 0 else args.warm_minutes * 60.0
    warm = min(warm, total)
    t0 = time.time()
    epoch, phase, log = 0, "greedy", []
    # Bootstrapped value learning is not monotone, so the shipped checkpoint is
    # the reigning *champion* rather than whatever is live when the clock runs
    # out. The champion only changes when the live network beats it over
    # `gate_games` paired games, which is a comparison that stays informative
    # for the whole run because the reference moves with it.
    champ = {"score": -1.0, "t": 0.0, "state": None, "promotions": 0}
    # Gate rarely: each gate is minutes of eval, so only big runs bother. The
    # first gate is `gate_every` seconds into the ReBeL phase; a run shorter
    # than that has no gates and ships the latest weights.
    next_gate = warm + args.gate_every
    gate_curve = []
    # The marker-differential payoff at the horizon distorts the game being
    # solved, so it is annealed away as soon as horizon games become rare, and
    # evaluation always runs on the real game (value 0).
    cap_v = args.cap_value
    warchest.set_cap_value(cap_v)
    probe = None
    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} hidden={args.hidden} dg={args.dg} rank={args.rank} depth={args.depth} "
          f"iters={args.iters} budget={total:.0f}s warm={warm:.0f}s device={dev} "
          f"draft={'random' if args.random_draft else 'starter'} "
          f"gate_every={args.gate_every:.0f}s promote={args.promote} "
          f"train_gen_ratio={args.train_gen_ratio} "
          f"augment={not args.no_augment} cap={args.cap}", flush=True)

    while True:
        el = time.time() - t0
        if el >= total:
            break
        if phase == "greedy" and el >= warm:
            # Freeze the warm-started network into slot 1: the initial
            # checkpoint the ReBeL phase has to beat. Slot 2 is the champion,
            # which starts as the same network and is only replaced when the
            # live one measurably beats it.
            value.push(1)
            value.push(CHAMP_SLOT)
            champ["state"] = {k: v.detach().cpu().clone()
                              for k, v in value.state_dict().items()}
            torch.save({"value": value.state_dict(), "hidden": args.hidden,
                        "dg": args.dg, "rank": args.rank}, f"{args.out}/ckpt_init.pt")
            # Drop the warm-phase data. Its job was to initialise the *network*,
            # not to serve as bootstrap targets: it comes from a different
            # policy and its targets are not bootstrapped. Keeping it is
            # actively harmful because the warm phase outproduces ReBeL by
            # ~20x, so it dominates the buffer for the whole run -- at depth 2
            # a 9-minute ReBeL phase contributed 7% of the buffer and the
            # network simply kept fitting greedy play (`runs/diagC`,
            # final_vs_init 0.478, i.e. no improvement at all).
            buf.clear()
            phase = "rebel"
            print(f"[t={el:6.1f}s] --- initial checkpoint saved, switching to ReBeL ---",
                  flush=True)

        tg = time.time()
        kw = dict(random_draft=args.random_draft)
        if phase == "greedy":
            d = warchest.gen_data(args.warm_games, args.seed * 1_000_003 + epoch, "greedy",
                                  temp=args.temp, eval_mix=args.eval_mix, **kw)
        else:
            d = warchest.gen_data(args.rebel_games, args.seed * 1_000_003 + epoch, "rebel",
                                  depth=args.depth, iters=args.iters, explore=args.explore,
                                  mc_mix=args.mc_mix, **kw)
        gen_s = time.time() - tg
        # Utilities live in [-1, 1]; so does the true value function, so clip
        # the bootstrapped targets to that range.
        vx = np.asarray(d["vx"], np.float32).reshape(-1, PUBFEAT)
        cc = np.asarray(d["cc"], np.uint8).reshape(-1, CCOUNTS)
        cw = np.asarray(d["cw"], np.float32)
        cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
        coff = np.asarray(d["coff"], np.int64)
        buf.add(vx.astype(np.float16), cc, cw.astype(np.float16),
                cy.astype(np.float16), coff)
        # A frozen batch from the warm phase. If the network's spread on it
        # collapses, the value function has gone degenerate -- the failure mode
        # a falling training loss hides.
        if probe is None and len(buf) >= 2048:
            probe = make_batch(buf.sample(2048, rng), rng, dev, False)
        tgt_mean, tgt_std = float(cy.mean()), float(cy.std())

        tt = time.time()
        # Hold a fixed train:generation sample ratio (the reference's
        # `train_gen_ratio: 4`) instead of a fixed step count. The step count
        # then tracks how much fresh data the generator actually produced, which
        # is what keeps the ratio stable across depths -- a fixed count swings
        # the ratio by ~18x between depth 1 and depth 2, and over-trains the
        # thin first epochs after the buffer is cleared.
        steps = max(1, round(args.train_gen_ratio * len(vx) / args.batch))
        lv = train_steps(value, opt, buf, steps, args.batch, rng, dev,
                         augment=not args.no_augment)
        train_s = time.time() - tt
        value.push(0)
        with torch.no_grad():
            probe_std = float(value(*probe[:5], probe[6]).std()) \
                if probe is not None else float("nan")

        # Anneal the horizon payoff to zero on a fixed schedule over the first
        # `anneal_frac` of the ReBeL phase. It must not react to the observed
        # horizon rate: paying out a marker differential at the cap makes
        # stalling a winning line, so a feedback rule that raises the payoff
        # when games stop finishing reinforces exactly the failure it sees.
        # Once it reaches zero the agent is solving the real game, where the
        # clock running out is a draw and only a real win scores.
        if phase == "rebel":
            span = max(args.anneal_frac * (total - warm), 1.0)
            cap_v = args.cap_value * max(0.0, 1.0 - (el - warm) / span)
            warchest.set_cap_value(cap_v)

        # Step the learning rate down at fixed fractions of the ReBeL phase.
        if phase == "rebel" and next_decay < len(lr_decays) and \
                (el - warm) >= lr_decays[next_decay] * (total - warm):
            for pg in opt.param_groups:
                pg["lr"] /= 2
            print(f"[t={el:6.1f}s] --- lr -> {opt.param_groups[0]['lr']:.2e} ---", flush=True)
            next_decay += 1

        # Periodic gate against the fixed reference opponent. Always scored on
        # the real game (horizon payoff 0, so running out the clock is a draw)
        # regardless of what the generator is currently training against --
        # otherwise gate scores drift with the anneal and checkpoint selection
        # would prefer whichever weights exploit the shaped payoff best.
        gate = None
        scores = {}
        promoted = False
        if phase == "rebel" and time.time() - t0 >= next_gate:
            warchest.set_cap_value(0.0)
            seedg = 900 + epoch
            # The selection gate: live network against the reigning champion.
            # Both fixed references saturate -- vs Greedy at ~0.97 and vs the
            # initial checkpoint at ~0.94 within ten minutes -- so on a long run
            # neither can order two late checkpoints, and selecting on them is
            # selecting on noise. The champion moves, so this stays a live
            # measurement for the whole run.
            w, l, dr = warchest.eval_match(args.gate_games, seedg + 2, "rebel", "rebel",
                                           depth=args.depth, iters=args.iters, temp=args.temp,
                                           slot_a=0, slot_b=CHAMP_SLOT,
                                           random_draft=args.random_draft)
            gate = (w + 0.5 * dr) / max(w + l + dr, 1)
            scores["champ"] = gate
            # The fixed references stay in the log as a curve, since they are
            # comparable across runs in a way the champion score is not.
            if args.gate_vs in ("greedy", "both"):
                w, l, dr = warchest.eval_match(args.gate_games, seedg, "rebel", "greedy",
                                               depth=args.depth, iters=args.iters, temp=args.temp,
                                               slot_a=0, random_draft=args.random_draft)
                scores["greedy"] = (w + 0.5 * dr) / max(w + l + dr, 1)
            if args.gate_vs in ("init", "both"):
                w, l, dr = warchest.eval_match(args.gate_games, seedg + 1, "rebel", "rebel",
                                               depth=args.depth, iters=args.iters, temp=args.temp,
                                               slot_a=0, slot_b=1, random_draft=args.random_draft)
                scores["init"] = (w + 0.5 * dr) / max(w + l + dr, 1)
            warchest.set_cap_value(cap_v)
            promoted = gate >= args.promote
            if promoted:
                champ = {"score": round(gate, 4), "t": round(time.time() - t0, 1),
                         "state": {k: v.detach().cpu().clone() for k, v in
                                   value.state_dict().items()},
                         "promotions": champ["promotions"] + 1}
                value.push(CHAMP_SLOT)
            gate_curve.append({"t": round(time.time() - t0, 1), "promoted": promoted,
                               **{k: round(v, 3) for k, v in scores.items()}})
            # Checkpoint to disk at every gate. A nine-hour run that keeps its
            # only copy of the champion in Python memory loses everything to a
            # crash or a sleep at hour seven.
            torch.save({"value": champ["state"], "hidden": args.hidden, "dg": args.dg, "rank": args.rank,
                        "score": champ["score"], "t": champ["t"],
                        "promotions": champ["promotions"]},
                       f"{args.out}/ckpt_champion.pt")
            torch.save({"value": value.state_dict(), "hidden": args.hidden,
                        "dg": args.dg, "rank": args.rank}, f"{args.out}/ckpt_live.pt")
            next_gate = time.time() - t0 + args.gate_every

        dec = max(d["decisions"], 1)
        rec = {"t": round(time.time() - t0, 1), "epoch": epoch, "phase": phase,
               "games": d["games"], "decisions": dec, "loss": round(lv, 5),
               "cap_frac": round(d["cap_hits"] / max(d["games"], 1), 3),
               "configs": round(d["configs"] / dec, 1), "cap_value": round(cap_v, 4),
               "steps": steps,
               "tgt_mean": round(tgt_mean, 4), "tgt_std": round(tgt_std, 4),
               "probe_std": round(probe_std, 4),
               "gen_s": round(gen_s, 2), "train_s": round(train_s, 2), "buf": len(buf),
               "lr": opt.param_groups[0]["lr"]}
        log.append(rec)
        # Rewritten every epoch, not just at gates: this is the file `plot.py`
        # reads, and a run should be watchable from its first minute rather than
        # only after the first gate. It is a few hundred kilobytes even on a
        # long run, so the cost is nothing against a multi-second epoch.
        with open(f"{args.out}/log.json", "w") as f:
            json.dump({"epochs": log, "gate": gate_curve,
                       "champ": {k: champ[k] for k in ("score", "t", "promotions")}},
                      f, indent=1)
        gstr = "  GATE " + " ".join(f"{k}={v:.3f}" for k, v in scores.items()) if scores else ""
        print(f"[t={rec['t']:6.1f}s] {phase:6s} ep{epoch:3d} games={rec['games']:4d} "
              f"dec={dec:6d} cap={rec['cap_frac']:.2f} cfgs={rec['configs']:5.1f} "
              f"L={lv:.5f} tgt={tgt_mean:+.3f}/{tgt_std:.3f} pstd={probe_std:.3f} "
              f"capv={cap_v:.3f} lr={rec['lr']:.1e} gen={gen_s:.1f}s train={train_s:.1f}s"
              + (f"{gstr}{'  *PROMOTED*' if promoted else ''}"
                 if gate is not None else ""), flush=True)
        epoch += 1

    # Ship the champion. If no gate ever ran (a run shorter than `gate_every`)
    # or nothing ever cleared the promotion threshold, the live network is what
    # there is -- and "no promotion in N gates" is itself the finding.
    if champ["state"] is not None and champ["promotions"] > 0:
        value.load_state_dict(champ["state"])
        value.push(0)
        print(f"\nshipping champion from t={champ['t']}s ({champ['promotions']} promotions, "
              f"last gate score {champ['score']:.3f})", flush=True)
    elif gate_curve:
        print(f"\nno promotion in {len(gate_curve)} gates -- shipping the live network. "
              f"Champion scores: {[g['champ'] for g in gate_curve]}", flush=True)
    torch.save({"value": value.state_dict(), "hidden": args.hidden, "dg": args.dg, "rank": args.rank},
               f"{args.out}/ckpt_final.pt")
    with open(f"{args.out}/log.json", "w") as f:
        json.dump({"epochs": log, "gate": gate_curve,
                   "champ": {k: champ[k] for k in ("score", "t", "promotions")}}, f, indent=1)

    if args.dump_buffer:
        # Oldest row first, so a recency split is an honest held-out set.
        x, cc, cp, cw, cy, seg = buf.ordered()
        np.savez(args.dump_buffer, x=x, cc=cc, cp=cp, cw=cw, cy=cy, seg=seg,
                 pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
                 ccounts=np.int32(CCOUNTS), cnorm=np.float32(CNORM))
        print(f"dumped {len(x)} buffer rows ({len(cy)} configs) to {args.dump_buffer}",
              flush=True)

    # ------------------------------------------------------------- evaluation
    warchest.set_cap_value(0.0)
    print(f"\n=== evaluation on the real game (horizon payoff 0; training ended at "
          f"{cap_v:.3f}) ===", flush=True)
    n = args.eval_games
    kw = dict(depth=args.depth, iters=args.iters, temp=args.temp,
              random_draft=args.random_draft)

    def report(name, res):
        w, l, dr = res
        tot = max(w + l + dr, 1)
        score = (w + 0.5 * dr) / tot
        se = (score * (1 - score) / tot) ** 0.5
        print(f"{name:38s} W{w:4d} L{l:4d} D{dr:4d}   score {score:.3f} +- {2*se:.3f}",
              flush=True)
        return score

    r = {}
    r["final_vs_greedy"] = report("final checkpoint vs Greedy",
                                  warchest.eval_match(n, 303, "rebel", "greedy", slot_a=0, **kw))
    r["final_vs_init"] = report("final checkpoint vs initial checkpoint",
                                warchest.eval_match(n, 101, "rebel", "rebel",
                                                    slot_a=0, slot_b=1, **kw))
    with open(f"{args.out}/eval.json", "w") as f:
        json.dump(r, f, indent=1)

    ok = r["final_vs_init"] > 0.5 and r["final_vs_greedy"] > 0.5
    print(f"\nGOAL: the run produced a checkpoint better than the initial one that also "
          f"beats Greedy -> {'PASS' if ok else 'FAIL'}", flush=True)
    print("      (ReBeL self-play must carry the warm-started network past its own "
          "start; final_vs_init is the headline)", flush=True)


if __name__ == "__main__":
    main()
