"""ReBeL training for War Chest.

Two phases inside one wall-clock budget:

1. **Warm start** (`--warm-frac` of the budget). Both players are a stochastic
   one-ply greedy bot on a public-information evaluation; value targets blend
   that evaluation (squashed into (-1, 1)) with the realised game outcome.
   ReBeL never plays a policy directly — every move comes out of CFR using the
   value network at the leaves — so the value network is the natural place to
   inject a starting behaviour. Without it CFR searches on noise and no game
   ever ends inside the horizon. The network at the end of this phase is
   snapshot 0, labelled `init`: where ReBeL started, and the zero point the Elo
   curve is read against.

2. **ReBeL** (the rest). Self-play where every decision solves a depth-limited
   CFR subgame over public belief states; the targets are the CFR root values,
   one per config in each player's belief support.

Everything except the gradient step runs in Rust across all cores; Python ships
weights down and pulls tensors back once per epoch.

A training row is a public state plus, for each player, the whole belief: the
exact configs in support, their probabilities, and the value the solve gave
each. The config lists are ragged, so they live in a flat arena and a batch is
assembled by gathering spans -- see `Buffer`.

The run saves a snapshot every `--snapshot-every` minutes and does not try to
decide which one is best while it is training. `ladder.py` plays them against
each other, and against Greedy and Random, once the run is over and turns the
results into Elo — a curve of strength against training time, which is the
thing we actually wanted to know.
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
import ladder
import mirror
from value_net import Mlp

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM


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

    def sample(self, batch, rng, recent_mix=0.0, recent_frac=0.2):
        """A batch, part of it drawn from the newest rows only.

        Uniform sampling over a 2M-row buffer is not neutral here: the targets
        are bootstrapped, so an old row's target was written by an old network
        and is wrong by however much the network has moved since. Sampling
        purely by recency is not the fix either, because held-out error falls
        monotonically with the number of *distinct* positions trained on
        (40k -> 0.0122, 284k -> 0.0082), and a recency-only sampler throws that
        away to refit a small window.

        So: a mixture. `recent_mix` of the batch comes from the newest
        `recent_frac` of the buffer and the rest from all of it, which draws a
        row in the fresh slice `1 + mix / ((1 - mix) * frac)` times as often as
        an old one -- 6x at the defaults, or 3x the average rate -- while
        leaving every row reachable. Two uniform draws, and no weight vector to
        rebuild each epoch.
        """
        ids = rng.integers(self.lo, self.rows, size=batch)
        k = int(batch * recent_mix)
        if k > 0:
            span = max(1, int((self.rows - self.lo) * recent_frac))
            ids[:k] = rng.integers(self.rows - span, self.rows, size=k)
        return self.gather(ids)

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


def train_steps(net, opt, buf, steps, batch, rng, device, augment=True,
                recent_mix=0.0, recent_frac=0.2):
    if len(buf) < batch:
        return float("nan")
    tot = 0.0
    for _ in range(steps):
        parts = make_batch(buf.sample(batch, rng, recent_mix, recent_frac),
                           rng, device, augment)
        loss = value_loss(net, *parts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        tot += loss.detach().item()
    return tot / steps


def write_log(args, epochs, snaps):
    """The run's whole record: settings, per-epoch stats, snapshot manifest.

    One file, rewritten in place, so `plot.py` and `ladder.py` have a single
    thing to read and a run that is still going is readable at any moment.
    """
    with open(f"{args.out}/log.json", "w") as f:
        json.dump({"cfg": vars(args), "epochs": epochs, "snapshots": snaps},
                  f, indent=1)


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
    # Fraction of each batch drawn from the newest slice of the buffer, and how
    # big that slice is. See `Buffer.sample`.
    ap.add_argument("--recent-mix", type=float, default=0.5)
    ap.add_argument("--recent-frac", type=float, default=0.2)
    ap.add_argument("--warm-games", type=int, default=96)
    ap.add_argument("--rebel-games", type=int, default=48)
    ap.add_argument("--train-gen-ratio", type=float, default=4.0)
    # depth 1 puts *zero* opponent decision nodes in the subgame, which reduces
    # CFR to 1-ply value iteration over the network. 2 is the reference's
    # setting for liar's dice and the minimum that is actually ReBeL.
    ap.add_argument("--depth", type=int, default=2)
    # CFR iterations per subgame. Measured on real mid-game positions against a
    # converged T=512 reference (`examples/solvererr.rs`), mean |error| in the
    # root value is:
    #
    #     T=8  0.0098   (8% of the spread of the values themselves)
    #     T=16 0.0036   (3%)
    #     T=32 0.0016   (1.3%)
    #
    # This is *bias*, not noise -- the same position gives the same wrong number
    # every time -- so the network fits it happily and converges to the fixed
    # point of the under-solved operator. No training loss curve can show it.
    # Earlier runs traded that bias for throughput and settled on 16, on the
    # grounds that the lost data was worth more. That is the wrong trade to keep
    # making: the whole claim of ReBeL is that the targets are the values of a
    # *solved* subgame, and at T=16 they are the values of a subgame we stopped
    # solving early. 64 costs roughly 2.5x the generation rate of 16.
    ap.add_argument("--iters", type=int, default=64)
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
    # Save the network this often. Snapshots cost a file write and nothing else:
    # no games are played during training, and no snapshot is treated as better
    # than another until the ladder says so.
    ap.add_argument("--snapshot-every", type=float, default=6.0,
                    help="minutes between snapshots")
    # Paired games per pairing in the closing Elo ladder. 0 skips it, for when
    # the ladder will be run separately (`python train/ladder.py <run>`).
    ap.add_argument("--ladder-games", type=int, default=60)
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
    # The marker-differential payoff at the horizon distorts the game being
    # solved, so it is annealed away as soon as horizon games become rare, and
    # evaluation always runs on the real game (value 0).
    cap_v = args.cap_value
    warchest.set_cap_value(cap_v)
    probe = None

    # Snapshots. Nothing selects between them during the run. Bootstrapped value
    # learning is not monotone, so there is a real question about which weights
    # are best -- but a match large enough to answer it costs minutes of the
    # budget (300 paired games: standard error 0.029, about the size of the gap
    # between neighbouring snapshots), and answering it from a noisy match is
    # how you ship a checkpoint chosen by a coin flip. The ladder rates all of
    # them at the end, off the clock.
    snaps = []

    def snapshot(label, el):
        # "init" and "final" are the two the reader always wants named; the rest
        # are numbered, and the manifest carries the time each was taken at.
        # Relabelling instead of resaving keeps the ladder from rating the same
        # weights twice when the clock runs out just after a periodic snapshot.
        if snaps and el - snaps[-1]["t"] < 30.0:
            snaps[-1]["label"] = label
            return
        path = f"{args.out}/snap_{len(snaps):02d}.pt"
        torch.save({"value": value.state_dict(), "hidden": args.hidden,
                    "dg": args.dg, "rank": args.rank, "t": round(el, 1),
                    "label": label}, path)
        snaps.append({"label": label, "t": round(el, 1),
                      "file": os.path.basename(path)})
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    next_snap = float("inf")
    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} hidden={args.hidden} dg={args.dg} rank={args.rank} depth={args.depth} "
          f"iters={args.iters} budget={total:.0f}s warm={warm:.0f}s device={dev} "
          f"draft={'random' if args.random_draft else 'starter'} "
          f"snapshot_every={args.snapshot_every:.1f}min "
          f"train_gen_ratio={args.train_gen_ratio} "
          f"recent_mix={args.recent_mix}/{args.recent_frac} "
          f"augment={not args.no_augment} cap={args.cap}", flush=True)

    while True:
        el = time.time() - t0
        if el >= total:
            break
        if phase == "greedy" and el >= warm:
            # The warm-started network is snapshot 0: where the ReBeL phase
            # started, and the zero point the Elo curve is read against.
            snapshot("init", el)
            next_snap = el + args.snapshot_every * 60.0
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
            print(f"[t={el:6.1f}s] --- switching to ReBeL ---", flush=True)

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
                         augment=not args.no_augment, recent_mix=args.recent_mix,
                         recent_frac=args.recent_frac)
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

        # Snapshot on a wall-clock schedule. No games are played and nothing
        # is compared: the snapshot is just the weights, and the ladder decides
        # afterwards what they were worth.
        if phase == "rebel" and time.time() - t0 >= next_snap:
            snapshot(f"s{len(snaps)}", time.time() - t0)
            next_snap = time.time() - t0 + args.snapshot_every * 60.0

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
        # Rewritten every epoch: this is the file `plot.py` reads, and a run
        # should be watchable from its first minute. It is a few hundred
        # kilobytes even on a long run, so the cost is nothing against a
        # multi-second epoch.
        write_log(args, log, snaps)
        print(f"[t={rec['t']:6.1f}s] {phase:6s} ep{epoch:3d} games={rec['games']:4d} "
              f"dec={dec:6d} cap={rec['cap_frac']:.2f} cfgs={rec['configs']:5.1f} "
              f"L={lv:.5f} tgt={tgt_mean:+.3f}/{tgt_std:.3f} pstd={probe_std:.3f} "
              f"capv={cap_v:.3f} lr={rec['lr']:.1e} gen={gen_s:.1f}s train={train_s:.1f}s",
              flush=True)
        epoch += 1

    snapshot("final", time.time() - t0)
    write_log(args, log, snaps)

    if args.dump_buffer:
        # Oldest row first, so a recency split is an honest held-out set.
        x, cc, cp, cw, cy, seg = buf.ordered()
        np.savez(args.dump_buffer, x=x, cc=cc, cp=cp, cw=cw, cy=cy, seg=seg,
                 pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
                 ccounts=np.int32(CCOUNTS), cnorm=np.float32(CNORM))
        print(f"dumped {len(x)} buffer rows ({len(cy)} configs) to {args.dump_buffer}",
              flush=True)

    # ------------------------------------------------------------- the ladder
    # Every snapshot against every other, plus Greedy and Random, on the real
    # game. This is the only strength measurement the run makes, and it makes it
    # once, at the end, where it can afford enough games to mean something.
    if args.ladder_games > 0:
        ladder.run(args.out, games=args.ladder_games, depth=args.depth,
                   iters=args.iters, temp=args.temp, random_draft=args.random_draft)


if __name__ == "__main__":
    main()
