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
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest
import ladder
import mirror
from value_net import Mlp, AUX, AFEAT

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
NTYPE = warchest.NTYPE
ROW_BYTES = warchest.ROW_BYTES
ROW_IDS = warchest.ROW_IDS
ROW_AUX = warchest.ROW_AUX


def public_sizes(cc, cp, seg, n):
    """Per-row per-player hand/face-down/bag sizes, from the row's config
    support. `seg` must be non-decreasing with values `2 * row + seat`; all
    configs in a support share the sizes, so the first config of each span
    decides."""
    starts = np.searchsorted(seg, np.arange(2 * n, dtype=np.int64), side="left")
    cs = cc[starts]
    return (cs[:, :5].sum(1).astype(np.uint8).reshape(n, 2),
            cs[:, 5:10].sum(1).astype(np.uint8).reshape(n, 2),
            cs[:, 10:].sum(1).astype(np.uint8).reshape(n, 2))


def expand_batch(rows, hand, fd, bag):
    """Expand packed replay rows into the public encoding, in one batch.
    The expansion itself runs in Rust — one source of truth with the
    solver's leaf encoding."""
    n = len(rows)
    return np.asarray(warchest.expand_rows(rows.ravel(), hand, fd, bag), np.float32).reshape(n, -1)


class Buffer:
    """Fixed-capacity FIFO over rows whose config lists are ragged.

    Two rings advance together: one over rows, one over the config arena the
    rows point into. A row's configs sit at an *absolute* arena offset, so a row
    is live exactly while both rings still hold it -- and because both are
    written in order, the rows the arena has evicted are always the oldest ones,
    which is a single monotone pointer rather than a validity test per row.

    Bootstrapped targets are averaged over whatever history the buffer holds, so
    its length is a real algorithmic knob and not just a memory setting -- the
    reference implementation runs a 2M buffer. A row is the frozen compact
    format (`ROW_BYTES` raw bytes: hex facts, piles, unit ids, scalars, aux) --
    ~223 bytes instead of the ~1.9 KB the old float encoding cost -- and the
    network input is expanded from it when a batch is made. Counts are stored
    as the `uint8` they are and everything else as float16, which is what makes
    the cap affordable: a row costs `ROW_BYTES` bytes plus 20 per config.

    Preallocated and written with wraparound rather than grown by
    concatenation. The concatenate form rebuilt the whole buffer every epoch:
    at an 800k cap that copies ~2.6 GB per epoch and transiently holds two
    copies of it, which is most of a 16 GB machine. `np.zeros` maps zero pages
    lazily, so reserving the full capacity up front costs nothing until it is
    actually filled.

    Solve offsets are kept in row space (`soff`), so a dump can be split at
    solve boundaries for honest offline comparisons.
    """

    def __init__(self, cap, ccap):
        self.cap, self.ccap = cap, ccap
        self.x = np.zeros((cap, ROW_BYTES), np.uint8)
        self.soff = np.zeros(0, np.int64)
        self.cstart = np.zeros(cap, np.int64)   # absolute arena offset
        self.clen = np.zeros((cap, 2), np.int32)
        self.cc = np.zeros((ccap, CCOUNTS), np.uint8)
        self.cp = np.zeros(ccap, np.uint8)
        self.cw = np.zeros(ccap, np.float16)
        self.cy = np.zeros(ccap, np.float16)
        self.rows = 0   # rows ever written
        self.cfgs = 0   # configs ever written
        self.lo = 0     # oldest row whose configs are still in the arena

    def add(self, x, cc, cw, cy, coff, soff):
        n = len(x)
        lens = np.diff(coff).reshape(n, 2)
        cp = np.repeat(np.tile([0, 1], n).astype(np.uint8), lens.ravel())
        starts = self.cfgs + coff[:-1:2]
        base = self.rows
        for i in range(0, n, 4096):
            j = min(i + 4096, n)
            sl = np.arange(i, j) + base
            self.x[sl % self.cap] = x[i:j]
            self.cstart[sl % self.cap] = starts[i:j]
            self.clen[sl % self.cap] = lens[i:j]
        m = len(cw)
        sl = (np.arange(m) + self.cfgs) % self.ccap
        self.cc[sl], self.cp[sl], self.cw[sl], self.cy[sl] = cc, cp, cw, cy
        self.rows += n
        self.cfgs += m
        # Solve offsets in absolute row space (first entry 0, trailing count).
        self.soff = np.concatenate([self.soff, np.asarray(soff, np.int64)[1:] + base])
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
        """Assemble a batch from absolute row ids.

        Returns `(rows, cc, cp, cw, cy, seg)`; the aux targets and unit ids
        live inside the packed rows and are read out by `make_batch`.
        """
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

        TurboReBeL's per-solve rows are thinned to the ~8 log-spaced iterates
        plus the live belief before they reach the buffer, so rows inside one
        solve are not near-duplicates and plain row sampling is unbiased.
        """
        ids = rng.integers(self.lo, self.rows, size=batch)
        k = int(batch * recent_mix)
        if k > 0:
            span = max(1, int((self.rows - self.lo) * recent_frac))
            ids[:k] = rng.integers(self.rows - span, self.rows, size=k)
        return self.gather(ids)

    def sample_old(self, batch, rng, recent_frac=0.2):
        """A batch from outside the recent slice — the stale majority — for
        the age-bucket loss. A diagnostic, not training."""
        span = max(1, int((self.rows - self.lo) * recent_frac))
        hi = max(self.lo + 1, self.rows - span)
        return self.gather(rng.integers(self.lo, hi, size=batch))

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

    The mirror runs on the packed rows (hex arrays permuted, owners and
    players swapped, unit ids exchanged) and the config side is one bit: a
    config carries the seat it belongs to as a feature, and `seg` is
    `2 * row + seat`. The network input is then expanded from the (possibly
    mirrored) rows by the Rust encoder.
    """
    rows, cc, cp, cw, cy, seg = parts
    n = len(rows)
    # Public sizes name seats, so they are read off the config support before
    # the mirror (the seat flip below would scramble `seg`'s order).
    hand, fd, bag = public_sizes(cc, cp, seg, n)
    if augment:
        which = rng.random(n) < 0.5
        rows[which] = mirror.mirror_rows(rows[which])
        flip = which[seg // 2]
        cp = np.where(flip, 1 - cp, cp)
        seg = np.where(flip, seg ^ 1, seg)
        hand[which] = hand[which][:, ::-1]
        fd[which] = fd[which][:, ::-1]
        bag[which] = bag[which][:, ::-1]
    x = expand_batch(rows, hand, fd, bag)
    unit_ids = rows[:, ROW_IDS:ROW_IDS + NTYPE]
    ay = rows[:, ROW_AUX:ROW_AUX + 2 * AUX].view(np.float16).reshape(-1, AUX).astype(np.float32)
    # Every config gets its own holding-tower row, duplicates included.
    #
    # This used to deduplicate: the key was the row's unit ids (post-mirror),
    # the seat and the 15 counts, and `inv` mapped each config back to its
    # representative. It removed about 32% of config rows and cost more than it
    # saved. `np.unique` over that key was ~105 ms of the ~173 ms step, on the
    # same cores the generation workers want; computing the tower for every row
    # instead took the whole step to ~72 ms, and 228 -> 101 ms with 320 workers
    # running. The result is unchanged, not approximated: with `inv` the
    # identity, `crow` in `Mlp.forward` collapses to `seg // 2` and the two
    # gathers become no-ops, so the tower sees exactly the inputs it saw before.
    phi = np.concatenate([cc.astype(np.float32) / CNORM,
                          cp[:, None].astype(np.float32)], 1)
    inv = np.arange(len(cc), dtype=np.int64)
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    return (t(x), t(unit_ids, torch.long), t(phi), t(inv, torch.long), t(cw),
            t(seg, torch.long), t(cy), 2 * len(rows), t(ay))


def value_loss(net, xpub, unit_ids, phi, inv, w, seg, y, nseg):
    # Belief-weighted Huber over every config in the support. Weighting by the
    # belief is what makes the loss match the distribution CFR queries: a config
    # the belief gives 1% to is worth 1% of the gradient.
    v = net(xpub, unit_ids, phi, inv, w, seg, nseg)
    per = F.smooth_l1_loss(v, y, reduction="none", beta=0.5)
    return (per * w).sum() / w.sum().clamp(min=1e-6)


def policy_loss(net, d, ids, device):
    """Cross-entropy from the policy head to the solves' own reference strategy,
    weighted by the belief -- the same weighting the value loss uses, and for the
    same reason: a config the belief gives 1% to is worth 1% of the gradient.

    Trained on the fresh epoch only, never from the replay buffer. A value target
    is bootstrapped and gains from being averaged over a long history; a strategy
    is not, and the epoch regenerates every one of them.

    Action lists are ragged across solves, so they are padded to the widest in
    the batch and masked. The target gives illegal actions probability exactly
    zero, so legality needs no separate mask in the loss; the solver masks by
    real legality when it reads the head.
    """
    prow, pact, paoff, coff = d["prow"][ids], d["pact"][ids], d["paoff"], d["coff"]
    pa, pp = d["pa"].reshape(-1, AFEAT), d["pp"]
    cc, cw = d["cc"].reshape(-1, CCOUNTS), d["cw"]
    na = (paoff[ids + 1] - paoff[ids]).astype(np.int64)
    S, NA = len(ids), int(na.max())

    # The row's configs: both players for the belief block, the acting player's
    # alone for the strategy, which is indexed by them.
    both = [np.arange(coff[2 * r], coff[2 * r + 2]) for r in prow]
    mine = [np.arange(coff[2 * r + p], coff[2 * r + p + 1]) for r, p in zip(prow, pact)]
    nc = np.array([len(m) for m in mine])
    seg = np.concatenate([2 * j + (np.arange(len(b)) >= coff[2 * r + 1] - coff[2 * r])
                          for j, (b, r) in enumerate(zip(both, prow))]).astype(np.int64)
    both, mine = np.concatenate(both), np.concatenate(mine)

    apad = np.zeros((S, NA, AFEAT), np.float32)
    amask = np.zeros((S, NA), bool)
    tgt = np.zeros((int(nc.sum()), NA), np.float32)
    at, ct = 0, 0
    for j, i in enumerate(ids):
        apad[j, :na[j]] = pa[paoff[i]:paoff[i] + na[j]]
        amask[j, :na[j]] = True
        tgt[ct:ct + nc[j], :na[j]] = pp[at:at + nc[j] * na[j]].reshape(nc[j], na[j])
        at, ct = at + nc[j] * na[j], ct + nc[j]

    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    phi = lambda idx, seats: t(np.concatenate(
        [cc[idx].astype(np.float32) / CNORM, seats[:, None].astype(np.float32)], 1))
    csol = t(np.repeat(np.arange(S), nc), torch.long)

    # Expand the selected rows (one per solve) from the packed format.
    rows = d["rows"].reshape(-1, ROW_BYTES)[prow]
    first = np.stack([coff[2 * prow], coff[2 * prow + 1]], 1)  # [S, 2] span starts
    cs = cc[first]
    hand = cs[:, :, :5].sum(2).astype(np.uint8)
    fd = cs[:, :, 5:10].sum(2).astype(np.uint8)
    bag = cs[:, :, 10:].sum(2).astype(np.uint8)
    x = t(np.asarray(
        warchest.expand_rows(rows.ravel(), hand, fd, bag), np.float32).reshape(S, -1))
    e = net.cards(x, t(rows[:, ROW_IDS:ROW_IDS + NTYPE], torch.long))
    zb = net.holdings(phi(both, seg & 1), e[t(seg // 2, torch.long)])
    b = torch.zeros(2 * S, zb.shape[1], dtype=zb.dtype, device=device)
    b.index_add_(0, t(seg, torch.long), zb * t(cw[both]).unsqueeze(1))
    h = F.relu(net.ln0(net.public_trunk(x, e)))
    h = F.relu(net.ln1(net.w1(h) + net.wb(b.reshape(S, -1))))

    q = net.actions(t(apad.reshape(-1, AFEAT)),
                    e.repeat_interleave(NA, 0)).reshape(S, NA, -1)
    k = net.wp(h)[csol] + net.wk(net.holdings(phi(mine, np.repeat(pact, nc)), e[csol]))
    logit = (k.unsqueeze(1) * q[csol]).sum(-1).masked_fill(~t(amask, torch.bool)[csol], -1e30)
    ce = -(t(tgt) * logit.log_softmax(-1)).sum(-1)
    w = t(cw[mine])
    return (ce * w).sum() / w.sum().clamp(min=1e-6)


def train_steps(net, opt, buf, steps, batch, rng, device, augment=True,
                recent_mix=0.0, recent_frac=0.2, aux_weight=0.0, policy_weight=0.0,
                d=None, policy_batch=64):
    """Returns the mean value loss and the mean policy loss.

    The side tasks are summed into the same backward as the value loss, not
    stepped separately. Adam divides a gradient by its own magnitude, so a
    constant in front of a *standalone* step almost cancels -- a weight only
    means anything when it sets one term's size against another's in one sum.
    """
    if len(buf) < batch:
        return float("nan"), float("nan")
    live = policy_weight > 0.0 and d is not None and len(d["prow"]) > 0
    # Drawn unconditionally, and from its own stream: sampling policy labels out
    # of `rng` would shift every later value batch, so turning the policy on
    # would change the value batches too and any comparison against it would be
    # measuring the sampling, not the side task.
    prng = np.random.default_rng(rng.integers(1 << 62))
    tot, ptot = 0.0, 0.0
    for _ in range(steps):
        parts = make_batch(buf.sample(batch, rng, recent_mix, recent_frac),
                           rng, device, augment)
        loss = value_loss(net, *parts[:-1])
        ay = parts[-1]
        # What is reported is the value loss alone, so the column means the same
        # thing whether or not the side tasks are on.
        tot += loss.detach().item()
        # Both side tasks share the trunk and are dropped at play time, so they
        # are extra gradient per row for nothing at inference.
        if aux_weight > 0.0:
            loss = loss + aux_weight * net.aux_loss(parts[0], parts[1], ay)
        if live:
            n = len(d["prow"])
            pl = policy_loss(net, d, prng.choice(n, min(policy_batch, n), replace=False), device)
            ptot += pl.detach().item()
            loss = loss + policy_weight * pl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    return tot / steps, (ptot / steps if live else float("nan"))


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
    # Width of the second public matrix, the belief projection, the second
    # LayerNorm and the readouts. Metadata like every other width; default
    # equals --hidden, which is the network this split came from.
    ap.add_argument("--head", type=int, default=0,
                    help="readout width (default: --hidden)")
    # Tower depths (comma lists of widths; empty = the classic shape). These
    # are the experiment knobs the flexible format exists for.
    ap.add_argument("--pub", default="", help="public tower widths, e.g. 384,384")
    ap.add_argument("--hmlp", default="", help="extra per-iteration head widths")
    ap.add_argument("--card", default="", help="card describer hidden widths")
    ap.add_argument("--slot", default="", help="holding tower hidden widths")
    ap.add_argument("--nres", type=int, default=1, help="holding residual blocks")
    # Width of a config embedding, and so of one player's belief block. This is
    # the rank of the value function's dependence on the private state, and it
    # is also what the belief is summarised into -- the one place where a fixed
    # width is a real approximation, since a belief is a distribution over a
    # config space too large to enumerate.
    ap.add_argument("--dg", type=int, default=64)
    # Rank of the value readout's inner product -- see `Mlp`.
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--policy", type=float, default=0.0,
                    help="weight on the policy head's loss. Its labels are free -- every\nsolve already computes the reference strategy -- and the head is what\na warm start and any action shortlist read. Off by default: it trains\nthe shared trunk, so it changes the value network and has to be gated\nas its own change.")
    ap.add_argument("--aux", type=float, default=0.0,
                    help="weight on the auxiliary heads' loss. They predict dense facts\nabout how the game went -- markers three rounds on, the initiative\nflip, the result -- so every row gives the shared trunk a gradient\nthe single value number does not. Off by default until gated.")
    ap.add_argument("--de", type=int, default=32,
                    help="width of a card embedding: what the describer summarises a\ncard's rulebook facts into, and what every part that names a card reads")
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
    ap.add_argument("--cfr", default="linear",
                    help="the regret rule: linear, plus, dcfr, pcfr, sapcfr. "
                         "See docs/REBEL.md; they are one formula with four numbers.")
    ap.add_argument("--warm", type=float, default=0.0,
                    help="iterations the policy head's strategy is worth when a solve "
                         "is seeded from it. 0 starts uniform.")
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
    # Work package B: run the ReBeL solves on the CUDA service (one thread
    # owns GPU-0; the trainer stays on --device). The service must be present
    # at startup or the run fails loudly rather than falling back to CPU.
    ap.add_argument("--gpu", action="store_true",
                    help="run solves on the in-process CUDA service")
    # One service per listed CUDA device. Two services on a two-card box beat
    # one by nearly the card count: the trainer's own device still has plenty
    # of room left over, because this network is small and the trainer is not
    # what the cards are busy with.
    ap.add_argument("--gpu-devices", default="0",
                    help="comma-separated CUDA devices for the solve services")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    # With the GPU service, the CPU cores belong to the Rust workers.
    torch.set_num_threads(2 if args.gpu else (os.cpu_count() or 8))
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    towers = lambda s: [int(x) for x in s.split(",") if x.strip()] or None
    value = Mlp(args.hidden, args.dg, args.rank, args.de,
                head=(args.head or args.hidden),
                pub=towers(args.pub), hmlp=towers(args.hmlp),
                card=towers(args.card), slot=towers(args.slot),
                nres=args.nres).to(dev)
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    # Step-decay plan: halve the lr at each listed fraction of the ReBeL phase.
    lr_decays = sorted(float(x) for x in args.lr_decay_frac.split(",") if x.strip())
    next_decay = 0
    value.push(0)
    gpu_devices = [int(x) for x in args.gpu_devices.split(",") if x.strip()]
    if args.gpu:
        dims, w, b, ln = value.dims, *value.flat()
        warchest.gpu_start(dims, w, b, ln, devices=gpu_devices)
        print(f"[gpu] solve services up on {gpu_devices}", flush=True)
    # Buffer capacity is the knob the data-scaling curve points at, so every
    # byte per row is a row we cannot hold. Public features are float16; counts
    # are the uint8 they already are; probabilities and targets live in [-1, 1]
    # where float16 resolves to ~0.001, a fiftieth of the network's own error.
    buf = Buffer(args.cap, args.cap * args.cfgs_per_row)

    gen_box = None
    total = args.minutes * 60.0
    warm = total * args.warm_frac if args.warm_minutes < 0 else args.warm_minutes * 60.0
    warm = min(warm, total)
    t0 = time.time()
    epoch, phase, log = 0, "greedy", []
    # Fresh subgames per second over the whole ReBeL phase: the rate
    # docs/GPU_PERF_GOAL.md is about. Generation overlaps training, so
    # per-epoch `gen_s` is not it -- only cumulative solves over cumulative
    # ReBeL wall time counts every cost, including the trainer's own.
    rebel_t0, rebel_solves = None, 0
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
        # The window is a quarter of the snapshot cadence: at 10-minute
        # snapshots a 30-minute run's final lands ~30 s after the last timed
        # snapshot, and rating both would waste ladder pairings on near-twins.
        if snaps and el - snaps[-1]["t"] < args.snapshot_every * 60.0 / 4.0:
            snaps[-1]["label"] = label
            return
        path = f"{args.out}/snap_{len(snaps):02d}.pt"
        torch.save({"value": value.state_dict(), "spec": value.spec(),
                    "hidden": args.hidden, "head": args.head or args.hidden,
                    "dg": args.dg, "rank": args.rank, "de": args.de, "t": round(el, 1),
                    "label": label}, path)
        snaps.append({"label": label, "t": round(el, 1),
                      "file": os.path.basename(path)})
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    next_snap = float("inf")
    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} hidden={args.hidden} head={args.head or args.hidden} dg={args.dg} rank={args.rank} depth={args.depth} "
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
            # The solve services were started before warm training. Publish
            # the warm-started weights before the first ReBeL batch; otherwise
            # that whole batch runs on the freshly initialised network and the
            # first upload does not happen until after it returns.
            if args.gpu:
                flat = value.flat()
                for i in range(len(gpu_devices)):
                    warchest.gpu_set_weights(value.dims, *flat, device=i)
            next_snap = el + args.snapshot_every * 60.0
            # Drop the warm-phase data. Its job was to initialise the *network*,
            # not to serve as bootstrap targets: it comes from a different
            # policy and its targets are not bootstrapped. Keeping it is
            # actively harmful because the warm phase outproduces ReBeL by
            # ~20x, so it dominates the buffer for the whole run -- at depth 2
            # a 9-minute ReBeL phase contributed 7% of the buffer and the
            # network simply kept fitting greedy play (`runs/diagC`,
            # the run ended no stronger than it started).
            buf.clear()
            phase = "rebel"
            rebel_t0 = time.time()
            rebel_solves = 0
            print(f"[t={el:6.1f}s] --- switching to ReBeL ---", flush=True)

        tg = time.time()
        kw = dict(random_draft=args.random_draft)

        def start_gen(gen_seed):
            # One background thread; gpu_gen_data releases the GIL, so GPU 0
            # generates the next batch while this thread trains on the last.
            box = {}

            def go():
                box["d"] = warchest.gpu_gen_data(
                    args.rebel_games, gen_seed, "rebel",
                    depth=args.depth, iters=args.iters, explore=args.explore,
                    temp=args.temp, eval_mix=args.eval_mix,
                    mc_mix=args.mc_mix, cfr=args.cfr, warm=args.warm, **kw)

            th = threading.Thread(target=go, daemon=True)
            th.start()
            return th, box

        if phase == "greedy":
            d = warchest.gen_data(args.warm_games, args.seed * 1_000_003 + epoch, "greedy",
                                  temp=args.temp, eval_mix=args.eval_mix, **kw)
        elif args.gpu:
            # Generation overlaps training: batch N+1 is produced (from the
            # weights published after batch N-1's training) while batch N
            # trains. The service drains between calls, so a publication
            # never lands mid-solve. One batch of weight staleness, same as
            # ReBeL's periodic publication.
            if gen_box is None:
                gen_box = start_gen(args.seed * 1_000_003 + epoch)
            th, box = gen_box
            th.join()
            d = box["d"]
            flat = value.flat()
            for i in range(len(gpu_devices)):
                warchest.gpu_set_weights(value.dims, *flat, device=i)
            gen_box = start_gen(args.seed * 1_000_003 + epoch + 1)
        else:
            d = warchest.gen_data(args.rebel_games, args.seed * 1_000_003 + epoch, "rebel",
                                  depth=args.depth, iters=args.iters, explore=args.explore,
                                  mc_mix=args.mc_mix, cfr=args.cfr, warm=args.warm, **kw)
        gen_s = time.time() - tg
        # Utilities live in [-1, 1]; so does the true value function, so clip
        # the bootstrapped targets to that range. Rows stay packed (raw
        # bytes); the public encoding is expanded per batch.
        rows = np.asarray(d["rows"], np.uint8).reshape(-1, ROW_BYTES)
        cc = np.asarray(d["cc"], np.uint8).reshape(-1, CCOUNTS)
        cw = np.asarray(d["cw"], np.float32)
        cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
        coff = np.asarray(d["coff"], np.int64)
        soff = np.asarray(d["soff"], np.int64)
        # TurboReBeL exposes the solve count so the train:generation ratio
        # can count solves (the sampling unit of the data) instead of rows,
        # which turbo multiplies by ~T for near-duplicate data.
        solves = max(1, int(d["solves"]))
        if phase == "rebel":
            if rebel_t0 is None:
                rebel_t0, rebel_solves = time.time(), 0
            rebel_solves += solves
        sps = rebel_solves / max(time.time() - rebel_t0, 1e-9) if rebel_t0 else 0.0
        buf.add(rows, cc, cw.astype(np.float16), cy.astype(np.float16), coff, soff)
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
        #
        # The sample unit is the *solve*, not the row: TurboReBeL multiplies
        # rows per solve by ~T, and counting rows would inflate the step count
        # by the same factor for near-duplicate data. One solve is one sample,
        # matching the buffer's sampling unit.
        steps = max(1, round(args.train_gen_ratio * solves / args.batch))
        lv, lp = train_steps(value, opt, buf, steps, args.batch, rng, dev, aux_weight=args.aux,
                             policy_weight=(args.policy if phase == "rebel" else 0.0), d=d,
                         augment=not args.no_augment, recent_mix=args.recent_mix,
                         recent_frac=args.recent_frac)
        train_s = time.time() - tt
        value.push(0)
        with torch.no_grad():
            probe_std = float(value(*probe[:6], probe[7]).std()) \
                if probe is not None else float("nan")
            # Age-bucket loss: bootstrapped targets are written by past
            # versions of the net, so old rows carry stale labels. This curve
            # makes that staleness visible: if old-row loss falls while
            # fresh-row loss rises, training is overfitting the buffer.
            if len(buf) >= args.batch:
                old_parts = make_batch(
                    buf.sample_old(args.batch, rng, args.recent_frac), rng, dev, False)
                loss_old = float(value_loss(value, *old_parts[:-1]))
                new_parts = make_batch(
                    buf.sample(args.batch, rng, recent_mix=1.0,
                               recent_frac=args.recent_frac), rng, dev, False)
                loss_new = float(value_loss(value, *new_parts[:-1]))
            else:
                loss_old = loss_new = float("nan")

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
               "loss_policy": round(lp, 4),
               "rows": len(rows), "solves": solves,
               "loss_old": round(loss_old, 5), "loss_new": round(loss_new, 5),
               "horizon_frac": round(d["horizon_hits"] / max(d["games"], 1), 3),
               "node_caps": int(d["node_caps"]), "dropped": int(d["dropped"]),
               "configs": round(d["configs"] / dec, 1), "cap_value": round(cap_v, 4),
               "steps": steps,
               "tgt_mean": round(tgt_mean, 4), "tgt_std": round(tgt_std, 4),
               "probe_std": round(probe_std, 4),
               "gen_s": round(gen_s, 2), "train_s": round(train_s, 2), "buf": len(buf),
               "solves_per_s": round(sps, 1),
               "lr": opt.param_groups[0]["lr"]}
        log.append(rec)
        # Rewritten every epoch: this is the file `plot.py` reads, and a run
        # should be watchable from its first minute. It is a few hundred
        # kilobytes even on a long run, so the cost is nothing against a
        # multi-second epoch.
        write_log(args, log, snaps)
        print(f"[t={rec['t']:6.1f}s] {phase:6s} ep{epoch:3d} games={rec['games']:4d} "
              f"dec={dec:6d} rows={len(rows):6d} horizon={rec['horizon_frac']:.2f} "
              f"nodecap={rec['node_caps']} drop={rec['dropped']} "
              f"cfgs={rec['configs']:5.1f} L={lv:.5f} P={lp:.3f} old={loss_old:.5f} new={loss_new:.5f} "
              f"tgt={tgt_mean:+.3f}/{tgt_std:.3f} pstd={probe_std:.3f} "
              f"capv={cap_v:.3f} lr={rec['lr']:.1e} gen={gen_s:.1f}s train={train_s:.1f}s "
              f"sps={sps:.0f}",
              flush=True)
        epoch += 1

    snapshot("final", time.time() - t0)
    write_log(args, log, snaps)

    if args.dump_buffer:
        # Oldest row first, so a recency split is an honest held-out set.
        # The dump carries the frozen row format (version + rules hash) and
        # the solve offsets, so offline comparisons can split at solve
        # boundaries and refuse dumps from a different rules build.
        rows, cc, cp, cw, cy, seg = buf.ordered()
        # Solve boundaries in dump-row space; the oldest partial solve starts
        # before the dump, so 0 is prepended.
        lo = buf.lo
        soff = np.concatenate([[0], buf.soff[(buf.soff > lo) & (buf.soff < buf.rows)] - lo,
                               [len(rows)]])
        np.savez(args.dump_buffer, rows=rows, cc=cc, cp=cp, cw=cw, cy=cy, seg=seg,
                 soff=soff, pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
                 ccounts=np.int32(CCOUNTS), cnorm=np.float32(CNORM),
                 row_bytes=np.int32(ROW_BYTES), version=np.int32(warchest.ROW_FORMAT_VERSION),
                 rules_hash=np.uint64(warchest.rules_table_hash()))
        print(f"dumped {len(rows)} buffer rows ({len(cy)} configs) to {args.dump_buffer}",
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
