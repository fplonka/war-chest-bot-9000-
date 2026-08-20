"""ReBeL training for War Chest.

Two phases inside one wall-clock budget:

1. **Warm start** (`warm_minutes` of the budget). Both players are a stochastic
   one-ply greedy bot on a public-information evaluation; value targets blend
   that evaluation (squashed into (-1, 1)) with the realised game outcome.
   ReBeL never plays a policy directly — every move comes out of CFR using the
   value network at the leaves — so the value network is the natural place to
   inject a starting behaviour. Without it CFR searches on noise and no game
   ever ends inside the horizon. The network at the end of this phase is
   snapshot 0, labelled `init`: where ReBeL started, and the zero point the Elo
   curve is read against.

2. **GT-CFR** (the rest). Self-play where every decision grows a search tree
   along trajectories sampled from its changing strategy. Interior search
   queries become value targets, one value per config in each player's belief.

Generation runs in Rust across all CPU cores while the previous batch trains
on the GPU. Python publishes weights between generation batches.

A training row is a public state plus, for each player, the whole belief: the
exact configs in support, their probabilities, and the value the solve gave
each. The config lists are ragged, so they live in a flat arena and a batch is
assembled by gathering spans -- see `Buffer`.

The run snapshots every `snapshot_every` minutes of ReBeL. When training
ends, the snapshots play Greedy (random drafts) and a report is written.

    python train/train.py out=seat
    python train/train.py out=seat note="centre the seat bit at ±0.5"
    python train/train.py out=smoke minutes=6 warm_minutes=2
"""

import argparse
import collections
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest
import config
import mirror
from export_weights import load as load_checkpoint
from value_net import AFEAT, Net

ROOT = pathlib.Path(__file__).resolve().parent.parent

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES
N_KINDS = warchest.N_KINDS
NSLOT = warchest.NSLOT
N_HEXES = warchest.N_HEXES


def action_feats(pa):
    """The five stored bytes of each action into the head's one-hot input.

    The layout is the contract with `Net::action_feats`: kind, the coin slot it
    spends with the last column meaning none, then the three squares it names
    with the last column meaning no square.
    """
    feat = np.zeros((len(pa), AFEAT), np.float32)
    if not len(pa):
        return feat
    idx = np.arange(len(pa))
    feat[idx, pa[:, 0]] = 1.0
    feat[idx, N_KINDS + pa[:, 1]] = 1.0
    at = N_KINDS + NSLOT + 1
    for k in range(3):
        h = np.where(pa[:, 2 + k] == 255, N_HEXES, pa[:, 2 + k].astype(np.int64))
        feat[idx, at + h] = 1.0
        at += N_HEXES + 1
    return feat


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
    format (`ROW_BYTES` raw bytes: hex facts, piles, unit ids, and scalars) --
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
        # The policy target, per row: the root's actions, and the legal cells
        # with their probability. Only main-line rows carry one, so both arenas
        # are sized off the row cap rather than the config cap.
        self.pastart = np.zeros(cap, np.int64)
        self.palen = np.zeros(cap, np.int32)
        self.pcstart = np.zeros(cap, np.int64)
        self.pclen = np.zeros(cap, np.int32)
        self.acap = cap * 24
        self.pcap = cap * 96
        self.pa = np.zeros((self.acap, ACT_BYTES), np.uint8)
        self.pci = np.zeros(self.pcap, np.uint16)
        self.pact = np.zeros(self.pcap, np.uint8)
        self.pp = np.zeros(self.pcap, np.float16)
        self.acts = 0
        self.cells = 0
        self.rows = 0   # rows ever written
        # (rows written, when) at each insertion, trimmed to the live window.
        self.stamps = collections.deque()
        self.cfgs = 0   # configs ever written
        self.lo = 0     # oldest row whose configs are still in the arena

    def add(self, x, cc, cw, cy, coff, soff, pol=None):
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
        if pol is not None:
            pa, paoff, pcoff, pci, pact, pprob = pol
            alen = np.diff(paoff).astype(np.int32)
            clen = np.diff(pcoff).astype(np.int32)
            for i in range(0, n, 4096):
                j = min(i + 4096, n)
                sl = (np.arange(i, j) + base) % self.cap
                self.pastart[sl] = self.acts + paoff[i:j]
                self.palen[sl] = alen[i:j]
                self.pcstart[sl] = self.cells + pcoff[i:j]
                self.pclen[sl] = clen[i:j]
            na, nc = len(pa), len(pci)
            self.pa[(np.arange(na) + self.acts) % self.acap] = pa
            at = (np.arange(nc) + self.cells) % self.pcap
            self.pci[at], self.pact[at], self.pp[at] = pci, pact, pprob
            self.acts += na
            self.cells += nc
        self.rows += n
        self.cfgs += m
        # Solve offsets in absolute row space (first entry 0, trailing count).
        self.soff = np.concatenate([self.soff, np.asarray(soff, np.int64)[1:] + base])
        self.stamps.append((base, time.time()))
        # Advance past every row the arena no longer holds in full.
        floor = self.cfgs - self.ccap
        self.lo = max(self.lo, self.rows - self.cap)
        while self.lo < self.rows and self.cstart[self.lo % self.cap] < floor:
            self.lo += 1
        # Drop offsets whose rows have been evicted. Without this, soff grows
        # for the whole run and the concatenate above copies all of it every
        # chunk — quadratic in the run length.
        if self.soff.size:
            i = int(np.searchsorted(self.soff, self.lo, "right"))
            if i > self.soff.size // 2:
                self.soff = self.soff[i:].copy()

    def span_seconds(self):
        while len(self.stamps) > 1 and self.stamps[1][0] <= self.lo:
            self.stamps.popleft()
        return time.time() - self.stamps[0][1] if self.stamps else 0.0

    def clear(self):
        self.lo = self.rows
        self.stamps.clear()
        self.soff = np.zeros(0, np.int64)

    def __len__(self):
        return self.rows - self.lo

    def gather(self, ids):
        """Assemble a batch from absolute row ids.

        Returns `(rows, cc, cp, cw, cy, seg)`.
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
        # The policy target, remapped onto the batch. A row with no target
        # contributes no cells, which is how a query solve and the warm start
        # drop out of the policy loss without a mask.
        alen, clen = self.palen[s].astype(np.int64), self.pclen[s].astype(np.int64)
        ai = (np.repeat(self.pastart[s], alen)
              + (np.arange(int(alen.sum()), dtype=np.int64)
                 - np.repeat(np.concatenate([[0], np.cumsum(alen)[:-1]]), alen)))
        ci = (np.repeat(self.pcstart[s], clen)
              + (np.arange(int(clen.sum()), dtype=np.int64)
                 - np.repeat(np.concatenate([[0], np.cumsum(clen)[:-1]]), clen)))
        # An action's index becomes batch-global, and a cell names the query
        # (row, acting config) it belongs to.
        abase = np.concatenate([[0], np.cumsum(alen)[:-1]])
        cellrow = np.repeat(np.arange(len(ids), dtype=np.int64), clen)
        # A cell names its config within its own row; the batch arena puts that
        # row's configs at `rowbase`, so the two add to an arena index.
        rowbase = np.concatenate([[0], np.cumsum(lens)[:-1]])
        pol = (self.pa[ai % self.acap],
               np.repeat(abase, clen) + self.pact[ci % self.pcap],
               cellrow,
               np.repeat(rowbase, clen) + self.pci[ci % self.pcap].astype(np.int64),
               self.pp[ci % self.pcap].astype(np.float32),
               np.repeat(np.arange(len(ids), dtype=np.int64), alen))
        return (self.x[s], self.cc[at], self.cp[at],
                self.cw[at].astype(np.float32), self.cy[at].astype(np.float32),
                seg, pol)

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


def make_batch(parts, rng, device):
    """Numpy replay batch -> the two canonical player queries per row."""
    del rng
    rows, cc, cp, cw, cy, seg, pol = parts
    n = len(rows)
    hand, fd, bag = public_sizes(cc, cp, seg, n)
    views = np.empty((2 * n, ROW_BYTES), np.uint8)
    views[0::2] = rows
    views[1::2] = mirror.mirror_rows(rows)
    sizes = []
    for a in (hand, fd, bag):
        pair = np.empty((2 * n, 2), np.uint8)
        pair[0::2] = a
        pair[1::2] = a[:, ::-1]
        sizes.append(pair)
    x = expand_batch(views, *sizes)
    phi = cc.astype(np.float32) / CNORM
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    pa, pact, pcrow, pcfg, pprob, parow = pol
    policy = (t(action_feats(pa)), t(parow, torch.long), t(pact, torch.long),
              t(pcrow, torch.long), t(pcfg, torch.long), t(pprob))
    return (t(x), t(phi), t(cw), t(seg, torch.long), t(cy), 2 * n, policy)


def forward_values(net, parts):
    # `nseg` is the sixth element, not the last one: a batch carries the policy
    # target after it.
    return net(*parts[:4], parts[5])


def losses(net, xpub, phi, w, seg, y, nseg, policy=None, wp=1.0, stats=None):
    """Value Huber, mean per belief support and then across queries, plus the
    policy cross-entropy Student of Games trains the second head with."""
    v = net(xpub, phi, w, seg, nseg)
    if stats is not None:
        expected = torch.zeros(nseg, dtype=v.dtype, device=v.device)
        expected.index_add_(0, seg, v.detach() * w)
        residual = (expected[0::2] + expected[1::2]).abs().max().item()
        stats["zero_sum_max"] = max(stats["zero_sum_max"], residual)
    per = F.smooth_l1_loss(v, y, reduction="none", beta=0.5)
    total = torch.zeros(nseg, dtype=per.dtype, device=per.device)
    count = torch.zeros(nseg, dtype=per.dtype, device=per.device)
    total.index_add_(0, seg, per)
    count.index_add_(0, seg, torch.ones_like(per))
    loss = (total / count.clamp(min=1)).mean()
    # `L` and `L/var` are the *value* loss, as they were before the policy head
    # existed, so the run report still compares with every run before it. The
    # policy term is reported beside them, never folded into them.
    if stats is not None:
        stats["value_loss"] = float(loss.detach())
    if policy is not None and wp > 0.0:
        pl = policy_loss(net, xpub, phi, seg, nseg, policy)
        if pl is not None:
            if stats is not None:
                stats["policy_loss"] = float(pl)
            loss = loss + wp * pl
    return loss


def policy_loss(net, xpub, phi, seg, nseg, policy):
    """Cross entropy of the policy head against the search's root average.

    The head scores a `(config, action)` cell as `<f_p(c), e(a)>`, so the batch
    is exactly the cells the solves stored. Each cell's softmax runs over its
    own `(row, config)` group, which is one information state.
    """
    feat, parow, pact, _pcrow, pcfg, target = policy
    if feat.shape[0] == 0 or pact.shape[0] == 0:
        return None
    cards = net.cards(xpub)
    physical = xpub[0::2]
    board = net.board(physical, net.tokens(physical, cards[0::2]))
    _f, _g, fp = net.configs(phi, cards[:, :NSLOT], seg)
    e = net.actions(feat, board, parow)

    # `pcfg` is already an index into the batch's own config arena, so the cell
    # reads its config vector directly.
    logit = (fp[pcfg] * e[pact]).sum(1)

    # One softmax per information state, over its own cells. `pcfg` is already
    # unique across the batch, so it alone names the group.
    group = pcfg
    uniq, inv = torch.unique(group, return_inverse=True)
    top = torch.full((len(uniq),), -1e30, device=logit.device)
    top = top.scatter_reduce(0, inv, logit, reduce="amax")
    ex = (logit - top[inv]).exp()
    tot = torch.zeros(len(uniq), device=ex.device).index_add_(0, inv, ex)
    logp = (logit - top[inv]) - tot[inv].clamp(min=1e-30).log()
    per = -(target * logp)
    out = torch.zeros(len(uniq), device=per.device).index_add_(0, inv, per)
    return out.mean()


@torch.no_grad()
def diagnostics(net, buf, probe, batch, rng, device, batch_fn, recent_frac):
    """The spread of predictions on a fixed probe batch, and the value loss on
    stale rows against the freshest slice — the gap between the two is how far
    the bootstrapped targets have drifted under the network."""
    nan = float("nan")
    spread = float(forward_values(net, probe).std()) if probe is not None else nan
    if len(buf) < batch:
        return spread, nan, nan
    old = batch_fn(buf.sample_old(batch, rng, recent_frac), rng, device)
    new = batch_fn(buf.sample(batch, rng, recent_mix=1.0, recent_frac=recent_frac),
                   rng, device)
    return (spread,
            float(losses(net, *old, wp=0.0)),
            float(losses(net, *new, wp=0.0)))


def train_steps(net, opt, buf, steps, batch, rng, device,
                recent_mix=0.0, recent_frac=0.2, profile_cuda=False,
                batch_fn=make_batch, policy_w=0.0):
    """Mean loss over `steps` Adam updates -- value, plus the policy head's
    cross entropy at weight `policy_w`."""
    stat = {"sample_s": 0.0, "prepare_s": 0.0, "forward_wall_s": 0.0,
            "backward_wall_s": 0.0, "batch_configs": 0, "steps": steps,
            "gpu_forward_s": 0.0, "gpu_backward_s": 0.0,
            "zero_sum_max": 0.0, "grad_clipped": 0,
            "policy_loss": 0.0, "value_loss": 0.0, "policy_sum": 0.0}
    if len(buf) < batch:
        return float("nan"), stat
    tot = 0.0
    event_pairs = []
    stream = torch.cuda.current_stream(device) if profile_cuda and device.type == "cuda" else None
    for _ in range(steps):
        ts = time.perf_counter()
        sampled = buf.sample(batch, rng, recent_mix, recent_frac)
        stat["sample_s"] += time.perf_counter() - ts
        stat["batch_configs"] += len(sampled[-1])
        ts = time.perf_counter()
        parts = batch_fn(sampled, rng, device)
        stat["prepare_s"] += time.perf_counter() - ts
        if stream is not None:
            f0 = torch.cuda.Event(enable_timing=True)
            f1 = torch.cuda.Event(enable_timing=True)
            b1 = torch.cuda.Event(enable_timing=True)
            f0.record(stream)
        ts = time.perf_counter()
        value = losses(net, *parts, wp=policy_w, stats=stat)
        tot += stat.get("value_loss", float(value.detach()))
        stat["policy_sum"] += stat.get("policy_loss", 0.0)
        stat["forward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            f1.record(stream)
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        value.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        stat["grad_clipped"] += int(grad_norm > 5.0)
        opt.step()
        stat["backward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            b1.record(stream)
            event_pairs.append((f0, f1, b1))
    if event_pairs:
        torch.cuda.synchronize(device)
        stat["gpu_forward_s"] = sum(a.elapsed_time(b) for a, b, _ in event_pairs) / 1000.0
        stat["gpu_backward_s"] = sum(b.elapsed_time(c) for _, b, c in event_pairs) / 1000.0
    return tot / steps, stat




def physical_cpus():
    """Return one Linux hardware thread from each physical core."""
    cpus = set()
    root = "/sys/devices/system/cpu"
    if not os.path.isdir(root):
        return []
    for name in os.listdir(root):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        path = os.path.join(root, name, "topology", "thread_siblings_list")
        try:
            first = open(path).read().strip().split(",", 1)[0].split("-", 1)[0]
            cpus.add(int(first))
        except (OSError, ValueError):
            pass
    return sorted(cpus)

def refuse_if_machine_busy():
    """Catch a second run started by accident."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raw = ""
    for line in raw.strip().splitlines():
        bits = [b.strip() for b in line.split(",")]
        if len(bits) != 3:
            continue
        idx, util, mem = bits[0], float(bits[1]), float(bits[2])
        if util >= 25 or mem >= 2048:
            raise SystemExit(
                f"GPU {idx} is busy ({util:.0f}% util, {mem:.0f} MiB). "
                "Another run already going?")
    n = os.cpu_count() or 1
    load = os.getloadavg()[0]
    if load >= 0.5 * n:
        raise SystemExit(
            f"CPU load {load:.1f} on {n} CPUs. Another run already going?")


def write_log(args, epochs, snaps):
    """The run's whole record: settings, per-epoch stats, snapshot manifest.

    One file, rewritten in place, so `tools/arena.py` and `tools/monitor.py` have a
    single thing to read and a run that is still going is readable at any
    moment.
    """
    path = f"{args.out}/log.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"cfg": dataclasses.asdict(args), "epochs": epochs,
                   "snapshots": snaps}, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description="Train one run, then rate its snapshots against Greedy.")
    ap.add_argument("over", nargs="*", help="knob=value (production defaults)")
    over = config.parse(ap.parse_args().over)
    name = over.pop("out", None)
    if not name:
        raise SystemExit("pass out=<name>")
    args = dataclasses.replace(config.BASELINE, **over)
    args.git = config.git_sha()
    args.out = name if name.startswith("runs/") else f"runs/{name}"
    refuse_if_machine_busy()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists")
    os.makedirs(args.out)
    logf = open(f"{args.out}/train.log", "w")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logf.write(s)
            return len(s)
        def flush(self):
            sys.__stdout__.flush()
            logf.flush()
    sys.stdout = sys.stderr = Tee()
    print(f"[train] {args.out} at {args.git} seed={args.seed} {over or 'baseline'}",
          flush=True)
    if args.note:
        print(f"[train] {args.note}", flush=True)
    if args.gen_workers == 0:
        cores = physical_cpus()
        args.gen_workers = len(cores) or (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise SystemExit(f"device must be a CUDA device, got {args.device!r}")
    if args.replay_ratio <= 0.0:
        raise SystemExit("replay_ratio must be positive")
    if args.target_every <= 0.0:
        raise SystemExit("target_every must be positive minutes")
    if args.gen_solves <= 0 or args.gen_workers <= 0:
        raise SystemExit("gen_solves and resolved gen_workers must be positive")
    torch.cuda.set_device(dev)
    if args.train_stream_priority > 0:
        raise SystemExit("train_stream_priority must be zero or negative")
    if args.train_stream_priority < 0:
        default_stream = torch.cuda.current_stream(dev)
        train_stream = torch.cuda.Stream(
            device=dev, priority=args.train_stream_priority)
        train_stream.wait_stream(default_stream)
        torch.cuda.set_stream(train_stream)
        print(f"[train] CUDA stream priority {args.train_stream_priority}", flush=True)

    value = Net().to(dev)
    if args.init_weights:
        value.load_state_dict(load_checkpoint(args.init_weights).state_dict())
        args.warm_minutes = 0.0
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    lr_decays = sorted(float(x) for x in args.lr_decay_frac.split(",") if x.strip())
    next_decay = 0
    value.push()
    buf = Buffer(args.cap, args.cap * args.cfgs_per_row)
    import gpu_batch
    gpu_batch.warmup(dev)
    batcher = gpu_batch.make_batch
    print(f"[train] search inference on cuda:{args.gen_devices}, "
          f"training on {dev}", flush=True)

    total = args.minutes * 60.0
    warm = args.warm_minutes * 60.0
    if not 0.0 <= warm <= total:
        raise SystemExit("warm_minutes must be between zero and the run length")
    if args.snapshot_every <= 0:
        raise SystemExit("snapshot_every must be positive minutes")
    snap_gap = args.snapshot_every * 60.0
    t0 = time.time()
    epoch, log = 0, []
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
        # Never collapse a short run's trained result into `init`: those are
        # opposite ends of the experiment even when the whole run is shorter
        # than the ordinary snapshot de-duplication window.
        if snaps and snaps[-1]["label"] != "init" and \
                el - snaps[-1]["t"] < snap_gap / 4.0:
            snaps[-1]["label"] = label
            return
        path = f"{args.out}/snap_{len(snaps):02d}.pt"
        torch.save({"value": value.state_dict(), "t": round(el, 1),
                    "label": label, "git": args.git,
                    "search": {"nodes": args.nodes, "expand": args.expand,
                               "iters": args.iters, "cfr": args.cfr,
                               "node_cap": args.node_cap,
                               "config_cap": args.config_cap}}, path)
        snaps.append({"label": label, "t": round(el, 1),
                      "file": os.path.basename(path)})
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    def run_search_pipeline():
        """Overlap small GT-CFR batches with each other and with training."""
        nonlocal probe, cap_v, next_decay, next_snap, epoch, rebel_solves

        deadline = t0 + total
        # One process, many solver threads, one forward pass per round. A
        # process per core could not batch inference at all: the solves have to
        # share an address space for their leaf rows to end up in one batch.
        farm = warchest.SolveFarm(
            args.seed,
            args.gen_workers,
            nodes=args.nodes,
            expand=args.expand,
            iters=args.iters,
            explore=args.explore,
            random_draft=args.random_draft,
            cfr=args.cfr,
            node_cap=args.node_cap,
            config_cap=args.config_cap,
            query_rate=args.query_rate,
            recursive_rate=args.recursive_rate,
            devices=[int(d) for d in args.gen_devices.split(",")])

        optimizer_rows = 0
        optimizer_steps = 0
        generated_rows = 0
        window = collections.Counter()
        totals = collections.Counter()
        next_report = time.time() + 10.0
        next_target = rebel_t0 + args.target_every * 60.0

        while True:
            now = time.time()
            if now >= deadline:
                break
            gen_t = time.time()
            data = farm.collect(args.gen_solves)
            gen_s = time.time() - gen_t

            tc = time.time()
            rows = np.asarray(data["rows"], np.uint8).reshape(-1, ROW_BYTES)
            cc = np.asarray(data["cc"], np.uint8).reshape(-1, CCOUNTS)
            cw = np.asarray(data["cw"], np.float32)
            cy = np.clip(np.asarray(data["cy"], np.float32), -1.0, 1.0)
            coff = np.asarray(data["coff"], np.int64)
            soff = np.asarray(data["soff"], np.int64)
            conv_s = time.time() - tc
            ta = time.time()
            if len(rows):
                pol = (np.asarray(data["pa"], np.uint8).reshape(-1, ACT_BYTES),
                       np.asarray(data["paoff"], np.int64),
                       np.asarray(data["pcoff"], np.int64),
                       np.asarray(data["pci"], np.uint16),
                       np.asarray(data["pcell"], np.uint8),
                       np.asarray(data["pprob"], np.float16))
                buf.add(rows, cc, cw.astype(np.float16),
                        cy.astype(np.float16), coff, soff, pol)
            add_s = time.time() - ta

            solves = int(data["solves"])
            rebel_solves += solves
            generated_rows += len(rows)
            window["results"] += 1
            window["rows"] += len(rows)
            window["solves"] += solves
            window["target_n"] += cy.size
            window["target_sum"] += float(cy.sum(dtype=np.float64))
            window["target_square_sum"] += float(
                np.square(cy.astype(np.float64)).sum())
            window["gen_s"] += gen_s
            window["conv_s"] += conv_s
            window["add_s"] += add_s
            for name in (
                    "games", "decisions", "horizon_hits", "node_caps",
                    "plays_attack", "plays_pass", "plays_deploy",
                    "plays_bolster", "plays_maneuver", "plays_recruit",
                    "configs", "query_rows"):
                amount = int(data.get(name, 0))
                totals[name] += amount
                window[name] += amount

            debt = max(0.0, args.replay_ratio * generated_rows - optimizer_rows)
            nsteps = int(debt // args.batch) if len(buf) >= args.batch else 0
            train_s = 0.0
            lv = 0.0
            train_stat = {
                name: 0.0 for name in (
                    "sample_s", "prepare_s", "forward_wall_s",
                    "backward_wall_s", "gpu_forward_s", "gpu_backward_s",
                    "batch_configs", "zero_sum_max", "grad_clipped")
            }
            if nsteps:
                tt = time.time()
                lv, train_stat = train_steps(
                    value, opt, buf, nsteps, args.batch, rng, dev,
                    recent_mix=args.recent_mix,
                    recent_frac=args.recent_frac,
                    profile_cuda=os.environ.get("WARCHEST_TRAIN_PROFILE") == "1",
                    batch_fn=batcher, policy_w=args.policy_w)
                train_s = time.time() - tt
                optimizer_steps += nsteps
                optimizer_rows += nsteps * args.batch
                window["loss_sum"] += lv * nsteps
                window["policy_sum"] += train_stat["policy_sum"]
                window["train_steps"] += nsteps
                window["batch_configs"] += train_stat["batch_configs"]
                window["gpu_forward_s"] += train_stat["gpu_forward_s"]
                window["gpu_backward_s"] += train_stat["gpu_backward_s"]
                window["grad_clipped"] += train_stat["grad_clipped"]
                window["zero_sum_max"] = max(
                    window["zero_sum_max"], train_stat["zero_sum_max"])
            window["train_s"] += train_s

            now = time.time()
            if now >= next_target:
                # `push` bumps the version the farm watches; its threads
                # pick the new weights up at their next chunk.
                value.push()
                print(
                    f"[t={now - t0:6.1f}s] --- target network refresh ---",
                    flush=True)
                while next_target <= now:
                    next_target += args.target_every * 60.0
            rebel_elapsed = max(0.0, now - rebel_t0)
            span = max(args.anneal_frac * (total - warm), 1.0)
            cap_v = args.cap_value * max(0.0, 1.0 - rebel_elapsed / span)
            warchest.set_cap_value(cap_v)
            while next_decay < len(lr_decays) and \
                    rebel_elapsed >= lr_decays[next_decay] * (total - warm):
                for pg in opt.param_groups:
                    pg["lr"] /= 2
                print(
                    f"[t={now - t0:6.1f}s] --- lr -> "
                    f"{opt.param_groups[0]['lr']:.2e} ---",
                    flush=True)
                next_decay += 1
            if now - t0 >= next_snap:
                snapshot(f"s{len(snaps)}", now - t0)
                next_snap = now - t0 + snap_gap

            if now < next_report:
                epoch += 1
                continue
            next_report = now + 10.0
            steps = int(window["train_steps"])
            lv = window["loss_sum"] / max(steps, 1)
            if probe is None and len(buf) >= 2048:
                probe = batcher(buf.sample(2048, rng), rng, dev)
            probe_std, loss_old, loss_new = diagnostics(
                value, buf, probe, args.batch, rng, dev, batcher,
                args.recent_frac)
            target_n = max(int(window["target_n"]), 1)
            target_mean = window["target_sum"] / target_n
            target_var = max(
                0.0,
                window["target_square_sum"] / target_n
                - target_mean * target_mean)
            dec = max(int(window["decisions"]), 1)
            games = max(int(window["games"]), 1)
            raw_sps = rebel_solves / max(rebel_elapsed, 1e-9)
            gen_s = window["gen_s"] / max(window["results"], 1)
            train_s = window["train_s"]
            rec = {
                "t": round(now - t0, 1),
                "epoch": epoch,
                "phase": "rebel",
                "games": int(window["games"]),
                "decisions": int(window["decisions"]),
                "rows": int(window["rows"]),
                "solves": int(window["solves"]),
                "loss": round(lv, 5),
                "loss_old": round(loss_old, 5),
                "loss_new": round(loss_new, 5),
                "zero_sum_max": round(window["zero_sum_max"], 5),
                "grad_clip_frac": round(
                    window["grad_clipped"] / max(steps, 1), 4),
                "horizon_frac": round(window["horizon_hits"] / games, 3),
                "node_caps": int(window["node_caps"]),
                # How many solves shared a forward pass. It should sit near the
                # thread count; well below means the round is waiting on
                # stragglers instead of batching them.
                "calls_per_round": round(
                    int(data["round_calls"]) / max(int(data["rounds"]), 1), 2),
                "rows_per_round": round(
                    int(data["round_rows"]) / max(int(data["rounds"]), 1), 1),
                # Milliseconds a round spends inside the device backend — the
                # batch plus the concatenation and split around it. The rest of
                # a round is CFR on the cores.
                "device_ms_per_round": round(
                    1e-6 * int(data["round_nanos"]) / max(int(data["rounds"]), 1), 2),
                # Rows the query solver produced, i.e. targets taken off
                # the line of play. Zero means the coverage path is dead.
                "query_rows": int(window["query_rows"]),
                "plays": {
                    name: int(window[f"plays_{name}"])
                    for name in (
                        "attack", "pass", "deploy", "bolster",
                        "maneuver", "recruit")
                },
                "configs": round(window["configs"] / dec, 1),
                "cap_value": round(cap_v, 4),
                "steps": steps,
                "optimizer_steps": optimizer_steps,
                "optimizer_rows": optimizer_rows,
                "optimizer_debt": round(
                    max(0.0, args.replay_ratio * generated_rows
                        - optimizer_rows), 1),
                "replay_rows": generated_rows,
                "rows_per_s": round(
                    generated_rows / max(rebel_elapsed, 1e-9), 1),
                "effective_train_ratio": round(
                    optimizer_rows / max(rebel_solves, 1), 3),
                "train_row_ratio": round(
                    optimizer_rows / max(generated_rows, 1), 3),
                "tgt_mean": round(target_mean, 4),
                "tgt_std": round(target_var ** 0.5, 4),
                "probe_std": round(probe_std, 4),
                "gen_s": round(gen_s, 2),
                "train_s": round(train_s, 2),
                "conv_s": round(window["conv_s"], 2),
                "add_s": round(window["add_s"], 2),
                "gpu_forward_s": round(window["gpu_forward_s"], 2),
                "gpu_backward_s": round(window["gpu_backward_s"], 2),
                "batch_configs": round(
                    window["batch_configs"] / max(steps, 1), 1),
                "buf": len(buf),
                "buf_s": round(buf.span_seconds(), 1),
                "solves_per_s": round(raw_sps, 1),
                "lr": opt.param_groups[0]["lr"],
                "policy_loss": window["policy_sum"] / max(window["train_steps"], 1),
            }
            log.append(rec)
            write_log(args, log, snaps)
            print(
                f"[t={rec['t']:6.1f}s] GT-CFR solves={rebel_solves} "
                f"rate={raw_sps:.1f}/s rows={rec['rows']} "
                f"games={rec['games']} qrows={rec['query_rows']} "
                f"caps={totals['node_caps']} "
                f"L={lv:.5f} L/var={lv / max(target_var, 1e-9):.2f} "
                f"Lp={rec['policy_loss']:.3f} "
                f"tgt={target_mean:+.3f}/{target_var ** 0.5:.3f} "
                f"gen={gen_s:.2f}s train={train_s:.2f}s "
                f"gpu={window['gpu_forward_s'] + window['gpu_backward_s']:.2f}s",
                flush=True)
            window.clear()
            epoch += 1
        # Dropping the farm stops its threads once they finish the solve they
        # are in, which is also what flushes their last rows.
        del farm

        elapsed = max(deadline - rebel_t0, 1e-9)
        print(
            f"[GT-CFR-summary] solves={rebel_solves} "
            f"optimizer_rows={optimizer_rows} "
            f"rate={rebel_solves / elapsed:.1f}/s "
            f"horizon={totals['horizon_hits'] / max(totals['games'], 1):.2f} "
            f"games={totals['games']} caps={totals['node_caps']}",
            flush=True)

    next_snap = float("inf")
    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} architecture=gt-cfr "
          f"nodes={args.nodes} expand={args.expand} iters={args.iters} "
          f"budget={total:.0f}s warm={warm:.0f}s "
          f"snapshot_every={args.snapshot_every:g}min device={dev} "
          f"draft={'random' if args.random_draft else 'starter'} "
          f"replay_ratio={args.replay_ratio} "
          f"recent_mix={args.recent_mix}/{args.recent_frac} "
          f"canonical_views=2 cap={args.cap} "
          f"matmul={torch.get_float32_matmul_precision()}", flush=True)

    while True:
        el = time.time() - t0
        if el >= warm:
            break
        tg = time.time()
        d = warchest.gen_data(args.warm_games, args.seed * 1_000_003 + epoch, "greedy",
                              temp=args.temp, eval_mix=args.eval_mix,
                              random_draft=args.random_draft)
        gen_s = time.time() - tg
        tr = time.time()
        rows = np.asarray(d["rows"], np.uint8).reshape(-1, ROW_BYTES)
        cc = np.asarray(d["cc"], np.uint8).reshape(-1, CCOUNTS)
        cw = np.asarray(d["cw"], np.float32)
        cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
        coff = np.asarray(d["coff"], np.int64)
        soff = np.asarray(d["soff"], np.int64)
        solves = max(1, int(d["solves"]))
        conv_s = time.time() - tr
        tr = time.time()
        buf.add(rows, cc, cw.astype(np.float16), cy.astype(np.float16), coff, soff)
        add_s = time.time() - tr
        if probe is None and len(buf) >= 2048:
            probe = batcher(buf.sample(2048, rng), rng, dev)
        tgt_mean, tgt_std = float(cy.mean()), float(cy.std())
        tt = time.time()
        # One optimizer row per deterministic warm row; repeated fitting only
        # reduces the number of independent games seen before ReBeL starts.
        steps = max(1, round(solves / args.batch))
        lv, train_stat = train_steps(
            value, opt, buf, steps, args.batch, rng, dev,
            recent_mix=args.recent_mix, recent_frac=args.recent_frac,
            batch_fn=batcher)
        train_s = time.time() - tt
        value.push()
        probe_std, loss_old, loss_new = diagnostics(
            value, buf, probe, args.batch, rng, dev, batcher, args.recent_frac)
        dec = max(d["decisions"], 1)
        rec = {"t": round(time.time() - t0, 1), "epoch": epoch, "phase": "greedy",
               "games": d["games"], "decisions": dec, "loss": round(lv, 5),
               "rows": len(rows), "solves": solves,
               "loss_old": round(loss_old, 5), "loss_new": round(loss_new, 5),
               "zero_sum_max": round(train_stat["zero_sum_max"], 5),
               "horizon_frac": round(d["horizon_hits"] / max(d["games"], 1), 3),
               "node_caps": int(d["node_caps"]),
               "plays_attack": int(d.get("plays_attack", 0)),
               "plays_pass": int(d.get("plays_pass", 0)),
               "plays_deploy": int(d.get("plays_deploy", 0)),
               "plays_bolster": int(d.get("plays_bolster", 0)),
               "plays_maneuver": int(d.get("plays_maneuver", 0)),
               "plays_recruit": int(d.get("plays_recruit", 0)),
               "configs": round(d["configs"] / dec, 1), "cap_value": round(cap_v, 4),
               "steps": steps,
               "tgt_mean": round(tgt_mean, 4), "tgt_std": round(tgt_std, 4),
               "probe_std": round(probe_std, 4),
               "gen_s": round(gen_s, 2), "train_s": round(train_s, 2),
               "conv_s": round(conv_s, 2), "add_s": round(add_s, 2), "buf": len(buf),
               "buf_s": round(buf.span_seconds(), 1),
               "solves_per_s": 0.0,
               "lr": opt.param_groups[0]["lr"]}
        log.append(rec)
        write_log(args, log, snaps)
        print(f"[t={rec['t']:6.1f}s] greedy ep{epoch:3d} games={rec['games']:4d} "
              f"dec={dec:6d} rows={len(rows):6d} horizon={rec['horizon_frac']:.2f} "
              f"L={lv:.5f} L/var={lv / max(tgt_std ** 2, 1e-9):.2f} "
              f"tgt={tgt_mean:+.3f}/{tgt_std:.3f} pstd={probe_std:.3f} "
              f"gen={gen_s:.1f}s train={train_s:.1f}s",
              flush=True)
        epoch += 1

    snapshot("init", time.time() - t0)
    next_snap = (time.time() - t0) + snap_gap
    # Warm-up initialises the network. Its Adam moments and its rows are a
    # different objective; they must not steer ReBeL.
    buf.clear()
    probe = None
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    rebel_t0 = time.time()
    rebel_solves = 0
    print(f"[t={time.time() - t0:6.1f}s] --- switching to GT-CFR ---", flush=True)
    run_search_pipeline()

    snapshot("final", time.time() - t0)
    write_log(args, log, snaps)
    if args.ladder_games:
        # Every snapshot becomes an immutable bot for the ladder, and a bot
        # solves on whatever cards it finds. A ladder is thousands of solves at
        # the training budget; on the CPU that was an hour a run, which is far
        # too dear for the thing that says whether the run learned anything.
        arena = [sys.executable, str(ROOT / "tools" / "arena.py")]
        subprocess.run(arena + ["pack", args.out], check=True)
        tag = pathlib.Path(args.out).name
        bots = sorted(str(p) for p in (ROOT / "bots").glob(f"{tag}.*"))
        # Greedy first, so ratings are quoted against the one reference that
        # means the same thing from one run to the next.
        greedy = ROOT / "bots" / "greedy"
        anchor = [str(greedy)] if (greedy / "bot.json").exists() else []
        subprocess.run(arena + ["ladder", *anchor, *bots,
                                "--games", str(args.ladder_games),
                                "--out", f"{args.out}/ladder.json"], check=True)


if __name__ == "__main__":
    main()
