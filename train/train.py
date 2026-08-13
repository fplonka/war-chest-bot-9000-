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

2. **ReBeL** (the rest). Self-play where every decision solves a depth-limited
   CFR subgame over public belief states; the targets are the CFR root values,
   one per config in each player's belief support.

Everything except the gradient step runs in Rust across all cores; Python ships
weights down and pulls tensors back once per epoch.

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
from value_net import Mlp

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
NTYPE = warchest.NTYPE
ROW_BYTES = warchest.ROW_BYTES
ROW_IDS = warchest.ROW_IDS


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
        # (rows written, when) at each insertion, trimmed to the live window.
        self.stamps = collections.deque()
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
            t(seg, torch.long), t(cy), 2 * len(rows))


def zero_sum_residual(v, w, seg, nseg):
    """Per position, how far the two players' belief-weighted values are from
    cancelling. War Chest is zero-sum, so this is zero for the true value
    function. Nothing in a per-config regression asks for it, and under
    bootstrapping a constant shared by both players lands in both their targets,
    so the loop carries it rather than correcting it. `seg` is `2 * row + seat`,
    so the two seats of a position are neighbours."""
    num = torch.zeros(nseg, dtype=v.dtype, device=v.device).index_add_(0, seg, w * v)
    den = torch.zeros(nseg, dtype=v.dtype, device=v.device).index_add_(0, seg, w)
    m = num / den.clamp(min=1e-9)
    return m[0::2] + m[1::2]


def probe_stats(net, probe):
    """Spread of the network's values on the fixed probe batch, and the RMS of
    its zero-sum residual — the violation, measured on the raw network, which
    is the only place it can be seen."""
    if probe is None:
        return float("nan"), float("nan")
    v = net(*probe[:6], probe[7])
    return (float(v.std()),
            float(zero_sum_residual(v, probe[4], probe[5], probe[7]).pow(2).mean().sqrt()))


def value_loss(net, xpub, unit_ids, phi, inv, w, seg, y, nseg, zero_sum_w=0.0):
    # Belief-weighted Huber over every config in the support. Weighting by the
    # belief is what makes the loss match the distribution CFR queries: a config
    # the belief gives 1% to is worth 1% of the gradient.
    v = net(xpub, unit_ids, phi, inv, w, seg, nseg)
    per = F.smooth_l1_loss(v, y, reduction="none", beta=0.5)
    loss = (per * w).sum() / w.sum().clamp(min=1e-6)
    if zero_sum_w:
        # Learn the constraint instead of projecting it on afterwards: one
        # implementation, in the trainer, and the violation stays measurable on
        # the raw network. The projection did the same arithmetic in three
        # places and lost its gate (`runs/zsum`).
        loss = loss + zero_sum_w * zero_sum_residual(v, w, seg, nseg).pow(2).mean()
    return loss


def train_steps(net, opt, buf, steps, batch, rng, device, augment=True,
                recent_mix=0.0, recent_frac=0.2, profile_cuda=False,
                batch_fn=make_batch, zero_sum_w=0.0):
    """Mean value loss over `steps` Adam updates."""
    if len(buf) < batch:
        return float("nan"), {}
    tot = 0.0
    stat = {"sample_s": 0.0, "prepare_s": 0.0, "forward_wall_s": 0.0,
            "backward_wall_s": 0.0, "batch_configs": 0, "steps": steps,
            "gpu_forward_s": 0.0, "gpu_backward_s": 0.0}
    event_pairs = []
    stream = torch.cuda.current_stream(device) if profile_cuda and device.type == "cuda" else None
    for _ in range(steps):
        ts = time.perf_counter()
        sampled = buf.sample(batch, rng, recent_mix, recent_frac)
        stat["sample_s"] += time.perf_counter() - ts
        stat["batch_configs"] += len(sampled[1])
        ts = time.perf_counter()
        parts = batch_fn(sampled, rng, device, augment)
        stat["prepare_s"] += time.perf_counter() - ts
        if stream is not None:
            f0 = torch.cuda.Event(enable_timing=True)
            f1 = torch.cuda.Event(enable_timing=True)
            b1 = torch.cuda.Event(enable_timing=True)
            f0.record(stream)
        ts = time.perf_counter()
        loss = value_loss(net, *parts, zero_sum_w=zero_sum_w)
        tot += loss.detach().item()
        stat["forward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            f1.record(stream)
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
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
    """Lowest id in each hyperthread sibling group: one hardware thread per core."""
    cores = set()
    root = "/sys/devices/system/cpu"
    if not os.path.isdir(root):
        return cores
    for name in os.listdir(root):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        path = os.path.join(root, name, "topology", "thread_siblings_list")
        try:
            text = open(path).read().strip().replace("-", ",")
        except OSError:
            continue
        ids = [int(x) for x in text.split(",") if x]
        if ids:
            cores.add(min(ids))
    return cores


def pin_one_thread_per_core():
    if not hasattr(os, "sched_setaffinity"):
        return
    cores = physical_cpus()
    if not cores:
        return
    os.sched_setaffinity(0, cores)
    print(f"[cpu] pinned to {len(cores)} physical cores", flush=True)


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

    One file, rewritten in place, so `report.py` and `ladder.py` have a single
    thing to read and a run that is still going is readable at any moment.
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
    ap.add_argument("over", nargs="*", help="knob=value (defaults are gpu_golden8)")
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
    pin_one_thread_per_core()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise SystemExit(f"device must be a CUDA device, got {args.device!r}")
    if args.train_gen_ratio <= 0.0:
        raise SystemExit("train_gen_ratio must be positive")
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

    towers = lambda s: [int(x) for x in s.split(",") if x.strip()] or None
    value = Mlp(args.hidden, args.dg, args.rank, args.de,
                head=(args.head or args.hidden),
                pub=towers(args.pub), hmlp=towers(args.hmlp),
                card=towers(args.card), slot=towers(args.slot),
                nres=args.nres).to(dev)
    if args.init_weights:
        initial = load_checkpoint(args.init_weights)
        if list(initial.dims) != list(value.dims):
            raise ValueError(
                f"initial shape {initial.dims} does not match requested shape {value.dims}")
        value.load_state_dict(initial.state_dict())
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    lr_decays = sorted(float(x) for x in args.lr_decay_frac.split(",") if x.strip())
    next_decay = 0
    value.push(0)
    gpu_devices = [int(x) for x in args.gpu_devices.split(",") if x.strip()]
    os.environ.setdefault("WARCHEST_DIRECT", "1")
    os.environ.setdefault("WARCHEST_WAVE_LANES", "3")
    os.environ.setdefault("WARCHEST_WAVE_ROWS", "196608")
    os.environ.setdefault("WARCHEST_WAVE_JOBS", "256")
    os.environ.setdefault("WARCHEST_WAVE_US", "75000")
    dims, w, b, ln = value.dims, *value.flat()
    warchest.gpu_start(dims, w, b, ln, devices=gpu_devices)
    print(f"[gpu] solve services up on {gpu_devices}", flush=True)
    buf = Buffer(args.cap, args.cap * args.cfgs_per_row)
    import gpu_batch
    gpu_batch.warmup(dev)
    batcher = gpu_batch.make_batch

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
        torch.save({"value": value.state_dict(), "spec": value.spec(),
                    "hidden": args.hidden, "head": args.head or args.hidden,
                    "dg": args.dg, "rank": args.rank, "de": args.de, "t": round(el, 1),
                    "label": label, "git": args.git,
                    "search": {"depth": args.depth, "iters": args.iters,
                               "cfr": args.cfr, "warm": 0.0}}, path)
        snaps.append({"label": label, "t": round(el, 1),
                      "file": os.path.basename(path)})
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    def run_gpu_stream():
        """Continuous solve -> replay -> optimizer pipeline for the production
        pure-bootstrap configuration. Generation is backpressured by a bounded
        Rust/Python chunk queue; optimizer work is paid from exact sample debt,
        and immutable GPU weights are published on a fixed step cadence."""
        nonlocal probe, cap_v, next_decay, next_snap, epoch, rebel_solves

        gen = warchest.gpu_stream_start(
            args.seed * 1_000_003 + epoch,
            depth=args.depth, iters=args.iters, explore=args.explore,
            random_draft=args.random_draft, cfr=args.cfr, warm=0.0,
            eval_mix=args.eval_mix, workers=args.gpu_workers,
            actors_per_worker=args.gpu_actors,
            inflight_per_worker=args.gpu_inflight, chunk_solves=args.gpu_chunk)
        deadline = t0 + total
        drain = max(0.0, min(args.gpu_drain_seconds, total - warm))
        stop_at = deadline - drain
        publish_steps = max(1, args.gpu_publish_steps)
        optimizer_rows = 0
        optimizer_steps = 0
        publications = 0
        stopping = False
        done = False
        next_report = time.time() + 10.0
        counter_names = (
            "games", "decisions", "horizon_hits", "node_caps",
            "oversize_routes", "card_exclusive_routes", "exact_fallbacks", "censored_games",
            "dropped", "configs")
        totals = {name: 0 for name in counter_names}
        window = {name: 0 for name in counter_names}
        window.update(rows=0, solves=0, target_n=0, target_sum=0.0,
                      target_sq=0.0, conv_s=0.0, add_s=0.0,
                      train_s=0.0, gpu_wait_s=0.0,
                      loss_sum=0.0, train_steps=0, sample_s=0.0,
                      prepare_s=0.0, forward_wall_s=0.0,
                      backward_wall_s=0.0, gpu_forward_s=0.0,
                      gpu_backward_s=0.0, batch_configs=0)

        def emit_report(now):
            nonlocal probe, epoch
            elapsed = max(now - rebel_t0, 1e-9)
            debt = max(0.0, args.train_gen_ratio * rebel_solves - optimizer_rows)
            credit = optimizer_rows / args.train_gen_ratio
            raw_sps = rebel_solves / elapsed
            balanced_sps = min(rebel_solves, credit) / elapsed
            if probe is None and len(buf) >= 2048:
                probe = batcher(buf.sample(2048, rng), rng, dev, False)
            with torch.no_grad():
                probe_std, probe_zs = probe_stats(value, probe)
                if len(buf) >= args.batch:
                    old_parts = batcher(
                        buf.sample_old(args.batch, rng, args.recent_frac), rng, dev, False)
                    loss_old = float(value_loss(value, *old_parts))
                    new_parts = batcher(
                        buf.sample(args.batch, rng, recent_mix=1.0,
                                   recent_frac=args.recent_frac), rng, dev, False)
                    loss_new = float(value_loss(value, *new_parts))
                else:
                    loss_old = loss_new = float("nan")
            tn = max(window["target_n"], 1)
            tgt_mean = window["target_sum"] / tn
            tgt_var = max(0.0, window["target_sq"] / tn - tgt_mean * tgt_mean)
            dec = max(window["decisions"], 1)
            games = max(window["games"], 1)
            lv = window["loss_sum"] / max(window["train_steps"], 1)
            rec = {
                "t": round(now - t0, 1), "epoch": epoch, "phase": "rebel",
                "games": window["games"], "decisions": window["decisions"],
                "rows": window["rows"], "solves": window["solves"],
                "loss": round(lv, 5),
                "loss_old": round(loss_old, 5), "loss_new": round(loss_new, 5),
                "horizon_frac": round(window["horizon_hits"] / games, 3),
                "node_caps": window["node_caps"],
                "oversize_routes": window["oversize_routes"],
                "card_exclusive_routes": window["card_exclusive_routes"],
                "exact_fallbacks": window["exact_fallbacks"],
                "censored_games": window["censored_games"],
                "dropped": window["dropped"],
                "configs": round(window["configs"] / dec, 1),
                "cap_value": round(cap_v, 4),
                "steps": window["train_steps"],
                "optimizer_steps": optimizer_steps,
                "optimizer_rows": optimizer_rows,
                "optimizer_debt": round(debt, 1),
                "publications": publications,
                "weight_age_steps": optimizer_steps % publish_steps,
                "tgt_mean": round(tgt_mean, 4),
                "tgt_std": round(tgt_var ** 0.5, 4),
                "probe_std": round(probe_std, 4),
                "probe_zs": round(probe_zs, 4),
                "gen_s": round(elapsed, 2),
                "train_s": round(window["train_s"], 2),
                "sample_s": round(window["sample_s"], 2),
                "prepare_s": round(window["prepare_s"], 2),
                "forward_wall_s": round(window["forward_wall_s"], 2),
                "backward_wall_s": round(window["backward_wall_s"], 2),
                "gpu_forward_s": round(window["gpu_forward_s"], 2),
                "gpu_backward_s": round(window["gpu_backward_s"], 2),
                "batch_configs": round(
                    window["batch_configs"] / max(window["train_steps"], 1), 1),
                "target_configs_per_row": round(
                    window["target_n"] / max(window["rows"], 1), 1),
                "conv_s": round(window["conv_s"], 2),
                "add_s": round(window["add_s"], 2),
                "gpu_wait_s": round(window["gpu_wait_s"], 2),
                "buf": len(buf),
                "buf_s": round(buf.span_seconds(), 1),
                "solves_per_s": round(raw_sps, 1),
                "balanced_solves_per_s": round(balanced_sps, 1),
                "lr": opt.param_groups[0]["lr"],
                "deadline_remaining": round(max(0.0, deadline - now), 1),
            }
            log.append(rec)
            write_log(args, log, snaps)
            print(
                f"[t={rec['t']:6.1f}s] rebel stream solves={rebel_solves} "
                f"raw={raw_sps:.0f}/s balanced={balanced_sps:.0f}/s "
                f"debt={debt:.0f} rows steps={optimizer_steps} "
                f"horizon={rec['horizon_frac']:.2f} games={rec['games']} "
                f"over={totals['oversize_routes']} card={totals['card_exclusive_routes']} "
                f"drop={totals['dropped']} "
                f"L={lv:.5f} L/var={lv / max(tgt_var, 1e-9):.2f} "
                f"tgt={tgt_mean:+.3f}/{tgt_var ** 0.5:.3f} "
                f"cfg/b={rec['batch_configs']:.0f} prep={window['prepare_s']:.2f}s "
                f"gpu={window['gpu_forward_s'] + window['gpu_backward_s']:.2f}s "
                f"conv={window['conv_s']:.2f}s add={window['add_s']:.2f}s "
                f"train={window['train_s']:.2f}s",
                flush=True)
            epoch += 1
            for name in counter_names:
                window[name] = 0
            for name in ("rows", "solves", "target_n", "target_sum", "target_sq",
                         "conv_s", "add_s", "train_s", "gpu_wait_s",
                         "loss_sum", "train_steps", "sample_s", "prepare_s",
                         "forward_wall_s", "backward_wall_s", "gpu_forward_s",
                         "gpu_backward_s", "batch_configs"):
                window[name] = 0

        try:
            while True:
                now = time.time()
                if not stopping and now >= stop_at:
                    gen.stop()
                    stopping = True
                    print(f"[t={now - t0:6.1f}s] --- stopping GPU admission; draining ---",
                          flush=True)

                data = None
                if not done:
                    try:
                        data = gen.next(timeout=0.05)
                    except StopIteration:
                        done = True

                if data is not None:
                    tc = time.time()
                    rows = np.asarray(data["rows"], np.uint8).reshape(-1, ROW_BYTES)
                    cc = np.asarray(data["cc"], np.uint8).reshape(-1, CCOUNTS)
                    cw = np.asarray(data["cw"], np.float32)
                    cy = np.clip(np.asarray(data["cy"], np.float32), -1.0, 1.0)
                    coff = np.asarray(data["coff"], np.int64)
                    soff = np.asarray(data["soff"], np.int64)
                    solves = int(data["solves"])
                    window["conv_s"] += time.time() - tc
                    ta = time.time()
                    if len(rows) > 0:
                        buf.add(rows, cc, cw.astype(np.float16), cy.astype(np.float16), coff, soff)
                    window["add_s"] += time.time() - ta
                    rebel_solves += solves
                    window["solves"] += solves
                    window["rows"] += len(rows)
                    window["target_n"] += cy.size
                    window["target_sum"] += float(cy.sum(dtype=np.float64))
                    window["target_sq"] += float(np.square(cy.astype(np.float64)).sum())
                    window["gpu_wait_s"] += float(data.get("gpu_wait_s", 0.0))
                    for name in counter_names:
                        v = int(data.get(name, 0))
                        totals[name] += v
                        window[name] += v

                debt = max(0.0, args.train_gen_ratio * rebel_solves - optimizer_rows)
                if debt >= args.batch and len(buf) >= args.batch:
                    until_publish = publish_steps - optimizer_steps % publish_steps
                    nsteps = min(int(debt // args.batch), until_publish)
                    tt = time.time()
                    lv, train_stat = train_steps(
                        value, opt, buf, nsteps, args.batch, rng, dev,
                        augment=not args.no_augment,
                        recent_mix=args.recent_mix, recent_frac=args.recent_frac,
                        profile_cuda=os.environ.get("WARCHEST_TRAIN_PROFILE") == "1",
                        batch_fn=batcher, zero_sum_w=args.zero_sum_w)
                    window["train_s"] += time.time() - tt
                    window["loss_sum"] += lv * nsteps
                    window["train_steps"] += nsteps
                    for name in ("sample_s", "prepare_s", "forward_wall_s",
                                 "backward_wall_s", "gpu_forward_s", "gpu_backward_s",
                                 "batch_configs"):
                        window[name] += train_stat[name]
                    optimizer_steps += nsteps
                    optimizer_rows += nsteps * args.batch
                    if optimizer_steps % publish_steps == 0:
                        value.push(0)
                        flat = value.flat()
                        for i in range(len(gpu_devices)):
                            warchest.gpu_set_weights(value.dims, *flat, device=i)
                        publications += 1

                now = time.time()
                rebel_elapsed = max(0.0, now - rebel_t0)
                span = max(args.anneal_frac * (total - warm), 1.0)
                cap_v = args.cap_value * max(0.0, 1.0 - rebel_elapsed / span)
                warchest.set_cap_value(cap_v)
                while next_decay < len(lr_decays) and \
                        rebel_elapsed >= lr_decays[next_decay] * (total - warm):
                    for pg in opt.param_groups:
                        pg["lr"] /= 2
                    print(f"[t={now - t0:6.1f}s] --- lr -> {opt.param_groups[0]['lr']:.2e} ---",
                          flush=True)
                    next_decay += 1
                if now - t0 >= next_snap:
                    snapshot(f"s{len(snaps)}", now - t0)
                    next_snap = now - t0 + snap_gap
                if now >= next_report:
                    emit_report(now)
                    next_report = now + 10.0

                debt = max(0.0, args.train_gen_ratio * rebel_solves - optimizer_rows)
                if done and debt < args.batch:
                    break
        finally:
            if not stopping:
                gen.stop()

        now = time.time()
        if any(window[name] for name in ("solves", "games", "decisions", "train_steps")):
            emit_report(now)
        # The run's denominator is the fixed ReBeL wall-clock interval, not an
        # early exit made flattering by a short drain. Usually this is only a
        # few seconds of reserve left after all submitted waves have completed.
        while time.time() < deadline:
            time.sleep(min(0.05, deadline - time.time()))
        elapsed = max(time.time() - rebel_t0, 1e-9)
        debt = max(0.0, args.train_gen_ratio * rebel_solves - optimizer_rows)
        credit = optimizer_rows / args.train_gen_ratio
        raw_sps = rebel_solves / elapsed
        balanced_sps = min(rebel_solves, credit) / elapsed
        print(
            f"[gpu-summary] solves={rebel_solves} optimizer_rows={optimizer_rows} "
            f"debt={debt:.0f} raw={raw_sps:.1f}/s balanced={balanced_sps:.1f}/s "
            f"horizon={totals['horizon_hits'] / max(totals['games'], 1):.2f} "
            f"games={totals['games']} "
            f"over={totals['oversize_routes']} card={totals['card_exclusive_routes']} "
            f"exact={totals['exact_fallbacks']} "
            f"censored={totals['censored_games']} dropped={totals['dropped']} "
            f"overrun={max(0.0, time.time() - deadline):.2f}s",
            flush=True)

    next_snap = float("inf")
    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} hidden={args.hidden} "
          f"head={args.head or args.hidden} dg={args.dg} rank={args.rank} "
          f"depth={args.depth} iters={args.iters} budget={total:.0f}s "
          f"warm={warm:.0f}s snapshot_every={args.snapshot_every:g}min "
          f"device={dev} draft={'random' if args.random_draft else 'starter'} "
          f"train_gen_ratio={args.train_gen_ratio} "
          f"recent_mix={args.recent_mix}/{args.recent_frac} "
          f"augment={not args.no_augment} cap={args.cap} "
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
            probe = batcher(buf.sample(2048, rng), rng, dev, False)
        tgt_mean, tgt_std = float(cy.mean()), float(cy.std())
        tt = time.time()
        steps = max(1, round(args.train_gen_ratio * solves / args.batch))
        lv, _ = train_steps(
            value, opt, buf, steps, args.batch, rng, dev,
            augment=not args.no_augment,
            recent_mix=args.recent_mix, recent_frac=args.recent_frac,
            batch_fn=batcher, zero_sum_w=args.zero_sum_w)
        train_s = time.time() - tt
        value.push(0)
        with torch.no_grad():
            probe_std, probe_zs = probe_stats(value, probe)
            if len(buf) >= args.batch:
                old_parts = batcher(
                    buf.sample_old(args.batch, rng, args.recent_frac), rng, dev, False)
                loss_old = float(value_loss(value, *old_parts))
                new_parts = batcher(
                    buf.sample(args.batch, rng, recent_mix=1.0,
                               recent_frac=args.recent_frac), rng, dev, False)
                loss_new = float(value_loss(value, *new_parts))
            else:
                loss_old = loss_new = float("nan")
        dec = max(d["decisions"], 1)
        rec = {"t": round(time.time() - t0, 1), "epoch": epoch, "phase": "greedy",
               "games": d["games"], "decisions": dec, "loss": round(lv, 5),
               "rows": len(rows), "solves": solves,
               "loss_old": round(loss_old, 5), "loss_new": round(loss_new, 5),
               "horizon_frac": round(d["horizon_hits"] / max(d["games"], 1), 3),
               "node_caps": int(d["node_caps"]),
               "oversize_routes": int(d.get("oversize_routes", 0)),
               "card_exclusive_routes": int(d.get("card_exclusive_routes", 0)),
               "exact_fallbacks": int(d.get("exact_fallbacks", 0)),
               "dropped": int(d["dropped"]),
               "configs": round(d["configs"] / dec, 1), "cap_value": round(cap_v, 4),
               "steps": steps,
               "tgt_mean": round(tgt_mean, 4), "tgt_std": round(tgt_std, 4),
               "probe_std": round(probe_std, 4),
               "probe_zs": round(probe_zs, 4),
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
    flat = value.flat()
    for i in range(len(gpu_devices)):
        warchest.gpu_set_weights(value.dims, *flat, device=i)
    next_snap = (time.time() - t0) + snap_gap
    # Warm-up initialises the network. Its Adam moments and its rows are a
    # different objective; they must not steer ReBeL.
    buf.clear()
    opt = torch.optim.Adam(value.parameters(), lr=args.lr)
    rebel_t0 = time.time()
    rebel_solves = 0
    print(f"[t={time.time() - t0:6.1f}s] --- switching to ReBeL ---", flush=True)
    run_gpu_stream()

    snapshot("final", time.time() - t0)
    write_log(args, log, snaps)
    import ladder
    import report
    ladder.run([args.out], games=args.ladder_games, gpu=True)
    report.write([args.out])


if __name__ == "__main__":
    main()
