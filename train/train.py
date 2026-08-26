"""Student of Games training for War Chest.

Self-play where every decision grows a search tree along trajectories sampled
from its changing strategy. The root of each solve becomes a value target --
one value per config in each player's belief -- and the leaves it queried
become roots for solves of their own, which is how the value function becomes
accurate away from the line of play.

There is a short greedy warm start. A game that reaches the play cap scores
`cap_value * delta_markers`, the win condition graded rather than an invented
evaluation. The payoff fades with the fraction of finished games that hit the
horizon; evaluation always runs on the real game.

Generation runs in Rust across all CPU cores while the previous batch trains
on the GPU. Python publishes weights between generation batches.

A training row is a public state plus, for each player, the whole belief: the
exact configs in support, their probabilities, and the value the solve gave
each. The config lists are ragged, so they live in a flat arena and a batch is
assembled by gathering spans -- see `Buffer`.

The run snapshots every `snapshot_every` minutes. Training starts with a
short greedy warm phase labelled by the public static evaluation, then the
SoG phase. When training ends, the snapshots play a ladder and a report is
written.

    python train/train.py out=seat
    python train/train.py out=seat note="centre the seat bit at +-0.5"
    python train/train.py out=smoke minutes=6
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
from export_weights import load as load_checkpoint
from gpu_batch import make_batch, warmup
from replay import (Buffer, ROW_COLUMNS, SOURCE_COUNT, SOURCE_PLAY,
                    SOURCE_QUERY, SOURCE_WARM)
from value_net import Net

ROOT = pathlib.Path(__file__).resolve().parent.parent

PUBFEAT = warchest.PUBFEAT
CFEAT = warchest.CFEAT
CCOUNTS = warchest.CCOUNTS
CNORM = warchest.CNORM
ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES
NSLOT = warchest.NSLOT

# What `SolveFarm.collect` reports about the device rounds, cumulative
# since the farm started. `rounds` is the denominator of the other three.
ROUND_KEYS = ("rounds", "round_calls", "round_rows", "round_nanos", "budget_hits")


def forward_values(net, parts):
    # `nseg` is the sixth element, not the last one: a batch carries the policy
    # target after it.
    return net(*parts[:4], parts[5])


def losses(net, xpub, phi, w, seg, y, nseg, policy=None, wp=1.0):
    """Return the value-and-policy loss and its device-side measurements."""
    stats = {}
    v, board, heads, policy_config = net.forward_parts(xpub, phi, w, seg, nseg)
    expected = torch.zeros(nseg, dtype=v.dtype, device=v.device)
    expected.index_add_(0, seg, v.detach() * w)
    residual = expected[0::2] + expected[1::2]
    # Device scalars: `train_steps` reads them back once a call, so the step
    # loop itself never synchronizes.
    stats["zero_sum_max"] = residual.abs().max()
    stats["zero_sum_square_sum"] = residual.square().sum()
    stats["zero_sum_n"] = len(residual)
    per = F.smooth_l1_loss(v, y, reduction="none", beta=0.5)
    total = torch.zeros(nseg, dtype=per.dtype, device=per.device)
    count = torch.zeros(nseg, dtype=per.dtype, device=per.device)
    total.index_add_(0, seg, per)
    count.index_add_(0, seg, torch.ones_like(per))
    loss = (total / count).mean()
    # `L` and `L/var` are the *value* loss, as they were before the policy head
    # existed, so the run report still compares with every run before it. The
    # policy term is reported beside them, never folded into them.
    stats["value_loss"] = loss.detach()
    if policy is not None and wp > 0.0:
        policy_value, policy_stats = policy_loss(
            net, xpub, w, seg, nseg, policy, board, heads, policy_config)
        stats.update(policy_stats)
        if policy_value is not None:
            loss = loss + wp * policy_value
    return loss, stats


def policy_loss(net, xpub, weight, seg, nseg, policy, board, heads,
                policy_config):
    """Cross entropy of the policy head against the search's root average.

    The head scores a `(config, action)` cell as `<f_p(c), e(a)>`, so the batch
    is exactly the cells the solves stored. Each cell's softmax runs over its
    own `(row, config)` group, which is one information state. ``board``,
    ``heads``, and ``policy_config`` come from the value path, so shared network
    work runs once per step.
    """
    stats = {}
    feat, parow, pact, pcfg, group, target, _group_count = policy
    if feat.shape[0] == 0 or pact.shape[0] == 0:
        return None, stats
    h = heads
    fp = policy_config
    action_query = torch.zeros(feat.shape[0], dtype=torch.long, device=feat.device)
    action_query.scatter_(0, pact, seg[pcfg])
    e = net.actions(feat, board, h, parow, action_query)

    # `pcfg` is an index into the batch's own config arena, so the cell reads
    # its config vector directly.
    logit = (fp[pcfg] * e[pact]).sum(1)

    # The group ids are compact and sorted, but their count changes with every
    # sampled batch. Reduce into the cell-sized arena and mask its unused tail;
    # no Python integer controls a compiled tensor shape.
    slots = torch.arange(logit.shape[0], device=logit.device)
    used = slots <= group[-1]
    groups = used.sum()
    top = torch.full_like(logit, -1e30)
    top.scatter_reduce_(0, group, logit, reduce="amax")
    ex = (logit - top[group]).exp()
    tot = torch.zeros_like(logit).index_add_(0, group, ex)
    logp = (logit - top[group]) - tot[group].clamp(min=1e-30).log()
    per = -(target * logp)
    out = torch.zeros_like(logit).index_add_(0, group, per)
    loss = (out * used).sum() / groups
    target_mass = torch.zeros_like(target).index_add_(0, group, target)
    q = target / target_mass[group].clamp(min=1e-30)
    target_entropy = torch.zeros_like(target).index_add_(
        0, group, -(q * q.clamp(min=1e-30).log()))
    prior = logp.exp()
    prior_entropy = torch.zeros_like(target).index_add_(
        0, group, -(prior * logp))
    search_ce = torch.zeros_like(target).index_add_(
        0, group, -(q * logp))
    for key, value in zip((
            "policy_loss", "policy_target_entropy", "policy_prior_entropy",
            "policy_search_kl"), (
            loss, (target_entropy * used).sum() / groups,
            (prior_entropy * used).sum() / groups,
            ((search_ce - target_entropy) * used).sum() / groups)):
        stats[key] = value.detach()
    return loss, stats


@torch.no_grad()
def diagnostics(net, buf, probe, batch, rng, device, recent_frac):
    """Held replay error, prediction spread, and outcome calibration."""
    nan = float("nan")
    out = {
        "probe_std": float(forward_values(net, probe).std()) if probe is not None else nan,
        "loss_old": nan,
        "loss_new": nan,
        "value_outcome_rmse": nan,
        "value_outcome_mae": nan,
        "value_outcome_bias": nan,
        "value_outcome_corr": nan,
        "value_calibration_slope": nan,
    }
    if len(buf) < batch:
        return out
    old = make_batch(buf.sample_old(batch, rng, recent_frac), device)
    new = make_batch(buf.sample(batch, rng, recent_mix=1.0, recent_frac=recent_frac),
                     device)
    out["loss_old"] = float(losses(net, *old, wp=0.0)[0])
    out["loss_new"] = float(losses(net, *new, wp=0.0)[0])
    calibration = buf.sample_calibration(batch, rng)
    if calibration is None:
        return out
    sampled, at, outcome = calibration
    parts = make_batch(sampled, device)
    pred = forward_values(net, parts)[torch.as_tensor(at, device=device)].float().cpu().numpy()
    error = pred - outcome
    pc = pred - pred.mean()
    oc = outcome - outcome.mean()
    cov = float(np.mean(pc * oc))
    out.update({
        "value_outcome_rmse": float(np.sqrt(np.mean(error * error))),
        "value_outcome_mae": float(np.mean(np.abs(error))),
        "value_outcome_bias": float(error.mean()),
        "value_outcome_corr": cov / max(float(pc.std() * oc.std()), 1e-12),
        "value_calibration_slope": cov / max(float(np.mean(pc * pc)), 1e-12),
    })
    return out


def train_steps(net, opt, buf, steps, batch, rng, device,
                recent_mix=0.0, recent_frac=0.2, profile_cuda=False,
                policy_w=0.0, *, deadline, loss_fn=losses):
    """Mean loss over up to `steps` Adam updates -- value, plus the policy
    head's cross entropy at weight `policy_w` -- stopping between updates at
    `deadline`. `loss_fn` is `losses`, or its torch.compile form on the box:
    the forward and its backward become one fused graph, which is where most
    of a step's device time went."""
    policy_metrics = (
        "policy_loss", "policy_target_entropy", "policy_prior_entropy",
        "policy_search_kl")
    z = lambda: torch.zeros((), device=device)
    stat = {"sample_s": 0.0, "prepare_s": 0.0, "forward_wall_s": 0.0,
            "backward_wall_s": 0.0, "batch_configs": 0, "steps": 0,
            "enqueue_s": 0.0,
            "gpu_forward_s": 0.0, "gpu_backward_s": 0.0,
            "zero_sum_max": z(), "zero_sum_square_sum": z(),
            "zero_sum_n": 0, "grad_clipped": 0,
            "grad_norm_sum": z(), "grad_norm_max": z(),
            "policy_steps": 0, "sample_ages": [], "sample_delays": [],
            "sample_sources": [],
            "sample_warm": 0, "sample_play": 0, "sample_query": 0,
            "sample_td1_targets": 0, "sample_targets": 0,
            **{f"{key}_sum": z() for key in policy_metrics}}
    if len(buf) < batch:
        return float("nan"), stat
    enqueue_t = time.perf_counter()
    tot = 0.0
    event_pairs = []
    stream = torch.cuda.current_stream(device) if profile_cuda and device.type == "cuda" else None
    for _ in range(steps):
        if time.time() >= deadline:
            break
        ts = time.perf_counter()
        ids = buf.sample_ids(batch, rng, recent_mix, recent_frac)
        ring = ids % buf.cap
        stat["sample_ages"].append(time.time() - buf.created_at[ring])
        delay = buf.written_at[ring] - buf.created_at[ring]
        stat["sample_delays"].append(delay)
        source_id = buf.source[ring]
        source = np.bincount(source_id, minlength=SOURCE_COUNT)
        stat["sample_sources"].append(source_id.copy())
        stat["sample_warm"] += int(source[SOURCE_WARM])
        stat["sample_play"] += int(source[SOURCE_PLAY])
        stat["sample_query"] += int(source[SOURCE_QUERY])
        stat["sample_td1_targets"] += 2 * int(buf.td1[ring].sum())
        stat["sample_targets"] += int(buf.clen[ring].sum())
        sampled = buf.gather(ids)
        stat["sample_s"] += time.perf_counter() - ts
        stat["batch_configs"] += len(sampled[1])
        ts = time.perf_counter()
        parts = make_batch(sampled, device)
        stat["prepare_s"] += time.perf_counter() - ts
        if stream is not None:
            f0 = torch.cuda.Event(enable_timing=True)
            f1 = torch.cuda.Event(enable_timing=True)
            b1 = torch.cuda.Event(enable_timing=True)
            f0.record(stream)
        ts = time.perf_counter()
        value, loss_stats = loss_fn(net, *parts, wp=policy_w)
        tot += loss_stats["value_loss"]
        stat["zero_sum_max"] = torch.maximum(
            stat["zero_sum_max"], loss_stats["zero_sum_max"])
        stat["zero_sum_square_sum"] = stat["zero_sum_square_sum"] \
            + loss_stats["zero_sum_square_sum"]
        stat["zero_sum_n"] += loss_stats["zero_sum_n"]
        if "policy_loss" in loss_stats:
            stat["policy_steps"] += 1
            for key in policy_metrics:
                stat[f"{key}_sum"] = stat[f"{key}_sum"] + loss_stats[key]
        stat["forward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            f1.record(stream)
        ts = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        value.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        stat["grad_norm_sum"] = stat["grad_norm_sum"] + grad_norm
        stat["grad_norm_max"] = torch.maximum(stat["grad_norm_max"], grad_norm)
        stat["grad_clipped"] = stat["grad_clipped"] + (grad_norm > 5.0).long()
        opt.step()
        stat["steps"] += 1
        stat["backward_wall_s"] += time.perf_counter() - ts
        if stream is not None:
            b1.record(stream)
            event_pairs.append((f0, f1, b1))
    if stat["steps"]:
        stat["enqueue_s"] = time.perf_counter() - enqueue_t
        # The loop above never synchronizes, so the device and the host stay
        # overlapped; one readback a call drains it.
        cpu = lambda t: float(t.detach().cpu()) if torch.is_tensor(t) else float(t)
        for key in ("zero_sum_max", "zero_sum_square_sum", "zero_sum_n",
                    "grad_clipped", "grad_norm_sum", "grad_norm_max",
                    *(f"{k}_sum" for k in policy_metrics)):
            stat[key] = cpu(stat[key])
        tot = cpu(tot)
    if event_pairs:
        torch.cuda.synchronize(device)
        stat["gpu_forward_s"] = sum(a.elapsed_time(b) for a, b, _ in event_pairs) / 1000.0
        stat["gpu_backward_s"] = sum(b.elapsed_time(c) for _, b, c in event_pairs) / 1000.0
    if profile_cuda and stat["steps"]:
        n = stat["steps"]
        print(f"[profile] steps={n} rows={batch}"
              f" step_enqueue={1e3 * stat['enqueue_s'] / n:.1f}ms"
              f" sample_enqueue={1e3 * stat['sample_s'] / n:.1f}ms"
              f" prepare_enqueue={1e3 * stat['prepare_s'] / n:.1f}ms"
              f" fwd_enqueue={1e3 * stat['forward_wall_s'] / n:.1f}ms"
              f" bwd_enqueue={1e3 * stat['backward_wall_s'] / n:.1f}ms"
              f" fwd_event={1e3 * stat['gpu_forward_s'] / n:.1f}ms"
              f" bwd_event={1e3 * stat['gpu_backward_s'] / n:.1f}ms",
              flush=True)
    return tot / stat["steps"] if stat["steps"] else float("nan"), stat


def ingest(buf, data):
    """Append one `gen_data` / farm collect dict onto the replay buffer."""
    x = np.asarray(data["rows"], np.uint8).reshape(-1, ROW_BYTES)
    if not len(x):
        return 0
    cols = {}
    for name, key, dtype, shape in ROW_COLUMNS:
        value = np.asarray(data[key], dtype)
        cols[name] = value.reshape((-1, *shape)) if shape else value
    cc = np.asarray(data["cc"], np.uint8).reshape(-1, CCOUNTS)
    cw = np.asarray(data["cw"], np.float32)
    cy = np.clip(np.asarray(data["cy"], np.float32), -1.0, 1.0)
    coff = np.asarray(data["coff"], np.int64)
    soff = np.asarray(data["soff"], np.int64)
    pol = (np.asarray(data["pa"], np.uint8).reshape(-1, ACT_BYTES),
           np.asarray(data["paoff"], np.int64),
           np.asarray(data["pcoff"], np.int64),
           np.asarray(data["pci"], np.uint16),
           np.asarray(data["pcell"], np.uint16),
           np.asarray(data["pprob"], np.float16))
    buf.add(x, cols, cc, cw.astype(np.float16), cy.astype(np.float16), coff, soff,
            pol)
    return len(x)


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
    cfg = dataclasses.asdict(args)
    cfg["resume"] = ""
    with open(tmp, "w") as f:
        json.dump({"cfg": cfg, "epochs": epochs,
                   "snapshots": snaps}, f, indent=1)
    os.replace(tmp, path)


def cpu_state(net):
    return {name: value.detach().cpu().clone()
            for name, value in net.state_dict().items()}


def publish_state(state):
    net = Net()
    net.load_state_dict(state)
    net.push()


def compiled_loss():
    """Compile the measured loss path, with eager as the safe fallback."""
    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(losses, mode="default", dynamic=True)
    except Exception as error:
        print(f"[train] torch.compile unavailable; using eager: "
              f"{type(error).__name__}: {error}", flush=True)
        return losses

    failed = False
    last_counts = None

    def run(*args, **kwargs):
        nonlocal failed, last_counts
        if failed:
            return losses(*args, **kwargs)
        try:
            result = compiled(*args, **kwargs)
            counters = torch._dynamo.utils.counters
            counts = (counters["stats"].get("unique_graphs", 0),
                      counters["frames"].get("total", 0))
            if counts != last_counts:
                print(f"[train] torch.compile unique_graphs={counts[0]} "
                      f"frames={counts[1]}", flush=True)
                last_counts = counts
            return result
        except Exception as error:
            failed = True
            print(f"[train] torch.compile failed; using eager: "
                  f"{type(error).__name__}: {error}", flush=True)
            return losses(*args, **kwargs)

    return run


def main():
    ap = argparse.ArgumentParser(
        description="Train one run, then rate its snapshots against Greedy.")
    ap.add_argument("over", nargs="*", help="knob=value (production defaults)")
    over = config.parse(ap.parse_args().over)
    resume = over.pop("resume", "")
    name = over.pop("out", None)
    checkpoint = None
    if resume:
        if over:
            raise SystemExit("resume accepts no changed training knobs")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        args = config.Cfg(**checkpoint["cfg"])
        expected = pathlib.Path(args.out)
        requested = pathlib.Path(name if name and name.startswith("runs/")
                                 else f"runs/{name}") if name else expected
        if requested != expected:
            raise SystemExit(f"resume belongs to {expected}, not {requested}")
        args.resume = resume
    else:
        if not name:
            raise SystemExit("pass out=<name>")
        args = dataclasses.replace(config.BASELINE, **over)
        args.git = config.git_sha()
        args.out = name if name.startswith("runs/") else f"runs/{name}"
    refuse_if_machine_busy()
    if resume:
        if not os.path.isdir(args.out):
            raise SystemExit(f"resume output {args.out} does not exist")
    else:
        if os.path.exists(args.out):
            raise SystemExit(f"{args.out} exists")
        os.makedirs(args.out)
    logf = open(f"{args.out}/train.log", "a" if resume else "w")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logf.write(s)
            return len(s)
        def flush(self):
            sys.__stdout__.flush()
            logf.flush()
    sys.stdout = sys.stderr = Tee()
    if resume:
        print(f"[resume] {resume}", flush=True)
    else:
        print(f"[train] {args.out} at {args.git} seed={args.seed} "
              f"{over or 'baseline'}", flush=True)
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
    diag_rng = np.random.default_rng(args.seed ^ 0xD1A6_0571)
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise SystemExit(f"device must be a CUDA device, got {args.device!r}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; training requires a working GPU")
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

    torch.cuda.reset_peak_memory_stats(dev)
    value = Net().to(dev)
    if args.init_weights:
        value.load_state_dict(load_checkpoint(args.init_weights).state_dict())
    opt = torch.optim.Adam(value.parameters(), lr=args.lr, fused=True)
    lr_decays = sorted(float(x) for x in args.lr_decay_frac.split(",") if x.strip())
    value.push()
    target_state = cpu_state(value)
    buf = Buffer(args.cap, args.cap * args.cfgs_per_row, dev)
    warmup(dev)
    # One training step and one probe at a training-sized batch, so the
    # caching allocator holds that peak before the farm carves. Configs in a
    # batch are `rows × cfgs_per_row`. The solve slot's config cap is not a
    # batch width: at s=512 it is thousands, and a dummy that wide does not
    # fit next to the intern table.
    # 2048 is the intern-table floor: a 2048× table does not fit the encoder.
    n = max(args.batch, 2048)
    k = n * args.cfgs_per_row
    x = torch.zeros(2 * n, PUBFEAT, device=dev)
    phi = torch.zeros(k, CFEAT, device=dev)
    w = torch.ones(k, device=dev)
    seg = torch.arange(k, device=dev) % (2 * n)
    y = torch.zeros(k, device=dev)
    parts = (x, phi, w, seg, y, 2 * n, None)
    opt.zero_grad(set_to_none=True)
    losses(value, *parts, wp=0.0)[0].backward()
    opt.step()
    forward_values(value, parts)
    torch.cuda.synchronize(dev)
    # The replay and batch paths are measured. Default-mode compile is now the
    # measured loss path; a graph failure falls back to eager in compiled_loss.
    step_loss = compiled_loss()
    if checkpoint:
        value.load_state_dict(checkpoint["value"])
        opt.load_state_dict(checkpoint["optimizer"])
        target_state = checkpoint["target"]
        publish_state(target_state)
        rng.bit_generator.state = checkpoint["numpy_rng"]
        diag_rng.bit_generator.state = checkpoint["diag_numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    peak = torch.cuda.max_memory_reserved(dev)
    print(f"[train] torch+replay peak {peak / (1 << 20):.0f} MiB reserved on {dev} "
          f"(rows={n} configs={k}); farm carves mem_get_info free",
          flush=True)
    print(f"[train] search inference on cuda:{args.gen_devices}, "
          f"training on {dev}", flush=True)

    total = args.minutes * 60.0
    if args.snapshot_every <= 0:
        raise SystemExit("snapshot_every must be positive minutes")
    snap_gap = args.snapshot_every * 60.0
    elapsed = float(checkpoint["elapsed"]) if checkpoint else 0.0
    next_snap = float(checkpoint["next_snapshot"]) if checkpoint else snap_gap
    t0 = time.time() - elapsed
    epoch = int(checkpoint["epoch"]) if checkpoint else 0
    log = checkpoint["epochs"] if checkpoint else []
    # Fresh subgames per second over the whole ReBeL phase: the rate
    # docs/GPU_PERF_GOAL.md is about. Generation overlaps training, so
    # per-epoch `gen_s` is not it -- only cumulative solves over cumulative
    # ReBeL wall time counts every cost, including the trainer's own.
    progress = checkpoint["progress"] if checkpoint else {
        "sog_start": None,
        "sog_solves": 0,
        "optimizer_rows": 0,
        "optimizer_steps": 0,
        "generated_rows": 0,
        "next_target": None,
        "next_decay": 0,
        "farm_runs": 0,
        "totals": {},
    }
    sog_t0 = (t0 + progress["sog_start"]
              if progress["sog_start"] is not None else None)
    sog_solves = int(progress["sog_solves"])
    next_decay = int(progress["next_decay"])
    # The marker-differential payoff at the horizon distorts the game being
    # solved, so it tracks the recent fraction of finished games that hit the
    # horizon. Evaluation always runs on the real game (value 0).
    cap_v = float(checkpoint["cap_value"]) if checkpoint else args.cap_value
    warchest.set_cap_value(cap_v)
    probe = None

    # Snapshots. Nothing selects between them during the run. Bootstrapped value
    # learning is not monotone, so there is a real question about which weights
    # are best -- but a match large enough to answer it costs minutes of the
    # budget (300 paired games: standard error 0.029, about the size of the gap
    # between neighbouring snapshots), and answering it from a noisy match is
    # how you ship a checkpoint chosen by a coin flip. The ladder rates all of
    # them at the end, off the clock.
    snaps = checkpoint["snapshots"] if checkpoint else []
    saved_replay_rows = int(checkpoint["replay_rows"]) if checkpoint else 0
    grace_rows = (min(saved_replay_rows, 2 * args.batch)
                  if checkpoint else 0)
    if checkpoint:
        gib = args.cap * (ROW_BYTES + 40 + args.cfgs_per_row * 20
                          + 24 * ACT_BYTES + 96 * 6 + 34) / (1 << 30)
        print(f"[resume] replay is intentionally not persisted: its default-cap "
              f"snapshot can reach {gib:.2f} GiB "
              f"({gib * 1440 / args.snapshot_every:.0f} GiB/day at "
              f"{args.snapshot_every:g}-minute cadence)", flush=True)
        print(f"[resume] model, optimizer, target network, LR position, RNG, "
              f"and {elapsed:.1f}s wall budget restored; training waits for "
              f"{grace_rows} fresh replay rows", flush=True)

    def snapshot(label, el):
        # "init" and "final" are the two the reader always wants named; the rest
        # are numbered, and the manifest carries the time each was taken at.
        path = f"{args.out}/snap_{len(snaps):02d}.pt"
        entry = {"label": label, "t": round(el, 1),
                 "file": os.path.basename(path)}
        snaps.append(entry)
        cfg = dataclasses.asdict(args)
        cfg["resume"] = ""
        state = {
            "value": value.state_dict(),
            "optimizer": opt.state_dict(),
            "target": target_state,
            "numpy_rng": rng.bit_generator.state,
            "diag_numpy_rng": diag_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "elapsed": float(el),
            "next_snapshot": float(el + snap_gap),
            "epoch": epoch,
            "epochs": log,
            "snapshots": snaps,
            "progress": progress,
            "cap_value": cap_v,
            "replay_rows": len(buf),
            "cfg": cfg,
            "t": round(el, 1),
            "label": label,
            "git": args.git,
            "search": {"s": args.s, "c": args.c, "cfr": args.cfr},
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    def tick(rec=None, line=None):
        """Log one epoch if given, then snapshot if due. Both generators use this."""
        nonlocal epoch, next_snap
        if rec is not None:
            rec.setdefault("t", round(time.time() - t0, 1))
            rec.setdefault("epoch", epoch)
            rec.setdefault("buf", len(buf))
            rec.setdefault("lr", opt.param_groups[0]["lr"])
            log.append(rec)
            write_log(args, log, snaps)
            print(line, flush=True)
        epoch += 1
        now = time.time()
        if now - t0 >= next_snap:
            snapshot(f"s{len(snaps)}", now - t0)
            next_snap = now - t0 + snap_gap

    def fit(nsteps, deadline):
        """Adam updates on the shared replay. Returns (loss, enqueue_s, stats)."""
        if nsteps < 1 or len(buf) < args.batch:
            return float("nan"), 0.0, {}
        lv, st = train_steps(
            value, opt, buf, nsteps, args.batch, rng, dev,
            recent_mix=args.recent_mix, recent_frac=args.recent_frac,
            profile_cuda=os.environ.get("WARCHEST_TRAIN_PROFILE") == "1",
            policy_w=args.policy_w, deadline=deadline,
            loss_fn=step_loss)
        return lv, st["enqueue_s"], st

    def run_search_pipeline():
        """Overlap small GT-CFR batches with each other and with training."""
        nonlocal probe, cap_v, next_decay, sog_solves, target_state

        deadline = t0 + total
        if time.time() >= deadline:
            return
        # One process, one driver a card and a worker per core. A process per
        # core could not batch inference at all: the solves have to share an
        # address space for their leaf rows to end up in one batch.
        farm = warchest.SolveFarm(
            args.seed + 1_000_003 * progress["farm_runs"],
            args.gen_workers,
            s=args.s,
            c=args.c,
            batch=args.round_batch,
            rounds=args.rounds,
            explore=args.explore,
            random_draft=args.random_draft,
            cfr=args.cfr,
            p_td1=args.p_td1,
            query_rate=args.query_rate,
            recursive_rate=args.recursive_rate,
            devices=[int(d) for d in args.gen_devices.split(",")])

        progress["farm_runs"] += 1
        optimizer_rows = int(progress["optimizer_rows"])
        optimizer_steps = int(progress["optimizer_steps"])
        generated_rows = int(progress["generated_rows"])
        window = collections.Counter()
        totals = collections.Counter(progress["totals"])
        window_shapes = []
        window_targets = []
        window_target_weights = []
        window_sample_ages = []
        window_sample_delays = []
        window_sample_sources = []
        # The farm's round counters are cumulative, so an epoch's figures are
        # the difference against the last report and not an average since the
        # farm started.
        round_at = dict.fromkeys(ROUND_KEYS, 0)
        stage_names = warchest.stage_names()
        ent_at = [0] * 8
        next_report = time.time() + 10.0
        next_target = t0 + float(progress["next_target"])
        grace_report = time.time()

        def save_progress():
            progress.update({
                "sog_solves": sog_solves,
                "optimizer_rows": optimizer_rows,
                "optimizer_steps": optimizer_steps,
                "generated_rows": generated_rows,
                "next_target": next_target - t0,
                "next_decay": next_decay,
                "totals": dict(totals),
            })

        while True:
            now = time.time()
            if now >= deadline:
                break
            gen_t = time.time()
            data = farm.collect(args.gen_solves)
            gen_s = time.time() - gen_t

            ta = time.time()
            n = ingest(buf, data)
            add_s = time.time() - ta
            cy = np.clip(np.asarray(data["cy"], np.float32), -1.0, 1.0)
            cw = np.asarray(data["cw"], np.float32)
            window_targets.append(cy)
            window_target_weights.append(cw)

            solves = int(data["solves"])
            sog_solves += solves
            generated_rows += n
            window["results"] += 1
            window["rows"] += n
            window["solves"] += solves
            window["target_n"] += cy.size
            window["target_sum"] += float(cy.sum(dtype=np.float64))
            window["target_square_sum"] += float(
                np.square(cy.astype(np.float64)).sum())
            window["gen_s"] += gen_s
            window["add_s"] += add_s
            window_shapes.extend(data.get("shapes") or [])
            for name in (
                    "games", "decisions", "horizon_hits",
                    "white_wins", "black_wins", "draws",
                    "plays_attack", "plays_pass", "plays_deploy",
                    "plays_bolster", "plays_maneuver", "plays_recruit",
                    "plays_claim_initiative", "configs", "query_rows"):
                amount = int(data.get(name, 0))
                totals[name] += amount
                window[name] += amount
            # Stay at the full payoff until a game finishes; then scale by the
            # fraction of this window that still died at the play cap.
            if window["games"]:
                cap_v = args.cap_value * (window["horizon_hits"] / window["games"])
                warchest.set_cap_value(cap_v)

            debt = max(0.0, args.replay_ratio * generated_rows - optimizer_rows)
            regenerating = len(buf) < grace_rows
            nsteps = (int(debt // args.batch)
                      if len(buf) >= args.batch and not regenerating else 0)
            lv, train_s, train_stat = fit(nsteps, deadline)
            trained = train_stat.get("steps", 0)
            if trained:
                optimizer_steps += trained
                optimizer_rows += trained * args.batch
                window["loss_sum"] += lv * trained
                window["train_steps"] += trained
                window["policy_steps"] += train_stat["policy_steps"]
                for key in (
                        "policy_loss", "policy_target_entropy", "policy_prior_entropy",
                        "policy_search_kl"):
                    window[f"{key}_sum"] += train_stat[f"{key}_sum"]
                window["batch_configs"] += train_stat["batch_configs"]
                window["gpu_forward_s"] += train_stat["gpu_forward_s"]
                window["gpu_backward_s"] += train_stat["gpu_backward_s"]
                window["grad_clipped"] += train_stat["grad_clipped"]
                window["grad_norm_sum"] += train_stat["grad_norm_sum"]
                window["grad_norm_max"] = max(
                    window["grad_norm_max"], train_stat["grad_norm_max"])
                for key in ("sample_warm", "sample_play", "sample_query",
                            "sample_td1_targets", "sample_targets"):
                    window[key] += train_stat[key]
                window_sample_ages.extend(train_stat["sample_ages"])
                window_sample_delays.extend(train_stat["sample_delays"])
                window_sample_sources.extend(train_stat["sample_sources"])
                window["zero_sum_max"] = max(
                    window["zero_sum_max"], train_stat["zero_sum_max"])
                window["zero_sum_square_sum"] += train_stat["zero_sum_square_sum"]
                window["zero_sum_n"] += train_stat["zero_sum_n"]
            window["train_s"] += train_s

            now = time.time()
            if now >= deadline:
                save_progress()
                break
            if now >= next_target:
                # `push` bumps the version the farm watches; its threads
                # pick the new weights up at their next chunk.
                value.push()
                target_state = cpu_state(value)
                print(
                    f"[t={now - t0:6.1f}s] --- target network refresh ---",
                    flush=True)
                while next_target <= now:
                    next_target += args.target_every * 60.0
            sog_elapsed = max(0.0, now - sog_t0)
            while next_decay < len(lr_decays) and \
                    sog_elapsed >= lr_decays[next_decay] * total:
                for pg in opt.param_groups:
                    pg["lr"] /= 2
                print(
                    f"[t={now - t0:6.1f}s] --- lr -> "
                    f"{opt.param_groups[0]['lr']:.2e} ---",
                    flush=True)
                next_decay += 1
            save_progress()
            if regenerating:
                if now >= grace_report:
                    print(f"[t={now - t0:6.1f}s] --- resume replay "
                          f"{len(buf)}/{grace_rows}; training paused ---",
                          flush=True)
                    grace_report = now + 10.0
                tick()
                continue
            if now < next_report:
                tick()
                continue
            next_report = now + 10.0
            steps = int(window["train_steps"])
            lv = window["loss_sum"] / max(steps, 1)
            if probe is None and len(buf) >= 2048:
                probe = make_batch(buf.sample(2048, diag_rng), dev)
            diag = diagnostics(value, buf, probe, args.batch, diag_rng, dev,
                               args.recent_frac)
            target_n = max(int(window["target_n"]), 1)
            target_mean = window["target_sum"] / target_n
            target_var = max(
                0.0,
                window["target_square_sum"] / target_n
                - target_mean * target_mean)
            targets = np.concatenate(window_targets)
            target_weights = np.concatenate(window_target_weights)
            weight_mass = max(float(target_weights.sum()), 1e-12)
            belief_mean = float(np.dot(targets, target_weights) / weight_mass)
            belief_var = max(float(np.dot((targets - belief_mean) ** 2,
                                          target_weights) / weight_mass), 0.0)
            target_q = np.quantile(targets, [0.05, 0.5, 0.95])
            sample_ages = (np.concatenate(window_sample_ages)
                           if window_sample_ages else np.zeros(1))
            sample_delays = (np.concatenate(window_sample_delays)
                             if window_sample_delays else np.zeros(1))
            sample_sources = (np.concatenate(window_sample_sources)
                              if window_sample_sources else np.zeros(1, np.uint8))
            sample_n = max(window["sample_warm"] + window["sample_play"]
                           + window["sample_query"], 1)
            policy_steps = max(int(window["policy_steps"]), 1)
            policy = {
                key: window[f"{key}_sum"] / policy_steps
                for key in (
                    "policy_loss", "policy_target_entropy", "policy_prior_entropy",
                    "policy_search_kl")
            }
            weight_norm = float(torch.sqrt(sum(
                p.detach().float().square().sum() for p in value.parameters())))
            dec = max(int(window["decisions"]), 1)
            games = max(int(window["games"]), 1)
            raw_sps = sog_solves / max(sog_elapsed, 1e-9)
            gen_s = window["gen_s"] / max(window["results"], 1)
            train_s = window["train_s"]
            delay_mean = lambda source: float(
                sample_delays[sample_sources == source].mean()) \
                if np.any(sample_sources == source) else 0.0
            target_age_max = (time.time() - buf.created_at[buf.lo % buf.cap]
                              if len(buf) else 0.0)
            live = np.arange(buf.lo, buf.rows, dtype=np.int64) % buf.cap
            live_source = buf.source[live]
            live_n = max(len(live), 1)
            live_td1 = int(buf.td1[live].sum())
            live_targets = max(int(buf.clen[live].sum()), 1)
            now_at = {k: int(data[k]) for k in ROUND_KEYS}
            rounds = max(now_at["rounds"] - round_at["rounds"], 1)
            per_round = {k: (now_at[k] - round_at[k]) / rounds
                         for k in ROUND_KEYS[1:]}
            hits = now_at["budget_hits"] - round_at["budget_hits"]
            round_at = now_at
            leaf = warchest.leaf_breakdown()
            leaf_breakdown = {
                "ms_per_round": {
                    name: round(value / rounds, 3)
                    for name, value in zip(stage_names[:-3], leaf[:-3])
                },
                "bytes_per_round": {
                    name: round(value * 1e6 / rounds)
                    for name, value in zip(stage_names[-3:-1], leaf[-3:-1])
                },
            }
            stage_line = "/".join(
                f"{name}={value:g}"
                for name, value in leaf_breakdown["ms_per_round"].items()
                if value > 0)
            names = tuple(warchest.ENT_NAMES)
            now_ent = [int(x) for x in (data.get("entity_hits") or [0] * 8)]
            ent_hits = [now_ent[i] - ent_at[i] for i in range(8)]
            ent_at = now_ent
            bounds = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536)
            if window_shapes:
                a = np.asarray(window_shapes, np.uint32)
                def pct(col, q):
                    v = np.sort(a[:, col])
                    return int(v[int(round((len(v) - 1) * q))])
                shape = {
                    names[i]: {
                        "p50": pct(i, 0.50),
                        "p90": pct(i, 0.90),
                        "p99": pct(i, 0.99),
                        "max": int(a[:, i].max()),
                    }
                    for i in range(8)
                }
                node_histogram = {}
                stop_census = {}
                for kind_id, kind in enumerate(warchest.SOLVE_KIND_NAMES):
                    ka = a[a[:, 9] == kind_id]
                    hist = {}
                    for lo, hi in zip(bounds[:-1], bounds[1:]):
                        hist[f"{lo}-{hi - 1}"] = int(
                            ((ka[:, 0] >= lo) & (ka[:, 0] < hi)).sum())
                    hist[f"{bounds[-1]}+"] = int((ka[:, 0] >= bounds[-1]).sum())
                    node_histogram[kind] = hist
                    stops = {}
                    for stop_id, stop in enumerate(warchest.STOP_NAMES):
                        nodes = ka[ka[:, 8] == stop_id, 0]
                        if nodes.size:
                            nodes.sort()
                            stops[stop] = {
                                "count": int(nodes.size),
                                "node_p50": int(nodes[int(round(
                                    (nodes.size - 1) * 0.5))]),
                            }
                    stop_census[kind] = stops
            else:
                shape = {n: {"p50": 0, "p90": 0, "p99": 0, "max": 0} for n in names}
                empty_hist = {
                    **{f"{lo}-{hi - 1}": 0
                       for lo, hi in zip(bounds[:-1], bounds[1:])},
                    f"{bounds[-1]}+": 0,
                }
                node_histogram = {
                    k: dict(empty_hist) for k in warchest.SOLVE_KIND_NAMES}
                stop_census = {k: {} for k in warchest.SOLVE_KIND_NAMES}
            rec = {
                "t": round(now - t0, 1),
                "epoch": epoch,
                "phase": "sog",
                "games": int(window["games"]),
                # Decisive against drawn, per epoch. A finished game is the
                # earliest evidence that self-play is going anywhere at all, and
                # a run that only ever draws is failing differently from one
                # that never finishes a game.
                "white_wins": int(window["white_wins"]),
                "black_wins": int(window["black_wins"]),
                "draws": int(window["draws"]),
                "decisions": int(window["decisions"]),
                "rows": int(window["rows"]),
                "solves": int(window["solves"]),
                "loss": round(lv, 5),
                "total_loss": round(lv + args.policy_w * policy["policy_loss"], 5),
                "loss_old": round(diag["loss_old"], 5),
                "loss_new": round(diag["loss_new"], 5),
                "zero_sum_max": round(window["zero_sum_max"], 5),
                "zero_sum_rms": round((window["zero_sum_square_sum"]
                                        / max(window["zero_sum_n"], 1)) ** 0.5, 5),
                "grad_norm": round(window["grad_norm_sum"] / max(steps, 1), 4),
                "grad_norm_max": round(window["grad_norm_max"], 4),
                "weight_norm": round(weight_norm, 4),
                "grad_clip_frac": round(
                    window["grad_clipped"] / max(steps, 1), 4),
                "horizon_frac": round(window["horizon_hits"] / games, 3),
                # How many solves shared a forward pass. It should sit near the
                # thread count; well below means the round is waiting on
                # stragglers instead of batching them. A round carries
                # `round_batch` regret updates, so a solve rides in ceil(64 /
                # round_batch) rounds rather than sixty-five, and the two
                # figures below rise with the knob: none of the three compares
                # across a change to it.
                "calls_per_round": round(per_round["round_calls"], 2),
                "rows_per_round": round(per_round["round_rows"], 1),
                # Milliseconds a round spends inside the device backend — the
                # batch plus the concatenation and split around it. The rest of
                # a round is CFR on the cores.
                "device_ms_per_round": round(
                    1e-6 * per_round["round_nanos"], 2),
                "leaf_breakdown": leaf_breakdown,
                # Rows the query solver produced, i.e. targets taken off
                # the line of play. Zero means the coverage path is dead.
                "query_rows": int(window["query_rows"]),
                "plays": {
                    name: int(window[f"plays_{name}"])
                    for name in (
                        "attack", "pass", "deploy", "bolster",
                        "maneuver", "recruit", "claim_initiative")
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
                "replay_warm_frac": round(
                    float(np.count_nonzero(live_source == SOURCE_WARM) / live_n), 4),
                "replay_play_frac": round(
                    float(np.count_nonzero(live_source == SOURCE_PLAY) / live_n), 4),
                "replay_query_frac": round(
                    float(np.count_nonzero(live_source == SOURCE_QUERY) / live_n), 4),
                "replay_td1_row_frac": round(live_td1 / live_n, 5),
                "replay_td1_target_frac": round(
                    2 * live_td1 / live_targets, 5),
                "rows_per_s": round(
                    generated_rows / max(sog_elapsed, 1e-9), 1),
                "effective_train_ratio": round(
                    optimizer_rows / max(sog_solves, 1), 3),
                "train_row_ratio": round(
                    optimizer_rows / max(generated_rows, 1), 3),
                "tgt_mean": round(target_mean, 4),
                "tgt_std": round(target_var ** 0.5, 4),
                "tgt_belief_mean": round(belief_mean, 4),
                "tgt_belief_std": round(belief_var ** 0.5, 4),
                "tgt_p05": round(float(target_q[0]), 4),
                "tgt_p50": round(float(target_q[1]), 4),
                "tgt_p95": round(float(target_q[2]), 4),
                "tgt_abs95_frac": round(float(np.mean(np.abs(targets) >= 0.95)), 4),
                "probe_std": round(diag["probe_std"], 4),
                "value_outcome_rmse": round(diag["value_outcome_rmse"], 4),
                "value_outcome_mae": round(diag["value_outcome_mae"], 4),
                "value_outcome_bias": round(diag["value_outcome_bias"], 4),
                "value_outcome_corr": round(diag["value_outcome_corr"], 4),
                "value_calibration_slope": round(diag["value_calibration_slope"], 4),
                "gen_s": round(gen_s, 2),
                "train_s": round(train_s, 2),
                "train_enqueue_s": round(train_s, 2),
                "add_s": round(window["add_s"], 2),
                "gpu_forward_s": round(window["gpu_forward_s"], 2),
                "gpu_backward_s": round(window["gpu_backward_s"], 2),
                "batch_configs": round(
                    window["batch_configs"] / max(steps, 1), 1),
                "buf_s": round(buf.span_seconds(), 1),
                "target_age_max": round(target_age_max, 1),
                "sample_age_mean": round(float(sample_ages.mean()), 1),
                "sample_age_p50": round(float(np.quantile(sample_ages, 0.5)), 1),
                "sample_age_p90": round(float(np.quantile(sample_ages, 0.9)), 1),
                "sample_delay_mean": round(float(sample_delays.mean()), 1),
                "sample_delay_p90": round(float(np.quantile(sample_delays, 0.9)), 1),
                "sample_warm_delay": round(delay_mean(SOURCE_WARM), 1),
                "sample_play_delay": round(delay_mean(SOURCE_PLAY), 1),
                "sample_query_delay": round(delay_mean(SOURCE_QUERY), 1),
                "sample_warm_frac": round(window["sample_warm"] / sample_n, 4),
                "sample_play_frac": round(window["sample_play"] / sample_n, 4),
                "sample_query_frac": round(window["sample_query"] / sample_n, 4),
                "sample_td1_target_frac": round(
                    window["sample_td1_targets"] / max(window["sample_targets"], 1), 5),
                "solves_per_s": round(raw_sps, 1),
                **{key: round(value, 5) for key, value in policy.items()},
                "policy_weighted_loss": round(args.policy_w * policy["policy_loss"], 5),
                "budget_hits": int(hits),
                "entity_hits": {names[i]: ent_hits[i] for i in range(8)},
                "slots": int(data.get("slots", 0)),
                "slots_used": int(data.get("slots_used", 0)),
                "slots_per_card": int(data.get("slots_per_card", 0)),
                "slot_bytes": int(data.get("slot_bytes", 0)),
                "shape": shape,
                "node_histogram": node_histogram,
                "stop_census": stop_census,
            }
            rec["budget_hit_rate"] = round(
                rec["budget_hits"] / max(len(window_shapes), rec["solves"], 1), 3)
            tick(rec,
                f"[t={rec['t']:6.1f}s] GT-CFR solves={sog_solves} "
                f"rate={raw_sps:.1f}/s rows={rec['rows']} "
                f"games={rec['games']} "
                f"W{rec['white_wins']}/B{rec['black_wins']}/D{rec['draws']} "
                f"qrows={rec['query_rows']} "
                f"L={lv:.5f} L/var={lv / max(target_var, 1e-9):.2f} "
                f"Lp={rec['policy_loss']:.3f} "
                f"tgt={target_mean:+.3f}/{target_var ** 0.5:.3f} "
                f"gen={gen_s:.2f}s train_enqueue={train_s:.2f}s "
                f"add={window['add_s']:.2f}s "
                f"gpu_events={window['gpu_forward_s'] + window['gpu_backward_s']:.2f}s "
                f"slots={rec['slots_used']}/{rec['slots']} "
                f"spc={rec['slots_per_card']} "
                f"slot={rec['slot_bytes'] / (1 << 20):.1f}MiB "
                f"hits={rec['budget_hits']} "
                f"hit_rate={rec['budget_hit_rate']} "
                f"stages={stage_line} "
                f"ehits={'/'.join(str(ent_hits[i]) for i in range(8))} "
                f"p90={'/'.join(str(shape[n]['p90']) for n in names)}")
            window.clear()
            window_shapes.clear()
            window_targets.clear()
            window_target_weights.clear()
            window_sample_ages.clear()
            window_sample_delays.clear()
            window_sample_sources.clear()
        # Dropping the farm stops its threads once they finish the solve they
        # are in, which is also what flushes their last rows.
        del farm
        save_progress()

        elapsed = max(deadline - sog_t0, 1e-9)
        print(
            f"[GT-CFR-summary] solves={sog_solves} "
            f"optimizer_rows={optimizer_rows} "
            f"rate={sog_solves / elapsed:.1f}/s "
            f"horizon={totals['horizon_hits'] / max(totals['games'], 1):.2f} "
            f"games={totals['games']} "
            f"W{totals['white_wins']}/B{totals['black_wins']}/D{totals['draws']}",
            flush=True)

    print(f"[cfg] PUBFEAT={PUBFEAT} CFEAT={CFEAT} architecture=gt-cfr "
          f"s={args.s} c={args.c} "
          f"budget={total:.0f}s warm={args.warm_minutes:g}min "
          f"snapshot_every={args.snapshot_every:g}min device={dev} "
          f"draft={'random' if args.random_draft else 'starter'} "
          f"replay_ratio={args.replay_ratio} "
          f"recent_mix={args.recent_mix}/{args.recent_frac} "
          f"canonical_views=2 cap={args.cap} "
          f"matmul={torch.get_float32_matmul_precision()}", flush=True)

    if not checkpoint:
        snapshot("init", 0.0)
    warm = args.warm_minutes * 60.0
    if not 0.0 <= warm <= total:
        raise SystemExit("warm_minutes must be between zero and the run length")
    if sog_t0 is None:
        while True:
            el = time.time() - t0
            if el >= warm:
                break
            tg = time.time()
            d = warchest.gen_data(
                args.warm_games, args.seed * 1_000_003 + epoch,
                explore=args.explore, random_draft=args.random_draft,
                agent="greedy", temp=args.temp)
            gen_s = time.time() - tg
            n = ingest(buf, d)
            steps = max(1, n // args.batch) if len(buf) >= args.batch else 0
            lv, train_s, _ = fit(steps, t0 + total)
            cy = np.clip(np.asarray(d["cy"], np.float32), -1.0, 1.0)
            rec = {
                "t": round(time.time() - t0, 1), "epoch": epoch, "phase": "warm",
                "games": int(d.get("games", 0)), "rows": n,
                "loss": round(float(lv), 5) if steps else None,
                "tgt_mean": round(float(cy.mean()) if cy.size else 0.0, 4),
                "tgt_std": round(float(cy.std()) if cy.size else 0.0, 4),
                "horizon_frac": round(int(d.get("horizon_hits", 0)) /
                                      max(int(d.get("games", 0)), 1), 3),
                "gen_s": round(gen_s, 2), "train_s": round(train_s, 2),
                "train_enqueue_s": round(train_s, 2),
            }
            tick(rec,
                f"[t={rec['t']:6.1f}s] warm ep{epoch:3d} games={rec['games']:4d} "
                f"rows={n:6d} L={lv if steps else float('nan'):.5f} "
                f"tgt={rec['tgt_mean']:+.3f}/{rec['tgt_std']:.3f} "
                f"gen={gen_s:.1f}s train_enqueue={train_s:.1f}s")
        value.push()
        target_state = cpu_state(value)
        sog_t0 = time.time()
        progress["sog_start"] = sog_t0 - t0
        progress["next_target"] = (sog_t0 - t0) + args.target_every * 60.0
    # The warm rows stay in replay and age out through the FIFO. Clearing them
    # here collapsed an earlier run, so a fresh run carries them into self-play.
    run_search_pipeline()

    snapshot("final", time.time() - t0)
    write_log(args, log, snaps)
    if args.ladder_games:
        # Every snapshot becomes an immutable bot for the ladder, and a bot
        # solves on whatever cards it finds. A ladder is thousands of solves at
        # the training budget; on the CPU that was an hour a run, which is far
        # too dear for the thing that says whether the run learned anything.
        arena = [sys.executable, str(ROOT / "tools" / "arena.py")]
        bot_dir = ROOT / "bots"
        subprocess.run(arena + ["pack", args.out, "--out", str(bot_dir)], check=True)
        subprocess.run(arena + ["pack-greedy", "--out", str(bot_dir)], check=True)
        tag = pathlib.Path(args.out).name
        bots = [str(bot_dir / f"{tag}.{snap['label']}") for snap in snaps
                if snap["label"] != "final"]
        final = str(bot_dir / f"{tag}.final")
        # Greedy first, so ratings are quoted against the one reference that
        # means the same thing from one run to the next. Without it a ladder
        # only says which snapshot beats which other snapshot, which every run
        # can satisfy while learning nothing, so a missing or unrunnable anchor
        # is a failure rather than something to drop.
        greedy = bot_dir / "greedy"
        subprocess.run(arena + ["ladder", str(greedy), *bots, final,
                                "--games", str(args.ladder_games),
                                "--out", f"{args.out}/ladder.json"], check=True)


if __name__ == "__main__":
    main()
