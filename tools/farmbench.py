"""The farm at a fixed workload: solves/s, and the three numbers that set it.

`rate = threads / (rounds_per_solve * round_time)`. Every driver change moves
one of those three, so the bench reports all three and not only the rate.

The workload is a corpus of roots sampled from real play. `farmprobe` plays
games forward instead, and a solve's cost varies twenty-six fold with how far
into a game its root sits, so its rate moves two-fold with nothing but which
phase its threads happened to reach -- 16.6, 16.2, 12.1 and 8.0 solves/s were
measured across consecutive probes of one build. Here the threads walk one
fixed corpus on interleaved strides and cycle it, so the mix of costs in flight
is the same at every moment and the same between two builds.

    python tools/farmbench.py --make roots.bin --games 64
    python tools/farmbench.py --roots roots.bin --threads 72 --devices 0,1

What it cannot do is claim a rate for a training run: the trainer holds a card,
publishes weights mid-run, and its position distribution shifts as play
improves. Rank builds here; claim rates with train.py.
"""

import argparse
import sys
import time

sys.path.insert(0, "train")
import warchest  # noqa: E402
from export_weights import load as load_checkpoint  # noqa: E402
from value_net import Net  # noqa: E402

KEYS = ("rounds", "round_calls", "round_rows", "round_nanos")


def make(args):
    n = warchest.save_roots(args.games, args.seed, args.make, cap=args.cap)
    print(f"wrote {n} roots to {args.make}")


def bench(args, devices, threads):
    farm = warchest.SolveFarm(
        args.seed,
        threads,
        s=args.s,
        c=args.c,
        cfr=args.cfr,
        recursive_rate=args.recursive_rate,
        devices=devices,
        cohorts=args.cohorts,
        roots=args.roots,
    )
    # Warm: the kernels compile, the pools fill, and every thread reaches its
    # first solve. None of that is what we are measuring.
    warm = farm.collect(solves=4 * threads)
    base = [int(warm[k]) for k in KEYS]

    start = time.time()
    mark, solves = start, 0
    while time.time() - start < args.seconds:
        d = farm.collect(solves=threads)
        solves += int(d["solves"])
        now = time.time()
        if now - mark < args.window:
            continue
        dt = now - mark
        rounds, calls, rows, nanos = (int(d[k]) - b for k, b in zip(KEYS, base))
        base = [int(d[k]) for k in KEYS]
        rate = solves / dt
        per_round = calls / max(rounds, 1)
        # A round waits for the slowest thread in its cohort, so what a solver
        # thread does between rounds is the round's floor. With more threads
        # than cores this is queueing as much as work, which is the whole
        # question about the thread-per-solve shape.
        # `awake()` reports milliseconds already.
        ms, spans, longest = warchest.awake()
        awake = ms / max(spans, 1)
        print(
            f"threads={threads:3d} t={now - start:5.0f}s: "
            f"{rate:7.1f} solves/s | "
            f"threads {threads:3d}  "
            f"rounds/solve {rounds * per_round / max(solves, 1):5.1f}  "
            f"round {1e3 * dt / max(rounds, 1):6.1f} ms | "
            f"{per_round:5.1f} calls/round  {rows / max(rounds, 1):7.0f} rows/round  "
            f"device {1e-9 * nanos / dt:4.0%}  "
            f"awake {awake:5.1f}/{longest:6.1f} ms",
            flush=True,
        )
        mark, solves = now, 0
    return farm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", help="corpus to solve")
    p.add_argument("--make", help="write a corpus here instead of benching")
    p.add_argument("--games", type=int, default=64, help="games to sample a corpus from")
    p.add_argument("--cap", type=int, default=4096, help="roots to keep")
    p.add_argument("--threads", default="72")
    p.add_argument("--cohorts", type=int, default=2,
                   help="independent cohorts of solves; one lane of the card each")
    p.add_argument("--devices", default="0,1")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--window", type=float, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--s", type=int, default=512,
                   help="expansion simulations a solve runs")
    p.add_argument("--c", type=float, default=8.0,
                   help="expansions per regret update")
    p.add_argument("--cfr", default="sog", help="the run's regret rule")
    p.add_argument("--recursive-rate", type=float, default=0.1)
    p.add_argument("--weights", help="a checkpoint to solve with, e.g. runs/NAME/snap_02.pt")
    args = p.parse_args()

    # The network steers PUCT, so it steers where the tree grows and what a
    # solve costs. A random net is a different workload from a trained one.
    net = Net()
    if args.weights:
        net.load_state_dict(load_checkpoint(args.weights).state_dict())
    net.push()
    if args.make:
        return make(args)
    if not args.roots:
        p.error("one of --roots or --make")
    devices = [int(d) for d in args.devices.split(",")]
    for threads in (int(t) for t in args.threads.split(",")):
        bench(args, devices, threads)
    warchest.prof_dump()
    b = warchest.leaf_breakdown()
    if b:
        # The engine owns the names, so a rename cannot mislabel a column here.
        # The last two ride the same accumulator as byte counts, which the
        # shared 1e6 scaling turns into megabytes.
        names = warchest.stage_names()
        print("stages, ms: "
              + "  ".join(f"{n}={v:.0f}" for n, v in zip(names[:-2], b) if v > 0))
        print("  ".join(f"{n}={v:.0f} MB" for n, v in zip(names[-2:], b[-2:])))


if __name__ == "__main__":
    main()
