"""The farm at a fixed workload: solves/s, and the three numbers that set it.

`rate = threads / (rounds_per_solve * round_time)`. Every driver change moves
one of those three, so the bench reports the round time beside the rate. Rounds
per solve is not one of the farm's counters -- a solve makes two to four calls
a round -- so what stands in for it is calls per solve.

The workload is a corpus of roots taken from the stream a run drives, so it
carries the run's own mix of self-play roots and query roots -- the two cost
about two-fold different in device calls, and a corpus of only one of them
ranks builds on a workload the run never sees. `farmprobe` plays games forward
instead, and a solve's cost varies twenty-six fold with how far into a game its
root sits, so its rate moves two-fold with nothing but which phase its threads
happened to reach -- 16.6, 16.2, 12.1 and 8.0 solves/s were measured across
consecutive probes of one build. Here the threads walk one shuffled corpus on
interleaved strides and cycle it, so the mix of costs in flight is the same at
every moment and the same between two builds.

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
from config import BASELINE as PROD  # noqa: E402
from export_weights import load as load_checkpoint  # noqa: E402
from value_net import Net  # noqa: E402

KEYS = ("rounds", "round_calls", "round_rows", "round_nanos")


def make(args):
    n = warchest.save_roots(
        args.games,
        args.seed,
        args.make,
        cap=args.cap,
        random_draft=PROD.random_draft,
        explore=PROD.explore,
        query_rate=args.query_rate,
        recursive_rate=args.recursive_rate,
        cpu=args.cpu,
    )
    print(f"wrote {n} roots to {args.make}")


def bench(args, devices, threads):
    farm = warchest.SolveFarm(
        args.seed,
        threads,
        s=args.s,
        c=args.c,
        batch=args.batch,
        rounds=args.rounds,
        cfr=args.cfr,
        recursive_rate=args.recursive_rate,
        devices=devices,
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
        print(
            f"workers={threads:3d} t={now - start:5.0f}s: "
            f"{rate:7.1f} solves/s | "
            f"calls/solve {calls / max(solves, 1):5.1f}  "
            f"round {1e3 * dt / max(rounds, 1):6.1f} ms | "
            f"{per_round:5.1f} calls/round  {rows / max(rounds, 1):7.0f} rows/round  "
            f"device {1e-9 * nanos / dt:4.0%}",
            flush=True,
        )
        mark, solves = now, 0
    return farm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", help="corpus to solve")
    p.add_argument("--make", help="write a corpus here instead of benching")
    p.add_argument("--cpu", action="store_true",
                   help="accept the ~50x slower CPU path used by --make")
    p.add_argument("--games", type=int, default=64, help="games to sample a corpus from")
    p.add_argument("--cap", type=int, default=4096, help="roots to keep")
    p.add_argument("--threads", default="72")
    p.add_argument("--devices", default="0,1")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--window", type=float, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--s", type=int, default=PROD.s,
                   help="expansion simulations a solve runs")
    p.add_argument("--c", type=float, default=PROD.c,
                   help="expansions per regret update")
    p.add_argument("--batch", type=int, default=PROD.round_batch,
                   help="regret updates one round of a solve carries")
    p.add_argument("--rounds", type=int, default=PROD.rounds,
                   help="round boundaries tree growth may pass through")
    p.add_argument("--cfr", default=PROD.cfr, help="the run's regret rule")
    p.add_argument("--query-rate", type=float, default=PROD.query_rate,
                   help="leaves a self-play solve queues, when making a corpus")
    p.add_argument("--recursive-rate", type=float, default=PROD.recursive_rate)
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
