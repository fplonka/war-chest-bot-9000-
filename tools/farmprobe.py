"""What the farm gets through, and where the time goes.

Solves per second is the headline. The rest says why it is what it is:

* `calls/round` is how many solves shared a forward pass. It cannot exceed the
  thread count, and well below it means rounds are firing on patience with
  most threads still in CFR.
* `device` is the share of wall clock spent inside the backend — the batch
  itself, plus the concatenation and the split around it. The remainder is CFR
  on the cores.

    python tools/farmprobe.py --threads 36,54,72 --devices 0,1

Each thread count is timed in its own process-lifetime farm, so the pools and
the kernel compile are warm before anything is measured.
"""

import argparse
import sys
import time

sys.path.insert(0, "train")
import warchest  # noqa: E402
from value_net import Net  # noqa: E402

KEYS = ("rounds", "round_calls", "round_rows", "round_nanos")


def probe(devices, threads, seconds, args):
    farm = warchest.SolveFarm(
        1234,
        threads,
        nodes=args.nodes,
        expand=args.expand,
        iters=args.iters,
        explore=0.1,
        random_draft=True,
        cfr="dcfr",
        node_cap=args.node_cap,
        config_cap=256,
        query_rate=0.9,
        recursive_rate=0.1,
        devices=devices,
        cohorts=args.cohorts,
        grow_every=args.grow_every,
    )
    warm = farm.collect(solves=threads)
    base = [int(warm[k]) for k in KEYS]

    # Windows, because the rate is not constant: a farm can start fast and
    # decay, and an average over the whole probe hides that entirely.
    start = time.time()
    mark, solves, done = start, 0, 0
    while time.time() - start < seconds:
        d = farm.collect(solves=args.chunk or threads)
        solves += int(d["solves"])
        done += int(d["solves"])
        now = time.time()
        if now - mark < args.window:
            continue
        dt = now - mark
        rounds, calls, rows, nanos = (int(d[k]) - b for k, b in zip(KEYS, base))
        base = [int(d[k]) for k in KEYS]
        print(
            f"devices={devices} threads={threads:3d} t={now - start:5.0f}s: "
            f"{solves / dt:7.1f} solves/s  "
            f"({done / (now - start):5.1f} mean)  "
            f"{rounds / dt:6.1f} rounds/s  "
            f"{calls / max(rounds, 1):5.2f} calls/round  "
            f"{rows / max(rounds, 1):7.0f} rows/round  "
            f"device {1e-9 * nanos / dt:4.0%}",
            flush=True,
        )
        mark, solves = now, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", default="36")
    p.add_argument("--grow-every", type=int, default=1,
                   help="CFR iterations a solve runs before waking the host to grow")
    p.add_argument("--cohorts", type=int, default=2,
                   help="independent cohorts of solves; one lane of the card each")
    p.add_argument("--devices", default="0")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--window", type=float, default=30)
    # A window can only close when a collect returns, so a big chunk hides the
    # rate behind minutes of silence when the rate is what is being measured.
    p.add_argument("--chunk", type=int, default=0, help="solves per collect")
    p.add_argument("--nodes", type=int, default=8192)
    p.add_argument("--expand", type=int, default=8)
    p.add_argument("--iters", type=int, default=64)
    p.add_argument("--node-cap", type=int, default=65536)
    args = p.parse_args()

    Net().push()
    devices = [int(d) for d in args.devices.split(",")]
    for threads in (int(t) for t in args.threads.split(",")):
        probe(devices, threads, args.seconds, args)
    # Only says anything if the module was built with the `prof` feature. NET
    # is the time a solver spends inside `submit`, so it is the device plus the
    # wait for the round, not the device alone.
    warchest.prof_dump()
    b = warchest.leaf_breakdown()
    if b:
        # The device stages are only filled under WARCHEST_STAGES, which costs
        # a stream synchronise apiece; without it they read zero and the host's
        # own are all there is. The engine owns the names.
        names = warchest.stage_names()
        print("stages, ms: "
              + "  ".join(f"{n}={v:.0f}" for n, v in zip(names[:-3], b) if v > 0))
        print("  ".join(f"{n}={v:.0f} MB" for n, v in zip(names[-3:], b[-3:])))
    census = getattr(warchest, "solve_census", lambda: [])()
    if census:
        total = sum(b for _, b in census)
        print(f"fattest solve {total / 1e6:.1f} MB: "
              + "  ".join(f"{n}={b / 1e6:.1f}" for n, b in census[:10] if b))


if __name__ == "__main__":
    main()
