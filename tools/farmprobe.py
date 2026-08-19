"""What the farm gets through, per device set.

Reports solves/s and the batching shape — calls per round is how many solves
shared a forward pass, and it should sit at the thread count.

    python tools/farmprobe.py [--threads 36] [--seconds 20]
"""
import argparse
import sys
import time




sys.path.insert(0, "train")
import warchest  # noqa: E402
from value_net import Net  # noqa: E402


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
        config_cap=256,
        query_rate=0.9,
        recursive_rate=0.1,
        devices=devices,
    )
    # One collect to warm the pools and the kernel compile before timing.
    warm = farm.collect(solves=threads)
    keys = ("rounds", "round_calls", "round_rows")
    base = [int(warm[k]) for k in keys]

    solves, start = 0, time.time()
    while time.time() - start < seconds:
        d = farm.collect(solves=4 * threads)
        solves += int(d["solves"])
    dt = time.time() - start
    rounds, calls, rows = (int(d[k]) - b for k, b in zip(keys, base))
    print(
        f"devices={devices} threads={threads}: "
        f"{solves / dt:7.1f} solves/s  "
        f"{rounds / dt:6.1f} rounds/s  "
        f"{calls / max(rounds, 1):5.2f} calls/round  "
        f"{rows / max(rounds, 1):7.0f} rows/round",
        flush=True,
    )
    return solves / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=36)
    p.add_argument("--devices", default="0")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--nodes", type=int, default=256)
    p.add_argument("--expand", type=int, default=4)
    p.add_argument("--iters", type=int, default=16)
    args = p.parse_args()

    net = Net()
    net.push()
    print(f"params {sum(p.numel() for p in net.parameters())}", flush=True)

    probe([int(d) for d in args.devices.split(",")], args.threads, args.seconds, args)
    print("probe done; anything below here is shutdown", flush=True)


if __name__ == "__main__":
    main()
