"""Measure the continuous live ReBeL solve stream without training.

This is a scheduling diagnostic, not the performance gate: the real metric is
the balanced rate reported by ``train.py``. It uses a real checkpoint, random
drafts, live games, depth-2 trees, and the production 64 CFR iterations, so it
is useful for choosing wave and actor parameters before spending a training
run.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warchest  # noqa: E402
from export_weights import load  # noqa: E402


COUNTERS = (
    "solves",
    "games",
    "decisions",
    "oversize_routes",
    "card_exclusive_routes",
    "exact_fallbacks",
    "censored_games",
    "dropped",
    "node_caps",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--actors", type=int, default=128,
                    help="live game actors per worker")
    ap.add_argument("--inflight", type=int, default=32,
                    help="maximum submitted solves per worker")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="completed solves per Python result chunk")
    ap.add_argument("--report-every", type=float, default=2.0)
    ap.add_argument("--cap-value", type=float,
                    help="override the engine's horizon marker payoff")
    args = ap.parse_args()

    net = load(args.checkpoint)
    if args.cap_value is not None:
        warchest.set_cap_value(args.cap_value)
    # The CPU builder reads the global Rust-side network shape while CUDA reads
    # the flat bank passed to gpu_start. Keep them on the same checkpoint, just
    # as train.py does before it starts the services.
    net.push(0)
    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    warchest.gpu_start(net.dims, *net.flat(), devices=devices)
    gen = warchest.gpu_stream_start(
        args.seed,
        depth=2,
        iters=64,
        explore=0.25,
        random_draft=True,
        cfr="linear",
        warm=0.0,
        eval_mix=0.5,
        workers=args.workers,
        actors_per_worker=args.actors,
        inflight_per_worker=args.inflight,
        chunk_solves=args.chunk,
    )

    totals = {name: 0 for name in COUNTERS}
    started = time.monotonic()
    next_report = args.report_every
    stopped = False
    prestop = None
    try:
        while True:
            elapsed = time.monotonic() - started
            if not stopped and elapsed >= args.seconds:
                prestop = {**totals, "seconds": elapsed}
                gen.stop()
                stopped = True
            try:
                data = gen.next(timeout=0.2)
            except StopIteration:
                break
            if data is not None:
                for name in COUNTERS:
                    totals[name] += int(data.get(name, 0))
            elapsed = time.monotonic() - started
            if elapsed >= next_report and not stopped:
                print(json.dumps({
                    "phase": "live",
                    "seconds": round(elapsed, 3),
                    "solves": totals["solves"],
                    "solves_per_s": round(totals["solves"] / elapsed, 1),
                    "oversize_routes": totals["oversize_routes"],
                    "card_exclusive_routes": totals["card_exclusive_routes"],
                }), flush=True)
                next_report += args.report_every
    finally:
        if not stopped:
            gen.stop()
        warchest.gpu_stop()

    elapsed = time.monotonic() - started
    prestop = prestop or {**totals, "seconds": elapsed}
    print(json.dumps({
        "phase": "prestop",
        **prestop,
        "solves_per_s": round(prestop["solves"] / prestop["seconds"], 1),
    }), flush=True)
    print(json.dumps({
        "phase": "drained",
        **totals,
        "seconds": round(elapsed, 3),
        "solves_per_s": round(totals["solves"] / elapsed, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
