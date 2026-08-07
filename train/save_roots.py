"""Save the plan section 6 root sample: `cap` complete solver roots
(state + both beliefs) from random-draft self-play, for GPU tree sizing.
Optionally push a checkpoint first so the sample matches the trained
agent's positions (the dump run's own game distribution).

    python train/save_roots.py --ckpt runs/pre_cuda_random/snap_03.pt \
        --games 300 --cap 1000 --out runs/pre_cuda_random/roots.bin
"""

import argparse

import warchest
from export_weights import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="",
                    help="checkpoint to push before sampling (default: the pushed net)")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="roots.bin")
    args = ap.parse_args()
    if args.ckpt:
        net = load(args.ckpt)
        net.push(0)
        print(f"pushed {args.ckpt} (dims {net.dims})")
    n = warchest.save_roots(args.games, args.seed, args.out, args.cap, True)
    print(f"saved {n} roots to {args.out}")


if __name__ == "__main__":
    main()
