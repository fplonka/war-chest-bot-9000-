"""Save the plan section 6 root sample: `cap` complete solver roots
(state + both beliefs) from random-draft self-play, for GPU tree sizing.

    python train/save_roots.py --games 200 --cap 1000 --out runs/pre_cuda_random/roots.bin
"""

import argparse

import warchest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="roots.bin")
    args = ap.parse_args()
    n = warchest.save_roots(args.games, args.seed, args.out, args.cap, True)
    print(f"saved {n} roots to {args.out}")


if __name__ == "__main__":
    main()
