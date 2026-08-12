"""A frozen set of solved positions, and every checkpoint's error against it.

    python train/truth.py build --ckpt runs/best/snap_04.pt --out data/truth.npz
    python train/truth.py score runs/*/snap_*.pt

Why this exists
---------------
Until now the only way to ask "is this value function any good" was to play
matches, which is the noisiest instrument in the project: a few hundred games
resolve nothing finer than about 70 Elo. Yet the quantity the training loop is
actually chasing is not noisy at all. A ReBeL value target is the CFR root value
of a depth-limited subgame -- a deterministic function of the network's own
input -- so a position solved *to convergence* has one right answer, and a
checkpoint either reproduces it or does not.

So: solve a few thousand positions past the point where the answer is still
moving, write them down, and never touch them again. Every checkpoint then gets
one number, in seconds, with no games and no variance.

READ THIS BEFORE TRUSTING THE NUMBER
------------------------------------
**This set measures similarity to the network that built it, not quality.**
Measured, not feared. The same five `runs/gpu_golden8` snapshots, scored against
two sets built the same way from two different leaf evaluators:

| checkpoint | Elo | set built from snap_02 | set built from snap_04 |
|---|---:|---:|---:|
| snap_02 | +281 | 0.00847 | 0.0637 |
| snap_03 | +473 | **0.00773** | 0.0350 |
| snap_04 | +593 | 0.00895 | **0.0163** |

Each set ranks its own builder's neighbourhood best, and on the snap_02 set the
strongest network we have comes out *worse* than one 312 Elo below it. The
ranking does not survive changing the ruler.

The reason is structural and was always in the design: the targets are
`CFR(leaves = builder)`, so a network equal to the builder reproduces them up to
the search itself, and every other network pays for its distance from the
builder on top of whatever it gets wrong. The set is a fixed point of the
*builder's* operator, not of the game.

What it is still good for: telling a badly-trained network from a decently
trained one (both sets agree snap_00 and snap_01 are far worse), and tracking
one run's progress over its own snapshots. What it must not do is arbitrate
between experiment arms -- it systematically favours whichever arm most
resembles the builder, and if the builder came from the control's configuration
that is a bias pointed straight at the answer. **The ladder arbitrates.**

The fix, unimplemented: anchor the targets on terminal positions instead of on a
network. Late-game positions solved deep enough that every leaf of the subgame
is terminal have exact game values, no evaluator involved, and would be a ruler
that stays true forever. That is the set worth building.

One rule survives unchanged: **never rebuild a set mid-experiment.** Build a new
one under a new filename, keep both, and say which one a result came from.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import warchest
from dump import Dump
from export_weights import load as load_checkpoint
from offline import evaluate
from train import CCOUNTS, CNORM, CFEAT, PUBFEAT, ROW_BYTES

# Iterations for the reference solve. At T=256 with dcfr the remaining target
# error is 0.00008 (`runs/solvererr_g8`), a thousandth of the network's own
# ~0.088, so nothing measured against this set is limited by it. Higher was
# tried and is simply 4x the build cost for the same ruler.
REFERENCE_ITERS = 256
REFERENCE_CFR = "dcfr"
DEFAULT_SET = os.environ.get("WARCHEST_TRUTH", "data/truth.npz")


def build(args):
    """Play games with a trained net and solve every decision to convergence."""
    net = load_checkpoint(args.ckpt)
    net.push(0)
    # The horizon's marker payoff is a training aid. A ruler must measure the
    # real game, so it is off here and in every ladder.
    warchest.set_cap_value(0.0)
    print(f"[truth] {args.games} games, T={args.iters} {args.cfr}, "
          f"leaves from {args.ckpt}", flush=True)
    d = warchest.gen_data(args.games, args.seed, "rebel", depth=args.depth,
                          iters=args.iters, explore=args.explore, temp=2.0,
                          eval_mix=0.0, cfr=args.cfr, warm=0.0,
                          random_draft=args.random_draft)

    rows = np.asarray(d["rows"], np.uint8).reshape(-1, ROW_BYTES)
    cc = np.asarray(d["cc"], np.uint8).reshape(-1, CCOUNTS)
    cw = np.asarray(d["cw"], np.float32)
    cy = np.asarray(d["cy"], np.float32)
    coff = np.asarray(d["coff"], np.int64)
    # `coff` holds two entries per row (one per seat); `seg` is 2*row + seat,
    # which is the layout `dump.py` documents and every batcher expects.
    lens = np.diff(coff)
    seg = np.repeat(np.arange(len(lens)), lens)
    cp = (seg & 1).astype(np.uint8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, rows=rows, cc=cc, cp=cp, cw=cw, cy=cy,
             soff=np.asarray(d["soff"], np.int64), seg=seg,
             pubfeat=np.int32(PUBFEAT), cfeat=np.int32(CFEAT),
             ccounts=np.int32(CCOUNTS),
             cnorm=np.float32(CNORM),
             row_bytes=np.int32(ROW_BYTES),
             version=np.int32(warchest.ROW_FORMAT_VERSION),
             rules_hash=np.uint64(warchest.rules_table_hash()))
    print(f"[truth] wrote {args.out}: {len(rows)} positions, {len(cy)} configs, "
          f"target spread {cy.std():.4f}", flush=True)


def errors(paths, set_path=DEFAULT_SET, device="cpu"):
    """`{path: (huber, rms)}` against the frozen set, or `{}` if there is none.

    Callers that have no set must still work: the ladder is the older
    instrument and does not depend on this one.
    """
    if not os.path.exists(set_path):
        return {}
    try:
        d = Dump(set_path)
    except SystemExit as e:
        print(f"[truth] skipped: {e}", flush=True)
        return {}
    parts = d.rows(0, len(d))
    dev, rng = torch.device(device), np.random.default_rng(0)
    return {p: evaluate(load_checkpoint(p).to(dev), parts, rng, dev) for p in paths}


def score(args):
    """Every checkpoint's belief-weighted error against the frozen set."""
    if not os.path.exists(args.set):
        raise SystemExit(f"no such set {args.set}; build one first")
    out = errors(args.ckpts, args.set, args.device)
    print(f"[truth] {args.set}\n\n{'checkpoint':>44s} {'huber':>9s} {'rms':>9s}",
          flush=True)
    for path, (huber, rms) in out.items():
        print(f"{path:>44s} {huber:>9.6f} {rms:>9.6f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="solve a fresh set of positions to convergence")
    b.add_argument("--ckpt", required=True, help="the leaf evaluator; use the strongest we have")
    b.add_argument("--out", default=DEFAULT_SET)
    b.add_argument("--games", type=int, default=64)
    b.add_argument("--iters", type=int, default=REFERENCE_ITERS)
    b.add_argument("--cfr", default=REFERENCE_CFR)
    b.add_argument("--depth", type=int, default=2)
    b.add_argument("--explore", type=float, default=0.25)
    b.add_argument("--random-draft", action="store_true",
                   help="widen the position distribution beyond the fixed armies")
    b.add_argument("--seed", type=int, default=20260810)
    b.set_defaults(fn=build)

    s = sub.add_parser("score", help="rate checkpoints against a frozen set")
    s.add_argument("ckpts", nargs="+")
    s.add_argument("--set", default=DEFAULT_SET)
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
