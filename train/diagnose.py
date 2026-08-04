"""Ask how much of a dumped replay buffer's target signal is actually learnable.

Every architecture tried so far plateaus at the same held-out error, which is
the signature of a floor that has nothing to do with the network. There are
three candidate explanations and they need separating before any more
architecture work is justified:

  1. **Drift.** Targets are bootstrapped, so a row recorded early in a run and
     a row recorded late carry values computed from different networks. Across
     a whole buffer the target stops being a function of the input alone.
  2. **A bug.** Contradictory labels, a mis-shaped mask, a target written to
     the wrong head.
  3. **Nothing wrong.** The value really does have irreducible spread given the
     network only sees a hand key rather than a full config.

This script is deliberately **model-free**: it never trains anything, so it
cannot be fooled by an optimiser or an architecture. It finds rows whose inputs
are byte-identical and measures how much their targets disagree. That
disagreement is a hard lower bound on the error *any* network can achieve --
identical inputs must produce identical outputs.

The age gap is what separates (1) from (2). If duplicates recorded far apart
disagree while duplicates recorded close together agree, the floor is drift. If
even near-simultaneous duplicates disagree, something is wrong.

    python train/diagnose.py runs/feat01/buf.npz
"""

import argparse
import sys

import numpy as np


def duplicate_groups(vx, max_rows):
    """Group row indices by exact feature bytes."""
    x = np.ascontiguousarray(vx[:max_rows])
    view = x.view(np.void, )  # noqa: E203 - one void scalar per row
    view = np.ascontiguousarray(x).view([("b", np.void, x.dtype.itemsize * x.shape[1])]).ravel()
    order = np.argsort(view, kind="stable")
    srt = view[order]
    # Boundaries of runs of equal rows.
    new = np.ones(len(srt), dtype=bool)
    new[1:] = srt[1:] != srt[:-1]
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, len(srt)))
    return order, starts, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--max-rows", type=int, default=200_000,
                    help="rows to scan for duplicates (from the start of the dump)")
    args = ap.parse_args()

    d = np.load(args.dump)
    vx, vy, vm = d["vx"], d["vy"], d["vm"]
    n = min(args.max_rows, len(vx))
    print(f"[data] {len(vx)} rows total, scanning the first {n} for duplicates")
    print(f"[data] FEAT={vx.shape[1]} dtype={vx.dtype}")

    tgt_all = vy[vm > 0]
    print(f"[data] target spread over all rows: std {tgt_all.std():.4f} "
          f"mean {tgt_all.mean():+.4f}")

    order, starts, sizes = duplicate_groups(vx, n)
    dup = starts[sizes > 1]
    dupsz = sizes[sizes > 1]
    print(f"\n[dups] {len(dup)} distinct inputs occur more than once; "
          f"{int(dupsz.sum())} rows are in a duplicate group "
          f"({100 * dupsz.sum() / n:.1f}% of scanned rows)")
    if len(dup) == 0:
        print("       no exact duplicates -- cannot bound the noise this way")
        sys.exit(0)
    print(f"[dups] group sizes: max {dupsz.max()}, median {int(np.median(dupsz))}")

    # For every duplicate group, the spread of the targets on the hand keys the
    # mask actually supports, and how far apart in the run the rows were taken.
    within, gaps, sizes_kept = [], [], []
    for s, k in zip(dup, dupsz):
        idx = order[s:s + k]
        m = vm[idx]
        y = vy[idx]
        sup = m.min(axis=0) > 0        # keys supported in *every* row of the group
        if not sup.any():
            continue
        yy = y[:, sup]
        # Spread around the group mean, pooled over the supported keys.
        within.append(np.sqrt(((yy - yy.mean(axis=0)) ** 2).mean()))
        gaps.append(int(idx.max() - idx.min()))
        sizes_kept.append(k)
    if not within:
        print("       duplicate groups share no supported hand key")
        sys.exit(0)
    within = np.array(within)
    gaps = np.array(gaps)

    rms = float(np.sqrt((within ** 2).mean()))
    print(f"\n[noise] within-duplicate target RMS: {rms:.4f}")
    print(f"        (compare: overall target std {tgt_all.std():.4f}, and the "
          f"best held-out RMS any architecture reached, ~0.092)")
    print(f"        share of duplicate groups whose targets agree exactly: "
          f"{100 * (within == 0).mean():.1f}%")

    # Drift versus bug: does disagreement grow with how far apart the rows were
    # recorded? Rows are appended in generation order, so index distance is age.
    print("\n[drift] within-duplicate RMS by how far apart the rows were recorded")
    edges = [0, 1, 100, 1_000, 10_000, 50_000, 1 << 62]
    names = ["same batch", "<100 rows", "<1k", "<10k", "<50k", ">=50k"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        sel = (gaps >= lo) & (gaps < hi)
        if sel.sum() == 0:
            continue
        print(f"        {nm:>12s}  n={int(sel.sum()):6d}  "
              f"rms={float(np.sqrt((within[sel] ** 2).mean())):.4f}")

    print("\n[read] If RMS is near zero for close rows and rises with distance, the "
          "floor is\n       target drift: no architecture can fix it, and the knob is "
          "buffer age.\n       If close rows already disagree, the target computation "
          "is not a function\n       of the encoding -- that is a bug worth finding.")


if __name__ == "__main__":
    main()
