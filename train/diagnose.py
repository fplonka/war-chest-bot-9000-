"""Ask how much of a dumped replay buffer's target signal is actually learnable.

When held-out error stops falling, there are three candidate explanations and
they need separating before any architecture work is justified:

  1. **Drift.** Targets are bootstrapped, so a row recorded early in a run and
     a row recorded late carry values computed from different networks. Across
     a whole buffer the target stops being a function of the input alone.
  2. **A bug.** Contradictory labels, a target written against the wrong
     config, a belief that does not match the values beside it.
  3. **Nothing wrong.** Some irreducible spread the network cannot see.

This script is deliberately **model-free**: it never trains anything, so it
cannot be fooled by an optimiser or an architecture. It finds rows whose whole
input is byte-identical -- public encoding, config lists and belief weights --
and measures how much their targets disagree. That disagreement is a hard lower
bound on the error *any* network can achieve: identical inputs must produce
identical outputs.

The age gap is what separates (1) from (2). If duplicates recorded far apart
disagree while duplicates recorded close together agree, the floor is drift. If
even near-simultaneous duplicates disagree, something is wrong.

    python train/diagnose.py runs/mine/buf.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dump import Dump


def row_keys(d, n):
    """One hashable key per row: everything the network is given, verbatim."""
    keys = []
    for r in range(n):
        a, b = int(d.row_start[r]), int(d.row_start[r + 1])
        keys.append(
            d.x[r].tobytes()
            + d.cc[a:b].tobytes()
            + d.cw[a:b].tobytes()
        )
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--max-rows", type=int, default=200_000,
                    help="rows to scan for duplicates (from the start of the dump)")
    args = ap.parse_args()

    d = Dump(args.dump)
    n = min(args.max_rows, len(d))
    print(f"[data] {len(d)} rows total ({len(d.cy)} configs), scanning the first {n}")
    print(f"[data] PUBFEAT={d.pubfeat} CCOUNTS={d.ccounts} ROW_BYTES={d.row_bytes}")
    print(f"[data] target spread over all configs: std {d.cy.std():.4f} "
          f"mean {d.cy.mean():+.4f}")

    groups = {}
    for r, k in enumerate(row_keys(d, n)):
        groups.setdefault(k, []).append(r)
    dup = [v for v in groups.values() if len(v) > 1]
    ndup = sum(len(v) for v in dup)
    print(f"\n[dups] {len(dup)} distinct inputs occur more than once; {ndup} rows are "
          f"in a duplicate group ({100 * ndup / max(n, 1):.1f}% of scanned rows)")
    if not dup:
        print("       no exact duplicates -- cannot bound the noise this way")
        sys.exit(0)
    szs = np.array([len(v) for v in dup])
    print(f"[dups] group sizes: max {szs.max()}, median {int(np.median(szs))}")

    # For every duplicate group, the belief-weighted spread of the per-config
    # targets around the group mean, and how far apart the rows were recorded.
    within, gaps = [], []
    for rows in dup:
        ys, ws = [], None
        for r in rows:
            a, b = int(d.row_start[r]), int(d.row_start[r + 1])
            ys.append(d.cy[a:b].astype(np.float64))
            ws = d.cw[a:b].astype(np.float64)
        y = np.stack(ys)
        dev = y - y.mean(axis=0)
        within.append(np.sqrt((ws * (dev ** 2)).sum() / max(ws.sum() * len(rows), 1e-9)))
        gaps.append(max(rows) - min(rows))
    within, gaps = np.array(within), np.array(gaps)

    rms = float(np.sqrt((within ** 2).mean()))
    print(f"\n[noise] within-duplicate target RMS: {rms:.4f}")
    print(f"        (compare: overall target std {d.cy.std():.4f})")
    print(f"        share of duplicate groups whose targets agree exactly: "
          f"{100 * (within == 0).mean():.1f}%")

    # Drift versus bug: does disagreement grow with how far apart the rows were
    # recorded? Rows are stored in generation order, so index distance is age.
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
