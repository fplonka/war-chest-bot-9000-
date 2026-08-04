"""Reading a dumped replay buffer.

A dump is a *noise-free supervised dataset*: a ReBeL value target is a
deterministic function of the network's own input -- the CFR root value of the
subgame at `(state, ctx, beliefs)`, which is exactly what the row encodes. Solve
the same position twice with the same weights and the same number comes back. So
questions about the value function can be settled here in minutes rather than by
training runs whose headline score wanders by +-0.05 on its own.

A dump holds, oldest row first:

    x    [rows, PUBFEAT]   the public encoding
    cc   [configs, CCOUNTS] hand, face-down and bag counts, per config
    cp   [configs]         which player the config belongs to
    cw   [configs]         its belief probability
    cy   [configs]         the value the solve gave it
    seg  [configs]         `2 * row + player`, non-decreasing

Rows are stored oldest-first so a split by recency is possible. That is the only
honest split: rows from one epoch come from the same handful of games and are
heavily correlated, so a random split leaks its answers into the training set.
"""

import numpy as np


class Dump:
    def __init__(self, path):
        d = np.load(path)
        self.x, self.cc, self.cp = d["x"], d["cc"], d["cp"]
        self.cw, self.cy, self.seg = d["cw"], d["cy"], d["seg"]
        self.pubfeat = int(d["pubfeat"])
        self.ccounts = int(d["ccounts"])
        self.cnorm = float(d["cnorm"])
        # `seg` is emitted in row order, so a row range is a contiguous config
        # range and slicing is two binary searches rather than a scan.
        self.row_start = np.searchsorted(self.seg, 2 * np.arange(len(self.x) + 1))

    def __len__(self):
        return len(self.x)

    def rows(self, lo, hi):
        """Rows `[lo, hi)` as a self-contained batch, `seg` renumbered from 0."""
        a, b = int(self.row_start[lo]), int(self.row_start[hi])
        return (self.x[lo:hi], self.cc[a:b], self.cp[a:b],
                self.cw[a:b].astype(np.float32), self.cy[a:b].astype(np.float32),
                self.seg[a:b] - 2 * lo)

    def check(self, pubfeat, ccounts):
        if self.pubfeat != pubfeat or self.ccounts != ccounts:
            raise SystemExit(
                f"dump has PUBFEAT={self.pubfeat} CCOUNTS={self.ccounts}, module has "
                f"{pubfeat}/{ccounts} -- rebuild or redump")


def subset(parts, ids):
    """Pick a set of rows out of an assembled batch, renumbering `seg`."""
    x, cc, cp, cw, cy, seg = parts
    row = seg // 2
    keep = np.isin(row, ids)
    # `ids` must be sorted for the renumbering below to stay in row order.
    newid = np.full(len(x), -1, np.int64)
    newid[ids] = np.arange(len(ids))
    return (x[ids], cc[keep], cp[keep], cw[keep], cy[keep],
            2 * newid[row[keep]] + (seg[keep] & 1))
