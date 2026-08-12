"""Reading a dumped replay buffer.

A dump is a *noise-free supervised dataset*: a ReBeL value target is a
deterministic function of the network's own input -- the CFR root value of the
subgame at `(state, ctx, beliefs)`, which is exactly what the row encodes. Solve
the same position twice with the same weights and the same number comes back. So
questions about the value function can be settled here in minutes rather than by
training runs whose headline score wanders by +-0.05 on its own.

A dump holds, oldest row first:

    rows  [rows, ROW_BYTES] the packed frozen row format (see `rebel::ROW_*`);
                           the network input is expanded from these
    cc    [configs, CPRIVATE] counts and future forced-play flags, per config
    cp    [configs]         which player the config belongs to
    cw    [configs]         its belief probability
    cy    [configs]         the value the solve gave it
    seg   [configs]         `2 * row + player`, non-decreasing
    soff  [solves + 1]      solve boundaries in row space (first 0, trailing
                            row count) -- a solve is the sampling unit, and
                            the only honest split boundary

Rows are stored oldest-first so a split by recency is possible. That is the only
honest split: rows from one epoch come from the same handful of games and are
heavily correlated, so a random split leaks its answers into the training set.
`version` pins the row, feature, and target semantics; `rules_hash` pins the
rules tables. An incompatible dump is refused.
"""

import numpy as np

import warchest


class Dump:
    def __init__(self, path):
        d = np.load(path)
        self.x, self.cc, self.cp = d["rows"], d["cc"], d["cp"]
        self.cw, self.cy, self.seg = d["cw"], d["cy"], d["seg"]
        self.soff = np.asarray(d.get("soff", [0, len(self.x)]), np.int64)
        self.pubfeat = int(d["pubfeat"])
        self.cprivate = int(d.get("cprivate", 0))
        self.ccounts = int(d["ccounts"])
        self.cnorm = float(d["cnorm"])
        self.row_bytes = int(d.get("row_bytes", 0))
        self.version = int(d.get("version", 0))
        self.rules_hash = int(d.get("rules_hash", 0))
        # `seg` is emitted in row order, so a row range is a contiguous config
        # range and slicing is two binary searches rather than a scan.
        self.row_start = np.searchsorted(self.seg, 2 * np.arange(len(self.x) + 1))
        self.check(warchest.PUBFEAT, warchest.CPRIVATE, warchest.CCOUNTS)

    def __len__(self):
        return len(self.x)

    def rows(self, lo, hi):
        """Rows `[lo, hi)` as a self-contained batch, `seg` renumbered from 0."""
        a, b = int(self.row_start[lo]), int(self.row_start[hi])
        return (self.x[lo:hi], self.cc[a:b], self.cp[a:b],
                self.cw[a:b].astype(np.float32), self.cy[a:b].astype(np.float32),
                self.seg[a:b] - 2 * lo)

    def check(self, pubfeat, cprivate, ccounts):
        if (self.pubfeat, self.cprivate, self.ccounts) != (pubfeat, cprivate, ccounts):
            raise SystemExit(
                f"dump has PUBFEAT/CPRIVATE/CCOUNTS="
                f"{self.pubfeat}/{self.cprivate}/{self.ccounts}, module has "
                f"{pubfeat}/{cprivate}/{ccounts} -- rebuild or redump"
            )
        if self.row_bytes != warchest.ROW_BYTES:
            raise SystemExit(
                f"dump has ROW_BYTES={self.row_bytes}, module has {warchest.ROW_BYTES}"
            )
        if self.version != warchest.ROW_FORMAT_VERSION:
            raise SystemExit(
                f"dump has format version {self.version}, module has "
                f"{warchest.ROW_FORMAT_VERSION} -- rebuild or redump"
            )
        if self.rules_hash != warchest.rules_table_hash():
            raise SystemExit(
                f"dump was written by a different rules build (hash {self.rules_hash} "
                f"vs {warchest.rules_table_hash()}) -- redump"
            )

    def solve_splits(self, lo=0, hi=None):
        """The solve boundaries strictly inside `[lo, hi)`, as split points."""
        hi = len(self.x) if hi is None else hi
        return [s for s in self.soff if lo < s < hi]


def subset(parts, ids):
    """Pick a set of rows out of an assembled batch, renumbering `seg`."""
    rows, cc, cp, cw, cy, seg = parts
    row = seg // 2
    keep = np.isin(row, ids)
    # `ids` must be sorted for the renumbering below to stay in row order.
    newid = np.full(len(rows), -1, np.int64)
    newid[ids] = np.arange(len(ids))
    return (rows[ids], cc[keep], cp[keep], cw[keep], cy[keep],
            2 * newid[row[keep]] + (seg[keep] & 1))
