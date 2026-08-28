"""Read compact replay snapshots."""

import numpy as np

import warchest


class Dump:
    def __init__(self, path):
        d = np.load(path)
        self.x = d["rows"]
        self.cc = d["cc"]
        self.cw, self.cy, self.seg = d["cw"], d["cy"], d["seg"]
        self.soff = np.asarray(d.get("soff", [0, len(self.x)]), np.int64)
        self.pubfeat = int(d["pubfeat"])
        self.ccounts = int(d["ccounts"])
        self.cnorm = float(d["cnorm"])
        self.row_bytes = int(d.get("row_bytes", 0))
        self.rules_hash = int(d.get("rules_hash", 0))
        # `seg` is emitted in row order, so a row range is a contiguous config
        # range and slicing is two binary searches rather than a scan.
        self.row_start = np.searchsorted(self.seg, 2 * np.arange(len(self.x) + 1))

    def __len__(self):
        return len(self.x)

    def rows(self, lo, hi):
        """Rows `[lo, hi)` as a self-contained batch, `seg` renumbered from 0."""
        a, b = int(self.row_start[lo]), int(self.row_start[hi])
        return (self.x[lo:hi], self.cc[a:b], self.cw[a:b].astype(np.float32),
                self.cy[a:b].astype(np.float32), self.seg[a:b] - 2 * lo)

    def check(self, pubfeat, ccounts):
        if self.pubfeat != pubfeat or self.ccounts != ccounts:
            raise SystemExit(
                f"dump has PUBFEAT={self.pubfeat} CCOUNTS={self.ccounts}, module has "
                f"{pubfeat}/{ccounts} -- rebuild or redump"
            )
        if self.row_bytes != warchest.ROW_BYTES:
            raise SystemExit(
                f"dump has ROW_BYTES={self.row_bytes}, module has {warchest.ROW_BYTES}"
            )
        if self.rules_hash != warchest.rules_table_hash():
            raise SystemExit(
                f"dump was written by a different rules build (hash {self.rules_hash} "
                f"vs {warchest.rules_table_hash()}) -- redump"
            )
