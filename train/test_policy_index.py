"""Host check: gathered policy cells index inside the batch arenas.

A 16k-row burst is what train30f added when the horizon wave landed. After
that write, every live row's `pcfg` / `pact` must sit inside the batch it
builds. The same check covers a fat write that used to wrap the cell ring
over still-live thin rows: those thin rows must have been retired.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from train import ACT_BYTES, CCOUNTS, ROW_BYTES, Buffer


def dump(n, ncfg, na, ncells):
    rows = np.zeros((n, ROW_BYTES), np.uint8)
    cc = np.zeros((n * ncfg, CCOUNTS), np.uint8)
    cw = np.full(n * ncfg, 1.0 / max(ncfg, 1), np.float32)
    cy = np.zeros(n * ncfg, np.float32)
    per = max(ncfg // 2, 1)
    coff = [0]
    for _ in range(n):
        coff.append(coff[-1] + per)
        coff.append(coff[-1] + (ncfg - per))
    coff = np.asarray(coff, np.int64)
    soff = np.arange(n + 1, dtype=np.int64)
    pa = np.zeros((n * na, ACT_BYTES), np.uint8)
    paoff = np.arange(n + 1, dtype=np.int64) * na
    pcoff = np.arange(n + 1, dtype=np.int64) * ncells
    pci = np.tile(np.arange(ncells, dtype=np.int64) % max(ncfg, 1), n).astype(np.uint16)
    pact = np.tile(np.arange(ncells, dtype=np.int64) % max(na, 1), n).astype(np.uint16)
    pprob = np.full(n * ncells, 1.0 / max(ncells, 1), np.float32)
    source = np.ones(n, np.uint8)
    truth = np.zeros((n, 2), np.uint32)
    outcome = np.full((n, 2), np.nan, np.float32)
    created = np.zeros(n, np.float64)
    td1 = np.zeros(n, np.uint8)
    pol = (pa, paoff, pcoff, pci, pact, pprob)
    return rows, cc, cw, cy, coff, soff, source, truth, outcome, created, td1, pol


def check(buf, ids):
    _x, cc, _cw, _cy, _seg, pol = buf.gather(ids)
    pa, _parow, pact, pcfg, _pp = pol
    n_cfg, n_act = len(cc), len(pa)
    assert pcfg.size == 0 or int(pcfg.max()) < n_cfg, (int(pcfg.max()), n_cfg)
    assert pact.size == 0 or int(pact.max()) < n_act, (int(pact.max()), n_act)
    return n_cfg, n_act


def main():
    rng = np.random.default_rng(0)
    n, ncfg, na, ncells = 16000, 40, 12, 24
    buf = Buffer(n * 2, n * ncfg * 2)
    buf.add(*dump(n, ncfg, na, ncells))
    check(buf, np.arange(buf.lo, buf.rows))
    print(f"16k burst: {n} live rows, pcfg < {n * ncfg}")
    for _ in range(20):
        ids = rng.integers(buf.lo, buf.rows, size=256)
        check(buf, ids)

    cap = 128
    ring = Buffer(cap, cap * 10_000)
    thin, fat = 50, 30
    ncfg_t, ncfg_f = 6, 400
    ring.add(*dump(thin, ncfg_t, 4, ncfg_t))
    ring.add(*dump(fat, ncfg_f, 8, ncfg_f))
    live = ring.cells - int(ring.pcstart[ring.lo % ring.cap])
    assert live <= ring.pcap, (live, ring.pcap, ring.lo)
    check(ring, np.arange(ring.lo, ring.rows))
    print(f"fat wrap: lo={ring.lo} live_cells={live} pcap={ring.pcap}")


if __name__ == "__main__":
    main()
