"""Checks for the replay rotate-and-swap augmentation."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mirror
import warchest


class FixedRng:
    def random(self, n):
        return np.array([0.1, 0.9][:n], dtype=np.float64)


def main():
    rows = np.zeros((2, warchest.ROW_BYTES), np.uint8)
    rows[0, warchest.ROW_HEX_OWNER] = 0
    rows[0, warchest.ROW_HEX_SLOT] = 1
    rows[0, warchest.ROW_HEX_MARKER] = 1
    rows[0, warchest.ROW_IDS:warchest.ROW_IDS + warchest.NTYPE] = np.arange(
        warchest.NTYPE)
    rows[0, warchest.ROW_INITIATIVE] = 0
    rows[0, warchest.ROW_TO_ACT] = 0
    rows[0, warchest.ROW_STACK_OWED] = 1
    rows[1, warchest.ROW_INITIATIVE] = 1
    rows[1, warchest.ROW_TO_ACT] = 1

    assert np.array_equal(mirror.mirror_rows(mirror.mirror_rows(rows)), rows)

    lens = [2, 3, 1, 2]
    seg = np.repeat(np.arange(4, dtype=np.int64), lens)
    cc = np.arange(len(seg) * warchest.CCOUNTS, dtype=np.uint8).reshape(
        len(seg), warchest.CCOUNTS)
    cp = seg & 1
    cw = np.arange(len(seg), dtype=np.float32)
    cy = cw + 100
    pa = np.zeros((3, warchest.ACT_BYTES), np.uint8)
    pa[:, 0] = 1
    pa[:, 1] = 2
    pa[0, 2:5] = [0, 1, 255]
    pa[1, 2:5] = [2, 3, 4]
    pa[2, 2:5] = [5, 6, 7]
    policy = (
        pa,
        np.array([0, 1, 2], np.int64),
        np.array([0, 0, 1], np.int64),
        np.array([0, 2, 5], np.int64),
        np.ones(3, np.float32),
        np.array([0, 1, 1], np.int64),
    )
    got = mirror.augment((rows, cc, cp, cw, cy, seg, policy), FixedRng())
    mrows, mcc, mcp, mcw, mcy, mseg, (mpa, _, _, mpcfg, _, _) = got

    np.testing.assert_array_equal(mrows[0], mirror.mirror_rows(rows[[0]])[0])
    np.testing.assert_array_equal(mrows[1], rows[1])
    np.testing.assert_array_equal(mcc, cc)
    np.testing.assert_array_equal(mseg, [1, 1, 0, 0, 0, 2, 3, 3])
    np.testing.assert_array_equal(mcp, [1, 1, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(mcw, cw)
    np.testing.assert_array_equal(mcy, cy)
    np.testing.assert_array_equal(mpcfg, [0, 2, 5])
    expected = pa[0, 2:5].copy()
    valid = expected < warchest.N_HEXES
    expected[valid] = mirror.HEXMAP[expected[valid]]
    np.testing.assert_array_equal(mpa[0, 2:5], expected)
    np.testing.assert_array_equal(mpa[1:], pa[1:])
    print("mirror augmentation OK")


if __name__ == "__main__":
    main()
