"""The replay symmetry moves rows, players, configs, and policy entities together."""

import numpy as np

import mirror
import warchest


def main():
    pairs = np.frombuffer(bytes(warchest.mirror_row_pairs(4, 23)), np.uint8)
    rows = pairs.reshape(-1, warchest.ROW_BYTES)[0::2][:2].copy()
    seg = np.arange(4, dtype=np.int64)
    desc = np.asarray([
        [0, 0, 6, 0, 7, 255],
        [1, 9, 255, 36, 12, 4],
        [2, 2, 8, 3, 8, 13],
    ], np.uint8)
    policy = (desc, np.asarray([0, 0, 1]), np.asarray([0, 1, 2]),
              np.asarray([0, 1, 2]), np.asarray([0.2, 0.8, 1.0], np.float32))
    flipped = np.asarray([True, False])

    once = mirror.mirror_batch(rows, seg, policy, flipped)
    assert np.array_equal(once[0][0], mirror.mirror_rows(rows[:1])[0])
    assert np.array_equal(once[0][1], rows[1])
    assert np.array_equal(once[1], [1, 0, 2, 3])
    assert np.array_equal(once[2][0][0, 1:3], [5, 1])
    assert once[2][0][0, 5] == 255
    assert np.array_equal(once[2][0][2], desc[2])

    twice = mirror.mirror_batch(*once, flipped)
    assert np.array_equal(twice[0], rows)
    assert np.array_equal(twice[1], seg)
    assert all(np.array_equal(a, b) for a, b in zip(twice[2], policy))
    print("replay symmetry OK")


if __name__ == "__main__":
    main()
