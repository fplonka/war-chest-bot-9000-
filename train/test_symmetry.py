import numpy as np

import mirror
import warchest


def main():
    rows = np.frombuffer(bytes(warchest.sample_rows(2, 23)), np.uint8)
    rows = rows.reshape(-1, warchest.ROW_BYTES)[:2].copy()
    cc = np.arange(4 * warchest.CCOUNTS, dtype=np.uint8).reshape(4, -1)
    player = np.asarray([0, 1, 0, 1], np.uint8)
    weight = np.ones(4, np.float32)
    target = np.asarray([-0.5, 0.5, -0.25, 0.25], np.float32)
    seg = np.arange(4, dtype=np.int64)
    desc = np.asarray([[0, 0, 6, 0, 7, 255],
                       [1, 9, 255, 36, 12, 4],
                       [2, 2, 8, 3, 8, 13]], np.uint8)
    policy = (desc, np.asarray([0, 1, 2]), np.asarray([0, 0, 1]),
              np.asarray([0, 1, 2]), np.asarray([0.2, 0.8, 1.0], np.float32),
              np.asarray([0, 0, 1]))
    parts = (rows, cc, player, weight, target, seg, policy)
    flipped = np.asarray([True, False])

    once = mirror.mirror_batch(parts, flipped)
    assert np.array_equal(once[0][0], mirror.mirror_rows(rows[:1])[0])
    assert np.array_equal(once[0][1], rows[1])
    assert np.array_equal(once[2], [1, 0, 0, 1])
    assert np.array_equal(once[5], [1, 0, 2, 3])
    assert np.array_equal(once[6][0][0, 1:3], [5, 1])
    assert once[6][0][0, 5] == 255
    assert np.array_equal(once[6][0][2], desc[2])

    twice = mirror.mirror_batch(once, flipped)
    for got, want in zip(twice[:6], parts[:6]):
        assert np.array_equal(got, want)
    for got, want in zip(twice[6], parts[6]):
        assert np.array_equal(got, want)
    print("replay symmetry OK")


if __name__ == "__main__":
    main()
