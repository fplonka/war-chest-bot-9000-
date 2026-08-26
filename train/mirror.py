"""Rotate a replay row and exchange its two player segments."""

import numpy as np
import warchest

N_HEXES = warchest.N_HEXES
NSLOT = warchest.NSLOT
PILE_COUNTS = warchest.PILE_COUNTS

_coords = np.asarray(warchest.hex_coords(), dtype=np.int16)
HEXMAP = np.asarray([
    np.flatnonzero(np.all(_coords == (6 - x, 6 - y), axis=1))[0]
    for x, y in _coords
], dtype=np.int64)


def _flip_seat(values):
    return np.where(values == 0, 1, np.where(values == 1, 0, values))


def _swap(values, start, width):
    first = values[:, start:start + width].copy()
    values[:, start:start + width] = values[:, start + width:start + 2 * width]
    values[:, start + width:start + 2 * width] = first


def mirror_rows(rows):
    """Return packed rows rotated 180 degrees with the seats exchanged."""
    out = rows.copy()
    for start in (warchest.ROW_HEX_OWNER, warchest.ROW_HEX_SLOT,
                  warchest.ROW_HEX_HEIGHT, warchest.ROW_HEX_MARKER):
        out[:, start:start + N_HEXES] = rows[:, start + HEXMAP]
    out[:, warchest.ROW_HEX_OWNER:warchest.ROW_HEX_OWNER + N_HEXES] = _flip_seat(
        out[:, warchest.ROW_HEX_OWNER:warchest.ROW_HEX_OWNER + N_HEXES])
    out[:, warchest.ROW_HEX_MARKER:warchest.ROW_HEX_MARKER + N_HEXES] = _flip_seat(
        out[:, warchest.ROW_HEX_MARKER:warchest.ROW_HEX_MARKER + N_HEXES])

    _swap(out, warchest.ROW_IDS, NSLOT)
    _swap(out, warchest.ROW_PILES, NSLOT * PILE_COUNTS)
    for start in (warchest.ROW_HAND_SIZE, warchest.ROW_FD_SIZE,
                  warchest.ROW_BAG_SIZE):
        _swap(out, start, 1)
    out[:, warchest.ROW_INITIATIVE] = _flip_seat(out[:, warchest.ROW_INITIATIVE])
    out[:, warchest.ROW_TO_ACT] = _flip_seat(out[:, warchest.ROW_TO_ACT])

    for depth in range(warchest.CONT_CAP):
        start = warchest.ROW_STACK_OWED + depth * 8
        bits = np.unpackbits(rows[:, start:start + 8], axis=1, bitorder="little")
        mapped = np.zeros_like(bits)
        mapped[:, HEXMAP] = bits[:, :N_HEXES]
        out[:, start:start + 8] = np.packbits(
            mapped, axis=1, bitorder="little")[:, :8]
    return out


def _mirror_actions(actions):
    out = actions.copy()
    for column in (2, 3, 4):
        values = actions[:, column]
        valid = values < N_HEXES
        out[valid, column] = HEXMAP[values[valid]]
    return out


def augment(parts, rng):
    """Apply the symmetry independently to half of a gathered training batch.

    A config's segment is its ``seg`` label. Flipping that label exchanges the
    two seat segments without copying the ragged config arena, and keeps truth
    indices valid for calibration diagnostics.
    """
    rows, cc, cp, cw, cy, seg, policy = parts
    n = len(rows)
    mirrored = rng.random(n) < 0.5
    if not mirrored.any():
        return parts

    rows = rows.copy()
    rows[mirrored] = mirror_rows(rows[mirrored])
    config_rows = mirrored[seg // 2]
    seg = seg.copy()
    seg[config_rows] ^= 1
    cp = cp.copy()
    cp[config_rows] ^= 1

    pa, pact, pcrow, pcfg, pprob, parow = policy
    pa = pa.copy()
    if len(pa):
        action_rows = mirrored[parow]
        pa[action_rows] = _mirror_actions(pa[action_rows])
    policy = (pa, pact, pcrow, pcfg, pprob, parow)
    return rows, cc, cp, cw, cy, seg, policy
