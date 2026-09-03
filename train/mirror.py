import numpy as np

import warchest

N_HEXES = warchest.N_HEXES
NSLOT = warchest.NSLOT
NTYPE = warchest.NTYPE
HEXMAP = np.asarray(warchest.hex_mirror(), np.int64)
FLIP = np.asarray([1, 0, 2, 3], np.int64)


def mirror_rows(rows):
    packed = bytes(warchest.mirror_rows(np.ascontiguousarray(rows).ravel()))
    return np.frombuffer(packed, np.uint8).reshape(len(rows), -1)


def mirror_batch(parts, flipped):
    rows, cc, player, weight, target, seg, policy, control = parts
    flipped = np.asarray(flipped, dtype=bool)
    if flipped.shape != (len(rows),):
        raise ValueError("one symmetry choice is required per replay row")
    if not flipped.any():
        return parts

    rows = rows.copy()
    rows[flipped] = mirror_rows(rows[flipped])
    control = control.copy()
    control[flipped] = FLIP[control[flipped][:, HEXMAP]]
    turned = flipped[seg // 2]
    player = player.copy()
    player[turned] ^= 1
    seg = seg.copy()
    seg[turned] ^= 1

    desc, pact, pcrow, pcfg, probability, parow = policy
    desc = desc.copy()
    moved = flipped[parow]
    actions = desc[moved]
    for field in (1, 2):
        here = actions[:, field] < NTYPE
        actions[here, field] = (actions[here, field] + NSLOT) % NTYPE
    for field in (3, 4, 5):
        here = actions[:, field] < N_HEXES
        actions[here, field] = HEXMAP[actions[here, field]]
    desc[moved] = actions
    policy = (desc, pact, pcrow, pcfg, probability, parow)
    return rows, cc, player, weight, target, seg, policy, control
