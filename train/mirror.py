"""The board's 180-degree symmetry, as a transform on encoded rows.

Rotating the board by 180 degrees maps white's two starting locations exactly
onto black's and permutes the six neutral ones, so *rotate the board and swap
the two players* is an exact symmetry of War Chest: the rotated position is a
legal position with the seats exchanged, and its value is the original value
with the two players' roles exchanged.

That makes every training row usable twice. The measured problem with this value
network is not capacity -- five architectures spanning 2.6x in cost all landed
within 4% of each other -- but data: held-out error sits at 2.7x the training
error and stops improving while training error keeps falling. Doubling the
effective dataset for free is the direct answer to that, and unlike a bigger
network it costs nothing at generation time.

Applied on the fly to a fraction of each batch rather than stored, so the
replay buffer does not double in memory.

Only the public encoding is permuted here. A row's configs carry the seat they
belong to as a feature, so swapping the seats on that side is one bit flip per
config -- see `train.py`'s batch assembly.

Correctness
-----------
`self_check` asserts the properties that any indexing mistake would break:
the transform is an involution, quantities that must be invariant under a
rotation (whether a hex is a location, the round number, which kind of
continuation is pending) do not move, and quantities that must swap (each
player's board presence) actually do.
"""

import numpy as np

import warchest

N_HEXES = warchest.N_HEXES
HEX_CH = warchest.HEX_CH
NSLOT = warchest.NSLOT
PUBFEAT = warchest.PUBFEAT


def _feature_permutation():
    """Where each feature index moves to under rotate-plus-swap.

    Returns `perm` such that `mirrored[i] = original[perm[i]]`, plus the list of
    indices that are flipped (`x -> 1 - x`) rather than permuted: the two
    "is player 0" flags, which negate when the seats swap.
    """
    perm = np.arange(PUBFEAT, dtype=np.int64)
    hexmap = np.asarray(warchest.hex_mirror(), dtype=np.int64)

    # --- per-hex block: the hex moves, and within it the two owner-valued
    # channel pairs swap. Channel order is
    #   0,1 owner | 2 height | 3..3+NSLOT slot one-hot | +2 marker owner |
    #   +1 is-location | +1 pending mask
    mk = 3 + NSLOT               # first location-marker channel
    for h in range(N_HEXES):
        src, dst = h * HEX_CH, int(hexmap[h]) * HEX_CH
        for c in range(HEX_CH):
            perm[dst + c] = src + c
        # owner one-hot and location-marker owner both swap players. The slot
        # one-hot does *not* need touching: it indexes the owner's own slot
        # table, and the identity and card blocks swap alongside.
        perm[dst + 0], perm[dst + 1] = src + 1, src + 0
        perm[dst + mk], perm[dst + mk + 1] = src + mk + 1, src + mk

    def swap_pair(off, width):
        """Two consecutive per-player blocks of `width` exchange places."""
        for k in range(width):
            perm[off + k] = off + width + k
            perm[off + width + k] = off + k

    swap_pair(warchest.OFF_ZONES, warchest.ZONE_FEATS)
    swap_pair(warchest.OFF_IDENT, NSLOT * warchest.N_UNITS)
    swap_pair(warchest.OFF_CARDS, NSLOT * warchest.CARD_FEATS)
    swap_pair(warchest.OFF_PLAYER, warchest.PLAYER_SCALARS)

    # --- globals: round, plies-remaining, initiative-moved and the pending
    # blocks are all seat-independent and stay put. The two "is player 0" flags
    # are not permutations of anything -- they invert.
    g = warchest.OFF_GLOBAL
    flip = [g + 3, g + 4]  # active == 0, to_act == 0
    return perm, np.asarray(flip, dtype=np.int64)


PERM, FLIP = _feature_permutation()


def mirror_x(x):
    """Mirror a batch of public encodings. `x` is `[n, PUBFEAT]`."""
    out = x[:, PERM]
    out[:, FLIP] = 1.0 - out[:, FLIP]
    return out


def self_check(vx, n=512):
    """Assert the properties an indexing mistake would break."""
    x = vx[:n].astype(np.float32)
    mx = mirror_x(x)

    assert np.allclose(mirror_x(mx), x), "mirror is not an involution on features"

    # Whether a hex is a location is a property of the board, so a rotation
    # that is not a symmetry of the location set would move it.
    loc = slice(5 + NSLOT, None, HEX_CH)
    a = x[:, :N_HEXES * HEX_CH][:, loc]
    b = mx[:, :N_HEXES * HEX_CH][:, loc]
    assert np.array_equal(a, b), "the location map is not rotation-invariant"

    # Seat-independent globals must not move.
    g = warchest.OFF_GLOBAL
    for off, what in [(0, "round"), (1, "plies remaining"), (2, "initiative moved")]:
        assert np.allclose(x[:, g + off], mx[:, g + off]), f"{what} moved under the mirror"
    pk = slice(g + warchest.GLOBAL_SCALARS,
               g + warchest.GLOBAL_SCALARS + warchest.PEND_KINDS)
    assert np.array_equal(x[:, pk], mx[:, pk]), "the pending-kind one-hot moved"

    # Seat-dependent quantities must swap. Board presence is the clearest:
    # player 0's occupied hexes become player 1's.
    own0 = x[:, 0:N_HEXES * HEX_CH:HEX_CH].sum(axis=1)
    own1 = x[:, 1:N_HEXES * HEX_CH:HEX_CH].sum(axis=1)
    mown0 = mx[:, 0:N_HEXES * HEX_CH:HEX_CH].sum(axis=1)
    assert np.allclose(own1, mown0), "board occupancy did not swap players"

    # The flags that identify a seat must invert.
    assert np.allclose(x[:, g + 4], 1.0 - mx[:, g + 4]), "the to-act flag did not invert"
    return True
