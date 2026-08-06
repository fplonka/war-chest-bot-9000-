"""The board's 180-degree symmetry, as a transform on packed replay rows.

Rotating the board by 180 degrees maps white's two starting locations exactly
onto black's and permutes the six neutral ones, so *rotate the board and swap
the two players* is an exact symmetry of War Chest: the rotated position is a
legal position with the seats exchanged, and its value is the original value
with the two players' roles exchanged.

That makes every training row usable twice, for free and with no cost at
generation time.

Applied on the fly to a fraction of each batch rather than stored, so the
replay buffer does not double. The transform permutes the frozen row fields
(`warchest.ROW_*` byte slices) directly; the network input is expanded from
the mirrored row afterwards, so the encoder itself never has to know about the
mirror.

Correctness
-----------
`self_check` asserts the properties that any indexing mistake would break:
the transform is an involution, quantities that must be invariant under a
rotation (whether a hex is a location, the round number) do not move, and
quantities that must swap (each player's board presence, the unit ids) do.
The strongest check: expansion and mirror commute — `expand(mirror(row))`
must equal the feature-level mirror of `expand(row)`.
"""

import numpy as np

import warchest

N_HEXES = warchest.N_HEXES
NTYPE = warchest.NTYPE
NSLOT = warchest.NSLOT
ROW_BYTES = warchest.ROW_BYTES
PILE_COUNTS = warchest.PILE_COUNTS
AUX = warchest.AUX
NONE = 255

HEXMAP = np.asarray(warchest.hex_mirror(), dtype=np.int64)

def _swap(a, lo, width):
    """Two consecutive per-player blocks of `width` exchange places."""
    tmp = a[:, lo:lo + width].copy()
    a[:, lo:lo + width] = a[:, lo + width:lo + 2 * width]
    a[:, lo + width:lo + 2 * width] = tmp


def mirror_rows(rows):
    """Mirror a batch of packed replay rows. `rows` is `[n, ROW_BYTES]` u8."""
    out = rows.copy()
    o = warchest.ROW_HEX_OWNER
    s = warchest.ROW_HEX_SLOT
    h = warchest.ROW_HEX_HEIGHT
    m = warchest.ROW_HEX_MARKER
    # Hex arrays permute under the rotation; owner and marker values swap
    # seats (0 <-> 1, NONE stays).
    for (lo, w) in [(o, N_HEXES), (s, N_HEXES), (h, N_HEXES), (m, N_HEXES)]:
        out[:, lo:lo + w] = rows[:, lo + HEXMAP]
    def flip(v):
        return np.where(v == 0, 1, np.where(v == 1, 0, v))
    out[:, o:o + N_HEXES] = flip(out[:, o:o + N_HEXES])
    out[:, m:m + N_HEXES] = flip(out[:, m:m + N_HEXES])
    # Player-major blocks swap halves: unit ids (NSLOT per player), piles
    # (NSLOT * PILE_COUNTS per player).
    _swap(out, warchest.ROW_IDS, NSLOT)
    _swap(out, warchest.ROW_PILES, NSLOT * PILE_COUNTS)
    # Initiative holder and the player to act swap; initiative-moved and the
    # horizon do not.
    out[:, warchest.ROW_INITIATIVE] = flip(out[:, warchest.ROW_INITIATIVE])
    out[:, warchest.ROW_TO_ACT] = flip(out[:, warchest.ROW_TO_ACT])
    # Aux targets name seats: the two marker counts swap, the result class
    # inverts (0 = white wins, 2 = black wins, 1 = neither), the initiative
    # flip is a flip either way.
    aux = out[:, warchest.ROW_AUX:warchest.ROW_AUX + 2 * AUX].view(np.float16).reshape(-1, AUX)
    aux = aux[:, [1, 0, 2, 3]].copy()
    aux[:, 3] = 2.0 - aux[:, 3]
    out[:, warchest.ROW_AUX:warchest.ROW_AUX + 2 * AUX] = aux.view(np.uint8).reshape(-1, 2 * AUX)
    return out


# The feature-level mirror of the *expanded* encoding, kept only as the
# oracle for `self_check`: the two mirrors must commute. Layout channels:
#   per hex: owner(2), height, marker(2), is-location, coin-type one-hot
#   piles(4 per type), card facts(25 per type), player scalars(6), globals(3)
def _feature_permutation():
    HEX_FACTS = warchest.HEX_FACTS
    HEX_CH = warchest.HEX_CH
    PUBFEAT = warchest.PUBFEAT
    PLAYER_SCALARS = warchest.PLAYER_SCALARS
    GLOBAL_SCALARS = warchest.GLOBAL_SCALARS
    CARD_FEATS = warchest.CARD_FEATS
    OFF_PILES = warchest.OFF_PILES
    OFF_CARDS = warchest.OFF_CARDS
    OFF_LOOSE = warchest.OFF_LOOSE
    perm = np.arange(PUBFEAT, dtype=np.int64)
    for h in range(N_HEXES):
        src, dst = h * HEX_CH, int(HEXMAP[h]) * HEX_CH
        for c in range(HEX_CH):
            perm[dst + c] = src + c
        perm[dst + 0], perm[dst + 1] = src + 1, src + 0
        perm[dst + 3], perm[dst + 4] = src + 4, src + 3
        for k in range(NSLOT):
            perm[dst + HEX_FACTS + k] = src + HEX_FACTS + NSLOT + k
            perm[dst + HEX_FACTS + NSLOT + k] = src + HEX_FACTS + k

    def swap_pair(off, width):
        for k in range(width):
            perm[off + k] = off + width + k
            perm[off + width + k] = off + k

    swap_pair(OFF_PILES, NSLOT * PILE_COUNTS)
    swap_pair(OFF_CARDS, NSLOT * CARD_FEATS)
    swap_pair(OFF_LOOSE, PLAYER_SCALARS)
    # Globals: plies and initiative-moved stay; to-act flips.
    g = OFF_LOOSE + 2 * PLAYER_SCALARS
    flip = [g + 2]
    return perm, np.asarray(flip, dtype=np.int64)


PERM, FLIP = _feature_permutation()


def mirror_x(x):
    """Mirror expanded feature rows. Only used as the oracle for `self_check`."""
    out = x[:, PERM]
    out[:, FLIP] = 1.0 - out[:, FLIP]
    return out


def self_check(vx, n=512):
    """Assert the properties an indexing mistake would break.

    `vx` is a batch of *expanded* rows (`[n, PUBFEAT]` f32) — the check
    verifies the row-level mirror through the encoder itself: rebuilding a
    mirror-expanded row must give the feature-level mirror.
    """
    x = vx[:n].astype(np.float32)
    mx = mirror_x(x)
    assert np.allclose(mirror_x(mx), x), "mirror is not an involution on features"

    # Whether a hex is a location is a property of the board, so a rotation
    # that is not a symmetry of the location set would move it.
    loc = slice(5, None, warchest.HEX_CH)
    a = x[:, :warchest.N_HEXES * warchest.HEX_CH][:, loc]
    b = mx[:, :warchest.N_HEXES * warchest.HEX_CH][:, loc]
    assert np.array_equal(a, b), "the location map is not rotation-invariant"

    # Seat-independent globals must not move.
    g = warchest.OFF_LOOSE + 2 * warchest.PLAYER_SCALARS
    for off, what in [(0, "plies remaining"), (1, "initiative moved")]:
        assert np.allclose(x[:, g + off], mx[:, g + off]), f"{what} moved under the mirror"

    # Seat-dependent quantities must swap. Board presence is the clearest:
    # player 0's occupied hexes become player 1's.
    own0 = x[:, 0:warchest.N_HEXES * warchest.HEX_CH:warchest.HEX_CH].sum(axis=1)
    own1 = x[:, 1:warchest.N_HEXES * warchest.HEX_CH:warchest.HEX_CH].sum(axis=1)
    mown0 = mx[:, 0:warchest.N_HEXES * warchest.HEX_CH:warchest.HEX_CH].sum(axis=1)
    assert np.allclose(own1, mown0), "board occupancy did not swap players"

    # The flag that identifies the seat must invert.
    assert np.allclose(x[:, g + 2], 1.0 - mx[:, g + 2]), "the to-act flag did not invert"
    return True


def self_check_rows(rows, cc, cp, seg):
    """The row-level checks: involution, and expansion commutes with the
    feature mirror."""
    mr = mirror_rows(rows)
    assert np.array_equal(mirror_rows(mr), rows), "row mirror is not an involution"
    # Unit ids swap players; piles swap players.
    ids = rows[:, warchest.ROW_IDS:warchest.ROW_IDS + NTYPE]
    mids = mr[:, warchest.ROW_IDS:warchest.ROW_IDS + NTYPE]
    assert np.array_equal(ids[:, :NSLOT], mids[:, NSLOT:]), "unit ids did not swap"
    # Expansion commutes: expand(mirror(rows)) == mirror_x(expand(rows)).
    from train import expand_batch, public_sizes
    n = len(rows)
    hand, fd, bag = public_sizes(cc, cp, seg, n)
    hand_m = hand[:, ::-1].copy()
    fd_m = fd[:, ::-1].copy()
    bag_m = bag[:, ::-1].copy()
    a = expand_batch(mr, hand_m, fd_m, bag_m)
    b = mirror_x(expand_batch(rows, hand, fd, bag))
    assert np.allclose(a, b, atol=1e-6), "row mirror and feature mirror diverge"
    return True
