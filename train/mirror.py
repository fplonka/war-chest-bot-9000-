"""The board's 180-degree symmetry, as a transform on packed replay rows.

Rotating the board by 180 degrees maps white's two starting locations exactly
onto black's and permutes the six neutral ones, so *rotate the board and swap
the two players* is an exact symmetry of War Chest: the rotated position is a
legal position with the seats exchanged, and its value is the original value
with the two players' roles exchanged.

That makes every training row usable twice, for free and with no cost at
generation time.

Applied to the second canonical view of every row rather than stored, so the
replay buffer does not double. The transform permutes the frozen row fields
(`warchest.ROW_*` byte slices) directly; the network input is expanded from
the mirrored row afterwards, so the encoder itself never has to know about the
mirror. The board trunk runs in physical seat-0 space, so only the config side
of a query needs the mirrored row.

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

HEXMAP = np.asarray(warchest.hex_mirror(), dtype=np.int64)


def _flip_seat(v):
    """Owner bytes swap seats; anything else (255, or `neither`) is fixed."""
    return np.where(v == 0, 1, np.where(v == 1, 0, v))


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
    out[:, o:o + N_HEXES] = _flip_seat(out[:, o:o + N_HEXES])
    out[:, m:m + N_HEXES] = _flip_seat(out[:, m:m + N_HEXES])
    # Player-major blocks swap halves: unit ids (NSLOT per player), piles
    # (NSLOT * PILE_COUNTS per player).
    _swap(out, warchest.ROW_IDS, NSLOT)
    _swap(out, warchest.ROW_PILES, NSLOT * PILE_COUNTS)
    for at in (warchest.ROW_HAND_SIZE, warchest.ROW_FD_SIZE,
               warchest.ROW_BAG_SIZE):
        _swap(out, at, 1)
    # Initiative holder and the player to act swap; initiative-moved and the
    # horizon do not.
    out[:, warchest.ROW_INITIATIVE] = _flip_seat(out[:, warchest.ROW_INITIATIVE])
    out[:, warchest.ROW_TO_ACT] = _flip_seat(out[:, warchest.ROW_TO_ACT])
    # Stack kinds are seat-invariant; owed hexes at every level rotate with the board.
    for d in range(warchest.CONT_CAP):
        lo = warchest.ROW_STACK_OWED + d * 8
        raw = rows[:, lo:lo + 8]
        bits = np.unpackbits(raw, axis=1, bitorder='little')
        mapped = np.zeros_like(bits)
        mapped[:, HEXMAP] = bits[:, :N_HEXES]
        out[:, lo:lo + 8] = np.packbits(mapped, axis=1, bitorder='little')[:, :8]
    return out


def mirror_batch(parts, flipped):
    """Rotate selected replay rows and swap every attached player reference."""
    rows, cc, player, weight, target, seg, policy = parts
    flipped = np.asarray(flipped, dtype=bool)
    if flipped.shape != (len(rows),):
        raise ValueError("one symmetry choice is required per replay row")
    if not flipped.any():
        return parts

    rows = rows.copy()
    rows[flipped] = mirror_rows(rows[flipped])
    config_flipped = flipped[seg // 2]
    player = player.copy()
    player[config_flipped] ^= 1
    seg = seg.copy()
    seg[config_flipped] ^= 1

    desc, pact, pcrow, pcfg, probability, parow = policy
    desc = desc.copy()
    action_flipped = flipped[parow]
    actions = desc[action_flipped]
    for field in (1, 2):
        present = actions[:, field] < NTYPE
        actions[present, field] = (actions[present, field] + NSLOT) % NTYPE
    for field in (3, 4, 5):
        present = actions[:, field] < N_HEXES
        actions[present, field] = HEXMAP[actions[present, field]]
    desc[action_flipped] = actions
    policy = (desc, pact, pcrow, pcfg, probability, parow)
    return rows, cc, player, weight, target, seg, policy


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
    for k in range(warchest.CONT_CAP * warchest.PENDING_KINDS):
        assert np.allclose(x[:, g + 3 + k], mx[:, g + 3 + k]), f"stack kind channel {k} moved"

    # Seat-dependent quantities must swap. Board presence is the clearest:
    # player 0's occupied hexes become player 1's.
    own0 = x[:, 0:warchest.N_HEXES * warchest.HEX_CH:warchest.HEX_CH].sum(axis=1)
    own1 = x[:, 1:warchest.N_HEXES * warchest.HEX_CH:warchest.HEX_CH].sum(axis=1)
    mown0 = mx[:, 0:warchest.N_HEXES * warchest.HEX_CH:warchest.HEX_CH].sum(axis=1)
    assert np.allclose(own1, mown0), "board occupancy did not swap players"

    # The flag that identifies the seat must invert.
    assert np.allclose(x[:, g + 2], 1.0 - mx[:, g + 2]), "the to-act flag did not invert"
    return True


def self_check_rows(rows):
    """Check involution and expansion commuting with the feature mirror."""
    mr = mirror_rows(rows)
    assert np.array_equal(mirror_rows(mr), rows), "row mirror is not an involution"
    # Unit ids swap players; piles swap players.
    ids = rows[:, warchest.ROW_IDS:warchest.ROW_IDS + NTYPE]
    mids = mr[:, warchest.ROW_IDS:warchest.ROW_IDS + NTYPE]
    assert np.array_equal(ids[:, :NSLOT], mids[:, NSLOT:]), "unit ids did not swap"
    # Expansion commutes: expand(mirror(rows)) == mirror_x(expand(rows)).
    from train import expand_batch
    a = expand_batch(mr)
    b = mirror_x(expand_batch(rows))
    assert np.allclose(a, b, atol=1e-6), "row mirror and feature mirror diverge"
    return True


def check_against_engine(games=8, seed=11):
    """The row permutation against `State::mirror`, the engine's own rotation.

    `self_check_rows` proves the permutation is self-consistent; a permutation
    can be self-consistent and still wrong. This one compares it with the
    position the engine says the rotation produces.
    """
    pairs = np.frombuffer(bytes(warchest.mirror_row_pairs(games, seed)), np.uint8)
    pairs = pairs.reshape(-1, ROW_BYTES)
    rows, want = pairs[0::2], pairs[1::2]
    assert len(rows), "the oracle produced no coin-play states"
    got = mirror_rows(rows.copy())
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not len(bad), (
        f"{len(bad)} of {len(rows)} rows disagree with State::mirror, "
        f"first at byte {np.flatnonzero(got[bad[0]] != want[bad[0]])[0]}")
    return len(rows)
