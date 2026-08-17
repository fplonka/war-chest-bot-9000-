"""Device-side expansion of compact replay rows for the production trainer.

The replay ring deliberately stores raw bytes.  Expanding a 1,024-row batch
and up to ~100k ragged configurations on the contended host took 110--180 ms,
then copied the resulting 10+ MiB of floats to GPU 1.  These kernels copy the
compact bytes and expand them directly on the training device.  Encoder tables
come from the Rust module, so this is another implementation of the frozen
layout, not another source of game facts.
"""

from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

import mirror
import warchest


@triton.jit
def _expand_rows(
        rows, sizes, cards, locations, out,
        ROW_BYTES: tl.constexpr, PUBFEAT: tl.constexpr,
        N_HEXES: tl.constexpr, HEX_CH: tl.constexpr, HEX_FACTS: tl.constexpr,
        HEX_BLOCK: tl.constexpr, NTYPE: tl.constexpr, NSLOT: tl.constexpr,
        PILE_COUNTS: tl.constexpr, CARD_FEATS: tl.constexpr,
        OFF_PILES: tl.constexpr, OFF_CARDS: tl.constexpr,
        OFF_LOOSE: tl.constexpr, PLAYER_SCALARS: tl.constexpr,
        ROW_IDS: tl.constexpr, ROW_HEX_OWNER: tl.constexpr,
        ROW_HEX_SLOT: tl.constexpr, ROW_HEX_HEIGHT: tl.constexpr,
        ROW_HEX_MARKER: tl.constexpr, ROW_PILES: tl.constexpr,
        ROW_INITIATIVE: tl.constexpr, ROW_INIT_MOVED: tl.constexpr,
        ROW_TO_ACT: tl.constexpr, ROW_PLIES: tl.constexpr,
        MAX_COINS: tl.constexpr, MAX_PLIES: tl.constexpr,
        BLOCK: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    valid = j < PUBFEAT
    base = r * ROW_BYTES
    value = tl.zeros((BLOCK,), tl.float32)

    # Per-hex facts and the player/slot type one-hot.
    is_hex = j < HEX_BLOCK
    h = tl.minimum(j // HEX_CH, N_HEXES - 1)
    ch = j - h * HEX_CH
    owner = tl.load(rows + base + ROW_HEX_OWNER + h, mask=is_hex, other=255).to(tl.int32)
    slot = tl.load(rows + base + ROW_HEX_SLOT + h, mask=is_hex, other=255).to(tl.int32)
    height = tl.load(rows + base + ROW_HEX_HEIGHT + h, mask=is_hex, other=0).to(tl.float32)
    marker = tl.load(rows + base + ROW_HEX_MARKER + h, mask=is_hex, other=255).to(tl.int32)
    location = tl.load(locations + h, mask=is_hex, other=0).to(tl.float32)
    value = tl.where(is_hex & (ch == 0) & (owner == 0), 1.0, value)
    value = tl.where(is_hex & (ch == 1) & (owner == 1), 1.0, value)
    value = tl.where(is_hex & (ch == 2) & (owner != 255), height / 5.0, value)
    value = tl.where(is_hex & (ch == 3) & (marker == 0), 1.0, value)
    value = tl.where(is_hex & (ch == 4) & (marker == 1), 1.0, value)
    value = tl.where(is_hex & (ch == 5), location, value)
    one = ch - HEX_FACTS
    expected = owner * NSLOT + slot
    value = tl.where(
        is_hex & (one >= 0) & (one < NTYPE) & (owner < 2) &
        (slot < NSLOT) & (one == expected), 1.0, value)

    # Four public pile counts for each of the ten drafted card types.
    is_pile = (j >= OFF_PILES) & (j < OFF_CARDS)
    pi = tl.minimum(j - OFF_PILES, NTYPE * PILE_COUNTS - 1)
    pile = tl.load(rows + base + ROW_PILES + pi, mask=is_pile, other=0).to(tl.float32)
    value = tl.where(is_pile, pile / 5.0, value)

    # Static rulebook facts, indexed by the row's ten unit ids.
    is_card = (j >= OFF_CARDS) & (j < OFF_LOOSE)
    ci = tl.minimum(j - OFF_CARDS, NTYPE * CARD_FEATS - 1)
    typ = ci // CARD_FEATS
    fact = ci - typ * CARD_FEATS
    unit = tl.load(rows + base + ROW_IDS + typ, mask=is_card, other=0).to(tl.int32)
    card = tl.load(cards + unit * CARD_FEATS + fact, mask=is_card, other=0.0)
    value = tl.where(is_card, card, value)

    # Marker counts need a reduction over the 37 location-marker bytes.
    hh = tl.arange(0, 64)
    marks = tl.load(
        rows + base + ROW_HEX_MARKER + hh,
        mask=hh < N_HEXES, other=255).to(tl.int32)
    on0 = tl.sum((marks == 0).to(tl.int32), axis=0)
    on1 = tl.sum((marks == 1).to(tl.int32), axis=0)

    is_player = (j >= OFF_LOOSE) & (j < OFF_LOOSE + 2 * PLAYER_SCALARS)
    li = j - OFF_LOOSE
    player = tl.minimum(li // PLAYER_SCALARS, 1)
    field = li - player * PLAYER_SCALARS
    on_board = tl.where(player == 0, on0, on1).to(tl.float32)
    in_hand = 6.0 - on_board
    hand = tl.load(sizes + r * 6 + player, mask=is_player, other=0).to(tl.float32)
    fd = tl.load(sizes + r * 6 + 2 + player, mask=is_player, other=0).to(tl.float32)
    bag = tl.load(sizes + r * 6 + 4 + player, mask=is_player, other=0).to(tl.float32)
    initiative = tl.load(rows + base + ROW_INITIATIVE).to(tl.int32)
    initiative = tl.where(initiative <= 1, initiative, 0)
    pv = tl.where(field == 0, in_hand / 6.0, 0.0)
    pv = tl.where(field == 1, on_board / 6.0, pv)
    pv = tl.where(field == 2, hand / 3.0, pv)
    pv = tl.where(field == 3, fd / MAX_COINS, pv)
    pv = tl.where(field == 4, bag / MAX_COINS, pv)
    pv = tl.where(field == 5, (initiative == player).to(tl.float32), pv)
    value = tl.where(is_player, pv, value)

    g = j - (OFF_LOOSE + 2 * PLAYER_SCALARS)
    is_global = (g >= 0) & (g < 3)
    pl0 = tl.load(rows + base + ROW_PLIES).to(tl.int32)
    pl1 = tl.load(rows + base + ROW_PLIES + 1).to(tl.int32)
    plies = (pl0 + pl1 * 256).to(tl.float32) / MAX_PLIES
    moved = tl.load(rows + base + ROW_INIT_MOVED).to(tl.float32)
    to_act = tl.load(rows + base + ROW_TO_ACT).to(tl.int32)
    gv = tl.where(g == 0, plies, 0.0)
    gv = tl.where(g == 1, moved, gv)
    gv = tl.where(g == 2, (to_act == 0).to(tl.float32), gv)
    value = tl.where(is_global, gv, value)
    tl.store(out + r * PUBFEAT + j, value, mask=valid)


@lru_cache(maxsize=None)
def _tables(device_text):
    device = torch.device(device_text)
    cards = torch.as_tensor(
        np.asarray(warchest.card_features_table(), np.float32), device=device)
    locations = torch.as_tensor(
        np.asarray(warchest.hex_location_flags(), np.uint8), device=device)
    return cards, locations


def make_batch(parts, rng, device):
    """Compact replay batch -> two canonical query rows on ``device``."""
    del rng
    from train import public_sizes

    rows, aux, cc, cp, cw, cy, seg = parts
    n = len(rows)
    hand, fd, bag = public_sizes(cc, cp, seg, n)
    views = np.empty((2 * n, warchest.ROW_BYTES), np.uint8)
    views[0::2] = rows
    views[1::2] = mirror.mirror_rows(rows)
    # The trunk and its ownership head run once in physical seat-0 space.
    owner = aux
    size_parts = []
    for a in (hand, fd, bag):
        pair = np.empty((2 * n, 2), np.uint8)
        pair[0::2] = a
        pair[1::2] = a[:, ::-1]
        size_parts.append(pair)
    sizes = np.concatenate(size_parts, axis=1)
    rows = np.ascontiguousarray(views)
    cc = np.ascontiguousarray(cc)
    t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=device)
    rows_t = t(rows, torch.uint8)
    sizes_t = t(sizes, torch.uint8)
    cards, locations = _tables(str(device))
    x = torch.empty((2 * n, warchest.PUBFEAT), dtype=torch.float32, device=device)
    _expand_rows[(2 * n,)](
        rows_t, sizes_t, cards, locations, x,
        ROW_BYTES=warchest.ROW_BYTES, PUBFEAT=warchest.PUBFEAT,
        N_HEXES=warchest.N_HEXES, HEX_CH=warchest.HEX_CH,
        HEX_FACTS=warchest.HEX_FACTS, HEX_BLOCK=warchest.HEX_BLOCK,
        NTYPE=warchest.NTYPE, NSLOT=warchest.NSLOT,
        PILE_COUNTS=warchest.PILE_COUNTS, CARD_FEATS=warchest.CARD_FEATS,
        OFF_PILES=warchest.OFF_PILES, OFF_CARDS=warchest.OFF_CARDS,
        OFF_LOOSE=warchest.OFF_LOOSE, PLAYER_SCALARS=warchest.PLAYER_SCALARS,
        ROW_IDS=warchest.ROW_IDS, ROW_HEX_OWNER=warchest.ROW_HEX_OWNER,
        ROW_HEX_SLOT=warchest.ROW_HEX_SLOT, ROW_HEX_HEIGHT=warchest.ROW_HEX_HEIGHT,
        ROW_HEX_MARKER=warchest.ROW_HEX_MARKER, ROW_PILES=warchest.ROW_PILES,
        ROW_INITIATIVE=warchest.ROW_INITIATIVE,
        ROW_INIT_MOVED=warchest.ROW_INIT_MOVED, ROW_TO_ACT=warchest.ROW_TO_ACT,
        ROW_PLIES=warchest.ROW_PLIES, MAX_COINS=21.0,
        MAX_PLIES=warchest.MAX_MAIN_PLAYS, BLOCK=1024, num_warps=4)

    phi = t(cc, torch.float32) / float(warchest.CNORM)
    return (x, phi, t(cw, torch.float32), t(seg, torch.long),
            t(cy, torch.float32), t(owner, torch.int64), 2 * n)


def warmup(device):
    """Compile the expansion kernel before the run's wall-clock starts."""
    rows = np.zeros((1, warchest.ROW_BYTES), np.uint8)
    rows[:, warchest.ROW_HEX_OWNER:warchest.ROW_HEX_OWNER + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_SLOT:warchest.ROW_HEX_SLOT + warchest.N_HEXES] = 255
    rows[:, warchest.ROW_HEX_MARKER:warchest.ROW_HEX_MARKER + warchest.N_HEXES] = 255
    cc = np.zeros((2, warchest.CCOUNTS), np.uint8)
    aux = np.zeros((1, warchest.N_LOCATIONS), np.uint8)
    parts = (rows, aux, cc, np.asarray([0, 1], np.uint8),
             np.asarray([1.0, 1.0], np.float32), np.zeros(2, np.float32),
             np.asarray([0, 1], np.int64))
    make_batch(parts, np.random.default_rng(0), device)
    torch.cuda.synchronize(device)
