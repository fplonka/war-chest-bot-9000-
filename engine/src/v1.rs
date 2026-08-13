//! The encoding a pre-describer checkpoint was trained with.
//!
//! Frozen. The card describer changed the public encoding's width and layout,
//! which means a checkpoint from before it cannot read a row written after it —
//! and a gate is worthless if the new architecture cannot be played against the
//! pool it is meant to beat. So the old encoder stays, exactly as it was, and a
//! net carries the version it was trained with (`Mlp::v1`, keyed off `dims`).
//!
//! Nothing here is maintained or extended. Delete this module when the pool has
//! rotated past every checkpoint that needs it.

use crate::board::{board, NONE, N_HEXES};
use crate::rebel::{
    pending_kind, reserve, Ctx, GLOBAL_SCALARS, MAX_COINS, MAX_ROUND, NSLOT, PEND_KINDS, PEND_SLOT,
    PLAYER_SCALARS,
};
use crate::state::{Cont, State, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_SUPPLY};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

pub const HEX_CH_V1: usize = 2 + 1 + NSLOT + 2 + 1 + 1;
const HEX_BLOCK_V1: usize = N_HEXES * HEX_CH_V1;
const ZONE_FEATS_V1: usize = 4 * NSLOT;
const OFF_ZONES_V1: usize = HEX_BLOCK_V1;
const OFF_IDENT_V1: usize = OFF_ZONES_V1 + 2 * ZONE_FEATS_V1;
const OFF_CARDS_V1: usize = OFF_IDENT_V1 + 2 * NSLOT * N_UNITS;
const OFF_PLAYER_V1: usize = OFF_CARDS_V1 + 2 * NSLOT * CARD_FEATS;
const OFF_GLOBAL_V1: usize = OFF_PLAYER_V1 + 2 * PLAYER_SCALARS;
/// 972, against the current encoding's 957.
pub const PUBFEAT_V1: usize = OFF_GLOBAL_V1 + GLOBAL_SCALARS + PEND_KINDS + PEND_SLOT;

/// The public encoding as it was before the card describer.
pub fn write_public_features_v1(s: &State, ctx: &Ctx, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PUBFEAT_V1);
    out.fill(0.0);
    let bd = board();

    // Which board unit, if any, still owes a maneuver at this decision node.
    // `pending_kind` below says *what kind* of trigger is open; without this it
    // does not say *whose*, and the Footman tactic can owe two at once.
    //
    // Only the hex-valued payloads are encoded. The Warrior Priest's drawn coin
    // lives in a private zone, so encoding it would leak. `FootmanInstantDeploy`'s
    // coin is public (a Recruit reveals the unit taken) and is encoded with the
    // globals instead.
    let mut pending_hexes = crate::state::HexSet(0);
    let mark = |h: u8, set: &mut crate::state::HexSet| {
        if (h as usize) < N_HEXES {
            set.insert(h);
        }
    };
    match *s.pending() {
        Cont::SwordsmanMove { hex }
        | Cont::BerserkerChain { hex, .. }
        | Cont::CavalryAttack { hex }
        | Cont::MercenaryManeuver { hex } => mark(hex, &mut pending_hexes),
        Cont::FootmanManeuver { hexes } => {
            for h in hexes.iter() {
                mark(h, &mut pending_hexes);
            }
        }
        Cont::RoyalGuardChoice { rg_hex, .. } | Cont::WarriorPriestDraw { rg_hex, .. } => {
            mark(rg_hex, &mut pending_hexes)
        }
        Cont::_AttackPost { atk_hex } => mark(atk_hex, &mut pending_hexes),
        _ => {}
    }

    let mut i = 0;
    for h in 0..N_HEXES {
        let owner = s.hex_owner[h];
        if owner != NONE {
            out[i + owner as usize] = 1.0;
            // Divisor is the largest coin count on any card. Bolstering has no
            // height limit (RULES.md section 5) and heights of 4 and 5 are
            // ~20% of occupied-hex observations under random play, so the
            // previous /3 collapsed them onto the same value.
            out[i + 2] = s.hex_height[h] as f32 / 5.0;
            let k = ctx.slot_of[owner as usize][s.hex_type[h] as usize];
            if k >= 0 {
                out[i + 3 + k as usize] = 1.0;
            }
        }
        if s.loc_marker[h] != NONE {
            out[i + 3 + NSLOT + s.loc_marker[h] as usize] = 1.0;
        }
        out[i + 5 + NSLOT] = bd.is_location[h] as u8 as f32;
        out[i + 6 + NSLOT] = ((pending_hexes.0 >> h) & 1) as f32;
        i += HEX_CH_V1;
    }
    debug_assert_eq!(i, OFF_ZONES_V1);

    for p in 0..2usize {
        let res = reserve(s, p as u8, ctx);
        for k in 0..NSLOT {
            let u = ctx.slots[p][k] as usize;
            out[i + k] = res[k] as f32 / 5.0;
            out[i + NSLOT + k] = s.zones[p][Z_FACEUP][u] as f32 / 5.0;
            out[i + 2 * NSLOT + k] = s.zones[p][Z_SUPPLY][u] as f32 / 5.0;
            out[i + 3 * NSLOT + k] = s.zones[p][Z_ELIM][u] as f32 / 5.0;
        }
        i += ZONE_FEATS_V1;
    }
    debug_assert_eq!(i, OFF_IDENT_V1);

    for p in 0..2usize {
        for k in 0..NSLOT {
            out[i + ctx.slots[p][k] as usize] = 1.0;
            i += N_UNITS;
        }
    }
    debug_assert_eq!(i, OFF_CARDS_V1);

    // What each drafted card actually does. Under a fixed draft these are
    // constant across games and carry no information, but they are what lets a
    // draft the network has never seen be encoded at all, so they are the
    // prerequisite for `--random-draft`.
    for p in 0..2usize {
        for k in 0..NSLOT {
            write_card_features(ctx.slots[p][k], &mut out[i..i + CARD_FEATS]);
            i += CARD_FEATS;
        }
    }
    debug_assert_eq!(i, OFF_PLAYER_V1);

    for p in 0..2usize {
        let fd: u8 = s.zones[p][Z_FACEDOWN].iter().sum();
        out[i] = s.markers_hand[p] as f32 / 6.0;
        out[i + 1] = s.markers_on_board(p as u8) as f32 / 6.0;
        out[i + 2] = s.hand_size(p as u8) as f32 / 3.0;
        // Divisors are the true maxima, not estimates. A player's coins are
        // bounded by the whole reserve, and both of these saturated under the
        // previous /10 and /12: the face-down count reaches 14 and the bag
        // reaches 18, so the two most dynamic scalars in the encoding were
        // being clipped in exactly the late-game states that matter most.
        out[i + 3] = fd as f32 / MAX_COINS;
        out[i + 4] = s.bag_size(p as u8) as f32 / MAX_COINS;
        out[i + 5] = s.turns_taken[p] as f32 / 3.0;
        out[i + 6] = (s.initiative == p as u8) as u8 as f32;
        out[i + 7] = (s.first_player == p as u8) as u8 as f32;
        i += PLAYER_SCALARS;
    }
    debug_assert_eq!(i, OFF_GLOBAL_V1);

    out[i] = (s.round as f32 / MAX_ROUND).min(1.0);
    // plies_remaining: PBS values near the horizon are not well defined without it.
    let cap = crate::state::MAX_MAIN_PLAYS;
    out[i + 1] = (cap - s.main_plays.min(cap)) as f32 / cap as f32;
    out[i + 2] = s.initiative_moved as u8 as f32;
    out[i + 3] = (s.active == 0) as u8 as f32;
    out[i + 4] = (s.to_act() == 0) as u8 as f32;
    i += GLOBAL_SCALARS;
    out[i + pending_kind(s)] = 1.0;
    i += PEND_KINDS;
    // The coin a Footman-V2 instant deploy is holding. Public, unlike the
    // Warrior Priest's drawn coin (see the pending-mask note above).
    if let Cont::FootmanInstantDeploy { coin } = *s.pending() {
        let k = ctx.slot_of[s.to_act() as usize][coin as usize];
        if k >= 0 {
            out[i + k as usize] = 1.0;
        }
    }
    i += PEND_SLOT;
    debug_assert_eq!(i, PUBFEAT_V1);
}
