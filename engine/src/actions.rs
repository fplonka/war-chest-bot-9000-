//! The Action enum, its stable integer encoding, and human-readable Display.
//!
//! Encoding (u32), documented and stable:
//!   bits  0..5   kind   (0..31)
//!   bits  6..11  field a
//!   bits 12..17  field b
//!   bits 18..23  field c
//!   bits 24..29  field d
//! Field meaning depends on kind (see `encode`/`decode`). Unit indices (0..19)
//! and hex indices (0..36) both fit in 6 bits; NONE (255) is encoded as 63.
//!
//! `kind` values (stable — do not renumber):
//!   0 Deploy        a=unit b=hex
//!   1 Bolster       a=unit b=hex
//!   2 ClaimInit
//!   3 Recruit       a=unit
//!   4 Pass
//!   5 Move          a=from b=to
//!   6 Control       a=from
//!   7 Attack        a=from b=target
//!   8 TacticArcher      a=from b=target
//!   9 TacticCavalryMove a=from b=to     (cavalry step; attack chosen after)
//!  10 TacticCavalryAtk  a=from b=target (the follow-up attack)
//!  11 TacticCrossbow    a=from b=target
//!  12 TacticEnsign      a=from(target unit) b=to
//!  13 TacticLancer      a=from b=to c=target
//!  14 TacticLightCav    a=from b=to
//!  15 TacticMarshal     a=granter b=target-unit c=enemy-target
//!  16 TacticRoyalGuard  a=from b=to
//!  17 TacticFootmanMove a=from b=to
//!  18 TacticFootmanCtrl a=from
//!  19 TacticFootmanAtk  a=from b=target
//!  20 DrawCoin         a=unit           (chance outcome: which coin was drawn)
//!  21 RGSoakSupply                       (defender removes RG coin from supply)
//!  22 RGSoakStack                        (defender removes from the RG stack)
//!  23 SwordsmanMove    a=from b=to  (b=63 => decline)
//!  24 BerserkerMove    a=from b=to
//!  25 BerserkerCtrl    a=from
//!  26 BerserkerAttack  a=from b=target
//!  27 BerserkerStop
//!  28 FootmanDone            (used to end a footman maneuver segment as no-op? unused)
//!  29 MercMove        a=from b=to
//!  30 MercCtrl        a=from
//!  31 MercAttack      a=from b=target
//! Kinds >=32 spill into the high bit region; we keep a second byte via field d
//! for the remaining variants:
//!  see Kind2 encoded when kind==31 is insufficient. To stay simple we instead
//! widen `kind` to 6 bits (0..63) and shift fields up by one bit each.

pub const KIND_BITS: u32 = 6;
pub const FIELD_BITS: u32 = 6;
pub const FIELD_NONE: u32 = 63;

use crate::board::NONE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    // --- main coin plays ---
    Deploy { unit: u8, hex: u8 },
    Bolster { unit: u8, hex: u8 },
    // Facedown plays spend a specific coin from hand (`coin`), which matters for
    // future bag composition. Recruit also names the recruited supply type.
    ClaimInitiative { coin: u8 },
    Recruit { coin: u8, unit: u8 },
    Pass { coin: u8 },
    Move { from: u8, to: u8 },
    Control { from: u8 },
    Attack { from: u8, target: u8 },
    TacArcher { from: u8, target: u8 },
    TacCavalryMove { from: u8, to: u8 },
    TacCavalryAttack { from: u8, target: u8 },
    TacCrossbow { from: u8, target: u8 },
    TacEnsign { from: u8, to: u8 },
    TacLancer { from: u8, to: u8, target: u8 },
    TacLightCav { from: u8, to: u8 },
    TacMarshal { unit_hex: u8, target: u8 },
    TacRoyalGuard { from: u8, to: u8 },
    // Footman tactic: spend a footman coin (`coin`), then maneuver each footman
    // unit of that type on the board (a sequence of FootMove/FootControl/FootAttack).
    TacFootman { coin: u8 },
    // Footman tactic per-unit maneuvers:
    FootMove { from: u8, to: u8 },
    FootControl { from: u8 },
    FootAttack { from: u8, target: u8 },
    // --- chance ---
    DrawCoin { unit: u8 },
    // --- defender royal guard choice ---
    RGSoakSupply,
    RGSoakStack,
    // --- swordsman post-attack move ---
    SwordsmanMove { from: u8, to: u8 },
    SwordsmanDecline,
    // --- berserker chain ---
    BerserkMove { from: u8, to: u8 },
    BerserkControl { from: u8 },
    BerserkAttack { from: u8, target: u8 },
    BerserkStop,
    // --- mercenary free maneuver (after recruit) ---
    MercMove { from: u8, to: u8 },
    MercControl { from: u8 },
    MercAttack { from: u8, target: u8 },
    MercDecline,
    // --- footman v2 instant deploy ---
    FootmanInstantDeploy { hex: u8 },
    FootmanInstantDecline,
    // --- warrior priest forced play is expressed with the ordinary main-play
    //     variants above; no dedicated variant needed ---
    // --- ensign/marshal granted maneuver by the target unit ---
    GrantMove { from: u8, to: u8 },
    GrantAttack { from: u8, target: u8 },
    GrantControl { from: u8 },
}

// Stable kind numbers.
const K_DEPLOY: u32 = 0;
const K_BOLSTER: u32 = 1;
const K_CLAIM: u32 = 2;
const K_RECRUIT: u32 = 3;
const K_PASS: u32 = 4;
const K_MOVE: u32 = 5;
const K_CONTROL: u32 = 6;
const K_ATTACK: u32 = 7;
const K_ARCHER: u32 = 8;
const K_CAVMOVE: u32 = 9;
const K_CAVATK: u32 = 10;
const K_CROSSBOW: u32 = 11;
const K_ENSIGN: u32 = 12;
const K_LANCER: u32 = 13;
const K_LIGHTCAV: u32 = 14;
const K_MARSHAL: u32 = 15;
const K_ROYALGUARD: u32 = 16;
const K_FOOTMOVE: u32 = 17;
const K_FOOTCTRL: u32 = 18;
const K_FOOTATK: u32 = 19;
const K_DRAW: u32 = 20;
const K_RG_SUPPLY: u32 = 21;
const K_RG_STACK: u32 = 22;
const K_SWORD_MOVE: u32 = 23;
const K_SWORD_DECLINE: u32 = 24;
const K_BERS_MOVE: u32 = 25;
const K_BERS_CTRL: u32 = 26;
const K_BERS_ATK: u32 = 27;
const K_BERS_STOP: u32 = 28;
const K_MERC_MOVE: u32 = 29;
const K_MERC_CTRL: u32 = 30;
const K_MERC_ATK: u32 = 31;
const K_MERC_DECLINE: u32 = 32;
const K_FID_DEPLOY: u32 = 33;
const K_FID_DECLINE: u32 = 34;
const K_GRANT_MOVE: u32 = 35;
const K_GRANT_ATK: u32 = 36;
const K_GRANT_CTRL: u32 = 37;
const K_TAC_FOOTMAN: u32 = 38;

#[inline]
fn e(v: u8) -> u32 {
    if v == NONE {
        FIELD_NONE
    } else {
        v as u32
    }
}
#[inline]
fn pack(kind: u32, a: u32, b: u32, c: u32, d: u32) -> u32 {
    kind | (a << KIND_BITS)
        | (b << (KIND_BITS + FIELD_BITS))
        | (c << (KIND_BITS + 2 * FIELD_BITS))
        | (d << (KIND_BITS + 3 * FIELD_BITS))
}

impl Action {
    pub fn encode(&self) -> u32 {
        use Action::*;
        match *self {
            Deploy { unit, hex } => pack(K_DEPLOY, e(unit), e(hex), 0, 0),
            Bolster { unit, hex } => pack(K_BOLSTER, e(unit), e(hex), 0, 0),
            ClaimInitiative { coin } => pack(K_CLAIM, e(coin), 0, 0, 0),
            Recruit { coin, unit } => pack(K_RECRUIT, e(coin), e(unit), 0, 0),
            Pass { coin } => pack(K_PASS, e(coin), 0, 0, 0),
            Move { from, to } => pack(K_MOVE, e(from), e(to), 0, 0),
            Control { from } => pack(K_CONTROL, e(from), 0, 0, 0),
            Attack { from, target } => pack(K_ATTACK, e(from), e(target), 0, 0),
            TacArcher { from, target } => pack(K_ARCHER, e(from), e(target), 0, 0),
            TacCavalryMove { from, to } => pack(K_CAVMOVE, e(from), e(to), 0, 0),
            TacCavalryAttack { from, target } => pack(K_CAVATK, e(from), e(target), 0, 0),
            TacCrossbow { from, target } => pack(K_CROSSBOW, e(from), e(target), 0, 0),
            TacEnsign { from, to } => pack(K_ENSIGN, e(from), e(to), 0, 0),
            TacLancer { from, to, target } => pack(K_LANCER, e(from), e(to), e(target), 0),
            TacLightCav { from, to } => pack(K_LIGHTCAV, e(from), e(to), 0, 0),
            TacMarshal { unit_hex, target } => pack(K_MARSHAL, e(unit_hex), e(target), 0, 0),
            TacRoyalGuard { from, to } => pack(K_ROYALGUARD, e(from), e(to), 0, 0),
            TacFootman { coin } => pack(K_TAC_FOOTMAN, e(coin), 0, 0, 0),
            FootMove { from, to } => pack(K_FOOTMOVE, e(from), e(to), 0, 0),
            FootControl { from } => pack(K_FOOTCTRL, e(from), 0, 0, 0),
            FootAttack { from, target } => pack(K_FOOTATK, e(from), e(target), 0, 0),
            DrawCoin { unit } => pack(K_DRAW, e(unit), 0, 0, 0),
            RGSoakSupply => pack(K_RG_SUPPLY, 0, 0, 0, 0),
            RGSoakStack => pack(K_RG_STACK, 0, 0, 0, 0),
            SwordsmanMove { from, to } => pack(K_SWORD_MOVE, e(from), e(to), 0, 0),
            SwordsmanDecline => pack(K_SWORD_DECLINE, 0, 0, 0, 0),
            BerserkMove { from, to } => pack(K_BERS_MOVE, e(from), e(to), 0, 0),
            BerserkControl { from } => pack(K_BERS_CTRL, e(from), 0, 0, 0),
            BerserkAttack { from, target } => pack(K_BERS_ATK, e(from), e(target), 0, 0),
            BerserkStop => pack(K_BERS_STOP, 0, 0, 0, 0),
            MercMove { from, to } => pack(K_MERC_MOVE, e(from), e(to), 0, 0),
            MercControl { from } => pack(K_MERC_CTRL, e(from), 0, 0, 0),
            MercAttack { from, target } => pack(K_MERC_ATK, e(from), e(target), 0, 0),
            MercDecline => pack(K_MERC_DECLINE, 0, 0, 0, 0),
            FootmanInstantDeploy { hex } => pack(K_FID_DEPLOY, e(hex), 0, 0, 0),
            FootmanInstantDecline => pack(K_FID_DECLINE, 0, 0, 0, 0),
            GrantMove { from, to } => pack(K_GRANT_MOVE, e(from), e(to), 0, 0),
            GrantAttack { from, target } => pack(K_GRANT_ATK, e(from), e(target), 0, 0),
            GrantControl { from } => pack(K_GRANT_CTRL, e(from), 0, 0, 0),
        }
    }

    pub fn decode(code: u32) -> Option<Action> {
        use Action::*;
        let kind = code & ((1 << KIND_BITS) - 1);
        let a = (code >> KIND_BITS) & ((1 << FIELD_BITS) - 1);
        let b = (code >> (KIND_BITS + FIELD_BITS)) & ((1 << FIELD_BITS) - 1);
        let c = (code >> (KIND_BITS + 2 * FIELD_BITS)) & ((1 << FIELD_BITS) - 1);
        let d8 = |v: u32| -> u8 {
            if v == FIELD_NONE {
                NONE
            } else {
                v as u8
            }
        };
        let _ = d8; // used below
        let f = |v: u32| -> u8 {
            if v == FIELD_NONE {
                NONE
            } else {
                v as u8
            }
        };
        Some(match kind {
            K_DEPLOY => Deploy {
                unit: f(a),
                hex: f(b),
            },
            K_BOLSTER => Bolster {
                unit: f(a),
                hex: f(b),
            },
            K_CLAIM => ClaimInitiative { coin: f(a) },
            K_RECRUIT => Recruit {
                coin: f(a),
                unit: f(b),
            },
            K_PASS => Pass { coin: f(a) },
            K_MOVE => Move {
                from: f(a),
                to: f(b),
            },
            K_CONTROL => Control { from: f(a) },
            K_ATTACK => Attack {
                from: f(a),
                target: f(b),
            },
            K_ARCHER => TacArcher {
                from: f(a),
                target: f(b),
            },
            K_CAVMOVE => TacCavalryMove {
                from: f(a),
                to: f(b),
            },
            K_CAVATK => TacCavalryAttack {
                from: f(a),
                target: f(b),
            },
            K_CROSSBOW => TacCrossbow {
                from: f(a),
                target: f(b),
            },
            K_ENSIGN => TacEnsign {
                from: f(a),
                to: f(b),
            },
            K_LANCER => TacLancer {
                from: f(a),
                to: f(b),
                target: f(c),
            },
            K_LIGHTCAV => TacLightCav {
                from: f(a),
                to: f(b),
            },
            K_MARSHAL => TacMarshal {
                unit_hex: f(a),
                target: f(b),
            },
            K_ROYALGUARD => TacRoyalGuard {
                from: f(a),
                to: f(b),
            },
            K_TAC_FOOTMAN => TacFootman { coin: f(a) },
            K_FOOTMOVE => FootMove {
                from: f(a),
                to: f(b),
            },
            K_FOOTCTRL => FootControl { from: f(a) },
            K_FOOTATK => FootAttack {
                from: f(a),
                target: f(b),
            },
            K_DRAW => DrawCoin { unit: f(a) },
            K_RG_SUPPLY => RGSoakSupply,
            K_RG_STACK => RGSoakStack,
            K_SWORD_MOVE => SwordsmanMove {
                from: f(a),
                to: f(b),
            },
            K_SWORD_DECLINE => SwordsmanDecline,
            K_BERS_MOVE => BerserkMove {
                from: f(a),
                to: f(b),
            },
            K_BERS_CTRL => BerserkControl { from: f(a) },
            K_BERS_ATK => BerserkAttack {
                from: f(a),
                target: f(b),
            },
            K_BERS_STOP => BerserkStop,
            K_MERC_MOVE => MercMove {
                from: f(a),
                to: f(b),
            },
            K_MERC_CTRL => MercControl { from: f(a) },
            K_MERC_ATK => MercAttack {
                from: f(a),
                target: f(b),
            },
            K_MERC_DECLINE => MercDecline,
            K_FID_DEPLOY => FootmanInstantDeploy { hex: f(a) },
            K_FID_DECLINE => FootmanInstantDecline,
            K_GRANT_MOVE => GrantMove {
                from: f(a),
                to: f(b),
            },
            K_GRANT_ATK => GrantAttack {
                from: f(a),
                target: f(b),
            },
            K_GRANT_CTRL => GrantControl { from: f(a) },
            _ => return None,
        })
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::board::board;
        use crate::units::def;
        let b = board();
        let h = |x: u8| -> String {
            if x == NONE {
                "-".into()
            } else {
                b.coord_str(x as usize)
            }
        };
        let u = |x: u8| -> &'static str {
            if x == NONE {
                "-"
            } else {
                def(x).name
            }
        };
        use Action::*;
        match *self {
            Deploy { unit, hex } => write!(out, "Deploy {} @ {}", u(unit), h(hex)),
            Bolster { unit, hex } => write!(out, "Bolster {} @ {}", u(unit), h(hex)),
            ClaimInitiative { coin } => write!(out, "ClaimInitiative (spend {})", u(coin)),
            Recruit { coin, unit } => write!(out, "Recruit {} (spend {})", u(unit), u(coin)),
            Pass { coin } => write!(out, "Pass (spend {})", u(coin)),
            Move { from, to } => write!(out, "Move {}->{}", h(from), h(to)),
            Control { from } => write!(out, "Control @ {}", h(from)),
            Attack { from, target } => write!(out, "Attack {}->{}", h(from), h(target)),
            TacArcher { from, target } => write!(out, "Archer {}=>{}", h(from), h(target)),
            TacCavalryMove { from, to } => write!(out, "Cavalry move {}->{}", h(from), h(to)),
            TacCavalryAttack { from, target } => {
                write!(out, "Cavalry attack {}->{}", h(from), h(target))
            }
            TacCrossbow { from, target } => write!(out, "Crossbow {}=>{}", h(from), h(target)),
            TacEnsign { from, to } => write!(out, "Ensign moves {}->{}", h(from), h(to)),
            TacLancer { from, to, target } => {
                write!(out, "Lancer {}->{} atk {}", h(from), h(to), h(target))
            }
            TacLightCav { from, to } => write!(out, "LightCav {}->{}", h(from), h(to)),
            TacMarshal { unit_hex, target } => {
                write!(out, "Marshal: {} attacks {}", h(unit_hex), h(target))
            }
            TacRoyalGuard { from, to } => write!(out, "RoyalGuard {}->{}", h(from), h(to)),
            TacFootman { coin } => write!(out, "Footman tactic ({})", u(coin)),
            FootMove { from, to } => write!(out, "Footman move {}->{}", h(from), h(to)),
            FootControl { from } => write!(out, "Footman control @ {}", h(from)),
            FootAttack { from, target } => write!(out, "Footman attack {}->{}", h(from), h(target)),
            DrawCoin { unit } => write!(out, "Draw {}", u(unit)),
            RGSoakSupply => write!(out, "RG soak from supply"),
            RGSoakStack => write!(out, "RG soak from stack"),
            SwordsmanMove { from, to } => write!(out, "Swordsman move {}->{}", h(from), h(to)),
            SwordsmanDecline => write!(out, "Swordsman decline"),
            BerserkMove { from, to } => write!(out, "Berserk move {}->{}", h(from), h(to)),
            BerserkControl { from } => write!(out, "Berserk control @ {}", h(from)),
            BerserkAttack { from, target } => {
                write!(out, "Berserk attack {}->{}", h(from), h(target))
            }
            BerserkStop => write!(out, "Berserk stop"),
            MercMove { from, to } => write!(out, "Merc move {}->{}", h(from), h(to)),
            MercControl { from } => write!(out, "Merc control @ {}", h(from)),
            MercAttack { from, target } => write!(out, "Merc attack {}->{}", h(from), h(target)),
            MercDecline => write!(out, "Merc decline"),
            FootmanInstantDeploy { hex } => write!(out, "Footman instant deploy @ {}", h(hex)),
            FootmanInstantDecline => write!(out, "Footman instant decline"),
            GrantMove { from, to } => write!(out, "Granted move {}->{}", h(from), h(to)),
            GrantAttack { from, target } => {
                write!(out, "Granted attack {}->{}", h(from), h(target))
            }
            GrantControl { from } => write!(out, "Granted control @ {}", h(from)),
        }
    }
}
