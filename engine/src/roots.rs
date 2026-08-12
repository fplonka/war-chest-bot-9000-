//! Serialization for subgame roots: a public `State` plus both beliefs, as
//! collected during a training run for GPU tree sizing. A tool-only format —
//! explicit field-by-field little-endian writes, no dependencies, and a magic
//! + version so a mismatched reader fails loudly.

use std::io::{Read, Write};

use crate::board::N_HEXES;
use crate::rebel::{Belief, Config, NSLOT};
use crate::state::{Cont, ContStack, State, CONT_CAP, N_PLAYERS, N_ZONES};
use crate::units::N_UNITS;

pub const ROOTS_MAGIC: u32 = 0x5710_7207;
pub const ROOTS_VERSION: u32 = 2;

fn w8<W: Write>(w: &mut W, x: u8) -> std::io::Result<()> {
    w.write_all(&[x])
}
fn wu32<W: Write>(w: &mut W, x: u32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn wf32<W: Write>(w: &mut W, x: f32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn r8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn ru32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn rf32<R: Read>(r: &mut R) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_cont<W: Write>(w: &mut W, c: &Cont) -> std::io::Result<()> {
    use Cont::*;
    match *c {
        Draw { player } => {
            w8(w, 0)?;
            w8(w, player)
        }
        MainPlay => w8(w, 1),
        RoyalGuardChoice { defender, rg_hex } => {
            w8(w, 2)?;
            w8(w, defender)?;
            w8(w, rg_hex)
        }
        SwordsmanMove { hex } => {
            w8(w, 3)?;
            w8(w, hex)
        }
        BerserkerChain { hex, v2 } => {
            w8(w, 4)?;
            w8(w, hex)?;
            w8(w, v2 as u8)
        }
        FootmanManeuver { hexes } => {
            w8(w, 5)?;
            wu32(w, hexes.0 as u32)
        }
        CavalryAttack { hex } => {
            w8(w, 6)?;
            w8(w, hex)
        }
        MercenaryManeuver { hex } => {
            w8(w, 7)?;
            w8(w, hex)
        }
        FootmanInstantDeploy { coin } => {
            w8(w, 8)?;
            w8(w, coin)
        }
        WarriorPriestDraw { player, rg_hex } => {
            w8(w, 9)?;
            w8(w, player)?;
            w8(w, rg_hex)
        }
        WarriorPriestPlay { player, coin } => {
            w8(w, 10)?;
            w8(w, player)?;
            w8(w, coin)
        }
        _AttackPost { atk_hex } => {
            w8(w, 11)?;
            w8(w, atk_hex)
        }
    }
}

fn read_cont<R: Read>(r: &mut R) -> std::io::Result<Cont> {
    use Cont::*;
    Ok(match r8(r)? {
        0 => Draw { player: r8(r)? },
        1 => MainPlay,
        2 => RoyalGuardChoice {
            defender: r8(r)?,
            rg_hex: r8(r)?,
        },
        3 => SwordsmanMove { hex: r8(r)? },
        4 => BerserkerChain {
            hex: r8(r)?,
            v2: r8(r)? != 0,
        },
        5 => FootmanManeuver {
            hexes: crate::state::HexSet(ru32(r)? as u64),
        },
        6 => CavalryAttack { hex: r8(r)? },
        7 => MercenaryManeuver { hex: r8(r)? },
        8 => FootmanInstantDeploy { coin: r8(r)? },
        9 => WarriorPriestDraw {
            player: r8(r)?,
            rg_hex: r8(r)?,
        },
        10 => WarriorPriestPlay {
            player: r8(r)?,
            coin: r8(r)?,
        },
        11 => _AttackPost { atk_hex: r8(r)? },
        t => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad cont tag {t}"),
            ))
        }
    })
}

fn write_state<W: Write>(w: &mut W, s: &State) -> std::io::Result<()> {
    for h in 0..N_HEXES {
        w8(w, s.hex_type[h])?;
        w8(w, s.hex_owner[h])?;
        w8(w, s.hex_height[h])?;
        w8(w, s.loc_marker[h])?;
    }
    for p in 0..N_PLAYERS {
        for z in 0..N_ZONES {
            for u in 0..N_UNITS {
                w8(w, s.zones[p][z][u])?;
            }
        }
    }
    w8(w, s.markers_hand[0])?;
    w8(w, s.markers_hand[1])?;
    w8(w, s.initiative)?;
    w8(w, s.initiative_moved as u8)?;
    wu32(w, s.round as u32)?;
    w8(w, s.first_player)?;
    w8(w, s.active)?;
    w8(w, s.turns_taken[0])?;
    w8(w, s.turns_taken[1])?;
    wu32(w, s.main_plays as u32)?;
    w8(w, s.winner)?;
    w8(w, s.adjudicated_draw as u8)?;
    write_cont(w, &s.pending)?;
    w8(w, s.conts.len() as u8)?;
    let mut stack = s.conts;
    while let Some(c) = stack.pop() {
        write_cont(w, &c)?;
    }
    w8(w, s.wp_v2_triggered as u8)?;
    w8(w, s.interrupt as u8)?;
    Ok(())
}

fn read_state<R: Read>(r: &mut R) -> std::io::Result<State> {
    let mut s = State::blank(0);
    for h in 0..N_HEXES {
        s.hex_type[h] = r8(r)?;
        s.hex_owner[h] = r8(r)?;
        s.hex_height[h] = r8(r)?;
        s.loc_marker[h] = r8(r)?;
    }
    for p in 0..N_PLAYERS {
        for z in 0..N_ZONES {
            for u in 0..N_UNITS {
                s.zones[p][z][u] = r8(r)?;
            }
        }
    }
    s.markers_hand[0] = r8(r)?;
    s.markers_hand[1] = r8(r)?;
    s.initiative = r8(r)?;
    s.initiative_moved = r8(r)? != 0;
    s.round = ru32(r)? as u16;
    s.first_player = r8(r)?;
    s.active = r8(r)?;
    s.turns_taken[0] = r8(r)?;
    s.turns_taken[1] = r8(r)?;
    s.main_plays = ru32(r)? as u16;
    s.winner = r8(r)?;
    s.adjudicated_draw = r8(r)? != 0;
    s.pending = read_cont(r)?;
    let n = r8(r)? as usize;
    assert!(n <= CONT_CAP, "bad cont stack length");
    // Written in pop order (top first); rebuild by pushing in reverse.
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        items.push(read_cont(r)?);
    }
    s.conts = ContStack::default();
    for c in items.into_iter().rev() {
        s.conts.push(c);
    }
    s.wp_v2_triggered = r8(r)? != 0;
    s.interrupt = r8(r)? != 0;
    Ok(s)
}

fn write_config<W: Write>(w: &mut W, c: &Config) -> std::io::Result<()> {
    for k in 0..NSLOT {
        w8(w, c.hand[k])?;
    }
    for k in 0..NSLOT {
        w8(w, c.fd[k])?;
    }
    w8(w, c.pending_coin.map_or(0xff, |p| p))?;
    w8(w, c.queued_coin.map_or(0xff, |p| p))?;
    Ok(())
}

fn read_config<R: Read>(r: &mut R) -> std::io::Result<Config> {
    let mut c = Config::default();
    for k in 0..NSLOT {
        c.hand[k] = r8(r)?;
    }
    for k in 0..NSLOT {
        c.fd[k] = r8(r)?;
    }
    let p = r8(r)?;
    c.pending_coin = if p == 0xff { None } else { Some(p) };
    let q = r8(r)?;
    c.queued_coin = if q == 0xff { None } else { Some(q) };
    Ok(c)
}

/// Write one root.
pub fn write_root<W: Write>(w: &mut W, state: &State, bel: &[Belief; 2]) -> std::io::Result<()> {
    write_state(w, state)?;
    for p in 0..2 {
        wu32(w, bel[p].cfg.len() as u32)?;
        for c in &bel[p].cfg {
            write_config(w, c)?;
        }
        for x in &bel[p].p {
            wf32(w, *x)?;
        }
    }
    Ok(())
}

/// Read one root.
pub fn read_root<R: Read>(r: &mut R) -> std::io::Result<(State, [Belief; 2])> {
    let s = read_state(r)?;
    let mut bel = [Belief::default(), Belief::default()];
    for p in 0..2 {
        let n = ru32(r)? as usize;
        if n > 100_000 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "belief too large",
            ));
        }
        bel[p].cfg = (0..n)
            .map(|_| read_config(r))
            .collect::<std::io::Result<_>>()?;
        bel[p].p = (0..n).map(|_| rf32(r)).collect::<std::io::Result<_>>()?;
    }
    Ok((s, bel))
}

/// The whole file: magic, version, count, then roots.
pub fn write_roots<W: Write>(w: &mut W, roots: &[(State, [Belief; 2])]) -> std::io::Result<()> {
    wu32(w, ROOTS_MAGIC)?;
    wu32(w, ROOTS_VERSION)?;
    wu32(w, roots.len() as u32)?;
    for (s, b) in roots {
        write_root(w, s, b)?;
    }
    Ok(())
}

/// Load every root from a file.
pub fn read_roots<R: Read>(r: &mut R) -> std::io::Result<Vec<(State, [Belief; 2])>> {
    if ru32(r)? != ROOTS_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad roots magic",
        ));
    }
    if ru32(r)? != ROOTS_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad roots version",
        ));
    }
    let n = ru32(r)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_root(r)?);
    }
    Ok(out)
}
