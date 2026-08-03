//! pyo3 bindings: the `warchest` python module.
//!
//! Action dicts use a stable `kind` string plus named params and an `actor`
//! (the player to act on that action, 0 = white, 1 = black). `apply` accepts an
//! action dict; only `kind` and its params are required (actor is ignored on
//! input). `state_dict` dumps the full state as a nested dict for verification.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::state::*;
use crate::units::{def, index_of_id, N_UNITS};

#[pyclass]
struct Game {
    s: State,
}

// --- action <-> dict ---------------------------------------------------------

fn hexopt(v: u8) -> Option<u8> {
    if v == NONE {
        None
    } else {
        Some(v)
    }
}

/// Serialize an action to a python dict: {kind, actor, ...params}.
fn action_to_dict(py: Python<'_>, s: &State, a: &Action) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("actor", s.to_act())?;
    d.set_item("code", a.encode())?;
    d.set_item("label", format!("{}", a))?;
    use Action::*;
    let set2 = |k: &str, x: u8| d.set_item(k, hexopt(x));
    match *a {
        Deploy { unit, hex } => {
            d.set_item("kind", "deploy")?;
            d.set_item("unit", unit)?;
            set2("hex", hex)?;
        }
        Bolster { unit, hex } => {
            d.set_item("kind", "bolster")?;
            d.set_item("unit", unit)?;
            set2("hex", hex)?;
        }
        ClaimInitiative { coin } => {
            d.set_item("kind", "claim_initiative")?;
            d.set_item("coin", coin)?;
        }
        Recruit { coin, unit } => {
            d.set_item("kind", "recruit")?;
            d.set_item("coin", coin)?;
            d.set_item("unit", unit)?;
        }
        Pass { coin } => {
            d.set_item("kind", "pass")?;
            d.set_item("coin", coin)?;
        }
        Move { from, to } => {
            d.set_item("kind", "move")?;
            set2("from", from)?;
            set2("to", to)?;
        }
        Control { from } => {
            d.set_item("kind", "control")?;
            set2("from", from)?;
        }
        Attack { from, target } => {
            d.set_item("kind", "attack")?;
            set2("from", from)?;
            set2("target", target)?;
        }
        TacArcher { from, target } => tac(&d, "archer", from, Some(target), None)?,
        TacCavalryMove { from, to } => tac(&d, "cavalry_move", from, None, Some(to))?,
        TacCavalryAttack { from, target } => tac(&d, "cavalry_attack", from, Some(target), None)?,
        TacCrossbow { from, target } => tac(&d, "crossbow", from, Some(target), None)?,
        TacEnsign { from, to } => tac(&d, "ensign", from, None, Some(to))?,
        TacLancer { from, to, target } => {
            d.set_item("kind", "lancer")?;
            set2("from", from)?;
            set2("to", to)?;
            set2("target", target)?;
        }
        TacLightCav { from, to } => tac(&d, "light_cavalry", from, None, Some(to))?,
        TacMarshal { unit_hex, target } => {
            d.set_item("kind", "marshal")?;
            set2("from", unit_hex)?;
            set2("target", target)?;
        }
        TacRoyalGuard { from, to } => tac(&d, "royal_guard", from, None, Some(to))?,
        TacFootman { coin } => {
            d.set_item("kind", "footman_tactic")?;
            d.set_item("coin", coin)?;
        }
        FootMove { from, to } => tac(&d, "foot_move", from, None, Some(to))?,
        FootControl { from } => tac(&d, "foot_control", from, None, None)?,
        FootAttack { from, target } => tac(&d, "foot_attack", from, Some(target), None)?,
        DrawCoin { unit } => {
            d.set_item("kind", "draw")?;
            d.set_item("unit", hexopt(unit))?;
        }
        RGSoakSupply => {
            d.set_item("kind", "rg_soak_supply")?;
        }
        RGSoakStack => {
            d.set_item("kind", "rg_soak_stack")?;
        }
        SwordsmanMove { from, to } => tac(&d, "swordsman_move", from, None, Some(to))?,
        SwordsmanDecline => {
            d.set_item("kind", "swordsman_decline")?;
        }
        BerserkMove { from, to } => tac(&d, "berserk_move", from, None, Some(to))?,
        BerserkControl { from } => tac(&d, "berserk_control", from, None, None)?,
        BerserkAttack { from, target } => tac(&d, "berserk_attack", from, Some(target), None)?,
        BerserkStop => {
            d.set_item("kind", "berserk_stop")?;
        }
        MercMove { from, to } => tac(&d, "merc_move", from, None, Some(to))?,
        MercControl { from } => tac(&d, "merc_control", from, None, None)?,
        MercAttack { from, target } => tac(&d, "merc_attack", from, Some(target), None)?,
        MercDecline => {
            d.set_item("kind", "merc_decline")?;
        }
        FootmanInstantDeploy { hex } => {
            d.set_item("kind", "footman_instant_deploy")?;
            set2("hex", hex)?;
        }
        FootmanInstantDecline => {
            d.set_item("kind", "footman_instant_decline")?;
        }
        GrantMove { from, to } => tac(&d, "grant_move", from, None, Some(to))?,
        GrantAttack { from, target } => tac(&d, "grant_attack", from, Some(target), None)?,
        GrantControl { from } => tac(&d, "grant_control", from, None, None)?,
    }
    Ok(d.into())
}

fn tac(
    d: &Bound<'_, PyDict>,
    kind: &str,
    from: u8,
    target: Option<u8>,
    to: Option<u8>,
) -> PyResult<()> {
    d.set_item("kind", kind)?;
    d.set_item("from", hexopt(from))?;
    if let Some(t) = target {
        d.set_item("target", hexopt(t))?;
    }
    if let Some(t) = to {
        d.set_item("to", hexopt(t))?;
    }
    Ok(())
}

/// Parse an action dict back into an Action. Accepts either a bare `code`
/// (fast path) or `kind` + params.
fn action_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Action> {
    // Fast path: an explicit integer code.
    if let Ok(Some(code)) = d.get_item("code") {
        if let Ok(c) = code.extract::<u32>() {
            if let Some(a) = Action::decode(c) {
                return Ok(a);
            }
        }
    }
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("action dict needs 'kind' or 'code'")
        })?
        .extract()?;
    let g = |k: &str| -> Option<u8> {
        d.get_item(k)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<u8>().ok())
    };
    let gh = |k: &str| -> u8 { g(k).unwrap_or(NONE) };
    let a = match kind.as_str() {
        "deploy" => Action::Deploy {
            unit: gh("unit"),
            hex: gh("hex"),
        },
        "bolster" => Action::Bolster {
            unit: gh("unit"),
            hex: gh("hex"),
        },
        "claim_initiative" => Action::ClaimInitiative { coin: gh("coin") },
        "recruit" => Action::Recruit {
            coin: gh("coin"),
            unit: gh("unit"),
        },
        "pass" => Action::Pass { coin: gh("coin") },
        "move" => Action::Move {
            from: gh("from"),
            to: gh("to"),
        },
        "control" => Action::Control { from: gh("from") },
        "attack" => Action::Attack {
            from: gh("from"),
            target: gh("target"),
        },
        "archer" => Action::TacArcher {
            from: gh("from"),
            target: gh("target"),
        },
        "cavalry_move" => Action::TacCavalryMove {
            from: gh("from"),
            to: gh("to"),
        },
        "cavalry_attack" => Action::TacCavalryAttack {
            from: gh("from"),
            target: gh("target"),
        },
        "crossbow" => Action::TacCrossbow {
            from: gh("from"),
            target: gh("target"),
        },
        "ensign" => Action::TacEnsign {
            from: gh("from"),
            to: gh("to"),
        },
        "lancer" => Action::TacLancer {
            from: gh("from"),
            to: gh("to"),
            target: gh("target"),
        },
        "light_cavalry" => Action::TacLightCav {
            from: gh("from"),
            to: gh("to"),
        },
        "marshal" => Action::TacMarshal {
            unit_hex: gh("from"),
            target: gh("target"),
        },
        "royal_guard" => Action::TacRoyalGuard {
            from: gh("from"),
            to: gh("to"),
        },
        "footman_tactic" => Action::TacFootman { coin: gh("coin") },
        "foot_move" => Action::FootMove {
            from: gh("from"),
            to: gh("to"),
        },
        "foot_control" => Action::FootControl { from: gh("from") },
        "foot_attack" => Action::FootAttack {
            from: gh("from"),
            target: gh("target"),
        },
        "draw" => Action::DrawCoin { unit: gh("unit") },
        "rg_soak_supply" => Action::RGSoakSupply,
        "rg_soak_stack" => Action::RGSoakStack,
        "swordsman_move" => Action::SwordsmanMove {
            from: gh("from"),
            to: gh("to"),
        },
        "swordsman_decline" => Action::SwordsmanDecline,
        "berserk_move" => Action::BerserkMove {
            from: gh("from"),
            to: gh("to"),
        },
        "berserk_control" => Action::BerserkControl { from: gh("from") },
        "berserk_attack" => Action::BerserkAttack {
            from: gh("from"),
            target: gh("target"),
        },
        "berserk_stop" => Action::BerserkStop,
        "merc_move" => Action::MercMove {
            from: gh("from"),
            to: gh("to"),
        },
        "merc_control" => Action::MercControl { from: gh("from") },
        "merc_attack" => Action::MercAttack {
            from: gh("from"),
            target: gh("target"),
        },
        "merc_decline" => Action::MercDecline,
        "footman_instant_deploy" => Action::FootmanInstantDeploy { hex: gh("hex") },
        "footman_instant_decline" => Action::FootmanInstantDecline,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown action kind '{}'",
                other
            )))
        }
    };
    Ok(a)
}

// --- state dict --------------------------------------------------------------

fn zones_dict(py: Python<'_>, s: &State, p: u8) -> PyResult<PyObject> {
    let names = [
        "bag",
        "hand",
        "faceup_discard",
        "facedown_discard",
        "supply",
        "eliminated",
    ];
    let d = PyDict::new_bound(py);
    for (zi, name) in names.iter().enumerate() {
        let zd = PyDict::new_bound(py);
        for u in 0..N_UNITS {
            let c = s.zones[p as usize][zi][u];
            if c > 0 {
                zd.set_item(def(u as u8).id, c)?;
            }
        }
        d.set_item(*name, zd)?;
    }
    Ok(d.into())
}

fn state_to_dict(py: Python<'_>, s: &State) -> PyResult<PyObject> {
    let b = board();
    let d = PyDict::new_bound(py);
    d.set_item("round", s.round)?;
    d.set_item("main_plays", s.main_plays)?;
    d.set_item("active", s.active)?;
    d.set_item("first_player", s.first_player)?;
    d.set_item("initiative", s.initiative)?;
    d.set_item("initiative_moved", s.initiative_moved)?;
    d.set_item("to_act", s.to_act())?;
    d.set_item("is_chance", s.is_chance())?;
    d.set_item("terminal", s.is_terminal())?;
    d.set_item("adjudicated_draw", s.adjudicated_draw)?;
    d.set_item("winner", s.winner().map(|w| w as i32).unwrap_or(-1))?;
    d.set_item("pending", format!("{:?}", s.pending()))?;

    // board occupancy: list of {hex, coord, unit, owner, height}.
    let occ = PyList::empty_bound(py);
    for h in 0..N_HEXES {
        if s.hex_type[h] != NONE {
            let e = PyDict::new_bound(py);
            e.set_item("hex", h)?;
            e.set_item("coord", b.coord_str(h))?;
            e.set_item("unit", def(s.hex_type[h]).id)?;
            e.set_item("owner", s.hex_owner[h])?;
            e.set_item("height", s.hex_height[h])?;
            occ.append(e)?;
        }
    }
    d.set_item("board", occ)?;

    // control markers: {coord: owner}.
    let marks = PyDict::new_bound(py);
    for h in 0..N_HEXES {
        if s.loc_marker[h] != NONE {
            marks.set_item(b.coord_str(h), s.loc_marker[h])?;
        }
    }
    d.set_item("markers", marks)?;

    d.set_item("markers_hand", vec![s.markers_hand[0], s.markers_hand[1]])?;
    d.set_item(
        "markers_on_board",
        vec![s.markers_on_board(0), s.markers_on_board(1)],
    )?;

    let players = PyList::empty_bound(py);
    for p in 0..2u8 {
        players.append(zones_dict(py, s, p)?)?;
    }
    d.set_item("zones", players)?;
    Ok(d.into())
}

// --- Game --------------------------------------------------------------------

#[pymethods]
impl Game {
    /// Construct from a draft dict:
    /// {white_units:[4 ids], black_units:[4 ids], first_player:"white"|"black"}.
    #[staticmethod]
    fn new(draft: &Bound<'_, PyDict>) -> PyResult<Game> {
        let get_units = |key: &str| -> PyResult<Vec<u16>> {
            let v = draft.get_item(key)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("draft needs '{}'", key))
            })?;
            let ids: Vec<u16> = v.extract()?;
            if ids.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "'{}' must have 4 unitTypeIds",
                    key
                )));
            }
            for &id in &ids {
                if index_of_id(id).is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unitTypeId {} is out of scope",
                        id
                    )));
                }
            }
            Ok(ids)
        };
        let white = get_units("white_units")?;
        let black = get_units("black_units")?;
        let fp: String = draft
            .get_item("first_player")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("draft needs 'first_player'"))?
            .extract()?;
        let first = match fp.as_str() {
            "white" => WHITE,
            "black" => BLACK,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "first_player must be 'white' or 'black'",
                ))
            }
        };
        Ok(Game {
            s: State::from_draft(&white, &black, first),
        })
    }

    /// List legal actions for whoever is to act (each as a dict).
    fn legal_actions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty_bound(py);
        for a in self.s.legal_actions() {
            list.append(action_to_dict(py, &self.s, &a)?)?;
        }
        Ok(list.into())
    }

    /// Apply an action dict, mutating the game in place.
    fn apply(&mut self, action: &Bound<'_, PyDict>) -> PyResult<()> {
        let a = action_from_dict(action)?;
        // Validate legality to fail loudly on a bad replay/action.
        let code = a.encode();
        if !self.s.legal_actions().iter().any(|x| x.encode() == code) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "illegal action {} for pending {:?}",
                a,
                self.s.pending()
            )));
        }
        self.s.apply_inplace(a);
        Ok(())
    }

    /// Full state as a nested dict.
    fn state_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        state_to_dict(py, &self.s)
    }

    fn is_terminal(&self) -> bool {
        self.s.is_terminal()
    }

    /// Winner as 0 (white) / 1 (black) / None.
    fn winner(&self) -> Option<u8> {
        self.s.winner()
    }

    fn adjudicated_draw(&self) -> bool {
        self.s.adjudicated_draw
    }

    fn main_plays(&self) -> u16 {
        self.s.main_plays
    }

    fn to_act(&self) -> u8 {
        self.s.to_act()
    }

    fn is_chance(&self) -> bool {
        self.s.is_chance()
    }

    fn clone(&self) -> Game {
        Game { s: self.s.clone() }
    }
}

#[pymodule]
fn warchest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add("MAX_MAIN_PLAYS", crate::state::MAX_MAIN_PLAYS)?;
    Ok(())
}
