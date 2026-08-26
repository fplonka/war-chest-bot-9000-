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
use crate::units::{def, index_of_id, write_card_features, CARD_FEATS, N_UNITS};

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
pub(crate) fn action_to_dict(py: Python<'_>, s: &State, a: &Action) -> PyResult<PyObject> {
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
pub(crate) fn action_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Action> {
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

pub(crate) fn state_to_dict(py: Python<'_>, s: &State) -> PyResult<PyObject> {
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

// ----------------------------------------------------------------- SoG API
//
// The training loop lives in Python (PyTorch), but every game, every subgame
// solve and every network evaluation runs here: Python only ships weights down
// and pulls tensors back once per epoch.

use crate::net::Net;
use crate::farm::{Backend, Farm, Work};
use crate::search::{Budget, Cfg, Cfr, Ent, Nets};
use crate::selfplay::{run_games, Agent, Collect, Data, GameCfg};
use numpy::{IntoPyArray, PyReadonlyArray1};
use parking_lot::RwLock;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicU64, Ordering};

/// The live network. Empty until the trainer pushes weights: the
/// phase plays with no network at all. One process holds one network — two
/// checkpoints meet each other as two arena bots, not as two slots here.
static NETS: LazyLock<RwLock<Arc<Nets>>> =
    LazyLock::new(|| RwLock::new(Arc::new(Nets::default())));
static NET_VERSION: AtomicU64 = AtomicU64::new(0);

pub(crate) fn nets() -> &'static RwLock<Arc<Nets>> {
    &NETS
}

fn check_nets() -> PyResult<()> {
    if nets().read().value.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no weights pushed"));
    }
    Ok(())
}

/// Install value-network weights: the flat arrays `Net::from_flat` documents.
#[pyfunction]
fn set_weights(
    w: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    ln: PyReadonlyArray1<f32>,
) -> PyResult<()> {
    let value = Net::from_flat(w.as_slice()?, b.as_slice()?, ln.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    *nets().write() = Arc::new(Nets { value, device: false });
    NET_VERSION.fetch_add(1, Ordering::Release);
    Ok(())
}

/// Install the weights a bot directory carries. The binary format is the one
/// `train/export_weights.py` writes, so anything that can play a bot loads it
/// the same way and nothing needs torch to do it.
#[pyfunction]
fn set_weights_bin(path: &str) -> PyResult<()> {
    let value = Net::load_bin(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}: {}", path, e)))?;
    *nets().write() = Arc::new(Nets { value, device: false });
    NET_VERSION.fetch_add(1, Ordering::Release);
    Ok(())
}

/// An action's own name, for a player describing their own move.
#[pyfunction]
fn action_label(code: u32) -> PyResult<String> {
    Action::decode(code)
        .map(|a| format!("{}", a))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no action {}", code)))
}

/// What an *observation* looks like to the player who did not make it.
///
/// `obs_key` drops the coin behind a face-down play, so the code it leaves
/// decodes to an action naming some arbitrary coin. Rendering that would put a
/// private coin on the screen, so the three plays that hide one are described
/// by what was actually seen and nothing more.
#[pyfunction]
fn obs_label(key: u32) -> String {
    use crate::actions::Action::*;
    match Action::decode(key) {
        Some(Pass { .. }) => "passes with a face-down coin".into(),
        Some(ClaimInitiative { .. }) => "claims initiative with a face-down coin".into(),
        Some(Recruit { unit, .. }) => format!(
            "recruits {} with a face-down coin",
            crate::units::def(unit).name
        ),
        Some(a) => format!("{}", a),
        None => "plays a face-down coin".into(),
    }
}

fn cfr_of(name: &str) -> PyResult<Cfr> {
    Cfr::named(name).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown cfr '{}': one of {}",
            name,
            Cfr::NAMED.map(|(n, _)| n).join(", ")
        ))
    })
}

fn rate(name: &str, value: f32) -> PyResult<f32> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and in [0, 1], got {value}"
        )))
    }
}

fn data_to_dict(py: Python<'_>, d: Data) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("nv", d.nv)?;
    out.set_item("games", d.games)?;
    out.set_item("decisions", d.decisions)?;
    out.set_item("white_wins", d.wins[0])?;
    out.set_item("black_wins", d.wins[1])?;
    out.set_item("draws", d.draws)?;
    out.set_item("cap_hits", d.cap_hits)?;
    out.set_item("horizon_hits", d.cap_hits)?;
    out.set_item("configs", d.configs)?;
    out.set_item("query_rows", d.queries)?;
    out.set_item("dropped", d.dropped)?;
    out.set_item("plays_attack", d.plays[0])?;
    out.set_item("plays_pass", d.plays[1])?;
    out.set_item("plays_deploy", d.plays[2])?;
    out.set_item("plays_bolster", d.plays[3])?;
    out.set_item("plays_maneuver", d.plays[4])?;
    out.set_item("plays_recruit", d.plays[5])?;
    out.set_item("plays_claim_initiative", d.plays[6])?;
    assert_eq!(
        d.coff.len(),
        if d.nv == 0 { 0 } else { 2 * d.nv + 1 },
        "config offsets do not match the row count"
    );
    assert_eq!(d.truth.len(), 2 * d.nv, "truth does not match rows");
    assert_eq!(d.outcome.len(), 2 * d.nv, "outcomes do not match rows");
    assert_eq!(d.created.len(), d.nv, "creation times do not match rows");
    assert_eq!(d.query.len(), d.nv, "query labels do not match rows");
    assert_eq!(d.td1.len(), d.nv, "TD(1) labels do not match rows");
    // Internal `soff` holds one start per solve; the exposed array appends
    // the total row count as the trailing entry, so `len - 1` is the number
    // of solves.
    let n_solves = d.soff.len();
    let mut soff = d.soff.clone();
    soff.push(d.nv as u32);
    if !soff.is_empty() {
        assert_eq!(soff[0], 0, "solve offsets must start at row 0");
        assert!(
            soff.windows(2).all(|w| w[0] < w[1]),
            "solve offsets must be strictly increasing"
        );
    }
    out.set_item("rows", d.rows.into_pyarray_bound(py))?;
    out.set_item("row_bytes", crate::pbs::ROW_BYTES)?;
    out.set_item("cc", d.cc.into_pyarray_bound(py))?;
    out.set_item("cw", d.cw.into_pyarray_bound(py))?;
    out.set_item("cy", d.cy.into_pyarray_bound(py))?;
    out.set_item("coff", d.coff.into_pyarray_bound(py))?;
    // The policy target: the root's actions per row, and per config the legal
    // cells with their probability.
    out.set_item("pa", d.pa.into_pyarray_bound(py))?;
    out.set_item("paoff", d.paoff.into_pyarray_bound(py))?;
    out.set_item("pcoff", d.pcoff.into_pyarray_bound(py))?;
    out.set_item("pci", d.pci.into_pyarray_bound(py))?;
    out.set_item("pcell", d.pcell.into_pyarray_bound(py))?;
    out.set_item("pprob", d.pprob.into_pyarray_bound(py))?;
    out.set_item("truth", d.truth.into_pyarray_bound(py))?;
    out.set_item("outcome", d.outcome.into_pyarray_bound(py))?;
    out.set_item("created", d.created.into_pyarray_bound(py))?;
    out.set_item("query", d.query.into_pyarray_bound(py))?;
    out.set_item("td1", d.td1.into_pyarray_bound(py))?;
    out.set_item("soff", soff.into_pyarray_bound(py))?;
    out.set_item("solves", n_solves)?;
    Ok(out.into())
}

/// Many solves in flight in one process, sharing one forward pass.
///
/// This replaces the process-per-core generator: threads are cheap, a network
/// replica per core was not, and inference only batches if the solves are in
/// the same address space.
#[pyclass]
struct SolveFarm {
    farm: Farm,
    net_version: u64,
}

#[pymethods]
impl SolveFarm {
    #[new]
    #[pyo3(signature = (seed, workers, s=512, c=8.0, batch=8, rounds=0, explore=0.1, random_draft=true, cfr="sog", p_td1=0.2, query_rate=0.9, recursive_rate=0.1, devices=vec![0], roots=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        seed: u64,
        workers: usize,
        s: u32,
        c: f32,
        batch: usize,
        rounds: u8,
        explore: f32,
        random_draft: bool,
        cfr: &str,
        p_td1: f32,
        query_rate: f32,
        recursive_rate: f32,
        devices: Vec<usize>,
        roots: Option<&str>,
    ) -> PyResult<SolveFarm> {
        let query_rate = rate("query_rate", query_rate)?;
        let recursive_rate = rate("recursive_rate", recursive_rate)?;
        let cfg = Cfg { s, c, batch, rounds, cfr: cfr_of(cfr)?, budget: Budget::for_s(s), ..Default::default() };
        // A corpus makes this a bench rather than a run: the same roots in the
        // same order, so the mix of solve costs in flight does not drift.
        let work = match roots {
            None => Work::Play(GameCfg {
                agents: [Agent::Sog { cfg }; 2],
                collect: Collect::Sog,
                explore,
                random_draft,
                p_td1,
                query_rate,
                recursive_rate,
            }),
            Some(path) => {
                let f = std::fs::File::open(path)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
                let roots = crate::roots::read_roots(&mut std::io::BufReader::new(f))
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                if roots.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err("empty root corpus"));
                }
                Work::Roots {
                    roots: Arc::new(roots),
                    cfg,
                    recursive_rate,
                }
            }
        };
        let version = NET_VERSION.load(Ordering::Acquire);
        let backend = backend_for(&devices, (**nets().read()).value.clone(), cfg)?;
        Ok(SolveFarm {
            farm: Farm::new(seed, workers, work, backend),
            net_version: version,
        })
    }

    /// Run rounds until at least `solves` rows are ready, then hand them over.
    #[pyo3(signature = (solves=1))]
    fn collect(&mut self, py: Python<'_>, solves: usize) -> PyResult<PyObject> {
        if solves == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "solves must be positive",
            ));
        }
        check_nets()?;
        let version = NET_VERSION.load(Ordering::Acquire);
        if self.net_version != version {
            self.farm
                .publish((**nets().read()).value.clone())
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            self.net_version = version;
        }
        let d = py.allow_threads(|| self.farm.drive(solves));
        if self.farm.broken() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "a card could not answer a round; the solves it held are gone. \
                 The driver prints the reason, and out of memory is the usual \
                 one.",
            ));
        }
        let out = data_to_dict(py, d)?;
        let dict = out.bind(py).downcast::<PyDict>()?.clone();
        // How well the batching is working: calls per round is how many solves
        // shared a forward pass.
        let s = self.farm.stats();
        let read = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        dict.set_item("rounds", read(&s.rounds))?;
        dict.set_item("round_calls", read(&s.calls))?;
        dict.set_item("round_rows", read(&s.rows))?;
        dict.set_item("round_nanos", read(&s.nanos))?;
        // What the population is, rather than what it is guessed to be: solves
        // in flight, what the host budget allowed at the last admission, and
        // the largest a solve has grown to in host bytes.
        dict.set_item("slots", s.slots())?;
        dict.set_item("slots_used", s.used())?;
        dict.set_item("slots_per_card", s.slots_per_card())?;
        dict.set_item("slot_bytes", s.slot_bytes())?;
        dict.set_item("budget_hits", s.budget_hits())?;
        dict.set_item("entity_hits", s.entity_hits())?;
        dict.set_item("shapes", s.take_shapes())?;
        Ok(out)
    }
}

/// The devices. There is no CPU fallback on purpose: a run that cannot reach a
/// GPU is two orders of magnitude off and should say so rather than crawl.
fn backend_for(
    _devices: &[usize],
    _value: crate::net::Net,
    #[allow(unused)] cfg: Cfg,
) -> PyResult<Backend> {
    #[cfg(feature = "gpu")]
    {
        let max_slots = crate::farm::host_slots(cfg.budget);
        return crate::cuda::Device::new(_devices, _value, cfg, max_slots)
            .map(Backend::Cuda)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err);
    }
    #[cfg(not(feature = "gpu"))]
    Err(pyo3::exceptions::PyRuntimeError::new_err(
        "built without the `gpu` feature: there is no CPU inference path",
    ))
}

/// Run `games` self-play games across all cores and return the training data.
#[pyfunction]
#[pyo3(signature = (games, seed, s=512, c=8.0, explore=0.25, random_draft=true, p_td1=0.0, cfr="sog", agent="sog", temp=2.0, cpu=false))]
#[allow(clippy::too_many_arguments)]
fn gen_data(
    py: Python<'_>,
    games: usize,
    seed: u64,
    s: u32,
    c: f32,
    explore: f32,
    random_draft: bool,
    p_td1: f32,
    cfr: &str,
    agent: &str,
    temp: f32,
    cpu: bool,
) -> PyResult<PyObject> {
    let cfg = Cfg { s, c, cfr: cfr_of(cfr)?, budget: Budget::for_s(s), ..Default::default() };
    let (agent, collect, p_td1) = match agent {
        "sog" if cpu => {
            eprintln!("\n*** cpu=True: CPU SELF-PLAY IS ~50x SLOWER. YOU DO NOT WANT THIS. ***\n");
            (Agent::Sog { cfg }, Collect::Sog, p_td1)
        }
        "sog" => return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "GPU self-play requires SolveFarm; pass cpu=True only for the ~50x slower test path",
        )),
        "greedy" => (Agent::Greedy { temp }, Collect::Static, 0.0),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown agent {other}"
            )))
        }
    };
    let gc = GameCfg {
        agents: [agent, agent],
        collect,
        explore,
        random_draft,
        p_td1,
        // The batch generator takes a row at every decision already; the query
        // solver belongs to the streaming generator.
        query_rate: 0.0,
        recursive_rate: 0.0,
    };
    let n = Arc::clone(&nets().read());
    let d = py.allow_threads(|| run_games(games, seed, &n, &gc));
    data_to_dict(py, d)
}

/// Write a corpus of solve roots for the tools that need a fixed workload.
///
/// The rates belong to the caller because the corpus is only worth anything if
/// it is the mix a real run solves: pass the run's own `query_rate` and
/// `recursive_rate`, not a convenient number.
///
/// The search here decides which positions arise, not what a root *is*: a
/// belief support comes from the draw history. So this runs a cheap search by
/// default -- the corpus is generated on the cores, and a full-budget solve at
/// every decision of every game takes the better part of an hour where a
/// thirty-two expansion one takes a minute.
#[pyfunction]
#[pyo3(signature = (games, seed, path, cap=4096, random_draft=true, s=32, c=4.0,
                    explore=0.1, query_rate=0.9, recursive_rate=0.1, cpu=false))]
#[allow(clippy::too_many_arguments)]
fn save_roots(
    py: Python<'_>,
    games: usize,
    seed: u64,
    path: &str,
    cap: usize,
    random_draft: bool,
    s: u32,
    c: f32,
    explore: f32,
    query_rate: f32,
    recursive_rate: f32,
    cpu: bool,
) -> PyResult<usize> {
    let query_rate = rate("query_rate", query_rate)?;
    let recursive_rate = rate("recursive_rate", recursive_rate)?;
    if !cpu {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "save_roots uses CPU search; pass cpu=True to accept the ~50x slower path",
        ));
    }
    eprintln!("\n*** cpu=True: CPU ROOT GENERATION IS ~50x SLOWER. YOU DO NOT WANT THIS. ***\n");
    let cfg = Cfg { s, c, budget: Budget::for_s(s), ..Default::default() };
    let gc = GameCfg {
        agents: [Agent::Sog { cfg }; 2],
        collect: Collect::Sog,
        explore,
        random_draft,
        p_td1: 0.0,
        query_rate,
        recursive_rate,
    };
    let n = Arc::clone(&nets().read());
    let roots =
        py.allow_threads(|| crate::selfplay::collect_roots(games, seed, &n, &gc, cap));
    let f = std::fs::File::create(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut w = std::io::BufWriter::new(f);
    crate::roots::write_roots(&mut w, &roots)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(roots.len())
}

/// Print generation phase timers. Empty unless the extension was built with
/// the `prof` feature.
#[pyfunction]
fn prof_dump() {
    eprintln!(
        "  sizes: TNode {} B  State {} B",
        std::mem::size_of::<crate::search::TNode>(),
        std::mem::size_of::<crate::State>(),
    );
    crate::prof::dump();
    // Rows by kind, which is what says whether the trunk or the join is worth
    // optimising: a trunk row is two orders of magnitude the work of a join.
    crate::prof::dump_work();
}

/// Set the fixed training payoff per marker of lead.
#[pyfunction]
fn set_cap_value(v: f32) {
    crate::state::set_cap_marker_value(v);
}

#[pyfunction]
fn infer(
    xpub: PyReadonlyArray1<f32>,
    phi: PyReadonlyArray1<f32>,
    weight: PyReadonlyArray1<f32>,
    seg: PyReadonlyArray1<u32>,
    queries: usize,
) -> PyResult<Vec<f32>> {
    check_nets()?;
    let guard = nets().read();
    Ok(guard.value.forward(
        xpub.as_slice()?,
        phi.as_slice()?,
        weight.as_slice()?,
        seg.as_slice()?,
        queries,
    ))
}

/// The policy readout for one node: `logit(c, a) = <f_p(c), e(a)>` over the
/// `(config, action)` cells named by `cfg` and `act`.
///
/// The counterpart of `infer` for the policy head, so `test_parity.py` holds
/// both readouts to torch through the same door.
#[pyfunction]
fn infer_policy(
    xpub: PyReadonlyArray1<f32>,
    phi: PyReadonlyArray1<f32>,
    weight: PyReadonlyArray1<f32>,
    seg: PyReadonlyArray1<u32>,
    feat: PyReadonlyArray1<f32>,
    cfg: PyReadonlyArray1<u32>,
    act: PyReadonlyArray1<u32>,
    queries: usize,
) -> PyResult<Vec<f32>> {
    check_nets()?;
    let guard = nets().read();
    Ok(guard.value.forward_policy(
        xpub.as_slice()?,
        phi.as_slice()?,
        weight.as_slice()?,
        seg.as_slice()?,
        feat.as_slice()?,
        cfg.as_slice()?,
        act.as_slice()?,
        queries,
    ))
}

/// What `leaf_breakdown` returns, in order. The last two are megabytes; the
/// rest are milliseconds. Exposed so a reporting tool cannot drift from the
/// engine's own list, which is how two of them came to be mislabelled.
#[pyfunction]
fn stage_names() -> Vec<String> {
    #[cfg(feature = "gpu")]
    {
        return crate::cuda::STAGES.iter().map(|s| s.to_string()).collect();
    }
    #[cfg(not(feature = "gpu"))]
    Vec::new()
}

/// Where the device's leaf pass spends its wall clock, in ms since the last
/// call: host marshalling, uploads, launches, the download.
#[pyfunction]
fn leaf_breakdown() -> Vec<f64> {
    #[cfg(feature = "gpu")]
    {
        return crate::cuda::leaf_breakdown().to_vec();
    }
    #[cfg(not(feature = "gpu"))]
    Vec::new()
}

/// What the fattest solve on a card holds, array by array, in bytes. Solves
/// in flight is memory-bound, so this is the list of things to argue with.
#[pyfunction]
fn solve_census() -> Vec<(String, usize)> {
    #[cfg(feature = "gpu")]
    {
        return crate::cuda::CENSUS
            .lock()
            .iter()
            .map(|&(n, b)| (n.to_string(), b))
            .collect();
    }
    #[cfg(not(feature = "gpu"))]
    Vec::new()
}

#[pyfunction]
fn budget_for_s(s: u32) -> [usize; 8] {
    let b = Budget::for_s(s);
    Ent::ALL.map(|e| b.cap(e))
}

#[pyfunction]
fn host_slot_bytes(s: u32) -> usize {
    Budget::for_s(s).host_slot_bytes()
}

/// All 37 hexes' axial coords, indexed by hex. The browser UI's board
/// geometry; mirrors `Board::coord`.
#[pyfunction]
fn hex_coords() -> Vec<(i8, i8)> {
    board().coord.to_vec()
}

/// The unit table as (unitTypeId, name, coin count) triples, for the UI to
/// label hands and drafts from the engine's source of truth.
#[pyfunction]
fn units_info() -> Vec<(u16, &'static str, u8)> {
    (0..N_UNITS)
        .map(|u| {
            let d = def(u as u8);
            (d.id, d.name, d.coins)
        })
        .collect()
}

/// Frozen encoder constants for the device-side replay expander. Export the
/// tables from the rules engine instead of restating card or board facts in
/// the trainer.
#[pyfunction]
fn card_features_table() -> Vec<f32> {
    let mut out = vec![0.0; N_UNITS * CARD_FEATS];
    for u in 0..N_UNITS {
        write_card_features(u as u8, &mut out[u * CARD_FEATS..(u + 1) * CARD_FEATS]);
    }
    out
}

/// Which hexes are control locations, as a `[N_HEXES]` 0/1 mask.
#[pyfunction]
fn hex_location_flags() -> Vec<u8> {
    board().is_location.iter().map(|&x| x as u8).collect()
}

/// The trunk's neighbour gather: `[N_HEXES * 6]`, hex-major, fixed direction
/// order, a missing neighbour written as `N_HEXES` so torch can gather from a
/// zero-padded 38th row without a mask. `board::neighbour_gather` is the
/// definition; the Rust trunk reads the same table.
#[pyfunction]
fn hex_neighbours() -> Vec<u8> {
    crate::board::neighbour_gather()
}

/// The control-location hex indices used by the public feature encoder.
#[pyfunction]
fn location_hexes() -> Vec<u8> {
    board().location_hexes.to_vec()
}

/// `N_HEXES` indices: where each hex lands under a 180-degree rotation of the
/// board, `(x, y) -> (6 - x, 6 - y)` in axial coordinates.
///
/// That rotation maps white's two starting locations exactly onto black's and
/// permutes the six neutral ones, so rotating the board and swapping the two
/// players is an exact symmetry of the game. It is the basis of the training
/// augmentation: every position can be presented a second way, for free.
#[pyfunction]
fn hex_mirror() -> Vec<u32> {
    (0..crate::board::N_HEXES)
        .map(|h| crate::state::mirror_hex(h) as u32)
        .collect()
}

/// Packed rows for coin-play states off random playouts, each followed by the
/// packed row of the same position rotated 180 degrees with the seats swapped.
/// `State::mirror` is the engine's own answer; `train/mirror.py` permutes row
/// bytes to get there and is checked against this.
#[pyfunction]
fn mirror_row_pairs(games: usize, seed: u64) -> Vec<u8> {
    use crate::pbs::{pack_row, Ctx, ROW_BYTES};
    let mut rng = crate::rng::Rng::new(seed);
    let mut out = Vec::new();
    for _ in 0..games {
        let mut s = crate::selfplay::make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        while !s.is_terminal() {
            let acts = s.legal_actions();
            if acts.is_empty() {
                break;
            }
            if !s.is_terminal() && !s.is_chance() {
                let m = s.mirror();
                let mctx = Ctx::new(&m);
                let at = out.len();
                out.resize(at + 2 * ROW_BYTES, 0);
                pack_row(&s, &ctx, &mut out[at..at + ROW_BYTES]);
                pack_row(&m, &mctx, &mut out[at + ROW_BYTES..at + 2 * ROW_BYTES]);
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    out
}

/// Expand packed replay rows into the public encoding, in one batch.
///
/// `rows` is `[n * ROW_BYTES]` u8 (see `pbs::ROW_*`).
/// Returns `[n, PUBFEAT]` f32, the exact layout `write_public_features`
/// produces for a live state.
#[pyfunction]
fn rules_table_hash() -> u64 {
    crate::pbs::rules_table_hash()
}

#[pyfunction]
fn expand_rows(rows: PyReadonlyArray1<u8>) -> PyResult<Vec<f32>> {
    use crate::pbs::{expand_row, PUBFEAT, ROW_BYTES};
    let rows = rows.as_slice()?;
    let n = rows.len() / ROW_BYTES;
    if rows.len() != n * ROW_BYTES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rows is not a multiple of ROW_BYTES",
        ));
    }
    let mut out = vec![0.0f32; n * PUBFEAT];
    for r in 0..n {
        let row = &rows[r * ROW_BYTES..(r + 1) * ROW_BYTES];
        expand_row(row, &mut out[r * PUBFEAT..(r + 1) * PUBFEAT]);
    }
    Ok(out)
}

/// Expand CUDA-resident packed rows on PyTorch's current stream.
#[pyfunction]
fn expand_rows_cuda(
    rows: u64,
    cards: u64,
    locations: u64,
    out: u64,
    n: usize,
    stream: usize,
    device: i32,
) -> PyResult<()> {
    #[cfg(feature = "gpu")]
    {
        return crate::cuda::expand_rows_torch(rows, cards, locations, out, n, stream, device)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err);
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (rows, cards, locations, out, n, stream, device);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "warchest was built without CUDA",
        ))
    }
}

#[pymodule]
fn warchest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hex_mirror, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_row_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(hex_coords, m)?)?;
    m.add_function(wrap_pyfunction!(units_info, m)?)?;
    m.add_function(wrap_pyfunction!(card_features_table, m)?)?;
    m.add_function(wrap_pyfunction!(hex_location_flags, m)?)?;
    m.add_class::<Game>()?;
    m.add_class::<SolveFarm>()?;
    m.add_class::<crate::arena::PyTable>()?;
    m.add("MAX_MAIN_PLAYS", crate::state::MAX_MAIN_PLAYS)?;
    m.add("PUBFEAT", crate::pbs::PUBFEAT)?;
    m.add("CFEAT", crate::pbs::CFEAT)?;
    m.add("CCOUNTS", crate::pbs::CCOUNTS)?;
    m.add("CNORM", crate::pbs::CNORM)?;
    m.add("N_HEXES", crate::board::N_HEXES)?;
    m.add("N_LOCATIONS", crate::board::N_LOCATIONS)?;
    m.add("N_UNITS", crate::units::N_UNITS)?;
    m.add("NSLOT", crate::pbs::NSLOT)?;
    m.add("N_KINDS", crate::actions::N_KINDS)?;
    m.add("ACT_BYTES", crate::search::ACT_BYTES)?;
    m.add("CARD_FEATS", crate::units::CARD_FEATS)?;
    // Block offsets in the public half of the encoding. Exported so the
    // training side can build the mirror permutation from one source of truth
    // rather than restating the layout.
    m.add("NTYPE", crate::pbs::NTYPE)?;
    m.add("HEX_CH", crate::pbs::HEX_CH)?;
    m.add("HEX_FACTS", crate::pbs::HEX_FACTS)?;
    m.add("HEX_BLOCK", crate::pbs::HEX_BLOCK)?;
    m.add("PILE_COUNTS", crate::pbs::PILE_COUNTS)?;
    m.add("PLAYER_SCALARS", crate::pbs::PLAYER_SCALARS)?;
    m.add("GLOBAL_SCALARS", crate::pbs::GLOBAL_SCALARS)?;
    m.add("PENDING_KINDS", crate::state::PENDING_KINDS)?;
    m.add("CONT_CAP", crate::state::CONT_CAP)?;
    m.add("LOOSE", crate::pbs::LOOSE)?;
    m.add("OFF_PILES", crate::pbs::OFF_PILES)?;
    m.add("OFF_CARDS", crate::pbs::OFF_CARDS)?;
    m.add("OFF_LOOSE", crate::pbs::OFF_LOOSE)?;
    m.add_function(wrap_pyfunction!(expand_rows, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows_cuda, m)?)?;
    m.add("ROW_BYTES", crate::pbs::ROW_BYTES)?;
    m.add("ROW_IDS", crate::pbs::ROW_IDS)?;
    m.add("ROW_HEX_OWNER", crate::pbs::ROW_HEX_OWNER)?;
    m.add("ROW_HEX_SLOT", crate::pbs::ROW_HEX_SLOT)?;
    m.add("ROW_HEX_HEIGHT", crate::pbs::ROW_HEX_HEIGHT)?;
    m.add("ROW_HEX_MARKER", crate::pbs::ROW_HEX_MARKER)?;
    m.add("ROW_PILES", crate::pbs::ROW_PILES)?;
    m.add("ROW_HAND_SIZE", crate::pbs::ROW_HAND_SIZE)?;
    m.add("ROW_FD_SIZE", crate::pbs::ROW_FD_SIZE)?;
    m.add("ROW_BAG_SIZE", crate::pbs::ROW_BAG_SIZE)?;
    m.add("ROW_INITIATIVE", crate::pbs::ROW_INITIATIVE)?;
    m.add("ROW_INIT_MOVED", crate::pbs::ROW_INIT_MOVED)?;
    m.add("ROW_TO_ACT", crate::pbs::ROW_TO_ACT)?;
    m.add("ROW_PLIES", crate::pbs::ROW_PLIES)?;
    m.add("ROW_STACK_KIND", crate::pbs::ROW_STACK_KIND)?;
    m.add("ROW_STACK_OWED", crate::pbs::ROW_STACK_OWED)?;
    m.add_function(wrap_pyfunction!(rules_table_hash, m)?)?;
    m.add_function(wrap_pyfunction!(hex_neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(location_hexes, m)?)?;
    m.add_function(wrap_pyfunction!(set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(set_weights_bin, m)?)?;
    m.add_function(wrap_pyfunction!(action_label, m)?)?;
    m.add_function(wrap_pyfunction!(obs_label, m)?)?;
    m.add_function(wrap_pyfunction!(set_cap_value, m)?)?;
    m.add_function(wrap_pyfunction!(prof_dump, m)?)?;
    m.add_function(wrap_pyfunction!(save_roots, m)?)?;
    m.add_function(wrap_pyfunction!(gen_data, m)?)?;
        m.add_function(wrap_pyfunction!(infer, m)?)?;
        m.add_function(wrap_pyfunction!(infer_policy, m)?)?;
        m.add_function(wrap_pyfunction!(leaf_breakdown, m)?)?;
        m.add_function(wrap_pyfunction!(stage_names, m)?)?;
        m.add_function(wrap_pyfunction!(solve_census, m)?)?;
    m.add("ENT_NAMES", Ent::NAME.iter().copied().collect::<Vec<&str>>())?;
    m.add("STOP_NAMES", crate::search::StopReason::NAMES.to_vec())?;
    m.add("SOLVE_KIND_NAMES", crate::selfplay::SolveKind::NAMES.to_vec())?;
    m.add_function(wrap_pyfunction!(budget_for_s, m)?)?;
    m.add_function(wrap_pyfunction!(host_slot_bytes, m)?)?;
    Ok(())
}
