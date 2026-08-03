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

// --------------------------------------------------------------- ReBeL API
//
// The training loop lives in Python (PyTorch), but every game, every subgame
// solve and every network evaluation runs here: Python only ships weights down
// and pulls tensors back once per epoch.

use crate::net::Mlp;
use crate::search::{Cfg, Nets};
use crate::selfplay::{eval_match as rs_eval_match, run_games, Agent, Collect, Data, GameCfg};
use numpy::{IntoPyArray, PyReadonlyArray1};
use std::sync::{OnceLock, RwLock};

/// Two independent weight slots, so a match can pit one checkpoint against
/// another (the final network against the initial one, say).
pub const N_SLOTS: usize = 2;

fn nets() -> &'static RwLock<Vec<Nets>> {
    static NETS: OnceLock<RwLock<Vec<Nets>>> = OnceLock::new();
    NETS.get_or_init(|| RwLock::new(vec![Nets::default(); N_SLOTS]))
}

fn split_mlp(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> PyResult<Mlp> {
    let mut mlp = Mlp {
        dims: dims.to_vec(),
        w: Vec::new(),
        b: Vec::new(),
        ln_w: Vec::new(),
        ln_b: Vec::new(),
    };
    let (mut wi, mut bi) = (0usize, 0usize);
    for l in 0..dims.len() - 1 {
        let (i, o) = (dims[l], dims[l + 1]);
        if wi + i * o > w.len() || bi + o > b.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weight buffer too short for the given dims",
            ));
        }
        mlp.w.push(w[wi..wi + i * o].to_vec());
        mlp.b.push(b[bi..bi + o].to_vec());
        wi += i * o;
        bi += o;
    }
    // LayerNorm parameters for the hidden layers, weight then bias per layer.
    // An empty buffer means the network has no norm.
    if !ln.is_empty() {
        let hidden = dims.len() - 2;
        let need: usize = 2 * dims[1..dims.len() - 1].iter().sum::<usize>();
        if ln.len() != need {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "layernorm buffer is {} floats, expected {}",
                ln.len(),
                need
            )));
        }
        let mut li = 0usize;
        for l in 0..hidden {
            let o = dims[l + 1];
            mlp.ln_w.push(ln[li..li + o].to_vec());
            li += o;
            mlp.ln_b.push(ln[li..li + o].to_vec());
            li += o;
        }
    }
    Ok(mlp)
}

/// Install value-network weights. `w` holds every layer's `[in, out]` matrix
/// concatenated row-major (torch's `weight.t()`), `b` every bias.
#[pyfunction]
#[pyo3(signature = (dims, w, b, slot=0, ln=None))]
fn set_weights(
    dims: Vec<usize>,
    w: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    slot: usize,
    ln: Option<PyReadonlyArray1<f32>>,
) -> PyResult<()> {
    if slot >= N_SLOTS {
        return Err(pyo3::exceptions::PyValueError::new_err("slot out of range"));
    }
    let empty: [f32; 0] = [];
    let ln_slice = match &ln {
        Some(a) => a.as_slice()?,
        None => &empty,
    };
    let mlp = split_mlp(&dims, w.as_slice()?, b.as_slice()?, ln_slice)?;
    nets().write().unwrap()[slot].value = mlp;
    Ok(())
}

fn agent_of(name: &str, cfg: Cfg, temp: f32, slot: usize) -> PyResult<Agent> {
    Ok(match name {
        "greedy" => Agent::Greedy { temp },
        "uniform" => Agent::Uniform,
        "rebel" => Agent::Rebel { cfg, slot },
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown agent '{}'",
                other
            )))
        }
    })
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
    out.set_item("configs", d.configs)?;
    out.set_item("vx", d.vx.into_pyarray_bound(py))?;
    out.set_item("vy", d.vy.into_pyarray_bound(py))?;
    out.set_item("vm", d.vm.into_pyarray_bound(py))?;
    Ok(out.into())
}

/// Run `games` self-play games across all cores and return the training data.
/// `mode` is "greedy" (Monte-Carlo warm start) or "rebel".
#[pyfunction]
#[pyo3(signature = (games, seed, mode, depth=1, iters=16, explore=0.25, temp=2.0, random_draft=false, eval_mix=0.5))]
#[allow(clippy::too_many_arguments)]
fn gen_data(
    py: Python<'_>,
    games: usize,
    seed: u64,
    mode: &str,
    depth: usize,
    iters: usize,
    explore: f32,
    temp: f32,
    random_draft: bool,
    eval_mix: f32,
) -> PyResult<PyObject> {
    let cfg = Cfg { depth, iters };
    let (agent, collect) = match mode {
        "greedy" => (Agent::Greedy { temp }, Collect::Mc),
        "rebel" => (Agent::Rebel { cfg, slot: 0 }, Collect::Rebel),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown mode '{}'",
                other
            )))
        }
    };
    let gc = GameCfg {
        agents: [agent, agent],
        collect,
        explore,
        eval: false,
        random_draft,
        eval_mix,
    };
    let d = py.allow_threads(|| {
        let n = nets().read().unwrap();
        run_games(games, seed, &n, &gc)
    });
    data_to_dict(py, d)
}

/// Head-to-head evaluation with alternating colours and paired drafts.
#[pyfunction]
#[pyo3(signature = (games, seed, a, b, depth=1, iters=16, temp=2.0, slot_a=0, slot_b=1, random_draft=false))]
#[allow(clippy::too_many_arguments)]
fn eval_match(
    py: Python<'_>,
    games: usize,
    seed: u64,
    a: &str,
    b: &str,
    depth: usize,
    iters: usize,
    temp: f32,
    slot_a: usize,
    slot_b: usize,
    random_draft: bool,
) -> PyResult<(usize, usize, usize)> {
    let cfg = Cfg { depth, iters };
    let (aa, bb) = (agent_of(a, cfg, temp, slot_a)?, agent_of(b, cfg, temp, slot_b)?);
    Ok(py.allow_threads(|| {
        let n = nets().read().unwrap();
        rs_eval_match(games, seed, &n, aa, bb, random_draft)
    }))
}

/// Set the horizon payoff per marker of lead. The trainer anneals it to 0.
#[pyfunction]
fn set_cap_value(v: f32) {
    crate::state::set_cap_marker_value(v);
}

/// Run the Rust value network forward on `x` (`rows * FEAT`, row-major) and
/// return `rows * 2*NHAND` outputs. Exists so the Python side can assert that
/// the inference path used to generate targets is numerically the same network
/// that PyTorch trains -- a silent divergence there would corrupt every target
/// while every other test kept passing.
#[pyfunction]
#[pyo3(signature = (x, rows, slot=0))]
fn infer(x: PyReadonlyArray1<f32>, rows: usize, slot: usize) -> PyResult<Vec<f32>> {
    if slot >= N_SLOTS {
        return Err(pyo3::exceptions::PyValueError::new_err("slot out of range"));
    }
    let guard = nets().read().unwrap();
    let mlp = &guard[slot].value;
    if mlp.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no weights in slot"));
    }
    let (mut scratch, mut out) = (Vec::new(), Vec::new());
    mlp.forward(x.as_slice()?, rows, &mut scratch, &mut out);
    Ok(out)
}

#[pymodule]
fn warchest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add("MAX_MAIN_PLAYS", crate::state::MAX_MAIN_PLAYS)?;
    m.add("FEAT", crate::rebel::FEAT)?;
    m.add("NHAND", crate::rebel::NHAND)?;
    m.add_function(wrap_pyfunction!(set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(set_cap_value, m)?)?;
    m.add_function(wrap_pyfunction!(gen_data, m)?)?;
    m.add_function(wrap_pyfunction!(eval_match, m)?)?;
    m.add_function(wrap_pyfunction!(infer, m)?)?;
    Ok(())
}
