use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::board::board;
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

#[cfg(feature = "gpu")]
use crate::farm::Farm;
use crate::net::Net;
use crate::search::{Budget, Cfg, Cfr, Ent};
use crate::selfplay::{run_static_games, Agent, Collect, Data, GameCfg};
use numpy::{IntoPyArray, PyReadonlyArray1};
use parking_lot::RwLock;
use std::sync::{Arc, LazyLock};

static NETS: LazyLock<RwLock<Arc<Net>>> = LazyLock::new(|| RwLock::new(Arc::new(Net::default())));

pub(crate) fn nets() -> &'static RwLock<Arc<Net>> {
    &NETS
}

#[pyfunction]
fn set_weights(w: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>, ln: PyReadonlyArray1<f32>) -> PyResult<()> {
    let value = Net::from_flat(w.as_slice()?, b.as_slice()?, ln.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    *nets().write() = Arc::new(value);
    Ok(())
}

#[cfg(feature = "gpu")]
fn check_nets() -> PyResult<()> {
    if nets().read().is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("no weights pushed"));
    }
    Ok(())
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
    const PLAYS: [&str; 7] = [
        "attack", "pass", "deploy", "bolster", "maneuver", "recruit", "claim_initiative",
    ];
    for (name, v) in [
        ("nv", d.nv),
        ("games", d.games),
        ("decisions", d.decisions),
        ("white_wins", d.wins[0]),
        ("black_wins", d.wins[1]),
        ("draws", d.draws),
        ("horizon_hits", d.cap_hits),
        ("configs", d.configs),
        ("query_rows", d.queries),
        ("dropped", d.dropped),
    ] {
        out.set_item(name, v)?;
    }
    for (name, v) in PLAYS.iter().zip(d.plays) {
        out.set_item(format!("plays_{name}"), v)?;
    }
    for (name, got, want) in [
        ("config offsets", d.coff.len(), if d.nv == 0 { 0 } else { 2 * d.nv + 1 }),
        ("truth", d.truth.len(), 2 * d.nv),
        ("outcomes", d.outcome.len(), 2 * d.nv),
        ("creation times", d.created.len(), d.nv),
        ("query labels", d.query.len(), d.nv),
        ("TD(1) labels", d.td1.len(), d.nv),
    ] {
        assert_eq!(got, want, "{name} do not match the row count");
    }
    let n_solves = d.soff.len();
    let mut soff = d.soff.clone();
    soff.push(d.nv as u32);
    assert_eq!(soff[0], 0, "solve offsets must start at row 0");
    assert!(
        soff.windows(2).all(|w| w[0] < w[1]),
        "solve offsets must be strictly increasing"
    );
    macro_rules! arrays {
        ($($name:ident = $value:expr),* $(,)?) => {
            $( out.set_item(stringify!($name), $value.into_pyarray_bound(py))?; )*
        };
    }
    arrays! {
        rows = d.rows, cc = d.cc, cw = d.cw, cy = d.cy, coff = d.coff,
        pa = d.pa, paoff = d.paoff, pcoff = d.pcoff, pci = d.pci,
        pcell = d.pcell, pprob = d.pprob, truth = d.truth, outcome = d.outcome,
        created = d.created, query = d.query, td1 = d.td1, soff = soff,
    }
    out.set_item("row_bytes", crate::pbs::ROW_BYTES)?;
    out.set_item("solves", n_solves)?;
    Ok(out.into())
}

#[pyclass]
struct SolveFarm {
    #[cfg(feature = "gpu")]
    farm: Farm,
}

#[pymethods]
impl SolveFarm {
    #[new]
    #[pyo3(signature = (seed, workers, s=512, c=8.0, batch=8, rounds=0, explore=0.1, random_draft=true, cfr="sog", puct=1.5, prior_temp=1.0, p_td1=0.2, query_rate=0.9, recursive_rate=0.1, devices=vec![0]))]
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
        puct: f32,
        prior_temp: f32,
        p_td1: f32,
        query_rate: f32,
        recursive_rate: f32,
        devices: Vec<usize>,
    ) -> PyResult<SolveFarm> {
        let explore = rate("explore", explore)?;
        let query_rate = rate("query_rate", query_rate)?;
        let recursive_rate = rate("recursive_rate", recursive_rate)?;
        let cfg = Cfg {
            s,
            c,
            batch,
            rounds,
            cfr: cfr_of(cfr)?,
            puct,
            prior_temp,
            budget: Budget::for_s(s),
        };
        let gc = GameCfg {
            agents: [Agent::Sog { cfg }; 2],
            collect: Collect::Sog,
            explore,
            random_draft,
            p_td1,
            query_rate,
            recursive_rate,
        };
        #[cfg(feature = "gpu")]
        {
            let net = Arc::clone(&nets().read());
            let device = device_for(&devices, &net, cfg)?;
            Ok(SolveFarm {
                farm: Farm::new(seed, workers, gc, net, device),
            })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (seed, workers, gc, devices, cfg);
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "built without the `gpu` feature: there is no CUDA solve farm",
            ))
        }
    }

    #[pyo3(signature = (solves=1))]
    fn collect(&mut self, py: Python<'_>, solves: usize) -> PyResult<PyObject> {
        #[cfg(feature = "gpu")]
        {
            if solves == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err("solves must be positive"));
            }
            check_nets()?;
            self.farm
                .publish(Arc::clone(&nets().read()))
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
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
            let s = self.farm.stats();
            let read = |a: &std::sync::atomic::AtomicU64| {
                a.load(std::sync::atomic::Ordering::Relaxed)
            };
            dict.set_item("rounds", read(&s.rounds))?;
            dict.set_item("round_calls", read(&s.calls))?;
            dict.set_item("round_rows", read(&s.rows))?;
            dict.set_item("round_nanos", read(&s.nanos))?;
            dict.set_item("slots", read(&s.slots))?;
            dict.set_item("slots_used", read(&s.slots))?;
            dict.set_item("slots_per_card", read(&s.slots_per_card))?;
            dict.set_item("slot_bytes", read(&s.slot_bytes))?;
            dict.set_item("budget_hits", read(&s.budget_hits))?;
            dict.set_item("entity_hits", s.entity_hits())?;
            dict.set_item("shapes", s.take_shapes())?;
            Ok(out)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (py, solves);
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "built without the `gpu` feature: there is no CUDA solve farm",
            ))
        }
    }
}

#[cfg(feature = "gpu")]
fn device_for(devices: &[usize], value: &crate::net::Net, cfg: Cfg) -> PyResult<crate::cuda::Device> {
    let max_slots = crate::farm::host_slots(cfg.budget);
    crate::cuda::Device::new(devices, value, cfg, max_slots).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
#[pyo3(signature = (games, seed, explore=0.1, random_draft=true, temp=2.0))]
fn gen_data(
    py: Python<'_>,
    games: usize,
    seed: u64,
    explore: f32,
    random_draft: bool,
    temp: f32,
) -> PyResult<PyObject> {
    let gc = GameCfg {
        agents: [Agent::Greedy { temp }; 2],
        collect: Collect::Static,
        explore,
        random_draft,
        p_td1: 0.0,
        query_rate: 0.0,
        recursive_rate: 0.0,
    };
    let n = Arc::clone(&nets().read());
    let d = py.allow_threads(|| run_static_games(games, seed, &n, &gc));
    data_to_dict(py, d)
}

#[pyfunction]
fn card_features_table() -> Vec<f32> {
    let mut out = vec![0.0; N_UNITS * CARD_FEATS];
    for u in 0..N_UNITS {
        write_card_features(u as u8, &mut out[u * CARD_FEATS..(u + 1) * CARD_FEATS]);
    }
    out
}

#[pyfunction]
fn hex_location_flags() -> Vec<u8> {
    board().is_location.iter().map(|&x| x as u8).collect()
}

#[pyfunction]
fn hex_neighbours() -> Vec<u8> {
    crate::board::neighbour_gather()
}

#[pyfunction]
fn hex_mirror() -> Vec<u8> {
    (0..crate::board::N_HEXES).map(|h| crate::state::mirror_hex(h) as u8).collect()
}


#[pyfunction]
fn sample_rows(games: usize, seed: u64) -> Vec<u8> {
    use crate::pbs::{pack_row, Ctx, ROW_BYTES};
    let mut rng = crate::rng::Rng::new(seed);
    let mut out = Vec::new();
    for _ in 0..games {
        let mut s = crate::selfplay::make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        for _ in 0..160 {
            if s.is_terminal() {
                break;
            }
            if s.is_valued() {
                let at = out.len();
                out.resize(at + ROW_BYTES, 0);
                pack_row(&s, &ctx, &mut out[at..at + ROW_BYTES]);
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    out
}

#[pyfunction]
fn mirror_rows(rows: PyReadonlyArray1<u8>) -> PyResult<Vec<u8>> {
    use crate::pbs::{mirror_row, ROW_BYTES};
    let rows = rows.as_slice()?;
    if rows.len() % ROW_BYTES != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("rows are not whole"));
    }
    let mut out = vec![0u8; rows.len()];
    for (src, dst) in rows.chunks_exact(ROW_BYTES).zip(out.chunks_exact_mut(ROW_BYTES)) {
        mirror_row(src, dst);
    }
    Ok(out)
}

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
        crate::cuda::expand_rows_torch(rows, cards, locations, out, n, stream, device)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
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
    m.add_function(wrap_pyfunction!(mirror_rows, m)?)?;
    m.add_function(wrap_pyfunction!(sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(card_features_table, m)?)?;
    m.add_function(wrap_pyfunction!(hex_location_flags, m)?)?;
    m.add_class::<SolveFarm>()?;
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
    m.add("ROW_HEX_OWNER", crate::pbs::ROW_HEX_OWNER)?;
    m.add("ROW_HEX_SLOT", crate::pbs::ROW_HEX_SLOT)?;
    m.add("ROW_HEX_MARKER", crate::pbs::ROW_HEX_MARKER)?;
    m.add_function(wrap_pyfunction!(rules_table_hash, m)?)?;
    m.add_function(wrap_pyfunction!(hex_neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(hex_mirror, m)?)?;
    m.add_function(wrap_pyfunction!(set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(gen_data, m)?)?;
    m.add("ENT_NAMES", Ent::NAME.to_vec())?;
    m.add("STOP_NAMES", crate::search::STOP_NAMES.to_vec())?;
    m.add("SOLVE_KIND_NAMES", crate::selfplay::SolveKind::NAMES.to_vec())?;
    Ok(())
}
