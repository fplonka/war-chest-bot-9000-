//! End-to-end gates for the GPU wave. Storage and CUDA kernels are FP32; the
//! production GEMMs may down-convert internally for tensor-core throughput.
//! These compile everywhere and execute only on a CUDA test host.

use crate::net::{Net, NetLayout};
use crate::rng::Rng;
use crate::search::{Cfg, Cfr, Nets, Solver};
use crate::selfplay::{collect_roots, Agent, Collect, GameCfg};
use crate::serialize::PackedJob;
use std::sync::{Mutex, MutexGuard, OnceLock};

use super::client::SolveResult;
use super::service::spawn;

const N_CARRIED: usize = 3;

const TEST_CFG: Cfg = Cfg {
    depth: 2,
    iters: 8,
    snapshots: true,
    cfr: Cfr::LINEAR,
    node_cap: 0,
    gpu_build: false,
    keep_states: false,
};

fn gpu_guard() -> MutexGuard<'static, ()> {
    static GPU: OnceLock<Mutex<()>> = OnceLock::new();
    GPU.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn test_weights() -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dims = crate::net::MODEL_TAG.to_vec();
    let l = NetLayout::new(&dims).expect("dims");
    let mut rng = Rng::new(0xD15EA5E);
    let w: Vec<f32> = (0..l.w_len)
        .map(|_| (rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 0.1)
        .collect();
    let b = vec![0.0; l.b_len];
    let mut ln = vec![0.0; l.ln_len];
    // A norm's `(gamma, beta)` offsets are adjacent, so their gap is its width.
    for &(gain, shift) in &l.norms {
        ln[gain..shift].fill(1.0);
    }
    (dims, w, b, ln)
}

fn nets(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Nets {
    Nets {
        value: Net::from_flat(dims, w, b, ln).expect("test weights"),
    }
}

fn skew(live: &[Vec<f32>; 2], i: usize) -> [Vec<f32>; 2] {
    let one = |v: &Vec<f32>| {
        let mut x: Vec<f32> = v
            .iter()
            .enumerate()
            .map(|(c, &p)| p + if c % (i + 2) == 0 { 0.5 } else { 0.0 })
            .collect();
        let sum: f32 = x.iter().sum();
        for p in &mut x {
            *p /= sum;
        }
        x
    };
    [one(&live[0]), one(&live[1])]
}

fn fixtures<'a>(nets: &'a Nets) -> Vec<(Solver<'a>, PackedJob)> {
    let setup = Cfg {
        depth: 1,
        iters: 1,
        snapshots: false,
        ..Default::default()
    };
    let gc = GameCfg {
        agents: [Agent::Rebel {
            cfg: setup,
            slot: 0,
        }; 2],
        collect: Collect::None,
        explore: 0.0,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let roots = collect_roots(1, 0xABCD, std::slice::from_ref(nets), &gc, 4000);
    assert!(roots.len() >= 4, "fixture game produced too few roots");
    let round_end = roots
        .windows(2)
        .position(|x| x[0].0.round != x[1].0.round)
        .expect("fixture never crossed a round draw");
    let mut picks = vec![0, round_end, roots.len() / 2, roots.len() - 1];
    picks.sort_unstable();
    picks.dedup();
    assert_eq!(picks.len(), 4, "fixture indices collapsed");
    picks
        .into_iter()
        .map(|i| {
            let (state, belief) = &roots[i];
            let mut sv = Solver::new(
                state,
                crate::rebel::Ctx::new(state),
                nets,
                TEST_CFG,
                belief.clone(),
            );
            sv.leaf_values(0);
            let live = [belief[0].p.clone(), belief[1].p.clone()];
            let carried: Vec<_> = (0..N_CARRIED).map(|k| skew(&live, k)).collect();
            let job = PackedJob::from_solver(&sv, &carried);
            (sv, job)
        })
        .collect()
}

fn cmp(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let (at, ratio, diff, tolerance) = got
        .iter()
        .zip(want)
        .enumerate()
        .map(|(i, (&a, &b))| {
            assert!(a.is_finite() && b.is_finite(), "{name}: non-finite at {i}");
            let diff = (a - b).abs();
            let tolerance = atol + rtol * a.abs().max(b.abs());
            (i, diff / tolerance.max(f32::MIN_POSITIVE), diff, tolerance)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0, 0.0, atol));
    if std::env::var_os("WARCHEST_REPORT_DIFFS").is_some() {
        eprintln!(
            "{name}: worst error/tolerance {ratio:.3}, diff {diff:.3e}, tolerance {tolerance:.3e} at {at}"
        );
    }
    assert!(
        ratio <= 1.0,
        "{name}: error/tolerance {ratio:.2}, diff {diff:.3e} > {tolerance:.3e} at {at}: gpu {:.8e}, reference {:.8e}",
        got.get(at).copied().unwrap_or(0.0),
        want.get(at).copied().unwrap_or(0.0),
    );
}

fn assert_probabilities(name: &str, values: &[f32], offsets: &[u32]) {
    assert_eq!(offsets.first().copied(), Some(0), "{name}: first offset");
    assert_eq!(
        offsets.last().copied().map(|x| x as usize),
        Some(values.len()),
        "{name}: final offset"
    );
    for (row, bounds) in offsets.windows(2).enumerate() {
        let lo = bounds[0] as usize;
        let hi = bounds[1] as usize;
        assert!(lo <= hi && hi <= values.len(), "{name}: bad row {row}");
        if lo == hi {
            continue;
        }
        let mut sum = 0.0f32;
        for (col, &x) in values[lo..hi].iter().enumerate() {
            assert!(x.is_finite(), "{name}: non-finite at row {row}, col {col}");
            assert!(
                (-2e-6..=1.0 + 2e-6).contains(&x),
                "{name}: {x} outside probability bounds at row {row}, col {col}"
            );
            sum += x;
        }
        assert!((sum - 1.0).abs() <= 2e-4, "{name}: row {row} sums to {sum}");
    }
}

fn assert_result_invariants(job: &PackedJob, result: &SolveResult) {
    assert_eq!(result.strategy.len(), job.tables.cells, "strategy shape");
    assert_probabilities("strategy", &result.strategy, &job.tables.legal_off);

    assert_eq!(result.root_values.len(), job.carried.len(), "root count");
    for (root, pair) in result.root_values.iter().enumerate() {
        for p in 0..2 {
            assert_eq!(
                pair[p].len(),
                job.root[p].len(),
                "root {root}, player {p}: shape"
            );
            assert!(
                pair[p].iter().all(|x| x.is_finite()),
                "root {root}, player {p}: non-finite"
            );
        }
    }

    let carries = &result.carries;
    let snapshots = job.meta.snap_iters.len().saturating_sub(1);
    assert_eq!(carries.snapshots, snapshots, "carry snapshot count");
    assert_eq!(
        carries.coff.len(),
        2 * carries.exit_nodes.len() + 1,
        "carry offset shape"
    );
    assert_eq!(
        carries.data.len(),
        carries.snapshots * carries.snapshot_configs,
        "carry data shape"
    );
    for snap in 0..carries.snapshots {
        let values = carries.snapshot(snap);
        assert_probabilities(&format!("carry snapshot {snap}"), &values, &carries.coff);
    }
}

fn flatten_pairs(v: &[[Vec<f32>; 2]]) -> Vec<f32> {
    v.iter()
        .flat_map(|x| x[0].iter().chain(&x[1]))
        .copied()
        .collect()
}

#[test]
fn full_wave_oracle() {
    let _gpu = gpu_guard();
    let (dims, w, b, ln) = test_weights();
    let nets = nets(&dims, &w, &b, &ln);
    let set = fixtures(&nets);
    let gpu = spawn(0, dims, w, b, ln, true).expect("GPU executor");
    let pending: Vec<_> = set
        .iter()
        .map(|(_, j)| gpu.submit(j.clone()).expect("submit"))
        .collect();
    let got: Vec<_> = pending
        .into_iter()
        .map(|h| h.wait().expect("solve"))
        .collect();

    for (tree, ((mut sv, job), result)) in set.into_iter().zip(got).enumerate() {
        assert_result_invariants(&job, &result);
        sv.multistep(TEST_CFG.iters);
        let strategy_tol = (5e-3, 2e-3);
        cmp(
            &format!("tree {tree} strategy"),
            &result.strategy,
            sv.snaps.last().expect("CPU reference strategy"),
            strategy_tol.0,
            strategy_tol.1,
        );
        let want_roots = sv.value_under(&job.carried);
        let root_tol = (1e-3, 2e-4);
        cmp(
            &format!("tree {tree} root values"),
            &flatten_pairs(&result.root_values),
            &flatten_pairs(&want_roots),
            root_tol.0,
            root_tol.1,
        );
        let leaf = sv
            .leaf_rows
            .first()
            .or_else(|| sv.term_leaves.first())
            .copied()
            .expect("leaf");
        let got_carry = result.carries.select(leaf as u32).expect("carry leaf");
        let want_carry = sv.carried_beliefs(leaf);
        cmp(
            &format!("tree {tree} carried beliefs"),
            &flatten_pairs(&got_carry),
            &flatten_pairs(&want_carry),
            // Carries stay FP16 even when the GEMMs run in precise mode.
            6e-4,
            6e-4,
        );
    }
}

#[test]
fn zero_network_oracle() {
    let _gpu = gpu_guard();
    let (dims, mut w, b, ln) = test_weights();
    w.fill(0.0);
    let nets = nets(&dims, &w, &b, &ln);
    let set = fixtures(&nets);
    let gpu = spawn(0, dims, w, b, ln, true).expect("GPU executor");
    let pending: Vec<_> = set
        .iter()
        .map(|(_, j)| gpu.submit(j.clone()).expect("submit"))
        .collect();
    let got: Vec<_> = pending
        .into_iter()
        .map(|h| h.wait().expect("solve"))
        .collect();

    for (tree, ((mut sv, job), result)) in set.into_iter().zip(got).enumerate() {
        assert_result_invariants(&job, &result);
        sv.multistep(TEST_CFG.iters);
        cmp(
            &format!("zero tree {tree} strategy"),
            &result.strategy,
            sv.snaps.last().expect("CPU reference strategy"),
            2e-4,
            2e-4,
        );
        let want_roots = sv.value_under(&job.carried);
        cmp(
            &format!("zero tree {tree} root values"),
            &flatten_pairs(&result.root_values),
            &flatten_pairs(&want_roots),
            2e-4,
            2e-4,
        );
        let leaf = sv
            .leaf_rows
            .first()
            .or_else(|| sv.term_leaves.first())
            .copied()
            .expect("leaf");
        let got_carry = result.carries.select(leaf as u32).expect("carry leaf");
        let want_carry = sv.carried_beliefs(leaf);
        cmp(
            &format!("zero tree {tree} carried beliefs"),
            &flatten_pairs(&got_carry),
            &flatten_pairs(&want_carry),
            2e-4,
            2e-4,
        );
    }
}

#[test]
fn wave_composition_stays_bounded() {
    let _gpu = gpu_guard();
    let (dims, w, b, ln) = test_weights();
    let nets = nets(&dims, &w, &b, &ln);
    let set = fixtures(&nets);
    for measured_id in 0..set.len() {
        let job = set[measured_id].1.clone();
        let gpu = spawn(0, dims.clone(), w.clone(), b.clone(), ln.clone(), true).expect("GPU");
        let alone = gpu.submit(job.clone()).unwrap().wait().unwrap();
        let mut pending: Vec<_> = (0..15)
            .map(|i| gpu.submit(set[i % set.len()].1.clone()).unwrap())
            .collect();
        let measured = gpu.submit(job).unwrap();
        pending.push(measured);
        let mut results: Vec<_> = pending.into_iter().map(|h| h.wait().unwrap()).collect();
        let together = results.pop().unwrap();
        let after = gpu
            .submit(set[measured_id].1.clone())
            .unwrap()
            .wait()
            .unwrap();
        assert_result_invariants(&set[measured_id].1, &alone);
        assert_result_invariants(&set[measured_id].1, &together);
        assert_result_invariants(&set[measured_id].1, &after);
        let company_strategy_tol = (5e-3, 2e-3);
        cmp(
            &format!("tree {measured_id} strategy depends on wave company"),
            &together.strategy,
            &alone.strategy,
            company_strategy_tol.0,
            company_strategy_tol.1,
        );
        let reuse_tol = (1e-6, 1e-6);
        cmp(
            &format!("tree {measured_id} strategy depends on reused capacity"),
            &after.strategy,
            &alone.strategy,
            reuse_tol.0,
            reuse_tol.1,
        );
        cmp(
            &format!("tree {measured_id} root values depend on reused capacity"),
            &flatten_pairs(&after.root_values),
            &flatten_pairs(&alone.root_values),
            reuse_tol.0,
            reuse_tol.1,
        );
        let company_root_tol = (1e-3, 2e-4);
        cmp(
            &format!("tree {measured_id} root values depend on wave company"),
            &flatten_pairs(&together.root_values),
            &flatten_pairs(&alone.root_values),
            company_root_tol.0,
            company_root_tol.1,
        );
    }
}

/// The production head multiplies in FP16 on the tensor cores, so it is a
/// different numerical machine from the exact path the oracles above pin. What
/// training consumes is the root value; bound that against exact math, and
/// require the same structural invariants. The strategy is a ratio of
/// differences between leaf values and magnifies the same rounding by two
/// orders, which is what the looser bound here records.
#[test]
fn tensor_core_head_tracks_exact_math() {
    let _gpu = gpu_guard();
    let (dims, w, b, ln) = test_weights();
    let nets = nets(&dims, &w, &b, &ln);
    let set = fixtures(&nets);
    let solve = |precise: bool| {
        let gpu = spawn(0, dims.clone(), w.clone(), b.clone(), ln.clone(), precise)
            .expect("GPU executor");
        let pending: Vec<_> = set
            .iter()
            .map(|(_, j)| gpu.submit(j.clone()).expect("submit"))
            .collect();
        pending
            .into_iter()
            .map(|h| h.wait().expect("solve"))
            .collect::<Vec<_>>()
    };
    let exact = solve(true);
    let fast = solve(false);
    for (tree, ((_, job), got)) in set.iter().zip(&fast).enumerate() {
        assert_result_invariants(job, got);
        // v5 pools all ten projected type tokens through GELU into the board
        // stem. Fast-F16 therefore accumulates their projection error once per
        // leaf instead of only on occupied hexes.
        cmp(
            &format!("tree {tree} fast root values"),
            &flatten_pairs(&got.root_values),
            &flatten_pairs(&exact[tree].root_values),
            1e-2,
            2e-3,
        );
        let mean_strategy_error = got
            .strategy
            .iter()
            .zip(&exact[tree].strategy)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / got.strategy.len() as f32;
        assert!(
            mean_strategy_error <= 5e-3,
            "tree {tree} fast strategy: mean absolute error {mean_strategy_error:.3e}"
        );
        // Near-tied CFR regrets amplify a small leaf perturbation at isolated
        // cells. Bound the whole policy tightly above and retain a guard
        // against a gross pointwise error here.
        cmp(
            &format!("tree {tree} fast strategy"),
            &got.strategy,
            &exact[tree].strategy,
            4e-1,
            1e-1,
        );
    }
}
