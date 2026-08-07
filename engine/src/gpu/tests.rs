//! Work-package-B oracle tests: the CPU solver is the oracle, the GPU service
//! must reproduce it. Run on a CUDA box with `cargo test --features gpu`
//! (the laptop has no CUDA). The phase tests compare kernel arenas with the
//! matching Rust functions phase by phase (plan B4.2); the full-solve test
//! compares the trip-1/trip-2 outputs (B4.3); the batch-invariance test pins
//! bit-for-bit determinism across live-set compositions (B4.4).

use crate::rng::Rng;
use crate::search::{Back, Cfg, Nets, Solver};
use crate::selfplay::{collect_roots, Agent, Collect, GameCfg};
use crate::serialize::Job;

use super::client::GpuClient;
use super::service::{spawn, Service};

/// The arenas after a probe phase.
pub struct ProbeOut {
    pub reach: Vec<f32>,
    pub vals: Vec<f32>,
    pub regret: Vec<f32>,
    pub inst: Vec<f32>,
    pub cur: Vec<f32>,
    pub sum_strat: Vec<f32>,
    pub avg: Vec<f32>,
    pub snaps: Vec<f32>,
    pub xb: Vec<f32>,
    pub u: Vec<f32>,
}

/// Deterministic random weights, scaled so activations stay near +-1 (the
/// scale real training produces), keeping the plan's absolute tolerances
/// (1e-5 phases, 1e-4 full solve) honest.
fn test_weights() -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(0xD15EA5E);
    let dims: Vec<usize> = vec![
        crate::rebel::PUBFEAT, 384, 384, crate::rebel::CFEAT, 64, 64,
        crate::rebel::AFEAT, 32, 64, 0,
    ];
    let (h, hd, dg, rk, de, dc) = (384usize, 384usize, 64usize, 64usize, 32usize, 64usize);
    let (af, hf, xd) = (
        crate::rebel::AFEAT + de,
        4 + de,
        crate::board::N_HEXES * (crate::rebel::HEX_FACTS + de) + 2 * de + crate::rebel::LOOSE,
    );
    let cf = crate::units::CARD_FEATS;
    let n_w = cf * dc + dc * de + crate::units::N_UNITS * de + (4 + de) * de
        + xd * h + h * hd + 2 * dg * hd + hf * dg + dg * dg + dg * dg
        + dg * (rk + 1) + hd * rk + af * rk + dg * rk + hd * rk;
    let n_b = dc + de + de + h + hd + dg + dg + dg + (rk + 1) + 4 * rk;
    let n_ln = h + hd + h + hd;
    let mut w = Vec::with_capacity(n_w);
    for _ in 0..n_w {
        w.push((rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 0.6);
    }
    let b = vec![0.0f32; n_b];
    let mut ln = vec![0.0f32; n_ln];
    for i in 0..h + hd {
        ln[i] = 1.0;
    }
    (dims, w, b, ln)
}

fn test_nets() -> Nets {
    let (dims, w, b, ln) = test_weights();
    Nets { value: crate::net::Mlp::from_flat(&dims, &w, &b, &ln).expect("weights") }
}

const TEST_CFG: Cfg = Cfg {
    depth: 2,
    iters: 8,
    snapshots: true,
    cfr: crate::search::Cfr::LINEAR,
    warm: 0.0,
    node_cap: 0,
};

/// A solver on a real subgame root (collected under the same weights), its
/// serialized job, and the live belief (the Phase-2 root list).
fn test_solve<'a>(
    nets: &'a Nets,
    cfg: Cfg,
    random: bool,
) -> (Solver<'a>, Job, Vec<[Vec<f32>; 2]>) {
    let nets_arr = [Nets { value: crate::net::Mlp::from_flat(
        &test_weights().0, &test_weights().1, &test_weights().2, &test_weights().3,
    ).unwrap() }];
    let gc = GameCfg {
        agents: [
            Agent::Rebel { cfg: Cfg { depth: 2, iters: 4, snapshots: false, ..Default::default() }, slot: 0 },
            Agent::Rebel { cfg: Cfg { depth: 2, iters: 4, snapshots: false, ..Default::default() }, slot: 0 },
        ],
        collect: Collect::None,
        explore: 0.0,
        random_draft: random,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let roots = collect_roots(6, 0xABCD, &nets_arr, &gc, 4);
    let (s, bel) = roots.into_iter().next().expect("root");
    let ctx = crate::rebel::Ctx::new(&s);
    let sv = Solver::new(&s, ctx, nets, cfg, bel.clone());
    let carried = vec![[bel[0].p.clone(), bel[1].p.clone()]];
    let job = Job::from_solver(&sv, &carried);
    (sv, job, carried)
}

/// An in-process service (no thread) with the test weights, for the phase
/// probes.
fn start_probe() -> Service {
    let (dims, w, b, ln) = test_weights();
    let (_tx, rx) = std::sync::mpsc::channel();
    Service::new(rx, dims, w, b, ln).expect("service")
}

/// A running service thread with the test weights, for the client round
/// trips.
fn start_client() -> GpuClient {
    let (dims, w, b, ln) = test_weights();
    spawn(dims, w, b, ln).expect("spawn")
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn cmp(name: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let scale = a.iter().chain(b.iter()).map(|x| x.abs()).fold(0.0, f32::max);
    let d = maxdiff(a, b);
    assert!(
        d <= atol + rtol * scale,
        "{name}: max diff {d:.3e} > {atol} + {rtol}*{scale:.3e}"
    );
}

fn flatten(v: &[Vec<f32>]) -> Vec<f32> {
    v.iter().flatten().cloned().collect()
}

/// Phase oracle tests (plan B4.2): each kernel's output arena against the
/// matching Rust function's, in B3's order. Phase 0 = strategy init,
/// 1 = belief sums + head, 2 = readout, 3 = backward sweep,
/// 4 = regret matching, 5 = forward reach, 6 = average.
#[test]
fn phase_oracle() {
    let mut svc = start_probe();
    let nets = test_nets();
    let (mut sv, job, _) = test_solve(&nets, TEST_CFG, true);

    // Phase 0: init — uniform cur/avg, zero regrets, the reach-weighted
    // seed, snapshot 0. Exact.
    let p0 = svc.probe_phase(job.clone(), 0).unwrap();
    cmp("p0 cur", &p0.cur, &sv.cur, 0.0, 1e-6);
    cmp("p0 avg", &p0.avg, &flatten(&sv.avg), 0.0, 1e-6);
    cmp("p0 sum_strat", &p0.sum_strat, &flatten(&sv.sum_strat), 1e-6, 1e-6);
    cmp("p0 reach", &p0.reach, &sv.reach, 0.0, 1e-6);
    cmp("p0 snaps", &p0.snaps, &flatten(&sv.snaps), 0.0, 1e-6);

    // Phase 1: belief sums + head (xb, u).
    let p1 = svc.probe_phase(job.clone(), 1).unwrap();
    sv.leaf_values(0);
    cmp("p1 xb", &p1.xb, &sv.xb[..sv.leaf_rows.len() * 2 * 64], 1e-5, 1e-5);
    cmp("p1 u", &p1.u, &sv.ob[..sv.leaf_rows.len() * 64], 1e-5, 1e-5);

    // Phase 2: readout (vals).
    let p2 = svc.probe_phase(job.clone(), 2).unwrap();
    sv.readout(0);
    cmp("p2 vals", &p2.vals, &sv.vals, 1e-5, 1e-5);

    // Phase 3: backward sweep (inst + vals).
    let p3 = svc.probe_phase(job.clone(), 3).unwrap();
    let cur = std::mem::take(&mut sv.cur);
    sv.backprop(0, &cur, Back::Regret);
    sv.cur = cur;
    cmp("p3 inst", &p3.inst, &sv.inst, 1e-5, 1e-5);
    cmp("p3 vals", &p3.vals, &sv.vals, 1e-5, 1e-5);

    // Phase 4: regret matching (regret, cur, sum_strat discount).
    let p4 = svc.probe_phase(job.clone(), 4).unwrap();
    sv.rm_block(0);
    cmp("p4 regret", &p4.regret, &sv.regret, 1e-5, 1e-5);
    cmp("p4 cur", &p4.cur, &sv.cur, 1e-5, 1e-5);
    cmp("p4 sum_strat", &p4.sum_strat, &flatten(&sv.sum_strat), 1e-5, 1e-5);

    // Phase 5: forward reach (reach under the new strategy).
    let p5 = svc.probe_phase(job.clone(), 5).unwrap();
    sv.precompute_reaches();
    cmp("p5 reach", &p5.reach, &sv.reach, 1e-5, 1e-5);

    // Phase 6: average (avg after one full step).
    let p6 = svc.probe_phase(job.clone(), 6).unwrap();
    sv.avg_block(0);
    cmp("p6 avg", &p6.avg, &flatten(&sv.avg), 1e-5, 1e-5);
}

/// Full-solve oracle test (plan B4.3): the GPU solve against the CPU solve,
/// same tree, same weights. Tolerance 1e-4 on the root values, the reference
/// strategy, and the carried beliefs.
#[test]
fn full_solve_oracle() {
    let nets = test_nets();
    let (mut sv, job, carried) = test_solve(&nets, TEST_CFG, true);
    let client = start_client();
    // CPU reference.
    sv.multistep(TEST_CFG.iters);
    let vals = sv.value_under(&carried);
    let leaf = *sv.leaf_rows.first().expect("leaves");
    let cpu_carried = sv.carried_beliefs(leaf);
    let reference = sv.snaps.last().cloned().unwrap_or_default();
    // GPU.
    let trip1 = client.solve(job, &carried).expect("trip1");
    let gpu_carried = client.carried_beliefs(trip1.id, leaf as u32).expect("trip2");
    cmp("reference strategy", &trip1.strategy, &reference, 1e-4, 1e-4);
    assert_eq!(trip1.root_values.len(), vals.len());
    for (k, (g, c)) in trip1.root_values.iter().zip(vals.iter()).enumerate() {
        cmp(&format!("root[{k}].0"), &g[0], &c[0], 1e-4, 1e-4);
        cmp(&format!("root[{k}].1"), &g[1], &c[1], 1e-4, 1e-4);
    }
    assert_eq!(gpu_carried.len(), cpu_carried.len());
    for (k, (g, c)) in gpu_carried.iter().zip(cpu_carried.iter()).enumerate() {
        cmp(&format!("carried[{k}].0"), &g[0], &c[0], 1e-4, 1e-4);
        cmp(&format!("carried[{k}].1"), &g[1], &c[1], 1e-4, 1e-4);
    }
}


/// Zero-network determinism (plan B5.4): with all-zero weights every strategy
/// is uniform at every iterate, so fixed-seed games must match the CPU build
/// move for move. The whole `Data` (rows, targets, policy labels, stats)
/// must come out identical.
#[test]
fn zero_network_determinism() {
    let (dims, _, _, ln0) = test_weights();
    // All-zero weights with the identity LayerNorms (scale 1, bias 0).
    let nw = dims.clone();
    let (dims, mut w, mut b, ln) = test_weights();
    let _ = (&nw, &mut w, &mut b, &ln0);
    w.iter_mut().for_each(|x| *x = 0.0);
    b.iter_mut().for_each(|x| *x = 0.0);
    let nets = [Nets { value: crate::net::Mlp::from_flat(&dims, &w, &b, &ln).expect("zero net") }];
    let gc = GameCfg {
        agents: [
            Agent::Rebel { cfg: TEST_CFG, slot: 0 },
            Agent::Rebel { cfg: TEST_CFG, slot: 0 },
        ],
        collect: Collect::Rebel,
        explore: 0.25,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let gpu = start_client();
    let seed = 0x5EED;
    let cpu = crate::selfplay::run_games(4, seed, &nets, &gc);
    let gpu_data = crate::selfplay::run_games_gpu(4, seed, &nets, &gc, &gpu);
    assert_data_eq(&cpu, &gpu_data);
}

fn assert_data_eq(a: &crate::selfplay::Data, b: &crate::selfplay::Data) {
    assert_eq!(a.rows, b.rows, "rows differ");
    assert_eq!(a.cc, b.cc, "config counts differ");
    assert_eq!(a.cw, b.cw, "belief weights differ");
    assert_eq!(a.cy, b.cy, "targets differ");
    assert_eq!(a.nv, b.nv, "row count differs");
    assert_eq!(a.decisions, b.decisions, "decision count differs");
    assert_eq!(a.configs, b.configs, "config count differs");
    assert_eq!(a.games, b.games, "game count differs");
    assert_eq!(a.wins, b.wins, "wins differ");
    assert_eq!(a.draws, b.draws, "draws differ");
    assert_eq!(a.cap_hits, b.cap_hits, "cap hits differ");
    assert_eq!(a.pa, b.pa, "policy actions differ");
    assert_eq!(a.pp, b.pp, "policy probabilities differ");
    assert_eq!(a.prow, b.prow, "policy rows differ");
    assert_eq!(a.pact, b.pact, "policy players differ");
}

/// Batch invariance (plan B4.4): solve one tree alone, then the same tree
/// inside a full live set. The results must match bit for bit.
#[test]
fn batch_invariance() {
    let nets = test_nets();
    let (_, job_a, carried_a) = test_solve(&nets, TEST_CFG, true);
    let leaf_a = job_a.tables.leaf_rows[0];
    let (_, job_b, carried_b) = test_solve(&nets, TEST_CFG, false);
    let client = start_client();
    // Solo.
    let solo = client.solve(job_a.clone(), &carried_a).expect("solo trip1");
    let solo_carried = client.carried_beliefs(solo.id, leaf_a).expect("solo trip2");
    // Paired: submit A and B, let them share ticks.
    let ha = client.submit(job_a).expect("submit a");
    let hb = client.submit(job_b).expect("submit b");
    let ta = ha.wait().expect("a trip1");
    let _tb = hb.wait().expect("b trip1");
    assert_eq!(solo.strategy.len(), ta.strategy.len());
    assert_eq!(solo.strategy, ta.strategy, "strategy must be bit-identical");
    for (g, c) in solo.root_values.iter().zip(ta.root_values.iter()) {
        assert_eq!(g[0], c[0], "root values must be bit-identical");
        assert_eq!(g[1], c[1]);
    }
    // Trip 2 for the paired A must match the solo A.
    let paired_carried = client.carried_beliefs(ta.id, leaf_a).expect("paired trip2");
    assert_eq!(solo_carried.len(), paired_carried.len());
    for (g, c) in solo_carried.iter().zip(paired_carried.iter()) {
        assert_eq!(g[0], c[0], "carried beliefs must be bit-identical");
        assert_eq!(g[1], c[1]);
    }
    let _ = carried_b;
}
