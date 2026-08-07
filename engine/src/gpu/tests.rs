//! Work-package-B oracle tests: the CPU solver is the oracle and the GPU
//! service must reproduce it. These need a CUDA device, so they run on a box
//! with one (`cargo test --features gpu`), not on the laptop.
//!
//! The phase tests compare each kernel's arena with the matching Rust
//! function, in the order a tick runs them (plan B4.2); the full-solve test
//! compares the trip-1 and trip-2 outputs (B4.3); the batch-invariance test
//! pins that a solve's answer does not depend on what shares its tick (B4.4).

use crate::rng::Rng;
use crate::search::{Back, Cfg, Nets, Solver};
use crate::selfplay::{collect_roots, Agent, Collect, GameCfg};
use crate::serialize::Job;

use super::client::GpuClient;
use super::service::{spawn, Service, Step};

const DG: usize = 64;
const RK: usize = 64;

/// The classic shape, in the v3 tower format: card [64], pub [384],
/// head 384, no extra head layers, no slot hiddens, one residual block.
fn test_dims() -> Vec<usize> {
    vec![3, 32, DG, RK, 384, 1, 1, 64, 1, 384, 0, 0]
}

/// Deterministic random weights, scaled so activations stay near +-1 — the
/// range real training produces, which is what makes the plan's absolute
/// tolerances (1e-5 per phase, 1e-4 per solve) meaningful. The lengths come
/// from the shared layout, so the test cannot disagree with the service.
fn test_weights() -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dims = test_dims();
    let l = crate::net::V3Layout::new(&dims).expect("dims");
    let mut rng = Rng::new(0xD15EA5E);
    let w = (0..l.w_len)
        .map(|_| (rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 0.6)
        .collect();
    let b = vec![0.0f32; l.b_len];
    let mut ln = vec![0.0f32; l.ln_len];
    // LayerNorm gains start at one; their shifts stay at zero.
    for (g, _) in l.pub_ln.iter() {
        for x in ln[*g..*g + 384].iter_mut() {
            *x = 1.0;
        }
    }
    for x in ln[l.ln1.0..l.ln1.0 + l.head_in].iter_mut() {
        *x = 1.0;
    }
    (dims, w, b, ln)
}

fn nets_from(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Nets {
    Nets { value: crate::net::Mlp::from_flat(dims, w, b, ln).expect("weights") }
}

const TEST_CFG: Cfg = Cfg {
    depth: 2,
    iters: 8,
    snapshots: true,
    cfr: crate::search::Cfr::LINEAR,
    warm: 0.0,
    node_cap: 0,
    gpu_build: false,
};

/// Real subgame roots and the jobs that describe them. One tree exercises
/// perhaps half the branches in the sweeps — chance nodes, terminal leaves and
/// ragged config counts all vary from root to root — so every oracle test runs
/// over a set of them and `assert_varied` refuses a degenerate set.
fn test_solves(nets: &Nets, cfg: Cfg, random: bool) -> Vec<(Solver<'_>, Job)> {
    let inner = Cfg { depth: 2, iters: 4, snapshots: false, ..Default::default() };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg: inner, slot: 0 }, Agent::Rebel { cfg: inner, slot: 0 }],
        collect: Collect::None,
        explore: 0.0,
        random_draft: random,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    // Roots come out in game order, so a fixed stride through them gives
    // four opening subgames — and an opening subgame has no terminal leaf.
    // The last root of a completed game is the game's final decision, so its
    // subgame does terminate; that one plus a spread of earlier roots covers
    // the shapes. One game is enough and takes about a second.
    let pool = collect_roots(1, 0xABCD, std::slice::from_ref(nets), &gc, 4000);
    assert!(pool.len() >= N_TREES, "not enough roots to choose from");
    let last = pool.len() - 1;
    let step = last / (N_TREES - 1);
    let pick: Vec<usize> = (0..N_TREES - 1).map(|i| i * step).chain([last]).collect();
    let out: Vec<_> = pool
        .into_iter()
        .enumerate()
        .filter(|(i, _)| pick.contains(i))
        .map(|(_, (state, bel))| {
            let ctx = crate::rebel::Ctx::new(&state);
            let mut sv = Solver::new(&state, ctx, nets, cfg, bel.clone());
            // Build the leaf batch before serializing. `TreeTables::from_solver`
            // starts from the solver's interned config table and adds what is
            // missing, so serializing first would leave the job and the solver
            // with two independently ordered tables — and every comparison in
            // the oracle assumes one order.
            sv.leaf_values(0);
            // Several carried roots, not one: the value stage indexes them by
            // step, and a single root would leave that indexing untested.
            let live = [bel[0].p.clone(), bel[1].p.clone()];
            let carried: Vec<_> = (0..N_CARRIED).map(|i| skew(&live, i)).collect();
            let job = Job::from_solver(&sv, &carried);
            (sv, job)
        })
        .collect();
    assert_varied(&out);
    out
}

const N_TREES: usize = 4;
const N_CARRIED: usize = 3;

/// A belief that is not the live one, so the carried roots differ from each
/// other and from the root: tilt the mass towards config `i`, renormalised.
fn skew(live: &[Vec<f32>; 2], i: usize) -> [Vec<f32>; 2] {
    let one = |v: &Vec<f32>| {
        let mut w: Vec<f32> = v.iter().enumerate()
            .map(|(c, &p)| p + if c % (i + 2) == 0 { 0.5 } else { 0.0 })
            .collect();
        let tot: f32 = w.iter().sum();
        for x in w.iter_mut() {
            *x /= tot;
        }
        w
    };
    [one(&live[0]), one(&live[1])]
}

/// The fixture must actually reach the code it claims to test. Passing on
/// four trees means nothing if all four are one decision node deep.
fn assert_varied(set: &[(Solver, Job)]) {
    fn t(j: &Job) -> &crate::serialize::TreeTables { &j.tables }
    assert!(set.len() >= 2, "need several trees");
    assert!(
        set.iter().any(|(_, j)| t(j).node_kind.iter().any(|&k| k == 1)),
        "no chance node in any tree: the draw transitions are untested"
    );
    assert!(
        set.iter().any(|(_, j)| t(j).nterm > 0),
        "no terminal leaf in any tree: the utility path is untested"
    );
    assert!(
        set.iter().all(|(_, j)| t(j).nlevels >= 3),
        "a tree is too shallow to exercise the level sweeps"
    );
    assert!(
        set.iter().any(|(_, j)| {
            let c = &t(j).cfg_off;
            (0..t(j).nodes).map(|i| c[2 * i + 1] - c[2 * i]).max()
                > (0..t(j).nodes).map(|i| c[2 * i + 1] - c[2 * i]).min()
        }),
        "every node has the same config count: the ragged path is untested"
    );
}

fn start_probe() -> Service {
    let (dims, w, b, ln) = test_weights();
    let (_tx, rx) = std::sync::mpsc::channel();
    Service::new(0, rx, dims, w, b, ln).expect("service")
}

fn start_client() -> GpuClient {
    let (dims, w, b, ln) = test_weights();
    spawn(0, dims, w, b, ln).expect("spawn")
}

fn cmp(name: &str, a: &[f32], b: &[f32], atol: f32, rtol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let scale = a.iter().chain(b.iter()).map(|x| x.abs()).fold(0.0, f32::max);
    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    assert!(d <= atol + rtol * scale, "{name}: max diff {d:.3e} > {atol} + {rtol}*{scale:.3e}");
}

fn flatten(v: &[Vec<f32>]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
}

/// Phase oracle (plan B4.2): each kernel's arena against the matching Rust
/// function, in the order `Service::iterate` runs them, over every tree in
/// the fixture. Each probe replays the iteration from admission and stops one
/// phase later, so the first failure names the first wrong kernel — and
/// naming the tree tells you which shape provoked it.
#[test]
fn phase_oracle() {
    let (dims, w, b, ln) = test_weights();
    let nets = nets_from(&dims, &w, &b, &ln);
    let mut svc = start_probe();
    for (t, (mut sv, job)) in test_solves(&nets, TEST_CFG, true).into_iter().enumerate() {
        let at = |s: &str| format!("tree {t} {s}");
        let (rows, hd) = (sv.leaf_rows.len(), 384);
        // The job's config table extends the solver's: the serializer interns
        // any support member the leaf batch missed, appending after the rows
        // the batch already holds. Those extra rows have no CPU counterpart,
        // so the towers are compared over the shared prefix.
        let (nz, ng) = (sv.ncfg * DG, sv.ncfg * (RK + 1));
        assert!(job.tables.ncfg >= sv.ncfg, "the job's config table must extend the solver's");

        // The build: the card table, the holding tower, and the trunk. Each
        // ends one independent chain, so three comparisons localise a build
        // failure without exposing the scratch in between.
        let p = svc.probe(job.clone(), Step::Build).unwrap();
        cmp(&at("build e"), &p.e, &sv.ce, 1e-5, 1e-5);
        cmp(&at("build z"), &p.z[..nz], &sv.cz[..nz], 1e-5, 1e-5);
        cmp(&at("build g"), &p.g[..ng], &sv.cg[..ng], 1e-5, 1e-5);
        cmp(&at("build h0"), &p.h0, &sv.h0[..rows * hd], 1e-5, 1e-5);

        // Admission: uniform strategy, zero regrets, the reach-weighted seed.
        let p = svc.probe(job.clone(), Step::None).unwrap();
        cmp(&at("init cur"), &p.cur, &sv.cur, 0.0, 1e-6);
        cmp(&at("init avg"), &p.avg, &flatten(&sv.avg), 0.0, 1e-6);
        cmp(&at("init sum_strat"), &p.sum_strat, &flatten(&sv.sum_strat), 1e-6, 1e-6);
        cmp(&at("init reach"), &p.reach, &sv.reach, 0.0, 1e-6);

        // The head: the belief sums, then the two GEMMs around LayerNorm.
        let p = svc.probe(job.clone(), Step::Head).unwrap();
        cmp(&at("head xb"), &p.xb, &sv.xb[..rows * 2 * DG], 1e-5, 1e-5);
        cmp(&at("head u"), &p.u, &sv.ob[..rows * RK], 1e-5, 1e-5);

        let p = svc.probe(job.clone(), Step::Readout).unwrap();
        sv.readout(0);
        cmp(&at("readout vals"), &p.vals, &sv.vals, 1e-5, 1e-5);

        let p = svc.probe(job.clone(), Step::Backprop).unwrap();
        let cur = std::mem::take(&mut sv.cur);
        sv.backprop(0, &cur, Back::Regret);
        sv.cur = cur;
        cmp(&at("backprop inst"), &p.inst, &sv.inst, 1e-5, 1e-5);
        cmp(&at("backprop vals"), &p.vals, &sv.vals, 1e-5, 1e-5);

        let p = svc.probe(job.clone(), Step::Regret).unwrap();
        sv.rm_block(0);
        cmp(&at("regret"), &p.regret, &sv.regret, 1e-5, 1e-5);
        cmp(&at("regret cur"), &p.cur, &sv.cur, 1e-5, 1e-5);
        cmp(&at("regret sum_strat"), &p.sum_strat, &flatten(&sv.sum_strat), 1e-5, 1e-5);

        let p = svc.probe(job.clone(), Step::Propagate).unwrap();
        sv.precompute_reaches();
        cmp(&at("propagate reach"), &p.reach, &sv.reach, 1e-5, 1e-5);

        let p = svc.probe(job, Step::Average).unwrap();
        sv.avg_block(0);
        cmp(&at("average avg"), &p.avg, &flatten(&sv.avg), 1e-5, 1e-5);
    }
}

/// Full-solve oracle (plan B4.3): the GPU solve against the CPU solve, same
/// trees and same weights, on both trips' outputs. Every tree in the fixture
/// is submitted at once, so the live set holds solves in different stages and
/// the tick's per-stage grouping is exercised rather than assumed.
#[test]
fn full_solve_oracle() {
    let (dims, w, b, ln) = test_weights();
    let nets = nets_from(&dims, &w, &b, &ln);
    let set = test_solves(&nets, TEST_CFG, true);
    let gpu = start_client();

    // Submit them all before waiting on any, so they interleave.
    let pending: Vec<_> = set
        .iter()
        .map(|(_, job)| gpu.submit(job.clone()).expect("submit"))
        .collect();
    let trips: Vec<_> = pending.into_iter().map(|h| h.wait().expect("trip 1")).collect();

    for (t, ((mut sv, job), t1)) in set.into_iter().zip(trips).enumerate() {
        let at = |s: String| format!("tree {t} {s}");
        let leaf = sv.leaf_rows[0];
        let got = gpu.carried_beliefs(t1.id, leaf as u32).expect("trip 2");
        let carried = job.carried;

        sv.multistep(TEST_CFG.iters);
        cmp(&at("strategy".into()), &t1.strategy, &flatten(&sv.avg), 1e-4, 1e-4);

        let want = sv.value_under(&carried);
        assert_eq!(t1.root_values.len(), want.len(), "one value set per carried root");
        assert!(want.len() > 1, "several carried roots, or the step index is untested");
        for (i, (g, e)) in t1.root_values.iter().zip(&want).enumerate() {
            cmp(&at(format!("root {i} p0")), &g[0], &e[0], 1e-4, 1e-4);
            cmp(&at(format!("root {i} p1")), &g[1], &e[1], 1e-4, 1e-4);
        }

        let exp = sv.carried_beliefs(leaf);
        assert_eq!(got.len(), exp.len(), "one belief per kept snapshot");
        assert!(!exp.is_empty(), "no kept snapshot, so trip 2 is untested");
        for (i, (g, e)) in got.iter().zip(&exp).enumerate() {
            cmp(&at(format!("belief {i} p0")), &g[0], &e[0], 1e-4, 1e-4);
            cmp(&at(format!("belief {i} p1")), &g[1], &e[1], 1e-4, 1e-4);
        }
    }
}

/// Batch invariance (plan B4.4): a solve alone and the same solve inside a
/// busy live set must agree bit for bit. A difference means a reduction
/// crossed a solve boundary, or cuBLAS split a row differently.
#[test]
fn batch_invariance() {
    let (dims, w, b, ln) = test_weights();
    let nets = nets_from(&dims, &w, &b, &ln);
    let set = test_solves(&nets, TEST_CFG, true);
    let job = set[0].1.clone();
    let carried = job.carried.clone();
    let gpu = start_client();

    let alone = gpu.solve(job.clone(), &carried).expect("alone");
    // Fill the live set with *different* trees, staggered, so the measured
    // solve shares its ticks with solves of other shapes at other iterations
    // and in other stages — the conditions a real live set produces.
    let crowd: Vec<_> = (0..16)
        .map(|i| gpu.submit(set[i % set.len()].1.clone()).expect("submit"))
        .collect();
    let together = gpu.solve(job, &carried).expect("together");
    for h in crowd {
        let _ = h.wait();
    }
    assert_eq!(alone.strategy, together.strategy, "strategy depends on company");
    assert_eq!(alone.root_values, together.root_values, "root values depend on company");
}

/// With an all-zero network every leaf value is zero, so no regret ever
/// becomes positive and regret matching leaves the strategy uniform at every
/// iterate. The reference strategy must then be exactly `1/k` on each
/// config's `k` legal actions and zero elsewhere (plan B5.3).
///
/// That is a property of the answer, not a comparison with another
/// implementation: it needs no oracle and admits no tolerance. The reaches do
/// still flow through the sweeps, and they are ordinary floats — so the
/// strategy is checked exactly and the CPU comparison is left to the tests
/// above, which is the honest division.
#[test]
fn zero_network_uniformity() {
    let (dims, w, b, ln) = test_weights();
    let zw = vec![0.0f32; w.len()];
    let nets = nets_from(&dims, &zw, &b, &ln);
    let (_, job) = test_solves(&nets, TEST_CFG, false).into_iter().next().unwrap();
    let carried = job.carried.clone();
    let t = job.tables.clone();
    let gpu = spawn(0, dims, zw, b, ln).expect("spawn");
    let strategy = gpu.solve(job, &carried).expect("trip 1").strategy;

    let legal = |cell: usize| (t.legal_bits[cell >> 3] >> (cell & 7)) & 1 == 1;
    let mut checked = 0;
    for i in 0..t.nodes {
        if t.node_kind[i] != 0 {
            continue;
        }
        let me = t.node_player[i] as usize;
        let nc = (t.cfg_off[2 * i + me + 1] - t.cfg_off[2 * i + me]) as usize;
        let na = t.obs_start[t.obs_off[i + 1] as usize - 1] as usize;
        let so = t.soff[i] as usize;
        for c in 0..nc {
            let cells: Vec<usize> = (0..na).map(|a| so + c * na + a).collect();
            let k = cells.iter().filter(|&&x| legal(x)).count();
            let want = 1.0f32 / k.max(1) as f32;
            for &cell in &cells {
                let got = strategy[cell];
                let exp = if legal(cell) { want } else { 0.0 };
                // The average strategy reaches uniformity by dividing an
                // accumulated sum, not by writing 1/k, so it lands within an
                // ulp rather than on the nose. Illegal cells are exactly zero.
                assert!(
                    (got - exp).abs() <= 1e-6 * exp.max(1.0),
                    "node {i} config {c} cell {cell}: {got} is not the uniform {exp}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no decision config was checked");
}
