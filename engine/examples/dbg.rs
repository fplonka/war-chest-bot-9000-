//! Step-by-step solver dump for the torch-spec debug: writes the solver's
//! internal arenas at init and after each phase, for comparison with
//! train/cfr_spec.py. Usage: dbg <out-dir> <solve-index>.
use std::fs;

use warchest::net::Mlp;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};

fn dump(name: &str, dir: &std::path::Path, v: &[f32]) {
    let mut b: Vec<u8> = Vec::new();
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(dir.join(name), b).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).cloned().unwrap_or_else(|| "/tmp/dbg".into()));
    let which: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    fs::create_dir_all(&dir).unwrap();
    let dir = dir.as_path();
    let net = Mlp::load_bin("/tmp/oracle_test/weights.bin").unwrap();
    let nets = [Nets { value: net }];
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg: Cfg { depth: 2, iters: 4, snapshots: false, ..Default::default() }, slot: 0 }; 2],
        collect: Collect::None, explore: 0.0, random_draft: true,
        eval_mix: 0.0, mc_mix: 0.0,
    };
    let roots = collect_roots(4, 0xC0FFEE, &nets, &gc, 6);
    let (s, bel) = &roots[which];
    let ctx = warchest::rebel::Ctx::new(s);
    let mut sv = Solver::new(s, &ctx, &nets[0],
        Cfg { depth: 2, iters: 8, snapshots: true, ..Default::default() }, bel.clone());
    dump("reach0", dir, &sv.reach);
    dump("cur0", dir, &sv.cur);
    dump("sum0", dir, &sv.sum_strat.iter().flatten().cloned().collect::<Vec<_>>());
    dump("avg0", dir, &sv.avg.iter().flatten().cloned().collect::<Vec<_>>());
    // Leaf values for traverser 0 (phases 1-3). This also runs the build
    // GEMMs (ensure_leaf_batch), filling h0/ce/cz/cg.
    sv.leaf_values(0);
    let dg = 64usize;
    let rows = sv.leaf_rows.len();
    dump("h0", dir, &sv.h0[..rows * 384]);
    dump("ce", dir, &sv.ce);
    dump("cz", dir, &sv.cz[..sv.ncfg * 64]);
    dump("cg", dir, &sv.cg[..sv.ncfg * 65]);
    dump("xb1", dir, &sv.xb[..rows * 2 * dg]);
    dump("ob1", dir, &sv.ob[..rows * 384]);
    dump("vals1", dir, &sv.vals);
    // backprop (phase 4) for traverser 0: inst + vals.
    let cur = std::mem::take(&mut sv.cur);
    sv.backprop(0, &cur, warchest::search::Back::Regret);
    sv.cur = cur;
    dump("inst2", dir, &sv.inst);
    dump("vals2", dir, &sv.vals);
    // RM (phase 5a).
    // (step does RM + propagate + AVG; replicate via step and dump)
    drop(sv);
    let mut sv2 = Solver::new(s, &ctx, &nets[0],
        Cfg { depth: 2, iters: 8, snapshots: true, ..Default::default() }, bel.clone());
    sv2.step(0);
    dump("reach3", dir, &sv2.reach);
    dump("cur3", dir, &sv2.cur);
    dump("sum3", dir, &sv2.sum_strat.iter().flatten().cloned().collect::<Vec<_>>());
    dump("avg3", dir, &sv2.avg.iter().flatten().cloned().collect::<Vec<_>>());
    println!("dumped to {}", dir.display());
}
