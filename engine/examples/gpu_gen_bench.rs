//! The end-to-end generation benchmark: real games, real tree builds, the
//! real worker loop, against the GPU service — the number that decides when
//! performance work stops (the handoff's stop condition).
//!
//! Weights are all-zero, so every leaf value is zero, CFR stays uniform, and
//! the CPU and GPU builds play the *same* seeded games (zeros survive any
//! summation order — the same trick docs/PERF.md used). The matmuls still
//! run at full size; zeros multiply as slowly as real numbers.
//!
//! Prints, for the GPU path and a same-seeds CPU reference: games/s,
//! solves/s, training rows/s, and the workers' summed GPU-wait share. The
//! tallies (decisions, wins, draws) of the two paths are printed side by
//! side: with zero weights they must match, and a mismatch is a logic bug in
//! one of the paths, not float noise.
//!
//! Usage: `gpu_gen_bench [games] [iters] [depth]` (defaults 64, 64, 2).

use std::time::Instant;

use warchest::net::V3Layout;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{run_games, run_games_gpu, Agent, Collect, GameCfg};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (games, iters, depth) = (arg(1, 64), arg(2, 64), arg(3, 2));

    // The classic shape, all-zero weights.
    let dims: Vec<usize> = vec![3, 32, 64, 64, 384, 1, 1, 64, 1, 384, 0, 0];
    let l = V3Layout::new(&dims).expect("dims");
    let (w, b, mut ln) = (vec![0.0f32; l.w_len], vec![0.0f32; l.b_len], vec![0.0f32; l.ln_len]);
    for &(g, _) in l.pub_ln.iter().chain([&l.ln1]) {
        for x in ln[g..g + 1].iter_mut() {
            *x = 1.0; // gains irrelevant at zero weights; keep LN well-formed
        }
    }
    let nets = [Nets {
        value: warchest::net::Mlp::from_flat(&dims, &w, &b, &ln).expect("weights"),
    }];
    let cfg = Cfg { depth, iters, snapshots: true, node_cap: 200_000, ..Default::default() };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg, slot: 0 }; 2],
        collect: Collect::Rebel,
        explore: 0.25,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };

    let gpu = warchest::gpu::service::spawn(0, dims, w, b, ln).expect("gpu service");
    let t0 = Instant::now();
    let d = run_games_gpu(games, 0xBE9C, &nets, &gc, &[gpu], &|_| 0);
    let el = t0.elapsed().as_secs_f64();
    let workers = std::thread::available_parallelism().map(|n| n.get() - 2).unwrap_or(8);
    println!("== gpu path ==");
    println!("games         {games} in {el:.1}s = {:.2} games/s", games as f64 / el);
    println!("solves/s      {:.0}", d.decisions as f64 / el);
    println!("train rows/s  {:.0}", d.nv as f64 / el);
    println!("worker wait   {:.0}% of {} workers' time",
             100.0 * d.gpu_wait_s as f64 / (el * workers as f64), workers);
    println!("tallies       decisions {} wins {:?} draws {} caps {}",
             d.decisions, d.wins, d.draws, d.cap_hits);

    let t0 = Instant::now();
    let c = run_games(games, 0xBE9C, &nets, &gc);
    let el_cpu = t0.elapsed().as_secs_f64();
    println!("== cpu path (same seeds) ==");
    println!("games         {games} in {el_cpu:.1}s = {:.2} games/s", games as f64 / el_cpu);
    println!("solves/s      {:.0}", c.decisions as f64 / el_cpu);
    println!("tallies       decisions {} wins {:?} draws {} caps {}",
             c.decisions, c.wins, c.draws, c.cap_hits);
    println!("== ratio ==   {:.1}x", el_cpu / el);
    if (d.decisions, d.wins, d.draws) != (c.decisions, c.wins, c.draws) {
        println!("!! TALLY MISMATCH: with zero weights the two paths must play");
        println!("!! identical games; one of them has a logic bug.");
    }
}
