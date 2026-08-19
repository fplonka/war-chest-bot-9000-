//! Throughput harness for the GT-CFR generation loop.
//!
//! Runs the same workload `gen_data(mode="rebel")` runs, with weights exported
//! from a real checkpoint so branching and game length match training, and
//! reports games/sec and decisions/sec. Iterating on this instead of on a
//! 10-minute training run is the difference between a 20-second and a
//! 20-minute edit-measure cycle.
//!
//! Usage: `rebelbench <weights.bin> [games] [nodes] [expand] [iters] [threads]`

use std::time::Instant;
use warchest::net::Net;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{run_games, Agent, Collect, GameCfg};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let games: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(48);
    let nodes: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(1024);
    let expand: usize = a.get(4).and_then(|x| x.parse().ok()).unwrap_or(1);
    let iters: usize = a.get(5).and_then(|x| x.parse().ok()).unwrap_or(64);
    let threads: usize = a.get(6).and_then(|x| x.parse().ok()).unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let mut nets = Nets::default();
    nets.value = Net::load_bin(&path).expect("weights file");
    warchest::state::set_cap_marker_value(0.0);

    let cfg = Cfg {
        nodes,
        expand,
        iters,
        ..Default::default()
    };
    let agent = Agent::Rebel { cfg };
    let gc = GameCfg {
        agents: [agent, agent],
        collect: Collect::Rebel,
        explore: 0.25,
        random_draft: false,
        eval_mix: 0.5,
        mc_mix: 0.0,
        query_rate: 0.0,
        recursive_rate: 0.0,
    };

    // Warm the allocator / caches, then measure.
    let _ = run_games(4, 1, &nets, &gc);
    let t0 = Instant::now();
    let d = run_games(games, 12345, &nets, &gc);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "games {} in {:.2}s -> {:.3} games/s | decisions {} -> {:.0} dec/s | queries {} -> {:.0} query/s | cfgs/dec {:.1} | horizon {:.2} | node-caps {}",
        d.games,
        secs,
        d.games as f64 / secs,
        d.decisions,
        d.decisions as f64 / secs,
        d.nv,
        d.nv as f64 / secs,
        d.configs as f64 / d.decisions.max(1) as f64,
        d.cap_hits as f64 / d.games.max(1) as f64,
        d.node_caps,
    );
    warchest::prof::dump();
    warchest::prof::dump_work();
}
