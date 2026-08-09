//! The end-to-end generation benchmark: real games, real tree builds, and the
//! real worker loop against the GPU service. It is a controlled diagnostic;
//! only balanced throughput in `train.py` satisfies the production target.
//!
//! By default weights are all-zero, so every leaf value is zero, CFR stays
//! uniform, and the CPU and GPU builds play the *same* seeded games (zeros
//! survive any summation order — the same trick docs/PERF.md used). The
//! matmuls still run at full size; zeros multiply as slowly as real numbers.
//! Set `GPU_WEIGHTS` to an `export_weights.py` dump to benchmark the real
//! trained policy distribution. That mode is intended to be used with
//! `GPU_ONLY=1`: ordinary floating-point differences can change later moves,
//! so a same-seed CPU game is no longer a useful exact oracle.
//! `GPU_SEED` and `GPU_CAP_VALUE` override the seed and horizon payoff, so a
//! trainer epoch can be replayed instead of silently measuring a different
//! game distribution.
//!
//! Prints, for the GPU path and a same-seeds CPU reference: games/s,
//! solves/s, training rows/s, and the workers' summed GPU-wait share. The
//! tallies (decisions, wins, draws) of the two paths are printed side by
//! side: with zero weights they must match, and a mismatch is a logic bug in
//! one of the paths, not float noise.
//!
//! Usage: `gpu_gen_bench [games] [iters] [depth]` (defaults 64, 64, 2).

use std::time::Instant;

use warchest::net::{Mlp, V3Layout};
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{run_games, run_games_gpu, Agent, Collect, GameCfg};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (games, iters, depth) = (arg(1, 64), arg(2, 64), arg(3, 2));
    let seed = std::env::var("GPU_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xBE9C);
    let cap_value = std::env::var("GPU_CAP_VALUE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(warchest::state::CAP_MARKER_VALUE_DEFAULT);
    warchest::state::set_cap_marker_value(cap_value);

    let weight_path = std::env::var("GPU_WEIGHTS").ok();
    let (dims, w, b, ln) = if let Some(path) = weight_path.as_deref() {
        Mlp::load_flat_bin(path).unwrap_or_else(|e| panic!("load {path}: {e}"))
    } else {
        // The classic shape, all-zero weights.
        let dims: Vec<usize> = vec![3, 32, 64, 64, 384, 1, 1, 64, 1, 384, 0, 0];
        let l = V3Layout::new(&dims).expect("dims");
        let (w, b, mut ln) = (
            vec![0.0f32; l.w_len],
            vec![0.0f32; l.b_len],
            vec![0.0f32; l.ln_len],
        );
        for &(gain, bias) in l.pub_ln.iter().chain([&l.ln1]) {
            for x in &mut ln[gain..bias] {
                *x = 1.0;
            }
        }
        (dims, w, b, ln)
    };
    println!(
        "weights       {} dims {:?}",
        weight_path.as_deref().unwrap_or("all-zero"),
        dims
    );
    println!("game          seed {seed} cap-value {cap_value}");
    let nets = [Nets {
        value: Mlp::from_flat(&dims, &w, &b, &ln).expect("weights"),
    }];
    let cfg = Cfg {
        depth,
        iters,
        snapshots: true,
        node_cap: 200_000,
        ..Default::default()
    };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg, slot: 0 }; 2],
        collect: Collect::Rebel,
        explore: 0.25,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };

    // `WARCHEST_GPU_DEVICES=0,1` benchmarks the pair of cards the way
    // training uses them; the default is the single service on device 0.
    let devices: Vec<usize> = std::env::var("WARCHEST_GPU_DEVICES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![0]);
    println!("devices       {devices:?}");
    let gpus: Vec<_> = devices
        .iter()
        .map(|&d| {
            warchest::gpu::service::spawn(d, dims.clone(), w.clone(), b.clone(), ln.clone())
                .unwrap_or_else(|e| panic!("gpu service on device {d}: {e}"))
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t0 = Instant::now();
    let d = run_games_gpu(games, seed, &nets, &gc, &gpus, &|_| {
        next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    });
    let el = t0.elapsed().as_secs_f64();
    let per = warchest::selfplay::gen_workers_per();
    let workers = std::env::var("WARCHEST_GEN_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2).max(1))
                .unwrap_or(8)
        })
        .min(games.div_ceil(per).max(1));
    let solves = d.soff.len();
    println!("== gpu path ==");
    println!(
        "games         {games} in {el:.1}s = {:.2} games/s",
        games as f64 / el
    );
    println!("solves/s      {:.0}", solves as f64 / el);
    println!("decisions/s   {:.0}", d.decisions as f64 / el);
    println!("train rows/s  {:.0}", d.nv as f64 / el);
    println!(
        "worker wait   {:.0}% of {} workers' time ({} games each)",
        100.0 * d.gpu_wait_s as f64 / (el * workers as f64),
        workers,
        per
    );
    println!(
        "tallies       solves {} decisions {} wins {:?} draws {} horizon {} node-caps {} dropped {}",
        solves, d.decisions, d.wins, d.draws, d.cap_hits, d.node_caps, d.dropped
    );
    warchest::prof::dump_shape();
    warchest::prof::dump_gpu();
    warchest::prof::dump();
    if std::env::var_os("GPU_ONLY").is_some() {
        return;
    }
    if weight_path.is_some() {
        eprintln!("warning: trained-weight CPU/GPU tallies need not match exactly");
    }

    let t0 = Instant::now();
    let c = run_games(games, seed, &nets, &gc);
    let el_cpu = t0.elapsed().as_secs_f64();
    let cpu_solves = c.soff.len();
    println!("== cpu path (same seeds) ==");
    println!(
        "games         {games} in {el_cpu:.1}s = {:.2} games/s",
        games as f64 / el_cpu
    );
    println!("solves/s      {:.0}", cpu_solves as f64 / el_cpu);
    println!("decisions/s   {:.0}", c.decisions as f64 / el_cpu);
    println!(
        "tallies       solves {} decisions {} wins {:?} draws {} horizon {} node-caps {} dropped {}",
        cpu_solves, c.decisions, c.wins, c.draws, c.cap_hits, c.node_caps, c.dropped
    );
    println!("== ratio ==   {:.1}x", el_cpu / el);
    if weight_path.is_none() && (d.decisions, d.wins, d.draws) != (c.decisions, c.wins, c.draws) {
        println!("!! TALLY MISMATCH: with zero weights the two paths must play");
        println!("!! identical games; one of them has a logic bug.");
    }
}
