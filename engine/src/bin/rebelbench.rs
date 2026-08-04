//! Throughput harness for the ReBeL generation loop — the thing the training
//! run is actually bottlenecked on.
//!
//! Runs the same workload `gen_data(mode="rebel")` runs, with weights exported
//! from a real checkpoint so branching and game length match training, and
//! reports games/sec and decisions/sec. Iterating on this instead of on a
//! 10-minute training run is the difference between a 20-second and a
//! 20-minute edit-measure cycle.
//!
//! Usage: `rebelbench <weights.bin> [games] [depth] [iters] [threads]`

use std::time::Instant;
use warchest::net::Mlp;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{run_games, Agent, Collect, GameCfg};

fn read_u32(b: &[u8], at: &mut usize) -> usize {
    let v = u32::from_le_bytes(b[*at..*at + 4].try_into().unwrap()) as usize;
    *at += 4;
    v
}

fn read_f32s(b: &[u8], at: &mut usize) -> Vec<f32> {
    let n = read_u32(b, at);
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = *at + i * 4;
        v.push(f32::from_le_bytes(b[o..o + 4].try_into().unwrap()));
    }
    *at += n * 4;
    v
}

fn load(path: &str) -> Mlp {
    let raw = std::fs::read(path).expect("weights file");
    let mut at = 0usize;
    let nd = read_u32(&raw, &mut at);
    let dims: Vec<usize> = (0..nd).map(|_| read_u32(&raw, &mut at)).collect();
    let (w, b, ln) = (
        read_f32s(&raw, &mut at),
        read_f32s(&raw, &mut at),
        read_f32s(&raw, &mut at),
    );
    let mut mlp = Mlp {
        dims: dims.clone(),
        w: Vec::new(),
        b: Vec::new(),
        ln_w: Vec::new(),
        ln_b: Vec::new(),
    };
    let (mut wi, mut bi) = (0usize, 0usize);
    for l in 0..dims.len() - 1 {
        let (i, o) = (dims[l], dims[l + 1]);
        mlp.w.push(w[wi..wi + i * o].to_vec());
        mlp.b.push(b[bi..bi + o].to_vec());
        wi += i * o;
        bi += o;
    }
    let mut li = 0usize;
    for l in 0..dims.len() - 2 {
        let o = dims[l + 1];
        mlp.ln_w.push(ln[li..li + o].to_vec());
        li += o;
        mlp.ln_b.push(ln[li..li + o].to_vec());
        li += o;
    }
    mlp
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let games: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(48);
    let depth: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(2);
    let iters: usize = a.get(4).and_then(|x| x.parse().ok()).unwrap_or(8);
    let threads: usize = a.get(5).and_then(|x| x.parse().ok()).unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    let mut nets = vec![Nets::default(), Nets::default()];
    let mlp = load(&path);
    println!("dims {:?}", mlp.dims);
    nets[0].value = mlp.clone();
    nets[1].value = mlp;
    warchest::state::set_cap_marker_value(0.0);

    let cfg = Cfg {
        depth,
        iters,
        average: true,
    };
    let agent = Agent::Rebel { cfg, slot: 0 };
    let gc = GameCfg {
        agents: [agent, agent],
        collect: Collect::Rebel,
        explore: 0.25,
        eval: false,
        random_draft: false,
        eval_mix: 0.5,
    };

    // Warm the allocator / caches, then measure.
    let _ = run_games(4, 1, &nets, &gc);
    let t0 = Instant::now();
    let d = run_games(games, 12345, &nets, &gc);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "games {} in {:.2}s -> {:.3} games/s | decisions {} -> {:.0} dec/s | samples {} -> {:.0} tgt/s | cfgs/dec {:.1} | cap {:.2}",
        d.games,
        secs,
        d.games as f64 / secs,
        d.decisions,
        d.decisions as f64 / secs,
        d.nv,
        d.nv as f64 / secs,
        d.configs as f64 / d.decisions.max(1) as f64,
        d.cap_hits as f64 / d.games.max(1) as f64,
    );
    warchest::prof::dump_shape();
    warchest::prof::dump();
}
