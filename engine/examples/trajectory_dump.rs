//! Deterministic all-zero-network trajectory oracle.
//!
//! Scheduling and storage rewrites must play byte-identical games when the
//! network is all zero. This prints one stable hash over every public row,
//! target, policy label, arena offset, result, and counter a game produced.
//! Run the same seeds on both sides of a structural change and diff stdout.

use warchest::rng::Rng;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{play_game, Agent, Collect, Data, GameCfg};

/// Frozen from the dense CPU solver at `0aaa466`. These cover 255 solves,
/// 1,259 training rows, random drafts, two horizon games, and one game with
/// real solver-node-cap fallbacks. Instrumentation-only counters are printed
/// but deliberately excluded from the hash.
const DENSE_CPU: [u64; 2] = [0x011e76703114eb0b, 0x3ec52ff1e4932892];

fn bytes(h: &mut u64, xs: &[u8]) {
    for &x in xs {
        *h ^= x as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn u64s(h: &mut u64, xs: impl IntoIterator<Item = u64>) {
    for x in xs {
        bytes(h, &x.to_le_bytes());
    }
}

fn hash_data(z: f32, d: &Data) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    bytes(&mut h, &z.to_bits().to_le_bytes());
    for v in [&d.rows, &d.cc, &d.pact] {
        u64s(&mut h, [v.len() as u64]);
        bytes(&mut h, v);
    }
    for v in [&d.cw, &d.cy, &d.pa, &d.pp] {
        u64s(&mut h, [v.len() as u64]);
        u64s(&mut h, v.iter().map(|x| x.to_bits() as u64));
    }
    for v in [&d.prow, &d.paoff, &d.coff, &d.soff] {
        u64s(&mut h, [v.len() as u64]);
        u64s(&mut h, v.iter().map(|&x| x as u64));
    }
    u64s(
        &mut h,
        [
            d.nv,
            d.games,
            d.decisions,
            d.wins[0],
            d.wins[1],
            d.draws,
            d.cap_hits,
            0, // counter added after the frozen dense baseline
            d.dropped,
            d.configs,
        ]
        .map(|x| x as u64),
    );
    h
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let games: u64 = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(4);
    let iters: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(8);
    let nets = [Nets::default()];
    let cfg = Cfg {
        depth: 2,
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
        eval_mix: 0.5,
        mc_mix: 0.0,
    };
    warchest::state::set_cap_marker_value(0.04);
    for seed in 0..games {
        let mut d = Data::default();
        let z = play_game(Rng::new(seed * 1_000_003 + 17), &nets, &gc, &mut d, None);
        let hash = hash_data(z, &d);
        if let Some(&want) = DENSE_CPU.get(seed as usize) {
            assert_eq!(hash, want, "trajectory {seed} diverged from dense CPU");
        }
        println!(
            "seed {seed} {:016x} games {} decisions {} solves {} rows {} configs {} horizon {} nodecaps {} dropped {}",
            hash,
            d.games,
            d.decisions,
            d.soff.len(),
            d.nv,
            d.configs,
            d.cap_hits,
            d.node_caps,
            d.dropped,
        );
    }
}
