//! Benchmark: applies/sec and full random playouts/sec on one core.
//! Uses the crate's tiny inline RNG (no external deps).

use std::time::Instant;
use warchest::rng::Rng;
use warchest::state::{State, BLACK, WHITE};

// A pool of legal draftable unitTypeIds (the 19 in-scope unit types).
const POOL: [u16; 19] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54,
];

fn random_draft(rng: &mut Rng) -> (Vec<u16>, Vec<u16>, u8) {
    let pick4 = |rng: &mut Rng| -> Vec<u16> {
        let mut chosen: Vec<u16> = Vec::new();
        while chosen.len() < 4 {
            let id = POOL[rng.below(POOL.len())];
            if !chosen.contains(&id) {
                chosen.push(id);
            }
        }
        chosen
    };
    let w = pick4(rng);
    let b = pick4(rng);
    let first = if rng.next_u64() & 1 == 0 {
        WHITE
    } else {
        BLACK
    };
    (w, b, first)
}

/// Play one random game; return (applies, hit_cap).
fn playout(rng: &mut Rng, cap: u32) -> (u32, bool) {
    let (w, b, first) = random_draft(rng);
    let mut s = State::from_draft(&w, &b, first);
    let mut applies = 0u32;
    while !s.is_terminal() && applies < cap {
        let acts = s.legal_actions();
        if acts.is_empty() {
            break;
        }
        let a = acts[rng.below(acts.len())];
        s.apply_inplace(a);
        applies += 1;
    }
    (applies, applies >= cap && !s.is_terminal())
}

fn main() {
    let mut rng = Rng::new(0xC0FFEE);
    let cap = 2000u32;

    let mut total_applies: u64 = 0;
    let mut games: u64 = 0;
    let mut cap_hits: u64 = 0;

    let t0 = Instant::now();
    let budget = std::time::Duration::from_secs(3);
    while t0.elapsed() < budget {
        for _ in 0..64 {
            let (n, hit) = playout(&mut rng, cap);
            total_applies += n as u64;
            games += 1;
            if hit {
                cap_hits += 1;
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();

    let applies_per_sec = total_applies as f64 / secs;
    let games_per_sec = games as f64 / secs;

    println!("=== warchest-engine benchmark (single core) ===");
    println!("wall time:         {:.3} s", secs);
    println!("games played:      {}", games);
    println!("total applies:     {}", total_applies);
    println!(
        "avg game length:   {:.1} applies",
        total_applies as f64 / games as f64
    );
    println!("cap hits:          {} (cap = {})", cap_hits, cap);
    println!();
    println!("applies/sec/core:  {:.0}", applies_per_sec);
    println!("playouts/sec/core: {:.0}", games_per_sec);
    println!();
    if applies_per_sec >= 100_000.0 {
        println!("TARGET MET: >= 100k applies/sec/core");
    } else {
        println!("TARGET MISSED: < 100k applies/sec/core");
    }
}
