//! GT-CFR generation throughput probe (run with `-- --ignored --nocapture`).
use std::time::Instant;
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{play_game, Agent, Collect, Data, GameCfg};

#[test]
#[ignore]
fn throughput_probe() {
    for iters in [16usize, 8] {
        let nets = Nets::default();
        let cfg = Cfg {
            s: iters as u32,
            c: 1.0,
            ..Default::default()
        };
        let n = 8;
        let t0 = Instant::now();
        let mut games = 0u64;
        let mut decisions = 0u64;
        for seed in 0..n {
            let rng = Rng::new(seed * 31 + 5);
            let mut d = Data::default();
            let gc = GameCfg {
                agents: [Agent::Sog { cfg }, Agent::Sog { cfg }],
                collect: Collect::Sog,
                explore: 0.0,
                random_draft: false,
                eval_mix: 0.0,
                mc_mix: 0.0,
                query_rate: 0.0,
                recursive_rate: 0.0,
            };
            play_game(rng, &nets, &gc, &mut d, None);
            games += 1;
            decisions += d.decisions as u64;
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "GT-CFR iters={}: {:.2} games/s, {:.0} decisions/s ({} games, {:.2}s)",
            iters,
            games as f64 / secs,
            decisions as f64 / secs,
            games,
            secs
        );
    }
    // Eval mode (avg strategy, no targets):
    {
        let nets = Nets::default();
        let cfg = Cfg {
            s: 16,
            c: 1.0,
            ..Default::default()
        };
        let t0 = Instant::now();
        let mut games = 0u64;
        for seed in 0..16u64 {
            let rng = Rng::new(seed * 31 + 5);
            let mut d = Data::default();
            let gc = GameCfg {
                agents: [Agent::Sog { cfg }, Agent::Sog { cfg }],
                collect: Collect::None,
                explore: 0.0,
                random_draft: false,
                eval_mix: 0.0,
                mc_mix: 0.0,
                query_rate: 0.0,
                recursive_rate: 0.0,
            };
            play_game(rng, &nets, &gc, &mut d, None);
            games += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("GT-CFR iters=16: {:.2} games/s", games as f64 / secs);
    }
}
