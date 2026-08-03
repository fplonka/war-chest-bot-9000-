//! Throughput probe (run with `-- --ignored --nocapture`): ReBeL generation
//! games/s at depth 2 with the trajectory walk.
use std::time::Instant;
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{play_game, Agent, Collect, Data, GameCfg};

#[test]
#[ignore]
fn throughput_probe() {
    for iters in [16usize, 8] {
        let nets = [Nets::default()];
        let cfg = Cfg { depth: 2, iters };
        let n = 8;
        let t0 = Instant::now();
        let mut games = 0u64;
        let mut decisions = 0u64;
        for seed in 0..n {
            let mut rng = Rng::new(seed * 31 + 5);
            let mut d = Data::default();
            let gc = GameCfg {
                agents: [
                    Agent::Rebel { cfg, slot: 0 },
                    Agent::Rebel { cfg, slot: 0 },
                ],
                collect: Collect::Rebel,
                explore: 0.0,
                eval: false,
                random_draft: false,
                eval_mix: 0.0,
            };
            play_game(&mut rng, &nets, &gc, &mut d);
            games += 1;
            decisions += d.decisions as u64;
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "depth2 iters={} walk: {:.2} games/s, {:.0} decisions/s ({} games, {:.2}s)",
            iters,
            games as f64 / secs,
            decisions as f64 / secs,
            games,
            secs
        );
    }
    // Eval mode (avg strategy, no targets):
    {
        let nets = [Nets::default()];
        let cfg = Cfg { depth: 2, iters: 16 };
        let t0 = Instant::now();
        let mut games = 0u64;
        for seed in 0..16u64 {
            let mut rng = Rng::new(seed * 31 + 5);
            let mut d = Data::default();
            let gc = GameCfg {
                agents: [
                    Agent::Rebel { cfg, slot: 0 },
                    Agent::Rebel { cfg, slot: 0 },
                ],
                collect: Collect::None,
                explore: 0.0,
                eval: true,
                random_draft: false,
                eval_mix: 0.0,
            };
            play_game(&mut rng, &nets, &gc, &mut d);
            games += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("depth2 iters=16 walk eval: {:.2} games/s", games as f64 / secs);
    }
}
