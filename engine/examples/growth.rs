//! Where a GT-CFR solve's nodes actually go, iteration by iteration.
//!
//! One expansion adds a coin play *and* every draw, tactic and forced play
//! beneath it, so the node budget is spent far faster than `expand` suggests.
//! That schedule decides how a solve can be executed: the iterations after
//! growth stops run on a tree nobody touches, and those are the ones worth
//! handing to an accelerator whole.
//!
//! `cargo run --release --example growth -- [nodes] [expand] [iters] [roots]`

use warchest::rng::Rng;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};

fn random_net(seed: u64) -> warchest::net::Net {
    let mut r = Rng::new(seed);
    let l = warchest::net::NetLayout::new();
    let mut draw = |n: usize| -> Vec<f32> {
        (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect()
    };
    let (w, b) = (draw(l.w_len), draw(l.b_len));
    let mut ln = vec![0.0; l.ln_len];
    for n in &l.norms {
        ln[n.g..n.g + n.width].fill(1.0);
    }
    warchest::net::Net::from_flat(&w, &b, &ln).expect("random net")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    let cfg = Cfg {
        nodes: arg(1, 8192),
        expand: arg(2, 8),
        iters: arg(3, 64),
        ..Default::default()
    };
    let roots = arg(4, 12);

    // Small random weights, not a trained network: which leaf a trajectory
    // picks is then arbitrary, but what an expansion *costs* in nodes is a
    // property of the rules and the position, and that is what is measured
    // here. An empty net would send every agent down the greedy path instead.
    let nets = Nets {
        value: random_net(0x5EED),
        gate: None,
    };
    let small = Cfg { nodes: 64, expand: 1, iters: 4, ..cfg };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg: small }; 2],
        collect: Collect::Rebel,
        explore: 0.1,
        random_draft: true,
        eval_mix: 1.0,
        mc_mix: 0.0,
        query_rate: 0.9,
        recursive_rate: 0.1,
    };
    let positions = collect_roots(8, 99, &nets, &gc, roots);
    println!(
        "nodes={} expand={} iters={} over {} real roots",
        cfg.nodes,
        cfg.expand,
        cfg.iters,
        positions.len()
    );

    let mut rng = Rng::new(0x67CF);
    let (mut stop_sum, mut stopped, mut node_iters, mut static_iters) = (0usize, 0usize, 0u64, 0u64);
    let mut conc = 0.0f64;
    let mut contract_ns = 0u64;
    for (n, (s, belief)) in positions.iter().enumerate() {
        let ctx = warchest::rebel::Ctx::new(s);
        let mut sv = Solver::new(s, ctx, &nets, cfg, belief.clone());
        let mut stop = None;
        let mut contract = warchest::contract::Contract::of(&sv);
        sv.grown.clear();
        for t in 0..cfg.iters {
            sv.step();
            for _ in 0..cfg.expand {
                if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                    break;
                }
            }
            node_iters += sv.nodes.len() as u64;
            // What describing the tree for a device costs, against the sweeps
            // that description exists to feed.
            let tc = std::time::Instant::now();
            let grown = std::mem::take(&mut sv.grown);
            contract.extend(&sv, &grown);
            contract_ns += tc.elapsed().as_nanos() as u64;
            std::hint::black_box(contract.nodes());
            if stop.is_none() && sv.nodes.len() >= cfg.nodes {
                stop = Some(t + 1);
            }
            if stop.is_some_and(|at| t + 1 > at) {
                static_iters += sv.nodes.len() as u64;
            }
        }
        // The same two sweeps, walked over the tree and gathered over the flat
        // description, on the tree as it finally stands. Which is faster
        // decides whether the flat form should simply replace the walk on the
        // host, quite apart from any device.
        {
            let reps = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                sv.precompute_reaches();
            }
            let walk_reach = t0.elapsed().as_secs_f64() / reps as f64;
            let root = [&sv.root_belief[0].p[..], &sv.root_belief[1].p[..]];
            let mut out = vec![0.0f32; sv.reach.len()];
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                contract.reach(root, &sv.cur, &mut out);
            }
            let flat_reach = t1.elapsed().as_secs_f64() / reps as f64;
            println!(
                "  reach: walk {:.2} ms, flat {:.2} ms  ({:.2}x)",
                1e3 * walk_reach,
                1e3 * flat_reach,
                walk_reach / flat_reach
            );
        }
        println!(
            "  root {n:2}: {:6} nodes, growth stopped at iteration {:?}, \
             visit concentration {:.3}",
            sv.nodes.len(),
            stop,
            sv.visit_concentration(),
        );
        conc += sv.visit_concentration() as f64;
        if let Some(at) = stop {
            stop_sum += at;
            stopped += 1;
        }
    }
    if stopped > 0 {
        println!(
            "growth stops at iteration {:.1} of {} on {}/{} roots",
            stop_sum as f64 / stopped as f64,
            cfg.iters,
            stopped,
            positions.len()
        );
    }
    println!(
        "{:.0}% of node-iterations run on a tree that never changes again",
        100.0 * static_iters as f64 / node_iters.max(1) as f64
    );
    println!(
        "mean visit concentration {:.3} (uniform over k actions would be 1/k)",
        conc / positions.len().max(1) as f64
    );
    // Where a solve's CPU time goes, which is what the device has to absorb.
    // Only says anything with `--features prof`.
    println!(
        "contract updates: {:.0} cpu-ms total over {} solves",
        contract_ns as f64 / 1e6,
        positions.len()
    );
    warchest::prof::dump();
    warchest::prof::dump_work();
}
