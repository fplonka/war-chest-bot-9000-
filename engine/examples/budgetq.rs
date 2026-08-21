//! Whether a cheaper search budget is a worse one.
//!
//! `budget` prices a budget; this asks what it buys. Two questions, because
//! they have different answers and only the first is cheap:
//!
//! **Is the tree we build one we can solve?** `nash_conv` is the
//! exploitability of the finite search game — how far the average strategy is
//! from the equilibrium *of the tree the search itself built*. It is comparable
//! only within one tree, which is exactly what is wanted here: take the
//! production budget, stop growing, and keep iterating. If exploitability keeps
//! falling long past sixty-four iterations then CFR error dominates, the tree
//! is larger than the iterations can solve, and trading tree for iterations is
//! free.
//!
//! **Does a smaller tree move the value target?** The target a solve exists to
//! produce is the root's counterfactual values. Each budget is compared against
//! one reference solve — the largest tree here, iterated far past convergence —
//! on the same root under the same weights.
//!
//! **What does a bigger round cost?** `Cfg::batch` regret updates ride in one
//! round against a frozen tree, which is what turns sixty-five host rounds a
//! solve into nine. The sweep asks what the target pays for that, against the
//! spread between two seeds at `batch = 1`.
//!
//! `cargo run --release --example budgetq -- <weights.bin> [roots] [games]`

use rayon::prelude::*;
use warchest::rng::Rng;
use warchest::search::{Cfg, Cfr, Nets, Policy, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};

/// The root's counterfactual values for both players, belief-weighted into one
/// number apiece — what a training row carries.
fn root_target(sv: &mut Solver) -> [f32; 2] {
    let v = sv.root_values();
    let mut out = [0.0f32; 2];
    for p in 0..2 {
        out[p] = (0..sv.root_belief[p].len())
            .map(|c| sv.root_belief[p].p[c] * v[p][c])
            .sum();
    }
    out
}

fn cfg_of(s: u32, c: f32) -> Cfg {
    cfg_temp(s, c, 1.0)
}

/// The same, with a chosen round size. `batch` regret updates ride in one
/// round: the tree is frozen for all of them and grows once at the end, from
/// every trajectory the round sampled.
fn cfg_batch(s: u32, c: f32, batch: usize) -> Cfg {
    Cfg { batch, ..cfg_of(s, c) }
}

/// The same, with the policy prior flattened. `prior_temp` divides the policy
/// head's logits before the softmax, so a large one is a uniform prior -- which
/// is the whole policy path switched off without touching it.
fn cfg_temp(s: u32, c: f32, prior_temp: f32) -> Cfg {
    Cfg { s, c, prior_temp, cfr: Cfr::SOG, ..Default::default() }
}

/// Total variation between two root policies, averaged over the acting
/// player's configs. Two solves of one root have the same action list and the
/// same belief, so the layouts line up; an unexpanded root gives neither.
fn policy_gap(a: &Policy, b: &Policy) -> Option<f64> {
    if a.p.len() != b.p.len() || a.off.len() < 2 {
        return None;
    }
    let tv: f64 = a
        .off
        .windows(2)
        .map(|w| {
            let (i, j) = (w[0] as usize, w[1] as usize);
            0.5 * (i..j).map(|k| (a.p[k] - b.p[k]).abs() as f64).sum::<f64>()
        })
        .sum();
    Some(tv / (a.off.len() - 1) as f64)
}

/// One solve run to its end, and the two halves of what its target carries.
fn solve(
    st: &warchest::state::State,
    bel: &[warchest::pbs::Belief; 2],
    nets: &std::sync::Arc<Nets>,
    cfg: Cfg,
    seed: u64,
    extra: usize,
) -> ([f32; 2], Policy, usize) {
    let ctx = warchest::pbs::Ctx::new(st);
    let mut sv = Solver::new(st, ctx, std::sync::Arc::clone(nets), cfg, bel.clone(), Rng::new(seed));
    sv.run_alone();
    if extra > 0 {
        sv.multistep(extra);
        sv.finish();
    }
    (root_target(&mut sv), sv.root_policy(), sv.shape().nodes)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let weights = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let roots: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(32);
    let games: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(16);

    let nets = std::sync::Arc::new(Nets { value: warchest::net::Net::load_bin(&weights).expect("weights file"), device: false });
    let small = Cfg { s: 32, c: 4.0, cfr: Cfr::SOG, ..Default::default() };
    let gc = GameCfg {
        agents: [Agent::Sog { cfg: small }; 2],
        collect: Collect::Sog,
        explore: 0.1,
        random_draft: true,
        p_td1: 0.0,
        query_rate: 0.9,
        recursive_rate: 0.1,
    };
    // The same corpus the farm bench solves, when one is given: a shape
    // measured on one set of roots and a rate measured on another cannot be
    // divided into each other.
    //
    // Otherwise roots off fresh play. `collect_roots` concatenates whole games
    // and then truncates, so asking for a small cap takes every root from the
    // first game or two -- and a solve's cost varies twenty-six fold with how
    // far into a game its root sits. Take the whole harvest and stride it.
    let all = match std::env::var("WARCHEST_ROOTS") {
        Ok(path) => {
            let f = std::fs::File::open(&path).expect("roots corpus");
            warchest::roots::read_roots(&mut std::io::BufReader::new(f)).expect("roots corpus")
        }
        Err(_) => collect_roots(games, 99, &nets, &gc, usize::MAX),
    };
    let step = (all.len() / roots.max(1)).max(1);
    let positions: Vec<_> = all.into_iter().step_by(step).take(roots).collect();
    println!("{} roots from {games} games", positions.len());

    // ---- is the production tree one that sixty-four iterations solves?
    //
    // Grown at the production budget, then iterated with growth off. The tree
    // is identical across the row, so the only thing moving is CFR.
    const EXTRA: [usize; 4] = [0, 64, 192, 448];
    let conv: Vec<(f64, f64, usize)> = positions
        .par_iter()
        .enumerate()
        .map(|(i, (st, bel))| {
            let ctx = warchest::pbs::Ctx::new(st);
            let mut sv = Solver::new(st, ctx, std::sync::Arc::clone(&nets), cfg_of(512, 8.0), bel.clone(), Rng::new(0x51D5 ^ i as u64));
            sv.run_alone();
            let mut out = Vec::new();
            let mut done = 0usize;
            for &e in &EXTRA {
                if e > done {
                    sv.multistep(e - done);
                    sv.finish();
                    done = e;
                }
                let c = sv.nash_conv();
                out.push((c.nash as f64, c.zero_sum as f64, 64 + done));
            }
            out
        })
        .filter(|v| !v.is_empty())
        .reduce(Vec::new, |a, b| {
            if a.is_empty() {
                return b;
            }
            a.iter().zip(&b).map(|(x, y)| (x.0 + y.0, x.1 + y.1, x.2)).collect()
        });
    let n = positions.len() as f64;
    println!("\nexploitability of the tree SoG(512,8) builds, growth off:");
    println!("{:>10} {:>12} {:>12}", "iters", "nash_conv", "v0+v1");
    for (nash, zs, it) in &conv {
        println!("{:>10} {:>12.5} {:>12.5}", it, nash / n, zs / n);
    }

    // ---- does a cheaper budget move the value target?
    let budgets: Vec<(u32, f32)> = vec![
        (512, 8.0), (512, 4.0), (256, 4.0), (256, 2.0), (256, 1.0),
        (128, 2.0), (128, 1.0), (64, 1.0), (32, 1.0),
    ];
    // The reference: the same tree the production budget builds, solved far
    // past where its own exploitability stops moving. It is not ground truth --
    // the tree is finite and its leaves are network values -- but it is the
    // best answer available *on the largest tree here*, which is what a
    // cheaper budget has to be judged against.
    //
    // One iteration a round, which is the schedule this started from, so the
    // yardstick does not move when `batch` does.
    let refs: Vec<([f32; 2], Policy)> = positions
        .par_iter()
        .enumerate()
        .map(|(i, (st, bel))| {
            let (t, p, _) = solve(st, bel, &nets, cfg_batch(512, 2.0, 1), 0x51D5 ^ i as u64, 768);
            (t, p)
        })
        .collect();

    println!("\nroot value target against a converged SoG(512,2) reference:");
    println!("{:>12} {:>10} {:>12} {:>12}", "SoG(s,c)", "iters", "|dv| mean", "|v0+v1|");
    for &(s, c) in &budgets {
        let (err, zs, k) = positions
            .par_iter()
            .enumerate()
            .map(|(i, (st, bel))| {
                let r = refs[i].0;
                let ctx = warchest::pbs::Ctx::new(st);
                let mut sv = Solver::new(st, ctx, std::sync::Arc::clone(&nets), cfg_of(s, c), bel.clone(), Rng::new(0x51D5 ^ i as u64));
                sv.run_alone();
                let t = root_target(&mut sv);
                (
                    ((t[0] - r[0]).abs() + (t[1] - r[1]).abs()) as f64 / 2.0,
                    (t[0] + t[1]).abs() as f64,
                    1.0,
                )
            })
            .reduce(|| (0.0f64, 0.0f64, 0.0f64), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let k = k.max(1.0);
        println!(
            "{:>12} {:>10} {:>12.5} {:>12.5}",
            format!("({s},{c})"),
            cfg_of(s, c).iters(),
            err / k,
            zs / k
        );
    }

    // ---- what a round of several iterations costs.
    //
    // `batch` regret updates ride in one round against a frozen tree, so the
    // trajectories that choose where it grows are up to `batch - 1` updates
    // ahead of it and more of them collide on one leaf. What that buys is one
    // host round where there were `batch`. `batch = 1` twice on two seeds is
    // the noise floor the rest of the column is read against.
    println!("\nround size at SoG(512,8), against the same reference:");
    println!("{:>8} {:>10} {:>12} {:>12}", "batch", "nodes", "|dv| mean", "policy tv");
    for (label, b, salt) in [("1", 1, 0u64), ("1'", 1, 0xA5A5), ("2", 2, 0), ("4", 4, 0), ("8", 8, 0), ("16", 16, 0)] {
        let (err, tv, nodes, tvk, k) = positions
            .par_iter()
            .enumerate()
            .map(|(i, (st, bel))| {
                let (r, rp) = &refs[i];
                let cfg = cfg_batch(512, 8.0, b);
                let (t, p, n) = solve(st, bel, &nets, cfg, 0x51D5 ^ salt ^ i as u64, 0);
                let gap = policy_gap(&p, rp);
                (
                    ((t[0] - r[0]).abs() + (t[1] - r[1]).abs()) as f64 / 2.0,
                    gap.unwrap_or(0.0),
                    n as f64,
                    gap.is_some() as u8 as f64,
                    1.0,
                )
            })
            .reduce(
                || (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
            );
        let (k, tvk) = (k.max(1.0), tvk.max(1.0));
        println!("{:>8} {:>10.0} {:>12.5} {:>12.5}", label, nodes / k, err / k, tv / tvk);
    }

    // ---- is the policy prior worth what it costs?
    //
    // `refresh_priors` is two thirds of the host's CPU and it is all network:
    // the action encoder and a dot product per legal cell, run on a core, for
    // every grown node of every iteration. What it buys is the `p` half of
    // PUCT in the expansion phase. Flattening it costs nothing and switches
    // the whole path off, so the same target error says whether the head is
    // steering the tree anywhere the search would not have gone.
    println!("\nwhat the policy prior buys, at SoG(512,8):");
    println!("{:>12} {:>12} {:>12}", "prior_temp", "|dv| mean", "nash_conv");
    for temp in [1.0f32, 1e6] {
        let (err, nash, k) = positions
            .par_iter()
            .enumerate()
            .map(|(i, (st, bel))| {
                let r = refs[i].0;
                let ctx = warchest::pbs::Ctx::new(st);
                let mut sv = Solver::new(st, ctx, std::sync::Arc::clone(&nets), cfg_temp(512, 8.0, temp), bel.clone(), Rng::new(0x51D5 ^ i as u64));
                sv.run_alone();
                let t = root_target(&mut sv);
                let c = sv.nash_conv();
                (
                    ((t[0] - r[0]).abs() + (t[1] - r[1]).abs()) as f64 / 2.0,
                    c.nash as f64,
                    1.0,
                )
            })
            .reduce(|| (0.0f64, 0.0f64, 0.0f64), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let k = k.max(1.0);
        println!("{:>12} {:>12.5} {:>12.5}", temp, err / k, nash / k);
    }
}
