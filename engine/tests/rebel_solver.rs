//! Ground truth for the subgame solver.
//!
//! The belief tests in `rebel_pbs.rs` check that the PBS is tracked correctly.
//! This one checks that the *solver on top of it* computes the right numbers,
//! by constructing real positions a few plies from the horizon — so the entire
//! remaining game fits inside one subgame and the value network is never
//! consulted — and comparing against a completely separate vanilla CFR that
//! walks world states with explicit information-set keys.
//!
//! In a two-player zero-sum game the value is unique, so two correct solvers
//! must agree on it. Any error in reach propagation, in the `(config, action)`
//! transition map, in the grouping of private actions under one public
//! observation, or in the counterfactual-value convention shows up here as a
//! disagreement.

use std::collections::HashMap;

use warchest::rebel::*;
use warchest::rng::Rng;
use warchest::search::{node_actions, Cfg, Nets, Solver};
use warchest::selfplay::make_game;
use warchest::state::{State, MAX_MAIN_PLAYS, Z_BAG, Z_FACEDOWN, Z_HAND};

/// A real position `plies` coin plays from the horizon, reached by random play
/// so it is a state the engine actually produces.
fn micro_position(seed: u64, warmup: usize, plies: u16) -> Option<State> {
    let mut rng = Rng::new(seed);
    let mut s = make_game(&mut rng, false);
    for _ in 0..warmup {
        if s.is_terminal() {
            return None;
        }
        let acts = s.legal_actions();
        s.apply_inplace(acts[rng.below(acts.len())]);
    }
    // Only a plain coin-play node will do: micro-decisions and chance nodes
    // make the remaining game depend on things this test does not model.
    if s.is_terminal() || s.is_chance() || !matches!(s.pending(), warchest::state::Cont::MainPlay) {
        return None;
    }
    if s.hand_size(0) < 2 || s.hand_size(1) < 2 {
        return None;
    }
    s.main_plays = MAX_MAIN_PLAYS - plies;
    Some(s)
}

fn uniform_belief(s: &State, ctx: &Ctx, p: u8) -> Belief {
    let res = reserve(s, p, ctx);
    let truth = true_config(s, p, ctx);
    let cfgs = enumerate_configs(&res, truth.hand_size(), truth.fd_size());
    let n = cfgs.len() as f32;
    Belief {
        p: vec![1.0 / n; cfgs.len()],
        cfg: cfgs,
    }
}

// --------------------------------------------------------- vanilla CFR oracle

type Key = (Vec<u32>, Config);

#[derive(Default)]
struct Tab {
    regret: HashMap<Key, Vec<f64>>,
    strat: HashMap<Key, Vec<f64>>,
}

fn regret_match(r: &[f64]) -> Vec<f64> {
    let pos: Vec<f64> = r.iter().map(|&x| x.max(0.0)).collect();
    let sum: f64 = pos.iter().sum();
    if sum > 0.0 {
        pos.iter().map(|x| x / sum).collect()
    } else {
        vec![1.0 / r.len() as f64; r.len()]
    }
}

/// Textbook vanilla CFR over world states. Information sets are keyed by the
/// public observation history plus the acting player's own config — nothing in
/// here knows about public belief states.
fn cfr(
    t: &mut Tab,
    s: &State,
    ctx: &Ctx,
    hist: &mut Vec<u32>,
    reach: [f64; 2],
    trav: usize,
) -> f64 {
    if s.is_terminal() {
        return s.utility(trav) as f64;
    }
    let p = s.to_act();
    let cfg = true_config(s, p, ctx);
    let (acts, _, _) = node_actions(s, p, ctx, std::slice::from_ref(&cfg));
    let key = (hist.clone(), cfg);
    let n = acts.len();
    let sigma = regret_match(t.regret.entry(key.clone()).or_insert_with(|| vec![0.0; n]));

    let mut util = vec![0.0; n];
    let mut node_util = 0.0;
    for (i, a) in acts.iter().enumerate() {
        let mut ns = s.clone();
        ns.apply_inplace(*a);
        let mut r = reach;
        r[p as usize] *= sigma[i];
        hist.push(obs_key(a));
        util[i] = cfr(t, &ns, ctx, hist, r, trav);
        hist.pop();
        node_util += sigma[i] * util[i];
    }
    if p as usize == trav {
        let opp = reach[1 - trav];
        let reg = t.regret.get_mut(&key).unwrap();
        for i in 0..n {
            reg[i] += opp * (util[i] - node_util);
        }
        let st = t.strat.entry(key).or_insert_with(|| vec![0.0; n]);
        for i in 0..n {
            st[i] += reach[trav] * sigma[i];
        }
    }
    node_util
}

/// Game value for player 0, averaged over the prior on both private configs.
fn oracle_value(s: &State, ctx: &Ctx, bel: &[Belief; 2], iters: usize) -> f64 {
    let mut t = Tab::default();
    let mut value = 0.0;
    for it in 0..iters {
        let trav = it % 2;
        let mut total = 0.0;
        for (i0, c0) in bel[0].cfg.iter().enumerate() {
            for (i1, c1) in bel[1].cfg.iter().enumerate() {
                let mut w = s.clone();
                set_config(&mut w, 0, ctx, c0);
                set_config(&mut w, 1, ctx, c1);
                let pr = [bel[0].p[i0] as f64, bel[1].p[i1] as f64];
                let mut hist = Vec::new();
                let v = cfr(&mut t, &w, ctx, &mut hist, pr, trav);
                if trav == 0 {
                    total += pr[0] * pr[1] * v;
                }
            }
        }
        // Average the traverser-0 values over the second half of the run, which
        // is where CFR's iterates have settled.
        if trav == 0 && it * 2 >= iters {
            value += total;
        }
    }
    value / (iters as f64 / 4.0)
}

// ------------------------------------------------------------------ the test

#[test]
fn subgame_solver_matches_tabular_cfr_on_micro_endgames() {
    let nets = Nets::default();
    let mut checked = 0;
    for seed in 0..3000u64 {
        let Some(s) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [
            uniform_belief(&s, &ctx, 0),
            uniform_belief(&s, &ctx, 1),
        ];
        // Keep the exhaustive side affordable.
        if bel[0].len() * bel[1].len() > 64 {
            continue;
        }

        let cfg = Cfg {
            depth: 8,
            iters: 500,
            average: true,
        };
        let mut sv = Solver::new(&s, &ctx, &nets, cfg, bel.clone());
        // If any leaf were non-terminal the (empty) network would silently
        // return zero and the comparison would be meaningless.
        assert!(
            sv.nodes.iter().all(|n| !n.leaf || n.s.is_terminal()),
            "the whole remaining game must fit inside the subgame"
        );
        if sv.nodes.len() > 8_000 {
            continue;
        }
        // A position where every line ends in the same score tests nothing.
        let mut outcomes: Vec<i32> = sv
            .nodes
            .iter()
            .filter(|n| n.leaf && n.s.is_terminal())
            .map(|n| (n.s.utility(0) * 1000.0) as i32)
            .collect();
        outcomes.sort_unstable();
        outcomes.dedup();
        if outcomes.len() < 2 {
            continue;
        }
        sv.multistep(cfg.iters);

        let v0: f64 = (0..bel[0].len())
            .map(|c| bel[0].p[c] as f64 * sv.root_values(0)[c] as f64)
            .sum();
        let v1: f64 = (0..bel[1].len())
            .map(|c| bel[1].p[c] as f64 * sv.root_values(1)[c] as f64)
            .sum();
        // A zero-sum game solved consistently: the two players' root values
        // must cancel. This is the single most useful invariant on the
        // counterfactual-value convention.
        assert!(
            (v0 + v1).abs() < 0.02,
            "seed {}: root values are not zero-sum: {:.4} + {:.4}",
            seed,
            v0,
            v1
        );

        let exact = oracle_value(&s, &ctx, &bel, 100);
        assert!(
            (v0 - exact).abs() < 0.03,
            "seed {}: subgame solver says {:.4}, tabular CFR says {:.4} \
             ({} public nodes, {}x{} configs)",
            seed,
            v0,
            exact,
            sv.nodes.len(),
            bel[0].len(),
            bel[1].len()
        );
        checked += 1;
        eprintln!(
            "  seed {:4}: solver {:+.4}  tabular {:+.4}  zero-sum {:+.4}  ({} nodes, {}x{} configs)",
            seed, v0, exact, v0 + v1, sv.nodes.len(), bel[0].len(), bel[1].len()
        );
        if checked >= 4 {
            break;
        }
    }
    assert!(checked >= 4, "only {} positions exercised", checked);
    eprintln!("verified {} micro-endgames against tabular CFR", checked);
}

/// How badly does a short solve misprice a position?
///
/// ReBeL's value target is the running mean of the root values over CFR
/// iterations. Run too few and that mean sits closer to a best-response value
/// than to the equilibrium — a bias in the *same direction for both players*,
/// which is exactly the kind of error that a bootstrapped training loop
/// amplifies instead of averaging away. This measures it against the exact
/// value so the iteration count can be chosen on evidence.
#[test]
fn cfr_iteration_count_bias() {
    let nets = Nets::default();
    let mut rows = 0;
    let budgets = [4usize, 16, 64, 256];
    let mut err = vec![0.0f64; budgets.len()];
    let mut nzs = vec![0.0f64; budgets.len()];
    eprintln!("     exact   {}", budgets.iter().map(|b| format!("{:>9}", b)).collect::<String>());
    for seed in 0..3000u64 {
        let Some(s) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
        if bel[0].len() * bel[1].len() > 64 {
            continue;
        }
        let probe = Solver::new(&s, &ctx, &nets, Cfg { depth: 8, iters: 1, average: true }, bel.clone());
        if !probe.nodes.iter().all(|n| !n.leaf || n.s.is_terminal()) || probe.nodes.len() > 8_000 {
            continue;
        }
        let mut o: Vec<i32> = probe
            .nodes
            .iter()
            .filter(|n| n.leaf && n.s.is_terminal())
            .map(|n| (n.s.utility(0) * 1000.0) as i32)
            .collect();
        o.sort_unstable();
        o.dedup();
        if o.len() < 2 {
            continue;
        }
        let exact = oracle_value(&s, &ctx, &bel, 100);
        let mut line = format!("  {:+.4}   ", exact);
        for (bi, &t) in budgets.iter().enumerate() {
            let mut sv = Solver::new(&s, &ctx, &nets, Cfg { depth: 8, iters: t, average: true }, bel.clone());
            sv.multistep(t);
            let v0: f64 = (0..bel[0].len())
                .map(|c| bel[0].p[c] as f64 * sv.root_values(0)[c] as f64)
                .sum();
            let v1: f64 = (0..bel[1].len())
                .map(|c| bel[1].p[c] as f64 * sv.root_values(1)[c] as f64)
                .sum();
            err[bi] += (v0 - exact).abs();
            nzs[bi] += (v0 + v1).abs();
            line += &format!("{:+9.4}", v0);
        }
        eprintln!("{}", line);
        rows += 1;
        if rows >= 6 {
            break;
        }
    }
    eprintln!("\nmean |value error| by iteration count:");
    for (bi, &t) in budgets.iter().enumerate() {
        eprintln!(
            "  T={:4}: err {:.4}   |v0+v1| {:.4}",
            t,
            err[bi] / rows as f64,
            nzs[bi] / rows as f64
        );
    }
    assert!(rows >= 4);
}

/// A real position a few plies from the horizon, reached by random play, whose
/// remaining game spans a round boundary: small hands, so both players empty
/// them within the remaining main plays and the draws happen inside the
/// subgame.
fn draw_position(seed: u64, warmup: usize, plies: u16) -> Option<State> {
    let mut rng = Rng::new(seed);
    let mut s = make_game(&mut rng, false);
    for _ in 0..warmup {
        if s.is_terminal() {
            return None;
        }
        let acts = s.legal_actions();
        s.apply_inplace(acts[rng.below(acts.len())]);
    }
    if s.is_terminal() || !matches!(s.pending(), warchest::state::Cont::MainPlay) {
        return None;
    }
    // A real board reached by actual play; force the hands to one coin each
    // so both players empty them — and the round boundary with its draws —
    // inside the remaining `plies` main plays. Everything else stays in the
    // bag, so the config space is tiny.
    let ctx = Ctx::new(&s);
    for p in 0..2u8 {
        for u in 0..warchest::units::N_UNITS {
            let c = s.zones[p as usize][Z_HAND][u];
            s.zones[p as usize][Z_BAG][u] += c;
            s.zones[p as usize][Z_HAND][u] = 0;
            let c = s.zones[p as usize][Z_FACEDOWN][u];
            s.zones[p as usize][Z_BAG][u] += c;
            s.zones[p as usize][Z_FACEDOWN][u] = 0;
        }
        let mut cfg = Config::default();
        // A unit whose coins are all deployed (or eliminated) has no reserve
        // left; the forced one-coin hand must fit the reserve, so use the
        // first slot that still has coins in the bag.
        let hand_slot = (0..NSLOT)
            .find(|&k| s.zones[p as usize][Z_BAG][ctx.slots[p as usize][k] as usize] > 0)
            .unwrap_or(NSLOT);
        if hand_slot == NSLOT {
            return None;
        }
        cfg.hand[hand_slot] = 1;
        set_config(&mut s, p, &ctx, &cfg);
    }
    s.main_plays = MAX_MAIN_PLAYS - plies;
    Some(s)
}

/// The draw pass-through, checked structurally on real positions: a chance
/// node must have exactly one public child, the child's config support must be
/// exactly `belief_after_draw`'s support (same list, same order — the
/// invariant the self-play walk asserts at runtime), the idle player's support
/// must pass through untouched, the chance-matrix rows must be proper
/// distributions, and the solved root values must stay zero-sum with draws
/// inside the tree. The chance transition itself is already verified against
/// brute-force enumeration in `rebel_pbs.rs`.
#[test]
fn draw_pass_through_consistency() {
    let nets = Nets::default();
    let mut checked = 0;
    let mut cnt = [0usize; 5]; // 0 rejected, 1 built, 2 toolarge, 3 nochance, 4 solved
    for seed in 0..4000u64 {
        let Some(s) = draw_position(seed, 60 + (seed as usize % 120), 4) else {
            cnt[0] += 1;
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [
            uniform_belief(&s, &ctx, 0),
            uniform_belief(&s, &ctx, 1),
        ];
        if bel[0].len() * bel[1].len() > 1000 {
            cnt[0] += 1;
            continue;
        }
        // Bound the build: positions whose root decision branches widely
        // produce subgames too big to build, let alone solve.
        let (acts, _, _) = node_actions(&s, s.to_act(), &ctx, &bel[s.to_act() as usize].cfg);
        if acts.len() > 14 {
            cnt[0] += 1;
            continue;
        }
        let mut sv = Solver::new(
            &s,
            &ctx,
            &nets,
            Cfg {
                depth: 5,
                iters: 80,
                average: true,
            },
            bel.clone(),
        );
        if sv.nodes.len() > 20_000 {
            cnt[2] += 1;
            continue;
        }
        if !sv.nodes.iter().any(|n| n.chance) {
            cnt[3] += 1;
            continue;
        }
        cnt[1] += 1;
        for i in 0..sv.nodes.len() {
            if !sv.nodes[i].chance {
                continue;
            }
            let n = &sv.nodes[i];
            assert_eq!(n.child.len(), 1, "a draw must have exactly one public child");
            let me = n.player as usize;
            let ch = n.child[0];
            assert_eq!(
                sv.nodes[ch].cfgs[1 - me],
                n.cfgs[1 - me],
                "idle player's support must pass through the draw untouched"
            );
            // One node stands for a whole run of this player's draws, so the
            // oracle applies `belief_after_draw` once per step, walking the
            // state alongside it.
            assert!(n.draw_steps >= 1, "a draw node covers at least one draw");
            let mut ws = n.s.clone();
            let mut b = Belief {
                cfg: n.cfgs[me].to_vec(),
                p: vec![1.0; n.cfgs[me].len()],
            };
            for _ in 0..n.draw_steps {
                let res = reserve(&ws, n.player, &ctx);
                let fu = faceup_counts(&ws, n.player, &ctx);
                b = belief_after_draw(&b, &res, &fu);
                let acts = ws.legal_actions();
                ws.apply_inplace(acts[0]);
            }
            assert_eq!(
                sv.nodes[ch].cfgs[me].to_vec(),
                b.cfg,
                "post-draw support must equal belief_after_draw's, in order"
            );
            for ci in 0..n.draw.rows() {
                let sum: f32 = n.draw.row(ci).1.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "draw row {} sums to {}",
                    ci,
                    sum
                );
            }
            assert_eq!(
                sv.nodes[ch].s.hand_size(n.player),
                n.s.hand_size(n.player) + n.draw_steps,
                "each covered draw adds exactly one coin to the hand"
            );
        }
        sv.multistep(80);
        let v0: f64 = (0..bel[0].len())
            .map(|c| bel[0].p[c] as f64 * sv.root_values(0)[c] as f64)
            .sum();
        let v1: f64 = (0..bel[1].len())
            .map(|c| bel[1].p[c] as f64 * sv.root_values(1)[c] as f64)
            .sum();
        assert!(
            (v0 + v1).abs() < 0.05,
            "seed {}: root values not zero-sum with draws in the tree: {:.4} + {:.4}",
            seed,
            v0,
            v1
        );
        checked += 1;
        cnt[4] += 1;
        eprintln!(
            "  seed {:4}: zero-sum {:+.4} ({} nodes)",
            seed,
            v0 + v1,
            sv.nodes.len()
        );
        if checked >= 6 {
            break;
        }
    }
    eprintln!(
        "draw pass-through: rejected={} built={} toolarge={} nochance={} solved={}",
        cnt[0], cnt[1], cnt[2], cnt[3], cnt[4]
    );
    assert!(checked >= 4, "only {} draw positions exercised", checked);
}
