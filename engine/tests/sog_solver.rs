//! Ground truth for the subgame solver.
//!
//! The belief tests in `pbs.rs` check that the PBS is tracked correctly.
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
use std::sync::Arc;

use warchest::pbs::*;
use warchest::rng::Rng;
use warchest::net::Net;
use warchest::search::{node_actions, Budget, Cfg, Cfr, Ent, Solver, StopReason};
use warchest::selfplay::make_game;
use warchest::state::{Cont, State, MAX_MAIN_PLAYS, Z_BAG, Z_FACEDOWN, Z_HAND};
use warchest::units::{
    ARCHER, CAVALRY, CROSSBOWMAN, FOOTMAN, LANCER, PIKEMAN, ROYAL_COIN, SWORDSMAN, WARRIOR_PRIEST,
};

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
    let cfgs = enumerate_configs(
        &res,
        truth.hand_size(),
        truth.fd_size(),
        truth.inflight.is_some(),
    );
    let n = cfgs.len() as f32;
    Belief {
        p: vec![1.0 / n; cfgs.len()],
        cfg: cfgs,
    }
}

/// The same belief, thinned to at most `cap` configs and renormalised.
///
/// A narrower support is still a public belief state, and both solvers are
/// asked about the same one, so the comparison stays exact. This is what makes
/// the exhaustive side affordable: it walks the whole remaining game once per
/// config pair on every iteration, and a real mid-game reserve allows hundreds
/// of configs. Taking every `n/cap`-th config spreads the support over the
/// enumeration rather than keeping one corner of it.
fn thinned(mut b: Belief, cap: usize) -> Belief {
    if b.len() > cap {
        let step = b.len().div_ceil(cap);
        b.cfg = b.cfg.iter().step_by(step).cloned().collect();
    }
    b.p = vec![1.0 / b.cfg.len() as f32; b.cfg.len()];
    b
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

/// The largest subgame the exhaustive oracle is asked to walk, and the widest
/// belief it is asked to average over. Together these are what keep the test to
/// a second: the oracle walks the whole remaining game once per config pair on
/// every one of its iterations.
const NODE_CAP: usize = 1_200;
const CFG_CAP: usize = 3;

#[test]
fn subgame_solver_matches_tabular_cfr_on_micro_endgames() {
    let nets = Arc::new(Net::default());
    let cfg = Cfg {
        s: 200,
        c: 1.0,
        budget: Budget::unbounded(),
        ..Default::default()
    };
    let mut checked = 0;
    let mut skipped = [0usize; 3]; // too large, one-sided, no position
    for seed in 0..3000u64 {
        let Some(s) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            skipped[2] += 1;
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [
            thinned(uniform_belief(&s, &ctx, 0), CFG_CAP),
            thinned(uniform_belief(&s, &ctx, 1), CFG_CAP),
        ];

        let mut probe = Solver::new(&s, ctx, Arc::clone(&nets), cfg, bel.clone(), Rng::new(seed));
        if !probe.grow_full(NODE_CAP) {
            skipped[0] += 1;
            continue;
        }
        // If any leaf were non-terminal the (empty) network would silently
        // return zero and the comparison would be meaningless.
        assert!(
            probe
                .nodes
                .iter()
                .zip(&probe.states)
                .all(|(n, st)| !n.leaf || st.is_terminal()),
            "the whole remaining game must fit inside the subgame"
        );
        let mut outcomes: Vec<f64> = probe
            .nodes
            .iter()
            .zip(&probe.states)
            .filter(|(n, st)| n.leaf && st.is_terminal())
            .map(|(_, st)| st.utility(0) as f64)
            .collect();
        outcomes.sort_by(f64::total_cmp);
        let (lo, hi) = (outcomes[0], outcomes[outcomes.len() - 1]);
        let nodes = probe.nodes.len();
        drop(probe);

        let exact = oracle_value(&s, &ctx, &bel, 100);
        // The position has to be contested. Where the equilibrium value sits at
        // the best score one side can reach, that side can force it and so can
        // any strategy that merely avoids the losing line — such a position
        // passes the comparison below even on a solver that has stopped
        // solving. A value well inside the range of reachable outcomes is one
        // only a real equilibrium finds.
        let margin = 0.2 * (hi - lo);
        if margin <= 0.0 || exact - lo < margin || hi - exact < margin {
            skipped[1] += 1;
            continue;
        }

        // The exhaustive side is the expensive one, so it runs once and every
        // variant is held to it. In a two-player zero-sum game the value is
        // unique, so any regret rule that converges must land on the same
        // number — which makes this the whole correctness net for the family.
        for (name, rule) in Cfr::NAMED {
            let mut sv = Solver::new(&s, ctx, Arc::clone(&nets), Cfg { cfr: rule, ..cfg }, bel.clone(), Rng::new(seed));
            assert!(sv.grow_full(NODE_CAP));
            // Exploitability early, before the solve has gone anywhere. Read
            // mid-flight on purpose: a fixed-policy pass must leave the solve
            // able to continue.
            sv.multistep(2);
            sv.finish();
            let early = sv.nash_conv().nash as f64;
            sv.multistep(cfg.iters() - 2);
            sv.finish();
            let late = sv.nash_conv().nash as f64;

            let vals = sv.root_values();
            let v0: f64 = (0..bel[0].len())
                .map(|c| bel[0].p[c] as f64 * vals[0][c] as f64)
                .sum();
            let v1: f64 = (0..bel[1].len())
                .map(|c| bel[1].p[c] as f64 * vals[1][c] as f64)
                .sum();
            // A zero-sum game solved consistently: the two players' root values
            // must cancel. This is the single most useful invariant on the
            // counterfactual-value convention.
            assert!(
                (v0 + v1).abs() < 0.02,
                "seed {seed} {name}: root values are not zero-sum: {v0:.4} + {v1:.4}"
            );
            assert!(
                (v0 - exact).abs() < 0.01,
                "seed {seed} {name}: subgame solver says {v0:.4}, tabular CFR says \
                 {exact:.4} ({}x{} configs)",
                bel[0].len(),
                bel[1].len()
            );
            // A best response can never do worse than the strategy it answers,
            // so NashConv is non-negative; and the whole solve must beat its
            // own first two iterations.
            assert!(
                late > -1e-3,
                "seed {seed} {name}: NashConv is negative: {late:.5}"
            );
            assert!(
                late < early.max(1e-3),
                "seed {seed} {name}: NashConv did not fall: {early:.5} -> {late:.5}"
            );
            eprintln!(
                "  seed {seed:4} {name:>7}: value {v0:+.4} (exact {exact:+.4}, outcomes \
                 {lo:+.3}..{hi:+.3}, {nodes} nodes)  zero-sum {:+.4}  NashConv {early:.4} -> \
                 {late:.4}",
                v0 + v1
            );
        }
        checked += 1;
        if checked >= 3 {
            break;
        }
    }
    eprintln!(
        "verified {} contested micro-endgames against tabular CFR, for every regret rule \
         (skipped {} too large, {} one-sided, {} not a coin play)",
        checked, skipped[0], skipped[1], skipped[2]
    );
    assert!(checked >= 3, "only {} positions exercised", checked);
}

/// A tree with nothing left to grow must stop expanding, not spin.
///
/// Growth stops at a terminal and at the round boundary, so a small endgame's
/// whole frontier goes non-expandable long before a large `s` is spent. Every
/// simulation after that samples a leaf nothing may grow and is thrown away --
/// which is the failure the deleted node ceiling used to cause, and what
/// `exhausted` exists to stop.
#[test]
fn a_solve_stops_expanding_once_the_tree_is_exhausted() {
    let nets = Arc::new(Net::default());
    // Far more expansions than the whole subgame holds.
    let cfg = Cfg { s: 4_000, c: 1.0, budget: Budget::unbounded(), ..Default::default() };
    let mut checked = 0;
    for seed in 0..3000u64 {
        let Some(s) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [
            thinned(uniform_belief(&s, &ctx, 0), CFG_CAP),
            thinned(uniform_belief(&s, &ctx, 1), CFG_CAP),
        ];
        let mut probe = Solver::new(&s, ctx, Arc::clone(&nets), cfg, bel.clone(), Rng::new(seed));
        if !probe.grow_full(NODE_CAP) {
            continue;
        }
        let whole = probe.nodes.len();

        let mut sv = Solver::new(
            &s,
            ctx,
            Arc::clone(&nets),
            cfg,
            bel.clone(),
            Rng::new(0x5A17 + seed),
        );
        sv.run_alone();
        assert!(
            sv.nodes[0].exhausted,
            "seed {seed}: {} of {whole} nodes grown and the root is still open",
            sv.nodes.len()
        );
        assert_eq!(sv.stop_reason(), StopReason::Exhausted);
        assert_eq!(sv.oracle().trace.iters, cfg.iters() as u64);
        assert_eq!(
            sv.nodes.len(),
            whole,
            "seed {seed}: the search sealed the root without growing the subgame"
        );
        // The budget that is left buys nothing and must not be spent.
        assert!(
            !sv.expand_once(),
            "seed {seed}: an exhausted tree still handed back a leaf"
        );
        checked += 1;
        if checked >= 3 {
            break;
        }
    }
    assert!(checked >= 3, "only {checked} endgames exercised");
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
/// brute-force enumeration in `pbs.rs`.
#[test]
fn draw_pass_through_consistency() {
    let nets = Arc::new(Net::default());
    let mut checked = 0;
    let mut cnt = [0usize; 5]; // 0 rejected, 1 built, 2 toolarge, 3 nochance, 4 solved
    for seed in 0..4000u64 {
        let Some(s) = draw_position(seed, 60 + (seed as usize % 120), 4) else {
            cnt[0] += 1;
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
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
            ctx,
            Arc::clone(&nets),
            Cfg { s: 80, c: 1.0, budget: Budget::unbounded(), ..Default::default() },
            bel.clone(),
            Rng::new(seed),
        );
        if !sv.grow_full(20_000) {
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
            assert_eq!(
                n.child.len(),
                1,
                "a draw must have exactly one public child"
            );
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
            let mut ws = sv.states[i].clone();
            let mut b = Belief {
                cfg: n.cfgs[me].to_vec(),
                p: vec![1.0; n.cfgs[me].len()],
            };
            for _ in 0..n.draw_steps {
                let res = reserve(&ws, n.player, &ctx);
                let fu = faceup_counts(&ws, n.player, &ctx);
                b = belief_after_draw(&b, &res, &fu, false);
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
                assert!((sum - 1.0).abs() < 1e-5, "draw row {} sums to {}", ci, sum);
            }
            assert_eq!(
                sv.states[ch].hand_size(n.player),
                sv.states[i].hand_size(n.player) + n.draw_steps,
                "each covered draw adds exactly one coin to the hand"
            );
        }
        sv.multistep(80);
        sv.finish();
        let vals = vec![sv.root_values()];
        let v0: f64 = (0..bel[0].len())
            .map(|c| bel[0].p[c] as f64 * vals[0][0][c] as f64)
            .sum();
        let v1: f64 = (0..bel[1].len())
            .map(|c| bel[1].p[c] as f64 * vals[0][1][c] as f64)
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


/// A Warrior Priest draw inside a subgame: the private mid-round draw is a
/// chance node like a round-start draw, but its children carry the pending
/// forced-play coin, and the forced play is a config-dependent decision node
/// whose legal set is exactly the pending coin's plays. Checks the structure
/// (`belief_after_draw`-consistent support, one public child, per-config
/// legality), that the forced play is walked through, and that the solve runs
/// to completion.
#[test]
fn warrior_priest_draw_walks_through_the_tree() {
    use warchest::state::Z_BAG;
    let nets = Arc::new(Net::default());
    // White: WP at W1, enemy at E1. Hand holds one WP coin (the trigger);
    // the bag holds a WP coin and a Swordsman coin, so a draw can leave
    // either of two pendings. The root belief carries two configs so the
    // draw's children span both pendings.
    let mut s = State::blank(warchest::state::WHITE);
    s.set_unit(17, warchest::state::WHITE, WARRIOR_PRIEST, 1); // (2,3)
    s.set_unit(19, warchest::state::BLACK, FOOTMAN, 3); // (4,3)
                                                        // Full 5-type reserve per player, as `Ctx::new` requires. Only the WP and
                                                        // Swordsman coins are actually reachable.
    for u in [WARRIOR_PRIEST, SWORDSMAN, PIKEMAN, CROSSBOWMAN, ROYAL_COIN] {
        s.add_zone(warchest::state::WHITE, Z_BAG, u, 1);
    }
    for u in [FOOTMAN, ARCHER, CAVALRY, LANCER, ROYAL_COIN] {
        s.add_zone(warchest::state::BLACK, Z_BAG, u, 1);
    }
    // Two coins in White's hand and one in Black's, so the round does not end
    // the moment White plays the trigger: a solve stops growing at the draw
    // that starts the next round, so a mid-round draw is only in the tree when
    // the round has plays left in it.
    s.add_zone(warchest::state::WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    s.add_zone(warchest::state::WHITE, Z_HAND, PIKEMAN, 1);
    s.add_zone(warchest::state::BLACK, Z_HAND, FOOTMAN, 1);
    let ctx = Ctx::new(&s);
    let wp = ctx.slot_of[0][WARRIOR_PRIEST as usize] as u8;
    let sw = ctx.slot_of[0][SWORDSMAN as usize] as u8;
    assert_ne!(wp, sw);
    // Hand size is public, so every config in the belief holds exactly the
    // coins the state shows. The draw's children come from the bag.
    let mut c1 = Config::default();
    c1.hand[wp as usize] = 1;
    c1.hand[ctx.slot_of[0][PIKEMAN as usize] as usize] = 1;
    let mut c2 = Config::default();
    c2.hand[ctx.slot_of[1][FOOTMAN as usize] as usize] = 1;
    let bel = [Belief::point(c1), Belief::point(c2)];
    let mut sv = Solver::new(
        &s,
        ctx,
        Arc::clone(&nets),
        Cfg { s: 8, c: 1.0, budget: Budget::unbounded(), ..Default::default() },
        bel.clone(),
        Rng::new(0x5EED),
    );
    assert!(sv.grow_full(20_000));

    // Find the WP draw node: a chance node whose state is a WarriorPriestDraw.
    let draws: Vec<usize> = (0..sv.nodes.len())
        .filter(|&i| {
            sv.nodes[i].chance && matches!(sv.states[i].pending(), Cont::WarriorPriestDraw { .. })
        })
        .collect();
    assert!(!draws.is_empty(), "expected a WP draw in the grown tree");
    let d = draws[0];
    let n = &sv.nodes[d];
    assert_eq!(n.child.len(), 1, "a draw has exactly one public child");
    assert_eq!(n.draw_steps, 1, "a WP draw is a single draw");
    // The child support must be exactly `belief_after_draw(set_pending=true)`,
    // in order — the invariant the self-play walk asserts at runtime.
    let res = reserve(&sv.states[d], n.player, &ctx);
    let fu = faceup_counts(&sv.states[d], n.player, &ctx);
    let oracle = belief_after_draw(
        &Belief {
            cfg: n.cfgs[n.player as usize].to_vec(),
            p: vec![1.0; n.nc(n.player as usize)],
        },
        &res,
        &fu,
        true,
    );
    let ch = n.child[0];
    assert_eq!(
        sv.nodes[ch].cfgs[n.player as usize].to_vec(),
        oracle.cfg,
        "post-draw support must equal belief_after_draw's, in order"
    );
    // Every child carries its drawn coin in flight (no fizzle here: the bag is
    // not empty), and every draw row is a proper distribution.
    for c in sv.nodes[ch].cfgs[n.player as usize].iter() {
        assert!(c.inflight.is_some(), "a WP draw child carries its coin");
    }
    for ci in 0..n.draw.rows() {
        let sum: f32 = n.draw.row(ci).1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "draw row {} sums to {}", ci, sum);
    }

    // The child is a WarriorPriestPlay decision node. Its actions come from
    // both in-flight coins and its per-config legality is that coin.
    let wpn = &sv.nodes[ch];
    assert!(matches!(
        sv.states[ch].pending(),
        Cont::WarriorPriestPlay { .. }
    ));
    assert!(!wpn.leaf && !wpn.chance);
    assert!(wpn.na() > 0);
    let me = wpn.player as usize;
    for (ci, c) in wpn.cfgs[me].iter().enumerate() {
        let pend = c.inflight.expect("in flight");
        for a in 0..wpn.na() {
            let legal = wpn
                .legal_row(ci)
                .any(|cell| wpn.legal_action[cell] as usize == a);
            assert_eq!(
                legal,
                wpn.aslot[a] == pend as i8,
                "WP play legality must be the in-flight coin"
            );
        }
    }
    // At least two distinct drawn coins are represented.
    let mut drawn: Vec<u8> = wpn.cfgs[me].iter().map(|c| c.inflight.unwrap()).collect();
    drawn.sort_unstable();
    drawn.dedup();
    assert!(
        drawn.len() >= 2,
        "expected both drawn coins in the support, got {:?}",
        drawn
    );

    // The forced play's children hold nothing in flight: the coin is spent.
    for &c in wpn.child.iter() {
        for cc in sv.nodes[c].cfgs[me].iter() {
            assert!(cc.inflight.is_none(), "the forced play spends the coin");
        }
    }

    // Every non-terminal leaf is a valued decision or a chance node the
    // round boundary froze.
    for i in 0..sv.nodes.len() {
        if sv.nodes[i].leaf && !sv.states[i].is_terminal() {
            assert!(
                sv.states[i].is_valued() || sv.states[i].is_chance(),
                "non-terminal leaf {} is {:?}",
                i,
                sv.states[i].pending()
            );
        }
    }

    // The solve runs to completion and the final fixed-policy pass is finite.
    sv.multistep(8);
    sv.finish();
    let vals = vec![sv.root_values()];
    assert!(vals[0][0].iter().all(|v| v.is_finite()));
    assert!(vals[0][1].iter().all(|v| v.is_finite()));
}

#[test]
fn expanding_a_coin_play_stops_at_micro_decisions() {
    let nets = Arc::new(Net::default());
    let mut rng = Rng::new(1234);
    for _ in 0..400 {
        let mut s = make_game(&mut rng, false);
        while !s.is_terminal() {
            let acts = s.legal_actions();
            if matches!(s.pending(), Cont::MainPlay)
                && acts.iter().any(|a| {
                    let mut child = s.clone();
                    child.apply_inplace(*a);
                    !child.is_terminal()
                        && !matches!(child.pending(), Cont::MainPlay | Cont::Draw { .. })
                })
            {
                let ctx = Ctx::new(&s);
                let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
                let sv = Solver::new(
                    &s,
                    ctx,
                    Arc::clone(&nets),
                    Cfg::default(),
                    bel,
                    Rng::new(rng.next_u64()),
                );
                assert!(
                    sv.nodes.iter().zip(&sv.states).any(|(n, state)| {
                        n.leaf && state.is_valued() && !matches!(state.pending(), Cont::MainPlay)
                    }),
                    "a micro-decision was grown through instead of left as a value leaf"
                );
                return;
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    panic!("no compound play found in 400 random games");
}

/// A solve never grows past its budget, in any term.
///
/// This is the property the whole of admission rests on. A slot's arenas are
/// allocated once at the budget and reused for every solve that runs in it, so
/// a solve that could exceed the budget in even one term would write past an
/// arena -- and admission, which is a free list and measures nothing, would
/// have no way to know. Every term is checked, at a budget far under what these
/// positions want, so every term's guard is exercised.
#[test]
fn a_solve_never_grows_past_its_budget() {
    let nets = Arc::new(Net::default());
    let budget = Budget {
        nodes: 512,
        rows: 256,
        boards: 128,
        configs: 256,
        cidx: 4_096,
        reach: 8_192,
        cells: 2_048,
        draws: 2_048,
    };
    let cfg = Cfg { s: 4_000, c: 1.0, budget, ..Default::default() };
    let mut checked = 0;
    let mut hits = 0;
    for seed in 0..3000u64 {
        let Some(st) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            continue;
        };
        let ctx = Ctx::new(&st);
        let bel = [
            thinned(uniform_belief(&st, &ctx, 0), CFG_CAP),
            thinned(uniform_belief(&st, &ctx, 1), CFG_CAP),
        ];
        let mut sv = Solver::new(&st, ctx, Arc::clone(&nets), cfg, bel, Rng::new(seed));
        sv.run_alone();
        for e in Ent::ALL {
            assert!(
                sv.used(e) <= budget.cap(e),
                "seed {seed}: {} {} > {}",
                e.name(),
                sv.used(e),
                budget.cap(e)
            );
        }
        if sv.nodes[0].leaf {
            // The first expansion did not fit. The root stayed a leaf, which
            // is the truncation: there is no average to read.
        } else {
            let root = sv.average_strategy(0, 0);
            assert!(!root.is_empty(), "seed {seed}: a truncated solve has no average");
            assert!(
                root.iter().sum::<f32>() > 0.99 && root.iter().sum::<f32>() < 1.01,
                "seed {seed}: the root average sums to {}",
                root.iter().sum::<f32>()
            );
        }
        hits += sv.budget_hit() as usize;
        checked += 1;
        if checked >= 200 {
            break;
        }
    }
    assert!(checked >= 200, "only {checked} positions exercised");
    // The budget above is far under what the game wants, so it has to bite.
    // A run that reports no hits at all is a run whose guards are not reached
    // and whose assertions above therefore prove nothing.
    assert!(hits > checked / 2, "the budget bit on only {hits} of {checked} solves");
    eprintln!("{hits} of {checked} solves reached the budget");
}

fn roomy_budget() -> Budget {
    Budget {
        nodes: 1_000_000,
        rows: 1_000_000,
        boards: 1_000_000,
        configs: 1_000_000,
        cidx: 10_000_000,
        reach: 10_000_000,
        cells: 10_000_000,
        draws: 10_000_000,
    }
}

fn tight_in(e: Ent) -> Budget {
    let mut b = roomy_budget();
    match e {
        Ent::Node => b.nodes = 512,
        Ent::Cell => b.cells = 2_048,
        Ent::Reach => b.reach = 8_192,
        Ent::Draw => b.draws = 64,
        Ent::Row => b.rows = 64,
        Ent::Board => b.boards = 32,
        Ent::Config => b.configs = 16,
        Ent::Cidx => b.cidx = 512,
    }
    b
}

/// A budget tight in one entity still yields a tree that fits the slot, and a
/// normalised root policy. Eight cases, one entity at a time: the other seven
/// are roomy, so a miss in that entity's append-point reserve is what would
/// grow the contract past the cap.
#[test]
fn a_solve_fits_a_budget_tight_in_one_entity() {
    let nets = Arc::new(Net::default());
    for e in Ent::ALL {
        let budget = tight_in(e);
        let cfg = Cfg { s: 512, c: 1.0, budget, ..Default::default() };
        let mut checked = 0;
        let mut hits = 0;
        for seed in 0..8_000u64 {
            let Some(st) = micro_position(seed, 60 + (seed as usize % 120), 6) else {
                continue;
            };
            let ctx = Ctx::new(&st);
            let bel = [
                thinned(uniform_belief(&st, &ctx, 0), CFG_CAP),
                thinned(uniform_belief(&st, &ctx, 1), CFG_CAP),
            ];
            let mut sv = Solver::new(&st, ctx, Arc::clone(&nets), cfg, bel, Rng::new(seed));
            sv.run_alone();
            for x in Ent::ALL {
                assert!(
                    sv.used(x) <= budget.cap(x),
                    "{} seed {seed}: {} {} > {}",
                    e.name(),
                    x.name(),
                    sv.used(x),
                    budget.cap(x)
                );
            }
            if sv.nodes[0].leaf || sv.nodes[0].chance {
                continue;
            }
            let me = sv.nodes[0].player as usize;
            for c in 0..sv.nodes[0].nc(me) {
                let row = sv.average_strategy(0, c);
                let sum: f32 = row.iter().sum();
                assert!(
                    !row.is_empty() && sum > 0.99 && sum < 1.01,
                    "{} seed {seed} config {c}: the root policy sums to {sum}",
                    e.name()
                );
            }
            hits += sv.budget_hit() as usize;
            checked += 1;
            if checked >= 200 {
                break;
            }
        }
        assert!(checked >= 200, "{}: only {checked} grown roots", e.name());
        assert!(hits > 0, "{}: the tight budget never bit", e.name());
        eprintln!("{}: {hits} of {checked} solves reached the budget", e.name());
    }
}
