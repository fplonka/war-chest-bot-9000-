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
use warchest::search::{node_actions, Cfg, Cfr, Nets, Solver};
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
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
        // Keep the exhaustive side affordable.
        if bel[0].len() * bel[1].len() > 64 {
            continue;
        }

        let cfg = Cfg {
            nodes: 200_000,
            iters: 500,
            ..Default::default()
        };
        {
            let mut sv = Solver::new(&s, ctx, &nets, cfg, bel.clone());
            sv.grow_full();
            // If any leaf were non-terminal the (empty) network would silently
            // return zero and the comparison would be meaningless.
            assert!(
                sv.nodes
                    .iter()
                    .zip(&sv.states)
                    .all(|(n, st)| !n.leaf || st.is_terminal()),
                "the whole remaining game must fit inside the subgame"
            );
            if sv.nodes.len() > 8_000 {
                continue;
            }
            // A position where every line ends in the same score tests nothing.
            let mut outcomes: Vec<i32> = sv
                .nodes
                .iter()
                .zip(&sv.states)
                .filter(|(n, st)| n.leaf && st.is_terminal())
                .map(|(_, st)| (st.utility(0) * 1000.0) as i32)
                .collect();
            outcomes.sort_unstable();
            outcomes.dedup();
            if outcomes.len() < 2 {
                continue;
            }
        }

        // The exhaustive side is the expensive one, so it runs once and every
        // variant is held to it. In a two-player zero-sum game the value is
        // unique, so any regret rule that converges must land on the same
        // number — which makes this the whole correctness net for the family.
        let exact = oracle_value(&s, &ctx, &bel, 100);
        for (name, rule) in Cfr::NAMED {
            let mut sv = Solver::new(&s, ctx, &nets, Cfg { cfr: rule, ..cfg }, bel.clone());
            sv.grow_full();
            // Exploitability early, before the solve has gone anywhere. Read
            // mid-flight on purpose: a fixed-policy pass must leave the solve
            // able to continue.
            sv.multistep(2);
            sv.finish();
            let early = sv.nash_conv().nash as f64;
            sv.multistep(cfg.iters - 2);
            sv.finish();
            let late = sv.nash_conv().nash as f64;

            let vals = vec![sv.root_values()];
            let v0: f64 = (0..bel[0].len())
                .map(|c| bel[0].p[c] as f64 * vals[0][0][c] as f64)
                .sum();
            let v1: f64 = (0..bel[1].len())
                .map(|c| bel[1].p[c] as f64 * vals[0][1][c] as f64)
                .sum();
            // A zero-sum game solved consistently: the two players' root values
            // must cancel. This is the single most useful invariant on the
            // counterfactual-value convention.
            assert!(
                (v0 + v1).abs() < 0.02,
                "seed {seed} {name}: root values are not zero-sum: {v0:.4} + {v1:.4}"
            );
            assert!(
                (v0 - exact).abs() < 0.03,
                "seed {seed} {name}: subgame solver says {v0:.4}, tabular CFR says \
                 {exact:.4} ({}x{} configs)",
                bel[0].len(),
                bel[1].len()
            );
            // A best response can never do worse than the strategy it answers,
            // so NashConv is non-negative; and 500 iterations of any of these
            // rules must beat 2.
            assert!(
                late > -1e-3,
                "seed {seed} {name}: NashConv is negative: {late:.5}"
            );
            assert!(
                late < early.max(1e-3),
                "seed {seed} {name}: NashConv did not fall: {early:.5} -> {late:.5}"
            );
            eprintln!(
                "  seed {seed:4} {name:>7}: value {v0:+.4} (exact {exact:+.4})  \
                 zero-sum {:+.4}  NashConv {early:.4} -> {late:.4}",
                v0 + v1
            );
        }
        checked += 1;
        if checked >= 4 {
            break;
        }
    }
    assert!(checked >= 4, "only {} positions exercised", checked);
    eprintln!(
        "verified {} micro-endgames against tabular CFR, for every regret rule",
        checked
    );
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
    eprintln!(
        "     exact   {}",
        budgets
            .iter()
            .map(|b| format!("{:>9}", b))
            .collect::<String>()
    );
    for seed in 0..3000u64 {
        let Some(s) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
        if bel[0].len() * bel[1].len() > 64 {
            continue;
        }
        let mut probe = Solver::new(
            &s,
            ctx,
            &nets,
            Cfg {
                nodes: 200_000,
                iters: 1,
                ..Default::default()
            },
            bel.clone(),
        );
        probe.grow_full();
        if !probe
            .nodes
            .iter()
            .zip(&probe.states)
            .all(|(n, st)| !n.leaf || st.is_terminal())
            || probe.nodes.len() > 8_000
        {
            continue;
        }
        let mut o: Vec<i32> = probe
            .nodes
            .iter()
            .zip(&probe.states)
            .filter(|(n, st)| n.leaf && st.is_terminal())
            .map(|(_, st)| (st.utility(0) * 1000.0) as i32)
            .collect();
        o.sort_unstable();
        o.dedup();
        if o.len() < 2 {
            continue;
        }
        let exact = oracle_value(&s, &ctx, &bel, 100);
        let mut line = format!("  {:+.4}   ", exact);
        for (bi, &t) in budgets.iter().enumerate() {
            let mut sv = Solver::new(
                &s,
                ctx,
                &nets,
                Cfg {
                    nodes: 200_000,
                    iters: t,
                    ..Default::default()
                },
                bel.clone(),
            );
            sv.grow_full();
            sv.multistep(t);
            sv.finish();
            let vals = vec![sv.root_values()];
            let v0: f64 = (0..bel[0].len())
                .map(|c| bel[0].p[c] as f64 * vals[0][0][c] as f64)
                .sum();
            let v1: f64 = (0..bel[1].len())
                .map(|c| bel[1].p[c] as f64 * vals[0][1][c] as f64)
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
            &nets,
            Cfg {
                nodes: 20_000,
                node_cap: 20_000,
                iters: 80,
                ..Default::default()
            },
            bel.clone(),
        );
        sv.grow_full();
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
/// legality), that every non-terminal leaf is a MainPlay state, and that the
/// solve runs to completion.
#[test]
fn warrior_priest_draw_walks_through_the_tree() {
    use warchest::state::Z_BAG;
    let nets = Nets::default();
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
    s.add_zone(warchest::state::WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    let ctx = Ctx::new(&s);
    let wp = ctx.slot_of[0][WARRIOR_PRIEST as usize] as u8;
    let sw = ctx.slot_of[0][SWORDSMAN as usize] as u8;
    assert_ne!(wp, sw);
    // Hand size is public, so every config in the belief holds exactly the one
    // coin the state shows. The draw's two children come from the bag.
    let mut c1 = Config::default();
    c1.hand[wp as usize] = 1;
    let bel = [Belief::point(c1), Belief::point(Config::default())];
    let mut sv = Solver::new(
        &s,
        ctx,
        &nets,
        Cfg {
            nodes: 20_000,
            node_cap: 20_000,
            iters: 8,
            ..Default::default()
        },
        bel.clone(),
    );
    sv.grow_full();

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

    // Every non-terminal leaf is a MainPlay state.
    for i in 0..sv.nodes.len() {
        if sv.nodes[i].leaf && !sv.states[i].is_terminal() {
            assert!(
                matches!(sv.states[i].pending(), Cont::MainPlay),
                "non-terminal leaf {} is not a MainPlay state",
                i
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
fn growing_a_coin_play_finishes_its_micro_decisions() {
    let nets = Nets::default();
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
                    &nets,
                    Cfg {
                        nodes: 200_000,
                        ..Default::default()
                    },
                    bel,
                );
                assert!(
                    sv.nodes
                        .iter()
                        .zip(&sv.states)
                        .skip(1)
                        .any(|(n, state)| {
                            !n.leaf && !matches!(state.pending(), Cont::MainPlay)
                        }),
                    "the compound play's micro-decision was not grown"
                );
                assert!(
                    sv.nodes.iter().zip(&sv.states).all(|(n, state)| {
                        !n.leaf
                            || state.is_terminal()
                            || matches!(state.pending(), Cont::MainPlay)
                    }),
                    "a micro-decision remained as a value leaf"
                );
                return;
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    panic!("no compound play found in 400 random games");
}

/// Growing less often searches worse, and this is what it costs.
///
/// `grow_every` runs several iterations between host wakes. On the farm that
/// is worth a great deal -- 28.6 solves a second at one, 33.3 at two, 47.4 at
/// four -- because the cards are idle through every host turnaround it
/// removes. It is a search knob all the same: the second and later expansion
/// phases of a round select from a tree the host has not grown, so GT-CFR
/// builds a different tree, and the rate cannot say whether it is a worse one.
///
/// `nash_conv` can. It is the exploitability of the finite search game, so it
/// says whether the strategy the solve arrives at is nearer equilibrium. Same
/// roots, same iteration count, same node budget, same random stream; only the
/// size of a round differs. Over forty-six micro-endgames:
///
/// ```text
/// every=1 0.01598   every=2 0.02048 (+28%)   every=4 0.02641 (+65%)
/// ```
///
/// So it is not free at any setting, and the default stays at one. A dozen
/// roots had said two was free -- it is worth knowing that a sample that size
/// will say that.
#[test]
fn growing_less_often_searches_worse() {
    let nets = Nets::default();
    let (mut sum, mut checked) = ([0.0f64; 3], 0usize);
    const EVERY: [usize; 3] = [1, 2, 4];
    for seed in 0..20000u64 {
        let Some(s) = micro_position(seed, 60 + (seed as usize % 120), 3) else {
            continue;
        };
        let ctx = Ctx::new(&s);
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
        if bel[0].len() * bel[1].len() > 64 {
            continue;
        }
        let mut conv = [0.0f64; 3];
        for (i, &every) in EVERY.iter().enumerate() {
            // Sized so the solve finishes rather than striking the node cap:
            // `iters * expand` expansions of about seventeen nodes each.
            let cfg = Cfg {
                nodes: 3_000,
                iters: 32,
                expand: 4,
                grow_every: every,
                ..Default::default()
            };
            let mut sv = Solver::new(&s, ctx, &nets, cfg, bel.clone());
            // The same stream for every variant, so what differs is the size
            // of a round and not the draws.
            let mut rng = warchest::rng::Rng::new(0x51D5 ^ seed);
            sv.solve(&mut rng);
            if sv.capped() {
                conv = [f64::NAN; 3];
                break;
            }
            conv[i] = sv.nash_conv().nash as f64;
        }
        if conv.iter().any(|c| c.is_nan()) {
            continue;
        }
        for i in 0..3 {
            sum[i] += conv[i];
        }
        checked += 1;
        if checked == 60 {
            break;
        }
    }
    assert!(checked >= 30, "only {checked} positions were solvable");
    let mean: Vec<f64> = sum.iter().map(|v| v / checked as f64).collect();
    println!(
        "NashConv over {checked} roots: every=1 {:.5}  every=2 {:.5}  every=4 {:.5}",
        mean[0], mean[1], mean[2]
    );
    // The ordering is the finding, and it is what a change to the search
    // would disturb. If growing less often ever stops costing exploitability,
    // the farm can have its 1.66x and this test should be the thing that says
    // so.
    assert!(
        mean[0] < mean[1] && mean[1] < mean[2],
        "growing less often no longer costs exploitability: {:.5} {:.5} {:.5} \
         -- re-measure the farm, `grow_every` may be worth taking now",
        mean[0],
        mean[1],
        mean[2]
    );
}
