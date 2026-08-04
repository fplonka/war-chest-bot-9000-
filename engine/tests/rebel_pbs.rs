//! Correctness tests for the public belief state.
//!
//! Belief bookkeeping bugs do not crash — they just make the bot quietly wrong —
//! so these are the load-bearing tests of the whole ReBeL stack.
//!
//! 1. `features_do_not_leak_private_information`: the PBS encoding must be a
//!    function of the public state and the beliefs alone. Swapping a player's
//!    true config for any other config consistent with the same public counts
//!    must not move a single feature.
//! 2. `belief_tracker_matches_brute_force`: the incremental tracker is compared
//!    against an exhaustive enumeration of every world consistent with the
//!    observation sequence, weighted by exact draw probabilities and the
//!    announced policy. The brute-force side goes through the engine only — it
//!    never touches `Belief`, `advance_config` or `belief_after_draw`.

use std::collections::HashMap;

use warchest::rebel::*;
use warchest::rng::Rng;
use warchest::search::node_actions;
use warchest::selfplay::make_game;
use warchest::state::{State, Z_BAG, Z_FACEDOWN, Z_FACEUP};
use warchest::Action;

/// Instantiate a world from the shared public state plus both configs.
fn world(pubs: &State, ctx: &Ctx, c: &(Config, Config)) -> State {
    let mut w = pubs.clone();
    set_config(&mut w, 0, ctx, &c.0);
    set_config(&mut w, 1, ctx, &c.1);
    w
}

fn pick(c: &(Config, Config), p: u8) -> &Config {
    if p == 0 {
        &c.0
    } else {
        &c.1
    }
}

fn with(c: &(Config, Config), p: u8, n: Config) -> (Config, Config) {
    if p == 0 {
        (n, c.1)
    } else {
        (c.0, n)
    }
}

/// Uniform over the private actions legal for this config — the announced
/// policy both sides of the test use.
#[allow(clippy::type_complexity)]
fn uniform_row(
    s: &State,
    ctx: &Ctx,
    p: u8,
    c: &Config,
) -> (Vec<Action>, Vec<i8>, Vec<bool>, Vec<f64>) {
    let (acts, aslot, fdown) = node_actions(s, p, ctx, std::slice::from_ref(c));
    let legal: Vec<bool> = aslot
        .iter()
        .map(|&k| k < 0 || c.hand[k as usize] > 0)
        .collect();
    let n = legal.iter().filter(|&&x| x).count() as f64;
    let probs = legal
        .iter()
        .map(|&l| if l && n > 0.0 { 1.0 / n } else { 0.0 })
        .collect();
    (acts, aslot, fdown, probs)
}

/// The engine's own chance distribution: the bag, or the refilled discard pile
/// when the bag has run out. Bag emptiness is public, so every world reshuffles
/// at the same moment.
fn draw_weights(s: &State, p: u8, acts: &[Action]) -> Vec<f64> {
    let bag_total: u8 = s.zones[p as usize][Z_BAG].iter().sum();
    acts.iter()
        .map(|a| match a {
            Action::DrawCoin { unit } => {
                let u = *unit as usize;
                if bag_total > 0 {
                    s.zones[p as usize][Z_BAG][u] as f64
                } else {
                    (s.zones[p as usize][Z_FACEUP][u] + s.zones[p as usize][Z_FACEDOWN][u]) as f64
                }
            }
            _ => 0.0,
        })
        .collect()
}

#[test]
fn features_do_not_leak_private_information() {
    let mut rng = Rng::new(99);
    let mut checked = 0;
    for g in 0..12u64 {
        let mut s = make_game(&mut Rng::new(g + 1), false);
        let ctx = Ctx::new(&s);
        let bel = [
            Belief::point(Config::default()),
            Belief::point(Config::default()),
        ];
        for _ in 0..150 {
            if s.is_terminal() {
                break;
            }
            for p in 0..2u8 {
                let res = reserve(&s, p, &ctx);
                let truth = true_config(&s, p, &ctx);
                let all = enumerate_configs(&res, truth.hand_size(), truth.fd_size());
                if all.len() < 2 {
                    continue;
                }
                let mut a = vec![0.0f32; FEAT];
                let mut b = vec![0.0f32; FEAT];
                let mut sa = s.clone();
                set_config(&mut sa, p, &ctx, &all[0]);
                write_features(&sa, &ctx, &bel, &mut a);
                let mut sb = s.clone();
                set_config(&mut sb, p, &ctx, &all[all.len() - 1]);
                write_features(&sb, &ctx, &bel, &mut b);
                assert_eq!(
                    a, b,
                    "features changed when only player {}'s hidden config changed",
                    p
                );
                checked += 1;
            }
            let acts = s.legal_actions();
            if acts.is_empty() {
                break;
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    assert!(checked > 500, "not enough positions exercised: {}", checked);
}

#[test]
fn belief_tracker_matches_brute_force() {
    for seed in 0..6u64 {
        run_one(seed);
    }
}

fn run_one(seed: u64) {
    let mut rng = Rng::new(seed + 1);
    let mut s = make_game(&mut rng, false);
    let ctx = Ctx::new(&s);
    let mut bel = [
        Belief::point(Config::default()),
        Belief::point(Config::default()),
    ];
    // Exhaustive posterior over joint private configs. The public state is
    // shared by every world, so a world is just the pair.
    let mut worlds: HashMap<(Config, Config), f64> = HashMap::new();
    worlds.insert((Config::default(), Config::default()), 1.0);

    let mut steps = 0;
    while !s.is_terminal() && steps < 60 && worlds.len() < 20_000 {
        let p = s.to_act();
        if s.is_chance() {
            let mut next: HashMap<(Config, Config), f64> = HashMap::new();
            for (c, w) in worlds.iter() {
                let ws = world(&s, &ctx, c);
                let acts = ws.legal_actions();
                let wts = draw_weights(&ws, p, &acts);
                let tot: f64 = wts.iter().sum();
                assert!(tot > 0.0, "a world had nothing to draw");
                for (i, a) in acts.iter().enumerate() {
                    if wts[i] <= 0.0 {
                        continue;
                    }
                    let mut nx = ws.clone();
                    nx.apply_inplace(*a);
                    let nc = with(c, p, true_config(&nx, p, &ctx));
                    *next.entry(nc).or_insert(0.0) += w * wts[i] / tot;
                }
            }
            worlds = next;

            let res = reserve(&s, p, &ctx);
            let fu = faceup_counts(&s, p, &ctx);
            bel[p as usize] = belief_after_draw(&bel[p as usize], &res, &fu);

            let acts = s.legal_actions();
            let wts = draw_weights(&s, p, &acts);
            let ai = rng.weighted_index(&wts);
            s.apply_inplace(acts[ai]);
            compare(&worlds, &bel, seed, steps, "draw");
            steps += 1;
            continue;
        }

        // Decision node: sample the real action from the announced policy.
        let truth = true_config(&s, p, &ctx);
        let (acts, _, _, probs) = uniform_row(&s, &ctx, p, &truth);
        let chosen = rng.weighted_index(&probs);
        let obs = obs_key(&acts[chosen]);

        // Brute force: keep the worlds whose policy could have produced this
        // observation, weighted by how much probability they put on it.
        let mut next: HashMap<(Config, Config), f64> = HashMap::new();
        for (c, w) in worlds.iter() {
            let ws = world(&s, &ctx, c);
            let (wacts, _, _, wprobs) = uniform_row(&ws, &ctx, p, pick(c, p));
            for (i, a) in wacts.iter().enumerate() {
                if wprobs[i] <= 0.0 || obs_key(a) != obs {
                    continue;
                }
                let mut nx = ws.clone();
                nx.apply_inplace(*a);
                let nc = with(c, p, true_config(&nx, p, &ctx));
                *next.entry(nc).or_insert(0.0) += w * wprobs[i];
            }
        }
        let tot: f64 = next.values().sum();
        assert!(tot > 0.0, "brute force lost every world at step {}", steps);
        for v in next.values_mut() {
            *v /= tot;
        }
        worlds = next;

        // Tracker: the same update, done incrementally over configs.
        let cfgs = bel[p as usize].cfg.clone();
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in cfgs.iter().enumerate() {
            let (cacts, aslot, fdown, cprobs) = uniform_row(&s, &ctx, p, c);
            for (i, a) in cacts.iter().enumerate() {
                if cprobs[i] <= 0.0 || obs_key(a) != obs {
                    continue;
                }
                if let Some(n) = advance_config(c, aslot[i], fdown[i]) {
                    pairs.push((n, bel[p as usize].p[ci] * cprobs[i] as f32));
                }
            }
        }
        bel[p as usize] = Belief::from_pairs(pairs);

        s.apply_inplace(acts[chosen]);
        compare(&worlds, &bel, seed, steps, "decision");
        steps += 1;
    }
    // Uniform play discards face down constantly, so the exhaustive side hits
    // its world budget after a couple of rounds. That is the point at which it
    // stops being a usable oracle, not a failure.
    assert!(
        steps > 8,
        "seed {} produced too short a trace ({} steps, {} worlds)",
        seed,
        steps,
        worlds.len()
    );
    eprintln!(
        "seed {}: verified {} steps against {} exhaustively enumerated worlds",
        seed,
        steps,
        worlds.len()
    );
}

fn compare(
    worlds: &HashMap<(Config, Config), f64>,
    bel: &[Belief; 2],
    seed: u64,
    step: usize,
    what: &str,
) {
    for p in 0..2u8 {
        let mut exact: HashMap<Config, f64> = HashMap::new();
        for (c, w) in worlds.iter() {
            *exact.entry(*pick(c, p)).or_insert(0.0) += w;
        }
        let tot: f64 = exact.values().sum();
        for v in exact.values_mut() {
            *v /= tot;
        }
        let b = &bel[p as usize];
        for (c, w) in exact.iter() {
            if *w < 1e-9 {
                continue;
            }
            let got = b.index_of(c).map(|i| b.p[i] as f64).unwrap_or(0.0);
            assert!(
                (got - w).abs() < 1e-5,
                "seed {} step {} ({}) player {}: config {:?} exact {:.9} tracker {:.9}",
                seed,
                step,
                what,
                p,
                c,
                w,
                got
            );
        }
        for (i, c) in b.cfg.iter().enumerate() {
            if b.p[i] < 1e-9 {
                continue;
            }
            let w = exact.get(c).copied().unwrap_or(0.0);
            assert!(
                (b.p[i] as f64 - w).abs() < 1e-5,
                "seed {} step {} ({}) player {}: tracker has a phantom config {:?} at {:.9}",
                seed,
                step,
                what,
                p,
                c,
                b.p[i]
            );
        }
    }
}

/// The belief block the solver writes per leaf per CFR iteration must equal the
/// straightforward definition: the hand-key marginal, and the belief-weighted
/// bag and face-down composition.
///
/// `write_belief_block` accumulates the two hidden components and derives the
/// bag as `reserve - E[hand] - E[facedown]` instead of forming each config's
/// bag first — half the arithmetic in a loop that runs once per leaf per player
/// per iteration, but a different expression, so it gets an oracle.
#[test]
fn belief_block_matches_the_direct_definition() {
    let mut rng = Rng::new(0xB33F);
    let mut checked = 0usize;
    for seed in 0..400u64 {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9));
        let reserve: [u8; NSLOT] = std::array::from_fn(|_| r.below(6) as u8);
        let hand_size = r.below(4) as u8;
        let fd_size = r.below(4) as u8;
        let cfgs = enumerate_configs(&reserve, hand_size, fd_size);
        if cfgs.is_empty() {
            continue;
        }
        // Unnormalised weights, including the all-zero case the smoothing
        // fallback exists for.
        let w: Vec<f32> = if seed % 17 == 0 {
            vec![0.0; cfgs.len()]
        } else {
            cfgs.iter().map(|_| rng.unit_f64() as f32).collect()
        };

        let mut got = vec![0.0f32; BELIEF_DIM];
        write_belief_block(&cfgs, None, &w, &reserve, &mut got);

        let mut bel = Belief {
            cfg: cfgs.clone(),
            p: w.clone(),
        };
        bel.normalize();
        let hm = bel.hand_marginal();
        let (bag, fd) = bel.composition(&reserve);
        for h in 0..NHAND {
            assert!(
                (got[h] - hm[h]).abs() < 2e-6,
                "hand marginal {} differs: {} vs {}",
                h,
                got[h],
                hm[h]
            );
        }
        for k in 0..NSLOT {
            assert!(
                (got[NHAND + k] - bag[k]).abs() < 2e-6,
                "bag composition {} differs: {} vs {}",
                k,
                got[NHAND + k],
                bag[k]
            );
            assert!(
                (got[NHAND + NSLOT + k] - fd[k]).abs() < 2e-6,
                "face-down composition {} differs: {} vs {}",
                k,
                got[NHAND + NSLOT + k],
                fd[k]
            );
        }
        checked += 1;
    }
    assert!(checked > 100, "only {} belief blocks exercised", checked);
}
