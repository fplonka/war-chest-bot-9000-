//! Correctness tests for the public belief state.
//!
//! Belief bookkeeping bugs do not crash — they just make the bot quietly wrong —
//! so these are the load-bearing tests of the whole ReBeL stack.
//!
//! 1. `features_do_not_leak_private_information`: the public encoding must be a
//!    function of the public state alone. Swapping a player's true config for
//!    any other config consistent with the same public counts must not move a
//!    single feature.
//! 2. `a_solve_reads_only_the_beliefs`: the same property one level up. The
//!    value network is now asked about specific configs, so the leak the first
//!    test guards is no longer the only way private information could reach it;
//!    this one solves the same public position in two different worlds and
//!    requires bit-identical values and strategies.
//! 3. `config_features_separate_every_config`: the value function's argument
//!    must actually identify the config. If two distinct private states shared
//!    a feature vector the network could not tell them apart, which is the bug
//!    the hand-keyed encoding had by construction.
//! 4. `the_value_function_separates_configs_sharing_a_hand`: end to end — two
//!    configs with the same hand and different face-down piles must get
//!    different leaf values, and therefore different play.
//! 5. `belief_tracker_matches_brute_force`: the incremental tracker is compared
//!    against an exhaustive enumeration of every world consistent with the
//!    observation sequence, weighted by exact draw probabilities and the
//!    announced policy. The brute-force side goes through the engine only — it
//!    never touches `Belief`, `advance_config` or `belief_after_draw`.

use std::collections::HashMap;

use warchest::net::Net;
use warchest::rebel::*;
use warchest::rng::Rng;
use warchest::search::{node_actions, Cfg, Nets, Solver};
use warchest::selfplay::make_game;
use warchest::state::{Cont, State, Z_BAG, Z_FACEDOWN, Z_FACEUP};
use warchest::units::{write_card_features, CARD_FEATS};
use warchest::Action;

/// A network with random weights, for tests that need the value function to
/// actually distinguish things rather than return zero. LayerNorms start at the
/// identity (gamma = 1, beta = 0); everything else is small and uniform.
fn random_net(seed: u64) -> Net {
    let mut r = Rng::new(seed);
    let dims = warchest::net::MODEL_TAG;
    let layout = warchest::net::NetLayout::new(&dims).unwrap();
    let mut draw = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| (r.unit_f64() as f32 - 0.5) * scale)
            .collect()
    };
    let w = draw(layout.w_len, 0.2);
    let b = draw(layout.b_len, 0.2);
    let mut ln = vec![0.0; layout.ln_len];
    for &(gamma, beta) in &layout.norms {
        ln[gamma..beta].fill(1.0);
    }
    Net::from_flat(&dims, &w, &b, &ln).expect("random net")
}

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
    let legal: Vec<bool> = aslot.iter().map(|&k| action_legal(c, k)).collect();
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

/// Swapping a player's hidden config for any other consistent with the same
/// public counts must not move a single feature. Run over both drafts: the
/// starter matchup never produces a Footman, Mercenary or Royal Guard trigger,
/// so it would not exercise the pending-maneuver mask at all.
fn leak_check(random_draft: bool) -> (usize, usize) {
    let mut rng = Rng::new(99);
    let (mut checked, mut pending_seen) = (0, 0);
    for g in 0..12u64 {
        let mut s = make_game(&mut Rng::new(g + 1), random_draft);
        let ctx = Ctx::new(&s);
        for _ in 0..150 {
            if s.is_terminal() {
                break;
            }
            for p in 0..2u8 {
                let res = reserve(&s, p, &ctx);
                let truth = true_config(&s, p, &ctx);
                let all = enumerate_configs(
                    &res,
                    truth.hand_size(),
                    truth.fd_size(),
                    truth.inflight.is_some(),
                );
                if all.len() < 2 {
                    continue;
                }
                let mut a = vec![0.0f32; PUBFEAT];
                let mut b = vec![0.0f32; PUBFEAT];
                let mut sa = s.clone();
                set_config(&mut sa, p, &ctx, &all[0]);
                write_public_features(&sa, &ctx, &mut a);
                let mut sb = s.clone();
                set_config(&mut sb, p, &ctx, &all[all.len() - 1]);
                write_public_features(&sb, &ctx, &mut b);
                assert_eq!(
                    a, b,
                    "features changed when only player {}'s hidden config changed",
                    p
                );
                // The pending-maneuver mask is channel `6 + NSLOT` of the
                // hex-major block.
                if (0..warchest::board::N_HEXES).any(|h| a[h * HEX_CH + 6] != 0.0) {
                    pending_seen += 1;
                }
                checked += 1;
            }
            let acts = s.legal_actions();
            if acts.is_empty() {
                break;
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    (checked, pending_seen)
}

#[test]
fn features_do_not_leak_private_information() {
    let (checked, _) = leak_check(false);
    assert!(checked > 500, "not enough positions exercised: {}", checked);
}

#[test]
fn features_do_not_leak_private_information_random_draft() {
    let (checked, pending) = leak_check(true);
    assert!(checked > 500, "not enough positions exercised: {}", checked);
    // A feature block that never fires is dead weight the encoding is paying
    // for; assert the pending-maneuver mask is actually reached.
    assert!(
        pending > 0,
        "the pending-maneuver mask never fired over {} positions",
        checked
    );
}

#[test]
fn belief_tracker_matches_brute_force() {
    for seed in 0..6u64 {
        run_one(seed);
    }
}

fn run_one(seed: u64) {
    // The starter matchup: no Warrior Priest, so the pending-coin machinery is
    // dormant — the comparison still runs, and is the regression test for the
    // ordinary path.
    run_one_draft(seed, &[17, 12, 4, 9], &[1, 3, 8, 16]);
}

/// The same exhaustive-vs-incremental comparison on a draft with both Warrior
/// Priests: private mid-round draws put `inflight` into the config, so the
/// belief update, the walk and the brute force all carry it.
#[test]
fn belief_tracker_matches_brute_force_with_warrior_priests() {
    for seed in 0..4u64 {
        run_one_draft(seed + 100, &[18, 17, 12, 4], &[54, 1, 3, 8]);
    }
}

fn run_one_draft(seed: u64, white: &[u16], black: &[u16]) {
    let mut rng = Rng::new(seed + 1);
    let mut s = State::from_draft(white, black, warchest::state::WHITE);
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
            let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
            bel[p as usize] = belief_after_draw(&bel[p as usize], &res, &fu, wp);

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

/// The config key must stay inside the packed `u64` budget (the key shares a
/// word with the element index in the solver's sort): 38 bits of config +
/// `IDX_BITS` index bits must not overflow. Also pins that the key
/// distinguishes pendings and that hand slots never exceed the two-bit width.
#[test]
fn config_key_packing_has_headroom() {
    for seed in 0..200u64 {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut c = Config::default();
        for k in 0..NSLOT {
            // A hand holds at most 3 and a face-down slot at most 5.
            c.hand[k] = r.below(4) as u8;
            c.fd[k] = r.below(6) as u8;
        }
        c.inflight = if r.next_u64() & 1 == 0 {
            None
        } else {
            Some(r.below(NSLOT) as u8)
        };
        let key = c.key();
        assert!(
            key < (1u64 << (64 - IDX_BITS)),
            "config key must leave room for the element index: {:#x}",
            key
        );
        // Same counts, different pending -> different key; equal -> equal.
        let mut d = c;
        assert_eq!(c.key(), d.key());
        d.inflight = match c.inflight {
            None => Some(0),
            Some(p) if p + 1 < NSLOT as u8 => Some(p + 1),
            Some(_) => None,
        };
        if c.inflight != d.inflight {
            assert_ne!(c.key(), d.key());
        }
    }
    // Explicit extreme: every slot at its maximum with a pending coin.
    let mut c = Config::default();
    for k in 0..NSLOT {
        c.hand[k] = 3;
        c.fd[k] = 5;
    }
    c.inflight = Some(NSLOT as u8 - 1);
    assert!(c.key() < (1u64 << (64 - IDX_BITS)));
    // The hand width is two bits; `key`'s debug_assert is the overflow test
    // for a slot value of 4 or more (it fires in every debug test build).
}

/// The reachable-config census, re-run with the Warrior Priests in the draft
/// pool: how big the belief supports get. Reported, not asserted — the
/// numbers feed the docs, and the solver's sizing depends on them.
#[test]
fn reachable_config_census_with_warrior_priests() {
    let mut sizes: Vec<usize> = Vec::new();
    let mut rng = Rng::new(0xC0FFEE);
    for g in 0..200u64 {
        let mut s = make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        for _ in 0..300 {
            if s.is_terminal() {
                break;
            }
            for p in 0..2u8 {
                let res = reserve(&s, p, &ctx);
                let truth = true_config(&s, p, &ctx);
                sizes.push(
                    enumerate_configs(
                        &res,
                        truth.hand_size(),
                        truth.fd_size(),
                        truth.inflight.is_some(),
                    )
                    .len(),
                );
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    sizes.sort_unstable();
    let n = sizes.len();
    let (med, p99) = (sizes[n / 2], sizes[(n as f64 * 0.99) as usize]);
    let mean = sizes.iter().sum::<usize>() as f64 / n as f64;
    eprintln!(
        "census: {} positions, median {} mean {:.1} p99 {}",
        n, med, mean, p99
    );
}

/// A subgame root survives the roots-file round trip: state, both beliefs
/// and every continuation. The GPU sizing tools read this format.
#[test]
fn roots_round_trip() {
    for seed in 0..30u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut s = make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        let mut bel = [
            Belief::point(Config::default()),
            Belief::point(Config::default()),
        ];
        for _ in 0..40 {
            if s.is_terminal() {
                break;
            }
            if matches!(s.pending(), Cont::MainPlay) {
                let mut buf: Vec<u8> = Vec::new();
                let mut w = std::io::Cursor::new(&mut buf);
                warchest::roots::write_root(&mut w, &s, &bel).unwrap();
                let mut r = std::io::Cursor::new(&buf);
                let (s2, bel2) = warchest::roots::read_root(&mut r).unwrap();
                assert_eq!(s, s2, "seed {}: state did not round trip", seed);
                assert_eq!(bel[0].cfg, bel2[0].cfg);
                assert_eq!(bel[0].p, bel2[0].p);
                assert_eq!(bel[1].cfg, bel2[1].cfg);
                assert_eq!(bel[1].p, bel2[1].p);
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
            let p = s.to_act();
            if s.is_chance() {
                let res = reserve(&s, p, &ctx);
                let fu = faceup_counts(&s, p, &ctx);
                let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
                bel[p as usize] = belief_after_draw(&bel[p as usize], &res, &fu, wp);
            } else {
                let truth = true_config(&s, p, &ctx);
                bel[p as usize] = Belief::point(truth);
            }
        }
    }
}

/// A packed replay row must expand to exactly the features the solver's own
/// encoder writes for the same state — the two producers of the public
/// encoding share one core, and this pins that they cannot drift. Walked
/// over random positions from random drafts (Warrior Priests included).
#[test]
fn packed_row_expands_to_the_same_features() {
    for seed in 0..40u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut s = make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        let mut checked = 0;
        for _ in 0..120 {
            if s.is_terminal() {
                break;
            }
            if matches!(s.pending(), Cont::MainPlay) {
                let mut row = [0u8; ROW_BYTES];
                pack_row(&s, &ctx, &mut row);
                let mut hs = [0u8; 2];
                let mut fd = [0u8; 2];
                let mut bg = [0u8; 2];
                for p in 0..2usize {
                    let res = reserve(&s, p as u8, &ctx);
                    let truth = true_config(&s, p as u8, &ctx);
                    hs[p] = truth.hand_size();
                    fd[p] = truth.fd_size();
                    let mut bag = 0u8;
                    for k in 0..NSLOT {
                        bag += res[k] - truth.hand[k] - truth.fd[k];
                    }
                    bg[p] = bag;
                }
                let mut direct = vec![0.0f32; PUBFEAT];
                write_public_features(&s, &ctx, &mut direct);
                let mut expanded = vec![0.0f32; PUBFEAT];
                expand_row(&row, &hs, &fd, &bg, &mut expanded);
                assert_eq!(
                    direct, expanded,
                    "seed {}: packed row and live state encode differently",
                    seed
                );
                checked += 1;
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
        assert!(
            checked > 20,
            "seed {} exercised too few MainPlay states",
            seed
        );
    }
}

// ------------------------------------------------- the value function's argument

/// Two different private states must never share a feature vector. This is the
/// property that makes `v(PBS, config)` a function *of the config*: without it
/// the network is being asked about an equivalence class, which is what the
/// hand-keyed encoding was and why it changed the game being solved.
#[test]
fn config_features_separate_every_config() {
    let mut seen: HashMap<Vec<u32>, Config> = HashMap::new();
    let mut checked = 0usize;
    for seed in 0..300u64 {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let reserve: [u8; NSLOT] = std::array::from_fn(|_| r.below(6) as u8);
        for hand_size in 0..=HAND_CAP as u8 {
            for fd_size in 0..4u8 {
                let cfgs = enumerate_configs(&reserve, hand_size, fd_size, false);
                seen.clear();
                for c in &cfgs {
                    for p in 0..2usize {
                        let mut phi = vec![0.0f32; CFEAT];
                        write_config_feats(c, &reserve, &mut phi);
                        // Bit patterns, so this compares exactly rather than
                        // up to a tolerance chosen to make it pass.
                        let key: Vec<u32> = phi.iter().map(|x| x.to_bits()).collect();
                        if let Some(prev) = seen.insert(key, *c) {
                            assert_eq!(prev, *c, "two configs share a feature vector");
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "only {checked} config vectors exercised");
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
    let n = cfgs.len().max(1) as f32;
    Belief {
        p: vec![1.0 / n; cfgs.len()],
        cfg: cfgs,
    }
}

/// A subgame close enough to the horizon that every leaf is terminal has no
/// network rows at all. The card table is only built when there is something to
/// encode, so anything hoisted out of the per-row loop must not read it.
///
/// The solver oracle already reaches these positions, but with an empty network,
/// which returns before the batch is touched. This one uses a real one -- the
/// combination a benchmark hit and the suite did not.
#[test]
fn a_subgame_of_only_terminal_leaves_solves() {
    let nets = Nets {
        value: random_net(5),
    };
    let mut checked = 0usize;
    for seed in 0..600u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut s = make_game(&mut rng, false);
        for _ in 0..60 + seed % 100 {
            if s.is_terminal() {
                break;
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
        if s.is_terminal() || s.is_chance() {
            continue;
        }
        // One coin play from the horizon: every child is terminal.
        s.main_plays = warchest::state::MAX_MAIN_PLAYS - 1;
        let ctx = Ctx::new(&s);
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
        let cfg = Cfg {
            depth: 2,
            iters: 8,
            snapshots: true,
            keep_states: true,
            ..Default::default()
        };
        let mut sv = Solver::new(&s, ctx, &nets, cfg, bel.clone());
        assert!(
            sv.nodes
                .iter()
                .zip(&sv.states)
                .all(|(n, st)| !n.leaf || st.is_terminal()),
            "expected every leaf terminal"
        );
        sv.multistep(cfg.iters);
        let v = sv.value_under(&[[bel[0].p.clone(), bel[1].p.clone()]]);
        assert!(v[0][0].iter().all(|x| x.is_finite()));
        checked += 1;
        if checked >= 3 {
            break;
        }
    }
    assert!(checked >= 3, "only {checked} positions exercised");
}

/// The card describer is the whole of what makes an unseen draft readable: the
/// network keys on what a card *does* rather than on which card it is. That only
/// works if the rulebook facts tell the draftable cards apart. If two share a
/// vector the describer merges them silently, and every part that reads the
/// embedding — the hex block, the pile summary, the holding tower, the action
/// tower — is built on the merge.
///
/// The Royal Coin is in play for both sides and is drafted by nobody, so it is
/// checked alongside the pool.
#[test]
fn card_features_separate_every_draftable_unit() {
    let mut seen: HashMap<Vec<u32>, u8> = HashMap::new();
    let units: Vec<u8> = warchest::selfplay::DRAFT_POOL
        .iter()
        .filter_map(|&id| warchest::units::index_of_id(id))
        .chain([warchest::units::ROYAL_COIN])
        .collect();
    assert_eq!(
        units.len(),
        20,
        "the draft pool did not resolve to unit indices"
    );
    for u in units {
        let mut f = vec![0.0f32; CARD_FEATS];
        write_card_features(u, &mut f);
        let key: Vec<u32> = f.iter().map(|x| x.to_bits()).collect();
        if let Some(prev) = seen.insert(key, u) {
            assert_eq!(
                prev, u,
                "units {} and {} share a card vector, so the describer cannot \
                 tell them apart",
                prev, u
            );
        }
    }
}

/// The counts must be the ones the name says, and the bag must be the derived
/// one. A transposition here would be invisible to every other test: the
/// network would happily learn whatever permutation it was given.
#[test]
fn config_counts_are_hand_facedown_bag() {
    let reserve = [4u8, 3, 5, 2, 1];
    let c = Config {
        hand: [1, 0, 2, 0, 0],
        fd: [2, 1, 0, 0, 1],
        inflight: None,
    };
    let mut cnt = [0u8; CCOUNTS];
    config_counts(&c, &reserve, &mut cnt);
    assert_eq!(&cnt[..NSLOT], &c.hand, "hand block");
    assert_eq!(&cnt[NSLOT..2 * NSLOT], &c.fd, "face-down block");
    assert_eq!(&cnt[2 * NSLOT..], &[1u8, 2, 3, 2, 0], "bag block");
    let mut phi = vec![0.0f32; CFEAT];
    write_config_feats(&c, &reserve, &mut phi);
    for k in 0..CCOUNTS {
        assert_eq!(phi[k], cnt[k] as f32 / CNORM);
    }
}

// ------------------------------------------------------ no leak through a solve

/// Find a mid-game decision node whose acting player has at least two configs
/// sharing a hand — the situation the whole rearchitecture is about.
fn position_with_ambiguous_facedown(seed: u64) -> Option<(State, Ctx, [Belief; 2])> {
    let mut rng = Rng::new(seed);
    let mut s = make_game(&mut Rng::new(seed), false);
    for _ in 0..40 + rng.below(60) {
        if s.is_terminal() {
            return None;
        }
        let acts = s.legal_actions();
        if acts.is_empty() {
            return None;
        }
        s.apply_inplace(acts[rng.below(acts.len())]);
    }
    while !s.is_terminal() && s.is_chance() {
        let acts = s.legal_actions();
        s.apply_inplace(acts[rng.below(acts.len())]);
    }
    if s.is_terminal() || s.is_chance() || !matches!(s.pending(), Cont::MainPlay) {
        return None;
    }
    let ctx = Ctx::new(&s);
    let mut bel = Vec::new();
    for p in 0..2u8 {
        let res = reserve(&s, p, &ctx);
        let truth = true_config(&s, p, &ctx);
        let cfg = enumerate_configs(
            &res,
            truth.hand_size(),
            truth.fd_size(),
            truth.inflight.is_some(),
        );
        if cfg.is_empty() {
            return None;
        }
        let w = 1.0 / cfg.len() as f32;
        bel.push(Belief {
            p: vec![w; cfg.len()],
            cfg,
        });
    }
    let me = s.to_act() as usize;
    let mut hands: HashMap<[u8; NSLOT], usize> = HashMap::new();
    for c in &bel[me].cfg {
        *hands.entry(c.hand).or_insert(0) += 1;
    }
    if !hands.values().any(|&n| n > 1) {
        return None;
    }
    Some((s, ctx, [bel[0].clone(), bel[1].clone()]))
}

/// A solve must be a function of the public state and the beliefs — nothing
/// else. The tree is built from a `State` that still carries somebody's true
/// hidden coins, so "the solver never looks at them" is a property worth
/// checking rather than assuming: instantiate the same public position in two
/// different worlds and require the results to agree bit for bit.
#[test]
fn a_solve_reads_only_the_beliefs() {
    let mut nets = Nets::default();
    nets.value = random_net(0xA11CE);
    let cfg = Cfg {
        depth: 2,
        iters: 8,
        snapshots: true,
        ..Default::default()
    };
    let mut checked = 0usize;
    for seed in 1..80u64 {
        let Some((s, ctx, bel)) = position_with_ambiguous_facedown(seed) else {
            continue;
        };
        // Two worlds with the same public projection: the first and the last
        // config in each player's support.
        let mut runs = Vec::new();
        for pick in [0usize, 1] {
            let mut w = s.clone();
            for p in 0..2usize {
                let cs = &bel[p].cfg;
                let c = if pick == 0 { cs[0] } else { cs[cs.len() - 1] };
                set_config(&mut w, p as u8, &ctx, &c);
            }
            let mut sv = Solver::new(&w, ctx, &nets, cfg, bel.clone());
            sv.multistep(cfg.iters);
            let vals = sv.value_under(&[[bel[0].p.clone(), bel[1].p.clone()]]);
            let strat: Vec<f32> = (0..bel[w.to_act() as usize].cfg.len())
                .flat_map(|c| sv.average_strategy(0, c).to_vec())
                .collect();
            runs.push((vals[0][0].clone(), vals[0][1].clone(), strat));
        }
        assert_eq!(
            runs[0].0, runs[1].0,
            "player 0 root values moved with the true world"
        );
        assert_eq!(
            runs[0].1, runs[1].1,
            "player 1 root values moved with the true world"
        );
        assert_eq!(
            runs[0].2, runs[1].2,
            "the root strategy moved with the true world"
        );
        checked += 1;
    }
    assert!(checked > 20, "only {checked} positions exercised");
}

/// The regression test for the thing this architecture exists to fix.
///
/// Two configs that share a hand but hold different face-down piles have
/// different bags and therefore different futures. They must get different
/// values, and because CFR derives its strategy from those values, different
/// play. Under the old hand-keyed head both were identically zero-difference by
/// construction — the network could not express the distinction and the solver
/// could not act on it.
#[test]
fn the_value_function_separates_configs_sharing_a_hand() {
    let mut nets = Nets::default();
    nets.value = random_net(0xBEEF);
    let cfg = Cfg {
        depth: 2,
        iters: 8,
        snapshots: true,
        ..Default::default()
    };
    let (mut positions, mut val_differs, mut strat_differs) = (0usize, 0usize, 0usize);
    for seed in 1..80u64 {
        let Some((s, ctx, bel)) = position_with_ambiguous_facedown(seed) else {
            continue;
        };
        let me = s.to_act() as usize;
        let mut sv = Solver::new(&s, ctx, &nets, cfg, bel.clone());
        sv.multistep(cfg.iters);
        let vals = sv.value_under(&[[bel[0].p.clone(), bel[1].p.clone()]]);
        let v = &vals[0][me];
        for i in 0..bel[me].cfg.len() {
            for j in 0..i {
                if bel[me].cfg[i].hand != bel[me].cfg[j].hand {
                    continue;
                }
                positions += 1;
                if (v[i] - v[j]).abs() > 1e-9 {
                    val_differs += 1;
                }
                let (a, b) = (sv.average_strategy(0, i), sv.average_strategy(0, j));
                if a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-9) {
                    strat_differs += 1;
                }
            }
        }
    }
    assert!(positions > 50, "only {positions} same-hand pairs found");
    // Not every pair must differ — two face-down piles can leave the same bag
    // when the coins came from the same slot — but the overwhelming majority
    // must, and under the old architecture the count was exactly zero except
    // where a round-start draw happened to fall inside the horizon.
    let vf = val_differs as f64 / positions as f64;
    let sf = strat_differs as f64 / positions as f64;
    assert!(
        vf > 0.9,
        "only {:.0}% of same-hand config pairs got distinct values",
        vf * 100.0
    );
    assert!(
        sf > 0.5,
        "only {:.0}% of same-hand config pairs got distinct play",
        sf * 100.0
    );
}

/// `normalize_weights` is what turns a reach vector into the belief the network
/// reads, and it has to agree with `Belief::normalize` — including the fallback
/// to uniform when the reaches have underflowed to zero, which is where a
/// mismatch would silently produce a different query than the trainer saw.
#[test]
fn normalized_weights_match_belief_normalize() {
    let mut rng = Rng::new(0xB33F);
    let mut checked = 0usize;
    for seed in 0..400u64 {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9));
        let reserve: [u8; NSLOT] = std::array::from_fn(|_| r.below(6) as u8);
        let cfgs = enumerate_configs(&reserve, r.below(4) as u8, r.below(4) as u8, false);
        if cfgs.is_empty() {
            continue;
        }
        let w: Vec<f32> = if seed % 17 == 0 {
            vec![0.0; cfgs.len()]
        } else {
            cfgs.iter().map(|_| rng.unit_f64() as f32).collect()
        };
        let mut got = vec![0.0f32; cfgs.len()];
        normalize_weights(&w, &mut got);
        let mut bel = Belief {
            cfg: cfgs.clone(),
            p: w.clone(),
        };
        bel.normalize();
        for (a, b) in got.iter().zip(bel.p.iter()) {
            assert!((a - b).abs() < 2e-7, "{a} vs {b}");
        }
        checked += 1;
    }
    assert!(checked > 100, "only {checked} weight vectors exercised");
}

#[test]
fn from_pairs_keeps_zero_weight_configs() {
    // Support is reachability, never weight. A belief weight is a product of
    // one strategy probability per decision; regret matching floors those at
    // 1e-6, so after enough of them a reachable config's weight reaches
    // exactly 0.0 in f32. Dropping it would shift every later strategy row
    // index by one — the walk-desync panic that killed
    // runs/t256_h384_dg64_s12 at epoch 168.
    let a = Config::default();
    let b = Config {
        hand: [1, 0, 0, 0, 0],
        fd: [0; NSLOT],
        inflight: None,
    };
    let c = Config {
        hand: [0; NSLOT],
        fd: [1, 0, 0, 0, 0],
        inflight: None,
    };
    let bel = Belief::from_pairs(vec![
        (b, 0.25),
        (c, -0.5), // a negative weight is still dropped
        (a, 0.0),  // underflowed to exactly zero: kept, in sorted position
    ]);
    assert_eq!(
        bel.cfg,
        vec![a, b],
        "zero-weight config must stay in the support"
    );
    assert_eq!(bel.p[0], 0.0, "the kept config's weight stays exactly zero");
    assert_eq!(bel.p[1], 1.0, "kept configs are renormalized");
}

#[test]
fn zero_weight_config_survives_the_walk_update() {
    // The crash shape, end to end: the Bayes update multiplies a config's
    // prior by its strategy probability at every decision, and a config the
    // strategy keeps calling unlikely reaches exactly 0.0 in f32. The subgame
    // tree keeps every reachable config, so a belief that dropped the
    // underflowed one would no longer match the tree's config list and the
    // walk's support assertion would fire mid-run. Drive the update with a
    // prior small enough that every product underflows, then require the new
    // support to equal the tree child's config list element for element — the
    // invariant the desync assert protects: support is reachability, never
    // weight.
    let nets = Nets::default();
    let mut rng = Rng::new(777);
    for _ in 0..200 {
        let mut s = make_game(&mut rng, false);
        for _ in 0..40 + rng.below(60) {
            if s.is_terminal() {
                break;
            }
            let acts = s.legal_actions();
            if acts.is_empty() {
                break;
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
        while !s.is_terminal() && s.is_chance() {
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
        if s.is_terminal() || s.is_chance() || !matches!(s.pending(), Cont::MainPlay) {
            continue;
        }
        let ctx = Ctx::new(&s);
        let me = s.to_act() as usize;
        let mut bel = Vec::new();
        for p in 0..2u8 {
            let res = reserve(&s, p, &ctx);
            let truth = true_config(&s, p, &ctx);
            let cfg = enumerate_configs(
                &res,
                truth.hand_size(),
                truth.fd_size(),
                truth.inflight.is_some(),
            );
            if cfg.len() < 2 {
                break;
            }
            let w = 1.0 / cfg.len() as f32;
            bel.push(Belief {
                p: vec![w; cfg.len()],
                cfg,
            });
        }
        if bel.len() != 2 {
            continue;
        }
        let mut bel = [bel[0].clone(), bel[1].clone()];
        // Config 0 of the acting player gets a prior so small that every
        // product with a strategy probability underflows to exactly 0.0 in
        // f32 (denormals end near 1.4e-45).
        bel[me].p[0] = 1e-46;
        bel[me].normalize();
        let mut sv = Solver::new(
            &s,
            ctx,
            &nets,
            Cfg {
                depth: 2,
                iters: 8,
                snapshots: false,
                ..Default::default()
            },
            bel.clone(),
        );
        sv.multistep(8);
        let n0 = &sv.nodes[0];
        // An action the underflowed config can actually play, so the tree's
        // child support includes it.
        let Some(chosen) = n0
            .legal_row(1)
            .map(|cell| n0.legal_action[cell] as usize)
            .next()
        else {
            continue;
        };
        let child = n0.child[n0.obs_child[chosen]];
        if sv.nodes[child].chance {
            // A draw child's config list is the post-draw support, which this
            // pre-draw update does not model; keep searching.
            continue;
        }
        // The walk's Bayes update on the public observation of `chosen`.
        let obs = obs_key(&n0.acts[chosen]);
        let mut pairs = Vec::new();
        for (ci, c) in bel[me].cfg.iter().enumerate() {
            let row = sv.average_strategy(0, ci);
            for (cell, &p) in n0.legal_row(ci).zip(row) {
                let a = n0.legal_action[cell] as usize;
                if obs_key(&n0.acts[a]) != obs {
                    continue;
                }
                if let Some(n) = advance_config(c, n0.aslot[a], n0.fdown[a]) {
                    pairs.push((n, bel[me].p[ci] * p));
                }
            }
        }
        let new_bel = Belief::from_pairs(pairs);
        assert_eq!(
            &*sv.nodes[child].cfgs[me],
            &new_bel.cfg[..],
            "walk update dropped a reachable config (weight underflow); the tree keeps it, so the desync assert would fire"
        );
        return;
    }
    panic!("no usable position in 200 random games");
}
