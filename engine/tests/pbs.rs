use std::collections::HashMap;

use warchest::pbs::*;
use warchest::rng::Rng;
use warchest::net::Net;
use warchest::search::{node_actions, Cfg, Solver};
use std::sync::Arc;
use warchest::selfplay::make_game;
use warchest::state::{Cont, State, Z_BAG, Z_FACEDOWN, Z_FACEUP};
use warchest::units::{write_card_features, CARD_FEATS};
use warchest::Action;

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

#[allow(clippy::type_complexity)]
fn uniform_row(s: &State, ctx: &Ctx, p: u8, c: &Config) -> (Vec<Action>, Vec<i8>, Vec<bool>, Vec<f64>) {
    let (acts, aslot, fdown) = node_actions(s, p, ctx, std::slice::from_ref(c));
    let legal: Vec<bool> = aslot.iter().map(|&k| action_legal(c, k)).collect();
    let n = legal.iter().filter(|&&x| x).count() as f64;
    let probs = legal
        .iter()
        .map(|&l| if l && n > 0.0 { 1.0 / n } else { 0.0 })
        .collect();
    (acts, aslot, fdown, probs)
}

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
                let all = enumerate_configs(&res, truth.hand_size(), truth.fd_size(), truth.inflight.is_some());
                if all.len() < 2 {
                    continue;
                }
                let mut a = vec![0.0f32; PUBFEAT];
                let mut b = vec![0.0f32; PUBFEAT];
                let mut row = [0u8; ROW_BYTES];
                let mut sa = s.clone();
                set_config(&mut sa, p, &ctx, &all[0]);
                pack_row(&sa, &ctx, &mut row);
                expand_row(&row, &mut a);
                let mut sb = s.clone();
                set_config(&mut sb, p, &ctx, &all[all.len() - 1]);
                pack_row(&sb, &ctx, &mut row);
                expand_row(&row, &mut b);
                assert_eq!(a, b, "features changed when only player {}'s hidden config changed", p);
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
    for random_draft in [false, true] {
        let (checked, pending) = leak_check(random_draft);
        assert!(checked > 500, "not enough positions exercised: {}", checked);
        assert!(
            pending > 0,
            "the pending-maneuver mask never fired over {} positions",
            checked
        );
    }
}

#[test]
fn belief_tracker_matches_brute_force() {
    for seed in 0..6u64 {
        run_one_draft(seed, &[17, 12, 4, 9], &[1, 3, 8, 16]);
    }
    for seed in 0..4u64 {
        run_one_draft(seed + 100, &[18, 17, 12, 4], &[54, 1, 3, 8]);
    }
}

fn run_one_draft(seed: u64, white: &[u16], black: &[u16]) {
    let mut rng = Rng::new(seed + 1);
    let mut s = State::from_draft(white, black, warchest::state::WHITE);
    let ctx = Ctx::new(&s);
    let mut bel = [Belief::point(Config::default()), Belief::point(Config::default())];
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

        let truth = true_config(&s, p, &ctx);
        let (acts, _, _, probs) = uniform_row(&s, &ctx, p, &truth);
        let chosen = rng.weighted_index(&probs);
        let obs = obs_key(&acts[chosen]);

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

fn compare(worlds: &HashMap<(Config, Config), f64>, bel: &[Belief; 2], seed: u64, step: usize, what: &str) {
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

#[test]
fn config_key_packing_has_headroom() {
    for seed in 0..200u64 {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut c = Config::default();
        for k in 0..NSLOT {
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
    let mut c = Config::default();
    for k in 0..NSLOT {
        c.hand[k] = 3;
        c.fd[k] = 5;
    }
    c.inflight = Some(NSLOT as u8 - 1);
    assert!(c.key() < (1u64 << (64 - IDX_BITS)));
}

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
                    let mut phi = vec![0.0f32; CFEAT];
                    write_config_feats(c, &reserve, &mut phi);
                    let key: Vec<u32> = phi.iter().map(|x| x.to_bits()).collect();
                    if let Some(prev) = seen.insert(key, *c) {
                        assert_eq!(prev, *c, "two configs share a feature vector");
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 10_000, "only {checked} config vectors exercised");
}

#[test]
fn card_features_separate_every_draftable_unit() {
    let mut seen: HashMap<Vec<u32>, u8> = HashMap::new();
    let units: Vec<u8> = warchest::selfplay::DRAFT_POOL
        .iter()
        .filter_map(|&id| warchest::units::index_of_id(id))
        .chain([warchest::units::ROYAL_COIN])
        .collect();
    assert_eq!(units.len(), 20, "the draft pool did not resolve to unit indices");
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
        let mut bel = Belief { cfg: cfgs.clone(), p: w.clone(), };
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
    let bel = Belief::from_pairs(vec![(b, 0.25), (c, -0.5), (a, 0.0)]);
    assert_eq!(bel.cfg, vec![a, b], "zero-weight config must stay in the support");
    assert_eq!(bel.p[0], 0.0, "the kept config's weight stays exactly zero");
    assert_eq!(bel.p[1], 1.0, "kept configs are renormalized");
}

#[test]
fn a_mirrored_row_is_the_mirrored_state_packed() {
    let mut checked = 0usize;
    for seed in 0..40u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut s = make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        for _ in 0..120 {
            if s.is_terminal() {
                break;
            }
            if s.is_valued() {
                let (mut a, mut b, mut got) = ([0u8; ROW_BYTES], [0u8; ROW_BYTES], [0u8; ROW_BYTES]);
                pack_row(&s, &ctx, &mut a);
                pack_row(&s.mirror(), &ctx.mirrored(), &mut b);
                mirror_row(&a, &mut got);
                assert_eq!(got, b, "seed {seed}: mirror_row disagrees with State::mirror");
                let mut back = [0u8; ROW_BYTES];
                mirror_row(&got, &mut back);
                assert_eq!(back, a, "seed {seed}: mirror is not an involution");
                checked += 1;
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    assert!(checked > 500, "only {checked} rows mirrored");
}

#[test]
fn a_contract_describes_every_node_of_the_tree() {
    let nets = Arc::new(Net::default());
    let cfg = Cfg { s: 8, c: 1.0, ..Default::default() };
    let mut checked = 0usize;
    for seed in 1..90u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut s = make_game(&mut rng, true);
        for _ in 0..40 + seed % 60 {
            if s.is_terminal() {
                break;
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
        if s.is_terminal() || s.is_chance() || !matches!(s.pending(), Cont::MainPlay) {
            continue;
        }
        let ctx = Ctx::new(&s);
        let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
        let sv = Solver::new(&s, ctx, Arc::clone(&nets), cfg, bel, Rng::new(seed));
        let c = warchest::contract::Contract::of(&sv);
        assert_eq!(c.nodes(), sv.nodes.len(), "seed {seed}: node count");
        for i in 0..c.nodes() {
            assert_eq!(c.nc[i], sv.nc[i], "seed {seed} node {i}: config counts");
            assert_eq!(c.roff[i], sv.roff[i], "seed {seed} node {i}: reach offset");
            assert_eq!(c.voff[i], sv.voff[i], "seed {seed} node {i}: value offset");
            assert_eq!(c.soff[i], sv.soff[i], "seed {seed} node {i}: strategy offset");
        }
        assert!(c.levels() >= 1, "seed {seed}: no levels");
        checked += 1;
    }
    assert!(checked > 20, "only {checked} trees described");
}
