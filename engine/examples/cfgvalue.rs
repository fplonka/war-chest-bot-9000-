//! How much does the value function separate configs that share a hand?
//!
//! Two configs with the same hand and different face-down piles have different
//! bags, and so different futures. The value function has to tell them apart:
//! if it cannot, the strategy CFR derives from it cannot either, and the agent
//! is playing a game in which players may not remember what they buried.
//!
//! Three quantities per sampled position, none of which needs a training run:
//!
//!   * **bag spread** -- within a group of configs sharing a hand, the
//!     belief-weighted RMS deviation of each config's bag composition from the
//!     group mean, in coins. This is the *information* at stake and it does not
//!     involve the network at all. If it were zero the hand would be a
//!     sufficient statistic and the question would not arise.
//!
//!   * **root strategy divergence** -- how often two configs sharing a hand get
//!     different action distributions out of the solve. This is the behaviour
//!     the information is supposed to buy.
//!
//!   * **value spread** -- the belief-weighted RMS deviation of each config's
//!     CFR root value from its group mean. Read it against the value network's
//!     own held-out error and against the spread of the targets.
//!
//! This example was written to measure what the previous architecture threw
//! away: values were keyed by hand, so the last two were zero by construction
//! except where a round-start draw happened to fall inside the horizon (8% of
//! same-hand pairs; the rest got bit-identical play). It is kept as the
//! instrument that says the current architecture does not.
//!
//! `cargo run --release --example cfgvalue -- weights.bin [positions] [iters]`

use warchest::net::Mlp;
use warchest::rebel::{enumerate_configs, reserve, true_config, Belief, Config, Ctx, NSLOT};
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{eval_static, make_game};
use warchest::state::{Cont, State, Z_HAND};

const DEPTHS: [usize; 4] = [1, 2, 3, 4];

/// One-ply greedy on the public evaluation, to reach realistic mid-game
/// positions. Chance nodes resolve uniformly over the listed draws.
fn step(s: &mut State, rng: &mut Rng, greedy: bool) {
    let acts = s.legal_actions();
    if acts.is_empty() {
        return;
    }
    if !greedy || matches!(s.pending(), Cont::Draw { .. }) {
        s.apply_inplace(acts[rng.below(acts.len())]);
        return;
    }
    let p = s.to_act();
    let (mut best, mut pick) = (f32::NEG_INFINITY, acts[0]);
    for a in &acts {
        let mut t = *s;
        t.apply_inplace(*a);
        let v = eval_static(&t, p);
        if v > best {
            (best, pick) = (v, *a);
        }
    }
    s.apply_inplace(pick);
}

/// Uniform belief over every config consistent with what is publicly visible.
fn open_belief(s: &State, ctx: &Ctx, p: u8) -> Belief {
    let res = reserve(s, p, ctx);
    let truth = true_config(s, p, ctx);
    let cfg = enumerate_configs(&res, truth.hand_size(), truth.fd_size());
    let cfg = if cfg.is_empty() { vec![Config::default()] } else { cfg };
    let w = 1.0 / cfg.len() as f32;
    Belief { p: vec![w; cfg.len()], cfg }
}

/// Belief-weighted RMS deviation of `v` within each group of configs sharing a
/// hand, plus the belief mass sitting in groups that hold more than one config.
fn within_hand(cfg: &[Config], w: &[f32], v: &[f32]) -> (f64, f64) {
    let mut acc: std::collections::BTreeMap<[u8; NSLOT], (f64, f64, usize)> = Default::default();
    for (i, c) in cfg.iter().enumerate() {
        let e = acc.entry(c.hand).or_insert((0.0, 0.0, 0));
        e.0 += w[i] as f64 * v[i] as f64;
        e.1 += w[i] as f64;
        e.2 += 1;
    }
    let (mut sq, mut tot, mut multi) = (0.0, 0.0, 0.0);
    for (i, c) in cfg.iter().enumerate() {
        let (num, den, n) = acc[&c.hand];
        let d = v[i] as f64 - num / den.max(1e-12);
        sq += w[i] as f64 * d * d;
        tot += w[i] as f64;
        if n > 1 {
            multi += w[i] as f64;
        }
    }
    ((sq / tot.max(1e-12)).sqrt(), multi / tot.max(1e-12))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let want: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(120);
    let iters: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(16);
    let greedy = a.get(4).map(|x| x != "random").unwrap_or(true);
    let skip: usize = a.get(5).and_then(|x| x.parse().ok()).unwrap_or(20);

    let mut nets = Nets::default();
    nets.value = Mlp::load_bin(&path).expect("weights file");
    warchest::state::set_cap_marker_value(0.0);
    println!(
        "dims {:?}, T={iters}, positions from {} play, sampled {}-{} plies in",
        nets.value.dims,
        if greedy { "greedy" } else { "random" },
        skip,
        skip + 60
    );

    // Per depth: value residual (all positions / only those spanning a draw),
    // and the count in each bucket.
    let mut vres = [0.0f64; DEPTHS.len()];
    let mut vres_draw = [0.0f64; DEPTHS.len()];
    let mut n_all = [0usize; DEPTHS.len()];
    let mut n_draw = [0usize; DEPTHS.len()];
    // Value scale, to read the residual against.
    let (mut ref_sum, mut ref_sq, mut ref_n) = (0.0f64, 0.0f64, 0usize);
    // The information at stake: bag composition spread within a hand group.
    let (mut bag_rms, mut bag_max, mut multi_mass) = (0.0f64, 0.0f64, 0.0f64);
    let (mut ncfg, mut nkey) = (0.0f64, 0.0f64);
    // Root-strategy divergence between configs that share a hand.
    let (mut pol_pairs, mut pol_diff) = (0usize, 0usize);
    let (mut pol_sum, mut pol_worst) = (0.0f64, 0.0f64);
    let mut positions = 0usize;

    let mut game = 0u64;
    while positions < want {
        game += 1;
        let mut rng = Rng::new(game * 6_364_136_223 + 11);
        let mut s = make_game(&mut Rng::new(game), false);
        for _ in 0..rng.below(60) + skip {
            if s.is_terminal() {
                break;
            }
            step(&mut s, &mut rng, greedy);
        }
        let mut guard = 0;
        while !s.is_terminal() && s.is_chance() && guard < 40 {
            step(&mut s, &mut rng, greedy);
            guard += 1;
        }
        if s.is_terminal() || s.is_chance() {
            continue;
        }
        let ctx = Ctx::new(&s);
        let bel = [open_belief(&s, &ctx, 0), open_belief(&s, &ctx, 1)];

        // ---- information dropped, for the player to act. No solve needed.
        let me = s.to_act() as usize;
        let res = reserve(&s, me as u8, &ctx);
        let b = &bel[me];
        let mut keys = std::collections::BTreeMap::new();
        for c in b.cfg.iter() {
            keys.entry(c.hand).or_insert_with(Vec::new).push(*c);
        }
        ncfg += b.cfg.len() as f64;
        nkey += keys.len() as f64;
        for slot in 0..NSLOT {
            let v: Vec<f32> = b.cfg.iter().map(|c| c.bag(&res)[slot] as f32).collect();
            let (r, m) = within_hand(&b.cfg, &b.p, &v);
            bag_rms += r * r;
            bag_max = bag_max.max(r);
            if slot == 0 {
                multi_mass += m;
            }
        }

        // ---- how many coin plays are left in the round, i.e. how deep a
        // subgame has to be before a round-start draw comes inside it.
        let plays_left: usize = (0..2)
            .map(|p| s.zones[p][Z_HAND].iter().map(|&x| x as usize).sum::<usize>())
            .sum();

        // ---- does the root STRATEGY differ between configs sharing a hand?
        // This is the quantity that matters, because it is the behaviour. Two
        // configs sharing a hand have identical legal action sets, so if they
        // also share every leaf value their regrets are identical and CFR hands
        // them the same distribution over actions -- the agent cannot act on
        // its own bag even though it knows it. Under the hand-keyed value
        // function that held everywhere except at a round-start draw.
        {
            let cfg = Cfg { depth: 2, iters, snapshots: true, ..Default::default() };
            let mut sv = Solver::new(&s, ctx, &nets, cfg, bel.clone());
            sv.multistep(iters);
            let mut worst = 0.0f64;
            for i in 0..bel[me].cfg.len() {
                for j in 0..i {
                    if bel[me].cfg[i].hand != bel[me].cfg[j].hand {
                        continue;
                    }
                    let (a, b) = (sv.average_strategy(0, i), sv.average_strategy(0, j));
                    let d: f64 = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| (*x - *y).abs() as f64)
                        .fold(0.0, f64::max);
                    worst = worst.max(d);
                    pol_pairs += 1;
                    pol_sum += d;
                    if d > 1e-4 {
                        pol_diff += 1;
                    }
                }
            }
            pol_worst = pol_worst.max(worst);
        }

        // ---- value residual at each depth.
        for (di, &d) in DEPTHS.iter().enumerate() {
            let cfg = Cfg { depth: d, iters, snapshots: true, ..Default::default() };
            let mut sv = Solver::new(&s, ctx, &nets, cfg, bel.clone());
            sv.multistep(iters);
            let vals = sv.value_under(&[[bel[0].p.clone(), bel[1].p.clone()]]);
            for p in 0..2 {
                let v = &vals[0][p];
                let (r, _) = within_hand(&bel[p].cfg, &bel[p].p, v);
                vres[di] += r;
                n_all[di] += 1;
                if plays_left <= d {
                    vres_draw[di] += r;
                    n_draw[di] += 1;
                }
                if di == 1 && p == 0 {
                    for (i, x) in v.iter().enumerate() {
                        let w = bel[p].p[i] as f64;
                        ref_sum += w * *x as f64;
                        ref_sq += w * (*x as f64) * (*x as f64);
                        ref_n += 1;
                    }
                }
            }
        }
        positions += 1;
        if positions % 20 == 0 {
            println!("  ... {positions}/{want}");
        }
    }

    let pf = positions as f64;
    // `ref_sum`/`ref_sq` are belief-weighted, so each position contributes
    // weight 1; the divisor is the position count, not the config count.
    let _ = ref_n;
    let mean = ref_sum / pf;
    let var = ref_sq / pf - mean * mean;
    println!("\n{positions} positions, {:.1} configs each over {:.1} distinct hands",
             ncfg / pf, nkey / pf);
    println!("value scale (depth 2): mean {mean:+.4}, spread {:.4}", var.max(0.0).sqrt());

    println!("\n-- information at stake: how much a hand fails to pin down --");
    println!("  belief mass in hands holding >1 config     : {:>7.1}%", 100.0 * multi_mass / pf);
    println!("  within-hand RMS bag deviation, per slot    : {:>7.4} coins",
             (bag_rms / pf / NSLOT as f64).sqrt());
    println!("  worst single slot seen                     : {bag_max:>7.4} coins");

    println!("\n-- root strategy, between configs sharing a hand (depth 2) --");
    println!("  same-hand config pairs compared             : {pol_pairs}");
    println!("  pairs whose action distributions differ     : {:>7.1}%",
             100.0 * pol_diff as f64 / pol_pairs.max(1) as f64);
    println!("  mean / worst max-abs difference            : {:.5} / {pol_worst:.5}",
             pol_sum / pol_pairs.max(1) as f64);

    println!("\n-- how far the value separates configs sharing a hand --");
    println!("{:>6}  {:>14}  {:>16}  {:>10}", "depth", "all positions", "spans a draw", "n(draw)");
    for (di, d) in DEPTHS.iter().enumerate() {
        let all = vres[di] / n_all[di].max(1) as f64;
        let dr = vres_draw[di] / n_draw[di].max(1) as f64;
        println!("{d:>6}  {all:>14.5}  {dr:>16.5}  {:>10}", n_draw[di]);
    }
    println!(
        "\nread: the last two blocks are what the hand-keyed architecture could not\n\
         produce. It scored 8% strategy divergence and a value spread that was\n\
         zero except where a round-start draw fell inside the horizon; anything\n\
         near those numbers means the config has stopped reaching the value."
    );
}
