//! How good is a solve, at the iteration count generation uses, under each
//! regret rule?
//!
//! Two questions, one harness, no training runs and no noise.
//!
//! **How far from solved?** `NashConv` — what a best response to the solve's
//! own average strategy would gain, summed over the players. Zero means the
//! strategy is an equilibrium of the subgame it induces, which is the fixed
//! point ReBeL's whole argument rests on. This is the number that compares two
//! *different* regret rules, because it is absolute: it does not grade an
//! algorithm against its own answer, which is what the previous version of this
//! tool did and why it could not have chosen between them.
//!
//! **How wrong is the target?** The value target is the root value under the
//! average strategy (`value_under`, exactly what TurboReBeL's Phase 2 records),
//! and its error is the distance to a converged reference. This is the quantity
//! the value network actually fits, and an under-converged solve does not make
//! it *noisy* — the same position gives the same number every time — it makes it
//! **biased**. A biased target function is perfectly learnable, so the network
//! fits it happily and converges to the fixed point of the under-solved
//! operator. No training loss curve would ever show that.
//!
//! Both are read off one solve per rule per position: the rungs are readings
//! taken as the solve passes them.
//!
//! Beliefs here are uniform over the configs consistent with the public counts
//! rather than the true Bayes posterior. That keeps the harness small, and what
//! governs how hard a subgame is to solve — its tree shape and the size of the
//! belief support — is matched either way. The absolute values would shift under
//! the real posterior; the convergence behaviour is what is being measured.
//!
//! `cargo run --release --example solvererr -- weights.bin [positions] [depth] [greedy|random] [skip]`

use warchest::board::N_HEXES;
use warchest::net::Mlp;
use warchest::rebel::{enumerate_configs, reserve, true_config, Belief, Config, Ctx};
use warchest::rng::Rng;
use warchest::search::{Cfg, Cfr, Nets, Solver};
use warchest::selfplay::{eval_static, make_game};
use warchest::state::{Cont, State};

/// The iteration counts to report, and the converged solve they are measured
/// against. The reference is the rule TurboReBeL itself runs, taken far past
/// every rung.
const LADDER: [usize; 7] = [4, 8, 16, 32, 64, 128, 256];
const TMAX: usize = 512;
const REFERENCE: Cfr = Cfr::DISCOUNTED;

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
    let cfg = enumerate_configs(&res, truth.hand_size(), truth.fd_size(), truth.inflight.is_some());
    let cfg = if cfg.is_empty() {
        vec![Config::default()]
    } else {
        cfg
    };
    let w = 1.0 / cfg.len() as f32;
    Belief {
        p: vec![w; cfg.len()],
        cfg,
    }
}

/// One rule's readings at every rung of one position, and at `TMAX`.
struct Run {
    /// Player 0's root value per config.
    vals: Vec<Vec<f32>>,
    nash: Vec<f64>,
    zero_sum: Vec<f64>,
}

/// One solve, read off at each rung as it passes: the fixed-policy passes
/// restore the solve's reaches, so a reading does not disturb what follows.
fn solve(
    s: &State,
    ctx: &Ctx,
    nets: &Nets,
    bel: &[Belief; 2],
    depth: usize,
    rule: Cfr,
    warm: f32,
) -> Run {
    let cfg = Cfg {
        depth,
        iters: TMAX,
        snapshots: true,
        cfr: rule,
        warm,
        ..Default::default()
    };
    let mut sv = Solver::new(s, *ctx, nets, cfg, bel.clone());
    sv.warm_start(warm);
    let root = [[bel[0].p.clone(), bel[1].p.clone()]];
    let mut r = Run {
        vals: Vec::new(),
        nash: Vec::new(),
        zero_sum: Vec::new(),
    };
    let mut done = 0usize;
    for t in LADDER.iter().copied().chain([TMAX]) {
        sv.multistep(t - done);
        done = t;
        let c = sv.nash_conv();
        r.nash.push(c.nash as f64);
        r.zero_sum.push(c.zero_sum.abs() as f64);
        r.vals.push(sv.value_under(&root)[0][0].clone());
    }
    r
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let want: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(100);
    let depth: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(2);
    // How the sampled positions are reached. Greedy play drives coins onto the
    // board and drains the reserve, which collapses the belief support to a
    // handful of configs -- far from the ~24 a real ReBeL decision carries, and
    // a smaller support is an easier subgame. Random play keeps reserves full
    // and brackets the real distribution from the other side.
    let greedy = a.get(4).map(|x| x != "random").unwrap_or(true);
    // How deep into the game to sample. Games run ~250 plies, so the default
    // window covers the opening and early middlegame; a later window is what
    // checks that the finding is not an artefact of one phase.
    let skip: usize = a.get(5).and_then(|x| x.parse().ok()).unwrap_or(20);
    // Non-zero turns the sweep into A4's test: every rule is run cold and again
    // seeded from the policy head, worth this many iterations. The decision rule
    // is whether warm at T/2 beats cold at T.
    let warm: f32 = a.get(6).and_then(|x| x.parse().ok()).unwrap_or(0.0);

    let mut nets = Nets::default();
    nets.value = Mlp::load_bin(&path).expect("weights file");
    println!(
        "dims {:?}, depth {depth}, reference {REFERENCE:?} at T={TMAX},\n\
         positions from {} play, sampled {}-{} plies in, warm start {warm}",
        nets.value.dims,
        if greedy { "greedy" } else { "random" },
        skip,
        skip + 60
    );
    warchest::state::set_cap_marker_value(0.0);

    // With a warm start every rule appears twice, cold then warm, so the two
    // columns sit side by side in every table.
    let rules: Vec<(String, Cfr, f32)> = Cfr::NAMED
        .iter()
        .flat_map(|(n, c)| {
            let mut v = vec![(n.to_string(), *c, 0.0)];
            if warm > 0.0 {
                v.push((format!("{n}+warm"), *c, warm));
            }
            v
        })
        .collect();
    let rungs = LADDER.len() + 1;
    // Per rule, per rung, summed over every config of every position: the
    // target's absolute and signed error against the reference, and NashConv.
    let mut err = vec![vec![0.0f64; rungs]; rules.len()];
    let mut signed = vec![vec![0.0f64; rungs]; rules.len()];
    let mut nash = vec![vec![0.0f64; rungs]; rules.len()];
    let mut asym = vec![vec![0.0f64; rungs]; rules.len()];
    let (mut n, mut positions, mut support) = (0usize, 0usize, 0usize);
    let (mut ref_sum, mut ref_sq) = (0.0f64, 0.0f64);

    let mut game = 0u64;
    while positions < want {
        game += 1;
        let mut rng = Rng::new(game * 6_364_136_223 + 11);
        let mut s = make_game(&mut Rng::new(game), false);
        // Skip the opening: the first plies are near-identical across games and
        // their subgames are trivial.
        for _ in 0..rng.below(60) + skip {
            if s.is_terminal() {
                break;
            }
            step(&mut s, &mut rng, greedy);
        }
        // Advance to a decision node the solver can root at.
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

        let runs: Vec<Run> = rules
            .iter()
            .map(|(_, r, w)| solve(&s, &ctx, &nets, &bel, depth, *r, *w))
            .collect();
        // Every rule is graded against the same numbers, which is what makes
        // the columns comparable.
        let ri = rules
            .iter()
            .position(|(_, r, w)| *r == REFERENCE && *w == 0.0)
            .unwrap();
        let reference = runs[ri].vals[rungs - 1].clone();

        for (k, run) in runs.iter().enumerate() {
            for t in 0..rungs {
                nash[k][t] += run.nash[t];
                asym[k][t] += run.zero_sum[t];
                for (a, b) in run.vals[t].iter().zip(reference.iter()) {
                    let d = (a - b) as f64;
                    err[k][t] += d.abs();
                    signed[k][t] += d;
                }
            }
        }
        for r in &reference {
            ref_sum += *r as f64;
            ref_sq += (*r as f64) * (*r as f64);
        }
        n += reference.len();
        support += reference.len();
        positions += 1;
        if positions % 10 == 0 {
            println!("  ... {positions}/{want} positions");
        }
    }

    let mean = ref_sum / n as f64;
    let spread = (ref_sq / n as f64 - mean * mean).max(0.0).sqrt();
    println!(
        "\n{positions} positions, {:.1} configs each, reference value spread {spread:.4}",
        support as f64 / positions as f64
    );

    let table = |title: &str, acc: &[Vec<f64>], div: f64| {
        println!("\n{title}\n");
        print!("{:>7}", "T");
        for (name, _, _) in rules.iter() {
            print!("{name:>12}");
        }
        println!();
        for t in 0..rungs {
            print!("{:>7}", if t < LADDER.len() { LADDER[t] } else { TMAX });
            for a in acc.iter() {
                print!("{:>12.5}", a[t] / div);
            }
            println!();
        }
    };
    table(
        "NashConv — what a best response to the solve would gain.",
        &nash,
        positions as f64,
    );
    table("mean |target error| against the reference.", &err, n as f64);
    table("signed mean target error.", &signed, n as f64);
    table(
        "|v_0 + v_1| — how far the network is from antisymmetric.",
        &asym,
        positions as f64,
    );

    println!(
        "\nread: NashConv picks the regret rule — it is absolute, so the columns\n\
         compare. The target error says what the choice costs the value\n\
         network: judge it against the spread of the values themselves\n\
         ({spread:.4}) and against the network's own held-out error (~0.09).\n\
         A signed mean far below the absolute one is error that averages away\n\
         through the bootstrap; comparable, and it is a bias that compounds\n\
         every time the operator is applied.\n\
         The T={TMAX} row is zero for the reference rule by construction. What\n\
         it shows for the others is whether they agree at convergence, which\n\
         they must if the fixed point is unique.\n\
         The last table is a property of the *network*, not of the solve, and\n\
         should barely move across a row: the subgame is only as zero-sum as\n\
         the value network is antisymmetric, and CFR's guarantees assume it is.\n\
         (board is {N_HEXES} hexes; beliefs are uniform over consistent configs)"
    );
}
