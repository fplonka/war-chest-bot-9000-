//! How wrong is the CFR value target at the iteration count self-play uses?
//!
//! Generation runs `T` alternating linear-CFR iterations per subgame. The
//! only evidence that any particular T is enough came from micro-endgames
//! solved against exact values, where mean |error| was 0.0035 at T=8 — but a
//! micro-endgame converges almost immediately, so that number says very
//! little about the ~540-node depth-2 subgames self-play actually solves.
//!
//! This matters in a way the offline loss work cannot see. An under-converged
//! solve does not produce *noisy* targets — the same position gives the same
//! number every time — it produces **biased** ones. A biased target function is
//! perfectly learnable, so the value network fits it happily and converges to
//! the fixed point of the under-solved operator. Nothing in a training loss
//! curve or a held-out fit would ever show it.
//!
//! So: take real mid-game positions, solve each once to `TMAX`, and read the
//! fixed-policy root value under the average strategy off at every
//! intermediate `T` (`value_under` — the exact quantity TurboReBeL's Phase 2
//! uses as its target). The difference between the reading at `T` and at
//! `TMAX` *is* the target error at that `T`. No training run, no noise.
//!
//! Beliefs here are uniform over the configs consistent with the public counts
//! rather than the true Bayes posterior. That keeps the harness small, and what
//! governs how hard a subgame is to solve — its tree shape and the size of the
//! belief support — is matched either way. The absolute values would shift
//! under the real posterior; the convergence behaviour is what is being
//! measured.
//!
//! `cargo run --release --example solvererr -- weights.bin [positions] [depth]`

use warchest::board::N_HEXES;
use warchest::net::Mlp;
use warchest::rebel::{enumerate_configs, reserve, true_config, Belief, Config, Ctx};
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{eval_static, make_game};
use warchest::state::{Cont, State};

/// The iteration counts to report, and the converged reference they are
/// measured against.
const LADDER: [usize; 6] = [2, 4, 8, 16, 32, 64];
const TMAX: usize = 512;

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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let want: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(200);
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

    let mut nets = Nets::default();
    nets.value = Mlp::load_bin(&path).expect("weights file");
    println!("dims {:?}, depth {depth}, reference T={TMAX}, positions from {} play,\n         sampled {}-{} plies in",
             nets.value.dims, if greedy { "greedy" } else { "random" }, skip, skip + 60);
    warchest::state::set_cap_marker_value(0.0);

    // err[i] accumulates |value(LADDER[i]) - value(TMAX)| over every config of
    // every sampled position; `spread` is the spread of the reference values
    // themselves, which is the scale the error has to be read against.
    let mut err = [0.0f64; LADDER.len()];
    // The *signed* error, which is the number that decides how much T matters.
    // A zero-mean error behaves like noise: it averages out through the
    // bootstrap and adds in quadrature with the network's own error, where
    // shrinking it below a few percent is invisible. A signed error is a bias
    // that compounds every time the operator is applied, and displaces the
    // fixed point by roughly bias/(1 - gamma) -- with gamma near 1 on an
    // undiscounted 256-ply horizon, that is an amplification, not a wash.
    let mut signed = [0.0f64; LADDER.len()];
    // Whether the sign is consistent *within* a position, which is what makes
    // it survive averaging over a subgame's configs.
    let mut same_sign = [0usize; LADDER.len()];
    let mut n = 0usize;
    let (mut ref_sum, mut ref_sq) = (0.0f64, 0.0f64);
    let mut support = 0usize;
    let mut positions = 0usize;

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
        let cfg = Cfg { depth, iters: TMAX, snapshots: true };
        let mut sv = Solver::new(&s, &ctx, &nets, cfg, bel.clone());

        // One solve, read off at each rung: `value_under` is the fixed-policy
        // root value under the average strategy run so far — exactly the
        // target TurboReBeL's Phase 2 would have taken had the solve stopped
        // there.
        let mut snap: Vec<Vec<f32>> = Vec::new();
        let mut done = 0usize;
        for t in LADDER {
            sv.multistep(t - done);
            done = t;
            let vals = sv.value_under(&[[bel[0].p.clone(), bel[1].p.clone()]]);
            snap.push(vals[0][0].clone());
        }
        sv.multistep(TMAX - done);
        let vals = sv.value_under(&[[bel[0].p.clone(), bel[1].p.clone()]]);
        let reference = vals[0][0].clone();

        for (i, v) in snap.iter().enumerate() {
            let mut pos = 0usize;
            for (a, b) in v.iter().zip(reference.iter()) {
                let d = (a - b) as f64;
                err[i] += d.abs();
                signed[i] += d;
                pos += (d > 0.0) as usize;
            }
            // Did this position's configs mostly err the same way?
            let k = reference.len().max(1);
            if pos * 4 >= k * 3 || pos * 4 <= k {
                same_sign[i] += 1;
            }
        }
        for r in &reference {
            ref_sum += *r as f64;
            ref_sq += (*r as f64) * (*r as f64);
        }
        n += reference.len();
        support += reference.len();
        positions += 1;
        if positions % 25 == 0 {
            println!("  ... {positions}/{want} positions");
        }
    }

    let mean = ref_sum / n as f64;
    let spread = (ref_sq / n as f64 - mean * mean).max(0.0).sqrt();
    println!(
        "\n{positions} positions, {:.1} configs each, reference value spread {spread:.4}",
        support as f64 / positions as f64
    );
    println!(
        "\n{:>6}  {:>12}  {:>10}  {:>13}  {:>12}",
        "T", "mean |err|", "vs spread", "signed mean", "one-sided"
    );
    for (i, t) in LADDER.iter().enumerate() {
        let e = err[i] / n as f64;
        let sg = signed[i] / n as f64;
        println!(
            "{t:>6}  {e:>12.5}  {:>9.1}%  {sg:>+13.5}  {:>11.0}%",
            100.0 * e / spread.max(1e-9),
            100.0 * same_sign[i] as f64 / positions as f64
        );
    }
    println!(
        "\n`signed mean` is the whole question. If it is far smaller than\n\
         `mean |err|`, the error is noise that averages out through the\n\
         bootstrap and T past ~16 buys nothing. If the two are comparable, it\n\
         is a bias that compounds and caps how good the value function can get.\n\
         `one-sided` is the share of positions whose configs mostly erred the\n\
         same way, which is what lets a bias survive averaging."
    );
    println!(
        "\nread: the target error at the T generation uses, against the spread of\n\
         the values themselves and against the value network's own held-out\n\
         error (~0.09). Raising T only helps if this is a comparable fraction.\n\
         (board is {N_HEXES} hexes; beliefs are uniform over consistent configs)"
    );
}
