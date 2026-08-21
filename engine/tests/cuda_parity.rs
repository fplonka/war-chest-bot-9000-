//! The device must solve as the CPU does.
//!
//! The whole search lives on the card now — the CFR loop, the readout, the
//! belief pooling and the policy head — so no single call has an answer the CPU
//! can produce on its own, and nothing but the sampled expansion leaves and the
//! final read crosses the bus at all. There are two ways to hold that to the
//! CPU network anyway.
//!
//! One is to ask a solve for the state it *keeps*. `Device::resident` copies a
//! solve's board vectors, its three config rows and its policy prior back off
//! the card, and every one of them has a CPU counterpart the same solver
//! computed on the host path. That is the call-by-call comparison, and it is
//! the only check the policy prior has: the prior steers expansion alone, so a
//! wrong one degrades the search silently and moves no target at all.
//!
//! The other is the solve itself. A fixed tree gives both backends the same
//! numbers to make, and the target a solve produces is downstream of every pass
//! it takes — the reach sweep, the join, the terminals, backpropagation, the
//! regret update and the value pass under the average — so a drift in any of
//! them lands there.
//!
//! Needs a GPU, so it only builds under `--features gpu`.
#![cfg(feature = "gpu")]

use std::sync::Arc;

use warchest::cuda::Device;
use warchest::farm::{Backend, Call, Reply};
use warchest::net::{Net, NetLayout};
use warchest::pbs::{enumerate_configs, reserve, true_config, Belief, Ctx};
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets, Solved, Solver, Step};
use warchest::selfplay::{make_game, Agent, Collect, Data, GameCfg, GameStream};
use warchest::state::State;

fn random_net(seed: u64) -> Net {
    let mut r = warchest::rng::Rng::new(seed);
    let l = NetLayout::new();
    let mut draw = |n: usize| -> Vec<f32> {
        (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect()
    };
    let (w, b) = (draw(l.w_len), draw(l.b_len));
    // Scales at one and shifts at zero, so the norms behave like real ones
    // rather than crushing the signal.
    let mut ln = vec![0.0; l.ln_len];
    for n in &l.norms {
        ln[n.g..n.g + n.width].fill(1.0);
    }
    Net::from_flat(&w, &b, &ln).expect("random net")
}

fn cfg(s: u32, c: f32) -> Cfg {
    // Not one. At `prior_temp = 1` the softmax the prior is formed at is the
    // identity in its temperature, so a backend that ignored the number
    // altogether would agree with one that applied it.
    Cfg { s, c, prior_temp: 1.7, ..Default::default() }
}

fn game_cfg(s: u32, c: f32) -> GameCfg {
    GameCfg {
        agents: [Agent::Sog { cfg: cfg(s, c) }; 2],
        collect: Collect::Sog,
        explore: 0.1,
        random_draft: true,
        p_td1: 0.0,
        // Root rows only, so a target is one solve's root value and a
        // divergence names the solve it came from.
        query_rate: 0.0,
        recursive_rate: 0.0,
    }
}

/// Run one game stream per `(seed, s)` against `backend`, every stream's calls
/// in the same round, and hand back what each produced.
///
/// A stream's games are a function of its seed alone, so the same seed run
/// alone and run beside others plays the same games and must produce the same
/// numbers. That is what makes batching testable.
fn generate(
    net: &Net,
    backend: Backend,
    streams: &[(u64, u32)],
    games: usize,
    c: f32,
) -> Vec<Data> {
    let nets = Arc::new(Nets { value: net.clone(), device: backend.keeps_the_solve() });
    let n = streams.len();
    let mut streams: Vec<GameStream> = streams
        .iter()
        .map(|&(seed, s)| GameStream::new(seed, game_cfg(s, c)))
        .collect();
    let mut out: Vec<Data> = (0..n).map(|_| Data::default()).collect();
    let mut live: Vec<Option<Solver>> = (0..n)
        .map(|i| {
            let mut sv = streams[i].next_solve(&nets, &mut out[i]);
            sv.pin(i);
            Some(sv)
        })
        .collect();
    let mut replies: Vec<Vec<Reply>> = (0..n).map(|_| Vec::new()).collect();
    while out.iter().any(|d| d.soff.len() < games) {
        let mut calls: Vec<Call> = Vec::new();
        let mut spans = vec![0usize; n];
        for i in 0..n {
            let Some(sv) = live[i].as_mut() else { continue };
            match sv.advance(&replies[i]) {
                Step::Calls(cs) => {
                    spans[i] = cs.len();
                    calls.extend(cs);
                }
                Step::Done(solved) => {
                    let sv = live[i].take().expect("a live solve");
                    streams[i].keep(&sv, solved, &mut out[i]);
                    if out[i].soff.len() < games {
                        let mut next = streams[i].next_solve(&nets, &mut out[i]);
                        next.pin(i);
                        live[i] = Some(next);
                    }
                }
            }
        }
        if calls.is_empty() {
            continue;
        }
        let mut rest = backend.run(&calls, 0).expect("the backend answered the round");
        for (i, k) in spans.into_iter().enumerate() {
            let tail = rest.split_off(k);
            replies[i] = rest;
            rest = tail;
        }
    }
    out
}

/// One stream, which is what a comparison against the CPU wants.
fn generate_one(net: &Net, backend: Backend, games: usize, s: u32, c: f32) -> Data {
    generate(net, backend, &[(0x51E5, s)], games, c)
        .pop()
        .expect("one stream")
}

/// One solve of the first position a stream reaches, run to the end on
/// `backend`, and the solver it leaves behind.
///
/// Pinned to slot zero of card zero, which is where `Device::resident` then
/// looks for it.
fn one_solve(net: &Net, backend: &Backend, s: u32, c: f32) -> Solver {
    let nets = Arc::new(Nets { value: net.clone(), device: backend.keeps_the_solve() });
    let mut data = Data::default();
    let sv = GameStream::new(0x51E5, game_cfg(s, c)).next_solve(&nets, &mut data);
    run_solve(backend, sv).0
}

/// Drive one solve to its end on `backend`, in slot zero of card zero.
fn run_solve(backend: &Backend, mut sv: Solver) -> (Solver, Option<Solved>) {
    sv.pin(0);
    let mut replies: Vec<Reply> = Vec::new();
    loop {
        match sv.advance(&replies) {
            Step::Calls(calls) => replies = backend.run(&calls, 0).expect("the backend answered"),
            Step::Done(solved) => return (sv, solved),
        }
    }
}

/// Largest relative difference, with an absolute floor so values near zero do
/// not dominate the ratio.
fn worst(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{what}: length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs() / (x.abs().max(y.abs()).max(1e-2)))
        .fold(0.0, f32::max)
}

/// A whole solve, both ways, on the same tree.
///
/// `c = 0` is what makes this a comparison. With it neither side grows, so both
/// solve the tree `Solver::new` built and every number is of the same thing.
/// With growth on they cannot be: the expansion phase samples, and the last
/// bits of a regret decide which leaf a trajectory takes — so the two trees
/// part company at the first such choice and never come back.
///
/// What is compared is the target a solve produces: the root's counterfactual
/// value for every config, which is the end of every path through the loop —
/// the reach sweep, the network at the leaves, the terminals, backpropagation,
/// the regret update, the average strategy and the value pass under it.
#[test]
fn the_cfr_loop_agrees_on_a_fixed_tree() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let host = generate_one(&net, Backend::Reference(net.clone()), 3, 8, 0.0);
    let card = generate_one(
        &net,
        Backend::Cuda(Device::new(&[0], net.clone()).expect("device")),
        3,
        8,
        0.0,
    );
    assert!(!host.cy.is_empty(), "the reference produced no targets");
    assert_eq!(
        host.cy.len(),
        card.cy.len(),
        "the two backends solved a different number of positions"
    );
    let bad = worst(&host.cy, &card.cy, "targets");
    let rel = |x: f32, y: f32| (x - y).abs() / (x.abs().max(y.abs()).max(1e-2));
    let off = host.cy.iter().zip(&card.cy).filter(|(&x, &y)| rel(x, y) > 1e-3).count();
    let first = host.cy.iter().zip(&card.cy).position(|(&x, &y)| rel(x, y) > 1e-3);
    let f = first.unwrap_or(0);
    let lo = f.saturating_sub(2);
    let hi = (f + 6).min(host.cy.len());
    eprintln!("first differing target {first:?} of {}", host.cy.len());
    eprintln!("  host {:?}", &host.cy[lo..hi]);
    eprintln!("  card {:?}", &card.cy[lo..hi]);
    eprintln!("  config offsets {:?}", &host.coff[..8.min(host.coff.len())]);
    let pbad = worst(&host.pprob, &card.pprob, "policy");
    eprintln!("  worst policy difference {pbad:e} over {} cells", host.pprob.len());
    eprintln!(
        "worst {bad:e}; {off} of {} targets differ; first few {:?} vs {:?}",
        host.cy.len(),
        &host.cy[..8.min(host.cy.len())],
        &card.cy[..8.min(card.cy.len())],
    );
    // The same bound the shared-round test holds a target to, and for the same
    // reason: a target is an average of counterfactual values over a fixed
    // tree, so the iterations damp the network's own f32 disagreement rather
    // than amplifying it. Measured, the worst is 4e-6 against values near 2.5.
    assert!(bad < 1e-4, "worst target difference {bad:e}");
}

/// With growth on, the device must still produce a sane solve.
///
/// The trees differ, so the numbers cannot be compared; what can be is that
/// every target is a finite value inside the game's range, and that the run
/// produced as many of them as the reference did positions.
#[test]
fn growth_on_the_device_produces_sane_targets() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let card = generate_one(
        &net,
        Backend::Cuda(Device::new(&[0], net.clone()).expect("device")),
        3,
        32,
        4.0,
    );
    let host = generate_one(&net, Backend::Reference(net.clone()), 3, 32, 4.0);
    assert!(!card.cy.is_empty(), "the device produced no targets");
    assert!(card.cy.iter().all(|v| v.is_finite()), "a target is not finite");
    // The trees differ, so the numbers do; the *scale* must not. A run whose
    // regrets or reaches were carried over from the solve before would blow up
    // here long before it produced a plausible spread.
    let scale = |d: &[f32]| d.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (a, b) = (scale(&host.cy), scale(&card.cy));
    assert!(b < 2.0 * a, "device targets reach {b} against the reference's {a}");
}

/// A solve must not depend on which other solves shared its rounds.
///
/// This is the one thing the tests above cannot see. They run a single stream,
/// so every round holds one solve and everything the batch carries is that
/// solve's own. A real run holds thirty-odd, each at a different point of its
/// own iterations, and anything the device reads from *the batch* where it
/// should read it from *the solve* is wrong for every member but one — silently,
/// and only in the shape a run actually has.
///
/// The streams are given different iteration counts so their step counts drift
/// apart; with equal counts the gate keeps them in lockstep and the same
/// mistake reads as correct. `c = 0` fixes the trees, so a stream's numbers are
/// a function of its seed alone.
#[test]
fn a_solve_does_not_depend_on_the_round_it_rides_in() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let streams = [(0x51E5u64, 8u32), (0x0A13, 11), (0x77C1, 13), (0x2E57, 17)];
    let device = || Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let together = generate(&net, device(), &streams, 3, 0.0);
    // A shared round must not move a solve at all, so the same run twice is
    // the control: whatever this reports is the floor the comparison sits on.
    let twice = generate(&net, device(), &streams, 3, 0.0);
    let rel = |x: f32, y: f32| (x - y).abs() / (x.abs().max(y.abs()).max(1e-2));
    let count = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(&x, &y)| rel(x, y) > 1e-3).count();
    let mut bad = 0.0f32;
    for (i, &s) in streams.iter().enumerate() {
        let alone = generate(&net, device(), &[s], 3, 0.0).pop().expect("one stream");
        assert_eq!(
            alone.cy.len(),
            together[i].cy.len(),
            "stream {i} solved a different number of positions in a shared round"
        );
        let (t, p) = (
            worst(&alone.cy, &together[i].cy, "targets"),
            worst(&alone.pprob, &together[i].pprob, "policy"),
        );
        eprintln!(
            "stream {i} iters={}: targets {t:e} ({} of {} differ)  policy {p:e} ({} of {} differ)  \
             repeat {:e}/{:e}",
            s.1,
            count(&alone.cy, &together[i].cy), alone.cy.len(),
            count(&alone.pprob, &together[i].pprob), alone.pprob.len(),
            worst(&twice[i].cy, &together[i].cy, "targets"),
            worst(&twice[i].pprob, &together[i].pprob, "policy"),
        );
        assert!(t < 1e-4, "stream {i}: sharing a round moved its targets by {t:e}");
        bad = bad.max(p);
    }
    // The policy tolerance is loose, and deliberately so. A round of four
    // solves and a round of one give the leaf pass different GEMM shapes, so
    // cuBLAS sums in a different order; regret matching then turns a 1e-7
    // difference in an accumulated regret into a visible difference in the
    // strategy at a cell whose regrets are near zero. Running the same four
    // streams with *matched* iteration counts gives the same 2.9e-2, so this
    // is arithmetic order and not a step count read from the wrong solve --
    // and the targets above, which is what a run trains on, are unmoved.
    assert!(bad < 5e-2, "sharing a round moved a solve's policy by {bad:e}");
}


/// Every array a solve keeps on the card, against the CPU network that makes
/// the same ones on the host path.
///
/// `c = 0` fixes the tree, so both solvers build the same nodes in the same
/// order and hold their arrays in the same layout. The two are then the same
/// arithmetic twice: `k_trunk` and the board head against `Net::board`, the
/// config encoder against `Net::configs`, and the action encoder with
/// `k_prior` against `Solver::refresh_priors`.
///
/// The prior is the reason this test exists. It is read by the expansion phase
/// and by nothing else, so a wrong one picks worse leaves to grow and leaves
/// every target, every policy and every belief looking exactly as it should.
///
/// The bound is 2e-4, and it is what a forward pass can honestly be held to.
/// None of these arrays is behind a loop -- each is one pass over the weights
/// -- so the only thing that separates the two backends is the order f32 sums
/// are accumulated in: cuBLAS against `Lin::run`, and a warp reduction against
/// a serial dot. Measured, that is 5.5e-5 at worst, in the join cache, which
/// has the longest chain of sums; the prior, which is a softmax over a handful
/// of dots, is 2.6e-6. Four times the worst leaves room for another card's
/// cuBLAS picking a different algorithm and none for a real drift.
#[test]
fn the_resident_state_agrees_with_the_cpu_network() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let host = one_solve(&net, &Backend::Reference(net.clone()), 8, 0.0);
    let device = Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let card = one_solve(&net, &device, 8, 0.0);
    let Backend::Cuda(d) = &device else { unreachable!("just built") };
    let got = d.resident(0, 0).expect("the card gave its solve back");

    assert!(!host.pb.is_empty(), "the reference solve made no board vectors");
    assert!(host.ncfg > 0, "the reference solve made no config rows");
    assert_eq!(host.ncells, card.ncells, "the two backends built different trees");
    let cells = host.ncells;
    assert!(cells > 0, "the fixed tree has no strategy cells");
    for (what, h, c) in [
        ("board vectors", &host.pb[..], &got.p[..]),
        ("join cache", &host.jp[..], &got.jp[..]),
        ("f(c)", &host.cf[..], &got.f[..]),
        ("g(c)", &host.cg[..], &got.g[..]),
        ("f_p(c)", &host.cp[..], &got.fp[..]),
        ("prior", &host.prior[..cells], &got.prior[..cells]),
    ] {
        let bad = worst(h, c, what);
        eprintln!("{what}: worst {bad:e} over {} values", h.len());
        assert!(bad < 2e-4, "{what} differ by {bad:e}");
    }
    // A prior that was never written would read as the uniform start the
    // scatter lays down, and would then agree with a host that had also never
    // written one. Both sides must actually be a policy.
    let uniform = got.prior[..cells].windows(2).all(|w| w[0] == w[1]);
    assert!(!uniform, "the card's prior is still the uniform start");
}


/// A subgame scored entirely from the game, on the card.
///
/// One coin play from the horizon, so every leaf below the root is terminal and
/// the only network row is the root's own. What the solve is then made of is
/// the terminal path -- the utilities, and the backpropagation that carries
/// them up through a chance node -- which the fixed-tree comparison barely
/// touches, because a mid-game subgame has few terminals and they are deep.
#[test]
fn a_subgame_scored_from_the_game_agrees_with_the_cpu() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let nets = Arc::new(Nets { value: net.clone(), device: true });
    let host_nets = Arc::new(Nets { value: net.clone(), device: false });
    let backend = Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let host = Backend::Reference(net.clone());
    let uniform = |s: &State, ctx: &Ctx, p: u8| {
        let truth = true_config(s, p, ctx);
        let cfg = enumerate_configs(
            &reserve(s, p, ctx),
            truth.hand_size(),
            truth.fd_size(),
            truth.inflight.is_some(),
        );
        let n = cfg.len().max(1) as f32;
        Belief { p: vec![1.0 / n; cfg.len()], cfg }
    };
    // An ordinary solve first, so slot zero holds another tree's rows and
    // arenas when this one takes it over.
    one_solve(&net, &backend, 8, 1.0);
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
        s.main_plays = warchest::state::MAX_MAIN_PLAYS - 1;
        let ctx = Ctx::new(&s);
        let bel = [uniform(&s, &ctx, 0), uniform(&s, &ctx, 1)];
        let mut sv = Solver::new(
            &s,
            ctx,
            Arc::clone(&nets),
            cfg(8, 1.0),
            bel.clone(),
            Rng::new(seed),
        );
        // The root itself carries a row -- it is a coin play, and a coin play
        // is where the network is defined. Everything under it is terminal.
        assert_eq!(sv.leaf_rows.len(), 1, "the subgame reaches the network more than once");
        sv.collect(0);
        let got = run_solve(&backend, sv).1.expect("a collected solve keeps a row");
        let mut want = Solver::new(
            &s,
            ctx,
            Arc::clone(&host_nets),
            cfg(8, 1.0),
            bel,
            Rng::new(seed),
        );
        want.collect(0);
        let want = run_solve(&host, want).1.expect("a collected solve keeps a row");
        for p in 0..2 {
            let bad = worst(&want.value[p], &got.value[p], "root value");
            assert!(bad < 1e-4, "player {p}'s root value differs by {bad:e}");
        }
        checked += 1;
        if checked >= 2 {
            return;
        }
    }
    panic!("only {checked} such positions were reached in 600 seeds");
}
