//! The device must solve as the CPU does.
//!
//! The whole CFR loop lives on the card now, so a round of calls no longer has
//! an answer the CPU can produce on its own: the reaches, the regrets and the
//! expansion trajectories never come back. What both backends can be asked for
//! is a *solve* — the same position, the same weights, the same seed — and that
//! is what this compares.
//!
//! Two levels. `the_network_agrees` still checks the trunk and the config
//! encoder call by call, because those cross the bus in both directions and a
//! drift there is arithmetic rather than search. `a_solve_agrees` runs real
//! self-play through each backend and compares what a solve produces: the
//! root's values, its policy, and the beliefs it harvests.
//!
//! Needs a GPU, so it only builds under `--features gpu`.
#![cfg(feature = "gpu")]

use std::sync::Arc;

use warchest::cuda::Device;
use warchest::farm::{Backend, Call, Reply};
use warchest::net::{Net, NetLayout};
use warchest::search::{Cfg, Nets, Solver, Step};
use warchest::selfplay::{Agent, Collect, Data, GameCfg, GameStream};

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
    Cfg { s, c, ..Default::default() }
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
///
/// `watch` sees every round before its replies are handed back, which is how
/// the network comparison below gets at the two calls that still cross the bus.
fn generate_with(
    net: &Net,
    backend: Backend,
    streams: &[(u64, u32)],
    games: usize,
    c: f32,
    mut watch: impl FnMut(&[Call], &[Reply]),
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
        let answered = backend.run(&calls, 0).expect("the backend answered the round");
        watch(&calls, &answered);
        let mut rest = answered;
        for (i, k) in spans.into_iter().enumerate() {
            let tail = rest.split_off(k);
            replies[i] = rest;
            rest = tail;
        }
    }
    out
}

fn generate(net: &Net, backend: Backend, streams: &[(u64, u32)], games: usize, c: f32) -> Vec<Data> {
    generate_with(net, backend, streams, games, c, |_, _| {})
}

/// One stream, which is what a comparison against the CPU wants.
fn generate_one(net: &Net, backend: Backend, games: usize, s: u32, c: f32) -> Data {
    generate(net, backend, &[(0x51E5, s)], games, c)
        .pop()
        .expect("one stream")
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

/// The trunk and the config encoder, call by call.
///
/// These are the two passes whose answers still cross the bus, so they can be
/// held to the CPU network directly. `Call::run` is that network.
#[test]
fn the_network_agrees() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let device = Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let (mut seen, mut bad) = ([0usize; 2], 0.0f32);
    generate_with(&net, device, &[(0x51E5, 32)], 4, 4.0, |calls, replies| {
        for (c, r) in calls.iter().zip(replies) {
            match c {
                Call::Trunk { .. } => {
                    seen[0] += 1;
                    bad = bad.max(worst(&c.run(&net).a, &r.a, "trunk"));
                }
                Call::Configs { .. } => {
                    seen[1] += 1;
                    bad = bad.max(worst(&c.run(&net).c, &r.c, "configs"));
                }
                _ => {}
            }
        }
    });
    assert!(seen.iter().all(|&n| n > 0), "saw {seen:?} of the two calls");
    assert!(bad < 1e-3, "worst network difference {bad:e}");
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
    assert!(bad < 1e-2, "worst target difference {bad:e}");
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
