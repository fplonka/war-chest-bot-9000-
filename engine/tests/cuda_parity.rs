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
//! Neither of those reaches the expansion phase, and a third way is needed for
//! it. Growth is a discrete function of the CFR arenas, so two backends whose
//! arenas differ in the last bits build different trees however faithfully they
//! copy each other's rule -- the comparison has to be of the rule, on one set
//! of numbers. `Device::resident` hands the arenas back and
//! `Solver::replay_expansion` runs the host's own trajectories against them.
//!
//! Needs a GPU, so it only builds under `--features gpu`.
#![cfg(feature = "gpu")]

use std::sync::Arc;

use warchest::contract::NO_ROW;
use warchest::cuda::Device;
use warchest::farm::{Backend, Call, Reply};
use warchest::net::{Net, NetLayout};
use warchest::pbs::{enumerate_configs, reserve, true_config, Belief, Ctx};
use warchest::rng::Rng;
use warchest::search::{Arenas, Cfg, Nets, Solved, Solver, Step};
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
    game_cfg_of(cfg(s, c))
}

fn game_cfg_of(cfg: Cfg) -> GameCfg {
    GameCfg {
        agents: [Agent::Sog { cfg }; 2],
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

/// What one stream produced: its training rows, and the size of every tree it
/// built. The trees are the sharper signal -- a solve that read another's
/// sampled leaves grows somewhere else entirely, where a solve that only saw a
/// different summation order grows the same tree and moves in the last bits.
struct Run {
    data: Data,
    nodes: Vec<usize>,
}

/// Run one game stream per `(seed, cfg)` against `backend`, every stream's
/// calls in the same round, and hand back what each produced.
///
/// A stream's games are a function of its seed alone, so the same seed run
/// alone and run beside others plays the same games and must produce the same
/// numbers. That is what makes batching testable.
fn generate(
    net: &Net,
    backend: Backend,
    streams: &[(u64, Cfg)],
    games: usize,
) -> Vec<Run> {
    let nets = Arc::new(Nets { value: net.clone(), device: backend.keeps_the_solve() });
    let n = streams.len();
    let mut nodes: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    let mut streams: Vec<GameStream> = streams
        .iter()
        .map(|&(seed, cfg)| GameStream::new(seed, game_cfg_of(cfg)))
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
                    nodes[i].push(sv.shape().nodes);
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
    out.into_iter().zip(nodes).map(|(data, nodes)| Run { data, nodes }).collect()
}

/// One stream, which is what a comparison against the CPU wants.
fn generate_one(net: &Net, backend: Backend, games: usize, s: u32, c: f32) -> Data {
    generate(net, backend, &[(0x51E5, cfg(s, c))], games)
        .pop()
        .expect("one stream")
        .data
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

/// The growth rule itself, held to the card on the card's own numbers.
///
/// Two whole solves with growth on cannot be compared. The trees part company
/// at the first close call, and they will have one: a cuBLAS leaf pass and a
/// host one differ in the last bits, and growth turns that into a different
/// node. Measured on the host alone, perturbing the network by one part in
/// `1e7` changes the node count of a third of a run's solves, by as much as
/// forty percent. So a test that asked two backends for the same tree would be
/// measuring the network, not the rule.
///
/// What can be compared is the rule. `Device::resident` hands back the arenas
/// an expansion phase reads, and `Solver::replay_expansion` runs the host's
/// own `sample_leaf` against them. Given the same numbers and the same stream
/// the two must agree simulation for simulation -- which is what holds
/// `k_expand`, `puct_choice`, `pick_live` and `live_cell` to `sample_leaf`,
/// `Solver::puct_choice`, `pick_live` and `Solver::live_cell`.
///
/// The phase compared is a solve's first, because `visits` is the one arena
/// the phase writes and before the first phase it is known to be zero. The
/// visits the replay leaves behind are then compared with the card's, so the
/// agreement is over every step of every trajectory and not just its end.
///
/// Four solves ride the round, at four budgets and four growth rates, because
/// the farm batches and a kernel that read a bound from the batch where it
/// should read it from the solve is right for one member and wrong for the
/// rest. `c` is what the phase widths come from -- a solve's first phase owes
/// `floor(c)` trajectories -- so the four ask for 3, 5, 8 and 13 of them and
/// the launch is a ragged one.
#[test]
fn growth_is_the_same_rule_as_the_reference() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let device = Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let Backend::Cuda(d) = &device else { unreachable!("just built") };
    let nets = Arc::new(Nets { value: net.clone(), device: true });
    let streams = [
        (0x51E5u64, 128u32, 3.0f32),
        (0x0A13, 192, 5.0),
        (0x77C1, 256, 8.0),
        (0x2E57, 320, 13.0),
    ];
    let n = streams.len();
    let mut data: Vec<Data> = (0..n).map(|_| Data::default()).collect();
    let mut gs: Vec<GameStream> = streams
        .iter()
        .map(|&(seed, s, c)| {
            // One expansion phase a round, deliberately. The replay reads the
            // arenas the card holds *after* the round and calls them the ones
            // the phase read, and that is only true when the round held one
            // phase: a round of `batch` regret updates moves `cur`, `sum`,
            // `qval` and `reach` between its phases, and the card hands back
            // the last state alone. Batched growth is the next test's job.
            GameStream::new(seed, game_cfg_of(Cfg { batch: 1, ..cfg(s, c) }))
        })
        .collect();

    let mut checked = 0usize;
    for _ in 0..4 {
        // One round, holding every stream's fresh solve. The first call a
        // solve raises already owes an expansion phase: at `c = 8` every
        // regret update earns eight trajectories.
        let mut live: Vec<Solver> = (0..n)
            .map(|i| {
                let mut sv = gs[i].next_solve(&nets, &mut data[i]);
                sv.pin(i);
                sv
            })
            .collect();
        let mut calls: Vec<Call> = Vec::new();
        let mut spans = vec![0usize; n];
        let mut sims = vec![0usize; n];
        for i in 0..n {
            let Step::Calls(cs) = live[i].advance(&[]) else {
                panic!("a fresh solve asks for a round")
            };
            // One expanding iterate a round, which `batch = 1` guarantees:
            // the snapshot taken afterwards is the state the phase read only
            // if there was exactly one phase.
            let owed: Vec<usize> = cs
                .iter()
                .filter_map(|c| match c {
                    Call::Iterate { expand, .. } if *expand > 0 => Some(*expand),
                    _ => None,
                })
                .collect();
            assert_eq!(
                owed.len(),
                1,
                "solve {i} asked for {} expansion phases in one round",
                owed.len()
            );
            sims[i] = owed[0];
            spans[i] = cs.len();
            calls.extend(cs);
        }
        let mut rest = device.run(&calls, 0).expect("the backend answered the round");
        let mut replies: Vec<Vec<Reply>> = Vec::new();
        for &k in &spans {
            let tail = rest.split_off(k);
            replies.push(rest);
            rest = tail;
        }

        for i in 0..n {
            assert_eq!(
                sims[i], streams[i].2 as usize,
                "solve {i}: the first phase owes floor(c) trajectories"
            );
            let got = d.resident(0, i).expect("the card gave its solve back");
            let theirs = &replies[i].last().expect("the round answered").leaves;
            assert_eq!(theirs.len(), sims[i], "solve {i}: the card sampled a short row");
            // Before a solve's first phase nothing has visited anything, so
            // the arenas the card holds now are the ones it grew from apart
            // from the visits, which are known.
            let zero = vec![0.0f32; got.visits.len()];
            let mine = live[i].replay_expansion(
                &Arenas {
                    reach: &got.reach,
                    cur: &got.cur,
                    sum: &got.sum,
                    qval: &got.qval,
                    visits: &zero,
                    prior: &got.prior,
                },
                sims[i],
            );
            let mine: Vec<u32> = mine
                .iter()
                .map(|l| l.map_or(NO_ROW, |x| x as u32))
                .collect();
            assert_eq!(
                &mine, theirs,
                "solve {i}: the reference and the card sampled different leaves"
            );
            assert_eq!(
                &live[i].cfr().visits[..got.visits.len()],
                &got.visits[..],
                "solve {i}: the trajectories passed through different cells"
            );
            checked += 1;
        }

        // Finish them the ordinary way, so the next round's solves come from a
        // played position rather than four openings.
        for i in 0..n {
            let mut r = std::mem::take(&mut replies[i]);
            let solved = loop {
                match live[i].advance(&r) {
                    Step::Calls(cs) => r = device.run(&cs, 0).expect("the backend answered"),
                    Step::Done(sd) => break sd,
                }
            };
            gs[i].keep(&live[i], solved, &mut data[i]);
        }
    }
    assert_eq!(checked, 4 * n, "every solve's first phase is compared");
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
    let streams: [(u64, Cfg); 4] = [
        (0x51E5, cfg(8, 0.0)),
        (0x0A13, cfg(11, 0.0)),
        (0x77C1, cfg(13, 0.0)),
        (0x2E57, cfg(17, 0.0)),
    ];
    let device = || Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let together = generate(&net, device(), &streams, 3);
    // A shared round must not move a solve at all, so the same run twice is
    // the control: whatever this reports is the floor the comparison sits on.
    let twice = generate(&net, device(), &streams, 3);
    let rel = |x: f32, y: f32| (x - y).abs() / (x.abs().max(y.abs()).max(1e-2));
    let count = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(&x, &y)| rel(x, y) > 1e-3).count();
    let mut bad = 0.0f32;
    for (i, &s) in streams.iter().enumerate() {
        let alone = generate(&net, device(), &[s], 3).pop().expect("one stream").data;
        let together = &together[i].data;
        assert_eq!(
            alone.cy.len(),
            together.cy.len(),
            "stream {i} solved a different number of positions in a shared round"
        );
        let (t, p) = (
            worst(&alone.cy, &together.cy, "targets"),
            worst(&alone.pprob, &together.pprob, "policy"),
        );
        eprintln!(
            "stream {i} iters={}: targets {t:e} ({} of {} differ)  policy {p:e} ({} of {} differ)  \
             repeat {:e}/{:e}",
            s.1.s,
            count(&alone.cy, &together.cy), alone.cy.len(),
            count(&alone.pprob, &together.pprob), alone.pprob.len(),
            worst(&twice[i].data.cy, &together.cy, "targets"),
            worst(&twice[i].data.pprob, &together.pprob, "policy"),
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

/// One round, holding every solver that still asks for one.
///
/// Stops at the first solver that is done and says which, without advancing
/// the rest: a caller either is waiting for that one or has one that stopped
/// filling the round.
fn shared_round(
    backend: &Backend,
    live: &mut [Solver],
    replies: &mut [Vec<Reply>],
) -> Option<(usize, Option<Solved>)> {
    let mut calls: Vec<Call> = Vec::new();
    let mut spans = vec![0usize; live.len()];
    for (i, sv) in live.iter_mut().enumerate() {
        match sv.advance(&replies[i]) {
            Step::Calls(cs) => {
                spans[i] = cs.len();
                calls.extend(cs);
            }
            Step::Done(solved) => return Some((i, solved)),
        }
    }
    let mut rest = backend.run(&calls, 0).expect("the backend answered the round");
    for (i, k) in spans.into_iter().enumerate() {
        let tail = rest.split_off(k);
        replies[i] = rest;
        rest = tail;
    }
    None
}

/// A ragged round must not move the smallest solve in it either.
///
/// `a_solve_does_not_depend_on_the_round_it_rides_in` shares a round between
/// four solves of the same shape: at `c = 0` nothing grows, so each of them is
/// the fourteen-node tree `Solver::new` built, and every level of a launch is
/// as wide for one member as for the next. A grid sized by the widest solve of
/// the round is right by construction on that shape. A run's rounds are not
/// that shape -- tree sizes there span two orders of magnitude -- and the
/// ragged one is what the flat work list has to get right: a level's items
/// come from solves of different depths, and the ones a shorter round drops
/// have to leave a prefix of every level behind them.
///
/// So the small solve here rides beside two that were grown first, to some
/// tens of times its size and several levels deeper, and is asked for the
/// targets it produces alone.
///
/// Only the small solve is read. The partners grow, so their own numbers are
/// not a function of their seed alone --
/// `growth_is_the_same_rule_as_the_reference` says why -- and for that same
/// reason no leaf either side sampled can be compared across two batch
/// compositions. Holding the sampling is the replay's job, not this test's.
///
/// The bounds are the neighbouring test's, and hold for its reason: a round of
/// three and a round of one give the leaf pass different GEMM shapes.
#[test]
fn a_ragged_round_does_not_move_the_small_solve() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let nets = Arc::new(Nets { value: net.clone(), device: true });
    // The solve under test: eight iterations over a tree that never grows, so
    // it is one round from beginning to end and the same one every time.
    let small = || {
        let mut g = GameStream::new(0x51E5, game_cfg(8, 0.0));
        let mut data = Data::default();
        let mut sv = g.next_solve(&nets, &mut data);
        sv.pin(0);
        (g, data, sv)
    };

    let (alone, tiny) = {
        let device = Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
        let (mut g, mut data, sv) = small();
        let (sv, solved) = run_solve(&device, sv);
        let tiny = sv.nodes.len();
        g.keep(&sv, solved, &mut data);
        (data, tiny)
    };

    let device = Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    // Two partners, in slots of their own, grown on their own for twelve
    // rounds. Both budgets run for more than twice that many, so neither can
    // finish and quietly stop making the round ragged.
    let mut big: Vec<Solver> = [(0x0A13u64, 256u32, 8.0f32), (0x77C1, 320, 13.0)]
        .iter()
        .enumerate()
        .map(|(i, &(seed, s, c))| {
            let mut data = Data::default();
            let mut sv = GameStream::new(seed, game_cfg(s, c)).next_solve(&nets, &mut data);
            sv.pin(i + 1);
            sv
        })
        .collect();
    let mut replies: Vec<Vec<Reply>> = big.iter().map(|_| Vec::new()).collect();
    for _ in 0..12 {
        if let Some((i, _)) = shared_round(&device, &mut big, &mut replies) {
            panic!("partner {i} finished before it had grown");
        }
    }
    let grown: Vec<usize> = big.iter().map(|sv| sv.nodes.len()).collect();
    eprintln!("small solve {tiny} nodes, partners {grown:?}");
    assert!(
        grown.iter().all(|&n| n > 20 * tiny),
        "the partners did not grow, so the round is not ragged: {grown:?} against {tiny}"
    );

    // The same solve again, now at the head of a round it shares with them.
    let (mut g, mut data, sv) = small();
    big.insert(0, sv);
    replies.insert(0, Vec::new());
    let solved = loop {
        match shared_round(&device, &mut big, &mut replies) {
            None => continue,
            Some((0, solved)) => break solved,
            Some((i, _)) => panic!("partner {i} finished while it was filling the round"),
        }
    };
    g.keep(&big[0], solved, &mut data);

    assert_eq!(
        alone.cy.len(),
        data.cy.len(),
        "the ragged round solved a different number of configs"
    );
    let (t, p) = (
        worst(&alone.cy, &data.cy, "targets"),
        worst(&alone.pprob, &data.pprob, "policy"),
    );
    eprintln!("ragged round: targets {t:e}  policy {p:e}");
    assert!(t < 1e-4, "a ragged round moved the small solve's targets by {t:e}");
    assert!(p < 5e-2, "a ragged round moved the small solve's policy by {p:e}");
}


/// The same question with the tree growing and a round carrying several
/// regret updates -- which is what production runs, and what none of the tests
/// above reach.
///
/// A round of `batch = 8` samples eight expansion phases before the host grows
/// anything, and the card lays them out phase-major over the whole round:
/// `at = phase * (parts * sims) + part * sims`, with `sims` the widest growth
/// rate in the round and every solve dropping out of the grid once
/// `iter >= t.todo`. Every one of those three is an index into a batch, so a
/// solve that read the wrong one takes another solve's leaves and grows
/// somewhere else. The streams are given different budgets so their rounds are
/// ragged in both directions -- different iteration counts, different growth
/// rates -- because a batch of equal members is the one shape where an index
/// off by a solve still lands on the right numbers.
///
/// Growth makes this strictly sharper than the `c = 0` test, not weaker: a
/// wrong index moves the tree, and a tree is discrete.
#[test]
fn a_growing_solve_does_not_depend_on_the_round_it_rides_in() {
    if Device::count() == 0 {
        eprintln!("no cuda device; skipping");
        return;
    }
    let net = random_net(0x9E37);
    let batched = |s: u32, c: f32| Cfg { batch: 8, ..cfg(s, c) };
    let streams: [(u64, Cfg); 4] = [
        (0x51E5, batched(32, 4.0)),
        (0x0A13, batched(48, 3.0)),
        (0x77C1, batched(64, 8.0)),
        (0x2E57, batched(80, 5.0)),
    ];
    let device = || Backend::Cuda(Device::new(&[0], net.clone()).expect("device"));
    let together = generate(&net, device(), &streams, 2);
    for (i, &s) in streams.iter().enumerate() {
        let alone = generate(&net, device(), &[s], 2).pop().expect("one stream");
        // The trees first. They are counts, so they are equal or they are not,
        // and an index read from the batch shows up here before it shows up
        // anywhere else.
        assert_eq!(
            alone.nodes, together[i].nodes,
            "stream {i} (s={}, c={}) built different trees alone and in company",
            s.1.s, s.1.c
        );
        let t = worst(&alone.data.cy, &together[i].data.cy, "targets");
        eprintln!("stream {i} s={} c={}: trees {:?}  targets {t:e}", s.1.s, s.1.c, alone.nodes);
        assert!(t < 1e-3, "stream {i}: sharing a round moved its targets by {t:e}");
    }
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
        ("prior", &host.cfr().prior[..cells], &got.prior[..cells]),
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
