#![cfg(feature = "gpu")]

use std::sync::{Arc, OnceLock};

use warchest::contract::NO_ROW;
use warchest::cuda::Device;
use warchest::contract::{Call, Reply};
use warchest::net::Net;
use warchest::pbs::{expand_row, pack_row, true_config, Ctx, PUBFEAT, ROW_BYTES};
use warchest::rng::Rng;
use warchest::resolve::{gadget_iteration, PlaySolved, SolveOutput};
use warchest::search::{Budget, Cfg, Cfr, Policy, Solver, Step};
use warchest::selfplay::{make_game, Agent, Collect, Data, GameCfg, GameStream};

struct Backend(TestDevice);

impl Backend {
    fn run(&self, calls: &[Call], card: usize) -> Option<Vec<Reply>> {
        self.0.run(calls, card)
    }
}

fn cfg(s: u32, c: f32) -> Cfg {
    Cfg { s, c, prior_temp: 1.7, ..Default::default() }
}

fn game_cfg(s: u32, c: f32) -> GameCfg {
    game_cfg_of(cfg(s, c))
}

fn game_cfg_of(cfg: Cfg) -> GameCfg {
    GameCfg {
        agents: [Agent::Sog { cfg }; 2],
        collect: Collect::Sog,
        static_explore: 0.0,
        random_draft: true,
        p_td1: 0.0,
        query_rate: 0.0,
        recursive_rate: 0.0,
    }
}

struct Run { data: Data }

const GPU_SLOTS: usize = 32;
static DEVICE: OnceLock<Device> = OnceLock::new();

fn shared_device() -> &'static Device {
    DEVICE.get_or_init(|| {
        let net = Net::random(0x9E37);
        let cfg = Cfg { budget: Budget::for_s(512), ..Default::default() };
        Device::new(&[0], &net, cfg, GPU_SLOTS).expect("device")
    })
}

#[derive(Clone, Copy)]
struct TestDevice { device: &'static Device, slot_base: usize, }

impl TestDevice {
    fn run(&self, calls: &[Call], lane: usize) -> Option<Vec<Reply>> {
        let calls: Vec<Call> = calls.iter().map(|call| shift_call(call, self.slot_base)).collect();
        self.device.run(&calls, lane)
    }

    fn resident(&self, card: usize, solve: usize) -> Result<warchest::cuda::Resident, String> {
        self.device.resident(card, self.slot_base + solve)
    }

    fn expand_rows(&self, rows: &[u8]) -> Result<Vec<f32>, String> {
        self.device.expand_rows(rows)
    }
}

fn shift_call(call: &Call, base: usize) -> Call {
    let mut shifted = call.clone();
    match &mut shifted {
        Call::Trunk { solve, .. }
        | Call::Configs { solve, .. }
        | Call::Gadget { solve, .. }
        | Call::Tree { solve, .. }
        | Call::Iterate { solve, .. }
        | Call::ReadPlay { solve, .. }
        | Call::ReadRefresh { solve, .. }
        | Call::ReadTarget { solve, .. } => *solve += base,
    }
    shifted
}

fn gpu(slot_base: usize) -> TestDevice {
    assert!(slot_base < GPU_SLOTS, "test slot range starts outside the device");
    TestDevice { device: shared_device(), slot_base }
}

#[test]
fn packed_rows_expand_on_the_card() {
    let mut rng = Rng::new(0x0A11_CE55);
    let mut rows = Vec::with_capacity(4096 * ROW_BYTES);
    while rows.len() / ROW_BYTES < 4096 {
        let mut state = make_game(&mut rng, true);
        let ctx = Ctx::new(&state);
        for _ in 0..160 {
            if state.is_terminal() {
                break;
            }
            if state.is_valued() {
                let mirror = state.mirror();
                for (s, c) in [(&state, ctx), (&mirror, ctx.mirrored())] {
                    let at = rows.len();
                    rows.resize(at + ROW_BYTES, 0);
                    pack_row(s, &c, &mut rows[at..at + ROW_BYTES]);
                }
            }
            let actions = state.legal_actions();
            state.apply_inplace(actions[rng.below(actions.len())]);
        }
    }
    rows.truncate(4096 * ROW_BYTES);
    let mut want = vec![0.0; 4096 * PUBFEAT];
    for (row, out) in rows.chunks_exact(ROW_BYTES).zip(want.chunks_exact_mut(PUBFEAT)) {
        expand_row(row, out);
    }
    let got = gpu(0).expand_rows(&rows).expect("expand rows");
    for (i, (&a, &b)) in want.iter().zip(&got).enumerate() {
        if a == 0.0 || a == 1.0 {
            assert_eq!(a.to_bits(), b.to_bits(), "one-hot feature {i}");
        } else {
            assert!((a - b).abs() <= 1e-6, "scalar feature {i}: {a} vs {b}");
        }
    }
}

fn generate(
    net: &Net,
    backend: Backend,
    streams: &[(u64, Cfg)],
    games: usize,
) -> Vec<Run> {
    let nets = Arc::new(net.clone());
    let n = streams.len();
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
            let step = sv.advance(&replies[i]);
            match step {
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
            let mut leaves: Vec<u32> = replies[i]
                .iter()
                .flat_map(|r| r.leaves.iter().copied())
                .filter(|&l| l != NO_ROW)
                .collect();
            let all = leaves.len();
            leaves.sort_unstable();
            leaves.dedup();
            assert_eq!(leaves.len(), all, "stream {i}: a round took a leaf twice");
        }
    }
    out.into_iter().map(|data| Run { data }).collect()
}

fn run_solve(backend: &Backend, mut sv: Solver) -> (Solver, Result<SolveOutput, String>) {
    sv.pin(0);
    let mut replies: Vec<Reply> = Vec::new();
    loop {
        let step = sv.advance(&replies);
        match step {
            Step::Calls(calls) => replies = backend.run(&calls, 0).expect("the backend answered"),
            Step::Done(solved) => return (sv, solved),
        }
    }
}

fn policy(solved: &SolveOutput) -> &Policy {
    match solved {
        SolveOutput::Play(s) => match s.as_ref() {
            PlaySolved::Continue(s) => &s.policy,
            PlaySolved::Terminal(s) => &s.policy,
        },
        SolveOutput::Target(s) => &s.policy,
        SolveOutput::Refresh(_) => panic!("refresh has no policy"),
    }
}

#[test]
fn fresh_batched_solves_use_supplied_beliefs_for_their_first_priors() {
    let net = Arc::new(Net::random(0x9E37));
    let cfg = Cfg { s: 1, c: 1.0, batch: 8, ..Default::default() };
    let mut data = Data::default();
    let seed = GameStream::new(0x51E5, game_cfg_of(cfg)).next_solve(&net, &mut data);
    let mut beliefs = [seed.root_belief.clone(), seed.root_belief.clone()];
    for (i, belief) in beliefs.iter_mut().flatten().enumerate() {
        belief.p.fill(0.0);
        let at = (i / 2) * (belief.len() - 1);
        belief.p[at] = 1.0;
    }
    let (mut solves, mut calls) = (Vec::new(), Vec::new());
    for (i, belief) in beliefs.into_iter().enumerate() {
        let state = seed.nodes[0].state;
        let actor = state.to_act();
        let actual = true_config(&state, actor, &Ctx::new(&state));
        let mut sv = Solver::initial_play(&state, Ctx::new(&state),
            Arc::clone(&net), cfg, belief, Rng::new(0xB3113F + i as u64), actual).unwrap();
        sv.pin(i);
        let Step::Calls(fresh) = sv.advance(&[]) else { panic!("a fresh solve asks for work") };
        calls.extend(fresh);
        solves.push(sv);
    }
    let device = gpu(30);
    device.run(&calls, 0).expect("the card answered the first round");
    let mut priors = Vec::new();
    for (i, sv) in solves.iter().enumerate() {
        let resident = device.resident(0, i).expect("the fresh solve is resident");
        let belief = [&sv.root_belief[0].p[..], &sv.root_belief[1].p[..]].concat();
        assert_eq!(&resident.reach[..belief.len()], &belief);
        let root = &sv.nodes[0];
        let cells = root.soff as usize..root.soff as usize + root.legal_action.len();
        priors.push(resident.prior[cells].to_vec());
    }
    assert!(priors[0].iter().zip(&priors[1]).any(|(&a, &b)| (a - b).abs() > 1e-5));
}

#[test]
fn a_solve_may_change_pipeline_streams() {
    let net = Net::random(0x9E37);
    let device = Backend(gpu(0));
    let nets = Arc::new(net);
    let fresh = || {
        let mut data = Data::default();
        GameStream::new(0x51E5, game_cfg(32, 0.0)).next_solve(&nets, &mut data)
    };
    let want = run_solve(&device, fresh()).1.expect("a finished solve");
    let mut sv = fresh();
    sv.pin(0);
    let mut replies = Vec::new();
    let mut lane = 0;
    loop {
        match sv.advance(&replies) {
            Step::Calls(calls) => {
                replies = device.run(&calls, lane).expect("the other pipeline answered");
                lane ^= 1;
            }
            Step::Done(got) => {
                let got = got.expect("a finished solve");
                assert_eq!(policy(&want).off, policy(&got).off);
                assert_eq!(policy(&want).act, policy(&got).act);
                let bad = worst(&policy(&want).p, &policy(&got).p, "root policy");
                assert!(bad < 1e-6, "the policy changed across streams by {bad:e}");
                break;
            }
        }
    }
    device.run(&[], lane).expect("the context stayed healthy");
}

fn worst(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{what}: length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs() / (x.abs().max(y.abs()).max(1e-2)))
        .fold(0.0, f32::max)
}

fn worst_scaled(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{what}: length {} vs {}", a.len(), b.len());
    let scale = (a.iter().map(|x| x * x).sum::<f32>() / a.len().max(1) as f32).sqrt().max(1e-2);
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs() / scale).fold(0.0, f32::max)
}

const TF32: f32 = 3e-3;

#[test]
fn a_solve_does_not_depend_on_the_round_it_rides_in() {
    let net = Net::random(0x9E37);
    let streams: [(u64, Cfg); 4] = [
        (0x51E5, cfg(8, 0.0)),
        (0x0A13, cfg(11, 0.0)),
        (0x77C1, cfg(13, 0.0)),
        (0x2E57, cfg(17, 0.0)),
    ];
    let device = || Backend(gpu(7));
    let together = generate(&net, device(), &streams, 3);
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
            worst_scaled(&alone.cy, &together.cy, "targets"),
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
        assert!(t < 2.0 * TF32, "stream {i}: sharing a round moved its targets by {t:e}");
        bad = bad.max(p);
    }
    eprintln!("worst policy difference across streams {bad:e}");
}

fn shared_round(
    backend: &Backend,
    live: &mut [Solver],
    replies: &mut [Vec<Reply>],
) -> Option<(usize, Result<SolveOutput, String>)> {
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

#[test]
fn a_ragged_round_does_not_move_the_small_solve() {
    let net = Net::random(0x9E37);
    let nets = Arc::new(net);
    let small = || {
        let mut g = GameStream::new(0x51E5, game_cfg(8, 0.0));
        let mut data = Data::default();
        let mut sv = g.next_solve(&nets, &mut data);
        sv.pin(0);
        (g, data, sv)
    };

    let (alone, tiny) = {
        let device = Backend(gpu(11));
        let (mut g, mut data, sv) = small();
        let (sv, solved) = run_solve(&device, sv);
        let tiny = sv.nodes.len();
        g.keep(&sv, solved, &mut data);
        (data, tiny)
    };

    let device = Backend(gpu(11));
    let mut big: Vec<Solver> = [(0x0A13u64, 1024u32, 8.0f32), (0x77C1, 1664, 13.0)]
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
        worst_scaled(&alone.cy, &data.cy, "targets"),
        worst(&alone.pprob, &data.pprob, "policy"),
    );
    eprintln!("ragged round: targets {t:e}  policy {p:e}");
    assert!(t < 2.0 * TF32, "a ragged round moved the small solve's targets by {t:e}");
    assert!(p < 5e-2, "a ragged round moved the small solve's policy by {p:e}");
}


#[test]
fn played_session_carries_across_multiple_steps() {
    let net = Arc::new(Net::random(0x9E37));
    let cfg = Cfg { s: 2, c: 0.0, batch: 2, ..Default::default() };
    let device = Backend(gpu(24));
    let mut stream = GameStream::new(0x5E5510, game_cfg_of(cfg));
    let mut data = Data::default();
    for decision in 0..4 {
        let mut solver = stream.next_solve(&net, &mut data);
        solver.pin(0);
        let mut replies = Vec::new();
        let mut saw_gadget = false;
        let solved = loop {
            match solver.advance(&replies) {
                Step::Calls(calls) => {
                    saw_gadget |= calls.iter().any(|call| matches!(call, Call::Gadget { .. }));
                    replies = device.run(&calls, 0).expect("session solve");
                }
                Step::Done(solved) => break solved,
            }
        };
        assert_eq!(saw_gadget, decision > 0);
        stream.keep(&solver, solved, &mut data);
    }
}

#[test]
fn gadget_and_carry_match_one_cpu_iteration() {
    let net = Arc::new(Net::random(0x9E37));
    let cfg = Cfg { s: 1, c: 0.0, batch: 1, cfr: Cfr::DISCOUNTED, ..Default::default() };
    let device = Backend(gpu(26));
    let mut stream = GameStream::new(0xC411, game_cfg_of(cfg));
    let mut data = Data::default();
    let first = stream.next_solve(&net, &mut data);
    let (first, solved) = run_solve(&device, first);
    stream.keep(&first, solved, &mut data);
    let mut second = stream.next_solve(&net, &mut data);
    second.pin(0);
    let Step::Calls(calls) = second.advance(&[]) else { panic!("a re-solve asks for work") };
    device.run(&calls, 0).expect("one gadget iteration");
    let resident = device.0.resident(0, 0).expect("resident re-solve");
    let n = resident.gadget.len() / 6;
    assert!(n > 0, "the second played solve has a gadget");
    let previous = &resident.gadget[..n];
    let term = &resident.gadget[n..2 * n];
    let opponent = 1 - second.nodes[0].player as usize;
    let vo = opponent * second.nvals + second.nodes[0].voff as usize;
    let follow = &resident.vals[vo..vo + n];
    let mut regret = vec![[0.0; 2]; n];
    let mut strategy = vec![[0.5; 2]; n];
    let mut sum = vec![[0.0; 2]; n];
    gadget_iteration(term, follow, &mut regret, &mut strategy, &mut sum, 0, cfg.cfr);
    for k in 0..n {
        assert!((resident.gadget[2 * n + k] - regret[k][0]).abs() < 2e-6);
        assert!((resident.gadget[3 * n + k] - regret[k][1]).abs() < 2e-6);
        assert!((resident.gadget[4 * n + k] - strategy[k][0]).abs() < 2e-6);
        assert!((resident.gadget[5 * n + k] - strategy[k][1]).abs() < 2e-6);
    }
    let mix = |follow: &[f32]| {
        let total: f32 = follow.iter().sum();
        previous.iter().zip(follow).map(|(&p, &f)| 0.5 * (p + f / total)).collect::<Vec<_>>()
    };
    let expected = mix(&strategy.iter().map(|s| s[1]).collect::<Vec<_>>());
    let root = &second.nodes[0];
    let root_at = root.roff as usize + if opponent == 0 { 0 } else { root.nc[0] as usize };
    assert!(worst(&expected, &resident.reach[root_at..root_at + n], "gadget range") < 2e-6);
    let counts = root.nc.map(|x| x as usize);
    for p in 0..2 {
        let start = if p == 0 { 0 } else { counts[0] };
        let expected = if p == opponent {
            mix(&vec![0.5; n])
        } else {
            second.root_belief[p].p.clone()
        };
        assert!(worst(&expected, &resident.carry[start..start + counts[p]], "carried root reach") < 2e-6);
    }
    let value_at = counts[0] + counts[1] + if opponent == 0 { 0 } else { counts[0] };
    assert_eq!(&resident.carry[value_at..value_at + n], follow);
}

#[test]
fn cfr_average_uses_evaluated_strategies_and_global_steps() {
    let net = Arc::new(Net::random(0x9E37));
    let device = Backend(gpu(28));
    let fresh = |s, batch| {
        let cfg = Cfg { s, c: 0.0, batch, cfr: Cfr::DISCOUNTED, ..Default::default() };
        let mut data = Data::default();
        GameStream::new(0x51E5, game_cfg_of(cfg)).next_solve(&net, &mut data)
    };

    let one = fresh(1, 1);
    let row = one.nodes[0].legal_row(0);
    let so = one.nodes[0].soff as usize;
    let initial = one.cur[so + row.start..so + row.end].to_vec();
    let solved = run_solve(&device, one).1.expect("one-update solve");
    assert!(worst(&initial, &policy(&solved).p[..initial.len()], "one-update average") < 2e-6);

    let solve = |batch| run_solve(&device, fresh(3, batch)).1.expect("three-update solve");
    let together = solve(3);
    let split = solve(1);
    assert!(worst(&policy(&together).p, &policy(&split).p, "split average") < 2e-6);
}

#[test]
fn k_iterates_together_match_k_iterates_alone() {
    const K: usize = 4;
    let net = Net::random(0x9E37);
    let device = gpu(20);
    let nets = Arc::new(net);
    let mut setup = Vec::new();
    let mut iterates = Vec::new();
    for i in 0..2 * K {
        let mut data = Data::default();
        let mut sv = GameStream::new(0x51E5, game_cfg(8, 0.0)).next_solve(&nets, &mut data);
        sv.pin(i);
        match sv.advance(&[]) {
            Step::Calls(cs) => {
                for c in cs {
                    if matches!(c, Call::Iterate { .. }) {
                        iterates.push(c);
                    } else {
                        setup.push(c);
                    }
                }
            }
            Step::Done(_) => panic!("a fresh solve is already done"),
        }
    }
    device.run(&setup, 0).expect("setup");
    assert_eq!(iterates.len(), 2 * K, "each copy owes one iterate");
    for (i, call) in iterates.iter_mut().enumerate() {
        let Call::Iterate { cfr, puct, .. } = call else { unreachable!() };
        *cfr = Cfr::NAMED[i % K % Cfr::NAMED.len()].1;
        *puct = 1.0 + (i % K) as f32;
    }

    let batched = device.run(&iterates[..K], 0).expect("batched iterate");
    let mut serial = Vec::new();
    for c in &iterates[K..] {
        serial.extend(device.run(std::slice::from_ref(c), 0).expect("serial iterate"));
    }
    assert_eq!(batched.len(), K);
    assert_eq!(serial.len(), K);

    for i in 0..K {
        assert_eq!(
            batched[i].leaves, serial[i].leaves,
            "slot {i}: batched iterate sampled different leaves"
        );
        let a = device.resident(0, i).expect("batched resident");
        let b = device.resident(0, i + K).expect("serial resident");
        for (what, x, y) in [
            ("cur", a.cur.as_slice(), b.cur.as_slice()),
            ("sum", a.sum.as_slice(), b.sum.as_slice()),
            ("qval", a.qval.as_slice(), b.qval.as_slice()),
            ("visits", a.visits.as_slice(), b.visits.as_slice()),
            ("reach", a.reach.as_slice(), b.reach.as_slice()),
        ] {
            let bad = worst(x, y, what);
            eprintln!("slot {i} {what}: worst {bad:e} over {} values", x.len());
            assert!(bad < 1e-3, "slot {i} {what} differ by {bad:e}");
        }
    }
}
