//! The device must answer a round exactly as the CPU network would.
//!
//! The calls are not synthetic. Real solves run against a gate, every call
//! they raise is captured, and the same captured round then goes through both
//! backends. That way the shapes, the card tables, the belief pooling and the
//! ragged config counts are whatever the solver actually produces, including
//! the mixture of queried seats that forces the device to batch calls whose
//! `player` differs.
//!
//! Needs a GPU, so it only builds under `--features gpu`.
#![cfg(feature = "gpu")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use warchest::cuda::Device;
use warchest::farm::{Backend, Call, Gate, Reply};
use warchest::net::{Net, NetLayout};
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{Agent, Collect, GameCfg, GameStream};

fn random_net(seed: u64) -> Net {
    let mut r = warchest::rng::Rng::new(seed);
    let l = NetLayout::new();
    let mut draw = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (r.unit_f64() as f32 - 0.5) * 0.2)
            .collect()
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

/// Run real solves and keep the rounds they raise.
///
/// Answers every round with the CPU network, because the solver has to make
/// progress to reach the later call kinds at all.
fn capture(net: &Net, threads: usize, want: usize) -> Vec<Vec<Call>> {
    let cfg = Cfg {
        nodes: 64,
        expand: 4,
        iters: 8,
        ..Default::default()
    };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg }; 2],
        collect: Collect::Rebel,
        explore: 0.1,
        random_draft: true,
        eval_mix: 1.0,
        mc_mix: 0.0,
        query_rate: 0.9,
        recursive_rate: 0.1,
    };
    let gate = Arc::new(Gate::default());
    let stopping = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = (0..threads)
        .map(|t| {
            let (gate, stopping, net) = (gate.clone(), stopping.clone(), net.clone());
            std::thread::spawn(move || {
                let _member = gate.enter();
                let nets = Nets {
                    value: net,
                    gate: Some(gate.clone()),
                };
                let mut stream = GameStream::new(0x51E5 ^ t as u64, gc);
                while !stopping.load(Ordering::Relaxed) {
                    stream.generate(&nets, 4);
                }
            })
        })
        .collect();

    let mut rounds: Vec<Vec<Call>> = Vec::new();
    let mut kinds = [0usize; 3];
    // Keep going until every call kind has been seen a few times, so a pass
    // that is never exercised cannot pass by being absent.
    while rounds.len() < want || kinds.iter().any(|&n| n < 3) {
        let got = gate.round(|calls| {
            rounds.push(calls.to_vec());
            for c in calls {
                kinds[c.kind()] += 1;
            }
            calls.iter().map(|c| c.run(net)).collect()
        });
        assert!(got.is_some(), "the gate closed while capturing");
    }
    stopping.store(true, Ordering::Relaxed);
    // The threads are mid-solve and still need every one of their remaining
    // rounds answered before they can notice the stop flag and leave. Closing
    // first would strand them inside a solve.
    while gate
        .serve_until_idle(|calls| calls.iter().map(|c| c.run(net)).collect())
        .is_some()
    {}
    gate.close();
    for w in workers {
        let _ = w.join();
    }
    rounds
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

const KIND: [&str; 3] = ["trunk", "configs", "join"];

/// Hold a device to the CPU network over captured rounds.
///
/// Reports the worst difference of every kind rather than stopping at the
/// first, because which kinds are wrong is most of the diagnosis. A call is
/// also run alone, which separates an error in the arithmetic from one in the
/// batching: alone, a batch holds exactly one call.
fn compare(device: &Device, reference: &Backend, rounds: &[Vec<Call>]) -> ([usize; 3], f32) {
    let mut seen = [0usize; 3];
    let (mut batched, mut alone) = ([0.0f32; 3], [0.0f32; 3]);
    let mut sample = None;
    for calls in rounds {
        let want: Vec<Reply> = reference.run(calls);
        let got: Vec<Reply> = device.run(calls);
        assert_eq!(got.len(), calls.len(), "one reply per call");
        for (i, call) in calls.iter().enumerate() {
            let k = call.kind();
            seen[k] += 1;
            let d = worst(&want[i].a, &got[i].a, "a").max(worst(&want[i].b, &got[i].b, "b"));
            let solo = device.run(std::slice::from_ref(call));
            let s = worst(&want[i].a, &solo[0].a, "a").max(worst(&want[i].b, &solo[0].b, "b"));
            if d > batched[k] {
                batched[k] = d;
                if k == 0 && sample.is_none() {
                    let n = want[i].a.len().min(8);
                    sample = Some((want[i].a[..n].to_vec(), got[i].a[..n].to_vec()));
                }
            }
            alone[k] = alone[k].max(s);
        }
    }
    for k in 0..3 {
        println!(
            "{:>7}: {:4} calls, worst batched {:e}, worst alone {:e}",
            KIND[k], seen[k], batched[k], alone[k]
        );
    }
    if let Some((want, got)) = sample {
        println!("first trunk row want {want:?}");
        println!("first trunk row got  {got:?}");
    }
    let top = batched.iter().chain(&alone).fold(0.0f32, |a, &b| a.max(b));
    assert!(top < 2e-3, "worst difference {top:e}");
    (seen, top)
}

/// A call's answer must not depend on what it is batched with.
///
/// This is the property the concatenation rests on, and it is the one that
/// broke: a call carrying a tail of its caller's scratch buffer answers
/// correctly on its own and shifts every later call in the batch. Growing
/// prefixes catch that, because the first call stays right while the rest
/// drift.
#[test]
fn a_call_answers_the_same_whatever_it_is_batched_with() {
    let net = random_net(0x9E37_79B9);
    let rounds = capture(&net, 6, 4);
    let device = Device::new(&[0], net).expect("cuda device 0");
    let mut checked = 0;
    for calls in &rounds {
        for kind in 0..3 {
            let group: Vec<Call> = calls.iter().filter(|c| c.kind() == kind).cloned().collect();
            if group.len() < 2 {
                continue;
            }
            let alone: Vec<Reply> = group
                .iter()
                .map(|c| device.run(std::slice::from_ref(c)).remove(0))
                .collect();
            for take in 2..=group.len() {
                let got = device.run(&group[..take]);
                for i in 0..take {
                    let d = worst(&alone[i].a, &got[i].a, "a")
                        .max(worst(&alone[i].b, &got[i].b, "b"));
                    assert!(
                        d < 2e-3,
                        "{} call {i} in a batch of {take} moved by {d}",
                        KIND[kind]
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no batch of two or more was ever formed");
    println!("{checked} batch prefixes agreed with the calls run alone");
}

#[test]
fn the_device_matches_the_cpu_network_on_real_rounds() {
    let net = random_net(0x9E37_79B9);
    let rounds = capture(&net, 6, 12);
    let reference = Backend::Reference(net.clone(), Default::default());
    let device = Device::new(&[0], net).expect("cuda device 0");
    let (seen, top) = compare(&device, &reference, &rounds);
    assert!(
        seen.iter().all(|&n| n >= 3),
        "not every call kind was exercised: {seen:?}"
    );
    println!("one card: worst relative difference {top:e} over {seen:?} calls");
}

/// The same, with the round split across two cards. A shard holds only part of
/// each kind, so this catches anything that is right only when one batch holds
/// every call — a card-table base, an owner offset, a row that assumed it knew
/// the whole batch.
#[test]
fn two_cards_match_the_cpu_network_as_well() {
    if Device::count() < 2 {
        eprintln!("one card only; skipping the shard test");
        return;
    }
    let net = random_net(0x1234_5677);
    let rounds = capture(&net, 6, 8);
    let reference = Backend::Reference(net.clone(), Default::default());
    let device = Device::new(&[0, 1], net).expect("cuda devices 0 and 1");
    let (seen, top) = compare(&device, &reference, &rounds);
    assert!(
        seen.iter().all(|&n| n >= 3),
        "not every call kind was exercised: {seen:?}"
    );
    println!("two cards: worst relative difference {top:e} over {seen:?} calls");
}
