//! Work-package-B oracle dump: a serialized solve job, its weights, and the
//! CPU solver's reference outputs for the same solve.
//!
//! `train/cfr_spec.py` (the torch CFR specification) consumes this directory:
//! it replays the job with torch ops and must reproduce the reference within
//! tolerance. The CUDA service's phase kernels are checked against the Rust
//! solver directly (in-crate tests), so this dump exists to validate the
//! *specification* the kernels are written against.
//!
//! Usage: `oracle_dump <out-dir> [weights.bin] [roots] [iters] [depth]`
//!   weights.bin: optional, the `export_weights.py` format. Default: fresh
//!   deterministic random weights in the production shape (h384 dg64 r64).
//!   roots: how many subgame roots to dump (default 3).
//!   iters: CFR iterations per solve (default 16).
//!   depth: subgame depth (default 2).
//!
//! Each root writes `<out-dir>/solve_<k>.bin` (the job), `weights.bin` (the
//! weights, one copy shared by every solve), and `<out-dir>/solve_<k>.ref`
//! (the reference outputs: the reference strategy, every kept snapshot, the
//! Phase-2 root values, the carried beliefs at the exit leaf, and NashConv).

use std::fs;

use warchest::net::Mlp;
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};
use warchest::serialize::Job;

/// The production shape the pre-CUDA runs train. The service reads dims from
/// the weights, so any shape works — this one is just the default.
const DIMS: [usize; 10] = [
    warchest::rebel::PUBFEAT,
    384, // hidden
    384, // head
    warchest::rebel::CFEAT,
    64, // dg
    64, // rank
    warchest::rebel::AFEAT,
    32, // de
    64, // dc
    0,  // encoder: flat
];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out_dir = a.get(1).cloned().unwrap_or_else(|| "oracle".into());
    let weights_path = a.get(2).filter(|s| !s.is_empty() && *s != "-").cloned();
    let n_roots: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let depth: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(2);
    fs::create_dir_all(&out_dir).expect("out dir");

    // Weights: load, or make deterministic random ones in the production shape.
    let (dims, w, b, ln, raw) = if let Some(p) = &weights_path {
        let raw = fs::read(p).expect("weights file");
        let (dims, w, b, ln) = parse_weights(&raw);
        (dims, w, b, ln, raw)
    } else {
        let (w, b, ln) = random_flat();
        let raw = flat_to_bytes(&DIMS, &w, &b, &ln);
        (DIMS.to_vec(), w, b, ln, raw)
    };
    let net = Mlp::from_flat(&dims, &w, &b, &ln).expect("weights parse");
    fs::write(format!("{out_dir}/weights.bin"), &raw).expect("weights.bin");
    let nets = [Nets { value: net }];

    let cfg = Cfg {
        depth,
        iters,
        snapshots: true,
        ..Default::default()
    };
    // Roots from real ReBeL solves under the dump's own weights: the tree
    // shapes are the ones the GPU will actually see.
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg: Cfg { depth: 2, iters: 4, snapshots: false, ..Default::default() }, slot: 0 }; 2],
        collect: Collect::None,
        explore: 0.0,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let roots = collect_roots(8, 0xC0FFEE, &nets, &gc, n_roots + 6);
    let mut dumped = 0usize;
    for (k, (s, bel)) in roots.into_iter().enumerate() {
        if dumped >= n_roots {
            break;
        }
        let ctx = warchest::rebel::Ctx::new(&s);
        let root_bel = [bel[0].p.clone(), bel[1].p.clone()];
        let mut sv = Solver::new(&s, &ctx, &nets[0], cfg, bel);
        if sv.capped() {
            continue;
        }
        // Phase 1, then Phase 2 for the live belief, then the carried beliefs
        // at the walk's likely exit leaf (the first non-terminal leaf).
        sv.multistep(iters);
        let roots_v: Vec<[Vec<f32>; 2]> = vec![root_bel];
        let vals = sv.value_under(&roots_v);
        let leaf = *sv.leaf_rows.first().expect("non-terminal leaves");
        let carried = sv.carried_beliefs(leaf);
        let conv = sv.nash_conv();
        let job = Job::from_solver(&sv, &roots_v);
        fs::write(format!("{out_dir}/solve_{k}.bin"), job.to_bytes()).expect("job");
        fs::write(format!("{out_dir}/solve_{k}.ref"), ref_bytes(&sv, &vals, leaf, &carried, conv))
            .expect("ref");
        println!(
            "solve {k}: nodes={} leaves={} ncfg={} ncells={} iters={iters} -> solve_{k}.bin",
            sv.nodes.len(),
            sv.leaf_rows.len(),
            sv.ncfg,
            sv.ncells
        );
        dumped += 1;
    }
    println!("dumped {dumped} solves to {out_dir}");
}

/// The reference outputs, as `train/cfr_spec.py` reads them:
/// `u32 magic, u32 nsnaps, u32 ncells, f32 reference[ncells],
///  f32 snaps[nsnaps * ncells], u32 nroots, per root f32[nc0] + f32[nc1],
///  u32 leaf, u32 ncarried, per carried f32[nc0] + f32[nc1],
///  f32 nash, f32 zero_sum`.
fn ref_bytes(
    sv: &Solver,
    vals: &[[Vec<f32>; 2]],
    leaf: usize,
    carried: &[[Vec<f32>; 2]],
    conv: warchest::search::Conv,
) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::new();
    let mut w = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
    let mut f = |b: &mut Vec<u8>, v: f32| b.extend_from_slice(&v.to_le_bytes());
    w(&mut b, 0x5743_5252); // "WCRR"
    w(&mut b, sv.snaps.len() as u32);
    w(&mut b, sv.ncells as u32);
    for &x in sv.snaps.last().expect("snapshots") {
        f(&mut b, x);
    }
    for snap in &sv.snaps {
        for &x in snap {
            f(&mut b, x);
        }
    }
    w(&mut b, vals.len() as u32);
    for v in vals {
        for &x in &v[0] {
            f(&mut b, x);
        }
        for &x in &v[1] {
            f(&mut b, x);
        }
    }
    w(&mut b, leaf as u32);
    w(&mut b, carried.len() as u32);
    for c in carried {
        for &x in &c[0] {
            f(&mut b, x);
        }
        for &x in &c[1] {
            f(&mut b, x);
        }
    }
    w(&mut b, 1);
    f(&mut b, conv.nash);
    w(&mut b, 1);
    f(&mut b, conv.zero_sum);
    b
}

/// The `export_weights.py` byte format: `u32 n_dims, dims, u32 n_w, w,
/// u32 n_b, b, u32 n_ln, ln` (all little-endian).
fn flat_to_bytes(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Vec<u8> {
    let mut raw = Vec::new();
    raw.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        raw.extend_from_slice(&(d as u32).to_le_bytes());
    }
    for arr in [w, b, ln] {
        raw.extend_from_slice(&(arr.len() as u32).to_le_bytes());
        for &x in arr {
            raw.extend_from_slice(&x.to_le_bytes());
        }
    }
    raw
}

fn parse_weights(raw: &[u8]) -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut at = 0usize;
    let u32_at = |b: &[u8], at: &mut usize| -> usize {
        let v = u32::from_le_bytes(b[*at..*at + 4].try_into().unwrap()) as usize;
        *at += 4;
        v
    };
    let f32s_at = |b: &[u8], at: &mut usize| -> Vec<f32> {
        let n = u32_at(b, at);
        let v = (0..n)
            .map(|i| f32::from_le_bytes(b[*at + i * 4..*at + i * 4 + 4].try_into().unwrap()))
            .collect();
        *at += n * 4;
        v
    };
    let nd = u32_at(raw, &mut at);
    let dims: Vec<usize> = (0..nd).map(|_| u32_at(raw, &mut at)).collect();
    (dims, f32s_at(raw, &mut at), f32s_at(raw, &mut at), f32s_at(raw, &mut at))
}

/// Deterministic random flat arrays in the production shape, with the
/// LayerNorms at their identity start (scale 1, bias 0).
fn random_flat() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(0xD15EA5E);
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| (rng.next_u64() as f32 / u64::MAX as f32) * 2.0 - 1.0)
            .collect()
    };
    let (h, hd, dg, rk, de, dc) = (DIMS[1], DIMS[2], DIMS[4], DIMS[5], DIMS[7], DIMS[8]);
    let (af, hf, xd) = (
        DIMS[6] + de,
        4 + de,
        warchest::board::N_HEXES * (warchest::rebel::HEX_FACTS + de) + 2 * de
            + warchest::rebel::LOOSE,
    );
    let n_w = warchest::units::CARD_FEATS * dc
        + dc * de
        + warchest::units::N_UNITS * de
        + (4 + de) * de
        + xd * h
        + h * hd
        + 2 * dg * hd
        + hf * dg
        + dg * dg
        + dg * dg
        + dg * (rk + 1)
        + hd * rk
        + af * rk
        + dg * rk
        + hd * rk;
    let n_b = dc + de + de + h + hd + dg + dg + dg + (rk + 1) + 4 * rk;
    let n_ln = h + hd + h + hd;
    let w = rnd(n_w);
    let b = rnd(n_b);
    let mut ln = vec![0.0f32; n_ln];
    for i in 0..h + hd {
        ln[i] = 1.0;
    }
    (w, b, ln)
}
