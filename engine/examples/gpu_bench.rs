//! What the GPU solve service actually delivers, in the units training cares
//! about (plan B5.6).
//!
//! Training speed is rows of training data per second. A solve yields a fixed
//! handful of rows whatever `T` is — that is the point of TurboReBeL — so the
//! quantity to maximise is **solves per second**, and the quantity that
//! explains it is CFR iterations per second. Both are reported, along with the
//! GEMM row count, which is what the head's cost is proportional to.
//!
//! Workers are simulated rather than played out: each thread holds a real
//! serialized job and runs the two-trip protocol against the service in a
//! loop. That measures admission, the ticks, and both round trips — everything
//! the service owns — without spending the run's wall clock on CPU game logic
//! that this benchmark is not about.
//!
//! Usage: `gpu_bench [live] [seconds] [iters] [job-index]`
//!
//! `job-index` repeats one fixture shape, which is useful for profiling or
//! reducing a memory-checker failure. Without it the benchmark rotates over
//! the full production-shaped fixture.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use warchest::net::Mlp;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};
use warchest::serialize::Job;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (live, secs, iters) = (arg(1, 128), arg(2, 20), arg(3, 64));

    let weight_path = std::env::var("GPU_WEIGHTS").ok();
    let (dims, mut w, mut b, mut ln) = if let Some(path) = weight_path.as_deref() {
        Mlp::load_flat_bin(path).unwrap_or_else(|e| panic!("load {path}: {e}"))
    } else {
        weights()
    };
    let nets = [Nets {
        value: Mlp::from_flat(&dims, &w, &b, &ln).expect("weights"),
    }];
    eprintln!(
        "fixture weights {} dims {:?}",
        weight_path.as_deref().unwrap_or("deterministic random"),
        dims
    );
    // A spread of real trees, so the live set is as ragged as production's.
    let inner = Cfg {
        depth: 2,
        iters: 4,
        snapshots: false,
        ..Default::default()
    };
    let gc = GameCfg {
        agents: [Agent::Rebel {
            cfg: inner,
            slot: 0,
        }; 2],
        collect: Collect::None,
        explore: 0.0,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let cfg = Cfg {
        depth: 2,
        iters,
        snapshots: true,
        ..Default::default()
    };
    eprint!("building jobs... ");
    let t0 = Instant::now();
    let mut jobs: Vec<Job> = collect_roots(2, 0xB0A7, &nets, &gc, 4000)
        .into_iter()
        .step_by(7)
        .take(64)
        .map(|(state, bel)| {
            let ctx = warchest::rebel::Ctx::new(&state);
            let sv = Solver::new(&state, ctx, &nets[0], cfg, bel.clone());
            let carried = vec![[bel[0].p.clone(), bel[1].p.clone()]];
            Job::from_solver(&sv, &carried)
        })
        .filter(|j| !j.tables.leaf_rows.is_empty())
        .collect();
    assert!(!jobs.is_empty(), "no usable jobs");
    if let Some(which) = a.get(4).and_then(|s| s.parse::<usize>().ok()) {
        let job = jobs
            .get(which)
            .unwrap_or_else(|| panic!("job index {which} is out of range 0..{}", jobs.len()))
            .clone();
        eprintln!(
            "selected job {which}: nodes {} rows {} levels {} reach {} cells {}",
            job.tables.nodes,
            job.tables.rows,
            job.tables.nlevels,
            job.tables.reach_len,
            job.tables.ncells,
        );
        jobs = vec![job];
    }
    let mut sizes: Vec<usize> = jobs.iter().map(|j| j.tables.rows).collect();
    sizes.sort_unstable();
    let rows: usize = sizes.iter().sum::<usize>() / sizes.len();
    eprintln!(
        "{} jobs in {:.1}s: rows/solve mean {rows}, median {}, max {}",
        jobs.len(),
        t0.elapsed().as_secs_f64(),
        sizes[sizes.len() / 2],
        sizes[sizes.len() - 1],
    );

    // For exact speed A/Bs, keep the trained-policy fixture distribution but
    // drive both service binaries with an all-zero network. The submitted Job
    // sequence is then static, while every matmul and elementwise pass still
    // runs at the production shape and iteration count.
    if std::env::var_os("GPU_ZERO_WEIGHTS").is_some() {
        w.fill(0.0);
        b.fill(0.0);
        ln.fill(0.0);
        eprintln!("service weights all-zero");
    } else {
        eprintln!("service weights match fixture weights");
    }

    let gpu = warchest::gpu::service::spawn(0, dims, w, b, ln).expect("gpu service");

    if let Some(check) = std::env::var("GPU_BENCH_CHECK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let mut hash = 0xcbf29ce484222325u64;
        for job in jobs.iter().take(check) {
            let carried = job.carried.clone();
            let leaf = job.tables.leaf_rows[0];
            let t1 = gpu.solve(job.clone(), &carried).expect("check trip 1");
            hash_f32s(&mut hash, &t1.strategy);
            for v in &t1.root_values {
                hash_f32s(&mut hash, &v[0]);
                hash_f32s(&mut hash, &v[1]);
            }
            let t2 = gpu.carried_beliefs(t1.id, leaf).expect("check trip 2");
            for v in &t2 {
                hash_f32s(&mut hash, &v[0]);
                hash_f32s(&mut hash, &v[1]);
            }
        }
        println!(
            "answer hash   {hash:016x} over {} jobs",
            check.min(jobs.len())
        );
    }

    let done = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    std::thread::scope(|s| {
        for k in 0..live {
            let (gpu, jobs, done, stop) = (gpu.clone(), &jobs, done.clone(), stop.clone());
            s.spawn(move || {
                let mut i = k % jobs.len();
                while stop.load(Ordering::Relaxed) == 0 {
                    let job = jobs[i].clone();
                    let carried = job.carried.clone();
                    let leaf = job.tables.leaf_rows[0];
                    let Ok(t1) = gpu.solve(job, &carried) else {
                        break;
                    };
                    // The walk leaves the tree: trip 2 frees the solve.
                    let _ = gpu.carried_beliefs(t1.id, leaf);
                    done.fetch_add(1, Ordering::Relaxed);
                    i = (i + 1) % jobs.len();
                }
            });
        }
        std::thread::sleep(Duration::from_secs(secs as u64));
        stop.store(1, Ordering::Relaxed);
    });

    let el = start.elapsed().as_secs_f64();
    let n = done.load(Ordering::Relaxed) as f64;
    // A generation solve emits one training row per kept iterate, which is
    // what `snapshot_iters` keeps: log-spaced plus the final one.
    let per_solve = (iters as f64).log2().floor() as usize + 2;
    println!("live set      {live}");
    println!("iterations    {iters}");
    println!("solves        {n:.0} in {el:.1}s");
    println!("solves/sec    {:.0}", n / el);
    println!("cfr iters/sec {:.0}", n * iters as f64 / el);
    println!("gemm rows/sec {:.2e}", n * iters as f64 * rows as f64 / el);
    println!("train rows/s  {:.0}", n * per_solve as f64 / el);
    // With the `prof` feature: where the live set actually sits. A solve is
    // only advanced by a tick while it is iterating or valuing, so the split
    // between those and carry/drain is what says whether the service is
    // compute-bound or round-trip-bound.
    warchest::prof::dump_gpu();
    warchest::prof::dump();
}

fn hash_f32s(hash: &mut u64, values: &[f32]) {
    // FNV-1a over lengths and raw IEEE-754 bits. This is deliberately stricter
    // than a numerical oracle: use it only for changes which claim to preserve
    // operation order, while the CPU/GPU oracle remains the correctness gate
    // for changes which intentionally use a different floating-point backend.
    for byte in (values.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain(values.iter().flat_map(|x| x.to_bits().to_le_bytes()))
    {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

/// The shape the service is built for, with weights that are the right size
/// and otherwise arbitrary — the benchmark measures time, not values.
fn weights() -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // The classic shape, in the v3 tower format.
    let dims: Vec<usize> = vec![3, 32, 64, 64, 384, 1, 1, 64, 1, 384, 0, 0];
    let l = warchest::net::V3Layout::new(&dims).expect("dims");
    let mut rng = warchest::rng::Rng::new(7);
    let w = (0..l.w_len)
        .map(|_| (rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 0.6)
        .collect();
    let b = vec![0.0f32; l.b_len];
    let mut ln = vec![0.0f32; l.ln_len];
    for &(g, _) in &l.pub_ln {
        for x in ln[g..g + 384].iter_mut() {
            *x = 1.0;
        }
    }
    for x in ln[l.ln1.0..l.ln1.0 + l.head_in].iter_mut() {
        *x = 1.0;
    }
    (dims, w, b, ln)
}
