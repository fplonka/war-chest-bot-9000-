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
//! Usage: `gpu_bench [live] [seconds] [iters]`

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};
use warchest::serialize::Job;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let (live, secs, iters) = (arg(1, 128), arg(2, 20), arg(3, 64));

    // A spread of real trees, so the live set is as ragged as production's.
    let nets = [Nets::default()];
    let inner = Cfg { depth: 2, iters: 4, snapshots: false, ..Default::default() };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg: inner, slot: 0 }; 2],
        collect: Collect::None,
        explore: 0.0,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let cfg = Cfg { depth: 2, iters, snapshots: true, ..Default::default() };
    eprint!("building jobs... ");
    let t0 = Instant::now();
    let jobs: Vec<Job> = collect_roots(2, 0xB0A7, &nets, &gc, 4000)
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

    let (dims, w, b, ln) = weights();
    let gpu = warchest::gpu::service::spawn(dims, w, b, ln).expect("gpu service");

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
                    let Ok(t1) = gpu.solve(job, &carried) else { break };
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
}

/// The shape the service is built for, with weights that are the right size
/// and otherwise arbitrary — the benchmark measures time, not values.
fn weights() -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dims: Vec<usize> = vec![
        warchest::rebel::PUBFEAT, 384, 384, warchest::rebel::CFEAT, 64, 64,
        warchest::rebel::AFEAT, 32, 64, 0,
    ];
    let (ow, ob, oln) = warchest::gpu::service::weight_offsets(&dims).expect("dims");
    let mut rng = warchest::rng::Rng::new(7);
    let w = (0..*ow.last().unwrap())
        .map(|_| (rng.next_u64() as f32 / u64::MAX as f32 - 0.5) * 0.6)
        .collect();
    let b = vec![0.0f32; *ob.last().unwrap()];
    let mut ln = vec![0.0f32; *oln.last().unwrap()];
    for (i, g) in ln.iter_mut().enumerate() {
        // The two LayerNorm gains are the first and third blocks.
        if i < dims[1] || (i >= 2 * dims[1] && i < 2 * dims[1] + dims[2]) {
            *g = 1.0;
        }
    }
    (dims, w, b, ln)
}
