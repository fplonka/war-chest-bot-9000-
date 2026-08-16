//! Deterministic throughput gate for the v5 contiguous-wave executor.
//!
//! Usage: `wave_tape <weights.bin> <roots.bin> [roots=256] [seconds=30]`
//!
//! The root sample supplies the measured production distribution.  Every job
//! uses depth 2, 64 linear-CFR iterations, the production snapshot schedule,
//! and eight Phase-2 roots (seven carried averages plus the live belief).  The
//! timer excludes executor/NVRTC startup, but includes queue fill and drain,
//! wave packing, transfers, graph capture/launch, and result materialisation.

#[cfg(not(feature = "gpu"))]
compile_error!("wave_tape requires --features gpu");

use std::collections::VecDeque;
use std::io::BufReader;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use warchest::gpu::client::PreparedJob;
use warchest::net::Net;
use warchest::rebel::Ctx;
use warchest::roots;
use warchest::search::{Cfg, Cfr, Nets, Solver};
use warchest::serialize::{IndexWidth, PackedJob, WorkVector};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weights = args.get(1).expect("weights.bin");
    let roots_path = args.get(2).expect("roots.bin");
    let root_limit = arg(&args, 3, 256);
    let seconds = arg(&args, 4, 30);

    let (dims, w, b, ln) = Net::load_flat_bin(weights).expect("weights");
    let net = Net::from_flat(&dims, &w, &b, &ln).expect("network");
    let nets = Nets { value: net };
    let root_file = std::fs::File::open(roots_path).expect("roots file");
    let roots = roots::read_roots(&mut BufReader::new(root_file)).expect("roots");
    let cfg = Cfg {
        depth: 2,
        iters: 64,
        snapshots: true,
        cfr: Cfr::LINEAR,
        node_cap: 200_000,
        gpu_build: true,
        keep_states: false,
    };

    let build0 = Instant::now();
    let mut jobs = Vec::new();
    let mut capped = 0usize;
    for (state, belief) in roots.into_iter().take(root_limit) {
        let sv = Solver::new(&state, Ctx::new(&state), &nets, cfg, belief.clone());
        if sv.capped() {
            capped += 1;
            continue;
        }
        let live = [belief[0].p.clone(), belief[1].p.clone()];
        let carried = vec![live; 8];
        jobs.push(PackedJob::from_solver(&sv, &carried));
    }
    assert!(!jobs.is_empty(), "the tape produced no valid jobs");
    let build_s = build0.elapsed().as_secs_f64();
    report_tape(&jobs, capped, build_s);
    if std::env::var_os("WARCHEST_TAPE_LARGEST_FIRST").is_some() {
        jobs.sort_by_key(|job| std::cmp::Reverse(job.work().mutable_bytes));
    }
    let jobs: Vec<_> = jobs.into_iter().map(PreparedJob::new).collect();

    let devices: Vec<usize> = std::env::var("WARCHEST_TAPE_DEVICES")
        .unwrap_or_else(|_| "0".into())
        .split(',')
        .map(|x| x.trim().parse().expect("CUDA device"))
        .collect();
    assert!(!devices.is_empty(), "WARCHEST_TAPE_DEVICES is empty");
    let gpus: Vec<_> = devices
        .iter()
        .map(|&device| {
            // Production GEMM path: the tape measures what training runs, not
            // the oracle-comparison SGEMM.
            warchest::gpu::service::spawn(
                device,
                dims.clone(),
                w.clone(),
                b.clone(),
                ln.clone(),
                false,
            )
            .expect("GPU executor")
        })
        .collect();
    // Warm up cuBLAS, NVRTC, graph creation, and the allocator outside the
    // tape timer.  The real trainer creates its services before the ReBeL
    // clock starts as well.
    for (device, gpu) in gpus.iter().enumerate() {
        gpu.submit_prepared(jobs[device % jobs.len()].clone())
            .expect("warm submit")
            .wait()
            .expect("warm solve");
    }

    let queue = std::env::var("WARCHEST_TAPE_QUEUE")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(128usize)
        .max(1);
    let producers = std::env::var("WARCHEST_TAPE_PRODUCERS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(1usize)
        .clamp(1, queue);
    let producer_queue = queue.div_ceil(producers);
    let deadline = Duration::from_secs(seconds as u64);
    let barrier = Arc::new(Barrier::new(producers + 1));
    let (completed, elapsed) = std::thread::scope(|scope| {
        let mut threads = Vec::with_capacity(producers);
        for producer in 0..producers {
            let gpus = &gpus;
            let jobs = &jobs;
            let barrier = barrier.clone();
            threads.push(scope.spawn(move || {
                let mut submitted = producer;
                let mut completed = 0usize;
                let mut pending = VecDeque::new();
                barrier.wait();
                let start = Instant::now();
                while pending.len() < producer_queue {
                    let gpu = gpus
                        .iter()
                        .min_by_key(|gpu| gpu.queued_work())
                        .expect("GPU service");
                    let h = gpu
                        .submit_prepared(jobs[submitted % jobs.len()].clone())
                        .expect("submit");
                    pending.push_back(h);
                    submitted += producers;
                }
                while start.elapsed() < deadline {
                    pending.pop_front().expect("pending").wait().expect("solve");
                    completed += 1;
                    let gpu = gpus
                        .iter()
                        .min_by_key(|gpu| gpu.queued_work())
                        .expect("GPU service");
                    let h = gpu
                        .submit_prepared(jobs[submitted % jobs.len()].clone())
                        .expect("submit");
                    pending.push_back(h);
                    submitted += producers;
                }
                while let Some(h) = pending.pop_front() {
                    h.wait().expect("drain");
                    completed += 1;
                }
                (completed, start.elapsed().as_secs_f64())
            }));
        }
        barrier.wait();
        threads
            .into_iter()
            .map(|h| h.join().expect("tape producer"))
            .fold((0usize, 0.0f64), |(cn, et), (c, e)| (cn + c, et.max(e)))
    });
    println!(
        "completed={completed} elapsed_s={elapsed:.3} solves_per_s={:.1} devices={} queue={queue} producers={producers}",
        completed as f64 / elapsed,
        devices.len(),
    );
}

fn arg(args: &[String], at: usize, default: usize) -> usize {
    args.get(at)
        .map(|x| x.parse().expect("integer argument"))
        .unwrap_or(default)
}

fn report_tape(jobs: &[PackedJob], capped: usize, build_s: f64) {
    let narrow = jobs
        .iter()
        .filter(|j| j.index_width() == IndexWidth::Narrow)
        .count();
    let mut schedules = std::collections::BTreeMap::new();
    for j in jobs {
        *schedules.entry(j.meta.snap_iters.clone()).or_insert(0usize) += 1;
    }
    let mut rows: Vec<_> = jobs.iter().map(|j| j.work().network_rows).collect();
    let mut cells: Vec<_> = jobs.iter().map(|j| j.work().legal_cells).collect();
    let mut bytes: Vec<_> = jobs
        .iter()
        .map(|j| {
            let WorkVector {
                table_bytes,
                mutable_bytes,
                carried_output_bytes,
                ..
            } = j.work();
            table_bytes + mutable_bytes + carried_output_bytes
        })
        .collect();
    let mut legal_widths = Vec::new();
    let mut child_widths = Vec::new();
    let mut draw_widths = Vec::new();
    for job in jobs {
        legal_widths.extend(
            job.tables
                .legal_off
                .windows(2)
                .map(|x| (x[1] - x[0]) as usize),
        );
        child_widths.extend(
            job.tables
                .node_child_start
                .windows(2)
                .map(|x| (x[1] - x[0]) as usize),
        );
        for (node, &kind) in job.tables.node_kind.iter().enumerate() {
            if kind != 1 {
                continue;
            }
            let lo = job.tables.draw_row_off[node] as usize;
            let hi = job.tables.draw_row_off[node + 1] as usize;
            draw_widths.extend(
                job.tables.draw_row_start[lo..hi]
                    .windows(2)
                    .map(|x| (x[1] - x[0]) as usize),
            );
        }
    }
    rows.sort_unstable();
    cells.sort_unstable();
    bytes.sort_unstable();
    legal_widths.sort_unstable();
    child_widths.sort_unstable();
    draw_widths.sort_unstable();
    println!(
        "tape_jobs={} node_caps={} build_s={build_s:.3} build_ms_per_job={:.2}",
        jobs.len(),
        capped,
        1e3 * build_s / (jobs.len() + capped) as f64,
    );
    println!(
        "narrow/wide={}/{} snapshot_schedules={schedules:?}",
        narrow,
        jobs.len() - narrow,
    );
    println!(
        "network_rows p50/p95/p99/max={}/{}/{}/{}",
        pct(&rows, 0.50),
        pct(&rows, 0.95),
        pct(&rows, 0.99),
        rows[rows.len() - 1],
    );
    println!(
        "legal_cells p50/p95/p99/max={}/{}/{}/{}",
        pct(&cells, 0.50),
        pct(&cells, 0.95),
        pct(&cells, 0.99),
        cells[cells.len() - 1],
    );
    println!(
        "owned_MiB p50/p95/p99/max={:.1}/{:.1}/{:.1}/{:.1}",
        pct(&bytes, 0.50) as f64 / 1048576.0,
        pct(&bytes, 0.95) as f64 / 1048576.0,
        pct(&bytes, 0.99) as f64 / 1048576.0,
        bytes[bytes.len() - 1] as f64 / 1048576.0,
    );
    report_width("legal_cells_per_config", &legal_widths);
    report_width("children_per_node", &child_widths);
    report_width("draw_entries_per_config", &draw_widths);
}

fn report_width(name: &str, values: &[usize]) {
    if values.is_empty() {
        return;
    }
    println!(
        "{name} p50/p95/p99/max={}/{}/{}/{}",
        pct(values, 0.50),
        pct(values, 0.95),
        pct(values, 0.99),
        values[values.len() - 1],
    );
}

fn pct(sorted: &[usize], q: f64) -> usize {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}
