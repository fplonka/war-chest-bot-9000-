//! Cost-bucketed dispatcher for one v5 CUDA wave executor.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::serialize::{PackedJob, WorkVector};

use super::client::{Cmd, GpuClient, SolveResult};
use super::device::Executor;
use super::wave::Wave;

pub struct Service;

struct Pending {
    job: PackedJob,
    tag: usize,
    reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    version: u64,
    queued: Instant,
}

pub fn spawn(
    device: usize,
    dims: Vec<usize>,
    w: Vec<f32>,
    b: Vec<f32>,
    ln: Vec<f32>,
) -> Result<GpuClient, String> {
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name(format!("warchest-wave-{device}"))
        .spawn(move || match Executor::new(device, dims, w, b, ln) {
            Ok(exec) => {
                let _ = ready_tx.send(Ok(()));
                run(rx, exec);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })
        .map_err(|e| format!("start GPU wave executor: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(GpuClient::new(tx, thread)),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => {
            let _ = thread.join();
            Err("GPU wave executor died during startup".into())
        }
    }
}

fn run(rx: mpsc::Receiver<Cmd>, mut exec: Executor) {
    let row_target = env_usize("WARCHEST_WAVE_ROWS", 48 * 1024).max(1);
    let max_jobs = env_usize("WARCHEST_WAVE_JOBS", 64).clamp(1, 256);
    let latency = Duration::from_micros(env_usize("WARCHEST_WAVE_US", 800) as u64);
    let mut current_version = 0u64;
    let mut pending = VecDeque::new();
    let mut shutdown = false;

    while !shutdown || !pending.is_empty() {
        if pending.is_empty() && !shutdown {
            match rx.recv() {
                Ok(cmd) => handle(cmd, &mut exec, &mut current_version, &mut pending, &mut shutdown),
                Err(_) => shutdown = true,
            }
        }
        if shutdown && pending.is_empty() {
            break;
        }

        // Give the oldest bucket a short, bounded fill window. Publications
        // stamp only later submissions, so a wave never mixes weight ages.
        let deadline = pending
            .front()
            .map(|p| p.queued + latency)
            .unwrap_or_else(Instant::now);
        while !shutdown && Instant::now() < deadline && bucket_rows(&pending) < row_target {
            match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(cmd) => handle(cmd, &mut exec, &mut current_version, &mut pending, &mut shutdown),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }

        let Some(seed) = pending.pop_front() else {
            continue;
        };
        let class = cost_class(seed.job.work());
        let mut batch = vec![seed];
        let mut rows = batch[0].job.tables.rows;
        let mut i = 0;
        while i < pending.len() && batch.len() < max_jobs && rows < row_target {
            let take = {
                let p = &pending[i];
                p.version == batch[0].version
                    && cost_class(p.job.work()) == class
                    && Wave::compatible(&p.job, &batch[0].job)
            };
            if take {
                let p = pending.remove(i).expect("pending index");
                rows += p.job.tables.rows;
                batch.push(p);
            } else {
                i += 1;
            }
        }
        let version = batch[0].version;
        let count = batch.len();
        let mut jobs = Vec::with_capacity(count);
        let mut tickets = Vec::with_capacity(count);
        for p in batch {
            jobs.push(p.job);
            tickets.push((p.tag, p.reply));
        }
        let result = Wave::pack(&jobs).and_then(|wave| exec.solve(wave, version));
        match result {
            Ok(results) if results.len() == count => {
                for ((tag, reply), value) in tickets.into_iter().zip(results) {
                    let _ = reply.send((tag, Ok(value)));
                }
            }
            Ok(results) => {
                let e = format!(
                    "GPU wave returned {} results for {} jobs",
                    results.len(),
                    count
                );
                for (tag, reply) in tickets {
                    let _ = reply.send((tag, Err(e.clone())));
                }
            }
            Err(e) => {
                for (tag, reply) in tickets {
                    let _ = reply.send((tag, Err(e.clone())));
                }
            }
        }
    }
}

fn handle(
    cmd: Cmd,
    exec: &mut Executor,
    current_version: &mut u64,
    pending: &mut VecDeque<Pending>,
    shutdown: &mut bool,
) {
    match cmd {
        Cmd::Submit { job, tag, reply } => pending.push_back(Pending {
            job,
            tag,
            reply,
            version: *current_version,
            queued: Instant::now(),
        }),
        Cmd::Publish {
            version,
            dims,
            w,
            b,
            ln,
        } => match exec.publish(version, dims, w, b, ln) {
            Ok(()) => *current_version = version,
            Err(e) => eprintln!("GPU weight publication {version} failed: {e}"),
        },
        Cmd::Shutdown => *shutdown = true,
    }
}

fn bucket_rows(pending: &VecDeque<Pending>) -> usize {
    let Some(seed) = pending.front() else {
        return 0;
    };
    let class = cost_class(seed.job.work());
    pending
        .iter()
        .filter(|p| {
            p.version == seed.version
                && cost_class(p.job.work()) == class
                && Wave::compatible(&p.job, &seed.job)
        })
        .map(|p| p.job.tables.rows)
        .sum()
}

/// Logarithmic cost classes keep a whale from setting every common wave's
/// capacity while preserving oldest-first service within each compatible
/// bucket. Class 31 is exclusive.
fn cost_class(w: WorkVector) -> u8 {
    let bytes = w
        .table_bytes
        .saturating_add(w.mutable_bytes)
        .saturating_add(w.carried_output_bytes);
    if bytes >= (2usize << 30) || w.legal_cells >= 8_000_000 {
        return 31;
    }
    let work = w
        .network_rows
        .max(w.legal_cells / 8)
        .max(w.reach_slots / 16)
        .max(w.reverse_nonzeros / 16)
        .max(1);
    (usize::BITS - work.leading_zeros()) as u8
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}
