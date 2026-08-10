//! Cost-bucketed dispatcher for one v5 CUDA device and its reusable lanes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::serialize::{PackedJob, WorkVector};

use super::client::{Cmd, GpuClient, SolveResult};
use super::device::Executor;
use super::wave::Wave;

pub struct Service;

struct Pending {
    job: Arc<PackedJob>,
    work: WorkVector,
    tag: usize,
    reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    version: u64,
    cost: u64,
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
    let queued_work = Arc::new(AtomicU64::new(0));
    let service_work = queued_work.clone();
    let thread = std::thread::Builder::new()
        .name(format!("warchest-dispatch-{device}"))
        .spawn(move || dispatcher(device, dims, w, b, ln, rx, ready_tx, service_work))
        .map_err(|e| format!("start GPU wave executor: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(GpuClient::new(tx, thread, queued_work)),
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

fn dispatcher(
    device: usize,
    dims: Vec<usize>,
    w: Vec<f32>,
    b: Vec<f32>,
    ln: Vec<f32>,
    rx: mpsc::Receiver<Cmd>,
    ready: mpsc::Sender<Result<(), String>>,
    queued_work: Arc<AtomicU64>,
) {
    let lanes = env_usize("WARCHEST_WAVE_LANES", 2).clamp(1, 5);
    let whale_lanes = env_usize("WARCHEST_WAVE_WHALE_LANES", 1).clamp(1, lanes);
    let (lane_ready_tx, lane_ready_rx) = mpsc::channel();
    let mut senders = Vec::with_capacity(lanes);
    let mut joins = Vec::with_capacity(lanes);
    let mut lane_work = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let (tx, lane_rx) = mpsc::channel();
        let lane_ready = lane_ready_tx.clone();
        let global = queued_work.clone();
        let local = Arc::new(AtomicU64::new(0));
        let local_run = local.clone();
        let (ldims, lw, lb, lln) = (dims.clone(), w.clone(), b.clone(), ln.clone());
        match std::thread::Builder::new()
            .name(format!("warchest-wave-{device}-{lane}"))
            .spawn(move || match Executor::new(device, ldims, lw, lb, lln) {
                Ok(exec) => {
                    let _ = lane_ready.send(Ok(()));
                    run(device, lane, lane_rx, exec, global, local_run);
                }
                Err(e) => {
                    let _ = lane_ready.send(Err(e));
                }
            }) {
            Ok(join) => {
                senders.push(tx);
                joins.push(join);
                lane_work.push(local);
            }
            Err(e) => {
                let _ = ready.send(Err(format!("start GPU lane {lane}: {e}")));
                for tx in &senders {
                    let _ = tx.send(Cmd::Shutdown);
                }
                for join in joins {
                    let _ = join.join();
                }
                return;
            }
        }
    }
    drop(lane_ready_tx);
    for lane in 0..lanes {
        match lane_ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = ready.send(Err(format!("GPU lane {lane}: {e}")));
                for tx in &senders {
                    let _ = tx.send(Cmd::Shutdown);
                }
                for join in joins {
                    let _ = join.join();
                }
                return;
            }
            Err(_) => {
                let _ = ready.send(Err("GPU lane died during startup".into()));
                return;
            }
        }
    }
    let _ = ready.send(Ok(()));

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Submit {
                job,
                work,
                tag,
                cost,
                reply,
            } => {
                if work.requires_card_exclusive_route() {
                    // A 4 GiB contiguous mutable arena, or a 6 GiB combined
                    // reservation, cannot reliably coexist with the ordinary
                    // retained buffers and trainer on a 24 GiB card. This tail
                    // is rare: drain and trim the card, run it on lane 0, and
                    // trim again before admission resumes. Common whales made
                    // from a smaller arena plus tables stay lane-local.
                    while lane_work.iter().any(|x| x.load(Ordering::Acquire) != 0) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    if let Err(e) = trim_lanes(&senders) {
                        queued_work.fetch_sub(cost, Ordering::Relaxed);
                        let _ = reply.send((tag, Err(e)));
                        continue;
                    }
                    lane_work[0].fetch_add(cost, Ordering::Release);
                    if let Err(e) = senders[0].send(Cmd::Submit {
                        job,
                        work,
                        tag,
                        cost,
                        reply,
                    }) {
                        lane_work[0].fetch_sub(cost, Ordering::Release);
                        queued_work.fetch_sub(cost, Ordering::Relaxed);
                        if let Cmd::Submit { tag, reply, .. } = e.0 {
                            let _ = reply.send((tag, Err("GPU giant-wave lane is gone".into())));
                        }
                        continue;
                    }
                    while lane_work[0].load(Ordering::Acquire) != 0 {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    continue;
                }
                // Keep common 4 GiB whales on a bounded set of lanes. Sending
                // each one to the currently emptiest lane eventually leaves
                // every lane retaining a whale-sized buffer, which exhausts a
                // 24 GiB card even though every individual solve fits. One is
                // the safe default; machines with headroom can opt into more.
                // Ordinary work can still use these lanes while they are idle.
                let lane = dispatch_lane(&lane_work, work.requires_exclusive_route(), whale_lanes);
                lane_work[lane].fetch_add(cost, Ordering::Relaxed);
                let cmd = Cmd::Submit {
                    job,
                    work,
                    tag,
                    cost,
                    reply,
                };
                if let Err(e) = senders[lane].send(cmd) {
                    lane_work[lane].fetch_sub(cost, Ordering::Relaxed);
                    queued_work.fetch_sub(cost, Ordering::Relaxed);
                    if let Cmd::Submit { tag, reply, .. } = e.0 {
                        let _ = reply.send((tag, Err(format!("GPU lane {lane} is gone"))));
                    }
                }
            }
            Cmd::Publish {
                version,
                dims,
                w,
                b,
                ln,
            } => {
                for tx in &senders {
                    let _ = tx.send(Cmd::Publish {
                        version,
                        dims: dims.clone(),
                        w: w.clone(),
                        b: b.clone(),
                        ln: ln.clone(),
                    });
                }
            }
            Cmd::Trim { ready } => {
                let _ = ready.send(Err("trim is an internal lane command".into()));
            }
            Cmd::Shutdown => break,
        }
    }
    for tx in &senders {
        let _ = tx.send(Cmd::Shutdown);
    }
    for join in joins {
        let _ = join.join();
    }
}

fn trim_lanes(senders: &[mpsc::Sender<Cmd>]) -> Result<(), String> {
    for tx in senders {
        let (done_tx, done_rx) = mpsc::sync_channel(0);
        tx.send(Cmd::Trim { ready: done_tx })
            .map_err(|_| "GPU lane is gone during giant-wave trim".to_string())?;
        done_rx
            .recv()
            .map_err(|_| "GPU lane died during giant-wave trim".to_string())??;
    }
    Ok(())
}

fn run(
    device: usize,
    lane: usize,
    rx: mpsc::Receiver<Cmd>,
    mut exec: Executor,
    queued_work: Arc<AtomicU64>,
    lane_work: Arc<AtomicU64>,
) {
    let row_target = env_usize("WARCHEST_WAVE_ROWS", 48 * 1024).max(1);
    let max_jobs = env_usize("WARCHEST_WAVE_JOBS", 64).clamp(1, 256);
    let latency = Duration::from_micros(env_usize("WARCHEST_WAVE_US", 800) as u64);
    let profile = std::env::var_os("WARCHEST_GPU_PROFILE").is_some();
    let mut current_version = 0u64;
    let mut pending = VecDeque::new();
    let mut shutdown = false;

    while !shutdown || !pending.is_empty() {
        if pending.is_empty() && !shutdown {
            match rx.recv() {
                Ok(cmd) => handle(
                    cmd,
                    &mut exec,
                    &mut current_version,
                    &mut pending,
                    &mut shutdown,
                ),
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
                Ok(cmd) => handle(
                    cmd,
                    &mut exec,
                    &mut current_version,
                    &mut pending,
                    &mut shutdown,
                ),
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
        let class = cost_class(seed.work);
        let mut batch = vec![seed];
        let mut rows = batch[0].work.network_rows;
        let mut i = 0;
        while class != 31 && i < pending.len() && batch.len() < max_jobs && rows < row_target {
            let take = {
                let p = &pending[i];
                p.version == batch[0].version
                    && cost_class(p.work) == class
                    && Wave::compatible(&p.job, &batch[0].job)
            };
            if take {
                let p = pending.remove(i).expect("pending index");
                rows += p.work.network_rows;
                batch.push(p);
            } else {
                i += 1;
            }
        }
        let version = batch[0].version;
        let count = batch.len();
        let profile_shape = profile.then(|| {
            let cells = batch.iter().map(|p| p.work.legal_cells).sum::<usize>();
            let reach = batch.iter().map(|p| p.work.reach_slots).sum::<usize>();
            let reverse = batch.iter().map(|p| p.work.reverse_nonzeros).sum::<usize>();
            let table_bytes = batch.iter().map(|p| p.work.table_bytes).sum::<usize>();
            let mutable_bytes = batch.iter().map(|p| p.work.mutable_bytes).sum::<usize>();
            let max_bytes = batch
                .iter()
                .map(|p| {
                    p.work
                        .table_bytes
                        .saturating_add(p.work.mutable_bytes)
                        .saturating_add(p.work.carried_output_bytes)
                })
                .max()
                .unwrap_or(0);
            let oldest_ms = batch
                .iter()
                .map(|p| p.queued.elapsed().as_secs_f64() * 1e3)
                .fold(0.0f64, f64::max);
            (
                cells,
                reach,
                reverse,
                table_bytes,
                mutable_bytes,
                max_bytes,
                oldest_ms,
            )
        });
        let mut jobs = Vec::with_capacity(count);
        let mut tickets = Vec::with_capacity(count);
        for p in batch {
            jobs.push(p.job);
            tickets.push((
                p.tag,
                p.reply,
                p.cost,
                p.work.requires_exclusive_route(),
                p.work.requires_card_exclusive_route(),
            ));
        }
        let pack_started = Instant::now();
        let packed = Wave::pack(&jobs);
        let packed_at = Instant::now();
        let result = packed.and_then(|wave| exec.solve(wave, version));
        let result = if tickets[0].4 {
            result.and_then(|values| {
                exec.trim()?;
                Ok(values)
            })
        } else {
            result
        };
        if let Some((cells, reach, reverse, table_bytes, mutable_bytes, max_bytes, oldest_ms)) =
            profile_shape
        {
            eprintln!(
                "v5_service device={device} lane={lane} class={class} jobs={count} rows={rows} cells={cells} reach={reach} reverse={reverse} table_mib={:.1} mutable_mib={:.1} max_job_mib={:.1} oldest_ms={oldest_ms:.2} pack_ms={:.2} solve_ms={:.2}",
                table_bytes as f64 / 1048576.0,
                mutable_bytes as f64 / 1048576.0,
                max_bytes as f64 / 1048576.0,
                1e3 * (packed_at - pack_started).as_secs_f64(),
                1e3 * packed_at.elapsed().as_secs_f64(),
            );
        }
        match result {
            Ok(results) if results.len() == count => {
                for ((tag, reply, cost, oversize, card), mut value) in
                    tickets.into_iter().zip(results)
                {
                    value.oversize_route = oversize;
                    value.card_exclusive_route = card;
                    let _ = reply.send((tag, Ok(value)));
                    queued_work.fetch_sub(cost, Ordering::Relaxed);
                    lane_work.fetch_sub(cost, Ordering::Relaxed);
                }
            }
            Ok(results) => {
                let e = format!(
                    "GPU wave returned {} results for {} jobs",
                    results.len(),
                    count
                );
                for (tag, reply, cost, _, _) in tickets {
                    let _ = reply.send((tag, Err(e.clone())));
                    queued_work.fetch_sub(cost, Ordering::Relaxed);
                    lane_work.fetch_sub(cost, Ordering::Relaxed);
                }
            }
            Err(e) => {
                for (tag, reply, cost, _, _) in tickets {
                    let _ = reply.send((tag, Err(e.clone())));
                    queued_work.fetch_sub(cost, Ordering::Relaxed);
                    lane_work.fetch_sub(cost, Ordering::Relaxed);
                }
            }
        }
        retain_needed_banks(&mut exec, current_version, &pending);
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
        Cmd::Submit {
            job,
            work,
            tag,
            cost,
            reply,
        } => pending.push_back(Pending {
            job,
            work,
            tag,
            reply,
            version: *current_version,
            queued: Instant::now(),
            cost,
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
        Cmd::Trim { ready } => {
            let _ = ready.send(exec.trim());
        }
        Cmd::Shutdown => *shutdown = true,
    }
    retain_needed_banks(exec, *current_version, pending);
}

fn retain_needed_banks(exec: &mut Executor, current: u64, pending: &VecDeque<Pending>) {
    let mut keep = Vec::with_capacity(pending.len() + 1);
    keep.push(current);
    keep.extend(pending.iter().map(|p| p.version));
    keep.sort_unstable();
    keep.dedup();
    exec.retain_weight_versions(&keep);
}

fn bucket_rows(pending: &VecDeque<Pending>) -> usize {
    let Some(seed) = pending.front() else {
        return 0;
    };
    let class = cost_class(seed.work);
    pending
        .iter()
        .filter(|p| {
            p.version == seed.version
                && cost_class(p.work) == class
                && Wave::compatible(&p.job, &seed.job)
        })
        .map(|p| p.work.network_rows)
        .sum()
}

/// Coarse physical capacity classes keep a whale from setting every common
/// wave's shape without splitting an ordinary production tape into tiny
/// power-of-two batches. Class 31 is isolated to one job and one lane.
fn cost_class(w: WorkVector) -> u8 {
    let bytes = w
        .table_bytes
        .saturating_add(w.mutable_bytes)
        .saturating_add(w.carried_output_bytes);
    if w.requires_exclusive_route() || w.legal_cells >= 8_000_000 {
        return 31;
    }
    let work = w
        .network_rows
        .max(w.legal_cells / 8)
        .max(w.reach_slots / 16)
        .max(w.reverse_nonzeros / 16)
        .max(1);
    if bytes >= (512usize << 20) || work >= 4_000_000 {
        3
    } else if bytes >= (64usize << 20) || work >= 500_000 {
        2
    } else if bytes >= (16usize << 20) || work >= 125_000 {
        1
    } else {
        0
    }
}

fn dispatch_lane(lane_work: &[Arc<AtomicU64>], whale: bool, whale_lanes: usize) -> usize {
    let eligible = if whale {
        &lane_work[..whale_lanes.min(lane_work.len())]
    } else {
        lane_work
    };
    eligible
        .iter()
        .enumerate()
        .min_by_key(|(_, x)| x.load(Ordering::Relaxed))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whales_stay_on_their_bounded_lane_set() {
        let lanes = [9, 2, 5]
            .into_iter()
            .map(|x| Arc::new(AtomicU64::new(x)))
            .collect::<Vec<_>>();
        assert_eq!(dispatch_lane(&lanes, true, 1), 0);
        assert_eq!(dispatch_lane(&lanes, true, 2), 1);
        assert_eq!(dispatch_lane(&lanes, false, 1), 1);
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}
