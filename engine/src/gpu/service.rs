//! Cost-bucketed dispatcher for one v5 CUDA device and its reusable lanes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::serialize::{PackedJob, WorkVector};

use super::client::{Cmd, GpuClient, SolveResult};
use super::device::{Blas, Executor};
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

struct HeldPending {
    route_id: u64,
    pending: Pending,
}

struct LaneState {
    blocked: Vec<bool>,
    route_refs: Vec<usize>,
    guard_refs: Vec<usize>,
}

struct RouteTicket {
    route_id: u64,
    target: usize,
    work: WorkVector,
    cost: u64,
    held_ready: mpsc::Receiver<()>,
}

enum RouteCmd {
    Run(RouteTicket),
    Shutdown,
}

/// Start one device's wave service. `precise` forces plain SGEMM instead of
/// the tensor-core head: the CPU-oracle tests need exact math to compare
/// against, production wants the four-times rate.
pub fn spawn(
    device: usize,
    dims: Vec<usize>,
    w: Vec<f32>,
    b: Vec<f32>,
    ln: Vec<f32>,
    precise: bool,
) -> Result<GpuClient, String> {
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let queued_work = Arc::new(AtomicU64::new(0));
    let service_work = queued_work.clone();
    let thread = std::thread::Builder::new()
        .name(format!("warchest-dispatch-{device}"))
        .spawn(move || dispatcher(device, dims, w, b, ln, precise, rx, ready_tx, service_work))
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
    precise: bool,
    rx: mpsc::Receiver<Cmd>,
    ready: mpsc::Sender<Result<(), String>>,
    queued_work: Arc<AtomicU64>,
) {
    // Per-device, because one card also runs the trainer: with ten solve
    // lanes beside it an optimizer step costs about 240 ms against 72-101 ms
    // measured alone, and that contention comes straight back out of
    // generation. `WARCHEST_WAVE_LANES=12,6` gives the free card more lanes.
    let lanes = env_list("WARCHEST_WAVE_LANES", device, 2).clamp(1, 12);
    let whale_lanes = env_usize("WARCHEST_WAVE_WHALE_LANES", 1).clamp(1, lanes);
    let route_profile = std::env::var_os("WARCHEST_ROUTE_PROFILE").is_some();
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
            .spawn(move || {
                match Executor::new(device, ldims, lw, lb, lln, |h| Blas::new(h, precise)) {
                    Ok(exec) => {
                        let _ = lane_ready.send(Ok(()));
                        run(device, lane, lane_rx, exec, global, local_run);
                    }
                    Err(e) => {
                        let _ = lane_ready.send(Err(e));
                    }
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
    let lane_state = Arc::new(Mutex::new(LaneState {
        blocked: vec![false; lanes],
        route_refs: vec![0; lanes],
        guard_refs: vec![0; lanes],
    }));
    let (route_tx, route_rx) = mpsc::channel();
    let route_senders = senders.clone();
    let route_work = lane_work.clone();
    let route_state = lane_state.clone();
    let route_global = queued_work.clone();
    let route_join = match std::thread::Builder::new()
        .name(format!("warchest-routes-{device}"))
        .spawn(move || {
            run_routes(
                device,
                route_rx,
                route_senders,
                route_work,
                route_state,
                route_global,
                route_profile,
            )
        }) {
        Ok(join) => join,
        Err(e) => {
            let _ = ready.send(Err(format!("start GPU route worker: {e}")));
            for tx in &senders {
                let _ = tx.send(Cmd::Shutdown);
            }
            for join in joins {
                let _ = join.join();
            }
            return;
        }
    };
    let _ = ready.send(Ok(()));

    let mut deferred = VecDeque::new();
    let mut next_route_id = 1u64;
    loop {
        flush_deferred(
            &mut deferred,
            &senders,
            &lane_work,
            &lane_state,
            whale_lanes,
            &queued_work,
        );
        let cmd = if deferred.is_empty() {
            match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            }
        } else {
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(cmd) => cmd,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        };
        match cmd {
            Cmd::Submit {
                job,
                work,
                tag,
                cost,
                reply,
            } => {
                if work.requires_arena_guard_route() || work.requires_card_exclusive_route() {
                    // Put the job on its whale lane now so it records the
                    // weight version at submission order. A separate worker
                    // drains and trims the bounded lane set; meanwhile the
                    // dispatcher keeps the other lanes fed.
                    let route_id = next_route_id;
                    next_route_id = next_route_id.wrapping_add(1);
                    let target = claim_route_target(&lane_work, &lane_state, whale_lanes);
                    // More than one route can be held on the same whale lane.
                    // Buffer the acknowledgement so a later hold cannot park
                    // the lane ahead of an earlier route's release command.
                    let (held_tx, held_rx) = mpsc::sync_channel(1);
                    let hold = Cmd::Hold {
                        route_id,
                        job,
                        work,
                        tag,
                        cost,
                        reply,
                        ready: held_tx,
                    };
                    if let Err(e) = senders[target].send(hold) {
                        release_route_target(&lane_state, target);
                        queued_work.fetch_sub(cost, Ordering::Relaxed);
                        if let Cmd::Hold { tag, reply, .. } = e.0 {
                            let _ = reply.send((tag, Err("GPU guarded-wave lane is gone".into())));
                        }
                        continue;
                    }
                    let ticket = RouteTicket {
                        route_id,
                        target,
                        work,
                        cost,
                        held_ready: held_rx,
                    };
                    if let Err(e) = route_tx.send(RouteCmd::Run(ticket)) {
                        let RouteCmd::Run(ticket) = e.0 else {
                            unreachable!()
                        };
                        let _ = senders[target].send(Cmd::CancelHeld {
                            route_id: ticket.route_id,
                            error: "GPU route worker is gone".into(),
                        });
                        release_route_target(&lane_state, target);
                    }
                    continue;
                }
                deferred.push_back(Cmd::Submit {
                    job,
                    work,
                    tag,
                    cost,
                    reply,
                });
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
            Cmd::Hold {
                tag, reply, ready, ..
            } => {
                let _ = ready.send(());
                let _ = reply.send((tag, Err("hold is an internal lane command".into())));
            }
            Cmd::ReleaseHeld { .. } | Cmd::CancelHeld { .. } => {}
            Cmd::Shutdown => break,
        }
    }
    let _ = route_tx.send(RouteCmd::Shutdown);
    let _ = route_join.join();
    {
        let mut state = lane_state.lock().unwrap_or_else(|e| e.into_inner());
        state.blocked.fill(false);
        state.route_refs.fill(0);
        state.guard_refs.fill(0);
    }
    while !deferred.is_empty() {
        flush_deferred(
            &mut deferred,
            &senders,
            &lane_work,
            &lane_state,
            whale_lanes,
            &queued_work,
        );
    }
    for tx in &senders {
        let _ = tx.send(Cmd::Shutdown);
    }
    for join in joins {
        let _ = join.join();
    }
}

fn flush_deferred(
    deferred: &mut VecDeque<Cmd>,
    senders: &[mpsc::Sender<Cmd>],
    lane_work: &[Arc<AtomicU64>],
    lane_state: &Arc<Mutex<LaneState>>,
    whale_lanes: usize,
    queued_work: &Arc<AtomicU64>,
) {
    let mut i = 0;
    while i < deferred.len() {
        let (whale, cost) = match &deferred[i] {
            Cmd::Submit { work, cost, .. } => (work.requires_exclusive_route(), *cost),
            _ => {
                i += 1;
                continue;
            }
        };
        let lane = {
            let state = lane_state.lock().unwrap_or_else(|e| e.into_inner());
            let end = if whale {
                whale_lanes.min(lane_work.len())
            } else {
                lane_work.len()
            };
            let selected = (0..end)
                .filter(|&lane| !state.blocked[lane])
                .min_by_key(|&lane| lane_work[lane].load(Ordering::Relaxed));
            if let Some(lane) = selected {
                // Blocking a lane and assigning work both take this mutex, so
                // a route cannot observe an empty lane between selection and
                // accounting for this submission.
                lane_work[lane].fetch_add(cost, Ordering::Relaxed);
            }
            selected
        };
        let Some(lane) = lane else {
            i += 1;
            continue;
        };
        let cmd = deferred.remove(i).expect("deferred submission index");
        if let Err(e) = senders[lane].send(cmd) {
            lane_work[lane].fetch_sub(cost, Ordering::Relaxed);
            queued_work.fetch_sub(cost, Ordering::Relaxed);
            if let Cmd::Submit { tag, reply, .. } = e.0 {
                let _ = reply.send((tag, Err(format!("GPU lane {lane} is gone"))));
            }
        }
    }
}

fn claim_route_target(
    lane_work: &[Arc<AtomicU64>],
    lane_state: &Arc<Mutex<LaneState>>,
    whale_lanes: usize,
) -> usize {
    let mut state = lane_state.lock().unwrap_or_else(|e| e.into_inner());
    let end = whale_lanes.min(lane_work.len()).max(1);
    let target = (0..end)
        .min_by_key(|&lane| {
            (
                state.route_refs[lane],
                lane_work[lane].load(Ordering::Relaxed),
            )
        })
        .unwrap_or(0);
    state.route_refs[target] += 1;
    state.blocked[target] = true;
    target
}

fn release_route_target(lane_state: &Arc<Mutex<LaneState>>, target: usize) {
    let mut state = lane_state.lock().unwrap_or_else(|e| e.into_inner());
    state.route_refs[target] = state.route_refs[target].saturating_sub(1);
    state.blocked[target] = state.route_refs[target] != 0 || state.guard_refs[target] != 0;
}

fn claim_guard_lanes(
    lane_work: &[Arc<AtomicU64>],
    lane_state: &Arc<Mutex<LaneState>>,
    target: usize,
    card: bool,
) -> Vec<usize> {
    let mut state = lane_state.lock().unwrap_or_else(|e| e.into_inner());
    let mut guarded = if card {
        (0..lane_work.len()).collect::<Vec<_>>()
    } else {
        let helper = (0..lane_work.len())
            .filter(|&lane| lane != target && !state.blocked[lane])
            .min_by_key(|&lane| lane_work[lane].load(Ordering::Relaxed))
            .or_else(|| {
                (0..lane_work.len())
                    .filter(|&lane| lane != target)
                    .min_by_key(|&lane| lane_work[lane].load(Ordering::Relaxed))
            });
        let mut lanes = vec![target];
        if let Some(helper) = helper {
            lanes.push(helper);
        }
        lanes
    };
    guarded.sort_unstable();
    for &lane in &guarded {
        state.guard_refs[lane] += 1;
        state.blocked[lane] = true;
    }
    guarded
}

fn release_guard_lanes(lane_state: &Arc<Mutex<LaneState>>, guarded: &[usize]) {
    let mut state = lane_state.lock().unwrap_or_else(|e| e.into_inner());
    for &lane in guarded {
        state.guard_refs[lane] = state.guard_refs[lane].saturating_sub(1);
        state.blocked[lane] = state.route_refs[lane] != 0 || state.guard_refs[lane] != 0;
    }
}

fn cancel_held(
    sender: &mpsc::Sender<Cmd>,
    route_id: u64,
    error: String,
    cost: u64,
    queued_work: &Arc<AtomicU64>,
) {
    if sender.send(Cmd::CancelHeld { route_id, error }).is_err() {
        queued_work.fetch_sub(cost, Ordering::Relaxed);
    }
}

fn run_routes(
    device: usize,
    rx: mpsc::Receiver<RouteCmd>,
    senders: Vec<mpsc::Sender<Cmd>>,
    lane_work: Vec<Arc<AtomicU64>>,
    lane_state: Arc<Mutex<LaneState>>,
    queued_work: Arc<AtomicU64>,
    route_profile: bool,
) {
    while let Ok(RouteCmd::Run(ticket)) = rx.recv() {
        let RouteTicket {
            route_id,
            target,
            work,
            cost,
            held_ready,
        } = ticket;
        let card = work.requires_card_exclusive_route();
        let route_started = Instant::now();
        // The target has been blocked since submission. Claim the remaining
        // memory guard now, then keep dispatching on every unguarded lane while
        // these lanes finish their already-accounted work.
        let guarded = claim_guard_lanes(&lane_work, &lane_state, target, card);
        if held_ready.recv().is_err() {
            release_guard_lanes(&lane_state, &guarded);
            release_route_target(&lane_state, target);
            continue;
        }
        while guarded
            .iter()
            .any(|&lane| lane_work[lane].load(Ordering::Acquire) != 0)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        let drained_at = Instant::now();
        if let Err(e) = trim_lanes(guarded.iter().map(|&lane| &senders[lane])) {
            cancel_held(&senders[target], route_id, e, cost, &queued_work);
            release_guard_lanes(&lane_state, &guarded);
            release_route_target(&lane_state, target);
            continue;
        }
        let trimmed_at = Instant::now();
        lane_work[target].fetch_add(cost, Ordering::Release);
        if senders[target].send(Cmd::ReleaseHeld { route_id }).is_err() {
            lane_work[target].fetch_sub(cost, Ordering::Release);
            queued_work.fetch_sub(cost, Ordering::Relaxed);
            release_guard_lanes(&lane_state, &guarded);
            release_route_target(&lane_state, target);
            continue;
        }
        while lane_work[target].load(Ordering::Acquire) != 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        if route_profile {
            let table_reserved = work
                .table_bytes
                .checked_next_power_of_two()
                .unwrap_or(work.table_bytes);
            let name = if card {
                "v5_card_route"
            } else {
                "v5_arena_guard"
            };
            let helper = guarded
                .iter()
                .copied()
                .find(|&lane| lane != target)
                .unwrap_or(target);
            eprintln!(
                "{name} device={device} lane={target} helper={helper} mutable_mib={:.1} table_mib={:.1} reserved_mib={:.1} drain_ms={:.1} trim_ms={:.1} solve_ms={:.1}",
                work.mutable_bytes as f64 / 1048576.0,
                table_reserved as f64 / 1048576.0,
                work.mutable_bytes.saturating_add(table_reserved) as f64 / 1048576.0,
                1e3 * (drained_at - route_started).as_secs_f64(),
                1e3 * (trimmed_at - drained_at).as_secs_f64(),
                1e3 * trimmed_at.elapsed().as_secs_f64(),
            );
        }
        release_guard_lanes(&lane_state, &guarded);
        release_route_target(&lane_state, target);
    }
}

fn trim_lanes<'a>(senders: impl IntoIterator<Item = &'a mpsc::Sender<Cmd>>) -> Result<(), String> {
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
    let mut held = VecDeque::new();
    let mut shutdown = false;
    // One wave's schedule stays queued on the card while the lane assembles
    // the next one and answers the last one. Without this a lane went quiet
    // between waves, and because every lane waits on the same card they went
    // quiet together: that convoy was the whole of the cards' idle time.
    let mut flight: Option<Flight> = None;

    while !shutdown || !pending.is_empty() || flight.is_some() {
        if pending.is_empty() && !shutdown && flight.is_none() {
            match rx.recv() {
                Ok(cmd) => handle(
                    cmd,
                    &mut exec,
                    &mut current_version,
                    &mut pending,
                    &mut held,
                    &mut shutdown,
                    &queued_work,
                ),
                Err(_) => shutdown = true,
            }
        }
        if shutdown && pending.is_empty() && flight.is_none() {
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
                    &mut held,
                    &mut shutdown,
                    &queued_work,
                ),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }

        let Some(seed) = pending.pop_front() else {
            if let Some(flown) = flight.take() {
                land(flown, &mut exec, device, lane, &queued_work, &lane_work);
                retain_needed_banks(&mut exec, current_version, &pending, &held);
            }
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
                p.work.requires_arena_guard_route(),
            ));
        }
        let pack_started = Instant::now();
        let packed = Wave::pack(&jobs);
        let pack_ms = 1e3 * pack_started.elapsed().as_secs_f64();
        let exclusive = tickets[0].4 || tickets[0].5;
        let shape = Shape {
            device,
            lane,
            class,
            count,
            rows,
            pack_ms,
            launched: Instant::now(),
            profile_shape,
        };
        // `launch` needs the lane buffers the flying wave still owns, so the
        // previous wave has to land first. Its results are unpacked and
        // answered afterwards, by which time the card is busy again.
        let landed = flight
            .take()
            .map(|flown| (exec.collect(flown.inflight), flown.tickets, flown.shape));
        let launched = packed.and_then(|wave| exec.launch(wave, version));
        match launched {
            Ok(inflight) => {
                flight = Some(Flight {
                    inflight,
                    tickets,
                    shape,
                });
            }
            Err(e) => answer(Err(e), tickets, shape, &queued_work, &lane_work),
        }
        if let Some((harvest, tickets, shape)) = landed {
            answer(
                harvest.and_then(|h| h.unpack()),
                tickets,
                shape,
                &queued_work,
                &lane_work,
            );
        }
        // An exclusive wave owns the card by construction, and `trim` gives
        // its arena straight back, so it is not pipelined.
        if exclusive {
            if let Some(flown) = flight.take() {
                land(flown, &mut exec, device, lane, &queued_work, &lane_work);
            }
            if let Err(e) = exec.trim() {
                eprintln!("GPU lane {device}/{lane} trim after an exclusive wave failed: {e}");
            }
        }
        retain_needed_banks(&mut exec, current_version, &pending, &held);
    }
    for held in held {
        let _ = held.pending.reply.send((
            held.pending.tag,
            Err("GPU lane shut down with a held route".into()),
        ));
        queued_work.fetch_sub(held.pending.cost, Ordering::Relaxed);
    }
}

/// A wave whose schedule is queued on this lane's card, with everything
/// needed to answer it once it lands.
struct Flight {
    inflight: super::device::InFlight,
    tickets: Vec<Ticket>,
    shape: Shape,
}

type Ticket = (
    usize,
    mpsc::Sender<(usize, Result<SolveResult, String>)>,
    u64,
    bool,
    bool,
    bool,
);

/// What the `WARCHEST_GPU_PROFILE` line reports about one wave.
struct Shape {
    device: usize,
    lane: usize,
    class: u8,
    count: usize,
    rows: usize,
    pack_ms: f64,
    launched: Instant,
    #[allow(clippy::type_complexity)]
    profile_shape: Option<(usize, usize, usize, usize, usize, usize, f64)>,
}

/// Wait for a flying wave and answer it. Used when there is no next batch to
/// launch first, and before any command that needs the lane quiescent.
fn land(
    flown: Flight,
    exec: &mut Executor,
    _device: usize,
    _lane: usize,
    queued_work: &Arc<AtomicU64>,
    lane_work: &Arc<AtomicU64>,
) {
    let result = exec.collect(flown.inflight).and_then(|h| h.unpack());
    answer(result, flown.tickets, flown.shape, queued_work, lane_work);
}

fn answer(
    result: Result<Vec<SolveResult>, String>,
    tickets: Vec<Ticket>,
    shape: Shape,
    queued_work: &Arc<AtomicU64>,
    lane_work: &Arc<AtomicU64>,
) {
    let (device, lane, class, count, rows) = (
        shape.device,
        shape.lane,
        shape.class,
        shape.count,
        shape.rows,
    );
    if let Some((cells, reach, reverse, table_bytes, mutable_bytes, max_bytes, oldest_ms)) =
        shape.profile_shape
    {
        eprintln!(
            "v5_service device={device} lane={lane} class={class} jobs={count} rows={rows} cells={cells} reach={reach} reverse={reverse} table_mib={:.1} mutable_mib={:.1} max_job_mib={:.1} oldest_ms={oldest_ms:.2} pack_ms={:.2} solve_ms={:.2}",
            table_bytes as f64 / 1048576.0,
            mutable_bytes as f64 / 1048576.0,
            max_bytes as f64 / 1048576.0,
            shape.pack_ms,
            1e3 * shape.launched.elapsed().as_secs_f64(),
        );
    }
    match result {
        Ok(results) if results.len() == count => {
            for ((tag, reply, cost, oversize, card, _), mut value) in
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
            for (tag, reply, cost, _, _, _) in tickets {
                let _ = reply.send((tag, Err(e.clone())));
                queued_work.fetch_sub(cost, Ordering::Relaxed);
                lane_work.fetch_sub(cost, Ordering::Relaxed);
            }
        }
        Err(e) => {
            for (tag, reply, cost, _, _, _) in tickets {
                let _ = reply.send((tag, Err(e.clone())));
                queued_work.fetch_sub(cost, Ordering::Relaxed);
                lane_work.fetch_sub(cost, Ordering::Relaxed);
            }
        }
    }
}

fn handle(
    cmd: Cmd,
    exec: &mut Executor,
    current_version: &mut u64,
    pending: &mut VecDeque<Pending>,
    held: &mut VecDeque<HeldPending>,
    shutdown: &mut bool,
    queued_work: &Arc<AtomicU64>,
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
        Cmd::Hold {
            route_id,
            job,
            work,
            tag,
            cost,
            reply,
            ready,
        } => {
            held.push_back(HeldPending {
                route_id,
                pending: Pending {
                    job,
                    work,
                    tag,
                    reply,
                    version: *current_version,
                    queued: Instant::now(),
                    cost,
                },
            });
            let _ = ready.send(());
        }
        Cmd::ReleaseHeld { route_id } => {
            if let Some(i) = held.iter().position(|x| x.route_id == route_id) {
                let held = held.remove(i).expect("held route index");
                pending.push_back(held.pending);
            }
        }
        Cmd::CancelHeld { route_id, error } => {
            if let Some(i) = held.iter().position(|x| x.route_id == route_id) {
                let held = held.remove(i).expect("held route index");
                let _ = held.pending.reply.send((held.pending.tag, Err(error)));
                queued_work.fetch_sub(held.pending.cost, Ordering::Relaxed);
            }
        }
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
    retain_needed_banks(exec, *current_version, pending, held);
}

fn retain_needed_banks(
    exec: &mut Executor,
    current: u64,
    pending: &VecDeque<Pending>,
    held: &VecDeque<HeldPending>,
) {
    let mut keep = Vec::with_capacity(pending.len() + held.len() + 1);
    keep.push(current);
    keep.extend(pending.iter().map(|p| p.version));
    keep.extend(held.iter().map(|p| p.pending.version));
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
///
/// Only jobs of the same class share a wave, so every extra class divides the
/// lane's queue and shrinks the waves it can form. `WARCHEST_WAVE_CLASS_SHIFT`
/// coarsens the ordinary classes: 2 collapses all four into one, which is what
/// the mature live stream wants because its queue holds a mixture and a
/// per-solve cost that falls steeply with jobs per wave.
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
    let class = if bytes >= (512usize << 20) || work >= 4_000_000 {
        3
    } else if bytes >= (64usize << 20) || work >= 500_000 {
        2
    } else if bytes >= (16usize << 20) || work >= 125_000 {
        1
    } else {
        0
    };
    class >> class_shift()
}

fn class_shift() -> u8 {
    static SHIFT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *SHIFT.get_or_init(|| env_usize("WARCHEST_WAVE_CLASS_SHIFT", 0).min(2) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_claims_keep_targets_blocked_until_the_queue_drains() {
        let lanes = [9, 2, 5]
            .into_iter()
            .map(|x| Arc::new(AtomicU64::new(x)))
            .collect::<Vec<_>>();
        let state = Arc::new(Mutex::new(LaneState {
            blocked: vec![false; 3],
            route_refs: vec![0; 3],
            guard_refs: vec![0; 3],
        }));
        let first = claim_route_target(&lanes, &state, 1);
        let second = claim_route_target(&lanes, &state, 1);
        assert_eq!((first, second), (0, 0));
        let guarded = claim_guard_lanes(&lanes, &state, first, false);
        assert_eq!(guarded, vec![0, 1]);
        assert_eq!(state.lock().unwrap().blocked, vec![true, true, false]);

        release_guard_lanes(&state, &guarded);
        release_route_target(&state, first);
        assert_eq!(state.lock().unwrap().blocked, vec![true, false, false]);
        release_route_target(&state, second);
        assert_eq!(state.lock().unwrap().blocked, vec![false, false, false]);
    }
}

/// Read a per-device setting: one value, or a comma-separated list indexed by
/// device with the last entry covering any further cards.
fn env_list(name: &str, device: usize, default: usize) -> usize {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    let parts: Vec<usize> = raw
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    if parts.is_empty() {
        return default;
    }
    *parts
        .get(device)
        .unwrap_or(parts.last().expect("non-empty"))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}
