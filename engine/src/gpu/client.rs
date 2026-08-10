//! Actor-side handle for the v5 wave executor.
//!
//! A submission has one completion. The result already owns the final sparse
//! strategy, Phase-2 root values, and every retained snapshot belief at every
//! possible walk exit. The actor selects its eventual exit locally; there is
//! no resident solve id and no second device rendezvous.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::serialize::{PackedJob, WorkVector};

pub type CarriedBeliefs = Vec<[Vec<f32>; 2]>;

/// Pageable binary16 host store streamed out while a wave materialises its
/// requested snapshots. `exit_nodes` is `leaf_rows` followed by terminal
/// leaves; `coff` gives both player spans within one snapshot's flat config
/// block. Only the exit selected by the real game walk is expanded to FP32.
pub struct CarryStore {
    pub exit_nodes: Vec<u32>,
    pub coff: Vec<u32>,
    pub snapshots: usize,
    pub snapshot_configs: usize,
    pub data: Vec<u16>,
}

impl CarryStore {
    pub fn empty() -> CarryStore {
        CarryStore {
            exit_nodes: Vec::new(),
            coff: vec![0],
            snapshots: 0,
            snapshot_configs: 0,
            data: Vec::new(),
        }
    }

    /// Select and copy the two belief spans for the leaf the real game walk
    /// eventually reached. The large solve arena has already been released.
    pub fn select(&self, node: u32) -> Result<CarriedBeliefs, String> {
        let exit = self
            .exit_nodes
            .iter()
            .position(|&x| x == node)
            .ok_or_else(|| format!("carry store has no exit node {node}"))?;
        let mut out = Vec::with_capacity(self.snapshots);
        for s in 0..self.snapshots {
            let base = s * self.snapshot_configs;
            let p0 = self.coff[2 * exit] as usize..self.coff[2 * exit + 1] as usize;
            let p1 = self.coff[2 * exit + 1] as usize..self.coff[2 * exit + 2] as usize;
            out.push([
                decode_probability(&self.data[base + p0.start..base + p0.end]),
                decode_probability(&self.data[base + p1.start..base + p1.end]),
            ]);
        }
        Ok(out)
    }

    #[cfg(all(test, feature = "gpu"))]
    pub(crate) fn snapshot(&self, slot: usize) -> Vec<f32> {
        let at = slot * self.snapshot_configs;
        let mut out = decode_f16(&self.data[at..at + self.snapshot_configs]);
        for bounds in self.coff.windows(2) {
            normalize(&mut out[bounds[0] as usize..bounds[1] as usize]);
        }
        out
    }

    pub fn owned_bytes(&self) -> usize {
        4 * (self.exit_nodes.len() + self.coff.len()) + 2 * self.data.len()
    }
}

fn decode_f16(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&x| f16_to_f32(x)).collect()
}

fn decode_probability(src: &[u16]) -> Vec<f32> {
    let mut out = decode_f16(src);
    normalize(&mut out);
    out
}

fn normalize(values: &mut [f32]) {
    let sum: f32 = values.iter().sum();
    if sum > 0.0 {
        for x in values {
            *x /= sum;
        }
    }
}

fn f16_to_f32(x: u16) -> f32 {
    let sign = (x as u32 & 0x8000) << 16;
    let exp = (x >> 10) & 0x1f;
    let man = x & 0x03ff;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        let magnitude = man as f32 * (1.0 / 16_777_216.0);
        return if sign == 0 { magnitude } else { -magnitude };
    }
    let bits = if exp == 0x1f {
        sign | 0x7f80_0000 | (man as u32) << 13
    } else {
        sign | ((exp as u32 + 112) << 23) | (man as u32) << 13
    };
    f32::from_bits(bits)
}

/// One completed solve. All indices are solve-local again by the time this
/// crosses the actor boundary.
pub struct SolveResult {
    pub strategy: Vec<f32>,
    pub root_values: Vec<[Vec<f32>; 2]>,
    pub carries: CarryStore,
    pub weight_version: u64,
    pub oversize_route: bool,
    pub card_exclusive_route: bool,
}

pub(crate) enum Cmd {
    Submit {
        job: Arc<PackedJob>,
        work: WorkVector,
        tag: usize,
        cost: u64,
        reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    },
    Hold {
        route_id: u64,
        job: Arc<PackedJob>,
        work: WorkVector,
        tag: usize,
        cost: u64,
        reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
        ready: mpsc::SyncSender<()>,
    },
    ReleaseHeld {
        route_id: u64,
    },
    CancelHeld {
        route_id: u64,
        error: String,
    },
    Publish {
        version: u64,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    },
    Trim {
        ready: mpsc::SyncSender<Result<(), String>>,
    },
    Shutdown,
}

/// Immutable submission with its admission vector computed exactly once.
/// Cloning this handle is cheap and is useful for deterministic workload
/// tapes; production normally creates it and submits it once.
#[derive(Clone)]
pub struct PreparedJob {
    job: Arc<PackedJob>,
    work: WorkVector,
    cost: u64,
}

impl PreparedJob {
    pub fn new(job: PackedJob) -> PreparedJob {
        let work = job.work();
        let cost = job_cost(work, job.meta.iters);
        PreparedJob {
            job: Arc::new(job),
            work,
            cost,
        }
    }
}

/// Handle shared by actors assigned to one CUDA device.
#[derive(Clone)]
pub struct GpuClient {
    inner: Arc<ClientInner>,
}

struct ClientInner {
    tx: mpsc::Sender<Cmd>,
    thread: Mutex<Option<JoinHandle<()>>>,
    next_weight_version: AtomicU64,
    queued_work: Arc<AtomicU64>,
}

impl GpuClient {
    pub(crate) fn new(
        tx: mpsc::Sender<Cmd>,
        thread: JoinHandle<()>,
        queued_work: Arc<AtomicU64>,
    ) -> GpuClient {
        GpuClient {
            inner: Arc::new(ClientInner {
                tx,
                thread: Mutex::new(Some(thread)),
                next_weight_version: AtomicU64::new(1),
                queued_work,
            }),
        }
    }

    pub fn submit(&self, job: PackedJob) -> Result<SolveHandle, String> {
        self.submit_prepared(PreparedJob::new(job))
    }

    pub fn submit_prepared(&self, job: PreparedJob) -> Result<SolveHandle, String> {
        let (tx, rx) = mpsc::channel();
        self.submit_tagged_prepared(job, 0, tx)?;
        Ok(SolveHandle { rx })
    }

    pub fn submit_tagged(
        &self,
        job: PackedJob,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    ) -> Result<(), String> {
        self.submit_tagged_prepared(PreparedJob::new(job), tag, reply)
    }

    fn submit_tagged_prepared(
        &self,
        job: PreparedJob,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    ) -> Result<(), String> {
        let PreparedJob { job, work, cost } = job;
        self.inner.queued_work.fetch_add(cost, Ordering::Relaxed);
        if self
            .inner
            .tx
            .send(Cmd::Submit {
                job,
                work,
                tag,
                cost,
                reply,
            })
            .is_err()
        {
            self.inner.queued_work.fetch_sub(cost, Ordering::Relaxed);
            return Err("GPU wave executor is gone".into());
        }
        Ok(())
    }

    /// Publish an immutable weight bank. Already-dispatched waves keep their
    /// old version; newly formed waves use this version as a unit.
    pub fn set_weights(&self, dims: Vec<usize>, w: Vec<f32>, b: Vec<f32>, ln: Vec<f32>) -> u64 {
        let version = self
            .inner
            .next_weight_version
            .fetch_add(1, Ordering::Relaxed);
        let _ = self.inner.tx.send(Cmd::Publish {
            version,
            dims,
            w,
            b,
            ln,
        });
        version
    }

    /// Monotone work estimate currently queued or executing on this card.
    /// Training routes a new solve to the least-finish-time candidate instead
    /// of alternating cards regardless of their tails.
    pub fn queued_work(&self) -> u64 {
        self.inner.queued_work.load(Ordering::Relaxed)
    }
}

fn job_cost(w: WorkVector, configured_iters: usize) -> u64 {
    let iters = configured_iters.max(1) as u64;
    (w.network_rows as u64)
        .saturating_mul(iters)
        .saturating_mul(16)
        .saturating_add((w.legal_cells as u64).saturating_mul(iters))
        .saturating_add((w.reach_slots as u64).saturating_mul(iters))
        .saturating_add(w.table_bytes as u64 / 16)
        .max(1)
}

impl Drop for ClientInner {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Shutdown);
        if let Some(thread) = self.thread.get_mut().unwrap().take() {
            let _ = thread.join();
        }
    }
}

pub struct SolveHandle {
    rx: mpsc::Receiver<(usize, Result<SolveResult, String>)>,
}

impl SolveHandle {
    pub fn wait(self) -> Result<SolveResult, String> {
        self.rx
            .recv()
            .map_err(|_| "GPU wave executor is gone".to_string())?
            .1
    }
}
