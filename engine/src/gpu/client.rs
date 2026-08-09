//! Actor-side handle for the v5 wave executor.
//!
//! A submission has one completion. The result already owns the final sparse
//! strategy, Phase-2 root values, and every retained snapshot belief at every
//! possible walk exit. The actor selects its eventual exit locally; there is
//! no resident solve id and no second device rendezvous.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

use crate::serialize::PackedJob;

pub type CarriedBeliefs = Vec<[Vec<f32>; 2]>;

/// Pageable host store streamed out while a wave materialises its requested
/// snapshots. `exit_nodes` is `leaf_rows` followed by terminal leaves;
/// `coff` gives both player spans within one snapshot's flat config block.
pub struct CarryStore {
    pub exit_nodes: Vec<u32>,
    pub coff: Vec<u32>,
    pub snapshots: usize,
    pub snapshot_configs: usize,
    pub data: Vec<f32>,
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
                self.data[base + p0.start..base + p0.end].to_vec(),
                self.data[base + p1.start..base + p1.end].to_vec(),
            ]);
        }
        Ok(out)
    }

    pub fn owned_bytes(&self) -> usize {
        4 * (self.exit_nodes.len() + self.coff.len() + self.data.len())
    }
}

/// One completed solve. All indices are solve-local again by the time this
/// crosses the actor boundary.
pub struct SolveResult {
    pub strategy: Vec<f32>,
    pub root_values: Vec<[Vec<f32>; 2]>,
    pub carries: CarryStore,
    pub weight_version: u64,
}

pub(crate) enum Cmd {
    Submit {
        job: PackedJob,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    },
    Publish {
        version: u64,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    },
    Shutdown,
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
}

impl GpuClient {
    pub(crate) fn new(tx: mpsc::Sender<Cmd>, thread: JoinHandle<()>) -> GpuClient {
        GpuClient {
            inner: Arc::new(ClientInner {
                tx,
                thread: Mutex::new(Some(thread)),
                next_weight_version: AtomicU64::new(1),
            }),
        }
    }

    pub fn submit(&self, job: PackedJob) -> Result<SolveHandle, String> {
        let (tx, rx) = mpsc::channel();
        self.submit_tagged(job, 0, tx)?;
        Ok(SolveHandle { rx })
    }

    pub fn submit_tagged(
        &self,
        job: PackedJob,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<SolveResult, String>)>,
    ) -> Result<(), String> {
        self.inner
            .tx
            .send(Cmd::Submit { job, tag, reply })
            .map_err(|_| "GPU wave executor is gone".to_string())
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
