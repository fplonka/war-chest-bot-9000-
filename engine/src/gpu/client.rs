//! The worker-side client: submit a solve job, block for trip 1 (the
//! reference strategy + the carried roots' values), then send the exit leaf
//! for trip 2 (the carried beliefs). The service thread owns the CUDA
//! device; workers only move bytes and wait.

use std::sync::mpsc;

use crate::serialize::Job;

/// What trip 1 returns: the solve id (for the later trip 2), the reference
/// strategy (flat, `soff`-aligned, one distribution per config), and the
/// Phase-2 root values for every carried root, per player, in root-support
/// order.
pub struct Trip1 {
    pub id: u64,
    pub strategy: Vec<f32>,
    pub root_values: Vec<[Vec<f32>; 2]>,
}

/// What trip 2 returns: the beliefs at the exit leaf under each kept
/// per-iterate average strategy (t = 0..T-1), per player.
pub type Trip2 = Vec<[Vec<f32>; 2]>;

pub(crate) enum Cmd {
    Submit {
        job: Job,
        reply: mpsc::Sender<Result<Trip1, String>>,
    },
    Trip2 {
        id: u64,
        leaf: u32,
        reply: mpsc::Sender<Result<Trip2, String>>,
    },
    SetWeights {
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    },
    Shutdown,
}

/// Handle to the service, shared by all workers.
#[derive(Clone)]
pub struct GpuClient {
    tx: mpsc::Sender<Cmd>,
}

impl GpuClient {
    pub(crate) fn new(tx: mpsc::Sender<Cmd>) -> GpuClient {
        GpuClient { tx }
    }

    /// Submit one solve and block until trip 1. `carried` are the root
    /// vectors Phase 2 must value (the previous solve's carried beliefs, or
    /// the live belief for the first level).
    pub fn solve(&self, job: Job, carried: &[[Vec<f32>; 2]]) -> Result<Trip1, String> {
        let mut job = job;
        job.carried = carried.to_vec();
        let h = self.submit(job)?;
        h.wait()
    }

    /// Submit one solve without blocking. The worker may hold several
    /// pending solves (one per game) and wait on them in any order.
    pub fn submit(&self, job: Job) -> Result<SolveHandle, String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Cmd::Submit { job, reply: tx })
            .map_err(|_| "gpu service gone".to_string())?;
        Ok(SolveHandle { rx })
    }

    /// The walk left the tree at `leaf`: get the carried beliefs and free
    /// the solve.
    pub fn carried_beliefs(&self, id: u64, leaf: u32) -> Result<Trip2, String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Cmd::Trip2 { id, leaf, reply: tx })
            .map_err(|_| "gpu service gone".to_string())?;
        rx.recv().map_err(|_| "gpu service gone".to_string())?
    }

    /// Publish fresh weights (from the trainer). Applied between solves.
    pub fn set_weights(&self, dims: Vec<usize>, w: Vec<f32>, b: Vec<f32>, ln: Vec<f32>) {
        let _ = self.tx.send(Cmd::SetWeights { dims, w, b, ln });
    }
}

impl Drop for GpuClient {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Shutdown);
    }
}

/// A submitted solve; the worker blocks on `wait` for trip 1.
pub struct SolveHandle {
    rx: mpsc::Receiver<Result<Trip1, String>>,
}

impl SolveHandle {
    pub fn wait(self) -> Result<Trip1, String> {
        self.rx.recv().map_err(|_| "gpu service gone".to_string())?
    }
}
