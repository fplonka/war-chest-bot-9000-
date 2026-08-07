//! The worker-side client: submit a solve job, wait for trip 1 (the
//! reference strategy + the carried roots' values), then send the exit leaf
//! for trip 2 (the carried beliefs). The service thread owns the CUDA
//! device; workers only move bytes and wait.
//!
//! Replies are tagged: a worker that runs several games at once submits each
//! game's solve with its own tag on one shared channel and resumes whichever
//! game answers first, instead of waiting on them in a fixed order.

use std::sync::mpsc;

use crate::serialize::Job;

/// What trip 1 returns: the solve id (for the later trip 2), the reference
/// strategy (flat, `soff`-aligned), and the Phase-2 root values for every
/// carried root, per player, in root-support order.
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
        tag: usize,
        reply: mpsc::Sender<(usize, Result<Trip1, String>)>,
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

    /// Submit one solve and block until trip 1.
    pub fn solve(&self, job: Job, carried: &[[Vec<f32>; 2]]) -> Result<Trip1, String> {
        let mut job = job;
        job.carried = carried.to_vec();
        let h = self.submit(job)?;
        h.wait()
    }

    /// Submit one solve without blocking; the handle owns its own channel.
    pub fn submit(&self, job: Job) -> Result<SolveHandle, String> {
        let (tx, rx) = mpsc::channel();
        self.submit_tagged(job, 0, tx)?;
        Ok(SolveHandle { rx })
    }

    /// Submit one solve whose trip-1 reply lands on `reply` with `tag`. A
    /// worker gives every live game a tag and one shared channel, then
    /// resumes whichever game's solve finishes first.
    pub fn submit_tagged(
        &self,
        job: Job,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<Trip1, String>)>,
    ) -> Result<(), String> {
        self.tx
            .send(Cmd::Submit { job, tag, reply })
            .map_err(|_| "gpu service gone".to_string())
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

    /// Publish fresh weights (from the trainer). Applied when the live set
    /// drains; a different shape is refused — restart the service.
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
    rx: mpsc::Receiver<(usize, Result<Trip1, String>)>,
}

impl SolveHandle {
    pub fn wait(self) -> Result<Trip1, String> {
        Ok(self
            .rx
            .recv()
            .map_err(|_| "gpu service gone".to_string())?
            .1?)
    }
}
