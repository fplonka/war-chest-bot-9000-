
use parking_lot::{Condvar, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(feature = "gpu")]
use crate::cuda::Device;
use crate::search::{Budget, Cfr, Solver, Step};
use crate::net::Net;
use crate::selfplay::{Data, GameCfg, GameStream};

#[derive(Clone)]
pub enum Call {
    Trunk {
        solve: usize,
        at: usize,
        queries: usize,
        board_of: Vec<u32>,
        boards_at: usize,
        boards: usize,
        packed: Vec<u8>,
        cards: Vec<f32>,
        cidx: Vec<u32>,
        coff: Vec<u32>,
    },
    Configs {
        solve: usize,
        at: usize,
        phi: Vec<f32>,
        owner: Vec<u32>,
        cards: Vec<f32>,
        n: usize,
    },
    Tree {
        solve: usize,
        writes: Writes,
        fresh: bool,
        ncells: usize,
        nreach: usize,
        nvals: usize,
        levels: Vec<u32>,
        nterm: usize,
        seed: Option<u64>,
        prime: Vec<Prime>,
        acts: Vec<u32>,
        cells: Vec<u32>,
        prior_temp: f32,
    },
    Iterate {
        solve: usize,
        step: usize,
        iters: usize,
        expand: usize,
        query: Vec<QueryPick>,
        cfr: Cfr,
        puct: f32,
    },
    Read {
        solve: usize,
        touched: [bool; 2],
        vals_at: [(u32, u32); 2],
        policy_at: (u32, u32),
    },
}

#[derive(Clone, Copy)]
pub struct QueryPick {
    pub iter: u32,
    pub reach: u32,
    pub len: u32,
}

#[derive(Clone, Copy)]
pub struct Prime {
    pub node: u32,
    pub row: u32,
    pub at: u32,
    pub na: u32,
    pub cell_at: u32,
    pub nc: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dst {
    Kind, Player, Exhausted, Nc, Parent, Roff, Voff, Soff, Util,
    ChildAt, ChildN, Child,
    LegalBase, LegalOff, LegalChild, LegalTrans, CellRow, CellVal,
    RevBase, RevStart, RevSrc, RevCell,
    RvdBase, RvdStart, RvdSrc, RvdP,
    DrawBase, DrawStart, DrawTo, DrawP,
    LevelStart, LevelNode,
    Cur, Prior, LeafNode, Term, Rootb,
}

#[derive(Clone, Default)]
pub struct Writes {
    pub blob: Vec<u32>,
    pub runs: Vec<Run>,
}

#[derive(Clone, Copy)]
pub struct Run {
    pub dst: Dst,
    pub at: u32,
    pub len: u32,
    pub start: u32,
}

impl Writes {
    pub fn u32s(&mut self, d: Dst, at: usize, src: &[u32]) {
        self.run(d, at, src.iter().copied(), src.len());
    }

    pub fn f32s(&mut self, d: Dst, at: usize, src: &[f32]) {
        self.run(d, at, src.iter().map(|x| x.to_bits()), src.len());
    }

    pub fn u8s(&mut self, d: Dst, at: usize, src: &[u8]) {
        self.run(d, at, src.iter().map(|&x| x as u32), src.len());
    }

    pub fn f32s_both(&mut self, a: Dst, b: Dst, at: usize, src: &[f32]) {
        self.f32s(a, at, src);
        if !src.is_empty() {
            let last = *self.runs.last().expect("a non-empty run was just pushed");
            self.runs.push(Run { dst: b, ..last });
        }
    }

    fn run(&mut self, d: Dst, at: usize, src: impl Iterator<Item = u32>, n: usize) {
        if n == 0 {
            return;
        }
        let start = self.blob.len() as u32;
        self.blob.extend(src);
        debug_assert!(
            !self.runs.iter().any(|r| {
                r.dst == d && (at as u32) < r.at + r.len && r.at < (at + n) as u32
            }),
            "two runs of one round overlap in the same array"
        );
        self.runs.push(Run { dst: d, at: at as u32, len: n as u32, start });
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

#[derive(Default)]
pub struct Reply {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub leaves: Vec<u32>,
}

impl Call {
    #[cfg(any(test, feature = "gpu"))]
    pub fn kind(&self) -> usize {
        match self {
            Call::Trunk { .. } => 0,
            Call::Configs { .. } => 1,
            Call::Tree { .. } => 2,
            Call::Iterate { .. } => 3,
            Call::Read { .. } => 4,
        }
    }

    pub fn solve(&self) -> usize {
        match self {
            Call::Trunk { solve, .. }
            | Call::Configs { solve, .. }
            | Call::Tree { solve, .. }
            | Call::Iterate { solve, .. }
            | Call::Read { solve, .. } => *solve,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Call::Trunk { queries, .. } => *queries,
            Call::Configs { n, .. } => *n,
            Call::Tree { .. } | Call::Iterate { .. } | Call::Read { .. } => 0,
        }
    }
}


pub const CARD_ROWS: usize = 2;

pub fn host_slots(budget: Budget) -> usize {
    let slot = budget.host_slot_bytes() as u64;
    (host_free() / slot.max(1)) as usize
}

fn host_free() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
            return 0;
        };
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0);
                return kb * 1024;
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        u64::MAX
    }
}

struct Queue<T> {
    q: Mutex<std::collections::VecDeque<T>>,
    ready: Condvar,
    closed: AtomicBool,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Queue {
            q: Mutex::new(std::collections::VecDeque::new()),
            ready: Condvar::new(),
            closed: AtomicBool::new(false),
        }
    }
}

impl<T> Queue<T> {
    fn push(&self, x: T) {
        self.q.lock().push_back(x);
        self.ready.notify_one();
    }

    fn pop(&self) -> Option<T> {
        let mut q = self.q.lock();
        loop {
            if let Some(x) = q.pop_front() {
                return Some(x);
            }
            if self.closed.load(Ordering::Relaxed) {
                return None;
            }
            self.ready.wait(&mut q);
        }
    }

    fn drain(&self) -> Vec<T> {
        let mut q = self.q.lock();
        loop {
            if !q.is_empty() {
                return q.drain(..).collect();
            }
            if self.closed.load(Ordering::Relaxed) {
                return Vec::new();
            }
            self.ready.wait(&mut q);
        }
    }

    fn drain_wave(&self, wave: usize) -> Vec<T> {
        let mut q = self.q.lock();
        loop {
            let n = q.len();
            if n >= wave || (self.closed.load(Ordering::Relaxed) && n > 0) {
                let take = n.min(wave);
                let out: Vec<T> = q.drain(..take).collect();
                if !q.is_empty() {
                    self.ready.notify_one();
                }
                return out;
            }
            if self.closed.load(Ordering::Relaxed) {
                return Vec::new();
            }
            self.ready.wait(&mut q);
        }
    }

    fn close(&self) {
        let _held = self.q.lock();
        self.closed.store(true, Ordering::Relaxed);
        self.ready.notify_all();
    }

    fn closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }
}

struct Job {
    source: GameStream,
    solver: Solver,
    card: usize,
    slot: usize,
    replies: Vec<Reply>,
    data: Data,
}

pub struct Farm {
    device: Vec<Arc<Queue<(Job, Vec<Call>)>>>,
    ready: Arc<Queue<Job>>,
    #[cfg(feature = "gpu")]
    cuda: Arc<RwLock<Device>>,
    nets: Arc<RwLock<Arc<crate::net::Net>>>,
    collected: Arc<Mutex<Vec<Data>>>,
    workers: Vec<JoinHandle<()>>,
    drivers: Vec<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    broken: Arc<AtomicBool>,
    stats: Arc<Stats>,
}

#[derive(Default)]
pub struct Stats {
    pub rounds: AtomicU64,
    pub rows: AtomicU64,
    pub calls: AtomicU64,
    pub nanos: AtomicU64,
    slots: AtomicU64,
    used: AtomicU64,
    budget_hits: AtomicU64,
    entity_hits: [AtomicU64; 8],
    slot_bytes: AtomicU64,
    slots_per_card: AtomicU64,
    shapes: Mutex<Vec<[u32; 10]>>,
}

impl Stats {
    fn new(slots: usize, slot_bytes: usize, slots_per_card: usize) -> Stats {
        Stats {
            slots: AtomicU64::new(slots as u64),
            slot_bytes: AtomicU64::new(slot_bytes as u64),
            slots_per_card: AtomicU64::new(slots_per_card as u64),
            ..Default::default()
        }
    }

    pub fn slots(&self) -> u64 {
        self.slots.load(Ordering::Relaxed)
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    pub fn budget_hits(&self) -> u64 {
        self.budget_hits.load(Ordering::Relaxed)
    }

    pub fn entity_hits(&self) -> [u64; 8] {
        std::array::from_fn(|i| self.entity_hits[i].load(Ordering::Relaxed))
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes.load(Ordering::Relaxed)
    }

    pub fn slots_per_card(&self) -> u64 {
        self.slots_per_card.load(Ordering::Relaxed)
    }

    pub fn take_shapes(&self) -> Vec<[u32; 10]> {
        std::mem::take(&mut *self.shapes.lock())
    }
}

impl Farm {
    #[cfg(feature = "gpu")]
    pub fn new(seed: u64, workers: usize, gc: GameCfg, cuda: Device) -> Farm {
        assert!(workers > 0, "a farm needs at least one worker");
        let gpus = cuda.cards();
        let pipes = crate::cuda::PIPELINE;
        let per_gpu: Vec<usize> = (0..gpus).map(|g| cuda.slots(g)).collect();
        let n_slots: usize = per_gpu.iter().sum();
        let slot_bytes = cuda.slot_bytes();
        let slots_per_card = cuda.slots_per_card();
        let gc = Arc::new(gc);
        let nets = Arc::new(RwLock::new(Arc::new(cuda.net().clone())));
        let ready = Arc::new(Queue::default());
        let device: Vec<Arc<Queue<(Job, Vec<Call>)>>> =
            (0..gpus).map(|_| Arc::new(Queue::default())).collect();
        let collected = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let broken = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::new(n_slots, slot_bytes, slots_per_card));
        let cuda = Arc::new(RwLock::new(cuda));

        let hands: Vec<JoinHandle<()>> = (0..workers)
            .map(|t| {
                let (ready, device, nets, collected, stopping, stats) = (
                    Arc::clone(&ready),
                    device.clone(),
                    Arc::clone(&nets),
                    Arc::clone(&collected),
                    Arc::clone(&stopping),
                    Arc::clone(&stats),
                );
                std::thread::Builder::new()
                    .name(format!("host-{t}"))
                    .spawn(move || {
                        while let Some(job) = ready.pop() {
                            advance_job(
                                job, &device, &nets, &collected, &stopping, &stats,
                            );
                        }
                    })
                    .expect("spawn host thread")
            })
            .collect();

        let mut drivers = Vec::with_capacity(gpus * pipes);
        for g in 0..gpus {
            for p in 0..pipes {
                let (queue, ready, cuda, nets, gc, stats, broken) = (
                    Arc::clone(&device[g]),
                    Arc::clone(&ready),
                    Arc::clone(&cuda),
                    Arc::clone(&nets),
                    Arc::clone(&gc),
                    Arc::clone(&stats),
                    Arc::clone(&broken),
                );
                let n = per_gpu[g];
                let seed = seed.wrapping_mul(0x9E37_79B9) ^ g as u64 ^ (p as u64) << 32;
                let lane = g * pipes + p;
                drivers.push(
                    std::thread::Builder::new()
                        .name(format!("card-{g}.{p}"))
                        .spawn(move || {
                            drive_card(
                                g, lane, n, p == 0, n.div_ceil(pipes.max(1)), seed,
                                &queue, &ready, &cuda, &nets, &gc, &stats, &broken,
                            )
                        })
                        .expect("spawn driver thread"),
                );
            }
        }

        Farm {
            device,
            ready,
            cuda,
            nets,
            collected,
            workers: hands,
            drivers,
            stopping,
            broken,
            stats,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn publish(&mut self, net: Net) -> Result<(), String> {
        self.cuda.write().set_weights(net.clone())?;
        *self.nets.write() = Arc::new(net);
        Ok(())
    }

    pub fn drive(&mut self, solves: usize) -> Data {
        let mut out = Data::default();
        loop {
            for d in self.collected.lock().drain(..) {
                out.merge(d);
            }
            if out.soff.len() >= solves {
                return out;
            }
            if self.stopping.load(Ordering::Relaxed) || self.broken.load(Ordering::Relaxed) {
                return out;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    pub fn broken(&self) -> bool {
        self.broken.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

}

impl Drop for Farm {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Relaxed);
        self.ready.close();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        for q in &self.device {
            q.close();
        }
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}

fn advance_job(
    mut job: Job,
    device: &[Arc<Queue<(Job, Vec<Call>)>>],
    nets: &RwLock<Arc<crate::net::Net>>,
    collected: &Mutex<Vec<Data>>,
    stopping: &AtomicBool,
    stats: &Stats,
) {
    let mut replies = std::mem::take(&mut job.replies);
    loop {
        match job.solver.advance(&replies) {
            Step::Calls(calls) => return device[job.card].push((job, calls)),
            Step::Done(solved) => {
                let mask = job.solver.hit_mask();
                if mask != 0 {
                    stats.budget_hits.fetch_add(1, Ordering::Relaxed);
                    for i in 0..8 {
                        if mask & (1 << i) != 0 {
                            stats.entity_hits[i].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                let counts = job.solver.counts();
                let mut census = [0; 10];
                census[..9].copy_from_slice(&counts);
                census[9] = job.source.solve_kind() as u32;
                stats.shapes.lock().push(census);
                job.source.keep(&job.solver, solved, &mut job.data);
                collected.lock().push(std::mem::take(&mut job.data));
                if stopping.load(Ordering::Relaxed) {
                    stats.used.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
                let n = Arc::clone(&*nets.read());
                job.solver = job.source.next_solve(&n, &mut job.data);
                job.solver.pin(job.slot);
                replies = Vec::new();
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn drive_card(
    gpu: usize,
    lane: usize,
    n_slots: usize,
    seed_slots: bool,
    wave: usize,
    seed: u64,
    queue: &Queue<(Job, Vec<Call>)>,
    ready: &Queue<Job>,
    cuda: &RwLock<Device>,
    nets: &RwLock<Arc<crate::net::Net>>,
    gc: &GameCfg,
    stats: &Stats,
    broken: &AtomicBool,
) {
    if seed_slots {
        for slot in 0..n_slots {
            if ready.closed() {
                break;
            }
            stats.used.fetch_add(1, Ordering::Relaxed);
            let mut source = GameStream::new(
                seed ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                *gc,
            );
            let mut data = Data::default();
            let n = Arc::clone(&*nets.read());
            let mut solver = source.next_solve(&n, &mut data);
            solver.pin(slot);
            ready.push(Job { source, solver, card: gpu, slot, replies: Vec::new(), data });
        }
    }
    loop {
        let batch = queue.drain_wave(wave);
        if batch.is_empty() {
            return;
        }
        let mut jobs = Vec::with_capacity(batch.len());
        let mut spans = Vec::with_capacity(batch.len());
        let mut calls: Vec<Call> = Vec::new();
        for (job, cs) in batch {
            spans.push(cs.len());
            calls.extend(cs);
            jobs.push(job);
        }
        let at = std::time::Instant::now();
        let answered = cuda.read().run(&calls, lane);
        let spent = at.elapsed();
        let Some(replies) = answered else {
            broken.store(true, Ordering::Relaxed);
            return;
        };
        assert_eq!(replies.len(), calls.len(), "one reply per call");
        stats.rounds.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(calls.iter().map(Call::rows).sum::<usize>() as u64, Ordering::Relaxed);
        stats.calls.fetch_add(calls.len() as u64, Ordering::Relaxed);
        stats.nanos.fetch_add(spent.as_nanos() as u64, Ordering::Relaxed);
        let mut rest = replies;
        for (mut job, n) in jobs.into_iter().zip(spans) {
            let tail = rest.split_off(n);
            job.replies = rest;
            rest = tail;
            ready.push(job);
        }
    }
}


pub struct Cards {
    queues: Vec<Arc<Queue<Ask>>>,
    seats: Mutex<Vec<(usize, usize)>>,
    free: Condvar,
    drivers: Vec<JoinHandle<()>>,
}

struct Ask {
    calls: Vec<Call>,
    back: std::sync::mpsc::Sender<Vec<Reply>>,
}

pub struct Seat<'a> {
    cards: &'a Cards,
    pub lane: usize,
    pub slot: usize,
}

impl Drop for Seat<'_> {
    fn drop(&mut self) {
        self.cards.seats.lock().push((self.lane, self.slot));
        self.cards.free.notify_one();
    }
}

impl Cards {
    #[cfg(feature = "gpu")]
    pub fn new(device: Device) -> Cards {
        let n = device.cards();
        let pipes = crate::cuda::PIPELINE;
        let device = Arc::new(device);
        let queues: Vec<Arc<Queue<Ask>>> =
            (0..n * pipes).map(|_| Arc::new(Queue::default())).collect();
        let mut drivers = Vec::with_capacity(n * pipes);
        for g in 0..n {
            for p in 0..pipes {
                let lane = g * pipes + p;
                let (queue, device) = (Arc::clone(&queues[lane]), Arc::clone(&device));
                drivers.push(
                    std::thread::Builder::new()
                        .name(format!("card-{g}.{p}"))
                        .spawn(move || loop {
                            let batch = queue.drain();
                            if batch.is_empty() {
                                return;
                            }
                            let mut backs = Vec::with_capacity(batch.len());
                            let mut spans = Vec::with_capacity(batch.len());
                            let mut calls: Vec<Call> = Vec::new();
                            for ask in batch {
                                spans.push(ask.calls.len());
                                calls.extend(ask.calls);
                                backs.push(ask.back);
                            }
                            let Some(replies) = device.run(&calls, lane) else {
                                return;
                            };
                            let mut rest = replies;
                            for (back, n) in backs.into_iter().zip(spans) {
                                let tail = rest.split_off(n);
                                let _ = back.send(rest);
                                rest = tail;
                            }
                        })
                        .expect("spawn driver thread"),
                );
            }
        }
        let mut free = Vec::new();
        for g in 0..n {
            for s in 0..device.slots(g) {
                free.push((g * pipes + s % pipes, s));
            }
        }
        Cards { queues, seats: Mutex::new(free), free: Condvar::new(), drivers }
    }

    pub fn seat(&self) -> Seat<'_> {
        let mut seats = self.seats.lock();
        loop {
            if let Some((lane, slot)) = seats.pop() {
                return Seat { cards: self, lane, slot };
            }
            self.free.wait(&mut seats);
        }
    }

    pub fn round(&self, lane: usize, calls: Vec<Call>) -> Option<Vec<Reply>> {
        let (back, replies) = std::sync::mpsc::channel();
        self.queues[lane].push(Ask { calls, back });
        replies.recv().ok()
    }
}

impl Drop for Cards {
    fn drop(&mut self) {
        for q in &self.queues {
            q.close();
        }
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}
