
use parking_lot::{Condvar, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

#[cfg(feature = "gpu")]
use crate::cuda::Device;
use crate::search::{Budget, Solver, Step};
use crate::contract::{Call, Reply};
use crate::net::Net;
use crate::selfplay::{Data, GameCfg, GameStream};


fn batched<T>(items: Vec<(T, Vec<Call>)>) -> (Vec<T>, Vec<usize>, Vec<Call>) {
    let mut heads = Vec::with_capacity(items.len());
    let mut spans = Vec::with_capacity(items.len());
    let mut calls = Vec::new();
    for (head, cs) in items {
        spans.push(cs.len());
        calls.extend(cs);
        heads.push(head);
    }
    (heads, spans, calls)
}

fn deal(mut replies: Vec<Reply>, spans: &[usize]) -> Vec<Vec<Reply>> {
    spans
        .iter()
        .map(|&n| {
            let tail = replies.split_off(n);
            std::mem::replace(&mut replies, tail)
        })
        .collect()
}

pub fn host_slots(budget: Budget) -> usize {
    let slot = budget.storage().host_slot_bytes() as u64;
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

const OPEN: u8 = 0;
const DRAINING: u8 = 1;
const CLOSED: u8 = 2;

struct Queue<T> {
    q: Mutex<std::collections::VecDeque<T>>,
    ready: Condvar,
    state: AtomicU8,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Queue {
            q: Mutex::new(std::collections::VecDeque::new()),
            ready: Condvar::new(),
            state: AtomicU8::new(OPEN),
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
            if self.state.load(Ordering::Relaxed) == CLOSED {
                return None;
            }
            self.ready.wait(&mut q);
        }
    }

    fn take(&self, least: usize, most: usize) -> Vec<T> {
        let mut q = self.q.lock();
        loop {
            let n = q.len();
            if n >= least || (self.state.load(Ordering::Relaxed) != OPEN && n > 0) {
                let out: Vec<T> = q.drain(..n.min(most)).collect();
                if !q.is_empty() {
                    self.ready.notify_one();
                }
                return out;
            }
            if self.state.load(Ordering::Relaxed) == CLOSED {
                return Vec::new();
            }
            self.ready.wait(&mut q);
        }
    }

    fn set(&self, state: u8) {
        let _held = self.q.lock();
        self.state.store(state, Ordering::Relaxed);
        self.ready.notify_all();
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

type JobQueue = Arc<Queue<(Job, Vec<Call>)>>;
type BackQueue = Arc<Queue<(Back, Vec<Call>)>>;

pub struct Farm {
    device: Vec<JobQueue>,
    ready: Arc<Queue<Job>>,
    #[cfg(feature = "gpu")]
    cuda: Arc<RwLock<Device>>,
    nets: Arc<RwLock<Arc<crate::net::Net>>>,
    collected: Arc<Mutex<Vec<Data>>>,
    retired: Arc<Mutex<Vec<Job>>>,
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
    pub slots: AtomicU64,
    pub budget_hits: AtomicU64,
    pub entity_hits: [AtomicU64; 8],
    pub slot_bytes: AtomicU64,
    pub slots_per_card: AtomicU64,
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

    pub fn entity_hits(&self) -> [u64; 8] {
        std::array::from_fn(|i| self.entity_hits[i].load(Ordering::Relaxed))
    }

    pub fn take_shapes(&self) -> Vec<[u32; 10]> {
        std::mem::take(&mut *self.shapes.lock())
    }
}

impl Farm {
    #[cfg(feature = "gpu")]
    pub fn new(seed: u64, workers: usize, gc: GameCfg, net: Arc<Net>, cuda: Device) -> Farm {
        assert!(workers > 0, "a farm needs at least one worker");
        let gpus = cuda.cards();
        let pipes = crate::cuda::PIPELINE;
        let per_gpu: Vec<usize> = (0..gpus).map(|g| cuda.slots(g)).collect();
        let n_slots: usize = per_gpu.iter().sum();
        let slot_bytes = cuda.slot_bytes();
        let slots_per_card = cuda.slots_per_card();
        let nets = Arc::new(RwLock::new(Arc::clone(&net)));
        let ready = Arc::new(Queue::default());
        let device: Vec<JobQueue> =
            (0..gpus).map(|_| Arc::new(Queue::default())).collect();
        let collected = Arc::new(Mutex::new(Vec::new()));
        let retired = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let broken = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::new(n_slots, slot_bytes, slots_per_card));
        let cuda = Arc::new(RwLock::new(cuda));

        for g in 0..gpus {
            let card_seed = seed.wrapping_mul(0x9E37_79B9) ^ g as u64;
            for slot in 0..per_gpu[g] {
                let key = card_seed ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut source = GameStream::new(key, gc);
                let mut data = Data::default();
                let mut solver = source.next_solve(&net, &mut data);
                solver.pin(slot);
                ready.push(Job { source, solver, card: g, slot, replies: Vec::new(), data });
            }
        }

        let hands: Vec<JoinHandle<()>> = (0..workers)
            .map(|t| {
                let (ready, device, nets, collected, retired, stopping, stats) = (
                    Arc::clone(&ready),
                    device.clone(),
                    Arc::clone(&nets),
                    Arc::clone(&collected),
                    Arc::clone(&retired),
                    Arc::clone(&stopping),
                    Arc::clone(&stats),
                );
                std::thread::Builder::new()
                    .name(format!("host-{t}"))
                    .spawn(move || {
                        while let Some(job) = ready.pop() {
                            advance_job(
                                job, &device, &nets, &collected, &retired, &stopping, &stats,
                            );
                        }
                    })
                    .expect("spawn host thread")
            })
            .collect();

        let mut drivers = Vec::with_capacity(gpus * pipes);
        for g in 0..gpus {
            for p in 0..pipes {
                let (queue, ready, cuda, stats, broken) = (
                    Arc::clone(&device[g]),
                    Arc::clone(&ready),
                    Arc::clone(&cuda),
                    Arc::clone(&stats),
                    Arc::clone(&broken),
                );
                let lane = g * pipes + p;
                let wave = per_gpu[g].div_ceil(pipes.max(1));
                drivers.push(
                    std::thread::Builder::new()
                        .name(format!("card-{g}.{p}"))
                        .spawn(move || drive_card(lane, wave, &queue, &ready, &cuda, &stats, &broken))
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
            retired,
            workers: hands,
            drivers,
            stopping,
            broken,
            stats,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn publish(&mut self, next: Arc<Net>) -> Result<(), String> {
        if Arc::ptr_eq(&self.nets.read(), &next) {
            return Ok(());
        }
        self.drain()?;
        self.cuda.write().set_weights(&next)?;
        *self.nets.write() = Arc::clone(&next);
        let mut jobs = std::mem::take(&mut *self.retired.lock());
        assert_eq!(jobs.len(), self.stats.slots() as usize, "a drain lost a game stream");
        for job in &mut jobs {
            job.solver = job.source.next_solve(&next, &mut job.data);
            job.solver.pin(job.slot);
        }
        for q in &self.device {
            q.set(OPEN);
        }
        self.stopping.store(false, Ordering::Release);
        for job in jobs {
            self.ready.push(job);
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), String> {
        let gate = self.nets.write();
        self.stopping.store(true, Ordering::Release);
        drop(gate);
        for q in &self.device {
            q.set(DRAINING);
        }
        while self.retired.lock().len() != self.stats.slots() as usize {
            if self.broken() {
                return Err("a card could not finish its solves".into());
            }
            std::thread::sleep(Duration::from_micros(200));
        }
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
        let _ = self.drain();
        self.ready.set(CLOSED);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        for q in &self.device {
            q.set(CLOSED);
        }
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}

fn advance_job(
    mut job: Job,
    device: &[JobQueue],
    nets: &RwLock<Arc<crate::net::Net>>,
    collected: &Mutex<Vec<Data>>,
    retired: &Mutex<Vec<Job>>,
    stopping: &AtomicBool,
    stats: &Stats,
) {
    let replies = std::mem::take(&mut job.replies);
    match job.solver.advance(&replies) {
        Step::Calls(calls) => device[job.card].push((job, calls)),
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
            let n = nets.read();
            if stopping.load(Ordering::Acquire) {
                drop(n);
                retired.lock().push(job);
                return;
            }
            job.solver = job.source.next_solve(&n, &mut job.data);
            job.solver.pin(job.slot);
            let Step::Calls(calls) = job.solver.advance(&[]) else {
                unreachable!("a fresh solve finishes without a round")
            };
            let card = job.card;
            device[card].push((job, calls));
            drop(n);
        }
    }
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn drive_card(
    lane: usize,
    wave: usize,
    queue: &Queue<(Job, Vec<Call>)>,
    ready: &Queue<Job>,
    cuda: &RwLock<Device>,
    stats: &Stats,
    broken: &AtomicBool,
) {
    loop {
        let (jobs, spans, calls) = batched(queue.take(wave, wave));
        if jobs.is_empty() {
            return;
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
        for (mut job, replies) in jobs.into_iter().zip(deal(replies, &spans)) {
            job.replies = replies;
            ready.push(job);
        }
    }
}


type Back = std::sync::mpsc::Sender<Vec<Reply>>;

pub struct Cards {
    queues: Vec<BackQueue>,
    seats: Mutex<Vec<(usize, usize)>>,
    free: Condvar,
    drivers: Vec<JoinHandle<()>>,
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
        let queues: Vec<BackQueue> =
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
                            let (backs, spans, calls) = batched(queue.take(1, usize::MAX));
                            if backs.is_empty() {
                                return;
                            }
                            let Some(replies) = device.run(&calls, lane) else {
                                return;
                            };
                            for (back, replies) in backs.into_iter().zip(deal(replies, &spans)) {
                                let _ = back.send(replies);
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
        self.queues[lane].push((back, calls));
        replies.recv().ok()
    }
}

impl Drop for Cards {
    fn drop(&mut self) {
        for q in &self.queues {
            q.set(CLOSED);
        }
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}
