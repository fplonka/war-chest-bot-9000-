//! Where many solves share one forward pass.
//!
//! A solve alone is a bad shape for an accelerator: its leaf batch is a couple
//! of hundred rows and it wants one every CFR iteration. Student of Games says
//! the same thing about its own actors — each runs several games at once and
//! batches the network evaluations — so the unit of inference here is a *round*
//! across every solve that is ready, not a solve.
//!
//! A solve is a state machine, not a thread: it consumes the replies it was
//! waiting for, does its host-side work, and says what it wants next. So it
//! sits in one of two queues. Two drivers a GPU share that GPU's queue, so
//! one set can grow while the other iterates; a pool of workers, one per
//! core, does the host side. Neither waits for the other, which is what lets
//! one solve's growth overlap another's device work.
//!
//! How many solves are in flight is the number of slots. A slot is allocated
//! once at the budget; admission is a pop from a free list, and a solve that
//! would exceed the budget is truncated rather than grown.

use parking_lot::{Condvar, Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::net::Net;
use crate::pbs::Belief;
use crate::search::{Budget, Cfg, Cfr, Solved, Solver, Step};
use crate::selfplay::{Data, GameCfg, GameStream};
use crate::state::State;
use rayon::prelude::*;

/// One solve's network work for this round.
///
/// The three calls are the network's whole surface: the trunk over a new
/// leaf's physical state, the encoder over a new config, and the join that
/// every CFR iteration pays for. `cards` rides along with the first two
/// because a batch spans solves and each draft has its own card table.
#[derive(Clone)]
pub enum Call {
    /// New leaves: public states in, board vectors and the join cache out.
    ///
    /// `solve` names the solve these rows belong to and `at` the row they
    /// start at, so `at == 0` is a fresh solve and anything else extends one.
    /// Both exist because the board vectors and the join cache are properties
    /// of a leaf rather than of an iteration: the backend keeps them, and the
    /// sixty-four join calls that follow do not carry them again.
    ///
    /// Rows and boards are counted apart. The trunk reads the public state and
    /// nothing else, and coin plays commute, so a sixth to a quarter of a
    /// solve's rows repeat a public state an earlier row already carried.
    /// `xpub` holds the distinct ones and `board_of` says which board each of
    /// this call's rows reads.
    Trunk {
        solve: usize,
        at: usize,
        rows: usize,
        /// One entry per row of this call: the board it reads, indexed from
        /// the start of the solve.
        board_of: Vec<u32>,
        boards_at: usize,
        boards: usize,
        xpub: Vec<f32>,
        cards: Vec<f32>,
        /// The belief index of exactly these rows: which config each query's
        /// support names, and where each query's span starts. A leaf's support
        /// is fixed when the leaf is made, so it travels once, here.
        cidx: Vec<u32>,
        coff: Vec<u32>,
    },
    /// New configs: `f(c)` for the readout and `g(c)` for the pooling. Both
    /// stay with the backend for the same reason the board vectors do — they
    /// are properties of a config, and every iteration reads them.
    Configs {
        solve: usize,
        at: usize,
        phi: Vec<f32>,
        owner: Vec<u32>,
        cards: Vec<f32>,
        n: usize,
    },
    /// The tree, brought up to date with the host's.
    ///
    /// Growth is the only part of a solve the host still runs: it holds the
    /// game rules, so it turns the leaves an expansion sampled into decision
    /// nodes and describes them. `Contract` is append-only apart from the rows
    /// of those leaves, so `from` says where the rewriting starts.
    Tree {
        solve: usize,
        /// Everything the card has yet to be told about this solve, already in
        /// the shape the wire wants.
        writes: Writes,
        /// The first call of a solve. A card's solve slot is reused, so this
        /// is what tells the backend to forget the solve that held it before.
        fresh: bool,
        /// Arena lengths, so the device can fit what it holds.
        ncells: usize,
        nreach: usize,
        nvals: usize,
        /// Where each level of the tree starts. The launch loop walks levels,
        /// so the host needs this as well as the card.
        levels: Vec<u32>,
        /// How many terminal leaves the solve holds, for the grid that scores
        /// them.
        nterm: usize,
        /// The seed of the expansion's own random stream, sent once.
        seed: Option<u64>,
        /// Nodes whose policy prior the card is to fill this round, and the two
        /// things it needs that it does not already hold.
        prime: Vec<Prime>,
        /// Five words an action -- kind, coin slot, three hexes -- which the
        /// card expands into the encoder's one-hot input.
        acts: Vec<u32>,
        /// Which action each of a primed node's strategy cells stands for.
        cells: Vec<u32>,
        /// The softmax temperature the prior is formed at.
        prior_temp: f32,
    },
    /// One CFR iteration and one expansion phase.
    ///
    /// The whole of what a GT-CFR round asks of the device: reach forward, the
    /// network at every leaf, backpropagation and the regret update for both
    /// players, the average strategy, and the trajectories that say where the
    /// tree grows next. What comes back is the sampled leaves and nothing else.
    Iterate {
        solve: usize,
        /// Where this solve's iterate count stands, and how many iterations to
        /// run from there. A CFR iterate is weighted by how many came before
        /// it, and a round holds solves at different points of their own
        /// sixty-four, so the weights are the solve's and the backend computes
        /// them per solve rather than taking one set for the batch.
        step: usize,
        iters: usize,
        /// Distinct leaves to take after *each* of those iterations, so the
        /// call comes back with `iters * expand` slots. A phase draws
        /// trajectories until it has that many leaves no phase of this round
        /// has taken. Zero once the tree has spent its node budget, and then
        /// the round takes nothing and the host is only woken to end the
        /// solve.
        expand: usize,
        cfr: Cfr,
        puct: f32,
    },
    /// What the host needs back once the solve is done: the reference
    /// strategy at the nodes it asks about, and their values and beliefs.
    Read {
        solve: usize,
        touched: [bool; 2],
        /// Slices of the device arenas the host wants back: the root's value
        /// row per player, the root's strategy cells, and the reach at each
        /// leaf the caller sampled. The host knows every offset from its own
        /// copy of the tree, so nothing here has to be looked up on the card.
        ///
        /// An empty value row means the caller wants no values, and the value
        /// pass under the reference strategy does not run. Every solve reads
        /// its root policy; only a solve that is collected reads targets.
        vals_at: [(u32, u32); 2],
        policy_at: (u32, u32),
        reach_at: Vec<(u32, u32)>,
    },
}

/// One node whose policy prior a round is to fill.
///
/// The card holds everything this reads but two facts: what an action *is*, and
/// which action each strategy cell stands for. Both are a few kilobytes a node.
#[derive(Clone, Copy)]
pub struct Prime {
    pub node: u32,
    /// The node's row in this solve's leaf batch, where its board vector is.
    pub row: u32,
    /// Where its actions start in `acts`, and how many it offers.
    pub at: u32,
    pub na: u32,
    /// Where its cells start in `cells`.
    pub cell_at: u32,
    /// Configs the acting player holds here, which is the grid the prior takes.
    pub nc: u32,
}

/// One of a solve's resident arrays, as a run of the round's blob names it.
///
/// The order is the vocabulary the solver and the backend share and nothing
/// else depends on it; `Solve::plan` in the device driver is the other half.
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

/// Everything one solve tells the card to write this round, concatenated.
///
/// A solve builds this on the worker that grew the tree. The driver says where
/// each run lands on the card. Floats travel as their bits, because the scatter
/// kernel moves words and does not care which they are.
#[derive(Clone, Default)]
pub struct Writes {
    pub blob: Vec<u32>,
    pub runs: Vec<Run>,
}

/// One run: where it goes, and where its words are. `start` is explicit rather
/// than implied by order, so two arrays can be given the same words.
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

    /// Bytes the card holds as words. `Contract` keeps a node's kind and
    /// player in one byte each; the device reads them as `unsigned int`.
    pub fn u8s(&mut self, d: Dst, at: usize, src: &[u8]) {
        self.run(d, at, src.iter().map(|&x| x as u32), src.len());
    }

    /// The same words into two arrays. The tail a growth appends is uniform
    /// over each legal row, which is where CFR starts *and* what the prior is
    /// until the policy head has spoken -- so `cur` and `prior` hold the same
    /// numbers there and there is no reason to carry them twice.
    ///
    /// One call, not two, because an empty tail must add nothing to either: a
    /// round that grew no cells would otherwise hand the second array whatever
    /// run happened to be last.
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
        // The scatter kernel places every run at once, so two runs that reach
        // the same array must not touch the same words -- a thing a host-side
        // replay in run order would not notice.
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

/// What a call gives back. The trunk returns the board vector and its join
/// cache, the encoder returns `f`, `g` and the policy's `f_p`, and the join
/// returns `h` alone.
#[derive(Default)]
pub struct Reply {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    /// The leaves the round's phases took, phase by phase and distinct over
    /// the whole round. `NO_ROW` pads a phase that gave up short, which is a
    /// phase that spent its draws on leaves the round already held.
    pub leaves: Vec<u32>,
}

impl Call {
    /// Run this call on the CPU network. The batched driver runs the same
    /// arithmetic on the device; this is the reference both are held to.
    pub fn run(&self, net: &Net) -> Reply {
        let mut r = Reply::default();
        match self {
            Call::Trunk { xpub, cards, boards, .. } => {
                // One row a distinct public state in `xpub`, and one card
                // table per solve -- `board` reads the physical view of that.
                net.board(xpub, cards, *boards, CARD_ROWS, &mut r.a);
                net.join_cache(&r.a, *boards, &mut r.b);
            }
            Call::Configs { phi, owner, cards, n, .. } => {
                net.configs(phi, owner, *n, cards, &mut r.a, &mut r.b, &mut r.c);
            }
            Call::Tree { .. } | Call::Iterate { .. } | Call::Read { .. } => {
                unreachable!("the CFR loop needs the resident state")
            }
        }
        r
    }

    /// Which of the three batches this call belongs to. The device groups a
    /// round by kind and runs each group once.
    pub fn kind(&self) -> usize {
        match self {
            Call::Trunk { .. } => 0,
            Call::Configs { .. } => 1,
            Call::Tree { .. } => 2,
            Call::Iterate { .. } => 3,
            Call::Read { .. } => 4,
        }
    }

    /// Which solve raised this call. A backend keeps a solve's board vectors
    /// between its iterations, so every call of one solve must reach the same
    /// backend — which is what a round shards on.
    pub fn solve(&self) -> usize {
        match self {
            Call::Trunk { solve, .. }
            | Call::Configs { solve, .. }
            | Call::Tree { solve, .. }
            | Call::Iterate { solve, .. }
            | Call::Read { solve, .. } => *solve,
        }
    }

    /// Rows this call contributes to its batch, for the round report.
    pub fn rows(&self) -> usize {
        match self {
            Call::Trunk { rows, .. } => *rows,
            Call::Configs { n, .. } => *n,
            Call::Tree { .. } | Call::Iterate { .. } | Call::Read { .. } => 0,
        }
    }
}


/// A solve's card table: one row per seat view. Fixed at the draft, so it is
/// built once per solve and every leaf of that solve reads it.
pub const CARD_ROWS: usize = 2;

/// What actually runs a round's batch.
///
/// A run always uses the device. `Reference` is the CPU network answering each
/// call on its own — the oracle the device is held to in the parity test, and
/// what the farm's own tests drive. It is deliberately not a fallback: a run
/// that cannot reach a GPU should fail, not quietly become a hundred times
/// slower.
pub enum Backend {
    #[cfg(feature = "gpu")]
    Cuda(crate::cuda::Device),
    Reference(Net),
}

impl Backend {
    pub fn run(&self, calls: &[Call], #[allow(unused)] card: usize) -> Option<Vec<Reply>> {
        match self {
            // Every call in a round is independent, and this is the only
            // thread the reference backend has anything for.
            Backend::Reference(net) => Some(calls.par_iter().map(|c| c.run(net)).collect()),
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.run(calls, card),
        }
    }

    /// How many GPUs this backend has, and so how many queues the farm runs.
    pub fn cards(&self) -> usize {
        match self {
            Backend::Reference(_) => 1,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.cards(),
        }
    }

    /// Cards per GPU. Two, so one set can grow while the other iterates.
    pub fn pipelines(&self) -> usize {
        match self {
            Backend::Reference(_) => 1,
            #[cfg(feature = "gpu")]
            Backend::Cuda(_) => crate::cuda::PIPELINE,
        }
    }

    /// Slots this card holds. Admission pops one; none free means wait.
    pub fn slots(&self, #[allow(unused)] card: usize) -> usize {
        match self {
            Backend::Reference(_) => 0,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.slots(card),
        }
    }

    pub fn slot_bytes(&self) -> usize {
        match self {
            Backend::Reference(_) => 0,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.slot_bytes(),
        }
    }

    pub fn slots_per_card(&self) -> usize {
        match self {
            Backend::Reference(_) => 0,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.slots_per_card(),
        }
    }

    /// Whether this backend runs the CFR loop itself rather than answering
    /// network calls alone.
    pub fn keeps_the_solve(&self) -> bool {
        match self {
            Backend::Reference(_) => false,
            #[cfg(feature = "gpu")]
            Backend::Cuda(_) => true,
        }
    }

    /// The weights this backend evaluates with, for the readout the solver
    /// still does itself.
    pub fn net(&self) -> &Net {
        match self {
            Backend::Reference(net) => net,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.net(),
        }
    }

    /// Evaluate with new weights from here on. The backend itself survives:
    /// rebuilding it would tear down a CUDA context and recompile every kernel
    /// for a change that touches three arrays.
    pub fn set_net(&mut self, net: Net) -> Result<(), String> {
        match self {
            Backend::Reference(old) => {
                *old = net;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.set_weights(net),
        }
    }
}


/// Host-side slots that fit in the memory the process does not already hold.
///
/// A slot is `Budget::host_slot_bytes`. The farm never admits more than this, and
/// the card never carves more than this, so host OOM is not a thing that can
/// happen at admission.
pub fn host_slots(budget: Budget) -> usize {
    let slot = budget.host_slot_bytes() as u64;
    (host_free() / slot.max(1)) as usize
}

/// Bytes the farm may still hold in solves.
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

/// A queue with parked consumers, closed once when the farm winds down.
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

    /// Both card pipes wait on this queue with different wave sizes. Wake both
    /// so the pipe whose threshold was reached always sees the new work.
    fn push_wave(&self, x: T) {
        self.q.lock().push_back(x);
        self.ready.notify_all();
    }

    /// The oldest item, or `None` once the queue is closed and empty.
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

    fn try_pop(&self) -> Option<T> {
        self.q.lock().pop_front()
    }

    fn wake(&self) {
        self.ready.notify_all();
    }

    /// Wait until `wave` items are ready, then take that many. Leftover items
    /// wake the other pipe. Empty once the queue is closed and empty.
    fn drain_wave(&self, target: impl Fn() -> usize) -> Vec<T> {
        let mut q = self.q.lock();
        loop {
            let n = q.len();
            let wave = target();
            if wave > 0 && (n >= wave || (self.closed.load(Ordering::Relaxed) && n > 0)) {
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

/// Where a job's solves come from.
///
/// `Play` is the run: solves come from games played forward, and their rows are
/// kept. `Roots` is the bench, and it exists because the run cannot be
/// measured. A solve's cost varies twenty-six fold with how far into a game its
/// root sits, so a probe's rate moves by two-fold with nothing but which phase
/// its games happened to reach — 16.6, 16.2, 12.1 and 8.0 solves/s were
/// measured across consecutive probes of one build. Cycling a corpus of roots
/// sampled from a real run holds the *mix* of costs in flight stationary, which
/// is the one property that makes two builds comparable in seconds rather than
/// in half an hour.
pub enum Work {
    Play(GameCfg),
    Roots {
        roots: Arc<Vec<(State, [Belief; 2])>>,
        cfg: Cfg,
        recursive_rate: f32,
    },
}

/// One job's side of `Work`: what it does between two solves.
///
/// A source hands out the next solve and takes the finished one back. What
/// happens in between is a game, which is why this is a state machine and not
/// a loop — the solve it is waiting on belongs to the farm, not to a thread.
enum Source {
    Play(GameStream),
    /// Where in the corpus this job is, and its own random stream. Each job
    /// starts at its own place and cycles from there, so the roots in flight
    /// are a spread of the corpus at every moment of a bench and the same
    /// spread between two builds.
    Roots { at: usize, rng: crate::rng::Rng },
}

impl Source {
    fn new(work: &Work, seed: u64, i: usize) -> Source {
        match work {
            Work::Play(gc) => Source::Play(GameStream::new(seed, *gc)),
            Work::Roots { .. } => Source::Roots { at: i, rng: crate::rng::Rng::new(seed) },
        }
    }

    /// The next solve this source wants run.
    fn next(&mut self, work: &Work, nets: &Arc<crate::search::Nets>, out: &mut Data) -> Solver {
        match (self, work) {
            (Source::Play(stream), _) => stream.next_solve(nets, out),
            (Source::Roots { at, rng }, Work::Roots { roots, cfg, recursive_rate }) => {
                let (s, bel) = &roots[*at % roots.len()];
                *at += 1;
                crate::selfplay::query_solver(nets, *cfg, *recursive_rate, s, bel, rng)
            }
            (Source::Roots { .. }, Work::Play(_)) => unreachable!("a source matches its work"),
        }
    }

    /// Take the finished solve back and keep what it produced.
    fn take(&mut self, sv: &Solver, solved: Option<Solved>, out: &mut Data) {
        match self {
            Source::Play(stream) => stream.keep(sv, solved, out),
            // A bench root is solved for its own row and nothing follows it.
            Source::Roots { .. } => {
                crate::selfplay::keep_query(sv, solved, out);
            }
        }
    }
}

/// One solve in flight, and everything that outlives it.
struct Job {
    solver: Solver,
    /// The card that holds this solve's arenas, and which of its slots. A GPU
    /// keeps a solve's board vectors between its rounds, so both are fixed for
    /// as long as the job lives. Either pipe of that GPU may run the round.
    card: usize,
    slot: usize,
    /// What the last round gave back, waiting for the host work that reads it.
    replies: Vec<Reply>,
    next: Next,
}

enum Next {
    Stream { source: Source, data: Data },
    Arena { id: u32 },
}

pub struct ArenaDone {
    pub id: u32,
    pub solver: Solver,
}

struct ArenaState {
    pending: Mutex<VecDeque<(u32, Solver)>>,
    free: Mutex<Vec<(usize, usize)>>,
    done: Queue<ArenaDone>,
    active: Vec<AtomicU64>,
}

/// Many solves in flight in one process, and one thing that evaluates for all
/// of them.
///
/// Two kinds of thread. Two drivers a GPU share one queue, so one set of solves
/// can grow on the host while the other iterates on the card. A pool of
/// workers, one per core, does the host side: growth, which is the game's
/// rules, and the game around it. Neither ever waits for the other, so one
/// solve's growth overlaps another's device work — which a barrier could not
/// do, because it made a round cost whatever its slowest member cost.
pub struct Farm {
    /// One queue a GPU: solves whose next round either of its cards may run.
    device: Vec<Arc<Queue<(Job, Vec<Call>)>>>,
    /// Solves whose replies are in and whose host work wants a core.
    ready: Arc<Queue<Job>>,
    /// Shared because every card evaluates against it, and locked because a
    /// publish rewrites the weights under them.
    backend: Arc<RwLock<Backend>>,
    /// The copy a solve reads for the work it still does itself. Replaced
    /// whole, so a solve that has started keeps the weights it started with.
    nets: Arc<RwLock<Arc<crate::search::Nets>>>,
    collected: Arc<Mutex<Vec<Data>>>,
    workers: Vec<JoinHandle<()>>,
    drivers: Vec<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    broken: Arc<AtomicBool>,
    stats: Arc<Stats>,
    arena: Option<Arc<ArenaState>>,
}

/// What the rounds carried, for the utilisation report. Calls per round is the
/// number that matters: it is how many solves shared a forward pass.
#[derive(Default)]
pub struct Stats {
    pub solves: AtomicU64,
    pub rounds: AtomicU64,
    pub rows: AtomicU64,
    pub calls: AtomicU64,
    /// Time inside the backend. Against wall clock this says how much of a
    /// round is the batch and how much is everything around it.
    pub nanos: AtomicU64,
    /// Slots this farm holds, and how many currently have a solve.
    slots: AtomicU64,
    used: AtomicU64,
    /// Solves that hit the budget. A slot is a percentile; this is the rate
    /// that argues with it.
    budget_hits: AtomicU64,
    /// Solves that hit each entity's cap, in `Ent::ALL` order.
    entity_hits: [AtomicU64; 8],
    slot_bytes: AtomicU64,
    slots_per_card: AtomicU64,
    /// Finished-solve entity counts, drained by collect. Eight per solve.
    shapes: Mutex<Vec<[u32; 8]>>,
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

    pub fn take_shapes(&self) -> Vec<[u32; 8]> {
        std::mem::take(&mut *self.shapes.lock())
    }
}

impl Farm {
    /// Start `workers` host threads and two drivers per GPU, with one job per
    /// slot. A slot is allocated at the budget; there is nothing to admit
    /// against after that.
    pub fn new(seed: u64, workers: usize, work: Work, backend: Backend) -> Farm {
        Self::build(seed, workers, Some(work), backend)
    }

    pub fn arena(workers: usize, backend: Backend) -> Farm {
        Self::build(0, workers, None, backend)
    }

    fn build(seed: u64, workers: usize, work: Option<Work>, backend: Backend) -> Farm {
        assert!(workers > 0, "a farm needs at least one worker");
        let gpus = backend.cards();
        let pipes = backend.pipelines();
        let cuda = backend.keeps_the_solve();
        let per_gpu: Vec<usize> = (0..gpus)
            .map(|g| if cuda { backend.slots(g) } else { workers })
            .collect();
        let n_slots: usize = per_gpu.iter().sum();
        let slot_bytes = backend.slot_bytes();
        let slots_per_card = backend.slots_per_card();
        let work = work.map(Arc::new);
        let nets = Arc::new(RwLock::new(Arc::new(crate::search::Nets {
            value: backend.net().clone(),
            device: backend.keeps_the_solve(),
        })));
        let ready = Arc::new(Queue::default());
        let device: Vec<Arc<Queue<(Job, Vec<Call>)>>> =
            (0..gpus).map(|_| Arc::new(Queue::default())).collect();
        let collected = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let broken = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::new(n_slots, slot_bytes, slots_per_card));
        let arena = work.is_none().then(|| {
            let free = (0..per_gpu.iter().copied().max().unwrap_or(0))
                .flat_map(|slot| {
                    per_gpu
                        .iter()
                        .enumerate()
                        .filter(move |(_, n)| slot < **n)
                        .map(move |(card, _)| (card, slot))
                })
                .collect();
            Arc::new(ArenaState {
                pending: Mutex::new(VecDeque::new()),
                free: Mutex::new(free),
                done: Queue::default(),
                active: (0..gpus).map(|_| AtomicU64::new(0)).collect(),
            })
        });
        let backend = Arc::new(RwLock::new(backend));

        let hands: Vec<JoinHandle<()>> = (0..workers)
            .map(|t| {
                let (ready, device, nets, collected, work, stopping, stats, arena) = (
                    Arc::clone(&ready),
                    device.clone(),
                    Arc::clone(&nets),
                    Arc::clone(&collected),
                    work.clone(),
                    Arc::clone(&stopping),
                    Arc::clone(&stats),
                    arena.clone(),
                );
                std::thread::Builder::new()
                    .name(format!("host-{t}"))
                    .spawn(move || {
                        while let Some(job) = ready.pop() {
                            advance_job(
                                job, &device, &ready, &nets, work.as_deref(), &collected,
                                &stopping, &stats, arena.as_deref(),
                            );
                        }
                    })
                    .expect("spawn host thread")
            })
            .collect();

        let mut drivers = Vec::with_capacity(gpus * pipes);
        for g in 0..gpus {
            for p in 0..pipes {
                let (queue, ready, backend, nets, work, stats, broken, arena) = (
                    Arc::clone(&device[g]),
                    Arc::clone(&ready),
                    Arc::clone(&backend),
                    Arc::clone(&nets),
                    work.clone(),
                    Arc::clone(&stats),
                    Arc::clone(&broken),
                    arena.clone(),
                );
                let n = per_gpu[g];
                let seed = seed.wrapping_mul(0x9E37_79B9) ^ g as u64 ^ (p as u64) << 32;
                let lane = g * pipes + p;
                drivers.push(
                    std::thread::Builder::new()
                        .name(format!("card-{g}.{p}"))
                        .spawn(move || {
                            drive_card(
                                g, p, lane, gpus, pipes, n, p == 0 && work.is_some(), seed,
                                &queue, &ready, &backend, &nets, work.as_deref(), &stats,
                                &broken, arena.as_deref(),
                            )
                        })
                        .expect("spawn driver thread"),
                );
            }
        }

        Farm {
            device,
            ready,
            backend,
            nets,
            collected,
            workers: hands,
            drivers,
            stopping,
            broken,
            stats,
            arena,
        }
    }

    /// Install new weights, in the backend and in the copy a solve keeps for
    /// the work it does itself. The write lock waits for every round in flight,
    /// so no round is ever evaluated against two different networks.
    pub fn publish(&mut self, net: Net) -> Result<(), String> {
        self.backend.write().set_net(net.clone())?;
        let device = self.backend.read().keeps_the_solve();
        *self.nets.write() = Arc::new(crate::search::Nets { value: net, device });
        Ok(())
    }

    /// Wait until the farm has produced at least `solves` rows, then hand over
    /// everything it produced.
    pub fn drive(&mut self, solves: usize) -> Data {
        let mut out = Data::default();
        loop {
            for d in self.collected.lock().drain(..) {
                out.merge(d);
            }
            if out.soff.len() >= solves {
                return out;
            }
            // A card that could not answer a round takes its solves with it, so
            // waiting for them is waiting for ever -- which is what a run did
            // when a card filled, silently, with the panic printed to a log
            // nobody was reading.
            if self.stopping.load(Ordering::Relaxed) || self.broken.load(Ordering::Relaxed) {
                return out;
            }
            // Parked, not spinning: a core busy-waiting here is a core taken
            // from the work it is waiting for.
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Whether a card failed a round, which is not recoverable.
    pub fn broken(&self) -> bool {
        self.broken.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The network the farm evaluates with.
    pub fn value(&self) -> Arc<crate::search::Nets> {
        Arc::clone(&*self.nets.read())
    }

    pub fn solve(&self, solves: Vec<(u32, Solver)>) -> Result<Vec<ArenaDone>, String> {
        let arena = self.arena.as_ref().expect("only an arena farm accepts external solves");
        let count = solves.len();
        arena.pending.lock().extend(solves);
        {
            let mut free = arena.free.lock();
            let mut pending = arena.pending.lock();
            while !free.is_empty() && !pending.is_empty() {
                let (card, slot) = free.pop().unwrap();
                let (id, mut solver) = pending.pop_front().unwrap();
                solver.pin(slot);
                arena.active[card].fetch_add(1, Ordering::Relaxed);
                self.stats.used.fetch_add(1, Ordering::Relaxed);
                self.ready.push(Job {
                    solver,
                    card,
                    slot,
                    replies: Vec::new(),
                    next: Next::Arena { id },
                });
            }
        }
        for q in &self.device {
            q.wake();
        }

        let mut done = Vec::with_capacity(count);
        while done.len() < count {
            if let Some(solved) = arena.done.try_pop() {
                done.push(solved);
            } else if self.broken() {
                return Err("a card could not answer a round".into());
            } else {
                std::thread::sleep(Duration::from_micros(200));
            }
        }
        Ok(done)
    }
}

impl Drop for Farm {
    fn drop(&mut self) {
        // Close the host side first and let the cards answer whatever is still
        // in their queues, so no worker is left parked on a round that will
        // never come back.
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

/// One job's turn on the host: consume the last round's replies, do the growth
/// they made possible, and hand the job to its card.
///
/// A solve that finishes is replaced here rather than through the queue, and a
/// solve that wants nothing from the card -- which is every solve when there
/// are no weights yet -- runs to its end without ever leaving this loop.
fn advance_job(
    mut job: Job,
    device: &[Arc<Queue<(Job, Vec<Call>)>>],
    ready: &Queue<Job>,
    nets: &RwLock<Arc<crate::search::Nets>>,
    work: Option<&Work>,
    collected: &Mutex<Vec<Data>>,
    stopping: &AtomicBool,
    stats: &Stats,
    arena: Option<&ArenaState>,
) {
    let mut replies = std::mem::take(&mut job.replies);
    loop {
        match job.solver.advance(&replies) {
            Step::Calls(calls) => return device[job.card].push_wave((job, calls)),
            Step::Done(solved) => {
                stats.solves.fetch_add(1, Ordering::Relaxed);
                let mask = job.solver.hit_mask();
                if mask != 0 {
                    stats.budget_hits.fetch_add(1, Ordering::Relaxed);
                    for i in 0..8 {
                        if mask & (1 << i) != 0 {
                            stats.entity_hits[i].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                stats.shapes.lock().push(job.solver.counts());
                let Job { solver, card, slot, next, .. } = job;
                match next {
                    Next::Stream { mut source, mut data } => {
                        source.take(&solver, solved, &mut data);
                        collected.lock().push(std::mem::take(&mut data));
                        if stopping.load(Ordering::Relaxed) {
                            stats.used.fetch_sub(1, Ordering::Relaxed);
                            return;
                        }
                        let n = Arc::clone(&*nets.read());
                        let mut solver = source.next(work.unwrap(), &n, &mut data);
                        solver.pin(slot);
                        job = Job {
                            solver,
                            card,
                            slot,
                            replies: Vec::new(),
                            next: Next::Stream { source, data },
                        };
                        replies = Vec::new();
                    }
                    Next::Arena { id } => {
                        let arena = arena.unwrap();
                        arena.done.push(ArenaDone { id, solver });
                        if let Some((id, mut solver)) = arena.pending.lock().pop_front() {
                            solver.pin(slot);
                            ready.push(Job {
                                solver,
                                card,
                                slot,
                                replies: Vec::new(),
                                next: Next::Arena { id },
                            });
                        } else {
                            arena.free.lock().push((card, slot));
                            arena.active[card].fetch_sub(1, Ordering::Relaxed);
                            stats.used.fetch_sub(1, Ordering::Relaxed);
                            device[card].wake();
                        }
                        return;
                    }
                }
            }
        }
    }
}

/// One card's rounds, and the solves that occupy its GPU's slots.
#[allow(clippy::too_many_arguments)]
fn drive_card(
    gpu: usize,
    pipe: usize,
    lane: usize,
    gpus: usize,
    pipes: usize,
    n_slots: usize,
    seed_slots: bool,
    seed: u64,
    queue: &Queue<(Job, Vec<Call>)>,
    ready: &Queue<Job>,
    backend: &RwLock<Backend>,
    nets: &RwLock<Arc<crate::search::Nets>>,
    work: Option<&Work>,
    stats: &Stats,
    broken: &AtomicBool,
    arena: Option<&ArenaState>,
) {
    if seed_slots {
        let work = work.unwrap();
        for slot in 0..n_slots {
            if ready.closed() {
                break;
            }
            stats.used.fetch_add(1, Ordering::Relaxed);
            let mut source = Source::new(
                work,
                seed ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                slot * gpus + gpu,
            );
            let mut data = Data::default();
            let n = Arc::clone(&*nets.read());
            let mut solver = source.next(work, &n, &mut data);
            solver.pin(slot);
            ready.push(Job {
                solver,
                card: gpu,
                slot,
                replies: Vec::new(),
                next: Next::Stream { source, data },
            });
        }
    }
    loop {
        let batch = queue.drain_wave(|| {
            let active = arena.map_or(n_slots, |a| a.active[gpu].load(Ordering::Relaxed) as usize);
            (active + pipes - 1 - pipe) / pipes
        });
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
        let answered = backend.read().run(&calls, lane);
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::{Agent, Collect};

    /// Small random weights, so the network actually makes calls rather than
    /// being skipped as absent.
    fn small_net(seed: u64) -> Net {
        let mut r = crate::rng::Rng::new(seed);
        let l = crate::net::NetLayout::new();
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect()
        };
        let (w, b) = (draw(l.w_len), draw(l.b_len));
        let mut ln = vec![0.0; l.ln_len];
        for n in &l.norms {
            ln[n.g..n.g + n.width].fill(1.0);
        }
        Net::from_flat(&w, &b, &ln).expect("small net")
    }

    /// Many solves in flight, one thing that evaluates for all of them: the
    /// farm must produce well-formed rows, share a round between solves, and
    /// shut down without stranding anyone.
    #[test]
    fn a_farm_batches_many_solves_into_one_round() {
        const WORKERS: usize = 4;
        let cfg = crate::search::Cfg { s: 8, c: 1.0, ..Default::default() };
        let gc = GameCfg {
            agents: [Agent::Sog { cfg }; 2],
            collect: Collect::Sog,
            explore: 0.1,
            random_draft: true,
            p_td1: 0.0,
            query_rate: 0.9,
            recursive_rate: 0.1,
        };
        let mut farm = Farm::new(5, WORKERS, Work::Play(gc), Backend::Reference(small_net(0x2E57)));
        let data = farm.drive(48);
        assert!(data.soff.len() >= 48, "only {} solves", data.soff.len());
        assert_eq!(data.nv, data.soff.len(), "a solve must store one row");
        assert_eq!(data.coff.len(), 2 * data.nv + 1, "ragged arena is malformed");
        let read = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64;
        let s = farm.stats();
        assert!(read(&s.rounds) > 0.0, "no round ever ran");
        assert!(read(&s.rows) > 0.0, "rounds carried no rows");
        // The whole point: a round is one forward pass shared by every solve
        // that was ready, not one pass per solve.
        let calls = read(&s.calls) / read(&s.rounds);
        assert!(calls > 2.0, "rounds averaged only {calls:.1} calls");
    }

    /// The population a farm holds is its slots, one solve each.
    #[test]
    fn a_farm_holds_one_solve_per_slot() {
        const WORKERS: usize = 4;
        let cfg = crate::search::Cfg { s: 8, c: 1.0, ..Default::default() };
        let gc = GameCfg {
            agents: [Agent::Sog { cfg }; 2],
            collect: Collect::Sog,
            explore: 0.1,
            random_draft: true,
            p_td1: 0.0,
            query_rate: 0.9,
            recursive_rate: 0.1,
        };
        let mut farm = Farm::new(5, WORKERS, Work::Play(gc), Backend::Reference(small_net(0x2E57)));
        let got = farm.drive(24);
        assert!(got.soff.len() >= 24, "only {} solves", got.soff.len());
        let s = farm.stats();
        assert_eq!(s.slots(), WORKERS as u64, "a reference farm is one slot a worker");
        assert!(s.used() <= s.slots(), "{} used, {} slots", s.used(), s.slots());
    }

    #[test]
    fn an_arena_wave_reuses_its_slots() {
        const WORKERS: usize = 4;
        let cfg = crate::search::Cfg { s: 8, c: 1.0, ..Default::default() };
        let gc = GameCfg {
            agents: [Agent::Sog { cfg }; 2],
            collect: Collect::Sog,
            explore: 0.1,
            random_draft: true,
            p_td1: 0.0,
            query_rate: 0.0,
            recursive_rate: 0.0,
        };
        let farm = Farm::arena(WORKERS, Backend::Reference(small_net(0xA7E1)));
        let nets = farm.value();
        let solves = (0..7)
            .map(|id| {
                let mut stream = GameStream::new(id as u64 + 1, gc);
                let mut data = Data::default();
                (id, stream.next_solve(&nets, &mut data))
            })
            .collect();
        let mut done = farm.solve(solves).unwrap();
        done.sort_unstable_by_key(|x| x.id);
        assert_eq!(
            done.iter().map(|x| x.id).collect::<Vec<_>>(),
            (0..7).collect::<Vec<_>>()
        );
        assert_eq!(farm.stats().solves.load(Ordering::Relaxed), 7);
        assert_eq!(farm.stats().used(), 0);
    }
}
