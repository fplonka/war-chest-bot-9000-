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
//! sits in one of two queues. One driver a card takes whatever is waiting on
//! that card and runs it as one batch; a pool of workers, one per core, does
//! the host side. Neither waits for the other, which is what lets one solve's
//! growth overlap another's device work.
//!
//! How many solves are in flight is a question about memory, and it is asked on
//! both sides: the card's arenas and the host's. Each is a projection of what
//! the population will hold, not a reading of what it holds.

use parking_lot::{Condvar, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::net::Net;
use crate::pbs::Belief;
use crate::search::{Cfg, Cfr, Solved, Solver, Step};
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
/// The host used to run the policy head itself instead, and the round
/// downloaded a board vector per fresh leaf and a `f_p` row per fresh config so
/// that it could — a quarter of a megabyte a solve a round, for a handful of
/// nodes.
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
/// The backend used to build this itself: it held an `Arc<Contract>`, walked
/// it, widened bytes into words and copied every run into one buffer -- all on
/// the one driver thread a card has. That was a third of a round. A solve
/// builds its own now, on the worker that grew the tree, and the driver is left
/// with what only it can do: say where each run lands on the card.
///
/// Floats travel as their bits, because the scatter kernel moves words and
/// does not care which they are.
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

    /// How many cards this backend has, and so how many drivers the farm runs.
    pub fn cards(&self) -> usize {
        match self {
            Backend::Reference(_) => 1,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.cards(),
        }
    }

    /// Whether this card has room for another solve in flight.
    ///
    /// A solve's cost varies twenty-six fold with how far into a game its root
    /// sits, so how many fit is a question about bytes and the card is the only
    /// thing that can answer it. See `Device::room_for`.
    ///
    /// The reference backend keeps nothing resident -- a solve it serves lives
    /// entirely in host memory, where the farm's own budget bounds it -- so
    /// this is a plain count, and it is the only bound left on a machine whose
    /// memory cannot be read.
    pub fn has_room(&self, #[allow(unused)] card: usize, live: usize) -> bool {
        match self {
            Backend::Reference(_) => live < REFERENCE_IN_FLIGHT,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.room_for(card, live),
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


/// Solves the CPU reference serves at once. It keeps nothing on a card, so the
/// host budget is its real bound and this is the backstop for a machine whose
/// memory cannot be read.
const REFERENCE_IN_FLIGHT: usize = 64;

/// Host bytes the farm may hold in solves, reserved once before it starts.
///
/// A solve costs host bytes as well as device bytes, and more of them: the
/// tree, the states its nodes stand on and the description the card reads are
/// all here too. A farm that admitted on the card's level alone filled the host
/// instead, and the run was killed with no message but the exit code.
///
/// This used to be a level -- admit while the whole *process* holds less than a
/// fifth of the machine -- and it was the wrong quantity twice over. It counted
/// torch, two CUDA contexts and a replay buffer reserved at its cap, none of
/// which a solve has anything to do with, so the farm was throttled by the
/// trainer's memory. And a level always lags: a solve's arenas fill over its
/// whole life, so what the population holds now is what a younger one held.
/// Measured, that overshot by 4.7x and a bench was killed at 58 GB of 62.
///
/// So it is a reservation instead, and admission projects against it the way
/// the card already does (`Device::room_for`): a share of the machine, less
/// what the process was already holding before a single solve was admitted, so
/// everything outside the farm is charged once and never chased again.
///
/// The machine's own figures come from `/proc`, and where there is no `/proc`
/// there is no honest number to give: the farm is then bounded by the card and
/// by pacing alone, as it was on every non-Linux machine before. The rule that
/// spends this budget does not depend on where it came from, which is why it is
/// a number here and not a predicate.
fn host_budget() -> u64 {
    /// Share of the machine the farm's solves may hold.
    ///
    /// The other half is the trainer, the replay buffer at its cap, the
    /// allocator's retained pages and whatever else shares the box. Half is
    /// safe because the population no longer overshoots: what is admitted is
    /// projected at the largest a solve has ever grown to, so nothing already
    /// in flight can surprise it.
    const SHARE: f64 = 0.5;
    let Some((total, rss)) = machine() else {
        return u64::MAX;
    };
    ((SHARE * total as f64) as u64).saturating_sub(rss)
}

/// The machine's memory and what this process already holds, in bytes.
#[cfg(target_os = "linux")]
fn machine() -> Option<(u64, u64)> {
    let field = |path: &str, at: usize| -> Option<u64> {
        std::fs::read_to_string(path).ok()?.split_whitespace().nth(at)?.parse().ok()
    };
    // `statm` is in pages and `meminfo`'s first field is `MemTotal:` in kB.
    Some((field("/proc/meminfo", 1)? * 1024, field("/proc/self/statm", 1)? * 4096))
}

#[cfg(not(target_os = "linux"))]
fn machine() -> Option<(u64, u64)> {
    None
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

    /// Everything waiting, which is what a round is. Empty once the queue is
    /// closed and empty.
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
    source: Source,
    solver: Solver,
    /// The card that holds this solve's arenas, and which of its slots. A card
    /// keeps a solve's board vectors between its rounds, so both are fixed for
    /// as long as the job lives.
    card: usize,
    slot: usize,
    /// What the last round gave back, waiting for the host work that reads it.
    replies: Vec<Reply>,
    /// Rows this job has produced since it last handed any over.
    data: Data,
}

/// Many solves in flight in one process, and one thing that evaluates for all
/// of them.
///
/// Two kinds of thread. One driver a card takes whatever solves are waiting on
/// that card, runs their calls as one batch, and hands the replies back. A pool
/// of workers, one per core, does the host side: growth, which is the game's
/// rules, and the game around it. Neither ever waits for the other, so one
/// solve's growth overlaps another's device work — which a barrier could not
/// do, because it made a round cost whatever its slowest member cost.
pub struct Farm {
    /// One queue a card: solves whose next round that card must run.
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
}

/// What the rounds carried, for the utilisation report. Calls per round is the
/// number that matters: it is how many solves shared a forward pass.
#[derive(Default)]
pub struct Stats {
    pub rounds: AtomicU64,
    pub rows: AtomicU64,
    pub calls: AtomicU64,
    /// Time inside the backend. Against wall clock this says how much of a
    /// round is the batch and how much is everything around it.
    pub nanos: AtomicU64,
    /// Per card: solves admitted, how many its share of the host budget allowed
    /// when the last one was admitted, and the largest a solve on that card has
    /// grown to in host bytes.
    ///
    /// How many solves are in flight used to be folklore -- nothing measured it
    /// and nothing logged it, so a run that was slow because it was holding
    /// four solves looked exactly like one holding forty. These three are what
    /// the population is, and `live <= max(1, allowed)` holds card by card.
    live: Vec<AtomicU64>,
    allowed: Vec<AtomicU64>,
    peak: Vec<AtomicU64>,
}

impl Stats {
    fn new(cards: usize) -> Stats {
        let per = || (0..cards).map(|_| AtomicU64::new(0)).collect();
        Stats { live: per(), allowed: per(), peak: per(), ..Default::default() }
    }

    /// One more solve on this card, with the allowance it was admitted under.
    fn admit(&self, card: usize, allowed: u64) {
        self.allowed[card].store(allowed, Ordering::Relaxed);
        self.live[card].fetch_add(1, Ordering::Relaxed);
    }

    /// A solve on this card finished at `bytes`. Published before the solve is
    /// counted as done, so the admission that pacing then allows sees it.
    fn grew(&self, card: usize, bytes: u64) {
        self.peak[card].fetch_max(bytes, Ordering::Relaxed);
    }

    /// Solves in flight, over every card.
    pub fn live(&self) -> u64 {
        self.live.iter().map(|a| a.load(Ordering::Relaxed)).sum()
    }

    /// How many the budget allowed at the last admission, over every card.
    pub fn allowed(&self) -> u64 {
        self.allowed.iter().map(|a| a.load(Ordering::Relaxed)).sum()
    }

    /// The largest a solve has grown to in host bytes, over every card.
    pub fn host_peak(&self) -> u64 {
        self.peak.iter().map(|a| a.load(Ordering::Relaxed)).max().unwrap_or(0)
    }
}

impl Farm {
    /// Start `workers` host threads and one driver per card, against whatever
    /// share of the machine `host_budget` reserved for solves.
    ///
    /// How many solves are in flight is not settled here. Each card admits them
    /// as memory allows, on both sides, which is the only bound that means
    /// anything: a solve's cost varies twenty-six fold with how far into a game
    /// its root sits, so no thread count describes what fits.
    pub fn new(seed: u64, workers: usize, work: Work, backend: Backend) -> Farm {
        Farm::bounded(seed, workers, work, backend, host_budget())
    }

    /// The same farm, against a host budget the caller names. Only the
    /// admission test wants this: everything else takes the machine's.
    pub fn bounded(
        seed: u64,
        workers: usize,
        work: Work,
        backend: Backend,
        host_budget: u64,
    ) -> Farm {
        assert!(workers > 0, "a farm needs at least one worker");
        let cards = backend.cards();
        let work = Arc::new(work);
        let nets = Arc::new(RwLock::new(Arc::new(crate::search::Nets {
            value: backend.net().clone(),
            device: backend.keeps_the_solve(),
        })));
        let ready = Arc::new(Queue::default());
        let device: Vec<Arc<Queue<(Job, Vec<Call>)>>> =
            (0..cards).map(|_| Arc::new(Queue::default())).collect();
        let collected = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let broken = Arc::new(AtomicBool::new(false));
        // Solves each stream has finished, which is what paces its admissions.
        let done: Vec<Arc<AtomicUsize>> = (0..cards).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        let stats = Arc::new(Stats::new(cards));
        let backend = Arc::new(RwLock::new(backend));
        // Split evenly. A card's solves are its own and no card can spend
        // another's, so one share each is the only division that bounds the
        // whole process.
        let share = host_budget / cards as u64;

        let hands: Vec<JoinHandle<()>> = (0..workers)
            .map(|t| {
                let (ready, device, nets, collected, work, stopping, done, stats) = (
                    Arc::clone(&ready),
                    device.clone(),
                    Arc::clone(&nets),
                    Arc::clone(&collected),
                    Arc::clone(&work),
                    Arc::clone(&stopping),
                    done.clone(),
                    Arc::clone(&stats),
                );
                std::thread::Builder::new()
                    .name(format!("host-{t}"))
                    .spawn(move || {
                        while let Some(job) = ready.pop() {
                            advance_job(
                                job, &device, &nets, &work, &collected, &stopping, &done, &stats,
                            );
                        }
                    })
                    .expect("spawn host thread")
            })
            .collect();

        let drivers = (0..cards)
            .map(|c| {
                let (queue, ready, backend, nets, work, stats, broken, done) = (
                    Arc::clone(&device[c]),
                    Arc::clone(&ready),
                    Arc::clone(&backend),
                    Arc::clone(&nets),
                    Arc::clone(&work),
                    Arc::clone(&stats),
                    Arc::clone(&broken),
                    Arc::clone(&done[c]),
                );
                let seed = seed.wrapping_mul(0x9E37_79B9) ^ c as u64;
                std::thread::Builder::new()
                    .name(format!("card-{c}"))
                    .spawn(move || {
                        drive_card(
                            c, cards, seed, share, &queue, &ready, &backend, &nets, &work,
                            &stats, &broken, &done,
                        )
                    })
                    .expect("spawn driver thread")
            })
            .collect();

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
    nets: &RwLock<Arc<crate::search::Nets>>,
    work: &Work,
    collected: &Mutex<Vec<Data>>,
    stopping: &AtomicBool,
    done: &[Arc<AtomicUsize>],
    stats: &Stats,
) {
    let mut replies = std::mem::take(&mut job.replies);
    loop {
        match job.solver.advance(&replies) {
            Step::Calls(calls) => return device[job.card].push((job, calls)),
            Step::Done(solved) => {
                // What this solve grew to, before it is counted as finished:
                // finishing is what pays for the next admission, and that
                // admission must be projected against a peak that includes
                // this one. A solve is at its largest here and nowhere else.
                stats.grew(job.card, job.solver.host_bytes() as u64);
                // Released, and acquired where the driver reads it, so a
                // driver that sees this solve counted also sees its size.
                done[job.card].fetch_add(1, Ordering::Release);
                job.source.take(&job.solver, solved, &mut job.data);
                // Every solve, not every eighth. A job re-reads the network
                // between two solves anyway, so there is nothing a batch of
                // them buys -- and holding eight back per job is eight times
                // the solves in flight before a caller sees its first row.
                collected.lock().push(std::mem::take(&mut job.data));
                if stopping.load(Ordering::Relaxed) {
                    return;
                }
                // A publish lands between two solves rather than inside one.
                let n = Arc::clone(&*nets.read());
                job.solver = job.source.next(work, &n, &mut job.data);
                job.solver.pin(job.slot);
                replies = Vec::new();
            }
        }
    }
}

/// One card's rounds, and the solves it admits between them.
#[allow(clippy::too_many_arguments)]
fn drive_card(
    card: usize,
    cards: usize,
    seed: u64,
    // This card's share of the host budget, in bytes.
    share: u64,
    queue: &Queue<(Job, Vec<Call>)>,
    ready: &Queue<Job>,
    backend: &RwLock<Backend>,
    nets: &RwLock<Arc<crate::search::Nets>>,
    work: &Work,
    stats: &Stats,
    broken: &AtomicBool,
    done: &AtomicUsize,
) {
    let mut live = 0usize;
    loop {
        // Solves this stream has room for, admitted between rounds and never
        // retired. A solve's cost varies twenty-six fold with how far into a
        // game its root sits, so how many fit is a question about bytes, and it
        // is asked on both sides: the card's arenas and the host's.
        //
        // Both are projections, not levels. A solve's arenas fill over its
        // whole run, so what a population holds now is what a younger one held,
        // and admitting on that overshoots by whatever the solves already in
        // flight grow in the meantime -- measured at 4.7x, and it killed a
        // bench. So the question asked is what this population *will* hold:
        // every solve in flight, and the one being asked about, at the largest
        // a solve on this stream has ever reached.
        //
        // The peak is nought until the first solve has finished, and the answer
        // is yes until it has. What makes that safe is the pacing below: one
        // admitted per one *finished*, so the second is admitted only once the
        // first has run its whole life and the peak is a real one.
        //
        // Nothing new once the farm is winding down. The solves in flight have
        // to finish before a worker can leave, and a fresh one would be a whole
        // solve of that wait for rows nobody will collect.
        let paid = live <= done.load(Ordering::Acquire);
        let peak = stats.peak[card].load(Ordering::Relaxed);
        let allowed = if peak == 0 { u64::MAX } else { share / peak };
        let room = paid && (live as u64) < allowed && backend.read().has_room(card, live);
        if !ready.closed() && (live == 0 || room) {
            stats.admit(card, allowed);
            let mut source = Source::new(
                work,
                seed ^ (live as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                live * cards + card,
            );
            let mut data = Data::default();
            let n = Arc::clone(&*nets.read());
            let mut solver = source.next(work, &n, &mut data);
            solver.pin(live);
            ready.push(Job { source, solver, card, slot: live, replies: Vec::new(), data });
            live += 1;
        }
        let batch = queue.drain();
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
        let answered = backend.read().run(&calls, card);
        let spent = at.elapsed();
        let Some(replies) = answered else {
            // Not recoverable: the card is out of memory, or gone. Every solve
            // it holds dies with it, so say so rather than leave the caller
            // waiting for rows that will never come.
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


// ---------------------------------------------------------- a blocking client

/// One card's rounds, for callers that block on a whole solve.
///
/// The farm's solves are state machines because there are far more of them than
/// there are cores. A bot is the other shape: one game per thread, nothing to
/// remember between two solves, and a thread that is willing to wait. So the
/// thread's own stack is its continuation — it puts its calls on a card's queue
/// and waits for the round that carries them, which is shared with every other
/// thread that was ready at the same moment.
pub struct Cards {
    queues: Vec<Arc<Queue<Ask>>>,
    /// Seats nobody is sitting in, and how many have ever been handed out. A
    /// card keeps a solve's arenas while it runs, so two solves must never
    /// share a slot, and the number in use is therefore the number of threads
    /// solving at once.
    seats: Mutex<(Vec<(usize, usize)>, usize)>,
    drivers: Vec<JoinHandle<()>>,
}

/// One thread's calls, and where to send the replies.
struct Ask {
    calls: Vec<Call>,
    back: std::sync::mpsc::Sender<Vec<Reply>>,
}

/// A card and one of its solve slots, held for one solve and given back when
/// this is dropped.
pub struct Seat<'a> {
    cards: &'a Cards,
    pub card: usize,
    pub slot: usize,
}

impl Drop for Seat<'_> {
    fn drop(&mut self) {
        self.cards.seats.lock().0.push((self.card, self.slot));
    }
}

impl Cards {
    pub fn new(backend: Backend) -> Cards {
        let n = backend.cards();
        let backend = Arc::new(backend);
        let queues: Vec<Arc<Queue<Ask>>> = (0..n).map(|_| Arc::new(Queue::default())).collect();
        let drivers = (0..n)
            .map(|c| {
                let (queue, backend) = (Arc::clone(&queues[c]), Arc::clone(&backend));
                std::thread::Builder::new()
                    .name(format!("card-{c}"))
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
                        // A card that cannot answer takes its solves with it.
                        // Dropping the senders is what tells them.
                        let Some(replies) = backend.run(&calls, c) else {
                            return;
                        };
                        let mut rest = replies;
                        for (back, n) in backs.into_iter().zip(spans) {
                            let tail = rest.split_off(n);
                            let _ = back.send(rest);
                            rest = tail;
                        }
                    })
                    .expect("spawn driver thread")
            })
            .collect();
        Cards { queues, seats: Mutex::new((Vec::new(), 0)), drivers }
    }

    /// Take a card and one of its solve slots for the length of one solve.
    pub fn seat(&self) -> Seat<'_> {
        let mut seats = self.seats.lock();
        let (card, slot) = seats.0.pop().unwrap_or_else(|| {
            seats.1 += 1;
            ((seats.1 - 1) % self.queues.len(), (seats.1 - 1) / self.queues.len())
        });
        drop(seats);
        Seat { cards: self, card, slot }
    }

    /// Run these calls in the next round of `card`, and wait for them.
    /// `None` once the card is gone, which is not recoverable.
    pub fn round(&self, card: usize, calls: Vec<Call>) -> Option<Vec<Reply>> {
        let (back, replies) = std::sync::mpsc::channel();
        self.queues[card].push(Ask { calls, back });
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

    /// The population a farm holds is the one its host budget allows.
    ///
    /// This is the gate nothing used to reach: the old rule was a level on the
    /// process's own RSS, read from `/proc`, so on any machine without one it
    /// returned "yes" and no test ever ran it. The rule here is arithmetic on
    /// two numbers the farm keeps, so it runs everywhere, and the budget is an
    /// argument so a test can make it small.
    ///
    /// Three farms over the same work. With no budget at all a stream holds the
    /// one solve it must -- a stream that held none could never finish one, and
    /// nothing would ever pay for the next -- and no more. With room for a few,
    /// it holds no more than it was told it could. With no bound it holds many
    /// more than either, which is what says the first two were bounded by the
    /// budget and not by something else.
    #[test]
    fn admission_is_bounded_by_the_host_budget() {
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
        let run = |budget: u64| -> (u64, u64, u64) {
            let backend = Backend::Reference(small_net(0x2E57));
            let mut farm = Farm::bounded(5, WORKERS, Work::Play(gc), backend, budget);
            let got = farm.drive(24);
            assert!(got.soff.len() >= 24, "only {} solves at budget {budget}", got.soff.len());
            let s = farm.stats();
            let (live, allowed, peak) = (s.live(), s.allowed(), s.host_peak());
            eprintln!("budget {budget}: {live} live, {allowed} allowed, peak {peak} B");
            (live, allowed, peak)
        };

        let (live, _, peak) = run(0);
        assert!(peak > 0, "no solve ever reported its host size");
        assert_eq!(live, 1, "a farm with no host budget held {live} solves");

        let (live, allowed, _) = run(4 * peak);
        assert!(live <= allowed.max(1), "{live} solves in flight, {allowed} allowed");

        let (big, _, _) = run(u64::MAX);
        assert!(big > live, "an unbounded farm held {big}, a bounded one {live}");
    }
}
