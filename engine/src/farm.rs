//! Where many solves share one forward pass.
//!
//! A solve alone is a bad shape for an accelerator: its leaf batch is a couple
//! of hundred rows and it wants one every CFR iteration. Student of Games says
//! the same thing about its own actors — each runs several games at once and
//! batches the network evaluations — so the unit of inference here is a *round*
//! across every solve in flight, not a solve.
//!
//! The mechanism is a barrier. A solver thread submits its call and parks; when
//! every live thread has parked, the driver takes the union as one batch, runs
//! it, and wakes them together. A thread's own stack is its continuation, so
//! the solver above this stays straight-line code and nothing in it knows a
//! batch exists.
//!
//! Threads that leave (their solve ended) drop out of the count, so a round
//! never waits on a thread that is gone.

use parking_lot::{Condvar, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::net::{Net, C, CFGH, D, JW, POOL, TYPE};
use crate::rebel::{Belief, CFEAT, LOOSE, NSLOT, NTYPE, PUBFEAT};
use crate::search::{Cfg, Cfr};
use crate::selfplay::{solve_root, Data, GameCfg, GameStream};
use crate::state::State;
use rayon::prelude::*;

/// One thread's network work for this round.
///
/// The three calls are the network's whole surface: the trunk over a new
/// leaf's physical state, the encoder over a new config, and the join that
/// every CFR iteration pays for. `cards` rides along with the first two
/// because a batch spans solves and each draft has its own card table.
#[derive(Clone)]
pub enum Call {
    /// New leaves: physical rows in, board vectors and the join cache out.
    ///
    /// `solve` names the solve these rows belong to and `at` the row they
    /// start at, so `at == 0` is a fresh solve and anything else extends one.
    /// Both exist because the board vectors and the join cache are properties
    /// of a leaf rather than of an iteration: the backend keeps them, and the
    /// sixty-four join calls that follow do not carry them again.
    Trunk {
        solve: usize,
        at: usize,
        xpub: Vec<f32>,
        cards: Vec<f32>,
        /// The belief index of exactly these rows: which config each query's
        /// support names, and where each query's span starts. A leaf's support
        /// is fixed when the leaf is made, so it travels once, here.
        cidx: Vec<u32>,
        coff: Vec<u32>,
        rows: usize,
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
        /// The first call of a solve. A gate slot is reused, so this is what
        /// tells the backend to forget the solve that held it before.
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
        /// Expansion simulations after the last iteration. Zero once the tree
        /// has spent its node budget, which is what lets every remaining
        /// iteration ride in one call: the host has nothing left to do for
        /// them, so it should not be woken between them.
        expand: usize,
        cfr: Cfr,
        puct: f32,
    },
    /// What the host needs back once the solve is done: the reference
    /// strategy at the nodes it asks about, and their values and beliefs.
    Read {
        solve: usize,
        touched: [bool; 2],
        /// Materialise the reference strategy first. A solve does this once,
        /// as its last act; a harvest that follows reads what it left.
        finish: bool,
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

/// One of a solve's resident arrays, as a run of the round's blob names it.
///
/// The order is the vocabulary the solver and the backend share and nothing
/// else depends on it; `Solve::plan` in the device driver is the other half.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dst {
    Kind, Player, Nc, Parent, Roff, Voff, Soff, Util,
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
/// the single driver thread, while every solver thread sat parked with nothing
/// to do. That was a third of a round. A thread builds its own now, on a core
/// that is otherwise idle, and the driver is left with what only it can do:
/// say where each run lands on the card.
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
    /// The leaves an expansion phase sampled, or `NO_ROW` where a trajectory
    /// ran into a terminal or a config with no legal action there.
    pub leaves: Vec<u32>,
}

impl Call {
    /// Run this call on the CPU network. The batched driver runs the same
    /// arithmetic on the device; this is the reference both are held to.
    pub fn run(&self, net: &Net) -> Reply {
        let mut r = Reply::default();
        match self {
            Call::Trunk { xpub, cards, rows, .. } => {
                // One row a leaf in `xpub`, and one card table per solve --
                // `board` reads the physical view of that.
                net.board(xpub, cards, *rows, CARD_ROWS, &mut r.a);
                net.join_cache(&r.a, *rows, &mut r.b);
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

/// What one attempt at a round came to.
pub enum Round {
    /// A batch ran, carrying this many rows.
    Ran(usize),
    /// Nobody was ready inside the driver's patience.
    Empty,
}

// The mailbox this thread reads its answers from, set for as long as it is a
// member of the gate. A solver reaches `submit` through the shared `Nets`,
// which cannot carry anything per-thread.
thread_local! {
    static SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

#[derive(Default)]
struct Pending {
    /// Threads currently running a solve, so currently able to park.
    live: usize,
    parked: usize,
    /// Submitted and not yet taken by a round, each with the mailbox to answer.
    calls: Vec<(usize, Call)>,
    /// One mailbox per member. A thread reads only its own, so the driver may
    /// publish the next round's answers while this one is still waking up —
    /// which is the whole reason mailboxes are per thread and not per round.
    mail: Vec<Option<Vec<Reply>>>,
    /// Mailboxes of members that have left, to be handed out again.
    free: Vec<usize>,
    /// Set when the farm is winding down: parked threads wake and get nothing.
    closed: bool,
}

pub struct Gate {
    round: Mutex<Pending>,
    /// Solvers to driver: everyone is parked.
    full: Condvar,
    /// Driver to solvers: the replies are in.
    done: Condvar,
}

impl Default for Gate {
    fn default() -> Self {
        Gate {
            round: Mutex::new(Pending::default()),
            full: Condvar::new(),
            done: Condvar::new(),
        }
    }
}

impl Gate {
    /// Join the round count, and take a mailbox, for as long as the returned
    /// guard lives.
    pub fn enter(&self) -> Member<'_> {
        let mut g = self.round.lock();
        g.live += 1;
        let slot = g.free.pop().unwrap_or_else(|| {
            g.mail.push(None);
            g.mail.len() - 1
        });
        SLOT.with(|s| s.set(slot));
        Member { gate: self, slot }
    }

    /// Which mailbox this thread holds, which is also the solve it is running
    /// for as long as it is running one.
    pub fn slot() -> usize {
        SLOT.with(|s| s.get())
    }

    /// Submit one call and park until the driver has answered it.
    ///
    /// Returns `None` only when the farm is closing, which is the one case a
    /// caller cannot serve and must unwind from.
    pub fn submit(&self, call: Call) -> Option<Reply> {
        self.submit_all(vec![call]).map(|mut r| r.remove(0))
    }

    /// Submit several calls in one round.
    ///
    /// A GT-CFR iteration is two of these: the trunk and the config encoder
    /// over what the last growth added, and then the tree and the iteration
    /// itself. Parking twice an iteration rather than four times is the
    /// difference between the barrier costing a tenth of a round and a fifth.
    pub fn submit_all(&self, calls: Vec<Call>) -> Option<Vec<Reply>> {
        let slot = SLOT.with(|s| s.get());
        assert_ne!(slot, usize::MAX, "submitting from a thread that never entered");
        let mut g = self.round.lock();
        if g.closed {
            return None;
        }
        for call in calls {
            g.calls.push((slot, call));
        }
        g.parked += 1;
        if g.parked == g.live {
            self.full.notify_one();
        }
        while g.mail[slot].is_none() && !g.closed {
            self.done.wait(&mut g);
        }
        g.mail[slot].take()
    }

    /// Wait for a full round, hand the calls to `eval`, publish what it
    /// returns. Returns the rows the round carried, or `None` once closed.
    pub fn round<F>(&self, eval: F) -> Option<usize>
    where
        F: FnOnce(&[Call]) -> Option<Vec<Reply>>,
    {
        self.serve(eval, false, None)
    }

    /// The same, but give up once no thread is left in the count. This is what
    /// shutdown uses: threads finishing their last chunk still need their
    /// rounds answered, and once they have all left there is nothing to wait
    /// for.
    pub fn serve_until_idle<F>(&self, eval: F) -> Option<usize>
    where
        F: FnOnce(&[Call]) -> Option<Vec<Reply>>,
    {
        self.serve(eval, true, None)
    }

    /// Wait up to `patience` for every thread to park, then run with whoever
    /// is there.
    ///
    /// Waiting for all of them makes a round cost the *slowest* solve rather
    /// than the average, and solve cost has a fat tail — a round-start
    /// position with a broad belief is worth many ordinary ones. Measured, that
    /// stall took generation from 100 solves/s to 49 inside ninety seconds.
    /// A thread that misses a round simply joins the next one.
    pub fn round_before<F>(&self, patience: Duration, eval: F) -> Round
    where
        F: FnOnce(&[Call]) -> Option<Vec<Reply>>,
    {
        self.serve(eval, false, Some(patience))
            .map_or(Round::Empty, Round::Ran)
    }

    /// `eval` returns `None` when the round could not be answered at all --
    /// the card is out of memory, or the driver is gone. Every thread in the
    /// cohort is parked on this round, so there is no answer to give them and
    /// no later round that will: the gate closes and they unwind. It used to
    /// panic here instead, inside the lock and with the mailboxes unfilled,
    /// which left the whole cohort parked for ever on an idle card.
    fn serve<F>(&self, eval: F, exit_when_idle: bool, patience: Option<Duration>) -> Option<usize>
    where
        F: FnOnce(&[Call]) -> Option<Vec<Reply>>,
    {
        let deadline = patience.map(|p| std::time::Instant::now() + p);
        let mut g = self.round.lock();
        while !g.closed && (g.live == 0 || g.parked < g.live) {
            if exit_when_idle && g.live == 0 {
                return None;
            }
            match deadline {
                None => self.full.wait(&mut g),
                Some(t) => {
                    if self.full.wait_until(&mut g, t).timed_out() {
                        // Whoever is parked is a batch. Only give up if that
                        // is nobody at all.
                        if g.calls.is_empty() {
                            return None;
                        }
                        break;
                    }
                }
            }
        }
        if g.closed {
            return None;
        }
        let (slots, calls): (Vec<usize>, Vec<Call>) = g.calls.drain(..).unzip();

        // The lock is held across `eval` on purpose: every live thread is
        // parked on `done`, so none of them wants it, and a thread just
        // entering would otherwise take a mailbox this round is about to fill.
        let rows = calls.iter().map(Call::rows).sum();
        let Some(replies) = eval(&calls) else {
            g.closed = true;
            self.done.notify_all();
            self.full.notify_all();
            return None;
        };
        assert_eq!(replies.len(), calls.len(), "one reply per call");

        // A thread may raise several calls in one round; its mailbox holds
        // them in the order it submitted them.
        for (slot, reply) in slots.into_iter().zip(replies) {
            g.mail[slot].get_or_insert_with(Vec::new).push(reply);
        }
        g.parked = 0;
        self.done.notify_all();
        Some(rows)
    }

    /// Whether the gate has been closed, so a driver loop can stop.
    pub fn round_closed(&self) -> bool {
        self.round.lock().closed
    }

    /// Wake everyone and refuse further work.
    pub fn close(&self) {
        let mut g = self.round.lock();
        g.closed = true;
        self.full.notify_all();
        self.done.notify_all();
    }
}

/// Membership in the round count. A thread that leaves must stop being waited
/// for, or the last round of a run never fills.
pub struct Member<'a> {
    gate: &'a Gate,
    slot: usize,
}

impl Drop for Member<'_> {
    fn drop(&mut self) {
        let mut g = self.gate.round.lock();
        g.live -= 1;
        g.mail[self.slot] = None;
        g.free.push(self.slot);
        SLOT.with(|s| s.set(usize::MAX));
        // The thread that just left may have been the one the round was
        // waiting on — and if it was the last one out, a drain waiting for the
        // count to reach zero has to be told, or it waits for ever.
        if g.live == 0 || g.parked == g.live {
            self.gate.full.notify_one();
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
    pub fn run(&self, calls: &[Call], #[allow(unused)] lane: usize) -> Option<Vec<Reply>> {
        match self {
            // Every call in a round is independent, and the solver threads
            // that raised them are all parked, so the cores are free.
            Backend::Reference(net) => Some(calls.par_iter().map(|c| c.run(net)).collect()),
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.run(calls, lane),
        }
    }

    /// How many cohorts of solves this backend serves at once.
    pub fn lanes(&self) -> usize {
        match self {
            Backend::Reference(_) => 1,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.lanes(),
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

// ------------------------------------------------------------------- the farm

/// What a solver thread does with its turn.
///
/// `Play` is the run: threads play games forward and keep the rows. `Roots`
/// is the bench, and it exists because the run cannot be measured. A solve's
/// cost varies twenty-six fold with how far into a game its root sits, so a
/// probe's rate moves by two-fold with nothing but which phase its threads
/// happened to reach — 16.6, 16.2, 12.1 and 8.0 solves/s were measured across
/// consecutive probes of one build. Cycling a corpus of roots sampled from a
/// real run holds the *mix* of costs in flight stationary, which is the one
/// property that makes two builds comparable in seconds rather than in half
/// an hour.
pub enum Work {
    Play(GameCfg),
    Roots {
        roots: Arc<Vec<(State, [Belief; 2])>>,
        cfg: Cfg,
        recursive_rate: f32,
    },
}

/// Many solves in flight in one process, and one thing that evaluates for all
/// of them.
///
/// Threads equal cores: a round runs when every one of them has parked, so a
/// thread that is still computing holds the others up, and oversubscribing
/// would make that worse rather than better. What the parked cores cost is the
/// batch itself, which is short next to the CFR work between rounds.
pub struct Farm {
    /// One gate a cohort. A cohort's threads all park together and its round
    /// runs on its own lane of the device, so while one cohort's kernels are in
    /// flight the other's threads are awake growing trees and its driver is
    /// marshalling. The driver is busy about ninety per cent of a round and
    /// only a third of that is waiting for the card; two cohorts let each fill
    /// the other's gap.
    gates: Vec<Arc<Gate>>,
    /// Shared because every cohort evaluates against it, and locked because a
    /// publish rewrites the weights under them.
    backend: Arc<RwLock<Backend>>,
    /// One per cohort: a thread has to be told its own gate.
    nets: Vec<Arc<RwLock<Arc<crate::search::Nets>>>>,
    collected: Arc<Mutex<Vec<Data>>>,
    workers: Vec<JoinHandle<()>>,
    drivers: Vec<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    stats: Arc<Stats>,
}

/// What the rounds carried, for the utilisation report. Calls per round is the
/// number that matters: it is how many solves shared a forward pass, and it
/// should sit at the cohort's thread count.
#[derive(Default)]
pub struct Stats {
    pub rounds: std::sync::atomic::AtomicU64,
    pub rows: std::sync::atomic::AtomicU64,
    pub calls: std::sync::atomic::AtomicU64,
    /// Time inside the backend. Against wall clock this says how much of a
    /// round is the batch and how much is everything around it.
    pub nanos: std::sync::atomic::AtomicU64,
}

impl Farm {
    /// Start `threads` solver threads in each of the backend's lanes. They
    /// block on their first round until rows are asked for.
    pub fn new(seed: u64, threads: usize, work: Work, backend: Backend) -> Farm {
        assert!(threads > 0, "a farm needs at least one thread");
        let cohorts = backend.lanes();
        let work = Arc::new(work);
        let value = backend.net().clone();
        let device = backend.keeps_the_solve();
        let gates: Vec<Arc<Gate>> = (0..cohorts).map(|_| Arc::new(Gate::default())).collect();
        let nets: Vec<_> = gates
            .iter()
            .map(|g| {
                Arc::new(RwLock::new(Arc::new(crate::search::Nets {
                    value: value.clone(),
                    device,
                    gate: Some(Arc::clone(g)),
                })))
            })
            .collect();
        let collected = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        let backend = Arc::new(RwLock::new(backend));
        let workers = (0..cohorts * threads)
            .map(|i| {
                let (c, t) = (i / threads, i % threads);
                let (gate, nets, collected, stopping, work) = (
                    Arc::clone(&gates[c]),
                    Arc::clone(&nets[c]),
                    Arc::clone(&collected),
                    Arc::clone(&stopping),
                    Arc::clone(&work),
                );
                let seed = seed.wrapping_mul(0x9E37_79B9) ^ i as u64;
                std::thread::Builder::new()
                    .name(format!("solve-{c}-{t}"))
                    .spawn(move || {
                        let _member = gate.enter();
                        let mut source = Source::new(&work, seed, i, cohorts * threads);
                        while !stopping.load(Ordering::Relaxed) {
                            // One chunk is small so a `drive` call can end
                            // near the row count it was asked for, and so a
                            // publish lands between chunks rather than inside
                            // a solve.
                            let n = Arc::clone(&*nets.read());
                            let d = source.chunk(&work, &n);
                            collected.lock().push(d);
                        }
                    })
                    .expect("spawn solver thread")
            })
            .collect();
        let drivers = (0..cohorts)
            .map(|c| {
                let (gate, backend, stats, stopping) = (
                    Arc::clone(&gates[c]),
                    Arc::clone(&backend),
                    Arc::clone(&stats),
                    Arc::clone(&stopping),
                );
                std::thread::Builder::new()
                    .name(format!("round-{c}"))
                    .spawn(move || drive_cohort(&gate, &backend, c, &stats, &stopping))
                    .expect("spawn driver thread")
            })
            .collect();
        Farm { gates, backend, nets, collected, workers, drivers, stopping, stats }
    }

    /// Install new weights, in the backend and in the copy the solver threads
    /// keep for the readout. The write lock waits for every cohort's round in
    /// flight, so no solve is ever evaluated against two different networks.
    pub fn publish(&mut self, net: Net) -> Result<(), String> {
        self.backend.write().set_net(net.clone())?;
        let device = self.backend.read().keeps_the_solve();
        for (n, g) in self.nets.iter().zip(&self.gates) {
            *n.write() = Arc::new(crate::search::Nets {
                value: net.clone(),
                device,
                gate: Some(Arc::clone(g)),
            });
        }
        Ok(())
    }

    /// Wait until the cohorts have produced at least `solves` rows, then hand
    /// over everything they produced. The rounds run on their own threads.
    pub fn drive(&mut self, solves: usize) -> Data {
        let mut out = Data::default();
        loop {
            for d in self.collected.lock().drain(..) {
                out.merge(d);
            }
            if out.soff.len() >= solves {
                return out;
            }
            // Parked, not spinning. With a cohort a lane and the solver threads
            // outnumbering the cores several times over, a caller busy-waiting
            // here is a core taken from the work it is waiting for.
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The network the drivers evaluate with.
    pub fn value(&self) -> Arc<crate::search::Nets> {
        Arc::clone(&*self.nets[0].read())
    }
}

/// One cohort's rounds, until the farm winds down.
fn drive_cohort(
    gate: &Gate,
    backend: &RwLock<Backend>,
    lane: usize,
    stats: &Stats,
    stopping: &AtomicBool,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut patience = PATIENCE_MIN;
    loop {
        if stopping.load(Relaxed) {
            // Threads finishing their last chunk still need their rounds
            // answered; once they have all left there is nothing to wait for.
            let b = backend.read();
            if gate.serve_until_idle(|calls| b.run(calls, lane)).is_none() {
                return;
            }
            continue;
        }
        let (mut spent, mut seen) = (Duration::ZERO, 0usize);
        let ran = {
            let b = backend.read();
            gate.round_before(patience, |calls| {
                seen = calls.len();
                let at = std::time::Instant::now();
                let replies = b.run(calls, lane);
                spent = at.elapsed();
                replies
            })
        };
        if let Round::Ran(rows) = ran {
            stats.rounds.fetch_add(1, Relaxed);
            stats.rows.fetch_add(rows as u64, Relaxed);
            stats.calls.fetch_add(seen as u64, Relaxed);
            stats.nanos.fetch_add(spent.as_nanos() as u64, Relaxed);
            patience = spent.clamp(PATIENCE_MIN, PATIENCE_MAX);
        }
    }
}

impl Drop for Farm {
    fn drop(&mut self) {
        // Ask the threads to stop; the drivers keep answering rounds until
        // every thread has finished the chunk it was in and left the count.
        // Closing a gate first would strand whoever was parked, and they would
        // unwind out of a solve instead of ending one.
        self.stopping.store(true, Ordering::Relaxed);
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        for g in &self.gates {
            g.close();
        }
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}

/// One thread's side of `Work`: the state it carries between chunks.
enum Source {
    Play(GameStream),
    /// Where in the corpus this thread is, and its own random stream. Threads
    /// are interleaved and the corpus cycles, so the set of roots in flight is
    /// the same at every moment of a bench and the same between two builds.
    Roots { at: usize, step: usize, rng: crate::rng::Rng },
}

impl Source {
    fn new(work: &Work, seed: u64, t: usize, threads: usize) -> Source {
        match work {
            Work::Play(gc) => Source::Play(GameStream::new(seed, *gc)),
            Work::Roots { .. } => Source::Roots {
                at: t,
                step: threads,
                rng: crate::rng::Rng::new(seed),
            },
        }
    }

    fn chunk(&mut self, work: &Work, nets: &crate::search::Nets) -> Data {
        match (self, work) {
            (Source::Play(stream), _) => stream.generate(nets, CHUNK_SOLVES),
            (Source::Roots { at, step, rng }, Work::Roots { roots, cfg, recursive_rate }) => {
                let mut out = Data::default();
                // Bounded by attempts, not by rows: a root whose belief is over
                // the config cap yields nothing, and a corpus of those would
                // otherwise spin here for ever.
                for _ in 0..2 * CHUNK_SOLVES {
                    if out.soff.len() >= CHUNK_SOLVES {
                        break;
                    }
                    let (s, bel) = &roots[*at % roots.len()];
                    *at += *step;
                    solve_root(nets, *cfg, *recursive_rate, s, bel, rng, &mut out);
                }
                out
            }
            (Source::Roots { .. }, Work::Play(_)) => unreachable!("a source matches its work"),
        }
    }
}

/// Solves a thread runs before it hands its rows over and re-reads the
/// network. Small enough that a publish is never more than this stale.
const CHUNK_SOLVES: usize = 8;

/// How long a driver waits for a round to fill before running with whoever is
/// there.
///
/// It cannot be a small constant. A thread woken by a round does its share of
/// the host work -- growing the leaves the expansion sampled -- before it can
/// park again, and that is milliseconds. A patience shorter than it fires the
/// next round with only the threads whose work happened to be trivial, so a
/// round of seventy-two threads carried twenty-seven and the cards saw a
/// batch a third of the size they should.
///
/// So it tracks what a round costs instead. Waiting for a full round can never
/// cost more than running a partial one did, and when the host work is short
/// the wait ends early anyway because everyone has parked.
const PATIENCE_MIN: Duration = Duration::from_millis(2);
const PATIENCE_MAX: Duration = Duration::from_millis(50);

/// Shapes the driver needs to lay a batch out. Kept here so the device code
/// and the CPU reference agree on them by construction.
pub const TRUNK_IN: usize = PUBFEAT;
pub const TRUNK_OUT: usize = D;
pub const TRUNK_CACHE: usize = JW;
pub const CARDS_PER_ROW: usize = NTYPE * TYPE;
pub const CONFIG_IN: usize = CFEAT;
pub const CONFIG_F: usize = D;
pub const CONFIG_G: usize = POOL;
pub const JOIN_OUT: usize = D;
pub const BOARD_WIDTH: usize = 2 * C + LOOSE;
pub const CFG_SLOTS: usize = NSLOT;
pub const CFG_HIDDEN: usize = CFGH;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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

    /// A call whose reply is a function of its own contents, so a thread can
    /// tell its own answer from anybody else's.
    fn tagged(tag: usize) -> Call {
        Call::Configs {
            solve: 0,
            at: tag,
            phi: Vec::new(),
            owner: Vec::new(),
            cards: Vec::new(),
            n: tag,
        }
    }

    fn tag_of(c: &Call) -> f32 {
        match c {
            Call::Configs { at, .. } => *at as f32,
            _ => unreachable!(),
        }
    }

    /// Every thread must get back the reply to its own call, round after
    /// round, and threads that finish at different times must not strand the
    /// ones still running.
    #[test]
    fn a_round_answers_every_thread_with_its_own_reply() {
        const THREADS: usize = 8;
        let gate = Arc::new(Gate::default());
        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                let gate = Arc::clone(&gate);
                // Staggered lengths: thread `t` leaves after `t + 1` rounds,
                // so the round count shrinks under the driver.
                std::thread::spawn(move || {
                    let _member = gate.enter();
                    for r in 0..=t {
                        let tag = 1 + t * 100 + r;
                        let reply = gate.submit(tagged(tag)).expect("open gate");
                        assert_eq!(
                            reply.a[0], tag as f32,
                            "thread {t} round {r} got another thread's reply"
                        );
                    }
                })
            })
            .collect();

        let driver = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let mut rounds = 0usize;
                while gate
                    .round(|calls| Some(calls.iter().map(|c| Reply { a: vec![tag_of(c)], ..Default::default() }).collect()))
                    .is_some()
                {
                    rounds += 1;
                }
                rounds
            })
        };

        for w in workers {
            w.join().expect("worker");
        }
        gate.close();
        let rounds = driver.join().expect("driver");
        // The longest-lived thread submits THREADS times, and every round it
        // takes part in is a round.
        assert!(rounds >= THREADS, "only {rounds} rounds for {THREADS} threads");
    }

    /// The same, under the driver the farm actually uses.
    ///
    /// `round_before` serves whoever is parked when its patience runs out, so
    /// rounds are partial and the ticket a thread holds means something
    /// different from one round to the next. A thread slow to wake used to
    /// read the *next* round's replies at its own ticket and get another
    /// solve's answer. Uneven work between submits makes rounds partial, and
    /// the tag says whose answer arrived.
    #[test]
    fn a_partial_round_still_answers_every_thread_with_its_own_reply() {
        const THREADS: usize = 200;
        const ROUNDS: usize = 200;
        let gate = Arc::new(Gate::default());
        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    let _member = gate.enter();
                    for r in 0..ROUNDS {
                        let tag = 1 + t * ROUNDS + r;
                        let reply = gate.submit(tagged(tag)).expect("open gate");
                        assert_eq!(
                            reply.a[0], tag as f32,
                            "thread {t} round {r} got another thread's reply"
                        );
                        // Different threads take different times to come back,
                        // so the driver's patience expires on a partial round.
                        for _ in 0..(t * 37) % 91 {
                            std::hint::spin_loop();
                        }
                    }
                })
            })
            .collect();

        let driver = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                while !gate.round_closed() {
                    gate.round_before(Duration::from_micros(200), |calls| {
                        Some(
                            calls
                                .iter()
                                .map(|c| Reply { a: vec![tag_of(c)], ..Default::default() })
                                .collect(),
                        )
                    });
                }
            })
        };

        for w in workers {
            w.join().expect("worker");
        }
        gate.close();
        driver.join().expect("driver");
    }

    /// Many threads, one evaluator: the farm must produce well-formed rows and
    /// shut down without stranding anyone. Batches larger than one call are
    /// the whole point, so the round size is checked too.
    #[test]
    fn a_farm_batches_many_threads_into_one_round() {
        use crate::selfplay::{Agent, Collect};
        const THREADS: usize = 6;
        let cfg = crate::search::Cfg {
            nodes: 64,
            expand: 1,
            iters: 8,
            ..Default::default()
        };
        let gc = crate::selfplay::GameCfg {
            agents: [Agent::Rebel { cfg }; 2],
            collect: Collect::Rebel,
            explore: 0.1,
            random_draft: true,
            eval_mix: 1.0,
            mc_mix: 0.0,
            query_rate: 0.9,
            recursive_rate: 0.1,
        };
        let mut farm = Farm::new(5, THREADS, Work::Play(gc), Backend::Reference(small_net(0x2E57)));
        let data = farm.drive(48);
        assert!(data.soff.len() >= 48, "only {} solves", data.soff.len());
        assert_eq!(data.nv, data.soff.len(), "a solve must store one row");
        assert_eq!(data.coff.len(), 2 * data.nv + 1, "ragged arena is malformed");
        let read = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed) as f64;
        let s = farm.stats();
        assert!(read(&s.rounds) > 0.0, "no round ever ran");
        // The whole point: a round is one forward pass shared by every thread
        // that is running a solve, not one pass per solve.
        let calls = read(&s.calls) / read(&s.rounds);
        assert!(
            calls > (THREADS as f64) * 0.5,
            "rounds averaged only {calls:.1} calls for {THREADS} threads"
        );
        assert!(read(&s.rows) > 0.0, "rounds carried no rows");
    }

    /// Closing the gate must release a thread that is already parked, rather
    /// than leave it waiting for a round that will never fill.
    #[test]
    fn closing_releases_a_parked_thread() {
        let gate = Arc::new(Gate::default());
        let worker = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let _member = gate.enter();
                // Two members are counted, so this round can never fill on
                // its own and the thread parks until `close`.
                let _phantom = gate.enter();
                gate.submit(tagged(7)).is_none()
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        gate.close();
        assert!(worker.join().expect("worker"), "close must hand back None");
    }
}
