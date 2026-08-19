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
use crate::rebel::{CFEAT, LOOSE, NSLOT, NTYPE, PUBFEAT};
use crate::selfplay::{Data, GameCfg, GameStream};
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
    /// Every leaf, every iteration: the belief-conditioned head.
    ///
    /// The whole of what a CFR iteration asks the network.
    ///
    /// Everything else the query needs is already with the backend, so this
    /// carries only what actually changed: `w`, the normalised belief over
    /// every query's configs, and `opp`, the opponent's reach mass at each
    /// leaf. What comes back is the counterfactual value per config of the
    /// queried player — not the head, not the board vectors, not `f` or `g`.
    ///
    /// Pooling the beliefs, running the join and reading the values out were
    /// three steps with two round trips between them. They are one step here
    /// because the two intermediates never had a reader on the host.
    Leaf {
        solve: usize,
        w: Vec<f32>,
        opp: Vec<f32>,
        rows: usize,
        player: usize,
    },
}

/// What a call gives back. The trunk returns the board vector and its join
/// cache, the encoder returns `f`, `g` and the policy's `f_p`, and the join
/// returns `h` alone.
#[derive(Default)]
pub struct Reply {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
}

impl Call {
    /// Run this call on the CPU network. The batched driver runs the same
    /// arithmetic on the device; this is the reference both are held to.
    pub fn run(&self, net: &Net) -> Reply {
        let mut r = Reply::default();
        match self {
            Call::Trunk { xpub, cards, rows, .. } => {
                // Two seat views per leaf in `xpub`, and one card table per
                // solve — `board` reads the physical view of each.
                net.board(xpub, cards, *rows, CARD_ROWS, &mut r.a);
                net.join_cache(&r.a, *rows, &mut r.b);
            }
            Call::Configs { phi, owner, cards, n, .. } => {
                net.configs(phi, owner, *n, cards, &mut r.a, &mut r.b, &mut r.c);
            }
            Call::Leaf { .. } => unreachable!("a leaf query needs the resident state"),
        }
        r
    }

    /// Which of the three batches this call belongs to. The device groups a
    /// round by kind and runs each group once.
    pub fn kind(&self) -> usize {
        match self {
            Call::Trunk { .. } => 0,
            Call::Configs { .. } => 1,
            Call::Leaf { .. } => 2,
        }
    }

    /// Which solve raised this call. A backend keeps a solve's board vectors
    /// between its iterations, so every call of one solve must reach the same
    /// backend — which is what a round shards on.
    pub fn solve(&self) -> usize {
        match self {
            Call::Trunk { solve, .. }
            | Call::Configs { solve, .. }
            | Call::Leaf { solve, .. } => *solve,
        }
    }

    /// Rows this call contributes to its batch, for the round report.
    pub fn rows(&self) -> usize {
        match self {
            Call::Trunk { rows, .. } => *rows,
            Call::Configs { n, .. } => *n,
            Call::Leaf { rows, .. } => *rows,
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
    mail: Vec<Option<Reply>>,
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
        let slot = SLOT.with(|s| s.get());
        assert_ne!(slot, usize::MAX, "submitting from a thread that never entered");
        let mut g = self.round.lock();
        if g.closed {
            return None;
        }
        g.calls.push((slot, call));
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
        F: FnOnce(&[Call]) -> Vec<Reply>,
    {
        self.serve(eval, false, None)
    }

    /// The same, but give up once no thread is left in the count. This is what
    /// shutdown uses: threads finishing their last chunk still need their
    /// rounds answered, and once they have all left there is nothing to wait
    /// for.
    pub fn serve_until_idle<F>(&self, eval: F) -> Option<usize>
    where
        F: FnOnce(&[Call]) -> Vec<Reply>,
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
        F: FnOnce(&[Call]) -> Vec<Reply>,
    {
        self.serve(eval, false, Some(patience))
            .map_or(Round::Empty, Round::Ran)
    }

    fn serve<F>(&self, eval: F, exit_when_idle: bool, patience: Option<Duration>) -> Option<usize>
    where
        F: FnOnce(&[Call]) -> Vec<Reply>,
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
        let replies = eval(&calls);
        assert_eq!(replies.len(), calls.len(), "one reply per call");

        for (slot, reply) in slots.into_iter().zip(replies) {
            g.mail[slot] = Some(reply);
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

/// Pool the beliefs, run the join, read the values out — the CPU reference for
/// `Call::Leaf`, and the oracle the device kernels are held to.
///
/// The two intermediates never leave: `pooled` and the head `h` are produced
/// and consumed here, which is what makes the call small at both ends.
fn leaf(net: &Net, b: &Board, w: &[f32], opp: &[f32], rows: usize, player: usize) -> Reply {
    let queries = 2 * rows;
    let mut pooled = vec![0.0f32; queries * POOL];
    for q in 0..queries {
        let (lo, hi) = (b.coff[q] as usize, b.coff[q + 1] as usize);
        crate::net::accumulate(
            &b.g,
            &b.cidx[lo..hi],
            &w[lo..hi],
            POOL,
            &mut pooled[q * POOL..(q + 1) * POOL],
        );
    }
    let mut h = Vec::new();
    net.join(&b.p, &b.jp, &pooled, rows, player, &mut h);
    // One value per config of the queried player, counterfactual: the
    // network's value for that config times the opponent's reach into the leaf.
    let mut out = Vec::new();
    for r in 0..rows {
        let q = 2 * r + player;
        let (lo, hi) = (b.coff[q] as usize, b.coff[q + 1] as usize);
        let at = out.len();
        out.resize(at + (hi - lo), 0.0);
        net.values(&h[r * D..(r + 1) * D], &b.f, &b.cidx[lo..hi], &mut out[at..]);
        for v in &mut out[at..] {
            *v *= opp[r];
        }
    }
    Reply { a: out, ..Default::default() }
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
    Reference(Net, Resident),
}

/// What a backend keeps for a solve between its iterations.
///
/// A leaf's board vector and its projection into the join are computed once by
/// the trunk and read by every one of the sixty-four join calls that follow.
/// Sending them each time was 87% of what crossed the bus, and on a device
/// they were being sent back to the card that had just produced them.
///
/// Keyed by the gate slot, which is a solve while that solve is in flight. A
/// trunk call at row zero starts a new one and drops whatever was there.
#[derive(Default)]
pub struct Resident {
    slots: Mutex<Vec<Board>>,
}

#[derive(Default, Clone)]
struct Board {
    p: Vec<f32>,
    jp: Vec<f32>,
    /// `f(c)` and `g(c)` for every distinct config the solve has interned.
    f: Vec<f32>,
    g: Vec<f32>,
    /// Which config each query's support names, and where a query's span
    /// starts. Written by the trunk, since a leaf's support is fixed when the
    /// leaf is made.
    cidx: Vec<u32>,
    coff: Vec<u32>,
}

impl Resident {
    fn slot(&self, solve: usize) -> parking_lot::MappedMutexGuard<'_, Board> {
        let mut g = self.slots.lock();
        if g.len() <= solve {
            g.resize(solve + 1, Board::default());
        }
        parking_lot::MutexGuard::map(g, |v| &mut v[solve])
    }

    fn keep_configs(&self, solve: usize, at: usize, f: &[f32], g_: &[f32]) {
        let mut b = self.slot(solve);
        let (d, pool) = (crate::net::D, crate::net::POOL);
        let want = (at + f.len() / d) * d;
        if b.f.len() < want {
            b.f.resize(want, 0.0);
            b.g.resize(want / d * pool, 0.0);
        }
        b.f[at * d..want].copy_from_slice(f);
        b.g[at * pool..want / d * pool].copy_from_slice(g_);
    }

    fn keep(&self, solve: usize, at: usize, p: &[f32], jp: &[f32]) {
        let mut g = self.slots.lock();
        if g.len() <= solve {
            g.resize(solve + 1, Board::default());
        }
        let b = &mut g[solve];
        // Written at the row it names rather than appended, so replaying a
        // round -- which the parity test does, once alone and once batched --
        // leaves the same state instead of stacking it up.
        let (d, jw) = (crate::net::D, crate::net::JW);
        let want = (at + p.len() / d) * d;
        if b.p.len() < want {
            b.p.resize(want, 0.0);
            b.jp.resize(want / d * jw, 0.0);
        }
        b.p[at * d..want].copy_from_slice(p);
        b.jp[at * jw..want / d * jw].copy_from_slice(jp);
    }

    /// The belief index arrays, appended for the queries the trunk just made.
    fn keep_index(&self, solve: usize, at: usize, cidx: &[u32], coff: &[u32]) {
        let mut b = self.slot(solve);
        if at == 0 {
            b.cidx.clear();
            b.coff.clear();
            b.coff.push(0);
        }
        let base = b.cidx.len() as u32;
        b.cidx.extend_from_slice(cidx);
        b.coff.extend(coff.iter().skip(1).map(|x| x + base));
    }

    fn board(&self, solve: usize) -> Board {
        self.slots.lock()[solve].clone()
    }
}

impl Backend {
    pub fn run(&self, calls: &[Call]) -> Vec<Reply> {
        match self {
            // Every call in a round is independent, and the solver threads
            // that raised them are all parked, so the cores are free.
            Backend::Reference(net, res) => calls
                .par_iter()
                .map(|c| match c {
                    Call::Trunk { solve, at, cidx, coff, .. } => {
                        let r = c.run(net);
                        res.keep(*solve, *at, &r.a, &r.b);
                        res.keep_index(*solve, *at, cidx, coff);
                        Reply { a: r.a, ..Default::default() }
                    }
                    Call::Configs { solve, at, .. } => {
                        let r = c.run(net);
                        res.keep_configs(*solve, *at, &r.a, &r.b);
                        Reply { c: r.c, ..Default::default() }
                    }
                    Call::Leaf { solve, w, opp, rows, player } => {
                        leaf(net, &res.board(*solve), w, opp, *rows, *player)
                    }
                })
                .collect(),
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.run(calls),
        }
    }

    /// The weights this backend evaluates with, for the readout the solver
    /// still does itself.
    pub fn net(&self) -> &Net {
        match self {
            Backend::Reference(net, _) => net,
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.net(),
        }
    }

    /// Evaluate with new weights from here on. The backend itself survives:
    /// rebuilding it would tear down a CUDA context and recompile every kernel
    /// for a change that touches three arrays.
    pub fn set_net(&mut self, net: Net) -> Result<(), String> {
        match self {
            Backend::Reference(old, _) => {
                *old = net;
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.set_weights(net),
        }
    }
}

// ------------------------------------------------------------------- the farm

/// Many solves in flight in one process, and one thing that evaluates for all
/// of them.
///
/// Threads equal cores: a round runs when every one of them has parked, so a
/// thread that is still computing holds the others up, and oversubscribing
/// would make that worse rather than better. What the parked cores cost is the
/// batch itself, which is short next to the CFR work between rounds.
pub struct Farm {
    gate: Arc<Gate>,
    backend: Backend,
    /// Swapped when the trainer publishes; workers re-read it per chunk.
    nets: Arc<RwLock<Arc<crate::search::Nets>>>,
    collected: Arc<Mutex<Vec<Data>>>,
    workers: Vec<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
    /// What the rounds carried, for the utilisation report. Calls per round is
    /// the number that matters: it is how many solves shared a forward pass,
    /// and it should sit at the thread count.
    pub rounds: u64,
    pub round_rows: u64,
    pub round_calls: u64,
    /// Time inside the backend. Against wall clock this says how much of a
    /// round is the batch and how much is everything around it.
    pub round_nanos: u64,
}

impl Farm {
    /// Start `threads` solver threads. They block on the first round until
    /// `drive` is called, so nothing runs until the caller asks for rows.
    pub fn new(seed: u64, threads: usize, gc: GameCfg, backend: Backend) -> Farm {
        assert!(threads > 0, "a farm needs at least one thread");
        let value = backend.net().clone();
        let gate = Arc::new(Gate::default());
        let nets = Arc::new(RwLock::new(Arc::new(crate::search::Nets {
            value,
            gate: Some(Arc::clone(&gate)),
        })));
        let collected = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let workers = (0..threads)
            .map(|t| {
                let (gate, nets, collected, stopping) = (
                    Arc::clone(&gate),
                    Arc::clone(&nets),
                    Arc::clone(&collected),
                    Arc::clone(&stopping),
                );
                std::thread::Builder::new()
                    .name(format!("solve-{t}"))
                    .spawn(move || {
                        let _member = gate.enter();
                        let mut stream =
                            GameStream::new(seed.wrapping_mul(0x9E37_79B9) ^ t as u64, gc);
                        while !stopping.load(Ordering::Relaxed) {
                            // One chunk is small so a `drive` call can end
                            // near the row count it was asked for, and so a
                            // publish lands between chunks rather than inside
                            // a solve.
                            let n = Arc::clone(&*nets.read());
                            let d = stream.generate(&n, CHUNK_SOLVES);
                            collected.lock().push(d);
                        }
                    })
                    .expect("spawn solver thread")
            })
            .collect();
        Farm {
            gate,
            backend,
            nets,
            collected,
            workers,
            stopping,
            rounds: 0,
            round_rows: 0,
            round_calls: 0,
            round_nanos: 0,
        }
    }

    /// Install new weights, in the backend and in the copy the solver threads
    /// keep for the readout. Threads pick them up at their next chunk, so a
    /// solve is never evaluated against two different networks.
    pub fn publish(&mut self, net: Net) -> Result<(), String> {
        self.backend.set_net(net.clone())?;
        *self.nets.write() = Arc::new(crate::search::Nets {
            value: net,
            gate: Some(Arc::clone(&self.gate)),
        });
        Ok(())
    }

    /// Run rounds until the threads have produced at least `solves` rows, then
    /// return everything they produced. `eval` runs one batch.
    pub fn drive(&mut self, solves: usize) -> Data {
        let mut out = Data::default();
        let mut calls_seen = 0usize;
        loop {
            for d in self.collected.lock().drain(..) {
                out.merge(d);
            }
            if out.soff.len() >= solves {
                return out;
            }
            // Bounded, because a thread hands its chunk over between two
            // solves and is not parked while it does; blocking on a round here
            // would sit behind rows that are already finished.
            let backend = &self.backend;
            let mut spent = Duration::ZERO;
            match self.gate.round_before(PATIENCE, |calls| {
                calls_seen = calls.len();
                let at = std::time::Instant::now();
                let replies = backend.run(calls);
                spent = at.elapsed();
                replies
            }) {
                Round::Ran(rows) => {
                    self.rounds += 1;
                    self.round_rows += rows as u64;
                    self.round_calls += calls_seen as u64;
                    self.round_nanos += spent.as_nanos() as u64;
                }
                Round::Empty => {}
            }
        }
    }

    /// The network the driver evaluates with.
    pub fn value(&self) -> Arc<crate::search::Nets> {
        Arc::clone(&*self.nets.read())
    }
}

impl Drop for Farm {
    fn drop(&mut self) {
        // Ask the threads to stop, then keep answering rounds until they have
        // all finished the chunk they were in and left the count. Closing the
        // gate first would strand whoever was parked, and they would unwind
        // out of a solve instead of ending one.
        self.stopping.store(true, Ordering::Relaxed);
        let backend = &self.backend;
        while self.gate.serve_until_idle(|calls| backend.run(calls)).is_some() {}
        self.gate.close();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Solves a thread runs before it hands its rows over and re-reads the
/// network. Small enough that a publish is never more than this stale.
const CHUNK_SOLVES: usize = 8;

/// How long a driver waits for a round to fill before looking at what the
/// threads have already finished. Short next to the CFR work between rounds,
/// so a round that is genuinely coming is never missed by it.
const PATIENCE: Duration = Duration::from_millis(2);

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
        Call::Leaf {
            solve: 0,
            w: vec![tag as f32],
            opp: Vec::new(),
            rows: tag,
            player: 0,
        }
    }

    fn tag_of(c: &Call) -> f32 {
        match c {
            Call::Leaf { w, .. } => w[0],
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
                    .round(|calls| calls.iter().map(|c| Reply { a: vec![tag_of(c)], ..Default::default() }).collect())
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
                        calls
                            .iter()
                            .map(|c| Reply { a: vec![tag_of(c)], ..Default::default() })
                            .collect()
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
        let mut farm = Farm::new(5, THREADS, gc, Backend::Reference(small_net(0x2E57), Default::default()));
        let data = farm.drive(48);
        assert!(data.soff.len() >= 48, "only {} solves", data.soff.len());
        assert_eq!(data.nv, data.soff.len(), "a solve must store one row");
        assert_eq!(data.coff.len(), 2 * data.nv + 1, "ragged arena is malformed");
        assert!(farm.rounds > 0, "no round ever ran");
        // The whole point: a round is one forward pass shared by every thread
        // that is running a solve, not one pass per solve.
        let calls = farm.round_calls as f64 / farm.rounds as f64;
        assert!(
            calls > (THREADS as f64) * 0.5,
            "rounds averaged only {calls:.1} calls for {THREADS} threads"
        );
        assert!(farm.round_rows > 0, "rounds carried no rows");
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
