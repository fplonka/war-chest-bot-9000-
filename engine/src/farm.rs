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
    Trunk { xpub: Vec<f32>, cards: Vec<f32>, rows: usize },
    /// New configs: `f(c)` for the readout and `g(c)` for the pooling.
    Configs { phi: Vec<f32>, owner: Vec<u32>, cards: Vec<f32>, n: usize },
    /// Every leaf, every iteration: the belief-conditioned head.
    Join {
        p: Vec<f32>,
        jp: Vec<f32>,
        pooled: Vec<f32>,
        rows: usize,
        player: usize,
    },
}

/// What a call gives back. Two vectors covers all three: the trunk returns the
/// board vector and its join cache, the encoder returns `f` and `g`, and the
/// join returns `h` alone.
#[derive(Default)]
pub struct Reply {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl Call {
    /// Run this call on the CPU network. The batched driver runs the same
    /// arithmetic on the device; this is the reference both are held to.
    pub fn run(&self, net: &Net) -> Reply {
        let mut r = Reply::default();
        match self {
            Call::Trunk { xpub, cards, rows } => {
                // Two seat views per leaf in `xpub`, and one card table per
                // solve — `board` reads the physical view of each.
                net.board(xpub, cards, *rows, CARD_ROWS, &mut r.a);
                net.join_cache(&r.a, *rows, &mut r.b);
            }
            Call::Configs { phi, owner, cards, n } => {
                net.configs(phi, owner, *n, cards, &mut r.a, &mut r.b);
            }
            Call::Join { p, jp, pooled, rows, player } => {
                net.join(p, jp, pooled, *rows, *player, &mut r.a);
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
            Call::Join { .. } => 2,
        }
    }

    /// Rows this call contributes to its batch, for the round report.
    pub fn rows(&self) -> usize {
        match self {
            Call::Trunk { rows, .. } => *rows,
            Call::Configs { n, .. } => *n,
            Call::Join { rows, .. } => *rows,
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

#[derive(Default)]
struct Pending {
    /// Threads currently running a solve, so currently able to park.
    live: usize,
    parked: usize,
    calls: Vec<Option<Call>>,
    replies: Vec<Option<Reply>>,
    /// Bumped when the driver publishes replies, which is what a parked
    /// thread waits on rather than on a flag it could miss.
    epoch: u64,
    /// Answers published and not yet picked up. The next round may not start
    /// while this is non-zero.
    uncollected: usize,
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
    /// Join the round count for as long as the returned guard lives.
    pub fn enter(&self) -> Member<'_> {
        let mut g = self.round.lock();
        g.live += 1;
        Member { gate: self }
    }

    /// Submit one call and park until the driver has run the round.
    ///
    /// Returns `None` only when the farm is closing, which is the one case a
    /// caller cannot serve and must unwind from.
    pub fn submit(&self, call: Call) -> Option<Reply> {
        let mut g = self.round.lock();
        if g.closed {
            return None;
        }
        let epoch = g.epoch;
        let ticket = g.calls.len();
        g.calls.push(Some(call));
        g.parked += 1;
        if g.parked == g.live {
            self.full.notify_one();
        }
        while g.epoch == epoch && !g.closed {
            self.done.wait(&mut g);
        }
        if g.epoch == epoch {
            return None; // closed under us
        }
        let reply = g.replies.get_mut(ticket).and_then(|r| r.take());
        // Our round has been answered, and the driver holds the next one back
        // until every answer of this one is collected — so this slot is still
        // ours however late we woke.
        g.uncollected -= 1;
        if g.uncollected == 0 {
            self.full.notify_one();
        }
        reply
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
        let mut g = self.round.lock();
        // Publishing a round's answers overwrites the last round's. A thread
        // that woke late would then read this round's reply at its own ticket
        // — another solve's answer, of another solve's size, with nothing to
        // say it is wrong. So the previous round is not finished until every
        // answer has been picked up. That costs a lock and a move per thread,
        // not a solve, so it is not the straggler stall the patience avoids.
        while !g.closed && g.uncollected > 0 {
            self.full.wait(&mut g);
        }
        let deadline = patience.map(|p| std::time::Instant::now() + p);
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
        let calls: Vec<Call> = g.calls.drain(..).map(|c| c.expect("a filled slot")).collect();

        // The lock is held across `eval` on purpose. Every live thread is
        // parked on `done`, so none of them wants it; the only thread that
        // could take it is one just entering, and letting it in here would
        // give it ticket zero of a round that is already being answered. It
        // waits and joins the next round instead.
        let rows = calls.iter().map(Call::rows).sum();
        let replies = eval(&calls);
        assert_eq!(replies.len(), calls.len(), "one reply per call");

        g.replies = replies.into_iter().map(Some).collect();
        g.uncollected = g.replies.len();
        g.parked = 0;
        g.epoch += 1;
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
}

impl Drop for Member<'_> {
    fn drop(&mut self) {
        let mut g = self.gate.round.lock();
        g.live -= 1;
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
    pub fn run(&self, calls: &[Call]) -> Vec<Reply> {
        match self {
            // Every call in a round is independent, and the solver threads
            // that raised them are all parked, so the cores are free.
            Backend::Reference(net) => calls.par_iter().map(|c| c.run(net)).collect(),
            #[cfg(feature = "gpu")]
            Backend::Cuda(d) => d.run(calls),
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
        }
    }

    /// Install new weights, in the backend and in the copy the solver threads
    /// keep for the readout. Threads pick them up at their next chunk, so a
    /// solve is never evaluated against two different networks.
    pub fn publish(&mut self, backend: Backend) {
        *self.nets.write() = Arc::new(crate::search::Nets {
            value: backend.net().clone(),
            gate: Some(Arc::clone(&self.gate)),
        });
        self.backend = backend;
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
            match self.gate.round_before(PATIENCE, |calls| {
                calls_seen = calls.len();
                backend.run(calls)
            }) {
                Round::Ran(rows) => {
                    self.rounds += 1;
                    self.round_rows += rows as u64;
                    self.round_calls += calls_seen as u64;
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
        Call::Join {
            p: vec![tag as f32],
            jp: Vec::new(),
            pooled: Vec::new(),
            rows: tag,
            player: 0,
        }
    }

    fn tag_of(c: &Call) -> f32 {
        match c {
            Call::Join { p, .. } => p[0],
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
                    .round(|calls| calls.iter().map(|c| Reply { a: vec![tag_of(c)], b: Vec::new() }).collect())
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
                            .map(|c| Reply { a: vec![tag_of(c)], b: Vec::new() })
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
        let mut farm = Farm::new(5, THREADS, gc, Backend::Reference(small_net(0x2E57)));
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
