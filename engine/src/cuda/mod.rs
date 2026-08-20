//! The device backend: one round of the farm, on the GPU.
//!
//! The farm hands this a whole round — every solve in flight, one call each.
//! Those calls are not run one at a time. Calls of a kind are concatenated into
//! a single batch and the network runs **once per kind per round**, so a round
//! costs three chains of large GEMMs instead of a hundred small ones. That is
//! the entire reason the farm exists; a solve on its own is a couple of hundred
//! rows, which no accelerator is interested in.
//!
//! Two conventions carry the concatenation into the kernels:
//!
//! * a leaf's physical `xpub` row is `2 * r`. The paired canonical queries stay
//!   adjacent when calls are joined, so the copy `net::board` makes to pick the
//!   physical rows becomes a stride;
//! * anything that was constant within a call and varies across a batch — the
//!   card table a leaf reads, the seat a join asks about — becomes an index
//!   array.
//!
//! The arithmetic is `net.rs`, in the same order, and `tests/cuda_parity.rs`
//! holds it to `Backend::Reference` on the same weights.
//!
//! Every scratch buffer is allocated per pass. With event tracking off and one
//! stream per context, `CudaSlice` is a stream-ordered pool allocation, which
//! costs about as much as a kernel launch and keeps the code the same shape as
//! the CPU network.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchArgs, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use rayon::prelude::*;

use crate::board::{board, N_HEXES, NONE};
use crate::farm::{Call, Reply, CARD_ROWS};
use crate::net::{
    ln_block, Net, NetLayout, NormSpan, Span, C, CFGH, D, JBLOCKS, JOIN_IN, JW, LN_CFG, LN_H,
    LN_JOIN, LN_JOUT, LN_TRUNK, POOL, TYPE,
};
use crate::rebel::{
    CFEAT, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_LOOSE, OFF_PILES, PILE_COUNTS, PUBFEAT,
};

type Res<T> = Result<T, String>;

/// Where a leaf pass spends its wall clock: host marshalling, the uploads, the
/// launches, the one download. Cards are 20% busy while the host waits inside
/// this call, so which of these four it is decides everything.
pub static LEAF_NS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Report and reset.
pub fn leaf_breakdown() -> [f64; 4] {
    std::array::from_fn(|i| {
        LEAF_NS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
    })
}

const KERNELS: &str = include_str!("kernels.cu");

/// Everything in `kernels.cu`, resolved once at startup so a name that does not
/// exist is an error there rather than a wrong answer later.
struct Kernels {
    gelu: CudaFunction,
    add: CudaFunction,
    layernorm: CudaFunction,
    bias: CudaFunction,
    group_bias: CudaFunction,
    window: CudaFunction,
    gather: CudaFunction,
    seed_reach: CudaFunction,
    avg_block: CudaFunction,
    beliefs: CudaFunction,
    terminals: CudaFunction,
    expand: CudaFunction,
    finish: CudaFunction,
    tokens: CudaFunction,
    hex_facts: CudaFunction,
    type_pool: CudaFunction,
    stem: CudaFunction,
    neighbour_mix: CudaFunction,
    pool: CudaFunction,
    board_input: CudaFunction,
    cfg_slots: CudaFunction,
    sum_slots: CudaFunction,
    bag: CudaFunction,
    join_input: CudaFunction,
    belief_pool: CudaFunction,
    readout: CudaFunction,
    reach_sweep: CudaFunction,
    backprop_sweep: CudaFunction,
}

impl Kernels {
    fn load(m: &Arc<CudaModule>) -> Res<Kernels> {
        let get = |name: &str| {
            m.load_function(name)
                .map_err(|e| format!("kernel {name}: {e:?}"))
        };
        Ok(Kernels {
            gelu: get("k_gelu")?,
            add: get("k_add")?,
            layernorm: get("k_layernorm")?,
            bias: get("k_bias")?,
            group_bias: get("k_group_bias")?,
            window: get("k_window")?,
            gather: get("k_gather")?,
            seed_reach: get("k_seed_reach")?,
            avg_block: get("k_avg_block")?,
            beliefs: get("k_beliefs")?,
            terminals: get("k_terminals")?,
            expand: get("k_expand")?,
            finish: get("k_finish")?,
            tokens: get("k_tokens")?,
            hex_facts: get("k_hex_facts")?,
            type_pool: get("k_type_pool")?,
            stem: get("k_stem")?,
            neighbour_mix: get("k_neighbour_mix")?,
            pool: get("k_pool")?,
            board_input: get("k_board_input")?,
            cfg_slots: get("k_cfg_slots")?,
            sum_slots: get("k_sum_slots")?,
            bag: get("k_bag")?,
            join_input: get("k_join_input")?,
            belief_pool: get("k_belief_pool")?,
            readout: get("k_readout")?,
            reach_sweep: get("k_reach_sweep")?,
            backprop_sweep: get("k_backprop_sweep")?,
        })
    }
}

/// `LaunchArgs::launch` hands back the events it may have recorded. Event
/// tracking is off here, so it never records any and no call site wants them.
trait LaunchUnit {
    /// # Safety
    /// The same contract as `LaunchArgs::launch`: the arguments must match the
    /// kernel's signature and stay in bounds.
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError>;
}

impl LaunchUnit for LaunchArgs<'_> {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.launch(cfg).map(|_| ())
    }
}

/// `launch!(self, kernel, elements, args...)` — one kernel over `elements`
/// work items. The builder is the same nine lines every time, and spelling it
/// out buries the arithmetic it is there to express.
macro_rules! launch {
    ($card:expr, $kernel:ident, $n:expr, $($arg:expr),+ $(,)?) => {{
        let n = $n;
        unsafe {
            $card.stream
                .launch_builder(&$card.k.$kernel)
                $(.arg($arg))+
                .launch_unit(spread(n))
        }
        .map_err(err)
    }};
}

/// Threads per block for the elementwise kernels.
const THREADS: u32 = 256;

fn spread(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One block per row, and a power-of-two block width so the reduction inside
/// `k_layernorm` halves cleanly.
fn per_row(rows: usize, width: usize) -> LaunchConfig {
    let threads = width.next_power_of_two().clamp(32, 256) as u32;
    LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 4 * threads,
    }
}

/// The GPUs a run evaluates on.
///
/// A round is split across the cards by call, so each card builds and runs a
/// self-contained batch and nothing crosses the bus between them.
pub struct Device {
    cards: Vec<Card>,
    net: Net,
}

/// One device array of a solve's state.
///
/// It grows geometrically and keeps what it holds: regrets, visit counts and
/// the strategy sum accumulate across a solve's iterations, so a reallocation
/// that dropped them would silently restart the search.
struct Arr<T> {
    buf: Option<CudaSlice<T>>,
    cap: usize,
    /// How much of it the card has been told about, for the append-only pools.
    len: usize,
}

impl<T> Default for Arr<T> {
    fn default() -> Self {
        Arr { buf: None, cap: 0, len: 0 }
    }
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default> Arr<T> {
    fn fit(&mut self, stream: &Arc<CudaStream>, want: usize) -> Res<()> {
        if self.buf.is_some() && self.cap >= want {
            return Ok(());
        }
        let cap = want.next_power_of_two().max(1);
        let mut fresh = stream.alloc_zeros::<T>(cap).map_err(err)?;
        if self.cap > 0 {
            let old = self.buf.as_ref().expect("a capacity implies a buffer");
            let mut d = fresh.slice_mut(0..self.cap);
            stream.memcpy_dtod(&old.slice(0..self.cap), &mut d).map_err(err)?;
        }
        self.buf = Some(fresh);
        self.cap = cap;
        Ok(())
    }

    /// Grow to hold `at + host.len()` elements and write `host` at `at`.
    fn put(&mut self, stream: &Arc<CudaStream>, at: usize, host: &[T]) -> Res<()> {
        self.fit(stream, at + host.len())?;
        self.len = at + host.len();
        if host.is_empty() {
            return Ok(());
        }
        let dst = self.buf.as_mut().expect("just fitted");
        let mut d = dst.slice_mut(at..at + host.len());
        stream.memcpy_htod(host, &mut d).map_err(err)
    }

    /// Send whatever of `host` the card has not been told about yet. A rewind
    /// shortens `host`, which drops the tail and lets it be written again.
    fn append(&mut self, stream: &Arc<CudaStream>, host: &[T]) -> Res<()> {
        let at = self.len.min(host.len());
        self.put(stream, at, &host[at..])
    }

    /// Copy `n` elements of `src` starting at `from` to `at`.
    fn copy(&mut self, stream: &Arc<CudaStream>, at: usize, src: &CudaSlice<T>, from: usize, n: usize)
        -> Res<()> {
        self.fit(stream, at + n)?;
        if n == 0 {
            return Ok(());
        }
        let dst = self.buf.as_mut().expect("just fitted");
        let mut d = dst.slice_mut(at..at + n);
        stream.memcpy_dtod(&src.slice(from..from + n), &mut d).map_err(err)
    }

    /// Forget everything: a gate slot is reused by the next solve, and what
    /// the last one left in it is another tree's arithmetic.
    fn clear(&mut self, stream: &Arc<CudaStream>) -> Res<()> {
        self.len = 0;
        if let Some(b) = self.buf.as_mut() {
            stream.memset_zeros(b).map_err(err)?;
        }
        Ok(())
    }

    fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        self.buf.as_ref().map_or(0, |b| b.device_ptr(stream).0)
    }
}

/// Everything one solve keeps on its card: the network state that outlives an
/// iteration, the flat tree, and the CFR arenas themselves.
///
/// The whole CFR loop runs here. Nothing but the sampled expansion leaves
/// crosses the bus per iteration, which is the only arrangement the budget
/// allows: the arenas alone are tens of megabytes a round trip.
#[derive(Default)]
struct Solve {
    /// Board vectors and the join cache, once per leaf.
    p: Arr<f32>,
    jp: Arr<f32>,
    /// `f(c)` and `g(c)`, and the belief index that names them.
    f: Arr<f32>,
    g: Arr<f32>,
    cidx: Arr<u32>,
    coff: Arr<u32>,
    /// The same offsets on the host. A round has to know where each solve's
    /// queries start to lay the batch out, and reading that back off the card
    /// would sync the stream once per solve per iteration.
    host_coff: Vec<u32>,
    cells: usize,
    rows: usize,
    /// The flat tree, extended as `Contract` extends it.
    tree: Tree,
    /// The CFR arenas, laid out exactly as `Solver` lays them out.
    reach: Arr<f32>,
    vals: Arr<f32>,
    cur: Arr<f32>,
    regret: Arr<f32>,
    sum: Arr<f32>,
    qval: Arr<f32>,
    visits: Arr<f32>,
    prior: Arr<f32>,
    avg: Arr<f32>,
    rootb: Arr<f32>,
    leaf_node: Arr<u32>,
    term: Arr<u32>,
    nterm: usize,
    /// Level bounds, on the host, because they drive the launch loop.
    level_start: Vec<u32>,
    /// The expansion's own random stream, seeded once by the solver.
    seed: Arr<u64>,
}

/// Fields of `struct Tree` in `kernels.cu`, in order. Every one is eight bytes
/// wide, so the descriptor is positional and needs no packing rules.
const DESC: usize = 51;

impl Solve {
    fn describe(&self, s: &Arc<CudaStream>) -> [u64; DESC] {
        let t = &self.tree;
        [
            t.kind.ptr(s), t.player.ptr(s), t.nc.ptr(s), t.parent.ptr(s),
            t.roff.ptr(s), t.voff.ptr(s), t.soff.ptr(s), t.util.ptr(s),
            t.child_at.ptr(s), t.child_n.ptr(s), t.child.ptr(s),
            t.legal_base.ptr(s), t.legal_off.ptr(s), t.legal_child.ptr(s),
            t.legal_trans.ptr(s), t.cell_row.ptr(s),
            t.rev_base.ptr(s), t.rev_start.ptr(s), t.rev_src.ptr(s), t.rev_cell.ptr(s),
            t.rvd_base.ptr(s), t.rvd_start.ptr(s), t.rvd_src.ptr(s), t.rvd_p.ptr(s),
            t.draw_base.ptr(s), t.draw_start.ptr(s), t.draw_to.ptr(s), t.draw_p.ptr(s),
            t.level_start.ptr(s), t.level_node.ptr(s),
            self.reach.ptr(s), self.vals.ptr(s), self.cur.ptr(s), self.regret.ptr(s),
            self.sum.ptr(s), self.qval.ptr(s), self.visits.ptr(s), self.prior.ptr(s),
            self.avg.ptr(s), self.rootb.ptr(s),
            self.p.ptr(s), self.jp.ptr(s), self.f.ptr(s), self.g.ptr(s),
            self.cidx.ptr(s), self.coff.ptr(s),
            self.leaf_node.ptr(s), self.term.ptr(s), self.seed.ptr(s),
            self.level_start.len().saturating_sub(1) as u64,
            self.nterm as u64,
        ]
    }
}

struct Card {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    /// Indexed by solve, which is the gate slot the call came from.
    solves: parking_lot::Mutex<Vec<Solve>>,
    /// The weights exactly as `NetLayout` describes them.
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    /// Hex adjacency, `NONE` folded to `-1`.
    nb: CudaSlice<i32>,
    layout: NetLayout,
}

impl Device {
    /// Bring up one card per ordinal and upload the weights to each.
    pub fn new(ordinals: &[usize], net: Net) -> Res<Device> {
        if ordinals.is_empty() {
            return Err("no cuda device ordinals given".into());
        }
        if net.is_empty() {
            return Err("cannot start the device backend without weights".into());
        }
        let cards = ordinals
            .iter()
            .map(|&o| Card::new(o, &net))
            .collect::<Res<Vec<_>>>()?;
        Ok(Device { cards, net })
    }

    /// How many cards the driver can see.
    pub fn count() -> usize {
        CudaContext::device_count().unwrap_or(0).max(0) as usize
    }

    pub fn net(&self) -> &Net {
        &self.net
    }

    /// Point the cards at new weights.
    ///
    /// A publish used to build a whole new `Device`: a CUDA context per card
    /// and an NVRTC compile of every kernel, for a change that only ever
    /// touches three arrays. Nothing else about a card depends on the weights,
    /// and once a solve keeps state on the device the context cannot be torn
    /// down under it anyway.
    pub fn set_weights(&mut self, net: Net) -> Res<()> {
        if net.is_empty() {
            return Err("cannot publish empty weights to the device".into());
        }
        let flat = net.flat();
        for card in &mut self.cards {
            card.stream.context().bind_to_thread().map_err(err)?;
            card.stream.memcpy_htod(&flat.w, &mut card.w).map_err(err)?;
            card.stream.memcpy_htod(&flat.b, &mut card.b).map_err(err)?;
            card.stream.memcpy_htod(&flat.ln, &mut card.ln).map_err(err)?;
        }
        self.net = net;
        Ok(())
    }

    /// Evaluate a round. A device error is not recoverable and not worth
    /// limping past, so it stops the run.
    pub fn run(&self, calls: &[Call]) -> Vec<Reply> {
        match self.try_run(calls) {
            Ok(replies) => replies,
            Err(e) => panic!("cuda: {e}"),
        }
    }

    fn try_run(&self, calls: &[Call]) -> Res<Vec<Reply>> {
        // A solve's board vectors stay on the card that produced them, so a
        // solve is pinned to a card and cannot be dealt round-robin. Solves
        // are gate slots and there are many more of them than cards, so this
        // still splits a round about evenly. Config calls belong to no solve
        // and are dealt to keep both cards busy.
        let n = self.cards.len();
        let mut shards: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut spare = 0;
        for (i, c) in calls.iter().enumerate() {
            let card = if c.solve() == usize::MAX {
                spare += 1;
                (spare - 1) % n
            } else {
                c.solve() % n
            };
            shards[card].push(i);
        }
        let mut out: Vec<Reply> = (0..calls.len()).map(|_| Reply::default()).collect();
        let done = self
            .cards
            .par_iter()
            .zip(shards)
            .map(|(card, mine)| card.round(calls, &mine))
            .collect::<Res<Vec<_>>>()?;
        for part in done {
            for (i, reply) in part {
                out[i] = reply;
            }
        }
        Ok(out)
    }
}

impl Card {
    fn new(ordinal: usize, net: &Net) -> Res<Card> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("device {ordinal}: {e:?}"))?;
        // One stream per context and no sharing between them, so the read/write
        // events cudarc would otherwise create on every allocation buy nothing
        // and cost two event creations per buffer.
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let ptx = compile_ptx_with_opts(
            KERNELS,
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(ptx).map_err(err)?;
        let k = Kernels::load(&module)?;
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let flat = net.flat();
        let nb: Vec<i32> = board()
            .neighbors
            .iter()
            .flatten()
            .map(|&n| if n == NONE { -1 } else { n as i32 })
            .collect();
        Ok(Card {
            w: stream.memcpy_stod(&flat.w).map_err(err)?,
            b: stream.memcpy_stod(&flat.b).map_err(err)?,
            ln: stream.memcpy_stod(&flat.ln).map_err(err)?,
            nb: stream.memcpy_stod(&nb).map_err(err)?,
            stream,
            blas,
            k,
            solves: parking_lot::Mutex::new(Vec::new()),
            layout: NetLayout::new(),
        })
    }

    fn round(&self, calls: &[Call], mine: &[usize]) -> Res<Vec<(usize, Reply)>> {
        self.stream.context().bind_to_thread().map_err(err)?;
        let pick = |kind: usize| -> Vec<usize> {
            mine.iter()
                .copied()
                .filter(|&i| calls[i].kind() == kind)
                .collect()
        };
        let mut out = Vec::with_capacity(mine.len());
        // Named, because a driver error carries an errno and nothing else, and
        // five stages of a round are five very different suspects.
        fn at(stage: &'static str) -> impl Fn(String) -> String {
            move |e| format!("{stage}: {e}")
        }
        self.trunk(calls, &pick(0), &mut out).map_err(at("trunk"))?;
        self.configs(calls, &pick(1), &mut out).map_err(at("configs"))?;
        self.tree(calls, &pick(2)).map_err(at("tree"))?;
        self.iterate(calls, &pick(3), &mut out).map_err(at("iterate"))?;
        self.read(calls, &pick(4), &mut out).map_err(at("read"))?;
        Ok(out)
    }

    // ------------------------------------------------------------ primitives

    /// `out[rows, o] = inp[rows, i] @ w[i, o] + beta * out[rows, o]`, the
    /// row-major shape of `Lin::run`.
    ///
    /// cuBLAS is column-major, so the very same buffers read as their own
    /// transposes give this with no transposes and no repacking: computing
    /// `outᵀ[o, rows] = wᵀ[o, i] @ inpᵀ[i, rows]`.
    fn lin<A: DevicePtr<f32>, O: DevicePtrMut<f32>>(
        &self,
        s: Span,
        inp: &A,
        rows: usize,
        beta: f32,
        out: &mut O,
    ) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let cfg = GemmConfig {
            transa: CUBLAS_OP_N,
            transb: CUBLAS_OP_N,
            m: s.o as i32,
            n: rows as i32,
            k: s.i as i32,
            alpha: 1.0,
            lda: s.o as i32,
            ldb: s.i as i32,
            beta,
            ldc: s.o as i32,
        };
        let w = self.w.slice(s.w..s.w + s.i * s.o);
        unsafe { self.blas.gemm(cfg, &w, inp, out) }.map_err(err)
    }

    /// The per-column bias the GEMM does not carry. A span with no bias is a
    /// no-op, which is how `Lin::bias` behaves on an empty bias.
    fn bias(&self, s: Span, rows: usize, out: &mut CudaSlice<f32>) -> Res<()> {
        if s.b == usize::MAX || rows == 0 {
            return Ok(());
        }
        let bias = self.b.slice(s.b..s.b + s.o);
        let (rows, width) = (rows as i32, s.o as i32);
        launch!(self, bias, rows as usize * s.o, out, &bias, &rows, &width)
    }

    /// `Lin::run`: the GEMM and then the bias.
    fn run<A: DevicePtr<f32>>(
        &self,
        s: Span,
        inp: &A,
        rows: usize,
        out: &mut CudaSlice<f32>,
    ) -> Res<()> {
        self.lin(s, inp, rows, 0.0, out)?;
        self.bias(s, rows, out)
    }

    /// `Norm::apply` when `act`, `Norm::plain` when not.
    fn norm(&self, s: NormSpan, rows: usize, act: bool, x: &mut CudaSlice<f32>) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let g = self.ln.slice(s.g..s.g + s.width);
        let b = self.ln.slice(s.b..s.b + s.width);
        let (rows_i, width, act) = (rows as i32, s.width as i32, act as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.layernorm)
                .arg(x)
                .arg(&g)
                .arg(&b)
                .arg(&rows_i)
                .arg(&width)
                .arg(&act)
                .launch_unit(per_row(rows, s.width))
        }
        .map_err(err)
    }

    fn add(&self, x: &mut CudaSlice<f32>, y: &CudaSlice<f32>, n: usize) -> Res<()> {
        let n_i = n as i32;
        launch!(self, add, n, x, y, &n_i)
    }

    /// Reach the state of solve `solve`, creating it if this is its first call.
    fn slot<'g>(&self, g: &'g mut Vec<Solve>, solve: usize) -> &'g mut Solve {
        if g.len() <= solve {
            g.resize_with(solve + 1, Solve::default);
        }
        &mut g[solve]
    }

    fn alloc(&self, n: usize) -> Res<CudaSlice<f32>> {
        self.stream.alloc_zeros::<f32>(n.max(1)).map_err(err)
    }

    fn up<T: cudarc::driver::DeviceRepr>(&self, host: &[T]) -> Res<CudaSlice<T>> {
        self.stream.memcpy_stod(host).map_err(err)
    }

    fn down(&self, d: &CudaSlice<f32>, n: usize) -> Res<Vec<f32>> {
        self.stream.memcpy_dtov(&d.slice(0..n)).map_err(err)
    }

    // ----------------------------------------------------------------- trunk

    /// Every new leaf in the round: the board vector and the join cache.
    fn trunk(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        // Concatenate. `card_of_row` is what replaces `board`'s modulo: a leaf
        // reads the physical view of the card table its own solve drafted.
        let (mut xpub, mut cards, mut card_of_row) = (Vec::new(), Vec::new(), Vec::new());
        let mut rows = 0usize;
        for &i in mine {
            let Call::Trunk { xpub: xp, cards: cd, rows: n, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            // Concatenation only works if a call carries exactly its own rows.
            // A trailing tail from a caller's scratch buffer would shift every
            // later call in the batch and is invisible when a call runs alone.
            assert_eq!(xp.len(), 2 * n * PUBFEAT, "trunk xpub is not 2 rows a leaf");
            assert_eq!(cd.len(), CARD_ROWS * NTYPE * TYPE, "trunk card table");
            let base = (cards.len() / (NTYPE * TYPE)) as i32;
            xpub.extend_from_slice(xp);
            cards.extend_from_slice(cd);
            card_of_row.extend((0..*n).map(|r| base + ((2 * r) % CARD_ROWS) as i32));
            rows += n;
        }
        let cells = rows * N_HEXES;
        let stride = (2 * PUBFEAT) as i32;
        let (rows_i, cells_i) = (rows as i32, cells as i32);
        let (nhex, ntype, chan, nslot) = (N_HEXES as i32, NTYPE as i32, C as i32, NSLOT as i32);
        let l = &self.layout;

        let xpub = self.up(&xpub)?;
        let cards = self.up(&cards)?;
        let card_of_row = self.up(&card_of_row)?;

        // Tokens: projected pile counts, then the card token and seat on top.
        let mut piles = self.alloc(rows * NTYPE * PILE_COUNTS)?;
        let (off, width) = (OFF_PILES as i32, (NTYPE * PILE_COUNTS) as i32);
        launch!(self, window, rows * NTYPE * PILE_COUNTS, &xpub, &mut piles, &rows_i, &stride, &off, &width)?;
        let mut tokens = self.alloc(rows * NTYPE * TYPE)?;
        self.lin(l.pile, &piles, rows * NTYPE, 0.0, &mut tokens)?;
        let seat = self.w.slice(l.seat..l.seat + 2 * TYPE);
        let type_i = TYPE as i32;
        launch!(self, tokens, rows * NTYPE * TYPE, &cards, &card_of_row, &seat, &mut tokens, &rows_i, &ntype, &type_i, &nslot)?;

        // Stem.
        let mut projected = self.alloc(rows * NTYPE * C)?;
        self.run(l.tok_stem, &tokens, rows * NTYPE, &mut projected)?;
        let mut type_pool = self.alloc(rows * C)?;
        launch!(self, type_pool, rows * C, &projected, &mut type_pool, &rows_i, &ntype, &chan)?;
        let mut loose = self.alloc(rows * LOOSE)?;
        let (off, width) = (OFF_LOOSE as i32, LOOSE as i32);
        launch!(self, window, rows * LOOSE, &xpub, &mut loose, &rows_i, &stride, &off, &width)?;
        let mut glob = self.alloc(rows * C)?;
        self.run(l.glob_stem, &loose, rows, &mut glob)?;
        let mut facts = self.alloc(cells * HEX_FACTS)?;
        let mut occupant = self.stream.alloc_zeros::<i32>(cells.max(1)).map_err(err)?;
        let (hex_ch, hex_facts) = (HEX_CH as i32, HEX_FACTS as i32);
        launch!(self, hex_facts, cells, &xpub, &mut facts, &mut occupant, &rows_i, &stride, &nhex, &hex_ch, &hex_facts, &ntype)?;
        let mut x = self.alloc(cells * C)?;
        self.run(l.hex_stem, &facts, cells, &mut x)?;
        let pos = self.w.slice(l.pos..l.pos + N_HEXES * C);
        launch!(self, stem, cells * C, &mut x, &projected, &occupant, &pos, &glob, &type_pool, &cells_i, &nhex, &ntype, &chan)?;

        // Residual blocks over the board's adjacency.
        let mut a = self.alloc(cells * C)?;
        let mut mixed = self.alloc(cells * 2 * C)?;
        let mut y = self.alloc(cells * C)?;
        let mut pooled = self.alloc(rows * 2 * C)?;
        let mut gb = self.alloc(rows * C)?;
        let mut z = self.alloc(cells * C)?;
        for (i, blk) in l.blocks.iter().enumerate() {
            self.stream
                .memcpy_dtod(&x.slice(0..cells * C), &mut a)
                .map_err(err)?;
            self.norm(l.norms[ln_block(i, 0)], cells, true, &mut a)?;
            launch!(self, neighbour_mix, cells * C, &a, &self.nb, &mut mixed, &cells_i, &nhex, &chan)?;
            self.run(blk.mix, &mixed, cells, &mut y)?;
            launch!(self, pool, rows * C, &a, &mut pooled, &rows_i, &nhex, &chan)?;
            self.run(blk.pool, &pooled, rows, &mut gb)?;
            launch!(self, group_bias, cells * C, &mut y, &gb, &cells_i, &chan, &nhex)?;
            self.norm(l.norms[ln_block(i, 1)], cells, true, &mut y)?;
            self.run(blk.out, &y, cells, &mut z)?;
            self.add(&mut x, &z, cells * C)?;
        }
        self.norm(l.norms[LN_TRUNK], cells, true, &mut x)?;

        // The board head, and the half of the join that does not move between
        // CFR iterations.
        let width = 2 * C + LOOSE;
        let mut input = self.alloc(rows * width)?;
        let (off, loose_i) = (OFF_LOOSE as i32, LOOSE as i32);
        launch!(self, board_input, rows * width, &x, &xpub, &mut input, &rows_i, &nhex, &chan, &stride, &off, &loose_i)?;
        let mut p = self.alloc(rows * D)?;
        self.run(l.board_out, &input, rows, &mut p)?;
        let mut jp = self.alloc(rows * JW)?;
        self.run(l.join_p, &p, rows, &mut jp)?;

        // Keep them, per solve, for the iterations that follow. The host still
        // takes `p` back: the policy head builds its action embeddings against
        // a node's own board vector, and that runs there.
        let host_p = self.down(&p, rows * D)?;
        let mut at = 0;
        let mut g = self.solves.lock();
        for &i in mine {
            let n = calls[i].rows();
            let Call::Trunk { solve, at: row0, cidx, coff, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            let b = self.slot(&mut g, *solve);
            b.p.copy(&self.stream, row0 * D, &p, at * D, n * D)?;
            b.jp.copy(&self.stream, row0 * JW, &jp, at * JW, n * JW)?;
            // `coff` arrives relative to this call's own `cidx`, so it is
            // shifted onto the resident index before it is stored. Row zero
            // starts a fresh solve and writes the leading zero; every later
            // call overwrites it with its own first offset, the same number.
            if *row0 == 0 {
                b.cells = 0;
                b.host_coff.clear();
                b.host_coff.push(0);
            }
            let base = b.cells as u32;
            let shifted: Vec<u32> = coff.iter().map(|x| x + base).collect();
            b.host_coff.extend(shifted.iter().skip(1));
            b.cidx.put(&self.stream, b.cells, cidx)?;
            b.coff.put(&self.stream, 2 * row0, &shifted)?;
            b.cells += cidx.len();
            b.rows = row0 + n;
            out.push((
                i,
                Reply {
                    a: host_p[at * D..(at + n) * D].to_vec(),
                    ..Default::default()
                },
            ));
            at += n;
        }
        Ok(())
    }

    // --------------------------------------------------------------- configs

    /// `f(c)` for the readout and `g(c)` for the pooling, for every config the
    /// round asked about.
    fn configs(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let (mut phi, mut owner, mut cards) = (Vec::new(), Vec::new(), Vec::new());
        let mut n = 0usize;
        for &i in mine {
            let Call::Configs { phi: ph, owner: ow, cards: cd, n: k, .. } = &calls[i] else {
                unreachable!("config shard holds only config calls")
            };
            assert_eq!(ph.len(), k * CFEAT, "config phi is not one row a config");
            assert_eq!(ow.len(), *k, "config owner is not one entry a config");
            let base = (cards.len() / (NTYPE * TYPE)) as u32;
            phi.extend_from_slice(ph);
            owner.extend(ow.iter().map(|&q| q + base));
            cards.extend_from_slice(cd);
            n += k;
        }
        let views = cards.len() / (NTYPE * TYPE);
        let l = &self.layout;
        let (n_i, nslot, cfeat) = (n as i32, NSLOT as i32, CFEAT as i32);
        let (ntype, type_i, pool_i) = (NTYPE as i32, TYPE as i32, POOL as i32);

        let phi = self.up(&phi)?;
        let owner = self.up(&owner)?;
        let cards = self.up(&cards)?;

        let width = 3 + TYPE;
        let mut slots = self.alloc(n * NSLOT * width)?;
        launch!(self, cfg_slots, n * NSLOT * width, &phi, &owner, &cards, &mut slots, &n_i, &nslot, &cfeat, &ntype, &type_i)?;
        let mut hidden = self.alloc(n * NSLOT * CFGH)?;
        self.run(l.cfg1, &slots, n * NSLOT, &mut hidden)?;
        let hid = (n * NSLOT * CFGH) as i32;
        launch!(self, gelu, n * NSLOT * CFGH, &mut hidden, &hid)?;
        let mut u = self.alloc(n * CFGH)?;
        let cfgh = CFGH as i32;
        launch!(self, sum_slots, n * CFGH, &hidden, &mut u, &n_i, &nslot, &cfgh)?;
        self.norm(l.norms[LN_CFG], n, true, &mut u)?;
        let mut f = self.alloc(n * D)?;
        let mut g = self.alloc(n * POOL)?;
        let mut fp = self.alloc(n * D)?;
        self.run(l.cfg_f, &u, n, &mut f)?;
        self.run(l.cfg_g, &u, n, &mut g)?;
        // The policy's config vector, off the same encoding as the value's.
        self.run(l.cfg_p, &u, n, &mut fp)?;

        // The linear half of `g`, which pooling carries exactly.
        let mut bag = self.alloc(views * NTYPE * 3 * POOL)?;
        self.run(l.cfg_m, &cards, views * NTYPE, &mut bag)?;
        launch!(self, bag, n * POOL, &bag, &phi, &owner, &mut g, &n_i, &nslot, &ntype, &cfeat, &pool_i)?;

        // `f` and `g` stay: the readout and the belief pooling both run here
        // now, so neither has a reader on the host. `f_p` goes back, because
        // the policy prior is built there.
        let host_fp = self.down(&fp, n * D)?;
        let mut at = 0;
        {
            let mut solves = self.solves.lock();
            for &i in mine {
                let k = calls[i].rows();
                let Call::Configs { solve, at: base, .. } = &calls[i] else {
                    unreachable!("config shard holds only config calls")
                };
                let b = self.slot(&mut solves, *solve);
                b.f.copy(&self.stream, base * D, &f, at * D, k * D)?;
                b.g.copy(&self.stream, base * POOL, &g, at * POOL, k * POOL)?;
                at += k;
            }
        }
        let mut at = 0;
        for &i in mine {
            let k = calls[i].rows();
            out.push((
                i,
                Reply {
                    c: host_fp[at * D..(at + k) * D].to_vec(),
                    ..Default::default()
                },
            ));
            at += k;
        }
        Ok(())
    }

    // ------------------------------------------------------------ the CFR loop

    /// Bring each solve's tree, arenas and priors up to date with the host.
    ///
    /// Growth is the only thing the host still does inside a solve: it holds
    /// the game rules, so it turns the sampled leaves into decision nodes and
    /// describes them. Everything the description feeds stays here.
    fn tree(&self, calls: &[Call], mine: &[usize]) -> Res<()> {
        let mut g = self.solves.lock();
        for &i in mine {
            let Call::Tree {
                solve, contract, from, fresh, ncells, nreach, nvals,
                leaf_node, term, rootb, cur, prior_at, prior, seed,
            } = &calls[i] else {
                unreachable!("tree shard holds only tree calls")
            };
            let b = self.slot(&mut g, *solve);
            let s = &self.stream;
            if *fresh {
                // Regrets, visits and the strategy sum accumulate over a
                // solve, so the next solve to take this slot must not inherit
                // them. The rest is written before it is read.
                b.regret.clear(s)?;
                b.sum.clear(s)?;
                b.visits.clear(s)?;
                b.qval.clear(s)?;
            }
            b.tree.extend(s, contract, *from, *fresh)?;
            b.level_start.clear();
            b.level_start.extend_from_slice(&contract.level_start);
            b.nterm = term.len();
            b.leaf_node.put(s, 0, leaf_node)?;
            b.term.put(s, 0, term)?;
            if !rootb.is_empty() {
                b.rootb.put(s, 0, rootb)?;
                b.seed.put(s, 0, &[*seed])?;
            }
            // A node is given its cells when it is expanded, uniform over its
            // legal row, and until the policy head has spoken the prior is that
            // same uniform strategy.
            b.cur.put(s, *ncells - cur.len(), cur)?;
            b.prior.put(s, *prior_at, prior)?;
            b.regret.fit(s, *ncells)?;
            b.sum.fit(s, *ncells)?;
            b.qval.fit(s, *ncells)?;
            b.visits.fit(s, *ncells)?;
            b.avg.fit(s, *ncells)?;
            b.reach.fit(s, *nreach)?;
            b.vals.fit(s, *nvals)?;
        }
        Ok(())
    }

    /// One CFR iteration and one expansion phase, for every solve in the round.
    ///
    /// The whole iteration is here: reach forward, the network at every leaf,
    /// value backpropagation and the regret update for both traversers, the
    /// average strategy, and the trajectories that say where the tree grows
    /// next. The only thing that crosses the bus is the handful of leaves the
    /// expansion sampled.
    ///
    /// A level's nodes never depend on each other and neither do two solves, so
    /// one launch covers a whole level of the whole round: `blockIdx.y` is the
    /// solve and `blockIdx.x` the node within its level.
    fn iterate(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        fn at(stage: &'static str) -> impl Fn(String) -> String {
            move |e| format!("{stage}: {e}")
        }
        let mark = std::time::Instant::now();
        let parts = mine.len() as u32;
        let (mut desc, mut coff) = (Vec::with_capacity(mine.len() * DESC), vec![0u32]);
        let (mut part_of_row, mut local_row, mut base) = (Vec::new(), Vec::new(), Vec::new());
        let (mut rows, mut cells, mut nterm) = (0usize, 0u32, 0usize);
        let mut wide: Vec<u32> = Vec::new();
        {
            let g = self.solves.lock();
            for (part, &i) in mine.iter().enumerate() {
                let solve = calls[i].solve();
                let b = g
                    .get(solve)
                    .filter(|b| b.host_coff.len() == 2 * b.rows + 1)
                    .ok_or_else(|| format!("solve {solve} has no resident tree"))?;
                desc.extend_from_slice(&b.describe(&self.stream));
                coff.extend(b.host_coff[1..].iter().map(|x| x + cells));
                part_of_row.extend(std::iter::repeat(part as i32).take(b.rows));
                local_row.extend(0..b.rows as i32);
                base.push(cells as i32);
                cells += b.host_coff[2 * b.rows];
                rows += b.rows;
                nterm = nterm.max(b.nterm);
                while wide.len() + 1 < b.level_start.len() {
                    wide.push(0);
                }
                for (l, w) in b.level_start.windows(2).zip(wide.iter_mut()) {
                    *w = (*w).max(l[1] - l[0]);
                }
            }
        }
        let t_marshal = mark.elapsed();
        let mark = std::time::Instant::now();
        let trees = self.up(&desc)?;
        let coff_d = self.up(&coff)?;
        let part_d = self.up(&part_of_row)?;
        let local_d = self.up(&local_row)?;
        let base_d = self.up(&base)?;
        let t_up = mark.elapsed();
        let mark = std::time::Instant::now();

        let Call::Iterate { factors, predict, expand, puct, .. } = &calls[mine[0]] else {
            unreachable!("iterate shard holds only iterate calls")
        };
        let (da, db, dg) = *factors;
        let (predict, puct, expand) = (*predict, *puct, *expand);

        self.reaches(&trees, &wide, parts, 0, false).map_err(at("reach"))?;
        let (mut w, mut oppmass) = (self.alloc(cells as usize)?, self.alloc(rows)?);
        for traverser in 0..2usize {
            self.network(&trees, &part_d, &local_d, &base_d, &coff_d,
                         &mut w, &mut oppmass, rows, traverser).map_err(at("net"))?;
            self.terminals(&trees, nterm, parts, traverser).map_err(at("terminals"))?;
            self.backprop(&trees, &wide, parts, traverser, 0, (da, db, dg, predict))
                .map_err(at("backprop"))?;
        }
        // The regret update moved both players' strategies, so the reaches the
        // next iteration reads are stale until they are pushed down again --
        // and the average strategy is accumulated against those fresh ones.
        self.reaches(&trees, &wide, parts, 0, true).map_err(at("avg"))?;
        let leaves = self.expand(&trees, parts, expand, puct).map_err(at("expand"))?;
        let t_launch = mark.elapsed();
        let mark = std::time::Instant::now();
        let host = self.down_u32(&leaves, parts as usize * expand)?;
        let t_down = mark.elapsed();
        for (slot, n) in [t_marshal, t_up, t_launch, t_down].iter().enumerate() {
            LEAF_NS[slot].fetch_add(n.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        for (part, &i) in mine.iter().enumerate() {
            out.push((
                i,
                Reply {
                    leaves: host[part * expand..(part + 1) * expand].to_vec(),
                    ..Default::default()
                },
            ));
        }
        Ok(())
    }

    /// What the host needs back once a solve is done.
    ///
    /// The reference strategy, then a value pass under it: the same two sweeps
    /// and the same network, propagating and averaging under `avg` rather than
    /// under the regret-matching iterate. What crosses is the root's values and
    /// policy and the beliefs at the leaves the caller asks about — a few
    /// kilobytes against the tens of megabytes the arenas hold.
    fn read(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        for &i in mine {
            let Call::Read { solve, touched, vals_at, policy_at, reach_at } = &calls[i] else {
                unreachable!("read shard holds only read calls")
            };
            let (desc, wide, rows, cells, coff, nterm) = {
                let g = self.solves.lock();
                let b = g.get(*solve).ok_or_else(|| format!("solve {solve} is not resident"))?;
                let mut wide = vec![0u32; b.level_start.len().saturating_sub(1)];
                for (l, w) in b.level_start.windows(2).zip(wide.iter_mut()) {
                    *w = l[1] - l[0];
                }
                (b.describe(&self.stream), wide, b.rows,
                 b.host_coff[2 * b.rows] as usize, b.host_coff.clone(), b.nterm)
            };
            let trees = self.up(&desc)?;
            let coff_d = self.up(&coff)?;
            let part_d = self.up(&vec![0i32; rows])?;
            let local_d = self.up(&(0..rows as i32).collect::<Vec<i32>>())?;
            let base_d = self.up(&[0i32])?;
            let touched = (touched[0] as i32) | ((touched[1] as i32) << 1);
            self.finish(&trees, &wide, 1, touched)?;
            self.reaches(&trees, &wide, 1, 1, false)?;
            let (mut w, mut oppmass) = (self.alloc(cells)?, self.alloc(rows)?);
            let mut root = Vec::new();
            for traverser in 0..2usize {
                self.network(&trees, &part_d, &local_d, &base_d, &coff_d,
                             &mut w, &mut oppmass, rows, traverser)?;
                self.terminals(&trees, nterm, 1, traverser)?;
                self.backprop(&trees, &wide, 1, traverser, 1, (0.0, 0.0, 0.0, 0.0))?;
                // Backpropagating for the second player overwrites the first's
                // values, so the root's row is taken before the next pass runs.
                let g = self.solves.lock();
                let s = &g[*solve];
                let (at, n) = vals_at[traverser];
                root.extend(self.slice(&s.vals, at as usize, n as usize)?);
            }
            let g = self.solves.lock();
            let s = &g[*solve];
            let policy = self.slice(&s.avg, policy_at.0 as usize, policy_at.1 as usize)?;
            let mut beliefs = Vec::new();
            for &(at, n) in reach_at {
                beliefs.extend(self.slice(&s.reach, at as usize, n as usize)?);
            }
            out.push((i, Reply { a: root, b: policy, c: beliefs, ..Default::default() }));
        }
        Ok(())
    }

    fn slice(&self, a: &Arr<f32>, at: usize, n: usize) -> Res<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let buf = a.buf.as_ref().ok_or("reading an arena that was never written")?;
        self.stream.memcpy_dtov(&buf.slice(at..at + n)).map_err(err)
    }

    /// The grid one level of one round takes: `blockIdx.x` is the node within
    /// the level, `blockIdx.y` the solve. A solve whose tree is shallower than
    /// another's simply has no work at the deeper levels.
    fn grid(widest: u32, parts: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (widest, parts, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Push the reach probabilities down from the root beliefs, level by level.
    /// `also_avg` adds the reach-weighted iterate to the running strategy sum,
    /// which needs exactly the reaches this pass has just made current.
    fn reaches(&self, trees: &CudaSlice<u64>, wide: &[u32], parts: u32, avg: i32, also_avg: bool)
        -> Res<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.seed_reach)
                .arg(trees)
                .launch_unit(LaunchConfig {
                    grid_dim: (64, parts, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        for level in 1..wide.len() {
            if wide[level] == 0 {
                continue;
            }
            let level_i = level as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.reach_sweep)
                    .arg(trees).arg(&level_i).arg(&avg)
                    .launch_unit(Self::grid(wide[level], parts))
            }
            .map_err(err)?;
        }
        if also_avg {
            for level in 0..wide.len() {
                if wide[level] == 0 {
                    continue;
                }
                let level_i = level as i32;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.avg_block)
                        .arg(trees).arg(&level_i)
                        .launch_unit(Self::grid(wide[level], parts))
                }
                .map_err(err)?;
            }
        }
        Ok(())
    }

    /// Value backpropagation up the levels, for one traverser. `avg` averages
    /// under the reference strategy and leaves the regrets alone.
    #[allow(clippy::too_many_arguments)]
    fn backprop(
        &self,
        trees: &CudaSlice<u64>,
        wide: &[u32],
        parts: u32,
        traverser: usize,
        avg: i32,
        f: (f32, f32, f32, f32),
    ) -> Res<()> {
        let (tr, (da, db, dg, predict)) = (traverser as i32, f);
        for level in (0..wide.len()).rev() {
            if wide[level] == 0 {
                continue;
            }
            let level_i = level as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.backprop_sweep)
                    .arg(trees).arg(&level_i).arg(&tr).arg(&avg)
                    .arg(&da).arg(&db).arg(&dg).arg(&predict)
                    .launch_unit(Self::grid(wide[level], parts))
            }
            .map_err(err)?;
        }
        Ok(())
    }

    /// The network at every leaf of the round, for one traverser: normalise the
    /// beliefs, pool them, run the join, read the values out into each solve's
    /// own value arena. Nothing here touches the host.
    #[allow(clippy::too_many_arguments)]
    fn network(
        &self,
        trees: &CudaSlice<u64>,
        part_d: &CudaSlice<i32>,
        local_d: &CudaSlice<i32>,
        base_d: &CudaSlice<i32>,
        coff_d: &CudaSlice<u32>,
        w: &mut CudaSlice<f32>,
        oppmass: &mut CudaSlice<f32>,
        rows: usize,
        traverser: usize,
    ) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let l = &self.layout;
        let (rows_i, pool_i, d_i) = (rows as i32, POOL as i32, D as i32);
        let (queries, tr) = (2 * rows, traverser as i32);

        unsafe {
            self.stream
                .launch_builder(&self.k.beliefs)
                .arg(trees).arg(part_d).arg(local_d).arg(coff_d)
                .arg(&tr).arg(&mut *w).arg(&mut *oppmass).arg(&rows_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (rows as u32, 2, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;

        let mut pooled = self.alloc(queries * POOL)?;
        let queries_i = queries as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.belief_pool)
                .arg(trees).arg(part_d).arg(base_d).arg(coff_d).arg(&*w)
                .arg(&mut pooled).arg(&queries_i).arg(&pool_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (queries as u32, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;

        // The board vectors and the join cache, gathered out of the solves.
        let mut p = self.alloc(rows * D)?;
        let mut jp = self.alloc(rows * JW)?;
        let (jw_i, zero, one) = (JW as i32, 0i32, 1i32);
        launch!(self, gather, rows * D, trees, part_d, local_d, &zero, &mut p, &rows_i, &d_i)?;
        launch!(self, gather, rows * JW, trees, part_d, local_d, &one, &mut jp, &rows_i, &jw_i)?;

        let mut input = self.alloc(rows * JOIN_IN)?;
        let seat = self.up(&vec![tr; rows])?;
        launch!(self, join_input, rows * JOIN_IN, &pooled, &seat, &mut input, &rows_i, &pool_i)?;
        let mut z = self.alloc(rows * JW)?;
        self.stream.memcpy_dtod(&jp.slice(0..rows * JW), &mut z).map_err(err)?;
        self.lin(l.join_b, &input, rows, 1.0, &mut z)?;
        self.bias(l.join_b, rows, &mut z)?;
        let mut t = self.alloc(rows * JW)?;
        let mut dbuf = self.alloc(rows * JW)?;
        for i in 0..JBLOCKS {
            self.stream.memcpy_dtod(&z.slice(0..rows * JW), &mut t).map_err(err)?;
            self.norm(l.norms[LN_JOIN + i], rows, true, &mut t)?;
            self.run(l.join_w[i], &t, rows, &mut dbuf)?;
            self.add(&mut z, &dbuf, rows * JW)?;
        }
        self.norm(l.norms[LN_JOUT], rows, true, &mut z)?;
        let mut h = self.alloc(rows * D)?;
        self.stream.memcpy_dtod(&p.slice(0..rows * D), &mut h).map_err(err)?;
        self.lin(l.join_out, &z, rows, 1.0, &mut h)?;
        self.bias(l.join_out, rows, &mut h)?;
        self.norm(l.norms[LN_H], rows, false, &mut h)?;

        let bias = self.b.slice(l.value_bias..l.value_bias + 1);
        unsafe {
            self.stream
                .launch_builder(&self.k.readout)
                .arg(trees).arg(part_d).arg(local_d).arg(coff_d)
                .arg(&h).arg(&bias).arg(&*oppmass).arg(&tr).arg(&rows_i).arg(&d_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 4 * D as u32,
                })
        }
        .map_err(err)
    }

    /// Terminal leaves, scored from the game rather than from the network.
    fn terminals(&self, trees: &CudaSlice<u64>, most: usize, parts: u32, traverser: usize)
        -> Res<()> {
        if most == 0 {
            return Ok(());
        }
        let tr = traverser as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.terminals)
                .arg(trees).arg(&tr)
                .launch_unit(LaunchConfig {
                    grid_dim: (most.div_ceil(8) as u32, parts, 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)
    }

    /// The expansion phase: `sims` trajectories a solve, and the leaf each one
    /// reached. The simulations of one phase run in order, because each counts
    /// the visits it passes and the next is meant to see them.
    fn expand(&self, trees: &CudaSlice<u64>, parts: u32, sims: usize, puct: f32)
        -> Res<CudaSlice<u32>> {
        let n = (parts as usize * sims).max(1);
        let mut out = self.stream.alloc_zeros::<u32>(n).map_err(err)?;
        let (parts_i, sims_i) = (parts as i32, sims as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.expand)
                .arg(trees).arg(&mut out).arg(&parts_i).arg(&sims_i).arg(&puct)
                .launch_unit(LaunchConfig {
                    grid_dim: (parts.div_ceil(32).max(1), 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        Ok(out)
    }

    /// The reference strategy, once the tree has stopped growing.
    fn finish(&self, trees: &CudaSlice<u64>, wide: &[u32], parts: u32, touched: i32) -> Res<()> {
        for level in 0..wide.len() {
            if wide[level] == 0 {
                continue;
            }
            let level_i = level as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.finish)
                    .arg(trees).arg(&level_i).arg(&touched)
                    .launch_unit(Self::grid(wide[level], parts))
            }
            .map_err(err)?;
        }
        Ok(())
    }

    fn down_u32(&self, d: &CudaSlice<u32>, n: usize) -> Res<Vec<u32>> {
        self.stream.memcpy_dtov(&d.slice(0..n)).map_err(err)
    }
}

/// A solve's tree, as the CFR kernels read it. Every array is `contract.rs`
/// verbatim.
///
/// `Contract` is append-only apart from the rows of the leaves an expansion
/// just grew, so an update is the tail of every pool plus the per-node rows
/// from the earliest grown leaf onward. Uploading the whole description each
/// time would be tens of megabytes a growth on the large solves.
#[derive(Default)]
struct Tree {
    kind: Arr<u32>,
    player: Arr<u32>,
    nc: Arr<u32>,
    parent: Arr<u32>,
    roff: Arr<u32>,
    voff: Arr<u32>,
    soff: Arr<u32>,
    util: Arr<f32>,
    child_at: Arr<u32>,
    child_n: Arr<u32>,
    child: Arr<u32>,
    legal_base: Arr<u32>,
    legal_off: Arr<u32>,
    legal_child: Arr<u32>,
    legal_trans: Arr<u32>,
    cell_row: Arr<u32>,
    rev_base: Arr<u32>,
    rev_start: Arr<u32>,
    rev_src: Arr<u32>,
    rev_cell: Arr<u32>,
    rvd_base: Arr<u32>,
    rvd_start: Arr<u32>,
    rvd_src: Arr<u32>,
    rvd_p: Arr<f32>,
    draw_base: Arr<u32>,
    draw_start: Arr<u32>,
    draw_to: Arr<u32>,
    draw_p: Arr<f32>,
    level_start: Arr<u32>,
    level_node: Arr<u32>,
}

impl Tree {
    /// Bring the description up to date with `c`. `from` is the first node
    /// whose row may have changed — the earliest leaf this growth expanded.
    fn extend(&mut self, s: &Arc<CudaStream>, c: &crate::contract::Contract, from: usize,
              fresh: bool) -> Res<()> {
        if fresh {
            // The slot held another solve until a moment ago. `append` sends
            // whatever a pool has grown by, so a pool that starts shorter than
            // the last solve's would send nothing at all and the card would go
            // on reading the tree before it.
            for a in self.pools() {
                a.len = 0;
            }
            self.rvd_p.len = 0;
            self.draw_p.len = 0;
        }
        let wide: Vec<u32> = c.kind[from..].iter().map(|&x| x as u32).collect();
        self.kind.put(s, from, &wide)?;
        let wide: Vec<u32> = c.player[from..].iter().map(|&x| x as u32).collect();
        self.player.put(s, from, &wide)?;
        let wide: Vec<u32> = c.nc[from..].iter().flatten().copied().collect();
        self.nc.put(s, 2 * from, &wide)?;
        self.parent.put(s, from, &c.parent[from..])?;
        self.roff.put(s, from, &c.roff[from..])?;
        self.voff.put(s, from, &c.voff[from..])?;
        self.soff.put(s, from, &c.soff[from..])?;
        self.util.put(s, from, &c.util[from..])?;
        self.child_at.put(s, from, &c.child_at[from..])?;
        self.child_n.put(s, from, &c.child_n[from..])?;
        self.legal_base.put(s, from, &c.legal_base[from..])?;
        self.rev_base.put(s, from, &c.rev_base[from..])?;
        self.rvd_base.put(s, from, &c.rvd_base[from..])?;
        self.draw_base.put(s, from, &c.draw_base[from..])?;
        // The pools only ever grow, so their tail is the whole of the update.
        self.child.append(s, &c.child)?;
        self.legal_off.append(s, &c.legal_off)?;
        self.legal_child.append(s, &c.legal_child)?;
        self.legal_trans.append(s, &c.legal_trans)?;
        self.cell_row.append(s, &c.cell_row)?;
        self.rev_start.append(s, &c.rev_start)?;
        self.rev_src.append(s, &c.rev_src)?;
        self.rev_cell.append(s, &c.rev_cell)?;
        self.rvd_start.append(s, &c.rvd_start)?;
        self.rvd_src.append(s, &c.rvd_src)?;
        self.rvd_p.append(s, &c.rvd_p)?;
        self.draw_start.append(s, &c.draw_start)?;
        self.draw_to.append(s, &c.draw_to)?;
        self.draw_p.append(s, &c.draw_p)?;
        // Levels are recomputed whenever the tree grows, so they travel whole.
        // It is two entries a node between them.
        self.level_start.put(s, 0, &c.level_start)?;
        self.level_node.put(s, 0, &c.level_node)?;
        Ok(())
    }

    /// The append-only pools, which a fresh solve rewinds.
    fn pools(&mut self) -> [&mut Arr<u32>; 12] {
        [
            &mut self.child,
            &mut self.legal_off,
            &mut self.legal_child,
            &mut self.legal_trans,
            &mut self.cell_row,
            &mut self.rev_start,
            &mut self.rev_src,
            &mut self.rev_cell,
            &mut self.rvd_start,
            &mut self.rvd_src,
            &mut self.draw_start,
            &mut self.draw_to,
        ]
    }
}

/// The driver and cuBLAS error types are `Debug` only.
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
