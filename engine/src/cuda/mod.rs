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

/// A solve's board vectors, kept on the card that produced them.
///
/// The trunk runs once per leaf and the join runs once per leaf per iteration,
/// so these are read sixty-four times for every time they are written. Sending
/// them from the host each time was most of what crossed the bus, and it sent
/// the card back the very numbers it had just computed.
#[derive(Default)]
struct Boards {
    p: Option<CudaSlice<f32>>,
    jp: Option<CudaSlice<f32>>,
    /// `f(c)` and `g(c)`, and the belief index that names them. All of it is
    /// read every iteration and written once.
    f: Option<CudaSlice<f32>>,
    g: Option<CudaSlice<f32>>,
    cidx: Option<CudaSlice<u32>>,
    coff: Option<CudaSlice<u32>>,
    /// Cells and queries actually written, as against `caps`, which is what
    /// the buffers could hold.
    cells: usize,
    queries: usize,
    /// The same offsets on the host. A round has to know where each solve's
    /// queries start to lay the batch out, and reading that back off the card
    /// would sync the stream once per solve per iteration — which is the cost
    /// this whole design exists to avoid.
    host_coff: Vec<u32>,
    /// Elements each buffer can hold, one per buffer. They only grow, and each
    /// is bounded by the solve's budget, so a slot settles and stays there.
    caps: [usize; 6],
    /// The flat description the CFR sweeps read. Uploaded whole when it grows
    /// rather than incrementally: it is a few megabytes against the tens the
    /// sweeps then run over without touching the bus again, and an incremental
    /// upload would have to mirror `Contract::extend`'s layout rules on both
    /// sides of the bus for no measured gain.
    tree: Option<Tree>,
}

struct Card {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    /// Indexed by solve, which is the gate slot the call came from.
    boards: parking_lot::Mutex<Vec<Boards>>,
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
            boards: parking_lot::Mutex::new(Vec::new()),
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
        self.trunk(calls, &pick(0), &mut out)?;
        self.configs(calls, &pick(1), &mut out)?;
        self.leaf(calls, &pick(2), &mut out)?;
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

    /// Keep `n` rows of a trunk batch as solve `solve`'s board vectors,
    /// starting at row `row0` of that solve. Row zero starts a fresh solve.
    ///
    /// The buffer is grown to the solve's whole node budget on first use
    /// rather than reallocated per growth step: a lane that grows its buffers
    /// to the largest tree it has ever served and never gives them back is
    /// what filled a 24 GiB card in the architecture this replaces, so the
    /// size is bounded by the budget and reused rather than left to climb.
    fn keep(
        &self,
        solve: usize,
        row0: usize,
        n: usize,
        p: &CudaSlice<f32>,
        jp: &CudaSlice<f32>,
        from: usize,
    ) -> Res<()> {
        let mut g = self.boards.lock();
        if g.len() <= solve {
            g.resize_with(solve + 1, Boards::default);
        }
        let b = &mut g[solve];
        let (mut pc, mut jc) = (b.caps[0], b.caps[1]);
        self.stash(&mut b.p, &mut pc, row0, p, from, n, D)?;
        self.stash(&mut b.jp, &mut jc, row0, jp, from, n, JW)?;
        b.caps[0] = pc;
        b.caps[1] = jc;
        Ok(())
    }

    /// Grow `slot` to hold `want` elements and write `src[from..from+n]` at
    /// `at`. One helper for every resident array, since they differ only in
    /// width.
    fn stash<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default>(
        &self,
        slot: &mut Option<CudaSlice<T>>,
        have: &mut usize,
        at: usize,
        src: &CudaSlice<T>,
        from: usize,
        n: usize,
        width: usize,
    ) -> Res<()> {
        let want = (at + n) * width;
        if *have < want {
            let cap = want.next_power_of_two();
            let mut fresh = self.stream.alloc_zeros::<T>(cap).map_err(err)?;
            if let Some(old) = slot.as_ref() {
                if *have > 0 {
                    let keep = (*have).min(at * width);
                    if keep > 0 {
                        let mut d = fresh.slice_mut(0..keep);
                        self.stream
                            .memcpy_dtod(&old.slice(0..keep), &mut d)
                            .map_err(err)?;
                    }
                }
            }
            *slot = Some(fresh);
            *have = cap;
        }
        let dst = slot.as_mut().expect("just allocated");
        let mut d = dst.slice_mut(at * width..want);
        self.stream
            .memcpy_dtod(&src.slice(from * width..(from + n) * width), &mut d)
            .map_err(err)?;
        Ok(())
    }

    /// The belief index of the queries the trunk just made, appended to the
    /// solve's. `coff` arrives relative to this call's own `cidx`, so it is
    /// shifted onto the resident one before it is stored.
    fn keep_index(&self, solve: usize, row0: usize, cidx: &[u32], coff: &[u32]) -> Res<()> {
        let mut g = self.boards.lock();
        if g.len() <= solve {
            g.resize_with(solve + 1, Boards::default);
        }
        let b = &mut g[solve];
        let base = if row0 == 0 { 0 } else { b.cells as u32 };
        if row0 == 0 {
            b.cells = 0;
            b.queries = 0;
        }
        let shifted: Vec<u32> = coff.iter().map(|x| x + base).collect();
        if row0 == 0 {
            b.host_coff.clear();
            b.host_coff.push(0);
        }
        b.host_coff.extend(shifted.iter().skip(1));
        let up_i = self.up(cidx)?;
        let up_o = self.up(&shifted)?;
        let (mut ic, mut oc) = (b.caps[4], b.caps[5]);
        self.stash(&mut b.cidx, &mut ic, b.cells, &up_i, 0, cidx.len(), 1)?;
        // `coff` has one more entry than it has queries; row zero writes the
        // leading zero and every later call overwrites it with its own first
        // offset, which is the same number.
        self.stash(&mut b.coff, &mut oc, b.queries, &up_o, 0, shifted.len(), 1)?;
        b.caps[4] = ic;
        b.caps[5] = oc;
        b.cells += cidx.len();
        b.queries += shifted.len() - 1;
        Ok(())
    }

    /// Put a solve's tree description on the card.
    ///
    /// Whole, not incremental. It is a few megabytes against the tens of
    /// megabytes of sweep traffic it then saves, and mirroring
    /// `Contract::extend`'s layout rules across the bus would buy nothing
    /// measured.
    pub fn keep_tree(&self, solve: usize, c: &crate::contract::Contract) -> Res<()> {
        let t = Tree {
            parent: self.up(&c.parent)?,
            player: self.up(&c.player)?,
            kind: self.up(&c.kind)?,
            nc: self.up(&c.nc.iter().flatten().copied().collect::<Vec<u32>>())?,
            roff: self.up(&c.roff)?,
            voff: self.up(&c.voff)?,
            soff: self.up(&c.soff)?,
            child_at: self.up(&c.child_at)?,
            child_n: self.up(&c.child_n)?,
            child: self.up(&c.child)?,
            legal_base: self.up(&c.legal_base)?,
            legal_off: self.up(&c.legal_off)?,
            legal_child: self.up(&c.legal_child)?,
            legal_trans: self.up(&c.legal_trans)?,
            rev_base: self.up(&c.rev_base)?,
            rev_start: self.up(&c.rev_start)?,
            rev_src: self.up(&c.rev_src)?,
            rev_cell: self.up(&c.rev_cell)?,
            rvd_base: self.up(&c.rvd_base)?,
            rvd_start: self.up(&c.rvd_start)?,
            rvd_src: self.up(&c.rvd_src)?,
            rvd_p: self.up(&c.rvd_p)?,
            draw_base: self.up(&c.draw_base)?,
            draw_start: self.up(&c.draw_start)?,
            draw_to: self.up(&c.draw_to)?,
            draw_p: self.up(&c.draw_p)?,
            level_node: self.up(&c.level_node)?,
            level_start: c.level_start.clone(),
        };
        let mut g = self.boards.lock();
        if g.len() <= solve {
            g.resize_with(solve + 1, Boards::default);
        }
        g[solve].tree = Some(t);
        Ok(())
    }

    /// One CFR iteration's two sweeps, on the card.
    ///
    /// Reach walks the levels forward from level one -- the root's reach is
    /// the root belief and is seeded by the caller -- and backpropagation
    /// walks them backward. A level's nodes never depend on each other, which
    /// is what lets each level be one launch; `a_level_never_depends_on_itself`
    /// is what says so.
    #[allow(clippy::too_many_arguments)]
    pub fn sweep(
        &self,
        solve: usize,
        traverser: usize,
        cfr: crate::search::Cfr,
        factors: (f32, f32, f32),
        cur: &mut CudaSlice<f32>,
        reach: &mut CudaSlice<f32>,
        vals: &mut CudaSlice<f32>,
        regret: &mut CudaSlice<f32>,
        sum: &mut CudaSlice<f32>,
        qval: &mut CudaSlice<f32>,
    ) -> Res<()> {
        let g = self.boards.lock();
        let t = g
            .get(solve)
            .and_then(|b| b.tree.as_ref())
            .ok_or_else(|| format!("solve {solve} has no resident tree"))?;
        let levels = t.level_start.len().saturating_sub(1);
        let (da, db, dg) = factors;

        for level in 1..levels {
            let (lo, hi) = (t.level_start[level], t.level_start[level + 1]);
            let (lo_i, nodes) = (lo as i32, (hi - lo) as usize);
            if nodes == 0 {
                continue;
            }
            unsafe {
                self.stream
                    .launch_builder(&self.k.reach_sweep)
                    .arg(&t.level_node).arg(&lo_i)
                    .arg(&t.parent).arg(&t.player).arg(&t.nc).arg(&t.roff)
                    .arg(&t.rev_base).arg(&t.rev_start).arg(&t.rev_src).arg(&t.rev_cell)
                    .arg(&t.rvd_base).arg(&t.rvd_start).arg(&t.rvd_src).arg(&t.rvd_p)
                    .arg(&*cur).arg(&mut *reach).arg(&(nodes as i32))
                    .launch_unit(LaunchConfig {
                        grid_dim: (nodes as u32, 2, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(err)?;
        }

        let (tr, predict) = (traverser as i32, cfr.predict);
        for level in (0..levels).rev() {
            let (lo, hi) = (t.level_start[level], t.level_start[level + 1]);
            let (lo_i, nodes) = (lo as i32, (hi - lo) as usize);
            if nodes == 0 {
                continue;
            }
            unsafe {
                self.stream
                    .launch_builder(&self.k.backprop_sweep)
                    .arg(&t.level_node).arg(&lo_i)
                    .arg(&t.kind).arg(&t.player).arg(&t.nc).arg(&t.voff).arg(&t.soff)
                    .arg(&t.child_at).arg(&t.child_n).arg(&t.child)
                    .arg(&t.legal_base).arg(&t.legal_off)
                    .arg(&t.legal_child).arg(&t.legal_trans)
                    .arg(&t.draw_base).arg(&t.draw_start).arg(&t.draw_to).arg(&t.draw_p)
                    .arg(&mut *vals).arg(&mut *cur).arg(&mut *regret)
                    .arg(&mut *sum).arg(&mut *qval)
                    .arg(&tr).arg(&da).arg(&db).arg(&dg).arg(&predict)
                    .launch_unit(LaunchConfig {
                        grid_dim: (nodes as u32, 1, 1),
                        block_dim: (64, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(err)?;
        }
        Ok(())
    }

    fn alloc(&self, n: usize) -> Res<CudaSlice<f32>> {
        self.stream.alloc_zeros::<f32>(n.max(1)).map_err(err)
    }

    fn up<T: cudarc::driver::DeviceRepr>(&self, host: &[T]) -> Res<CudaSlice<T>> {
        self.stream.memcpy_stod(host).map_err(err)
    }

    /// A resident buffer's device address, for the pointer arrays that let a
    /// stage over a whole round reach every solve's own arrays in one launch.
    /// Event tracking is off, so the guard `device_ptr` returns records nothing.
    fn ptr<T>(&self, s: &CudaSlice<T>) -> u64 {
        s.device_ptr(&self.stream).0
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

        // Keep them, per solve, for the join calls that follow. The host still
        // takes `p` back: the policy head builds its action embeddings against
        // a node's own board vector, and that runs there.
        let host_p = self.down(&p, rows * D)?;
        let mut at = 0;
        for &i in mine {
            let n = calls[i].rows();
            let Call::Trunk { solve, at: row0, cidx, coff, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            self.keep(*solve, *row0, n, &p, &jp, at)?;
            self.keep_index(*solve, *row0, cidx, coff)?;
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
            let mut boards = self.boards.lock();
            for &i in mine {
                let k = calls[i].rows();
                let Call::Configs { solve, at: base, .. } = &calls[i] else {
                    unreachable!("config shard holds only config calls")
                };
                if boards.len() <= *solve {
                    boards.resize_with(solve + 1, Boards::default);
                }
                let b = &mut boards[*solve];
                let (mut fc, mut gc) = (b.caps[2], b.caps[3]);
                self.stash(&mut b.f, &mut fc, *base, &f, at, k, D)?;
                self.stash(&mut b.g, &mut gc, *base, &g, at, k, POOL)?;
                b.caps[2] = fc;
                b.caps[3] = gc;
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

    // ------------------------------------------------------------------ leaf

    /// One CFR iteration for every solve in the round.
    ///
    /// Pooling the beliefs, the join and the readout are one pass. The two
    /// intermediates -- the pooled block and the head -- are made and consumed
    /// on the card, so a round's traffic is the beliefs in and the values out
    /// and nothing else.
    ///
    /// `g` and `f` belong to a solve, not to the batch, so the pooling and the
    /// readout launch once per solve over slices of the batch's own arrays.
    /// Everything else is one launch for the whole round.
    fn leaf(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let mark = std::time::Instant::now();
        let (mut w, mut opp, mut player) = (Vec::new(), Vec::new(), Vec::new());
        let mut player_of_call: Vec<usize> = Vec::with_capacity(mine.len());
        let mut rows = 0usize;
        for &i in mine {
            let Call::Leaf { w: wv, opp: ov, rows: n, player: q, .. } = &calls[i] else {
                unreachable!("leaf shard holds only leaf calls")
            };
            assert_eq!(ov.len(), *n, "one opponent reach mass a leaf");
            w.extend_from_slice(wv);
            opp.extend_from_slice(ov);
            player.extend(std::iter::repeat(*q as i32).take(*n));
            player_of_call.push(*q);
            rows += n;
        }
        let l = &self.layout;
        let (rows_i, pool_i, d_i) = (rows as i32, POOL as i32, D as i32);
        let queries = 2 * rows;

        // Lay the round out. A solve's arrays stay where they are and travel as
        // pointers: cloning a `CudaSlice` allocates and copies on the device, so
        // the old per-part `Part { g, f }` duplicated every resident `f` and `g`
        // once per CFR iteration.
        let mut coff: Vec<u32> = Vec::with_capacity(queries + 1);
        coff.push(0);
        let mut part_of_row: Vec<i32> = Vec::with_capacity(rows);
        let mut local_row: Vec<i32> = Vec::with_capacity(rows);
        let mut base: Vec<i32> = Vec::with_capacity(mine.len());
        let (mut gp, mut fp, mut cip, mut pp, mut jpp) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(mine.len());
        {
            let boards = self.boards.lock();
            let mut cell = 0usize;
            for (part, &i) in mine.iter().enumerate() {
                let n = calls[i].rows();
                let solve = calls[i].solve();
                let b = boards
                    .get(solve)
                    .filter(|b| b.cidx.is_some() && b.f.is_some() && b.host_coff.len() > 2 * n)
                    .ok_or_else(|| format!("solve {solve} has nothing resident"))?;
                coff.extend(b.host_coff[1..=2 * n].iter().map(|x| x + cell as u32));
                part_of_row.extend(std::iter::repeat(part as i32).take(n));
                local_row.extend(0..n as i32);
                base.push(cell as i32);
                gp.push(self.ptr(b.g.as_ref().unwrap()));
                fp.push(self.ptr(b.f.as_ref().unwrap()));
                cip.push(self.ptr(b.cidx.as_ref().unwrap()));
                pp.push(self.ptr(b.p.as_ref().unwrap()));
                jpp.push(self.ptr(b.jp.as_ref().unwrap()));
                cell += b.host_coff[2 * n] as usize;
            }
        }
        // Where each row's values land. A leaf call asks about one player, so a
        // row contributes exactly that player's configs, and the row's own
        // query in `coff` already bounds them.
        let mut vlo: Vec<u32> = Vec::with_capacity(rows + 1);
        let mut cells = 0u32;
        let mut row0 = 0usize;
        for (&i, &q) in mine.iter().zip(&player_of_call) {
            let start = cells;
            for r in row0..row0 + calls[i].rows() {
                vlo.push(cells);
                let query = 2 * r + q;
                cells += coff[query + 1] - coff[query];
            }
            spans.push((start as usize, cells as usize));
            row0 += calls[i].rows();
        }
        vlo.push(cells);
        let cells = cells as usize;
        let t_marshal = mark.elapsed();
        let mark = std::time::Instant::now();
        let w_d = self.up(&w)?;
        let opp_d = self.up(&opp)?;
        let player_d = self.up(&player)?;
        let coff_d = self.up(&coff)?;
        let part_d = self.up(&part_of_row)?;
        let local_d = self.up(&local_row)?;
        let base_d = self.up(&base)?;
        let (gp, fp, cip, pp, jpp) = (
            self.up(&gp)?,
            self.up(&fp)?,
            self.up(&cip)?,
            self.up(&pp)?,
            self.up(&jpp)?,
        );
        let vlo_d = self.up(&vlo)?;
        let t_up = mark.elapsed();
        let mark = std::time::Instant::now();

        // The belief block, from each solve's resident `g`.
        let mut pooled = self.alloc(queries * POOL)?;
        let queries_i = queries as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.belief_pool)
                .arg(&gp).arg(&cip).arg(&part_d).arg(&base_d)
                .arg(&coff_d).arg(&w_d).arg(&mut pooled)
                .arg(&queries_i).arg(&pool_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (queries.max(1) as u32, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;

        // The join, over the resident board vectors.
        let mut p = self.alloc(rows * D)?;
        let mut jp = self.alloc(rows * JW)?;
        let jw_i = JW as i32;
        launch!(self, gather, rows * D, &pp, &part_d, &local_d, &mut p, &rows_i, &d_i)?;
        launch!(self, gather, rows * JW, &jpp, &part_d, &local_d, &mut jp, &rows_i, &jw_i)?;

        let mut input = self.alloc(rows * JOIN_IN)?;
        launch!(self, join_input, rows * JOIN_IN, &pooled, &player_d, &mut input, &rows_i, &pool_i)?;
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

        // The readout: one block a row, one warp a config of the queried
        // player, the row's head vector staged once in shared memory.
        let mut vals = self.alloc(cells.max(1))?;
        let bias = self.b.slice(l.value_bias..l.value_bias + 1);
        unsafe {
            self.stream
                .launch_builder(&self.k.readout)
                .arg(&fp).arg(&cip).arg(&part_d).arg(&base_d)
                .arg(&h).arg(&bias).arg(&coff_d).arg(&vlo_d)
                .arg(&player_d).arg(&opp_d)
                .arg(&mut vals).arg(&rows_i).arg(&d_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (rows.max(1) as u32, 1, 1),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 4 * D as u32,
                })
        }
        .map_err(err)?;
        let t_launch = mark.elapsed();
        let mark = std::time::Instant::now();
        let host = self.down(&vals, cells.max(1))?;
        let t_down = mark.elapsed();
        LEAF_NS[0].fetch_add(t_marshal.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        LEAF_NS[1].fetch_add(t_up.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        LEAF_NS[2].fetch_add(t_launch.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        LEAF_NS[3].fetch_add(t_down.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        for (&i, &(lo, hi)) in mine.iter().zip(&spans) {
            out.push((i, Reply { a: host[lo..hi].to_vec(), ..Default::default() }));
        }
        Ok(())
    }
}

/// A solve's tree, as the sweep kernels read it.
///
/// Every array here is `contract.rs` verbatim. The host keeps the level table
/// because it drives the launch loop, and reading it back would sync.
struct Tree {
    parent: CudaSlice<u32>,
    player: CudaSlice<u8>,
    kind: CudaSlice<u8>,
    nc: CudaSlice<u32>,
    roff: CudaSlice<u32>,
    voff: CudaSlice<u32>,
    soff: CudaSlice<u32>,
    child_at: CudaSlice<u32>,
    child_n: CudaSlice<u32>,
    child: CudaSlice<u32>,
    legal_base: CudaSlice<u32>,
    legal_off: CudaSlice<u32>,
    legal_child: CudaSlice<u32>,
    legal_trans: CudaSlice<u32>,
    rev_base: CudaSlice<u32>,
    rev_start: CudaSlice<u32>,
    rev_src: CudaSlice<u32>,
    rev_cell: CudaSlice<u32>,
    rvd_base: CudaSlice<u32>,
    rvd_start: CudaSlice<u32>,
    rvd_src: CudaSlice<u32>,
    rvd_p: CudaSlice<f32>,
    draw_base: CudaSlice<u32>,
    draw_start: CudaSlice<u32>,
    draw_to: CudaSlice<u32>,
    draw_p: CudaSlice<f32>,
    level_node: CudaSlice<u32>,
    level_start: Vec<u32>,
}

/// The driver and cuBLAS error types are `Debug` only.
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
