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
//! * packed public rows are expanded after upload. A leaf's physical `xpub`
//!   row is `2 * r`, so paired canonical queries stay adjacent and the copy
//!   `net::board` makes becomes a stride;
//! * anything that was constant within a call and varies across a batch — the
//!   card table a leaf reads, the seat a join asks about — becomes an index
//!   array.
//!
//! The arithmetic is `net.rs`, in the same order, and `tests/cuda_parity.rs`
//! holds it to `Backend::Reference` on the same weights. The join is one kernel;
//! the trunk and the join multiply on the tensor cores.
//!
//! Scratch is allocated at carve from `RoundCap`. A round that needs more rows
//! tiles. Nothing a round runs is a device allocation.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchArgs, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::board::{board, N_HEXES, NONE};
use crate::farm::{Call, Prime, Reply, CARD_ROWS};
use crate::net::{
    ln_block, Net, NetLayout, NormSpan, Span, AFEAT, AW, BLOCKS, C, CFGH, D, JBLOCKS, JOIN_IN, JW,
    LN_ACT, LN_CFG, LN_H, LN_JOIN, LN_TRUNK, POOL, TYPE,
};
use crate::pbs::{
    CFEAT, HEX_BLOCK, HEX_CH, HEX_FACTS, LOOSE, MAX_COINS, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE,
    OFF_PILES, PILE_COUNTS, PLAYER_SCALARS, PUBFEAT, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE,
    ROW_HAND_SIZE, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_IDS,
    ROW_INITIATIVE, ROW_INIT_MOVED, ROW_PILES, ROW_PLIES, ROW_STACK_KIND, ROW_STACK_OWED,
    ROW_TO_ACT,
};
use crate::search::{Budget, Cfg, Cfr, Ent};
use crate::state::{CONT_CAP, MAX_MAIN_PLAYS, PENDING_KINDS};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

mod slot;
use slot::{Arr, Solve, DESC, FIELDS, C_CUR, C_PRIOR, C_QVAL, C_SUM, C_VISITS, R_REACH, R_VALS, B_P, B_JP, G_F, G_G, G_FP, Y_BOARD_OF, Y_COFF};

type Res<T> = Result<T, String>;

/// Where an iteration spends its wall clock, by stage. The first four are the
/// host's own: marshalling, the uploads, issuing the launches, the download
/// that ends the round. The rest are device stages, and are only filled when
/// `WARCHEST_STAGES` is set -- separating them means synchronising after each,
/// which changes the thing being measured, so it is off by default.
///
/// The last two are byte counts rather than nanoseconds. They ride the same
/// accumulator, and `leaf_breakdown` scales everything by 1e6 -- which turns
/// nanoseconds into milliseconds and bytes into megabytes, so both read
/// correctly.
pub const STAGES: [&str; 22] = [
    "marshal", "upload", "launch", "download",
    "reach", "beliefs", "join", "readout", "terminals", "backprop", "expand",
    "trunk", "configs", "tree",
    "t-marshal", "t-upload", "priors",
    "describe", "scatter",
    "sent", "regrown",
    // Not a rate but a level: how much device memory every solve arena on this
    // process holds. Solves in flight is what the rate is linear in, and this
    // is the ceiling on it.
    "held",
];

/// The bytes each of a solve's arrays holds, largest first.
///
/// `held` says solves in flight is memory-bound; this says which array to
/// argue with. Reported for the largest solve a card holds, because the mean
/// is not what fills the pool.
pub static CENSUS: parking_lot::Mutex<Vec<(&'static str, usize)>> =
    parking_lot::Mutex::new(Vec::new());

/// Device bytes every card's solve arenas hold. Fixed at carve: a slot is
/// allocated once at the budget and never grows, so this is a level, not a
/// rate, and `leaf_breakdown` reports it without resetting.
pub static HELD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static LEAF_NS: [std::sync::atomic::AtomicU64; STAGES.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; STAGES.len()];

/// Report and reset.
pub fn leaf_breakdown() -> [f64; STAGES.len()] {
    std::array::from_fn(|i| {
        if i == STAGES.len() - 1 {
            return HELD.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
        }
        LEAF_NS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
    })
}

/// Whether to time the device stages, which costs a stream synchronise apiece.
fn timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WARCHEST_STAGES").is_some())
}

const KERNELS: &str = include_str!("kernels.cu");

/// Everything in `kernels.cu`, resolved once at startup so a name that does not
/// exist is an error there rather than a wrong answer later.
struct Kernels {
    expand_rows: CudaFunction,
    gelu: CudaFunction,
    norm_ip: CudaFunction,
    bias: CudaFunction,
    window: CudaFunction,
    scatter: CudaFunction,
    seed_reach: CudaFunction,
    avg_block: CudaFunction,
    terminals: CudaFunction,
    expand: CudaFunction,
    finish: CudaFunction,
    tokens: CudaFunction,
    act_feats: CudaFunction,
    act_boards: CudaFunction,
    act_add: CudaFunction,
    prior: CudaFunction,
    hex_facts: CudaFunction,
    type_pool: CudaFunction,
    stem: CudaFunction,
    trunk: CudaFunction,
    cfg_slots: CudaFunction,
    sum_slots: CudaFunction,
    bag: CudaFunction,
    join: CudaFunction,
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
            expand_rows: get("k_expand_rows")?,
            gelu: get("k_gelu")?,
            norm_ip: get("k_norm_ip")?,
            bias: get("k_bias")?,
            window: get("k_window")?,
            scatter: get("k_scatter")?,
            seed_reach: get("k_seed_reach")?,
            avg_block: get("k_avg_block")?,
            terminals: get("k_terminals")?,
            expand: get("k_expand")?,
            finish: get("k_finish")?,
            tokens: get("k_tokens")?,
            act_feats: get("k_act_feats")?,
            act_boards: get("k_act_boards")?,
            act_add: get("k_act_add")?,
            prior: get("k_prior")?,
            hex_facts: get("k_hex_facts")?,
            type_pool: get("k_type_pool")?,
            stem: get("k_stem")?,
            trunk: get("k_trunk")?,
            cfg_slots: get("k_cfg_slots")?,
            sum_slots: get("k_sum_slots")?,
            bag: get("k_bag")?,
            join: get("k_join")?,
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

/// One block per row, threads across the row. What every kernel that walks a
/// `[rows, width]` matrix wants: a flat index would need a division and a
/// modulo per element, which is twenty-odd cycles against the one operation
/// the kernel is there to do.
fn rows_of(rows: usize, width: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (width.next_power_of_two().clamp(32, 256) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One row per warp, eight warps to a block. What `k_norm_ip` wants: its rows
/// are a hundred-odd wide, which one warp reduces in five shuffles where a
/// block spends the same time in barriers.
fn warp_rows(rows: usize) -> LaunchConfig {
    const WARPS: u32 = 8;
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(WARPS).max(1), 1, 1),
        block_dim: (32, WARPS, 1),
        shared_mem_bytes: 0,
    }
}

pub struct Device {
    /// `PIPELINE` cards per GPU, flattened. The farm's queues are one per GPU;
    /// both cards of a GPU share one slot pool and pull from that queue.
    cards: Vec<Card>,
    n_gpus: usize,
    net: Net,
    slot_bytes: usize,
}

/// Two rounds in flight: while set A grows on the host, set B iterates on the card.
pub const PIPELINE: usize = 2;

/// Round to the eleven significand bits a tensor core keeps, matching
/// `cvt.rna.tf32.f32`: nearest, ties away from zero.
///
/// The trunk multiplies in TF32, whose operands are single-precision numbers
/// with the low thirteen mantissa bits clear. Rounding the weights here rather
/// than letting the tensor core drop the bits is what keeps the error
/// unbiased, and it costs nothing: it happens once, when the weights are
/// uploaded. The activations are rounded the same way as `k_trunk` stores
/// them.
fn tf32(v: f32) -> f32 {
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x1000) & 0xFFFF_E000)
}

/// JOIN_IN padded to a whole `mma.m16n8k8` k-tile. Seven zero columns, 5 %.
const JOIN_K: usize = JOIN_IN.next_multiple_of(8);
const _: () = assert!(JOIN_K == 136 && JOIN_IN == 129);

/// One matrix in the order a warp of `mma.sync.m16n8k8` reads it.
///
/// Lane `l` holds `w[k + l % 4][n + l / 4]` and the value four rows below it.
/// Stored as the net stores it that is eight thirty-two-byte transactions a
/// fragment; here the fragment is written out lane by lane, so it is one
/// eight-byte load a lane. `k_pad` may exceed `k_real`: the extra rows are
/// zero, which is how JOIN_IN = 129 becomes a 136-deep tile.
fn pack_mma(w: &[f32], base: usize, k_real: usize, k_pad: usize, n: usize, out: &mut Vec<f32>) {
    assert_eq!(k_pad % 8, 0, "a packed matrix is whole fragments deep");
    assert_eq!(n % 8, 0, "a packed matrix is whole fragments across");
    for kt in 0..k_pad / 8 {
        for nt in 0..n / 8 {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let at = |k: usize| {
                    let row = 8 * kt + k;
                    if row >= k_real {
                        0.0
                    } else {
                        tf32(w[base + row * n + 8 * nt + g])
                    }
                };
                out.push(at(t));
                out.push(at(t + 4));
            }
        }
    }
}

/// The trunk's two matrices a block, laid out in the order the tensor cores
/// read them. Returns the buffer and where each matrix starts, mix then out,
/// block by block.
fn fragwise(l: &NetLayout, w: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let mut out = Vec::new();
    let mut at = Vec::new();
    for blk in &l.blocks {
        for s in [blk.mix, blk.out] {
            assert_eq!(s.o, C, "the trunk's matrices are the channel width across");
            assert_eq!(s.i % 8, 0, "the trunk's matrices are whole fragments deep");
            at.push(out.len());
            pack_mma(w, s.w, s.i, s.i, s.o, &mut out);
        }
    }
    (out, at)
}

/// `join_b`, the three `join_w`, then `join_out`, each packed the way `k_join`
/// indexes them. `join_b` is padded to `JOIN_K`.
fn join_pack(l: &NetLayout, w: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    assert_eq!(l.join_b.i, JOIN_IN, "join_b is JOIN_IN deep");
    assert_eq!(l.join_b.o, JW, "join_b is JW across");
    pack_mma(w, l.join_b.w, l.join_b.i, JOIN_K, l.join_b.o, &mut out);
    for s in l.join_w {
        assert_eq!(s.i, JW, "a join residual is JW deep");
        assert_eq!(s.o, JW, "a join residual is JW across");
        pack_mma(w, s.w, s.i, s.i, s.o, &mut out);
    }
    let o = l.join_out;
    assert_eq!(o.i, JW, "join_out is JW deep");
    assert_eq!(o.o, D, "join_out is D across");
    pack_mma(w, o.w, o.i, o.i, o.o, &mut out);
    out
}

/// The board rounded up to whole sixteen-row `mma` tiles, and the shared row
/// stride `k_trunk` holds it at. Both are `TRUNK_ROWS` and `TRUNK_LDS` there.
const TRUNK_ROWS: usize = N_HEXES.next_multiple_of(16);
const TRUNK_LDS: usize = C + 4;
const _: () = assert!(C == 96 && TRUNK_ROWS == 48, "k_trunk says so with its own defines");

/// What one block of `k_trunk` asks for: the residual stream, the operand the
/// tensor cores read with its padding rows, the neighbour projections, and the
/// pooled row with its bias.
const TRUNK_SHARED: usize = (2 * N_HEXES + TRUNK_ROWS) * TRUNK_LDS * 4 + 3 * C * 4;

/// The running sums of the join's biases, in the order its norms read them.
/// See `Card::owed`, which is where they are kept and why.
fn owed_by_the_join(l: &NetLayout, b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity((JBLOCKS + 1) * JW + D);
    let mut run = vec![0.0f32; JW];
    for s in std::iter::once(l.join_b).chain(l.join_w) {
        for (o, &x) in run.iter_mut().zip(&b[s.b..s.b + s.o]) {
            *o += x;
        }
        out.extend_from_slice(&run);
    }
    out.extend_from_slice(&b[l.join_out.b..l.join_out.b + D]);
    out
}

/// Leaf rows a pass works on at once.
///
/// The intermediates of the leaf pass are 2,560 bytes a row -- the pooled
/// belief block and the head, since the join keeps its residual stream in
/// registers -- so a tile large enough to fill the card costs forty-two
/// megabytes and a handful of extra launches. Scratch is allocated at this
/// size when the card is carved, so a round cannot grow it.
const TILE: usize = 16384;

/// Join rows one block of `k_join` holds.
///
/// Shared is `JROWS * (JOIN_K + 4) * 4` = 35,840 B, so two blocks fit an SM.
/// Sixteen warps, `__launch_bounds__(512, 3)`, caps registers at 42 a thread,
/// and two resident blocks are 66 % of the file. Two cards share a GPU and
/// the sweeps that run beside this need somewhere to put a block.
const JROWS: usize = 64;
const _: () = assert!(JROWS <= JW && JROWS % 16 == 0);

/// Every device buffer a round holds, from the slot count and the budget.
///
/// Carve allocates these. `Device::new` prices a card from the same numbers, so
/// a larger budget cannot overflow a scratch and cannot OOM a carve that fitted.
struct RoundCap {
    mass: usize,
    pooled: usize,
    h: usize,
    z: usize,
    input: usize,
    leaves: usize,
    piles: usize,
    tokens: usize,
    projected: usize,
    type_pool: usize,
    loose: usize,
    glob: usize,
    facts: usize,
    occupant: usize,
    x: usize,
    bag: usize,
    packed: usize,
    xpub: usize,
    cards: usize,
    card_of_row: usize,
    phi: usize,
    owner: usize,
    cfg_cards: usize,
    blob: usize,
    at: usize,
    src: usize,
    dst: usize,
    start: usize,
    trees: usize,
    work: usize,
    coff: usize,
    part: usize,
    local: usize,
    base: usize,
    prime: usize,
    touched: usize,
}

impl RoundCap {
    fn of(n: usize, b: &Budget, s: u32) -> RoundCap {
        let cards = n * CARD_ROWS * NTYPE * TYPE;
        // Host-packed tree columns (not board `p`/`jp` or config `f`/`g`/`fp`,
        // which stay on the card), plus the trunk's extra copies of board_of,
        // cidx and coff on the same scatter.
        let packed = FIELDS[0] * b.nodes
            + FIELDS[1] * b.cells
            + FIELDS[2] * b.reach
            + FIELDS[3] * b.draws
            + FIELDS[4] * b.rows
            + 2 * b.configs
            + b.cidx;
        let trunk = b.rows + b.cidx + 2 * b.rows + 1;
        let blob = n * (packed + trunk);
        RoundCap {
            mass: 2 * n * b.rows,
            pooled: 2 * TILE * POOL,
            h: 2 * TILE * D,
            z: TILE * D,
            input: TILE * (2 * C + LOOSE),
            leaves: n * s.max(1) as usize,
            piles: TILE * NTYPE * PILE_COUNTS,
            tokens: TILE * NTYPE * TYPE,
            projected: TILE * NTYPE * C,
            type_pool: TILE * C,
            loose: TILE * LOOSE,
            glob: TILE * C,
            facts: TILE * N_HEXES * HEX_FACTS,
            occupant: TILE * N_HEXES,
            x: TILE * N_HEXES * C,
            bag: n * CARD_ROWS * NTYPE * 3 * POOL,
            packed: TILE * ROW_BYTES,
            xpub: TILE * PUBFEAT,
            cards,
            card_of_row: TILE,
            phi: TILE * CFEAT,
            owner: TILE,
            cfg_cards: cards,
            blob,
            at: TILE,
            src: TILE,
            dst: TILE,
            start: TILE + 1,
            trees: n * DESC,
            work: n * b.nodes,
            coff: n * (2 * b.rows + 1),
            part: n * b.rows,
            local: n * b.rows,
            base: n,
            prime: 12 * TILE + n * b.cells,
            touched: n,
        }
    }

    fn bytes(&self) -> usize {
        let w4 = self.mass
            + self.pooled
            + self.h
            + self.z
            + self.input
            + self.leaves
            + self.piles
            + self.tokens
            + self.projected
            + self.type_pool
            + self.loose
            + self.glob
            + self.facts
            + self.occupant
            + self.x
            + self.bag
            + self.xpub
            + self.cards
            + self.card_of_row
            + self.phi
            + self.owner
            + self.cfg_cards
            + self.blob
            + self.at
            + self.src
            + self.start
            + self.work
            + self.coff
            + self.part
            + self.local
            + self.base
            + self.prime
            + self.touched;
        w4 * 4 + (self.trees + self.dst) * 8 + self.packed
    }
}

/// Fill a staging buffer with exactly `src`.
fn copy<T: Copy>(src: &[T]) -> impl FnOnce(&mut [T]) -> usize + '_ {
    move |dst: &mut [T]| {
        dst[..src.len()].copy_from_slice(src);
        src.len()
    }
}

/// A page-locked host buffer that grows like a `Vec`.
///
/// Every byte a round sends goes through one of these, because a copy from
/// ordinary pageable memory is not asynchronous. The driver has to stage such a
/// copy through a pinned buffer of its own, and it blocks the calling thread
/// and drains the stream while it does -- so a round with one explicit
/// synchronise had ninety implicit ones, and the cards stood idle through all
/// of them. Page-locked memory is copied by the DMA engine directly, with the
/// host free to carry on.
///
/// The event is what makes reuse safe: a buffer must not be overwritten while
/// the copy that reads it is still in flight, and `fill` waits on the copy the
/// last round issued.
struct Host<T> {
    buf: Option<cudarc::driver::PinnedHostSlice<T>>,
    len: usize,
    sent: Option<cudarc::driver::CudaEvent>,
}

impl<T> Default for Host<T> {
    fn default() -> Self {
        Host { buf: None, len: 0, sent: None }
    }
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy> Host<T> {
    /// Make room for `want`, hand the buffer over, and keep what was written.
    /// `f` returns how many elements it filled.
    fn fill(
        &mut self,
        stream: &Arc<CudaStream>,
        want: usize,
        f: impl FnOnce(&mut [T]) -> usize,
    ) -> Res<()> {
        if let Some(e) = &self.sent {
            e.synchronize().map_err(err)?;
        }
        if self.buf.as_ref().is_none_or(|b| b.len() < want) {
            self.buf = Some(unsafe { stream.context().alloc_pinned::<T>(want.max(1)) }.map_err(err)?);
        }
        let b = self.buf.as_mut().expect("just fitted");
        self.len = f(b.as_mut_slice().map_err(err)?);
        assert!(self.len <= b.len(), "a fill wrote past the buffer");
        Ok(())
    }

    /// Send what was filled into `dst`, without waiting for it.
    fn send(&mut self, stream: &Arc<CudaStream>, dst: &mut CudaSlice<T>) -> Res<()> {
        if self.len == 0 {
            return Ok(());
        }
        let src = &self.buf.as_ref().expect("a length implies a buffer")
            .as_slice().map_err(err)?[..self.len];
        let mut view = dst.slice_mut(0..self.len);
        stream.memcpy_htod(src, &mut view).map_err(err)?;
        if self.sent.is_none() {
            self.sent = Some(stream.context().new_event(None).map_err(err)?);
        }
        self.sent.as_ref().expect("just made").record(stream).map_err(err)
    }

    /// Receive `src` into this buffer. The copy is DMA into pinned memory, so
    /// the driver does not stage it; we wait once, after it is queued.
    fn recv<Src: DevicePtr<T>>(&mut self, stream: &Arc<CudaStream>, src: &Src) -> Res<Vec<T>> {
        let n = src.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if let Some(e) = &self.sent {
            e.synchronize().map_err(err)?;
        }
        if self.buf.as_ref().is_none_or(|b| b.len() < n) {
            self.buf = Some(unsafe { stream.context().alloc_pinned::<T>(n.max(1)) }.map_err(err)?);
        }
        let buf = self.buf.as_mut().expect("just fitted");
        stream.memcpy_dtoh(src, buf).map_err(err)?;
        if self.sent.is_none() {
            self.sent = Some(stream.context().new_event(None).map_err(err)?);
        }
        self.sent.as_ref().expect("just made").record(stream).map_err(err)?;
        self.sent.as_ref().expect("just made").synchronize().map_err(err)?;
        self.len = n;
        Ok(self.buf.as_ref().expect("just fitted").as_slice().map_err(err)?[..n].to_vec())
    }
}

/// One host buffer and the device buffer it is sent to, kept together and kept
/// between rounds. A round's staging is a fixed set of these, by role.
#[derive(Default)]
struct Wire<T> {
    host: Host<T>,
    dev: Arr<T>,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Copy> Wire<T> {
    fn with_cap(stream: &Arc<CudaStream>, cap: usize) -> Res<Wire<T>> {
        let cap = cap.max(1);
        let mut host = Host::default();
        host.fill(stream, cap, |_| 0)?;
        Ok(Wire { host, dev: Arr::with_cap(stream, cap)? })
    }

    fn put(
        &mut self,
        stream: &Arc<CudaStream>,
        want: usize,
        f: impl FnOnce(&mut [T]) -> usize,
    ) -> Res<()> {
        self.host.fill(stream, want, f)?;
        let n = self.host.len;
        self.dev.room(n.max(1))?;
        let dst = self.dev.buf.as_mut().expect("room");
        self.host.send(stream, dst)
    }

    fn buf(&self) -> &CudaSlice<T> {
        self.dev.buf.as_ref().expect("carved")
    }
}

/// A work item packs the solve above the node's place inside its level. The
/// kernels' `WORK_BITS` is the same split.
const WORK_BITS: u32 = 20;

/// A set of solves laid out as one batch, and the device arrays that describe
/// it. Every stage of an iteration reads these, so laying them out once is what
/// makes a round of thirty solves one launch a stage rather than thirty.
///
/// The device arrays are carved once, at the card's slot count times the
/// budget. `lay` fills them; it does not allocate.
struct Batch {
    trees: Wire<u64>,
    /// One work item per (solve, node), bucketed by level: what a level's
    /// launch hands to its blocks. `level_at[l]` is where level `l`'s bucket
    /// starts.
    work: Wire<u32>,
    level_at: Vec<u32>,
    coff: Wire<u32>,
    part: Wire<i32>,
    local: Wire<i32>,
    base: Wire<i32>,
    prime: Wire<u32>,
    touched: Wire<i32>,
    /// Prefixes of the batch, one per solve count. The solves are laid out
    /// longest-running first, so the ones still owed an iteration are always a
    /// prefix -- and an iteration that fewer solves want is the same launch
    /// with a shorter grid and fewer rows, at no host cost.
    upto: Vec<Prefix>,
    parts: u32,
    cells: usize,
}

impl Default for Batch {
    fn default() -> Batch {
        Batch {
            trees: Wire::default(),
            work: Wire::default(),
            level_at: Vec::new(),
            coff: Wire::default(),
            part: Wire::default(),
            local: Wire::default(),
            base: Wire::default(),
            prime: Wire::default(),
            touched: Wire::default(),
            upto: vec![Prefix::default()],
            parts: 0,
            cells: 0,
        }
    }
}

impl Batch {
    /// The whole batch, for the passes every member takes part in.
    fn all(&self) -> &Prefix {
        self.upto.last().expect("a batch has at least the empty prefix")
    }
}

/// What the first `parts` solves of a batch come to.
#[derive(Default, Clone)]
struct Prefix {
    parts: u32,
    rows: usize,
    /// Work items these solves own, level by level. They are the first of each
    /// bucket, so a launch covering a level is that many blocks from
    /// `level_at[l]`.
    items: Vec<u32>,
    /// The most terminals any one of them holds.
    nterm: usize,
}

/// Every write a round makes to its solves' arrays, gathered to be sent as one.
///
/// The pieces are small and there are thousands of them; concatenated they are
/// one upload and one kernel. A solve hands over its words already
/// concatenated, so the driver copies each solve's blob once and records where
/// each run inside it lands.
#[derive(Default)]
struct Pack {
    blob: Vec<u32>,
    dst: Vec<u64>,
    at: Vec<u32>,
    src: Vec<u32>,
    /// Prefix sum of the piece lengths, so the kernel can find its piece.
    sum: Vec<u32>,
    /// Words the pieces move, which is more than the blob holds once a run has
    /// two destinations.
    moved: u32,
}

impl Pack {
    /// Add words to the blob and say where they landed.
    fn words(&mut self, w: &[u32]) -> u32 {
        let base = self.blob.len() as u32;
        self.blob.extend_from_slice(w);
        base
    }

    /// Keep the buffers, drop the contents. A round concatenates tens of
    /// megabytes, and building that from an empty `Vec` every time is a dozen
    /// reallocations and thousands of first-touch page faults -- the same cost
    /// `Stage` and `Scratch` are kept to avoid.
    fn clear(&mut self) {
        self.blob.clear();
        self.dst.clear();
        self.at.clear();
        self.src.clear();
        self.sum.clear();
        self.moved = 0;
    }

    fn piece(&mut self, dst: u64, at: u32, src: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.sum.push(self.moved);
        self.moved += len;
        self.dst.push(dst);
        self.at.push(at);
        self.src.push(src);
    }
}

struct Card {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Arc<Kernels>,
    /// Indexed by solve, which is the slot the farm pinned it to. Shared by
    /// both cards of a GPU.
    solves: Arc<parking_lot::Mutex<Vec<Solve>>>,
    /// Host staging for a round's batches, kept between rounds for the same
    /// reason the device scratch is: a round concatenates sixteen megabytes of
    /// public encodings, and building that from an empty `Vec` every time is
    /// twenty-odd reallocations and four thousand first-touch page faults --
    /// which measured at a quarter of the whole round.
    host: parking_lot::Mutex<Stage>,
    /// A round's writes, kept between rounds for the same reason.
    pack: parking_lot::Mutex<Pack>,
    /// Pinned landing pads for the downloads that end a round.
    down: parking_lot::Mutex<Host<u32>>,
    down_f: parking_lot::Mutex<Host<f32>>,
    /// The batch index `lay` fills, carved once.
    batch: parking_lot::Mutex<Batch>,
    /// Scratch for one pass, kept between rounds.
    ///
    /// A round's intermediates are hundreds of megabytes -- four hundred
    /// thousand join rows at `D` wide is four hundred of them for the head
    /// alone -- and allocating and freeing that every round is work the driver
    /// does instead of the arithmetic. They are grown by role and reused, for
    /// the same reason `docs/PERF.md` pools the host's five big buffers by
    /// role rather than from one shared pool.
    scratch: parking_lot::Mutex<Scratch>,
    /// The weights exactly as `NetLayout` describes them.
    w: CudaSlice<f32>,
    /// The two square matrices of every trunk block, laid out the way a warp
    /// of `mma.sync.m16n8k8` reads them. See `fragwise`.
    wt: CudaSlice<f32>,
    /// The join's five matrices, packed the same way. See `join_pack`.
    jw: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    /// Hex adjacency, `NONE` folded to `-1`.
    nb: CudaSlice<i32>,
    card_facts: CudaSlice<f32>,
    locations: CudaSlice<u8>,
    /// What the join's residual stream is owed.
    ///
    /// Every block of the join adds its matrix multiply's bias to the same
    /// stream, and the only thing that reads the stream is the norm at the top
    /// of the next block. So the biases are never stored: this holds their
    /// running sums, one per norm, and the norm adds the one it needs as it
    /// reads. Five passes over `[rows, JW]` a call become none.
    ///
    /// Layout: `JBLOCKS + 1` sums of width `JW`, then the head's own bias of
    /// width `D`.
    owed: CudaSlice<f32>,
    /// Where every weight the fused trunk reads lives, in the order
    /// `k_trunk` expects. Built once: the layout never moves, and a publish
    /// replaces the numbers rather than their offsets.
    plan: CudaSlice<i32>,
    layout: NetLayout,
}

impl Device {
    /// Bring up one card per ordinal, carve each into slots at `cfg.budget`.
    ///
    /// `max_slots` is the host's cap across every ordinal: the farm has already
    /// asked how many host-side arenas fit, and a card that carved more than
    /// that would sit empty.
    pub fn new(ordinals: &[usize], net: Net, cfg: Cfg, max_slots: usize) -> Res<Device> {
        if ordinals.is_empty() {
            return Err("no cuda device ordinals given".into());
        }
        if net.is_empty() {
            return Err("cannot start the device backend without weights".into());
        }
        let budget = cfg.budget;
        let mut cards = Vec::with_capacity(ordinals.len() * PIPELINE);
        let mut left = max_slots;
        let mut slot_bytes = 0usize;
        for (g, &o) in ordinals.iter().enumerate() {
            let gpu = Gpu::get(o)?;
            gpu.ctx.bind_to_thread().map_err(err)?;
            let mut pair: Vec<Card> = (0..PIPELINE)
                .map(|_| Card::on(&gpu, &net))
                .collect::<Res<_>>()?;
            let s0 = Arc::clone(&pair[0].stream);
            s0.context().bind_to_thread().map_err(err)?;
            // A first allocation on a cold context can include NVRTC scratch
            // in the free-memory delta, so a slot looks like the whole card
            // and the carve yields one. Warm, then measure a second probe.
            drop(Solve::at_budget(&s0, &budget)?);
            s0.synchronize().map_err(err)?;
            let free0 = cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64;
            let probe = Solve::at_budget(&s0, &budget)?;
            s0.synchronize().map_err(err)?;
            let measured = free0.saturating_sub(cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64);
            let accounted = probe.bytes() as u64;
            let slot = measured.clamp(accounted, accounted.saturating_mul(2));
            drop(probe);
            s0.synchronize().map_err(err)?;
            if g == 0 {
                slot_bytes = slot as usize;
            }
            let free = cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64;
            let tile = Card::carve_bytes(0, &cfg);
            let extra = Card::carve_bytes(1, &cfg).saturating_sub(tile);
            let per = slot + PIPELINE as u64 * extra;
            let usable = free.saturating_sub(PIPELINE as u64 * tile);
            // Slot cost is the allocation delta. Carve is many buffers; packing
            // free to the last byte OOMs, so a tenth stays for fragmentation.
            let fit = (usable - usable / 10) / per.max(1);
            let gpus_left = ordinals.len() - g;
            // `fit` prices a slot as `tile + n * extra`. Carve is
            // `carve_bytes(n)` per pipe, which matches that only while every
            // buffer is linear in n. Walk n down until the bytes that will
            // actually be allocated fit.
            let mut n = (fit as usize).min(left / gpus_left.max(1));
            while n > 0 {
                let need = n as u64 * slot + PIPELINE as u64 * Card::carve_bytes(n, &cfg);
                if need + need / 10 <= free {
                    break;
                }
                n -= 1;
            }
            if g == 0 && n == 0 {
                return Err(format!(
                    "a slot of {slot} bytes does not fit in {free} bytes free"
                ));
            }
            let mut solves = Vec::with_capacity(n);
            for _ in 0..n {
                solves.push(Solve::at_budget(&s0, &budget)?);
            }
            if let Some(s) = solves.first() {
                *CENSUS.lock() = s.census();
            }
            let solves = Arc::new(parking_lot::Mutex::new(solves));
            for (p, card) in pair.iter_mut().enumerate() {
                card.solves = Arc::clone(&solves);
                card.carve(n, &cfg).map_err(|e| {
                    format!("carve gpu {g} pipe {p} n={n} slot={slot} free={free}: {e}")
                })?;
                card.stream.synchronize().map_err(err)?;
            }
            eprintln!("cuda: gpu {g} carved {n} solve slots");
            left = left.saturating_sub(n);
            cards.extend(pair);
        }
        Ok(Device { cards, n_gpus: ordinals.len(), net, slot_bytes })
    }

    /// How many GPUs a round can be spread over. Each has `PIPELINE` cards.
    pub fn cards(&self) -> usize {
        self.n_gpus
    }

    /// Slots this GPU holds. Admission is a pop from a free list of these.
    pub fn slots(&self, gpu: usize) -> usize {
        self.cards[gpu * PIPELINE].solves.lock().len()
    }

    /// Slots across every GPU.
    pub fn total_slots(&self) -> usize {
        (0..self.n_gpus).map(|g| self.slots(g)).sum()
    }

    /// Bytes one solve slot holds, summed from the arrays allocated at the budget.
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Slots one physical GPU holds.
    pub fn slots_per_card(&self) -> usize {
        if self.n_gpus == 0 {
            0
        } else {
            self.total_slots() / self.n_gpus
        }
    }

    /// How many cards the driver can see.
    pub fn count() -> usize {
        CudaContext::device_count().unwrap_or(0).max(0) as usize
    }

    pub fn net(&self) -> &Net {
        &self.net
    }

    /// Expand packed public rows with the same kernel the trunk uses.
    pub fn expand_rows(&self, rows: &[u8]) -> Res<Vec<f32>> {
        if rows.len() % ROW_BYTES != 0 {
            return Err("packed rows are not a multiple of ROW_BYTES".into());
        }
        let n = rows.len() / ROW_BYTES;
        if n == 0 {
            return Ok(Vec::new());
        }
        let card = &self.cards[0];
        card.stream.context().bind_to_thread().map_err(err)?;
        let packed = card.stream.memcpy_stod(rows).map_err(err)?;
        let mut out = card.stream.alloc_zeros::<f32>(n * PUBFEAT).map_err(err)?;
        let n_i = n as i32;
        unsafe {
            card.stream
                .launch_builder(&card.k.expand_rows)
                .arg(&packed)
                .arg(&card.card_facts)
                .arg(&card.locations)
                .arg(&mut out)
                .arg(&n_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        card.stream.memcpy_dtov(&out).map_err(err)
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
            let (lw, _) = fragwise(&card.layout, &flat.w);
            card.stream.memcpy_htod(&lw, &mut card.wt).map_err(err)?;
            card.stream
                .memcpy_htod(&join_pack(&card.layout, &flat.w), &mut card.jw)
                .map_err(err)?;
            card.stream.memcpy_htod(&flat.b, &mut card.b).map_err(err)?;
            card.stream.memcpy_htod(&flat.ln, &mut card.ln).map_err(err)?;
            let owed = owed_by_the_join(&card.layout, &flat.b);
            card.stream.memcpy_htod(&owed, &mut card.owed).map_err(err)?;
        }
        self.net = net;
        Ok(())
    }

    /// Evaluate a round. A device error is not recoverable and not worth
    /// limping past, so it stops the run.
    /// `None` when the round could not be answered. The caller closes its
    /// gate on that, so the cohort unwinds instead of parking on a card that
    /// is never going to reply.
    pub fn run(&self, calls: &[Call], lane: usize) -> Option<Vec<Reply>> {
        // `lane` is `gpu * PIPELINE + pipe`. Slots are shared per GPU; the
        // farm pins a solve to a GPU for its whole life, and either pipe of
        // that GPU may run the round.
        let all: Vec<usize> = (0..calls.len()).collect();
        match self.cards[lane].round(calls, &all) {
            Ok(part) => {
                let mut out: Vec<Reply> = (0..calls.len()).map(|_| Reply::default()).collect();
                for (i, reply) in part {
                    out[i] = reply;
                }
                Some(out)
            }
            Err(e) => {
                eprintln!("cuda: lane {lane}: {e}");
                None
            }
        }
    }

    /// Everything one solve keeps on the card that the CPU network can also
    /// produce, copied back.
    ///
    /// Nothing in a run reads any of it. The trunk's board vectors, the config
    /// encoder's three rows and the policy prior are made here and consumed
    /// here, so a round carries none of them home -- which is exactly why the
    /// arithmetic that makes them needs a way to be asked. The parity test is
    /// the only caller.
    pub fn resident(&self, card: usize, solve: usize) -> Res<Resident> {
        let c = &self.cards[card];
        c.stream.context().bind_to_thread().map_err(err)?;
        let g = c.solves.lock();
        let s = g.get(solve).ok_or_else(|| format!("solve {solve} is not resident"))?;
        let mut h = c.down_f.lock();
        Ok(Resident {
            p: s.get_f32(&c.stream, Ent::Board, B_P, 0, s.ent[Ent::Board as usize].len() * D, &mut h)?,
            jp: s.get_f32(&c.stream, Ent::Board, B_JP, 0, s.ent[Ent::Board as usize].len() * JW, &mut h)?,
            f: s.get_f32(&c.stream, Ent::Config, G_F, 0, s.ent[Ent::Config as usize].len() * D, &mut h)?,
            g: s.get_f32(&c.stream, Ent::Config, G_G, 0, s.ent[Ent::Config as usize].len() * POOL, &mut h)?,
            fp: s.get_f32(&c.stream, Ent::Config, G_FP, 0, s.ent[Ent::Config as usize].len() * D, &mut h)?,
            prior: s.get_f32(&c.stream, Ent::Cell, C_PRIOR, 0, s.ent[Ent::Cell as usize].len(), &mut h)?,
            cur: s.get_f32(&c.stream, Ent::Cell, C_CUR, 0, s.ncells, &mut h)?,
            sum: s.get_f32(&c.stream, Ent::Cell, C_SUM, 0, s.ncells, &mut h)?,
            qval: s.get_f32(&c.stream, Ent::Cell, C_QVAL, 0, s.ncells, &mut h)?,
            visits: s.get_f32(&c.stream, Ent::Cell, C_VISITS, 0, s.ncells, &mut h)?,
            reach: s.get_f32(&c.stream, Ent::Reach, R_REACH, 0, s.nreach, &mut h)?,
        })
    }
}

/// What `Device::resident` hands back: one solve's network state, in the same
/// layout `Solver` keeps it in on the host.
pub struct Resident {
    /// A board vector a leaf row, and the join's projection of it.
    pub p: Vec<f32>,
    pub jp: Vec<f32>,
    /// The readout's `f(c)`, the pooling's `g(c)` and the policy's `f_p(c)`.
    pub f: Vec<f32>,
    pub g: Vec<f32>,
    pub fp: Vec<f32>,
    /// The PUCT prior, over every strategy cell of the tree.
    pub prior: Vec<f32>,
    /// The CFR arenas the expansion phase reads, in `Solver`'s own layout.
    /// Nothing in a run reads these either: the loop that makes them is here
    /// and so is the growth that consumes them, so a round carries none of
    /// them home. `Solver::replay_expansion` takes them to hold the host's
    /// growth rule to the card's on the card's own numbers.
    pub cur: Vec<f32>,
    pub sum: Vec<f32>,
    pub qval: Vec<f32>,
    pub visits: Vec<f32>,
    pub reach: Vec<f32>,
}

/// One GPU's context and kernels, shared by its two cards.
struct Gpu {
    ctx: Arc<CudaContext>,
    k: Arc<Kernels>,
    torch: Arc<CudaStream>,
}

static GPUS: LazyLock<parking_lot::Mutex<HashMap<usize, Arc<Gpu>>>> =
    LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

fn compile_options(major: i32, minor: i32) -> CompileOptions {
    let define = |name: &str, value: usize| format!("-D{name}={value}");
    CompileOptions {
        options: vec![
            format!("--gpu-architecture=compute_{major}{minor}"),
            format!("-DJ_ROWS={JROWS}"),
            format!("-DJ_W={JW}"),
            format!("-DJ_IN={JOIN_K}"),
            format!("-DJ_POOL={POOL}"),
            format!("-DJ_D={D}"),
            format!("-DJ_BLOCKS={JBLOCKS}"),
            define("ROW_BYTES", ROW_BYTES),
            define("PUBFEAT", PUBFEAT),
            define("N_HEXES", N_HEXES),
            define("HEX_CH", HEX_CH),
            define("HEX_FACTS", HEX_FACTS),
            define("HEX_BLOCK", HEX_BLOCK),
            define("NTYPE", NTYPE),
            define("NSLOT", NSLOT),
            define("PILE_COUNTS", PILE_COUNTS),
            define("CARD_FEATS", CARD_FEATS),
            define("OFF_PILES", OFF_PILES),
            define("OFF_CARDS", OFF_CARDS),
            define("OFF_LOOSE", OFF_LOOSE),
            define("PLAYER_SCALARS", PLAYER_SCALARS),
            define("ROW_IDS", ROW_IDS),
            define("ROW_HEX_OWNER", ROW_HEX_OWNER),
            define("ROW_HEX_SLOT", ROW_HEX_SLOT),
            define("ROW_HEX_HEIGHT", ROW_HEX_HEIGHT),
            define("ROW_HEX_MARKER", ROW_HEX_MARKER),
            define("ROW_PILES", ROW_PILES),
            define("ROW_HAND_SIZE", ROW_HAND_SIZE),
            define("ROW_FD_SIZE", ROW_FD_SIZE),
            define("ROW_BAG_SIZE", ROW_BAG_SIZE),
            define("ROW_INITIATIVE", ROW_INITIATIVE),
            define("ROW_INIT_MOVED", ROW_INIT_MOVED),
            define("ROW_TO_ACT", ROW_TO_ACT),
            define("ROW_PLIES", ROW_PLIES),
            define("ROW_STACK_KIND", ROW_STACK_KIND),
            define("ROW_STACK_OWED", ROW_STACK_OWED),
            define("PENDING_KINDS", PENDING_KINDS),
            define("CONT_CAP", CONT_CAP),
            define("MAX_MAIN_PLAYS", MAX_MAIN_PLAYS as usize),
            format!("-DMAX_COINS={MAX_COINS:.1}f"),
        ],
        include_paths: vec![
            "/usr/local/cuda/include".into(),
            "/usr/include".into(),
        ],
        ..Default::default()
    }
}

/// Launch the row expander on memory and a stream owned by PyTorch.
pub fn expand_rows_torch(
    rows: u64,
    cards: u64,
    locations: u64,
    out: u64,
    n: usize,
    stream: usize,
    ordinal: i32,
) -> Res<()> {
    use cudarc::driver::{result, sys};
    if n == 0 {
        return Ok(());
    }
    let torch_ctx = result::ctx::get_current()
        .map_err(err)?
        .ok_or("PyTorch has no current CUDA context")?;
    let device = result::device::get(ordinal).map_err(err)?;
    let primary = unsafe { result::primary_ctx::retain(device) }.map_err(err)?;
    let same_context = primary == torch_ctx;
    unsafe { result::primary_ctx::release(device) }.map_err(err)?;
    if !same_context {
        return Err("PyTorch does not use the device primary CUDA context".into());
    }
    let gpu = Gpu::get(ordinal as usize)?;

    let before = gpu.ctx.new_event(None).map_err(err)?;
    unsafe {
        result::event::record(before.cu_event(), stream as sys::CUstream).map_err(err)?;
    }
    gpu.torch.wait(&before).map_err(err)?;

    // These are borrowed PyTorch allocations. `leak` below only keeps cudarc
    // from freeing them; PyTorch retains their ownership.
    let rows = unsafe { gpu.torch.upgrade_device_ptr::<u8>(rows, n * ROW_BYTES) };
    let cards = unsafe { gpu.torch.upgrade_device_ptr::<f32>(cards, N_UNITS * CARD_FEATS) };
    let locations = unsafe { gpu.torch.upgrade_device_ptr::<u8>(locations, N_HEXES) };
    let mut out = unsafe { gpu.torch.upgrade_device_ptr::<f32>(out, n * PUBFEAT) };
    let n_i = n as i32;
    unsafe {
        gpu.torch
            .launch_builder(&gpu.k.expand_rows)
            .arg(&rows)
            .arg(&cards)
            .arg(&locations)
            .arg(&mut out)
            .arg(&n_i)
            .launch_unit(LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(err)?;
    }
    rows.leak();
    cards.leak();
    locations.leak();
    out.leak();

    let after = gpu.torch.record_event(None).map_err(err)?;
    unsafe {
        result::stream::wait_event(
            stream as sys::CUstream,
            after.cu_event(),
            sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )
        .map_err(err)
    }
}

impl Gpu {
    fn get(ordinal: usize) -> Res<Arc<Gpu>> {
        let mut gpus = GPUS.lock();
        if let Some(gpu) = gpus.get(&ordinal) {
            gpu.ctx.bind_to_thread().map_err(err)?;
            return Ok(Arc::clone(gpu));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("device {ordinal}: {e:?}"))?;
        // Stream ordering is explicit, so the read/write events cudarc would
        // otherwise create on every allocation buy nothing.
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let ptx = compile_ptx_with_opts(
            KERNELS,
            compile_options(major, minor),
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(err)?;
        let k = Kernels::load(&module)?;
        // A block of `k_trunk` holds three boards and asks for more than the
        // forty-eight kilobytes a kernel gets without saying so out loud.
        k.trunk
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                TRUNK_SHARED as i32,
            )
            .map_err(err)?;
        let torch = ctx.new_stream().map_err(err)?;
        let gpu = Arc::new(Gpu { ctx, k: Arc::new(k), torch });
        gpus.insert(ordinal, Arc::clone(&gpu));
        Ok(gpu)
    }
}

impl Card {
    fn carve_bytes(n: usize, cfg: &Cfg) -> u64 {
        RoundCap::of(n, &cfg.budget, cfg.s).bytes() as u64
    }

    fn on(gpu: &Gpu, net: &Net) -> Res<Card> {
        let stream = gpu.ctx.new_stream().map_err(err)?;
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let flat = net.flat();
        let nb: Vec<i32> = board()
            .neighbors
            .iter()
            .flatten()
            .map(|&n| if n == NONE { -1 } else { n as i32 })
            .collect();
        let mut card_facts = vec![0.0; N_UNITS * CARD_FEATS];
        for u in 0..N_UNITS {
            write_card_features(
                u as u8,
                &mut card_facts[u * CARD_FEATS..(u + 1) * CARD_FEATS],
            );
        }
        let locations: Vec<u8> = board().is_location.iter().map(|&x| x as u8).collect();
        let layout = NetLayout::new();
        let (fragwise, lanes) = fragwise(&layout, &flat.w);
        let mut plan: Vec<i32> = Vec::new();
        for (i, blk) in layout.blocks.iter().enumerate() {
            let (n0, n1) = (layout.norms[ln_block(i, 0)], layout.norms[ln_block(i, 1)]);
            plan.extend([blk.mix.w, blk.mix.b, blk.pool.w, blk.pool.b, blk.out.w,
                         blk.out.b, n0.g, n0.b, n1.g, n1.b].map(|x| x as i32));
            plan.extend([lanes[2 * i], lanes[2 * i + 1]].map(|x| x as i32));
        }
        let t = layout.norms[LN_TRUNK];
        plan.extend([t.g as i32, t.b as i32]);
        let owed = owed_by_the_join(&layout, &flat.b);
        // `k_join` reads the join's weights and norms as one run each, so the
        // fused kernel takes three slices rather than thirteen. `NetLayout`
        // lays them down back to back; this is where that is a requirement and
        // not a coincidence.
        let mut at = layout.join_b.w + JOIN_IN * JW;
        for span in layout.join_w {
            assert_eq!(span.w, at, "the join's weights are not one run");
            at += JW * JW;
        }
        assert_eq!(layout.join_out.w, at, "the join's weights are not one run");
        for i in 0..=JBLOCKS {
            let n = layout.norms[LN_JOIN + i];
            assert_eq!(n.g, layout.norms[LN_JOIN].g + 2 * i * JW, "the join's norms are not one run");
            assert_eq!(n.b, n.g + JW, "a join norm's shift does not follow its scale");
        }
        Ok(Card {
            plan: stream.memcpy_stod(&plan).map_err(err)?,
            owed: stream.memcpy_stod(&owed).map_err(err)?,
            w: stream.memcpy_stod(&flat.w).map_err(err)?,
            wt: stream.memcpy_stod(&fragwise).map_err(err)?,
            jw: stream.memcpy_stod(&join_pack(&layout, &flat.w)).map_err(err)?,
            b: stream.memcpy_stod(&flat.b).map_err(err)?,
            ln: stream.memcpy_stod(&flat.ln).map_err(err)?,
            nb: stream.memcpy_stod(&nb).map_err(err)?,
            card_facts: stream.memcpy_stod(&card_facts).map_err(err)?,
            locations: stream.memcpy_stod(&locations).map_err(err)?,
            stream,
            blas,
            k: Arc::clone(&gpu.k),
            solves: Arc::new(parking_lot::Mutex::new(Vec::new())),
            host: parking_lot::Mutex::new(Stage::default()),
            pack: parking_lot::Mutex::new(Pack::default()),
            down: parking_lot::Mutex::new(Host::default()),
            down_f: parking_lot::Mutex::new(Host::default()),
            batch: parking_lot::Mutex::new(Batch::default()),
            scratch: parking_lot::Mutex::new(Scratch::default()),
            layout,
        })
    }

    /// Allocate this card's round buffers, once, at `RoundCap`. Slots live on
    /// the GPU's shared pool.
    fn carve(&mut self, n: usize, cfg: &Cfg) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        self.stream.context().bind_to_thread().map_err(err)?;
        let cap = RoundCap::of(n, &cfg.budget, cfg.s);
        let s = &self.stream;
        let mut scratch = Scratch::default();
        scratch.mass = Arr::with_cap(s, cap.mass)?;
        scratch.pooled = Arr::with_cap(s, cap.pooled)?;
        scratch.h = Arr::with_cap(s, cap.h)?;
        scratch.z = Arr::with_cap(s, cap.z)?;
        scratch.input = Arr::with_cap(s, cap.input)?;
        scratch.leaves = Arr::with_cap(s, cap.leaves)?;
        scratch.piles = Arr::with_cap(s, cap.piles)?;
        scratch.tokens = Arr::with_cap(s, cap.tokens)?;
        scratch.projected = Arr::with_cap(s, cap.projected)?;
        scratch.type_pool = Arr::with_cap(s, cap.type_pool)?;
        scratch.loose = Arr::with_cap(s, cap.loose)?;
        scratch.glob = Arr::with_cap(s, cap.glob)?;
        scratch.facts = Arr::with_cap(s, cap.facts)?;
        scratch.occupant = Arr::with_cap(s, cap.occupant)?;
        scratch.x = Arr::with_cap(s, cap.x)?;
        scratch.bag = Arr::with_cap(s, cap.bag)?;
        let mut stage = Stage::default();
        stage.packed = Wire::with_cap(s, cap.packed)?;
        stage.xpub = Arr::with_cap(s, cap.xpub)?;
        stage.cards = Wire::with_cap(s, cap.cards)?;
        stage.card_of_row = Wire::with_cap(s, cap.card_of_row)?;
        stage.phi = Wire::with_cap(s, cap.phi)?;
        stage.owner = Wire::with_cap(s, cap.owner)?;
        stage.cfg_cards = Wire::with_cap(s, cap.cfg_cards)?;
        stage.blob = Wire::with_cap(s, cap.blob)?;
        stage.at = Wire::with_cap(s, cap.at)?;
        stage.src = Wire::with_cap(s, cap.src)?;
        stage.dst = Wire::with_cap(s, cap.dst)?;
        stage.start = Wire::with_cap(s, cap.start)?;
        let mut batch = Batch::default();
        batch.trees = Wire::with_cap(s, cap.trees)?;
        batch.work = Wire::with_cap(s, cap.work)?;
        batch.coff = Wire::with_cap(s, cap.coff)?;
        batch.part = Wire::with_cap(s, cap.part)?;
        batch.local = Wire::with_cap(s, cap.local)?;
        batch.base = Wire::with_cap(s, cap.base)?;
        batch.prime = Wire::with_cap(s, cap.prime)?;
        batch.touched = Wire::with_cap(s, cap.touched)?;
        *self.host.lock() = stage;
        *self.scratch.lock() = scratch;
        *self.batch.lock() = batch;
        Ok(())
    }

    fn round(&self, calls: &[Call], mine: &[usize]) -> Res<Vec<(usize, Reply)>> {
        self.stream.context().bind_to_thread().map_err(err)?;
        let mut slots: Vec<usize> = mine.iter().map(|&i| calls[i].solve()).collect();
        slots.sort_unstable();
        slots.dedup();
        {
            let solves = self.solves.lock();
            for &slot in &slots {
                let solve = solves.get(slot).ok_or_else(|| {
                    format!("round names solve slot {slot}, but only {} were carved", solves.len())
                })?;
                self.stream.wait(&solve.ready).map_err(err)?;
            }
        }
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
        // These three are always timed. They are host work as much as device
        // work -- the trunk marshals a batch, the tree copies a description --
        // and a stage nobody times is where the round's time turns out to be.
        // One pack for the round: every write it makes to a solve's arrays,
        // wherever it is planned, travels as one buffer and one kernel.
        let mut pack = self.pack.lock();
        pack.clear();
        self.wall(11, || self.trunk(calls, &pick(0), &mut pack)).map_err(at("trunk"))?;
        self.wall(12, || self.configs(calls, &pick(1))).map_err(at("configs"))?;
        self.wall(17, || self.tree(calls, &pick(2), &mut pack)).map_err(at("tree"))?;
        self.wall(18, || self.scatter(&mut pack)).map_err(at("scatter"))?;
        drop(pack);
        // After the scatter, which lays the uniform prior down over the cells
        // this growth appended, and before the iteration, whose expansion
        // phase reads what this writes.
        self.wall(16, || self.priors(calls, &pick(2))).map_err(at("priors"))?;
        self.iterate(calls, &pick(3), &mut out).map_err(at("iterate"))?;
        self.read(calls, &pick(4), &mut out).map_err(at("read"))?;
        {
            let solves = self.solves.lock();
            for &slot in &slots {
                solves[slot].ready.record(&self.stream).map_err(err)?;
            }
        }
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
        let (rows_i, width) = (rows as i32, s.o as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.bias)
                .arg(out).arg(&bias).arg(&rows_i).arg(&width)
                .launch_unit(rows_of(rows, s.o))
        }
        .map_err(err)
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

    /// `Norm::apply` when `act`, `Norm::plain` when not, in place. The join's
    /// own four norms are inside `k_join`; this serves everything else.
    fn norm(&self, s: NormSpan, rows: usize, act: bool, x: &mut CudaSlice<f32>) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let g = self.ln.slice(s.g..s.g + s.width);
        let b = self.ln.slice(s.b..s.b + s.width);
        let (rows_i, width, act) = (rows as i32, s.width as i32, act as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.norm_ip)
                .arg(x).arg(&g).arg(&b)
                .arg(&rows_i).arg(&width).arg(&act)
                .launch_unit(warp_rows(rows))
        }
        .map_err(err)
    }


    /// The slot the farm pinned this solve to. It was allocated at carve.
    fn slot<'g>(&self, g: &'g mut Vec<Solve>, solve: usize) -> &'g mut Solve {
        let n = g.len();
        g.get_mut(solve).unwrap_or_else(|| panic!("solve {solve} pinned to a card that holds {n} slots"))
    }

    // ----------------------------------------------------------------- trunk

    /// Every new leaf in the round: the board vector and the join cache.
    fn trunk(&self, calls: &[Call], mine: &[usize], pack: &mut Pack) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let each = |i: usize| -> (&[u8], &[f32], usize) {
            let Call::Trunk { packed, cards, boards, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            assert_eq!(packed.len(), boards * ROW_BYTES, "trunk input is not one packed row a board");
            assert_eq!(cards.len(), CARD_ROWS * NTYPE * TYPE, "trunk card table");
            (packed, cards, *boards)
        };
        let rows: usize = mine.iter().map(|&i| each(i).2).sum();
        {
            let mut g = self.solves.lock();
            for &i in mine {
                let Call::Trunk { solve, at: row0, .. } = &calls[i] else {
                    unreachable!("trunk shard holds only trunk calls")
                };
                if *row0 == 0 {
                    self.slot(&mut g, *solve).rewind_leaf();
                }
            }
        }
        let mark = std::time::Instant::now();
        let s = &self.stream;
        {
            let mut stage = self.host.lock();
            stage.cards.put(s, mine.len() * CARD_ROWS * NTYPE * TYPE, |dst| {
                let mut at = 0;
                for &i in mine {
                    let (_, cd, _) = each(i);
                    dst[at..at + cd.len()].copy_from_slice(cd);
                    at += cd.len();
                }
                at
            })?;
        }
        let mut board0 = 0usize;
        while board0 < rows {
            let n = TILE.min(rows - board0);
            {
                let mut stage = self.host.lock();
                stage.packed.put(s, n * ROW_BYTES, |dst| {
                    let (mut skip, mut wrote) = (board0, 0);
                    for &i in mine {
                        let (packed, _, boards) = each(i);
                        if skip >= boards {
                            skip -= boards;
                            continue;
                        }
                        let take = (boards - skip).min(n - wrote);
                        let a = skip * ROW_BYTES;
                        dst[wrote * ROW_BYTES..(wrote + take) * ROW_BYTES]
                            .copy_from_slice(&packed[a..a + take * ROW_BYTES]);
                        wrote += take;
                        skip = 0;
                        if wrote == n {
                            break;
                        }
                    }
                    wrote * ROW_BYTES
                })?;
                stage.card_of_row.put(s, n, |dst| {
                    let (mut skip, mut wrote, mut card) = (board0, 0, 0i32);
                    for &i in mine {
                        let boards = each(i).2;
                        if skip >= boards {
                            skip -= boards;
                            card += CARD_ROWS as i32;
                            continue;
                        }
                        let take = (boards - skip).min(n - wrote);
                        dst[wrote..wrote + take].fill(card);
                        wrote += take;
                        skip = 0;
                        card += CARD_ROWS as i32;
                        if wrote == n {
                            break;
                        }
                    }
                    wrote
                })?;
                let rows_i = n as i32;
                let Stage { packed, xpub, .. } = &mut *stage;
                let packed = packed.dev.buf.as_ref().expect("staged");
                let xpub = xpub.buf.as_mut().expect("carved");
                unsafe {
                    self.stream
                        .launch_builder(&self.k.expand_rows)
                        .arg(packed)
                        .arg(&self.card_facts)
                        .arg(&self.locations)
                        .arg(xpub)
                        .arg(&rows_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: (n as u32, 1, 1),
                            block_dim: (THREADS, 1, 1),
                            shared_mem_bytes: 0,
                        })
                }
                .map_err(err)?;
            }
            self.trunk_tile(calls, mine, board0, n)?;
            board0 += n;
        }
        LEAF_NS[14].fetch_add(mark.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        self.keep(calls, mine, pack)
    }

    /// Encode `n` boards already expanded at the head of `xpub` / `card_of_row`.
    fn trunk_tile(&self, calls: &[Call], mine: &[usize], board0: usize, n: usize) -> Res<()> {
        let s = &self.stream;
        let stage = self.host.lock();
        let xpub = stage.xpub.buf.as_ref().expect("carved");
        let cards = stage.cards.dev.buf.as_ref().expect("staged");
        let card_of_row = stage.card_of_row.dev.buf.as_ref().expect("staged");
        let cells = n * N_HEXES;
        let stride = PUBFEAT as i32;
        let (rows_i, cells_i) = (n as i32, cells as i32);
        let (nhex, ntype, chan, nslot) = (N_HEXES as i32, NTYPE as i32, C as i32, NSLOT as i32);
        let l = &self.layout;
        let mut sc = self.scratch.lock();
        sc.piles.room(n * NTYPE * PILE_COUNTS)?;
        sc.tokens.room(n * NTYPE * TYPE)?;
        sc.projected.room(n * NTYPE * C)?;
        sc.type_pool.room(n * C)?;
        sc.loose.room(n * LOOSE)?;
        sc.glob.room(n * C)?;
        sc.facts.room(cells * HEX_FACTS)?;
        sc.occupant.room(cells)?;
        sc.x.room(cells * C)?;
        sc.input.room(n * (2 * C + LOOSE))?;
        sc.h.room(n * D)?;
        sc.z.room(n * JW)?;
        let Scratch {
            piles, tokens, projected, type_pool, loose, glob, facts, occupant, x, input, h, z, ..
        } = &mut *sc;
        let piles = piles.buf.as_mut().unwrap();
        let tokens = tokens.buf.as_mut().unwrap();
        let projected = projected.buf.as_mut().unwrap();
        let type_pool = type_pool.buf.as_mut().unwrap();
        let loose = loose.buf.as_mut().unwrap();
        let glob = glob.buf.as_mut().unwrap();
        let facts = facts.buf.as_mut().unwrap();
        let occupant = occupant.buf.as_mut().unwrap();
        let x = x.buf.as_mut().unwrap();
        let input = input.buf.as_mut().unwrap();
        let p = h.buf.as_mut().unwrap();
        let jp = z.buf.as_mut().unwrap();

        let (off, width) = (OFF_PILES as i32, (NTYPE * PILE_COUNTS) as i32);
        launch!(self, window, n * NTYPE * PILE_COUNTS, xpub, &mut *piles, &rows_i, &stride, &off, &width)?;
        self.lin(l.pile, piles, n * NTYPE, 0.0, &mut *tokens)?;
        let seat = self.w.slice(l.seat..l.seat + 2 * TYPE);
        let type_i = TYPE as i32;
        launch!(self, tokens, n * NTYPE * TYPE, cards, card_of_row, &seat, &mut *tokens, &rows_i, &ntype, &type_i, &nslot)?;
        self.run(l.tok_stem, tokens, n * NTYPE, &mut *projected)?;
        launch!(self, type_pool, n * C, &mut *projected, &mut *type_pool, &rows_i, &ntype, &chan)?;
        let (off, width) = (OFF_LOOSE as i32, LOOSE as i32);
        launch!(self, window, n * LOOSE, xpub, &mut *loose, &rows_i, &stride, &off, &width)?;
        self.run(l.glob_stem, loose, n, &mut *glob)?;
        let (hex_ch, hex_facts) = (HEX_CH as i32, HEX_FACTS as i32);
        launch!(self, hex_facts, cells, xpub, &mut *facts, &mut *occupant, &rows_i, &stride, &nhex, &hex_ch, &hex_facts, &ntype)?;
        self.run(l.hex_stem, facts, cells, &mut *x)?;
        let pos = self.w.slice(l.pos..l.pos + N_HEXES * C);
        launch!(self, stem, cells * C, &mut *x, &*projected, &*occupant, &pos, &*glob, &*type_pool, &cells_i, &nhex, &ntype, &chan)?;
        let (off, loose_i, blocks_i) = (OFF_LOOSE as i32, LOOSE as i32, BLOCKS as i32);
        // One warp to each eight-channel output tile of the multiply, twelve
        // of them; read the other way round that is a warp to a hex and `C /
        // 32` channels to a lane, so a hex's row is exactly one warp wide and
        // its LayerNorm is a shuffle rather than a barrier.
        const SLOTS: u32 = (C / 8) as u32;
        assert_eq!(C % 32, 0, "k_trunk wants a whole number of warps a row");
        assert_eq!(TRUNK_ROWS / SLOTS as usize, 4, "k_trunk holds four hexes a thread");
        unsafe {
            self.stream
                .launch_builder(&self.k.trunk)
                .arg(&*x).arg(&self.nb).arg(&self.w).arg(&self.wt)
                .arg(&self.b).arg(&self.ln)
                .arg(&self.plan).arg(xpub).arg(&mut *input)
                .arg(&rows_i).arg(&nhex).arg(&chan).arg(&blocks_i)
                .arg(&stride).arg(&off).arg(&loose_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (32, SLOTS, 1),
                    shared_mem_bytes: TRUNK_SHARED as u32,
                })
        }
        .map_err(err)?;
        self.run(l.board_out, input, n, &mut *p)?;
        self.run(l.join_p, p, n, &mut *jp)?;
        let mut skip = board0;
        let mut src = 0;
        let mut g = self.solves.lock();
        for &i in mine {
            let Call::Trunk { solve, boards_at, boards: nb, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            if skip >= *nb {
                skip -= *nb;
                continue;
            }
            let take = (*nb - skip).min(n - src);
            let b = self.slot(&mut g, *solve);
            b.copy_board(s, *boards_at + skip, p, jp, src, take)?;
            src += take;
            skip = 0;
            if src == n {
                break;
            }
        }
        Ok(())
    }

    /// Keep the trunk's metadata, per solve. Board vectors were written by
    /// `trunk_tile` before this runs.
    fn keep(
        &self,
        calls: &[Call],
        mine: &[usize],
        pack: &mut Pack,
    ) -> Res<()> {
        let mut g = self.solves.lock();
        for &i in mine {
            let Call::Trunk {
                solve, at: row0, queries: nrows, board_of, cidx, coff, ..
            } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            let nrows = *nrows;
            let b = self.slot(&mut g, *solve);
            b.reserve(Ent::Row, row0 + nrows)?;
            let words = pack.words(board_of);
            let dst = b.ent[Ent::Row as usize].field(Y_BOARD_OF, &self.stream);
            pack.piece(dst, *row0 as u32, words, nrows as u32);
            // `coff` arrives relative to this call's own `cidx`, so it is
            // shifted onto the resident index before it is stored. Row zero
            // writes the leading zero; every later call overwrites it with its
            // own first offset, the same number.
            let base = b.cells as u32;
            let shifted: Vec<u32> = coff.iter().map(|x| x + base).collect();
            b.host_coff.extend(shifted.iter().skip(1));
            let words = pack.words(cidx);
            let dst = b.view(&self.stream, Ent::Cidx, 0, b.cells, cidx.len(), 1)?;
            pack.piece(dst, b.cells as u32, words, cidx.len() as u32);
            let words = pack.words(&shifted);
            let dst = b.ent[Ent::Row as usize].field(Y_COFF, &self.stream);
            pack.piece(dst, 2 * *row0 as u32, words, shifted.len() as u32);
            b.cells += cidx.len();
            b.rows = row0 + nrows;
        }
        Ok(())
    }

    // --------------------------------------------------------------- configs

    /// `f(c)` for the readout and `g(c)` for the pooling, for every config the
    /// round asked about.
    fn configs(&self, calls: &[Call], mine: &[usize]) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let each = |i: usize| -> (&[f32], &[u32], &[f32], usize) {
            let Call::Configs { phi, owner, cards, n, .. } = &calls[i] else {
                unreachable!("config shard holds only config calls")
            };
            assert_eq!(phi.len(), n * CFEAT, "config phi is not one row a config");
            assert_eq!(owner.len(), *n, "config owner is not one entry a config");
            (phi, owner, cards, *n)
        };
        let n: usize = mine.iter().map(|&i| each(i).3).sum();
        let s = &self.stream;
        let l = &self.layout;
        {
            let mut stage = self.host.lock();
            stage.cfg_cards.put(s, mine.len() * CARD_ROWS * NTYPE * TYPE, |dst| {
                let mut at = 0;
                for &i in mine {
                    let (_, _, cd, _) = each(i);
                    dst[at..at + cd.len()].copy_from_slice(cd);
                    at += cd.len();
                }
                at
            })?;
            let views = stage.cfg_cards.host.len / (NTYPE * TYPE);
            let cards = stage.cfg_cards.dev.buf.as_ref().expect("staged");
            let mut sc = self.scratch.lock();
            sc.bag.room(views * NTYPE * 3 * POOL)?;
            self.run(l.cfg_m, cards, views * NTYPE, sc.bag.buf.as_mut().unwrap())?;
        }
        let mut cfg0 = 0usize;
        while cfg0 < n {
            let k = TILE.min(n - cfg0);
            {
                let mut stage = self.host.lock();
                stage.phi.put(s, k * CFEAT, |dst| {
                    let (mut skip, mut wrote) = (cfg0, 0);
                    for &i in mine {
                        let (ph, _, _, kn) = each(i);
                        if skip >= kn {
                            skip -= kn;
                            continue;
                        }
                        let take = (kn - skip).min(k - wrote);
                        let a = skip * CFEAT;
                        dst[wrote * CFEAT..(wrote + take) * CFEAT]
                            .copy_from_slice(&ph[a..a + take * CFEAT]);
                        wrote += take;
                        skip = 0;
                        if wrote == k {
                            break;
                        }
                    }
                    wrote * CFEAT
                })?;
                stage.owner.put(s, k, |dst| {
                    let (mut skip, mut wrote, mut base) = (cfg0, 0, 0u32);
                    for &i in mine {
                        let (_, ow, cd, kn) = each(i);
                        if skip >= kn {
                            skip -= kn;
                            base += (cd.len() / (NTYPE * TYPE)) as u32;
                            continue;
                        }
                        let take = (kn - skip).min(k - wrote);
                        for (d, &q) in dst[wrote..wrote + take].iter_mut().zip(&ow[skip..skip + take]) {
                            *d = q + base;
                        }
                        wrote += take;
                        skip = 0;
                        base += (cd.len() / (NTYPE * TYPE)) as u32;
                        if wrote == k {
                            break;
                        }
                    }
                    wrote
                })?;
            }
            self.config_tile(calls, mine, cfg0, k)?;
            cfg0 += k;
        }
        Ok(())
    }

    fn config_tile(&self, calls: &[Call], mine: &[usize], cfg0: usize, k: usize) -> Res<()> {
        let s = &self.stream;
        let stage = self.host.lock();
        let phi = stage.phi.dev.buf.as_ref().expect("staged");
        let owner = stage.owner.dev.buf.as_ref().expect("staged");
        let cards = stage.cfg_cards.dev.buf.as_ref().expect("staged");
        let l = &self.layout;
        let (n_i, nslot, cfeat) = (k as i32, NSLOT as i32, CFEAT as i32);
        let (ntype, type_i, pool_i) = (NTYPE as i32, TYPE as i32, POOL as i32);
        let width = 3 + TYPE;
        let mut sc = self.scratch.lock();
        sc.tokens.room(k * NSLOT * width)?;
        sc.projected.room(k * NSLOT * CFGH)?;
        sc.facts.room(k * CFGH)?;
        sc.h.room(k * D)?;
        sc.pooled.room(k * POOL)?;
        sc.z.room(k * D)?;
        let Scratch { tokens, projected, facts, h, pooled, z, bag, .. } = &mut *sc;
        let slots = tokens.buf.as_mut().unwrap();
        let hidden = projected.buf.as_mut().unwrap();
        let u = facts.buf.as_mut().unwrap();
        let f = h.buf.as_mut().unwrap();
        let g = pooled.buf.as_mut().unwrap();
        let fp = z.buf.as_mut().unwrap();
        let bag = bag.buf.as_mut().unwrap();
        launch!(self, cfg_slots, k * NSLOT * width, phi, owner, cards, &mut *slots, &n_i, &nslot, &cfeat, &ntype, &type_i)?;
        self.run(l.cfg1, slots, k * NSLOT, &mut *hidden)?;
        let hid = (k * NSLOT * CFGH) as i32;
        launch!(self, gelu, k * NSLOT * CFGH, &mut *hidden, &hid)?;
        let cfgh = CFGH as i32;
        launch!(self, sum_slots, k * CFGH, hidden, &mut *u, &n_i, &nslot, &cfgh)?;
        self.norm(l.norms[LN_CFG], k, true, &mut *u)?;
        self.run(l.cfg_f, u, k, &mut *f)?;
        self.run(l.cfg_g, u, k, &mut *g)?;
        self.run(l.cfg_p, u, k, &mut *fp)?;
        launch!(self, bag, k * POOL, bag, phi, owner, &mut *g, &n_i, &nslot, &ntype, &cfeat, &pool_i)?;
        let mut skip = cfg0;
        let mut src = 0;
        let mut solves = self.solves.lock();
        for &i in mine {
            let kn = calls[i].rows();
            let Call::Configs { solve, at: base, .. } = &calls[i] else {
                unreachable!("config shard holds only config calls")
            };
            if skip >= kn {
                skip -= kn;
                continue;
            }
            let take = (kn - skip).min(k - src);
            let b = self.slot(&mut solves, *solve);
            b.copy_cfg(s, *base + skip, f, g, fp, src, take)?;
            src += take;
            skip = 0;
            if src == k {
                break;
            }
        }
        Ok(())
    }

    // ----------------------------------------------------------- the prior

    /// Fill the policy prior of every node this round primed.
    ///
    /// The action encoder and the policy readout, against arrays the card
    /// already holds: the node's own board vector, the config rows `f_p`, and
    /// the tree's cells. `Solver::refresh_priors` used to run this on the host,
    /// and the round downloaded a board vector per fresh leaf and an `f_p` row
    /// per fresh config so that it could -- a quarter of a megabyte a solve a
    /// round, for a handful of nodes.
    ///
    /// What the card does not hold is what an action *is* and which action each
    /// strategy cell stands for. Both ride in the tree call, and both are a few
    /// kilobytes a node.
    fn priors(&self, calls: &[Call], mine: &[usize]) -> Res<()> {
        let each = |i: usize| -> (&[Prime], &[u32], &[u32], f32) {
            let Call::Tree { prime, acts, cells, prior_temp, .. } = &calls[i] else {
                unreachable!("tree shard holds only tree calls")
            };
            (prime, acts, cells, *prior_temp)
        };
        let solves: Vec<usize> = mine
            .iter()
            .copied()
            .filter(|&i| !each(i).0.is_empty())
            .map(|i| calls[i].solve())
            .collect();
        if solves.is_empty() {
            return Ok(());
        }
        // One upload: the six per-node arrays, then the action's own node, then
        // the two pools. Floats travel as their bits, as everything else a
        // round scatters does.
        let (mut part, mut node, mut row) = (Vec::new(), Vec::new(), Vec::new());
        let (mut act_at, mut cell_at, mut inv_t) = (Vec::new(), Vec::new(), Vec::new());
        let (mut act_node, mut desc, mut cells) = (Vec::new(), Vec::new(), Vec::new());
        let (mut nas, mut ncs) = (Vec::new(), Vec::new());
        for (p, &i) in mine.iter().filter(|&&i| !each(i).0.is_empty()).enumerate() {
            let (prime, a, c, temp) = each(i);
            for q in prime {
                act_node.extend(std::iter::repeat(node.len() as u32).take(q.na as usize));
                part.push(p as u32);
                node.push(q.node);
                row.push(q.row);
                act_at.push(desc.len() as u32 / 5 + q.at);
                cell_at.push(cells.len() as u32 + q.cell_at);
                inv_t.push((1.0f32 / temp.max(1e-6)).to_bits());
                nas.push(q.na as usize);
                ncs.push(q.nc);
            }
            desc.extend_from_slice(a);
            cells.extend_from_slice(c);
        }
        let m = node.len();
        self.lay(&solves)?;
        let mut batch = self.batch.lock();
        let mut i = 0usize;
        while i < m {
            let mut j = i;
            let mut na_c = 0usize;
            let mut wide = 0u32;
            while j < m && (j - i) < TILE && na_c + nas[j] <= TILE {
                na_c += nas[j];
                wide = wide.max(ncs[j]);
                j += 1;
            }
            if j == i {
                na_c = nas[j];
                wide = ncs[j];
                j = i + 1;
            }
            let act0 = act_at[i] as usize;
            let cell0 = cell_at[i] as usize;
            let cell1 = if j < m { cell_at[j] as usize } else { cells.len() };
            let act_at_r: Vec<u32> = act_at[i..j].iter().map(|x| x - act_at[i]).collect();
            let cell_at_r: Vec<u32> = cell_at[i..j].iter().map(|x| x - cell_at[i]).collect();
            let act_node_r: Vec<u32> = act_node[act0..act0 + na_c].iter().map(|x| x - i as u32).collect();
            let mc = j - i;
            let flat: Vec<u32> = [
                &part[i..j],
                &node[i..j],
                &row[i..j],
                &act_at_r,
                &cell_at_r,
                &inv_t[i..j],
                &act_node_r,
                &desc[act0 * 5..(act0 + na_c) * 5],
                &cells[cell0..cell1],
            ]
            .concat();
            self.prior_tile(&mut batch, &flat, mc, na_c, cell1 - cell0, wide)?;
            i = j;
        }
        Ok(())
    }

    fn prior_tile(
        &self,
        batch: &mut Batch,
        flat: &[u32],
        m: usize,
        na: usize,
        ncells: usize,
        widest: u32,
    ) -> Res<()> {
        let s = &self.stream;
        batch.prime.put(s, flat.len(), copy(flat))?;
        let dev = batch.prime.buf();
        let at = |k: usize, n: usize| dev.slice(k..k + n);
        let (part_d, node_d, row_d) = (at(0, m), at(m, m), at(2 * m, m));
        let (act_at_d, cell_at_d, inv_d) = (at(3 * m, m), at(4 * m, m), at(5 * m, m));
        let act_node_d = at(6 * m, na);
        let desc_d = at(6 * m + na, 5 * na);
        let cells_d = at(6 * m + 6 * na, ncells);
        let l = &self.layout;
        let (na_i, m_i, d_i, aw_i) = (na as i32, m as i32, D as i32, AW as i32);
        let (nkinds, nslot, nhex, afeat) = (
            crate::actions::N_KINDS as i32,
            NSLOT as i32,
            N_HEXES as i32,
            AFEAT as i32,
        );
        let mut sc = self.scratch.lock();
        sc.x.room(na * AFEAT)?;
        sc.tokens.room(na * AW)?;
        sc.h.room((m * D).max(na * D))?;
        sc.projected.room(m * AW)?;
        let Scratch { x, tokens, h, projected, .. } = &mut *sc;
        let feat = x.buf.as_mut().unwrap();
        let z = tokens.buf.as_mut().unwrap();
        let hbuf = h.buf.as_mut().unwrap();
        let proj = projected.buf.as_mut().unwrap();
        launch!(self, act_feats, na * AFEAT, &desc_d, &mut *feat, &na_i, &nkinds, &nslot, &nhex, &afeat)?;
        self.run(l.act_in, feat, na, &mut *z)?;
        launch!(self, act_boards, m * D, batch.trees.buf(), &part_d, &row_d, &mut *hbuf, &m_i, &d_i)?;
        self.run(l.act_board, hbuf, m, &mut *proj)?;
        launch!(self, act_add, na * AW, &mut *z, proj, &act_node_d, &na_i, &aw_i)?;
        self.norm(l.norms[LN_ACT], na, true, &mut *z)?;
        // `e` reuses `h` after the board rows are done.
        self.run(l.act_out, z, na, &mut *hbuf)?;
        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (widest.div_ceil(WARPS).max(1), m as u32, 1),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.k.prior)
                .arg(batch.trees.buf()).arg(&part_d).arg(&node_d).arg(&row_d)
                .arg(&act_at_d).arg(&cell_at_d).arg(&cells_d).arg(&*hbuf)
                .arg(&inv_d).arg(&m_i).arg(&d_i)
                .launch_unit(cfg)
        }
        .map_err(err)
    }

    // ------------------------------------------------------------ the CFR loop

    /// Bring each solve's tree, arenas and priors up to date with the host.
    ///
    /// Growth is the only thing the host still does inside a solve: it holds
    /// the game rules, so it turns the sampled leaves into decision nodes and
    /// describes them. Everything the description feeds stays here.
    fn tree(&self, calls: &[Call], mine: &[usize], pack: &mut Pack) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let mut g = self.solves.lock();
        for &i in mine {
            let Call::Tree { solve, writes, fresh, ncells, nreach, nvals, levels, nterm, seed, .. }
                = &calls[i] else {
                unreachable!("tree shard holds only tree calls")
            };
            let b = self.slot(&mut g, *solve);
            let s = &self.stream;
            if *fresh {
                b.rewind_cfr(s)?;
            }
            // One copy of the solve's words, then a piece per run saying where
            // inside them each destination reads.
            let base = pack.words(&writes.blob);
            for r in &writes.runs {
                let dst = b.plan(s, r.dst, r.at as usize, r.len as usize)?;
                pack.piece(dst, r.at, base + r.start, r.len);
            }
            b.level_start.clear();
            b.level_start.extend_from_slice(levels);
            b.nterm = *nterm;
            if let Some(sd) = seed {
                b.seed.put(s, 0, &[*sd])?;
            }
            b.ncells = *ncells;
            b.nreach = *nreach;
            b.reserve(Ent::Cell, *ncells)?;
            b.nvals = *nvals;
            b.reserve(Ent::Reach, (*nreach).max(*nvals))?;
        }
        Ok(())
    }

    /// Send a round's writes: one buffer up, then the pieces a tile at a time.
    fn scatter(&self, pack: &mut Pack) -> Res<()> {
        if pack.moved == 0 {
            return Ok(());
        }
        let moved = pack.moved;
        pack.sum.push(moved);
        LEAF_NS[19].fetch_add(4 * moved as u64, std::sync::atomic::Ordering::Relaxed);
        let s = &self.stream;
        let mut stage = self.host.lock();
        stage.blob.put(s, pack.blob.len(), copy(&pack.blob))?;
        let np = pack.dst.len();
        let mut i = 0usize;
        while i < np {
            let k = TILE.min(np - i);
            let word0 = pack.sum[i];
            let tile_moved = pack.sum[i + k] - word0;
            let tile_sum: Vec<u32> = pack.sum[i..i + k + 1].iter().map(|x| x - word0).collect();
            stage.at.put(s, k, copy(&pack.at[i..i + k]))?;
            stage.src.put(s, k, copy(&pack.src[i..i + k]))?;
            stage.start.put(s, k + 1, copy(&tile_sum))?;
            stage.dst.put(s, k, copy(&pack.dst[i..i + k]))?;
            let Stage { blob, dst, at, src, start, .. } = &*stage;
            let blob = blob.dev.buf.as_ref().expect("staged");
            let dst = dst.dev.buf.as_ref().expect("staged");
            let at = at.dev.buf.as_ref().expect("staged");
            let src = src.dev.buf.as_ref().expect("staged");
            let start = start.dev.buf.as_ref().expect("staged");
            let (pieces, total_i) = (k as i32, tile_moved as i32);
            launch!(self, scatter, tile_moved as usize, blob, dst, at, src, start, &pieces, &total_i)?;
            i += k;
        }
        Ok(())
    }

    /// Lay a set of solves out as one batch. Fills the card's arena; does not allocate.
    fn lay(&self, solves: &[usize]) -> Res<()> {
        let mut batch = self.batch.lock();
        // These are gathered into ordinary vectors first because the shape of
        // the batch is not known until every solve has been read. They are
        // small -- a descriptor apiece and an entry per leaf row.
        let (mut desc, mut coff): (Vec<u64>, Vec<u32>) = (Vec::new(), vec![0]);
        let (mut part_of_row, mut local_row, mut base): (Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut rows, mut cells, mut nterm) = (0usize, 0u32, 0usize);
        // One work item per (solve, node), bucketed by level, so a level's
        // launch is exactly as many blocks as it has nodes. A grid sized by the
        // widest solve instead paid that width at every solve of the round.
        let (mut bucket, mut items): (Vec<Vec<u32>>, Vec<u32>) = (Vec::new(), Vec::new());
        // `upto[k]` is the batch made of the first `k` solves. Because the
        // items of a level are in solve order, those solves own a prefix of
        // every bucket, so running an iteration that only some of them want is
        // still a matter of a shorter grid with nothing rebuilt.
        let mut upto: Vec<Prefix> = vec![Prefix::default()];
        {
            let g = self.solves.lock();
            for (part, &solve) in solves.iter().enumerate() {
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
                while bucket.len() + 1 < b.level_start.len() {
                    bucket.push(Vec::new());
                    items.push(0);
                }
                for (l, w) in b.level_start.windows(2).enumerate() {
                    let n = w[1] - w[0];
                    assert!(
                        n <= u32::MAX >> (32 - WORK_BITS) && (part as u64) < 1 << (32 - WORK_BITS),
                        "a round of {} solves and a level of {n} nodes overflow a work item",
                        solves.len()
                    );
                    bucket[l].extend((0..n).map(|slot| (part as u32) << WORK_BITS | slot));
                    items[l] += n;
                }
                upto.push(Prefix {
                    parts: part as u32 + 1,
                    rows,
                    items: items.clone(),
                    nterm,
                });
            }
        }
        let mut level_at: Vec<u32> = Vec::with_capacity(bucket.len() + 1);
        let mut work: Vec<u32> = Vec::new();
        for v in &bucket {
            level_at.push(work.len() as u32);
            work.extend_from_slice(v);
        }
        level_at.push(work.len() as u32);
        let s = &self.stream;
        batch.trees.put(s, desc.len(), copy(&desc))?;
        batch.work.put(s, work.len(), copy(&work))?;
        batch.coff.put(s, coff.len(), copy(&coff))?;
        batch.part.put(s, part_of_row.len(), copy(&part_of_row))?;
        batch.local.put(s, local_row.len(), copy(&local_row))?;
        batch.base.put(s, base.len(), copy(&base))?;
        batch.level_at = level_at;
        batch.upto = upto;
        batch.parts = solves.len() as u32;
        batch.cells = cells as usize;
        Ok(())
    }

    /// A value pass under the reference strategy, for a whole batch: the
    /// reaches, the network at every leaf, the terminals, and backpropagation
    /// that averages rather than updating regret. This is what a solve's
    /// targets are read off.
    fn value_pass(&self, b: &Batch) -> Res<()> {
        let all = b.all();
        self.reaches(b, all, 1, false, 0)?;
        self.network(b, all)?;
        self.terminals(b.trees.buf(), all)?;
        self.backprop(b, all, 1, 0, Cfr::LINEAR)
    }

    /// Time a stage's wall clock, always. No synchronise: what these cost is
    /// the host work in them, and the launches they queue are paid for by
    /// whichever stage synchronises next.
    fn wall<T>(&self, slot: usize, f: impl FnOnce() -> Res<T>) -> Res<T> {
        let mark = std::time::Instant::now();
        let got = f()?;
        LEAF_NS[slot].fetch_add(
            mark.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(got)
    }

    /// Time one device stage, when `WARCHEST_STAGES` asks for it. The
    /// synchronise is the measurement: without it a launch returns before the
    /// kernel has run and every stage but the last reads as free.
    fn stage<T>(&self, slot: usize, f: impl FnOnce() -> Res<T>) -> Res<T> {
        if !timing() {
            return f();
        }
        let mark = std::time::Instant::now();
        let got = f()?;
        self.stream.synchronize().map_err(err)?;
        LEAF_NS[slot].fetch_add(
            mark.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(got)
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
    /// one launch covers a whole level of the whole round: `blockIdx.x` names a
    /// work item and the item names the solve and the node.
    fn iterate(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        fn at(stage: &'static str) -> impl Fn(String) -> String {
            move |e| format!("{stage}: {e}")
        }
        let mark = std::time::Instant::now();
        // Longest-running first. A round holds solves at different points of
        // their own sixty-four iterations, and once the tree is full a solve
        // asks for its whole tail at once -- so the set still owed an iteration
        // shrinks as the call proceeds, and sorting makes that set a prefix.
        let mut order: Vec<usize> = mine.to_vec();
        order.sort_by_key(|&i| std::cmp::Reverse(Self::asked(&calls[i]).0));
        let (rounds, puct, k) = {
            let Call::Iterate { iters, puct, cfr, .. } = &calls[order[0]] else {
                unreachable!("iterate shard holds only iterate calls")
            };
            (*iters, *puct, *cfr)
        };
        let mut sims = 0usize;
        {
            let mut g = self.solves.lock();
            for &i in &order {
                let Call::Iterate { solve, step, iters, expand, cfr, puct: p, .. } = &calls[i]
                else {
                    unreachable!("iterate shard holds only iterate calls")
                };
                // The regret rule is the run's, not a solve's; the step count
                // and the two counts below are the solve's own.
                assert_eq!((cfr.alpha, cfr.beta, cfr.gamma, cfr.predict, *p),
                           (k.alpha, k.beta, k.gamma, k.predict, puct),
                           "a round mixes two regret rules");
                let b = self.slot(&mut g, *solve);
                b.step = *step;
                b.todo = *iters;
                b.nexpand = *expand;
                sims = sims.max(*expand);
            }
        }
        let solves: Vec<usize> = order.iter().map(|&i| calls[i].solve()).collect();
        let t_marshal = mark.elapsed();
        let mark = std::time::Instant::now();
        self.lay(&solves).map_err(at("lay"))?;
        let b = self.batch.lock();
        let t_up = mark.elapsed();
        let mark = std::time::Instant::now();

        // Every iteration this round was asked for, back to back, against the
        // tree the host handed over -- which does not change until the round
        // ends. Solves drop out of the grid as they finish their share.
        //
        // One reach propagation an iteration, at its end -- which is what
        // `Solver::step` does. The device used to run a second one at the top
        // of each iteration, recomputing exactly what the previous iteration's
        // trailing sweep had left behind. What is left here is the one before
        // the loop, which is not redundant: the tree grew since the last round
        // and the new subtrees have no reaches yet.
        self.stage(4, || self.reaches(&b, b.all(), 0, false, 0)).map_err(at("reach"))?;
        for iter in 0..rounds {
            let live = order
                .iter()
                .position(|&i| Self::asked(&calls[i]).0 <= iter)
                .unwrap_or(order.len());
            let p = &b.upto[live];
            let it = iter as i32;
            self.network(&b, p).map_err(at("net"))?;
            self.stage(8, || self.terminals(b.trees.buf(), p)).map_err(at("terminals"))?;
            self.stage(9, || self.backprop(&b, p, 0, it, k)).map_err(at("backprop"))?;
            // The regret update moved both players' strategies, so the reaches
            // the next iteration reads are stale until they are pushed down
            // again -- and the average strategy accumulates against those.
            self.stage(4, || self.reaches(&b, p, 0, true, it)).map_err(at("avg"))?;
            // The phase reads the Q this iteration has just formed, so it
            // belongs inside the loop: a round that runs several iterations
            // samples several times and the host grows all of them at once.
            if sims > 0 {
                self.stage(10, || {
                    self.expand(b.trees.buf(), b.parts, sims, puct, iter, rounds)
                })
                .map_err(at("expand"))?;
            }
        }
        let t_launch = mark.elapsed();
        let mark = std::time::Instant::now();
        let each = b.parts as usize * sims;
        let host = self.sampled(rounds * each)?;
        let t_down = mark.elapsed();
        for (slot, n) in [t_marshal, t_up, t_launch, t_down].iter().enumerate() {
            LEAF_NS[slot].fetch_add(n.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        // The fattest solve this round held, array by array. `held` says the
        // rate is memory-bound; this says which array to argue with.
        {
            let solves = self.solves.lock();
            if let Some(c) = solves
                .iter()
                .map(Solve::census)
                .max_by_key(|c| c.iter().map(|&(_, b)| b).sum::<usize>())
            {
                let mut best = CENSUS.lock();
                if c.iter().map(|&(_, b)| b).sum::<usize>()
                    > best.iter().map(|&(_, b)| b).sum::<usize>()
                {
                    *best = c;
                }
            }
        }
        for (part, &i) in order.iter().enumerate() {
            let (iters, want) = Self::asked(&calls[i]);
            let mut leaves = Vec::with_capacity(iters * want);
            for phase in 0..iters {
                let at = phase * each + part * sims;
                leaves.extend_from_slice(&host[at..at + want]);
            }
            out.push((i, Reply { leaves, ..Default::default() }));
        }
        Ok(())
    }

    /// What one iterate call asks for: iterations, and trajectories after them.
    fn asked(c: &Call) -> (usize, usize) {
        match c {
            Call::Iterate { iters, expand, .. } => (*iters, *expand),
            _ => unreachable!("iterate shard holds only iterate calls"),
        }
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
        // Batched like an iteration, and for the same reason: a value pass is
        // most of a CFR iteration, and one solve's leaves are a couple of
        // hundred rows, which no accelerator is interested in.
        let solves: Vec<usize> = mine.iter().map(|&i| calls[i].solve()).collect();
        self.lay(&solves)?;
        let mut b = self.batch.lock();
        let touched: Vec<i32> = mine
            .iter()
            .map(|&i| match &calls[i] {
                Call::Read { touched, .. } => (touched[0] as i32) | ((touched[1] as i32) << 1),
                _ => unreachable!("read shard holds only read calls"),
            })
            .collect();
        b.touched.put(&self.stream, touched.len(), copy(&touched))?;
        self.finish(&b, b.all())?;

        // Only a solve that is collected pays for the values. Those are laid
        // out as their own batch so the pass runs over them alone.
        let want: Vec<usize> = mine
            .iter()
            .filter(|&&i| matches!(&calls[i],
                Call::Read { vals_at, .. } if vals_at[0].1 > 0 || vals_at[1].1 > 0))
            .map(|&i| calls[i].solve())
            .collect();
        drop(b);
        if !want.is_empty() {
            self.lay(&want)?;
            self.value_pass(&self.batch.lock())?;
        }

        let g = self.solves.lock();
        let mut h = self.down_f.lock();
        for &i in mine {
            let Call::Read { solve, vals_at, policy_at, reach_at, .. } = &calls[i] else {
                unreachable!("read shard holds only read calls")
            };
            let s = &g[*solve];
            let mut root = Vec::new();
            for &(at, n) in vals_at {
                root.extend(s.get_f32(&self.stream, Ent::Reach, R_VALS, at as usize, n as usize, &mut h)?);
            }
            let policy = s.get_f32(
                &self.stream,
                Ent::Cell,
                C_SUM,
                policy_at.0 as usize,
                policy_at.1 as usize,
                &mut h,
            )?;
            let mut beliefs = Vec::new();
            for &(at, n) in reach_at {
                beliefs.extend(s.get_f32(&self.stream, Ent::Reach, R_REACH, at as usize, n as usize, &mut h)?);
            }
            out.push((i, Reply { a: root, b: policy, c: beliefs, ..Default::default() }));
        }
        Ok(())
    }

    /// The grid one level of one round takes: a block to each of the level's
    /// work items, and `blockIdx.y` for whichever half the kernel splits on --
    /// the player in the reach sweep, the traverser in backpropagation.
    fn grid(items: u32, split: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (items, split, 1),
            // Four warps: the regret update gives a warp to each of a node's
            // configs, and two of them left half the block idle at every node
            // with more than two.
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Push the reach probabilities down from the root beliefs, level by level.
    /// `also_avg` adds the reach-weighted iterate to the running strategy sum,
    /// which needs exactly the reaches this pass has just made current.
    fn reaches(&self, b: &Batch, p: &Prefix, avg: i32, also_avg: bool, iter: i32)
        -> Res<()> {
        let (trees, work) = (b.trees.buf(), b.work.buf());
        unsafe {
            self.stream
                .launch_builder(&self.k.seed_reach)
                .arg(trees).arg(&iter)
                .launch_unit(LaunchConfig {
                    grid_dim: (64, p.parts, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        let sum = also_avg as i32;
        for level in 1..p.items.len() {
            if p.items[level] == 0 {
                continue;
            }
            let (at, level_i) = (b.level_at[level] as i32, level as i32);
            unsafe {
                self.stream
                    .launch_builder(&self.k.reach_sweep)
                    .arg(trees).arg(work).arg(&at).arg(&level_i).arg(&avg).arg(&sum).arg(&iter)
                    .launch_unit(Self::grid(p.items[level], 2))
            }
            .map_err(err)?;
        }
        // The root's own row. It is not a child of anything, so the sweep never
        // reaches it and its share of the sum is a launch of its own.
        if also_avg && p.items.first().is_some_and(|&n| n > 0) {
            let (at, level_i) = (b.level_at[0] as i32, 0i32);
            unsafe {
                self.stream
                    .launch_builder(&self.k.avg_block)
                    .arg(trees).arg(work).arg(&at).arg(&level_i).arg(&iter)
                    .launch_unit(Self::grid(p.items[0], 1))
            }
            .map_err(err)?;
        }
        Ok(())
    }

    /// Value backpropagation up the levels, for one traverser. `avg` averages
    /// under the reference strategy and leaves the regrets alone.
    fn backprop(&self, b: &Batch, p: &Prefix, avg: i32, iter: i32, k: Cfr) -> Res<()> {
        for level in (0..p.items.len()).rev() {
            if p.items[level] == 0 {
                continue;
            }
            let (at, level_i) = (b.level_at[level] as i32, level as i32);
            unsafe {
                self.stream
                    .launch_builder(&self.k.backprop_sweep)
                    .arg(b.trees.buf()).arg(b.work.buf()).arg(&at).arg(&level_i).arg(&avg).arg(&iter)
                    .arg(&k.alpha).arg(&k.beta).arg(&k.gamma).arg(&k.predict)
                    .launch_unit(Self::grid(p.items[level], 2))
            }
            .map_err(err)?;
        }
        Ok(())
    }

    /// The network at every leaf of the round, for both traversers at once.
    ///
    /// Pool the normalised beliefs, run the join, read the values out into each
    /// solve's own value arena. Three launches a tile, one stage each.
    #[allow(clippy::too_many_arguments)]
    fn network(&self, b: &Batch, p: &Prefix) -> Res<()> {
        let (trees, part_d, local_d, base_d, coff_d) =
            (b.trees.buf(), b.part.buf(), b.local.buf(), b.base.buf(), b.coff.buf());
        // The active solves are a prefix of the batch, so their leaf rows are a
        // prefix of the row arrays and the pass is the same launches over fewer
        // of them.
        let stride = p.rows;
        if stride == 0 {
            return Ok(());
        }
        let mut sc = self.scratch.lock();
        let l = &self.layout;
        let (stride_i, pool_i, d_i) = (stride as i32, POOL as i32, D as i32);

        // `mass` is indexed by the round's rows, so it belongs to the round
        // rather than to a tile; the reach mass of both seats of a leaf is
        // written by the tile that leaf falls in, and read by the same tile's
        // readout.
        sc.mass.room(2 * stride)?;

        let tile = TILE.min(stride);
        sc.pooled.room(2 * tile * POOL)?;
        sc.h.room(2 * tile * D)?;
        let Scratch { mass, pooled, h, .. } = &mut *sc;
        let mass = mass.buf.as_mut().unwrap();
        let pooled = pooled.buf.as_mut().unwrap();
        let h = h.buf.as_mut().unwrap();

        // `k_join` walks the join's five packed matrices, its four norms and
        // the biases they are owed as three runs rather than as thirteen
        // slices. `NetLayout` hands the unpacked weights out in that order,
        // which `Card::on` checks once so this does not have to.
        let ln = l.norms[LN_JOIN];
        let join_ln = self.ln.slice(ln.g..ln.g + 2 * (JBLOCKS + 1) * JW);
        let join_owed = self.owed.slice(0..(JBLOCKS + 1) * JW);

        let mut q0 = 0usize;
        while q0 < stride {
            let n = tile.min(stride - q0);
            let (q0_i, n_i) = (q0 as i32, n as i32);
            let rows = 2 * n;
            let (rows_i, queries_i) = (rows as i32, rows as i32);
            self.stage(5, || {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.belief_pool)
                        .arg(trees).arg(part_d).arg(local_d).arg(base_d).arg(coff_d)
                        .arg(&mut *pooled).arg(&mut *mass)
                        .arg(&(2 * q0_i)).arg(&queries_i).arg(&stride_i).arg(&pool_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (POOL as u32, 8, 1),
                            shared_mem_bytes: 8 * POOL as u32 * 4,
                        })
                }
                .map_err(err)
            })?;

            self.stage(6, || {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.join)
                        .arg(trees).arg(part_d).arg(local_d).arg(&*pooled)
                        .arg(&self.jw).arg(&join_ln).arg(&join_owed)
                        .arg(&mut *h).arg(&rows_i).arg(&n_i).arg(&q0_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: ((rows as u32).div_ceil(JROWS as u32), 1, 1),
                            block_dim: (32, (JW / 8) as u32, 1),
                            shared_mem_bytes: 0,
                        })
                }
                .map_err(err)
            })?;

            let bias = self.b.slice(l.value_bias..l.value_bias + 1);
            let hn = l.norms[LN_H];
            let owed = self.owed.slice((JBLOCKS + 1) * JW..(JBLOCKS + 1) * JW + D);
            let g = self.ln.slice(hn.g..hn.g + hn.width);
            let hb = self.ln.slice(hn.b..hn.b + hn.width);
            self.stage(7, || {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.readout)
                        .arg(trees).arg(part_d).arg(local_d).arg(coff_d)
                        .arg(&*h).arg(&bias).arg(&*mass).arg(&owed).arg(&g).arg(&hb)
                        .arg(&rows_i).arg(&stride_i).arg(&d_i).arg(&q0_i).arg(&n_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (32, 8, 1),
                            shared_mem_bytes: 4 * D as u32,
                        })
                }
                .map_err(err)
            })?;
            q0 += n;
        }
        Ok(())
    }

    /// Terminal leaves, scored from the game rather than from the network.
    fn terminals(&self, trees: &CudaSlice<u64>, p: &Prefix) -> Res<()> {
        if p.nterm == 0 {
            return Ok(());
        }
        unsafe {
            self.stream
                .launch_builder(&self.k.terminals)
                .arg(trees)
                .launch_unit(LaunchConfig {
                    grid_dim: (p.nterm.div_ceil(8) as u32, p.parts, 2),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)
    }

    /// The expansion phase: a solve's `sims` distinct leaves, and it draws
    /// trajectories until it has them. The draws of one phase run in order,
    /// because each counts the visits it passes and the next is meant to see
    /// them -- and so do the phases, which is why the round's `iters` phases
    /// share one buffer and one download.
    ///
    /// The kernel is handed the whole buffer rather than this phase's slice of
    /// it: the leaves the round's earlier phases took are what a leaf is
    /// checked against, and they are already in it.
    fn expand(&self, trees: &CudaSlice<u64>, parts: u32, sims: usize, puct: f32,
              iter: usize, iters: usize) -> Res<()> {
        let each = parts as usize * sims;
        let mut sc = self.scratch.lock();
        let out = sc.leaves.room((iters * each).max(1))?;
        let (parts_i, sims_i) = (parts as i32, sims as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.expand)
                .arg(trees).arg(out).arg(&parts_i).arg(&sims_i).arg(&puct)
                .arg(&(iter as i32))
                .arg(&(each as i32))
                .arg(&(crate::search::TRIES as i32))
                .launch_unit(LaunchConfig {
                    grid_dim: (parts.max(1), 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)
    }

    /// The leaves every phase of this round sampled, laid out by phase then by
    /// solve. The download is what ends the round: nothing else the iterations
    /// produced has a reader on the host.
    fn sampled(&self, n: usize) -> Res<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut sc = self.scratch.lock();
        let out = sc.leaves.room(n)?;
        self.down.lock().recv(&self.stream, &out.slice(0..n))
    }

    /// The reference strategy, once the tree has stopped growing.
    /// `touched` is per solve: which players' running sums have moved, or `-1`
    /// for a solve that is not asking for this at all.
    fn finish(&self, b: &Batch, p: &Prefix) -> Res<()> {
        let touched = b.touched.buf();
        for level in 0..p.items.len() {
            if p.items[level] == 0 {
                continue;
            }
            let (at, level_i) = (b.level_at[level] as i32, level as i32);
            unsafe {
                self.stream
                    .launch_builder(&self.k.finish)
                    .arg(b.trees.buf()).arg(b.work.buf()).arg(&at).arg(&level_i).arg(touched)
                    .launch_unit(Self::grid(p.items[level], 1))
            }
            .map_err(err)?;
        }
        Ok(())
    }

}

/// Everything a round sends, by role, host buffer and device buffer together.
///
/// Kept between rounds. A round concatenates tens of megabytes of public
/// encodings, and building that from an empty buffer every time is twenty-odd
/// reallocations and thousands of first-touch page faults -- which measured at
/// a quarter of the whole round. The page-locked half is why the copies no
/// longer block; see `Host`.
#[derive(Default)]
struct Stage {
    packed: Wire<u8>,
    xpub: Arr<f32>,
    cards: Wire<f32>,
    card_of_row: Wire<i32>,
    phi: Wire<f32>,
    owner: Wire<u32>,
    cfg_cards: Wire<f32>,
    blob: Wire<u32>,
    dst: Wire<u64>,
    at: Wire<u32>,
    src: Wire<u32>,
    start: Wire<u32>,
}

/// The intermediates of one pass, by role. Each is fully written before it is
/// read, so they are grown rather than cleared.
#[derive(Default)]
struct Scratch {
    /// `[2, rows]` reach mass per player.
    mass: Arr<f32>,
    /// `[2 * rows, POOL]` the pooled belief block.
    pooled: Arr<f32>,
    /// `[rows, D]` the head. The join's residual stream never reaches memory:
    /// `k_join` holds it in registers from the board seed to this. Also the
    /// trunk's board vectors and the config encoder's `f`.
    h: Arr<f32>,
    /// Trunk join-cache / config `f_p`.
    z: Arr<f32>,
    /// Trunk `board_out` input.
    input: Arr<f32>,
    /// `[parts * sims]` the leaves an expansion phase sampled.
    leaves: Arr<u32>,
    /// TILE trunk (and the config / prior passes that reuse these).
    piles: Arr<f32>,
    tokens: Arr<f32>,
    projected: Arr<f32>,
    type_pool: Arr<f32>,
    loose: Arr<f32>,
    glob: Arr<f32>,
    facts: Arr<f32>,
    occupant: Arr<i32>,
    x: Arr<f32>,
    bag: Arr<f32>,
}

/// The driver and cuBLAS error types are `Debug` only.
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
