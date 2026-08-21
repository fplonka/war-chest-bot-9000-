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

use crate::board::{board, N_HEXES, NONE};
use crate::farm::{Call, Dst, Prime, Reply, CARD_ROWS};
use crate::net::{
    ln_block, Net, NetLayout, NormSpan, Span, AFEAT, AW, BLOCKS, C, CFGH, D, JBLOCKS, JOIN_IN, JW,
    LN_ACT, LN_CFG, LN_H, LN_JOIN, LN_JOUT, LN_TRUNK, POOL, TYPE,
};
use crate::pbs::{
    CFEAT, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_LOOSE, OFF_PILES, PILE_COUNTS, PUBFEAT,
};
use crate::search::Cfr;

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

/// Device bytes every card's solve arenas hold right now -- a level, which
/// `leaf_breakdown` reports without resetting. `Device::room_for` asks the same
/// question per card, and is what admits a solve or holds it back.
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
    gelu: CudaFunction,
    norm: CudaFunction,
    norm_ip: CudaFunction,
    bias: CudaFunction,
    window: CudaFunction,
    gather: CudaFunction,
    scatter: CudaFunction,
    seed_reach: CudaFunction,
    avg_block: CudaFunction,
    beliefs: CudaFunction,
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
            norm: get("k_norm")?,
            norm_ip: get("k_norm_ip")?,
            bias: get("k_bias")?,
            window: get("k_window")?,
            gather: get("k_gather")?,
            scatter: get("k_scatter")?,
            seed_reach: get("k_seed_reach")?,
            avg_block: get("k_avg_block")?,
            beliefs: get("k_beliefs")?,
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

/// One row per warp, eight warps to a block. What `k_norm` wants: its rows are
/// a hundred-odd wide, which one warp reduces in five shuffles where a block
/// spends the same time in barriers.
fn warp_rows(rows: usize) -> LaunchConfig {
    const WARPS: u32 = 8;
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(WARPS).max(1), 1, 1),
        block_dim: (32, WARPS, 1),
        shared_mem_bytes: 0,
    }
}

/// Independent streams a GPU is fed by.
///
/// Each is a whole copy of a card's working state -- its own stream, staging,
/// scratch and solve table -- and the farm drives each with a thread of its
/// own. One is not enough: a driver spends about three fifths of a round
/// issuing launches rather than waiting for the card, so a single stream leaves
/// the GPU idle through most of it. Measured on two 3090s at the production
/// budget, generation runs at 21 solves/s with one stream a card and 75 to 86
/// with ten.
const STREAMS: usize = 10;

pub struct Device {
    /// One entry per stream, `STREAMS` of them per ordinal. Each is driven by
    /// one thread, which is what makes a round a batch: everything that thread
    /// found waiting goes in together.
    cards: Vec<Card>,
    net: Net,
}

/// The trunk's two square matrices a block, permuted so that one lane's three
/// channels of a weight row are four floats side by side.
///
/// `k_trunk` reads `m[k * c + lane + 32 * q]` for `q` in nought to two. Stored
/// as the net stores it that is three loads a row, and the inner product's
/// twelve multiplies then cost seven loads -- which leaves the kernel issuing
/// addresses rather than multiplying. Here the row is `TRUNK_LD` wide and a
/// lane's channels are at `4 * lane + q`: one sixteen-byte load, still five
/// hundred and twelve contiguous bytes across the warp. The fourth slot is
/// what makes the address a multiple of sixteen; it is never read.
///
/// Returns the buffer and where each matrix starts, mix then out, block by
/// block.
fn lanewise(l: &NetLayout, w: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let mut out = Vec::new();
    let mut at = Vec::new();
    for blk in &l.blocks {
        for s in [blk.mix, blk.out] {
            assert_eq!(s.o, C, "the trunk's matrices are square in the channels");
            at.push(out.len());
            let base = out.len();
            out.resize(base + s.i * TRUNK_LD, 0.0);
            for k in 0..s.i {
                for j in 0..s.o {
                    out[base + k * TRUNK_LD + 4 * (j % 32) + j / 32] = w[s.w + k * s.o + j];
                }
            }
        }
    }
    (out, at)
}

/// A lanewise weight row, wide enough that a lane's four floats start on a
/// sixteen-byte boundary.
const TRUNK_LD: usize = 128;

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

/// Card memory that is never admitted against, per ordinal.
///
/// Two things live in it. A round's own intermediates -- hundreds of megabytes,
/// four hundred of them for the join head alone at a large round -- which every
/// stream of an ordinal has its own set of. And the blocks the stream-ordered
/// allocator keeps back between one solve and the next, which is the larger
/// half and the reason this cannot be measured instead: a freed arena goes to a
/// pool rather than to the driver, so `mem_get_info` reports the pool as taken
/// and the card as full while most of it is available to the next solve.
///
/// What would make it wrong: a round whose intermediates grow -- a wider join
/// head, a larger `TILE`, more streams an ordinal -- or a pool that fragments
/// worse than `grow_to`'s size classes allow, so that blocks pile up unusable.
/// Both show in `leaf_breakdown`'s census: the card's free memory falls while
/// the solve arenas do not.
const ROUND_RESERVE: u64 = 6 << 30;

/// Leaf rows a pass works on at once.
///
/// The intermediates of the leaf pass are 5,640 bytes a row, so sizing them by
/// the whole round is a gigabyte a lane -- and with several cohorts of solves
/// in flight, memory is what bounds how many. A tile large enough to fill the
/// card costs ninety megabytes and a handful of extra launches.
const TILE: usize = 16384;

/// The capacity an array of `want` elements takes: a power of two.
///
/// Doubling rather than a quarter at a time. An arena that grows by a quarter
/// reallocates `log(final/first)/log(1.25)` times over a solve and copies
/// everything it holds each time -- five times its final size in
/// device-to-device traffic and, worse, three driver calls per growth on the
/// one thread a round can least afford. Not four times at a time either, even
/// though that is fewer still: headroom nobody is using is a solve that does
/// not fit.
///
/// Allocation is stream-ordered, so a freed buffer goes back to a pool rather
/// than to the driver, and it can only serve a request it is large enough for.
/// Doubling an arbitrary `want` gives arbitrary sizes -- one solve returns
/// 260,002 floats and the next asks for 259,884 -- so the pool held both and
/// grew until it had every size any slot had ever wanted, which is how both
/// cards came to read 24,027 MiB of 24,576. A size class fixes that.
///
/// The class has to be *coarse*, which is the part that is not obvious. A
/// power of two leaves up to half the array as slack, and the census says
/// that is real: the fattest solve holds 179 MB with nine of its arrays at
/// exactly 2^21 floats. Eight classes to an octave cuts the slack to an
/// eighth -- and ran the cards out of memory at twelve cohorts, where powers
/// of two had held. What a retained pool cares about is not how much slack a
/// block has but whether some other slot can use it, and eight times as many
/// classes is eight times fewer blocks that fit. Reuse beats slack.
fn grow_to(want: usize) -> usize {
    want.next_power_of_two().max(4096)
}

#[cfg(test)]
mod grow {
    use super::grow_to;

    #[test]
    fn a_size_class_is_never_smaller_and_never_twice_as_large() {
        for want in (1..1 << 22).step_by(9_973) {
            let got = grow_to(want);
            assert!(got >= want, "{want} -> {got} is smaller");
            assert!(got <= 4096.max(2 * want), "{want} -> {got} is more than double");
        }
    }

    #[test]
    fn an_octave_holds_one_class() {
        // What makes a freed block usable by another solve: every request in
        // an octave lands on the same size, not on eight nearby ones. This is
        // the property, and it is worth more than the slack it costs.
        let sizes: std::collections::BTreeSet<usize> =
            ((1 << 20) + 1..1 << 21).step_by(97).map(grow_to).collect();
        assert_eq!(sizes.len(), 1, "an octave holds {} classes", sizes.len());
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
            let cap = grow_to(want);
            self.buf = Some(unsafe { stream.context().alloc_pinned::<T>(cap) }.map_err(err)?);
        }
        let b = self.buf.as_mut().expect("just fitted");
        self.len = f(b.as_mut_slice().map_err(err)?);
        assert!(self.len <= b.len(), "a fill wrote past the buffer");
        Ok(())
    }

    /// The same, into a buffer of its own. `lay` needs this: a round can hold
    /// three batches at once and they must not share device memory.
    fn send_new(&mut self, stream: &Arc<CudaStream>) -> Res<CudaSlice<T>> {
        let mut dst = unsafe { stream.alloc::<T>(self.len.max(1)) }.map_err(err)?;
        self.send(stream, &mut dst)?;
        Ok(dst)
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
}

/// One host buffer and the device buffer it is sent to, kept together and kept
/// between rounds. A round's staging is a fixed set of these, by role.
#[derive(Default)]
struct Wire<T> {
    host: Host<T>,
    dev: Arr<T>,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Copy> Wire<T> {
    fn put(
        &mut self,
        stream: &Arc<CudaStream>,
        want: usize,
        f: impl FnOnce(&mut [T]) -> usize,
    ) -> Res<()> {
        self.host.fill(stream, want, f)?;
        let n = self.host.len;
        self.dev.room(stream, n.max(1))?;
        let dst = self.dev.buf.as_mut().expect("room");
        self.host.send(stream, dst)
    }
}

/// One device array of a solve's state.
///
/// It grows geometrically and keeps what it holds: regrets, visit counts and
/// the strategy sum accumulate across a solve's iterations, so a reallocation
/// that dropped them would silently restart the search.
struct Arr<T> {
    buf: Option<CudaSlice<T>>,
    cap: usize,
    /// Elements actually written, as against `cap`, which is a size class.
    /// `Device::resident` is what reads it.
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
        let cap = grow_to(want);
        HELD.fetch_add(
            ((cap - self.cap) * std::mem::size_of::<T>()) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Uninitialised, then the old contents, then zeros over what is new.
        // Allocating zeroed would clear the whole buffer and then overwrite
        // most of it with the copy -- twice the writes for the same answer.
        let mut fresh = unsafe { stream.alloc::<T>(cap) }.map_err(err)?;
        if self.cap > 0 {
            let old = self.buf.as_ref().expect("a capacity implies a buffer");
            let mut d = fresh.slice_mut(0..self.cap);
            stream.memcpy_dtod(&old.slice(0..self.cap), &mut d).map_err(err)?;
            LEAF_NS[20].fetch_add(
                (self.cap * std::mem::size_of::<T>()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let mut tail = fresh.slice_mut(self.cap..cap);
        stream.memset_zeros(&mut tail).map_err(err)?;
        self.buf = Some(fresh);
        self.cap = cap;
        Ok(())
    }

    /// Reserve room for `n` elements at `at` and hand back where they go. The
    /// write itself happens later, packed with every other one in the round.
    fn plan(&mut self, stream: &Arc<CudaStream>, at: usize, n: usize) -> Res<u64> {
        self.fit(stream, at + n)?;
        self.len = at + n;
        Ok(self.ptr(stream))
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

    /// Copy `n` elements of `src` starting at `from` to `at`.
    fn copy(&mut self, stream: &Arc<CudaStream>, at: usize, src: &CudaSlice<T>, from: usize, n: usize)
        -> Res<()> {
        self.fit(stream, at + n)?;
        self.len = at + n;
        if n == 0 {
            return Ok(());
        }
        let dst = self.buf.as_mut().expect("just fitted");
        let mut d = dst.slice_mut(at..at + n);
        stream.memcpy_dtod(&src.slice(from..from + n), &mut d).map_err(err)
    }

    /// Grow to `want` without preserving or zeroing what is there. For the
    /// pass intermediates, every one of which is fully written before it is
    /// read.
    ///
    /// # Safety
    /// The caller must write every element it then reads.
    fn room(&mut self, stream: &Arc<CudaStream>, want: usize) -> Res<&mut CudaSlice<T>> {
        if self.cap < want {
            self.cap = grow_to(want);
            self.buf = Some(unsafe { stream.alloc::<T>(self.cap) }.map_err(err)?);
        }
        Ok(self.buf.as_mut().expect("a capacity implies a buffer"))
    }

    /// Give the buffer back.
    ///
    /// A slot is reused by the next solve, and a solve's cost varies
    /// twenty-six fold. A slot that kept the largest tree it ever served would
    /// hold that much for the rest of the run, so the card would need the worst
    /// case times the number of slots rather than what is actually in flight.
    /// Allocation is stream-ordered, so this returns the pages to a pool the
    /// other slots draw from and costs about what a launch does.
    fn reset(&mut self) {
        HELD.fetch_sub(
            (self.cap * std::mem::size_of::<T>()) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.buf = None;
        self.cap = 0;
        self.len = 0;
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
    /// Board vectors and the join cache, once per distinct public state.
    p: Arr<f32>,
    jp: Arr<f32>,
    /// `[row]` -> the board it reads. Rows outnumber boards because coin plays
    /// commute, so a tree spanning one round holds the same public state at
    /// several places.
    board_of: Arr<u32>,
    /// `f(c)` and `g(c)`, and the belief index that names them.
    ///
    /// Both stay f32. Half storage would halve the largest byte flow in the
    /// design -- the readout gathers a row of `f` per config, forty million
    /// times a solve -- but it also turns a last-bit difference in the config
    /// encoder's matrix multiply, whose shape depends on which solves share the
    /// round, into a discrete one, and regret matching amplifies that. It moved
    /// the root policy by 1.4e-1 between batch compositions against a 5e-2
    /// bound. The readout and the pooling are about a tenth of device time; the
    /// policy target is not worth a tenth.
    f: Arr<f32>,
    g: Arr<f32>,
    /// The policy readout's config row, beside the value's `f`. The policy head
    /// runs here now, so this has no reader on the host either.
    fp: Arr<f32>,
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
    rootb: Arr<f32>,
    leaf_node: Arr<u32>,
    term: Arr<u32>,
    nterm: usize,
    /// Values per traverser: the arena holds both, so one launch backpropagates
    /// both.
    nvals: usize,
    /// Strategy cells and reach entries. The arenas that hold them are only
    /// ever fitted, never written from the host, so their own `len` stays at
    /// zero and `Device::resident` reads these instead.
    ncells: usize,
    nreach: usize,
    /// Level bounds, on the host, because they drive the launch loop.
    level_start: Vec<u32>,
    /// The expansion's own random stream, seeded once by the solver.
    seed: Arr<u64>,
    /// What this round asks of this solve: where its iterate count stands, how
    /// many iterations to run, and how many trajectories after them. All three
    /// differ across a round -- solves start at different times and a solve
    /// whose tree is full runs its whole tail in one call -- so they belong to
    /// the solve rather than to the batch.
    step: usize,
    todo: usize,
    nexpand: usize,
}

/// A set of solves laid out as one batch, and the device arrays that describe
/// it. Every stage of an iteration reads these, so laying them out once is what
/// makes a round of thirty solves one launch a stage rather than thirty.
struct Batch {
    trees: CudaSlice<u64>,
    coff: CudaSlice<u32>,
    part: CudaSlice<i32>,
    local: CudaSlice<i32>,
    base: CudaSlice<i32>,
    /// Prefixes of the batch, one per solve count. The solves are laid out
    /// longest-running first, so the ones still owed an iteration are always a
    /// prefix -- and an iteration that fewer solves want is the same launch
    /// with a shorter grid and fewer rows, at no host cost.
    upto: Vec<Prefix>,
    parts: u32,
    cells: usize,
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
    /// The widest level among them, level by level: the grid a launch covering
    /// that level needs.
    wide: Vec<u32>,
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

/// Fields of `struct Tree` in `kernels.cu`, in order. Every one is eight bytes
/// wide, so the descriptor is positional and needs no packing rules.
const DESC: usize = 59;

impl Solve {
    /// What this solve holds, array by array.
    fn census(&self) -> Vec<(&'static str, usize)> {
        let t = &self.tree;
        let f = std::mem::size_of::<f32>();
        let u = std::mem::size_of::<u32>();
        let mut v = vec![
            ("p", self.p.cap * f), ("jp", self.jp.cap * f),
            ("board_of", self.board_of.cap * u),
            ("f", self.f.cap * f), ("g", self.g.cap * f), ("fp", self.fp.cap * f),
            ("cidx", self.cidx.cap * u), ("coff", self.coff.cap * u),
            ("reach", self.reach.cap * f), ("vals", self.vals.cap * f),
            ("cur", self.cur.cap * f), ("regret", self.regret.cap * f),
            ("sum", self.sum.cap * f), ("qval", self.qval.cap * f),
            ("visits", self.visits.cap * f), ("prior", self.prior.cap * f),
            ("rootb", self.rootb.cap * f),
            ("leaf_node", self.leaf_node.cap * u), ("term", self.term.cap * u),
            ("legal_child", t.legal_child.cap * u),
            ("legal_trans", t.legal_trans.cap * u),
            ("cell_row", t.cell_row.cap * u), ("cell_val", t.cell_val.cap * u),
            ("legal_off", t.legal_off.cap * u), ("child", t.child.cap * u),
            ("rev_start", t.rev_start.cap * u), ("rev_src", t.rev_src.cap * u),
            ("rev_cell", t.rev_cell.cap * u),
            ("rvd_start", t.rvd_start.cap * u), ("rvd_src", t.rvd_src.cap * u),
            ("rvd_p", t.rvd_p.cap * f),
            ("draw_start", t.draw_start.cap * u), ("draw_to", t.draw_to.cap * u),
            ("draw_p", t.draw_p.cap * f),
            ("level_node", t.level_node.cap * u),
        ];
        v.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
        v
    }

    /// Device bytes this solve's arenas hold, all of them together. What the
    /// card admits solves against.
    fn bytes(&self) -> usize {
        self.census().iter().map(|&(_, b)| b).sum()
    }

    /// Where a run of the round's blob lands. The match is the other half of
    /// `farm::Dst`, and the only place the two vocabularies meet.
    fn plan(&mut self, s: &Arc<CudaStream>, d: Dst, at: usize, n: usize) -> Res<u64> {
        let t = &mut self.tree;
        match d {
            Dst::Kind => t.kind.plan(s, at, n),
            Dst::Player => t.player.plan(s, at, n),
            Dst::Exhausted => t.exhausted.plan(s, at, n),
            Dst::Nc => t.nc.plan(s, at, n),
            Dst::Parent => t.parent.plan(s, at, n),
            Dst::Roff => t.roff.plan(s, at, n),
            Dst::Voff => t.voff.plan(s, at, n),
            Dst::Soff => t.soff.plan(s, at, n),
            Dst::Util => t.util.plan(s, at, n),
            Dst::ChildAt => t.child_at.plan(s, at, n),
            Dst::ChildN => t.child_n.plan(s, at, n),
            Dst::Child => t.child.plan(s, at, n),
            Dst::LegalBase => t.legal_base.plan(s, at, n),
            Dst::LegalOff => t.legal_off.plan(s, at, n),
            Dst::LegalChild => t.legal_child.plan(s, at, n),
            Dst::LegalTrans => t.legal_trans.plan(s, at, n),
            Dst::CellRow => t.cell_row.plan(s, at, n),
            Dst::CellVal => t.cell_val.plan(s, at, n),
            Dst::RevBase => t.rev_base.plan(s, at, n),
            Dst::RevStart => t.rev_start.plan(s, at, n),
            Dst::RevSrc => t.rev_src.plan(s, at, n),
            Dst::RevCell => t.rev_cell.plan(s, at, n),
            Dst::RvdBase => t.rvd_base.plan(s, at, n),
            Dst::RvdStart => t.rvd_start.plan(s, at, n),
            Dst::RvdSrc => t.rvd_src.plan(s, at, n),
            Dst::RvdP => t.rvd_p.plan(s, at, n),
            Dst::DrawBase => t.draw_base.plan(s, at, n),
            Dst::DrawStart => t.draw_start.plan(s, at, n),
            Dst::DrawTo => t.draw_to.plan(s, at, n),
            Dst::DrawP => t.draw_p.plan(s, at, n),
            Dst::LevelStart => t.level_start.plan(s, at, n),
            Dst::LevelNode => t.level_node.plan(s, at, n),
            Dst::Cur => self.cur.plan(s, at, n),
            Dst::Prior => self.prior.plan(s, at, n),
            Dst::LeafNode => self.leaf_node.plan(s, at, n),
            Dst::Term => self.term.plan(s, at, n),
            Dst::Rootb => self.rootb.plan(s, at, n),
        }
    }

    fn describe(&self, s: &Arc<CudaStream>) -> [u64; DESC] {
        let t = &self.tree;
        [
            t.kind.ptr(s), t.player.ptr(s), t.exhausted.ptr(s), t.nc.ptr(s), t.parent.ptr(s),
            t.roff.ptr(s), t.voff.ptr(s), t.soff.ptr(s), t.util.ptr(s),
            t.child_at.ptr(s), t.child_n.ptr(s), t.child.ptr(s),
            t.legal_base.ptr(s), t.legal_off.ptr(s), t.legal_child.ptr(s),
            t.legal_trans.ptr(s), t.cell_row.ptr(s), t.cell_val.ptr(s),
            t.rev_base.ptr(s), t.rev_start.ptr(s), t.rev_src.ptr(s), t.rev_cell.ptr(s),
            t.rvd_base.ptr(s), t.rvd_start.ptr(s), t.rvd_src.ptr(s), t.rvd_p.ptr(s),
            t.draw_base.ptr(s), t.draw_start.ptr(s), t.draw_to.ptr(s), t.draw_p.ptr(s),
            t.level_start.ptr(s), t.level_node.ptr(s),
            self.reach.ptr(s), self.vals.ptr(s), self.cur.ptr(s), self.regret.ptr(s),
            self.sum.ptr(s), self.qval.ptr(s), self.visits.ptr(s), self.prior.ptr(s),
            // `avg` is `sum` normalised, written once by `k_finish` as a
            // solve's last act and read only by the value pass after it. So it
            // is the same array: four bytes a cell, and a solve holds up to a
            // million and a half of them.
            self.sum.ptr(s), self.rootb.ptr(s),
            self.p.ptr(s), self.jp.ptr(s), self.board_of.ptr(s),
            self.f.ptr(s), self.g.ptr(s), self.fp.ptr(s),
            self.cidx.ptr(s), self.coff.ptr(s),
            self.leaf_node.ptr(s), self.term.ptr(s), self.seed.ptr(s),
            self.level_start.len().saturating_sub(1) as u64,
            self.nterm as u64,
            self.nvals as u64,
            self.step as u64,
            self.todo as u64,
            self.nexpand as u64,
        ]
    }
}

struct Card {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    /// Indexed by solve, which is the slot the farm pinned it to.
    solves: parking_lot::Mutex<Vec<Solve>>,
    /// Host staging for a round's batches, kept between rounds for the same
    /// reason the device scratch is: a round concatenates sixteen megabytes of
    /// public encodings, and building that from an empty `Vec` every time is
    /// twenty-odd reallocations and four thousand first-touch page faults --
    /// which measured at a quarter of the whole round.
    host: parking_lot::Mutex<Stage>,
    /// A round's writes, kept between rounds for the same reason.
    pack: parking_lot::Mutex<Pack>,
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
    /// The two square matrices of every trunk block, laid out the way a lane
    /// reads them.
    ///
    /// `k_trunk` gives a thread the three channels `lane, lane + 32, lane + 64`
    /// of one weight row, and reading them where the net stores them is three
    /// loads. Here they are four floats side by side -- three used, one for the
    /// alignment a sixteen-byte load needs -- so a row is one load, and a warp
    /// still reads five hundred contiguous bytes. That and the same treatment
    /// of the board turn twelve multiplies per seven loads into forty-eight per
    /// eight, which is the difference between an issue-bound kernel and an
    /// arithmetic-bound one.
    wt: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    /// Hex adjacency, `NONE` folded to `-1`.
    nb: CudaSlice<i32>,
    /// Device bytes this card's solve arenas may hold.
    ///
    /// Whatever it had free when the backend came up, less `ROUND_RESERVE`,
    /// shared out between the streams of its ordinal.
    ///
    /// Measured rather than configured. A solve's cost varies twenty-six fold
    /// with how far into a game its root sits, so no count of threads describes
    /// what fits, and a run that guesses either wastes the card or fills it.
    budget: u64,
    /// The most any one solve on this card has ever held. `room_for` projects
    /// the population at it, which is what keeps admission from lagging behind
    /// arenas that are still filling.
    peak: std::sync::atomic::AtomicU64,
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
    /// Bring up one card per ordinal and upload the weights to each.
    pub fn new(ordinals: &[usize], net: Net) -> Res<Device> {
        if ordinals.is_empty() {
            return Err("no cuda device ordinals given".into());
        }
        if net.is_empty() {
            return Err("cannot start the device backend without weights".into());
        }
        // Only the context is shared between an ordinal's streams; the
        // weights are duplicated because they are a few megabytes against the
        // rounds they serve.
        let cards = ordinals
            .iter()
            .flat_map(|&o| (0..STREAMS).map(move |k| (o, k)))
            .map(|(o, k)| Card::new(o, &net, k > 0))
            .collect::<Res<Vec<_>>>()?;
        Ok(Device { cards, net })
    }

    /// How many cards a round can be spread over.
    pub fn cards(&self) -> usize {
        self.cards.len()
    }

    /// Whether this card can take another solve beside the `live` it holds.
    ///
    /// Not the level against the ceiling. A solve's arenas fill over its whole
    /// run, so what a population holds now is what a younger one held, and
    /// admitting on that overshoots by whatever the solves already in flight
    /// grow in the meantime. This projects instead: every solve in flight, and
    /// the one being asked about, at the largest a solve on this card has ever
    /// reached. Nothing that is already admitted can then surprise it.
    ///
    /// The peak is nought until the first solve has grown, and the answer is
    /// yes until it has. What makes that safe is the farm's pacing: it admits
    /// one solve per solve *finished*, so the second is admitted only once the
    /// first has run its whole life and the peak is a real one.
    pub fn room_for(&self, card: usize, live: usize) -> bool {
        let c = &self.cards[card];
        let widest = c.solves.lock().iter().map(|s| s.bytes() as u64).max().unwrap_or(0);
        let peak = widest.max(c.peak.fetch_max(widest, std::sync::atomic::Ordering::Relaxed));
        (live as u64 + 1) * peak <= c.budget
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
            let (lw, _) = lanewise(&card.layout, &flat.w);
            card.stream.memcpy_htod(&lw, &mut card.wt).map_err(err)?;
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
    pub fn run(&self, calls: &[Call], card: usize) -> Option<Vec<Reply>> {
        // A solve's board vectors stay on the card that produced them, so every
        // call of one solve reaches the same card. The farm pins a solve to a
        // card for its whole life, so a round is already the calls of one.
        let all: Vec<usize> = (0..calls.len()).collect();
        match self.cards[card].round(calls, &all) {
            Ok(part) => {
                let mut out: Vec<Reply> = (0..calls.len()).map(|_| Reply::default()).collect();
                for (i, reply) in part {
                    out[i] = reply;
                }
                Some(out)
            }
            Err(e) => {
                eprintln!("cuda: card {card}: {e}");
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
        let all = |a: &Arr<f32>| c.slice(a, 0, a.len);
        let cells = |a: &Arr<f32>| c.slice(a, 0, s.ncells);
        Ok(Resident {
            p: all(&s.p)?,
            jp: all(&s.jp)?,
            f: all(&s.f)?,
            g: all(&s.g)?,
            fp: all(&s.fp)?,
            prior: all(&s.prior)?,
            cur: cells(&s.cur)?,
            sum: cells(&s.sum)?,
            qval: cells(&s.qval)?,
            visits: cells(&s.visits)?,
            reach: c.slice(&s.reach, 0, s.nreach)?,
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

impl Card {
    fn new(ordinal: usize, net: &Net, own_stream: bool) -> Res<Card> {
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
                // `cuda_fp16.h`, for the half-precision config readout.
                include_paths: vec![
                    "/usr/local/cuda/include".into(),
                    "/usr/include".into(),
                ],
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        // A stream past the first needs one of its own, or the two drivers
        // serialise on the card they are meant to be filling in turn.
        let stream = if own_stream {
            ctx.new_stream().map_err(err)?
        } else {
            ctx.default_stream()
        };
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
        let layout = NetLayout::new();
        let (lanewise, lanes) = lanewise(&layout, &flat.w);
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
        // After the weights are up and before any solve is admitted. The box
        // is shared, so what another process already holds is simply not free.
        // Every stream of an ordinal reads the same figure and takes a share of
        // it, because they are all spending the one card's memory.
        let free = cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64;
        let budget = free.saturating_sub(ROUND_RESERVE) / STREAMS as u64;
        Ok(Card {
            plan: stream.memcpy_stod(&plan).map_err(err)?,
            owed: stream.memcpy_stod(&owed).map_err(err)?,
            w: stream.memcpy_stod(&flat.w).map_err(err)?,
            wt: stream.memcpy_stod(&lanewise).map_err(err)?,
            b: stream.memcpy_stod(&flat.b).map_err(err)?,
            ln: stream.memcpy_stod(&flat.ln).map_err(err)?,
            nb: stream.memcpy_stod(&nb).map_err(err)?,
            stream,
            blas,
            k,
            budget,
            peak: std::sync::atomic::AtomicU64::new(0),
            solves: parking_lot::Mutex::new(Vec::new()),
            host: parking_lot::Mutex::new(Stage::default()),
            pack: parking_lot::Mutex::new(Pack::default()),
            scratch: parking_lot::Mutex::new(Scratch::default()),
            layout,
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

    /// `Norm::apply` when `act`, `Norm::plain` when not, from `src` into `dst`,
    /// adding what the residual stream is owed as it reads. `owed` is an index
    /// into `Card::owed`. The two buffers may be the same, which is what a norm
    /// in place is.
    fn norm_owed(
        &self,
        s: NormSpan,
        rows: usize,
        act: bool,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        owed: Option<usize>,
    ) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let g = self.ln.slice(s.g..s.g + s.width);
        let b = self.ln.slice(s.b..s.b + s.width);
        let add = match owed {
            Some(at) => self.owed.slice(at..at + s.width),
            // A launch cannot take a null pointer through the builder, so a
            // norm with nothing owed reads the first sum and is told to ignore
            // it by `has`.
            None => self.owed.slice(0..s.width.min(JW)),
        };
        let (rows_i, width, act) = (rows as i32, s.width as i32, act as i32);
        let has = owed.is_some() as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.norm)
                .arg(src).arg(dst).arg(&g).arg(&b).arg(&add).arg(&has)
                .arg(&rows_i).arg(&width).arg(&act)
                .launch_unit(warp_rows(rows))
        }
        .map_err(err)
    }

    /// The same, in place.
    fn norm(&self, s: NormSpan, rows: usize, act: bool, x: &mut CudaSlice<f32>) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let g = self.ln.slice(s.g..s.g + s.width);
        let b = self.ln.slice(s.b..s.b + s.width);
        let add = self.owed.slice(0..s.width.min(JW));
        let (rows_i, width, act) = (rows as i32, s.width as i32, act as i32);
        let has = 0i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.norm_ip)
                .arg(x).arg(&g).arg(&b).arg(&add).arg(&has)
                .arg(&rows_i).arg(&width).arg(&act)
                .launch_unit(warp_rows(rows))
        }
        .map_err(err)
    }


    /// Reach the state of solve `solve`, creating it if this is its first call.
    fn slot<'g>(&self, g: &'g mut Vec<Solve>, solve: usize) -> &'g mut Solve {
        if g.len() <= solve {
            g.resize_with(solve + 1, Solve::default);
        }
        &mut g[solve]
    }

    /// Scratch for one pass.
    ///
    /// Uninitialised, not zeroed. Every buffer this hands out is fully written
    /// by the kernel that follows, and at four hundred thousand rows a round
    /// the zeroing alone was a gigabyte of writes an iteration -- the same
    /// mistake `docs/PERF.md` records fixing on the host, where `clear()` then
    /// `resize()` was a memset per layer per call.
    ///
    /// # Safety
    /// The caller must write every element before reading one.
    fn alloc(&self, n: usize) -> Res<CudaSlice<f32>> {
        unsafe { self.stream.alloc::<f32>(n.max(1)) }.map_err(err)
    }

    // ----------------------------------------------------------------- trunk

    /// Every new leaf in the round: the board vector and the join cache.
    fn trunk(&self, calls: &[Call], mine: &[usize], pack: &mut Pack) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        // Concatenate, straight into the page-locked buffers the copies read.
        // `card_of_row` is what replaces `board`'s modulo: a leaf reads the
        // physical view of the card table its own solve drafted.
        let mark = std::time::Instant::now();
        let s = &self.stream;
        let mut stage = self.host.lock();
        // Concatenation only works if a call carries exactly its own rows. A
        // trailing tail from a caller's scratch buffer would shift every later
        // call in the batch and is invisible when a call runs alone.
        let each = |i: usize| -> (&[f32], &[f32], usize) {
            let Call::Trunk { xpub, cards, boards, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            assert_eq!(xpub.len(), boards * PUBFEAT, "trunk xpub is not one row a board");
            assert_eq!(cards.len(), CARD_ROWS * NTYPE * TYPE, "trunk card table");
            (xpub, cards, *boards)
        };
        // The trunk runs on distinct public states; everything indexed by row
        // -- the belief index and `board_of` itself -- is counted apart.
        let rows: usize = mine.iter().map(|&i| each(i).2).sum();
        // A call carries one row a leaf, so a call is one copy. This used to
        // gather the physical row out of a pair, a leaf at a time, on the one
        // thread a card has -- and it was the largest single piece of a round
        // that was not the device.
        stage.xpub.put(s, rows * PUBFEAT, |dst| {
            let mut at = 0;
            for &i in mine {
                let (xp, _, _) = each(i);
                dst[at..at + xp.len()].copy_from_slice(xp);
                at += xp.len();
            }
            at
        })?;
        stage.cards.put(s, mine.len() * CARD_ROWS * NTYPE * TYPE, |dst| {
            let mut at = 0;
            for &i in mine {
                let (_, cd, _) = each(i);
                dst[at..at + cd.len()].copy_from_slice(cd);
                at += cd.len();
            }
            at
        })?;
        stage.card_of_row.put(s, rows, |dst| {
            let (mut at, mut card) = (0, 0i32);
            for &i in mine {
                let (_, _, n) = each(i);
                dst[at..at + n].fill(card);
                at += n;
                card += CARD_ROWS as i32;
            }
            at
        })?;
        let Stage { xpub, cards, card_of_row, .. } = &mut *stage;
        let xpub = xpub.dev.buf.as_ref().expect("staged");
        let cards = cards.dev.buf.as_ref().expect("staged");
        let card_of_row = card_of_row.dev.buf.as_ref().expect("staged");

        // Nothing new to encode: every fresh row repeats a board the solve
        // already holds.
        if rows == 0 {
            let none = self.alloc(0)?;
            return self.keep(calls, mine, &none, &none, pack);
        }
        let cells = rows * N_HEXES;
        let stride = PUBFEAT as i32;
        let (rows_i, cells_i) = (rows as i32, cells as i32);
        let (nhex, ntype, chan, nslot) = (N_HEXES as i32, NTYPE as i32, C as i32, NSLOT as i32);
        let l = &self.layout;
        LEAF_NS[14].fetch_add(mark.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);

        // Tokens: projected pile counts, then the card token and seat on top.
        let mut piles = self.alloc(rows * NTYPE * PILE_COUNTS)?;
        let (off, width) = (OFF_PILES as i32, (NTYPE * PILE_COUNTS) as i32);
        launch!(self, window, rows * NTYPE * PILE_COUNTS, xpub, &mut piles, &rows_i, &stride, &off, &width)?;
        let mut tokens = self.alloc(rows * NTYPE * TYPE)?;
        self.lin(l.pile, &piles, rows * NTYPE, 0.0, &mut tokens)?;
        let seat = self.w.slice(l.seat..l.seat + 2 * TYPE);
        let type_i = TYPE as i32;
        launch!(self, tokens, rows * NTYPE * TYPE, cards, card_of_row, &seat, &mut tokens, &rows_i, &ntype, &type_i, &nslot)?;

        // Stem.
        let mut projected = self.alloc(rows * NTYPE * C)?;
        self.run(l.tok_stem, &tokens, rows * NTYPE, &mut projected)?;
        let mut type_pool = self.alloc(rows * C)?;
        launch!(self, type_pool, rows * C, &projected, &mut type_pool, &rows_i, &ntype, &chan)?;
        let mut loose = self.alloc(rows * LOOSE)?;
        let (off, width) = (OFF_LOOSE as i32, LOOSE as i32);
        launch!(self, window, rows * LOOSE, xpub, &mut loose, &rows_i, &stride, &off, &width)?;
        let mut glob = self.alloc(rows * C)?;
        self.run(l.glob_stem, &loose, rows, &mut glob)?;
        let mut facts = self.alloc(cells * HEX_FACTS)?;
        // Fully written by `k_hex_facts`, one entry a cell.
        let mut occupant = unsafe { self.stream.alloc::<i32>(cells.max(1)) }.map_err(err)?;
        let (hex_ch, hex_facts) = (HEX_CH as i32, HEX_FACTS as i32);
        launch!(self, hex_facts, cells, xpub, &mut facts, &mut occupant, &rows_i, &stride, &nhex, &hex_ch, &hex_facts, &ntype)?;
        let mut x = self.alloc(cells * C)?;
        self.run(l.hex_stem, &facts, cells, &mut x)?;
        let pos = self.w.slice(l.pos..l.pos + N_HEXES * C);
        launch!(self, stem, cells * C, &mut x, &projected, &occupant, &pos, &glob, &type_pool, &cells_i, &nhex, &ntype, &chan)?;

        // The eight residual blocks and the head's input, in one launch with
        // the board resident in shared memory. See `k_trunk`: as separate
        // launches this was eighteen passes over `[cells, C]` of global memory
        // per block, which at the throughput target is more memory bandwidth
        // than the two cards have.
        let width = 2 * C + LOOSE;
        let mut input = self.alloc(rows * width)?;
        let (off, loose_i, blocks_i) = (OFF_LOOSE as i32, LOOSE as i32, BLOCKS as i32);
        // A warp to a hex, twelve of them: `k_trunk` gives each lane `C / 32`
        // channels, so a hex's row is exactly one warp wide and its LayerNorm
        // is a shuffle rather than a barrier.
        // `TRUNK_SPAN` and `TRUNK_MAXH` in the kernel; both are compile-time
        // there because a runtime trip count puts the accumulators in local
        // memory, which cost the kernel a factor of thirty.
        const SLOTS: u32 = 12;
        assert_eq!(C % 32, 0, "k_trunk wants a whole number of warps a row");
        assert_eq!(N_HEXES.div_ceil(SLOTS as usize), 4, "k_trunk holds four hexes a thread");
        let shared = (3 * N_HEXES * C + 3 * C) * 4;
        unsafe {
            self.stream
                .launch_builder(&self.k.trunk)
                .arg(&x).arg(&self.nb).arg(&self.w).arg(&self.wt)
                .arg(&self.b).arg(&self.ln)
                .arg(&self.plan).arg(xpub).arg(&mut input)
                .arg(&rows_i).arg(&nhex).arg(&chan).arg(&blocks_i)
                .arg(&stride).arg(&off).arg(&loose_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (32, SLOTS, 1),
                    shared_mem_bytes: shared as u32,
                })
        }
        .map_err(err)?;
        let mut p = self.alloc(rows * D)?;
        self.run(l.board_out, &input, rows, &mut p)?;
        let mut jp = self.alloc(rows * JW)?;
        self.run(l.join_p, &p, rows, &mut jp)?;

        self.keep(calls, mine, &p, &jp, pack)
    }

    /// Keep what the trunk made, per solve, for the iterations that follow.
    ///
    /// Nothing goes back: the readout, the belief pooling and the policy head
    /// all run here, so a board vector has no reader on the host at all.
    ///
    /// A call whose rows were all transpositions of boards the solve already
    /// holds contributes no boards, and then `p` and `jp` are empty and the
    /// copies below are of nothing. Its rows still arrive: `board_of` points
    /// them at the boards they share, and the belief index is their own.
    fn keep(
        &self,
        calls: &[Call],
        mine: &[usize],
        p: &CudaSlice<f32>,
        jp: &CudaSlice<f32>,
        pack: &mut Pack,
    ) -> Res<()> {
        let mut at = 0;
        let mut g = self.solves.lock();
        for &i in mine {
            let Call::Trunk {
                solve, at: row0, rows: nrows, board_of, boards_at, boards: n, cidx, coff, ..
            } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            let (n, nrows) = (*n, *nrows);
            let b = self.slot(&mut g, *solve);
            if *row0 == 0 {
                // A fresh solve in this slot. Everything the last one left is
                // another tree's, and the pages are worth more to whichever
                // slot is holding a large solve now. This comes before the
                // writes below, not after them.
                b.cells = 0;
                b.host_coff.clear();
                b.host_coff.push(0);
                for a in [&mut b.p, &mut b.jp, &mut b.f, &mut b.g, &mut b.fp] {
                    a.reset();
                }
                b.board_of.reset();
                b.cidx.reset();
                b.coff.reset();
                b.leaf_node.reset();
                b.term.reset();
            }
            b.p.copy(&self.stream, boards_at * D, &p, at * D, n * D)?;
            b.jp.copy(&self.stream, boards_at * JW, &jp, at * JW, n * JW)?;
            let words = pack.words(board_of);
            let dst = b.board_of.plan(&self.stream, *row0, nrows)?;
            pack.piece(dst, *row0 as u32, words, nrows as u32);
            // `coff` arrives relative to this call's own `cidx`, so it is
            // shifted onto the resident index before it is stored. Row zero
            // writes the leading zero; every later call overwrites it with its
            // own first offset, the same number.
            let base = b.cells as u32;
            let shifted: Vec<u32> = coff.iter().map(|x| x + base).collect();
            b.host_coff.extend(shifted.iter().skip(1));
            let words = pack.words(cidx);
            let dst = b.cidx.plan(&self.stream, b.cells, cidx.len())?;
            pack.piece(dst, b.cells as u32, words, cidx.len() as u32);
            let words = pack.words(&shifted);
            let dst = b.coff.plan(&self.stream, 2 * row0, shifted.len())?;
            pack.piece(dst, 2 * *row0 as u32, words, shifted.len() as u32);
            b.cells += cidx.len();
            b.rows = row0 + nrows;
            at += n;
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
        let mut stage = self.host.lock();
        stage.phi.put(s, n * CFEAT, |dst| {
            let mut at = 0;
            for &i in mine {
                let (ph, _, _, _) = each(i);
                dst[at..at + ph.len()].copy_from_slice(ph);
                at += ph.len();
            }
            at
        })?;
        stage.owner.put(s, n, |dst| {
            let (mut at, mut base) = (0, 0u32);
            for &i in mine {
                let (_, ow, cd, _) = each(i);
                for (d, &q) in dst[at..at + ow.len()].iter_mut().zip(ow) {
                    *d = q + base;
                }
                at += ow.len();
                base += (cd.len() / (NTYPE * TYPE)) as u32;
            }
            at
        })?;
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
        let Stage { phi, owner, cfg_cards, .. } = &mut *stage;
        let phi = phi.dev.buf.as_ref().expect("staged");
        let owner = owner.dev.buf.as_ref().expect("staged");
        let cards = cfg_cards.dev.buf.as_ref().expect("staged");
        let l = &self.layout;
        let (n_i, nslot, cfeat) = (n as i32, NSLOT as i32, CFEAT as i32);
        let (ntype, type_i, pool_i) = (NTYPE as i32, TYPE as i32, POOL as i32);

        let width = 3 + TYPE;
        let mut slots = self.alloc(n * NSLOT * width)?;
        launch!(self, cfg_slots, n * NSLOT * width, phi, owner, cards, &mut slots, &n_i, &nslot, &cfeat, &ntype, &type_i)?;
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
        self.run(l.cfg_m, cards, views * NTYPE, &mut bag)?;
        launch!(self, bag, n * POOL, &bag, phi, owner, &mut g, &n_i, &nslot, &ntype, &cfeat, &pool_i)?;


        // All three stay. The readout, the belief pooling and the policy head
        // all run here, so none of them has a reader on the host.
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
                b.fp.copy(&self.stream, base * D, &fp, at * D, k * D)?;
                at += k;
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
        let mut widest = 0u32;
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
                widest = widest.max(q.nc);
            }
            desc.extend_from_slice(a);
            cells.extend_from_slice(c);
        }
        let (m, na) = (node.len(), desc.len() / 5);
        let batch = self.lay(&solves)?;
        let flat: Vec<u32> = [
            &part[..], &node, &row, &act_at, &cell_at, &inv_t, &act_node, &desc, &cells,
        ]
        .concat();
        let dev = {
            let mut stage = self.host.lock();
            Self::alone(&self.stream, &mut stage.prime, &flat)?
        };
        let at = |k: usize, n: usize| dev.slice(k..k + n);
        let (part_d, node_d, row_d) = (at(0, m), at(m, m), at(2 * m, m));
        let (act_at_d, cell_at_d, inv_d) = (at(3 * m, m), at(4 * m, m), at(5 * m, m));
        let act_node_d = at(6 * m, na);
        let desc_d = at(6 * m + na, 5 * na);
        let cells_d = at(6 * m + 6 * na, cells.len());

        let l = &self.layout;
        let (na_i, m_i, d_i, aw_i) = (na as i32, m as i32, D as i32, AW as i32);
        let (nkinds, nslot, nhex, afeat) = (
            crate::actions::N_KINDS as i32,
            NSLOT as i32,
            N_HEXES as i32,
            AFEAT as i32,
        );
        let mut feat = self.alloc(na * AFEAT)?;
        launch!(self, act_feats, na * AFEAT, &desc_d, &mut feat, &na_i, &nkinds, &nslot, &nhex, &afeat)?;
        let mut z = self.alloc(na * AW)?;
        self.run(l.act_in, &feat, na, &mut z)?;
        let mut boards = self.alloc(m * D)?;
        launch!(self, act_boards, m * D, &batch.trees, &part_d, &row_d, &mut boards, &m_i, &d_i)?;
        let mut proj = self.alloc(m * AW)?;
        self.run(l.act_board, &boards, m, &mut proj)?;
        launch!(self, act_add, na * AW, &mut z, &proj, &act_node_d, &na_i, &aw_i)?;
        self.norm(l.norms[LN_ACT], na, true, &mut z)?;
        let mut e = self.alloc(na * D)?;
        self.run(l.act_out, &z, na, &mut e)?;

        // A warp to a config of a primed node: the dot is the warp's, so a row
        // of `f_p` and a row of `e` are each read once and coalesced.
        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (widest.div_ceil(WARPS).max(1), m as u32, 1),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.k.prior)
                .arg(&batch.trees).arg(&part_d).arg(&node_d).arg(&row_d)
                .arg(&act_at_d).arg(&cell_at_d).arg(&cells_d).arg(&e)
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
                // Regrets, visits and the strategy sum accumulate over a solve,
                // so the next solve to take this slot must not inherit them.
                // The tree's own arrays go back too: the caller rewinds what it
                // has told the card about, and holding the pages would cost the
                // card the worst case in every slot at once.
                for a in b.tree.pools() {
                    a.reset();
                }
                b.tree.rvd_p.reset();
                b.tree.draw_p.reset();
                for a in [&mut b.reach, &mut b.vals, &mut b.cur, &mut b.regret,
                          &mut b.sum, &mut b.qval, &mut b.visits, &mut b.prior] {
                    a.reset();
                }
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
            b.regret.fit(s, *ncells)?;
            b.sum.fit(s, *ncells)?;
            b.qval.fit(s, *ncells)?;
            b.visits.fit(s, *ncells)?;
            b.reach.fit(s, *nreach)?;
            b.nvals = *nvals;
            b.vals.fit(s, 2 * *nvals)?;
        }
        Ok(())
    }

    /// Send a round's writes: one buffer up, one kernel to place the pieces.
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
        stage.at.put(s, pack.at.len(), copy(&pack.at))?;
        stage.src.put(s, pack.src.len(), copy(&pack.src))?;
        stage.start.put(s, pack.sum.len(), copy(&pack.sum))?;
        stage.dst.put(s, pack.dst.len(), copy(&pack.dst))?;
        let Stage { blob, dst, at, src, start, .. } = &mut *stage;
        let blob = blob.dev.buf.as_ref().expect("staged");
        let dst = dst.dev.buf.as_ref().expect("staged");
        let at = at.dev.buf.as_ref().expect("staged");
        let src = src.dev.buf.as_ref().expect("staged");
        let start = start.dev.buf.as_ref().expect("staged");
        let (pieces, total_i) = (pack.dst.len() as i32, moved as i32);
        launch!(self, scatter, moved as usize, blob, dst, at, src, start, &pieces, &total_i)
    }

    /// Stage a small array through its page-locked buffer into a device buffer
    /// of its own. A round can hold three batches at once, so these cannot
    /// share one.
    fn alone<T>(s: &Arc<CudaStream>, h: &mut Host<T>, v: &[T]) -> Res<CudaSlice<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy,
    {
        h.fill(s, v.len(), copy(v))?;
        h.send_new(s)
    }

    /// Lay a set of solves out as one batch.
    fn lay(&self, solves: &[usize]) -> Res<Batch> {
        let mut stage = self.host.lock();
        // These are gathered into ordinary vectors first because the shape of
        // the batch is not known until every solve has been read. They are
        // small -- a descriptor apiece and an entry per leaf row.
        let (mut desc, mut coff): (Vec<u64>, Vec<u32>) = (Vec::new(), vec![0]);
        let (mut part_of_row, mut local_row, mut base): (Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut rows, mut cells, mut nterm) = (0usize, 0u32, 0usize);
        let mut wide: Vec<u32> = Vec::new();
        // `upto[k]` is the batch made of the first `k` solves. Running an
        // iteration that only some of them want is then a matter of a shorter
        // grid, with nothing rebuilt.
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
                while wide.len() + 1 < b.level_start.len() {
                    wide.push(0);
                }
                for (l, w) in b.level_start.windows(2).zip(wide.iter_mut()) {
                    *w = (*w).max(l[1] - l[0]);
                }
                upto.push(Prefix {
                    parts: part as u32 + 1,
                    rows,
                    wide: wide.clone(),
                    nterm,
                });
            }
        }
        let s = &self.stream;
        Ok(Batch {
            trees: Self::alone(s, &mut stage.desc, &desc)?,
            coff: Self::alone(s, &mut stage.lcoff, &coff)?,
            part: Self::alone(s, &mut stage.part_of_row, &part_of_row)?,
            local: Self::alone(s, &mut stage.local_row, &local_row)?,
            base: Self::alone(s, &mut stage.base, &base)?,
            upto,
            parts: solves.len() as u32,
            cells: cells as usize,
        })
    }

    /// A value pass under the reference strategy, for a whole batch: the
    /// reaches, the network at every leaf, the terminals, and backpropagation
    /// that averages rather than updating regret. This is what a solve's
    /// targets are read off.
    fn value_pass(&self, b: &Batch) -> Res<()> {
        let all = b.all();
        self.reaches(&b.trees, all, 1, false, 0)?;
        self.network(b, all)?;
        self.terminals(&b.trees, all)?;
        self.backprop(&b.trees, all, 1, 0, Cfr::LINEAR)
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
        let b = self.lay(&solves).map_err(at("lay"))?;
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
        self.stage(4, || self.reaches(&b.trees, b.all(), 0, false, 0)).map_err(at("reach"))?;
        for iter in 0..rounds {
            let live = order
                .iter()
                .position(|&i| Self::asked(&calls[i]).0 <= iter)
                .unwrap_or(order.len());
            let p = &b.upto[live];
            let it = iter as i32;
            self.network(&b, p).map_err(at("net"))?;
            self.stage(8, || self.terminals(&b.trees, p)).map_err(at("terminals"))?;
            self.stage(9, || self.backprop(&b.trees, p, 0, it, k)).map_err(at("backprop"))?;
            // The regret update moved both players' strategies, so the reaches
            // the next iteration reads are stale until they are pushed down
            // again -- and the average strategy accumulates against those.
            self.stage(4, || self.reaches(&b.trees, p, 0, true, it)).map_err(at("avg"))?;
            // The phase reads the Q this iteration has just formed, so it
            // belongs inside the loop: a round that runs several iterations
            // samples several times and the host grows all of them at once.
            if sims > 0 {
                self.stage(10, || {
                    self.expand(&b.trees, b.parts, sims, puct, iter, rounds)
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
        let all = self.lay(&solves)?;
        let touched: Vec<i32> = mine
            .iter()
            .map(|&i| match &calls[i] {
                Call::Read { touched, .. } => (touched[0] as i32) | ((touched[1] as i32) << 1),
                _ => unreachable!("read shard holds only read calls"),
            })
            .collect();
        let touched_d = Self::alone(&self.stream, &mut self.host.lock().touched, &touched)?;
        self.finish(&all.trees, all.all(), &touched_d)?;

        // Only a solve that is collected pays for the values. Those are laid
        // out as their own batch so the pass runs over them alone.
        let want: Vec<usize> = mine
            .iter()
            .filter(|&&i| matches!(&calls[i],
                Call::Read { vals_at, .. } if vals_at[0].1 > 0 || vals_at[1].1 > 0))
            .map(|&i| calls[i].solve())
            .collect();
        if !want.is_empty() {
            self.value_pass(&self.lay(&want)?)?;
        }

        let g = self.solves.lock();
        for &i in mine {
            let Call::Read { solve, vals_at, policy_at, reach_at, .. } = &calls[i] else {
                unreachable!("read shard holds only read calls")
            };
            let s = &g[*solve];
            let mut root = Vec::new();
            for &(at, n) in vals_at {
                root.extend(self.slice(&s.vals, at as usize, n as usize)?);
            }
            let policy = self.slice(&s.sum, policy_at.0 as usize, policy_at.1 as usize)?;
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
    fn reaches(&self, trees: &CudaSlice<u64>, p: &Prefix, avg: i32, also_avg: bool, iter: i32)
        -> Res<()> {
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
        for level in 1..p.wide.len() {
            if p.wide[level] == 0 {
                continue;
            }
            let level_i = level as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.reach_sweep)
                    .arg(trees).arg(&level_i).arg(&avg).arg(&sum).arg(&iter)
                    .launch_unit(Self::grid(p.wide[level], p.parts))
            }
            .map_err(err)?;
        }
        // The root's own row. It is not a child of anything, so the sweep never
        // reaches it and its share of the sum is a launch of its own.
        if also_avg && !p.wide.is_empty() {
            let level_i = 0i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.avg_block)
                    .arg(trees).arg(&level_i).arg(&iter)
                    .launch_unit(Self::grid(p.wide[0], p.parts))
            }
            .map_err(err)?;
        }
        Ok(())
    }

    /// Value backpropagation up the levels, for one traverser. `avg` averages
    /// under the reference strategy and leaves the regrets alone.
    fn backprop(&self, trees: &CudaSlice<u64>, p: &Prefix, avg: i32, iter: i32, k: Cfr)
        -> Res<()> {
        for level in (0..p.wide.len()).rev() {
            if p.wide[level] == 0 {
                continue;
            }
            let level_i = level as i32;
            let mut cfg = Self::grid(p.wide[level], p.parts);
            cfg.grid_dim.2 = 2;
            unsafe {
                self.stream
                    .launch_builder(&self.k.backprop_sweep)
                    .arg(trees).arg(&level_i).arg(&avg).arg(&iter)
                    .arg(&k.alpha).arg(&k.beta).arg(&k.gamma).arg(&k.predict)
                    .launch_unit(cfg)
            }
            .map_err(err)?;
        }
        Ok(())
    }

    /// The network at every leaf of the round, for both traversers at once.
    ///
    /// Normalise the beliefs, pool them, run the join, read the values out into
    /// each solve's own value arena. The beliefs and the pooling do not depend
    /// on which seat is asking, so they run once; the join and the readout do,
    /// and run over a batch of twice the rows rather than twice.
    #[allow(clippy::too_many_arguments)]
    fn network(&self, b: &Batch, p: &Prefix) -> Res<()> {
        let (trees, part_d, local_d, base_d, coff_d) =
            (&b.trees, &b.part, &b.local, &b.base, &b.coff);
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
        let s = &self.stream;

        // The beliefs are normalised once for the whole round: `w` is indexed
        // by the round's own cell offsets and `mass` by its rows, so neither
        // belongs to a tile.
        sc.w.room(s, b.cells)?;
        sc.mass.room(s, 2 * stride)?;
        {
            let Scratch { w, mass, .. } = &mut *sc;
            let (w, mass) = (w.buf.as_mut().unwrap(), mass.buf.as_mut().unwrap());
            self.stage(5, || {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.beliefs)
                        .arg(trees).arg(part_d).arg(local_d).arg(coff_d)
                        .arg(&mut *w).arg(&mut *mass).arg(&stride_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: ((stride as u32).div_ceil(8).max(1), 2, 1),
                            block_dim: (32, 8, 1),
                            shared_mem_bytes: 0,
                        })
                }
                .map_err(err)
            })?;
        }

        // Everything after that is a tile of leaves at a time. The pass's
        // intermediates are 5,640 bytes a leaf row, so sizing them by the whole
        // round cost a gigabyte a lane -- and lanes are what solves in flight
        // are now bounded by.
        let tile = TILE.min(stride);
        sc.pooled.room(s, 2 * tile * POOL)?;
        sc.h.room(s, 2 * tile * D)?;
        sc.z.room(s, 2 * tile * JW)?;
        sc.input.room(s, 2 * tile * JOIN_IN)?;
        sc.t.room(s, 2 * tile * JW)?;
        let Scratch { w, mass, pooled, h, z, input, t, .. } = &mut *sc;
        let (w, mass) = (w.buf.as_mut().unwrap(), mass.buf.as_mut().unwrap());
        let pooled = pooled.buf.as_mut().unwrap();
        let (h, z) = (h.buf.as_mut().unwrap(), z.buf.as_mut().unwrap());
        let (input, t) = (input.buf.as_mut().unwrap(), t.buf.as_mut().unwrap());

        let mut q0 = 0usize;
        while q0 < stride {
            let n = tile.min(stride - q0);
            let (q0_i, n_i) = (q0 as i32, n as i32);
            let rows = 2 * n;
            let (rows_i, queries_i) = (rows as i32, rows as i32);
            let jw_i = JW as i32;
            self.stage(5, || {
                unsafe {
                    self.stream
                        .launch_builder(&self.k.belief_pool)
                        .arg(trees).arg(part_d).arg(base_d).arg(coff_d).arg(&*w)
                        .arg(&mut *pooled).arg(&(2 * q0_i)).arg(&queries_i).arg(&pool_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (POOL as u32, 8, 1),
                            shared_mem_bytes: 8 * POOL as u32 * 4,
                        })
                }
                .map_err(err)
            })?;

            // The join cache, gathered out of the solves straight into the
            // buffer the residual chain accumulates onto.
            self.stage(6, || {
                let one = 1i32;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.gather)
                        .arg(trees).arg(part_d).arg(local_d).arg(&one)
                        .arg(&mut *z).arg(&rows_i).arg(&jw_i).arg(&q0_i).arg(&n_i)
                        .launch_unit(rows_of(rows, JW))
                }
                .map_err(err)?;
                unsafe {
                    self.stream
                        .launch_builder(&self.k.join_input)
                        .arg(&*pooled).arg(&mut *input).arg(&rows_i).arg(&pool_i).arg(&n_i)
                        .launch_unit(rows_of(rows, POOL))
                }
                .map_err(err)?;
                self.lin(l.join_b, &*input, rows, 1.0, &mut *z)?;
                // A residual block is a norm and a multiply. The norm reads `z`,
                // adds what the stream is owed and writes the scratch in one
                // pass; the multiply accumulates straight back into `z`.
                for i in 0..JBLOCKS {
                    self.norm_owed(l.norms[LN_JOIN + i], rows, true, z, t, Some(i * JW))?;
                    self.lin(l.join_w[i], &*t, rows, 1.0, &mut *z)?;
                }
                self.norm_owed(l.norms[LN_JOUT], rows, true, z, t, Some(JBLOCKS * JW))?;
                // Nothing seeds `h`: the readout adds the board vector as it
                // reads, so the multiply owns the buffer outright.
                self.lin(l.join_out, &*t, rows, 0.0, &mut *h)
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

    /// The expansion phase: `sims` trajectories a solve, and the leaf each one
    /// reached. The simulations of one phase run in order, because each counts
    /// the visits it passes and the next is meant to see them -- and so do the
    /// phases, which is why the round's `iters` phases share one buffer and
    /// one download.
    fn expand(&self, trees: &CudaSlice<u64>, parts: u32, sims: usize, puct: f32,
              iter: usize, iters: usize) -> Res<()> {
        let each = parts as usize * sims;
        let mut sc = self.scratch.lock();
        let out = sc.leaves.room(&self.stream, (iters * each).max(1))?;
        let mut view = out.slice_mut(iter * each..(iter + 1) * each);
        let (parts_i, sims_i) = (parts as i32, sims as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.expand)
                .arg(trees).arg(&mut view).arg(&parts_i).arg(&sims_i).arg(&puct)
                .arg(&(iter as i32))
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
        let out = sc.leaves.room(&self.stream, n)?;
        self.stream.memcpy_dtov(&out.slice(0..n)).map_err(err)
    }

    /// The reference strategy, once the tree has stopped growing.
    /// `touched` is per solve: which players' running sums have moved, or `-1`
    /// for a solve that is not asking for this at all.
    fn finish(&self, trees: &CudaSlice<u64>, p: &Prefix, touched: &CudaSlice<i32>) -> Res<()> {
        for level in 0..p.wide.len() {
            if p.wide[level] == 0 {
                continue;
            }
            let level_i = level as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.k.finish)
                    .arg(trees).arg(&level_i).arg(touched)
                    .launch_unit(Self::grid(p.wide[level], p.parts))
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
    xpub: Wire<f32>,
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
    /// The batch descriptors. These take a device buffer of their own per call,
    /// because a round can hold three batches at once.
    desc: Host<u64>,
    lcoff: Host<u32>,
    part_of_row: Host<i32>,
    local_row: Host<i32>,
    base: Host<i32>,
    touched: Host<i32>,
    /// Everything `Card::priors` sends: the per-node arrays and then the two
    /// pools, concatenated. Floats travel as their bits, as everything else a
    /// round scatters does.
    prime: Host<u32>,
}

/// The intermediates of one pass, by role. Each is fully written before it is
/// read, so they are grown rather than cleared.
#[derive(Default)]
struct Scratch {
    /// `[cells]` normalised beliefs, and `[2, rows]` reach mass per player.
    w: Arr<f32>,
    mass: Arr<f32>,
    /// `[2 * rows, POOL]` the pooled belief block.
    pooled: Arr<f32>,
    /// `[rows, D]` the head, `[rows, JW]` the residual stream, and the two
    /// buffers the join's input and its per-block scratch need.
    h: Arr<f32>,
    z: Arr<f32>,
    input: Arr<f32>,
    t: Arr<f32>,
    /// `[parts * sims]` the leaves an expansion phase sampled.
    leaves: Arr<u32>,
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
    /// Whether the subtree under a node has no expandable leaf left. The
    /// expansion trajectories will not descend into one.
    exhausted: Arr<u32>,
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
    cell_val: Arr<u32>,
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
    /// The append-only pools, which a fresh solve rewinds.
    fn pools(&mut self) -> [&mut Arr<u32>; 13] {
        [
            &mut self.child,
            &mut self.legal_off,
            &mut self.legal_child,
            &mut self.legal_trans,
            &mut self.cell_row,
            &mut self.cell_val,
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
