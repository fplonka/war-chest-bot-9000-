//! The GPU service thread: owns one CUDA device, keeps the live set of
//! solves resident, and advances all of them with ticks.
//!
//! The design in one paragraph. Three streams: `compute` runs ticks, `build`
//! admits jobs, `download` (owned by a separate thread) returns results — so
//! the service thread never waits on the device. All memory is cut from
//! slabs at startup; after that the service makes no CUDA allocations. A
//! tick is one fixed launch sequence over the whole live set: kernels are
//! one block per solve, read their stage from the descriptor, and return
//! early when a phase does not apply, so iterate, value and carry solves
//! share the same launches and nothing is staged per tick — the launch
//! headers change only when the live set's membership does. All per-tick
//! state (iteration counter, traverser, stage, DCFR discounts) advances on
//! the device in `advance_state`; the host runs the same arithmetic on
//! shadow copies to know when results are due.
//!
//! Admission is batched: up to `MAX_BATCH` jobs upload together and their
//! card/trunk/holding towers run as one GEMM chain over the packed batch,
//! then scatter kernels place each solve's h0 rows and z/g tables. The
//! compute stream waits on one build event, so ticks and builds overlap.
//!
//! The network shape (tower widths and depths) comes from the checkpoint;
//! kernels are NVRTC-compiled for it at startup. New weights of the same
//! shape apply when the live set drains; a different shape is an error —
//! restart the service.

use std::sync::mpsc;
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::CudaBlas;
use cudarc::driver::safe::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, LaunchConfig, PinnedHostSlice,
};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc;

use crate::net::V3Layout;
use crate::serialize::{Job, JobMeta};

use super::client::{Cmd, GpuClient, Trip1, Trip2};
use super::layout::{
    arena_offsets, cuda_preamble, pack_tables_into, packed_table_len, Arena, Derived, Desc, Sizes,
    N_ARENAS, STAGE_CARRY, STAGE_DRAIN, STAGE_ITERATE, STAGE_VALUE,
};

/// Live-set capacity, in solve slots. The service's throughput rises with the
/// resident set right up to the row pool's limit, and generation puts
/// `workers * WARCHEST_GEN_PER` games in flight, so this has to sit above what
/// a 72-thread box can offer.
pub(super) const CAP: usize = 1536;
/// Row-pool capacity, in network rows; the real bound on the live set. A
/// depth-2 random-draft solve carries ~850 rows, so the default holds around
/// nine hundred resident solves. The pools it sizes are the service's largest
/// device allocation (about 4 KiB per row), which is why a smaller CI card
/// can turn it down.
pub(super) fn max_rows() -> usize {
    std::env::var("WARCHEST_GPU_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(768 * 1024)
        .max(4096)
}
/// Threads per block.
const BLOCK: u32 = 256;
/// Admission batch bounds: jobs and distinct configs. One depth-2 random-draft
/// solve can legitimately carry >100k leaves, so the per-batch row bound is
/// the whole row pool rather than something a single valid job could exceed.
const MAX_BATCH: usize = 32;
const MAX_BATCH_ROWS: usize = 256 * 1024;
const MAX_BATCH_CFG: usize = 64 * 1024;
/// Trip 2 returns at most 15 kept snapshots and two config supports. Gather
/// its strided device rows here before one pinned host transfer.
const MAX_DOWNLOAD_FLOATS: usize = 2 * 16 * MAX_BATCH_CFG;
/// Keep enough complete ticks queued to hide host submission time, without
/// letting the host state machine run seconds ahead of the device.  A tick is
/// already dozens of kernels plus both head GEMMs, so two is ample depth.
/// Tunable because the queue depth is also the window in which a recycled
/// slot could be seen by an older launch — turning it down is how you test
/// whether a fault lives there.
fn max_queued_ticks() -> usize {
    std::env::var("WARCHEST_GPU_QUEUED_TICKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2usize)
        .clamp(1, 8)
}
/// Ints per batch-map entry: slot, row0-in-batch, nrows, cfg0-in-batch, ncfg.
const BMAP_INTS: usize = 5;
/// Depth-2 War Chest trees are shallow even when tactic micro-actions are
/// expanded. Keeping all level prefixes resident makes the hot path upload-
/// free; reject a malformed/pathological job instead of silently truncating.
const MAX_LEVELS: usize = 64;

/// Slab sizes, in MiB, overridable by environment (the T4 CI box is small).
fn env_mb(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        << 20
}

// ------------------------------------------------------------------ context

/// The `Ctx` struct of kernels.cu: weight pointers, row pools, build scratch.
#[repr(C)]
#[derive(Clone, Copy)]
struct Ctx {
    card_w: [*const f32; 8],
    card_b: [*const f32; 8],
    wid: *const f32,
    pile_w: *const f32,
    pile_b: *const f32,
    pub_w: [*const f32; 8],
    pub_b: [*const f32; 8],
    pub_lnw: [*const f32; 8],
    pub_lnb: [*const f32; 8],
    pub_out_w: *const f32,
    pub_out_b: *const f32,
    wb: *const f32,
    ln1w: *const f32,
    ln1b: *const f32,
    hmlp_w: [*const f32; 8],
    hmlp_b: [*const f32; 8],
    wu_w: *const f32,
    wu_b: *const f32,
    slot_w: [*const f32; 8],
    slot_b: [*const f32; 8],
    slot_out_w: *const f32,
    slot_out_b: *const f32,
    res_aw: [*const f32; 4],
    res_ab: [*const f32; 4],
    res_bw: [*const f32; 4],
    res_bb: [*const f32; 4],
    wg_w: *const f32,
    wg_b: *const f32,
    h0: *mut f32,
    xb: *mut f32,
    h: *mut f32,
    h2: *mut f32,
    u: *mut f32,
    bx: *mut f32,
    bh: *mut f32,
    bh2: *mut f32,
    bg: *mut f32,
    bmap: *const i32,
}

unsafe impl cudarc::driver::DeviceRepr for Ctx {}
unsafe impl cudarc::driver::ValidAsZeroBits for Ctx {}
unsafe impl Send for Ctx {}

impl Default for Ctx {
    fn default() -> Ctx {
        // SAFETY: every field is an integer or a device pointer.
        unsafe { std::mem::zeroed() }
    }
}

/// The `Group` struct of kernels.cu.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GroupDev {
    slots: *const i32,
    /// Cumulative task counts, one entry per live slot plus a sentinel. Null
    /// for the original one-block-per-solve and admission groups.
    prefix: *const i32,
    n: i32,
    mode: i32,
    p_player: i32,
    level: i32,
    total: i32,
}

unsafe impl cudarc::driver::DeviceRepr for GroupDev {}
unsafe impl cudarc::driver::ValidAsZeroBits for GroupDev {}
unsafe impl Send for GroupDev {}

/// The tick's launch headers, by fixed slot. They point at the live list and
/// change only when membership does.
#[derive(Clone, Copy)]
enum Hdr {
    Plain = 0,
    ReadIt,    // p_player = -1 (iterate readout/backprop)
    ReadV0,    // p_player = 0 (value pass)
    ReadV1,    // p_player = 1
    ReachVC,   // mode = 0 (value + carry)
    ReachIt,   // mode = 1 (iterate)
    ReachSnap, // mode = 2 (kept intermediate average)
    HmlpBase,  // + 2*k: layer k into h2 (mode 1) or h (mode 0)
}
const N_HDRS: usize = Hdr::HmlpBase as usize + 16;

/// Maps used by the flattened sweep kernels. Each prefix section has a fixed
/// `(CAP + 1)` stride so its device address never changes.
#[derive(Clone, Copy)]
enum FlatHdr {
    LeafRows = 0, // head/belief rows for iterate + value solves
    ReadIt,       // network + terminal leaves for iterate solves
    ReadV,        // network + terminal leaves for value solves
    NodesIt,      // current-traverser decision-node configs for RM/average
    LevelItBase,  // + public-tree level, iterate reach passes
}
/// Storage capacity. The live prefixes themselves are packed at the current
/// maximum level count, rather than uploading five 64-level padded sections
/// on every CFR iteration. The sections are: iterate reach, iterate backprop,
/// value backprop for player 0, the same for player 1, and value reach. The
/// two value-backprop sections cannot share one: they count *configs*, and a
/// node's config count differs between the players.
const N_FLAT_HDRS: usize = FlatHdr::LevelItBase as usize + 5 * MAX_LEVELS;
const FLAT_PREFIX_STRIDE: usize = CAP + 1;

/// Where each per-level flat-header section starts, for the current maximum
/// level count. `end` is one past the last live header.
#[derive(Clone, Copy)]
struct FlatBases {
    reach_it: usize,
    back_it: usize,
    back_v0: usize,
    back_v1: usize,
    reach_v: usize,
    end: usize,
}

/// How far into a tick to run; the phase-oracle test stops early.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Step {
    Build,
    None,
    Head,
    Readout,
    Backprop,
    Regret,
    Propagate,
    Average,
    All,
}

// ------------------------------------------------------------------ slabs

/// First-fit suballocation over one flat range. Every device blob — tables,
/// arenas, network rows — is a range from one of these, so the service never
/// calls the CUDA allocator after startup.
struct Slab {
    used: Vec<(usize, usize)>,
    cap: usize,
    high: usize,
}

impl Slab {
    fn new(cap: usize) -> Slab {
        Slab {
            used: Vec::new(),
            cap,
            high: 0,
        }
    }
    fn fits(&self, n: usize) -> bool {
        self.gap(Self::span(n)).is_some()
    }
    /// A range is freed by its start address, so every live range needs a
    /// start of its own. A zero-length one would sit on top of whichever
    /// range comes next, and freeing it would free that range as well — two
    /// solves would then be handed the same rows. A subgame whose leaves are
    /// all terminal asks for zero network rows, so this is reachable.
    fn span(n: usize) -> usize {
        n.max(1)
    }
    fn gap(&self, n: usize) -> Option<(usize, usize)> {
        let mut at = 0;
        for (i, &(start, len)) in self.used.iter().enumerate() {
            if start - at >= n {
                return Some((at, i));
            }
            at = start + len;
        }
        (at + n <= self.cap).then_some((at, self.used.len()))
    }
    fn alloc(&mut self, n: usize) -> Option<usize> {
        let n = Self::span(n);
        let (at, i) = self.gap(n)?;
        self.used.insert(i, (at, n));
        self.high = self.high.max(at + n);
        Some(at)
    }
    fn free(&mut self, at: usize) {
        let before = self.used.len();
        self.used.retain(|&(s, _)| s != at);
        if self.used.len() + 1 != before {
            eprintln!(
                "gpu: slab free of {at} removed {} ranges, not 1 — the allocator's \
                 starts are no longer unique",
                before - self.used.len()
            );
        }
        self.high = self.used.iter().map(|&(s, n)| s + n).max().unwrap_or(0);
    }
}

// ------------------------------------------------------------------ weights

struct Weights {
    dims: Vec<usize>,
    layout: V3Layout,
    _w: CudaSlice<f32>,
    _b: CudaSlice<f32>,
    _ln: CudaSlice<f32>,
    ctx: Ctx,
}

impl Weights {
    fn upload(
        stream: &Arc<CudaStream>,
        dims: &[usize],
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Weights, String> {
        let l = V3Layout::new(dims)?;
        if w.len() != l.w_len || b.len() != l.b_len || ln.len() != l.ln_len {
            return Err(format!(
                "gpu: weight sizes {}/{}/{} do not match dims {dims:?}",
                w.len(),
                b.len(),
                ln.len()
            ));
        }
        if l.card.len() > 8
            || l.pub_lin.len() > 8
            || l.hmlp.len() > 8
            || l.slot.len() > 8
            || l.res.len() > 4
        {
            return Err("gpu: tower deeper than the kernel pointer tables (8/4)".into());
        }
        let wb = htod(stream, &w)?;
        let bb = htod(stream, &b)?;
        let lb = htod(stream, &ln)?;
        let (wp, bp, lp) = (ptr(stream, &wb), ptr(stream, &bb), ptr(stream, &lb));
        let w_at = |o: usize| unsafe { wp.add(o) };
        let b_at = |o: usize| unsafe { bp.add(o) };
        let l_at = |o: usize| unsafe { lp.add(o) };
        let mut ctx = Ctx::default();
        for (k, s) in l.card.iter().enumerate() {
            ctx.card_w[k] = w_at(s.w);
            ctx.card_b[k] = b_at(s.b);
        }
        ctx.wid = w_at(l.wid);
        ctx.pile_w = w_at(l.pile.w);
        ctx.pile_b = b_at(l.pile.b);
        for (k, s) in l.pub_lin.iter().enumerate() {
            ctx.pub_w[k] = w_at(s.w);
            ctx.pub_b[k] = b_at(s.b);
            ctx.pub_lnw[k] = l_at(l.pub_ln[k].0);
            ctx.pub_lnb[k] = l_at(l.pub_ln[k].1);
        }
        ctx.pub_out_w = w_at(l.pub_out.w);
        ctx.pub_out_b = b_at(l.pub_out.b);
        ctx.wb = w_at(l.wb);
        ctx.ln1w = l_at(l.ln1.0);
        ctx.ln1b = l_at(l.ln1.1);
        for (k, s) in l.hmlp.iter().enumerate() {
            ctx.hmlp_w[k] = w_at(s.w);
            ctx.hmlp_b[k] = b_at(s.b);
        }
        ctx.wu_w = w_at(l.wu.w);
        ctx.wu_b = b_at(l.wu.b);
        for (k, s) in l.slot.iter().enumerate() {
            ctx.slot_w[k] = w_at(s.w);
            ctx.slot_b[k] = b_at(s.b);
        }
        ctx.slot_out_w = w_at(l.slot_out.w);
        ctx.slot_out_b = b_at(l.slot_out.b);
        for (k, (a, bbk)) in l.res.iter().enumerate() {
            ctx.res_aw[k] = w_at(a.w);
            ctx.res_ab[k] = b_at(a.b);
            ctx.res_bw[k] = w_at(bbk.w);
            ctx.res_bb[k] = b_at(bbk.b);
        }
        ctx.wg_w = w_at(l.wg.w);
        ctx.wg_b = b_at(l.wg.b);
        Ok(Weights {
            dims: dims.to_vec(),
            layout: l,
            _w: wb,
            _b: bb,
            _ln: lb,
            ctx,
        })
    }
}

// ------------------------------------------------------------------ kernels

macro_rules! kernels {
    ($($name:ident),* $(,)?) => {
        struct Kernels { $($name: CudaFunction,)* }
        impl Kernels {
            fn load(m: &Arc<CudaModule>) -> Result<Kernels, String> {
                Ok(Kernels { $($name: m.load_function(stringify!($name))
                    .map_err(|e| format!("kernel {}: {e:?}", stringify!($name)))?,)* })
            }
        }
    };
}

kernels! {
    hmlp_act, reach_prop, collect_root, advance_state,
    belief_sums_flat, head_entry_flat, readout_flat, backprop_level_flat,
    regret_match_flat, reach_seed_flat, reach_level_flat, average_flat,
    snapshot_beliefs_flat, gather_trip2,
    pack_cards, bias_act, cards_finish, pile_pe, assemble, pack_piles,
    trunk_norm, scatter_h0, holding_in, slot_sum, scatter_zg, init_strategy,
    seed_avg, seed_snapshot_beliefs, abi_probe,
}

// ------------------------------------------------------------------ solves

/// One resident solve: device ranges, the host shadow of the device state
/// machine, and the reply channels.
struct Solve {
    id: u64,
    meta: JobMeta,
    tbl_at: usize,
    arena_at: usize,
    row0: usize,
    nrows: usize,
    aoff: [u32; N_ARENAS + 1],
    /// The admission descriptor. The device mutates its own copy; the host
    /// mirror is rebuilt from the counters below when a field must change
    /// (trip 2's exit leaf), so the two never disagree.
    desc: Desc,
    /// Network leaf ids and their packed config spans. Trip 2 maps the walk's
    /// chosen tree node to the already-computed snapshot-belief rows.
    snap_leaves: Vec<u32>,
    snap_coff: Vec<u32>,
    snapshot_configs: usize,
    /// Public nodes in each BFS level; used to rebuild global flattened work
    /// prefixes when live-set membership changes.
    level_counts: Vec<usize>,
    /// Configs summed over each player's decision nodes: what regret matching
    /// and average-strategy accumulation launch one thread each for.
    decision_cfg_counts: [usize; 2],
    /// The same for the backward sweep, per level and per player. Backprop
    /// walks the traverser's configs at every non-leaf node of a level.
    sweep_cfg_level_counts: [Vec<usize>; 2],
    // Host shadow of the device state machine (advance_state).
    stage: i32,
    t: usize,
    step: usize,
    nroots: usize,
    nsnaps: usize,
    nc_root: [usize; 2],
    ncells: usize,
    trip1: Option<(usize, mpsc::Sender<(usize, Result<Trip1, String>)>)>,
    /// Trip 2 asked and the downloader owns the final read; release after.
    draining: bool,
}

// ------------------------------------------------------------ downloader

/// One request to the downloader thread: wait for `event`, copy the ranges,
/// reply. Raw device pointers are safe because the service keeps the solve's
/// ranges allocated until the downloader confirms (`done` channel).
enum Dl {
    Trip1 {
        event: CudaEvent,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<Trip1, String>)>,
        id: u64,
        /// Evaluation solves end after trip 1; generation solves remain
        /// resident for the later carry request.
        final_slot: Option<usize>,
        strategy: (u64, usize),
        root_vals: (u64, usize),
        nc_root: [usize; 2],
    },
    Trip2 {
        event: CudaEvent,
        reply: mpsc::Sender<Result<Trip2, String>>,
        slot: usize,
        id: u64,
        snap_beliefs: u64,
        leaf_configs: usize,
        leaf_off: [usize; 2],
        nc_leaf: [usize; 2],
        nsnaps: usize,
    },
}

/// Host staging for one download.
///
/// Pinned memory is what makes the copy fast, but a container's locked-memory
/// limit is small — 8 MiB on the Vast.ai box — and one large solve's strategy
/// exceeds it. A failed pinned allocation used to panic the downloader thread,
/// which then answered nobody: every worker parked on a reply that could never
/// arrive, and the whole run deadlocked with the GPU idle. So fall back to
/// ordinary pageable memory instead. It costs a slower copy on the rare
/// oversized solve and nothing at all on the common one.
enum Stage {
    Pinned(PinnedHostSlice<f32>),
    Paged(Vec<f32>),
}

impl Stage {
    fn len(&self) -> usize {
        match self {
            Stage::Pinned(p) => p.len(),
            Stage::Paged(v) => v.len(),
        }
    }

    fn as_mut_slice(&mut self) -> Result<&mut [f32], String> {
        match self {
            Stage::Pinned(p) => p.as_mut_slice().map_err(|e| format!("{e:?}")),
            Stage::Paged(v) => Ok(v.as_mut_slice()),
        }
    }
}

fn pinned_stage<'a>(
    stream: &Arc<CudaStream>,
    stage: &'a mut Option<Stage>,
    len: usize,
) -> Result<&'a mut [f32], String> {
    if stage.as_ref().map_or(0, Stage::len) < len {
        let cap = len.max(1).next_power_of_two();
        // Drop the old buffer before asking for a bigger one; the limit counts
        // what is locked right now.
        *stage = None;
        *stage = Some(match unsafe { stream.context().alloc_pinned(cap) } {
            Ok(p) => Stage::Pinned(p),
            Err(_) => Stage::Paged(vec![0.0; cap]),
        });
    }
    stage.as_mut().unwrap().as_mut_slice()
}

fn downloader(
    stream: Arc<CudaStream>,
    mut device_stage: CudaSlice<f32>,
    gather_trip2: CudaFunction,
    rx: Arc<std::sync::Mutex<mpsc::Receiver<Dl>>>,
    done: mpsc::Sender<(usize, u64)>,
) {
    let device_stage_at = ptr_mut(&stream, &mut device_stage) as usize as u64;
    let mut host_stage = None;
    loop {
        let idle = crate::timed!(DLIDLE);
        // Hold the receiver only for the recv itself: the copies below are
        // where the time goes and they must overlap across the pool.
        let got = rx.lock().map(|r| r.recv());
        drop(idle);
        let Ok(Ok(req)) = got else { break };
        let _busy = crate::timed!(DLBUSY);
        match req {
            Dl::Trip1 {
                event,
                tag,
                reply,
                id,
                final_slot,
                strategy,
                root_vals,
                nc_root,
            } => {
                // The ranges are disjoint on device but one pinned host buffer
                // lets both copies share one event wait and one stream sync.
                let host = match pinned_stage(&stream, &mut host_stage, strategy.1 + root_vals.1) {
                    Ok(host) => host,
                    Err(e) => {
                        let _ = reply.send((tag, Err(format!("gpu download staging: {e}"))));
                        if let Some(slot) = final_slot {
                            let _ = done.send((slot, id));
                        }
                        continue;
                    }
                };
                let _ = stream.wait(&event);
                unsafe {
                    let _ = cudarc::driver::result::memcpy_dtoh_async(
                        &mut host[..strategy.1],
                        strategy.0,
                        stream.cu_stream(),
                    );
                    let _ = cudarc::driver::result::memcpy_dtoh_async(
                        &mut host[strategy.1..strategy.1 + root_vals.1],
                        root_vals.0,
                        stream.cu_stream(),
                    );
                }
                {
                    let _t = crate::timed!(DLSYNC);
                    // Nobody else syncs this stream, so a fault in the gather
                    // or in these copies would first be noticed by whichever
                    // unrelated sync ran next — and blamed on it.
                    if let Err(e) = stream.synchronize() {
                        eprintln!("gpu: download stream (trip 1) failed: {e:?}");
                    }
                }
                let strategy = host[..strategy.1].to_vec();
                let flat = host[strategy.len()..strategy.len() + root_vals.1].to_vec();
                let stride = (nc_root[0] + nc_root[1]).max(1);
                let root_values = flat
                    .chunks_exact(stride)
                    .map(|c| [c[..nc_root[0]].to_vec(), c[nc_root[0]..].to_vec()])
                    .collect();
                let _ = reply.send((
                    tag,
                    Ok(Trip1 {
                        id,
                        strategy,
                        root_values,
                    }),
                ));
                if let Some(slot) = final_slot {
                    let _ = done.send((slot, id));
                }
            }
            Dl::Trip2 {
                event,
                reply,
                slot,
                id,
                snap_beliefs,
                leaf_configs,
                leaf_off,
                nc_leaf,
                nsnaps,
            } => {
                let n = nsnaps.saturating_sub(1);
                let stride = (nc_leaf[0] + nc_leaf[1]).max(1);
                let len = n * stride;
                let mut flat = Vec::new();
                if len > 0 {
                    let host = match pinned_stage(&stream, &mut host_stage, len) {
                        Ok(host) => host,
                        Err(e) => {
                            let _ = reply.send(Err(format!("gpu download staging: {e}")));
                            let _ = done.send((slot, id));
                            continue;
                        }
                    };
                    let _ = stream.wait(&event);
                    if len <= device_stage.len() {
                        let src = snap_beliefs;
                        let dst = device_stage_at;
                        let (leaf_configs, off0, off1, n0, n1, ns) = (
                            leaf_configs as i32,
                            leaf_off[0] as i32,
                            leaf_off[1] as i32,
                            nc_leaf[0] as i32,
                            nc_leaf[1] as i32,
                            n as i32,
                        );
                        let cfg = LaunchConfig {
                            grid_dim: ((len as u32).div_ceil(BLOCK), 1, 1),
                            block_dim: (BLOCK, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut b = stream.launch_builder(&gather_trip2);
                        b.arg(&src);
                        b.arg(&dst);
                        b.arg(&leaf_configs);
                        b.arg(&off0);
                        b.arg(&off1);
                        b.arg(&n0);
                        b.arg(&n1);
                        b.arg(&ns);
                        report(unsafe { b.launch(cfg) }, cfg.grid_dim.0);
                    } else {
                        for s in 0..n {
                            for p in 0..2 {
                                let count = nc_leaf[p];
                                if count == 0 {
                                    continue;
                                }
                                let dst = s * stride + if p == 1 { nc_leaf[0] } else { 0 };
                                let src =
                                    snap_beliefs + (4 * (s * leaf_configs + leaf_off[p])) as u64;
                                unsafe {
                                    let _ = cudarc::driver::result::memcpy_dtoh_async(
                                        &mut host[dst..dst + count],
                                        src,
                                        stream.cu_stream(),
                                    );
                                }
                            }
                        }
                    }
                    if len <= device_stage.len() {
                        unsafe {
                            let _ = cudarc::driver::result::memcpy_dtoh_async(
                                &mut host[..len],
                                device_stage_at,
                                stream.cu_stream(),
                            );
                        }
                    }
                    {
                        let _t = crate::timed!(DLSYNC);
                        if let Err(e) = stream.synchronize() {
                            eprintln!("gpu: download stream (trip 2) failed: {e:?}");
                        }
                    }
                    flat.extend_from_slice(&host[..len]);
                }
                let out: Trip2 = flat
                    .chunks_exact(stride)
                    .map(|c| [c[..nc_leaf[0]].to_vec(), c[nc_leaf[0]..].to_vec()])
                    .collect();
                let _ = reply.send(Ok(out));
                let _ = done.send((slot, id));
            }
        }
    }
}

/// Page-locked staging for table uploads.
///
/// `cuMemcpyHtoDAsync` out of ordinary pageable memory is synchronous inside
/// the driver, and it was running at about half a gigabyte a second — a third
/// of the service thread's time went into admitting jobs, time the tick loop
/// was not spending keeping the device fed. Packing straight into page-locked
/// memory makes the same upload a DMA the service thread does not wait for.
///
/// The buffer is a ring: a copy's source must stay untouched until it has
/// landed, so wrapping waits for the build stream to drain. Sized so that
/// wrapping happens every several batches, not every job.
struct Upload {
    buf: PinnedHostSlice<u8>,
    at: usize,
}

impl Upload {
    fn new(stream: &Arc<CudaStream>) -> Result<Upload, String> {
        let cap = env_mb("WARCHEST_GPU_UPLOAD_MB", 64);
        let buf = unsafe { stream.context().alloc_pinned(cap) }
            .map_err(|e| format!("upload staging: {e:?}"))?;
        Ok(Upload { buf, at: 0 })
    }

    /// A `len`-byte window of the ring, or `None` when one job's tables are
    /// larger than the whole buffer (the caller falls back to a pageable
    /// upload rather than refusing the job).
    fn reserve(&mut self, len: usize, build: &Arc<CudaStream>) -> Option<&mut [u8]> {
        let len = len.next_multiple_of(16);
        if len > self.buf.len() {
            return None;
        }
        if self.at + len > self.buf.len() {
            let _ = build.synchronize();
            self.at = 0;
        }
        let at = self.at;
        self.at += len;
        let slice = self.buf.as_mut_slice().ok()?;
        Some(&mut slice[at..at + len])
    }
}

// ------------------------------------------------------------------ service

pub struct Service {
    compute: Arc<CudaStream>,
    build: Arc<CudaStream>,
    blas: CudaBlas,
    build_blas: CudaBlas,
    f: Kernels,
    weights: Weights,
    incoming: Option<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)>,
    ctx: CudaSlice<Ctx>,
    // Slabs and pools.
    tables: CudaSlice<u8>,
    tbl_slab: Slab,
    arenas: CudaSlice<f32>,
    arena_slab: Slab,
    rows: Slab,
    _pools: Vec<CudaSlice<f32>>,
    // Live set.
    live: Vec<Option<Solve>>,
    free: Vec<usize>,
    next_id: u64,
    live_slots: CudaSlice<i32>,
    live_count: usize,
    live_dirty: bool,
    descs: CudaSlice<Desc>,
    g_meta: CudaSlice<GroupDev>,
    flat_meta: CudaSlice<GroupDev>,
    flat_prefix: CudaSlice<i32>,
    flat_totals: Vec<usize>,
    max_live_levels: usize,
    /// Reusable completion events which bound how far tick submission may run
    /// ahead of the device. `tick_fence_next` is the oldest event once full.
    tick_fences: Vec<CudaEvent>,
    tick_fence_next: usize,
    tick_fence_filled: usize,
    /// Admission's one mutable launch header. It is deliberately separate
    /// from `g_meta`: build and compute run concurrently, so even disjoint
    /// logical uses must not race through one host-to-device upload.
    build_meta: CudaSlice<GroupDev>,
    bmap: CudaSlice<i32>,
    // Plumbing.
    rx: mpsc::Receiver<Cmd>,
    upload: Upload,
    /// Diagnostic: synchronize and report after every tick phase.
    sync_phase: bool,
    dl_tx: Option<mpsc::Sender<Dl>>,
    dl_threads: Vec<std::thread::JoinHandle<()>>,
    done_rx: mpsc::Receiver<(usize, u64)>,
    waiting: std::collections::VecDeque<(Job, usize, mpsc::Sender<(usize, Result<Trip1, String>)>)>,
    /// Let staggered worker submissions coalesce while resident solves keep
    /// the compute stream busy.  Tiny admission batches strand the build
    /// kernels at one or two blocks; this bound is configurable so the
    /// latency/occupancy trade can be measured on the target workload.
    min_batch: usize,
    /// Ticks and solves served, for the log line.
    ticks: u64,
    solved: u64,
}

/// Spawn the service thread on CUDA device `device`; returns the client.
pub fn spawn(
    device: usize,
    dims: Vec<usize>,
    w: Vec<f32>,
    b: Vec<f32>,
    ln: Vec<f32>,
) -> Result<GpuClient, String> {
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name(format!("gpu-service-{device}"))
        .spawn(move || match Service::new(device, rx, dims, w, b, ln) {
            Ok(mut svc) => {
                let _ = ready_tx.send(Ok(()));
                svc.run()
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })
        .map_err(|e| format!("{e:?}"))?;
    let ready = ready_rx
        .recv()
        .map_err(|_| "gpu service thread died".to_string())?;
    if let Err(e) = ready {
        let _ = thread.join();
        return Err(e);
    }
    Ok(GpuClient::new(tx, thread))
}

impl Drop for Service {
    fn drop(&mut self) {
        // The service owns every address its streams and downloader use. Its
        // handle does not finish dropping until all queued device work and
        // the copy thread are gone, so a replacement service cannot inherit
        // a primary context with dangling work.
        if let Err(e) = self.build.synchronize() {
            eprintln!("gpu: build stream failed during shutdown: {e:?}");
        }
        if let Err(e) = self.compute.synchronize() {
            eprintln!("gpu: compute stream failed during shutdown: {e:?}");
        }
        self.dl_tx.take();
        for thread in std::mem::take(&mut self.dl_threads) {
            let _ = thread.join();
        }
        if let Err(e) = self.compute.context().synchronize() {
            eprintln!("gpu: context failed during shutdown: {e:?}");
        }
    }
}

impl Service {
    pub(crate) fn new(
        device: usize,
        rx: mpsc::Receiver<Cmd>,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Service, String> {
        let dev = CudaContext::new(device).map_err(|e| format!("cuda device {device}: {e:?}"))?;
        // This service owns all three streams and orders every cross-stream
        // handoff explicitly: build -> compute after admission, compute ->
        // download for results, and compute -> build before a head-row range
        // can be recycled.  cudarc's allocation event tracking would add a
        // wait and record for every CudaSlice kernel argument even though the
        // kernels reach the large pools through raw pointers in `Ctx`.  On the
        // hot path that was tens of thousands of redundant driver calls per
        // second, so keep one synchronization protocol rather than two.
        unsafe { dev.disable_event_tracking() };
        let compute = dev.new_stream().map_err(|e| format!("{e:?}"))?;
        let build = dev.new_stream().map_err(|e| format!("{e:?}"))?;
        let tick_fences = (0..max_queued_ticks())
            .map(|_| dev.new_event(None).map_err(|e| format!("{e:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        let blas = CudaBlas::new(compute.clone()).map_err(|e| format!("{e:?}"))?;
        let build_blas = CudaBlas::new(build.clone()).map_err(|e| format!("{e:?}"))?;

        let layout = V3Layout::new(&dims)?;
        let (maj, min) = (
            dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(|e| format!("{e:?}"))?,
            dev.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(|e| format!("{e:?}"))?,
        );
        let arch = format!("compute_{maj}{min}");
        let src = format!("{}\n{}", cuda_preamble(&layout), include_str!("kernels.cu"));
        let ptx = nvrtc::compile_ptx_with_opts(
            &src,
            nvrtc::CompileOptions {
                arch: Some(Box::leak(arch.into_boxed_str())),
                options: vec!["--generate-line-info".into()],
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = dev.load_module(ptx).map_err(|e| format!("module: {e:?}"))?;
        let f = Kernels::load(&module)?;

        // Pools: stable network rows, and the build scratch. Sizes derive
        // from the shape; the batch caps bound the scratch.
        let (pubw, hmlp, _, slotw) = layout.widths();
        let h_stride = std::iter::once(layout.head_in)
            .chain(hmlp.iter().copied())
            .max()
            .unwrap();
        let max_pub = pubw
            .iter()
            .copied()
            .chain([layout.head_in, layout.xdim()])
            .max()
            .unwrap();
        let slot_max = slotw
            .iter()
            .copied()
            .chain([layout.dg, layout.hfeat()])
            .max()
            .unwrap();
        let bh_len = (MAX_BATCH_ROWS * max_pub)
            .max(MAX_BATCH_CFG * crate::rebel::NSLOT * slot_max)
            .max(MAX_BATCH_ROWS * crate::rebel::NTYPE * layout.de);
        // The per-row head block of `bg` holds whichever of the packers is
        // widest, and `pile_pe` starts its own block at `rows * NTYPE * DE` —
        // so `DE` belongs in this maximum even though nothing packs `DE`
        // values per row. Leaving it out undersized the pool whenever a batch
        // carried more than about 200k rows, and `pile_pe` then wrote its
        // block past the end of the pool, into whatever the driver had placed
        // after it. The symptom was an illegal address minutes later, in an
        // unrelated kernel reading a table that had been overwritten with
        // somebody's card embeddings.
        let bg_len = (MAX_BATCH_ROWS
            * crate::rebel::NTYPE
            * crate::units::CARD_FEATS
                .max(crate::rebel::PILE_COUNTS)
                .max(layout.de)
            + MAX_BATCH * crate::rebel::NTYPE * layout.de)
            .max(MAX_BATCH_CFG * crate::rebel::NSLOT * layout.hfeat())
            .max(MAX_BATCH_CFG * (layout.rank + 1));
        let mut pools = Vec::new();
        let mut pool = |stream: &Arc<CudaStream>, n: usize| -> Result<*mut f32, String> {
            let mut s: CudaSlice<f32> =
                stream.alloc_zeros(n.max(1)).map_err(|e| format!("{e:?}"))?;
            let p = ptr_mut(stream, &mut s);
            pools.push(s);
            Ok(p)
        };
        let mut ctx0 = Ctx::default();
        let rows_cap = max_rows();
        ctx0.h0 = pool(&compute, rows_cap * layout.head_in)?;
        ctx0.xb = pool(&compute, rows_cap * 2 * layout.dg)?;
        ctx0.h = pool(&compute, rows_cap * h_stride)?;
        ctx0.h2 = pool(
            &compute,
            if hmlp.is_empty() {
                1
            } else {
                rows_cap * h_stride
            },
        )?;
        ctx0.u = pool(&compute, rows_cap * layout.rank)?;
        ctx0.bx = pool(&compute, MAX_BATCH_ROWS * layout.xdim())?;
        ctx0.bh = pool(&compute, bh_len)?;
        ctx0.bh2 = pool(&compute, bh_len)?;
        ctx0.bg = pool(&compute, bg_len)?;

        // Each batch stores the five-int map followed by one slot id per
        // entry. The tail is passed to build kernels as Group::slots.
        let mut bmap: CudaSlice<i32> = compute
            .alloc_zeros(MAX_BATCH * (BMAP_INTS + 1))
            .map_err(|e| format!("{e:?}"))?;
        ctx0.bmap = ptr_mut(&compute, &mut bmap) as *const i32;

        let mut weights = Weights::upload(&compute, &dims, w, b, ln)?;
        merge_ctx(&mut weights.ctx, &ctx0);
        let ctx = htod(&compute, &[weights.ctx])?;
        check_abi(&compute, &f.abi_probe)?;

        let tables: CudaSlice<u8> = compute
            .alloc_zeros(env_mb("WARCHEST_GPU_TABLE_MB", 512))
            .map_err(|e| format!("table slab: {e:?}"))?;
        let arenas: CudaSlice<f32> = compute
            .alloc_zeros(env_mb("WARCHEST_GPU_ARENA_MB", 1536) / 4)
            .map_err(|e| format!("arena slab: {e:?}"))?;
        let tbl_slab = Slab::new(tables.len());
        let arena_slab = Slab::new(arenas.len());

        // Results come back through a small pool, not one thread. One thread
        // was 76% busy at 640 solves a second — a single server at that
        // utilisation queues, and its queue is what left half the resident
        // solves sitting in carry or drain instead of iterating. Each worker
        // owns its own stream and staging so they never serialise on each
        // other; nothing about a reply depends on the order of the others.
        let (dl_tx, dl_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let dl_rx = Arc::new(std::sync::Mutex::new(dl_rx));
        let ndl = std::env::var("WARCHEST_GPU_DOWNLOADERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4usize)
            .clamp(1, 16);
        let mut dl_threads = Vec::new();
        for k in 0..ndl {
            let download = dev.new_stream().map_err(|e| format!("{e:?}"))?;
            let download_stage = download
                .alloc_zeros(MAX_DOWNLOAD_FLOATS)
                .map_err(|e| format!("download staging: {e:?}"))?;
            let gather_trip2 = f.gather_trip2.clone();
            let (rx, done) = (dl_rx.clone(), done_tx.clone());
            dl_threads.push(
                std::thread::Builder::new()
                    .name(format!("gpu-download-{device}-{k}"))
                    .spawn(move || downloader(download, download_stage, gather_trip2, rx, done))
                    .map_err(|e| format!("{e:?}"))?,
            );
        }
        drop(done_tx);

        let descs = compute.alloc_zeros(CAP).map_err(|e| format!("{e:?}"))?;
        let g_meta = compute.alloc_zeros(N_HDRS).map_err(|e| format!("{e:?}"))?;
        let flat_meta = compute
            .alloc_zeros(N_FLAT_HDRS)
            .map_err(|e| format!("{e:?}"))?;
        let flat_prefix = compute
            .alloc_zeros(N_FLAT_HDRS * FLAT_PREFIX_STRIDE)
            .map_err(|e| format!("{e:?}"))?;
        let build_meta = compute.alloc_zeros(1).map_err(|e| format!("{e:?}"))?;
        let live_slots = compute.alloc_zeros(CAP).map_err(|e| format!("{e:?}"))?;

        // Initialization is queued on `compute`, while admission runs on
        // `build`. Make every pointer table, allocation and weight upload
        // visible before the service can accept its first job.
        compute
            .synchronize()
            .map_err(|e| format!("gpu init: {e:?}"))?;

        let min_batch = std::env::var("WARCHEST_GPU_MIN_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
            .clamp(1, MAX_BATCH);

        Ok(Service {
            descs,
            g_meta,
            flat_meta,
            flat_prefix,
            flat_totals: vec![0; N_FLAT_HDRS],
            max_live_levels: 0,
            tick_fences,
            tick_fence_next: 0,
            tick_fence_filled: 0,
            build_meta,
            live_slots,
            live_count: 0,
            live_dirty: false,
            live: (0..CAP).map(|_| None).collect(),
            free: (0..CAP).rev().collect(),
            next_id: 1,
            waiting: std::collections::VecDeque::new(),
            min_batch,
            tables,
            tbl_slab,
            arenas,
            arena_slab,
            rows: Slab::new(rows_cap),
            _pools: pools,
            bmap,
            incoming: None,
            upload: Upload::new(&build)?,
            sync_phase: std::env::var_os("WARCHEST_GPU_SYNC_PHASE").is_some(),
            dl_tx: Some(dl_tx),
            dl_threads,
            done_rx,
            ticks: 0,
            solved: 0,
            compute,
            build,
            blas,
            build_blas,
            f,
            weights,
            ctx,
            rx,
        })
    }

    fn run(&mut self) {
        loop {
            // Drain commands; block only when nothing is in flight.
            loop {
                match self.rx.try_recv() {
                    Ok(cmd) => {
                        if self.handle(cmd) {
                            return;
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
            }
            while let Ok((slot, id)) = self.done_rx.try_recv() {
                self.release(slot, id);
            }
            // Weights apply only between solves: stop admitting, drain, swap.
            if self.incoming.is_some() && self.live_count == 0 && self.waiting.is_empty() {
                let (dims, w, b, ln) = self.incoming.take().unwrap();
                match Weights::upload(&self.compute, &dims, w, b, ln) {
                    Ok(mut fresh) => {
                        let keep = self.weights.ctx;
                        merge_ctx(&mut fresh.ctx, &keep);
                        self.weights = fresh;
                        let _ = self.compute.memcpy_htod(&[self.weights.ctx], &mut self.ctx);
                    }
                    Err(e) => eprintln!("gpu: bad weights: {e}"),
                }
            }
            let active = self
                .live
                .iter()
                .flatten()
                .any(|sv| sv.stage == STAGE_ITERATE || sv.stage == STAGE_VALUE);
            if self.incoming.is_none() && (self.waiting.len() >= self.min_batch || !active) {
                self.admit_batch();
            }
            // Only a solve that is iterating or valuing has anything for a
            // tick to advance. Solves sitting in carry or drain are waiting on
            // a worker's trip 2 or on the downloader, and ticking for them was
            // most of the service's ticks: an empty tick still rewrites the
            // launch headers, fires the state kernels and waits on a fence.
            //
            // Admission changes membership before `upload_headers` refreshes
            // the cached count. A freshly admitted first batch must enter
            // `tick`, which consumes `live_dirty`; otherwise the service
            // mistakes it for an empty live set and blocks on `recv` forever.
            let advancing = self.live_dirty
                || self
                    .live
                    .iter()
                    .flatten()
                    .any(|sv| sv.stage == STAGE_ITERATE || sv.stage == STAGE_VALUE);
            if advancing {
                self.tick();
            } else if self.waiting.is_empty() {
                // Idle: block for the next command instead of spinning.
                match self.rx.recv() {
                    Ok(cmd) => {
                        if self.handle(cmd) {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        }
    }

    /// Returns true when the owner requested a clean shutdown.
    fn handle(&mut self, cmd: Cmd) -> bool {
        match cmd {
            Cmd::Submit { job, tag, reply } => self.waiting.push_back((job, tag, reply)),
            Cmd::Trip2 { id, leaf, reply } => self.start_carry(id, leaf, reply),
            Cmd::SetWeights { dims, w, b, ln } => {
                if dims != self.weights.dims {
                    eprintln!(
                        "gpu: weight shape changed ({:?} -> {dims:?}); restart the service",
                        self.weights.dims
                    );
                    return false;
                }
                self.incoming = Some((dims, w, b, ln));
            }
            Cmd::Shutdown => return true,
        }
        false
    }

    // ------------------------------------------------------------ launching

    /// `#[track_caller]` and the error report below are the whole diagnostic
    /// for an out-of-bounds kernel: CUDA reports the fault asynchronously, at
    /// whichever later driver call notices, so the launch that caused it is
    /// invisible. Run with `CUDA_LAUNCH_BLOCKING=1` and every launch reports
    /// its own failure, named by the line in the tick that issued it.
    #[track_caller]
    fn fire(
        &self,
        which: fn(&Kernels) -> &CudaFunction,
        hdr: usize,
        grid: u32,
        stream: &Arc<CudaStream>,
    ) {
        if grid == 0 {
            return;
        }
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = which(&self.f).clone();
        let group = self.g_meta.slice(hdr..hdr + 1);
        let mut b = stream.launch_builder(&f);
        b.arg(&self.descs);
        b.arg(&group);
        b.arg(&self.ctx);
        report(unsafe { b.launch(cfg) }, grid);
    }

    /// Launch a kernel over a flattened row/node prefix. `arg` selects the
    /// dynamic solver pass (iterate traverser, fixed player, or reach mode)
    /// without rewriting a device header on every tick.
    #[track_caller]
    fn fire_flat(&self, which: fn(&Kernels) -> &CudaFunction, hdr: usize, grid: u32, arg: i32) {
        if grid == 0 {
            return;
        }
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = which(&self.f).clone();
        let group = self.flat_meta.slice(hdr..hdr + 1);
        let mut b = self.compute.launch_builder(&f);
        b.arg(&self.descs);
        b.arg(&group);
        b.arg(&self.ctx);
        b.arg(&arg);
        report(unsafe { b.launch(cfg) }, grid);
    }

    fn flat_warp_blocks(&self, hdr: usize) -> u32 {
        ((self.flat_totals[hdr] * 32) as u32).div_ceil(BLOCK)
    }

    /// Blocks for a one-thread-per-task launch, as the config-flattened
    /// kernels use.
    fn flat_thread_blocks(&self, hdr: usize) -> u32 {
        (self.flat_totals[hdr] as u32).div_ceil(BLOCK)
    }

    /// Packed bases for iterate reach, iterate backprop, value backprop for
    /// each player, and value reach. The four non-level headers precede them.
    fn flat_level_bases(&self) -> FlatBases {
        let n = self.max_live_levels;
        let reach_it = FlatHdr::LevelItBase as usize;
        FlatBases {
            reach_it,
            back_it: reach_it + n,
            back_v0: reach_it + 2 * n,
            back_v1: reach_it + 3 * n,
            reach_v: reach_it + 4 * n,
            end: reach_it + 5 * n,
        }
    }

    /// Rewrite the tick's launch headers; runs only when membership changed.
    fn upload_headers(&mut self) {
        let _t = crate::timed!(SVCHDR);
        let base = ptr(&self.compute, &self.live_slots);
        let mk = |mode: i32, p: i32, level: i32| GroupDev {
            slots: base,
            prefix: std::ptr::null(),
            n: self.live_count as i32,
            mode,
            p_player: p,
            level,
            total: 0,
        };
        let mut h = vec![GroupDev::default(); N_HDRS];
        h[Hdr::Plain as usize] = mk(0, 0, 0);
        h[Hdr::ReadIt as usize] = mk(0, -1, 0);
        h[Hdr::ReadV0 as usize] = mk(0, 0, 0);
        h[Hdr::ReadV1 as usize] = mk(0, 1, 0);
        h[Hdr::ReachVC as usize] = mk(0, 0, 0);
        h[Hdr::ReachIt as usize] = mk(1, 0, 0);
        h[Hdr::ReachSnap as usize] = mk(2, 0, 0);
        for k in 0..self.weights.layout.hmlp.len() {
            // Layer k reads the previous buffer and writes h (mode 0) or h2
            // (mode 1); the first layer reads h, so it writes h2.
            h[Hdr::HmlpBase as usize + k] = mk((k % 2 == 0) as i32, 0, k as i32);
        }
        let _ = self.compute.memcpy_htod(&h, &mut self.g_meta);

        let slots: Vec<usize> = (0..CAP).filter(|&s| self.live[s].is_some()).collect();
        self.max_live_levels = slots
            .iter()
            .map(|&s| self.live[s].as_ref().unwrap().level_counts.len())
            .max()
            .unwrap_or(0);

        let b = self.flat_level_bases();
        let active_hdrs = b.end;
        let mut prefix = vec![0i32; active_hdrs * FLAT_PREFIX_STRIDE];
        let count = |hdr: usize, sv: &Solve| -> usize {
            match hdr {
                x if x == FlatHdr::LeafRows as usize => {
                    ((sv.stage == STAGE_ITERATE || sv.stage == STAGE_VALUE) as usize)
                        * sv.desc.nleaf as usize
                }
                x if x == FlatHdr::ReadIt as usize => {
                    (sv.stage == STAGE_ITERATE) as usize * (sv.desc.nleaf + sv.desc.nterm) as usize
                }
                x if x == FlatHdr::ReadV as usize => {
                    (sv.stage == STAGE_VALUE) as usize * (sv.desc.nleaf + sv.desc.nterm) as usize
                }
                x if x == FlatHdr::NodesIt as usize => {
                    (sv.stage == STAGE_ITERATE) as usize * sv.decision_cfg_counts[sv.t & 1]
                }
                x if x < b.back_it => {
                    (sv.stage == STAGE_ITERATE) as usize
                        * sv.level_counts.get(x - b.reach_it).copied().unwrap_or(0)
                }
                x if x < b.back_v0 => {
                    (sv.stage == STAGE_ITERATE) as usize
                        * sv.sweep_cfg_level_counts[sv.t & 1]
                            .get(x - b.back_it)
                            .copied()
                            .unwrap_or(0)
                }
                x if x < b.back_v1 => {
                    (sv.stage == STAGE_VALUE) as usize
                        * sv.sweep_cfg_level_counts[0]
                            .get(x - b.back_v0)
                            .copied()
                            .unwrap_or(0)
                }
                x if x < b.reach_v => {
                    (sv.stage == STAGE_VALUE) as usize
                        * sv.sweep_cfg_level_counts[1]
                            .get(x - b.back_v1)
                            .copied()
                            .unwrap_or(0)
                }
                x => {
                    (sv.stage == STAGE_VALUE) as usize
                        * sv.level_counts.get(x - b.reach_v).copied().unwrap_or(0)
                }
            }
        };
        for hdr in 0..active_hdrs {
            let at = hdr * FLAT_PREFIX_STRIDE;
            let mut total = 0usize;
            for (i, &slot) in slots.iter().enumerate() {
                total += count(hdr, self.live[slot].as_ref().unwrap());
                prefix[at + i + 1] = total as i32;
            }
            self.flat_totals[hdr] = total;
        }
        let _ = self
            .compute
            .memcpy_htod(&prefix, &mut self.flat_prefix.slice_mut(..prefix.len()));
        let prefix_base = ptr(&self.compute, &self.flat_prefix);
        let flat: Vec<GroupDev> = (0..active_hdrs)
            .map(|hdr| GroupDev {
                slots: base,
                prefix: unsafe { prefix_base.add(hdr * FLAT_PREFIX_STRIDE) },
                n: self.live_count as i32,
                level: if hdr >= b.reach_v {
                    (hdr - b.reach_v) as i32
                } else if hdr >= b.back_v1 {
                    (hdr - b.back_v1) as i32
                } else if hdr >= b.back_v0 {
                    (hdr - b.back_v0) as i32
                } else if hdr >= b.back_it {
                    (hdr - b.back_it) as i32
                } else if hdr >= b.reach_it {
                    (hdr - b.reach_it) as i32
                } else {
                    0
                },
                total: self.flat_totals[hdr] as i32,
                ..GroupDev::default()
            })
            .collect();
        let _ = self
            .compute
            .memcpy_htod(&flat, &mut self.flat_meta.slice_mut(..flat.len()));
    }

    // ------------------------------------------------------------ the tick

    fn tick(&mut self) {
        let _t = crate::timed!(SVCTICK);
        // Reuse the oldest fence only after its tick has completed. This is
        // the service's intentional submission backpressure; cudarc's generic
        // per-allocation event tracking is disabled above.
        if self.tick_fence_filled == self.tick_fences.len() {
            let _f = crate::timed!(SVCFENCE);
            let _ = self.tick_fences[self.tick_fence_next].synchronize();
            self.tick_fence_filled -= 1;
        }
        self.tick_upto(Step::All);
        if self.live_count > 0 {
            let _ = self.tick_fences[self.tick_fence_next].record(&self.compute);
            self.tick_fence_next = (self.tick_fence_next + 1) % self.tick_fences.len();
            self.tick_fence_filled += 1;
        }
    }

    /// Synchronize and report, when `WARCHEST_GPU_SYNC_PHASE` is set. CUDA
    /// reports a faulting kernel at whatever driver call next notices, which
    /// is usually a different phase in a different tick; a sync per phase
    /// pins it to the phase that caused it without serialising every launch
    /// the way `CUDA_LAUNCH_BLOCKING` does (which hides races outright).
    fn phase(&self, name: &str) {
        if !self.sync_phase {
            return;
        }
        if let Err(e) = self.compute.synchronize() {
            eprintln!("gpu: phase '{name}' failed: {e:?}");
        }
    }

    fn tick_upto(&mut self, upto: Step) {
        if self.live_dirty {
            let slots: Vec<i32> = (0..CAP)
                .filter(|&s| self.live[s].is_some())
                .map(|s| s as i32)
                .collect();
            self.live_count = slots.len();
            if !slots.is_empty() {
                let _ = self
                    .compute
                    .memcpy_htod(&slots, &mut self.live_slots.slice_mut(..slots.len()));
            }
            self.live_dirty = false;
        }
        if self.live_count == 0 || upto <= Step::None {
            return;
        }
        #[cfg(feature = "prof")]
        {
            let mut stage = [0usize; 4];
            let mut active_rows = 0usize;
            for sv in self.live.iter().flatten() {
                stage[sv.stage as usize] += 1;
                if sv.stage == STAGE_ITERATE || sv.stage == STAGE_VALUE {
                    active_rows += sv.nrows;
                }
            }
            crate::prof::gpu_tick(
                self.live_count,
                stage[STAGE_ITERATE as usize],
                stage[STAGE_VALUE as usize],
                stage[STAGE_CARRY as usize],
                stage[STAGE_DRAIN as usize],
                self.waiting.len(),
                active_rows,
                self.rows.high,
            );
        }
        // Stage membership changes every tick even when slot membership does
        // not. Compact prefixes keep iterate, value and carry work out of one
        // another's grids; the small prefix upload is far cheaper than
        // launching hundreds of thousands of blocks which only return.
        self.upload_headers();
        let b = self.flat_level_bases();
        let l = &self.weights.layout;
        let n = self.live_count as u32;
        let hi = self.rows.high;
        let c = self.weights.ctx;

        // 1. Fixed-policy value passes re-seed, then propagate one global node level
        // at a time. Each node is a block; no large solve can strand the GPU
        // behind one serial block.
        self.fire(
            |k| &k.reach_seed_flat,
            Hdr::ReachVC as usize,
            n,
            &self.compute,
        );
        for lev in 1..self.max_live_levels {
            let hdr = b.reach_v + lev;
            self.fire_flat(|k| &k.reach_level_flat, hdr, self.flat_warp_blocks(hdr), 0);
        }
        self.phase("value reach");
        // 2. Belief sums: one warp per network leaf globally.
        let leaf_rows = FlatHdr::LeafRows as usize;
        self.fire_flat(
            |k| &k.belief_sums_flat,
            leaf_rows,
            self.flat_warp_blocks(leaf_rows),
            0,
        );
        self.phase("belief sums");
        // 3. The head: GEMM, entry norm, extra layers, readout GEMM.
        gemm(
            &self.blas,
            hi,
            l.head_in,
            2 * l.dg,
            c.xb as *const f32,
            2 * l.dg,
            c.wb,
            l.head_in,
            c.h,
            h_stride_of(l),
            0.0,
        );
        self.fire_flat(
            |k| &k.head_entry_flat,
            leaf_rows,
            self.flat_warp_blocks(leaf_rows),
            0,
        );
        let mut cur = c.h as *const f32;
        for (k, s) in l.hmlp.iter().enumerate() {
            let dst = if k % 2 == 0 { c.h2 } else { c.h };
            gemm(
                &self.blas,
                hi,
                s.o,
                s.i,
                cur,
                h_stride_of(l),
                self.weights_at(s.w),
                s.o,
                dst,
                h_stride_of(l),
                0.0,
            );
            self.fire(
                |f| &f.hmlp_act,
                Hdr::HmlpBase as usize + k,
                n,
                &self.compute,
            );
            cur = dst as *const f32;
        }
        gemm(
            &self.blas,
            hi,
            l.rank,
            l.head_out,
            cur,
            h_stride_of(l),
            self.weights_at(l.wu.w),
            l.rank,
            c.u,
            l.rank,
            0.0,
        );
        self.phase("head");
        if upto == Step::Head {
            return;
        }
        // 4. Iterate pass: readout + backprop for the traverser.
        let read_it = FlatHdr::ReadIt as usize;
        self.fire_flat(
            |k| &k.readout_flat,
            read_it,
            self.flat_warp_blocks(read_it),
            -1,
        );
        self.phase("iterate readout");
        if upto == Step::Readout {
            return;
        }
        for lev in (0..self.max_live_levels).rev() {
            let hdr = b.back_it + lev;
            self.fire_flat(
                |k| &k.backprop_level_flat,
                hdr,
                self.flat_thread_blocks(hdr),
                -1,
            );
        }
        self.phase("iterate backprop");
        if upto == Step::Backprop {
            return;
        }
        let nodes = FlatHdr::NodesIt as usize;
        self.fire_flat(
            |k| &k.regret_match_flat,
            nodes,
            self.flat_thread_blocks(nodes),
            0,
        );
        self.phase("regret matching");
        if upto == Step::Regret {
            return;
        }
        // 5. Value passes: both players against the average strategy.
        let read_v = FlatHdr::ReadV as usize;
        for (p, hdr) in [(0, Hdr::ReadV0), (1, Hdr::ReadV1)] {
            self.fire_flat(
                |k| &k.readout_flat,
                read_v,
                self.flat_warp_blocks(read_v),
                p,
            );
            self.phase("value readout");
            let base = if p == 0 { b.back_v0 } else { b.back_v1 };
            for lev in (0..self.max_live_levels).rev() {
                let level = base + lev;
                self.fire_flat(
                    |k| &k.backprop_level_flat,
                    level,
                    self.flat_thread_blocks(level),
                    p,
                );
                self.phase("value backprop");
            }
            self.fire(|k| &k.collect_root, hdr as usize, n, &self.compute);
            self.phase("collect root");
        }
        // 6. Iterate reach follows regret matching. Only the traverser's half
        // can have changed, so update it in place; the root needs no reseed.
        for lev in 1..self.max_live_levels {
            let hdr = b.reach_it + lev;
            self.fire_flat(|k| &k.reach_level_flat, hdr, self.flat_warp_blocks(hdr), 1);
            self.phase("iterate reach");
        }
        if upto == Step::Propagate {
            return;
        }
        self.fire_flat(|k| &k.average_flat, nodes, self.flat_thread_blocks(nodes), 0);
        self.phase("average");
        if upto == Step::Average {
            return;
        }
        // 7. When this iterate is kept, propagate its average once and store
        // the beliefs at every possible exit leaf. This replaces both the
        // full-tree strategy snapshot and trip 2's later replay.
        let snapshot_due = self.live.iter().flatten().any(|sv| {
            sv.stage == STAGE_ITERATE
                && sv.nsnaps > 1
                && sv.meta.snap_iters[1..sv.nsnaps - 1]
                    .binary_search(&(sv.t + 1))
                    .is_ok()
        });
        if snapshot_due {
            self.fire(
                |k| &k.reach_seed_flat,
                Hdr::ReachSnap as usize,
                n,
                &self.compute,
            );
            for lev in 1..self.max_live_levels {
                let hdr = b.reach_it + lev;
                self.fire_flat(|k| &k.reach_level_flat, hdr, self.flat_warp_blocks(hdr), 2);
            }
            self.fire_flat(
                |k| &k.snapshot_beliefs_flat,
                read_it,
                self.flat_warp_blocks(read_it),
                0,
            );
        }
        // 8. Advance the device and host state machines.
        let adv_grid = (self.live_count as u32).div_ceil(BLOCK);
        self.fire(
            |k| &k.advance_state,
            Hdr::Plain as usize,
            adv_grid.max(1),
            &self.compute,
        );
        self.phase("snapshot and advance");
        self.ticks += 1;
        self.advance_host();
    }

    fn weights_at(&self, off: usize) -> *const f32 {
        let (p, _s) = self.weights._w.device_ptr(&self.compute);
        unsafe { (p as usize as *const f32).add(off) }
    }

    /// The host shadow of `advance_state`, plus trip scheduling.
    fn advance_host(&mut self) {
        let mut trip1 = Vec::new();
        for s in 0..CAP {
            let Some(sv) = &mut self.live[s] else {
                continue;
            };
            match sv.stage {
                STAGE_ITERATE => {
                    sv.t += 1;
                    if sv.t == sv.meta.iters {
                        sv.step = 0;
                        if sv.nroots > 0 {
                            sv.stage = STAGE_VALUE;
                        } else {
                            sv.stage = STAGE_CARRY;
                            trip1.push(s);
                        }
                    }
                }
                STAGE_VALUE => {
                    sv.step += 1;
                    if sv.step >= sv.nroots {
                        sv.step = 0;
                        sv.stage = STAGE_CARRY;
                        trip1.push(s);
                    }
                }
                _ => {}
            }
        }
        for s in trip1 {
            self.send_trip1(s);
        }
    }

    // ------------------------------------------------------------ admission

    /// Admit a batch: everything queued, up to the batch caps and whatever
    /// fits in the slabs, in arrival order.
    fn admit_batch(&mut self) {
        let _t = crate::timed!(SVCADMIT);
        let mut batch = Vec::new();
        let (mut rows, mut cfgs) = (0usize, 0usize);
        while let Some((job, _, _)) = self.waiting.front() {
            let t = &job.tables;
            let (job_rows, job_cfgs) = (t.rows, t.ncfg);
            // A head-of-line job which can never fit must fail explicitly;
            // merely breaking here leaves every worker waiting forever while
            // the service spins on the same queue entry.
            let job_table_len = table_len(job);
            let job_arena_len = arena_len_of(job, &self.weights.layout);
            if t.rows > MAX_BATCH_ROWS
                || t.ncfg > MAX_BATCH_CFG
                || t.rows > self.rows.cap
                || job_table_len > self.tbl_slab.cap
                || job_arena_len > self.arena_slab.cap
            {
                if !batch.is_empty() {
                    break;
                }
                let report = job_size_report(job, &self.weights.layout);
                let (_, tag, reply) = self.waiting.pop_front().unwrap();
                let _ = reply.send((tag, Err(format!(
                    "gpu: job exceeds capacity: rows {}/{} (batch {MAX_BATCH_ROWS}), configs {}/{} (batch {MAX_BATCH_CFG}), table {}/{}, arena {}/{}\n{}",
                    job_rows,
                    self.rows.cap,
                    job_cfgs,
                    MAX_BATCH_CFG,
                    job_table_len,
                    self.tbl_slab.cap,
                    job_arena_len,
                    self.arena_slab.cap,
                    report,
                ))));
                continue;
            }
            if batch.len() == MAX_BATCH
                || rows + t.rows > MAX_BATCH_ROWS
                || cfgs + t.ncfg > MAX_BATCH_CFG
            {
                break;
            }
            if self.free.is_empty()
                || !self.rows.fits(t.rows)
                || !self.tbl_slab.fits(table_len(job))
                || !self
                    .arena_slab
                    .fits(arena_len_of(job, &self.weights.layout))
            {
                break;
            }
            let (job, tag, reply) = self.waiting.pop_front().unwrap();
            rows += job.tables.rows;
            cfgs += job.tables.ncfg;
            match self.admit(job, tag, reply.clone()) {
                Ok(slot) => batch.push(slot),
                Err(e) => {
                    let _ = reply.send((0, Err(e)));
                }
            }
        }
        if batch.is_empty() {
            return;
        }
        crate::prof::gpu_admit(batch.len(), rows, cfgs);
        self.build_batch(&batch);
        if self.sync_phase {
            if let Err(e) = self.build.synchronize() {
                eprintln!("gpu: phase 'admission build' failed: {e:?}");
            }
        }
        // The compute stream may not run these solves' ticks until the build
        // lands; one event orders the two streams without blocking the host.
        let ev = self.build.record_event(None).ok();
        if let Some(ev) = ev {
            let _ = self.compute.wait(&ev);
        }
        self.live_dirty = true;
    }

    /// Upload one job: tables into the table slab, zeroed arenas, descriptor.
    fn admit(
        &mut self,
        job: Job,
        tag: usize,
        reply: mpsc::Sender<(usize, Result<Trip1, String>)>,
    ) -> Result<usize, String> {
        if job.meta.warm > 0.0 {
            return Err("gpu: warm start is not implemented (plan A4)".into());
        }
        check_tables(&job)?;
        let t = &job.tables;
        let l = &self.weights.layout;
        if t.nlevels > MAX_LEVELS {
            return Err(format!(
                "gpu: tree has {} public levels, maximum is {MAX_LEVELS}",
                t.nlevels
            ));
        }
        let derived = Derived::new(&job);
        let nc = |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
        let nc_root = [nc(0, 0), nc(0, 1)];
        let nsnaps = if job.meta.snapshots {
            job.meta.snap_iters.len()
        } else {
            0
        };
        if job.meta.snap_iters.len() > 16 {
            return Err("gpu: more than 16 kept iterates".into());
        }
        let sizes = Sizes {
            reach_len: t.reach_len,
            vals_len: derived.vals_len(),
            ncells: t.ncells,
            nsnaps,
            leaf_configs: t.snapshot_configs,
            ncfg: t.ncfg,
            nc_root: nc_root[0] + nc_root[1],
            nroots: job.carried.len(),
            dg: l.dg,
            rk: l.rank,
            de: l.de,
        };
        let (aoff, arena_len) = arena_offsets(&sizes);
        let blob_len = packed_table_len(t, &derived);

        let slot = self.free.pop().ok_or("gpu: live set full")?;
        let row0 = self.rows.alloc(t.rows).ok_or("gpu: row pool full")?;
        let tbl_at = self.tbl_slab.alloc(blob_len).ok_or("gpu: table slab full")?;
        let arena_at = self
            .arena_slab
            .alloc(arena_len)
            .ok_or("gpu: arena slab full")?;

        // Pack straight into page-locked staging and upload from there; zero
        // the arena on the same stream.
        let build = self.build.clone();
        let toff = match self.upload.reserve(blob_len, &build) {
            Some(stage) => {
                let (toff, used) = pack_tables_into(t, &derived, stage);
                if used > blob_len {
                    return Err(format!(
                        "gpu: packed {used} table bytes into a {blob_len}-byte range"
                    ));
                }
                let mut dst = self.tables.slice_mut(tbl_at..tbl_at + used);
                build
                    .memcpy_htod(&stage[..used], &mut dst)
                    .map_err(|e| format!("{e:?}"))?;
                toff
            }
            None => {
                let mut blob = vec![0u8; blob_len];
                let (toff, used) = pack_tables_into(t, &derived, &mut blob);
                let mut dst = self.tables.slice_mut(tbl_at..tbl_at + used);
                build
                    .memcpy_htod(&blob[..used], &mut dst)
                    .map_err(|e| format!("{e:?}"))?;
                toff
            }
        };
        {
            let mut az = self.arenas.slice_mut(arena_at..arena_at + arena_len);
            build
                .memset_zeros(&mut az)
                .map_err(|e| format!("{e:?}"))?;
        }
        let (tbl_ptr, _) = self.tables.device_ptr(&self.build);
        let (arena_ptr, _) = self.arenas.device_ptr(&self.build);
        let mut desc = Desc {
            toff_sum: toff.iter().fold(0i32, |a, &x| a.wrapping_add(x as i32)),
            tbl: (tbl_ptr as usize + tbl_at) as *const u8,
            arena: (arena_ptr as usize as *mut f32).wrapping_add(arena_at),
            toff,
            aoff,
            nodes: t.nodes as i32,
            rows: t.rows as i32,
            nleaf: t.nleaf as i32,
            nterm: t.nterm as i32,
            ncells: t.ncells as i32,
            ncfg: t.ncfg as i32,
            nlevels: t.nlevels as i32,
            ndec: [
                derived.decision0.len() as i32,
                derived.decision1.len() as i32,
            ],
            nsweep: derived.sweep_order.len() as i32,
            nsnaps: nsnaps as i32,
            nroots: job.carried.len() as i32,
            iters: job.meta.iters as i32,
            row0: row0 as i32,
            nc_root: [nc_root[0] as i32, nc_root[1] as i32],
            nc_leaf: [0, 0],
            leaf: -1,
            stage: STAGE_ITERATE,
            traverser: 0,
            snap_t: if nsnaps > 0 { 1 } else { 0 },
            first_query: 1,
            snapshots: job.meta.snapshots as i32,
            alpha: job.meta.cfr.alpha,
            beta: job.meta.cfr.beta,
            gamma: job.meta.cfr.gamma,
            predict: job.meta.cfr.predict,
            ..Desc::default()
        };
        for (k, &it) in job.meta.snap_iters.iter().enumerate() {
            desc.snap_iters[k] = it as i32;
        }
        // The first iteration's DCFR discounts (advance_state computes the
        // rest): m = steps[traverser] + 1 = 1.
        let f = |p: f32| -> f32 {
            if p.is_infinite() {
                if p > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                let x = 1.0f32.powf(p);
                x / (x + 1.0)
            }
        };
        desc.da = f(job.meta.cfr.alpha);
        desc.db = f(job.meta.cfr.beta);
        desc.ds = (1.0f32 / 2.0).powf(job.meta.cfr.gamma);
        let _ = self
            .build
            .memcpy_htod(&[desc], &mut self.descs.slice_mut(slot..slot + 1));

        self.live[slot] = Some(Solve {
            id: self.next_id,
            meta: job.meta,
            tbl_at,
            arena_at,
            row0,
            nrows: t.rows,
            aoff,
            desc,
            snap_leaves: t.leaf_rows.iter().chain(&t.term_leaves).copied().collect(),
            snap_coff: t.snap_coff.clone(),
            snapshot_configs: t.snapshot_configs,
            level_counts: t
                .level_start
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .collect(),
            decision_cfg_counts: [
                *derived.dec_cfg0.last().unwrap_or(&0) as usize,
                *derived.dec_cfg1.last().unwrap_or(&0) as usize,
            ],
            sweep_cfg_level_counts: [
                level_cfg_counts(&derived.sweep_level_start, &derived.sweep_cfg0),
                level_cfg_counts(&derived.sweep_level_start, &derived.sweep_cfg1),
            ],
            stage: STAGE_ITERATE,
            t: 0,
            step: 0,
            nroots: job.carried.len(),
            nsnaps,
            nc_root,
            ncells: t.ncells,
            trip1: Some((tag, reply)),
            draining: false,
        });
        self.next_id += 1;
        Ok(slot)
    }

    /// The batched build: pack the batch, run each tower's GEMM chain once,
    /// scatter results into the solves, initialise their CFR state.
    fn build_batch(&mut self, slots: &[usize]) {
        let l = self.weights.layout.clone();
        let c = self.weights.ctx;
        let stream = self.build.clone();
        // Batch map + a slots list + per-batch headers, staged into g_meta
        // slots past the tick's fixed ones? No: build uses its own header
        // upload each batch (it is per-batch by nature). Reuse bmap buffer
        // for the slot list too: [map entries..., slots...].
        let mut map = Vec::with_capacity(slots.len() * BMAP_INTS);
        let (mut rows, mut cfgs, mut leaves) = (0usize, 0usize, 0usize);
        for &s in slots {
            let sv = self.live[s].as_ref().unwrap();
            map.extend_from_slice(&[
                s as i32,
                rows as i32,
                sv.nrows as i32,
                cfgs as i32,
                self.desc_ncfg(s) as i32,
            ]);
            rows += sv.nrows;
            cfgs += self.desc_ncfg(s);
            leaves += (sv.desc.nleaf + sv.desc.nterm) as usize;
        }
        let _ = stream.memcpy_htod(&map, &mut self.bmap.slice_mut(..map.len()));
        let nb = slots.len();
        let slot_ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        // Batch slot list sits after the map in bmap.
        let at = map.len();
        let _ = stream.memcpy_htod(&slot_ids, &mut self.bmap.slice_mut(at..at + nb));
        let (bmap_ptr, _) = self.bmap.device_ptr(&stream);
        let slots_dev = (bmap_ptr as usize + at * 4) as *const i32;

        let hdr = |n: i32, mode: i32, p: i32, level: i32, total: i32| GroupDev {
            slots: slots_dev,
            prefix: std::ptr::null(),
            n,
            mode,
            p_player: p,
            level,
            total,
        };
        // One reusable admission header, owned exclusively by build stream.
        let set = |g: GroupDev, me: &mut Self| {
            let _ = me.build.memcpy_htod(&[g], &mut me.build_meta);
        };
        let grid = |items: usize| (items.max(1) as u32).div_ceil(BLOCK);
        let wgrid = |items: usize| ((items.max(1) * 32) as u32).div_ceil(BLOCK);

        // ---- cards: pack -> GEMM chain -> finish (adds wid) --------------
        set(
            hdr(nb as i32, 0, 0, 0, (nb * crate::rebel::NTYPE) as i32),
            self,
        );
        self.fire_b(
            |k| &k.pack_cards,
            grid(nb * crate::rebel::NTYPE * crate::units::CARD_FEATS),
        );
        let nrows_card = nb * crate::rebel::NTYPE;
        let mut src = c.bg as *const f32;
        let mut buf = 0; // 0 -> bh, 1 -> bh2
        for (k, s) in l.card.iter().enumerate() {
            let dst = if buf == 0 { c.bh } else { c.bh2 };
            gemm(
                &self.build_blas,
                nrows_card,
                s.o,
                s.i,
                src,
                s.i,
                self.weights_at(s.w),
                s.o,
                dst,
                s.o,
                0.0,
            );
            if k + 1 < l.card.len() {
                set(hdr(nb as i32, 0, buf, k as i32, nrows_card as i32), self);
                self.fire_b(|f| &f.bias_act, grid(nrows_card * s.o));
                src = dst;
                buf ^= 1;
            } else if buf == 1 {
                // cards_finish reads bh; copy parity by re-running the GEMM
                // into bh would be silly — instead cards_finish reads the
                // buffer named by mode. (mode = which buffer holds the out.)
            }
        }
        let card_buf = (l.card.len() + 1) % 2; // which buffer the last GEMM wrote
        set(
            hdr(
                nb as i32,
                card_buf as i32,
                0,
                (l.card.len() - 1) as i32,
                nrows_card as i32,
            ),
            self,
        );
        self.fire_b(|k| &k.cards_finish, grid(nb * crate::rebel::NTYPE * l.de));
        // ---- pile: pe tail, count pack, count GEMM ------------------------
        // `pile_pe` writes after the per-row count block in bg: `n` is the
        // number of jobs it computes, while `total` is that row-block size.
        set(hdr(nb as i32, 0, 0, 0, rows as i32), self);
        self.fire_b(|k| &k.pile_pe, grid(nb * crate::rebel::NTYPE * l.de));
        set(hdr(nb as i32, 0, 0, 0, rows as i32), self);
        self.fire_b(
            |k| &k.pack_piles,
            grid(rows * crate::rebel::NTYPE * crate::rebel::PILE_COUNTS),
        );
        gemm(
            &self.build_blas,
            rows * crate::rebel::NTYPE,
            l.de,
            crate::rebel::PILE_COUNTS,
            c.bg as *const f32,
            crate::rebel::PILE_COUNTS,
            self.weights_at(l.pile.w),
            l.de,
            c.bh,
            l.de,
            0.0,
        );
        // ---- trunk: assemble -> pub chain -> pub_out -> scatter h0 --------
        set(hdr(nb as i32, 0, 0, 0, rows as i32), self);
        self.fire_b(|k| &k.assemble, rows as u32);
        let mut src = c.bx as *const f32;
        let mut buf = 0;
        for (k, s) in l.pub_lin.iter().enumerate() {
            let dst = if buf == 0 { c.bh } else { c.bh2 };
            gemm(
                &self.build_blas,
                rows,
                s.o,
                s.i,
                src,
                s.i,
                self.weights_at(s.w),
                s.o,
                dst,
                s.o,
                0.0,
            );
            set(hdr(nb as i32, 0, buf, k as i32, rows as i32), self);
            self.fire_b(|f| &f.trunk_norm, wgrid(rows));
            src = dst;
            buf ^= 1;
        }
        let dst = if buf == 0 { c.bh } else { c.bh2 };
        gemm(
            &self.build_blas,
            rows,
            l.head_in,
            l.pub_out.i,
            src,
            l.pub_out.i,
            self.weights_at(l.pub_out.w),
            l.head_in,
            dst,
            l.head_in,
            0.0,
        );
        set(hdr(nb as i32, 0, buf, 0, rows as i32), self);
        self.fire_b(|k| &k.scatter_h0, grid(rows * l.head_in));
        // ---- holding tower -------------------------------------------------
        set(hdr(nb as i32, 0, 0, 0, cfgs as i32), self);
        self.fire_b(|k| &k.holding_in, grid(cfgs * crate::rebel::NSLOT));
        let nsl = cfgs * crate::rebel::NSLOT;
        let mut src = c.bg as *const f32;
        let mut buf = 0;
        for (k, s) in l.slot.iter().enumerate() {
            let dst = if buf == 0 { c.bh } else { c.bh2 };
            gemm(
                &self.build_blas,
                nsl,
                s.o,
                s.i,
                src,
                s.i,
                self.weights_at(s.w),
                s.o,
                dst,
                s.o,
                0.0,
            );
            set(hdr(nb as i32, 2, buf, k as i32, nsl as i32), self);
            self.fire_b(|f| &f.bias_act, grid(nsl * s.o));
            src = dst;
            buf ^= 1;
        }
        let dst = if buf == 0 { c.bh } else { c.bh2 };
        gemm(
            &self.build_blas,
            nsl,
            l.dg,
            l.slot_out.i,
            src,
            l.slot_out.i,
            self.weights_at(l.slot_out.w),
            l.dg,
            dst,
            l.dg,
            0.0,
        );
        // slot_sum reads `p_player ? bh2 : bh` and writes the other: z ends
        // in the opposite buffer.
        set(hdr(nb as i32, 0, buf, 0, cfgs as i32), self);
        self.fire_b(|k| &k.slot_sum, wgrid(cfgs));
        let zbuf = buf ^ 1; // where z lives now
        for (k, (a, bres)) in l.res.iter().enumerate() {
            let (zp, rp) = if zbuf == 0 {
                (c.bh, c.bh2)
            } else {
                (c.bh2, c.bh)
            };
            gemm(
                &self.build_blas,
                cfgs,
                l.dg,
                l.dg,
                zp as *const f32,
                l.dg,
                self.weights_at(a.w),
                l.dg,
                rp,
                l.dg,
                0.0,
            );
            set(hdr(nb as i32, 3, zbuf ^ 1, k as i32, cfgs as i32), self);
            self.fire_b(|f| &f.bias_act, grid(cfgs * l.dg));
            gemm(
                &self.build_blas,
                cfgs,
                l.dg,
                l.dg,
                rp as *const f32,
                l.dg,
                self.weights_at(bres.w),
                l.dg,
                zp,
                l.dg,
                1.0,
            );
            set(hdr(nb as i32, 4, zbuf, k as i32, cfgs as i32), self);
            self.fire_b(|f| &f.bias_act, grid(cfgs * l.dg));
            let _ = zbuf; // z stays in zp
        }
        let zp = if zbuf == 0 { c.bh } else { c.bh2 };
        gemm(
            &self.build_blas,
            cfgs,
            l.rank + 1,
            l.dg,
            zp as *const f32,
            l.dg,
            self.weights_at(l.wg.w),
            l.rank + 1,
            c.bg,
            l.rank + 1,
            0.0,
        );
        set(hdr(nb as i32, 0, zbuf, 0, cfgs as i32), self);
        self.fire_b(|k| &k.scatter_zg, grid(cfgs * (l.dg + l.rank + 1)));
        // ---- CFR init: uniform strategy, initial reach, seeded average ----
        set(hdr(nb as i32, 0, 0, 0, 0), self);
        self.fire_b(|k| &k.init_strategy, nb as u32);
        set(hdr(nb as i32, 1, 0, 0, 0), self);
        self.fire_b(|k| &k.reach_prop, nb as u32);
        set(hdr(nb as i32, 0, 0, 0, 0), self);
        self.fire_b(|k| &k.seed_avg, nb as u32);
        set(hdr(nb as i32, 0, 0, 0, leaves as i32), self);
        self.fire_b(|k| &k.seed_snapshot_beliefs, wgrid(leaves));
        let _ = zbuf;
    }

    #[track_caller]
    fn fire_b(&self, which: fn(&Kernels) -> &CudaFunction, grid: u32) {
        if grid == 0 {
            return;
        }
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = which(&self.f).clone();
        let mut b = self.build.launch_builder(&f);
        b.arg(&self.descs);
        b.arg(&self.build_meta);
        b.arg(&self.ctx);
        report(unsafe { b.launch(cfg) }, grid);
        // Under `WARCHEST_GPU_SYNC_PHASE` the build is stepped the same way
        // the tick is: a sync per launch, named by the line that issued it.
        // A GEMM's fault lands on the next kernel's line, which is close
        // enough to name the chain it belongs to.
        if self.sync_phase {
            if let Err(e) = self.build.synchronize() {
                eprintln!(
                    "gpu: build step from {} (grid {grid}) failed: {e:?}",
                    std::panic::Location::caller()
                );
            }
        }
    }

    fn desc_ncfg(&self, slot: usize) -> usize {
        // ncfg is not mirrored on Solve; read it from the job we admitted.
        // (Stored at admission; kept simple by recomputing from aoff spans.)
        let sv = self.live[slot].as_ref().unwrap();
        let z = (sv.aoff[Arena::g as usize] - sv.aoff[Arena::z as usize]) as usize;
        z / self.weights.layout.dg
    }

    // ------------------------------------------------------------ the trips

    /// Trip 1: hand the reference strategy and the carried-root values to
    /// the downloader. The solve stays resident (carry stage), but its
    /// network rows are free the moment the value passes are done.
    fn send_trip1(&mut self, s: usize) {
        let sv = self.live[s].as_mut().unwrap();
        let Some((tag, tx)) = sv.trip1.take() else {
            return;
        };
        let (arena_ptr, _) = self.arenas.device_ptr(&self.compute);
        let at = |a: Arena| arena_ptr + ((sv.arena_at + sv.aoff[a as usize] as usize) * 4) as u64;
        let stride = sv.nc_root[0] + sv.nc_root[1];
        let (id, final_slot, strategy, root_vals, nc_root) = (
            sv.id,
            (sv.nsnaps == 0).then_some(s),
            (at(Arena::avg), sv.ncells),
            (at(Arena::root_vals), sv.nroots * stride),
            sv.nc_root,
        );
        // The head rows are dead from here on: value passes are done and the
        // carry stage never reads them. Freeing now is what keeps the head
        // GEMM's row span tight.
        self.rows.free(sv.row0);
        sv.nrows = 0;
        let final_desc = if sv.nsnaps == 0 {
            sv.draining = true;
            sv.stage = STAGE_DRAIN;
            sv.desc.stage = STAGE_DRAIN;
            Some(sv.desc)
        } else {
            None
        };
        self.solved += 1;
        if let Some(desc) = final_desc {
            let _ = self
                .compute
                .memcpy_htod(&[desc], &mut self.descs.slice_mut(s..s + 1));
        }
        // The completion point comes after publishing DRAIN for an evaluation
        // solve. Both consumers need it: download reads the result arenas,
        // while build may recycle this solve's descriptor, tables, arena and
        // head rows after the downloader reports completion.
        let ev = self.compute.record_event(None).expect("event");
        let _ = self.build.wait(&ev);
        let req = Dl::Trip1 {
            event: ev,
            tag,
            reply: tx,
            id,
            final_slot,
            strategy,
            root_vals,
            nc_root,
        };
        if let Some(tx) = &self.dl_tx {
            let _ = tx.send(req);
        }
    }

    /// The walk left the tree at `leaf`. Its beliefs under every intermediate
    /// average were stored when that average was created, so trip 2 is only a
    /// strided download from the chosen row — no tree replay or tick.
    fn start_carry(&mut self, id: u64, leaf: u32, reply: mpsc::Sender<Result<Trip2, String>>) {
        let Some(s) = (0..CAP).find(|&s| self.live[s].as_ref().is_some_and(|v| v.id == id)) else {
            let _ = reply.send(Err(format!("gpu: unknown solve id {id}")));
            return;
        };
        let sv = self.live[s].as_mut().unwrap();
        // One carry per solve. A second request would queue a second download,
        // and its completion would free a slot that by then belongs to someone
        // else.
        if sv.draining {
            let _ = reply.send(Err(format!("gpu: solve {id} is already draining")));
            return;
        }
        let Some(row) = sv.snap_leaves.iter().position(|&x| x == leaf) else {
            let _ = reply.send(Err(format!("gpu: exit node {leaf} is not a tree leaf")));
            return;
        };
        let leaf_off = [
            sv.snap_coff[2 * row] as usize,
            sv.snap_coff[2 * row + 1] as usize,
        ];
        let nc_leaf = [
            leaf_off[1] - leaf_off[0],
            sv.snap_coff[2 * row + 2] as usize - leaf_off[1],
        ];
        // The gather below reads `snap_beliefs[s * leaf_configs + off]` for
        // every kept iterate `s` and every config of this exit leaf, straight
        // off the device. Check that range against the arena the solve was
        // actually cut: an overrun here is invisible until CUDA reports an
        // illegal address on some unrelated stream, minutes later.
        let snap_len = (sv.aoff[Arena::snap_beliefs as usize + 1]
            - sv.aoff[Arena::snap_beliefs as usize]) as usize;
        let last = sv.nsnaps.saturating_sub(2) * sv.snapshot_configs
            + leaf_off[1]
            + nc_leaf[1];
        if sv.nsnaps > 1 && last > snap_len {
            let msg = format!(
                "gpu: trip 2 for exit leaf {leaf} (row {row} of {}) would read \
                 {last} of a {snap_len}-float snapshot arena ({} kept iterates \
                 x {} configs, leaf window {}..{})",
                sv.snap_leaves.len(),
                sv.nsnaps,
                sv.snapshot_configs,
                leaf_off[0],
                leaf_off[1] + nc_leaf[1],
            );
            let _ = reply.send(Err(msg));
            return;
        }
        sv.draining = true;
        sv.stage = STAGE_DRAIN;
        sv.desc.stage = STAGE_DRAIN;
        let final_desc = sv.desc;
        let (arena_ptr, _) = self.arenas.device_ptr(&self.compute);
        let snap_beliefs =
            arena_ptr + ((sv.arena_at + sv.aoff[Arena::snap_beliefs as usize] as usize) * 4) as u64;
        let (leaf_configs, nsnaps) = (sv.snapshot_configs, sv.nsnaps);
        let _ = self
            .compute
            .memcpy_htod(&[final_desc], &mut self.descs.slice_mut(s..s + 1));
        // Publish DRAIN before the completion event. The downloader's `done`
        // message permits immediate slot/table/arena reuse, so the build
        // stream must cross the same fence before touching recycled ranges.
        let ev = self.compute.record_event(None).expect("event");
        let _ = self.build.wait(&ev);
        let req = Dl::Trip2 {
            event: ev,
            reply,
            slot: s,
            id,
            snap_beliefs,
            leaf_configs,
            leaf_off,
            nc_leaf,
            nsnaps,
        };
        if let Some(tx) = &self.dl_tx {
            let _ = tx.send(req);
        }
    }

    /// Free a slot, its tables, its arena and its rows.
    ///
    /// `id` names the solve that finished, and it is not redundant with the
    /// slot: a slot is reused the moment it is freed, so a completion that
    /// arrives for a solve which has already left would otherwise free the
    /// ranges of whichever solve moved in. The victim keeps running, and its
    /// tables are quietly overwritten by the next admission that lands on the
    /// same bytes — which surfaces, much later, as an illegal address in an
    /// unrelated kernel reading a float where an index should be.
    fn release(&mut self, s: usize, id: u64) {
        match self.live[s].as_ref().map(|sv| sv.id) {
            None => return,
            Some(here) if here != id => {
                eprintln!(
                    "gpu: completion for solve {id} would have freed slot {s}, \
                     which now holds solve {here}"
                );
                return;
            }
            Some(_) => {}
        }
        if let Some(sv) = self.live[s].take() {
            // The downloader's result event can sit in front of later ticks
            // which were already queued while this slot was DRAINing. Fence
            // the build stream at the compute tail observed *now*, before the
            // slot, descriptor, tables or arena can be assigned to a new
            // solve. Otherwise a new descriptor can make those older kernels
            // treat the recycled slot as active and follow stale flat-prefix
            // entries into the new job.
            if let Ok(ev) = self.compute.record_event(None) {
                let _ = self.build.wait(&ev);
            }
            if sv.nrows > 0 {
                self.rows.free(sv.row0);
            }
            self.tbl_slab.free(sv.tbl_at);
            self.arena_slab.free(sv.arena_at);
            self.free.push(s);
            self.live_dirty = true;
        }
    }
}

/// Configs per backward-sweep level, from the level boundaries and the
/// running per-node config count they index into.
fn level_cfg_counts(level_start: &[u32], cfg_prefix: &[u32]) -> Vec<usize> {
    level_start
        .windows(2)
        .map(|w| (cfg_prefix[w[1] as usize] - cfg_prefix[w[0] as usize]) as usize)
        .collect()
}

/// Report a failed launch with the tick line that issued it. Silent on the
/// happy path; only ever prints when the device is already in trouble.
#[track_caller]
fn report<T, E: std::fmt::Debug>(r: Result<T, E>, grid: u32) {
    if let Err(e) = r {
        eprintln!(
            "gpu: launch from {} (grid {grid}) failed: {e:?}",
            std::panic::Location::caller()
        );
    }
}

fn h_stride_of(l: &V3Layout) -> usize {
    std::iter::once(l.head_in)
        .chain(l.hmlp.iter().map(|s| s.o))
        .max()
        .unwrap()
}

fn merge_ctx(dst: &mut Ctx, pools: &Ctx) {
    dst.h0 = pools.h0;
    dst.xb = pools.xb;
    dst.h = pools.h;
    dst.h2 = pools.h2;
    dst.u = pools.u;
    dst.bx = pools.bx;
    dst.bh = pools.bh;
    dst.bh2 = pools.bh2;
    dst.bg = pools.bg;
    dst.bmap = pools.bmap;
}

/// The table lengths every kernel indexes by a scalar of the descriptor.
///
/// The job reader checks the shape of a job it decodes, but a job built in
/// this process never passes through it, so nothing checks these. A short
/// table is not a crash where it is built: it is an out-of-range read on the
/// device, minutes later, reported as an illegal address in whichever kernel
/// happened to be running.
fn check_tables(job: &Job) -> Result<(), String> {
    let t = &job.tables;
    let want = [
        ("trans", t.trans.len(), t.ncells),
        ("legal_bits", t.legal_bits.len(), t.ncells.div_ceil(8)),
        ("soff", t.soff.len(), t.nodes + 1),
        ("bfs_order", t.bfs_order.len(), t.nodes),
        ("node_parent", t.node_parent.len(), t.nodes),
        ("cfg_off", t.cfg_off.len(), 2 * t.nodes + 1),
        ("reach_off", t.reach_off.len(), t.nodes + 1),
        ("level_start", t.level_start.len(), t.nlevels + 1),
        ("snap_coff", t.snap_coff.len(), 2 * (t.nleaf + t.nterm) + 1),
    ];
    for (name, got, want) in want {
        if got != want {
            return Err(format!(
                "gpu: job table {name} has {got} entries, not {want} \
                 (nodes {}, cells {}, levels {})",
                t.nodes, t.ncells, t.nlevels,
            ));
        }
    }
    if t.soff[t.nodes] as usize != t.ncells {
        return Err(format!(
            "gpu: job soff ends at {}, not at its {} cells",
            t.soff[t.nodes], t.ncells,
        ));
    }
    Ok(())
}

fn table_len(job: &Job) -> usize {
    let d = Derived::new(job);
    packed_table_len(&job.tables, &d)
}

fn arena_len_of(job: &Job, l: &V3Layout) -> usize {
    let t = &job.tables;
    let d = Derived::new(job);
    let nc = |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
    let nsnaps = if job.meta.snapshots {
        job.meta.snap_iters.len()
    } else {
        0
    };
    let sizes = Sizes {
        reach_len: t.reach_len,
        vals_len: d.vals_len(),
        ncells: t.ncells,
        nsnaps,
        leaf_configs: t.snapshot_configs,
        ncfg: t.ncfg,
        nc_root: nc(0, 0) + nc(0, 1),
        nroots: job.carried.len(),
        dg: l.dg,
        rk: l.rank,
        de: l.de,
    };
    arena_offsets(&sizes).1
}

/// A detailed one-shot report for an impossible job. This deliberately lives
/// on the rejection path: it makes a real capacity failure actionable without
/// adding bookkeeping to every successful admission.
fn job_size_report(job: &Job, l: &V3Layout) -> String {
    let t = &job.tables;
    let d = Derived::new(job);
    let nc = |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
    let nsnaps = if job.meta.snapshots {
        job.meta.snap_iters.len()
    } else {
        0
    };
    let nc_root = nc(0, 0) + nc(0, 1);

    let mut table: Vec<(&str, usize)> = vec![
        ("node_kind", t.node_kind.len()),
        ("node_player", t.node_player.len()),
        ("node_child_start", 4 * t.node_child_start.len()),
        ("node_child", 4 * t.node_child.len()),
        ("obs_off", 4 * t.obs_off.len()),
        ("obs_start", 4 * t.obs_start.len()),
        ("obs_act", 4 * t.obs_act.len()),
        ("obs_child", 4 * t.obs_child.len()),
        ("legal_bits", t.legal_bits.len()),
        ("trans", 4 * t.trans.len()),
        ("draw_off", 4 * t.draw_off.len()),
        ("draw_to", 4 * t.draw_to.len()),
        ("draw_p", 4 * t.draw_p.len()),
        ("draw_row_off", 4 * t.draw_row_off.len()),
        ("draw_row_start", 4 * t.draw_row_start.len()),
        ("cfg_off", 4 * t.cfg_off.len()),
        ("reach_off", 4 * t.reach_off.len()),
        ("soff", 4 * t.soff.len()),
        ("voff", 4 * d.voff.len()),
        ("act_off", 4 * d.act_off.len()),
        ("node_parent", 4 * t.node_parent.len()),
        ("rev_row_of", 4 * t.rev_row_of.len()),
        ("rev_start", 4 * t.rev_start.len()),
        ("rev_src", 4 * t.rev_src.len()),
        ("rev_cell", 4 * t.rev_cell.len()),
        ("rvd_row_of", 4 * t.rvd_row_of.len()),
        ("rvd_start", 4 * t.rvd_start.len()),
        ("rvd_src", 4 * t.rvd_src.len()),
        ("rvd_p", 4 * t.rvd_p.len()),
        ("leaf_rows", 4 * t.leaf_rows.len()),
        ("term_leaves", 4 * t.term_leaves.len()),
        ("terminal_utility", 4 * t.terminal_utility.len()),
        ("leaf_coff", 4 * t.leaf_coff.len()),
        ("leaf_cidx", 4 * t.leaf_cidx.len()),
        ("snap_coff", 4 * t.snap_coff.len()),
        ("leaf_raw", t.leaf_raw.len()),
        ("card_feat", 4 * t.card_feat.len()),
        ("cphi", 4 * t.cphi.len()),
        ("bfs_order", 4 * t.bfs_order.len()),
        ("level_start", 4 * t.level_start.len()),
        ("ids", t.ids.len()),
        ("root", 4 * d.root.len()),
        ("carried", 4 * d.carried.len()),
    ];
    table.sort_unstable_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));

    let mut arena: Vec<(&str, usize)> = vec![
        (
            "snapshot_beliefs",
            nsnaps.saturating_sub(1) * t.snapshot_configs,
        ),
        ("snapshot_reach", (nsnaps > 1) as usize * t.reach_len),
        ("regret", t.ncells),
        ("instant_strategy", t.ncells),
        ("current_strategy", t.ncells),
        ("strategy_sum", t.ncells),
        ("average_strategy", t.ncells),
        ("reach", t.reach_len),
        ("values", d.vals_len()),
        ("config_embedding", t.ncfg * l.dg),
        ("config_readout", t.ncfg * (l.rank + 1)),
        ("action_embedding", crate::rebel::NTYPE * l.de),
        ("root_values", job.carried.len() * nc_root),
    ];
    arena.sort_unstable_by_key(|&(_, floats)| std::cmp::Reverse(floats));

    let fmt = |name: &str, bytes: usize| format!("{name}={:.1}MiB", bytes as f64 / 1_048_576.0);
    let top_table = table
        .iter()
        .take(12)
        .map(|&(name, bytes)| fmt(name, bytes))
        .collect::<Vec<_>>()
        .join(", ");
    let top_arena = arena
        .iter()
        .map(|&(name, floats)| fmt(name, 4 * floats))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "shape: nodes={}, leaves={}, terminal={}, rows={}, cells={}, reach={}, values={}, snapshots={}, levels={}\ntable: {}\narena: {}",
        t.nodes,
        t.nleaf,
        t.nterm,
        t.rows,
        t.ncells,
        t.reach_len,
        d.vals_len(),
        nsnaps,
        t.nlevels,
        top_table,
        top_arena,
    )
}

/// Compare the device's view of `Desc` with the host's, field by field.
fn check_abi(stream: &Arc<CudaStream>, probe: &CudaFunction) -> Result<(), String> {
    let want = super::layout::abi_expected();
    let mut got = stream
        .alloc_zeros::<i32>(want.len())
        .map_err(|e| format!("{e:?}"))?;
    let mut b = stream.launch_builder(probe);
    b.arg(&mut got);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.map_err(|e| format!("abi probe: {e:?}"))?;
    let mut host = vec![0i32; want.len()];
    stream
        .memcpy_dtoh(&got, &mut host)
        .map_err(|e| format!("{e:?}"))?;
    stream.synchronize().map_err(|e| format!("{e:?}"))?;
    for ((name, &w), &g) in super::layout::abi_names().iter().zip(&want).zip(&host) {
        if w != g as usize {
            return Err(format!(
                "gpu: Desc.{name} is {w} on the host, {g} on the device"
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ helpers

fn htod<T: cudarc::driver::DeviceRepr + Unpin>(
    stream: &Arc<CudaStream>,
    v: &[T],
) -> Result<CudaSlice<T>, String> {
    let mut buf = unsafe { stream.alloc(v.len().max(1)) }.map_err(|e| format!("{e:?}"))?;
    stream
        .memcpy_htod(v, &mut buf)
        .map_err(|e| format!("{e:?}"))?;
    Ok(buf)
}

fn ptr<T>(stream: &Arc<CudaStream>, buf: &CudaSlice<T>) -> *const T {
    let (p, _sync) = buf.device_ptr(stream);
    p as usize as *const T
}

fn ptr_mut<T>(stream: &Arc<CudaStream>, buf: &mut CudaSlice<T>) -> *mut T {
    let (p, _sync) = buf.device_ptr_mut(stream);
    p as usize as *mut T
}

/// One row-major GEMM over raw device pointers: `C[m,n] = A[m,k] . B[k,n]`
/// plus `beta * C`, with explicit row strides. cuBLAS is column-major, so it
/// is handed the transposed product.
#[allow(clippy::too_many_arguments)]
#[track_caller]
fn gemm(
    blas: &CudaBlas,
    m: usize,
    n: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    beta: f32,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let alpha = 1.0f32;
    // SAFETY: device pointers the service owns; shapes from the layout.
    let r = unsafe {
        cudarc::cublas::result::sgemm(
            *blas.handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha,
            b,
            ldb as i32,
            a,
            lda as i32,
            &beta,
            c,
            ldc as i32,
        )
    };
    report(r, m as u32);
}

// ------------------------------------------------------------------ probe

/// The arenas after a probe, for the phase-oracle test.
#[cfg(test)]
pub struct ProbeOut {
    pub e: Vec<f32>,
    pub z: Vec<f32>,
    pub g: Vec<f32>,
    pub h0: Vec<f32>,
    pub reach: Vec<f32>,
    pub vals: Vec<f32>,
    pub regret: Vec<f32>,
    pub inst: Vec<f32>,
    pub cur: Vec<f32>,
    pub sum_strat: Vec<f32>,
    pub avg: Vec<f32>,
    pub xb: Vec<f32>,
    pub u: Vec<f32>,
}

#[cfg(test)]
impl Service {
    /// Admit one solve, run one tick up to `upto`, read the arenas back.
    /// Test-only, so it may synchronize as much as it likes.
    pub(crate) fn probe(&mut self, job: Job, upto: Step) -> Result<ProbeOut, String> {
        let (tx, _rx) = mpsc::channel();
        self.waiting.push_back((job, 0, tx));
        self.admit_batch();
        let s = (0..CAP)
            .find(|&s| self.live[s].is_some())
            .ok_or("probe: not admitted")?;
        self.tick_upto(upto);
        self.build.synchronize().map_err(|e| format!("{e:?}"))?;
        self.compute.synchronize().map_err(|e| format!("{e:?}"))?;
        self.compute
            .context()
            .check_err()
            .map_err(|e| format!("{upto:?}: {e:?}"))?;
        let sv = self.live[s].as_ref().unwrap();
        let l = &self.weights.layout;
        let span = |k: Arena| {
            (
                sv.arena_at + sv.aoff[k as usize] as usize,
                (sv.aoff[k as usize + 1] - sv.aoff[k as usize]) as usize,
            )
        };
        let arena = |k: Arena| -> Vec<f32> {
            let (at, n) = span(k);
            self.d2h_arena(at, n)
        };
        let cells = |k: Arena| -> Vec<f32> {
            let (at, _) = span(k);
            self.d2h_arena(at, sv.ncells)
        };
        let pool = |idx: usize, stride: usize, n: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; n];
            if n > 0 {
                let p = &self._pools[idx];
                let _ = self
                    .compute
                    .memcpy_dtoh(&p.slice(sv.row0 * stride..sv.row0 * stride + n), &mut v);
                let _ = self.compute.synchronize();
            }
            v
        };
        let nleaf = sv.desc.nleaf as usize;
        let out = ProbeOut {
            e: arena(Arena::e),
            z: arena(Arena::z),
            g: arena(Arena::g),
            h0: pool(0, l.head_in, sv.nrows * l.head_in),
            reach: arena(Arena::reach),
            vals: arena(Arena::vals),
            regret: cells(Arena::regret),
            inst: cells(Arena::inst),
            cur: cells(Arena::cur),
            sum_strat: cells(Arena::sum_strat),
            avg: cells(Arena::avg),
            xb: pool(1, 2 * l.dg, nleaf * 2 * l.dg),
            u: pool(4, l.rank, nleaf * l.rank),
        };
        let id = self.live[s].as_ref().map(|sv| sv.id).unwrap_or_default();
        self.release(s, id);
        Ok(out)
    }

    fn d2h_arena(&self, at: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        if n > 0 {
            let _ = self
                .compute
                .memcpy_dtoh(&self.arenas.slice(at..at + n), &mut v);
            let _ = self.compute.synchronize();
        }
        v
    }
}
