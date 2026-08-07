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
use cudarc::driver::safe::{CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc;

use crate::net::V3Layout;
use crate::serialize::{Job, JobMeta};

use super::client::{Cmd, GpuClient, Trip1, Trip2};
use super::layout::{
    arena_offsets, cuda_preamble, pack_tables, Arena, Derived, Desc, Sizes, N_ARENAS,
    STAGE_CARRY, STAGE_ITERATE, STAGE_VALUE,
};

/// Live-set capacity, in solve slots.
const CAP: usize = 256;
/// Row-pool capacity; the real bound on the live set.
const MAX_ROWS: usize = 256 * 1024;
/// Threads per block.
const BLOCK: u32 = 256;
/// Admission batch bounds: jobs, network rows, distinct configs.
const MAX_BATCH: usize = 32;
const MAX_BATCH_ROWS: usize = 32 * 1024;
const MAX_BATCH_CFG: usize = 64 * 1024;
/// Ints per batch-map entry: slot, row0-in-batch, nrows, cfg0-in-batch, ncfg.
const BMAP_INTS: usize = 5;

/// Slab sizes, in MiB, overridable by environment (the T4 CI box is small).
fn env_mb(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default) << 20
}

// ------------------------------------------------------------------ context

/// The `Ctx` struct of kernels.cu: weight pointers, row pools, build scratch.
#[repr(C)]
#[derive(Clone, Copy)]
struct Ctx {
    card_w: [*const f32; 8], card_b: [*const f32; 8],
    wid: *const f32,
    pile_w: *const f32, pile_b: *const f32,
    pub_w: [*const f32; 8], pub_b: [*const f32; 8],
    pub_lnw: [*const f32; 8], pub_lnb: [*const f32; 8],
    pub_out_w: *const f32, pub_out_b: *const f32,
    wb: *const f32, ln1w: *const f32, ln1b: *const f32,
    hmlp_w: [*const f32; 8], hmlp_b: [*const f32; 8],
    wu_w: *const f32, wu_b: *const f32,
    slot_w: [*const f32; 8], slot_b: [*const f32; 8],
    slot_out_w: *const f32, slot_out_b: *const f32,
    res_aw: [*const f32; 4], res_ab: [*const f32; 4],
    res_bw: [*const f32; 4], res_bb: [*const f32; 4],
    wg_w: *const f32, wg_b: *const f32,
    h0: *mut f32, xb: *mut f32, h: *mut f32, h2: *mut f32, u: *mut f32,
    bx: *mut f32, bh: *mut f32, bh2: *mut f32, bg: *mut f32,
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
    ReadIt,        // p_player = -1 (iterate readout/backprop)
    ReadV0,        // p_player = 0 (value pass)
    ReadV1,        // p_player = 1
    ReachVC,       // mode = 0 (value + carry)
    ReachIt,       // mode = 1 (iterate)
    HmlpBase,      // + 2*k: layer k into h2 (mode 1) or h (mode 0)
}
const N_HDRS: usize = Hdr::HmlpBase as usize + 16;

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
        Slab { used: Vec::new(), cap, high: 0 }
    }
    fn fits(&self, n: usize) -> bool {
        self.gap(n).is_some()
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
        let (at, i) = self.gap(n)?;
        self.used.insert(i, (at, n));
        self.high = self.high.max(at + n);
        Some(at)
    }
    fn free(&mut self, at: usize) {
        self.used.retain(|&(s, _)| s != at);
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
                w.len(), b.len(), ln.len()
            ));
        }
        if l.card.len() > 8 || l.pub_lin.len() > 8 || l.hmlp.len() > 8
            || l.slot.len() > 8 || l.res.len() > 4
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
        Ok(Weights { dims: dims.to_vec(), layout: l, _w: wb, _b: bb, _ln: lb, ctx })
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
    belief_sums, head_entry, hmlp_act, readout, backprop_solve, regret_match,
    reach_prop, average, collect_root, leaf_beliefs, advance_state,
    pack_cards, bias_act, cards_finish, pile_pe, assemble, pack_piles,
    trunk_norm, scatter_h0, holding_in, slot_sum, scatter_zg, init_strategy,
    seed_avg, abi_probe,
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
    /// Support offsets per (node, player), for the trip-2 leaf counts.
    cfg_off: Vec<u32>,
    // Host shadow of the device state machine (advance_state).
    stage: i32,
    t: usize,
    step: usize,
    nroots: usize,
    nsnaps: usize,
    nleaf_cfg: [usize; 2],
    nc_root: [usize; 2],
    ncells: usize,
    trip1: Option<(usize, mpsc::Sender<(usize, Result<Trip1, String>)>)>,
    trip2: Option<mpsc::Sender<Result<Trip2, String>>>,
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
        strategy: (u64, usize),
        root_vals: (u64, usize),
        nc_root: [usize; 2],
    },
    Trip2 {
        event: CudaEvent,
        reply: mpsc::Sender<Result<Trip2, String>>,
        slot: usize,
        beliefs: (u64, usize),
        nc_leaf: [usize; 2],
        nsnaps: usize,
    },
}

fn downloader(stream: Arc<CudaStream>, rx: mpsc::Receiver<Dl>, done: mpsc::Sender<usize>) {
    let fetch = |ev: &CudaEvent, (at, len): (u64, usize)| -> Vec<f32> {
        let mut host = vec![0.0f32; len];
        if len == 0 {
            return host;
        }
        // Order the copy after the tick that produced the data, then wait on
        // this thread only; the compute stream never stalls.
        let _ = stream.wait(ev);
        unsafe {
            let _ = cudarc::driver::result::memcpy_dtoh_async(
                &mut host,
                at,
                stream.cu_stream(),
            );
        }
        let _ = stream.synchronize();
        host
    };
    while let Ok(req) = rx.recv() {
        match req {
            Dl::Trip1 { event, tag, reply, id, strategy, root_vals, nc_root } => {
                let strategy = fetch(&event, strategy);
                let flat = fetch(&event, root_vals);
                let stride = (nc_root[0] + nc_root[1]).max(1);
                let root_values = flat
                    .chunks_exact(stride)
                    .map(|c| [c[..nc_root[0]].to_vec(), c[nc_root[0]..].to_vec()])
                    .collect();
                let _ = reply.send((tag, Ok(Trip1 { id, strategy, root_values })));
            }
            Dl::Trip2 { event, reply, slot, beliefs, nc_leaf, nsnaps } => {
                let flat = fetch(&event, beliefs);
                let stride = (nc_leaf[0] + nc_leaf[1]).max(1);
                let out: Trip2 = flat
                    .chunks_exact(stride)
                    .take(nsnaps.saturating_sub(1))
                    .map(|c| [c[..nc_leaf[0]].to_vec(), c[nc_leaf[0]..].to_vec()])
                    .collect();
                let _ = reply.send(Ok(out));
                let _ = done.send(slot);
            }
        }
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
    bmap: CudaSlice<i32>,
    // Plumbing.
    rx: mpsc::Receiver<Cmd>,
    dl_tx: mpsc::Sender<Dl>,
    done_rx: mpsc::Receiver<usize>,
    waiting: std::collections::VecDeque<(Job, usize, mpsc::Sender<(usize, Result<Trip1, String>)>)>,
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
    let client = GpuClient::new(tx);
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::Builder::new()
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
    ready_rx
        .recv()
        .map_err(|_| "gpu service thread died".to_string())??;
    Ok(client)
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
        let compute = dev.new_stream().map_err(|e| format!("{e:?}"))?;
        let build = dev.new_stream().map_err(|e| format!("{e:?}"))?;
        let download = dev.new_stream().map_err(|e| format!("{e:?}"))?;
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
            nvrtc::CompileOptions { arch: Some(Box::leak(arch.into_boxed_str())), ..Default::default() },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = dev.load_module(ptx).map_err(|e| format!("module: {e:?}"))?;
        let f = Kernels::load(&module)?;

        // Pools: stable network rows, and the build scratch. Sizes derive
        // from the shape; the batch caps bound the scratch.
        let (pubw, hmlp, _, slotw) = layout.widths();
        let h_stride = std::iter::once(layout.head_in).chain(hmlp.iter().copied()).max().unwrap();
        let max_pub = pubw.iter().copied().chain([layout.head_in, layout.xdim()]).max().unwrap();
        let slot_max = slotw.iter().copied().chain([layout.dg, layout.hfeat()]).max().unwrap();
        let bh_len = (MAX_BATCH_ROWS * max_pub)
            .max(MAX_BATCH_CFG * crate::rebel::NSLOT * slot_max)
            .max(MAX_BATCH_ROWS * crate::rebel::NTYPE * layout.de);
        let bg_len = (MAX_BATCH_ROWS * crate::rebel::NTYPE
            * crate::units::CARD_FEATS.max(crate::rebel::PILE_COUNTS)
            + MAX_BATCH * crate::rebel::NTYPE * layout.de)
            .max(MAX_BATCH_CFG * crate::rebel::NSLOT * layout.hfeat())
            .max(MAX_BATCH_CFG * (layout.rank + 1));
        let mut pools = Vec::new();
        let mut pool = |stream: &Arc<CudaStream>, n: usize| -> Result<*mut f32, String> {
            let mut s: CudaSlice<f32> = stream.alloc_zeros(n.max(1)).map_err(|e| format!("{e:?}"))?;
            let p = ptr_mut(stream, &mut s);
            pools.push(s);
            Ok(p)
        };
        let mut ctx0 = Ctx::default();
        ctx0.h0 = pool(&compute, MAX_ROWS * layout.head_in)?;
        ctx0.xb = pool(&compute, MAX_ROWS * 2 * layout.dg)?;
        ctx0.h = pool(&compute, MAX_ROWS * h_stride)?;
        ctx0.h2 = pool(&compute, if hmlp.is_empty() { 1 } else { MAX_ROWS * h_stride })?;
        ctx0.u = pool(&compute, MAX_ROWS * layout.rank)?;
        ctx0.bx = pool(&compute, MAX_BATCH_ROWS * layout.xdim())?;
        ctx0.bh = pool(&compute, bh_len)?;
        ctx0.bh2 = pool(&compute, bh_len)?;
        ctx0.bg = pool(&compute, bg_len)?;

        let mut bmap: CudaSlice<i32> = compute
            .alloc_zeros(MAX_BATCH * BMAP_INTS)
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

        let (dl_tx, dl_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name(format!("gpu-download-{device}"))
            .spawn(move || downloader(download, dl_rx, done_tx))
            .map_err(|e| format!("{e:?}"))?;

        Ok(Service {
            descs: compute.alloc_zeros(CAP).map_err(|e| format!("{e:?}"))?,
            g_meta: compute.alloc_zeros(N_HDRS).map_err(|e| format!("{e:?}"))?,
            live_slots: compute.alloc_zeros(CAP).map_err(|e| format!("{e:?}"))?,
            live_count: 0,
            live_dirty: false,
            live: (0..CAP).map(|_| None).collect(),
            free: (0..CAP).rev().collect(),
            next_id: 1,
            waiting: std::collections::VecDeque::new(),
            tables,
            tbl_slab,
            arenas,
            arena_slab,
            rows: Slab::new(MAX_ROWS),
            _pools: pools,
            bmap,
            incoming: None,
            dl_tx,
            done_rx,
            ticks: 0,
            solved: 0,
            compute, build, blas, build_blas, f, weights, ctx, rx,
        })
    }

    fn run(&mut self) {
        loop {
            // Drain commands; block only when nothing is in flight.
            loop {
                match self.rx.try_recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
            }
            while let Ok(slot) = self.done_rx.try_recv() {
                self.release(slot);
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
            if self.incoming.is_none() {
                self.admit_batch();
            }
            if self.live_count > 0 {
                self.tick();
            } else if self.waiting.is_empty() {
                // Idle: block for the next command instead of spinning.
                match self.rx.recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(_) => return,
                }
            }
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Submit { job, tag, reply } => self.waiting.push_back((job, tag, reply)),
            Cmd::Trip2 { id, leaf, reply } => self.start_carry(id, leaf, reply),
            Cmd::SetWeights { dims, w, b, ln } => {
                if dims != self.weights.dims {
                    eprintln!(
                        "gpu: weight shape changed ({:?} -> {dims:?}); restart the service",
                        self.weights.dims
                    );
                    return;
                }
                self.incoming = Some((dims, w, b, ln));
            }
            Cmd::Shutdown => {}
        }
    }

    // ------------------------------------------------------------ launching

    fn fire(&self, which: fn(&Kernels) -> &CudaFunction, hdr: usize, grid: u32, stream: &Arc<CudaStream>) {
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
        let _ = unsafe { b.launch(cfg) };
    }

    /// Rewrite the tick's launch headers; runs only when membership changed.
    fn upload_headers(&mut self) {
        let base = ptr(&self.compute, &self.live_slots);
        let mk = |mode: i32, p: i32, level: i32| GroupDev {
            slots: base,
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
        for k in 0..self.weights.layout.hmlp.len() {
            // Layer k reads the previous buffer and writes h (mode 0) or h2
            // (mode 1); the first layer reads h, so it writes h2.
            h[Hdr::HmlpBase as usize + k] = mk((k % 2 == 0) as i32, 0, k as i32);
        }
        let _ = self.compute.memcpy_htod(&h, &mut self.g_meta);
    }

    // ------------------------------------------------------------ the tick

    fn tick(&mut self) {
        self.tick_upto(Step::All);
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
            self.upload_headers();
            self.live_dirty = false;
        }
        if self.live_count == 0 || upto <= Step::None {
            return;
        }
        let l = &self.weights.layout;
        let n = self.live_count as u32;
        let hi = self.rows.high;
        let c = self.weights.ctx;

        // 1. Value/carry passes re-seed and propagate their strategy first.
        self.fire(|k| &k.reach_prop, Hdr::ReachVC as usize, n, &self.compute.clone());
        // 2. Belief sums into the xb pool.
        self.fire(|k| &k.belief_sums, Hdr::Plain as usize, n, &self.compute.clone());
        // 3. The head: GEMM, entry norm, extra layers, readout GEMM.
        gemm(&self.blas, hi, l.head_in, 2 * l.dg, c.xb as *const f32, 2 * l.dg,
             c.wb, l.head_in, c.h, h_stride_of(l), 0.0);
        self.fire(|k| &k.head_entry, Hdr::Plain as usize, n, &self.compute.clone());
        let mut cur = c.h as *const f32;
        for (k, s) in l.hmlp.iter().enumerate() {
            let dst = if k % 2 == 0 { c.h2 } else { c.h };
            gemm(&self.blas, hi, s.o, s.i, cur, h_stride_of(l), self.weights_at(s.w),
                 s.o, dst, h_stride_of(l), 0.0);
            self.fire(|f| &f.hmlp_act, Hdr::HmlpBase as usize + k, n, &self.compute.clone());
            cur = dst as *const f32;
        }
        gemm(&self.blas, hi, l.rank, l.head_out, cur, h_stride_of(l),
             self.weights_at(l.wu.w), l.rank, c.u, l.rank, 0.0);
        if upto == Step::Head {
            return;
        }
        // 4. Iterate pass: readout + backprop for the traverser.
        self.fire(|k| &k.readout, Hdr::ReadIt as usize, n, &self.compute.clone());
        if upto == Step::Readout {
            return;
        }
        self.fire(|k| &k.backprop_solve, Hdr::ReadIt as usize, n, &self.compute.clone());
        if upto == Step::Backprop {
            return;
        }
        self.fire(|k| &k.regret_match, Hdr::Plain as usize, n, &self.compute.clone());
        if upto == Step::Regret {
            return;
        }
        // 5. Value passes: both players against the average strategy.
        for hdr in [Hdr::ReadV0, Hdr::ReadV1] {
            self.fire(|k| &k.readout, hdr as usize, n, &self.compute.clone());
            self.fire(|k| &k.backprop_solve, hdr as usize, n, &self.compute.clone());
            self.fire(|k| &k.collect_root, hdr as usize, n, &self.compute.clone());
        }
        // 6. Iterate reach follows regret matching.
        self.fire(|k| &k.reach_prop, Hdr::ReachIt as usize, n, &self.compute.clone());
        if upto == Step::Propagate {
            return;
        }
        self.fire(|k| &k.average, Hdr::Plain as usize, n, &self.compute.clone());
        if upto == Step::Average {
            return;
        }
        // 7. Carry harvest, then the device state machine.
        self.fire(|k| &k.leaf_beliefs, Hdr::Plain as usize, n, &self.compute.clone());
        let adv_grid = (self.live_count as u32).div_ceil(BLOCK);
        self.fire(|k| &k.advance_state, Hdr::Plain as usize, adv_grid.max(1), &self.compute.clone());
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
        let mut trip2 = Vec::new();
        let mut done = Vec::new();
        for s in 0..CAP {
            let Some(sv) = &mut self.live[s] else { continue };
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
                _ => {
                    if sv.trip2.is_some() && !sv.draining {
                        sv.step += 1;
                        if sv.step + 1 >= sv.nsnaps.max(1) {
                            trip2.push(s);
                        }
                    } else if sv.trip2.is_none() && sv.nsnaps == 0 && sv.trip1.is_none() {
                        // Evaluation solve: finished at trip 1.
                        done.push(s);
                    }
                }
            }
        }
        for s in trip1 {
            self.send_trip1(s);
        }
        for s in trip2 {
            self.send_trip2(s);
        }
        for s in done {
            self.release(s);
        }
    }

    // ------------------------------------------------------------ admission

    /// Admit a batch: everything queued, up to the batch caps and whatever
    /// fits in the slabs, in arrival order.
    fn admit_batch(&mut self) {
        let mut batch = Vec::new();
        let (mut rows, mut cfgs) = (0usize, 0usize);
        while let Some((job, _, _)) = self.waiting.front() {
            let t = &job.tables;
            if batch.len() == MAX_BATCH
                || rows + t.rows > MAX_BATCH_ROWS
                || cfgs + t.ncfg > MAX_BATCH_CFG
            {
                break;
            }
            if self.free.is_empty()
                || !self.rows.fits(t.rows)
                || !self.tbl_slab.fits(table_len(job))
                || !self.arena_slab.fits(arena_len_of(job, &self.weights.layout))
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
        self.build_batch(&batch);
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
        let t = &job.tables;
        let l = &self.weights.layout;
        let derived = Derived::new(&job);
        let nc = |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
        let nc_root = [nc(0, 0), nc(0, 1)];
        let max_nc = (0..t.nodes).map(|i| nc(i, 0).max(nc(i, 1))).max().unwrap_or(0);
        let nsnaps = if job.meta.snapshots { job.meta.snap_iters.len() } else { 0 };
        if job.meta.snap_iters.len() > 16 {
            return Err("gpu: more than 16 kept iterates".into());
        }
        let sizes = Sizes {
            reach_len: t.reach_len,
            vals_len: derived.vals_len(),
            ncells: t.ncells,
            nsnaps,
            ncfg: t.ncfg,
            nc_root: nc_root[0] + nc_root[1],
            max_nc,
            nroots: job.carried.len(),
            dg: l.dg,
            rk: l.rank,
            de: l.de,
        };
        let (aoff, arena_len) = arena_offsets(&sizes);
        let (blob, toff) = pack_tables(t, &derived);

        let slot = self.free.pop().ok_or("gpu: live set full")?;
        let row0 = self.rows.alloc(t.rows).ok_or("gpu: row pool full")?;
        let tbl_at = self.tbl_slab.alloc(blob.len()).ok_or("gpu: table slab full")?;
        let arena_at = self.arena_slab.alloc(arena_len).ok_or("gpu: arena slab full")?;

        // Upload tables and zero the arena on the build stream.
        {
            let mut dst = self.tables.slice_mut(tbl_at..tbl_at + blob.len());
            self.build
                .memcpy_htod(&blob, &mut dst)
                .map_err(|e| format!("{e:?}"))?;
            let mut az = self.arenas.slice_mut(arena_at..arena_at + arena_len);
            self.build.memset_zeros(&mut az).map_err(|e| format!("{e:?}"))?;
        }
        let (tbl_ptr, _) = self.tables.device_ptr(&self.build);
        let (arena_ptr, _) = self.arenas.device_ptr(&self.build);
        let mut desc = Desc {
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
                if p > 0.0 { 1.0 } else { 0.0 }
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
            cfg_off: t.cfg_off.clone(),
            stage: STAGE_ITERATE,
            t: 0,
            step: 0,
            nroots: job.carried.len(),
            nsnaps,
            nleaf_cfg: [0, 0],
            nc_root,
            ncells: t.ncells,
            trip1: Some((tag, reply)),
            trip2: None,
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
        let (mut rows, mut cfgs) = (0usize, 0usize);
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
        }
        let _ = stream.memcpy_htod(&map, &mut self.bmap.slice_mut(..map.len()));
        // Group headers for the build launches live in the tail of g_meta.
        // Header layout: [N_HDRS-2] = grid-stride (total = per-kernel), we
        // instead upload per-launch headers one by one; the build is per
        // batch, a dozen small copies are irrelevant next to the GEMMs.
        let base = ptr(&stream, &self.live_slots);
        let _ = base;
        let nb = slots.len();
        let slot_ids: Vec<i32> = slots.iter().map(|&s| s as i32).collect();
        // Batch slot list sits after the map in bmap.
        let at = map.len();
        let _ = stream.memcpy_htod(&slot_ids, &mut self.bmap.slice_mut(at..at + nb));
        let (bmap_ptr, _) = self.bmap.device_ptr(&stream);
        let slots_dev = (bmap_ptr as usize + at * 4) as *const i32;

        let hdr = |n: i32, mode: i32, p: i32, level: i32, total: i32| GroupDev {
            slots: slots_dev,
            n,
            mode,
            p_player: p,
            level,
            total,
        };
        // One reusable header slot at the end of g_meta.
        let hslot = N_HDRS - 1;
        let mut set = |g: GroupDev, me: &mut Self| {
            let _ = me
                .build
                .memcpy_htod(&[g], &mut me.g_meta.slice_mut(hslot..hslot + 1));
        };
        let grid = |items: usize| (items.max(1) as u32).div_ceil(BLOCK);
        let wgrid = |items: usize| ((items.max(1) * 32) as u32).div_ceil(BLOCK);

        // ---- cards: pack -> GEMM chain -> finish (adds wid) --------------
        set(hdr(nb as i32, 0, 0, 0, (nb * crate::rebel::NTYPE) as i32), self);
        self.fire_b(|k| &k.pack_cards, hslot, grid(nb * crate::rebel::NTYPE * crate::units::CARD_FEATS));
        let nrows_card = nb * crate::rebel::NTYPE;
        let mut src = c.bg as *const f32;
        let mut buf = 0; // 0 -> bh, 1 -> bh2
        for (k, s) in l.card.iter().enumerate() {
            let dst = if buf == 0 { c.bh } else { c.bh2 };
            gemm(&self.build_blas, nrows_card, s.o, s.i, src, s.i,
                 self.weights_at(s.w), s.o, dst, s.o, 0.0);
            if k + 1 < l.card.len() {
                set(hdr(nb as i32, 0, buf, k as i32, nrows_card as i32), self);
                self.fire_b(|f| &f.bias_act, hslot, grid(nrows_card * s.o));
                src = dst;
                buf ^= 1;
            } else if buf == 1 {
                // cards_finish reads bh; copy parity by re-running the GEMM
                // into bh would be silly — instead cards_finish reads the
                // buffer named by mode. (mode = which buffer holds the out.)
            }
        }
        let card_buf = (l.card.len() + 1) % 2; // which buffer the last GEMM wrote
        set(hdr(nb as i32, card_buf as i32, 0, (l.card.len() - 1) as i32,
                nrows_card as i32), self);
        self.fire_b(|k| &k.cards_finish, hslot, grid(nb * crate::rebel::NTYPE * l.de));
        // ---- pile: pe tail, count pack, count GEMM ------------------------
        set(hdr(nb as i32, 0, 0, 0, nb as i32), self);
        self.fire_b(|k| &k.pile_pe, hslot, grid(nb * crate::rebel::NTYPE * l.de));
        set(hdr(nb as i32, 0, 0, 0, rows as i32), self);
        self.fire_b(|k| &k.pack_piles, hslot, grid(rows * crate::rebel::NTYPE * crate::rebel::PILE_COUNTS));
        gemm(&self.build_blas, rows * crate::rebel::NTYPE, l.de, crate::rebel::PILE_COUNTS,
             c.bg as *const f32, crate::rebel::PILE_COUNTS, self.weights_at(l.pile.w), l.de,
             c.bh, l.de, 0.0);
        // ---- trunk: assemble -> pub chain -> pub_out -> scatter h0 --------
        set(hdr(nb as i32, 0, 0, 0, rows as i32), self);
        self.fire_b(|k| &k.assemble, hslot, rows.max(1) as u32);
        let mut src = c.bx as *const f32;
        let mut buf = 0;
        for (k, s) in l.pub_lin.iter().enumerate() {
            let dst = if buf == 0 { c.bh } else { c.bh2 };
            gemm(&self.build_blas, rows, s.o, s.i, src, s.i, self.weights_at(s.w), s.o,
                 dst, s.o, 0.0);
            set(hdr(nb as i32, 0, buf, k as i32, rows as i32), self);
            self.fire_b(|f| &f.trunk_norm, hslot, wgrid(rows));
            src = dst;
            buf ^= 1;
        }
        let dst = if buf == 0 { c.bh } else { c.bh2 };
        gemm(&self.build_blas, rows, l.head_in, l.pub_out.i, src, l.pub_out.i,
             self.weights_at(l.pub_out.w), l.head_in, dst, l.head_in, 0.0);
        set(hdr(nb as i32, 0, buf, 0, rows as i32), self);
        self.fire_b(|k| &k.scatter_h0, hslot, grid(rows * l.head_in));
        // ---- holding tower -------------------------------------------------
        set(hdr(nb as i32, 0, 0, 0, cfgs as i32), self);
        self.fire_b(|k| &k.holding_in, hslot, grid(cfgs * crate::rebel::NSLOT));
        let nsl = cfgs * crate::rebel::NSLOT;
        let mut src = c.bg as *const f32;
        let mut buf = 0;
        for (k, s) in l.slot.iter().enumerate() {
            let dst = if buf == 0 { c.bh } else { c.bh2 };
            gemm(&self.build_blas, nsl, s.o, s.i, src, s.i, self.weights_at(s.w), s.o,
                 dst, s.o, 0.0);
            set(hdr(nb as i32, 2, buf, k as i32, nsl as i32), self);
            self.fire_b(|f| &f.bias_act, hslot, grid(nsl * s.o));
            src = dst;
            buf ^= 1;
        }
        let dst = if buf == 0 { c.bh } else { c.bh2 };
        gemm(&self.build_blas, nsl, l.dg, l.slot_out.i, src, l.slot_out.i,
             self.weights_at(l.slot_out.w), l.dg, dst, l.dg, 0.0);
        // slot_sum reads `p_player ? bh2 : bh` and writes the other: z ends
        // in the opposite buffer.
        set(hdr(nb as i32, 0, buf, 0, cfgs as i32), self);
        self.fire_b(|k| &k.slot_sum, hslot, wgrid(cfgs));
        let mut zbuf = buf ^ 1; // where z lives now
        for (k, (a, bres)) in l.res.iter().enumerate() {
            let (zp, rp) = if zbuf == 0 { (c.bh, c.bh2) } else { (c.bh2, c.bh) };
            gemm(&self.build_blas, cfgs, l.dg, l.dg, zp as *const f32, l.dg,
                 self.weights_at(a.w), l.dg, rp, l.dg, 0.0);
            set(hdr(nb as i32, 3, zbuf ^ 1, k as i32, cfgs as i32), self);
            self.fire_b(|f| &f.bias_act, hslot, grid(cfgs * l.dg));
            gemm(&self.build_blas, cfgs, l.dg, l.dg, rp as *const f32, l.dg,
                 self.weights_at(bres.w), l.dg, zp, l.dg, 1.0);
            set(hdr(nb as i32, 4, zbuf, k as i32, cfgs as i32), self);
            self.fire_b(|f| &f.bias_act, hslot, grid(cfgs * l.dg));
            let _ = zbuf; // z stays in zp
        }
        let zp = if zbuf == 0 { c.bh } else { c.bh2 };
        gemm(&self.build_blas, cfgs, l.rank + 1, l.dg, zp as *const f32, l.dg,
             self.weights_at(l.wg.w), l.rank + 1, c.bg, l.rank + 1, 0.0);
        set(hdr(nb as i32, 0, zbuf, 0, cfgs as i32), self);
        self.fire_b(|k| &k.scatter_zg, hslot, grid(cfgs * (l.dg + l.rank + 1)));
        // ---- CFR init: uniform strategy, initial reach, seeded average ----
        set(hdr(nb as i32, 0, 0, 0, 0), self);
        self.fire_b(|k| &k.init_strategy, hslot, nb as u32);
        set(hdr(nb as i32, 1, 0, 0, 0), self);
        self.fire_b(|k| &k.reach_prop, hslot, nb as u32);
        set(hdr(nb as i32, 0, 0, 0, 0), self);
        self.fire_b(|k| &k.seed_avg, hslot, nb as u32);
        let _ = zbuf;
    }

    fn fire_b(&self, which: fn(&Kernels) -> &CudaFunction, hdr: usize, grid: u32) {
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
        let mut b = self.build.launch_builder(&f);
        b.arg(&self.descs);
        b.arg(&group);
        b.arg(&self.ctx);
        let _ = unsafe { b.launch(cfg) };
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
        let ev = self.compute.record_event(None).ok();
        let sv = self.live[s].as_mut().unwrap();
        let Some((tag, tx)) = sv.trip1.take() else { return };
        let (arena_ptr, _) = self.arenas.device_ptr(&self.compute);
        let at = |a: Arena| arena_ptr + ((sv.arena_at + sv.aoff[a as usize] as usize) * 4) as u64;
        let stride = sv.nc_root[0] + sv.nc_root[1];
        let req = Dl::Trip1 {
            event: ev.expect("event"),
            tag,
            reply: tx,
            id: sv.id,
            strategy: (at(Arena::avg), sv.ncells),
            root_vals: (at(Arena::root_vals), sv.nroots * stride),
            nc_root: sv.nc_root,
        };
        // The head rows are dead from here on: value passes are done and the
        // carry stage never reads them. Freeing now is what keeps the head
        // GEMM's row span tight.
        self.rows.free(sv.row0);
        sv.nrows = 0;
        self.solved += 1;
        let _ = self.dl_tx.send(req);
    }

    /// The walk left the tree at `leaf`: rebuild the solve's descriptor from
    /// the host mirror (now with the exit leaf and its support counts) and
    /// re-upload it. The upload is stream-ordered before the next tick, and
    /// every mutable field is derivable from the mirror, so replacing the
    /// device copy wholesale is exact.
    fn start_carry(&mut self, id: u64, leaf: u32, reply: mpsc::Sender<Result<Trip2, String>>) {
        let Some(s) = (0..CAP).find(|&s| self.live[s].as_ref().is_some_and(|v| v.id == id)) else {
            let _ = reply.send(Err(format!("gpu: unknown solve id {id}")));
            return;
        };
        let sv = self.live[s].as_mut().unwrap();
        let iters = sv.meta.iters;
        let l = leaf as usize;
        sv.nleaf_cfg = [
            (sv.cfg_off[2 * l + 1] - sv.cfg_off[2 * l]) as usize,
            (sv.cfg_off[2 * l + 2] - sv.cfg_off[2 * l + 1]) as usize,
        ];
        sv.step = 0;
        sv.trip2 = Some(reply);
        let d = &mut sv.desc;
        d.leaf = l as i32;
        d.nc_leaf = [sv.nleaf_cfg[0] as i32, sv.nleaf_cfg[1] as i32];
        d.stage = STAGE_CARRY;
        d.step = 0;
        d.t = iters as i32;
        d.traverser = (iters & 1) as i32;
        d.steps = [iters.div_ceil(2) as i32, (iters / 2) as i32];
        d.first_query = 0;
        d.snap_t = sv.nsnaps as i32;
        let desc = *d;
        let _ = self
            .compute
            .memcpy_htod(&[desc], &mut self.descs.slice_mut(s..s + 1));
    }

    /// Trip 2: the carried beliefs at the exit leaf, downloaded after the
    /// carry replays finish.
    fn send_trip2(&mut self, s: usize) {
        let ev = self.compute.record_event(None).ok();
        let sv = self.live[s].as_mut().unwrap();
        let Some(tx) = sv.trip2.take() else { return };
        sv.draining = true;
        let (arena_ptr, _) = self.arenas.device_ptr(&self.compute);
        let at = arena_ptr
            + ((sv.arena_at + sv.aoff[Arena::beliefs as usize] as usize) * 4) as u64;
        let stride = sv.nleaf_cfg[0] + sv.nleaf_cfg[1];
        let n = sv.nsnaps.saturating_sub(1);
        let req = Dl::Trip2 {
            event: ev.expect("event"),
            reply: tx,
            slot: s,
            beliefs: (at, n * stride),
            nc_leaf: sv.nleaf_cfg,
            nsnaps: sv.nsnaps,
        };
        let _ = self.dl_tx.send(req);
    }

    fn release(&mut self, s: usize) {
        if let Some(sv) = self.live[s].take() {
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

fn table_len(job: &Job) -> usize {
    // pack_tables pads for alignment; a small overestimate is fine.
    let d = Derived::new(job);
    super::layout::pack_tables(&job.tables, &d).0.len()
}

fn arena_len_of(job: &Job, l: &V3Layout) -> usize {
    let t = &job.tables;
    let d = Derived::new(job);
    let nc = |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
    let nsnaps = if job.meta.snapshots { job.meta.snap_iters.len() } else { 0 };
    let sizes = Sizes {
        reach_len: t.reach_len,
        vals_len: d.vals_len(),
        ncells: t.ncells,
        nsnaps,
        ncfg: t.ncfg,
        nc_root: nc(0, 0) + nc(0, 1),
        max_nc: (0..t.nodes).map(|i| nc(i, 0).max(nc(i, 1))).max().unwrap_or(0),
        nroots: job.carried.len(),
        dg: l.dg,
        rk: l.rank,
        de: l.de,
    };
    arena_offsets(&sizes).1
}

/// Compare the device's view of `Desc` with the host's, field by field.
fn check_abi(stream: &Arc<CudaStream>, probe: &CudaFunction) -> Result<(), String> {
    let want = super::layout::abi_expected();
    let mut got = stream
        .alloc_zeros::<i32>(want.len())
        .map_err(|e| format!("{e:?}"))?;
    let mut b = stream.launch_builder(probe);
    b.arg(&mut got);
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    unsafe { b.launch(cfg) }.map_err(|e| format!("abi probe: {e:?}"))?;
    let mut host = vec![0i32; want.len()];
    stream.memcpy_dtoh(&got, &mut host).map_err(|e| format!("{e:?}"))?;
    stream.synchronize().map_err(|e| format!("{e:?}"))?;
    for ((name, &w), &g) in super::layout::abi_names().iter().zip(&want).zip(&host) {
        if w != g as usize {
            return Err(format!("gpu: Desc.{name} is {w} on the host, {g} on the device"));
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
    stream.memcpy_htod(v, &mut buf).map_err(|e| format!("{e:?}"))?;
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
fn gemm(
    blas: &CudaBlas,
    m: usize, n: usize, k: usize,
    a: *const f32, lda: usize,
    b: *const f32, ldb: usize,
    c: *mut f32, ldc: usize,
    beta: f32,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let alpha = 1.0f32;
    // SAFETY: device pointers the service owns; shapes from the layout.
    let _ = unsafe {
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
        self.release(s);
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
