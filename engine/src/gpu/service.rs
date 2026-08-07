//! The GPU service thread: owns GPU-0, keeps the live set of solves resident,
//! and advances it with ticks (plan section 8, B3).
//!
//! A tick runs the phases once each over the solves that need them. Solves in
//! the iterate stage take one CFR iteration; solves in the value stage take
//! one fixed-policy pass over a carried root; solves in the carry stage
//! propagate one kept snapshot to the exit leaf. Per-solve state comes from
//! the descriptor and per-phase switches from the launch group, so solves at
//! different iterations share a tick and no alignment is needed.
//!
//! Two invariants make this file short. Device memory is described once, in
//! `layout.rs`, and reached only through the accessors it generates. And
//! every kernel has the same signature, so `Service::launch` is the only
//! launch site — there is no per-kernel argument plumbing to get wrong.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::CudaBlas;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc;

use crate::serialize::{Job, JobMeta};

use super::client::{Cmd, GpuClient, Trip1, Trip2};
use super::layout::{
    arena_offsets, cuda_preamble, pack_tables, Arena, Derived, Desc, Sizes, Tbl, N_ARENAS,
    STAGE_CARRY, STAGE_ITERATE, STAGE_VALUE,
};

/// Live-set capacity, in solve slots.
const CAP: usize = 256;

/// Row-pool capacity. A solve's network rows are its non-terminal leaves. The
/// pool holds the whole live set at once, so a solve's rows are contiguous and
/// stable for its lifetime and no tick ever packs them; it is what actually
/// bounds the live set, since real trees vary by an order of magnitude and
/// `CAP` slots of the largest would not fit in memory.
const MAX_ROWS: usize = 256 * 1024;

/// Threads per block for the flat phases.
const BLOCK: u32 = 256;

/// How many launches one staging round may hold.
///
/// A sweep is one launch per level and a tick runs several sweeps, so this
/// scales with tree depth: the deepest trees measured have thirteen levels,
/// which puts a busy tick near a hundred launches. Sized well past that,
/// because overflowing it is a crash rather than a slowdown.
const MAX_LAUNCHES: usize = 512;

/// Per-phase timing, when `WARCHEST_GPU_TIMING` is set. Off by default: it
/// puts an event around every launch and synchronises once per tick, which is
/// exactly the overlap the service otherwise relies on.
/// Spans one flush may hold. A tick uses about twenty-five, and every
/// admission that lands in the same loop adds a dozen more.
const MAX_SPANS: usize = 1024;

struct Timing {
    events: Vec<cudarc::driver::CudaEvent>,
    names: Vec<&'static str>,
    total: std::collections::BTreeMap<&'static str, f32>,
    ticks: u64,
    since: std::time::Instant,
}

impl Timing {
    fn new(stream: &Arc<CudaStream>) -> Option<Timing> {
        if std::env::var("WARCHEST_GPU_TIMING").is_err() {
            return None;
        }
        let events = (0..2 * MAX_SPANS)
            // The default flag is DISABLE_TIMING, which makes elapsed_ms fail.
            .map(|_| {
                stream
                    .context()
                    .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .expect("event")
            })
            .collect();
        Some(Timing {
            events,
            names: Vec::new(),
            total: std::collections::BTreeMap::new(),
            ticks: 0,
            since: std::time::Instant::now(),
        })
    }

    /// Fold the tick's elapsed times into the running totals and report every
    /// few hundred ticks, so a long run prints a stable profile.
    fn flush(&mut self) {
        for (i, name) in self.names.iter().enumerate() {
            let ms = self.events[2 * i]
                .elapsed_ms(&self.events[2 * i + 1])
                .expect("timing events need CU_EVENT_DEFAULT");
            *self.total.entry(name).or_insert(0.0) += ms;
        }
        self.names.clear();
        self.ticks += 1;
        if self.ticks % 25 != 0 {
            return;
        }
        let sum: f32 = self.total.values().sum();
        let wall = self.since.elapsed().as_secs_f32() * 1000.0;
        let mut rows: Vec<_> = self.total.iter().collect();
        rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        eprintln!(
            "[gpu] {} ticks in {wall:.0} ms ({:.1} ticks/s); {sum:.0} ms on the device ({:.0}%):",
            self.ticks,
            self.ticks as f32 / (wall / 1000.0),
            100.0 * sum / wall,
        );
        for (name, ms) in rows {
            eprintln!("[gpu]   {name:<14} {ms:8.1} ms  {:5.1}%", 100.0 * ms / sum);
        }
    }
}

// ------------------------------------------------------------------ context

/// The `Ctx` struct of kernels.cu: the weights and the row pools, everything
/// a tick shares across solves.
#[repr(C)]
#[derive(Clone, Copy)]
struct Ctx {
    w0: *const f32, b0: *const f32, ln0w: *const f32, ln0b: *const f32,
    w1: *const f32, b1: *const f32, ln1w: *const f32, ln1b: *const f32,
    wb: *const f32, wu: *const f32, bu: *const f32,
    wc: *const f32, bc: *const f32, wh1: *const f32, bh1: *const f32,
    wh2: *const f32, bh2: *const f32, wg: *const f32, bg: *const f32,
    wd0: *const f32, bd0: *const f32, wd1: *const f32, bd1: *const f32,
    wid: *const f32, wpile: *const f32, bpile: *const f32,
    wq: *const f32, bq: *const f32, wk: *const f32, bk: *const f32,
    wp: *const f32, bp: *const f32,
    h0: *mut f32, xb: *mut f32, h: *mut f32, u: *mut f32,
    bx: *mut f32, bh: *mut f32, bgather: *mut f32,
    hidden: i32, head: i32, dg: i32, rk: i32, de: i32, dc: i32,
    af: i32, xd: i32, hf: i32, cfeat: i32, pubfeat: i32,
}

unsafe impl cudarc::driver::DeviceRepr for Ctx {}
unsafe impl cudarc::driver::ValidAsZeroBits for Ctx {}
unsafe impl Send for Ctx {}

/// The `Group` struct of kernels.cu: which solves a launch covers, where each
/// one's threads start, and the switches every solve in it shares.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GroupDev {
    slots: *const i32,
    starts: *const i32,
    n: i32,
    total: i32,
    mode: i32,
    p_player: i32,
    nplayers: i32,
    strat_src: i32,
    level: i32,
}

unsafe impl cudarc::driver::DeviceRepr for GroupDev {}
unsafe impl cudarc::driver::ValidAsZeroBits for GroupDev {}
unsafe impl Send for GroupDev {}

impl Default for Ctx {
    fn default() -> Ctx {
        // SAFETY: every field is an integer or a device pointer.
        unsafe { std::mem::zeroed() }
    }
}

/// The per-phase switches a launch carries: regret mode, the current
/// strategy, one player. Phases that need something else name it.
#[derive(Clone, Copy, Default, PartialEq)]
struct Phase {
    mode: i32,
    p_player: i32,
    nplayers: i32,
    strat_src: i32,
    level: i32,
}

/// The default phase, spelled short because most launches take it as is.
const P: Phase = Phase { mode: 0, p_player: 0, nplayers: 1, strat_src: 0, level: 0 };

/// `p_player` sentinel: each solve uses its own traverser rather than one
/// player fixed for the whole group.
const TRAVERSER: i32 = -1;

/// One staged launch: which group header it uses, and its grid.
#[derive(Clone, Copy)]
struct Launch {
    meta: usize,
    grid: u32,
}

/// Threads per item. Most phases give an item one thread; the two that walk a
/// row's config list give it a whole warp.
const THREAD: usize = 1;
const WARP: usize = 32;

/// The launches a CFR iteration needs, in the order they run.
#[derive(Default, Clone)]
struct IterateSet {
    belief_one: Option<Launch>,
    belief_two: Option<Launch>,
    head_norm: Option<Launch>,
    head_bias: Option<Launch>,
    readout: Option<Launch>,
    backprop: Vec<Launch>,
    regret: Option<Launch>,
    clear: Option<Launch>,
    seed: Option<Launch>,
    propagate: Vec<Launch>,
    average: Option<Launch>,
    snapshot: Option<Launch>,
}

/// The launches one fixed-policy value pass needs, per player.
#[derive(Default, Clone)]
struct ValuePass {
    readout: Option<Launch>,
    backprop: Vec<Launch>,
    collect: Option<Launch>,
}

#[derive(Default, Clone)]
struct ValueSet {
    clear: Option<Launch>,
    seed: Option<Launch>,
    propagate: Vec<Launch>,
    belief: Option<Launch>,
    head_norm: Option<Launch>,
    head_bias: Option<Launch>,
    pass: [ValuePass; 2],
}

#[derive(Default, Clone)]
struct CarrySet {
    clear: Option<Launch>,
    seed: Option<Launch>,
    propagate: Vec<Launch>,
    beliefs: Option<Launch>,
}

/// How far into a CFR iteration to run. Only the phase oracle test stops
/// short; the tick always runs `All`.
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

// ------------------------------------------------------------------ weights

/// The network's flat arrays on the device, plus the derived pointer table.
struct Weights {
    dims: Vec<usize>,
    _w: CudaSlice<f32>,
    _b: CudaSlice<f32>,
    _ln: CudaSlice<f32>,
    ctx: Ctx,
}

/// Slice bounds of the flat arrays, following `Mlp::from_flat`'s layout
/// (train/value_net.py::flat writes the same order). The last entry of each
/// scan is that array's total length, which is how callers size one.
pub fn weight_offsets(
    dims: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), String> {
    if dims.len() != 10 || dims[9] != 0 {
        return Err(format!("gpu: unsupported dims {dims:?}"));
    }
    let (h, hd, dg, rk, de, dc) = (dims[1], dims[2], dims[4], dims[5], dims[7], dims[8]);
    let (af, hf, xd) = (dims[6] + de, crate::net::hfeat(de), xdim(de));
    let (cf, nu, pc) = (crate::units::CARD_FEATS, crate::units::N_UNITS, crate::rebel::PILE_COUNTS);
    let scan = |lens: &[usize]| {
        let mut at = 0;
        let mut v = Vec::with_capacity(lens.len() + 1);
        for &n in lens {
            v.push(at);
            at += n;
        }
        v.push(at);
        v
    };
    Ok((
        scan(&[cf * dc, dc * de, nu * de, (pc + de) * de, xd * h, h * hd, 2 * dg * hd,
               hf * dg, dg * dg, dg * dg, dg * (rk + 1), hd * rk, af * rk, dg * rk, hd * rk]),
        scan(&[dc, de, de, h, hd, dg, dg, dg, rk + 1, rk, rk, rk, rk]),
        scan(&[h, h, hd, hd]),
    ))
}

/// The trunk input width: the hex block, the two per-player pile summaries,
/// and the loose scalars.
fn xdim(de: usize) -> usize {
    crate::board::N_HEXES * (crate::rebel::HEX_FACTS + de) + 2 * de + crate::rebel::LOOSE
}

impl Weights {
    fn upload(
        stream: &Arc<CudaStream>,
        pools: &Pools,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Weights, String> {
        let (ow, ob, oln) = weight_offsets(&dims)?;
        let want = |o: &[usize]| *o.last().unwrap();
        if w.len() != want(&ow) || b.len() != want(&ob) || ln.len() != want(&oln) {
            return Err(format!(
                "gpu: weight sizes {}/{}/{} do not match dims {dims:?}",
                w.len(), b.len(), ln.len()
            ));
        }
        let wb = htod(stream, &w)?;
        let bb = htod(stream, &b)?;
        let lb = htod(stream, &ln)?;
        let (wp, bp, lp) = (ptr(stream, &wb), ptr(stream, &bb), ptr(stream, &lb));
        let w_at = |i: usize| unsafe { wp.add(ow[i]) };
        let b_at = |i: usize| unsafe { bp.add(ob[i]) };
        let l_at = |i: usize| unsafe { lp.add(oln[i]) };
        let (h, hd, dg, rk, de, dc) = (dims[1], dims[2], dims[4], dims[5], dims[7], dims[8]);
        let ctx = Ctx {
            w0: w_at(4), b0: b_at(3), ln0w: l_at(0), ln0b: l_at(1),
            w1: w_at(5), b1: b_at(4), ln1w: l_at(2), ln1b: l_at(3),
            wb: w_at(6), wu: w_at(11), bu: b_at(9),
            wc: w_at(7), bc: b_at(5), wh1: w_at(8), bh1: b_at(6),
            wh2: w_at(9), bh2: b_at(7), wg: w_at(10), bg: b_at(8),
            wd0: w_at(0), bd0: b_at(0), wd1: w_at(1), bd1: b_at(1),
            wid: w_at(2), wpile: w_at(3), bpile: b_at(2),
            wq: w_at(12), bq: b_at(10), wk: w_at(13), bk: b_at(11),
            wp: w_at(14), bp: b_at(12),
            h0: pools.h0_ptr, xb: pools.xb_ptr, h: pools.h_ptr, u: pools.u_ptr,
            bx: pools.bx_ptr, bh: pools.bh_ptr, bgather: pools.bg_ptr,
            hidden: h as i32, head: hd as i32, dg: dg as i32, rk: rk as i32,
            de: de as i32, dc: dc as i32, af: (dims[6] + de) as i32,
            xd: xdim(de) as i32, hf: (crate::rebel::PILE_COUNTS + de) as i32,
            cfeat: crate::rebel::CFEAT as i32, pubfeat: crate::rebel::PUBFEAT as i32,
        };
        Ok(Weights { dims, _w: wb, _b: bb, _ln: lb, ctx })
    }
}

// ------------------------------------------------------------------ pools

/// The row pools: one contiguous buffer per network activation, indexed by a
/// solve's stable row base. Building a solve writes its `h0` rows straight in,
/// so the head's GEMMs run over the live set with no packing step.
struct Pools {
    _h0: CudaSlice<f32>,
    _xb: CudaSlice<f32>,
    _h: CudaSlice<f32>,
    _u: CudaSlice<f32>,
    _bx: CudaSlice<f32>,
    _bh: CudaSlice<f32>,
    _bg: CudaSlice<f32>,
    h0_ptr: *mut f32,
    xb_ptr: *mut f32,
    h_ptr: *mut f32,
    u_ptr: *mut f32,
    bx_ptr: *mut f32,
    bh_ptr: *mut f32,
    bg_ptr: *mut f32,
    /// Allocated row ranges, sorted by start; first fit.
    used: Vec<(usize, usize)>,
    /// One past the highest row in use, which is how far the head GEMMs run.
    high: usize,
}

impl Pools {
    fn new(stream: &Arc<CudaStream>, dims: &[usize]) -> Result<Pools, String> {
        let (h, hd, dg, rk, de) = (dims[1], dims[2], dims[4], dims[5], dims[7]);
        // The build's widest use of each scratch: the trunk input, the widest
        // hidden matrix, and the widest gather.
        let bx = MAX_BUILD_ROWS * xdim(de);
        let bh = MAX_BUILD_ROWS.max(MAX_CFG * crate::rebel::NSLOT) * h.max(dg);
        let bg = (MAX_BUILD_ROWS * crate::rebel::NTYPE * de + crate::rebel::NTYPE * de)
            .max(MAX_CFG * crate::rebel::NSLOT * crate::net::hfeat(de));
        let mut p = Pools {
            _h0: zeros(stream, MAX_ROWS * hd)?,
            _xb: zeros(stream, MAX_ROWS * 2 * dg)?,
            _h: zeros(stream, MAX_ROWS * hd.max(h))?,
            _u: zeros(stream, MAX_ROWS * rk)?,
            _bx: zeros(stream, bx)?,
            _bh: zeros(stream, bh)?,
            _bg: zeros(stream, bg)?,
            h0_ptr: std::ptr::null_mut(), xb_ptr: std::ptr::null_mut(),
            h_ptr: std::ptr::null_mut(), u_ptr: std::ptr::null_mut(),
            bx_ptr: std::ptr::null_mut(), bh_ptr: std::ptr::null_mut(),
            bg_ptr: std::ptr::null_mut(),
            used: Vec::new(),
            high: 0,
        };
        p.h0_ptr = ptr_mut(stream, &mut p._h0);
        p.xb_ptr = ptr_mut(stream, &mut p._xb);
        p.h_ptr = ptr_mut(stream, &mut p._h);
        p.u_ptr = ptr_mut(stream, &mut p._u);
        p.bx_ptr = ptr_mut(stream, &mut p._bx);
        p.bh_ptr = ptr_mut(stream, &mut p._bh);
        p.bg_ptr = ptr_mut(stream, &mut p._bg);
        Ok(p)
    }

    /// Where a run of `n` rows would go, by first fit. Ranges are kept
    /// sorted, so the gap before each one is the only place a run can start.
    /// `None` means the pool is full, which is what bounds the live set.
    fn find_gap(&self, n: usize) -> Option<(usize, usize)> {
        let mut at = 0;
        for (i, &(start, len)) in self.used.iter().enumerate() {
            if start - at >= n {
                return Some((at, i));
            }
            at = start + len;
        }
        (at + n <= MAX_ROWS).then_some((at, self.used.len()))
    }

    fn alloc(&mut self, n: usize) -> Option<usize> {
        let (at, i) = self.find_gap(n)?;
        self.used.insert(i, (at, n));
        self.high = self.high.max(at + n);
        Some(at)
    }

    fn free(&mut self, at: usize) {
        self.used.retain(|&(s, _)| s != at);
        self.high = self.used.iter().map(|&(s, n)| s + n).max().unwrap_or(0);
    }
}

/// Build-scratch bounds. A tree past these is rejected rather than silently
/// overrunning; the CPU solver's node cap keeps real trees far below.
const MAX_BUILD_ROWS: usize = 16 * 1024;
const MAX_CFG: usize = 32 * 1024;

// ------------------------------------------------------------------ kernels

/// Every kernel, by the name NVRTC exports. They all have the same signature,
/// so this table is the only thing that names them.
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
    belief_sums, head_norm, head_bias, readout, backprop, regret_match, propagate,
    average, snapshot, collect_root, leaf_beliefs, init_strategy, reach_clear, reach_seed,
    cards_finish, pile_pe, assemble, trunk_norm, holding_in, slot_sum, embed_relu,
    embed_bias, readout_bias, cards_relu, abi_probe,
}

// ------------------------------------------------------------------ solves

/// One resident solve: its device blobs, its descriptor, and the host mirror
/// of the state the tick advances.
struct Solve {
    id: u64,
    meta: JobMeta,
    _tables: CudaSlice<u8>,
    arenas: CudaSlice<f32>,
    desc: Desc,
    aoff: [u32; N_ARENAS + 1],
    stage: i32,
    t: usize,
    step: usize,
    snap_t: usize,
    traverser: usize,
    steps: [usize; 2],
    first_query: bool,
    row0: usize,
    nrows: usize,
    reach_len: usize,
    nleaf: usize,
    nterm: usize,
    nodes: usize,
    ncells: usize,
    nsnaps: usize,
    nroots: usize,
    nc_root: [usize; 2],
    nc_leaf: [usize; 2],
    /// Config counts per node, for sizing the trip-2 exit leaf.
    cfg_off: Vec<u32>,
    /// Node index boundaries per BFS level, so a per-level launch knows how
    /// many nodes this solve contributes to it.
    level_start: Vec<u32>,
    nlevels: usize,
    trip1: Option<mpsc::Sender<Result<Trip1, String>>>,
    trip2: Option<mpsc::Sender<Result<Trip2, String>>>,
}

// ------------------------------------------------------------------ service

pub struct Service {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    f: Kernels,
    pools: Pools,
    weights: Weights,
    incoming: Option<Weights>,
    ctx: CudaSlice<Ctx>,
    live: Vec<Option<Solve>>,
    free: Vec<usize>,
    next_id: u64,
    rx: mpsc::Receiver<Cmd>,
    descs: CudaSlice<Desc>,
    /// Group staging. Every launch of a tick is built here and uploaded in
    /// one pair of copies; `g_data_ptr` is the device address the staged
    /// group headers point into, which is stable because `g_data` outlives
    /// the service.
    g_data: CudaSlice<i32>,
    g_meta: CudaSlice<GroupDev>,
    g_data_ptr: *const i32,
    h_data: Vec<i32>,
    h_meta: Vec<GroupDev>,
    h_descs: Vec<Desc>,
    timing: Option<Timing>,
    /// Jobs that arrived when the live set was full. The row pool, not the
    /// slot count, is usually what fills; a worker blocked here costs nothing
    /// and the alternative is failing a solve the walk cannot do without.
    waiting: std::collections::VecDeque<(Job, mpsc::Sender<Result<Trip1, String>>)>,
}

/// Spawn the service thread; returns the worker-side client.
pub fn spawn(
    dims: Vec<usize>,
    w: Vec<f32>,
    b: Vec<f32>,
    ln: Vec<f32>,
) -> Result<GpuClient, String> {
    let (tx, rx) = mpsc::channel();
    let client = GpuClient::new(tx);
    std::thread::Builder::new()
        .name("gpu-service".into())
        .spawn(move || match Service::new(rx, dims, w, b, ln) {
            Ok(mut svc) => svc.run(),
            Err(e) => eprintln!("gpu service failed to start: {e}"),
        })
        .map_err(|e| format!("{e:?}"))?;
    Ok(client)
}

impl Service {
    pub(crate) fn new(
        rx: mpsc::Receiver<Cmd>,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Service, String> {
        let dev = CudaContext::new(0).map_err(|e| format!("cuda device: {e:?}"))?;
        let stream = dev.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("{e:?}"))?;
        let src = format!("{}\n{}", cuda_preamble(), include_str!("kernels.cu"));
        let ptx = nvrtc::compile_ptx_with_opts(
            &src,
            nvrtc::CompileOptions { arch: Some("compute_75"), ..Default::default() },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = dev.load_module(ptx).map_err(|e| format!("module: {e:?}"))?;
        let f = Kernels::load(&module)?;
        let pools = Pools::new(&stream, &dims)?;
        let weights = Weights::upload(&stream, &pools, dims, w, b, ln)?;
        let ctx = htod(&stream, &[weights.ctx])?;
        check_abi(&stream, &f.abi_probe)?;
        let mut g_data_buf = zeros::<i32>(&stream, MAX_LAUNCHES * (2 * CAP + 1))?;
        let g_data_ptr = ptr_mut(&stream, &mut g_data_buf) as *const i32;
        Ok(Service {
            descs: zeros(&stream, CAP)?,
            g_data: g_data_buf,
            g_meta: zeros(&stream, MAX_LAUNCHES)?,
            g_data_ptr,
            h_data: Vec::with_capacity(MAX_LAUNCHES * (2 * CAP + 1)),
            h_meta: Vec::with_capacity(MAX_LAUNCHES),
            timing: Timing::new(&stream),
            h_descs: Vec::with_capacity(CAP),
            waiting: std::collections::VecDeque::new(),
            live: (0..CAP).map(|_| None).collect(),
            free: (0..CAP).rev().collect(),
            next_id: 1,
            stream, blas, f, pools, weights, incoming: None, ctx, rx,
        })
    }

    fn run(&mut self) {
        loop {
            match self.rx.recv_timeout(Duration::from_micros(50)) {
                Ok(cmd) => self.handle(cmd),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            // Drain whatever else arrived while we were busy, so a burst of
            // submissions joins the same tick.
            while let Ok(cmd) = self.rx.try_recv() {
                self.handle(cmd);
            }
            // Admission is outside the tick: a tick only runs when something
            // is live, and with an empty live set that would never admit the
            // first job.
            self.admit_waiting();
            if self.live.iter().any(|s| s.is_some()) {
                self.tick();
            }
        }
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Submit { job, reply } => self.waiting.push_back((job, reply)),
            Cmd::Trip2 { id, leaf, reply } => self.start_carry(id, leaf, reply),
            Cmd::SetWeights { dims, w, b, ln } => {
                match Weights::upload(&self.stream, &self.pools, dims, w, b, ln) {
                    Ok(weights) => self.incoming = Some(weights),
                    Err(e) => eprintln!("gpu: bad weights: {e}"),
                }
            }
            Cmd::Shutdown => {}
        }
    }

    // ------------------------------------------------------------ launching

    /// Stage a launch over `slots`, with `threads(solve)` threads for each.
    ///
    /// Nothing is sent to the device here. A launch needs three small arrays —
    /// the slots, their thread offsets, and the group header — and uploading
    /// those per launch costs several driver round trips each, which for a
    /// tick of fifteen launches is more time than the kernels. So the whole
    /// tick is staged into one host buffer and uploaded once by `commit`.
    fn stage(
        &mut self,
        slots: &[usize],
        phase: Phase,
        threads: impl Fn(&Solve) -> usize,
    ) -> Option<Launch> {
        self.stage_wide(slots, phase, THREAD, threads)
    }

    fn stage_wide(
        &mut self,
        slots: &[usize],
        phase: Phase,
        unit: usize,
        threads: impl Fn(&Solve) -> usize,
    ) -> Option<Launch> {
        if slots.is_empty() {
            return None;
        }
        let slots_at = self.h_data.len();
        self.h_data.extend(slots.iter().map(|&s| s as i32));
        let starts_at = self.h_data.len();
        let mut total = 0usize;
        for &s in slots {
            self.h_data.push(total as i32);
            total += threads(self.live[s].as_ref().expect("live slot"));
        }
        self.h_data.push(total as i32);
        if total == 0 {
            self.h_data.truncate(slots_at);
            return None;
        }
        Some(self.stage_meta(slots_at, starts_at, slots.len(), total, unit, phase))
    }

    /// Stage a whole sweep as one launch per level.
    ///
    /// The levels have to run in order, which used to mean one block per solve
    /// and `__syncthreads` between levels — capping the sweep at one block's
    /// threads per solve however much of the device sat idle. A launch per
    /// level orders them just as well and lets every node of every solve at
    /// that level run at once. Solves shallower than a given level contribute
    /// no threads to it.
    fn stage_sweep(&mut self, slots: &[usize], phase: Phase, backward: bool) -> Vec<Launch> {
        self.stage_sweep_wide(slots, phase, backward, THREAD)
    }

    fn stage_sweep_wide(
        &mut self,
        slots: &[usize],
        phase: Phase,
        backward: bool,
        unit: usize,
    ) -> Vec<Launch> {
        if slots.is_empty() {
            return Vec::new();
        }
        let depth = slots
            .iter()
            .map(|&s| self.live[s].as_ref().expect("live slot").nlevels)
            .max()
            .unwrap_or(0);
        (0..depth)
            .filter_map(|k| {
                let phase = Phase { level: k as i32, ..phase };
                self.stage_wide(slots, phase, unit, |sv| {
                    // Backwards, level 0 of the sweep is the solve's deepest.
                    let lev = if backward { sv.nlevels.checked_sub(k + 1) } else { Some(k) };
                    match lev {
                        Some(l) if l + 1 < sv.level_start.len() => {
                            (sv.level_start[l + 1] - sv.level_start[l]) as usize
                        }
                        _ => 0,
                    }
                })
            })
            .collect()
    }

    fn stage_meta(
        &mut self,
        slots_at: usize,
        starts_at: usize,
        n: usize,
        total: usize,
        unit: usize,
        phase: Phase,
    ) -> Launch {
        assert!(
            self.h_data.len() <= self.g_data.len() && self.h_meta.len() < MAX_LAUNCHES,
            "group staging overflow: {} ints, {} launches",
            self.h_data.len(),
            self.h_meta.len(),
        );
        let base = self.g_data_ptr;
        self.h_meta.push(GroupDev {
            // SAFETY: both offsets are inside `g_data`, asserted above.
            slots: unsafe { base.add(slots_at) },
            starts: unsafe { base.add(starts_at) },
            n: n as i32,
            total: total as i32,
            mode: phase.mode,
            p_player: phase.p_player,
            nplayers: phase.nplayers,
            strat_src: phase.strat_src,
            level: phase.level,
        });
        let grid = if total > 0 { (total * unit).div_ceil(BLOCK as usize) } else { n };
        Launch { meta: self.h_meta.len() - 1, grid: grid as u32 }
    }

    /// Send everything staged since the last commit: two copies, whatever the
    /// launch count.
    fn commit(&mut self) {
        if self.h_meta.is_empty() {
            return;
        }
        let _ = self.stream.memcpy_htod(&self.h_data, &mut self.g_data.slice_mut(..self.h_data.len()));
        let _ = self.stream.memcpy_htod(&self.h_meta, &mut self.g_meta.slice_mut(..self.h_meta.len()));
    }

    /// Open a timed span. Paired with `end`; a no-op unless timing is on.
    fn begin(&mut self, name: &'static str) {
        let stream = self.stream.clone();
        let Some(t) = &mut self.timing else { return };
        let i = t.names.len();
        if i >= MAX_SPANS {
            return;
        }
        t.names.push(name);
        let _ = t.events[2 * i].record(&stream);
    }

    fn end(&mut self) {
        let stream = self.stream.clone();
        let Some(t) = &mut self.timing else { return };
        if t.names.is_empty() || t.names.len() > MAX_SPANS {
            return;
        }
        let i = t.names.len() - 1;
        let _ = t.events[2 * i + 1].record(&stream);
    }

    fn reset_staging(&mut self) {
        self.h_data.clear();
        self.h_meta.clear();
    }

    /// Run a staged launch. Ordering against the cuBLAS calls between them is
    /// the stream's, as it is for any two kernels.
    fn fire(&mut self, which: fn(&Kernels) -> &CudaFunction, l: Option<Launch>) {
        self.fire_named(which, l, "")
    }

    /// Run a staged sweep, level by level.
    fn fire_sweep(&mut self, which: fn(&Kernels) -> &CudaFunction, ls: &[Launch], name: &'static str) {
        for &l in ls {
            self.fire_named(which, Some(l), name);
        }
    }

    fn fire_named(
        &mut self,
        which: fn(&Kernels) -> &CudaFunction,
        l: Option<Launch>,
        name: &'static str,
    ) {
        let Some(l) = l else { return };
        let cfg = LaunchConfig {
            grid_dim: (l.grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let f = which(&self.f).clone();
        self.begin(name);
        {
            let group = self.g_meta.slice(l.meta..l.meta + 1);
            let mut b = self.stream.launch_builder(&f);
            b.arg(&self.descs);
            b.arg(&group);
            b.arg(&self.ctx);
            let _ = unsafe { b.launch(cfg) };
        }
        self.end();
    }

    // ------------------------------------------------------------ the tick

    fn tick(&mut self) {
        if let Some(w) = self.incoming.take() {
            // Weights change between solves, never inside one: a fresh set
            // only takes effect once the live set has drained.
            if self.live.iter().all(|s| s.is_none()) {
                self.weights = w;
                let _ = self.stream.memcpy_htod(&[self.weights.ctx], &mut self.ctx);
            } else {
                self.incoming = Some(w);
            }
        }
        let by = |st: i32| -> Vec<usize> {
            (0..CAP).filter(|&s| self.live[s].as_ref().is_some_and(|v| v.stage == st)).collect()
        };
        let (iter, value, carry) = (by(STAGE_ITERATE), by(STAGE_VALUE), by(STAGE_CARRY));
        if iter.is_empty() && value.is_empty() && carry.is_empty() {
            return;
        }
        self.upload_descs();
        let snap = self.snapshot_due(&iter);
        let carry = self.carry_active(&carry);

        // Stage every launch of the tick, send them in one go, then run them.
        self.reset_staging();
        let it = self.stage_iterate(&iter, &snap);
        let va = self.stage_value(&value);
        let ca = self.stage_carry(&carry);
        self.commit();
        self.fire_iterate(&iter, &it, Step::All);
        self.fire_value(&value, &va);
        self.fire_carry(&ca);

        if self.timing.is_some() {
            let _ = self.stream.synchronize();
            self.timing.as_mut().expect("timing").flush();
        }
        self.advance(&snap);
    }

    /// Stage one CFR iteration for every solve in the iterate stage.
    fn stage_iterate(&mut self, slots: &[usize], snap: &[usize]) -> IterateSet {
        if slots.is_empty() {
            return IterateSet::default();
        }
        let rk = self.weights.dims[5];
        // The traverser's side is all a later iteration needs; the first one
        // has to build both, because neither side is cached yet.
        let (mut one, mut two) = (Vec::new(), Vec::new());
        for &s in slots {
            if self.live[s].as_ref().unwrap().first_query { two.push(s) } else { one.push(s) }
        }
        // The readout and backward sweep run for each solve's own traverser,
        // which `p_player: TRAVERSER` defers to the descriptor.
        let rb = Phase { mode: 0, p_player: TRAVERSER, nplayers: 2, strat_src: 0, ..P };
        IterateSet {
            belief_one: self.stage_wide(&one, Phase { nplayers: 1, ..P }, WARP, |sv| sv.nleaf),
            belief_two: self.stage_wide(&two, Phase { nplayers: 2, ..P }, WARP, |sv| sv.nleaf * 2),
            head_norm: self.stage_wide(slots, P, WARP, |sv| sv.nleaf),
            head_bias: self.stage(slots, P, |sv| sv.nleaf * rk),
            readout: self.stage_wide(slots, rb, WARP, |sv| sv.nleaf + sv.nterm),
            backprop: self.stage_sweep_wide(slots, rb, true, WARP),
            regret: self.stage_wide(slots, P, WARP, |sv| sv.nodes),
            clear: self.stage(slots, P, |sv| sv.reach_len),
            seed: self.stage(slots, P, |sv| sv.nc_root[0] + sv.nc_root[1]),
            propagate: self.stage_sweep_wide(slots, P, false, WARP),
            average: self.stage_wide(slots, P, WARP, |sv| sv.nodes),
            snapshot: self.stage(snap, P, |sv| sv.ncells),
        }
    }

    /// Run a staged iteration. `upto` stops early; the phase oracle test uses
    /// it to compare one phase at a time against the CPU solver, and so
    /// exercises this exact sequence.
    fn fire_iterate(&mut self, slots: &[usize], l: &IterateSet, upto: Step) {
        if slots.is_empty() || upto <= Step::None {
            return;
        }
        self.fire_named(|k| &k.belief_sums, l.belief_one, "belief");
        self.fire_named(|k| &k.belief_sums, l.belief_two, "belief");
        self.head(slots, l.head_norm, l.head_bias);
        if upto == Step::Head {
            return;
        }
        self.fire_named(|k| &k.readout, l.readout, "readout");
        if upto == Step::Readout {
            return;
        }
        self.fire_sweep(|k| &k.backprop, &l.backprop, "backprop");
        if upto == Step::Backprop {
            return;
        }
        self.fire_named(|k| &k.regret_match, l.regret, "regret");
        if upto == Step::Regret {
            return;
        }
        self.fire_named(|k| &k.reach_clear, l.clear, "reach");
        self.fire_named(|k| &k.reach_seed, l.seed, "reach");
        self.fire_sweep(|k| &k.propagate, &l.propagate, "propagate");
        if upto == Step::Propagate {
            return;
        }
        self.fire_named(|k| &k.average, l.average, "average");
        if upto == Step::Average {
            return;
        }
        self.fire_named(|k| &k.snapshot, l.snapshot, "snapshot");
    }

    /// Stage one fixed-policy pass over the carried root a value-stage solve
    /// is on: re-seed the reach from it, then read out and sweep for each
    /// player, harvesting the root values into the solve's own arena.
    fn stage_value(&mut self, slots: &[usize]) -> ValueSet {
        if slots.is_empty() {
            return ValueSet::default();
        }
        let rk = self.weights.dims[5];
        let mut v = ValueSet {
            clear: self.stage(slots, Phase { strat_src: 1, ..P }, |sv| sv.reach_len),
            seed: self.stage(slots, Phase { strat_src: 1, ..P }, |sv| sv.nc_root[0] + sv.nc_root[1]),
            propagate: self.stage_sweep_wide(slots, Phase { strat_src: 1, ..P }, false, WARP),
            belief: self.stage_wide(slots, Phase { nplayers: 2, ..P }, WARP, |sv| sv.nleaf * 2),
            head_norm: self.stage_wide(slots, P, WARP, |sv| sv.nleaf),
            head_bias: self.stage(slots, P, |sv| sv.nleaf * rk),
            pass: Default::default(),
        };
        for p in 0..2 {
            let phase = Phase { mode: 1, p_player: p as i32, nplayers: 2, strat_src: 1, ..P };
            v.pass[p] = ValuePass {
                readout: self.stage_wide(slots, phase, WARP, |sv| sv.nleaf + sv.nterm),
                backprop: self.stage_sweep_wide(slots, phase, true, WARP),
                collect: self.stage(slots, phase, |sv| sv.nc_root[0] + sv.nc_root[1]),
            };
        }
        v
    }

    fn fire_value(&mut self, slots: &[usize], l: &ValueSet) {
        if slots.is_empty() {
            return;
        }
        self.fire_named(|k| &k.reach_clear, l.clear, "reach");
        self.fire_named(|k| &k.reach_seed, l.seed, "reach");
        self.fire_sweep(|k| &k.propagate, &l.propagate, "propagate");
        self.fire_named(|k| &k.belief_sums, l.belief, "belief");
        self.head(slots, l.head_norm, l.head_bias);
        for p in 0..2 {
            self.fire_named(|k| &k.readout, l.pass[p].readout, "readout");
            self.fire_sweep(|k| &k.backprop, &l.pass[p].backprop, "backprop");
            self.fire_named(|k| &k.collect_root, l.pass[p].collect, "collect");
        }
    }

    /// Stage one kept snapshot propagated to the trip-2 exit leaf, whose
    /// normalised reach is that snapshot's carried belief.
    fn stage_carry(&mut self, slots: &[usize]) -> CarrySet {
        CarrySet {
            clear: self.stage(slots, Phase { strat_src: 2, ..P }, |sv| sv.reach_len),
            seed: self.stage(slots, Phase { strat_src: 2, ..P }, |sv| sv.nc_root[0] + sv.nc_root[1]),
            propagate: self.stage_sweep_wide(slots, Phase { strat_src: 2, ..P }, false, WARP),
            beliefs: self.stage(slots, P, |_| 2),
        }
    }

    fn fire_carry(&mut self, l: &CarrySet) {
        self.fire_named(|k| &k.reach_clear, l.clear, "reach");
        self.fire_named(|k| &k.reach_seed, l.seed, "reach");
        self.fire_sweep(|k| &k.propagate, &l.propagate, "propagate");
        self.fire_named(|k| &k.leaf_beliefs, l.beliefs, "beliefs");
    }

    /// Carry-stage solves the walk has already given an exit leaf. The rest
    /// sit resident and idle until trip 2 asks.
    fn carry_active(&self, slots: &[usize]) -> Vec<usize> {
        slots
            .iter()
            .copied()
            .filter(|&s| self.live[s].as_ref().unwrap().trip2.is_some())
            .collect()
    }

    /// The iterate-stage solves whose next iteration is a kept one.
    fn snapshot_due(&self, slots: &[usize]) -> Vec<usize> {
        slots
            .iter()
            .copied()
            .filter(|&s| {
                let sv = self.live[s].as_ref().unwrap();
                sv.meta.snapshots && sv.meta.snap_iters.contains(&(sv.t + 1))
            })
            .collect()
    }

    /// The head: two cuBLAS GEMMs with the LayerNorm between them, over the
    /// span of the row pool this group occupies. The rows were laid out at
    /// admission and never move, so this is one call per GEMM however ragged
    /// the live set is — the packing step a per-solve layout would need does
    /// not exist. A group's span may include rows belonging to solves outside
    /// it; those rows are computed and ignored, which costs a little
    /// arithmetic and saves the packing.
    fn head(&mut self, slots: &[usize], norm: Option<Launch>, bias: Option<Launch>) {
        let Some((lo, hi)) = self.row_span(slots) else { return };
        let (hd, dg, rk) = (self.weights.dims[2], self.weights.dims[4], self.weights.dims[5]);
        let c = self.weights.ctx;
        let rows = hi - lo;
        let at = |p: *mut f32, w: usize| unsafe { p.add(lo * w) };
        // h = xb . Wb, then h = relu(LN1(h0 + h)), then u = h . Wu + bu.
        self.begin("gemm");
        gemm(&self.blas, rows, hd, 2 * dg,
             at(c.xb, 2 * dg), 2 * dg, c.wb, hd, at(c.h, hd), hd, 0.0);
        self.end();
        self.fire_named(|k| &k.head_norm, norm, "head_norm");
        self.begin("gemm");
        gemm(&self.blas, rows, rk, hd, at(c.h, hd), hd, c.wu, rk, at(c.u, rk), rk, 0.0);
        self.end();
        self.fire_named(|k| &k.head_bias, bias, "head_bias");
    }

    /// The rows a group covers, as one range of the pool.
    fn row_span(&self, slots: &[usize]) -> Option<(usize, usize)> {
        let mut span: Option<(usize, usize)> = None;
        for &s in slots {
            let sv = self.live[s].as_ref()?;
            let (lo, hi) = (sv.row0, sv.row0 + sv.nrows);
            span = Some(match span {
                Some((a, b)) => (a.min(lo), b.max(hi)),
                None => (lo, hi),
            });
        }
        span.filter(|&(lo, hi)| hi > lo)
    }

    /// Advance the host mirrors one step and post whatever became ready. The
    /// device state advanced in the launches above; this is the bookkeeping
    /// that decides what happens next tick.
    fn advance(&mut self, snapped: &[usize]) {
        let mut trip1 = Vec::new();
        let mut trip2 = Vec::new();
        let mut done = Vec::new();
        for s in 0..CAP {
            let Some(sv) = &mut self.live[s] else { continue };
            match sv.stage {
                STAGE_ITERATE => {
                    sv.first_query = false;
                    sv.steps[sv.traverser] += 1;
                    sv.t += 1;
                    if snapped.contains(&s) {
                        sv.snap_t += 1;
                    }
                    sv.traverser = sv.t % 2;
                    if sv.t == sv.meta.iters {
                        sv.step = 0;
                        if sv.nroots > 0 {
                            sv.stage = STAGE_VALUE;
                        } else {
                            trip1.push(s);
                            sv.stage = STAGE_CARRY;
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
                _ => match &sv.trip2 {
                    // An evaluation solve keeps no snapshots and so has no
                    // second trip: it is finished the moment trip 1 is out.
                    None if sv.nsnaps == 0 => done.push(s),
                    // Otherwise the solve stays resident until the walk says
                    // which leaf it left the tree at.
                    None => {}
                    Some(_) => {
                        sv.step += 1;
                        if sv.step + 1 >= sv.nsnaps {
                            trip2.push(s);
                            done.push(s);
                        }
                    }
                },
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

    fn upload_descs(&mut self) {
        let v = &mut self.h_descs;
        v.clear();
        v.resize(CAP, Desc::default());
        for s in 0..CAP {
            let Some(sv) = &self.live[s] else { continue };
            let mut d = sv.desc;
            d.t = sv.t as i32;
            d.stage = sv.stage;
            d.step = sv.step as i32;
            d.traverser = sv.traverser as i32;
            d.snap_t = sv.snap_t as i32;
            d.steps = [sv.steps[0] as i32, sv.steps[1] as i32];
            d.first_query = sv.first_query as i32;
            d.nc_leaf = [sv.nc_leaf[0] as i32, sv.nc_leaf[1] as i32];
            v[s] = d;
        }
        let _ = self.stream.memcpy_htod(&self.h_descs, &mut self.descs);
    }

    // ------------------------------------------------------------ admission

    /// Admit as many queued jobs as the live set has room for, stopping at
    /// the first that does not fit so jobs keep their arrival order.
    fn admit_waiting(&mut self) {
        while let Some((job, _)) = self.waiting.front() {
            if self.free.is_empty() || self.pools.find_gap(job.tables.rows).is_none() {
                return;
            }
            let (job, reply) = self.waiting.pop_front().expect("front exists");
            if let Err(e) = self.admit(job, reply.clone()) {
                let _ = reply.send(Err(e));
            }
        }
    }

    /// Admit one solve: upload its tables, cut its arenas, run the build, and
    /// put it in the iterate stage.
    fn admit(
        &mut self,
        job: Job,
        reply: mpsc::Sender<Result<Trip1, String>>,
    ) -> Result<(), String> {
        let t = &job.tables;
        let (dg, rk, de) = (self.weights.dims[4], self.weights.dims[5], self.weights.dims[7]);
        let derived = Derived::new(&job);
        let nc = |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
        let nc_root = [nc(0, 0), nc(0, 1)];
        let max_nc = (0..t.nodes).map(|i| nc(i, 0).max(nc(i, 1))).max().unwrap_or(0);
        if job.meta.warm > 0.0 {
            // Seeding CFR from the policy head is plan A4, which is gated on a
            // NashConv measurement that has not been made. Refusing is the
            // honest answer; silently solving cold would look like it worked.
            return Err("gpu: warm start is not implemented (plan A4)".into());
        }
        if t.rows > MAX_BUILD_ROWS || t.ncfg > MAX_CFG {
            return Err(format!("gpu: tree too large (rows {} cfg {})", t.rows, t.ncfg));
        }
        let nsnaps = if job.meta.snapshots { job.meta.snap_iters.len() } else { 0 };
        let sizes = Sizes {
            reach_len: t.reach.len(),
            vals_len: derived.vals_len(),
            ncells: t.ncells,
            nsnaps,
            ncfg: t.ncfg,
            nc_root: nc_root[0] + nc_root[1],
            max_nc,
            nroots: job.carried.len(),
            dg, rk, de,
        };
        let (aoff, arena_len) = arena_offsets(&sizes);
        let (blob, toff) = pack_tables(t, &derived);

        let slot = self.free.pop().ok_or("gpu: live set full")?;
        let row0 = match self.pools.alloc(t.rows) {
            Some(r) => r,
            None => {
                self.free.push(slot);
                return Err("gpu: row pool full".into());
            }
        };
        let tables = htod(&self.stream, &blob)?;
        let mut arenas = zeros(&self.stream, arena_len)?;
        // The initial reach is the uniform-strategy reach `Solver::new`
        // propagates before seeding the average.
        {
            let r0 = aoff[Arena::reach as usize] as usize;
            let mut dst = arenas.slice_mut(r0..r0 + t.reach.len());
            let _ = self.stream.memcpy_htod(&t.reach, &mut dst);
        }
        let desc = Desc {
            tbl: ptr(&self.stream, &tables),
            arena: ptr_mut(&self.stream, &mut arenas),
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
            row0: row0 as i32,
            nc_root: [nc_root[0] as i32, nc_root[1] as i32],
            snapshots: job.meta.snapshots as i32,
            alpha: job.meta.cfr.alpha,
            beta: job.meta.cfr.beta,
            gamma: job.meta.cfr.gamma,
            predict: job.meta.cfr.predict,
            warm: job.meta.warm,
            first_query: 1,
            ..Desc::default()
        };
        self.live[slot] = Some(Solve {
            id: self.next_id,
            meta: job.meta,
            _tables: tables,
            arenas,
            desc,
            aoff,
            stage: STAGE_ITERATE,
            t: 0,
            step: 0,
            // Snapshot 0 is the uniform average, taken below; the counter is
            // the slot the *next* kept iterate goes to, so it is bumped after
            // that first copy rather than before it.
            snap_t: 0,
            traverser: 0,
            steps: [0, 0],
            first_query: true,
            row0,
            nrows: t.rows,
            reach_len: t.reach.len(),
            nleaf: t.nleaf,
            nterm: t.nterm,
            nodes: t.nodes,
            ncells: t.ncells,
            nsnaps,
            nroots: job.carried.len(),
            nc_root,
            nc_leaf: [0, 0],
            cfg_off: t.cfg_off.clone(),
            level_start: t.level_start.clone(),
            nlevels: t.nlevels,
            trip1: Some(reply),
            trip2: None,
        });
        self.next_id += 1;
        self.upload_descs();
        self.build(slot)?;
        Ok(())
    }

    /// The build: the card table, the trunk (into the solve's h0 rows), the
    /// holding tower, and the action tower. Each GEMM is one cuBLAS call over
    /// the whole solve; the kernels between them add biases and gather.
    fn build(&mut self, slot: usize) -> Result<(), String> {
        let one = [slot];
        let c = self.weights.ctx;
        let (h, hd, dg, rk, de, dc) = (
            self.weights.dims[1], self.weights.dims[2], self.weights.dims[4],
            self.weights.dims[5], self.weights.dims[7], self.weights.dims[8],
        );
        let (rows, ncfg, row0, arena) = {
            let sv = self.live[slot].as_ref().unwrap();
            (sv.nrows, sv.desc.ncfg as usize, sv.row0, sv.desc.arena)
        };
        let cf = crate::units::CARD_FEATS;
        let ntype = crate::rebel::NTYPE;
        let nslot = crate::rebel::NSLOT;
        let hf = crate::net::hfeat(de);
        let at = |a: Arena| unsafe { arena.add(self.live[slot].as_ref().unwrap().aoff[a as usize] as usize) };
        let (a_e, a_z, a_g) = (at(Arena::e), at(Arena::z), at(Arena::g));
        let xpub = self.table_ptr(slot, Tbl::leaf_xpub);

        self.reset_staging();
        let l_cards_relu = self.stage(&one, P, |_| ntype * dc);
        let l_cards_finish = self.stage(&one, P, |_| ntype * de);
        let l_pile_pe = self.stage(&one, P, |_| ntype * de);
        let l_assemble = self.stage(&one, P, |sv| sv.nrows);
        let l_trunk_norm = self.stage_wide(&one, P, WARP, |sv| sv.nrows);
        let l_holding_in = self.stage(&one, P, |_| ncfg * nslot);
        let l_slot_sum = self.stage(&one, P, |_| ncfg);
        let l_embed_relu = self.stage(&one, P, |_| ncfg * dg);
        let l_embed_bias = self.stage(&one, P, |_| ncfg * dg);
        let l_readout_bias = self.stage(&one, P, |_| ncfg * (rk + 1));
        let l_init = self.stage(&one, P, |sv| sv.nodes);
        let l_snapshot = self.stage(&one, P, |sv| sv.ncells);
        self.commit();

        // The card table: e = relu(facts . Wd0 + bd0) . Wd1 + bd1 + wid[id].
        // The facts block is the same at every row, so it is read from row 0.
        let facts = unsafe { xpub.add(crate::rebel::OFF_CARDS) };
        gemm(&self.blas, ntype, dc, cf, facts, cf, c.wd0, dc, c.bh, dc, 0.0);
        self.fire_named(|k| &k.cards_relu, l_cards_relu, "build");
        gemm(&self.blas, ntype, de, dc, c.bh as *const f32, dc, c.wd1, de, a_e, de, 0.0);
        self.fire_named(|k| &k.cards_finish, l_cards_finish, "build");

        // The trunk: the pile summary's two halves, the assembled input, then
        // h_pub = relu(LN0(x . W0 + b0)) and h0 = h_pub . W1, written straight
        // into the solve's rows of the h0 pool.
        self.fire_named(|k| &k.pile_pe, l_pile_pe, "build");
        gemm_batched(&self.blas, ntype, de, crate::rebel::PILE_COUNTS,
             unsafe { xpub.add(crate::rebel::OFF_PILES) }, crate::rebel::PILE_COUNTS,
             crate::rebel::PUBFEAT, c.wpile, de, c.bgather, de, ntype * de, rows);
        self.fire_named(|k| &k.assemble, l_assemble, "build");
        gemm(&self.blas, rows, h, xdim(de), c.bx as *const f32, xdim(de), c.w0, h, c.bh, h, 0.0);
        self.fire_named(|k| &k.trunk_norm, l_trunk_norm, "build");
        gemm(&self.blas, rows, hd, h, c.bh as *const f32, h, c.w1, hd,
             unsafe { c.h0.add(row0 * hd) }, hd, 0.0);

        // The holding tower: z, then the residual, then g.
        self.fire_named(|k| &k.holding_in, l_holding_in, "build");
        gemm(&self.blas, ncfg * nslot, dg, hf, c.bgather as *const f32, hf, c.wc, dg, c.bh, dg, 0.0);
        self.fire_named(|k| &k.slot_sum, l_slot_sum, "build");
        gemm(&self.blas, ncfg, dg, dg, a_z as *const f32, dg, c.wh1, dg, c.bh, dg, 0.0);
        self.fire_named(|k| &k.embed_relu, l_embed_relu, "build");
        gemm(&self.blas, ncfg, dg, dg, c.bh as *const f32, dg, c.wh2, dg, a_z, dg, 1.0);
        self.fire_named(|k| &k.embed_bias, l_embed_bias, "build");
        gemm(&self.blas, ncfg, rk + 1, dg, a_z as *const f32, dg, c.wg, rk + 1, a_g, rk + 1, 0.0);
        self.fire_named(|k| &k.readout_bias, l_readout_bias, "build");

        // Uniform strategy and the reach-weighted seed, then snapshot 0.
        self.fire_named(|k| &k.init_strategy, l_init, "init");
        if self.live[slot].as_ref().unwrap().nsnaps > 0 {
            self.fire_named(|k| &k.snapshot, l_snapshot, "snapshot");
            self.live[slot].as_mut().unwrap().snap_t = 1;
        }
        Ok(())
    }

    /// The device address of one of a solve's uploaded tables.
    fn table_ptr(&self, slot: usize, table: Tbl) -> *const f32 {
        let sv = self.live[slot].as_ref().unwrap();
        unsafe { sv.desc.tbl.add(sv.desc.toff[table as usize] as usize) as *const f32 }
    }

    // ------------------------------------------------------------ the trips

    /// Trip 1: the reference strategy and the carried roots' values. Two
    /// downloads, both of whole arenas; nothing was copied out per tick.
    fn send_trip1(&mut self, s: usize) {
        let sv = self.live[s].as_mut().unwrap();
        let Some(tx) = sv.trip1.take() else { return };
        let stride = sv.nc_root[0] + sv.nc_root[1];
        let strategy = d2h(&self.stream, &sv.arenas, sv.aoff[Arena::avg as usize] as usize, sv.ncells);
        let flat = d2h(&self.stream, &sv.arenas, sv.aoff[Arena::root_vals as usize] as usize, sv.nroots * stride);
        let root_values = flat
            .chunks_exact(stride.max(1))
            .map(|c| [c[..sv.nc_root[0]].to_vec(), c[sv.nc_root[0]..].to_vec()])
            .collect();
        let _ = tx.send(Ok(Trip1 { id: sv.id, strategy, root_values }));
    }

    /// The walk left the tree at `leaf`: record it and let the carry stage
    /// replay each kept snapshot to that leaf.
    fn start_carry(&mut self, id: u64, leaf: u32, reply: mpsc::Sender<Result<Trip2, String>>) {
        let Some(sv) = self.live.iter_mut().flatten().find(|v| v.id == id) else {
            let _ = reply.send(Err(format!("gpu: unknown solve id {id}")));
            return;
        };
        let l = leaf as usize;
        sv.nc_leaf = [
            (sv.cfg_off[2 * l + 1] - sv.cfg_off[2 * l]) as usize,
            (sv.cfg_off[2 * l + 2] - sv.cfg_off[2 * l + 1]) as usize,
        ];
        sv.desc.leaf = l as i32;
        sv.step = 0;
        sv.trip2 = Some(reply);
    }

    /// Trip 2: the carried beliefs at the exit leaf, one per kept snapshot.
    fn send_trip2(&mut self, s: usize) {
        let sv = self.live[s].as_mut().unwrap();
        let Some(tx) = sv.trip2.take() else { return };
        let stride = sv.nc_leaf[0] + sv.nc_leaf[1];
        let n = sv.nsnaps.saturating_sub(1);
        let flat = d2h(&self.stream, &sv.arenas, sv.aoff[Arena::beliefs as usize] as usize, n * stride);
        let out = flat
            .chunks_exact(stride.max(1))
            .map(|c| [c[..sv.nc_leaf[0]].to_vec(), c[sv.nc_leaf[0]..].to_vec()])
            .collect();
        let _ = tx.send(Ok(out));
    }

    /// Test hook: admit one solve and run a CFR iteration up to `upto`, then
    /// read back the arenas the CPU solver can be compared against. It drives
    /// `iterate`, so what it checks is what the tick runs.
    #[cfg(test)]
    pub(crate) fn probe(&mut self, job: Job, upto: Step) -> Result<super::tests::ProbeOut, String> {
        let (tx, _rx) = mpsc::channel();
        self.admit(job, tx)?;
        let s = self.live.iter().position(|x| x.is_some()).expect("admitted");
        self.upload_descs();
        self.reset_staging();
        let l = self.stage_iterate(&[s], &[]);
        self.commit();
        self.fire_iterate(&[s], &l, upto);
        self.stream.synchronize().map_err(|e| format!("{e:?}"))?;
        self.stream.context().check_err().map_err(|e| format!("{upto:?}: {e:?}"))?;
        let (dg, rk) = (self.weights.dims[4], self.weights.dims[5]);
        let sv = self.live[s].as_ref().unwrap();
        let span = |k: Arena| {
            (sv.aoff[k as usize + 1] - sv.aoff[k as usize]) as usize
        };
        let a = |k: Arena, n: usize| d2h(&self.stream, &sv.arenas, sv.aoff[k as usize] as usize, n);
        let out = super::tests::ProbeOut {
            e: a(Arena::e, span(Arena::e)),
            z: a(Arena::z, span(Arena::z)),
            g: a(Arena::g, span(Arena::g)),
            h0: d2h(&self.stream, &self.pools._h0, sv.row0 * self.weights.dims[2],
                    sv.nrows * self.weights.dims[2]),
            reach: a(Arena::reach, span(Arena::reach)),
            vals: a(Arena::vals, span(Arena::vals)),
            regret: a(Arena::regret, sv.ncells),
            inst: a(Arena::inst, sv.ncells),
            cur: a(Arena::cur, sv.ncells),
            sum_strat: a(Arena::sum_strat, sv.ncells),
            avg: a(Arena::avg, sv.ncells),
            xb: d2h(&self.stream, &self.pools._xb, sv.row0 * 2 * dg, sv.nleaf * 2 * dg),
            u: d2h(&self.stream, &self.pools._u, sv.row0 * rk, sv.nleaf * rk),
        };
        self.release(s);
        Ok(out)
    }

    fn release(&mut self, s: usize) {
        if let Some(sv) = self.live[s].take() {
            self.pools.free(sv.row0);
        }
        self.free.push(s);
    }
}

/// Compare the device's view of `Desc` with the host's, field by field.
///
/// Everything else in this module rests on the two agreeing: the descriptor is
/// written by Rust and read by CUDA, and a padding difference would corrupt
/// every kernel while looking like a numerical bug. Both the probe kernel and
/// the expected values are generated from the one declaration in `layout.rs`,
/// so a field cannot be checked on one side only.
fn check_abi(stream: &Arc<CudaStream>, probe: &CudaFunction) -> Result<(), String> {
    let want = super::layout::abi_expected();
    let mut got = zeros::<i32>(stream, want.len())?;
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

fn zeros<T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<CudaSlice<T>, String> {
    stream.alloc_zeros(n.max(1)).map_err(|e| format!("{e:?}"))
}

fn htod<T: cudarc::driver::DeviceRepr + Unpin>(
    stream: &Arc<CudaStream>,
    v: &[T],
) -> Result<CudaSlice<T>, String> {
    let mut buf = unsafe { stream.alloc(v.len().max(1)) }.map_err(|e| format!("{e:?}"))?;
    stream.memcpy_htod(v, &mut buf).map_err(|e| format!("{e:?}"))?;
    Ok(buf)
}

/// Read part of a device buffer back to the host.
///
/// cudarc issues the copy asynchronously and, for an ordinary host slice,
/// attaches no synchronisation to it — so the wait belongs here. It costs
/// nothing in the steady state: downloads happen at a solve's two trip
/// boundaries, never inside a tick.
fn d2h(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>, off: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    if n > 0 {
        let _ = stream.memcpy_dtoh(&buf.slice(off..off + n), &mut v);
        let _ = stream.synchronize();
    }
    v
}

fn ptr<T>(stream: &Arc<CudaStream>, buf: &CudaSlice<T>) -> *const T {
    let (p, _sync) = buf.device_ptr(stream);
    p as usize as *const T
}

fn ptr_mut<T>(stream: &Arc<CudaStream>, buf: &mut CudaSlice<T>) -> *mut T {
    let (p, _sync) = buf.device_ptr_mut(stream);
    p as usize as *mut T
}

/// A batch of identical row-major GEMMs sharing one `B`: `C_i = A_i . B`,
/// where the `A_i` are `stride_a` apart. Used where a matrix is a block of
/// each row of a wider table and gathering it first would be pure copying.
#[allow(clippy::too_many_arguments)]
fn gemm_batched(
    blas: &CudaBlas,
    m: usize, n: usize, k: usize,
    a: *const f32, lda: usize, stride_a: usize,
    b: *const f32, ldb: usize,
    c: *mut f32, ldc: usize, stride_c: usize,
    batch: usize,
) {
    if m == 0 || n == 0 || k == 0 || batch == 0 {
        return;
    }
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: as `gemm` — device pointers the service owns, shapes from the
    // arena sizes the caller cut.
    let _ = unsafe {
        cudarc::cublas::result::sgemm_strided_batched(
            *blas.handle(), CUBLAS_OP_N, CUBLAS_OP_N,
            n as i32, m as i32, k as i32, &alpha,
            b, ldb as i32, 0,
            a, lda as i32, stride_a as i64,
            &beta, c, ldc as i32, stride_c as i64,
            batch as i32,
        )
    };
}

/// One row-major GEMM over raw device pointers: `C[m,n] = A[m,k] . B[k,n]`,
/// plus `beta * C`. Every matrix is stored row-major, so cuBLAS — which is
/// column-major — is handed the product transposed, with the row strides as
/// its leading dimensions.
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
    // SAFETY: the pointers are device addresses owned by the service's own
    // buffers, and the shapes come from the arena sizes the caller cut.
    // Swapping the operands is what turns the column-major call into the
    // row-major product above.
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
