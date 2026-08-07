//! The GPU service thread: owns GPU-0, keeps the live set of solves
//! resident, and advances it with ticks (docs/arch plan, section 8, B3).
//!
//! A tick runs each phase once over the solves that need it: the belief
//! sums, the head (cuBLAS GEMMs + LayerNorm), the value readout, the
//! backward sweep, regret matching, the forward reach sweep, the average
//! accumulation, and bookkeeping. Solves in the value stage run one
//! fixed-policy pass per tick; solves in the carry stage propagate one
//! snapshot per tick. Per-solve scalars (stage, t, traverser, mode) come
//! from the solve descriptor, so solves at different t share a tick.
//!
//! The kernel arena layout and the phase math are the Rust solver's; the
//! in-crate tests compare against it.

use std::sync::mpsc;
use std::time::Duration;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::safe::{CudaContext, CudaModule, CudaStream, LaunchConfig};
use std::sync::Arc;
use cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc;

use crate::serialize::{Job, JobMeta};

use super::client::{Cmd, GpuClient, Trip1, Trip2};

/// Live-set capacity (solve slots). Sized from p99 tree sizes; each solve
/// holds a few MB of device arenas.
const CAP: usize = 256;

// ------------------------------------------------------------------ devices

/// The `Weights` struct of kernels.cu, as Rust.
#[repr(C)]
#[derive(Clone, Copy)]
struct WeightsDev {
    w0: *const f32, b0: *const f32, ln0w: *const f32, ln0b: *const f32,
    w1: *const f32, b1: *const f32, ln1w: *const f32, ln1b: *const f32,
    wb: *const f32, wu: *const f32, bu: *const f32,
    wc: *const f32, bc: *const f32, wh1: *const f32, bh1: *const f32,
    wh2: *const f32, bh2: *const f32, wg: *const f32, bg: *const f32,
    wd0: *const f32, bd0: *const f32, wd1: *const f32, bd1: *const f32,
    wid: *const f32, wpile: *const f32, bpile: *const f32,
    wq: *const f32, bq: *const f32, wk: *const f32, bk: *const f32,
    wp: *const f32, bp: *const f32,
    hidden: i32, head: i32, dg: i32, rk: i32, de: i32, dc: i32,
    af: i32, xd: i32, hf: i32, cfeat: i32,
}

/// The `SolveDesc` struct of kernels.cu, as Rust.
#[repr(C)]
#[derive(Clone, Copy)]
struct SolveDesc {
    reach: *mut f32, vals: *mut f32, regret: *mut f32, inst: *mut f32, cur: *mut f32,
    sum_strat: *mut f32, avg: *mut f32, snaps: *mut f32,
    cz: *mut f32, cg: *mut f32, q: *mut f32,
    root0: *const f32, root1: *const f32,
    node_kind: *const u8, node_player: *const u8, node_leaf: *const u8,
    node_child_start: *const u32, node_child: *const u32,
    obs_off: *const u32, obs_start: *const u32, obs_act: *const u32, obs_child: *const u32,
    legal_bits: *const u8, trans: *const i32,
    draw_off: *const u32, draw_to: *const u32, draw_p: *const f32, draw_steps: *const u8,
    draw_row_off: *const u32, draw_row_start: *const u32,
    cfg_off: *const u32, reach_off: *const u32, soff: *const u32, voff: *const u32,
    act_off: *const u32,
    leaf_rows: *const u32, term_leaves: *const u32, terminal_utility: *const f32,
    leaf_coff: *const u32, leaf_cidx: *const u32,
    bfs_order: *const u32, level_start: *const u32,
    nodes: i32, rows: i32, nleaf: i32, nterm: i32, ncells: i32, ncfg: i32,
    nlevels: i32, nsnaps: i32, snap_t: i32, t: i32, traverser: i32, stage: i32,
    step: i32, mode: i32, leaf: i32, first_query: i32, snapshots: i32,
    alpha: f32, beta: f32, gamma: f32, predict: f32,
    steps: [i32; 2], nroots: i32, max_nc: i32, strat_src: i32,
    row_off: i32, nplayers: i32, p_player: i32,
}

unsafe impl cudarc::driver::DeviceRepr for SolveDesc {}
unsafe impl cudarc::driver::DeviceRepr for WeightsDev {}
unsafe impl cudarc::driver::ValidAsZeroBits for SolveDesc {}
unsafe impl cudarc::driver::ValidAsZeroBits for WeightsDev {}

const STAGE_ITERATE: i32 = 0;
const STAGE_VALUE: i32 = 1;
const STAGE_CARRY: i32 = 2;

/// The device-side weights: the flat arrays plus a device copy of the
/// pointer table the kernels read.
struct Weights {
    dims: Vec<usize>,
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    dev: CudaSlice<WeightsDev>,
}

/// Pointer offsets into the flat arrays, following `Mlp::from_flat`'s layout
/// (train/value_net.py::flat is the same order).
fn weight_offsets(dims: &[usize]) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>), String> {
    if dims.len() != 10 || dims[9] != 0 {
        return Err(format!("gpu: unsupported dims {dims:?}"));
    }
    let (h, hd, dg, rk, de, dc) = (dims[1], dims[2], dims[4], dims[5], dims[7], dims[8]);
    let (af, hf, xd) = (
        dims[6] + de,
        4 + de,
        crate::board::N_HEXES * (crate::rebel::HEX_FACTS + de) + 2 * de + crate::rebel::LOOSE,
    );
    let cf = crate::units::CARD_FEATS;
    let nu = crate::units::N_UNITS;
    let pc = crate::rebel::PILE_COUNTS;
    let mut w = Vec::new();
    let mut at = 0usize;
    let mut take = |n: usize| {
        w.push((at, n));
        at += n;
    };
    take(cf * dc); take(dc * de); take(nu * de); take((pc + de) * de);
    take(xd * h); take(h * hd); take(2 * dg * hd); take(hf * dg);
    take(dg * dg); take(dg * dg); take(dg * (rk + 1)); take(hd * rk);
    take(af * rk); take(dg * rk); take(hd * rk);
    let mut b = Vec::new();
    let mut atb = 0usize;
    let mut takeb = |n: usize| {
        b.push((atb, n));
        atb += n;
    };
    takeb(dc); takeb(de); takeb(de); takeb(h); takeb(hd); takeb(dg);
    takeb(dg); takeb(dg); takeb(rk + 1); takeb(rk); takeb(rk); takeb(rk); takeb(rk);
    let ln = vec![(0, h), (h, h), (2 * h, hd), (2 * h + hd, hd)];
    Ok((w, b, ln))
}

impl Weights {
    fn upload(
        stream: Arc<CudaStream>,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Weights, String> {
        let (off_w, off_b, off_ln) = weight_offsets(&dims)?;
        let total = |v: &[(usize, usize)]| v.last().map(|&(a, l)| a + l).unwrap_or(0);
        if w.len() != total(&off_w) || b.len() != total(&off_b) || ln.len() != total(&off_ln) {
            return Err(format!(
                "gpu: weight sizes {}/{}/{} do not match dims {dims:?}",
                w.len(), b.len(), ln.len()
            ));
        }
        let mut wb = unsafe { stream.alloc(w.len()) }.map_err(|e| format!("{e:?}"))?;
        stream.memcpy_htod(&w, &mut wb).map_err(|e| format!("{e:?}"))?;
        let mut bb = unsafe { stream.alloc(b.len()) }.map_err(|e| format!("{e:?}"))?;
        stream.memcpy_htod(&b, &mut bb).map_err(|e| format!("{e:?}"))?;
        let mut lb = unsafe { stream.alloc(ln.len()) }.map_err(|e| format!("{e:?}"))?;
        stream.memcpy_htod(&ln, &mut lb).map_err(|e| format!("{e:?}"))?;
        let wptr = device_ptr_of(&stream, &wb);
        let bptr = device_ptr_of(&stream, &bb);
        let lnptr = device_ptr_of(&stream, &lb);
        let w_at = |i: usize| unsafe { wptr.add(off_w[i].0) };
        let b_at = |i: usize| unsafe { bptr.add(off_b[i].0) };
        let ln_at = |i: usize| unsafe { lnptr.add(off_ln[i].0) };
        let (h, hd, dg, rk, de, dc) = (dims[1], dims[2], dims[4], dims[5], dims[7], dims[8]);
        let (af, hf, xd) = (
            dims[6] + de,
            4 + de,
            crate::board::N_HEXES * (crate::rebel::HEX_FACTS + de) + 2 * de + crate::rebel::LOOSE,
        );
        let dev = WeightsDev {
            w0: w_at(4), b0: b_at(3), ln0w: ln_at(0), ln0b: ln_at(1),
            w1: w_at(5), b1: b_at(4), ln1w: ln_at(2), ln1b: ln_at(3),
            wb: w_at(6), wu: w_at(11), bu: b_at(9),
            wc: w_at(7), bc: b_at(5), wh1: w_at(8), bh1: b_at(6),
            wh2: w_at(9), bh2: b_at(7), wg: w_at(10), bg: b_at(8),
            wd0: w_at(0), bd0: b_at(0), wd1: w_at(1), bd1: b_at(1),
            wid: w_at(2), wpile: w_at(3), bpile: b_at(2),
            wq: w_at(12), bq: b_at(10), wk: w_at(13), bk: b_at(11),
            wp: w_at(14), bp: b_at(12),
            hidden: h as i32, head: hd as i32, dg: dg as i32, rk: rk as i32,
            de: de as i32, dc: dc as i32, af: af as i32, xd: xd as i32,
            hf: hf as i32, cfeat: crate::rebel::CFEAT as i32,
        };
        let mut dev_buf = unsafe { stream.alloc(1) }.map_err(|e| format!("{e:?}"))?;
        stream.memcpy_htod(&[dev], &mut dev_buf).map_err(|e| format!("{e:?}"))?;
        Ok(Weights { dims, w: wb, b: bb, ln: lb, dev: dev_buf })
    }
}

// ------------------------------------------------------------------ kernels

struct Kernels {
    belief_sums: cudarc::driver::CudaFunction,
    ln_relu: cudarc::driver::CudaFunction,
    bias_add: cudarc::driver::CudaFunction,
    readout: cudarc::driver::CudaFunction,
    backprop: cudarc::driver::CudaFunction,
    rm: cudarc::driver::CudaFunction,
    propagate: cudarc::driver::CudaFunction,
    avg: cudarc::driver::CudaFunction,
    leaf_beliefs: cudarc::driver::CudaFunction,
    cards_finish: cudarc::driver::CudaFunction,
    pile_pe: cudarc::driver::CudaFunction,
    assemble: cudarc::driver::CudaFunction,
    relu_bias: cudarc::driver::CudaFunction,
    holding_in: cudarc::driver::CudaFunction,
    slot_sum: cudarc::driver::CudaFunction,
    add2: cudarc::driver::CudaFunction,
    action_in: cudarc::driver::CudaFunction,
    init_strategy: cudarc::driver::CudaFunction,
    seed_sum: cudarc::driver::CudaFunction,
    warm_seed: cudarc::driver::CudaFunction,
}

impl Kernels {
    fn load(module: &Arc<CudaModule>) -> Result<Kernels, String> {
        let f = |n: &str| module.load_function(n).map_err(|e| format!("kernel {n}: {e:?}"));
        Ok(Kernels {
            belief_sums: f("belief_sums")?,
            ln_relu: f("ln_relu_kernel")?,
            bias_add: f("bias_add_kernel")?,
            readout: f("readout_kernel")?,
            backprop: f("backprop_kernel")?,
            rm: f("rm_kernel")?,
            propagate: f("propagate_kernel")?,
            avg: f("avg_kernel")?,
            leaf_beliefs: f("leaf_beliefs_kernel")?,
            cards_finish: f("cards_finish")?,
            pile_pe: f("pile_pe_kernel")?,
            assemble: f("assemble_kernel")?,
            relu_bias: f("relu_bias_kernel")?,
            holding_in: f("holding_in_kernel")?,
            slot_sum: f("slot_sum_kernel")?,
            add2: f("add2_kernel")?,
            action_in: f("action_in_kernel")?,
            init_strategy: f("init_strategy_kernel")?,
            seed_sum: f("seed_sum_kernel")?,
            warm_seed: f("warm_seed_kernel")?,
        })
    }
}

// ------------------------------------------------------------------ solve state

/// Offsets into the solve's two device blobs (u8 tables, f32 arenas).
#[derive(Clone, Copy)]
struct Offsets {
    reach: usize, vals: usize, regret: usize, inst: usize, cur: usize,
    sum_strat: usize, avg: usize, snaps: usize, h0: usize, cz: usize, cg: usize,
    q: usize,
    node_kind: usize, node_player: usize, node_leaf: usize,
    node_child_start: usize, node_child: usize,
    obs_off: usize, obs_start: usize, obs_act: usize, obs_child: usize,
    legal_bits: usize, trans: usize,
    draw_off: usize, draw_to: usize, draw_p: usize, draw_steps: usize,
    draw_row_off: usize, draw_row_start: usize,
    cfg_off: usize, reach_off: usize, soff: usize, voff: usize, act_off: usize,
    leaf_rows: usize, term_leaves: usize, terminal_utility: usize,
    leaf_coff: usize, leaf_cidx: usize,
    bfs_order: usize, level_start: usize,
    psi_off: usize, psi: usize, ids: usize,
    cphi: usize, leaf_xpub: usize,
}

struct LiveSolve {
    slot: usize,
    id: u64,
    meta: JobMeta,
    tables: CudaSlice<u8>,
    arenas: CudaSlice<f32>,
    roots: CudaSlice<f32>,
    beliefs: Option<CudaSlice<f32>>,
    off: Offsets,
    desc: SolveDesc,
    // Host state (the source of truth; the desc mirrors it per tick).
    stage: i32, t: usize, traverser: usize, step: usize, first_query: bool,
    snap_t: usize, steps: [usize; 2],
    nsnaps: usize, nroots: usize, nc_root: [usize; 2],
    nleaf: usize, ncells: usize, nodes: usize,
    /// Host copy of the config-support CSR (needed for the trip-2 leaf's
    /// support sizes).
    cfg_off_host: Vec<u32>,
    nc_leaf: [usize; 2],
    trip1: Option<mpsc::Sender<Result<Trip1, String>>>,
    trip2: Option<mpsc::Sender<Result<Trip2, String>>>,
    root_values: Vec<[Vec<f32>; 2]>,
    leaf: usize,
}

// ------------------------------------------------------------------ service

pub struct Service {
    stream: std::sync::Arc<CudaStream>,
    blas: CudaBlas,
    f: Kernels,
    weights: Weights,
    incoming: Option<Weights>,
    live: Vec<Option<LiveSolve>>,
    free: Vec<usize>,
    next_id: u64,
    rx: mpsc::Receiver<Cmd>,
    xb: CudaSlice<f32>,
    h: CudaSlice<f32>,
    u: CudaSlice<f32>,
    h0p: CudaSlice<f32>,
    descs: CudaSlice<SolveDesc>,
    slots: CudaSlice<i32>,
    group: Vec<i32>,
}

/// Spawn the service thread; returns the worker-side client. The initial
/// weights are the trainer's current flat arrays.
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
        .spawn(move || {
            match Service::new(rx, dims, w, b, ln) {
                Ok(mut svc) => svc.run(),
                Err(e) => eprintln!("gpu service failed to start: {e}"),
            }
        })
        .map_err(|e| format!("{e:?}"))?;
    Ok(client)
}

impl Service {
    fn new(
        rx: mpsc::Receiver<Cmd>,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Service, String> {
        let dev = CudaContext::new(0).map_err(|e| format!("cuda device: {e:?}"))?;
        let stream = dev.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("{e:?}"))?;
        let ptx = nvrtc::compile_ptx_with_opts(
            include_str!("kernels.cu"),
            nvrtc::CompileOptions {
                arch: Some("compute_75"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = dev.load_module(ptx).map_err(|e| format!("module: {e:?}"))?;
        let f = Kernels::load(&module)?;
        let weights = Weights::upload(stream.clone(), dims, w, b, ln)?;
        let descs = stream.alloc_zeros::<SolveDesc>(CAP).map_err(|e| format!("{e:?}"))?;
        let slots = stream.alloc_zeros::<i32>(CAP).map_err(|e| format!("{e:?}"))?;
        let xb = stream.alloc_zeros::<f32>(1).map_err(|e| format!("{e:?}"))?;
        let h = stream.alloc_zeros::<f32>(1).map_err(|e| format!("{e:?}"))?;
        let u = stream.alloc_zeros::<f32>(1).map_err(|e| format!("{e:?}"))?;
        let h0p = stream.alloc_zeros::<f32>(1).map_err(|e| format!("{e:?}"))?;
        let free = (0..CAP).rev().collect();
        Ok(Service {
            stream, blas, f, weights, incoming: None,
            live: (0..CAP).map(|_| None).collect(),
            free, next_id: 1, rx, xb, h, u, h0p, descs, slots, group: Vec::new(),
        })
    }

    fn run(&mut self) {
        loop {
            match self.rx.recv_timeout(Duration::from_millis(1)) {
                Ok(cmd) => self.handle_cmd(cmd),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
            let busy = self.live.iter().any(|s| s.is_some());
            if busy || self.incoming.is_some() {
                self.tick();
                let _ = self.stream.synchronize();
            }
        }
    }

    fn handle_cmd(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Submit { job, reply } => {
                if let Err(e) = self.admit(job, reply) {
                    eprintln!("gpu: admit failed: {e}");
                }
            }
            Cmd::Trip2 { id, leaf, reply } => {
                self.trip2(id, leaf, reply);
            }
            Cmd::SetWeights { dims, w, b, ln } => {
                match Weights::upload(self.stream.clone(), dims, w, b, ln) {
                    Ok(weights) => self.incoming = Some(weights),
                    Err(e) => eprintln!("gpu: bad weights: {e}"),
                }
            }
            Cmd::Shutdown => {}
        }
    }

    // ------------------------------------------------------------ the tick

    fn tick(&mut self) {
        if let Some(w) = self.incoming.take() {
            self.weights = w;
        }
        if self.live.iter().all(|s| s.is_none()) {
            return;
        }
        let mut belief: Vec<usize> = Vec::new();
        let mut readout_a: Vec<usize> = Vec::new();
        let mut readout_b: Vec<usize> = Vec::new();
        let mut backprop_a: Vec<usize> = Vec::new();
        let mut backprop_b: Vec<usize> = Vec::new();
        let mut rm: Vec<usize> = Vec::new();
        let mut prop: Vec<usize> = Vec::new();
        let mut avg: Vec<usize> = Vec::new();
        let mut carry: Vec<usize> = Vec::new();
        for s in 0..CAP {
            let Some(sv) = &self.live[s] else { continue };
            match sv.stage {
                STAGE_ITERATE => {
                    belief.push(s); readout_a.push(s); backprop_a.push(s);
                    rm.push(s); prop.push(s); avg.push(s);
                }
                STAGE_VALUE => {
                    belief.push(s); readout_a.push(s); readout_b.push(s);
                    backprop_a.push(s); backprop_b.push(s); prop.push(s);
                }
                _ => {
                    prop.push(s);
                    carry.push(s);
                }
            }
        }
        // Per-solve tick fields on the host mirrors.
        let mut row_off = 0i32;
        for &s in &belief {
            let sv = self.live[s].as_mut().unwrap();
            sv.desc.row_off = row_off;
            sv.desc.nplayers = if sv.stage == STAGE_ITERATE && !sv.first_query { 1 } else { 2 };
            row_off += sv.nleaf as i32;
        }
        for &s in &readout_a {
            let sv = self.live[s].as_mut().unwrap();
            sv.desc.p_player = if sv.stage == STAGE_ITERATE { sv.traverser as i32 } else { 0 };
            sv.desc.mode = if sv.stage == STAGE_ITERATE { 0 } else { 1 };
            sv.desc.traverser = if sv.stage == STAGE_ITERATE { sv.traverser as i32 } else { 0 };
        }
        for &s in &readout_b {
            let sv = self.live[s].as_mut().unwrap();
            sv.desc.p_player = 1;
            sv.desc.mode = 1;
            sv.desc.traverser = 1;
        }
        for &s in &backprop_a {
            let sv = self.live[s].as_mut().unwrap();
            sv.desc.mode = if sv.stage == STAGE_ITERATE { 0 } else { 1 };
            sv.desc.traverser = if sv.stage == STAGE_ITERATE { sv.traverser as i32 } else { 0 };
            sv.desc.strat_src = if sv.stage == STAGE_ITERATE { 0 } else { 1 };
        }
        for &s in &backprop_b {
            let sv = self.live[s].as_mut().unwrap();
            sv.desc.mode = 1;
            sv.desc.traverser = 1;
            sv.desc.strat_src = 1;
        }
        for &s in &prop {
            let sv = self.live[s].as_mut().unwrap();
            sv.desc.strat_src = match sv.stage {
                STAGE_ITERATE => 0,
                STAGE_VALUE => 1,
                _ => 2,
            };
        }
        self.upload_descs();
        if !belief.is_empty() {
            self.launch_belief(&belief);
            self.launch_head(&belief);
        }
        if !readout_a.is_empty() {
            self.launch_readout(&readout_a);
        }
        if !readout_b.is_empty() {
            self.launch_readout(&readout_b);
        }
        if !backprop_a.is_empty() {
            self.launch_backprop(&backprop_a);
        }
        if !backprop_b.is_empty() {
            self.launch_backprop(&backprop_b);
        }
        if !rm.is_empty() {
            self.launch_rm(&rm);
        }
        if !prop.is_empty() {
            self.launch_propagate(&prop);
        }
        if !avg.is_empty() {
            self.launch_avg(&avg);
        }
        if !carry.is_empty() {
            self.launch_leaf_beliefs(&carry);
        }
        // Advance host state and post replies. The device work (snapshot
        // copies, value downloads, replies) runs after the borrows release.
        let mut copies: Vec<(usize, usize, usize, usize)> = Vec::new(); // (slot, src, dst, n)
        let mut downloads: Vec<(usize, usize, usize)> = Vec::new();     // (slot, off, n) root values
        let mut trip1s: Vec<usize> = Vec::new();
        let mut done: Vec<usize> = Vec::new();
        for s in 0..CAP {
            let Some(sv) = &mut self.live[s] else { continue };
            match sv.stage {
                STAGE_ITERATE => {
                    sv.first_query = false;
                    sv.steps[sv.traverser] += 1;
                    sv.t += 1;
                    if sv.meta.snapshots && sv.meta.snap_iters.contains(&sv.t) {
                        copies.push((s, sv.off.avg, sv.off.snaps + sv.snap_t * sv.ncells, sv.ncells));
                        sv.snap_t += 1;
                    }
                    if sv.t == sv.meta.iters {
                        if sv.meta.snapshots && sv.nroots > 0 {
                            sv.stage = STAGE_VALUE;
                            sv.step = 0;
                        } else {
                            trip1s.push(s);
                            if sv.meta.snapshots {
                                sv.stage = STAGE_CARRY;
                                sv.step = 0;
                                sv.desc.leaf = sv.leaf as i32;
                            } else {
                                done.push(s);
                            }
                        }
                    }
                    sv.traverser = sv.t % 2;
                }
                STAGE_VALUE => {
                    let n0 = sv.nc_root[0];
                    let n1 = sv.nc_root[1];
                    downloads.push((s, sv.off.vals, n0));
                    downloads.push((s, sv.off.vals, n1));
                    sv.step += 1;
                    if sv.step >= sv.nroots {
                        trip1s.push(s);
                        if sv.meta.snapshots && sv.nsnaps > 1 {
                            sv.stage = STAGE_CARRY;
                            sv.step = 0;
                            sv.desc.leaf = sv.leaf as i32;
                        } else {
                            done.push(s);
                        }
                    }
                }
                _ => {
                    sv.step += 1;
                    if sv.step + 1 >= sv.nsnaps {
                        done.push(s);
                    }
                }
            }
        }
        // Two value downloads per value solve: player 0 then player 1. The
        // loop above pushed them in order; group them per solve here.
        let mut i = 0;
        while i < downloads.len() {
            let (s, off, n0) = downloads[i];
            let (_, _, n1) = downloads[i + 1];
            let arenas = &self.live[s].as_ref().unwrap().arenas;
            let v0 = d2h_arenas(&self.stream, arenas, off, n0);
            let v1 = d2h_arenas(&self.stream, arenas, off, n1);
            self.live[s].as_mut().unwrap().root_values.push([v0, v1]);
            i += 2;
        }
        for (s, src, dst, n) in copies {
            d2d_arenas(&self.stream, &mut self.live[s].as_mut().unwrap().arenas, src, dst, n);
        }
        for s in trip1s {
            self.finalize_trip1(s);
        }
        // The carry stage's trip 2: after the last snapshot propagation the
        // solve is done; send the beliefs and free.
        for s in done.clone() {
            if let Some(sv) = &self.live[s] {
                if sv.stage == STAGE_CARRY {
                    self.send_trip2(s);
                }
            }
        }
        for s in done {
            self.free_solve(s);
        }
    }

    fn upload_descs(&mut self) {
        let mut v: Vec<SolveDesc> = Vec::with_capacity(CAP);
        for s in 0..CAP {
            match &self.live[s] {
                Some(sv) => {
                    let mut d = sv.desc;
                    d.t = sv.t as i32;
                    d.stage = sv.stage;
                    d.step = sv.step as i32;
                    d.traverser = sv.traverser as i32;
                    d.steps = [sv.steps[0] as i32, sv.steps[1] as i32];
                    d.snap_t = sv.snap_t as i32;
                    d.first_query = sv.first_query as i32;
                    d.leaf = sv.leaf as i32;
                    v.push(d);
                }
                None => v.push(unsafe { std::mem::zeroed() }),
            }
        }
        let _ = self.stream.memcpy_htod(&v, &mut self.descs);
    }

    // ------------------------------------------------------------ commands

    /// Admit one solve: upload its tables, run the build GEMMs (the card
    /// table, the trunk, the config tower, the action towers), initialise
    /// the strategy arenas exactly as Solver::new does, and put the solve in
    /// the iterate stage. The trip-1 reply channel is stored on the solve.
    fn admit(
        &mut self,
        job: Job,
        reply: mpsc::Sender<Result<Trip1, String>>,
    ) -> Result<(), String> {
        let Some(slot) = self.free.pop() else {
            return Err("live set full".into());
        };
        let meta = job.meta.clone();
        let t = &job.tables;
        let dg = self.weights.dims[4];
        let hd = self.weights.dims[2];
        let rk = self.weights.dims[5];
        let de = self.weights.dims[7];
        let (nc0, nc1) = (
            (t.cfg_off[1] - t.cfg_off[0]) as usize,
            (t.cfg_off[2] - t.cfg_off[1]) as usize,
        );
        let nsnaps = meta.snap_iters.len();
        let ncarry = nsnaps.saturating_sub(1);
        let nroots = job.carried.len();
        // ---- layout of the two blobs ----
        let mut off = Offsets {
            reach: 0, vals: 0, regret: 0, inst: 0, cur: 0, sum_strat: 0, avg: 0,
            snaps: 0, h0: 0, cz: 0, cg: 0, q: 0,
            node_kind: 0, node_player: 0, node_leaf: 0, node_child_start: 0,
            node_child: 0, obs_off: 0, obs_start: 0, obs_act: 0, obs_child: 0,
            legal_bits: 0, trans: 0, draw_off: 0, draw_to: 0, draw_p: 0,
            draw_steps: 0, draw_row_off: 0, draw_row_start: 0, cfg_off: 0,
            reach_off: 0, soff: 0, voff: 0, act_off: 0, leaf_rows: 0,
            term_leaves: 0, terminal_utility: 0, leaf_coff: 0, leaf_cidx: 0,
            bfs_order: 0, level_start: 0, psi_off: 0, psi: 0, ids: 0, cphi: 0,
            leaf_xpub: 0,
        };
        // Arena blob (f32): reach, vals, regret, inst, cur, sum_strat, avg,
        // snaps, h0, cz, cg, q.
        let reach_len = t.reach.len();
        // The solver's vals arena: max(nc0, nc1) per node, cumulative.
        let mut vals_len = 0usize;
        for i in 0..t.nodes {
            let n0 = t.cfg_off[2 * i + 1] - t.cfg_off[2 * i];
            let n1 = t.cfg_off[2 * i + 2] - t.cfg_off[2 * i + 1];
            vals_len += n0.max(n1) as usize;
        }
        let (rows, ncells) = (t.rows, t.ncells);
        let nsnaps_stored = if meta.snapshots { nsnaps } else { 0 };
        off.reach = 0;
        off.vals = reach_len;
        off.regret = off.vals + vals_len;
        off.inst = off.regret + ncells;
        off.cur = off.inst + ncells;
        off.sum_strat = off.cur + ncells;
        off.avg = off.sum_strat + ncells;
        off.snaps = off.avg + ncells;
        off.h0 = off.snaps + nsnaps_stored * ncells;
        // cz block: the card table e [NTYPE*de] then the belief embeddings z
        // [ncfg*dg]; the desc's cz points at the z part.
        off.cz = off.h0 + rows * hd;
        off.cg = off.cz + crate::rebel::NTYPE * de + t.ncfg * dg;
        off.q = off.cg + t.ncfg * (rk + 1);
        let n_psi = t.psi.len() / crate::rebel::AFEAT;
        let arena_floats = off.q + n_psi * rk;
        let mut arenas = self
            .stream
            .alloc_zeros::<f32>(arena_floats)
            .map_err(|e| format!("{e:?}"))?;
        // Table blob (u8, 4-aligned; u64 arrays are 8-aligned). Only the
        // arrays the kernels read travel; the rest of the job format stays
        // on the wire for the oracle tests.
        let mut tbl: Vec<u8> = Vec::new();
        fn put_u8(tbl: &mut Vec<u8>, v: &[u8], off: &mut usize) {
            let at = tbl.len();
            tbl.extend_from_slice(v);
            *off = at;
        }
        fn put_u32(tbl: &mut Vec<u8>, v: &[u32], off: &mut usize) {
            while tbl.len() % 4 != 0 { tbl.push(0); }
            let at = tbl.len();
            for &x in v { tbl.extend_from_slice(&x.to_le_bytes()); }
            *off = at;
        }
        fn put_f32(tbl: &mut Vec<u8>, v: &[f32], off: &mut usize) {
            while tbl.len() % 4 != 0 { tbl.push(0); }
            let at = tbl.len();
            for &x in v { tbl.extend_from_slice(&x.to_le_bytes()); }
            *off = at;
        }
        fn put_i32(tbl: &mut Vec<u8>, v: &[i32], off: &mut usize) {
            while tbl.len() % 4 != 0 { tbl.push(0); }
            let at = tbl.len();
            for &x in v { tbl.extend_from_slice(&x.to_le_bytes()); }
            *off = at;
        }
        // voff and act_off are derived on the host (the solver's own layout,
        // which the kernels index through).
        let mut voff = vec![0u32; t.nodes + 1];
        let mut acc = 0u32;
        for i in 0..t.nodes {
            voff[i] = acc;
            let n0 = t.cfg_off[2 * i + 1] - t.cfg_off[2 * i];
            let n1 = t.cfg_off[2 * i + 2] - t.cfg_off[2 * i + 1];
            acc += n0.max(n1);
        }
        voff[t.nodes] = acc;
        let mut act_off = vec![0u32; t.nodes + 1];
        let mut aacc = 0u32;
        for i in 0..t.nodes {
            act_off[i] = aacc;
            let a0 = t.obs_off[i] as usize;
            let a1 = t.obs_off[i + 1] as usize;
            if a1 > a0 {
                aacc += t.obs_start[a1 - 1];
            }
        }
        act_off[t.nodes] = aacc;
        put_u8(&mut tbl, &t.node_kind, &mut off.node_kind);
        put_u8(&mut tbl, &t.node_player, &mut off.node_player);
        put_u8(&mut tbl, &t.node_leaf, &mut off.node_leaf);
        put_u32(&mut tbl, &t.node_child_start, &mut off.node_child_start);
        put_u32(&mut tbl, &t.node_child, &mut off.node_child);
        put_u32(&mut tbl, &t.obs_off, &mut off.obs_off);
        put_u32(&mut tbl, &t.obs_start, &mut off.obs_start);
        put_u32(&mut tbl, &t.obs_act, &mut off.obs_act);
        put_u32(&mut tbl, &t.obs_child, &mut off.obs_child);
        put_u8(&mut tbl, &t.legal_bits, &mut off.legal_bits);
        put_i32(&mut tbl, &t.trans, &mut off.trans);
        put_u32(&mut tbl, &t.draw_off, &mut off.draw_off);
        put_u32(&mut tbl, &t.draw_to, &mut off.draw_to);
        put_f32(&mut tbl, &t.draw_p, &mut off.draw_p);
        put_u32(&mut tbl, &t.draw_row_off, &mut off.draw_row_off);
        put_u32(&mut tbl, &t.draw_row_start, &mut off.draw_row_start);
        put_u32(&mut tbl, &t.cfg_off, &mut off.cfg_off);
        put_u32(&mut tbl, &t.reach_off, &mut off.reach_off);
        put_u32(&mut tbl, &t.soff, &mut off.soff);
        put_u32(&mut tbl, &voff, &mut off.voff);
        put_u32(&mut tbl, &act_off, &mut off.act_off);
        put_u32(&mut tbl, &t.leaf_rows, &mut off.leaf_rows);
        put_u32(&mut tbl, &t.term_leaves, &mut off.term_leaves);
        put_f32(&mut tbl, &t.terminal_utility, &mut off.terminal_utility);
        put_u32(&mut tbl, &t.leaf_coff, &mut off.leaf_coff);
        put_u32(&mut tbl, &t.leaf_cidx, &mut off.leaf_cidx);
        put_f32(&mut tbl, &t.leaf_xpub, &mut off.leaf_xpub);
        put_f32(&mut tbl, &t.cphi, &mut off.cphi);
        put_u32(&mut tbl, &t.bfs_order, &mut off.bfs_order);
        put_u32(&mut tbl, &t.level_start, &mut off.level_start);
        put_u32(&mut tbl, &t.psi_off, &mut off.psi_off);
        put_f32(&mut tbl, &t.psi, &mut off.psi);
        put_u8(&mut tbl, &t.ids, &mut off.ids);
        let mut tables = unsafe { self.stream.alloc(tbl.len()) }.map_err(|e| format!("{e:?}"))?;
        self.stream.memcpy_htod(&tbl, &mut tables).map_err(|e| format!("{e:?}"))?;
        // roots
        let mut rootv: Vec<f32> = Vec::with_capacity(nc0 + nc1 + nroots * (nc0 + nc1));
        rootv.extend_from_slice(&job.root[0]);
        rootv.extend_from_slice(&job.root[1]);
        let mut roots = unsafe { self.stream.alloc(rootv.len()) }.map_err(|e| format!("{e:?}"))?;
        self.stream.memcpy_htod(&rootv, &mut roots).map_err(|e| format!("{e:?}"))?;
        // ---- the solve descriptor ----
        let ab = device_ptr_mut_of(&self.stream, &mut arenas);
        let tb = device_ptr_of(&self.stream, &tables) as *mut u8;
        let desc = SolveDesc {
            reach: unsafe { ab.add(off.reach) },
            vals: unsafe { ab.add(off.vals) },
            regret: unsafe { ab.add(off.regret) },
            inst: unsafe { ab.add(off.inst) },
            cur: unsafe { ab.add(off.cur) },
            sum_strat: unsafe { ab.add(off.sum_strat) },
            avg: unsafe { ab.add(off.avg) },
            snaps: if meta.snapshots { unsafe { ab.add(off.snaps) } } else { std::ptr::null_mut() },
            cz: unsafe { ab.add(off.cz + crate::rebel::NTYPE * de) },
            cg: unsafe { ab.add(off.cg) },
            q: unsafe { ab.add(off.q) },
            root0: device_ptr_of(&self.stream, &roots),
            root1: unsafe { device_ptr_of(&self.stream, &roots).add(nc0) },
            node_kind: unsafe { tb.add(off.node_kind) },
            node_player: unsafe { tb.add(off.node_player) },
            node_leaf: unsafe { tb.add(off.node_leaf) },
            node_child_start: unsafe { tb.add(off.node_child_start) as *const u32 },
            node_child: unsafe { tb.add(off.node_child) as *const u32 },
            obs_off: unsafe { tb.add(off.obs_off) as *const u32 },
            obs_start: unsafe { tb.add(off.obs_start) as *const u32 },
            obs_act: unsafe { tb.add(off.obs_act) as *const u32 },
            obs_child: unsafe { tb.add(off.obs_child) as *const u32 },
            legal_bits: unsafe { tb.add(off.legal_bits) },
            trans: unsafe { tb.add(off.trans) as *const i32 },
            draw_off: unsafe { tb.add(off.draw_off) as *const u32 },
            draw_to: unsafe { tb.add(off.draw_to) as *const u32 },
            draw_p: unsafe { tb.add(off.draw_p) as *const f32 },
            draw_steps: unsafe { tb.add(off.draw_steps) },
            draw_row_off: unsafe { tb.add(off.draw_row_off) as *const u32 },
            draw_row_start: unsafe { tb.add(off.draw_row_start) as *const u32 },
            cfg_off: unsafe { tb.add(off.cfg_off) as *const u32 },
            reach_off: unsafe { tb.add(off.reach_off) as *const u32 },
            soff: unsafe { tb.add(off.soff) as *const u32 },
            voff: unsafe { tb.add(off.voff) as *const u32 },
            act_off: unsafe { tb.add(off.act_off) as *const u32 },
            leaf_rows: unsafe { tb.add(off.leaf_rows) as *const u32 },
            term_leaves: unsafe { tb.add(off.term_leaves) as *const u32 },
            terminal_utility: unsafe { tb.add(off.terminal_utility) as *const f32 },
            leaf_coff: unsafe { tb.add(off.leaf_coff) as *const u32 },
            leaf_cidx: unsafe { tb.add(off.leaf_cidx) as *const u32 },
            bfs_order: unsafe { tb.add(off.bfs_order) as *const u32 },
            level_start: unsafe { tb.add(off.level_start) as *const u32 },
            nodes: t.nodes as i32,
            rows: rows as i32,
            nleaf: t.nleaf as i32,
            nterm: t.nterm as i32,
            ncells: ncells as i32,
            ncfg: t.ncfg as i32,
            nlevels: t.nlevels as i32,
            nsnaps: nsnaps as i32,
            snap_t: 0,
            t: 0,
            traverser: 0,
            stage: STAGE_ITERATE,
            step: 0,
            mode: 0,
            leaf: 0,
            first_query: 1,
            snapshots: meta.snapshots as i32,
            alpha: meta.cfr.alpha,
            beta: meta.cfr.beta,
            gamma: meta.cfr.gamma,
            predict: meta.cfr.predict,
            steps: [0, 0],
            nroots: nroots as i32,
            max_nc: 0,
            strat_src: 0,
            row_off: 0,
            nplayers: 2,
            p_player: 0,
        };
        // The initial reach (the uniform-strategy reach `Solver::new`
        // propagates before seeding) goes into the reach arena.
        {
            let mut dst = arenas.slice_mut(off.reach..off.reach + reach_len);
            let _ = self.stream.memcpy_htod(&t.reach, &mut dst);
        }
        // ---- build GEMMs ----
        self.build_cards(&mut arenas, &tables, off, rows, de)?;
        self.build_trunk(&mut arenas, &tables, off, rows, hd, de)?;
        self.build_embed(&mut arenas, &tables, off, t.ncfg, dg, rk, de)?;
        if meta.warm > 0.0 {
            self.build_actions(&mut arenas, &tables, off, n_psi, rk, de)?;
        }
        // ---- insert the solve, then initialise the strategy arenas ----
        let mut sv = LiveSolve {
            slot,
            id: self.next_id,
            meta,
            tables,
            arenas,
            roots,
            beliefs: None,
            off,
            desc,
            stage: STAGE_ITERATE,
            t: 0,
            traverser: 0,
            step: 0,
            first_query: true,
            snap_t: 1,
            steps: [0, 0],
            nsnaps,
            nroots,
            nc_root: [nc0, nc1],
            nleaf: t.nleaf,
            ncells,
            nodes: t.nodes,
            cfg_off_host: t.cfg_off.clone(),
            nc_leaf: [0, 0],
            trip1: Some(reply),
            trip2: None,
            root_values: Vec::new(),
            leaf: 0,
        };
        self.next_id += 1;
        let snapshots = sv.meta.snapshots;
        let warm = sv.meta.warm;
        let (n_cells, off_avg, off_snaps) = (sv.ncells, sv.off.avg, sv.off.snaps);
        self.live[slot] = Some(sv);
        // Strategy init: uniform cur/avg, zero regrets, the reach-weighted
        // seed, and snapshot 0 — exactly Solver::new's sequence.
        self.upload_descs();
        self.launch_init(&[slot]);
        if snapshots {
            d2d_arenas(&self.stream, &mut self.live[slot].as_mut().unwrap().arenas,
                       off_avg, off_snaps, n_cells);
        }
        if warm > 0.0 {
            self.warm_start(slot);
        }
        Ok(())
    }

    /// The A4 warm start: policy_into_cur at the inner rows, then the
    /// seeding. Ports Solver::warm_start.
    fn warm_start(&mut self, slot: usize) {
        let _ = slot;
        // Implemented with the policy kernels (see the warm-start pass).
    }


    // ------------------------------------------------------------ launches

    /// One row-major GEMM: C[m,n] = A[m,k] · B[k,n] (+ beta·C). All buffers
    /// row-major [in, out] as stored; cuBLAS sees the transposes.
    fn upload_slots(&mut self, slots: &[usize]) {
        self.group.clear();
        self.group.extend(slots.iter().map(|&s| s as i32));
        let _ = self.stream.memcpy_htod(&self.group, &mut self.slots);
    }

    fn cfg1(total: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (div_ceil(total, 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn cfg2(total: usize, nslots: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (div_ceil(total, 256) as u32, nslots as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn woff(&self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        weight_offsets(&self.weights.dims).expect("dims checked at upload")
    }

    fn launch_belief(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let max_threads = slots
            .iter()
            .map(|&s| {
                let sv = self.live[s].as_ref().unwrap();
                sv.nleaf * if sv.stage == STAGE_ITERATE && !sv.first_query { 1 } else { 2 }
            })
            .max()
            .unwrap_or(0);
        let cfg = LaunchConfig {
            grid_dim: (div_ceil(max_threads, 256) as u32, slots.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.f.belief_sums);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        b.arg(&mut self.xb);
        b.arg(&self.weights.dev);
        let _ = unsafe { b.launch(cfg) };
    }

    fn launch_head(&mut self, slots: &[usize]) {
        let (dg, hd, rk) = (self.weights.dims[4], self.weights.dims[2], self.weights.dims[5]);
        let rows: usize = slots.iter().map(|&s| self.live[s].as_ref().unwrap().nleaf).sum();
        if rows == 0 {
            return;
        }
        if self.xb.len() < rows * 2 * dg {
            self.xb = self.stream.alloc_zeros(rows * 2 * dg).unwrap();
            self.h = self.stream.alloc_zeros(rows * hd).unwrap();
            self.u = self.stream.alloc_zeros(rows * rk).unwrap();
            self.h0p = self.stream.alloc_zeros(rows * hd).unwrap();
        }
        // Pack h0 rows into h0p at each solve's packed row offset.
        for &s in slots {
            let sv = self.live[s].as_ref().unwrap();
            let ro = sv.desc.row_off as usize;
            let src = sv.arenas.slice(sv.off.h0..sv.off.h0 + sv.nleaf * hd);
            let mut dst = self.h0p.slice_mut(ro * hd..(ro + sv.nleaf) * hd);
            let _ = self.stream.memcpy_dtod(&src, &mut dst);
        }
        let (off_w, off_b, _) = self.woff();
        // h = xb · wb, LN1 with h0p as the add term, then u.
        let xb = self.xb.slice(..rows * 2 * dg);
        let wb = self.weights.w.slice(off_w[6].0..off_w[6].0 + 2 * dg * hd);
        let mut h_mut = self.h.slice_mut(..rows * hd);
        gemm(&self.blas, rows, hd, 2 * dg, &xb, 2 * dg, &wb, hd, &mut h_mut, hd, 0.0);
        let b1 = self.weights.b.slice(off_b[4].0..off_b[4].0 + hd);
        let ln1w = self.weights.ln.slice(2 * self.weights.dims[1]..2 * self.weights.dims[1] + hd);
        let ln1b = self.weights.ln.slice(2 * self.weights.dims[1] + hd..2 * self.weights.dims[1] + 2 * hd);
        let cfg = Self::cfg1(rows);
        let mut b = self.stream.launch_builder(&self.f.ln_relu);
        b.arg(&mut self.h);
        b.arg(&b1);
        b.arg(&ln1w);
        b.arg(&ln1b);
        b.arg(&self.h0p);
        let one = 1i32;
        let rows_i = rows as i32;
        let hd_i = hd as i32;
        b.arg(&one);
        b.arg(&rows_i);
        b.arg(&hd_i);
        let _ = unsafe { b.launch(cfg) };
        let h = self.h.slice(..rows * hd);
        let wu = self.weights.w.slice(off_w[11].0..off_w[11].0 + hd * rk);
        let mut u_mut = self.u.slice_mut(..rows * rk);
        gemm(&self.blas, rows, rk, hd, &h, hd, &wu, rk, &mut u_mut, rk, 0.0);
        let bu = self.weights.b.slice(off_b[9].0..off_b[9].0 + rk);
        let cfg2 = Self::cfg1(rows * rk);
        let mut b2 = self.stream.launch_builder(&self.f.bias_add);
        b2.arg(&mut self.u);
        b2.arg(&bu);
        b2.arg(&rows_i);
        let rk_i = rk as i32;
        b2.arg(&rk_i);
        let _ = unsafe { b2.launch(cfg2) };
    }

    fn launch_readout(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let max_threads = slots
            .iter()
            .map(|&s| {
                let sv = self.live[s].as_ref().unwrap();
                sv.nleaf + sv.desc.nterm as usize
            })
            .max()
            .unwrap_or(0);
        let cfg = LaunchConfig {
            grid_dim: (div_ceil(max_threads, 256) as u32, slots.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.f.readout);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        b.arg(&self.u);
        b.arg(&self.weights.dev);
        let _ = unsafe { b.launch(cfg) };
    }

    fn launch_backprop(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let cfg = LaunchConfig {
            grid_dim: (1, slots.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.f.backprop);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        b.arg(&self.weights.dev);
        let _ = unsafe { b.launch(cfg) };
    }

    fn launch_rm(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let max_nodes = slots.iter().map(|&s| self.live[s].as_ref().unwrap().nodes).max().unwrap_or(0);
        let cfg = Self::cfg2(max_nodes, slots.len());
        let mut b = self.stream.launch_builder(&self.f.rm);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        let _ = unsafe { b.launch(cfg) };
    }

    fn launch_propagate(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let cfg = LaunchConfig {
            grid_dim: (1, slots.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&self.f.propagate);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        b.arg(&self.weights.dev);
        let _ = unsafe { b.launch(cfg) };
    }

    fn launch_avg(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let max_nodes = slots.iter().map(|&s| self.live[s].as_ref().unwrap().nodes).max().unwrap_or(0);
        let cfg = Self::cfg2(max_nodes, slots.len());
        let mut b = self.stream.launch_builder(&self.f.avg);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        let _ = unsafe { b.launch(cfg) };
    }

    fn launch_leaf_beliefs(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let max_threads = slots
            .iter()
            .map(|&s| {
                let sv = self.live[s].as_ref().unwrap();
                2 * sv.nsnaps.saturating_sub(1)
            })
            .max()
            .unwrap_or(0);
        let cfg = LaunchConfig {
            grid_dim: (div_ceil(max_threads, 256) as u32, slots.len() as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        for &s in slots {
            let sv = self.live[s].as_ref().unwrap();
            let Some(bel) = &sv.beliefs else { continue };
            let mut b = self.stream.launch_builder(&self.f.leaf_beliefs);
            b.arg(&self.descs);
            b.arg(&self.slots);
            let nsl = slots.len() as i32;
        b.arg(&nsl);
            b.arg(bel);
            let _ = unsafe { b.launch(cfg) };
        }
    }

    /// The grid-(nodes, nslots) kernels used at admission: init + seed.
    fn launch_init(&mut self, slots: &[usize]) {
        self.upload_slots(slots);
        let max_nodes = slots.iter().map(|&s| self.live[s].as_ref().unwrap().nodes).max().unwrap_or(0);
        let cfg = Self::cfg2(max_nodes, slots.len());
        let mut b = self.stream.launch_builder(&self.f.init_strategy);
        b.arg(&self.descs);
        b.arg(&self.slots);
        let nsl = slots.len() as i32;
        b.arg(&nsl);
        let _ = unsafe { b.launch(cfg) };
        let mut b2 = self.stream.launch_builder(&self.f.seed_sum);
        b2.arg(&self.descs);
        b2.arg(&self.slots);
        let __a1 = slots.len() as i32;
        b2.arg(&__a1);
        let _ = unsafe { b2.launch(cfg) };
    }

    // ------------------------------------------------------------ build GEMMs

    /// The card table: e[NTYPE][de] = relu(facts·wd0 + bd0)·wd1 + bd1 + wid.
    /// Ports net.rs::cards. The facts block of row 0 is the same for every
    /// row of the solve (the draft is fixed per game).
    fn build_cards(
        &self,
        arenas: &mut CudaSlice<f32>,
        tables: &CudaSlice<u8>,
        off: Offsets,
        _rows: usize,
        de: usize,
    ) -> Result<(), String> {
        let (dc,) = (self.weights.dims[8],);
        let cf = crate::units::CARD_FEATS;
        // The facts block sits in row 0 of the uploaded xpub.
        let facts = tview::<f32>(
            tables,
            off.leaf_xpub + crate::rebel::OFF_CARDS,
            cf * crate::rebel::NTYPE,
        );
        let (off_w, off_b, _) = self.woff();
        let wd0 = self.weights.w.slice(off_w[0].0..off_w[0].0 + cf * dc);
        let wd1 = self.weights.w.slice(off_w[1].0..off_w[1].0 + dc * de);
        let wid = self.weights.w.slice(off_w[2].0..off_w[2].0 + crate::units::N_UNITS * de);
        let bd0 = self.weights.b.slice(off_b[0].0..off_b[0].0 + dc);
        let bd1 = self.weights.b.slice(off_b[1].0..off_b[1].0 + de);
        let hid = self.stream.alloc_zeros::<f32>(crate::rebel::NTYPE * dc).map_err(|e| format!("{e:?}"))?;
        let tmp = self.stream.alloc_zeros::<f32>(crate::rebel::NTYPE * de).map_err(|e| format!("{e:?}"))?;
        let mut hid_m = hid.clone();
        gemm(&self.blas, 
            crate::rebel::NTYPE, dc, cf,
            &facts, cf, &wd0, dc, &mut hid_m, dc, 0.0,
        );
        let cfg = Self::cfg1(crate::rebel::NTYPE * dc);
        let mut b = self.stream.launch_builder(&self.f.relu_bias);
        b.arg(&mut hid_m);
        b.arg(&bd0);
        let __a2 = crate::rebel::NTYPE as i32;
        b.arg(&__a2);
        let __a3 = dc as i32;
        b.arg(&__a3);
        let _ = unsafe { b.launch(cfg) };
        let mut tmp_m = tmp.clone();
        gemm(&self.blas, 
            crate::rebel::NTYPE, de, dc,
            &hid_m, dc, &wd1, de, &mut tmp_m, de, 0.0,
        );
        let mut e = arenas.slice_mut(off.cz..off.cz + crate::rebel::NTYPE * de);
        let ids = tview::<u8>(tables, off.ids, crate::rebel::NTYPE);
        let cfg2 = Self::cfg1(crate::rebel::NTYPE * de);
        let mut b2 = self.stream.launch_builder(&self.f.cards_finish);
        b2.arg(&mut e);
        b2.arg(&tmp_m);
        b2.arg(&bd1);
        b2.arg(&wid);
        b2.arg(&ids);
        let __a4 = crate::rebel::NTYPE as i32;
        b2.arg(&__a4);
        let __a5 = de as i32;
        b2.arg(&__a5);
        let _ = unsafe { b2.launch(cfg2) };
        Ok(())
    }

    /// The trunk: assemble the per-row input from xpub + e, then
    /// h0 = relu(LN0(x·w0 + b0))·w1. Ports net.rs::assemble + trunk.
    fn build_trunk(
        &self,
        arenas: &mut CudaSlice<f32>,
        tables: &CudaSlice<u8>,
        off: Offsets,
        rows: usize,
        hd: usize,
        de: usize,
    ) -> Result<(), String> {
        let (h,) = (self.weights.dims[1],);
        let (hf, xd) = (4 + de,
            crate::board::N_HEXES * (crate::rebel::HEX_FACTS + de) + 2 * de + crate::rebel::LOOSE);
        let (off_w, off_b, _) = self.woff();
        let wpile = self.weights.w.slice(off_w[3].0..off_w[3].0 + (4 + de) * de);
        let bpile = self.weights.b.slice(off_b[2].0..off_b[2].0 + de);
        let w0 = self.weights.w.slice(off_w[4].0..off_w[4].0 + xd * h);
        let b0 = self.weights.b.slice(off_b[3].0..off_b[3].0 + h);
        let w1 = self.weights.w.slice(off_w[5].0..off_w[5].0 + h * hd);
        let ln0w = self.weights.ln.slice(0..h);
        let ln0b = self.weights.ln.slice(h..2 * h);
        let e = arenas.slice(off.cz..off.cz + crate::rebel::NTYPE * de);
        // pe = bpile + e·wpile[4:] (once per solve), then assemble.
        let pe = self.stream.alloc_zeros::<f32>(crate::rebel::NTYPE * de).map_err(|e| format!("{e:?}"))?;
        let mut pe_m = pe.clone();
        let cfg = Self::cfg1(crate::rebel::NTYPE * de);
        let mut b = self.stream.launch_builder(&self.f.pile_pe);
        b.arg(&mut pe_m);
        b.arg(&e);
        b.arg(&wpile);
        b.arg(&bpile);
        let __a6 = de as i32;
        b.arg(&__a6);
        let _ = unsafe { b.launch(cfg) };
        let xpub = tview::<f32>(tables, off.leaf_xpub, rows * self.weights.dims[0]);
        let x = self.stream.alloc_zeros::<f32>(rows * xd).map_err(|e| format!("{e:?}"))?;
        let mut x_m = x.clone();
        let cfg2 = Self::cfg1(rows);
        let mut b2 = self.stream.launch_builder(&self.f.assemble);
        b2.arg(&mut x_m);
        b2.arg(&xpub);
        b2.arg(&e);
        b2.arg(&pe_m);
        let __a7 = rows as i32;
        b2.arg(&__a7);
        let __a8 = self.weights.dims[0] as i32;
        b2.arg(&__a8);
        let __a9 = de as i32;
        b2.arg(&__a9);
        let _ = unsafe { b2.launch(cfg2) };
        // scratch = x·w0; LN0+ReLU in place; h0 = scratch·w1.
        let mut scratch = self.stream.alloc_zeros::<f32>(rows * h).map_err(|e| format!("{e:?}"))?;
        gemm(&self.blas, rows, h, xd, &x_m, xd, &w0, h, &mut scratch, h, 0.0);
        let cfg3 = Self::cfg1(rows);
        let mut b3 = self.stream.launch_builder(&self.f.ln_relu);
        b3.arg(&mut scratch);
        b3.arg(&b0);
        b3.arg(&ln0w);
        b3.arg(&ln0b);
        b3.arg(&x_m);
        let __a10 = 0i32;
        b3.arg(&__a10);
        let __a11 = rows as i32;
        b3.arg(&__a11);
        let __a12 = h as i32;
        b3.arg(&__a12);
        let _ = unsafe { b3.launch(cfg3) };
        let mut h0 = arenas.slice_mut(off.h0..off.h0 + rows * hd);
        gemm(&self.blas, rows, hd, h, &scratch, h, &w1, hd, &mut h0, hd, 0.0);
        let _ = hf;
        Ok(())
    }

    /// The config tower: z, g from cphi. Ports net.rs::embed.
    fn build_embed(
        &self,
        arenas: &mut CudaSlice<f32>,
        tables: &CudaSlice<u8>,
        off: Offsets,
        n: usize,
        dg: usize,
        rk: usize,
        de: usize,
    ) -> Result<(), String> {
        let (hf,) = (4 + de,);
        let (off_w, off_b, _) = self.woff();
        let wc = self.weights.w.slice(off_w[7].0..off_w[7].0 + hf * dg);
        let bc = self.weights.b.slice(off_b[5].0..off_b[5].0 + dg);
        let wh1 = self.weights.w.slice(off_w[8].0..off_w[8].0 + dg * dg);
        let bh1 = self.weights.b.slice(off_b[6].0..off_b[6].0 + dg);
        let wh2 = self.weights.w.slice(off_w[9].0..off_w[9].0 + dg * dg);
        let bh2 = self.weights.b.slice(off_b[7].0..off_b[7].0 + dg);
        let wg = self.weights.w.slice(off_w[10].0..off_w[10].0 + dg * (rk + 1));
        let bg = self.weights.b.slice(off_b[8].0..off_b[8].0 + rk + 1);
        let cphi = tview::<f32>(tables, off.cphi, n * crate::rebel::CFEAT);
        let e = arenas.slice(off.cz..off.cz + crate::rebel::NTYPE * de);
        let inp = self.stream.alloc_zeros::<f32>(n * crate::rebel::NSLOT * hf).map_err(|e| format!("{e:?}"))?;
        let slot = self.stream.alloc_zeros::<f32>(n * crate::rebel::NSLOT * dg).map_err(|e| format!("{e:?}"))?;
        let res = self.stream.alloc_zeros::<f32>(n * dg).map_err(|e| format!("{e:?}"))?;
        let mut inp_m = inp.clone();
        let cfg = Self::cfg1(n * crate::rebel::NSLOT);
        let mut b = self.stream.launch_builder(&self.f.holding_in);
        b.arg(&mut inp_m);
        b.arg(&cphi);
        b.arg(&e);
        let __a13 = n as i32;
        b.arg(&__a13);
        let __a14 = crate::rebel::CFEAT as i32;
        b.arg(&__a14);
        let __a15 = de as i32;
        b.arg(&__a15);
        let _ = unsafe { b.launch(cfg) };
        let mut slot_m = slot.clone();
        gemm(&self.blas, n * crate::rebel::NSLOT, dg, hf, &inp_m, hf, &wc, dg, &mut slot_m, dg, 0.0);
        let mut z = arenas.slice_mut(off.cz + crate::rebel::NTYPE * de..off.cz + crate::rebel::NTYPE * de + n * dg);
        let cfg2 = Self::cfg1(n);
        let mut b2 = self.stream.launch_builder(&self.f.slot_sum);
        b2.arg(&mut z);
        b2.arg(&slot_m);
        b2.arg(&bc);
        let __a16 = n as i32;
        b2.arg(&__a16);
        let __a17 = dg as i32;
        b2.arg(&__a17);
        let _ = unsafe { b2.launch(cfg2) };
        // Residual: z = z + relu(z·wh1 + bh1)·wh2 + bh2.
        let mut res_m = res.clone();
        gemm(&self.blas, n, dg, dg, &z, dg, &wh1, dg, &mut res_m, dg, 0.0);
        let cfg3 = Self::cfg1(n * dg);
        let mut b3 = self.stream.launch_builder(&self.f.relu_bias);
        b3.arg(&mut res_m);
        b3.arg(&bh1);
        let __a18 = n as i32;
        b3.arg(&__a18);
        let __a19 = dg as i32;
        b3.arg(&__a19);
        let _ = unsafe { b3.launch(cfg3) };
        let mut z2 = arenas.slice_mut(off.cz + crate::rebel::NTYPE * de..off.cz + crate::rebel::NTYPE * de + n * dg);
        gemm(&self.blas, n, dg, dg, &res_m, dg, &wh2, dg, &mut z2, dg, 1.0);
        let mut b4 = self.stream.launch_builder(&self.f.bias_add);
        b4.arg(&mut z2);
        b4.arg(&bh2);
        let __a20 = n as i32;
        b4.arg(&__a20);
        let __a21 = dg as i32;
        b4.arg(&__a21);
        let _ = unsafe { b4.launch(cfg3) };
        // g = z·wg + bg.
        let mut z3 = self.stream.alloc_zeros(n * dg).map_err(|e| format!("{e:?}"))?;
        {
            let z_src = arenas.slice(off.cz + crate::rebel::NTYPE * de..off.cz + crate::rebel::NTYPE * de + n * dg);
            let _ = self.stream.memcpy_dtod(&z_src, &mut z3);
        }
        let mut g = arenas.slice_mut(off.cg..off.cg + n * (rk + 1));
        gemm(&self.blas, n, rk + 1, dg, &z3, dg, &wg, rk + 1, &mut g, rk + 1, 0.0);
        let mut b5 = self.stream.launch_builder(&self.f.bias_add);
        b5.arg(&mut g);
        b5.arg(&bg);
        let __a22 = n as i32;
        b5.arg(&__a22);
        let __a23 = (rk + 1) as i32;
        b5.arg(&__a23);
        let _ = unsafe { b5.launch(cfg3) };
        Ok(())
    }

    /// The action towers: q(a) = relu([psi | e_pay]·wq + bq) for every
    /// decision node's actions. Ports net.rs::embed_actions.
    fn build_actions(
        &self,
        arenas: &mut CudaSlice<f32>,
        tables: &CudaSlice<u8>,
        off: Offsets,
        total_na: usize,
        rk: usize,
        de: usize,
    ) -> Result<(), String> {
        let afeat = crate::rebel::AFEAT;
        let af = afeat + de;
        let (off_w, off_b, _) = self.woff();
        let wq = self.weights.w.slice(off_w[12].0..off_w[12].0 + af * rk);
        let bq = self.weights.b.slice(off_b[10].0..off_b[10].0 + rk);
        let psi = tview::<f32>(tables, off.psi, total_na * afeat);
        let e = arenas.slice(off.cz..off.cz + crate::rebel::NTYPE * de);
        let inp = self.stream.alloc_zeros::<f32>(total_na * af).map_err(|e| format!("{e:?}"))?;
        let mut inp_m = inp.clone();
        let cfg = Self::cfg1(total_na * de);
        let mut b = self.stream.launch_builder(&self.f.action_in);
        b.arg(&mut inp_m);
        b.arg(&psi);
        b.arg(&e);
        let __a24 = total_na as i32;
        b.arg(&__a24);
        let __a25 = afeat as i32;
        b.arg(&__a25);
        let __a26 = de as i32;
        b.arg(&__a26);
        let _ = unsafe { b.launch(cfg) };
        let mut q = arenas.slice_mut(off.q..off.q + total_na * rk);
        gemm(&self.blas, total_na, rk, af, &inp_m, af, &wq, rk, &mut q, rk, 0.0);
        let cfg2 = Self::cfg1(total_na * rk);
        let mut b2 = self.stream.launch_builder(&self.f.relu_bias);
        b2.arg(&mut q);
        b2.arg(&bq);
        let __a27 = total_na as i32;
        b2.arg(&__a27);
        let __a28 = rk as i32;
        b2.arg(&__a28);
        let _ = unsafe { b2.launch(cfg2) };
        Ok(())
    }

    // ------------------------------------------------------------ lifecycle

    /// Send trip 1: the reference strategy (the final average) and the
    /// Phase-2 root values collected during the value stage.
    fn finalize_trip1(&mut self, s: usize) {
        let (tx, id, strategy, root_values) = {
            let sv = self.live[s].as_mut().unwrap();
            let strategy = d2h_arenas(&self.stream, &sv.arenas, sv.off.avg, sv.ncells);
            (sv.trip1.take(), sv.id, strategy, sv.root_values.clone())
        };
        if let Some(tx) = tx {
            let _ = tx.send(Ok(Trip1 { id, strategy, root_values }));
        }
    }

    /// Send trip 2: the carried beliefs at the exit leaf, one per kept
    /// snapshot (t = 0..T-1), then free the solve.
    fn send_trip2(&mut self, s: usize) {
        let (tx, id, raw, nsnaps, max_nc, nc_leaf) = {
            let sv = self.live[s].as_mut().unwrap();
            let buf = sv.beliefs.as_ref().expect("belief buffer");
            let mut raw = vec![0.0f32; buf.len()];
            let _ = self.stream.memcpy_dtoh(buf, &mut raw);
            (sv.trip2.take(), sv.id, raw, sv.nsnaps, sv.desc.max_nc as usize, sv.nc_leaf)
        };
        let ncarry = nsnaps.saturating_sub(1);
        let mut out = Vec::with_capacity(ncarry);
        for i in 0..ncarry {
            let base = (i * 2) * max_nc;
            let p0 = raw[base..base + nc_leaf[0]].to_vec();
            let p1 = raw[base + max_nc..base + max_nc + nc_leaf[1]].to_vec();
            out.push([p0, p1]);
        }
        if let Some(tx) = tx {
            let _ = tx.send(Ok(out));
        }
        self.free_solve(s);
        let _ = id;
    }

    fn free_solve(&mut self, s: usize) {
        self.live[s] = None;
        self.free.push(s);
    }

    /// The walk left the tree at `leaf`: allocate the belief buffer, set the
    /// carry stage. The reply is posted when every kept snapshot has been
    /// propagated to the leaf.
    fn trip2(&mut self, id: u64, leaf: u32, reply: mpsc::Sender<Result<Trip2, String>>) {
        for s in 0..CAP {
            let Some(sv) = &mut self.live[s] else { continue };
            if sv.id != id {
                continue;
            }
            let l = leaf as usize;
            let n0 = (sv.cfg_off_host[2 * l + 1] - sv.cfg_off_host[2 * l]) as usize;
            let n1 = (sv.cfg_off_host[2 * l + 2] - sv.cfg_off_host[2 * l + 1]) as usize;
            let max_nc = n0.max(n1);
            let ncarry = sv.nsnaps.saturating_sub(1);
            let buf = match self.stream.alloc_zeros::<f32>(ncarry * 2 * max_nc) {
                Ok(b) => b,
                Err(e) => {
                    let _ = reply.send(Err(format!("{e:?}")));
                    return;
                }
            };
            sv.leaf = l;
            sv.nc_leaf = [n0, n1];
            sv.desc.leaf = l as i32;
            sv.desc.max_nc = max_nc as i32;
            sv.beliefs = Some(buf);
            sv.trip2 = Some(reply);
            sv.stage = STAGE_CARRY;
            sv.step = 0;
            return;
        }
        let _ = reply.send(Err(format!("gpu: unknown solve id {id}")));
    }
}



/// D2D copy inside one solve's arena blob (the snapshot copies), via a
/// temporary device buffer (the views would otherwise borrow the blob both
/// ways at once).
fn d2d_arenas(stream: &Arc<CudaStream>, arenas: &mut CudaSlice<f32>,
              src_off: usize, dst_off: usize, n: usize) {
    let src = arenas.slice(src_off..src_off + n);
    let mut tmp = stream.alloc_zeros(n).expect("d2d tmp");
    let _ = stream.memcpy_dtod(&src, &mut tmp);
    let mut dst = arenas.slice_mut(dst_off..dst_off + n);
    let _ = stream.memcpy_dtod(&tmp, &mut dst);
}

/// D2H of a range of one solve's arena blob.
fn d2h_arenas(stream: &Arc<CudaStream>, arenas: &CudaSlice<f32>, off: usize, n: usize) -> Vec<f32> {
    let view = arenas.slice(off..off + n);
    let mut v = vec![0.0f32; n];
    let _ = stream.memcpy_dtoh(&view, &mut v);
    v
}

/// One row-major GEMM: C[m,n] = A[m,k] · B[k,n] (+ beta·C). All buffers
/// row-major [in, out] as stored; cuBLAS sees the transposes.
fn gemm<A: cudarc::driver::DevicePtr<f32>, B: cudarc::driver::DevicePtr<f32>,
        C: cudarc::driver::DevicePtrMut<f32>>(
    blas: &CudaBlas,
    m: usize, n: usize, k: usize,
    a: &A, lda: usize,
    b: &B, ldb: usize,
    c: &mut C, ldc: usize,
    beta: f32,
) {
    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0,
        lda: ldb as i32,
        ldb: lda as i32,
        beta,
        ldc: m as i32,
    };
    let _ = unsafe { blas.gemm(cfg, b, a, c) };
}

/// Typed device view into the solve's table blob (byte offsets).
fn tview<T>(tables: &CudaSlice<u8>, byte_off: usize, n: usize) -> CudaView<'_, T> {
    unsafe {
        tables
            .slice(byte_off..byte_off + n * std::mem::size_of::<T>())
            .transmute::<T>(n)
            .expect("aligned table view")
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Raw device pointer of a buffer, on the service's single stream.
fn device_ptr_of<T>(stream: &Arc<CudaStream>, buf: &CudaSlice<T>) -> *const T {
    let (p, _sync) = buf.device_ptr(stream);
    p as usize as *const T
}

/// Raw mutable device pointer of a buffer.
fn device_ptr_mut_of<T>(stream: &Arc<CudaStream>, buf: &mut CudaSlice<T>) -> *mut T {
    let (p, _sync) = buf.device_ptr_mut(stream);
    p as usize as *mut T
}
