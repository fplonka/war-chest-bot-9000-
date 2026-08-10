//! One contiguous FP32 wave on one CUDA stream.
//!
//! The first correctness lane intentionally has no resident solve state: a
//! wave uploads one immutable byte blob and owns one flat FP32 arena until its
//! single completion is copied out. The dispatcher, not a device state
//! machine, decides which solves share the wave.

use std::collections::HashMap;
use std::mem::{align_of, size_of, MaybeUninit};
use std::sync::{Arc, Once, OnceLock};
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::CudaBlas;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
use cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::{result, sys, CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc;

use crate::gpu::client::{CarryStore, SolveResult};
use crate::net::V3Layout;
use crate::rebel::{CFEAT, GPU_ROW_BYTES, NSLOT, NTYPE, PILE_COUNTS};
use crate::units::CARD_FEATS;

use super::wave::Wave;

const BLOCK: u32 = 256;
const GRAPH_CLASSES: usize = 4;
const N_TABLES: usize = 51;
const N_ARENAS: usize = 21;

#[repr(usize)]
enum Table {
    NodeKind,
    NodePlayer,
    NodeNc,
    NodeChildStart,
    NodeChild,
    LegalRowOf,
    LegalOff,
    LegalValue,
    DrawOff,
    DrawTo,
    DrawP,
    DrawRowOff,
    DrawRowStart,
    ReachBase,
    Soff,
    Voff,
    NodeParent,
    RevRowOf,
    RevStart,
    RevSrc,
    RevCell,
    RvdRowOf,
    RvdStart,
    RvdSrc,
    RvdP,
    RowNode,
    RowJob,
    RowCfgOff,
    RowCfg,
    RawRows,
    CardFeat,
    Ids,
    ConfigJob,
    Cphi,
    Roots,
    Carried,
    NodeUtility,
    ExitNodes,
    ExitCoff,
    Decision0,
    Decision1,
    ReachTask0,
    ReachTask1,
    ReachLevel0,
    ReachLevel1,
    BackTask0,
    BackTask1,
    BackLevel0,
    BackLevel1,
    Readout,
    Jobs,
}

#[repr(usize)]
#[derive(Clone, Copy)]
enum Arena {
    Reach,
    SnapReach,
    Vals,
    Regret,
    Cur,
    Sum,
    SnapStrat,
    E,
    Z,
    G,
    H0,
    Xb,
    H,
    H2,
    U,
    RootValues,
    Carry,
    Bx,
    Bh,
    Bh2,
    Bg,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JobDev {
    node0: u32,
    nodes: u32,
    row0: u32,
    rows: u32,
    nleaf: u32,
    config0: u32,
    ncfg: u32,
    cell0: u32,
    ncells: u32,
    reach0: u32,
    reach_len: u32,
    vals0: u32,
    vals_len: u32,
    root0: u32,
    root_n0: u32,
    root_n1: u32,
    carried0: u32,
    nroots: u32,
    root_value0: u32,
    exit0: u32,
    nexits: u32,
    exit_cfg0: u32,
    snapshot_configs: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WaveDev {
    table: *const u8,
    arena: *mut f32,
    toff: [u64; N_TABLES],
    aoff: [u64; N_ARENAS],
    jobs: i32,
    nodes: i32,
    rows: i32,
    nleaf: i32,
    ncfg: i32,
    cells: i32,
    reach_len: i32,
    vals_len: i32,
    exits: i32,
    snapshot_configs: i32,
    carry_snapshots: i32,
    nlevels: i32,
    decision_n: [i32; 2],
    reach_task_n: [i32; 2],
    back_task_n: [i32; 2],
    readout_n: i32,
}

unsafe impl cudarc::driver::DeviceRepr for WaveDev {}
unsafe impl cudarc::driver::ValidAsZeroBits for WaveDev {}
unsafe impl Send for WaveDev {}

#[repr(C)]
#[derive(Clone, Copy)]
struct WeightDev {
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
}

unsafe impl cudarc::driver::DeviceRepr for WeightDev {}
unsafe impl cudarc::driver::ValidAsZeroBits for WeightDev {}
unsafe impl Send for WeightDev {}

impl Default for WeightDev {
    fn default() -> Self {
        // SAFETY: this is a plain table of device pointers.
        unsafe { std::mem::zeroed() }
    }
}

struct WeightBank {
    layout: V3Layout,
    _w: CudaSlice<f32>,
    _w16: CudaSlice<u16>,
    _b: CudaSlice<f32>,
    _ln: CudaSlice<f32>,
    dev: CudaSlice<WeightDev>,
}

macro_rules! kernels {
    ($($name:ident),* $(,)?) => {
        struct Kernels { $($name: CudaFunction,)* }
        impl Kernels {
            fn load(module: &Arc<CudaModule>) -> Result<Self, String> {
                Ok(Self {
                    $($name: module.load_function(stringify!($name))
                        .map_err(|e| format!("kernel {}: {e:?}", stringify!($name)))?,)*
                })
            }
        }
    };
}

kernels! {
    pack_cards, bias_act, cards_finish, pile_pe, pack_piles, assemble,
    trunk_norm, holding_in, slot_sum, finish_zg,
    pack_head_static_f16,
    init_strategy, seed_reach, reach_sweep, seed_sum,
    belief_sums, belief_sums_f16, head_entry, head_entry_f16, head_act, readout, backprop_sweep,
    normalize_strategy, gather_carry, collect_root,
}

pub struct Executor {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: Kernels,
    dims: Vec<usize>,
    banks: HashMap<u64, WeightBank>,
    buffers: Option<DeviceBuffers>,
    graphs: Vec<GraphExec>,
    next_graph: usize,
    backprop_blocks: u32,
    reach_blocks: u32,
    sweep_block: u32,
}

struct GraphExec {
    raw: sys::CUgraphExec,
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        let raw = std::mem::replace(&mut self.raw, std::ptr::null_mut());
        if !raw.is_null() {
            let _ = unsafe { result::graph::exec_destroy(raw) };
        }
    }
}

struct CapturedGraph(sys::CUgraph);

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let raw = std::mem::replace(&mut self.0, std::ptr::null_mut());
        if !raw.is_null() {
            let _ = unsafe { result::graph::destroy(raw) };
        }
    }
}

impl Executor {
    pub fn new(
        device: usize,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Self, String> {
        let context =
            CudaContext::new(device).map_err(|e| format!("cuda device {device}: {e:?}"))?;
        // Raw pointers in WaveDev/WeightDev have one explicit stream ordering
        // protocol; per-slice event tracking would duplicate it at every node.
        unsafe { context.disable_event_tracking() };
        let stream = context
            .new_stream()
            .map_err(|e| format!("CUDA stream: {e:?}"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS: {e:?}"))?;
        let layout = V3Layout::new(&dims)?;
        validate_layout(&layout)?;
        let (major, minor) = (
            context
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(|e| format!("CUDA capability: {e:?}"))?,
            context
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(|e| format!("CUDA capability: {e:?}"))?,
        );
        let arch = format!("compute_{major}{minor}");
        let cuda_root = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".into());
        // cudarc 0.17's `use_fast_math` option only enables contraction; pass
        // NVRTC's actual umbrella flag so division, square root and denormal
        // handling use the native fast paths too. The production path is fast
        // math by default; the opt-out exists only for numerical diagnosis.
        let mut nvrtc_options = vec!["--generate-line-info".into(), "--use_fast_math".into()];
        if std::env::var_os("WARCHEST_GPU_PRECISE_MATH").is_some() {
            nvrtc_options.pop();
        }
        let source = format!(
            "{}\n{}",
            cuda_preamble(&layout),
            include_str!("wave_kernels.cu")
        );
        let ptx = nvrtc::compile_ptx_with_opts(
            &source,
            nvrtc::CompileOptions {
                arch: Some(Box::leak(arch.into_boxed_str())),
                include_paths: vec![
                    format!("{cuda_root}/include"),
                    format!("{cuda_root}/targets/x86_64-linux/include/cccl"),
                ],
                options: nvrtc_options,
                ..Default::default()
            },
        )
        .map_err(|e| format!("v5 NVRTC: {e:?}"))?;
        let module = context
            .load_module(ptx)
            .map_err(|e| format!("v5 CUDA module: {e:?}"))?;
        let kernels = Kernels::load(&module)?;
        let sms = context
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| format!("CUDA multiprocessor count: {e:?}"))? as u32;
        let sweep_blocks_per_sm = std::env::var("WARCHEST_SWEEP_BLOCKS_PER_SM")
            .ok()
            .and_then(|x| x.parse::<u32>().ok())
            .filter(|&x| x > 0);
        let kernel_cap = |name: &str, default: u32| {
            std::env::var(name)
                .ok()
                .and_then(|x| x.parse::<u32>().ok())
                .filter(|&x| x > 0)
                .or(sweep_blocks_per_sm)
                .unwrap_or(default)
        };
        let sweep_block = std::env::var("WARCHEST_SWEEP_BLOCK")
            .ok()
            .and_then(|x| x.parse::<u32>().ok())
            .filter(|&x| (32..=1024).contains(&x) && x % 32 == 0)
            .unwrap_or(BLOCK);
        let backprop_blocks_per_sm = kernels
            .backprop_sweep
            .occupancy_max_active_blocks_per_multiprocessor(sweep_block, 0, None)
            .map_err(|e| format!("backprop sweep occupancy: {e:?}"))?
            // The unrestricted six-block grids overpopulate cooperative level
            // barriers. On the heterogeneous 1,000-root tape, backprop 4 plus
            // reach 2 averaged 709/s versus 693/s at 3/3 and 642/s at 6/6.
            .min(kernel_cap("WARCHEST_BACKPROP_BLOCKS_PER_SM", 4));
        let backprop_blocks = sms.saturating_mul(backprop_blocks_per_sm).max(1);
        let reach_blocks_per_sm = kernels
            .reach_sweep
            .occupancy_max_active_blocks_per_multiprocessor(sweep_block, 0, None)
            .map_err(|e| format!("reach sweep occupancy: {e:?}"))?
            .min(kernel_cap("WARCHEST_REACH_BLOCKS_PER_SM", 2));
        let reach_blocks = sms.saturating_mul(reach_blocks_per_sm).max(1);
        if std::env::var_os("WARCHEST_GPU_PROFILE").is_some() {
            eprintln!(
                "v5_sweeps block={sweep_block} backprop_blocks_per_sm={backprop_blocks_per_sm} reach_blocks_per_sm={reach_blocks_per_sm}"
            );
        }
        let initial = WeightBank::upload(&stream, &dims, w, b, ln)?;
        let mut banks = HashMap::new();
        banks.insert(0, initial);
        Ok(Self {
            stream,
            blas,
            kernels,
            dims,
            banks,
            buffers: None,
            graphs: Vec::with_capacity(GRAPH_CLASSES),
            next_graph: 0,
            backprop_blocks,
            reach_blocks,
            sweep_block,
        })
    }

    pub fn publish(
        &mut self,
        version: u64,
        dims: Vec<usize>,
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<(), String> {
        if dims != self.dims {
            return Err(format!(
                "GPU weight shape changed from {:?} to {:?}; restart executors",
                self.dims, dims
            ));
        }
        let bank = WeightBank::upload(&self.stream, &dims, w, b, ln)?;
        self.banks.insert(version, bank);
        Ok(())
    }

    /// Release wave-sized allocations while retaining the lane's CUDA
    /// context, kernels, cuBLAS handle, and immutable weight banks.
    pub fn trim(&mut self) -> Result<(), String> {
        self.stream
            .synchronize()
            .map_err(|e| format!("synchronize before GPU trim: {e:?}"))?;
        self.buffers = None;
        self.graphs.clear();
        self.next_graph = 0;
        Ok(())
    }

    /// Retire immutable banks once this lane has no queued wave stamped with
    /// their version. Lane command order proves that no unseen older submit can
    /// arrive after the publication that introduced a newer version.
    pub fn retain_weight_versions(&mut self, keep: &[u64]) {
        self.banks.retain(|version, _| keep.contains(version));
    }

    pub fn solve(&mut self, wave: Wave, version: u64) -> Result<Vec<SolveResult>, String> {
        let profile = std::env::var_os("WARCHEST_GPU_PROFILE").is_some();
        let started = Instant::now();
        let profile_jobs = wave.jobs.len();
        let profile_rows = wave.row_node.len();
        let profile_cells = wave.legal_value.len();
        if wave.meta.warm > 0.0 {
            return Err("v5 FP32 baseline does not yet implement policy-head warm starts".into());
        }
        if wave.jobs.iter().any(|j| j.rows.len() != j.network_leaves) {
            return Err("wave has policy rows despite warm=0".into());
        }
        let bank = self
            .banks
            .get(&version)
            .ok_or_else(|| format!("GPU weight version {version} is unavailable"))?;
        let device = DeviceWave::upload(&self.stream, &wave, &bank.layout, self.buffers.take())?;
        let uploaded = Instant::now();
        // Arbitrary live waves often make cuBLAS choose a different captured
        // topology, in which case graph update cannot reuse the executable and
        // capture+instantiation is pure overhead. Keep a direct-stream A/B path
        // while tuning that crossover; it executes the identical kernels and
        // GEMMs in the identical order.
        let direct = std::env::var_os("WARCHEST_DIRECT").is_some();
        if !direct {
            self.stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| format!("begin v5 CUDA Graph: {e:?}"))?;
        }
        let queued = (|| -> Result<(), String> {
            self.build_towers(&device, bank)?;
            self.initialise(&device, bank, &wave)?;

            if wave.meta.iters == 0 {
                self.materialize(&device, bank, &wave, [false, false])?;
            }
            for t in 0..wave.meta.iters {
                let p = t & 1;
                self.run_head(&device, bank, t == 0, p)?;
                self.launch_readout(&device, bank, p)?;
                let m = (t / 2 + 1) as f32;
                let cfr = wave.meta.cfr;
                let (da, db, ds) = (
                    factor(m, cfr.alpha),
                    factor(m, cfr.beta),
                    (m / (m + 1.0)).powf(cfr.gamma),
                );
                self.launch_backprop_sweep(&device, bank, p, false, da, db, ds, cfr.predict)?;
                self.launch_reach_sweep(&device, bank, p, false, false, true)?;

                let completed = t + 1;
                if let Some(s) = wave.meta.snap_iters.iter().position(|&x| x == completed) {
                    if wave.meta.snapshots || completed == wave.meta.iters {
                        self.materialize(&device, bank, &wave, [true, completed >= 2])?;
                        if wave.meta.snapshots && s + 1 < wave.meta.snap_iters.len() {
                            self.snapshot_carry(&device, bank, &wave, s as i32)?;
                        }
                    }
                }
            }

            let max_roots = wave.jobs.iter().map(|j| j.nroots).max().unwrap_or(0);
            for root in 0..max_roots {
                launch!(
                    self,
                    device,
                    bank,
                    seed_reach,
                    wave.jobs.len() as u32,
                    0i32,
                    root as i32
                )?;
                self.full_reach(&device, bank, false, true)?;
                self.run_head(&device, bank, true, 0)?;
                for p in 0..2 {
                    self.launch_readout(&device, bank, p)?;
                    self.launch_backprop_sweep(&device, bank, p, true, 0.0, 0.0, 0.0, 0.0)?;
                    launch!(
                        self,
                        device,
                        bank,
                        collect_root,
                        wave.jobs.len() as u32,
                        root as i32,
                        p as i32
                    )?;
                }
            }
            Ok(())
        })();
        let graph = if direct {
            queued?;
            None
        } else {
            let graph = unsafe { result::stream::end_capture(self.stream.cu_stream()) }
                .map_err(|e| format!("end v5 CUDA Graph capture: {e:?}"))?;
            queued?;
            if graph.is_null() {
                return Err("v5 CUDA Graph capture produced no graph".into());
            }
            let graph = CapturedGraph(graph);
            let (slot, reused) = update_graph(&mut self.graphs, &mut self.next_graph, graph.0)?;
            Some((slot, reused))
        };
        let captured = Instant::now();
        if let Some((slot, _)) = graph {
            unsafe { result::graph::launch(self.graphs[slot].raw, self.stream.cu_stream()) }
                .map_err(|e| format!("launch v5 CUDA Graph: {e:?}"))?;
        }

        let strategy = copy_arena(
            &self.stream,
            &device,
            Arena::SnapStrat,
            0,
            wave.legal_value.len(),
        )?;
        let root_total = wave.jobs.last().map_or(0, |j| j.root_values.end);
        let root_values = copy_arena(&self.stream, &device, Arena::RootValues, 0, root_total)?;
        let carry_n = device.host.carry_snapshots as usize * wave.snapshot_configs;
        let carries = copy_arena_f16(&self.stream, &device, Arena::Carry, carry_n)?;
        let queued_output = Instant::now();
        self.stream
            .synchronize()
            .map_err(|e| format!("GPU wave completion: {e:?}"))?;
        let completed = Instant::now();
        self.buffers = Some(device.into_buffers());
        let result = unpack(wave, strategy, root_values, carries, version);
        if profile {
            let unpacked = Instant::now();
            let graph_reused = graph.is_some_and(|(_, reused)| reused);
            eprintln!(
                "v5_device jobs={profile_jobs} rows={profile_rows} cells={profile_cells} direct={direct} graph_reused={graph_reused} upload_ms={:.2} capture_ms={:.2} queue_ms={:.2} gpu_ms={:.2} unpack_ms={:.2} total_ms={:.2}",
                1e3 * (uploaded - started).as_secs_f64(),
                1e3 * (captured - uploaded).as_secs_f64(),
                1e3 * (queued_output - captured).as_secs_f64(),
                1e3 * (completed - queued_output).as_secs_f64(),
                1e3 * (unpacked - completed).as_secs_f64(),
                1e3 * (unpacked - started).as_secs_f64(),
            );
        }
        result
    }

    fn build_towers(&self, d: &DeviceWave, bank: &WeightBank) -> Result<(), String> {
        let l = &bank.layout;
        let rows = d.host.rows as usize;
        let cfgs = d.host.ncfg as usize;
        let jobs = d.host.jobs as usize;

        launch!(
            self,
            d,
            bank,
            pack_cards,
            threads_usize(jobs * NTYPE * CARD_FEATS)
        )?;
        let mut src = d.ptr(Arena::Bg);
        let mut which = 0usize;
        for (k, layer) in l.card.iter().enumerate() {
            let dst = d.ptr_mut(if which == 0 { Arena::Bh } else { Arena::Bh2 });
            gemm(
                &self.blas,
                jobs * NTYPE,
                layer.o,
                layer.i,
                src,
                layer.i,
                bank.w_ptr(layer.w),
                layer.o,
                dst,
                layer.o,
                0.0,
            )?;
            if k + 1 < l.card.len() {
                launch!(
                    self,
                    d,
                    bank,
                    bias_act,
                    threads_usize(jobs * NTYPE * layer.o),
                    0i32,
                    k as i32,
                    (jobs * NTYPE) as i32,
                    which as i32
                )?;
                src = dst;
                which ^= 1;
            }
        }
        launch!(
            self,
            d,
            bank,
            cards_finish,
            threads_usize(jobs * NTYPE * l.de),
            which as i32
        )?;

        launch!(self, d, bank, pile_pe, threads_usize(jobs * NTYPE * l.de))?;
        launch!(
            self,
            d,
            bank,
            pack_piles,
            threads_usize(rows * NTYPE * PILE_COUNTS)
        )?;
        gemm(
            &self.blas,
            rows * NTYPE,
            l.de,
            PILE_COUNTS,
            d.ptr(Arena::Bg),
            PILE_COUNTS,
            bank.w_ptr(l.pile.w),
            l.de,
            d.ptr_mut(Arena::Bh),
            l.de,
            0.0,
        )?;
        launch!(self, d, bank, assemble, rows as u32)?;

        let mut src = d.ptr(Arena::Bx);
        let mut which = 0usize;
        for (k, layer) in l.pub_lin.iter().enumerate() {
            let dst = d.ptr_mut(if which == 0 { Arena::Bh } else { Arena::Bh2 });
            gemm(
                &self.blas,
                rows,
                layer.o,
                layer.i,
                src,
                layer.i,
                bank.w_ptr(layer.w),
                layer.o,
                dst,
                layer.o,
                0.0,
            )?;
            launch!(
                self,
                d,
                bank,
                trunk_norm,
                warps(rows),
                k as i32,
                rows as i32,
                which as i32
            )?;
            src = dst;
            which ^= 1;
        }
        if fast_gemm() && l.hmlp.is_empty() {
            // The public residual is fixed for all CFR iterations. Produce it
            // once in the idle tower scratch, fold its bias, and retain it as
            // FP16; the dynamic residual and LayerNorm reduction remain FP32.
            let dst = d.ptr_mut(if which == 0 { Arena::Bh } else { Arena::Bh2 });
            gemm(
                &self.blas,
                rows,
                l.head_in,
                l.pub_out.i,
                src,
                l.pub_out.i,
                bank.w_ptr(l.pub_out.w),
                l.head_in,
                dst,
                l.head_in,
                0.0,
            )?;
            launch!(
                self,
                d,
                bank,
                pack_head_static_f16,
                threads_usize(rows * l.head_in),
                which as i32
            )?;
        } else {
            gemm(
                &self.blas,
                rows,
                l.head_in,
                l.pub_out.i,
                src,
                l.pub_out.i,
                bank.w_ptr(l.pub_out.w),
                l.head_in,
                d.ptr_mut(Arena::H0),
                l.head_in,
                0.0,
            )?;
        }
        launch!(self, d, bank, holding_in, threads_usize(cfgs * NSLOT))?;
        let mut src = d.ptr(Arena::Bg);
        let mut which = 0usize;
        for (k, layer) in l.slot.iter().enumerate() {
            let dst = d.ptr_mut(if which == 0 { Arena::Bh } else { Arena::Bh2 });
            gemm(
                &self.blas,
                cfgs * NSLOT,
                layer.o,
                layer.i,
                src,
                layer.i,
                bank.w_ptr(layer.w),
                layer.o,
                dst,
                layer.o,
                0.0,
            )?;
            launch!(
                self,
                d,
                bank,
                bias_act,
                threads_usize(cfgs * NSLOT * layer.o),
                2i32,
                k as i32,
                (cfgs * NSLOT) as i32,
                which as i32
            )?;
            src = dst;
            which ^= 1;
        }
        let dst = d.ptr_mut(if which == 0 { Arena::Bh } else { Arena::Bh2 });
        gemm(
            &self.blas,
            cfgs * NSLOT,
            l.dg,
            l.slot_out.i,
            src,
            l.slot_out.i,
            bank.w_ptr(l.slot_out.w),
            l.dg,
            dst,
            l.dg,
            0.0,
        )?;
        launch!(self, d, bank, slot_sum, warps(cfgs), which as i32)?;
        let zwhich = which ^ 1;
        for (k, (a, b)) in l.res.iter().enumerate() {
            let z = d.ptr_mut(if zwhich == 0 { Arena::Bh } else { Arena::Bh2 });
            let tmp = d.ptr_mut(if zwhich == 0 { Arena::Bh2 } else { Arena::Bh });
            gemm(
                &self.blas,
                cfgs,
                l.dg,
                l.dg,
                z,
                l.dg,
                bank.w_ptr(a.w),
                l.dg,
                tmp,
                l.dg,
                0.0,
            )?;
            launch!(
                self,
                d,
                bank,
                bias_act,
                threads_usize(cfgs * l.dg),
                3i32,
                k as i32,
                cfgs as i32,
                (zwhich ^ 1) as i32
            )?;
            gemm(
                &self.blas,
                cfgs,
                l.dg,
                l.dg,
                tmp,
                l.dg,
                bank.w_ptr(b.w),
                l.dg,
                z,
                l.dg,
                1.0,
            )?;
            launch!(
                self,
                d,
                bank,
                bias_act,
                threads_usize(cfgs * l.dg),
                4i32,
                k as i32,
                cfgs as i32,
                zwhich as i32
            )?;
        }
        let z = d.ptr(if zwhich == 0 { Arena::Bh } else { Arena::Bh2 });
        gemm(
            &self.blas,
            cfgs,
            l.rank + 1,
            l.dg,
            z,
            l.dg,
            bank.w_ptr(l.wg.w),
            l.rank + 1,
            d.ptr_mut(Arena::G),
            l.rank + 1,
            0.0,
        )?;
        launch!(
            self,
            d,
            bank,
            finish_zg,
            threads_usize(cfgs * (l.dg + l.rank + 1)),
            zwhich as i32
        )?;
        Ok(())
    }

    fn initialise(&self, d: &DeviceWave, bank: &WeightBank, w: &Wave) -> Result<(), String> {
        for p in 0..2 {
            launch!(
                self,
                d,
                bank,
                init_strategy,
                threads(w.decision[p].len() as i32),
                p as i32
            )?;
        }
        launch!(self, d, bank, seed_reach, w.jobs.len() as u32, 0i32, -1i32)?;
        self.full_reach(d, bank, false, false)?;
        for p in 0..2 {
            launch!(
                self,
                d,
                bank,
                seed_sum,
                threads(w.decision[p].len() as i32),
                p as i32
            )?;
        }
        if d.host.carry_snapshots > 0 {
            launch!(
                self,
                d,
                bank,
                gather_carry,
                warps(w.exit_nodes.len()),
                0i32,
                0i32
            )?;
        }
        Ok(())
    }

    fn full_reach(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        snap: bool,
        strat_snap: bool,
    ) -> Result<(), String> {
        if d.host.reach_task_n.iter().all(|&n| n == 0) {
            return Ok(());
        }
        let f = self.kernels.reach_sweep.clone();
        let mut args = self.stream.launch_builder(&f);
        let (player, snap, strat_snap, accumulate) = (-1i32, snap as i32, strat_snap as i32, 0i32);
        args.arg(&d.buffers.dev)
            .arg(&bank.dev)
            .arg(&player)
            .arg(&snap)
            .arg(&strat_snap)
            .arg(&accumulate);
        let cfg = LaunchConfig {
            grid_dim: (self.reach_blocks, 1, 1),
            block_dim: (self.sweep_block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch_cooperative(cfg) }
            .map_err(|e| format!("CUDA paired reach sweep launch: {e:?}"))?;
        Ok(())
    }

    fn materialize(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        w: &Wave,
        touched: [bool; 2],
    ) -> Result<(), String> {
        for p in 0..2 {
            launch!(
                self,
                d,
                bank,
                normalize_strategy,
                threads(w.decision[p].len() as i32),
                p as i32,
                touched[p] as i32
            )?;
        }
        Ok(())
    }

    fn snapshot_carry(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        w: &Wave,
        slot: i32,
    ) -> Result<(), String> {
        launch!(self, d, bank, seed_reach, w.jobs.len() as u32, 1i32, -1i32)?;
        self.full_reach(d, bank, true, true)?;
        launch!(
            self,
            d,
            bank,
            gather_carry,
            warps(w.exit_nodes.len()),
            slot,
            1i32
        )?;
        Ok(())
    }

    fn run_head(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        both: bool,
        traverser: usize,
    ) -> Result<(), String> {
        let l = &bank.layout;
        let rows = d.host.rows as usize;
        // Production has no extra head MLP. Retain the dynamic GEMM output and
        // static public residual in FP16, then expand them to form the residual
        // sum and all LayerNorm reductions in FP32. The following tensor-core
        // GEMM consumes the post-ReLU value directly as FP16 as well.
        if fast_gemm() && l.hmlp.is_empty() {
            launch!(
                self,
                d,
                bank,
                belief_sums_f16,
                warps(d.host.nleaf as usize),
                traverser as i32,
                both as i32
            )?;
            gemm_f16_to_f16(
                &self.blas,
                rows,
                l.head_in,
                2 * l.dg,
                d.ptr(Arena::Xb).cast(),
                2 * l.dg,
                bank.w16_ptr(l.wb),
                l.head_in,
                d.ptr_mut(Arena::H).cast(),
                l.head_in,
            )?;
            launch!(self, d, bank, head_entry_f16, warps(rows))?;
            gemm_f16(
                &self.blas,
                rows,
                l.rank,
                l.head_in,
                d.ptr(Arena::H2).cast(),
                l.head_in,
                bank.w16_ptr(l.wu.w),
                l.rank,
                d.ptr_mut(Arena::U),
                l.rank,
            )?;
            return Ok(());
        }
        launch!(
            self,
            d,
            bank,
            belief_sums,
            warps(d.host.nleaf as usize),
            traverser as i32,
            both as i32
        )?;
        gemm(
            &self.blas,
            rows,
            l.head_in,
            2 * l.dg,
            d.ptr(Arena::Xb),
            2 * l.dg,
            bank.w_ptr(l.wb),
            l.head_in,
            d.ptr_mut(Arena::H),
            h_stride(l),
            0.0,
        )?;
        launch!(self, d, bank, head_entry, warps(rows))?;

        let mut src = d.ptr(Arena::H);
        let mut src_width = l.head_in;
        let mut which = 0usize;
        for (level, layer) in l.hmlp.iter().enumerate() {
            let next = which ^ 1;
            let dst = d.ptr_mut(if next == 0 { Arena::H } else { Arena::H2 });
            gemm(
                &self.blas,
                rows,
                layer.o,
                src_width,
                src,
                h_stride(l),
                bank.w_ptr(layer.w),
                layer.o,
                dst,
                h_stride(l),
                0.0,
            )?;
            launch!(
                self,
                d,
                bank,
                head_act,
                threads_usize(rows * layer.o),
                level as i32,
                rows as i32,
                next as i32
            )?;
            src = dst;
            src_width = layer.o;
            which = next;
        }
        gemm(
            &self.blas,
            rows,
            l.rank,
            src_width,
            src,
            h_stride(l),
            bank.w_ptr(l.wu.w),
            l.rank,
            d.ptr_mut(Arena::U),
            l.rank,
            0.0,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_backprop_sweep(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        player: usize,
        fixed: bool,
        da: f32,
        db: f32,
        ds: f32,
        predict: f32,
    ) -> Result<(), String> {
        if d.host.back_task_n[player] == 0 {
            return Ok(());
        }
        let f = self.kernels.backprop_sweep.clone();
        let mut args = self.stream.launch_builder(&f);
        let player = player as i32;
        let mode = fixed as i32;
        args.arg(&d.buffers.dev)
            .arg(&bank.dev)
            .arg(&player)
            .arg(&mode)
            .arg(&da)
            .arg(&db)
            .arg(&ds)
            .arg(&predict);
        let cfg = LaunchConfig {
            grid_dim: (self.backprop_blocks, 1, 1),
            block_dim: (self.sweep_block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch_cooperative(cfg) }
            .map_err(|e| format!("CUDA backprop sweep launch: {e:?}"))?;
        Ok(())
    }

    fn launch_reach_sweep(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        player: usize,
        snap: bool,
        strat_snap: bool,
        accumulate: bool,
    ) -> Result<(), String> {
        if d.host.reach_task_n[player] == 0 {
            return Ok(());
        }
        let f = self.kernels.reach_sweep.clone();
        let mut args = self.stream.launch_builder(&f);
        let player = player as i32;
        let snap = snap as i32;
        let strat_snap = strat_snap as i32;
        let accumulate = accumulate as i32;
        args.arg(&d.buffers.dev)
            .arg(&bank.dev)
            .arg(&player)
            .arg(&snap)
            .arg(&strat_snap)
            .arg(&accumulate);
        let cfg = LaunchConfig {
            grid_dim: (self.reach_blocks, 1, 1),
            block_dim: (self.sweep_block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch_cooperative(cfg) }
            .map_err(|e| format!("CUDA reach sweep launch: {e:?}"))?;
        Ok(())
    }

    fn launch_readout(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        player: usize,
    ) -> Result<(), String> {
        launch!(
            self,
            d,
            bank,
            readout,
            warps(d.host.readout_n as usize),
            player as i32
        )
    }
}

fn update_graph(
    slots: &mut Vec<GraphExec>,
    next: &mut usize,
    graph: sys::CUgraph,
) -> Result<(usize, bool), String> {
    for (at, exec) in slots.iter().enumerate() {
        let mut info = MaybeUninit::<sys::CUgraphExecUpdateResultInfo>::zeroed();
        let call = unsafe { sys::cuGraphExecUpdate_v2(exec.raw, graph, info.as_mut_ptr()) };
        let status = call.result();
        let info = unsafe { info.assume_init() };
        if let Err(error) = status {
            if std::env::var_os("WARCHEST_GRAPH_PROFILE").is_some() {
                eprintln!(
                    "v5_graph_update slot={at} call=Err({error:?}) result={:?}",
                    info.result
                );
            }
            continue;
        }
        if std::env::var_os("WARCHEST_GRAPH_PROFILE").is_some() {
            eprintln!(
                "v5_graph_update slot={at} call={status:?} result={:?}",
                info.result
            );
        }
        if info.result == sys::CUgraphExecUpdateResult_enum::CU_GRAPH_EXEC_UPDATE_SUCCESS {
            return Ok((at, true));
        }
    }
    let raw =
        unsafe { result::graph::instantiate(graph, CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) }
            .map_err(|e| format!("instantiate v5 CUDA Graph: {e:?}"))?;
    let at = if slots.len() < GRAPH_CLASSES {
        slots.push(GraphExec { raw });
        slots.len() - 1
    } else {
        let at = *next % GRAPH_CLASSES;
        slots[at] = GraphExec { raw };
        *next = next.wrapping_add(1);
        at
    };
    Ok((at, false))
}

impl WeightBank {
    fn upload(
        stream: &Arc<CudaStream>,
        dims: &[usize],
        w: Vec<f32>,
        b: Vec<f32>,
        ln: Vec<f32>,
    ) -> Result<Self, String> {
        let layout = V3Layout::new(dims)?;
        validate_layout(&layout)?;
        if w.len() != layout.w_len || b.len() != layout.b_len || ln.len() != layout.ln_len {
            return Err(format!(
                "GPU weight sizes {}/{}/{} do not match {:?} ({}/{}/{})",
                w.len(),
                b.len(),
                ln.len(),
                dims,
                layout.w_len,
                layout.b_len,
                layout.ln_len
            ));
        }
        let w16: Vec<u16> = w.iter().map(|&x| crate::selfplay::f32_to_f16(x)).collect();
        let wb = htod(stream, &w)?;
        let wb16 = htod(stream, &w16)?;
        let bb = htod(stream, &b)?;
        let lb = htod(stream, &ln)?;
        let (wp, bp, lp) = (ptr(stream, &wb), ptr(stream, &bb), ptr(stream, &lb));
        let mut d = WeightDev::default();
        let wa = |x| unsafe { wp.add(x) };
        let ba = |x| unsafe { bp.add(x) };
        let la = |x| unsafe { lp.add(x) };
        for (k, s) in layout.card.iter().enumerate() {
            d.card_w[k] = wa(s.w);
            d.card_b[k] = ba(s.b);
        }
        d.wid = wa(layout.wid);
        d.pile_w = wa(layout.pile.w);
        d.pile_b = ba(layout.pile.b);
        for (k, s) in layout.pub_lin.iter().enumerate() {
            d.pub_w[k] = wa(s.w);
            d.pub_b[k] = ba(s.b);
            d.pub_lnw[k] = la(layout.pub_ln[k].0);
            d.pub_lnb[k] = la(layout.pub_ln[k].1);
        }
        d.pub_out_w = wa(layout.pub_out.w);
        d.pub_out_b = ba(layout.pub_out.b);
        d.wb = wa(layout.wb);
        d.ln1w = la(layout.ln1.0);
        d.ln1b = la(layout.ln1.1);
        for (k, s) in layout.hmlp.iter().enumerate() {
            d.hmlp_w[k] = wa(s.w);
            d.hmlp_b[k] = ba(s.b);
        }
        d.wu_w = wa(layout.wu.w);
        d.wu_b = ba(layout.wu.b);
        for (k, s) in layout.slot.iter().enumerate() {
            d.slot_w[k] = wa(s.w);
            d.slot_b[k] = ba(s.b);
        }
        d.slot_out_w = wa(layout.slot_out.w);
        d.slot_out_b = ba(layout.slot_out.b);
        for (k, (a, b)) in layout.res.iter().enumerate() {
            d.res_aw[k] = wa(a.w);
            d.res_ab[k] = ba(a.b);
            d.res_bw[k] = wa(b.w);
            d.res_bb[k] = ba(b.b);
        }
        d.wg_w = wa(layout.wg.w);
        d.wg_b = ba(layout.wg.b);
        let dev = htod(stream, &[d])?;
        Ok(Self {
            layout,
            _w: wb,
            _w16: wb16,
            _b: bb,
            _ln: lb,
            dev,
        })
    }

    fn w_ptr(&self, off: usize) -> *const f32 {
        unsafe { ptr(self.dev.stream(), &self._w).add(off) }
    }

    fn w16_ptr(&self, off: usize) -> *const u16 {
        unsafe { ptr(self.dev.stream(), &self._w16).add(off) }
    }
}

struct DeviceBuffers {
    tables: CudaSlice<u8>,
    arena: CudaSlice<f32>,
    dev: CudaSlice<WaveDev>,
}

struct DeviceWave {
    buffers: DeviceBuffers,
    _jobs: Vec<JobDev>,
    host: WaveDev,
}

impl DeviceWave {
    fn upload(
        stream: &Arc<CudaStream>,
        w: &Wave,
        l: &V3Layout,
        reuse: Option<DeviceBuffers>,
    ) -> Result<Self, String> {
        let jobs = job_devices(w)?;
        let (toff, table_len) = table_layout(w, &jobs)?;
        let (aoff, arena_len) = arena_layout(w, l)?;
        // Before a multi-GiB growth, drop both old allocations so neither is
        // live while the other grows. Once a lane has paid for a whale-sized
        // pair, retain it for later waves instead of reallocating every whale.
        let table_need = table_len.max(1);
        let arena_need = arena_len.max(1);
        let reservation = table_need
            .checked_next_power_of_two()
            .unwrap_or(table_need)
            .saturating_add(
                arena_need
                    .checked_next_power_of_two()
                    .unwrap_or(arena_need)
                    .saturating_mul(size_of::<f32>()),
            );
        let reuse = reuse.filter(|x| {
            reservation < (4usize << 30)
                || (x.tables.len() >= table_need && x.arena.len() >= arena_need)
        });
        let (tables, arena, dev) = match reuse {
            Some(x) => (Some(x.tables), Some(x.arena), Some(x.dev)),
            None => (None, None, None),
        };
        let mut tables = grow_buffer(stream, tables, table_need, "wave tables")?;
        let table_blob = pack_tables(w, &jobs, &toff, table_len)?;
        stream
            .memcpy_htod(&table_blob, &mut tables.slice_mut(..table_len))
            .map_err(|e| format!("wave table H2D: {e:?}"))?;
        let mut arena = grow_buffer(stream, arena, arena_need, "wave arena")?;
        stream
            .memset_zeros(&mut arena.slice_mut(..arena_need))
            .map_err(|e| format!("wave arena zero: {e:?}"))?;
        let carry_snapshots = if w.meta.snapshots {
            w.meta.snap_iters.len().saturating_sub(1)
        } else {
            0
        };
        let host = WaveDev {
            table: ptr_mut(stream, &mut tables),
            arena: ptr_mut(stream, &mut arena),
            toff,
            aoff,
            jobs: i32n(w.jobs.len(), "jobs")?,
            nodes: i32n(w.node_kind.len(), "nodes")?,
            rows: i32n(w.row_node.len(), "rows")?,
            nleaf: i32n(
                w.jobs.iter().map(|j| j.network_leaves).sum(),
                "network leaves",
            )?,
            ncfg: i32n(w.config_job.len(), "configs")?,
            cells: i32n(w.legal_value.len(), "legal cells")?,
            reach_len: i32n(w.reach_len, "reach")?,
            vals_len: i32n(*w.voff.last().unwrap_or(&0) as usize, "values")?,
            exits: i32n(w.exit_nodes.len(), "exits")?,
            snapshot_configs: i32n(w.snapshot_configs, "snapshot configs")?,
            carry_snapshots: i32n(carry_snapshots, "carry snapshots")?,
            nlevels: i32n(w.work.levels, "levels")?,
            decision_n: [
                i32n(w.decision[0].len(), "p0 decisions")?,
                i32n(w.decision[1].len(), "p1 decisions")?,
            ],
            reach_task_n: [
                i32n(w.reach_task[0].len(), "p0 reach")?,
                i32n(w.reach_task[1].len(), "p1 reach")?,
            ],
            back_task_n: [
                i32n(w.back_task[0].len(), "p0 back")?,
                i32n(w.back_task[1].len(), "p1 back")?,
            ],
            readout_n: i32n(w.readout.len(), "readouts")?,
        };
        let dev = match dev {
            Some(mut dev) => {
                stream
                    .memcpy_htod(&[host], &mut dev)
                    .map_err(|e| format!("wave header H2D: {e:?}"))?;
                dev
            }
            None => htod(stream, &[host])?,
        };
        if std::env::var_os("WARCHEST_GPU_DEBUG").is_some() {
            eprintln!(
                "v5 wave jobs={} rows={} cfgs={} cells={} arena={} floats offsets={:?} arena_ptr={:p} dev_ptr={:p}",
                host.jobs,
                host.rows,
                host.ncfg,
                host.cells,
                arena_len,
                host.aoff,
                host.arena,
                ptr(stream, &dev),
            );
        }
        Ok(Self {
            buffers: DeviceBuffers { tables, arena, dev },
            _jobs: jobs,
            host,
        })
    }

    fn into_buffers(self) -> DeviceBuffers {
        self.buffers
    }

    fn ptr(&self, a: Arena) -> *const f32 {
        unsafe {
            ptr(self.buffers.dev.stream(), &self.buffers.arena)
                .add(self.host.aoff[a as usize] as usize)
        }
    }

    fn ptr_mut(&self, a: Arena) -> *mut f32 {
        self.ptr(a) as *mut f32
    }
}

fn grow_buffer<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    current: Option<CudaSlice<T>>,
    need: usize,
    name: &str,
) -> Result<CudaSlice<T>, String> {
    if let Some(buffer) = current {
        if buffer.len() >= need {
            return Ok(buffer);
        }
    }
    let capacity = need.checked_next_power_of_two().unwrap_or(need);
    unsafe { stream.alloc(capacity) }
        .map_err(|e| format!("{name} allocation ({capacity} elements): {e:?}"))
}

struct TableLayout {
    len: usize,
    off: [u64; N_TABLES],
}

impl TableLayout {
    fn new() -> Self {
        Self {
            len: 0,
            off: [0; N_TABLES],
        }
    }

    fn put<T: Copy>(&mut self, slot: Table, values: &[T]) -> Result<(), String> {
        let align = align_of::<T>();
        let pad = (align - self.len % align) % align;
        self.len = self
            .len
            .checked_add(pad)
            .ok_or("wave table size overflow")?;
        self.off[slot as usize] = self.len as u64;
        let n = values
            .len()
            .checked_mul(size_of::<T>())
            .ok_or("wave table size overflow")?;
        self.len = self.len.checked_add(n).ok_or("wave table size overflow")?;
        Ok(())
    }
}

macro_rules! wave_table_fields {
    ($put:ident, $w:ident, $jobs:ident) => {
        $put!(NodeKind, $w.node_kind.as_slice());
        $put!(NodePlayer, $w.node_player.as_slice());
        $put!(NodeNc, $w.node_nc.as_slice());
        $put!(NodeChildStart, $w.node_child_start.as_slice());
        $put!(NodeChild, $w.node_child.as_slice());
        $put!(LegalRowOf, $w.legal_row_of.as_slice());
        $put!(LegalOff, $w.legal_off.as_slice());
        $put!(LegalValue, $w.legal_value.as_slice());
        $put!(DrawOff, $w.draw_off.as_slice());
        $put!(DrawTo, $w.draw_to.as_slice());
        $put!(DrawP, $w.draw_p.as_slice());
        $put!(DrawRowOff, $w.draw_row_off.as_slice());
        $put!(DrawRowStart, $w.draw_row_start.as_slice());
        $put!(ReachBase, $w.reach_base.as_slice());
        $put!(Soff, $w.soff.as_slice());
        $put!(Voff, $w.voff.as_slice());
        $put!(NodeParent, $w.node_parent.as_slice());
        $put!(RevRowOf, $w.rev_row_of.as_slice());
        $put!(RevStart, $w.rev_start.as_slice());
        $put!(RevSrc, $w.rev_src.as_slice());
        $put!(RevCell, $w.rev_cell.as_slice());
        $put!(RvdRowOf, $w.rvd_row_of.as_slice());
        $put!(RvdStart, $w.rvd_start.as_slice());
        $put!(RvdSrc, $w.rvd_src.as_slice());
        $put!(RvdP, $w.rvd_p.as_slice());
        $put!(RowNode, $w.row_node.as_slice());
        $put!(RowJob, $w.row_job.as_slice());
        $put!(RowCfgOff, $w.row_cfg_off.as_slice());
        $put!(RowCfg, $w.row_cfg.as_slice());
        $put!(RawRows, $w.raw_rows.as_slice());
        $put!(CardFeat, $w.card_feat.as_slice());
        $put!(Ids, $w.ids.as_slice());
        $put!(ConfigJob, $w.config_job.as_slice());
        $put!(Cphi, $w.cphi.as_slice());
        $put!(Roots, $w.roots.as_slice());
        $put!(Carried, $w.carried.as_slice());
        $put!(NodeUtility, $w.node_utility.as_slice());
        $put!(ExitNodes, $w.exit_nodes.as_slice());
        $put!(ExitCoff, $w.exit_coff.as_slice());
        $put!(Decision0, $w.decision[0].as_slice());
        $put!(Decision1, $w.decision[1].as_slice());
        $put!(ReachTask0, $w.reach_task[0].as_slice());
        $put!(ReachTask1, $w.reach_task[1].as_slice());
        $put!(ReachLevel0, $w.reach_level[0].as_slice());
        $put!(ReachLevel1, $w.reach_level[1].as_slice());
        $put!(BackTask0, $w.back_task[0].as_slice());
        $put!(BackTask1, $w.back_task[1].as_slice());
        $put!(BackLevel0, $w.back_level[0].as_slice());
        $put!(BackLevel1, $w.back_level[1].as_slice());
        $put!(Readout, $w.readout.as_slice());
        $put!(Jobs, $jobs);
    };
}

fn job_devices(w: &Wave) -> Result<Vec<JobDev>, String> {
    w.jobs
        .iter()
        .map(|j| {
            Ok(JobDev {
                node0: u32n(j.nodes.start, "job node")?,
                nodes: u32n(j.nodes.len(), "job nodes")?,
                row0: u32n(j.rows.start, "job row")?,
                rows: u32n(j.rows.len(), "job rows")?,
                nleaf: u32n(j.network_leaves, "job leaves")?,
                config0: u32n(j.configs.start, "job config")?,
                ncfg: u32n(j.configs.len(), "job configs")?,
                cell0: u32n(j.cells.start, "job cell")?,
                ncells: u32n(j.cells.len(), "job cells")?,
                reach0: u32n(j.reach.start, "job reach")?,
                reach_len: u32n(j.reach.len(), "job reach length")?,
                vals0: u32n(j.vals.start, "job values")?,
                vals_len: u32n(j.vals.len(), "job values length")?,
                root0: u32n(j.root.start, "job root")?,
                root_n0: u32n(j.root_nc[0], "job root p0")?,
                root_n1: u32n(j.root_nc[1], "job root p1")?,
                carried0: u32n(j.carried.start, "job carried")?,
                nroots: u32n(j.nroots, "job roots")?,
                root_value0: u32n(j.root_values.start, "job root values")?,
                exit0: u32n(j.exits.start, "job exit")?,
                nexits: u32n(j.exits.len(), "job exits")?,
                exit_cfg0: u32n(j.exit_configs.start, "job exit configs")?,
                snapshot_configs: u32n(j.exit_configs.len(), "job snapshot configs")?,
            })
        })
        .collect()
}

fn table_layout(w: &Wave, jobs: &[JobDev]) -> Result<([u64; N_TABLES], usize), String> {
    let mut layout = TableLayout::new();
    macro_rules! put {
        ($slot:ident, $values:expr) => {
            layout.put(Table::$slot, $values)?
        };
    }
    wave_table_fields!(put, w, jobs);
    Ok((layout.off, layout.len))
}

fn pack_tables(
    w: &Wave,
    jobs: &[JobDev],
    off: &[u64; N_TABLES],
    len: usize,
) -> Result<Vec<u8>, String> {
    let mut tables = vec![0u8; len];
    macro_rules! put {
        ($slot:ident, $values:expr) => {
            copy_table(&mut tables, off[Table::$slot as usize] as usize, $values)?
        };
    }
    wave_table_fields!(put, w, jobs);
    Ok(tables)
}

fn copy_table<T: Copy>(tables: &mut [u8], at: usize, values: &[T]) -> Result<(), String> {
    let n = values
        .len()
        .checked_mul(size_of::<T>())
        .ok_or("wave table size overflow")?;
    if n == 0 {
        return Ok(());
    }
    // SAFETY: every table element is Copy and Task/ReadTask/JobDev use repr(C).
    let raw = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), n) };
    tables[at..at + n].copy_from_slice(raw);
    Ok(())
}

fn arena_layout(w: &Wave, l: &V3Layout) -> Result<([u64; N_ARENAS], usize), String> {
    let rows = w.row_node.len();
    let cfgs = w.config_job.len();
    let jobs = w.jobs.len();
    let cells = w.legal_value.len();
    let reach = w.reach_len;
    let vals = *w.voff.last().unwrap_or(&0) as usize;
    let roots = w.jobs.last().map_or(0, |j| j.root_values.end);
    let carry_snaps = if w.meta.snapshots {
        w.meta.snap_iters.len().saturating_sub(1)
    } else {
        0
    };
    let fast_head = fast_gemm() && l.hmlp.is_empty();
    let (pubw, _, _, slotw) = l.widths();
    let max_pub = pubw
        .into_iter()
        .chain([l.xdim(), l.head_in])
        .max()
        .unwrap_or(1);
    let slot_max = slotw
        .into_iter()
        .chain([l.hfeat(), l.dg])
        .max()
        .unwrap_or(1);
    let card_max = l.card.iter().map(|x| x.o).max().unwrap_or(l.de);
    let bh = (rows * max_pub)
        .max(cfgs * NSLOT * slot_max)
        .max(jobs * NTYPE * card_max)
        .max(rows * NTYPE * l.de)
        .max(cfgs * l.dg);
    let bg = (jobs * NTYPE * CARD_FEATS)
        .max(rows * NTYPE * PILE_COUNTS)
        .max(rows * NTYPE * l.de + jobs * NTYPE * l.de)
        .max(cfgs * NSLOT * l.hfeat())
        .max(cfgs * (l.rank + 1));
    let sizes = [
        reach,
        reach.max(vals),
        vals,
        cells,
        cells,
        cells,
        cells,
        jobs * NTYPE * l.de,
        cfgs * l.dg,
        cfgs * (l.rank + 1),
        if fast_head {
            (rows * l.head_in).div_ceil(2)
        } else {
            rows * l.head_in
        },
        if fast_head {
            (rows * 2 * l.dg).div_ceil(2)
        } else {
            rows * 2 * l.dg
        },
        if fast_head {
            (rows * l.head_in).div_ceil(2)
        } else {
            rows * h_stride(l)
        },
        if fast_head {
            (rows * l.head_in).div_ceil(2)
        } else {
            rows * h_stride(l)
        },
        rows * l.rank,
        roots,
        (carry_snaps * w.snapshot_configs).div_ceil(2),
        rows * l.xdim(),
        bh,
        bh,
        bg,
    ];
    let mut off = [0u64; N_ARENAS];
    let mut at = 0usize;
    for (i, &n) in sizes.iter().enumerate() {
        at = at
            .checked_add(31)
            .ok_or("wave FP32 arena alignment overflow")?
            & !31;
        off[i] = at as u64;
        at = at.checked_add(n).ok_or("wave FP32 arena size overflow")?;
    }
    Ok((off, at))
}

fn unpack(
    w: Wave,
    strategy: Vec<f32>,
    root_values: Vec<f32>,
    carry: Vec<u16>,
    version: u64,
) -> Result<Vec<SolveResult>, String> {
    let snapshots = if w.meta.snapshots {
        w.meta.snap_iters.len().saturating_sub(1)
    } else {
        0
    };
    let mut out = Vec::with_capacity(w.jobs.len());
    for j in &w.jobs {
        let mut roots = Vec::with_capacity(j.nroots);
        let stride = j.root_nc[0] + j.root_nc[1];
        for r in 0..j.nroots {
            let at = j.root_values.start + r * stride;
            roots.push([
                root_values[at..at + j.root_nc[0]].to_vec(),
                root_values[at + j.root_nc[0]..at + stride].to_vec(),
            ]);
        }
        let mut data = Vec::with_capacity(snapshots * j.exit_configs.len());
        for s in 0..snapshots {
            let at = s * w.snapshot_configs + j.exit_configs.start;
            data.extend_from_slice(&carry[at..at + j.exit_configs.len()]);
        }
        let node0 = j.nodes.start as u32;
        let cfg0 = j.exit_configs.start as u32;
        out.push(SolveResult {
            strategy: strategy[j.cells.clone()].to_vec(),
            root_values: roots,
            carries: CarryStore {
                exit_nodes: w.exit_nodes[j.exits.clone()]
                    .iter()
                    .map(|&x| x - node0)
                    .collect(),
                coff: w.exit_coff[2 * j.exits.start..=2 * j.exits.end]
                    .iter()
                    .map(|&x| x - cfg0)
                    .collect(),
                snapshots,
                snapshot_configs: j.exit_configs.len(),
                data,
            },
            weight_version: version,
            oversize_route: false,
        });
    }
    Ok(out)
}

fn copy_arena(
    stream: &Arc<CudaStream>,
    d: &DeviceWave,
    arena: Arena,
    start: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let at = d.host.aoff[arena as usize] as usize + start;
    stream
        .memcpy_dtov(&d.buffers.arena.slice(at..at + len))
        .map_err(|e| format!("wave result copy: {e:?}"))
}

fn copy_arena_f16(
    stream: &Arc<CudaStream>,
    d: &DeviceWave,
    arena: Arena,
    len: usize,
) -> Result<Vec<u16>, String> {
    let words = copy_arena(stream, d, arena, 0, len.div_ceil(2))?;
    let mut out = Vec::with_capacity(len);
    for word in words {
        let bits = word.to_bits();
        out.push(bits as u16);
        if out.len() < len {
            out.push((bits >> 16) as u16);
        }
    }
    Ok(out)
}

fn validate_layout(l: &V3Layout) -> Result<(), String> {
    if l.card.is_empty() || l.pub_lin.is_empty() {
        return Err("v5 requires non-empty card and public towers".into());
    }
    if l.card.len() > 8
        || l.pub_lin.len() > 8
        || l.hmlp.len() > 8
        || l.slot.len() > 8
        || l.res.len() > 4
    {
        return Err("network tower exceeds v5 kernel pointer tables".into());
    }
    Ok(())
}

fn h_stride(l: &V3Layout) -> usize {
    std::iter::once(l.head_in)
        .chain(l.hmlp.iter().map(|x| x.o))
        .max()
        .unwrap_or(l.head_in)
}

fn factor(t: f32, p: f32) -> f32 {
    if p.is_infinite() {
        if p > 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        let x = t.powf(p);
        x / (x + 1.0)
    }
}

fn threads(n: i32) -> u32 {
    if n <= 0 {
        0
    } else {
        (n as u32).div_ceil(BLOCK)
    }
}
fn threads_usize(n: usize) -> u32 {
    if n == 0 {
        0
    } else {
        (n as u32).div_ceil(BLOCK)
    }
}
fn warps(n: usize) -> u32 {
    if n == 0 {
        0
    } else {
        ((n as u32) * 32).div_ceil(BLOCK)
    }
}

macro_rules! launch {
    ($me:expr, $wave:expr, $bank:expr, $kernel:ident, $grid:expr $(, $arg:expr)* $(,)?) => {{
        let grid = $grid;
        if grid != 0 {
            let f = $me.kernels.$kernel.clone();
            let mut args = $me.stream.launch_builder(&f);
            args.arg(&$wave.buffers.dev).arg(&$bank.dev);
            $(let scalar = $arg; args.arg(&scalar);)*
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { args.launch(cfg) }
                .map_err(|e| format!("CUDA {} launch: {e:?}", stringify!($kernel)))?;
        }
        Ok::<(), String>(())
    }};
}
use launch;

fn htod<T: cudarc::driver::DeviceRepr + Unpin>(
    stream: &Arc<CudaStream>,
    values: &[T],
) -> Result<CudaSlice<T>, String> {
    let mut out =
        unsafe { stream.alloc(values.len().max(1)) }.map_err(|e| format!("CUDA alloc: {e:?}"))?;
    if !values.is_empty() {
        stream
            .memcpy_htod(values, &mut out)
            .map_err(|e| format!("CUDA H2D: {e:?}"))?;
    }
    Ok(out)
}

fn ptr<T>(stream: &Arc<CudaStream>, value: &CudaSlice<T>) -> *const T {
    let (p, _) = value.device_ptr(stream);
    p as usize as *const T
}

fn ptr_mut<T>(stream: &Arc<CudaStream>, value: &mut CudaSlice<T>) -> *mut T {
    let (p, _) = value.device_ptr_mut(stream);
    p as usize as *mut T
}

fn fast_gemm() -> bool {
    static FAST_F16: OnceLock<bool> = OnceLock::new();
    *FAST_F16.get_or_init(|| std::env::var_os("WARCHEST_GPU_PRECISE_GEMM").is_none())
}

#[allow(clippy::too_many_arguments)]
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
) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    let alpha = 1.0f32;
    static LOG_MODE: Once = Once::new();
    let fast_f16 = fast_gemm();
    LOG_MODE.call_once(|| {
        eprintln!(
            "v5 GEMM compute={}",
            if fast_f16 { "fast-f16" } else { "fp32" }
        );
    });
    let result = if fast_f16 {
        use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F;
        use cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        use cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F;

        unsafe {
            cudarc::cublas::result::gemm_ex(
                *blas.handle(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha as *const f32).cast(),
                b.cast(),
                CUDA_R_32F,
                ldb as i32,
                a.cast(),
                CUDA_R_32F,
                lda as i32,
                (&beta as *const f32).cast(),
                c.cast(),
                CUDA_R_32F,
                ldc as i32,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
    } else {
        unsafe {
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
        }
    };
    result.map_err(|e| format!("cuBLAS GEMM {m}x{k}x{n}: {e:?}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_f16(
    blas: &CudaBlas,
    m: usize,
    n: usize,
    k: usize,
    a: *const u16,
    lda: usize,
    b: *const u16,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    use cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    use cudarc::cublas::sys::cudaDataType_t::{CUDA_R_16F, CUDA_R_32F};
    let (alpha, beta) = (1.0f32, 0.0f32);
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            b.cast(),
            CUDA_R_16F,
            ldb as i32,
            a.cast(),
            CUDA_R_16F,
            lda as i32,
            (&beta as *const f32).cast(),
            c.cast(),
            CUDA_R_32F,
            ldc as i32,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(|e| format!("cuBLAS FP16 GEMM {m}x{k}x{n}: {e:?}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_f16_to_f16(
    blas: &CudaBlas,
    m: usize,
    n: usize,
    k: usize,
    a: *const u16,
    lda: usize,
    b: *const u16,
    ldb: usize,
    c: *mut u16,
    ldc: usize,
) -> Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
    use cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    use cudarc::cublas::sys::cudaDataType_t::CUDA_R_16F;
    let (alpha, beta) = (1.0f32, 0.0f32);
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            (&alpha as *const f32).cast(),
            b.cast(),
            CUDA_R_16F,
            ldb as i32,
            a.cast(),
            CUDA_R_16F,
            lda as i32,
            (&beta as *const f32).cast(),
            c.cast(),
            CUDA_R_16F,
            ldc as i32,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(|e| format!("cuBLAS FP16-output GEMM {m}x{k}x{n}: {e:?}"))?;
    Ok(())
}

fn u32n(x: usize, what: &str) -> Result<u32, String> {
    u32::try_from(x).map_err(|_| format!("wave {what} exceeds u32"))
}
fn i32n(x: usize, what: &str) -> Result<i32, String> {
    i32::try_from(x).map_err(|_| format!("wave {what} exceeds i32 launch range"))
}

fn cuda_preamble(l: &V3Layout) -> String {
    let arr = |name: &str, values: &[usize]| {
        let body = if values.is_empty() {
            "0".into()
        } else {
            values
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!("static __device__ const int {name}[] = {{{body}}};\n")
    };
    let (pubw, hmlp, card, slot) = l.widths();
    let locations = crate::board::board()
        .is_location
        .iter()
        .map(|&x| if x { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "#define DE {}\n#define DG {}\n#define RK {}\n#define HEADW {}\n\
         #define HEADOUT {}\n#define H_STRIDE {}\n#define XD {}\n#define HF {}\n\
         #define NCARD {}\n#define NPUB {}\n#define NHMLP {}\n#define NSLOTL {}\n#define NRES {}\n\
         {}{}{}{}\
         #define N_HEXES {}\n#define NSLOT {}\n#define NTYPE {}\n#define HEX_FACTS {}\n\
         #define PILE_COUNTS {}\n#define CARD_FEATS {}\n#define CFEAT {}\n\
         #define MAX_COINS {:.1}f\n#define MAX_PLIES {:.1}f\n#define LOOSE {}\n\
         #define GPU_ROW_BYTES {}\n#define GR_HEX_OWNER {}\n#define GR_HEX_SLOT {}\n\
         #define GR_HEX_HEIGHT {}\n#define GR_HEX_MARKER {}\n#define GR_PILES {}\n\
         #define GR_MARKERS {}\n#define GR_HAND {}\n#define GR_FD {}\n#define GR_BAG {}\n\
         #define GR_INITIATIVE {}\n#define GR_INIT_MOVED {}\n#define GR_TO_ACT {}\n#define GR_PLIES {}\n\
         static __device__ const unsigned char HEX_LOCATION[N_HEXES] = {{{locations}}};\n",
        l.de, l.dg, l.rank, l.head_in, l.head_out, h_stride(l), l.xdim(), l.hfeat(),
        l.card.len(), l.pub_lin.len(), l.hmlp.len(), l.slot.len(), l.res.len(),
        arr("CARDW", &card), arr("PUBW", &pubw), arr("HMLPW", &hmlp), arr("SLOTW", &slot),
        crate::board::N_HEXES, NSLOT, NTYPE, crate::rebel::HEX_FACTS,
        PILE_COUNTS, CARD_FEATS, CFEAT, crate::rebel::MAX_COINS,
        crate::state::MAX_MAIN_PLAYS as f32, crate::rebel::LOOSE,
        GPU_ROW_BYTES, crate::rebel::GPU_ROW_HEX_OWNER, crate::rebel::GPU_ROW_HEX_SLOT,
        crate::rebel::GPU_ROW_HEX_HEIGHT, crate::rebel::GPU_ROW_HEX_MARKER,
        crate::rebel::GPU_ROW_PILES, crate::rebel::GPU_ROW_MARKERS,
        crate::rebel::GPU_ROW_HAND, crate::rebel::GPU_ROW_FD, crate::rebel::GPU_ROW_BAG,
        crate::rebel::GPU_ROW_INITIATIVE, crate::rebel::GPU_ROW_INIT_MOVED,
        crate::rebel::GPU_ROW_TO_ACT, crate::rebel::GPU_ROW_PLIES,
    )
}

#[cfg(test)]
mod reservation_tests {
    use super::*;
    use crate::serialize::PackedJob;

    #[test]
    fn one_job_admission_matches_device_arena_growth() {
        let mut job = PackedJob::stub();
        job.carried.push([vec![1.0], vec![1.0]]);
        let reserved = job.work().mutable_bytes;
        let wave = Wave::pack(&[job]).expect("pack stub wave");
        let layout = V3Layout::new(&wave.meta.net_dims).expect("network layout");
        let (_, floats) = arena_layout(&wave, &layout).expect("device arena layout");
        let allocated = floats
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(floats.max(1))
            * size_of::<f32>();
        assert_eq!(reserved, allocated);
    }
}
