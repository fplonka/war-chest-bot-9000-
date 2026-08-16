//! One contiguous FP32 wave on one CUDA stream.
//!
//! The first correctness lane intentionally has no resident solve state: a
//! wave uploads one immutable byte blob and owns one flat FP32 arena until its
//! single completion is copied out. The dispatcher, not a device state
//! machine, decides which solves share the wave.

use std::collections::HashMap;
use std::mem::{align_of, size_of, MaybeUninit};
use std::sync::{Arc, Once};
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::CudaBlas;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
use cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use cudarc::driver::{result, sys, CudaSlice, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc;

use crate::board::N_HEXES;
use crate::gpu::client::{CarryStore, SolveResult};
use crate::net::{V5Layout, BLOCKS, C, CFGH, D, JBLOCKS, JOIN_IN, JW, POOL, TYPE};
use crate::rebel::{CFEAT, GPU_ROW_BYTES, LOOSE, NSLOT, NTYPE, PILE_COUNTS};
use crate::serialize::TRUNK_CHUNK_ROWS;
use crate::units::CARD_FEATS;

use super::wave::Wave;

const BLOCK: u32 = 256;
/// Leaf tasks a single `readout` block values. Its blocks share the candidate
/// pool through L1, so a tile turns repeat candidate reads into cache hits.
const READOUT_TILE: usize = 4;
const GRAPH_CLASSES: usize = 4;
const N_TABLES: usize = 51;
const N_ARENAS: usize = 31;

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
    ValsBase,
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
    ConfigPlayer,
}

#[repr(usize)]
#[derive(Clone, Copy)]
enum Arena {
    // CFR state, from `initialise` onwards.
    Reach,
    SnapReach,
    Vals,
    Regret,
    Cur,
    Sum,
    SnapStrat,
    // Built once per wave and read by every iteration.
    Cards,
    Bag,
    F,
    G,
    P,
    Jp,
    // The per-iteration join.
    Pooled,
    Jin,
    Z,
    Jt,
    H,
    // Results.
    RootValues,
    Carry,
    // Trunk and encoder scratch, dead before the first iteration. `A` carries
    // one zero hex per row for the neighbour gather, and is written over by
    // the mix GEMM's output; `Mix` is written over by the block output. Both
    // are rebuilt from the residual stream at the top of the next block.
    Tok,
    Ts,
    X,
    A,
    Mix,
    Pool,
    Gb,
    Board,
    Pack,
    Hidden,
    Cfg,
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

/// Only what device code dereferences: biases, LayerNorm pairs, the two
/// embeddings, and the three matrices narrow enough that a kernel applies them
/// itself instead of paying for a pack buffer and a tiny-K GEMM. Every other
/// matrix reaches cuBLAS as a raw offset into the weight blob.
#[repr(C)]
#[derive(Clone, Copy)]
struct WeightDev {
    card_b: [*const f32; 2],
    pile_w: *const f32,
    seat: *const f32,
    hex_stem_w: *const f32,
    hex_stem_b: *const f32,
    pos: *const f32,
    glob_stem_w: *const f32,
    mix_b: [*const f32; BLOCKS],
    pool_b: [*const f32; BLOCKS],
    out_b: [*const f32; BLOCKS],
    pre_lnw: [*const f32; BLOCKS + 1],
    pre_lnb: [*const f32; BLOCKS + 1],
    mid_lnw: [*const f32; BLOCKS],
    mid_lnb: [*const f32; BLOCKS],
    board_b: *const f32,
    cfg1_b: *const f32,
    cfg_lnw: *const f32,
    cfg_lnb: *const f32,
    cfg_f_b: *const f32,
    cfg_g_b: *const f32,
    join_b_b: *const f32,
    join_w_b: [*const f32; JBLOCKS],
    join_lnw: [*const f32; JBLOCKS],
    join_lnb: [*const f32; JBLOCKS],
    jout_lnw: *const f32,
    jout_lnb: *const f32,
    join_out_b: *const f32,
    h_lnw: *const f32,
    h_lnb: *const f32,
    value_bias: *const f32,
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
    layout: V5Layout,
    _w: CudaSlice<f32>,
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
    pack_cards, bias_gelu, cards_finish, tokens, stem,
    trunk_norm, gather_mix, hex_pool, block_mid, block_out,
    board_pool, board_bias, config_pack, slot_sum, config_finish, norm,
    init_strategy, seed_reach, reach_sweep, seed_sum,
    belief_sums, join_input, join_block, join_finish, readout,
    backprop_sweep, normalize_strategy, gather_carry, collect_root,
}

/// Compile the wave kernels for one layout and derive the launch geometry the
/// occupancy of those kernels implies. Split out of `Executor::new` because a
/// shape change has to do it again: a ladder that rates a new network
/// architecture against the pool it has to beat publishes two shapes to one
/// card, and everything below the kernels is wave-sized and rebuilt anyway.
fn build_kernels(context: &Arc<CudaContext>) -> Result<(Kernels, u32, u32, u32), String> {
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
    let source = format!("{}\n{}", cuda_preamble(), include_str!("wave_kernels.cu"));
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
    Ok((kernels, backprop_blocks, reach_blocks, sweep_block))
}

/// The cuBLAS handle together with the math mode its GEMMs run in. `fast` puts
/// the value head on the tensor cores -- FP16 multiplies with FP32
/// accumulation, four times the card's FP32 rate -- which is what a depth-two
/// solve needs and what production runs. The CPU-oracle tests build a precise
/// executor so that they still compare exact math.
pub struct Blas {
    handle: CudaBlas,
    fast: bool,
}

impl Blas {
    /// `precise` pins plain SGEMM. Otherwise `WARCHEST_GPU_GEMM=precise` can
    /// still pin it for a whole run, which is how a production build is
    /// A/B'd against exact math.
    pub fn new(handle: CudaBlas, precise: bool) -> Result<Self, String> {
        let fast = match std::env::var_os("WARCHEST_GPU_GEMM") {
            Some(mode) if mode == "fast" => true,
            Some(mode) if mode == "precise" => false,
            Some(mode) => {
                return Err(format!(
                    "WARCHEST_GPU_GEMM is fast or precise, not {mode:?}"
                ))
            }
            None => !precise,
        };
        Ok(Self { handle, fast })
    }
}

pub struct Executor {
    stream: Arc<CudaStream>,
    blas: Blas,
    kernels: Kernels,
    dims: Vec<usize>,
    banks: HashMap<u64, WeightBank>,
    buffers: Option<DeviceBuffers>,
    /// Host blob the wave tables are assembled into before the H2D copy. One
    /// per lane, grown to its high-water mark and reused.
    staging: Vec<u8>,
    graphs: Vec<GraphExec>,
    next_graph: usize,
    backprop_blocks: u32,
    reach_blocks: u32,
    sweep_block: u32,
}

/// A wave whose schedule is queued on the lane's stream. It owns the lane's
/// device buffers until `collect` synchronises, so nothing may reuse them.
pub struct InFlight {
    device: DeviceWave,
    wave: Wave,
    version: u64,
    profile: bool,
    started: Instant,
    uploaded: Instant,
    captured: Instant,
    direct: bool,
    graph_reused: bool,
}

/// A finished wave's raw device output, still to be turned into per-job
/// results. Separating this from `collect` lets the lane launch the next wave
/// before it does the unpacking.
pub struct Harvest {
    wave: Wave,
    strategy: Vec<f32>,
    root_values: Vec<f32>,
    carries: Vec<u16>,
    version: u64,
    profile: bool,
    started: Instant,
    uploaded: Instant,
    captured: Instant,
    queued_output: Instant,
    completed: Instant,
    direct: bool,
    graph_reused: bool,
}

impl Harvest {
    pub fn unpack(self) -> Result<Vec<SolveResult>, String> {
        let jobs = self.wave.jobs.len();
        let rows = self.wave.row_node.len();
        let cells = self.wave.legal_value.len();
        let result = unpack(
            self.wave,
            self.strategy,
            self.root_values,
            self.carries,
            self.version,
        );
        if self.profile {
            let unpacked = Instant::now();
            let (direct, reused) = (self.direct, self.graph_reused);
            eprintln!(
                "v5_device jobs={jobs} rows={rows} cells={cells} direct={direct} graph_reused={reused} upload_ms={:.2} capture_ms={:.2} queue_ms={:.2} gpu_ms={:.2} unpack_ms={:.2} total_ms={:.2}",
                1e3 * (self.uploaded - self.started).as_secs_f64(),
                1e3 * (self.captured - self.uploaded).as_secs_f64(),
                1e3 * (self.queued_output - self.captured).as_secs_f64(),
                1e3 * (self.completed - self.queued_output).as_secs_f64(),
                1e3 * (unpacked - self.completed).as_secs_f64(),
                1e3 * (unpacked - self.started).as_secs_f64(),
            );
        }
        result
    }
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
        blas: impl FnOnce(CudaBlas) -> Result<Blas, String>,
    ) -> Result<Self, String> {
        let context =
            CudaContext::new(device).map_err(|e| format!("cuda device {device}: {e:?}"))?;
        // Raw pointers in WaveDev/WeightDev have one explicit stream ordering
        // protocol; per-slice event tracking would duplicate it at every node.
        unsafe { context.disable_event_tracking() };
        let stream = context
            .new_stream()
            .map_err(|e| format!("CUDA stream: {e:?}"))?;
        let blas = blas(CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS: {e:?}"))?)?;
        let (kernels, backprop_blocks, reach_blocks, sweep_block) = build_kernels(&context)?;
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
            staging: Vec::new(),
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
                "cannot publish model format {dims:?}; expected {:?}",
                self.dims
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

    /// Queue a wave's whole 64-iteration schedule and return without waiting
    /// for any of it. The lane then has the card's service time to spend on
    /// host work -- answering the previous wave and assembling the next one --
    /// instead of standing idle between waves, which is where the cards' whole
    /// remaining idle time sits.
    pub fn launch(&mut self, wave: Wave, version: u64) -> Result<InFlight, String> {
        let profile = std::env::var_os("WARCHEST_GPU_PROFILE").is_some()
            || (std::env::var_os("WARCHEST_ROUTE_PROFILE").is_some()
                && wave.work.requires_arena_guard_route());
        let started = Instant::now();
        let bank = self
            .banks
            .get(&version)
            .ok_or_else(|| format!("GPU weight version {version} is unavailable"))?;
        let mut device = DeviceWave::upload(
            &self.stream,
            &wave,
            self.buffers.take(),
            &mut self.staging,
        )?;
        let uploaded = Instant::now();
        // Arbitrary live waves often make cuBLAS choose a different captured
        // topology, in which case graph update cannot reuse the executable and
        // capture+instantiation is pure overhead. Keep a direct-stream A/B path
        // while tuning that crossover; it executes the identical kernels and
        // GEMMs in the identical order.
        // `WARCHEST_DIRECT=0` puts the wave back on CUDA Graphs.
        let direct = std::env::var_os("WARCHEST_DIRECT").is_some_and(|x| x != "0");
        if !direct {
            self.stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| format!("begin v5 CUDA Graph: {e:?}"))?;
        }
        let queued = (|| -> Result<(), String> {
            self.build_network(&device, bank)?;
            device.clear_solve_state(&self.stream)?;
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
                for p in 0..2 {
                    self.run_head(&device, bank, true, p)?;
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

        Ok(InFlight {
            device,
            wave,
            version,
            profile,
            started,
            uploaded,
            captured,
            direct,
            graph_reused: graph.is_some_and(|(_, reused)| reused),
        })
    }

    /// Wait for a launched wave and take its results off the card. The result
    /// copies are pageable, so issuing them here is what blocks until the
    /// schedule finishes; that is the point at which the lane means to wait.
    pub fn collect(&mut self, f: InFlight) -> Result<Harvest, String> {
        let InFlight {
            device,
            wave,
            version,
            profile,
            started,
            uploaded,
            captured,
            direct,
            graph_reused,
        } = f;
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
        Ok(Harvest {
            wave,
            strategy,
            root_values,
            carries,
            version,
            profile,
            started,
            uploaded,
            captured,
            queued_output,
            completed,
            direct,
            graph_reused,
        })
    }

    pub fn solve(&mut self, wave: Wave, version: u64) -> Result<Vec<SolveResult>, String> {
        let f = self.launch(wave, version)?;
        self.collect(f)?.unpack()
    }

    /// Everything that does not move between CFR iterations: the card
    /// describer, the trunk, the board vector and its cached join projection,
    /// and the config encoder.
    fn build_network(&self, d: &DeviceWave, bank: &WeightBank) -> Result<(), String> {
        let l = &bank.layout;
        let rows = d.host.rows as usize;
        let cfgs = d.host.ncfg as usize;
        let jobs = d.host.jobs as usize;
        let queries = 2 * rows;

        // The printed-card tokens: two canonical views per solve, reused by the
        // trunk stem and by every config in the subgame.
        let cards = 2 * jobs * NTYPE;
        launch!(self, d, bank, pack_cards, threads_usize(cards * CARD_FEATS))?;
        gemm(
            &self.blas,
            cards,
            TYPE,
            CARD_FEATS,
            d.ptr(Arena::Pack),
            CARD_FEATS,
            bank.w_ptr(l.card[0].w),
            TYPE,
            d.ptr_mut(Arena::Hidden),
            TYPE,
            0.0,
        )?;
        launch!(
            self,
            d,
            bank,
            bias_gelu,
            threads_usize(cards * TYPE),
            0i32,
            cards as i32
        )?;
        gemm(
            &self.blas,
            cards,
            TYPE,
            TYPE,
            d.ptr(Arena::Hidden),
            TYPE,
            bank.w_ptr(l.card[1].w),
            TYPE,
            d.ptr_mut(Arena::Cards),
            TYPE,
            0.0,
        )?;
        launch!(self, d, bank, cards_finish, threads_usize(cards * TYPE))?;
        // The count-weighted half of `g`, one row per card token.
        gemm(
            &self.blas,
            cards,
            3 * POOL,
            TYPE,
            d.ptr(Arena::Cards),
            TYPE,
            bank.w_ptr(l.cfg_m.w),
            3 * POOL,
            d.ptr_mut(Arena::Bag),
            3 * POOL,
            0.0,
        )?;

        // The trunk is a per-row map, and one row carries 37 hex tokens through
        // eight residual blocks. Running it in row chunks is what keeps its
        // working tensors -- most of a wave's memory -- off the exclusive
        // one-job route.
        for start in (0..queries).step_by(TRUNK_CHUNK_ROWS) {
            let n = TRUNK_CHUNK_ROWS.min(queries - start);
            self.trunk(d, bank, start, n)?;
        }
        // Projecting the board vector into the join's first layer once per
        // solve is the whole reason that vector is allowed to be wide.
        gemm(
            &self.blas,
            queries,
            JW,
            D,
            d.ptr(Arena::P),
            D,
            bank.w_ptr(l.join_p.w),
            JW,
            d.ptr_mut(Arena::Jp),
            JW,
            0.0,
        )?;

        let slots = cfgs * NSLOT;
        launch!(self, d, bank, config_pack, threads_usize(slots))?;
        gemm(
            &self.blas,
            slots,
            CFGH,
            3 + TYPE,
            d.ptr(Arena::Pack),
            3 + TYPE,
            bank.w_ptr(l.cfg1.w),
            CFGH,
            d.ptr_mut(Arena::Hidden),
            CFGH,
            0.0,
        )?;
        launch!(
            self,
            d,
            bank,
            bias_gelu,
            threads_usize(slots * CFGH),
            1i32,
            slots as i32
        )?;
        launch!(self, d, bank, slot_sum, warps(cfgs))?;
        launch!(self, d, bank, norm, warps(cfgs), 0i32, cfgs as i32)?;
        gemm(
            &self.blas,
            cfgs,
            D,
            CFGH,
            d.ptr(Arena::Cfg),
            CFGH,
            bank.w_ptr(l.cfg_f.w),
            D,
            d.ptr_mut(Arena::F),
            D,
            0.0,
        )?;
        gemm(
            &self.blas,
            cfgs,
            POOL,
            CFGH,
            d.ptr(Arena::Cfg),
            CFGH,
            bank.w_ptr(l.cfg_g.w),
            POOL,
            d.ptr_mut(Arena::G),
            POOL,
            0.0,
        )?;
        launch!(self, d, bank, config_finish, threads_usize(cfgs * (D + POOL)))?;
        Ok(())
    }

    /// One chunk of canonical view rows through the trunk, ending in that
    /// chunk's rows of the board vector `P`.
    fn trunk(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        start: usize,
        n: usize,
    ) -> Result<(), String> {
        let l = &bank.layout;
        let cells = n * N_HEXES;
        launch!(
            self,
            d,
            bank,
            tokens,
            threads_usize(n * NTYPE * TYPE),
            start as i32,
            n as i32
        )?;
        gemm(
            &self.blas,
            n * NTYPE,
            C,
            TYPE,
            d.ptr(Arena::Tok),
            TYPE,
            bank.w_ptr(l.tok_stem.w),
            C,
            d.ptr_mut(Arena::Ts),
            C,
            0.0,
        )?;
        launch!(self, d, bank, stem, n as u32, start as i32)?;
        for (i, blk) in l.blocks.iter().enumerate() {
            launch!(
                self,
                d,
                bank,
                trunk_norm,
                warps(n * (N_HEXES + 1)),
                i as i32,
                n as i32
            )?;
            launch!(self, d, bank, gather_mix, threads_usize(cells * C), cells as i32)?;
            launch!(self, d, bank, hex_pool, threads_usize(n * C), n as i32)?;
            gemm(
                &self.blas,
                n,
                C,
                2 * C,
                d.ptr(Arena::Pool),
                2 * C,
                bank.w_ptr(blk.pool.w),
                C,
                d.ptr_mut(Arena::Gb),
                C,
                0.0,
            )?;
            // `y` lands over the activations it consumed; the next block's
            // `trunk_norm` rebuilds the whole padded tensor from `x`.
            gemm(
                &self.blas,
                cells,
                C,
                2 * C,
                d.ptr(Arena::Mix),
                2 * C,
                bank.w_ptr(blk.mix.w),
                C,
                d.ptr_mut(Arena::A),
                C,
                0.0,
            )?;
            launch!(self, d, bank, block_mid, warps(cells), i as i32, cells as i32)?;
            gemm(
                &self.blas,
                cells,
                C,
                C,
                d.ptr(Arena::A),
                C,
                bank.w_ptr(blk.out.w),
                C,
                d.ptr_mut(Arena::Mix),
                C,
                0.0,
            )?;
            launch!(
                self,
                d,
                bank,
                block_out,
                threads_usize(cells * C),
                i as i32,
                cells as i32
            )?;
        }
        launch!(
            self,
            d,
            bank,
            trunk_norm,
            warps(n * (N_HEXES + 1)),
            BLOCKS as i32,
            n as i32
        )?;
        let width = 2 * C + LOOSE;
        launch!(
            self,
            d,
            bank,
            board_pool,
            threads_usize(n * width),
            start as i32,
            n as i32
        )?;
        gemm(
            &self.blas,
            n,
            D,
            width,
            d.ptr(Arena::Board),
            width,
            bank.w_ptr(l.board_out.w),
            D,
            unsafe { d.ptr_mut(Arena::P).add(start * D) },
            D,
            0.0,
        )?;
        launch!(
            self,
            d,
            bank,
            board_bias,
            threads_usize(n * D),
            start as i32,
            n as i32
        )
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

    /// The only per-iteration path: pool the beliefs, then run the join. `both`
    /// re-pools both players, which only the first iteration and the
    /// fixed-policy root queries need; otherwise the player whose strategy just
    /// moved is the only block that changed.
    fn run_head(
        &self,
        d: &DeviceWave,
        bank: &WeightBank,
        both: bool,
        traverser: usize,
    ) -> Result<(), String> {
        let l = &bank.layout;
        let rows = d.host.rows as usize;
        let t = traverser as i32;
        launch!(
            self,
            d,
            bank,
            belief_sums,
            warps(d.host.nleaf as usize),
            t,
            both as i32
        )?;
        launch!(
            self,
            d,
            bank,
            join_input,
            threads_usize(rows * (JOIN_IN + JW)),
            t,
            rows as i32
        )?;
        gemm(
            &self.blas,
            rows,
            JW,
            JOIN_IN,
            d.ptr(Arena::Jin),
            JOIN_IN,
            bank.w_ptr(l.join_b.w),
            JW,
            d.ptr_mut(Arena::Z),
            JW,
            1.0,
        )?;
        for (i, span) in l.join_w.iter().enumerate() {
            launch!(self, d, bank, join_block, warps(rows), i as i32, rows as i32)?;
            gemm(
                &self.blas,
                rows,
                JW,
                JW,
                d.ptr(Arena::Jt),
                JW,
                bank.w_ptr(span.w),
                JW,
                d.ptr_mut(Arena::Z),
                JW,
                1.0,
            )?;
        }
        launch!(self, d, bank, join_finish, warps(rows), t, rows as i32)?;
        gemm(
            &self.blas,
            rows,
            D,
            JW,
            d.ptr(Arena::Z),
            JW,
            bank.w_ptr(l.join_out.w),
            D,
            d.ptr_mut(Arena::H),
            D,
            1.0,
        )?;
        launch!(self, d, bank, norm, warps(rows), 1i32, rows as i32)
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
            (d.host.readout_n as usize).div_ceil(READOUT_TILE) as u32,
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
        let layout = V5Layout::new(dims)?;
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
        let wb = htod(stream, &w)?;
        let bb = htod(stream, &b)?;
        let lb = htod(stream, &ln)?;
        let (wp, bp, lp) = (ptr(stream, &wb), ptr(stream, &bb), ptr(stream, &lb));
        let mut d = WeightDev::default();
        let wa = |x| unsafe { wp.add(x) };
        let ba = |x| unsafe { bp.add(x) };
        let ln_pair = |i: usize| {
            let (g, bt) = layout.norms[i];
            (unsafe { lp.add(g) }, unsafe { lp.add(bt) })
        };
        for k in 0..2 {
            d.card_b[k] = ba(layout.card[k].b);
        }
        d.pile_w = wa(layout.pile.w);
        d.seat = wa(layout.seat);
        d.hex_stem_w = wa(layout.hex_stem.w);
        d.hex_stem_b = ba(layout.hex_stem.b);
        d.pos = wa(layout.pos);
        d.glob_stem_w = wa(layout.glob_stem.w);
        for (i, blk) in layout.blocks.iter().enumerate() {
            d.mix_b[i] = ba(blk.mix.b);
            d.pool_b[i] = ba(blk.pool.b);
            d.out_b[i] = ba(blk.out.b);
            (d.pre_lnw[i], d.pre_lnb[i]) = ln_pair(2 * i);
            (d.mid_lnw[i], d.mid_lnb[i]) = ln_pair(2 * i + 1);
        }
        // The norms run in application order, so the trunk's final one follows
        // the blocks and the config encoder's follows that.
        (d.pre_lnw[BLOCKS], d.pre_lnb[BLOCKS]) = ln_pair(2 * BLOCKS);
        (d.cfg_lnw, d.cfg_lnb) = ln_pair(2 * BLOCKS + 1);
        d.board_b = ba(layout.board_out.b);
        d.cfg1_b = ba(layout.cfg1.b);
        d.cfg_f_b = ba(layout.cfg_f.b);
        d.cfg_g_b = ba(layout.cfg_g.b);
        d.join_b_b = ba(layout.join_b.b);
        for (i, span) in layout.join_w.iter().enumerate() {
            d.join_w_b[i] = ba(span.b);
            (d.join_lnw[i], d.join_lnb[i]) = ln_pair(2 * BLOCKS + 2 + i);
        }
        (d.jout_lnw, d.jout_lnb) = ln_pair(2 * BLOCKS + 2 + JBLOCKS);
        (d.h_lnw, d.h_lnb) = ln_pair(2 * BLOCKS + 3 + JBLOCKS);
        d.join_out_b = ba(layout.join_out.b);
        d.value_bias = ba(layout.value_bias);
        let dev = htod(stream, &[d])?;
        Ok(Self {
            layout,
            _w: wb,
            _b: bb,
            _ln: lb,
            dev,
        })
    }

    fn w_ptr(&self, off: usize) -> *const f32 {
        unsafe { ptr(self.dev.stream(), &self._w).add(off) }
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
    /// Floats the tower phase overwrites that the solve phase needs zeroed.
    solve_state: std::ops::Range<usize>,
}

impl DeviceWave {
    fn upload(
        stream: &Arc<CudaStream>,
        w: &Wave,
        reuse: Option<DeviceBuffers>,
        staging: &mut Vec<u8>,
    ) -> Result<Self, String> {
        let jobs = job_devices(w)?;
        let (toff, table_len) = table_layout(w, &jobs)?;
        let (aoff, arena_len, solve_state) = arena_layout(w)?;
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
        let step = Instant::now();
        let mut tables = grow_buffer(stream, tables, table_need, "wave tables")?;
        let allocated = step.elapsed();
        let step = Instant::now();
        pack_tables(w, &jobs, &toff, table_len, staging)?;
        let packed = step.elapsed();
        let step = Instant::now();
        stream
            .memcpy_htod(&staging[..table_len], &mut tables.slice_mut(..table_len))
            .map_err(|e| format!("wave table H2D: {e:?}"))?;
        let copied = step.elapsed();
        let step = Instant::now();
        let mut arena = grow_buffer(stream, arena, arena_need, "wave arena")?;
        stream
            .memset_zeros(&mut arena.slice_mut(..arena_need))
            .map_err(|e| format!("wave arena zero: {e:?}"))?;
        let zeroed = step.elapsed();
        if std::env::var_os("WARCHEST_UPLOAD_PROFILE").is_some() {
            eprintln!(
                "v5_upload table_kib={} arena_kib={} alloc_ms={:.2} pack_ms={:.2} htod_ms={:.2} zero_ms={:.2}",
                table_len / 1024,
                arena_need * size_of::<f32>() / 1024,
                1e3 * allocated.as_secs_f64(),
                1e3 * packed.as_secs_f64(),
                1e3 * copied.as_secs_f64(),
                1e3 * zeroed.as_secs_f64(),
            );
        }
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
            vals_len: i32n(w.vals_len, "values")?,
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
            solve_state,
        })
    }

    /// Restore the zeroed CFR state the tower GEMMs wrote over. Stream order
    /// puts this after every tower kernel and before `initialise`.
    fn clear_solve_state(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        if self.solve_state.is_empty() {
            return Ok(());
        }
        stream
            .memset_zeros(&mut self.buffers.arena.slice_mut(self.solve_state.clone()))
            .map_err(|e| format!("clear reused wave solve state: {e:?}"))
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
        let bytes = buffer.len().saturating_mul(size_of::<T>());
        // Keep it if it fits and is not hoarding. A lane that once served one
        // gibibyte-sized wave would otherwise hold that gibibyte for the rest
        // of the run, and eight lanes doing that is how a 24 GiB card fills up
        // and a long run dies of an out-of-memory error twenty minutes in.
        // Below the floor the excess is not worth an allocation, and the four
        // times margin keeps ordinary wave-to-wave variation from thrashing.
        if buffer.len() >= need && (bytes <= RETAIN_FLOOR_BYTES || buffer.len() <= 4 * need) {
            return Ok(buffer);
        }
        // Dropped here rather than after the allocation below, so a shrink
        // never has both the old and the new buffer live.
    }
    let capacity = need.checked_next_power_of_two().unwrap_or(need);
    unsafe { stream.alloc(capacity) }
        .map_err(|e| format!("{name} allocation ({capacity} elements): {e:?}"))
}

/// A retained lane buffer under this size is never given back: the memory is
/// not worth a `cuMemAlloc` on the wave's critical path.
const RETAIN_FLOOR_BYTES: usize = 128 << 20;

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
        $put!(ValsBase, $w.vals_base.as_slice());
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
        $put!(ConfigPlayer, $w.config_player.as_slice());
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

/// Fill a lane's reusable host staging blob with this wave's tables.
///
/// The buffer is grown and never cleared. Every byte the device addresses is
/// written by the field copies below; only the alignment padding between
/// tables keeps whatever the previous wave left there, and nothing reads it.
/// A fresh `vec![0u8; len]` per wave was 15.6 ms of the 113 ms lane budget --
/// mostly first-touch page faults on ~20 MiB of new pages, with the card idle
/// for all of it.
fn pack_tables(
    w: &Wave,
    jobs: &[JobDev],
    off: &[u64; N_TABLES],
    len: usize,
    tables: &mut Vec<u8>,
) -> Result<(), String> {
    if tables.len() < len {
        tables.resize(len, 0);
    }
    macro_rules! put {
        ($slot:ident, $values:expr) => {
            copy_table(tables, off[Table::$slot as usize] as usize, $values)?
        };
    }
    wave_table_fields!(put, w, jobs);
    Ok(())
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

/// Returns the per-arena offsets, the total float count, and the span that
/// `clear_solve_state` must re-zero once the network is built.
#[allow(clippy::type_complexity)]
fn arena_layout(w: &Wave) -> Result<([u64; N_ARENAS], usize, std::ops::Range<usize>), String> {
    let rows = w.row_node.len();
    let cfgs = w.config_job.len();
    let jobs = w.jobs.len();
    let cells = w.legal_value.len();
    let reach = w.reach_len;
    let vals = w.vals_len;
    let roots = w.jobs.last().map_or(0, |j| j.root_values.end);
    let carry_snaps = if w.meta.snapshots {
        w.meta.snap_iters.len().saturating_sub(1)
    } else {
        0
    };
    let queries = 2 * rows;
    let chunk = queries.min(TRUNK_CHUNK_ROWS);
    let cards = 2 * jobs * NTYPE * TYPE;
    let hidden = cards.max(cfgs * NSLOT * CFGH);
    let pack = (2 * jobs * NTYPE * CARD_FEATS).max(cfgs * NSLOT * (3 + TYPE));
    let sizes = [
        reach,
        reach.max(vals),
        vals,
        cells,
        cells,
        cells,
        cells,
        cards,
        2 * jobs * NTYPE * 3 * POOL,
        cfgs * D,
        cfgs * POOL,
        queries * D,
        queries * JW,
        queries * POOL,
        rows * JOIN_IN,
        rows * JW,
        rows * JW,
        rows * D,
        roots,
        (carry_snaps * w.snapshot_configs).div_ceil(2),
        chunk * NTYPE * TYPE,
        chunk * NTYPE * C,
        chunk * N_HEXES * C,
        chunk * (N_HEXES + 1) * C,
        chunk * N_HEXES * 2 * C,
        chunk * 2 * C,
        chunk * C,
        chunk * (2 * C + LOOSE),
        pack,
        hidden,
        cfgs * CFGH,
    ];
    if std::env::var_os("WARCHEST_ARENA_PROFILE").is_some() {
        let named = [
            "reach",
            "snap_reach",
            "vals",
            "regret",
            "cur",
            "sum",
            "snap_strat",
            "cards",
            "bag",
            "f",
            "g",
            "p",
            "jp",
            "pooled",
            "jin",
            "z",
            "jt",
            "h",
            "root_values",
            "carry",
            "tok",
            "ts",
            "x",
            "a",
            "mix",
            "pool",
            "gb",
            "board",
            "pack",
            "hidden",
            "cfg",
        ];
        let mut top: Vec<_> = named.iter().zip(sizes.iter()).collect();
        top.sort_by_key(|(_, &n)| std::cmp::Reverse(n));
        let report: Vec<String> = top
            .iter()
            .take(7)
            .map(|(name, &n)| format!("{name}={:.1}", n as f64 * 4.0 / 1048576.0))
            .collect();
        eprintln!(
            "v5_arena rows={rows} cfgs={cfgs} cells={cells} total_mib={:.1} {}",
            sizes.iter().sum::<usize>() as f64 * 4.0 / 1048576.0,
            report.join(" "),
        );
    }
    let mut off = [0u64; N_ARENAS];
    let mut at = 0usize;
    let place = |off: &mut [u64; N_ARENAS], at: &mut usize, a: Arena| -> Result<(), String> {
        *at = at
            .checked_add(31)
            .ok_or("wave FP32 arena alignment overflow")?
            & !31;
        off[a as usize] = *at as u64;
        *at = at
            .checked_add(sizes[a as usize])
            .ok_or("wave FP32 arena size overflow")?;
        Ok(())
    };
    for a in PERSISTENT {
        place(&mut off, &mut at, a)?;
    }
    // The trunk buffers are scratch for `build_network` and are dead before
    // `initialise` touches any CFR state, so the two phases share one region
    // and `clear_solve_state` re-zeroes the CFR side between them. The mix
    // input alone is 37 hexes by `2 * C` per row, which is most of a wave's
    // memory; keeping it alive beside the whole solve is what pushed mature
    // waves over the four-gibibyte exclusive-route boundary.
    let base = at
        .checked_add(31)
        .ok_or("wave FP32 arena alignment overflow")?
        & !31;
    let mut trunk = base;
    for a in TRUNK_SCRATCH {
        place(&mut off, &mut trunk, a)?;
    }
    let mut solve = base;
    for a in SOLVE_STATE {
        place(&mut off, &mut solve, a)?;
    }
    Ok((off, trunk.max(solve), base..solve))
}

/// Arenas that `build_network` produces and the CFR iterations then read.
const PERSISTENT: [Arena; 4] = [Arena::F, Arena::G, Arena::P, Arena::Jp];

/// Scratch that only exists while the network is being built. The card table
/// and its count-weighted embeddings belong here and not above: the trunk stem
/// and the config encoder are the only things that read them, and both run
/// before the first iteration.
const TRUNK_SCRATCH: [Arena; 13] = [
    Arena::Cards,
    Arena::Bag,
    Arena::Tok,
    Arena::Ts,
    Arena::X,
    Arena::A,
    Arena::Mix,
    Arena::Pool,
    Arena::Gb,
    Arena::Board,
    Arena::Pack,
    Arena::Hidden,
    Arena::Cfg,
];

/// State that only exists from `initialise` onwards.
const SOLVE_STATE: [Arena; 14] = [
    Arena::Reach,
    Arena::SnapReach,
    Arena::Vals,
    Arena::Regret,
    Arena::Cur,
    Arena::Sum,
    Arena::SnapStrat,
    Arena::Pooled,
    Arena::Jin,
    Arena::Z,
    Arena::Jt,
    Arena::H,
    Arena::RootValues,
    Arena::Carry,
];

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
            card_exclusive_route: false,
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

#[allow(clippy::too_many_arguments)]
fn gemm(
    blas: &Blas,
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
    let fast = blas.fast;
    static LOG_MODE: Once = Once::new();
    LOG_MODE.call_once(|| {
        eprintln!("v5 GEMM compute={}", if fast { "fast-f16" } else { "fp32" });
    });
    let result = if fast {
        use cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F;
        use cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        use cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *blas.handle.handle(),
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
                *blas.handle.handle(),
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

fn u32n(x: usize, what: &str) -> Result<u32, String> {
    u32::try_from(x).map_err(|_| format!("wave {what} exceeds u32"))
}
fn i32n(x: usize, what: &str) -> Result<i32, String> {
    i32::try_from(x).map_err(|_| format!("wave {what} exceeds i32 launch range"))
}

fn cuda_preamble() -> String {
    let bd = crate::board::board();
    let locations = bd
        .is_location
        .iter()
        .map(|&x| if x { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",");
    let mirrors = (0..N_HEXES)
        .map(|h| crate::state::mirror_hex(h).to_string())
        .collect::<Vec<_>>()
        .join(",");
    // A missing neighbour is `N_HEXES`, which addresses the zero hex the
    // activation tensor pads every row with, so the gather never branches.
    let neighbours = bd
        .neighbors
        .iter()
        .flatten()
        .map(|&n| {
            if n == crate::board::NONE {
                N_HEXES
            } else {
                n as usize
            }
            .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "#define WAVE_BLOCK {}\n#define READOUT_TILE {}\n\
         #define TYPE {}\n#define C {}\n#define BLOCKS {}\n#define D {}\n\
         #define POOL {}\n#define CFGH {}\n#define JW {}\n#define JBLOCKS {}\n\
         #define JOIN_IN {}\n#define N_LOCATIONS {}\n\
         #define N_HEXES {}\n#define NSLOT {}\n#define NTYPE {}\n#define HEX_FACTS {}\n\
         #define PILE_COUNTS {}\n#define CARD_FEATS {}\n#define CFEAT {}\n\
         #define CNORM {:.1}f\n#define MAX_COINS {:.1}f\n#define MAX_PLIES {:.1}f\n\
         #define LOOSE {}\n\
         #define GPU_ROW_BYTES {}\n#define GR_HEX_OWNER {}\n#define GR_HEX_SLOT {}\n\
         #define GR_HEX_HEIGHT {}\n#define GR_HEX_MARKER {}\n#define GR_PILES {}\n\
         #define GR_MARKERS {}\n#define GR_HAND {}\n#define GR_FD {}\n#define GR_BAG {}\n\
         #define GR_INITIATIVE {}\n#define GR_INIT_MOVED {}\n#define GR_TO_ACT {}\n#define GR_PLIES {}\n\
         static __device__ const unsigned char HEX_LOCATION[N_HEXES] = {{{locations}}};\n\
         static __device__ const unsigned char HEX_MIRROR[N_HEXES] = {{{mirrors}}};\n\
         static __device__ const unsigned char HEX_NEIGHBOUR[N_HEXES * 6] = {{{neighbours}}};\n",
        BLOCK,
        READOUT_TILE,
        TYPE, C, BLOCKS, D, POOL, CFGH, JW, JBLOCKS, JOIN_IN,
        bd.location_hexes.len(),
        N_HEXES, NSLOT, NTYPE, crate::rebel::HEX_FACTS,
        PILE_COUNTS, CARD_FEATS, CFEAT, crate::rebel::CNORM, crate::rebel::MAX_COINS,
        crate::state::MAX_MAIN_PLAYS as f32, LOOSE,
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
        let (_, floats, _) = arena_layout(&wave).expect("device arena layout");
        let allocated = floats
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(floats.max(1))
            * size_of::<f32>();
        assert_eq!(reserved, allocated);
    }
}
