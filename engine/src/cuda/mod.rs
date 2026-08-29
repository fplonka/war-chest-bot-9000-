
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::sys::{CUdevice_attribute_enum, CUfunction_attribute_enum};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchArgs, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::board::{board, N_HEXES, NONE};
use crate::contract::{Call, Prime, Reply, CARD_ROWS};
use crate::net::{
    ln_block, Net, NetLayout, NormSpan, Span, AW, BLOCKS, C, CFGH, D, JBLOCKS, JOIN_IN, JW,
    LN_ACT, LN_CFG, LN_H, LN_JOIN, LN_JOUT, LN_TRUNK, POOL, TYPE,
};
use crate::pbs::{
    CFEAT, HEX_BLOCK, HEX_CH, HEX_FACTS, LOOSE, MAX_COINS, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE,
    OFF_PILES, PILE_COUNTS, PLAYER_SCALARS, PUBFEAT, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE,
    ROW_HAND_SIZE, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_IDS,
    ROW_INITIATIVE, ROW_INIT_MOVED, ROW_PILES, ROW_PLIES, ROW_STACK_KIND, ROW_STACK_OWED,
    ROW_TO_ACT,
};
use crate::search::{Cfg, Cfr, Ent};
use crate::state::{CONT_CAP, MAX_MAIN_PLAYS, PENDING_KINDS};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

mod slot;
use slot::{Arr, Solve, DESC, FIELDS, C_CUR, C_PRIOR, C_QVAL, C_SUM, C_VISITS, R_REACH, R_VALS, B_P, B_JP, G_F, G_G, G_FP, Y_BOARD_OF, Y_COFF};

type Res<T> = Result<T, String>;

const KERNELS: &str = include_str!("kernels.cu");

macro_rules! kernels {
    ($($name:ident,)*) => {
        struct Kernels { $($name: CudaFunction,)* }

        impl Kernels {
            fn load(m: &Arc<CudaModule>) -> Res<Kernels> {
                $(let $name = {
                    let at = concat!("k_", stringify!($name));
                    m.load_function(at).map_err(|e| format!("kernel {at}: {e:?}"))?
                };)*
                Ok(Kernels { $($name,)* })
            }
        }
    };
}

kernels! {
    expand_rows,
    gelu,
    norm_ip,
    bias,
    window,
    scatter,
    seed_reach,
    avg_block,
    terminals,
    expand,
    finish,
    tokens,
    act_feats,
    prior_inputs,
    act_add,
    prior,
    hex_facts,
    type_pool,
    stem,
    trunk,
    cfg_slots,
    sum_slots,
    bag,
    leaf,
    reach_sweep,
    backprop_sweep,
}

trait LaunchUnit {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError>;
}

impl LaunchUnit for LaunchArgs<'_> {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.launch(cfg).map(|_| ())
    }
}

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

macro_rules! carved {
    ($($name:ident),* $(,)?) => { $( let $name = $name.buf.as_mut().expect("carved"); )* };
}

const THREADS: u32 = 256;

fn spread(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn rows_of(rows: usize, width: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (width.next_power_of_two().clamp(32, 256) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn warp_rows(rows: usize) -> LaunchConfig {
    const WARPS: u32 = 8;
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(WARPS).max(1), 1, 1),
        block_dim: (32, WARPS, 1),
        shared_mem_bytes: 0,
    }
}

pub struct Device {
    cards: Vec<Card>,
    n_gpus: usize,
    net: Net,
    slot_bytes: usize,
}

pub const PIPELINE: usize = 3;

fn tf32(v: f32) -> f32 {
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x1000) & 0xFFFF_E000)
}

const JOIN_K: usize = JOIN_IN.next_multiple_of(8);
const _: () = assert!(JOIN_K == 136 && JOIN_IN == 129);

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

const TRUNK_ROWS: usize = N_HEXES.next_multiple_of(16);
const TRUNK_LDS: usize = C + 4;

const TRUNK_SHARED: usize = (2 * N_HEXES + TRUNK_ROWS) * TRUNK_LDS * 4 + 3 * C * 4;
const _: () = {
    assert!(C % 32 == 0, "k_trunk wants a whole number of warps a row");
    assert!(TRUNK_ROWS % (C / 8) == 0, "k_trunk distributes whole hex rows");
};

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

const TILE: usize = 16384;

const JROWS: usize = 32;
const _: () = assert!(JROWS <= JW && JROWS % 16 == 0);

struct Shard {
    call: usize,
    at: usize,
    len: usize,
}

fn shards(sizes: &[usize], from: usize, want: usize) -> Vec<Shard> {
    let mut out = Vec::new();
    let (mut skip, mut taken) = (from, 0);
    for (call, &size) in sizes.iter().enumerate() {
        if taken == want {
            break;
        }
        if skip >= size {
            skip -= size;
            continue;
        }
        let len = (size - skip).min(want - taken);
        out.push(Shard { call, at: skip, len });
        taken += len;
        skip = 0;
    }
    out
}

fn copy<T: Copy>(src: &[T]) -> impl FnOnce(&mut [T]) -> usize + '_ {
    move |dst: &mut [T]| {
        dst[..src.len()].copy_from_slice(src);
        src.len()
    }
}

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

const WORK_BITS: u32 = 20;

struct Batch {
    trees: Wire<u64>,
    work: Wire<u32>,
    level_at: Vec<u32>,
    coff: Wire<u32>,
    part: Wire<i32>,
    local: Wire<i32>,
    base: Wire<i32>,
    prime: Wire<u32>,
    touched: Wire<i32>,
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
    fn all(&self) -> &Prefix {
        self.upto.last().expect("a batch has at least the empty prefix")
    }
}

#[derive(Default, Clone)]
struct Prefix {
    parts: u32,
    rows: usize,
    items: Vec<u32>,
    nterm: usize,
}

#[derive(Default)]
struct Pack {
    blob: Vec<u32>,
    dst: Vec<u64>,
    at: Vec<u32>,
    src: Vec<u32>,
    sum: Vec<u32>,
    moved: u32,
}

impl Pack {
    fn words(&mut self, w: &[u32]) -> u32 {
        let base = self.blob.len() as u32;
        self.blob.extend_from_slice(w);
        base
    }

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
    busy: parking_lot::Mutex<()>,
    blas: CudaBlas,
    k: Arc<Kernels>,
    solves: Arc<parking_lot::Mutex<Vec<Solve>>>,
    host: parking_lot::Mutex<Stage>,
    pack: parking_lot::Mutex<Pack>,
    down: parking_lot::Mutex<Host<u32>>,
    down_f: parking_lot::Mutex<Host<f32>>,
    batch: parking_lot::Mutex<Batch>,
    scratch: parking_lot::Mutex<Scratch>,
    w: CudaSlice<f32>,
    wt: CudaSlice<f32>,
    jw: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    nb: CudaSlice<i32>,
    card_facts: CudaSlice<f32>,
    locations: CudaSlice<u8>,
    owed: CudaSlice<f32>,
    plan: CudaSlice<i32>,
    layout: NetLayout,
}

impl Device {
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
            let fit = (usable - usable / 10) / per.max(1);
            let gpus_left = ordinals.len() - g;
            const SLOTS_KNEE: usize = 64;
            let mut n = (fit as usize).min(left / gpus_left.max(1)).min(SLOTS_KNEE);
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

    pub fn cards(&self) -> usize {
        self.n_gpus
    }

    pub fn slots(&self, gpu: usize) -> usize {
        self.cards[gpu * PIPELINE].solves.lock().len()
    }

    pub fn total_slots(&self) -> usize {
        (0..self.n_gpus).map(|g| self.slots(g)).sum()
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn slots_per_card(&self) -> usize {
        if self.n_gpus == 0 {
            0
        } else {
            self.total_slots() / self.n_gpus
        }
    }

    pub fn net(&self) -> &Net {
        &self.net
    }

    pub fn expand_rows(&self, rows: &[u8]) -> Res<Vec<f32>> {
        if rows.len() % ROW_BYTES != 0 {
            return Err("packed rows are not a multiple of ROW_BYTES".into());
        }
        let n = rows.len() / ROW_BYTES;
        if n == 0 {
            return Ok(Vec::new());
        }
        let card = &self.cards[0];
        let _busy = card.busy.lock();
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

    pub fn set_weights(&mut self, net: Net) -> Res<()> {
        if net.is_empty() {
            return Err("cannot publish empty weights to the device".into());
        }
        let flat = net.flat();
        for card in &mut self.cards {
            let _busy = card.busy.lock();
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

    pub fn run(&self, calls: &[Call], lane: usize) -> Option<Vec<Reply>> {
        let answered = self.cards[lane].round(calls).inspect_err(|e| {
            eprintln!("cuda: lane {lane}: {e}");
        });
        let mut out: Vec<Reply> = (0..calls.len()).map(|_| Reply::default()).collect();
        for (i, reply) in answered.ok()? {
            out[i] = reply;
        }
        Some(out)
    }

    pub fn resident(&self, card: usize, solve: usize) -> Res<Resident> {
        let c = &self.cards[card];
        let _busy = c.busy.lock();
        c.stream.context().bind_to_thread().map_err(err)?;
        let g = c.solves.lock();
        let s = g.get(solve).ok_or_else(|| format!("solve {solve} is not resident"))?;
        let mut h = c.down_f.lock();
        Ok(Resident {
            p: s.ent[Ent::Board as usize].get_f32(&c.stream, B_P, 0, s.ent[Ent::Board as usize].len() * D, &mut h)?,
            jp: s.ent[Ent::Board as usize].get_f32(&c.stream, B_JP, 0, s.ent[Ent::Board as usize].len() * JW, &mut h)?,
            f: s.ent[Ent::Config as usize].get_f32(&c.stream, G_F, 0, s.ent[Ent::Config as usize].len() * D, &mut h)?,
            g: s.ent[Ent::Config as usize].get_f32(&c.stream, G_G, 0, s.ent[Ent::Config as usize].len() * POOL, &mut h)?,
            fp: s.ent[Ent::Config as usize].get_f32(&c.stream, G_FP, 0, s.ent[Ent::Config as usize].len() * D, &mut h)?,
            prior: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_PRIOR, 0, s.ent[Ent::Cell as usize].len(), &mut h)?,
            cur: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_CUR, 0, s.ncells, &mut h)?,
            sum: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_SUM, 0, s.ncells, &mut h)?,
            qval: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_QVAL, 0, s.ncells, &mut h)?,
            visits: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_VISITS, 0, s.ncells, &mut h)?,
            reach: s.ent[Ent::Reach as usize].get_f32(&c.stream, R_REACH, 0, s.nreach, &mut h)?,
        })
    }
}

pub struct Resident {
    pub p: Vec<f32>,
    pub jp: Vec<f32>,
    pub f: Vec<f32>,
    pub g: Vec<f32>,
    pub fp: Vec<f32>,
    pub prior: Vec<f32>,
    pub cur: Vec<f32>,
    pub sum: Vec<f32>,
    pub qval: Vec<f32>,
    pub visits: Vec<f32>,
    pub reach: Vec<f32>,
}

struct Gpu {
    ctx: Arc<CudaContext>,
    k: Arc<Kernels>,
    torch: Arc<CudaStream>,
}

static GPUS: LazyLock<parking_lot::Mutex<HashMap<usize, Arc<Gpu>>>> =
    LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

macro_rules! defines {
    ($($name:ident),* $(,)?) => {
        vec![$(format!("-D{}={}", stringify!($name), $name)),*]
    };
}

fn compile_options(major: i32, minor: i32, trunk_blocks: usize) -> CompileOptions {
    let mut options = defines![
        ROW_BYTES, PUBFEAT, N_HEXES, HEX_CH, HEX_FACTS, HEX_BLOCK, NTYPE, NSLOT, PILE_COUNTS,
        CARD_FEATS, OFF_PILES, OFF_CARDS, OFF_LOOSE, PLAYER_SCALARS, ROW_IDS, ROW_HEX_OWNER,
        ROW_HEX_SLOT, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_PILES, ROW_HAND_SIZE, ROW_FD_SIZE,
        ROW_BAG_SIZE, ROW_INITIATIVE, ROW_INIT_MOVED, ROW_TO_ACT, ROW_PLIES, ROW_STACK_KIND,
        ROW_STACK_OWED, PENDING_KINDS, CONT_CAP, TRUNK_ROWS,
    ];
    options.extend([
        format!("--gpu-architecture=compute_{major}{minor}"),
        format!("-DTRUNK_MIN_BLOCKS={trunk_blocks}"),
        format!("-DTRUNK_C={C}"),
        format!("-DJ_ROWS={JROWS}"),
        format!("-DJ_W={JW}"),
        format!("-DJ_IN={JOIN_K}"),
        format!("-DJ_POOL={POOL}"),
        format!("-DJ_D={D}"),
        format!("-DJ_BLOCKS={JBLOCKS}"),
        format!("-DMAX_MAIN_PLAYS={}", MAX_MAIN_PLAYS as usize),
        format!("-DMAX_COINS={MAX_COINS:.1}f"),
    ]);
    CompileOptions {
        options,
        include_paths: vec!["/usr/local/cuda/include".into(), "/usr/include".into()],
        ..Default::default()
    }
}

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
        unsafe { ctx.disable_event_tracking() };
        let threads = 4 * C;
        let max_threads = ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(err)? as usize;
        if threads > max_threads {
            return Err(format!(
                "trunk channel width {C} needs {threads} threads per block; device {ordinal} allows {max_threads}"
            ));
        }
        let max_shared = ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map_err(err)? as usize;
        if TRUNK_SHARED > max_shared {
            return Err(format!(
                "trunk channel width {C} needs {TRUNK_SHARED} bytes of shared memory per block; device {ordinal} allows {max_shared}"
            ));
        }
        let sm_shared = ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
            .map_err(err)? as usize;
        let trunk_blocks = (sm_shared / TRUNK_SHARED).max(1);
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let source = slot::tree_source() + KERNELS;
        let ptx = compile_ptx_with_opts(&source, compile_options(major, minor, trunk_blocks))
            .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(err)?;
        let k = Kernels::load(&module)?;
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
        Plan { stream: None, bytes: 0 }.build(n, cfg).map_or(0, |(_, _, _, b)| b as u64)
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
            busy: parking_lot::Mutex::new(()),
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

    fn carve(&mut self, n: usize, cfg: &Cfg) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        self.stream.context().bind_to_thread().map_err(err)?;
        let plan = Plan { stream: Some(Arc::clone(&self.stream)), bytes: 0 };
        let (scratch, stage, batch, _) = plan.build(n, cfg)?;
        *self.host.lock() = stage;
        *self.scratch.lock() = scratch;
        *self.batch.lock() = batch;
        Ok(())
    }

    fn round(&self, calls: &[Call]) -> Res<Vec<(usize, Reply)>> {
        let mine: Vec<usize> = (0..calls.len()).collect();
        let mine = &mine[..];
        let _busy = self.busy.lock();
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
        fn at(stage: &'static str) -> impl Fn(String) -> String {
            move |e| format!("{stage}: {e}")
        }
        let mut pack = self.pack.lock();
        pack.clear();
        self.trunk(calls, &pick(0), &mut pack).map_err(at("trunk"))?;
        self.configs(calls, &pick(1)).map_err(at("configs"))?;
        self.tree(calls, &pick(2), &mut pack).map_err(at("tree"))?;
        self.scatter(&mut pack).map_err(at("scatter"))?;
        drop(pack);
        self.priors(calls, &pick(2)).map_err(at("priors"))?;
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

    fn slot<'g>(&self, g: &'g mut Vec<Solve>, solve: usize) -> &'g mut Solve {
        let n = g.len();
        g.get_mut(solve).unwrap_or_else(|| panic!("solve {solve} pinned to a card that holds {n} slots"))
    }

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
        let sizes: Vec<usize> = mine.iter().map(|&i| each(i).2).collect();
        let rows: usize = sizes.iter().sum();
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
                let tile = shards(&sizes, board0, n);
                stage.packed.put(s, n * ROW_BYTES, |dst| {
                    let mut wrote = 0;
                    for w in &tile {
                        let packed = each(mine[w.call]).0;
                        let a = w.at * ROW_BYTES;
                        dst[wrote..wrote + w.len * ROW_BYTES]
                            .copy_from_slice(&packed[a..a + w.len * ROW_BYTES]);
                        wrote += w.len * ROW_BYTES;
                    }
                    wrote
                })?;
                stage.card_of_row.put(s, n, |dst| {
                    let mut wrote = 0;
                    for w in &tile {
                        dst[wrote..wrote + w.len].fill((CARD_ROWS * w.call) as i32);
                        wrote += w.len;
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
            self.trunk_tile(calls, mine, &sizes, board0, n)?;
            board0 += n;
        }
        self.keep(calls, mine, pack)
    }

    fn encode_boards(&self, n: usize) -> Res<()> {
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
        let Scratch {
            piles, tokens, projected, type_pool, loose, glob, facts, occupant, x, input, ..
        } = &mut *sc;
        carved!(piles, tokens, projected, type_pool, loose, glob, facts, occupant, x, input);

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
        const SLOTS: u32 = (C / 8) as u32;
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
        .map_err(err)
    }

    fn trunk_tile(&self, calls: &[Call], mine: &[usize], sizes: &[usize], board0: usize, n: usize) -> Res<()> {
        self.encode_boards(n)?;
        let s = &self.stream;
        let mut sc = self.scratch.lock();
        sc.h.room(n * D)?;
        sc.z.room(n * JW)?;
        let Scratch { input, projected, x, h, z, .. } = &mut *sc;
        let p = h.buf.as_mut().unwrap();
        let jp = z.buf.as_mut().unwrap();
        self.run(self.layout.board_out, input.buf.as_ref().unwrap(), n, &mut *p)?;
        self.run(self.layout.join_p, p, n, &mut *jp)?;
        let mut src = 0;
        let mut g = self.solves.lock();
        for w in shards(sizes, board0, n) {
            let Call::Trunk { solve, boards_at, .. } = &calls[mine[w.call]] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            self.slot(&mut g, *solve).copy_board(
                s,
                *boards_at + w.at,
                p,
                jp,
                projected.buf.as_ref().unwrap(),
                x.buf.as_ref().unwrap(),
                src,
                w.len,
            )?;
            src += w.len;
        }
        Ok(())
    }

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
        let sizes: Vec<usize> = mine.iter().map(|&i| each(i).3).collect();
        let n: usize = sizes.iter().sum();
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
                let tile = shards(&sizes, cfg0, k);
                stage.phi.put(s, k * CFEAT, |dst| {
                    let mut wrote = 0;
                    for w in &tile {
                        let ph = each(mine[w.call]).0;
                        let a = w.at * CFEAT;
                        dst[wrote..wrote + w.len * CFEAT]
                            .copy_from_slice(&ph[a..a + w.len * CFEAT]);
                        wrote += w.len * CFEAT;
                    }
                    wrote
                })?;
                stage.owner.put(s, k, |dst| {
                    let mut wrote = 0;
                    for w in &tile {
                        let ow = each(mine[w.call]).1;
                        let base = (CARD_ROWS * w.call) as u32;
                        for (d, &q) in dst[wrote..wrote + w.len].iter_mut().zip(&ow[w.at..w.at + w.len]) {
                            *d = q + base;
                        }
                        wrote += w.len;
                    }
                    wrote
                })?;
            }
            self.config_tile(calls, mine, &sizes, cfg0, k)?;
            cfg0 += k;
        }
        Ok(())
    }

    fn config_tile(&self, calls: &[Call], mine: &[usize], sizes: &[usize], cfg0: usize, k: usize) -> Res<()> {
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
        carved!(tokens, projected, facts, h, pooled, z, bag);
        let (slots, hidden, u) = (tokens, projected, facts);
        let (f, g, fp) = (h, pooled, z);
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
        let mut src = 0;
        let mut solves = self.solves.lock();
        for w in shards(sizes, cfg0, k) {
            let Call::Configs { solve, at: base, .. } = &calls[mine[w.call]] else {
                unreachable!("config shard holds only config calls")
            };
            self.slot(&mut solves, *solve).copy_cfg(s, *base + w.at, f, g, fp, src, w.len)?;
            src += w.len;
        }
        Ok(())
    }

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
        let (mut want, mut desc, mut cells): (Vec<(u32, u32, Prime)>, Vec<u32>, Vec<u32>) =
            (Vec::new(), Vec::new(), Vec::new());
        for (p, &i) in mine.iter().filter(|&&i| !each(i).0.is_empty()).enumerate() {
            let (prime, a, c, temp) = each(i);
            let inv = (1.0f32 / temp.max(1e-6)).to_bits();
            let at = (desc.len() / crate::search::ACT_BYTES) as u32;
            let cell_at = cells.len() as u32;
            want.extend(prime.iter().map(|q| {
                let q = Prime { at: at + q.at, cell_at: cell_at + q.cell_at, ..*q };
                (p as u32, inv, q)
            }));
            desc.extend_from_slice(a);
            cells.extend_from_slice(c);
        }
        let m = want.len();
        self.lay(&solves)?;
        let mut batch = self.batch.lock();
        let mut i = 0usize;
        while i < m {
            let mut j = i;
            let mut na_c = 0usize;
            let mut wide = 0u32;
            while j < m && (j - i) < TILE && na_c + want[j].2.na as usize <= TILE {
                na_c += want[j].2.na as usize;
                wide = wide.max(want[j].2.nc);
                j += 1;
            }
            if j == i {
                na_c = want[j].2.na as usize;
                wide = want[j].2.nc;
                j = i + 1;
            }
            let (act0, cell0) = (want[i].2.at as usize, want[i].2.cell_at as usize);
            let cell1 = if j < m { want[j].2.cell_at as usize } else { cells.len() };
            let col = |f: &dyn Fn(&(u32, u32, Prime)) -> u32| -> Vec<u32> {
                want[i..j].iter().map(f).collect()
            };
            let act_node: Vec<u32> = (0..j - i)
                .flat_map(|k| std::iter::repeat(k as u32).take(want[i + k].2.na as usize))
                .collect();
            let flat: Vec<u32> = [
                col(&|r| r.0),
                col(&|r| r.2.node),
                col(&|r| r.2.row),
                col(&|r| r.2.at - want[i].2.at),
                col(&|r| r.2.cell_at - want[i].2.cell_at),
                col(&|r| r.1),
                act_node,
                desc[act0 * crate::search::ACT_BYTES..(act0 + na_c) * crate::search::ACT_BYTES].to_vec(),
                cells[cell0..cell1].to_vec(),
            ]
            .concat();
            self.prior_tile(&mut batch, &flat, j - i, na_c, cell1 - cell0, wide)?;
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
        let desc_d = at(6 * m + na, crate::search::ACT_BYTES * na);
        let cells_d = at(6 * m + (1 + crate::search::ACT_BYTES) * na, ncells);
        let l = &self.layout;
        let (na_i, m_i, d_i, aw_i) = (na as i32, m as i32, D as i32, AW as i32);
        let (ntype, nhex, chan) = (NTYPE as i32, N_HEXES as i32, C as i32);
        let mut sc = self.scratch.lock();
        sc.action.room(na * AW)?;
        sc.h.room((m * D).max(na * D))?;
        sc.facts.room(m * AW)?;
        sc.input.room(m * JOIN_IN)?;
        sc.z.room(m * JW)?;
        sc.pooled.room((m * JW).max(m * AW))?;
        let Scratch { action, h, input, z: join_z, pooled, facts, .. } = &mut *sc;
        carved!(action, h, facts, input, join_z, pooled);
        let (action_z, hbuf, board_proj) = (action, h, facts);
        let (join_in, temp) = (input, pooled);
        let kind = self.w.slice(l.act_kind..l.act_kind + crate::actions::N_KINDS * AW);
        let role = self.w.slice(l.act_role..l.act_role + 5 * AW);
        launch!(self, act_feats, na * AW, batch.trees.buf(), &part_d, &row_d,
                &desc_d, &act_node_d, &kind, &role, &mut *action_z,
                &na_i, &ntype, &nhex, &chan)?;
        let (pool_i, jw_i) = (POOL as i32, JW as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.prior_inputs)
                .arg(batch.trees.buf()).arg(&part_d).arg(&node_d).arg(&row_d)
                .arg(&mut *join_in).arg(&mut *hbuf).arg(&mut *join_z)
                .arg(&m_i).arg(&pool_i).arg(&d_i).arg(&jw_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (m as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        self.run(l.act_board, hbuf, m, &mut *board_proj)?;
        self.lin(l.join_b, join_in, m, 1.0, join_z)?;
        self.bias(l.join_b, m, join_z)?;
        for i in 0..JBLOCKS {
            let src = join_z.slice(0..m * JW);
            let mut dst = temp.slice_mut(0..m * JW);
            self.stream.memcpy_dtod(&src, &mut dst).map_err(err)?;
            self.norm(l.norms[LN_JOIN + i], m, true, temp)?;
            self.lin(l.join_w[i], temp, m, 1.0, join_z)?;
            self.bias(l.join_w[i], m, join_z)?;
        }
        {
            let src = join_z.slice(0..m * JW);
            let mut dst = temp.slice_mut(0..m * JW);
            self.stream.memcpy_dtod(&src, &mut dst).map_err(err)?;
        }
        self.norm(l.norms[LN_JOUT], m, true, temp)?;
        self.lin(l.join_out, temp, m, 1.0, hbuf)?;
        self.bias(l.join_out, m, hbuf)?;
        self.norm(l.norms[LN_H], m, false, hbuf)?;
        self.run(l.act_h, hbuf, m, &mut *temp)?;
        launch!(self, act_add, na * AW, &mut *action_z, board_proj, &act_node_d, &na_i, &aw_i)?;
        launch!(self, act_add, na * AW, &mut *action_z, temp, &act_node_d, &na_i, &aw_i)?;
        self.norm(l.norms[LN_ACT], na, true, &mut *action_z)?;
        self.run(l.act_out, action_z, na, &mut *hbuf)?;
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

    fn scatter(&self, pack: &mut Pack) -> Res<()> {
        if pack.moved == 0 {
            return Ok(());
        }
        let moved = pack.moved;
        pack.sum.push(moved);
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

    fn lay(&self, solves: &[usize]) -> Res<()> {
        let mut batch = self.batch.lock();
        let (mut desc, mut coff): (Vec<u64>, Vec<u32>) = (Vec::new(), vec![0]);
        let (mut part_of_row, mut local_row, mut base): (Vec<i32>, Vec<i32>, Vec<i32>) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut rows, mut cells, mut nterm) = (0usize, 0u32, 0usize);
        let (mut bucket, mut items): (Vec<Vec<u32>>, Vec<u32>) = (Vec::new(), Vec::new());
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

    fn value_pass(&self, b: &Batch) -> Res<()> {
        let all = b.all();
        self.reaches(b, all, 1, false, 0)?;
        self.network(b, all)?;
        self.terminals(b.trees.buf(), all)?;
        self.backprop(b, all, 1, 0, Cfr::LINEAR)
    }

    fn iterate(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        fn at(stage: &'static str) -> impl Fn(String) -> String {
            move |e| format!("{stage}: {e}")
        }
        let mut order: Vec<usize> = mine.to_vec();
        order.sort_by_key(|&i| std::cmp::Reverse(Self::asked(&calls[i]).0));
        let (rounds, puct, k) = {
            let Call::Iterate { iters, puct, cfr, .. } = &calls[order[0]] else {
                unreachable!("iterate shard holds only iterate calls")
            };
            (*iters, *puct, *cfr)
        };
        let mut sims = 0usize;
        let mut query_at = vec![0usize; calls.len()];
        let mut query_len = vec![0usize; calls.len()];
        let mut query_total = 0usize;
        {
            let mut g = self.solves.lock();
            for &i in &order {
                let Call::Iterate { solve, step, iters, expand, query, cfr, puct: p, .. } = &calls[i]
                else {
                    unreachable!("iterate shard holds only iterate calls")
                };
                assert_eq!((cfr.alpha, cfr.beta, cfr.gamma, cfr.predict, *p),
                           (k.alpha, k.beta, k.gamma, k.predict, puct),
                           "a round mixes two regret rules");
                let b = self.slot(&mut g, *solve);
                b.step = *step;
                b.todo = *iters;
                b.nexpand = *expand;
                sims = sims.max(*expand);
                query_at[i] = query_total;
                debug_assert!(
                    query.iter().all(|q| (q.iter as usize) < *iters),
                    "query pick is outside its solve's live iterations"
                );
                query_len[i] = query.iter().map(|q| q.len as usize).sum();
                query_total += query_len[i];
            }
        }
        let solves: Vec<usize> = order.iter().map(|&i| calls[i].solve()).collect();
        self.lay(&solves).map_err(at("lay"))?;
        let b = self.batch.lock();
        self.scratch.lock().queries.room(query_total)?;

        self.reaches(&b, b.all(), 0, false, 0).map_err(at("reach"))?;
        for iter in 0..rounds {
            let live = order
                .iter()
                .position(|&i| Self::asked(&calls[i]).0 <= iter)
                .unwrap_or(order.len());
            let p = &b.upto[live];
            let it = iter as i32;
            if query_total > 0 {
                let solves = self.solves.lock();
                let mut scratch = self.scratch.lock();
                let dst = scratch.queries.buf.as_mut().expect("query scratch is carved");
                for &i in &order[..live] {
                    let Call::Iterate { solve, query, .. } = &calls[i] else {
                        unreachable!("iterate shard holds only iterate calls")
                    };
                    let mut to = query_at[i];
                    for q in query {
                        if q.iter as usize == iter {
                            solves[*solve].ent[Ent::Reach as usize].copy_f32_to(
                                &self.stream,
                                R_REACH,
                                q.reach as usize,
                                dst,
                                to,
                                q.len as usize,
                            )?;
                        }
                        to += q.len as usize;
                    }
                }
            }
            self.network(&b, p).map_err(at("net"))?;
            self.terminals(b.trees.buf(), p).map_err(at("terminals"))?;
            self.backprop(&b, p, 0, it, k).map_err(at("backprop"))?;
            self.reaches(&b, p, 0, true, it).map_err(at("avg"))?;
            if sims > 0 {
                {
                    self.expand(b.trees.buf(), b.parts, sims, puct, iter, rounds)
                }
                .map_err(at("expand"))?;
            }
        }
        let each = b.parts as usize * sims;
        let host = self.sampled(rounds * each)?;
        let query_host = if query_total == 0 {
            Vec::new()
        } else {
            let scratch = self.scratch.lock();
            let src = scratch.queries.buf.as_ref().expect("query scratch is carved");
            self.down_f.lock().recv(&self.stream, &src.slice(..query_total))?
        };
        for (part, &i) in order.iter().enumerate() {
            let (iters, want) = Self::asked(&calls[i]);
            let mut leaves = Vec::with_capacity(iters * want);
            for phase in 0..iters {
                let at = phase * each + part * sims;
                leaves.extend_from_slice(&host[at..at + want]);
            }
            let q = query_at[i];
            let c = query_host[q..q + query_len[i]].to_vec();
            out.push((i, Reply { c, leaves, ..Default::default() }));
        }
        Ok(())
    }

    fn asked(c: &Call) -> (usize, usize) {
        match c {
            Call::Iterate { iters, expand, .. } => (*iters, *expand),
            _ => unreachable!("iterate shard holds only iterate calls"),
        }
    }

    fn read(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
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
            let Call::Read { solve, vals_at, policy_at, .. } = &calls[i] else {
                unreachable!("read shard holds only read calls")
            };
            let s = &g[*solve];
            let mut root = Vec::new();
            for &(at, n) in vals_at {
                root.extend(s.ent[Ent::Reach as usize].get_f32(&self.stream, R_VALS, at as usize, n as usize, &mut h)?);
            }
            let policy = s.ent[Ent::Cell as usize].get_f32(
                &self.stream,
                C_SUM,
                policy_at.0 as usize,
                policy_at.1 as usize,
                &mut h,
            )?;
            out.push((i, Reply { a: root, b: policy, ..Default::default() }));
        }
        Ok(())
    }

    fn grid(items: u32, split: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (items, split, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        }
    }

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

    #[allow(clippy::too_many_arguments)]
    fn network(&self, b: &Batch, p: &Prefix) -> Res<()> {
        let (trees, part_d, local_d, base_d, coff_d) =
            (b.trees.buf(), b.part.buf(), b.local.buf(), b.base.buf(), b.coff.buf());
        let stride = p.rows;
        if stride == 0 {
            return Ok(());
        }
        let l = &self.layout;
        let tile = TILE.min(stride);

        let ln = l.norms[LN_JOIN];
        let join_ln = self.ln.slice(ln.g..ln.g + 2 * (JBLOCKS + 1) * JW);
        let mut q0 = 0usize;
        while q0 < stride {
            let n = tile.min(stride - q0);
            let q0_i = q0 as i32;
            let rows = 2 * n;
            let rows_i = rows as i32;
            let bias = self.b.slice(l.value_bias..l.value_bias + 1);
            let hn = l.norms[LN_H];
            let g = self.ln.slice(hn.g..hn.g + hn.width);
            let hb = self.ln.slice(hn.b..hn.b + hn.width);
            unsafe {
                    self.stream
                        .launch_builder(&self.k.leaf)
                        .arg(trees).arg(part_d).arg(local_d).arg(base_d).arg(coff_d)
                        .arg(&self.jw).arg(&join_ln).arg(&self.owed)
                        .arg(&bias).arg(&g).arg(&hb)
                        .arg(&rows_i).arg(&q0_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: ((rows as u32).div_ceil(JROWS as u32), 1, 1),
                            block_dim: (32, (JW / 8) as u32, 1),
                            shared_mem_bytes: 0,
                        })
                }
            .map_err(err)?;
            q0 += n;
        }
        Ok(())
    }

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

    fn sampled(&self, n: usize) -> Res<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut sc = self.scratch.lock();
        let out = sc.leaves.room(n)?;
        self.down.lock().recv(&self.stream, &out.slice(0..n))
    }

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

#[derive(Default)]
struct Scratch {
    pooled: Arr<f32>,
    h: Arr<f32>,
    z: Arr<f32>,
    input: Arr<f32>,
    leaves: Arr<u32>,
    queries: Arr<f32>,
    piles: Arr<f32>,
    tokens: Arr<f32>,
    projected: Arr<f32>,
    type_pool: Arr<f32>,
    loose: Arr<f32>,
    glob: Arr<f32>,
    facts: Arr<f32>,
    occupant: Arr<i32>,
    x: Arr<f32>,
    action: Arr<f32>,
    bag: Arr<f32>,
}

struct Plan {
    stream: Option<Arc<CudaStream>>,
    bytes: usize,
}

impl Plan {
    fn arr<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default>(
        &mut self,
        cap: usize,
    ) -> Res<Arr<T>> {
        self.bytes += cap.max(1) * std::mem::size_of::<T>();
        match &self.stream {
            Some(s) => Arr::with_cap(s, cap),
            None => Ok(Arr::default()),
        }
    }

    fn wire<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Copy>(
        &mut self,
        cap: usize,
    ) -> Res<Wire<T>> {
        self.bytes += cap.max(1) * std::mem::size_of::<T>();
        match &self.stream {
            Some(s) => Wire::with_cap(s, cap),
            None => Ok(Wire::default()),
        }
    }

    fn build(mut self, n: usize, cfg: &Cfg) -> Res<(Scratch, Stage, Batch, usize)> {
        let b = &cfg.budget;
        let (nodes, rows, cidx) = (b.cap(Ent::Node), b.cap(Ent::Row), b.cap(Ent::Cidx));
        let cards = n * CARD_ROWS * NTYPE * TYPE;
        let resident = FIELDS[0] * nodes
            + FIELDS[1] * b.cap(Ent::Cell)
            + FIELDS[2] * b.cap(Ent::Reach)
            + FIELDS[3] * b.cap(Ent::Draw)
            + FIELDS[4] * rows
            + 2 * b.cap(Ent::Config)
            + cidx;
        let scratch = Scratch {
            pooled: self.arr(2 * TILE * POOL)?,
            h: self.arr(2 * TILE * D)?,
            z: self.arr(TILE * D)?,
            input: self.arr(TILE * (2 * C + LOOSE))?,
            leaves: self.arr(n * cfg.s.max(1) as usize)?,
            queries: self.arr(n * b.cap(Ent::Config))?,
            piles: self.arr(TILE * NTYPE * PILE_COUNTS)?,
            tokens: self.arr(TILE * NTYPE * TYPE)?,
            projected: self.arr(TILE * NTYPE * C)?,
            type_pool: self.arr(TILE * C)?,
            loose: self.arr(TILE * LOOSE)?,
            glob: self.arr(TILE * C)?,
            facts: self.arr(TILE * N_HEXES * HEX_FACTS)?,
            occupant: self.arr(TILE * N_HEXES)?,
            x: self.arr(TILE * N_HEXES * C)?,
            action: self.arr(TILE * AW)?,
            bag: self.arr(n * CARD_ROWS * NTYPE * 3 * POOL)?,
        };
        let stage = Stage {
            packed: self.wire(TILE * ROW_BYTES)?,
            xpub: self.arr(TILE * PUBFEAT)?,
            cards: self.wire(cards)?,
            card_of_row: self.wire(TILE)?,
            phi: self.wire(TILE * CFEAT)?,
            owner: self.wire(TILE)?,
            cfg_cards: self.wire(cards)?,
            blob: self.wire(n * (resident + rows + cidx + 2 * rows + 1))?,
            dst: self.wire(TILE)?,
            at: self.wire(TILE)?,
            src: self.wire(TILE)?,
            start: self.wire(TILE + 1)?,
        };
        let batch = Batch {
            trees: self.wire(n * DESC)?,
            work: self.wire(n * nodes)?,
            coff: self.wire(n * (2 * rows + 1))?,
            part: self.wire(n * rows)?,
            local: self.wire(n * rows)?,
            base: self.wire(n)?,
            prime: self.wire(12 * TILE + n * b.cap(Ent::Cell))?,
            touched: self.wire(n)?,
            ..Batch::default()
        };
        Ok((scratch, stage, batch, self.bytes))
    }
}

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
