//! One solve's eight entities, views into one size-class slab.
//!
//! A `Solve` holds one slab. Each entity is a `(offset, cap)` region; the
//! per-field arrays the kernels read are views at `base + (offset + k × cap)`.
//! `reserve` / `plan` / `put` only advance a length. Growth relocates to the
//! next class that fits.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::farm::Dst;
use crate::net::{D, JW, POOL};
use crate::search::Ent;
use crate::slab::{self, CLASSES, SHAPE};

use super::{err, Host, Res, HELD};

/// One column of `struct Tree` in `kernels.cu`, in that order. Width is how
/// many lanes the entity spends; zero means the previous pointer (avg = sum).
struct Col {
    ent: Ent,
    width: usize,
    dst: Option<Dst>,
    name: &'static str,
}

const TABLE: [Col; 52] = [
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Kind), name: "kind" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Player), name: "player" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Exhausted), name: "exhausted" },
    Col { ent: Ent::Node, width: 2, dst: Some(Dst::Nc), name: "nc" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Parent), name: "parent" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Roff), name: "roff" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Voff), name: "voff" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Soff), name: "soff" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Util), name: "util" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::ChildAt), name: "child_at" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::ChildN), name: "child_n" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::Child), name: "child" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::LegalBase), name: "legal_base" },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::LegalOff), name: "legal_off" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::LegalChild), name: "legal_child" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::LegalTrans), name: "legal_trans" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::CellRow), name: "cell_row" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::CellVal), name: "cell_val" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::RevBase), name: "rev_base" },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::RevStart), name: "rev_start" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::RevSrc), name: "rev_src" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::RevCell), name: "rev_cell" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::RvdBase), name: "rvd_base" },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::RvdStart), name: "rvd_start" },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::RvdSrc), name: "rvd_src" },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::RvdP), name: "rvd_p" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::DrawBase), name: "draw_base" },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::DrawStart), name: "draw_start" },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::DrawTo), name: "draw_to" },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::DrawP), name: "draw_p" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::LevelStart), name: "level_start" },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::LevelNode), name: "level_node" },
    Col { ent: Ent::Reach, width: 1, dst: None, name: "reach" },
    Col { ent: Ent::Reach, width: 2, dst: None, name: "vals" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::Cur), name: "cur" },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "regret" },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "sum" },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "qval" },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "visits" },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::Prior), name: "prior" },
    Col { ent: Ent::Cell, width: 0, dst: None, name: "avg" },
    Col { ent: Ent::Config, width: 2, dst: Some(Dst::Rootb), name: "rootb" },
    Col { ent: Ent::Board, width: D, dst: None, name: "p" },
    Col { ent: Ent::Board, width: JW, dst: None, name: "jp" },
    Col { ent: Ent::Row, width: 1, dst: None, name: "board_of" },
    Col { ent: Ent::Config, width: D, dst: None, name: "f" },
    Col { ent: Ent::Config, width: POOL, dst: None, name: "g" },
    Col { ent: Ent::Config, width: D, dst: None, name: "fp" },
    Col { ent: Ent::Cidx, width: 1, dst: None, name: "cidx" },
    Col { ent: Ent::Row, width: 2, dst: None, name: "coff" },
    Col { ent: Ent::Row, width: 1, dst: Some(Dst::LeafNode), name: "leaf_node" },
    Col { ent: Ent::Row, width: 1, dst: Some(Dst::Term), name: "term" },
];

pub const DESC: usize = TABLE.len() + 6;

const fn fields() -> [usize; 8] {
    let mut f = [0; 8];
    let mut i = 0;
    while i < TABLE.len() {
        f[TABLE[i].ent as usize] += TABLE[i].width;
        i += 1;
    }
    f
}

pub const FIELDS: [usize; 8] = fields();
pub const NODE_FIELDS: usize = FIELDS[Ent::Node as usize];

const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn lane(e: Ent, name: &str) -> usize {
    let mut acc = [0usize; 8];
    let mut i = 0;
    while i < TABLE.len() {
        if TABLE[i].ent as usize == e as usize
            && bytes_eq(TABLE[i].name.as_bytes(), name.as_bytes())
        {
            return acc[e as usize];
        }
        acc[TABLE[i].ent as usize] += TABLE[i].width;
        i += 1;
    }
    panic!("missing column")
}

pub const C_CUR: usize = lane(Ent::Cell, "cur");
pub const C_SUM: usize = lane(Ent::Cell, "sum");
pub const C_QVAL: usize = lane(Ent::Cell, "qval");
pub const C_VISITS: usize = lane(Ent::Cell, "visits");
pub const C_PRIOR: usize = lane(Ent::Cell, "prior");
pub const R_REACH: usize = lane(Ent::Reach, "reach");
pub const R_VALS: usize = lane(Ent::Reach, "vals");
pub const B_P: usize = lane(Ent::Board, "p");
pub const B_JP: usize = lane(Ent::Board, "jp");
pub const G_F: usize = lane(Ent::Config, "f");
pub const G_G: usize = lane(Ent::Config, "g");
pub const G_FP: usize = lane(Ent::Config, "fp");
pub const Y_BOARD_OF: usize = lane(Ent::Row, "board_of");
pub const Y_COFF: usize = lane(Ent::Row, "coff");

const _: () = {
    assert!(FIELDS[0] == 18);
    assert!(FIELDS[1] == 13);
    assert!(FIELDS[2] == 7);
    assert!(FIELDS[3] == 4);
    assert!(FIELDS[4] == 5);
    assert!(FIELDS[5] == D + JW);
    assert!(FIELDS[6] == 2 * D + POOL + 2);
    assert!(FIELDS[7] == 1);
    assert!(C_CUR == 7 && C_SUM == 9 && C_PRIOR == 12);
    assert!(B_P == 0 && B_JP == D);
    assert!(G_F == 2 && Y_COFF == 1);
    let mut i = 0;
    while i < 8 {
        assert!(FIELDS[i] == slab::FIELDS[i]);
        i += 1;
    }
};

pub(super) fn dst_slot(d: Dst) -> (Ent, usize, usize) {
    let mut acc = [0usize; 8];
    for c in &TABLE {
        if c.dst == Some(d) {
            return (c.ent, acc[c.ent as usize], c.width.max(1));
        }
        acc[c.ent as usize] += c.width;
    }
    unreachable!("every Dst is a Tree column")
}

/// One device array. Slot state is an `Entity`; a round's scratch is `RoundCap`.
pub struct Arr<T> {
    pub buf: Option<CudaSlice<T>>,
    pub cap: usize,
    pub len: usize,
}

impl<T> Default for Arr<T> {
    fn default() -> Self {
        Arr { buf: None, cap: 0, len: 0 }
    }
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default> Arr<T> {
    pub fn with_cap(stream: &Arc<CudaStream>, cap: usize) -> Res<Arr<T>> {
        let cap = cap.max(1);
        let mut buf = unsafe { stream.alloc::<T>(cap) }.map_err(err)?;
        stream.memset_zeros(&mut buf).map_err(err)?;
        HELD.fetch_add(
            (cap * std::mem::size_of::<T>()) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(Arr { buf: Some(buf), cap, len: 0 })
    }
}

impl<T> Drop for Arr<T> {
    fn drop(&mut self) {
        if self.buf.is_some() {
            HELD.fetch_sub(
                (self.cap * std::mem::size_of::<T>()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default> Arr<T> {
    pub fn put(&mut self, stream: &Arc<CudaStream>, at: usize, host: &[T]) -> Res<()> {
        if at + host.len() > self.cap {
            return Err(format!("scratch grew past its slot: {} > {}", at + host.len(), self.cap));
        }
        self.len = self.len.max(at + host.len());
        if host.is_empty() {
            return Ok(());
        }
        let dst = self.buf.as_mut().expect("a capacity implies a buffer");
        let mut d = dst.slice_mut(at..at + host.len());
        stream.memcpy_htod(host, &mut d).map_err(err)
    }

    pub fn room(&mut self, want: usize) -> Res<&mut CudaSlice<T>> {
        if want > self.cap {
            return Err(format!("scratch grew past its slot: {want} > {}", self.cap));
        }
        self.len = self.len.max(want);
        Ok(self.buf.as_mut().expect("a capacity implies a buffer"))
    }

    pub fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        self.buf.as_ref().map_or(0, |b| b.device_ptr(stream).0)
    }
}

/// One of a solve's eight regions: `cap × nfields` words inside the slab.
pub struct Entity {
    offset: usize,
    cap: usize,
    nfields: usize,
    len: usize,
}

impl Entity {
    fn field(&self, k: usize, base: u64) -> u64 {
        base + ((self.offset + k * self.cap) * 4) as u64
    }

    fn bytes(&self) -> usize {
        self.cap * self.nfields * 4
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn rewind(&mut self) {
        self.len = 0;
    }
}

/// Everything one solve keeps on its card.
pub struct Solve {
    pub class: usize,
    buf: Option<CudaSlice<u32>>,
    pub ent: [Entity; 8],
    pub host_coff: Vec<u32>,
    pub cells: usize,
    pub rows: usize,
    pub nterm: usize,
    pub nvals: usize,
    pub ncells: usize,
    pub nreach: usize,
    pub level_start: Vec<u32>,
    pub seed: Arr<u64>,
    pub step: usize,
    pub todo: usize,
    pub nexpand: usize,
}

impl Solve {
    pub fn empty(s: &Arc<CudaStream>) -> Res<Solve> {
        Ok(Solve {
            class: 0,
            buf: None,
            ent: std::array::from_fn(|e| Entity {
                offset: 0,
                cap: 1,
                nfields: FIELDS[e],
                len: 0,
            }),
            host_coff: vec![0],
            cells: 0,
            rows: 0,
            nterm: 0,
            nvals: 0,
            ncells: 0,
            nreach: 0,
            level_start: Vec::new(),
            seed: Arr::with_cap(s, 1)?,
            step: 0,
            todo: 0,
            nexpand: 0,
        })
    }

    fn base(&self, s: &Arc<CudaStream>) -> u64 {
        self.buf.as_ref().map_or(0, |b| b.device_ptr(s).0)
    }

    pub fn field(&self, e: Ent, k: usize, s: &Arc<CudaStream>) -> u64 {
        self.ent[e as usize].field(k, self.base(s))
    }

    pub fn caps(&self) -> [usize; 8] {
        std::array::from_fn(|e| self.ent[e].cap)
    }

    pub fn lens(&self) -> [usize; 8] {
        std::array::from_fn(|e| self.ent[e].len)
    }

    fn lay(&mut self, buf: CudaSlice<u32>, class: usize, caps: [usize; 8], lens: [usize; 8]) {
        let mut off = 0;
        for e in 0..8 {
            self.ent[e] = Entity {
                offset: off,
                cap: caps[e].max(1),
                nfields: FIELDS[e],
                len: lens[e],
            };
            off += self.ent[e].cap * FIELDS[e];
        }
        debug_assert!(off * 4 <= CLASSES[class]);
        self.class = class;
        self.buf = Some(buf);
    }

    pub fn bind(&mut self, buf: CudaSlice<u32>, class: usize) {
        self.lay(buf, class, slab::caps_for(CLASSES[class], &SHAPE), [0; 8]);
    }

    pub fn take_buf(&mut self) -> Option<(usize, CudaSlice<u32>)> {
        self.buf.take().map(|b| (self.class, b))
    }

    /// Copy used prefixes into `new` and keep it. Returns the old slab.
    pub fn relocate(
        &mut self,
        mut new: CudaSlice<u32>,
        class: usize,
        caps: [usize; 8],
        s: &Arc<CudaStream>,
    ) -> Res<CudaSlice<u32>> {
        let old = self.buf.take().ok_or("relocate of an empty solve")?;
        let lens = self.lens();
        for e in 0..8 {
            let n = lens[e];
            if n == 0 {
                continue;
            }
            let src_off = self.ent[e].offset;
            let dst_off = {
                let mut o = 0;
                for i in 0..e {
                    o += caps[i].max(1) * FIELDS[i];
                }
                o
            };
            let old_cap = self.ent[e].cap;
            let new_cap = caps[e].max(1);
            for k in 0..FIELDS[e] {
                let a = src_off + k * old_cap;
                let b = dst_off + k * new_cap;
                let src = old.slice(a..a + n);
                let mut dst = new.slice_mut(b..b + n);
                s.memcpy_dtod(&src, &mut dst).map_err(err)?;
            }
        }
        self.lay(new, class, caps, lens);
        Ok(old)
    }

    /// Fit `lens`. Relocates when a cap is short. `Ok(false)` under pressure.
    pub fn fit(
        &mut self,
        pool: &mut Pool,
        lens: [usize; 8],
        s: &Arc<CudaStream>,
    ) -> Res<bool> {
        for e in 0..8 {
            if lens[e] > self.ent[e].cap {
                let caps = slab::grow_caps(&self.caps(), &lens);
                let Some(want) = slab::class_of(&caps) else {
                    pool.stops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(false);
                };
                let Some((class, buf)) = pool.take(want, false) else {
                    pool.stops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(false);
                };
                let grown = (0..8).find(|&e| lens[e] > self.ent[e].cap).unwrap_or(0);
                pool.relocations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                pool.reloc_by[grown].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let old_class = self.class;
                let old = self.relocate(buf, class, caps, s)?;
                pool.put(old_class, old);
                break;
            }
        }
        for e in 0..8 {
            self.ent[e].len = self.ent[e].len.max(lens[e]);
        }
        Ok(true)
    }

    /// The one device-side guard. False in the error when `n` misses the slab.
    pub fn reserve(&mut self, e: Ent, n: usize) -> Res<()> {
        let a = &mut self.ent[e as usize];
        if n > a.cap {
            return Err(format!("{} grew past its slot: {n} > {}", e.name(), a.cap));
        }
        a.len = a.len.max(n);
        Ok(())
    }

    pub fn rewind_leaf(&mut self) {
        self.cells = 0;
        self.rows = 0;
        self.host_coff.clear();
        self.host_coff.push(0);
        self.ent[Ent::Row as usize].rewind();
        self.ent[Ent::Board as usize].rewind();
        self.ent[Ent::Config as usize].rewind();
        self.ent[Ent::Cidx as usize].rewind();
    }

    pub fn rewind_cfr(&mut self, s: &Arc<CudaStream>) -> Res<()> {
        self.nterm = 0;
        self.nvals = 0;
        self.ncells = 0;
        self.nreach = 0;
        self.step = 0;
        self.todo = 0;
        self.nexpand = 0;
        self.level_start.clear();
        self.ent[Ent::Node as usize].rewind();
        self.ent[Ent::Draw as usize].rewind();
        self.zero_ent(Ent::Cell, s)?;
        self.zero_ent(Ent::Reach, s)?;
        Ok(())
    }

    fn zero_ent(&mut self, e: Ent, s: &Arc<CudaStream>) -> Res<()> {
        let (off, n) = {
            let a = &self.ent[e as usize];
            (a.offset, a.cap * a.nfields)
        };
        if n == 0 {
            return Ok(());
        }
        if let Some(buf) = self.buf.as_mut() {
            let mut view = buf.slice_mut(off..off + n);
            s.memset_zeros(&mut view).map_err(err)?;
        }
        self.ent[e as usize].len = 0;
        Ok(())
    }

    pub fn census(&self) -> Vec<(&'static str, usize)> {
        let mut v: Vec<_> = Ent::ALL
            .iter()
            .map(|&e| (e.name(), self.ent[e as usize].bytes()))
            .collect();
        v.push(("seed", self.seed.cap * 8));
        v.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
        v
    }

    pub fn bytes(&self) -> usize {
        CLASSES.get(self.class).copied().unwrap_or(0) + self.seed.cap * 8
    }

    pub fn used_bytes(&self) -> usize {
        slab::words(&self.lens()) * 4
    }

    fn copy_f32(
        &mut self,
        s: &Arc<CudaStream>,
        e: Ent,
        field: usize,
        at: usize,
        src: &CudaSlice<f32>,
        from: usize,
        n: usize,
    ) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        let a = &self.ent[e as usize];
        let off = a.offset + field * a.cap + at;
        let dst = self.buf.as_mut().ok_or("copy into an empty solve")?;
        let mut view = dst.slice_mut(off..off + n);
        let mut fview = unsafe { view.transmute_mut::<f32>(n).expect("u32 and f32 are four bytes") };
        s.memcpy_dtod(&src.slice(from..from + n), &mut fview).map_err(err)
    }

    pub fn copy_board(
        &mut self,
        s: &Arc<CudaStream>,
        at: usize,
        p: &CudaSlice<f32>,
        jp: &CudaSlice<f32>,
        from: usize,
        n: usize,
    ) -> Res<()> {
        self.reserve(Ent::Board, at + n)?;
        self.copy_f32(s, Ent::Board, B_P, at * D, p, from * D, n * D)?;
        self.copy_f32(s, Ent::Board, B_JP, at * JW, jp, from * JW, n * JW)
    }

    pub fn copy_cfg(
        &mut self,
        s: &Arc<CudaStream>,
        at: usize,
        f: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        fp: &CudaSlice<f32>,
        from: usize,
        n: usize,
    ) -> Res<()> {
        self.reserve(Ent::Config, at + n)?;
        self.copy_f32(s, Ent::Config, G_F, at * D, f, from * D, n * D)?;
        self.copy_f32(s, Ent::Config, G_G, at * POOL, g, from * POOL, n * POOL)?;
        self.copy_f32(s, Ent::Config, G_FP, at * D, fp, from * D, n * D)
    }

    pub fn view(&mut self, s: &Arc<CudaStream>, e: Ent, field: usize, at: usize, n: usize, width: usize) -> Res<u64> {
        self.reserve(e, (at + n).div_ceil(width.max(1)))?;
        Ok(self.field(e, field, s))
    }

    pub fn plan(&mut self, s: &Arc<CudaStream>, d: Dst, at: usize, n: usize) -> Res<u64> {
        let (e, field, width) = dst_slot(d);
        self.view(s, e, field, at, n, width)
    }

    pub fn get_f32(
        &self,
        s: &Arc<CudaStream>,
        e: Ent,
        field: usize,
        at: usize,
        n: usize,
        host: &mut Host<f32>,
    ) -> Res<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let a = &self.ent[e as usize];
        let off = a.offset + field * a.cap + at;
        let buf = self.buf.as_ref().ok_or("reading an arena that was never written")?;
        let view = buf.slice(off..off + n);
        let fview = unsafe { view.transmute::<f32>(n).expect("u32 and f32 are four bytes") };
        host.recv(s, &fview)
    }

    pub fn describe(&self, s: &Arc<CudaStream>) -> [u64; DESC] {
        let mut out = [0u64; DESC];
        let mut acc = [0usize; 8];
        let mut i = 0;
        let base = self.base(s);
        for c in &TABLE {
            let e = c.ent as usize;
            out[i] = if c.width == 0 {
                self.ent[e].field(C_SUM, base)
            } else {
                self.ent[e].field(acc[e], base)
            };
            acc[e] += c.width;
            i += 1;
        }
        out[i] = self.seed.ptr(s);
        out[i + 1] = self.nterm as u64;
        out[i + 2] = self.nvals as u64;
        out[i + 3] = self.step as u64;
        out[i + 4] = self.todo as u64;
        out[i + 5] = self.nexpand as u64;
        out
    }
}

/// The card's size-class free lists. Carved once.
pub struct Pool {
    pub free: [Vec<CudaSlice<u32>>; 6],
    pub total: [usize; 6],
    pub relocations: std::sync::atomic::AtomicU64,
    pub reloc_by: [std::sync::atomic::AtomicU64; 8],
    pub stops: std::sync::atomic::AtomicU64,
    pub finish: parking_lot::Mutex<Vec<u32>>,
}

impl Pool {
    pub fn empty() -> Pool {
        Pool {
            free: std::array::from_fn(|_| Vec::new()),
            total: [0; 6],
            relocations: std::sync::atomic::AtomicU64::new(0),
            reloc_by: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)),
            stops: std::sync::atomic::AtomicU64::new(0),
            finish: parking_lot::Mutex::new(Vec::new()),
        }
    }

    pub fn can_admit(&self) -> bool {
        !self.free[0].is_empty() && !self.free[CLASSES.len() - 1].is_empty()
    }

    pub fn take(&mut self, want: usize, admit: bool) -> Option<(usize, CudaSlice<u32>)> {
        if admit {
            if !self.can_admit() {
                return None;
            }
            return self.free[0].pop().map(|b| (0, b));
        }
        for c in want..CLASSES.len() {
            if let Some(b) = self.free[c].pop() {
                return Some((c, b));
            }
        }
        None
    }

    pub fn put(&mut self, class: usize, buf: CudaSlice<u32>) {
        self.free[class].push(buf);
    }

    pub fn occupancy(&self) -> [usize; 6] {
        std::array::from_fn(|c| self.total[c].saturating_sub(self.free[c].len()))
    }

    pub fn note_finish(&self, bytes: usize) {
        self.finish.lock().push(bytes as u32);
    }

    pub fn take_finish(&self) -> Vec<u32> {
        std::mem::take(&mut *self.finish.lock())
    }
}
