//! One solve's eight entities, allocated once at the budget and never grown.
//!
//! A slot is a `Solve`. Each entity is one allocation, `cap × nfields × 4`
//! bytes; the per-field arrays the kernels read are views at `base + k × cap`.
//! `fit` / `plan` / `put` only advance a length.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::farm::Dst;
use crate::net::{D, JW, POOL};
use crate::search::{Budget, Ent};

use super::{err, Host, Res, HELD};

/// Fields of `struct Tree` in `kernels.cu`, in order. Every one is eight bytes
/// wide, so the descriptor is positional and needs no packing rules.
pub const DESC: usize = 58;

const fn sum(xs: &[usize]) -> usize {
    let mut s = 0;
    let mut i = 0;
    while i < xs.len() {
        s += xs[i];
        i += 1;
    }
    s
}

// kind player exhausted nc×2 parent roff voff soff util child_at child_n
// legal_base rev_base rvd_base draw_base level_start level_node
const NODE_W: [usize; 17] = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
pub const NODE_FIELDS: usize = sum(&NODE_W);
const CELL_W: [usize; 13] = [1; 13];
pub const CELL_FIELDS: usize = sum(&CELL_W);
// legal_off rev_start rvd_start draw_start reach vals×2
const REACH_W: [usize; 6] = [1, 1, 1, 1, 1, 2];
pub const REACH_FIELDS: usize = sum(&REACH_W);
const DRAW_W: [usize; 4] = [1; 4];
pub const DRAW_FIELDS: usize = sum(&DRAW_W);
// board_of leaf_node term coff×2 sentinel
const ROW_W: [usize; 5] = [1, 1, 1, 2, 1];
pub const ROW_FIELDS: usize = sum(&ROW_W);
pub const BOARD_FIELDS: usize = D + JW;
pub const CONFIG_FIELDS: usize = 2 * D + POOL + 2;
pub const CIDX_FIELDS: usize = 1;

pub const FIELDS: [usize; 8] = [
    NODE_FIELDS,
    CELL_FIELDS,
    REACH_FIELDS,
    DRAW_FIELDS,
    ROW_FIELDS,
    BOARD_FIELDS,
    CONFIG_FIELDS,
    CIDX_FIELDS,
];

// Field index (first lane) of each named array inside its entity.
const N_KIND: usize = 0;
const N_PLAYER: usize = 1;
const N_EXHAUSTED: usize = 2;
const N_NC: usize = 3;
const N_PARENT: usize = 5;
const N_ROFF: usize = 6;
const N_VOFF: usize = 7;
const N_SOFF: usize = 8;
const N_UTIL: usize = 9;
const N_CHILD_AT: usize = 10;
const N_CHILD_N: usize = 11;
const N_LEGAL_BASE: usize = 12;
const N_REV_BASE: usize = 13;
const N_RVD_BASE: usize = 14;
const N_DRAW_BASE: usize = 15;
const N_LEVEL_START: usize = 16;
const N_LEVEL_NODE: usize = 17;

const C_CHILD: usize = 0;
const C_LEGAL_CHILD: usize = 1;
const C_LEGAL_TRANS: usize = 2;
const C_CELL_ROW: usize = 3;
const C_CELL_VAL: usize = 4;
const C_REV_SRC: usize = 5;
const C_REV_CELL: usize = 6;
pub const C_CUR: usize = 7;
const C_REGRET: usize = 8;
pub const C_SUM: usize = 9;
pub const C_QVAL: usize = 10;
pub const C_VISITS: usize = 11;
pub const C_PRIOR: usize = 12;

const R_LEGAL_OFF: usize = 0;
const R_REV_START: usize = 1;
const R_RVD_START: usize = 2;
const R_DRAW_START: usize = 3;
pub const R_REACH: usize = 4;
pub const R_VALS: usize = 5;

const D_RVD_SRC: usize = 0;
const D_RVD_P: usize = 1;
const D_DRAW_TO: usize = 2;
const D_DRAW_P: usize = 3;

pub const Y_BOARD_OF: usize = 0;
const Y_LEAF_NODE: usize = 1;
const Y_TERM: usize = 2;
pub const Y_COFF: usize = 3;

pub const B_P: usize = 0;
pub const B_JP: usize = D;
pub const G_F: usize = 0;
pub const G_G: usize = D;
pub const G_FP: usize = D + POOL;
const G_ROOTB: usize = 2 * D + POOL;

/// One device array of a round's scratch. Slot state is an `Entity`; scratch
/// is still a typed buffer sized at TILE (or `n_slots` times a budget term).
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

    pub fn room(&mut self, _stream: &Arc<CudaStream>, want: usize) -> Res<&mut CudaSlice<T>> {
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

/// One of a solve's eight allocations: `cap × nfields` words.
pub struct Entity {
    buf: Option<CudaSlice<u32>>,
    cap: usize,
    nfields: usize,
    len: usize,
}

impl Entity {
    fn with_cap(stream: &Arc<CudaStream>, cap: usize, nfields: usize) -> Res<Entity> {
        let cap = cap.max(1);
        let words = cap * nfields;
        let mut buf = unsafe { stream.alloc::<u32>(words) }.map_err(err)?;
        stream.memset_zeros(&mut buf).map_err(err)?;
        HELD.fetch_add((words * 4) as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(Entity { buf: Some(buf), cap, nfields, len: 0 })
    }

    fn rewind(&mut self) {
        self.len = 0;
    }

    fn zero(&mut self, stream: &Arc<CudaStream>) -> Res<()> {
        if let Some(buf) = self.buf.as_mut() {
            stream.memset_zeros(buf).map_err(err)?;
        }
        self.len = 0;
        Ok(())
    }

    fn base(&self, stream: &Arc<CudaStream>) -> u64 {
        self.buf.as_ref().map_or(0, |b| b.device_ptr(stream).0)
    }

    pub fn field(&self, k: usize, stream: &Arc<CudaStream>) -> u64 {
        self.base(stream) + (k * self.cap * 4) as u64
    }

    fn bytes(&self) -> usize {
        self.cap * self.nfields * 4
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn copy_f32(
        &mut self,
        stream: &Arc<CudaStream>,
        field: usize,
        at: usize,
        src: &CudaSlice<f32>,
        from: usize,
        n: usize,
    ) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        let off = field * self.cap + at;
        let dst = self.buf.as_mut().expect("a capacity implies a buffer");
        let mut view = dst.slice_mut(off..off + n);
        let mut fview = unsafe { view.transmute_mut::<f32>(n).expect("u32 and f32 are four bytes") };
        stream.memcpy_dtod(&src.slice(from..from + n), &mut fview).map_err(err)
    }

    fn get_f32(
        &self,
        stream: &Arc<CudaStream>,
        field: usize,
        at: usize,
        n: usize,
        host: &mut Host<f32>,
    ) -> Res<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let off = field * self.cap + at;
        let buf = self.buf.as_ref().ok_or("reading an arena that was never written")?;
        let view = buf.slice(off..off + n);
        let fview = unsafe { view.transmute::<f32>(n).expect("u32 and f32 are four bytes") };
        host.recv(stream, &fview)
    }
}

impl Drop for Entity {
    fn drop(&mut self) {
        if self.buf.is_some() {
            HELD.fetch_sub(
                (self.cap * self.nfields * 4) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
}

/// Everything one solve keeps on its card.
pub struct Solve {
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
    pub fn at_budget(s: &Arc<CudaStream>, b: &Budget) -> Res<Solve> {
        Ok(Solve {
            ent: [
                Entity::with_cap(s, b.cap(Ent::Node), FIELDS[Ent::Node as usize])?,
                Entity::with_cap(s, b.cap(Ent::Cell), FIELDS[Ent::Cell as usize])?,
                Entity::with_cap(s, b.cap(Ent::Reach), FIELDS[Ent::Reach as usize])?,
                Entity::with_cap(s, b.cap(Ent::Draw), FIELDS[Ent::Draw as usize])?,
                Entity::with_cap(s, b.cap(Ent::Row), FIELDS[Ent::Row as usize])?,
                Entity::with_cap(s, b.cap(Ent::Board), FIELDS[Ent::Board as usize])?,
                Entity::with_cap(s, b.cap(Ent::Config), FIELDS[Ent::Config as usize])?,
                Entity::with_cap(s, b.cap(Ent::Cidx), FIELDS[Ent::Cidx as usize])?,
            ],
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

    /// The one device-side guard. False in the error when `n` misses the slot.
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
        self.ent[Ent::Cell as usize].zero(s)?;
        self.ent[Ent::Reach as usize].zero(s)?;
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
        self.census().iter().map(|&(_, b)| b).sum()
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
        let e = &mut self.ent[Ent::Board as usize];
        e.copy_f32(s, B_P, at * D, p, from * D, n * D)?;
        e.copy_f32(s, B_JP, at * JW, jp, from * JW, n * JW)
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
        let e = &mut self.ent[Ent::Config as usize];
        e.copy_f32(s, G_F, at * D, f, from * D, n * D)?;
        e.copy_f32(s, G_G, at * POOL, g, from * POOL, n * POOL)?;
        e.copy_f32(s, G_FP, at * D, fp, from * D, n * D)
    }

    pub fn view(&mut self, s: &Arc<CudaStream>, e: Ent, field: usize, at: usize, n: usize, width: usize) -> Res<u64> {
        self.reserve(e, (at + n).div_ceil(width.max(1)))?;
        Ok(self.ent[e as usize].field(field, s))
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
        self.ent[e as usize].get_f32(s, field, at, n, host)
    }

    pub fn describe(&self, s: &Arc<CudaStream>) -> [u64; DESC] {
        let n = &self.ent[Ent::Node as usize];
        let c = &self.ent[Ent::Cell as usize];
        let r = &self.ent[Ent::Reach as usize];
        let d = &self.ent[Ent::Draw as usize];
        let y = &self.ent[Ent::Row as usize];
        let b = &self.ent[Ent::Board as usize];
        let g = &self.ent[Ent::Config as usize];
        let x = &self.ent[Ent::Cidx as usize];
        [
            n.field(N_KIND, s),
            n.field(N_PLAYER, s),
            n.field(N_EXHAUSTED, s),
            n.field(N_NC, s),
            n.field(N_PARENT, s),
            n.field(N_ROFF, s),
            n.field(N_VOFF, s),
            n.field(N_SOFF, s),
            n.field(N_UTIL, s),
            n.field(N_CHILD_AT, s),
            n.field(N_CHILD_N, s),
            c.field(C_CHILD, s),
            n.field(N_LEGAL_BASE, s),
            r.field(R_LEGAL_OFF, s),
            c.field(C_LEGAL_CHILD, s),
            c.field(C_LEGAL_TRANS, s),
            c.field(C_CELL_ROW, s),
            c.field(C_CELL_VAL, s),
            n.field(N_REV_BASE, s),
            r.field(R_REV_START, s),
            c.field(C_REV_SRC, s),
            c.field(C_REV_CELL, s),
            n.field(N_RVD_BASE, s),
            r.field(R_RVD_START, s),
            d.field(D_RVD_SRC, s),
            d.field(D_RVD_P, s),
            n.field(N_DRAW_BASE, s),
            r.field(R_DRAW_START, s),
            d.field(D_DRAW_TO, s),
            d.field(D_DRAW_P, s),
            n.field(N_LEVEL_START, s),
            n.field(N_LEVEL_NODE, s),
            r.field(R_REACH, s),
            r.field(R_VALS, s),
            c.field(C_CUR, s),
            c.field(C_REGRET, s),
            c.field(C_SUM, s),
            c.field(C_QVAL, s),
            c.field(C_VISITS, s),
            c.field(C_PRIOR, s),
            c.field(C_SUM, s),
            g.field(G_ROOTB, s),
            b.field(B_P, s),
            b.field(B_JP, s),
            y.field(Y_BOARD_OF, s),
            g.field(G_F, s),
            g.field(G_G, s),
            g.field(G_FP, s),
            x.field(0, s),
            y.field(Y_COFF, s),
            y.field(Y_LEAF_NODE, s),
            y.field(Y_TERM, s),
            self.seed.ptr(s),
            self.nterm as u64,
            self.nvals as u64,
            self.step as u64,
            self.todo as u64,
            self.nexpand as u64,
        ]
    }
}

fn dst_slot(d: Dst) -> (Ent, usize, usize) {
    match d {
        Dst::Kind => (Ent::Node, N_KIND, 1),
        Dst::Player => (Ent::Node, N_PLAYER, 1),
        Dst::Exhausted => (Ent::Node, N_EXHAUSTED, 1),
        Dst::Nc => (Ent::Node, N_NC, 2),
        Dst::Parent => (Ent::Node, N_PARENT, 1),
        Dst::Roff => (Ent::Node, N_ROFF, 1),
        Dst::Voff => (Ent::Node, N_VOFF, 1),
        Dst::Soff => (Ent::Node, N_SOFF, 1),
        Dst::Util => (Ent::Node, N_UTIL, 1),
        Dst::ChildAt => (Ent::Node, N_CHILD_AT, 1),
        Dst::ChildN => (Ent::Node, N_CHILD_N, 1),
        Dst::Child => (Ent::Cell, C_CHILD, 1),
        Dst::LegalBase => (Ent::Node, N_LEGAL_BASE, 1),
        Dst::LegalOff => (Ent::Reach, R_LEGAL_OFF, 1),
        Dst::LegalChild => (Ent::Cell, C_LEGAL_CHILD, 1),
        Dst::LegalTrans => (Ent::Cell, C_LEGAL_TRANS, 1),
        Dst::CellRow => (Ent::Cell, C_CELL_ROW, 1),
        Dst::CellVal => (Ent::Cell, C_CELL_VAL, 1),
        Dst::RevBase => (Ent::Node, N_REV_BASE, 1),
        Dst::RevStart => (Ent::Reach, R_REV_START, 1),
        Dst::RevSrc => (Ent::Cell, C_REV_SRC, 1),
        Dst::RevCell => (Ent::Cell, C_REV_CELL, 1),
        Dst::RvdBase => (Ent::Node, N_RVD_BASE, 1),
        Dst::RvdStart => (Ent::Reach, R_RVD_START, 1),
        Dst::RvdSrc => (Ent::Draw, D_RVD_SRC, 1),
        Dst::RvdP => (Ent::Draw, D_RVD_P, 1),
        Dst::DrawBase => (Ent::Node, N_DRAW_BASE, 1),
        Dst::DrawStart => (Ent::Reach, R_DRAW_START, 1),
        Dst::DrawTo => (Ent::Draw, D_DRAW_TO, 1),
        Dst::DrawP => (Ent::Draw, D_DRAW_P, 1),
        Dst::LevelStart => (Ent::Node, N_LEVEL_START, 1),
        Dst::LevelNode => (Ent::Node, N_LEVEL_NODE, 1),
        Dst::Cur => (Ent::Cell, C_CUR, 1),
        Dst::Prior => (Ent::Cell, C_PRIOR, 1),
        Dst::LeafNode => (Ent::Row, Y_LEAF_NODE, 1),
        Dst::Term => (Ent::Row, Y_TERM, 1),
        Dst::Rootb => (Ent::Config, G_ROOTB, 2),
    }
}
