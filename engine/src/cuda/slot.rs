//! One solve's eight entities, allocated once at the budget and never grown.
//!
//! A slot is a `Solve`. Each entity is one allocation, `cap × nfields × 4`
//! bytes; the per-field arrays the kernels read are views at `base + k × cap`.
//! `reserve` / `plan` / `put` only advance a length.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::farm::Dst;
use crate::net::{D, JW, POOL};
use crate::search::{Budget, Ent};

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
};

fn dst_slot(d: Dst) -> (Ent, usize, usize) {
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

/// One of a solve's eight allocations: `cap × nfields` words.
pub struct Entity {
    arr: Arr<u32>,
    cap: usize,
    nfields: usize,
    len: usize,
}

impl Entity {
    fn with_cap(stream: &Arc<CudaStream>, cap: usize, nfields: usize) -> Res<Entity> {
        let cap = cap.max(1);
        Ok(Entity { arr: Arr::with_cap(stream, cap * nfields)?, cap, nfields, len: 0 })
    }

    fn rewind(&mut self) {
        self.len = 0;
    }

    fn zero(&mut self, stream: &Arc<CudaStream>) -> Res<()> {
        if let Some(buf) = self.arr.buf.as_mut() {
            stream.memset_zeros(buf).map_err(err)?;
        }
        self.len = 0;
        Ok(())
    }

    pub fn field(&self, k: usize, stream: &Arc<CudaStream>) -> u64 {
        self.arr.ptr(stream) + (k * self.cap * 4) as u64
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
        let dst = self.arr.buf.as_mut().expect("a capacity implies a buffer");
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
        let buf = self.arr.buf.as_ref().ok_or("reading an arena that was never written")?;
        let view = buf.slice(off..off + n);
        let fview = unsafe { view.transmute::<f32>(n).expect("u32 and f32 are four bytes") };
        host.recv(stream, &fview)
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
        let mut out = [0u64; DESC];
        let mut acc = [0usize; 8];
        let mut i = 0;
        for c in &TABLE {
            let e = c.ent as usize;
            out[i] = if c.width == 0 {
                self.ent[e].field(C_SUM, s)
            } else {
                self.ent[e].field(acc[e], s)
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
