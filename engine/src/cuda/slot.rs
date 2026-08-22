//! One solve's arenas, allocated once at the budget and never grown.
//!
//! A slot is a `Solve`. The farm pops one to admit a solve and the same slot
//! is reused for every solve that ever runs in it. `fit` / `plan` / `put` only
//! advance a length; they do not allocate.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::farm::Dst;
use crate::net::{D, JW, POOL};
use crate::search::Budget;

use super::{err, Res, HELD};

/// Fields of `struct Tree` in `kernels.cu`, in order. Every one is eight bytes
/// wide, so the descriptor is positional and needs no packing rules.
pub const DESC: usize = 58;

/// One device array of a solve's state, or of a round's scratch.
///
/// Slot arrays are allocated at the budget and never grow. Scratch arrays are
/// allocated at the round's TILE (or at `n_slots` times a budget term) the
/// same way: `room` will not allocate past `cap`.
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
    /// Allocate `cap` elements, zeroed, once.
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
    pub fn fit(&mut self, want: usize) -> Res<()> {
        if want > self.cap {
            return Err(format!("solve grew past its slot: {want} > {}", self.cap));
        }
        self.len = self.len.max(want);
        Ok(())
    }

    /// Reserve room for `n` elements at `at` and hand back where they go.
    pub fn plan(&mut self, stream: &Arc<CudaStream>, at: usize, n: usize) -> Res<u64> {
        self.fit(at + n)?;
        Ok(self.ptr(stream))
    }

    /// Write `host` at `at`. The slot must already hold the range.
    pub fn put(&mut self, stream: &Arc<CudaStream>, at: usize, host: &[T]) -> Res<()> {
        self.fit(at + host.len())?;
        if host.is_empty() {
            return Ok(());
        }
        let dst = self.buf.as_mut().expect("a capacity implies a buffer");
        let mut d = dst.slice_mut(at..at + host.len());
        stream.memcpy_htod(host, &mut d).map_err(err)
    }

    /// Copy `n` elements of `src` starting at `from` to `at`.
    pub fn copy(
        &mut self,
        stream: &Arc<CudaStream>,
        at: usize,
        src: &CudaSlice<T>,
        from: usize,
        n: usize,
    ) -> Res<()> {
        self.fit(at + n)?;
        if n == 0 {
            return Ok(());
        }
        let dst = self.buf.as_mut().expect("a capacity implies a buffer");
        let mut d = dst.slice_mut(at..at + n);
        stream.memcpy_dtod(&src.slice(from..from + n), &mut d).map_err(err)
    }

    /// Hand the buffer. Scratch is sized at carve; a round that asks for more
    /// is a bug in the sizing.
    pub fn room(&mut self, _stream: &Arc<CudaStream>, want: usize) -> Res<&mut CudaSlice<T>> {
        if want > self.cap {
            return Err(format!("scratch grew past its slot: {want} > {}", self.cap));
        }
        self.len = self.len.max(want);
        Ok(self.buf.as_mut().expect("a capacity implies a buffer"))
    }

    /// Staging: grow to `want`. The round's remainder in `round_bytes` is this.
    pub fn grow(&mut self, stream: &Arc<CudaStream>, want: usize) -> Res<&mut CudaSlice<T>> {
        if self.cap < want {
            let add = want.max(1) - self.cap;
            self.cap = want.max(1);
            self.buf = Some(unsafe { stream.alloc::<T>(self.cap) }.map_err(err)?);
            HELD.fetch_add(
                (add * std::mem::size_of::<T>()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.len = self.len.max(want);
        Ok(self.buf.as_mut().expect("a capacity implies a buffer"))
    }

    /// Rewind the length. The pages stay; the next solve overwrites them.
    pub fn rewind(&mut self) {
        self.len = 0;
    }

    /// Zero the whole allocation and rewind. For the CFR accumulators a fresh
    /// solve must not inherit.
    pub fn zero(&mut self, stream: &Arc<CudaStream>) -> Res<()> {
        if let Some(buf) = self.buf.as_mut() {
            stream.memset_zeros(buf).map_err(err)?;
        }
        self.len = 0;
        Ok(())
    }

    pub fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        self.buf.as_ref().map_or(0, |b| b.device_ptr(stream).0)
    }
}

/// A solve's tree, as the CFR kernels read it. Every array is `contract.rs`
/// verbatim.
pub struct Tree {
    pub kind: Arr<u32>,
    pub player: Arr<u32>,
    pub exhausted: Arr<u32>,
    pub nc: Arr<u32>,
    pub parent: Arr<u32>,
    pub roff: Arr<u32>,
    pub voff: Arr<u32>,
    pub soff: Arr<u32>,
    pub util: Arr<f32>,
    pub child_at: Arr<u32>,
    pub child_n: Arr<u32>,
    pub child: Arr<u32>,
    pub legal_base: Arr<u32>,
    pub legal_off: Arr<u32>,
    pub legal_child: Arr<u32>,
    pub legal_trans: Arr<u32>,
    pub cell_row: Arr<u32>,
    pub cell_val: Arr<u32>,
    pub rev_base: Arr<u32>,
    pub rev_start: Arr<u32>,
    pub rev_src: Arr<u32>,
    pub rev_cell: Arr<u32>,
    pub rvd_base: Arr<u32>,
    pub rvd_start: Arr<u32>,
    pub rvd_src: Arr<u32>,
    pub rvd_p: Arr<f32>,
    pub draw_base: Arr<u32>,
    pub draw_start: Arr<u32>,
    pub draw_to: Arr<u32>,
    pub draw_p: Arr<f32>,
    pub level_start: Arr<u32>,
    pub level_node: Arr<u32>,
}

impl Tree {
    fn at_budget(s: &Arc<CudaStream>, b: &Budget) -> Res<Tree> {
        let n = b.nodes;
        let c = b.cells;
        let r = b.reach + b.nodes;
        let d = b.draws;
        let u32 = |k| Arr::with_cap(s, k);
        let f32 = |k| Arr::with_cap(s, k);
        Ok(Tree {
            kind: u32(n)?,
            player: u32(n)?,
            exhausted: u32(n)?,
            nc: u32(2 * n)?,
            parent: u32(n)?,
            roff: u32(n)?,
            voff: u32(n)?,
            soff: u32(n)?,
            util: f32(n)?,
            child_at: u32(n)?,
            child_n: u32(n)?,
            child: u32(c)?,
            legal_base: u32(n)?,
            legal_off: u32(r)?,
            legal_child: u32(c)?,
            legal_trans: u32(c)?,
            cell_row: u32(c)?,
            cell_val: u32(c)?,
            rev_base: u32(n)?,
            rev_start: u32(c)?,
            rev_src: u32(c)?,
            rev_cell: u32(c)?,
            rvd_base: u32(n)?,
            rvd_start: u32(r)?,
            rvd_src: u32(d)?,
            rvd_p: f32(d)?,
            draw_base: u32(n)?,
            // CSR start indices, one per parent config of a chance node plus a
            // sentinel. That is `reach`, not `nodes`; sizing it at `n` was a
            // write past the slot on the first real root.
            draw_start: u32(r)?,
            draw_to: u32(d)?,
            draw_p: f32(d)?,
            level_start: u32(n)?,
            level_node: u32(n)?,
        })
    }

    fn rewind(&mut self) {
        for a in [
            &mut self.kind,
            &mut self.player,
            &mut self.exhausted,
            &mut self.nc,
            &mut self.parent,
            &mut self.roff,
            &mut self.voff,
            &mut self.soff,
            &mut self.child_at,
            &mut self.child_n,
            &mut self.child,
            &mut self.legal_base,
            &mut self.legal_off,
            &mut self.legal_child,
            &mut self.legal_trans,
            &mut self.cell_row,
            &mut self.cell_val,
            &mut self.rev_base,
            &mut self.rev_start,
            &mut self.rev_src,
            &mut self.rev_cell,
            &mut self.rvd_base,
            &mut self.rvd_start,
            &mut self.rvd_src,
            &mut self.draw_base,
            &mut self.draw_start,
            &mut self.draw_to,
            &mut self.level_start,
            &mut self.level_node,
        ] {
            a.rewind();
        }
        self.util.rewind();
        self.rvd_p.rewind();
        self.draw_p.rewind();
    }
}

/// Everything one solve keeps on its card: the network state that outlives an
/// iteration, the flat tree, and the CFR arenas themselves.
pub struct Solve {
    pub p: Arr<f32>,
    pub jp: Arr<f32>,
    pub board_of: Arr<u32>,
    pub f: Arr<f32>,
    pub g: Arr<f32>,
    pub fp: Arr<f32>,
    pub cidx: Arr<u32>,
    pub coff: Arr<u32>,
    pub host_coff: Vec<u32>,
    pub cells: usize,
    pub rows: usize,
    pub tree: Tree,
    pub reach: Arr<f32>,
    pub vals: Arr<f32>,
    pub cur: Arr<f32>,
    pub regret: Arr<f32>,
    pub sum: Arr<f32>,
    pub qval: Arr<f32>,
    pub visits: Arr<f32>,
    pub prior: Arr<f32>,
    pub rootb: Arr<f32>,
    pub leaf_node: Arr<u32>,
    pub term: Arr<u32>,
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
        let f32 = |k| Arr::with_cap(s, k);
        let u32 = |k| Arr::with_cap(s, k);
        Ok(Solve {
            p: f32(b.boards * D)?,
            jp: f32(b.boards * JW)?,
            board_of: u32(b.rows)?,
            f: f32(b.configs * D)?,
            g: f32(b.configs * POOL)?,
            fp: f32(b.configs * D)?,
            cidx: u32(b.cidx)?,
            coff: u32(2 * b.rows + 1)?,
            host_coff: vec![0],
            cells: 0,
            rows: 0,
            tree: Tree::at_budget(s, b)?,
            reach: f32(b.reach)?,
            vals: f32(2 * b.reach)?,
            cur: f32(b.cells)?,
            regret: f32(b.cells)?,
            sum: f32(b.cells)?,
            qval: f32(b.cells)?,
            visits: f32(b.cells)?,
            prior: f32(b.cells)?,
            rootb: f32(2 * b.configs)?,
            leaf_node: u32(b.rows)?,
            term: u32(b.rows)?,
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

    /// A fresh solve's leaf arrays: lengths back to zero. Trunk writes follow.
    pub fn rewind_leaf(&mut self) {
        self.cells = 0;
        self.rows = 0;
        self.host_coff.clear();
        self.host_coff.push(0);
        for a in [
            &mut self.p,
            &mut self.jp,
            &mut self.f,
            &mut self.g,
            &mut self.fp,
        ] {
            a.rewind();
        }
        self.board_of.rewind();
        self.cidx.rewind();
        self.coff.rewind();
        self.leaf_node.rewind();
        self.term.rewind();
    }

    /// A fresh solve's tree and CFR: lengths back to zero, accumulators cleared.
    pub fn rewind_cfr(&mut self, s: &Arc<CudaStream>) -> Res<()> {
        self.nterm = 0;
        self.nvals = 0;
        self.ncells = 0;
        self.nreach = 0;
        self.step = 0;
        self.todo = 0;
        self.nexpand = 0;
        self.level_start.clear();
        self.tree.rewind();
        self.reach.zero(s)?;
        self.vals.zero(s)?;
        self.cur.zero(s)?;
        self.regret.zero(s)?;
        self.sum.zero(s)?;
        self.qval.zero(s)?;
        self.visits.zero(s)?;
        self.prior.zero(s)?;
        Ok(())
    }

    pub fn census(&self) -> Vec<(&'static str, usize)> {
        let t = &self.tree;
        let f = std::mem::size_of::<f32>();
        let u = std::mem::size_of::<u32>();
        let mut v = vec![
            ("p", self.p.cap * f),
            ("jp", self.jp.cap * f),
            ("board_of", self.board_of.cap * u),
            ("f", self.f.cap * f),
            ("g", self.g.cap * f),
            ("fp", self.fp.cap * f),
            ("cidx", self.cidx.cap * u),
            ("coff", self.coff.cap * u),
            ("reach", self.reach.cap * f),
            ("vals", self.vals.cap * f),
            ("cur", self.cur.cap * f),
            ("regret", self.regret.cap * f),
            ("sum", self.sum.cap * f),
            ("qval", self.qval.cap * f),
            ("visits", self.visits.cap * f),
            ("prior", self.prior.cap * f),
            ("rootb", self.rootb.cap * f),
            ("leaf_node", self.leaf_node.cap * u),
            ("term", self.term.cap * u),
            ("kind", t.kind.cap * u),
            ("player", t.player.cap * u),
            ("exhausted", t.exhausted.cap * u),
            ("nc", t.nc.cap * u),
            ("parent", t.parent.cap * u),
            ("roff", t.roff.cap * u),
            ("voff", t.voff.cap * u),
            ("soff", t.soff.cap * u),
            ("util", t.util.cap * f),
            ("child_at", t.child_at.cap * u),
            ("child_n", t.child_n.cap * u),
            ("child", t.child.cap * u),
            ("legal_base", t.legal_base.cap * u),
            ("legal_off", t.legal_off.cap * u),
            ("legal_child", t.legal_child.cap * u),
            ("legal_trans", t.legal_trans.cap * u),
            ("cell_row", t.cell_row.cap * u),
            ("cell_val", t.cell_val.cap * u),
            ("rev_base", t.rev_base.cap * u),
            ("rev_start", t.rev_start.cap * u),
            ("rev_src", t.rev_src.cap * u),
            ("rev_cell", t.rev_cell.cap * u),
            ("rvd_base", t.rvd_base.cap * u),
            ("rvd_start", t.rvd_start.cap * u),
            ("rvd_src", t.rvd_src.cap * u),
            ("rvd_p", t.rvd_p.cap * f),
            ("draw_base", t.draw_base.cap * u),
            ("draw_start", t.draw_start.cap * u),
            ("draw_to", t.draw_to.cap * u),
            ("draw_p", t.draw_p.cap * f),
            ("level_start", t.level_start.cap * u),
            ("level_node", t.level_node.cap * u),
            ("seed", self.seed.cap * 8),
        ];
        v.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
        v
    }

    pub fn bytes(&self) -> usize {
        self.census().iter().map(|&(_, b)| b).sum()
    }

    pub fn plan(&mut self, s: &Arc<CudaStream>, d: Dst, at: usize, n: usize) -> Res<u64> {
        let t = &mut self.tree;
        match d {
            Dst::Kind => t.kind.plan(s, at, n),
            Dst::Player => t.player.plan(s, at, n),
            Dst::Exhausted => t.exhausted.plan(s, at, n),
            Dst::Nc => t.nc.plan(s, at, n),
            Dst::Parent => t.parent.plan(s, at, n),
            Dst::Roff => t.roff.plan(s, at, n),
            Dst::Voff => t.voff.plan(s, at, n),
            Dst::Soff => t.soff.plan(s, at, n),
            Dst::Util => t.util.plan(s, at, n),
            Dst::ChildAt => t.child_at.plan(s, at, n),
            Dst::ChildN => t.child_n.plan(s, at, n),
            Dst::Child => t.child.plan(s, at, n),
            Dst::LegalBase => t.legal_base.plan(s, at, n),
            Dst::LegalOff => t.legal_off.plan(s, at, n),
            Dst::LegalChild => t.legal_child.plan(s, at, n),
            Dst::LegalTrans => t.legal_trans.plan(s, at, n),
            Dst::CellRow => t.cell_row.plan(s, at, n),
            Dst::CellVal => t.cell_val.plan(s, at, n),
            Dst::RevBase => t.rev_base.plan(s, at, n),
            Dst::RevStart => t.rev_start.plan(s, at, n),
            Dst::RevSrc => t.rev_src.plan(s, at, n),
            Dst::RevCell => t.rev_cell.plan(s, at, n),
            Dst::RvdBase => t.rvd_base.plan(s, at, n),
            Dst::RvdStart => t.rvd_start.plan(s, at, n),
            Dst::RvdSrc => t.rvd_src.plan(s, at, n),
            Dst::RvdP => t.rvd_p.plan(s, at, n),
            Dst::DrawBase => t.draw_base.plan(s, at, n),
            Dst::DrawStart => t.draw_start.plan(s, at, n),
            Dst::DrawTo => t.draw_to.plan(s, at, n),
            Dst::DrawP => t.draw_p.plan(s, at, n),
            Dst::LevelStart => t.level_start.plan(s, at, n),
            Dst::LevelNode => t.level_node.plan(s, at, n),
            Dst::Cur => self.cur.plan(s, at, n),
            Dst::Prior => self.prior.plan(s, at, n),
            Dst::LeafNode => self.leaf_node.plan(s, at, n),
            Dst::Term => self.term.plan(s, at, n),
            Dst::Rootb => self.rootb.plan(s, at, n),
        }
    }

    pub fn describe(&self, s: &Arc<CudaStream>) -> [u64; DESC] {
        let t = &self.tree;
        [
            t.kind.ptr(s),
            t.player.ptr(s),
            t.exhausted.ptr(s),
            t.nc.ptr(s),
            t.parent.ptr(s),
            t.roff.ptr(s),
            t.voff.ptr(s),
            t.soff.ptr(s),
            t.util.ptr(s),
            t.child_at.ptr(s),
            t.child_n.ptr(s),
            t.child.ptr(s),
            t.legal_base.ptr(s),
            t.legal_off.ptr(s),
            t.legal_child.ptr(s),
            t.legal_trans.ptr(s),
            t.cell_row.ptr(s),
            t.cell_val.ptr(s),
            t.rev_base.ptr(s),
            t.rev_start.ptr(s),
            t.rev_src.ptr(s),
            t.rev_cell.ptr(s),
            t.rvd_base.ptr(s),
            t.rvd_start.ptr(s),
            t.rvd_src.ptr(s),
            t.rvd_p.ptr(s),
            t.draw_base.ptr(s),
            t.draw_start.ptr(s),
            t.draw_to.ptr(s),
            t.draw_p.ptr(s),
            t.level_start.ptr(s),
            t.level_node.ptr(s),
            self.reach.ptr(s),
            self.vals.ptr(s),
            self.cur.ptr(s),
            self.regret.ptr(s),
            self.sum.ptr(s),
            self.qval.ptr(s),
            self.visits.ptr(s),
            self.prior.ptr(s),
            self.sum.ptr(s),
            self.rootb.ptr(s),
            self.p.ptr(s),
            self.jp.ptr(s),
            self.board_of.ptr(s),
            self.f.ptr(s),
            self.g.ptr(s),
            self.fp.ptr(s),
            self.cidx.ptr(s),
            self.coff.ptr(s),
            self.leaf_node.ptr(s),
            self.term.ptr(s),
            self.seed.ptr(s),
            self.nterm as u64,
            self.nvals as u64,
            self.step as u64,
            self.todo as u64,
            self.nexpand as u64,
        ]
    }
}
