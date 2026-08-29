
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::board::N_HEXES;
use crate::contract::Dst;
use crate::net::{C, D, JW, POOL};
use crate::pbs::NTYPE;
use crate::search::{Budget, Ent};

use super::{err, Host, Res};

const CU: &str = "const unsigned int*";
const CF: &str = "const float*";
const FM: &str = "float*";

struct Col {
    ent: Ent,
    width: usize,
    dst: Option<Dst>,
    name: &'static str,
    ty: &'static str,
}

pub fn tree_source() -> String {
    let mut out = String::from("struct Tree {\n");
    for c in &TABLE {
        out += &format!("    {} {};\n", c.ty, c.name);
    }
    for tail in ["unsigned long long* seed", "unsigned long long nterm",
                 "unsigned long long nvals", "unsigned long long step",
                 "unsigned long long todo", "unsigned long long nexpand"] {
        out += &format!("    {tail};\n");
    }
    out + "};\n"
}

const TABLE: [Col; 54] = [
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Kind), name: "kind", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Player), name: "player", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Exhausted), name: "exhausted", ty: CU },
    Col { ent: Ent::Node, width: 2, dst: Some(Dst::Nc), name: "nc", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Parent), name: "parent", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Roff), name: "roff", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Voff), name: "voff", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Soff), name: "soff", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::Util), name: "util", ty: CF },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::ChildAt), name: "child_at", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::ChildN), name: "child_n", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::Child), name: "child", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::LegalBase), name: "legal_base", ty: CU },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::LegalOff), name: "legal_off", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::LegalChild), name: "legal_child", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::LegalTrans), name: "legal_trans", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::CellRow), name: "cell_row", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::CellVal), name: "cell_val", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::RevBase), name: "rev_base", ty: CU },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::RevStart), name: "rev_start", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::RevSrc), name: "rev_src", ty: CU },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::RevCell), name: "rev_cell", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::RvdBase), name: "rvd_base", ty: CU },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::RvdStart), name: "rvd_start", ty: CU },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::RvdSrc), name: "rvd_src", ty: CU },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::RvdP), name: "rvd_p", ty: CF },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::DrawBase), name: "draw_base", ty: CU },
    Col { ent: Ent::Reach, width: 1, dst: Some(Dst::DrawStart), name: "draw_start", ty: CU },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::DrawTo), name: "draw_to", ty: CU },
    Col { ent: Ent::Draw, width: 1, dst: Some(Dst::DrawP), name: "draw_p", ty: CF },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::LevelStart), name: "level_start", ty: CU },
    Col { ent: Ent::Node, width: 1, dst: Some(Dst::LevelNode), name: "level_node", ty: CU },
    Col { ent: Ent::Reach, width: 1, dst: None, name: "reach", ty: FM },
    Col { ent: Ent::Reach, width: 2, dst: None, name: "vals", ty: FM },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::Cur), name: "cur", ty: FM },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "regret", ty: FM },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "sum", ty: FM },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "qval", ty: FM },
    Col { ent: Ent::Cell, width: 1, dst: None, name: "visits", ty: FM },
    Col { ent: Ent::Cell, width: 1, dst: Some(Dst::Prior), name: "prior", ty: FM },
    Col { ent: Ent::Cell, width: 0, dst: None, name: "avg", ty: FM },
    Col { ent: Ent::Config, width: 2, dst: Some(Dst::Rootb), name: "rootb", ty: CF },
    Col { ent: Ent::Board, width: D, dst: None, name: "p", ty: CF },
    Col { ent: Ent::Board, width: JW, dst: None, name: "jp", ty: CF },
    Col { ent: Ent::Board, width: NTYPE * C, dst: None, name: "tokens", ty: CF },
    Col { ent: Ent::Board, width: N_HEXES * C, dst: None, name: "spatial", ty: CF },
    Col { ent: Ent::Row, width: 1, dst: None, name: "board_of", ty: CU },
    Col { ent: Ent::Config, width: D, dst: None, name: "f", ty: CF },
    Col { ent: Ent::Config, width: POOL, dst: None, name: "g", ty: CF },
    Col { ent: Ent::Config, width: D, dst: None, name: "fp", ty: CF },
    Col { ent: Ent::Cidx, width: 1, dst: None, name: "cidx", ty: CU },
    Col { ent: Ent::Row, width: 2, dst: None, name: "coff", ty: CU },
    Col { ent: Ent::Row, width: 1, dst: Some(Dst::LeafNode), name: "leaf_node", ty: CU },
    Col { ent: Ent::Row, width: 1, dst: Some(Dst::Term), name: "term", ty: CU },
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
pub const B_TOKENS: usize = lane(Ent::Board, "tokens");
pub const B_SPATIAL: usize = lane(Ent::Board, "spatial");
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
    assert!(FIELDS[5] == D + JW + NTYPE * C + N_HEXES * C);
    assert!(FIELDS[6] == 2 * D + POOL + 2);
    assert!(FIELDS[7] == 1);
    assert!(C_CUR == 7 && C_SUM == 9 && C_PRIOR == 12);
    assert!(B_P == 0 && B_JP == D);
    assert!(B_TOKENS == D + JW && B_SPATIAL == D + JW + NTYPE * C);
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
        Ok(Arr { buf: Some(buf), cap, len: 0 })
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

pub struct Entity {
    arr: Arr<u32>,
    stride: usize,
    limit: usize,
    nfields: usize,
    len: usize,
}

impl Entity {
    fn with_cap(stream: &Arc<CudaStream>, cap: usize, nfields: usize) -> Res<Entity> {
        Self::with_stride(stream, cap, cap, nfields)
    }

    fn with_stride(
        stream: &Arc<CudaStream>,
        limit: usize,
        stride: usize,
        nfields: usize,
    ) -> Res<Entity> {
        let stride = stride.max(1);
        Ok(Entity {
            arr: Arr::with_cap(stream, stride * nfields)?,
            stride,
            limit,
            nfields,
            len: 0,
        })
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
        self.arr.ptr(stream) + (k * self.stride * 4) as u64
    }

    fn bytes(&self) -> usize {
        self.stride * self.nfields * 4
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
        let off = field * self.stride + at;
        let dst = self.arr.buf.as_mut().expect("a capacity implies a buffer");
        let mut view = dst.slice_mut(off..off + n);
        let mut fview = unsafe { view.transmute_mut::<f32>(n).expect("u32 and f32 are four bytes") };
        stream.memcpy_dtod(&src.slice(from..from + n), &mut fview).map_err(err)
    }

    pub fn copy_f32_to(
        &self,
        stream: &Arc<CudaStream>,
        field: usize,
        at: usize,
        dst: &mut CudaSlice<f32>,
        to: usize,
        n: usize,
    ) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        let off = field * self.stride + at;
        let buf = self.arr.buf.as_ref().ok_or("copying an arena that was never written")?;
        let src = buf.slice(off..off + n);
        let src = unsafe { src.transmute::<f32>(n).expect("u32 and f32 are four bytes") };
        stream.memcpy_dtod(&src, &mut dst.slice_mut(to..to + n)).map_err(err)
    }

    pub fn get_f32(
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
        let off = field * self.stride + at;
        let buf = self.arr.buf.as_ref().ok_or("reading an arena that was never written")?;
        let view = buf.slice(off..off + n);
        let fview = unsafe { view.transmute::<f32>(n).expect("u32 and f32 are four bytes") };
        host.recv(stream, &fview)
    }
}

pub struct Solve {
    pub ready: cudarc::driver::CudaEvent,
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
        let ready = s.context().new_event(None).map_err(err)?;
        ready.record(s).map_err(err)?;
        Ok(Solve {
            ready,
            ent: [
                Entity::with_cap(s, b.cap(Ent::Node), FIELDS[Ent::Node as usize])?,
                Entity::with_cap(s, b.cap(Ent::Cell), FIELDS[Ent::Cell as usize])?,
                Entity::with_cap(s, b.cap(Ent::Reach), FIELDS[Ent::Reach as usize])?,
                Entity::with_cap(s, b.cap(Ent::Draw), FIELDS[Ent::Draw as usize])?,
                Entity::with_stride(
                    s,
                    b.cap(Ent::Row),
                    b.cap(Ent::Row) + 1,
                    FIELDS[Ent::Row as usize],
                )?,
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

    pub fn reserve(&mut self, e: Ent, n: usize) -> Res<()> {
        let a = &mut self.ent[e as usize];
        if n > a.limit {
            return Err(format!("{} grew past its slot: {n} > {}", e.name(), a.limit));
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

    pub fn bytes(&self) -> usize {
        self.ent.iter().map(Entity::bytes).sum::<usize>() + self.seed.cap * 8
    }

    pub fn copy_board(
        &mut self,
        s: &Arc<CudaStream>,
        at: usize,
        p: &CudaSlice<f32>,
        jp: &CudaSlice<f32>,
        tokens: &CudaSlice<f32>,
        spatial: &CudaSlice<f32>,
        from: usize,
        n: usize,
    ) -> Res<()> {
        self.reserve(Ent::Board, at + n)?;
        let e = &mut self.ent[Ent::Board as usize];
        e.copy_f32(s, B_P, at * D, p, from * D, n * D)?;
        e.copy_f32(s, B_JP, at * JW, jp, from * JW, n * JW)?;
        e.copy_f32(s, B_TOKENS, at * NTYPE * C, tokens, from * NTYPE * C, n * NTYPE * C)?;
        e.copy_f32(s, B_SPATIAL, at * N_HEXES * C, spatial, from * N_HEXES * C, n * N_HEXES * C)
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
