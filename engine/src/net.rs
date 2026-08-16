//! The fixed production value network. `train/value_net.py` is the same
//! network in torch and `flat()` there writes exactly the blob `V5Layout`
//! reads; the two must be changed together.
//!
//! Three pieces, split by how often CFR runs them:
//!
//! * the **trunk** — 37 hex tokens through `BLOCKS` pre-activation residual
//!   blocks over the board's own adjacency, each with a global-pooling bias —
//!   runs once per leaf per solve and produces the board vector `P`;
//! * the **config encoder** runs once per distinct config in the subgame and
//!   produces `f(c)` (the readout row) and `g(c)` (the pooling vector);
//! * the **join** runs on every CFR iteration and is the only per-iteration
//!   path: it modulates `P` by the two pooled beliefs and their marginals.
//!
//! The readout is then one dot product, `v(c) = <f(c), h> + bias`.

use crate::board::{board, N_HEXES, NONE};
use crate::rebel::{
    CFEAT, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE, OFF_PILES,
    PILE_COUNTS, PUBFEAT,
};
use crate::units::CARD_FEATS;

/// Coin-type token width.
pub const TYPE: usize = 64;
/// Hex channel width.
pub const C: usize = 128;
/// Trunk residual blocks.
pub const BLOCKS: usize = 8;
/// Board vector and readout width.
pub const D: usize = 256;
/// Pooled config embedding width.
pub const POOL: usize = 64;
/// Config encoder hidden width.
pub const CFGH: usize = 128;
/// Join width.
pub const JW: usize = 128;
/// Join residual blocks.
pub const JBLOCKS: usize = 3;
/// The join input that moves between CFR iterations: both pooled beliefs.
pub const JOIN_IN: usize = 2 * POOL;
/// The model format tag `V5Layout` accepts.
pub const MODEL_TAG: [usize; 1] = [5];

#[cfg(target_vendor = "apple")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// `c[m,n] = a[m,k] * b[k,n] + beta * c[m,n]`, all row-major.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(target_vendor = "apple")]
    unsafe {
        cblas_sgemm(
            101, 111, 111, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), lda as i32, b.as_ptr(),
            ldb as i32, beta, c.as_mut_ptr(), ldc as i32,
        );
        return;
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for i in 0..m {
            let row = &mut c[i * ldc..i * ldc + n];
            if beta == 0.0 {
                row.fill(0.0);
            } else if beta != 1.0 {
                row.iter_mut().for_each(|x| *x *= beta);
            }
            for p in 0..k {
                let av = a[i * lda + p];
                if av == 0.0 {
                    continue;
                }
                let brow = &b[p * ldb..p * ldb + n];
                for (o, &bv) in row.iter_mut().zip(brow) {
                    *o += av * bv;
                }
            }
        }
    }
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

pub fn fit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

pub fn accumulate(z: &[f32], idx: &[u32], weight: &[f32], width: usize, out: &mut [f32]) {
    debug_assert_eq!(idx.len(), weight.len());
    out.fill(0.0);
    for (&i, &w) in idx.iter().zip(weight) {
        let row = &z[i as usize * width..(i as usize + 1) * width];
        for (o, &x) in out.iter_mut().zip(row) {
            *o += w * x;
        }
    }
}

// ------------------------------------------------------------------- layout

#[derive(Clone, Copy, Debug, Default)]
pub struct Span {
    pub w: usize,
    pub b: usize,
    pub i: usize,
    pub o: usize,
}

/// One trunk block: the neighbour mix, the global-pooling bias, the output.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockSpan {
    pub mix: Span,
    pub pool: Span,
    pub out: Span,
}

/// Offsets of every array in the flat blob. The order here is the contract
/// with `value_net.py::flat`.
#[derive(Clone, Debug)]
pub struct V5Layout {
    pub card: [Span; 2],
    pub pile: Span,
    pub seat: usize,
    pub hex_stem: Span,
    pub tok_stem: Span,
    pub pos: usize,
    pub glob_stem: Span,
    pub blocks: [BlockSpan; BLOCKS],
    pub board_out: Span,
    pub cfg1: Span,
    pub cfg_f: Span,
    pub cfg_g: Span,
    pub cfg_m: Span,
    pub join_p: Span,
    pub join_b: Span,
    pub join_w: [Span; JBLOCKS],
    pub join_out: Span,
    pub value_bias: usize,
    /// `(gamma, beta)` offsets, in the order the norms are applied.
    pub norms: Vec<(usize, usize)>,
    pub w_len: usize,
    pub b_len: usize,
    pub ln_len: usize,
}

/// Widths of the LayerNorms, in blob order.
fn norm_widths() -> Vec<usize> {
    let mut v: Vec<usize> = (0..BLOCKS).flat_map(|_| [C, C]).collect();
    v.push(C); // trunk
    v.push(CFGH); // config
    v.extend(std::iter::repeat(JW).take(JBLOCKS));
    v.push(JW); // join output
    v.push(D); // h
    v
}

/// Walks the three flat arrays handing out offsets in blob order.
#[derive(Default)]
struct Cursor {
    w: usize,
    b: usize,
    ln: usize,
}

impl Cursor {
    fn lin(&mut self, i: usize, o: usize, bias: bool) -> Span {
        let s = Span {
            w: self.w,
            b: if bias { self.b } else { usize::MAX },
            i,
            o,
        };
        self.w += i * o;
        if bias {
            self.b += o;
        }
        s
    }

    fn embed(&mut self, n: usize) -> usize {
        let at = self.w;
        self.w += n;
        at
    }
}

impl V5Layout {
    pub fn new(dims: &[usize]) -> Result<Self, String> {
        if dims != MODEL_TAG {
            return Err(format!(
                "unsupported model format {dims:?}; expected {MODEL_TAG:?}"
            ));
        }
        let mut c = Cursor::default();
        let card = [c.lin(CARD_FEATS, TYPE, true), c.lin(TYPE, TYPE, true)];
        let pile = c.lin(PILE_COUNTS, TYPE, false);
        let seat = c.embed(2 * TYPE);
        let hex_stem = c.lin(HEX_FACTS, C, true);
        let tok_stem = c.lin(TYPE, C, false);
        let pos = c.embed(N_HEXES * C);
        let glob_stem = c.lin(LOOSE, C, false);
        let blocks = std::array::from_fn(|_| BlockSpan {
            mix: c.lin(2 * C, C, true),
            pool: c.lin(2 * C, C, true),
            out: c.lin(C, C, true),
        });
        let board_out = c.lin(2 * C + LOOSE, D, true);
        let cfg1 = c.lin(3 + TYPE, CFGH, true);
        let cfg_f = c.lin(CFGH, D, true);
        let cfg_g = c.lin(CFGH, POOL, true);
        let cfg_m = c.lin(TYPE, 3 * POOL, false);
        let join_p = c.lin(D, JW, false);
        let join_b = c.lin(JOIN_IN, JW, true);
        let join_w = std::array::from_fn(|_| c.lin(JW, JW, true));
        let join_out = c.lin(JW, D, true);
        let value_bias = c.b;
        c.b += 1;
        let norms = norm_widths()
            .into_iter()
            .map(|width| {
                let pair = (c.ln, c.ln + width);
                c.ln += 2 * width;
                pair
            })
            .collect();
        Ok(Self {
            card,
            pile,
            seat,
            hex_stem,
            tok_stem,
            pos,
            glob_stem,
            blocks,
            board_out,
            cfg1,
            cfg_f,
            cfg_g,
            cfg_m,
            join_p,
            join_b,
            join_w,
            join_out,
            value_bias,
            norms,
            w_len: c.w,
            b_len: c.b,
            ln_len: c.ln,
        })
    }
}

// --------------------------------------------------------------------- model

#[derive(Clone, Default)]
struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    i: usize,
    o: usize,
}

impl Lin {
    /// `out = input * w + b`, growing `out` as needed.
    fn run(&self, input: &[f32], rows: usize, out: &mut Vec<f32>) {
        fit(out, rows * self.o);
        gemm(
            rows, self.o, self.i, input, self.i, &self.w, self.o, 0.0, out, self.o,
        );
        self.bias(out, rows);
    }

    /// `out += input * w`, leaving whatever was already there.
    fn add(&self, input: &[f32], rows: usize, out: &mut [f32]) {
        gemm(
            rows, self.o, self.i, input, self.i, &self.w, self.o, 1.0, out, self.o,
        );
    }

    fn bias(&self, out: &mut [f32], rows: usize) {
        if self.b.is_empty() {
            return;
        }
        for row in out[..rows * self.o].chunks_exact_mut(self.o) {
            for (x, &bias) in row.iter_mut().zip(&self.b) {
                *x += bias;
            }
        }
    }
}

#[derive(Clone, Default)]
struct Block {
    mix: Lin,
    pool: Lin,
    out: Lin,
}

#[derive(Clone, Default)]
struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    /// LayerNorm then GELU, in place.
    fn apply(&self, x: &mut [f32], rows: usize) {
        let width = self.g.len();
        for row in x[..rows * width].chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..width {
                row[j] = gelu((row[j] - mean) * inv * self.g[j] + self.b[j]);
            }
        }
    }

    /// LayerNorm alone, in place. Only the readout normalisation needs this.
    fn plain(&self, x: &mut [f32], rows: usize) {
        let width = self.g.len();
        for row in x[..rows * width].chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..width {
                row[j] = (row[j] - mean) * inv * self.g[j] + self.b[j];
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Net {
    pub dims: Vec<usize>,
    card: [Lin; 2],
    pile: Lin,
    seat: Vec<f32>,
    hex_stem: Lin,
    tok_stem: Lin,
    pos: Vec<f32>,
    glob_stem: Lin,
    blocks: Vec<Block>,
    board_out: Lin,
    cfg1: Lin,
    cfg_f: Lin,
    cfg_g: Lin,
    cfg_m: Lin,
    join_p: Lin,
    join_b: Lin,
    join_w: Vec<Lin>,
    join_out: Lin,
    value_bias: f32,
    norms: Vec<Norm>,
}

/// Index of the LayerNorm applied after a trunk block's first / second stage.
const fn ln_block(i: usize, half: usize) -> usize {
    2 * i + half
}
const LN_TRUNK: usize = 2 * BLOCKS;
const LN_CFG: usize = LN_TRUNK + 1;
const LN_JOIN: usize = LN_CFG + 1;
const LN_JOUT: usize = LN_JOIN + JBLOCKS;
const LN_H: usize = LN_JOUT + 1;

impl Net {
    pub fn from_flat(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Result<Self, String> {
        let l = V5Layout::new(dims)?;
        if (w.len(), b.len(), ln.len()) != (l.w_len, l.b_len, l.ln_len) {
            return Err(format!(
                "weight sizes {}/{}/{} do not match v5 {}/{}/{}",
                w.len(),
                b.len(),
                ln.len(),
                l.w_len,
                l.b_len,
                l.ln_len
            ));
        }
        let layer = |s: Span| Lin {
            w: w[s.w..s.w + s.i * s.o].to_vec(),
            b: if s.b == usize::MAX {
                Vec::new()
            } else {
                b[s.b..s.b + s.o].to_vec()
            },
            i: s.i,
            o: s.o,
        };
        let norms = l
            .norms
            .iter()
            .zip(norm_widths())
            .map(|(&(g, bt), width)| Norm {
                g: ln[g..g + width].to_vec(),
                b: ln[bt..bt + width].to_vec(),
            })
            .collect();
        Ok(Self {
            dims: dims.to_vec(),
            card: l.card.map(layer),
            pile: layer(l.pile),
            seat: w[l.seat..l.seat + 2 * TYPE].to_vec(),
            hex_stem: layer(l.hex_stem),
            tok_stem: layer(l.tok_stem),
            pos: w[l.pos..l.pos + N_HEXES * C].to_vec(),
            glob_stem: layer(l.glob_stem),
            blocks: l
                .blocks
                .iter()
                .map(|s| Block {
                    mix: layer(s.mix),
                    pool: layer(s.pool),
                    out: layer(s.out),
                })
                .collect(),
            board_out: layer(l.board_out),
            cfg1: layer(l.cfg1),
            cfg_f: layer(l.cfg_f),
            cfg_g: layer(l.cfg_g),
            cfg_m: layer(l.cfg_m),
            join_p: layer(l.join_p),
            join_b: layer(l.join_b),
            join_w: l.join_w.iter().map(|&s| layer(s)).collect(),
            join_out: layer(l.join_out),
            value_bias: b[l.value_bias],
            norms,
        })
    }

    pub fn load_flat_bin(path: &str) -> std::io::Result<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let raw = std::fs::read(path)?;
        let mut at = 0;
        let u32_at = |at: &mut usize| {
            let v = u32::from_le_bytes(raw[*at..*at + 4].try_into().unwrap()) as usize;
            *at += 4;
            v
        };
        let nd = u32_at(&mut at);
        let dims = (0..nd).map(|_| u32_at(&mut at)).collect();
        let mut floats = || {
            let n = u32_at(&mut at);
            let out = (0..n)
                .map(|i| f32::from_le_bytes(raw[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                .collect();
            at += 4 * n;
            out
        };
        Ok((dims, floats(), floats(), floats()))
    }

    pub fn load_bin(path: &str) -> std::io::Result<Self> {
        let (dims, w, b, ln) = Self::load_flat_bin(path)?;
        Self::from_flat(&dims, &w, &b, &ln)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }
    pub fn pub_dim(&self) -> usize {
        PUBFEAT
    }
    pub fn cfeat(&self) -> usize {
        CFEAT
    }

    // ---------------------------------------------------------------- pieces

    /// `[rows, NTYPE, TYPE]` printed-card tokens. Fixed for a whole solve, so
    /// the solver runs this on the two canonical views only.
    pub fn cards(&self, xpub: &[f32], rows: usize, out: &mut Vec<f32>) {
        let mut facts = vec![0.0; rows * NTYPE * CARD_FEATS];
        for r in 0..rows {
            let src = &xpub[r * PUBFEAT + OFF_CARDS..r * PUBFEAT + OFF_CARDS + NTYPE * CARD_FEATS];
            facts[r * NTYPE * CARD_FEATS..(r + 1) * NTYPE * CARD_FEATS].copy_from_slice(src);
        }
        let mut hidden = Vec::new();
        self.card[0].run(&facts, rows * NTYPE, &mut hidden);
        hidden[..rows * NTYPE * TYPE]
            .iter_mut()
            .for_each(|x| *x = gelu(*x));
        self.card[1].run(&hidden, rows * NTYPE, out);
    }

    /// Card tokens plus this row's pile counts and the owner's seat.
    fn tokens(&self, xpub: &[f32], cards: &[f32], rows: usize, card_rows: usize) -> Vec<f32> {
        let mut piles = vec![0.0; rows * NTYPE * PILE_COUNTS];
        let n = NTYPE * PILE_COUNTS;
        for r in 0..rows {
            piles[r * n..(r + 1) * n]
                .copy_from_slice(&xpub[r * PUBFEAT + OFF_PILES..r * PUBFEAT + OFF_PILES + n]);
        }
        let mut out = vec![0.0; rows * NTYPE * TYPE];
        self.pile.add(&piles, rows * NTYPE, &mut out);
        for r in 0..rows {
            let card = &cards[(r % card_rows) * NTYPE * TYPE..(r % card_rows + 1) * NTYPE * TYPE];
            for t in 0..NTYPE {
                let seat = &self.seat[(t / NSLOT) * TYPE..(t / NSLOT + 1) * TYPE];
                let dst = &mut out[(r * NTYPE + t) * TYPE..(r * NTYPE + t + 1) * TYPE];
                for j in 0..TYPE {
                    dst[j] += card[t * TYPE + j] + seat[j];
                }
            }
        }
        out
    }

    /// The trunk stem: hex facts, the occupant's token, position, globals.
    fn stem(&self, xpub: &[f32], tokens: &[f32], rows: usize) -> Vec<f32> {
        let mut facts = vec![0.0; rows * N_HEXES * HEX_FACTS];
        let mut occ = vec![0.0; rows * N_HEXES * TYPE];
        let mut loose = vec![0.0; rows * LOOSE];
        for r in 0..rows {
            let src = &xpub[r * PUBFEAT..(r + 1) * PUBFEAT];
            loose[r * LOOSE..(r + 1) * LOOSE].copy_from_slice(&src[OFF_LOOSE..OFF_LOOSE + LOOSE]);
            for h in 0..N_HEXES {
                let hex = &src[h * HEX_CH..(h + 1) * HEX_CH];
                let at = (r * N_HEXES + h) * HEX_FACTS;
                facts[at..at + HEX_FACTS].copy_from_slice(&hex[..HEX_FACTS]);
                if let Some(t) = hex[HEX_FACTS..].iter().position(|&v| v != 0.0) {
                    let src = &tokens[(r * NTYPE + t) * TYPE..(r * NTYPE + t + 1) * TYPE];
                    let at = (r * N_HEXES + h) * TYPE;
                    occ[at..at + TYPE].copy_from_slice(src);
                }
            }
        }
        let mut x = Vec::new();
        self.hex_stem.run(&facts, rows * N_HEXES, &mut x);
        self.tok_stem.add(&occ, rows * N_HEXES, &mut x);
        let mut glob = Vec::new();
        self.glob_stem.run(&loose, rows, &mut glob);
        for r in 0..rows {
            for h in 0..N_HEXES {
                let dst = &mut x[(r * N_HEXES + h) * C..(r * N_HEXES + h + 1) * C];
                for j in 0..C {
                    dst[j] += self.pos[h * C + j] + glob[r * C + j];
                }
            }
        }
        x.truncate(rows * N_HEXES * C);
        x
    }

    /// `[rows, N_HEXES, C]` trunk output, already normalised and activated.
    fn trunk(&self, xpub: &[f32], tokens: &[f32], rows: usize) -> Vec<f32> {
        let bd = board();
        let cells = rows * N_HEXES;
        let mut x = self.stem(xpub, tokens, rows);
        let mut a = vec![0.0; cells * C];
        let mut mixed = vec![0.0; cells * 2 * C];
        let mut pooled = vec![0.0; rows * 2 * C];
        let (mut y, mut gb, mut z) = (Vec::new(), Vec::new(), Vec::new());
        for (i, blk) in self.blocks.iter().enumerate() {
            a.copy_from_slice(&x[..cells * C]);
            self.norms[ln_block(i, 0)].apply(&mut a, cells);
            for cell in 0..cells {
                let h = cell % N_HEXES;
                let (self_part, neigh) = mixed[cell * 2 * C..(cell + 1) * 2 * C].split_at_mut(C);
                self_part.copy_from_slice(&a[cell * C..(cell + 1) * C]);
                neigh.fill(0.0);
                for &n in &bd.neighbors[h] {
                    if n == NONE {
                        continue;
                    }
                    let base = (cell - h + n as usize) * C;
                    for (o, &v) in neigh.iter_mut().zip(&a[base..base + C]) {
                        *o += v;
                    }
                }
            }
            blk.mix.run(&mixed, cells, &mut y);
            for r in 0..rows {
                let (mean, max) = pooled[r * 2 * C..(r + 1) * 2 * C].split_at_mut(C);
                mean.fill(0.0);
                max.fill(f32::NEG_INFINITY);
                for h in 0..N_HEXES {
                    let src = &a[(r * N_HEXES + h) * C..(r * N_HEXES + h + 1) * C];
                    for j in 0..C {
                        mean[j] += src[j] / N_HEXES as f32;
                        max[j] = max[j].max(src[j]);
                    }
                }
            }
            blk.pool.run(&pooled, rows, &mut gb);
            for cell in 0..cells {
                let bias = &gb[(cell / N_HEXES) * C..(cell / N_HEXES + 1) * C];
                for (o, &v) in y[cell * C..(cell + 1) * C].iter_mut().zip(bias) {
                    *o += v;
                }
            }
            self.norms[ln_block(i, 1)].apply(&mut y, cells);
            blk.out.run(&y, cells, &mut z);
            for (o, &v) in x[..cells * C].iter_mut().zip(&z[..cells * C]) {
                *o += v;
            }
        }
        self.norms[LN_TRUNK].apply(&mut x, cells);
        x
    }

    /// The board vector `P`, one row per canonical query.
    pub fn board(
        &self,
        xpub: &[f32],
        cards: &[f32],
        rows: usize,
        card_rows: usize,
        out: &mut Vec<f32>,
    ) {
        let tokens = self.tokens(xpub, cards, rows, card_rows);
        let x = self.trunk(xpub, &tokens, rows);
        let width = 2 * C + LOOSE;
        let mut input = vec![0.0; rows * width];
        for r in 0..rows {
            let dst = &mut input[r * width..(r + 1) * width];
            dst[..C].fill(0.0);
            dst[C..2 * C].fill(f32::NEG_INFINITY);
            for h in 0..N_HEXES {
                let src = &x[(r * N_HEXES + h) * C..(r * N_HEXES + h + 1) * C];
                for j in 0..C {
                    dst[j] += src[j] / N_HEXES as f32;
                    dst[C + j] = dst[C + j].max(src[j]);
                }
            }
            dst[2 * C..]
                .copy_from_slice(&xpub[r * PUBFEAT + OFF_LOOSE..r * PUBFEAT + OFF_LOOSE + LOOSE]);
        }
        self.board_out.run(&input, rows, out);
    }

    /// The half of the join's first layer that does not move between CFR
    /// iterations. Projecting `P` once per solve is the whole reason the
    /// board vector is allowed to be wide.
    pub fn join_cache(&self, p: &[f32], rows: usize, out: &mut Vec<f32>) {
        self.join_p.run(p, rows, out);
    }

    /// `f(c)` (the readout row) and `g(c)` (the pooling vector) per config.
    /// `owner` is the canonical query whose five card tokens the config reads.
    pub fn configs(
        &self,
        phi: &[f32],
        owner: &[u32],
        n: usize,
        cards: &[f32],
        f_out: &mut Vec<f32>,
        g_out: &mut Vec<f32>,
    ) {
        let width = 3 + TYPE;
        let mut slots = vec![0.0; n * NSLOT * width];
        for c in 0..n {
            let q = owner[c] as usize;
            for k in 0..NSLOT {
                let row = &mut slots[(c * NSLOT + k) * width..(c * NSLOT + k + 1) * width];
                row[0] = phi[c * CFEAT + k];
                row[1] = phi[c * CFEAT + NSLOT + k];
                row[2] = phi[c * CFEAT + 2 * NSLOT + k];
                row[3..].copy_from_slice(&cards[(q * NTYPE + k) * TYPE..(q * NTYPE + k + 1) * TYPE]);
            }
        }
        let mut hidden = Vec::new();
        self.cfg1.run(&slots, n * NSLOT, &mut hidden);
        hidden[..n * NSLOT * CFGH]
            .iter_mut()
            .for_each(|x| *x = gelu(*x));
        let mut u = vec![0.0; n * CFGH];
        for c in 0..n {
            for k in 0..NSLOT {
                for j in 0..CFGH {
                    u[c * CFGH + j] += hidden[(c * NSLOT + k) * CFGH + j];
                }
            }
        }
        self.norms[LN_CFG].apply(&mut u, n);
        self.cfg_f.run(&u, n, f_out);
        self.cfg_g.run(&u, n, g_out);
        // The linear half of `g`: a count-weighted sum of per-zone card
        // embeddings. Pooling is linear over it, so `sum_c beta(c) g(c)`
        // carries the belief's exact expected holding of every card, bound to
        // that card. `bag` depends only on the card table, so it costs two
        // rows per solve and fifteen accumulations per config.
        let mut bag = Vec::new();
        let views = cards.len() / (NTYPE * TYPE);
        self.cfg_m.run(cards, views * NTYPE, &mut bag);
        let stride = 3 * POOL;
        for c in 0..n {
            let q = owner[c] as usize;
            let dst = &mut g_out[c * POOL..(c + 1) * POOL];
            for k in 0..NSLOT {
                let v = &bag[(q * NTYPE + k) * stride..(q * NTYPE + k + 1) * stride];
                for zone in 0..3 {
                    let count = phi[c * CFEAT + zone * NSLOT + k];
                    if count == 0.0 {
                        continue;
                    }
                    for (o, &e) in dst.iter_mut().zip(&v[zone * POOL..(zone + 1) * POOL]) {
                        *o += count * e;
                    }
                }
            }
        }
    }

    /// The per-iteration path. `p`, `jp` and `pooled` are all indexed by
    /// canonical query `2 * row + player`; `out` is `[rows, D]` for the one
    /// traverser asked for.
    pub fn join(
        &self,
        p: &[f32],
        jp: &[f32],
        pooled: &[f32],
        rows: usize,
        player: usize,
        out: &mut Vec<f32>,
    ) {
        let mut z = vec![0.0; rows * JW];
        let mut input = vec![0.0; rows * JOIN_IN];
        for r in 0..rows {
            let (q, o) = (2 * r + player, 2 * r + 1 - player);
            let dst = &mut input[r * JOIN_IN..(r + 1) * JOIN_IN];
            dst[..POOL].copy_from_slice(&pooled[q * POOL..(q + 1) * POOL]);
            dst[POOL..].copy_from_slice(&pooled[o * POOL..(o + 1) * POOL]);
            z[r * JW..(r + 1) * JW].copy_from_slice(&jp[q * JW..(q + 1) * JW]);
        }
        self.join_b.add(&input, rows, &mut z);
        self.join_b.bias(&mut z, rows);
        let (mut t, mut d) = (vec![0.0; rows * JW], Vec::new());
        for i in 0..JBLOCKS {
            t.copy_from_slice(&z);
            self.norms[LN_JOIN + i].apply(&mut t, rows);
            self.join_w[i].run(&t, rows, &mut d);
            for (o, &v) in z.iter_mut().zip(&d[..rows * JW]) {
                *o += v;
            }
        }
        self.norms[LN_JOUT].apply(&mut z, rows);
        fit(out, rows * D);
        for r in 0..rows {
            let src = &p[(2 * r + player) * D..(2 * r + player + 1) * D];
            out[r * D..(r + 1) * D].copy_from_slice(src);
        }
        self.join_out.add(&z, rows, &mut out[..rows * D]);
        self.join_out.bias(out, rows);
        self.norms[LN_H].plain(out, rows);
    }

    /// `v(c) = <f(c), h> + bias` for each config index in `idx`.
    pub fn values(&self, h: &[f32], f: &[f32], idx: &[u32], out: &mut [f32]) {
        for (o, &c) in out.iter_mut().zip(idx) {
            let row = &f[c as usize * D..(c as usize + 1) * D];
            let mut v = self.value_bias;
            for j in 0..D {
                v += row[j] * h[j];
            }
            *o = v;
        }
    }

    /// The whole network on one ragged canonical-query batch. Parity tests and
    /// the offline tools use this; the solver drives the pieces directly so it
    /// can cache everything that does not move between CFR iterations.
    pub fn forward(
        &self,
        xpub: &[f32],
        phi: &[f32],
        weight: &[f32],
        seg: &[u32],
        queries: usize,
    ) -> Vec<f32> {
        let n = weight.len();
        let mut cards = Vec::new();
        self.cards(xpub, queries, &mut cards);
        let (mut p, mut jp) = (Vec::new(), Vec::new());
        self.board(xpub, &cards, queries, queries, &mut p);
        self.join_cache(&p, queries, &mut jp);
        let (mut f, mut g) = (Vec::new(), Vec::new());
        self.configs(phi, seg, n, &cards, &mut f, &mut g);
        let mut pooled = vec![0.0; queries * POOL];
        for c in 0..n {
            let q = seg[c] as usize;
            for j in 0..POOL {
                pooled[q * POOL + j] += weight[c] * g[c * POOL + j];
            }
        }
        // `join` reads query `2 * row + player`, which is how the solver lays
        // its rows out; a flat query batch is the same thing with one row per
        // pair, so it is driven twice, once per seat.
        let rows = queries / 2;
        let mut out = vec![0.0; n];
        let mut h = Vec::new();
        for player in 0..2 {
            self.join(&p, &jp, &pooled, rows, player, &mut h);
            for c in 0..n {
                let q = seg[c] as usize;
                if q % 2 != player {
                    continue;
                }
                let r = q / 2;
                self.values(&h[r * D..(r + 1) * D], &f, &[c as u32], &mut out[c..c + 1]);
            }
        }
        out
    }
}
