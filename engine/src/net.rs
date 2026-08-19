//! The fixed production value network. `train/value_net.py` writes the blob
//! that `NetLayout` reads; the two must be changed together.
//!
//! Three pieces, split by how often CFR runs them:
//!
//! * the **trunk** — one physical board per leaf through `BLOCKS`
//!   pre-activation residual blocks over the board's adjacency;
//! * the **config encoder** — one `f(c)` readout and one `g(c)` pooling vector
//!   per distinct private config;
//! * the **join** — the per-iteration path, conditioned on the two beliefs and
//!   the queried physical seat.
//!
//! The readout is one dot product, `v(c) = <f(c), h> + bias`.
//!
//! Student of Games' network is a counterfactual value-**and-policy** network,
//! `f(beta) = (v, p)`, so there is a second readout of the same shape:
//! `logit(c, a) = <f_p(c), e(a)>`. `f_p` is a third head off the same config
//! encoder that produces `f` and `g`, and `e(a)` describes one action against
//! the board it is played on. Both readouts are a config vector dotted with a
//! situation vector, which is what lets the policy reuse the `(config, action)`
//! cells CFR already indexes rather than needing a table of its own.

use crate::actions::N_KINDS;
use crate::board::{board, NONE, N_HEXES};
use crate::rebel::{
    CFEAT, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE, OFF_PILES, PILE_COUNTS,
    PUBFEAT,
};
use crate::units::CARD_FEATS;
use std::cell::RefCell;

/// Coin-type token width.
pub const TYPE: usize = 64;
/// Hex channel width.
pub const C: usize = 96;
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
/// Both pooled beliefs and the queried physical seat.
pub const JOIN_IN: usize = 2 * POOL + 1;
/// Action encoder hidden width.
pub const AW: usize = 128;
/// How an action is described to the policy head: what kind it is, which coin
/// slot it spends (or none), and the three squares it can name — where the
/// acting piece stands, where it ends up, and what it strikes.
///
/// Every one of these is public, which is what lets one description serve every
/// config at a node: an action's private content is *whether it is legal*, and
/// the tree already carries that as the legal cells.
pub const AFEAT: usize = N_KINDS + (NSLOT + 1) + 3 * (N_HEXES + 1);

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
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
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
            101,
            111,
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            beta,
            c.as_mut_ptr(),
            ldc as i32,
        );
        return;
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        for row in 0..m {
            let out = &mut c[row * ldc..row * ldc + n];
            if beta == 0.0 {
                out.fill(0.0);
            } else if beta != 1.0 {
                out.iter_mut().for_each(|value| *value *= beta);
            }
            let input = &a[row * lda..row * lda + k];
            for (&value, weights) in input
                .iter()
                .zip(b.chunks_exact(ldb))
            {
                for (dst, &weight) in out.iter_mut().zip(&weights[..n]) {
                    *dst = value.mul_add(weight, *dst);
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

fn gelu_all(x: &mut [f32], arg: &mut Vec<f32>, th: &mut Vec<f32>) {
    #[cfg(target_vendor = "apple")]
    {
        for x in x.chunks_mut(1 << 16) {
            let n = x.len();
            fit(arg, n);
            fit(th, n);
            for (a, &v) in arg[..n].iter_mut().zip(x.iter()) {
                *a = 0.797_884_56 * (v + 0.044_715 * v * v * v);
            }
            let n = n as i32;
            unsafe { vvtanhf(th.as_mut_ptr(), arg.as_ptr(), &n) };
            for (v, &t) in x.iter_mut().zip(th.iter()) {
                *v *= 0.5 * (1.0 + t);
            }
        }
        return;
    }
    #[cfg(not(target_vendor = "apple"))]
    x.iter_mut().for_each(|v| *v = gelu(*v));
}
thread_local! {
    static SCRATCH: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

fn scratch(n: usize) -> Vec<f32> {
    let mut out = SCRATCH.with(|pool| pool.borrow_mut().pop().unwrap_or_default());
    out.resize(n, 0.0);
    out[..n].fill(0.0);
    out.truncate(n);
    out
}

fn recycle(mut value: Vec<f32>) {
    value.clear();
    SCRATCH.with(|pool| pool.borrow_mut().push(value));
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

/// One LayerNorm: where its scale and shift live, and how wide it is.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormSpan {
    pub g: usize,
    pub b: usize,
    pub width: usize,
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
pub struct NetLayout {
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
    /// The policy readout's config vector, beside `cfg_f`'s value one.
    pub cfg_p: Span,
    /// The action encoder: the action's own description, the board it is
    /// played on, and the projection out to the readout's width.
    pub act_in: Span,
    pub act_board: Span,
    pub act_out: Span,
    pub join_p: Span,
    pub join_b: Span,
    pub join_w: [Span; JBLOCKS],
    pub join_out: Span,
    pub value_bias: usize,
    /// The norms in the order they are applied; `LN_*` index into this.
    pub norms: Vec<NormSpan>,
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
    v.push(AW); // action encoder
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

impl NetLayout {
    pub fn new() -> Self {
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
        let cfg_p = c.lin(CFGH, D, true);
        let act_in = c.lin(AFEAT, AW, true);
        let act_board = c.lin(D, AW, false);
        let act_out = c.lin(AW, D, true);
        let join_p = c.lin(D, JW, false);
        let join_b = c.lin(JOIN_IN, JW, true);
        let join_w = std::array::from_fn(|_| c.lin(JW, JW, true));
        let join_out = c.lin(JW, D, true);
        let value_bias = c.b;
        c.b += 1;
        let norms = norm_widths()
            .into_iter()
            .map(|width| {
                let s = NormSpan {
                    g: c.ln,
                    b: c.ln + width,
                    width,
                };
                c.ln += 2 * width;
                s
            })
            .collect();
        Self {
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
            cfg_p,
            act_in,
            act_board,
            act_out,
            join_p,
            join_b,
            join_w,
            join_out,
            value_bias,
            norms,
            w_len: c.w,
            b_len: c.b,
            ln_len: c.ln,
        }
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
    fn apply(&self, x: &mut [f32], rows: usize, arg: &mut Vec<f32>, th: &mut Vec<f32>) {
        let width = self.g.len();
        let x = &mut x[..rows * width];
        for row in x.chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..width {
                row[j] = (row[j] - mean) * inv * self.g[j] + self.b[j];
            }
        }
        gelu_all(x, arg, th);
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

/// The weights exactly as they arrived, for a backend that indexes them with
/// `NetLayout` instead of unpacking them into layers. Shared, so cloning a
/// `Net` — which happens on every publish — does not copy them.
#[derive(Default)]
pub struct Flat {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub ln: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct Net {
    flat: std::sync::Arc<Flat>,
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
    cfg_p: Lin,
    act_in: Lin,
    act_board: Lin,
    act_out: Lin,
    join_p: Lin,
    join_b: Lin,
    join_w: Vec<Lin>,
    join_out: Lin,
    value_bias: f32,
    norms: Vec<Norm>,
}
/// Index of the LayerNorm applied after a trunk block's first / second stage.
pub const fn ln_block(i: usize, half: usize) -> usize {
    2 * i + half
}
pub const LN_TRUNK: usize = 2 * BLOCKS;
pub const LN_CFG: usize = LN_TRUNK + 1;
pub const LN_JOIN: usize = LN_CFG + 1;
pub const LN_JOUT: usize = LN_JOIN + JBLOCKS;
pub const LN_H: usize = LN_JOUT + 1;
pub const LN_ACT: usize = LN_H + 1;

impl Net {
    pub fn from_flat(w: &[f32], b: &[f32], ln: &[f32]) -> Result<Self, String> {
        let l = NetLayout::new();
        if (w.len(), b.len(), ln.len()) != (l.w_len, l.b_len, l.ln_len) {
            return Err(format!(
                "weight sizes {}/{}/{} do not match the network's {}/{}/{}",
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
            .map(|s| Norm {
                g: ln[s.g..s.g + s.width].to_vec(),
                b: ln[s.b..s.b + s.width].to_vec(),
            })
            .collect();
        Ok(Self {
            flat: std::sync::Arc::new(Flat {
                w: w.to_vec(),
                b: b.to_vec(),
                ln: ln.to_vec(),
            }),
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
            cfg_p: layer(l.cfg_p),
            act_in: layer(l.act_in),
            act_board: layer(l.act_board),
            act_out: layer(l.act_out),
            join_p: layer(l.join_p),
            join_b: layer(l.join_b),
            join_w: l.join_w.iter().map(|&s| layer(s)).collect(),
            join_out: layer(l.join_out),
            value_bias: b[l.value_bias],
            norms,
        })
    }

    pub fn load_flat_bin(
        path: &str,
    ) -> std::io::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let raw = std::fs::read(path)?;
        let mut at = 0;
        let u32_at = |at: &mut usize| {
            let v = u32::from_le_bytes(raw[*at..*at + 4].try_into().unwrap()) as usize;
            *at += 4;
            v
        };
        let mut floats = || {
            let n = u32_at(&mut at);
            let out = (0..n)
                .map(|i| f32::from_le_bytes(raw[at + 4 * i..at + 4 * i + 4].try_into().unwrap()))
                .collect();
            at += 4 * n;
            out
        };
        Ok((floats(), floats(), floats()))
    }

    pub fn load_bin(path: &str) -> std::io::Result<Self> {
        let (w, b, ln) = Self::load_flat_bin(path)?;
        Self::from_flat(&w, &b, &ln)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// A default-constructed `Net` has no weights, and every caller treats
    /// that as "no value function yet".
    pub fn is_empty(&self) -> bool {
        self.board_out.w.is_empty()
    }

    /// The weights as they arrived, for the device backend.
    pub fn flat(&self) -> &Flat {
        &self.flat
    }

    // ---------------------------------------------------------------- pieces

    /// `[rows, NTYPE, TYPE]` printed-card tokens. Fixed for a whole solve, so
    /// the solver runs this on the two canonical views only.
    pub fn cards(&self, xpub: &[f32], rows: usize, out: &mut Vec<f32>) {
        let mut facts = scratch(rows * NTYPE * CARD_FEATS);
        for r in 0..rows {
            let src = &xpub[r * PUBFEAT + OFF_CARDS..r * PUBFEAT + OFF_CARDS + NTYPE * CARD_FEATS];
            facts[r * NTYPE * CARD_FEATS..(r + 1) * NTYPE * CARD_FEATS].copy_from_slice(src);
        }
        let mut hidden = scratch(0);
        self.card[0].run(&facts, rows * NTYPE, &mut hidden);
        let (mut arg, mut th) = (scratch(0), scratch(0));
        gelu_all(&mut hidden[..rows * NTYPE * TYPE], &mut arg, &mut th);
        self.card[1].run(&hidden, rows * NTYPE, out);
        recycle(facts);
        recycle(hidden);
        recycle(arg);
        recycle(th);
    }

    /// Card tokens plus this row's pile counts and the owner's seat.
    fn tokens(&self, xpub: &[f32], cards: &[f32], rows: usize, card_rows: usize) -> Vec<f32> {
        let mut piles = scratch(rows * NTYPE * PILE_COUNTS);
        let n = NTYPE * PILE_COUNTS;
        for r in 0..rows {
            piles[r * n..(r + 1) * n]
                .copy_from_slice(&xpub[r * PUBFEAT + OFF_PILES..r * PUBFEAT + OFF_PILES + n]);
        }
        let mut out = scratch(rows * NTYPE * TYPE);
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
        recycle(piles);
        out
    }

    /// The trunk stem: hex facts, the occupant's projected token, position,
    /// globals, and a nonlinear pool over every drafted coin type.
    fn stem(&self, xpub: &[f32], tokens: &[f32], rows: usize) -> Vec<f32> {
        let mut facts = scratch(rows * N_HEXES * HEX_FACTS);
        let mut occ = scratch(rows * N_HEXES * C);
        let mut loose = scratch(rows * LOOSE);
        let mut projected = scratch(0);
        self.tok_stem.run(tokens, rows * NTYPE, &mut projected);
        let mut type_pool = scratch(rows * C);
        for r in 0..rows {
            for t in 0..NTYPE {
                let token = &projected[(r * NTYPE + t) * C..(r * NTYPE + t + 1) * C];
                for j in 0..C {
                    type_pool[r * C + j] += gelu(token[j]) / NTYPE as f32;
                }
            }
            let src = &xpub[r * PUBFEAT..(r + 1) * PUBFEAT];
            loose[r * LOOSE..(r + 1) * LOOSE].copy_from_slice(&src[OFF_LOOSE..OFF_LOOSE + LOOSE]);
            for h in 0..N_HEXES {
                let hex = &src[h * HEX_CH..(h + 1) * HEX_CH];
                let at = (r * N_HEXES + h) * HEX_FACTS;
                facts[at..at + HEX_FACTS].copy_from_slice(&hex[..HEX_FACTS]);
                if let Some(t) = hex[HEX_FACTS..].iter().position(|&v| v != 0.0) {
                    let src = &projected[(r * NTYPE + t) * C..(r * NTYPE + t + 1) * C];
                    let at = (r * N_HEXES + h) * C;
                    occ[at..at + C].copy_from_slice(src);
                }
            }
        }
        let mut x = scratch(0);
        self.hex_stem.run(&facts, rows * N_HEXES, &mut x);
        let mut glob = scratch(0);
        self.glob_stem.run(&loose, rows, &mut glob);
        for r in 0..rows {
            for h in 0..N_HEXES {
                let dst = &mut x[(r * N_HEXES + h) * C..(r * N_HEXES + h + 1) * C];
                let occupant = &occ[(r * N_HEXES + h) * C..(r * N_HEXES + h + 1) * C];
                for j in 0..C {
                    dst[j] +=
                        occupant[j] + self.pos[h * C + j] + glob[r * C + j] + type_pool[r * C + j];
                }
            }
        }
        x.truncate(rows * N_HEXES * C);
        recycle(facts);
        recycle(occ);
        recycle(loose);
        recycle(projected);
        recycle(type_pool);
        recycle(glob);
        x
    }

    /// `[rows, N_HEXES, C]` trunk output, already normalised and activated.
    fn trunk(&self, xpub: &[f32], tokens: &[f32], rows: usize) -> Vec<f32> {
        let bd = board();
        let cells = rows * N_HEXES;
        let mut x = self.stem(xpub, tokens, rows);
        let mut a = scratch(cells * C);
        let mut mixed = scratch(cells * 2 * C);
        let mut pooled = scratch(rows * 2 * C);
        let (mut y, mut gb, mut z) = (scratch(0), scratch(0), scratch(0));
        let (mut arg, mut th) = (scratch(0), scratch(0));
        for (i, blk) in self.blocks.iter().enumerate() {
            a.copy_from_slice(&x[..cells * C]);
            self.norms[ln_block(i, 0)].apply(&mut a, cells, &mut arg, &mut th);
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
            self.norms[ln_block(i, 1)].apply(&mut y, cells, &mut arg, &mut th);
            blk.out.run(&y, cells, &mut z);
            for (o, &v) in x[..cells * C].iter_mut().zip(&z[..cells * C]) {
                *o += v;
            }
        }
        self.norms[LN_TRUNK].apply(&mut x, cells, &mut arg, &mut th);
        recycle(a);
        recycle(mixed);
        recycle(pooled);
        recycle(y);
        recycle(gb);
        recycle(z);
        recycle(arg);
        recycle(th);
        x
    }

    /// One board vector per physical row. `xpub` contains paired canonical
    /// queries; the trunk reads the even, physical-seat-zero rows.
    pub fn board(
        &self,
        xpub: &[f32],
        cards: &[f32],
        rows: usize,
        card_rows: usize,
        out: &mut Vec<f32>,
    ) {
        let mut physical = scratch(rows * PUBFEAT);
        let mut physical_cards = scratch(rows * NTYPE * TYPE);
        for r in 0..rows {
            physical[r * PUBFEAT..(r + 1) * PUBFEAT]
                .copy_from_slice(&xpub[2 * r * PUBFEAT..(2 * r + 1) * PUBFEAT]);
            let cr = (2 * r) % card_rows;
            physical_cards[r * NTYPE * TYPE..(r + 1) * NTYPE * TYPE]
                .copy_from_slice(&cards[cr * NTYPE * TYPE..(cr + 1) * NTYPE * TYPE]);
        }
        let tokens = self.tokens(&physical, &physical_cards, rows, rows);
        let x = self.trunk(&physical, &tokens, rows);
        let width = 2 * C + LOOSE;
        let mut input = scratch(rows * width);
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
            dst[2 * C..].copy_from_slice(
                &physical[r * PUBFEAT + OFF_LOOSE..r * PUBFEAT + OFF_LOOSE + LOOSE],
            );
        }
        self.board_out.run(&input, rows, out);
        recycle(physical);
        recycle(physical_cards);
        recycle(tokens);
        recycle(x);
        recycle(input);
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
        p_out: &mut Vec<f32>,
    ) {
        let width = 3 + TYPE;
        let mut slots = scratch(n * NSLOT * width);
        for c in 0..n {
            let q = owner[c] as usize;
            for k in 0..NSLOT {
                let row = &mut slots[(c * NSLOT + k) * width..(c * NSLOT + k + 1) * width];
                row[0] = phi[c * CFEAT + k];
                row[1] = phi[c * CFEAT + NSLOT + k];
                row[2] = phi[c * CFEAT + 2 * NSLOT + k];
                row[3..]
                    .copy_from_slice(&cards[(q * NTYPE + k) * TYPE..(q * NTYPE + k + 1) * TYPE]);
            }
        }
        let mut hidden = scratch(0);
        self.cfg1.run(&slots, n * NSLOT, &mut hidden);
        let (mut arg, mut th) = (scratch(0), scratch(0));
        gelu_all(&mut hidden[..n * NSLOT * CFGH], &mut arg, &mut th);
        let mut u = scratch(n * CFGH);
        for c in 0..n {
            for k in 0..NSLOT {
                for j in 0..CFGH {
                    u[c * CFGH + j] += hidden[(c * NSLOT + k) * CFGH + j];
                }
            }
        }
        self.norms[LN_CFG].apply(&mut u, n, &mut arg, &mut th);
        self.cfg_f.run(&u, n, f_out);
        self.cfg_g.run(&u, n, g_out);
        // The policy's config vector, from the same encoding the value's comes
        // from: one description of a config, two readouts of it.
        self.cfg_p.run(&u, n, p_out);
        // The linear half of `g`: a count-weighted sum of per-zone card
        // embeddings. Pooling is linear over it, so `sum_c beta(c) g(c)`
        // carries the belief's exact expected holding of every card, bound to
        // that card. `bag` depends only on the card table, so it costs two
        // rows per solve and fifteen accumulations per config.
        let mut bag = scratch(0);
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
        recycle(slots);
        recycle(hidden);
        recycle(arg);
        recycle(th);
        recycle(u);
        recycle(bag);
    }

    /// The board projection is shared by a physical row. Belief order and the
    /// seat scalar select the queried player's value.
    pub fn join(
        &self,
        p: &[f32],
        jp: &[f32],
        pooled: &[f32],
        rows: usize,
        player: usize,
        out: &mut Vec<f32>,
    ) {
        let mut z = scratch(rows * JW);
        let mut input = scratch(rows * JOIN_IN);
        for r in 0..rows {
            let (q, o) = (2 * r + player, 2 * r + 1 - player);
            let dst = &mut input[r * JOIN_IN..(r + 1) * JOIN_IN];
            dst[..POOL].copy_from_slice(&pooled[q * POOL..(q + 1) * POOL]);
            dst[POOL..2 * POOL].copy_from_slice(&pooled[o * POOL..(o + 1) * POOL]);
            dst[2 * POOL] = if player == 0 { -1.0 } else { 1.0 };
            z[r * JW..(r + 1) * JW].copy_from_slice(&jp[r * JW..(r + 1) * JW]);
        }
        self.join_b.add(&input, rows, &mut z);
        self.join_b.bias(&mut z, rows);
        let (mut t, mut d, mut arg, mut th) =
            (scratch(rows * JW), scratch(0), scratch(0), scratch(0));
        for i in 0..JBLOCKS {
            t.copy_from_slice(&z);
            self.norms[LN_JOIN + i].apply(&mut t, rows, &mut arg, &mut th);
            self.join_w[i].run(&t, rows, &mut d);
            for (o, &v) in z.iter_mut().zip(&d[..rows * JW]) {
                *o += v;
            }
        }
        self.norms[LN_JOUT].apply(&mut z, rows, &mut arg, &mut th);
        fit(out, rows * D);
        for r in 0..rows {
            let src = &p[r * D..(r + 1) * D];
            out[r * D..(r + 1) * D].copy_from_slice(src);
        }
        self.join_out.add(&z, rows, &mut out[..rows * D]);
        self.join_out.bias(out, rows);
        self.norms[LN_H].plain(out, rows);
        recycle(z);
        recycle(input);
        recycle(t);
        recycle(d);
        recycle(arg);
        recycle(th);
    }

    /// One action's description, as the policy head reads it.
    ///
    /// `slot` is the coin slot the action spends, or `-1` for the
    /// micro-decisions that spend nothing; it comes from the node's own
    /// `aslot`, so nothing here has to re-derive it. `hexes` is
    /// `Action::hexes`. The layout is the contract with `value_net.py`.
    pub fn action_feats(kind: usize, slot: i8, hexes: [u8; 3], out: &mut [f32]) {
        debug_assert_eq!(out.len(), AFEAT);
        out.fill(0.0);
        out[kind] = 1.0;
        // `-1` lands in the slot past the last, which is the "spends nothing"
        // column rather than a wrap.
        out[N_KINDS + if slot < 0 { NSLOT } else { slot as usize }] = 1.0;
        let mut at = N_KINDS + NSLOT + 1;
        for h in hexes {
            out[at + if h == NONE { N_HEXES } else { h as usize }] = 1.0;
            at += N_HEXES + 1;
        }
        debug_assert_eq!(at, AFEAT);
    }

    /// `e(a)` for every action at one node: what the action is, against the
    /// board it is played on.
    ///
    /// `feat` is `[n, AFEAT]`, and `board` is that node's own board vector —
    /// the same `D` numbers the value readout dots against, so the action is
    /// described in the position rather than in the abstract. A node's actions
    /// are public, so this runs once per node and every config at it reads the
    /// result.
    pub fn actions(
        &self,
        feat: &[f32],
        boards: &[f32],
        board_of: &[u32],
        n: usize,
        out: &mut Vec<f32>,
    ) {
        debug_assert_eq!(feat.len(), n * AFEAT);
        debug_assert_eq!(board_of.len(), n);
        let rows = boards.len() / D;
        let mut z = scratch(0);
        self.act_in.run(feat, n, &mut z);
        // Every board once, then each action adds the one it belongs to. A
        // batch spans nodes, so which board an action reads is an index rather
        // than a property of the call — the same convention the device uses
        // for everything else that was per-call and is now per-row.
        let mut proj = scratch(0);
        self.act_board.run(boards, rows, &mut proj);
        for r in 0..n {
            let at = board_of[r] as usize * AW;
            for (o, &v) in z[r * AW..(r + 1) * AW].iter_mut().zip(&proj[at..at + AW]) {
                *o += v;
            }
        }
        let (mut arg, mut th) = (scratch(0), scratch(0));
        self.norms[LN_ACT].apply(&mut z[..n * AW], n, &mut arg, &mut th);
        self.act_out.run(&z, n, out);
        recycle(z);
        recycle(proj);
        recycle(arg);
        recycle(th);
    }

    /// `logit(c, a) = <f_p(c), e(a)>` over the legal cells of one node.
    ///
    /// `cfg` and `act` name, per cell, which config and which action it stands
    /// for — the arrays the tree already keeps to index its strategy, so the
    /// policy needs no table of its own.
    pub fn policy(&self, fp: &[f32], e: &[f32], cfg: &[u32], act: &[u32], out: &mut [f32]) {
        debug_assert_eq!(cfg.len(), out.len());
        debug_assert_eq!(act.len(), out.len());
        for (k, o) in out.iter_mut().enumerate() {
            let f = &fp[cfg[k] as usize * D..(cfg[k] as usize + 1) * D];
            let a = &e[act[k] as usize * D..(act[k] as usize + 1) * D];
            *o = f.iter().zip(a).map(|(x, y)| x * y).sum();
        }
    }

    /// `v(c) = <f(c), h> + bias` for each config index in `idx`.
    pub fn values(&self, h: &[f32], f: &[f32], idx: &[u32], out: &mut [f32]) {
        let mut lo = 0;
        while lo < idx.len() {
            let first = idx[lo] as usize;
            let mut hi = lo + 1;
            while hi < idx.len() && idx[hi] as usize == first + hi - lo {
                hi += 1;
            }
            if hi - lo >= 8 {
                gemm(
                    hi - lo,
                    1,
                    D,
                    &f[first * D..],
                    D,
                    h,
                    1,
                    0.0,
                    &mut out[lo..hi],
                    1,
                );
                for value in &mut out[lo..hi] {
                    *value += self.value_bias;
                }
            } else {
                for (value, &c) in out[lo..hi].iter_mut().zip(&idx[lo..hi]) {
                    let row = &f[c as usize * D..(c as usize + 1) * D];
                    *value = row
                        .iter()
                        .zip(h)
                        .fold(self.value_bias, |sum, (&x, &y)| sum + x * y);
                }
            }
            lo = hi;
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
        let rows = queries / 2;
        let (mut p, mut jp) = (Vec::new(), Vec::new());
        self.board(xpub, &cards, rows, queries, &mut p);
        self.join_cache(&p, rows, &mut jp);
        let (mut f, mut g, mut fp) = (Vec::new(), Vec::new(), Vec::new());
        self.configs(phi, seg, n, &cards, &mut f, &mut g, &mut fp);
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

    /// The policy readout over one query's `(config, action)` cells.
    ///
    /// The board trunk and the config encoder are the same ones `forward`
    /// runs; only the readout differs. Every cell here belongs to query
    /// `seg[cfg[k]]`, and an action's embedding is built against that query's
    /// board vector, so a batch may span queries.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_policy(
        &self,
        xpub: &[f32],
        phi: &[f32],
        seg: &[u32],
        feat: &[f32],
        cfg: &[u32],
        act: &[u32],
        queries: usize,
    ) -> Vec<f32> {
        let n = seg.len();
        let na = feat.len() / AFEAT;
        let mut cards = Vec::new();
        self.cards(xpub, queries, &mut cards);
        let rows = queries / 2;
        let mut p = Vec::new();
        self.board(xpub, &cards, rows, queries, &mut p);
        let (mut f, mut g, mut fp) = (Vec::new(), Vec::new(), Vec::new());
        self.configs(phi, seg, n, &cards, &mut f, &mut g, &mut fp);
        // An action belongs to the physical row of its cells' query, and the
        // paired seat views share that row.
        let mut e = Vec::new();
        let mut out = vec![0.0; cfg.len()];
        for r in 0..rows.max(1) {
            let mine: Vec<usize> = (0..cfg.len())
                .filter(|&k| seg[cfg[k] as usize] as usize / 2 == r)
                .collect();
            if mine.is_empty() {
                continue;
            }
            let board_of = vec![0u32; na];
            self.actions(feat, &p[r * D..(r + 1) * D], &board_of, na, &mut e);
            let (c, a): (Vec<u32>, Vec<u32>) =
                mine.iter().map(|&k| (cfg[k], act[k])).unzip();
            let mut got = vec![0.0; mine.len()];
            self.policy(&fp, &e, &c, &a, &mut got);
            for (k, v) in mine.into_iter().zip(got) {
                out[k] = v;
            }
        }
        out
    }
}
