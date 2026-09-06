use crate::actions::N_KINDS;
use crate::board::N_HEXES;
use crate::pbs::{HEX_FACTS, LOOSE, PILE_COUNTS};
use crate::units::{write_card_features, CARD_FEATS};

fn dense(w: &[f32], b: &[f32], i: usize, o: usize, input: &[f32], rows: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(rows * o, 0.0);
    for r in 0..rows {
        let dst = &mut out[r * o..(r + 1) * o];
        if !b.is_empty() {
            dst.copy_from_slice(b);
        }
        for (&x, weights) in input[r * i..(r + 1) * i].iter().zip(w.chunks_exact(o)) {
            for (d, &weight) in dst.iter_mut().zip(weights) {
                *d = x.mul_add(weight, *d);
            }
        }
    }
}

pub const TYPE: usize = 64;
pub const C: usize = 96;
pub const BLOCKS: usize = 8;
pub const D: usize = 256;
pub const ATTN: usize = 128;
pub const HEADS: usize = 4;
pub const HEAD: usize = ATTN / HEADS;

#[inline]
fn gelu(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Span {
    pub w: usize,
    pub b: usize,
    pub i: usize,
    pub o: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NormSpan {
    pub g: usize,
    pub b: usize,
    pub width: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlockSpan {
    pub mix: Span,
    pub pool: Span,
    pub out: Span,
}

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
    pub cfg_in: Span,
    pub cfg_seat: usize,
    pub token_in: Span,
    pub attn_q: Span,
    pub attn_k: Span,
    pub attn_v: Span,
    pub attn_out: Span,
    pub value_hidden: Span,
    pub value_out: Span,
    pub cfg_policy: Span,
    pub head_policy: Span,
    pub act_kind: usize,
    pub act_role: usize,
    pub act_board: Span,
    pub act_h: Span,
    pub act_out: Span,
    pub norms: Vec<NormSpan>,
    pub w_len: usize,
    pub b_len: usize,
    pub ln_len: usize,
}

fn norm_widths() -> Vec<usize> {
    let mut v: Vec<usize> = (0..BLOCKS).flat_map(|_| [C, C]).collect();
    v.push(C);
    v.push(ATTN);
    v.push(ATTN);
    v.push(C);
    v
}

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

impl Default for NetLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl NetLayout {
    pub fn new() -> Self {
        let mut c = Cursor::default();
        Self {
            card: [c.lin(CARD_FEATS, TYPE, true), c.lin(TYPE, TYPE, true)],
            pile: c.lin(PILE_COUNTS, TYPE, false),
            seat: c.embed(2 * TYPE),
            hex_stem: c.lin(HEX_FACTS, C, true),
            tok_stem: c.lin(TYPE, C, false),
            pos: c.embed(N_HEXES * C),
            glob_stem: c.lin(LOOSE, C, false),
            blocks: std::array::from_fn(|_| BlockSpan {
                mix: c.lin(2 * C, C, true),
                pool: c.lin(2 * C, C, true),
                out: c.lin(C, C, true),
            }),
            board_out: c.lin(2 * C + LOOSE, D, true),
            cfg_in: c.lin(3 + TYPE, ATTN, true),
            cfg_seat: c.embed(2 * ATTN),
            token_in: c.lin(C, ATTN, true),
            attn_q: c.lin(ATTN, ATTN, false),
            attn_k: c.lin(ATTN, ATTN, false),
            attn_v: c.lin(ATTN, ATTN, false),
            attn_out: c.lin(ATTN, ATTN, false),
            value_hidden: c.lin(3 * ATTN + D, ATTN, true),
            value_out: c.lin(ATTN, 1, true),
            cfg_policy: c.lin(3 * ATTN, D, true),
            head_policy: c.lin(2 * ATTN + D, D, true),
            act_kind: c.embed(N_KINDS * C),
            act_role: c.embed(5 * C),
            act_board: c.lin(D, C, false),
            act_out: c.lin(C, D, true),
            act_h: c.lin(D, C, false),
            norms: norm_widths()
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
                .collect(),
            w_len: c.w,
            b_len: c.b,
            ln_len: c.ln,
        }
    }
}

#[derive(Clone, Default)]
struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    i: usize,
    o: usize,
}

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
}
pub const fn ln_block(i: usize, half: usize) -> usize {
    2 * i + half
}
pub const LN_TRUNK: usize = 2 * BLOCKS;
pub const LN_CFG: usize = LN_TRUNK + 1;
pub const LN_ATTN: usize = LN_CFG + 1;
pub const LN_ACT: usize = LN_ATTN + 1;

impl Net {
    pub fn random(seed: u64) -> Net {
        let mut r = crate::rng::Rng::new(seed);
        let l = NetLayout::new();
        let mut draw =
            |n: usize| -> Vec<f32> { (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect() };
        let (w, b) = (draw(l.w_len), draw(l.b_len));
        let mut ln = vec![0.0; l.ln_len];
        for n in &l.norms {
            ln[n.g..n.g + n.width].fill(1.0);
        }
        Net::from_flat(&w, &b, &ln).expect("a random net matches the layout")
    }

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
        Ok(Self {
            flat: std::sync::Arc::new(Flat {
                w: w.to_vec(),
                b: b.to_vec(),
                ln: ln.to_vec(),
            }),
            card: l.card.map(layer),
        })
    }

    pub fn load_flat_bin(path: &str) -> std::io::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
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

    pub fn is_empty(&self) -> bool {
        self.card[0].w.is_empty()
    }

    pub fn flat(&self) -> &Flat {
        &self.flat
    }

    pub fn cards(&self, ids: &[u8], out: &mut Vec<f32>) {
        let mut facts = vec![0.0f32; ids.len() * CARD_FEATS];
        for (t, &id) in ids.iter().enumerate() {
            write_card_features(id, &mut facts[t * CARD_FEATS..(t + 1) * CARD_FEATS]);
        }
        let (a, b) = (&self.card[0], &self.card[1]);
        let mut hidden = Vec::new();
        dense(&a.w, &a.b, a.i, a.o, &facts, ids.len(), &mut hidden);
        for x in hidden.iter_mut() {
            *x = gelu(*x);
        }
        dense(&b.w, &b.b, b.i, b.o, &hidden, ids.len(), out);
    }
}
