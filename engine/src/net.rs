//! The fixed production value network.

use crate::board::N_HEXES;
use crate::rebel::{
    CFEAT, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE, OFF_PILES, PILE_COUNTS,
    PUBFEAT,
};
use crate::units::{CARD_FEATS, N_UNITS};

pub const UNIT: usize = 32;
pub const SLOT: usize = 64;
pub const CONFIG: usize = 128;
pub const PUBLIC: usize = 384;
pub const CONTEXT: usize = 384;
pub const JOINT: usize = 256;
pub const PUBLIC_IN: usize = N_HEXES * (HEX_FACTS + UNIT) + NTYPE * (PILE_COUNTS + UNIT) + LOOSE;

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
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    debug_assert!(a.len() >= (m - 1) * lda + k);
    debug_assert!(b.len() >= (k - 1) * ldb + n);
    debug_assert!(c.len() >= (m - 1) * ldc + n);
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
    }
    #[cfg(not(target_vendor = "apple"))]
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            lda as isize,
            1,
            b.as_ptr(),
            ldb as isize,
            1,
            beta,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
    }
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    const K: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (K * (x + 0.044_715 * x * x * x)).tanh())
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

#[derive(Clone, Copy, Debug, Default)]
pub struct Span {
    pub w: usize,
    pub b: usize,
    pub i: usize,
    pub o: usize,
}

#[derive(Clone, Debug)]
pub struct V4Layout {
    pub rule: [Span; 2],
    pub unit_id: usize,
    pub public: [Span; 3],
    pub belief_slot: Span,
    pub belief_config: Span,
    pub belief_item: Span,
    pub candidate_slot: Span,
    pub candidate: Span,
    pub context: [Span; 2],
    pub context_joint: Span,
    pub candidate_joint: Span,
    pub value: Span,
    pub joint_bias: usize,
    pub norms: [(usize, usize); 7],
    pub w_len: usize,
    pub b_len: usize,
    pub ln_len: usize,
}

impl V4Layout {
    pub fn new(dims: &[usize]) -> Result<Self, String> {
        if dims != [4] {
            return Err(format!("unsupported model format {dims:?}; expected [4]"));
        }
        let (mut w, mut b, mut ln) = (0, 0, 0);
        let lin = |i: usize, o: usize, bias: bool, w: &mut usize, b: &mut usize| {
            let s = Span {
                w: *w,
                b: if bias { *b } else { usize::MAX },
                i,
                o,
            };
            *w += i * o;
            if bias {
                *b += o;
            }
            s
        };
        let rule = [
            lin(CARD_FEATS, UNIT, true, &mut w, &mut b),
            lin(UNIT, UNIT, true, &mut w, &mut b),
        ];
        let unit_id = w;
        w += N_UNITS * UNIT;
        let public = [
            lin(PUBLIC_IN, PUBLIC, true, &mut w, &mut b),
            lin(PUBLIC, PUBLIC, true, &mut w, &mut b),
            lin(PUBLIC, PUBLIC, true, &mut w, &mut b),
        ];
        let belief_slot = lin(3 + UNIT, SLOT, true, &mut w, &mut b);
        let belief_config = lin(SLOT, CONFIG, true, &mut w, &mut b);
        let belief_item = lin(CONFIG, CONFIG, true, &mut w, &mut b);
        let candidate_slot = lin(3 + UNIT, SLOT, true, &mut w, &mut b);
        let candidate = lin(SLOT, CONFIG, true, &mut w, &mut b);
        let context = [
            lin(PUBLIC + 2 * CONFIG, CONTEXT, true, &mut w, &mut b),
            lin(CONTEXT, CONTEXT, true, &mut w, &mut b),
        ];
        let context_joint = lin(CONTEXT, JOINT, false, &mut w, &mut b);
        let candidate_joint = lin(CONFIG, JOINT, false, &mut w, &mut b);
        let joint_bias = b;
        b += JOINT;
        let value = lin(JOINT, 1, true, &mut w, &mut b);
        let mut norms = [(0, 0); 7];
        for (pair, width) in norms
            .iter_mut()
            .zip([PUBLIC, PUBLIC, PUBLIC, CONFIG, CONFIG, CONTEXT, CONTEXT])
        {
            *pair = (ln, ln + width);
            ln += 2 * width;
        }
        Ok(Self {
            rule,
            unit_id,
            public,
            belief_slot,
            belief_config,
            belief_item,
            candidate_slot,
            candidate,
            context,
            context_joint,
            candidate_joint,
            value,
            joint_bias,
            norms,
            w_len: w,
            b_len: b,
            ln_len: ln,
        })
    }
}

#[derive(Clone, Default)]
struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    i: usize,
    o: usize,
}

impl Lin {
    fn run(&self, input: &[f32], rows: usize, out: &mut Vec<f32>) {
        fit(out, rows * self.o);
        gemm(
            rows, self.o, self.i, input, self.i, &self.w, self.o, 0.0, out, self.o,
        );
        if !self.b.is_empty() {
            for row in out[..rows * self.o].chunks_exact_mut(self.o) {
                for (x, &bias) in row.iter_mut().zip(&self.b) {
                    *x += bias;
                }
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Mlp {
    pub dims: Vec<usize>,
    rule: [Lin; 2],
    unit_id: Vec<f32>,
    public: [Lin; 3],
    belief_slot: Lin,
    belief_config: Lin,
    belief_item: Lin,
    candidate_slot: Lin,
    candidate: Lin,
    context: [Lin; 2],
    context_joint: Lin,
    candidate_joint: Lin,
    value: Lin,
    joint_bias: Vec<f32>,
    norms: [(Vec<f32>, Vec<f32>); 7],
}

impl Mlp {
    pub fn from_flat(dims: &[usize], w: &[f32], b: &[f32], ln: &[f32]) -> Result<Self, String> {
        let layout = V4Layout::new(dims)?;
        if (w.len(), b.len(), ln.len()) != (layout.w_len, layout.b_len, layout.ln_len) {
            return Err(format!(
                "weight sizes {}/{}/{} do not match v4 {}/{}/{}",
                w.len(),
                b.len(),
                ln.len(),
                layout.w_len,
                layout.b_len,
                layout.ln_len
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
        let widths = [PUBLIC, PUBLIC, PUBLIC, CONFIG, CONFIG, CONTEXT, CONTEXT];
        let norms = std::array::from_fn(|i| {
            let (g, bt) = layout.norms[i];
            (
                ln[g..g + widths[i]].to_vec(),
                ln[bt..bt + widths[i]].to_vec(),
            )
        });
        Ok(Self {
            dims: dims.to_vec(),
            rule: layout.rule.map(layer),
            unit_id: w[layout.unit_id..layout.unit_id + N_UNITS * UNIT].to_vec(),
            public: layout.public.map(layer),
            belief_slot: layer(layout.belief_slot),
            belief_config: layer(layout.belief_config),
            belief_item: layer(layout.belief_item),
            candidate_slot: layer(layout.candidate_slot),
            candidate: layer(layout.candidate),
            context: layout.context.map(layer),
            context_joint: layer(layout.context_joint),
            candidate_joint: layer(layout.candidate_joint),
            value: layer(layout.value),
            joint_bias: b[layout.joint_bias..layout.joint_bias + JOINT].to_vec(),
            norms,
        })
    }

    pub fn load_flat_bin(
        path: &str,
    ) -> std::io::Result<(Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>)> {
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
    pub fn dg(&self) -> usize {
        CONFIG
    }
    pub fn rank(&self) -> usize {
        JOINT
    }
    pub fn de(&self) -> usize {
        UNIT
    }

    fn norm_gelu(rows: usize, width: usize, norm: &(Vec<f32>, Vec<f32>), x: &mut [f32]) {
        for row in x[..rows * width].chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..width {
                row[j] = gelu((row[j] - mean) * inv * norm.0[j] + norm.1[j]);
            }
        }
    }

    pub fn units(&self, xpub: &[f32], ids: &[u8], rows: usize, out: &mut Vec<f32>) {
        let mut facts = vec![0.0; rows * NTYPE * CARD_FEATS];
        for r in 0..rows {
            facts[r * NTYPE * CARD_FEATS..(r + 1) * NTYPE * CARD_FEATS].copy_from_slice(
                &xpub[r * PUBFEAT + OFF_CARDS..r * PUBFEAT + OFF_CARDS + NTYPE * CARD_FEATS],
            );
        }
        let mut hidden = Vec::new();
        self.rule[0].run(&facts, rows * NTYPE, &mut hidden);
        hidden[..rows * NTYPE * UNIT]
            .iter_mut()
            .for_each(|x| *x = gelu(*x));
        self.rule[1].run(&hidden, rows * NTYPE, out);
        for r in 0..rows {
            for t in 0..NTYPE {
                let id = ids[r * NTYPE + t] as usize;
                for j in 0..UNIT {
                    out[(r * NTYPE + t) * UNIT + j] += self.unit_id[id * UNIT + j];
                }
            }
        }
    }

    fn public_input(&self, xpub: &[f32], units: &[f32], rows: usize, unit_rows: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * PUBLIC_IN];
        let hex_units = N_HEXES * HEX_FACTS;
        let piles = hex_units + N_HEXES * UNIT;
        for r in 0..rows {
            let src = &xpub[r * PUBFEAT..(r + 1) * PUBFEAT];
            let dst = &mut out[r * PUBLIC_IN..(r + 1) * PUBLIC_IN];
            for h in 0..N_HEXES {
                let hex = &src[h * HEX_CH..(h + 1) * HEX_CH];
                dst[h * HEX_FACTS..(h + 1) * HEX_FACTS].copy_from_slice(&hex[..HEX_FACTS]);
                if let Some(t) = hex[HEX_FACTS..].iter().position(|&v| v != 0.0) {
                    dst[hex_units + h * UNIT..hex_units + (h + 1) * UNIT].copy_from_slice(
                        &units[((r % unit_rows) * NTYPE + t) * UNIT
                            ..((r % unit_rows) * NTYPE + t + 1) * UNIT],
                    );
                }
            }
            for t in 0..NTYPE {
                let at = piles + t * (PILE_COUNTS + UNIT);
                dst[at..at + PILE_COUNTS].copy_from_slice(
                    &src[OFF_PILES + t * PILE_COUNTS..OFF_PILES + (t + 1) * PILE_COUNTS],
                );
                dst[at + PILE_COUNTS..at + PILE_COUNTS + UNIT].copy_from_slice(
                    &units[((r % unit_rows) * NTYPE + t) * UNIT
                        ..((r % unit_rows) * NTYPE + t + 1) * UNIT],
                );
            }
            dst[piles + NTYPE * (PILE_COUNTS + UNIT)..]
                .copy_from_slice(&src[OFF_LOOSE..OFF_LOOSE + LOOSE]);
        }
        out
    }

    pub fn public(
        &self,
        xpub: &[f32],
        units: &[f32],
        rows: usize,
        unit_rows: usize,
        out: &mut Vec<f32>,
    ) {
        let mut a = self.public_input(xpub, units, rows, unit_rows);
        let mut b = Vec::new();
        for k in 0..3 {
            self.public[k].run(&a, rows, &mut b);
            Self::norm_gelu(rows, PUBLIC, &self.norms[k], &mut b);
            std::mem::swap(&mut a, &mut b);
        }
        fit(out, rows * PUBLIC);
        out[..rows * PUBLIC].copy_from_slice(&a[..rows * PUBLIC]);
    }

    fn config_slots(&self, phi: &[f32], owner: &[u8], units: &[f32], n: usize) -> Vec<f32> {
        let width = 3 + UNIT;
        let mut out = vec![0.0; n * NSLOT * width];
        for c in 0..n {
            let query = owner[c] as usize;
            for k in 0..NSLOT {
                let row = &mut out[(c * NSLOT + k) * width..(c * NSLOT + k + 1) * width];
                row[0] = phi[c * CFEAT + k];
                row[1] = phi[c * CFEAT + NSLOT + k];
                row[2] = phi[c * CFEAT + 2 * NSLOT + k];
                row[3..].copy_from_slice(
                    &units[(query * NTYPE + k) * UNIT..(query * NTYPE + k + 1) * UNIT],
                );
            }
        }
        out
    }

    pub fn embed(
        &self,
        phi: &[f32],
        owner: &[u8],
        n: usize,
        units: &[f32],
        belief: &mut Vec<f32>,
        candidate: &mut Vec<f32>,
    ) {
        let slots = self.config_slots(phi, owner, units, n);
        let mut bh = Vec::new();
        self.belief_slot.run(&slots, n * NSLOT, &mut bh);
        bh[..n * NSLOT * SLOT]
            .iter_mut()
            .for_each(|x| *x = gelu(*x));
        let mut sum = vec![0.0; n * SLOT];
        for c in 0..n {
            for k in 0..NSLOT {
                for j in 0..SLOT {
                    sum[c * SLOT + j] += bh[(c * NSLOT + k) * SLOT + j];
                }
            }
        }
        self.belief_config.run(&sum, n, &mut bh);
        Self::norm_gelu(n, CONFIG, &self.norms[3], &mut bh);
        self.belief_item.run(&bh, n, belief);

        self.candidate_slot.run(&slots, n * NSLOT, &mut bh);
        bh[..n * NSLOT * SLOT]
            .iter_mut()
            .for_each(|x| *x = gelu(*x));
        sum.fill(0.0);
        for c in 0..n {
            for k in 0..NSLOT {
                for j in 0..SLOT {
                    sum[c * SLOT + j] += bh[(c * NSLOT + k) * SLOT + j];
                }
            }
        }
        self.candidate.run(&sum, n, &mut bh);
        Self::norm_gelu(n, CONFIG, &self.norms[4], &mut bh);
        self.candidate_joint.run(&bh, n, candidate);
    }

    pub fn context(
        &self,
        belief: &[f32],
        rows: usize,
        public: &[f32],
        player: usize,
        out: &mut Vec<f32>,
    ) {
        let width = PUBLIC + 2 * CONFIG;
        let mut input = vec![0.0; rows * width];
        for r in 0..rows {
            let dst = &mut input[r * width..(r + 1) * width];
            dst[..PUBLIC]
                .copy_from_slice(&public[(2 * r + player) * PUBLIC..(2 * r + player + 1) * PUBLIC]);
            dst[PUBLIC..PUBLIC + CONFIG]
                .copy_from_slice(&belief[(2 * r + player) * CONFIG..(2 * r + player + 1) * CONFIG]);
            let other = 1 - player;
            dst[PUBLIC + CONFIG..]
                .copy_from_slice(&belief[(2 * r + other) * CONFIG..(2 * r + other + 1) * CONFIG]);
        }
        let mut hidden = Vec::new();
        self.context[0].run(&input, rows, &mut hidden);
        Self::norm_gelu(rows, CONTEXT, &self.norms[5], &mut hidden);
        self.context[1].run(&hidden, rows, &mut input);
        Self::norm_gelu(rows, CONTEXT, &self.norms[6], &mut input);
        self.context_joint.run(&input, rows, out);
    }

    fn contexts(&self, belief: &[f32], queries: usize, public: &[f32], out: &mut Vec<f32>) {
        let width = PUBLIC + 2 * CONFIG;
        let mut input = vec![0.0; queries * width];
        for q in 0..queries {
            let dst = &mut input[q * width..(q + 1) * width];
            dst[..PUBLIC].copy_from_slice(&public[q * PUBLIC..(q + 1) * PUBLIC]);
            dst[PUBLIC..PUBLIC + CONFIG].copy_from_slice(&belief[q * CONFIG..(q + 1) * CONFIG]);
            let other = q ^ 1;
            dst[PUBLIC + CONFIG..].copy_from_slice(&belief[other * CONFIG..(other + 1) * CONFIG]);
        }
        let mut hidden = Vec::new();
        self.context[0].run(&input, queries, &mut hidden);
        Self::norm_gelu(queries, CONTEXT, &self.norms[5], &mut hidden);
        self.context[1].run(&hidden, queries, &mut input);
        Self::norm_gelu(queries, CONTEXT, &self.norms[6], &mut input);
        self.context_joint.run(&input, queries, out);
    }

    pub fn values(&self, common: &[f32], candidate: &[f32], idx: &[u32], out: &mut [f32]) {
        for (o, &c) in out.iter_mut().zip(idx) {
            let private = &candidate[c as usize * JOINT..(c as usize + 1) * JOINT];
            let mut value = self.value.b[0];
            for j in 0..JOINT {
                value += gelu(common[j] + private[j] + self.joint_bias[j]) * self.value.w[j];
            }
            *o = value;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xpub: &[f32],
        ids: &[u8],
        phi: &[f32],
        weight: &[f32],
        seg: &[u32],
        queries: usize,
    ) -> Vec<f32> {
        let configs = weight.len();
        let mut units = Vec::new();
        self.units(xpub, ids, queries, &mut units);
        let mut public = Vec::new();
        self.public(xpub, &units, queries, queries, &mut public);
        let owner: Vec<u8> = seg.iter().map(|&x| x as u8).collect();
        let (mut items, mut private) = (Vec::new(), Vec::new());
        self.embed(phi, &owner, configs, &units, &mut items, &mut private);
        let mut pooled = vec![0.0; queries * CONFIG];
        for c in 0..configs {
            let q = seg[c] as usize;
            for j in 0..CONFIG {
                pooled[q * CONFIG + j] += weight[c] * items[c * CONFIG + j];
            }
        }
        let mut common = Vec::new();
        self.contexts(&pooled, queries, &public, &mut common);
        let mut out = vec![0.0; configs];
        for c in 0..configs {
            self.values(
                &common[seg[c] as usize * JOINT..(seg[c] as usize + 1) * JOINT],
                &private,
                &[c as u32],
                &mut out[c..c + 1],
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_layout_sizes() {
        let l = V4Layout::new(&[4]).unwrap();
        assert_eq!((l.w_len, l.b_len, l.ln_len), (1_543_072, 2_753, 4_352));
    }

    #[test]
    fn old_formats_are_rejected() {
        assert!(V4Layout::new(&[3]).is_err());
        assert!(V4Layout::new(&[4, 1]).is_err());
    }
}
