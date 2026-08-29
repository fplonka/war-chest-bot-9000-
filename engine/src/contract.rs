use crate::farm::{Dst, Writes};
use crate::search::{Solver, NO_TRANS};

pub const KIND_DECISION: u8 = 0;
pub const KIND_CHANCE: u8 = 1;
pub const KIND_LEAF: u8 = 2;

pub const NO_ROW: u32 = u32::MAX;

#[derive(Default, Clone)]
pub struct Contract {
    pub kind: Vec<u8>,
    pub player: Vec<u8>,
    pub nc: Vec<[u32; 2]>,
    pub parent: Vec<u32>,
    pub exhausted: Vec<u32>,
    pub level: Vec<u32>,
    pub roff: Vec<u32>,
    pub voff: Vec<u32>,
    pub soff: Vec<u32>,
    pub util: Vec<f32>,

    pub child_at: Vec<u32>,
    pub child_n: Vec<u32>,
    pub child: Vec<u32>,

    pub legal_base: Vec<u32>,
    pub legal_off: Vec<u32>,
    pub legal_child: Vec<u32>,
    pub legal_trans: Vec<u32>,
    pub cell_row: Vec<u32>,
    pub cell_val: Vec<u32>,

    pub rev_base: Vec<u32>,
    pub rev_start: Vec<u32>,
    pub rev_src: Vec<u32>,
    pub rev_cell: Vec<u32>,

    pub draw_base: Vec<u32>,
    pub draw_start: Vec<u32>,
    pub draw_to: Vec<u32>,
    pub draw_p: Vec<f32>,
    pub rvd_base: Vec<u32>,
    pub rvd_start: Vec<u32>,
    pub rvd_src: Vec<u32>,
    pub rvd_p: Vec<f32>,

    pub level_start: Vec<u32>,
    pub level_node: Vec<u32>,
    pub built: usize,
}

#[derive(Default, Clone)]
pub struct Sent {
    pub nodes: usize,
    pools: [usize; 15],
}

impl Contract {
    pub fn nodes(&self) -> usize {
        self.kind.len()
    }

    pub fn write_into(&self, w: &mut Writes, sent: &mut Sent, from: usize, rewrite: &[u32], resealed: &[u32]) {
        let n = self.nodes();
        let mut spans: Vec<(usize, usize)> = vec![(from, n - from)];
        spans.extend(rewrite.iter().map(|&g| (g as usize, 1)));
        w.u32s(Dst::Exhausted, from, &self.exhausted[from..]);
        let mut old: Vec<u32> = resealed.iter().copied().filter(|&i| (i as usize) < from).collect();
        old.sort_unstable();
        old.dedup();
        for i in old {
            w.u32s(Dst::Exhausted, i as usize, &self.exhausted[i as usize..i as usize + 1]);
        }
        for &(at, k) in &spans {
            w.u8s(Dst::Kind, at, &self.kind[at..at + k]);
            w.u8s(Dst::Player, at, &self.player[at..at + k]);
            let nc: Vec<u32> = self.nc[at..at + k].iter().flatten().copied().collect();
            w.u32s(Dst::Nc, 2 * at, &nc);
            w.u32s(Dst::Parent, at, &self.parent[at..at + k]);
            w.u32s(Dst::Roff, at, &self.roff[at..at + k]);
            w.u32s(Dst::Voff, at, &self.voff[at..at + k]);
            w.u32s(Dst::Soff, at, &self.soff[at..at + k]);
            w.f32s(Dst::Util, at, &self.util[at..at + k]);
            w.u32s(Dst::ChildAt, at, &self.child_at[at..at + k]);
            w.u32s(Dst::ChildN, at, &self.child_n[at..at + k]);
            w.u32s(Dst::LegalBase, at, &self.legal_base[at..at + k]);
            w.u32s(Dst::RevBase, at, &self.rev_base[at..at + k]);
            w.u32s(Dst::RvdBase, at, &self.rvd_base[at..at + k]);
            w.u32s(Dst::DrawBase, at, &self.draw_base[at..at + k]);
        }
        sent.nodes = n;
        let words: [(Dst, &[u32]); 13] = [
            (Dst::Child, &self.child),
            (Dst::LegalOff, &self.legal_off),
            (Dst::LegalChild, &self.legal_child),
            (Dst::LegalTrans, &self.legal_trans),
            (Dst::CellRow, &self.cell_row),
            (Dst::CellVal, &self.cell_val),
            (Dst::RevStart, &self.rev_start),
            (Dst::RevSrc, &self.rev_src),
            (Dst::RevCell, &self.rev_cell),
            (Dst::RvdStart, &self.rvd_start),
            (Dst::RvdSrc, &self.rvd_src),
            (Dst::DrawStart, &self.draw_start),
            (Dst::DrawTo, &self.draw_to),
        ];
        for (i, (d, v)) in words.into_iter().enumerate() {
            let at = sent.pools[i].min(v.len());
            w.u32s(d, at, &v[at..]);
            sent.pools[i] = v.len();
        }
        for (i, (d, v)) in [(Dst::RvdP, &self.rvd_p), (Dst::DrawP, &self.draw_p)]
            .into_iter()
            .enumerate()
        {
            let at = sent.pools[13 + i].min(v.len());
            w.f32s(d, at, &v[at..]);
            sent.pools[13 + i] = v.len();
        }
        w.u32s(Dst::LevelStart, 0, &self.level_start);
        w.u32s(Dst::LevelNode, 0, &self.level_node);
    }

    pub fn levels(&self) -> usize {
        self.level_start.len().saturating_sub(1)
    }

    pub fn of(sv: &Solver) -> Contract {
        let mut c = Contract::default();
        c.extend(sv, &[], &[]);
        c
    }

    pub fn extend(&mut self, sv: &Solver, grown: &[u32], resealed: &[u32]) {
        let first = self.built;
        self.ensure(sv.nodes.len());
        for &g in grown {
            if (g as usize) < first {
                self.describe(sv, g as usize);
            }
        }
        for i in first..sv.nodes.len() {
            self.describe(sv, i);
        }
        self.built = sv.nodes.len();
        for &r in resealed {
            let i = r as usize;
            if i < first {
                self.exhausted[i] = sv.nodes[i].exhausted as u32;
            }
        }
        self.levels_from(sv);
        for &g in grown {
            if (g as usize) < first {
                self.transpose_children(sv, g as usize);
            }
        }
        for i in first..self.built {
            self.transpose_children(sv, i);
        }
    }

    fn ensure(&mut self, n: usize) {
        self.kind.resize(n, KIND_LEAF);
        self.player.resize(n, 0);
        self.exhausted.resize(n, 0);
        self.parent.resize(n, NO_ROW);
        self.nc.resize(n, [0, 0]);
        self.roff.resize(n, 0);
        self.voff.resize(n, 0);
        self.soff.resize(n, 0);
        self.util.resize(n, 0.0);
        self.child_at.resize(n, 0);
        self.child_n.resize(n, 0);
        self.legal_base.resize(n, NO_ROW);
        self.draw_base.resize(n, NO_ROW);
        self.level.resize(n, 0);
        self.rev_base.resize(n, NO_ROW);
        self.rvd_base.resize(n, NO_ROW);
    }

    fn describe(&mut self, sv: &Solver, i: usize) {
        let c = self;
        {
            let t = &sv.nodes[i];
            c.kind[i] = if t.leaf {
                KIND_LEAF
            } else if t.chance {
                KIND_CHANCE
            } else {
                KIND_DECISION
            };
            c.player[i] = t.player;
            c.exhausted[i] = t.exhausted as u32;
            c.parent[i] = t.parent;
            c.nc[i] = t.nc;
            c.roff[i] = t.roff;
            c.voff[i] = t.voff;
            c.soff[i] = t.soff;
            c.util[i] = t.util;
            c.child_at[i] = c.child.len() as u32;
            c.child_n[i] = t.child.len() as u32;
            c.child.extend(t.child.iter().map(|&ch| ch as u32));

            c.legal_base[i] = if t.leaf || t.chance {
                NO_ROW
            } else {
                let base = c.legal_off.len() as u32;
                let me = t.player as usize;
                c.legal_off.extend_from_slice(&t.legal_off[..=t.nc[me] as usize]);
                let at = t.soff as usize;
                let end = at + t.legal_action.len();
                if c.legal_child.len() < end {
                    c.legal_child.resize(end, 0);
                    c.legal_trans.resize(end, NO_TRANS);
                    c.cell_row.resize(end, 0);
                    c.cell_val.resize(end, NO_ROW);
                }
                c.legal_child[at..end].copy_from_slice(&t.legal_child);
                c.legal_trans[at..end].copy_from_slice(&t.legal_trans);
                c.cell_row[at..end].copy_from_slice(&t.cell_row);
                for cell in 0..t.legal_action.len() {
                    let tr = t.legal_trans[cell];
                    c.cell_val[at + cell] = if tr == NO_TRANS {
                        NO_ROW
                    } else {
                        sv.nodes[t.legal_child[cell] as usize].voff + tr
                    };
                }
                base
            };

            let _ = &t;
            c.draw_base[i] = if t.chance {
                let base = c.draw_start.len() as u32;
                for r in 0..=t.draw.rows() {
                    c.draw_start.push(t.draw.start[r] + c.draw_to.len() as u32);
                }
                c.draw_to.extend_from_slice(&t.draw.to);
                c.draw_p.extend_from_slice(&t.draw.p);
                base
            } else {
                NO_ROW
            };
        }
    }

    fn levels_from(&mut self, sv: &Solver) {
        let c = self;
        let n = sv.nodes.len();
        for i in 1..n {
            let p = c.parent[i];
            c.level[i] = if p == NO_ROW { 0 } else { c.level[p as usize] + 1 };
        }
        let depth = c.level.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut count = vec![0u32; depth + 1];
        for &l in &c.level {
            count[l as usize + 1] += 1;
        }
        for l in 0..depth {
            count[l + 1] += count[l];
        }
        c.level_start = count.clone();
        c.level_node = vec![0; n];
        for i in 0..n {
            let l = c.level[i] as usize;
            c.level_node[count[l] as usize] = i as u32;
            count[l] += 1;
        }
    }

    fn transpose_children(&mut self, sv: &Solver, i: usize) {
        {
            let t = &sv.nodes[i];
            if t.leaf {
                return;
            }
            let me = t.player as usize;
            if t.chance {
                let ch = t.child[0];
                let kids = self.nc[ch][me] as usize;
                let base = self.rvd_start.len() as u32;
                self.rvd_base[ch] = base;
                let mut count = vec![0u32; kids + 1];
                for ci in 0..t.draw.rows() {
                    for &to in t.draw.row(ci).0 {
                        count[to as usize + 1] += 1;
                    }
                }
                for k in 0..kids {
                    count[k + 1] += count[k];
                }
                let at = self.rvd_src.len() as u32;
                self.rvd_start.extend(count.iter().map(|x| x + at));
                self.rvd_src.resize(self.rvd_src.len() + count[kids] as usize, 0);
                self.rvd_p.resize(self.rvd_p.len() + count[kids] as usize, 0.0);
                for ci in 0..t.draw.rows() {
                    let (to, pr) = t.draw.row(ci);
                    for k in 0..to.len() {
                        let slot = (at + count[to[k] as usize]) as usize;
                        self.rvd_src[slot] = ci as u32;
                        self.rvd_p[slot] = pr[k];
                        count[to[k] as usize] += 1;
                    }
                }
                return;
            }
            for (ci, &ch_u) in t.child.iter().enumerate() {
                let ch = ch_u;
                let kids = self.nc[ch][me] as usize;
                let base = self.rev_start.len() as u32;
                self.rev_base[ch] = base;
                let mut count = vec![0u32; kids + 1];
                for cell in 0..t.legal_action.len() {
                    if t.legal_child[cell] as usize != ch || t.legal_trans[cell] == NO_TRANS {
                        continue;
                    }
                    count[t.legal_trans[cell] as usize + 1] += 1;
                }
                for k in 0..kids {
                    count[k + 1] += count[k];
                }
                let at = self.rev_src.len() as u32;
                self.rev_start.extend(count.iter().map(|x| x + at));
                self.rev_src.resize(self.rev_src.len() + count[kids] as usize, 0);
                self.rev_cell.resize(self.rev_cell.len() + count[kids] as usize, 0);
                let (s0, s1) = (t.obs_start[ci] as usize, t.obs_start[ci + 1] as usize);
                for &au in &t.obs_act[s0..s1] {
                    let a = au as usize;
                    for &cell_u in &t.action_cell[t.action_off[a] as usize..t.action_off[a + 1] as usize] {
                        let cell = cell_u as usize;
                        if t.legal_child[cell] as usize != ch || t.legal_trans[cell] == NO_TRANS {
                            continue;
                        }
                        let to = t.legal_trans[cell] as usize;
                        let slot = (at + count[to]) as usize;
                        self.rev_src[slot] = t.cell_row[cell];
                        self.rev_cell[slot] = self.soff[i] + cell as u32;
                        count[to] += 1;
                    }
                }
            }
        }
    }
}
