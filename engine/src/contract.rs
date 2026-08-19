//! The flat description of a solve's tree that an accelerator reads.
//!
//! `Solver` holds a tree of `TNode`s: vectors of vectors, `Rc`-shared config
//! supports, and per-node offset tables. That shape is right for building and
//! growing a tree and wrong for a device, which wants contiguous arrays and one
//! task per thread with no pointer to chase and no owner to recover.
//!
//! Two things here are not just a re-layout of what `Solver` already holds.
//!
//! **The reach transition is transposed.** `Solver::propagate` scatters: it
//! walks a parent's legal cells and adds into whichever child config each one
//! leads to. On a device that is a write conflict between threads, so the same
//! arithmetic is stored the other way round — for every (node, config), the
//! list of parent configs and cells that feed it — and each thread sums its own
//! output. `rev_*` is that transpose for a decision parent and `rvd_*` for a
//! draw. Backpropagation needs no such thing: it is already a gather, because a
//! config reads its own legal row.
//!
//! **Nodes are grouped into levels.** Within one level no node is another's
//! ancestor, so a whole level's tasks are independent and the sweep is a
//! sequence of levels rather than a walk in node order.
//!
//! Everything is append-only under growth. `grow` turns one leaf into a
//! decision node and appends its children, so extending the contract rewrites
//! that node's own row and appends the rest — which is what lets a growing tree
//! ship a delta per iteration rather than itself.

use crate::search::{Solver, NO_TRANS};

/// What a node is, for a device that cannot afford a branch per field.
pub const KIND_DECISION: u8 = 0;
pub const KIND_CHANCE: u8 = 1;
pub const KIND_LEAF: u8 = 2;

/// No reverse row: the node's parent is not of that kind, so that gather does
/// not apply to it.
pub const NO_ROW: u32 = u32::MAX;

#[derive(Default, Clone)]
pub struct Contract {
    // ------------------------------------------------------------ per node
    pub kind: Vec<u8>,
    pub player: Vec<u8>,
    /// Config counts, player-major, so a task knows its own row length.
    pub nc: Vec<[u32; 2]>,
    pub parent: Vec<u32>,
    /// Depth from the root. Nodes of one level are mutually independent.
    pub level: Vec<u32>,
    /// Bases into the solver's own arenas, which the device mirrors exactly.
    pub roff: Vec<u32>,
    pub voff: Vec<u32>,
    pub soff: Vec<u32>,
    /// Terminal utility, for the player to act there.
    pub util: Vec<f32>,

    // -------------------------------------------------------- children, CSR
    pub child_off: Vec<u32>,
    pub child: Vec<u32>,

    // ------------------------------- legal cells, CSR by the acting config
    /// Where this node's `nc + 1` offsets start in `legal_off`.
    pub legal_base: Vec<u32>,
    pub legal_off: Vec<u32>,
    pub legal_child: Vec<u32>,
    pub legal_trans: Vec<u32>,
    /// The parent config each cell belongs to, which the average-strategy
    /// accumulation reads and the reverse gather does not.
    pub cell_row: Vec<u32>,

    // --------------------------- the reach transition, transposed, per node
    /// Where this node's `nc + 1` offsets start in `rev_start`, or `NO_ROW`
    /// when its parent is not a decision node.
    pub rev_base: Vec<u32>,
    pub rev_start: Vec<u32>,
    /// Parent config, and the parent cell whose strategy weights it.
    pub rev_src: Vec<u32>,
    pub rev_cell: Vec<u32>,

    // ------------------------------------ the draw transition, both ways
    /// Forward, by parent config: backpropagation pushes values through it.
    pub draw_base: Vec<u32>,
    pub draw_start: Vec<u32>,
    pub draw_to: Vec<u32>,
    pub draw_p: Vec<f32>,
    /// Transposed, by child config: the reach sweep gathers through it.
    pub rvd_base: Vec<u32>,
    pub rvd_start: Vec<u32>,
    pub rvd_src: Vec<u32>,
    pub rvd_p: Vec<f32>,

    // ------------------------------------------------------------- levels
    pub level_start: Vec<u32>,
    pub level_node: Vec<u32>,
}

impl Contract {
    pub fn nodes(&self) -> usize {
        self.kind.len()
    }

    pub fn levels(&self) -> usize {
        self.level_start.len().saturating_sub(1)
    }

    /// Describe a solver's tree as it stands.
    ///
    /// Rebuilt whole here. Growth appends, so a device can be handed the tail
    /// of each array instead; that is an upload question, not a correctness
    /// one, and this is the definition both sides are held to.
    pub fn of(sv: &Solver) -> Contract {
        let n = sv.nodes.len();
        let mut c = Contract {
            kind: Vec::with_capacity(n),
            player: Vec::with_capacity(n),
            nc: Vec::with_capacity(n),
            parent: vec![NO_ROW; n],
            level: vec![0; n],
            roff: Vec::with_capacity(n),
            voff: Vec::with_capacity(n),
            soff: Vec::with_capacity(n),
            util: Vec::with_capacity(n),
            child_off: Vec::with_capacity(n + 1),
            ..Default::default()
        };
        c.child_off.push(0);

        for i in 0..n {
            let t = &sv.nodes[i];
            c.kind.push(if t.leaf {
                KIND_LEAF
            } else if t.chance {
                KIND_CHANCE
            } else {
                KIND_DECISION
            });
            c.player.push(t.player);
            c.nc.push(sv.nc[i]);
            c.roff.push(sv.roff[i]);
            c.voff.push(sv.voff[i]);
            c.soff.push(sv.soff[i]);
            c.util.push(t.util);
            for &ch in &t.child {
                c.child.push(ch as u32);
                c.parent[ch] = i as u32;
            }
            c.child_off.push(c.child.len() as u32);

            // Legal cells, config-major. A leaf and a draw own none.
            c.legal_base.push(if t.leaf || t.chance {
                NO_ROW
            } else {
                let base = c.legal_off.len() as u32;
                let me = t.player as usize;
                for ci in 0..=t.nc(me) {
                    c.legal_off.push(t.legal_off[ci] + c.legal_child.len() as u32);
                }
                for cell in 0..t.legal_action.len() {
                    c.legal_child.push(t.legal_child[cell]);
                    c.legal_trans.push(t.legal_trans[cell]);
                    c.cell_row.push(t.cell_row[cell]);
                }
                base
            });

            // The draw transition, as the node stores it: parent config to
            // child config, with the chance factor.
            c.draw_base.push(if t.chance {
                let base = c.draw_start.len() as u32;
                for r in 0..=t.draw.rows() {
                    c.draw_start.push(t.draw.start[r] + c.draw_to.len() as u32);
                }
                c.draw_to.extend_from_slice(&t.draw.to);
                c.draw_p.extend_from_slice(&t.draw.p);
                base
            } else {
                NO_ROW
            });
        }

        // Children are always built after their parent, so node order is
        // already topological and one pass fixes every level.
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

        c.transpose(sv);
        c
    }

    /// Build the reverse of both transitions, which is the only part of the
    /// contract that is not a re-layout of something `Solver` already holds.
    fn transpose(&mut self, sv: &Solver) {
        let n = self.nodes();
        self.rev_base = vec![NO_ROW; n];
        self.rvd_base = vec![NO_ROW; n];
        // Counting pass per parent, then a fill pass, so each child's rows land
        // contiguously and in parent-cell order — which keeps the sum's
        // floating-point order the same as the scatter it replaces.
        for i in 0..n {
            let t = &sv.nodes[i];
            if t.leaf {
                continue;
            }
            let me = t.player as usize;
            if t.chance {
                let ch = t.child[0] as usize;
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
                continue;
            }
            // A decision node: bucket its cells by the child config each one
            // reaches. The fill walks the child's observations and then its
            // actions, which is exactly the order `propagate` adds them in, so
            // each output's sum keeps its floating-point order and the two
            // agree bit for bit rather than approximately.
            for (ci, &ch_u) in t.child.iter().enumerate() {
                let ch = ch_u as usize;
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
                    for &cell_u in
                        &t.action_cell[t.action_off[a] as usize..t.action_off[a + 1] as usize]
                    {
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

    /// Reach probabilities for every (node, player, config), level by level.
    ///
    /// This is `Solver::propagate` read backwards: the same products, summed by
    /// the thread that owns the output instead of by the thread that owns the
    /// input. It is the host reference the device kernel is held to, and it is
    /// what the equality test in this module compares against the scatter.
    pub fn reach(&self, root: [&[f32]; 2], cur: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        for p in 0..2 {
            let at = self.roff[0] as usize + if p == 1 { self.nc[0][0] as usize } else { 0 };
            out[at..at + root[p].len()].copy_from_slice(root[p]);
        }
        for level in 1..self.levels() {
            let lo = self.level_start[level] as usize;
            let hi = self.level_start[level + 1] as usize;
            for &node_u in &self.level_node[lo..hi] {
                let node = node_u as usize;
                let parent = self.parent[node] as usize;
                let me = self.player[parent] as usize;
                for p in 0..2 {
                    let n = self.nc[node][p] as usize;
                    let dst = self.block(node, p);
                    let src = self.block(parent, p);
                    if p != me {
                        // The idle player's information state does not move,
                        // and the child's support for them is the same list.
                        out.copy_within(src..src + n, dst);
                        continue;
                    }
                    for c in 0..n {
                        out[dst + c] = self.gather(node, c, src, cur, out);
                    }
                }
            }
        }
    }

    /// One output config's reach: the sum over whichever transition feeds it.
    #[inline]
    fn gather(&self, node: usize, c: usize, src: usize, cur: &[f32], out: &[f32]) -> f32 {
        let mut v = 0.0;
        if self.rev_base[node] != NO_ROW {
            let row = (self.rev_base[node] + c as u32) as usize;
            let (lo, hi) = (self.rev_start[row] as usize, self.rev_start[row + 1] as usize);
            for k in lo..hi {
                v += out[src + self.rev_src[k] as usize] * cur[self.rev_cell[k] as usize];
            }
        } else if self.rvd_base[node] != NO_ROW {
            let row = (self.rvd_base[node] + c as u32) as usize;
            let (lo, hi) = (self.rvd_start[row] as usize, self.rvd_start[row + 1] as usize);
            for k in lo..hi {
                v += out[src + self.rvd_src[k] as usize] * self.rvd_p[k];
            }
        }
        v
    }

    /// Where a node's block for one player starts in the reach arena.
    #[inline]
    fn block(&self, node: usize, p: usize) -> usize {
        self.roff[node] as usize + if p == 1 { self.nc[node][0] as usize } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::search::{Cfg, Nets};
    use crate::selfplay::{collect_roots, Agent, Collect, GameCfg};

    fn random_net(seed: u64) -> crate::net::Net {
        let mut r = Rng::new(seed);
        let l = crate::net::NetLayout::new();
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect()
        };
        let (w, b) = (draw(l.w_len), draw(l.b_len));
        let mut ln = vec![0.0; l.ln_len];
        for n in &l.norms {
            ln[n.g..n.g + n.width].fill(1.0);
        }
        crate::net::Net::from_flat(&w, &b, &ln).expect("random net")
    }

    /// The transposed reach must reproduce `Solver::propagate` exactly, not
    /// approximately: the two sum the same products in the same order, so any
    /// difference at all is a wrong edge rather than rounding.
    ///
    /// Run on real roots and after growth, because the transpose is rebuilt
    /// over a tree whose shape changes — a version that only held for the
    /// tree `Solver::new` builds would pass a static test and fail in a solve.
    #[test]
    fn the_transposed_reach_reproduces_the_scatter_exactly() {
        let nets = Nets {
            value: random_net(0x5EED),
            gate: None,
        };
        let cfg = Cfg {
            nodes: 512,
            expand: 4,
            iters: 8,
            ..Default::default()
        };
        let gc = GameCfg {
            agents: [Agent::Rebel {
                cfg: Cfg { nodes: 64, expand: 1, iters: 4, ..cfg },
            }; 2],
            collect: Collect::Rebel,
            explore: 0.1,
            random_draft: true,
            eval_mix: 1.0,
            mc_mix: 0.0,
            query_rate: 0.9,
            recursive_rate: 0.1,
        };
        let roots = collect_roots(3, 7, &nets, &gc, 4);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut rng = Rng::new(0xC047);
        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::rebel::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, &nets, cfg, belief.clone());
            for t in 0..cfg.iters {
                sv.step(t % 2);
                for _ in 0..cfg.expand {
                    if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                        break;
                    }
                }
                // `step` leaves the reaches consistent with `cur`; growth
                // appends rows that only the next propagate fills, so compare
                // straight after re-propagating the tree as it now stands.
                sv.precompute_reaches();
                let c = Contract::of(&sv);
                let mut got = vec![0.0f32; sv.reach.len()];
                let root = [&sv.root_belief[0].p[..], &sv.root_belief[1].p[..]];
                c.reach(root, &sv.cur, &mut got);
                assert_eq!(
                    got.len(),
                    sv.reach.len(),
                    "reach arena length disagrees at iteration {t}"
                );
                for (i, (a, b)) in got.iter().zip(sv.reach.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "reach[{i}] {a} vs {b} at iteration {t}, {} nodes",
                        sv.nodes.len()
                    );
                }
                checked += 1;
            }
        }
        assert!(checked >= 8, "only {checked} comparisons");
    }
}
