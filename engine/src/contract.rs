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

            // Legal cells. The offsets stay node-local, exactly as the node
            // holds them, and the cell arrays are indexed the way `cur` and
            // `regret` are — `soff[node] + cell` — so a task that has found
            // its cell has found its strategy entry too, with no second
            // mapping to carry. A leaf and a draw own no cells.
            c.legal_base.push(if t.leaf || t.chance {
                NO_ROW
            } else {
                let base = c.legal_off.len() as u32;
                let me = t.player as usize;
                c.legal_off.extend_from_slice(&t.legal_off[..=t.nc(me)]);
                let at = sv.soff[i] as usize;
                let end = at + t.legal_action.len();
                if c.legal_child.len() < end {
                    c.legal_child.resize(end, 0);
                    c.legal_trans.resize(end, NO_TRANS);
                    c.cell_row.resize(end, 0);
                }
                c.legal_child[at..end].copy_from_slice(&t.legal_child);
                c.legal_trans[at..end].copy_from_slice(&t.legal_trans);
                c.cell_row[at..end].copy_from_slice(&t.cell_row);
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

    /// One traverser's value backpropagation and regret update, level by level
    /// from the leaves up.
    ///
    /// Unlike the reach sweep this needs no transpose: a config reads its own
    /// legal row, and the row is stored action-ordered, so a thread per
    /// (node, config) sums the same products in the same order as
    /// `Solver::backprop`'s action-major walk. Leaf values must already be in
    /// `vals`; this fills every interior node and leaves `cur` holding the
    /// fresh regret-matching iterate.
    #[allow(clippy::too_many_arguments)]
    pub fn backprop(
        &self,
        traverser: usize,
        cfr: crate::search::Cfr,
        factors: (f32, f32, f32),
        vals: &mut [f32],
        cur: &mut [f32],
        regret: &mut [f32],
        sum: &mut [f32],
    ) {
        const EPS: f32 = 1e-6;
        let (da, db, dg) = factors;
        for level in (0..self.levels()).rev() {
            let lo = self.level_start[level] as usize;
            let hi = self.level_start[level + 1] as usize;
            for &node_u in &self.level_node[lo..hi] {
                let i = node_u as usize;
                if self.kind[i] == KIND_LEAF {
                    continue;
                }
                let me = self.player[i] as usize;
                let nc = self.nc[i][traverser] as usize;
                let vi = self.voff[i] as usize;
                if self.kind[i] == KIND_CHANCE {
                    let ch = self.child[self.child_off[i] as usize] as usize;
                    let cv = self.voff[ch] as usize;
                    if me == traverser {
                        // The chance factor is a real factor of the value,
                        // unlike the traverser's own strategy, which the
                        // counterfactual convention discards.
                        let base = self.draw_base[i] as usize;
                        for c in 0..nc {
                            let (a, b) = (
                                self.draw_start[base + c] as usize,
                                self.draw_start[base + c + 1] as usize,
                            );
                            let mut v = 0.0;
                            for k in a..b {
                                v += self.draw_p[k] * vals[cv + self.draw_to[k] as usize];
                            }
                            vals[vi + c] = v;
                        }
                    } else {
                        vals.copy_within(cv..cv + nc, vi);
                    }
                    continue;
                }
                if me != traverser {
                    // The traverser's information state is unchanged across an
                    // opponent decision, and the opponent's strategy is already
                    // in the reaches the leaf values carry.
                    vals[vi..vi + nc].fill(0.0);
                    for k in self.child_off[i] as usize..self.child_off[i + 1] as usize {
                        let cv = self.voff[self.child[k] as usize] as usize;
                        for c in 0..nc {
                            vals[vi + c] += vals[cv + c];
                        }
                    }
                    continue;
                }
                let so = self.soff[i] as usize;
                let lb = self.legal_base[i] as usize;
                for c in 0..nc {
                    let (a, b) = (
                        self.legal_off[lb + c] as usize,
                        self.legal_off[lb + c + 1] as usize,
                    );
                    let mut base = 0.0f32;
                    for cell in a..b {
                        if self.legal_trans[so + cell] == NO_TRANS {
                            continue;
                        }
                        let cv = self.voff[self.legal_child[so + cell] as usize] as usize;
                        base += vals[cv + self.legal_trans[so + cell] as usize] * cur[so + cell];
                    }
                    vals[vi + c] = base;
                    let mut total = 0.0f32;
                    for cell in a..b {
                        // Re-formed rather than retained: starting at +0 and
                        // adding keeps the arithmetic identical to the CPU's,
                        // including a cell with no successor.
                        let mut delta = 0.0f32;
                        if self.legal_trans[so + cell] != NO_TRANS {
                            let cv = self.voff[self.legal_child[so + cell] as usize] as usize;
                            delta += vals[cv + self.legal_trans[so + cell] as usize];
                        }
                        delta -= base;
                        let old = regret[so + cell];
                        let r = old * if old > 0.0 { da } else { db } + delta;
                        regret[so + cell] = r;
                        let v = (r + cfr.predict * delta).max(EPS);
                        cur[so + cell] = v;
                        total += v;
                    }
                    if total > 0.0 {
                        let inv = 1.0 / total;
                        for cell in a..b {
                            cur[so + cell] *= inv;
                        }
                    }
                }
                let cells = self.legal_off[lb + nc] as usize;
                for x in sum[so..so + cells].iter_mut() {
                    *x *= dg;
                }
            }
        }
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
                sv.step();
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

    /// `t^p / (t^p + 1)`, with the infinities that name "do not discount" and
    /// "discard entirely" evaluated rather than computed. A copy of the
    /// solver's own, which is private to it.
    fn factor(t: f32, p: f32) -> f32 {
        if p.is_infinite() {
            return if p > 0.0 { 1.0 } else { 0.0 };
        }
        let x = t.powf(p);
        x / (x + 1.0)
    }

    /// The contract's backpropagation and regret update must reproduce
    /// `Solver::backprop` exactly. A per-config gather over an action-ordered
    /// legal row sums the same products in the same order as the CPU's
    /// action-major walk, so again this is `to_bits` equality and not a
    /// tolerance.
    #[test]
    fn the_gathered_backprop_reproduces_the_scatter_exactly() {
        use crate::search::Back;
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
        let roots = collect_roots(3, 11, &nets, &gc, 3);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut rng = Rng::new(0xB4CC);
        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::rebel::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, &nets, cfg, belief.clone());
            for t in 0..cfg.iters {
                // Run whole iterations first, so the regrets and the running
                // strategy sum are both non-trivial before anything is
                // compared: a comparison against all-zero state proves nothing
                // about the discount factors.
                sv.step();
                for _ in 0..cfg.expand {
                    if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                        break;
                    }
                }
                sv.precompute_reaches();
                if t < 2 {
                    continue;
                }

                let traverser = t % 2;
                sv.leaf_values(traverser);
                let snap = (
                    sv.vals.clone(),
                    sv.cur.clone(),
                    sv.regret.clone(),
                    sv.sum_strat.clone(),
                );
                let k = sv.cfg.cfr;
                let m = sv.steps[traverser] as f32 + 1.0;
                let fs = (
                    factor(m, k.alpha),
                    factor(m, k.beta),
                    (m / (m + 1.0)).powf(k.gamma),
                );

                sv.backprop(traverser, &[], Back::Regret);
                let want = (
                    sv.vals.clone(),
                    sv.cur.clone(),
                    sv.regret.clone(),
                    sv.sum_strat.clone(),
                );

                // Put the solver back and run the contract over the same state.
                sv.vals = snap.0;
                sv.cur = snap.1;
                sv.regret = snap.2;
                sv.sum_strat = snap.3;
                let c = Contract::of(&sv);
                let mut sum = vec![0.0f32; sv.ncells];
                for i in 0..sv.nodes.len() {
                    let so = sv.soff[i] as usize;
                    let row = &sv.sum_strat[i];
                    sum[so..so + row.len()].copy_from_slice(row);
                }
                let (mut vals, mut cur, mut regret) =
                    (sv.vals.clone(), sv.cur.clone(), sv.regret.clone());
                c.backprop(traverser, k, fs, &mut vals, &mut cur, &mut regret, &mut sum);

                let same = |got: &[f32], want: &[f32], what: &str| {
                    assert_eq!(got.len(), want.len(), "{what} length at iteration {t}");
                    for (i, (a, b)) in got.iter().zip(want).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{what}[{i}] {a} vs {b} at iteration {t}, {} nodes",
                            c.nodes()
                        );
                    }
                };
                same(&vals, &want.0, "vals");
                same(&cur, &want.1, "cur");
                same(&regret, &want.2, "regret");
                for i in 0..sv.nodes.len() {
                    let so = sv.soff[i] as usize;
                    let row = &want.3[i];
                    same(&sum[so..so + row.len()], row, "sum_strat");
                }
                // Restore what the comparison consumed, so the next iteration
                // continues the same solve rather than a diverged one.
                sv.vals = vals;
                sv.cur = cur;
                sv.regret = regret;
                for i in 0..sv.nodes.len() {
                    let so = sv.soff[i] as usize;
                    let n = sv.sum_strat[i].len();
                    sv.sum_strat[i].copy_from_slice(&sum[so..so + n]);
                }
                sv.steps[traverser] += 1;
                checked += 1;
            }
        }
        assert!(checked >= 6, "only {checked} comparisons");
    }
}
