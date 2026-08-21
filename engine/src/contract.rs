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

use crate::farm::{Dst, Writes};
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

    // ------------------------------------------------------------- children
    /// Where a node's children start in `child`, and how many. Not CSR: a leaf
    /// that is grown gets its children appended at the end, long after its
    /// neighbours already have theirs, so the starts are not monotone and a
    /// single offset array cannot bracket them.
    pub child_at: Vec<u32>,
    pub child_n: Vec<u32>,
    pub child: Vec<u32>,

    // ------------------------------- legal cells, CSR by the acting config
    /// Where this node's `nc + 1` offsets start in `legal_off`.
    pub legal_base: Vec<u32>,
    pub legal_off: Vec<u32>,
    pub legal_child: Vec<u32>,
    /// The parent config each cell belongs to, which the average-strategy
    /// accumulation reads and the reverse gather does not.
    pub cell_row: Vec<u32>,
    /// The value slot each cell's child holds for the acting player, or
    /// `NO_ROW` where the cell has no transition. It is
    /// `voff[legal_child[cell]] + trans`, which the backward sweep used to
    /// form for itself: three loads deep, the middle one scattered, for every
    /// cell of every iteration. Tree shape only moves when the tree grows, so
    /// it is resolved once here.
    ///
    /// The transition comes back out of it as `cell_val - voff[legal_child]`,
    /// which is all the expansion walk ever wanted it for, so there is no
    /// array of transitions beside this one.
    pub cell_val: Vec<u32>,

    // --------------------------- the reach transition, transposed, per node
    /// Where this node's `nc + 1` offsets start in `rev_start`, or `NO_ROW`
    /// when its parent is not a decision node.
    pub rev_base: Vec<u32>,
    pub rev_start: Vec<u32>,
    /// The parent cell whose strategy weights this entry. Which of the
    /// parent's configs it belongs to is `cell_row[rev_cell[k]]`, so it is not
    /// stored twice.
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
    /// Nodes already described. Growth only ever appends past this, and
    /// re-describes the one leaf it turned into a decision node.
    pub built: usize,
}

/// How much of each append-only pool the card has already been told about.
///
/// The host owns this now. It used to live on the device side, where the
/// driver looked at how long each of its buffers was -- which meant the driver
/// had to be the one to walk the contract, on the one thread the round could
/// least afford.
#[derive(Default, Clone)]
pub struct Sent {
    /// Nodes described, in `Contract`'s own order.
    pub nodes: usize,
    pools: [usize; 13],
}

impl Contract {
    pub fn nodes(&self) -> usize {
        self.kind.len()
    }

    /// Everything the card has yet to be told about this tree.
    ///
    /// `from` is the first node whose row may have changed -- the earliest leaf
    /// this growth expanded -- and `rewrite` the handful of already-described
    /// rows it turned into decision nodes. Everything else the card holds.
    ///
    /// The appended tail leads because a run is placed by taking the buffer's
    /// address, and it is the only one that can grow the buffer and move it.
    pub fn write_into(&self, w: &mut Writes, sent: &mut Sent, from: usize, rewrite: &[u32]) {
        let n = self.nodes();
        let mut spans: Vec<(usize, usize)> = vec![(from, n - from)];
        spans.extend(rewrite.iter().map(|&g| (g as usize, 1)));
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
        // The pools only ever grow, apart from a rewind, so their tail is the
        // whole of the update.
        let words: [(Dst, &[u32]); 11] = [
            (Dst::Child, &self.child),
            (Dst::LegalOff, &self.legal_off),
            (Dst::LegalChild, &self.legal_child),
            (Dst::CellRow, &self.cell_row),
            (Dst::CellVal, &self.cell_val),
            (Dst::RevStart, &self.rev_start),
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
            let at = sent.pools[11 + i].min(v.len());
            w.f32s(d, at, &v[at..]);
            sent.pools[11 + i] = v.len();
        }
        // Levels are recomputed whenever the tree grows, so they travel whole.
        // It is two entries a node between them.
        w.u32s(Dst::LevelStart, 0, &self.level_start);
        w.u32s(Dst::LevelNode, 0, &self.level_node);
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
            child_at: Vec::with_capacity(n),
            child_n: Vec::with_capacity(n),
            ..Default::default()
        };

        for i in 0..n {
            c.describe(sv, i);
        }
        c.parent = vec![NO_ROW; n];
        for i in 0..n {
            c.link(i);
        }
        c.levels_from(sv);
        c.transpose(sv);
        c.built = n;
        c
    }

    /// Bring a description up to date with a tree that has grown.
    ///
    /// `grown` names the leaves that became decision or chance nodes since the
    /// last call. Everything else is append: the children they gained, and the
    /// nodes those children pulled in behind them.
    ///
    /// Rebuilding instead costs 2.2x the CFR sweeps this description exists to
    /// feed, measured at the frozen budget — so a rebuild per iteration is not
    /// a cadence that can pay for itself at any tree size.
    pub fn extend(&mut self, sv: &Solver, grown: &[u32]) {
        // A node can be created and grown between two calls -- `push_child`
        // grows through a draw or a tactic the moment it makes it -- and the
        // append below describes those in full. Only a leaf that was already
        // described needs its row rewritten.
        for &g in grown {
            let i = g as usize;
            if i < self.built {
                self.redescribe(sv, i);
            }
        }
        for i in self.built..sv.nodes.len() {
            self.parent.push(NO_ROW);
            self.level.push(0);
            self.rev_base.push(NO_ROW);
            self.rvd_base.push(NO_ROW);
            self.describe(sv, i);
        }
        let first = self.built;
        self.built = sv.nodes.len();
        // A node's parent is whoever listed it as a child, which is either a
        // node just grown or one just appended.
        for &g in grown {
            self.link(g as usize);
        }
        for i in first..self.built {
            self.link(i);
        }
        self.levels_from(sv);
        for &g in grown {
            self.transpose_children(sv, g as usize);
        }
        for i in first..self.built {
            self.transpose_children(sv, i);
        }
    }

    /// Point a node's children back at it.
    fn link(&mut self, i: usize) {
        let (a, n) = (self.child_at[i] as usize, self.child_n[i] as usize);
        for k in a..a + n {
            self.parent[self.child[k] as usize] = i as u32;
        }
    }

    /// Rewrite the row of a leaf that has become a decision or chance node.
    /// Everything it now owns is appended, so its old row's variable-length
    /// parts -- of which a leaf has none -- are simply orphaned.
    fn redescribe(&mut self, sv: &Solver, i: usize) {
        let at = self.kind.len();
        self.describe(sv, i);
        let moved = |v: &mut Vec<u32>| {
            v[i] = v[at];
            v.truncate(at);
        };
        self.kind[i] = self.kind[at];
        self.kind.truncate(at);
        self.player[i] = self.player[at];
        self.player.truncate(at);
        self.nc[i] = self.nc[at];
        self.nc.truncate(at);
        self.util[i] = self.util[at];
        self.util.truncate(at);
        moved(&mut self.roff);
        moved(&mut self.voff);
        moved(&mut self.soff);
        moved(&mut self.child_at);
        moved(&mut self.child_n);
        moved(&mut self.legal_base);
        moved(&mut self.draw_base);
    }

    /// Describe node `i`, appending everything it owns.
    ///
    /// Every array a node writes into is either indexed by the node (one push)
    /// or reached through a base this records, so nothing a previous node wrote
    /// moves. That is what lets a grown leaf be re-described in place later.
    fn describe(&mut self, sv: &Solver, i: usize) {
        let c = self;
        {
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
            c.child_at.push(c.child.len() as u32);
            c.child_n.push(t.child.len() as u32);
            // The children are listed but not linked back: a node grown after
            // its neighbours names children that do not exist yet, so `link`
            // runs once every node is present.
            c.child.extend(t.child.iter().map(|&ch| ch as u32));

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
                    c.cell_row.resize(end, 0);
                    c.cell_val.resize(end, NO_ROW);
                }
                c.legal_child[at..end].copy_from_slice(&t.legal_child);
                c.cell_row[at..end].copy_from_slice(&t.cell_row);
                for cell in 0..t.legal_action.len() {
                    let tr = t.legal_trans[cell];
                    c.cell_val[at + cell] = if tr == NO_TRANS {
                        NO_ROW
                    } else {
                        sv.voff[t.legal_child[cell] as usize] + tr
                    };
                }
                base
            });

            // The draw transition, as the node stores it: parent config to
            // child config, with the chance factor.
            let _ = &t;
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
    }

    /// Depth from the root for every node, and the nodes bucketed by it.
    ///
    /// Children are always built after their parent, so node order is already
    /// topological and one forward pass fixes every level.
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

    /// Build the reverse of both transitions, which is the only part of the
    /// contract that is not a re-layout of something `Solver` already holds.
    fn transpose(&mut self, sv: &Solver) {
        let n = self.nodes();
        self.rev_base = vec![NO_ROW; n];
        self.rvd_base = vec![NO_ROW; n];
        for i in 0..n {
            self.transpose_children(sv, i);
        }
    }

    /// The reverse transition into each of node `i`'s children.
    ///
    /// A counting pass then a fill pass, so a child's rows land contiguously
    /// and in parent-cell order — which keeps the sum's floating-point order
    /// the same as the scatter it replaces.
    fn transpose_children(&mut self, sv: &Solver, i: usize) {
        {
            let t = &sv.nodes[i];
            if t.leaf {
                return;
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
                return;
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
                let at = self.rev_cell.len() as u32;
                self.rev_start.extend(count.iter().map(|x| x + at));
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
                let pc = self.rev_cell[k] as usize;
                v += out[src + self.cell_row[pc] as usize] * cur[pc];
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

    /// The counterfactual value of one cell: the slot its child holds for the
    /// acting player, or zero where the cell reaches no information state of
    /// theirs. `action_value` in `kernels.cu` is the same three lines.
    #[inline]
    fn action_value(&self, vals: &[f32], cell: usize) -> f32 {
        let vc = self.cell_val[cell];
        if vc == NO_ROW { 0.0 } else { vals[vc as usize] }
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
                    let ch = self.child[self.child_at[i] as usize] as usize;
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
                    let (a, n) = (self.child_at[i] as usize, self.child_n[i] as usize);
                    for k in a..a + n {
                        let cv = self.voff[self.child[k] as usize] as usize;
                        for c in 0..nc {
                            vals[vi + c] += vals[cv + c];
                        }
                    }
                    continue;
                }
                let so = self.soff[i] as usize;
                let lb = self.legal_base[i] as usize;
                let cells = self.legal_off[lb + nc] as usize;
                for c in 0..nc {
                    let (a, b) = (
                        self.legal_off[lb + c] as usize,
                        self.legal_off[lb + c + 1] as usize,
                    );
                    let mut base = 0.0f32;
                    for cell in a..b {
                        base += self.action_value(vals, so + cell) * cur[so + cell];
                    }
                    vals[vi + c] = base;
                    let mut total = 0.0f32;
                    for cell in a..b {
                        let delta = self.action_value(vals, so + cell) - base;
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
            device: false,
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

    /// A whole solve driven through the flat description must reach the same
    /// strategy as one driven by the tree walk.
    ///
    /// The per-iteration tests pin the two sweeps against each other on a tree
    /// that the *walk* advanced. This drives the solve from the description
    /// instead, so an error in the sequencing -- extending too late, missing a
    /// growth, feeding a stale level table -- compounds across sixty-four
    /// iterations rather than being corrected by the next comparison. That is
    /// the failure a device would actually have.
    #[test]
    fn a_solve_driven_from_the_description_reaches_the_same_strategy() {
        let nets = Nets {
            value: random_net(0x5EED),
            device: false,
            gate: None,
        };
        let cfg = Cfg {
            nodes: 384,
            expand: 4,
            iters: 12,
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
        let roots = collect_roots(2, 47, &nets, &gc, 2);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::rebel::Ctx::new(s);
            // The same solve twice, from the same seed, so the expansion
            // trajectories match and only the sweep differs.
            let run = |flat: bool| {
                let mut rng = Rng::new(0xD12E);
                let mut sv = crate::search::Solver::new(s, ctx, &nets, cfg, belief.clone());
                let mut c = Contract::of(&sv);
                sv.grown.clear();
                for t in 0..cfg.iters {
                    if flat {
                        let grown = std::mem::take(&mut sv.grown);
                        c.extend(&sv, &grown);
                        let k = sv.cfg.cfr;
                        for p in 0..2 {
                            let m = sv.steps[p] as f32 + 1.0;
                            let fs = (
                                factor(m, k.alpha),
                                factor(m, k.beta),
                                (m / (m + 1.0)).powf(k.gamma),
                            );
                            sv.leaf_values(p);
                            let (mut vals, mut cur, mut regret) =
                                (sv.vals.clone(), sv.cur.clone(), sv.regret.clone());
                            let mut sum = vec![0.0f32; sv.ncells];
                            for i in 0..sv.nodes.len() {
                                let so = sv.soff[i] as usize;
                                let row = &sv.sum_strat[i];
                                sum[so..so + row.len()].copy_from_slice(row);
                            }
                            c.backprop(p, k, fs, &mut vals, &mut cur, &mut regret, &mut sum);
                            // The solver keeps one value arena where the device
                            // keeps one per traverser, so its expansion phase
                            // cannot re-form Q at selection time the way
                            // `k_expand` does. Capture it from the pass that has
                            // just made it current.
                            for i in 0..sv.nodes.len() {
                                let n = &sv.nodes[i];
                                if n.leaf || n.chance || n.player as usize != p {
                                    continue;
                                }
                                let so = sv.soff[i] as usize;
                                for cell in 0..n.legal_action.len() {
                                    sv.qval[so + cell] = c.action_value(&vals, so + cell);
                                }
                            }
                            sv.vals = vals;
                            sv.cur = cur;
                            sv.regret = regret;
                            for i in 0..sv.nodes.len() {
                                let so = sv.soff[i] as usize;
                                let n = sv.sum_strat[i].len();
                                sv.sum_strat[i].copy_from_slice(&sum[so..so + n]);
                            }
                        }
                        let root = [&sv.root_belief[0].p[..], &sv.root_belief[1].p[..]];
                        let mut out = vec![0.0f32; sv.reach.len()];
                        c.reach(root, &sv.cur, &mut out);
                        sv.reach = out;
                        sv.avg_block();
                        sv.steps[0] += 1;
                        sv.steps[1] += 1;
                    } else {
                        sv.step();
                    }
                    for _ in 0..cfg.expand {
                        if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                            break;
                        }
                    }
                    let _ = t;
                }
                sv.finish();
                (sv.avg.clone(), sv.cur.clone(), sv.regret.clone())
            };
            let (a_avg, a_cur, a_reg) = run(false);
            let (b_avg, b_cur, b_reg) = run(true);
            for (name, x, y) in [
                ("avg", &a_avg, &b_avg),
                ("cur", &a_cur, &b_cur),
                ("regret", &a_reg, &b_reg),
            ] {
                assert_eq!(x.len(), y.len(), "{name} length differs");
                for (k, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                    assert_eq!(p.to_bits(), q.to_bits(), "{name}[{k}] differs");
                }
            }
            checked += 1;
        }
        assert!(checked > 0, "no solve compared");
    }

    /// Every node of a level must have its parent in an earlier one.
    ///
    /// This is the property that makes a level's work independent, and it is
    /// the only reason a device may run a whole level's tasks at once. Nothing
    /// else checks it: a sweep over a mis-levelled tree still produces numbers,
    /// they are just read before they are written, and the answer is wrong in a
    /// way no downstream assertion notices.
    #[test]
    fn a_level_never_depends_on_itself() {
        let nets = Nets {
            value: random_net(0x5EED),
            device: false,
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
        let roots = collect_roots(2, 31, &nets, &gc, 2);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut rng = Rng::new(0x1E4E);
        let mut deepest = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::rebel::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, &nets, cfg, belief.clone());
            for _ in 0..cfg.iters {
                sv.step();
                for _ in 0..cfg.expand {
                    if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                        break;
                    }
                }
            }
            let c = Contract::of(&sv);
            assert!(c.levels() > 1, "a solved tree has more than one level");
            deepest = deepest.max(c.levels());

            let mut seen = vec![false; c.nodes()];
            for level in 0..c.levels() {
                let (lo, hi) = (c.level_start[level] as usize, c.level_start[level + 1] as usize);
                assert!(hi > lo, "level {level} is empty");
                for &node in &c.level_node[lo..hi] {
                    let i = node as usize;
                    assert_eq!(c.level[i] as usize, level, "node {i} is in the wrong bucket");
                    let p = c.parent[i];
                    if p == NO_ROW {
                        assert_eq!(level, 0, "only the root has no parent");
                    } else {
                        assert!(
                            seen[p as usize],
                            "node {i} at level {level} has parent {p} in the same level or later"
                        );
                    }
                }
                for &node in &c.level_node[lo..hi] {
                    seen[node as usize] = true;
                }
            }
            assert!(seen.iter().all(|&x| x), "a node belongs to no level");
        }
        println!("levels check ok, deepest tree {deepest} levels");
    }

    /// An incrementally extended description must equal one built from
    /// scratch, field for field.
    ///
    /// This is the whole safety of the append-only form. A description that
    /// drifts from the tree it claims to describe produces a sweep that is
    /// quietly solving a different game, and nothing downstream would say so.
    /// What the card ends up holding must be what the contract says.
    ///
    /// The description is sent as a delta -- the tail each growth appended and
    /// the handful of rows it rewrote -- so a wrong offset shows up not as a
    /// missing array but as a stale one, and only after the growth that should
    /// have replaced it. This mirrors the card: apply every run to plain
    /// vectors, exactly as the scatter kernel does, and require the result to
    /// equal the contract at every step.
    #[test]
    fn the_runs_a_solve_sends_rebuild_its_contract() {
        let nets = Nets { value: random_net(0x5EED), device: false, gate: None };
        let cfg = Cfg { nodes: 512, expand: 4, iters: 10, ..Default::default() };
        let gc = GameCfg {
            agents: [Agent::Rebel { cfg: Cfg { nodes: 64, expand: 1, iters: 4, ..cfg } }; 2],
            collect: Collect::Rebel,
            explore: 0.1,
            random_draft: true,
            eval_mix: 1.0,
            mc_mix: 0.0,
            query_rate: 0.9,
            recursive_rate: 0.1,
        };
        let roots = collect_roots(3, 23, &nets, &gc, 3);
        assert!(!roots.is_empty(), "no roots to test against");
        let mut rng = Rng::new(0xE47E);
        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::rebel::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, &nets, cfg, belief.clone());
            let mut inc = Contract::of(&sv);
            sv.grown.clear();
            // What the card holds, one vector an array.
            let mut card: Vec<Vec<u32>> = vec![Vec::new(); 35];
            let mut sent = Sent::default();
            let mut from = 0usize;
            let mut rewrite: Vec<u32> = Vec::new();
            for _ in 0..cfg.iters {
                let mut w = Writes::default();
                inc.write_into(&mut w, &mut sent, from, &rewrite);
                for r in &w.runs {
                    let v = &mut card[r.dst as usize];
                    let end = (r.at + r.len) as usize;
                    if v.len() < end {
                        v.resize(end, 0);
                    }
                    let src = r.start as usize;
                    v[r.at as usize..end].copy_from_slice(&w.blob[src..src + r.len as usize]);
                }
                let got = |d: Dst| &card[d as usize];
                let wide = |v: &[u8]| v.iter().map(|&x| x as u32).collect::<Vec<u32>>();
                assert_eq!(got(Dst::Kind), &wide(&inc.kind), "kind");
                assert_eq!(got(Dst::Player), &wide(&inc.player), "player");
                assert_eq!(got(Dst::Nc), &inc.nc.iter().flatten().copied().collect::<Vec<u32>>(), "nc");
                assert_eq!(got(Dst::Parent), &inc.parent, "parent");
                assert_eq!(got(Dst::Roff), &inc.roff, "roff");
                assert_eq!(got(Dst::Voff), &inc.voff, "voff");
                assert_eq!(got(Dst::Soff), &inc.soff, "soff");
                assert_eq!(got(Dst::Util), &inc.util.iter().map(|x| x.to_bits()).collect::<Vec<u32>>(), "util");
                assert_eq!(got(Dst::ChildAt), &inc.child_at, "child_at");
                assert_eq!(got(Dst::ChildN), &inc.child_n, "child_n");
                assert_eq!(got(Dst::Child), &inc.child, "child");
                assert_eq!(got(Dst::LegalBase), &inc.legal_base, "legal_base");
                assert_eq!(got(Dst::LegalOff), &inc.legal_off, "legal_off");
                assert_eq!(got(Dst::LegalChild), &inc.legal_child, "legal_child");
                assert_eq!(got(Dst::CellRow), &inc.cell_row, "cell_row");
                assert_eq!(got(Dst::CellVal), &inc.cell_val, "cell_val");
                assert_eq!(got(Dst::RevBase), &inc.rev_base, "rev_base");
                assert_eq!(got(Dst::RevStart), &inc.rev_start, "rev_start");
                assert_eq!(got(Dst::RevCell), &inc.rev_cell, "rev_cell");
                assert_eq!(got(Dst::RvdBase), &inc.rvd_base, "rvd_base");
                assert_eq!(got(Dst::RvdStart), &inc.rvd_start, "rvd_start");
                assert_eq!(got(Dst::RvdSrc), &inc.rvd_src, "rvd_src");
                assert_eq!(got(Dst::DrawBase), &inc.draw_base, "draw_base");
                assert_eq!(got(Dst::DrawStart), &inc.draw_start, "draw_start");
                assert_eq!(got(Dst::DrawTo), &inc.draw_to, "draw_to");
                assert_eq!(got(Dst::LevelStart), &inc.level_start, "level_start");
                assert_eq!(got(Dst::LevelNode), &inc.level_node, "level_node");
                checked += 1;

                sv.step();
                for _ in 0..cfg.expand {
                    if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                        break;
                    }
                }
                let grown = std::mem::take(&mut sv.grown);
                from = inc.built;
                rewrite = grown.iter().copied().filter(|&g| (g as usize) < from).collect();
                inc.extend(&sv, &grown);
            }
        }
        assert!(checked > 10, "only {checked} descriptions compared");
    }

    #[test]

    fn extending_a_contract_equals_rebuilding_it() {
        let nets = Nets {
            value: random_net(0x5EED),
            device: false,
            gate: None,
        };
        let cfg = Cfg {
            nodes: 512,
            expand: 4,
            iters: 10,
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
        let roots = collect_roots(3, 23, &nets, &gc, 3);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut rng = Rng::new(0xE47E);
        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::rebel::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, &nets, cfg, belief.clone());
            let mut inc = Contract::of(&sv);
            sv.grown.clear();
            for t in 0..cfg.iters {
                sv.step();
                for _ in 0..cfg.expand {
                    if sv.nodes.len() >= cfg.nodes || !sv.expand_once(&mut rng) {
                        break;
                    }
                }
                let grown = std::mem::take(&mut sv.grown);
                inc.extend(&sv, &grown);
                let full = Contract::of(&sv);

                // The layout is deliberately not compared. An extended
                // description appends a grown node's children at the end of
                // `child`, where a rebuild lays them in node order, so the
                // offsets differ while the meaning does not. What must agree
                // is what the description is *for*: the sweeps it drives.
                macro_rules! same {
                    ($($f:ident),*) => {$(
                        assert_eq!(
                            inc.$f, full.$f,
                            concat!(stringify!($f), " differs at iteration {}"),
                            t
                        );
                    )*};
                }
                same!(kind, player, nc, parent, level, roff, voff, soff, util);
                same!(child_n, level_start, level_node);
                assert_eq!(inc.built, full.built, "built differs");
                for i in 0..full.nodes() {
                    let kids = |c: &Contract| {
                        let (a, n) = (c.child_at[i] as usize, c.child_n[i] as usize);
                        c.child[a..a + n].to_vec()
                    };
                    assert_eq!(kids(&inc), kids(&full), "node {i}'s children differ");
                }

                let root = [&sv.root_belief[0].p[..], &sv.root_belief[1].p[..]];
                let (mut ra, mut rb) = (vec![0.0; sv.reach.len()], vec![0.0; sv.reach.len()]);
                inc.reach(root, &sv.cur, &mut ra);
                full.reach(root, &sv.cur, &mut rb);
                for (k, (x, y)) in ra.iter().zip(&rb).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "reach[{k}] differs at iteration {t}");
                }

                let k = sv.cfg.cfr;
                let fs = (factor(2.0, k.alpha), factor(2.0, k.beta), 0.5);
                let mut sum = vec![0.0f32; sv.ncells];
                let run = |c: &Contract| {
                    let (mut v, mut cu, mut rg, mut sm) =
                        (sv.vals.clone(), sv.cur.clone(), sv.regret.clone(), sum.clone());
                    c.backprop(0, k, fs, &mut v, &mut cu, &mut rg, &mut sm);
                    (v, cu, rg, sm)
                };
                let (va, ca, ga, sa) = run(&inc);
                let (vb, cb, gb, sb) = run(&full);
                for (name, x, y) in [
                    ("vals", &va, &vb),
                    ("cur", &ca, &cb),
                    ("regret", &ga, &gb),
                    ("sum", &sa, &sb),
                ] {
                    for (k, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                        assert_eq!(
                            p.to_bits(),
                            q.to_bits(),
                            "{name}[{k}] differs at iteration {t}"
                        );
                    }
                }
                sum.clear();
                checked += 1;
            }
        }
        assert!(checked >= 10, "only {checked} comparisons");
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
            device: false,
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
