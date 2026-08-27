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
use crate::search::{Ent, Solver, NO_TRANS};

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
    /// Whether the subtree under a node holds no expandable leaf. The
    /// expansion trajectories read it, and it is the one per-node fact that is
    /// not append-only: sealing a leaf seals a chain of its ancestors, so
    /// `write_into` resends exactly those rows.
    pub exhausted: Vec<u32>,
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
    pub legal_trans: Vec<u32>,
    /// The parent config each cell belongs to, which the average-strategy
    /// accumulation reads and the reverse gather does not.
    pub cell_row: Vec<u32>,
    /// The value slot each cell's child holds for the acting player, or
    /// `NO_ROW` where the cell has no transition. The backward sweep used to
    /// find it with `voff[legal_child[cell]] + legal_trans[cell]`: three loads
    /// deep, the middle one scattered, for every cell of every iteration. Tree
    /// shape only moves when the tree grows, so it is resolved once here.
    pub cell_val: Vec<u32>,

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
    pools: [usize; 15],
}

impl Contract {
    pub fn nodes(&self) -> usize {
        self.kind.len()
    }

    /// Host bytes this description holds. The farm admits solves against it.
    pub fn bytes(&self) -> usize {
        let u = |v: &Vec<u32>| v.capacity() * 4;
        self.nc.capacity() * 8
            + self.kind.capacity()
            + self.player.capacity()
            + u(&self.parent)
            + u(&self.exhausted)
            + u(&self.level)
            + u(&self.roff)
            + u(&self.voff)
            + u(&self.soff)
            + self.util.capacity() * 4
            + u(&self.child_at)
            + u(&self.child_n)
            + u(&self.child)
            + u(&self.legal_base)
            + u(&self.legal_off)
            + u(&self.legal_child)
            + u(&self.legal_trans)
            + u(&self.cell_row)
            + u(&self.cell_val)
            + u(&self.rev_base)
            + u(&self.rev_start)
            + u(&self.rev_src)
            + u(&self.rev_cell)
            + u(&self.draw_base)
            + u(&self.draw_start)
            + u(&self.draw_to)
            + self.draw_p.capacity() * 4
            + u(&self.rvd_base)
            + u(&self.rvd_start)
            + u(&self.rvd_src)
            + self.rvd_p.capacity() * 4
            + u(&self.level_start)
            + u(&self.level_node)
    }

    /// Everything the card has yet to be told about this tree.
    ///
    /// `from` is the first node whose row may have changed -- the earliest leaf
    /// this growth expanded -- and `rewrite` the handful of already-described
    /// rows it turned into decision nodes. Everything else the card holds.
    ///
    /// The appended tail leads because a run is placed by taking the buffer's
    /// address, and it is the only one that can grow the buffer and move it.
    pub fn write_into(
        &self,
        w: &mut Writes,
        sent: &mut Sent,
        from: usize,
        rewrite: &[u32],
        resealed: &[u32],
    ) {
        let n = self.nodes();
        let mut spans: Vec<(usize, usize)> = vec![(from, n - from)];
        spans.extend(rewrite.iter().map(|&g| (g as usize, 1)));
        // Exhaustion is the one row that changes without the node being
        // regrown, so the appended tail carries it and every older row that
        // moved goes on its own.
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
        // The pools only ever grow, apart from a rewind, so their tail is the
        // whole of the update.
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
        c.levels_from(sv);
        c.transpose(sv);
        c.built = n;
        c.check(sv);
        c
    }

    /// Every column of an entity is at most `Solver::used` for that entity.
    /// `level_start` is a prefix-sum of length `n_levels + 1`, so it is allowed
    /// one past the node count.
    fn check(&self, sv: &Solver) {
        let u = |e: Ent| sv.used(e);
        debug_assert_eq!(self.kind.len(), u(Ent::Node));
        debug_assert_eq!(self.level_node.len(), u(Ent::Node));
        debug_assert!(self.level_start.len() <= u(Ent::Node) + 1);
        for (n, what) in [
            (self.child.len(), "child"),
            (self.legal_child.len(), "legal_child"),
            (self.legal_trans.len(), "legal_trans"),
            (self.cell_row.len(), "cell_row"),
            (self.cell_val.len(), "cell_val"),
            (self.rev_src.len(), "rev_src"),
            (self.rev_cell.len(), "rev_cell"),
        ] {
            debug_assert!(n <= u(Ent::Cell), "{what} {n} > cell {}", u(Ent::Cell));
        }
        for (n, what) in [
            (self.legal_off.len(), "legal_off"),
            (self.rev_start.len(), "rev_start"),
            (self.rvd_start.len(), "rvd_start"),
            (self.draw_start.len(), "draw_start"),
        ] {
            debug_assert!(n <= u(Ent::Reach), "{what} {n} > reach {}", u(Ent::Reach));
        }
        debug_assert!(self.draw_to.len() <= u(Ent::Draw));
        debug_assert!(self.rvd_src.len() <= u(Ent::Draw));
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
    pub fn extend(&mut self, sv: &Solver, grown: &[u32], resealed: &[u32]) {
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
            self.level.push(0);
            self.rev_base.push(NO_ROW);
            self.rvd_base.push(NO_ROW);
            self.describe(sv, i);
        }
        let first = self.built;
        self.built = sv.nodes.len();
        // Sealing a leaf seals its ancestors, which are described already.
        for &r in resealed {
            let i = r as usize;
            if i < first {
                self.exhausted[i] = sv.nodes[i].exhausted as u32;
            }
        }
        self.levels_from(sv);
        // A node created and grown in the same step is in both `grown` and
        // `first..built`. Transpose it once; a second pass would append another
        // copy of its reverse edges and the slot is not that large.
        for &g in grown {
            if (g as usize) < first {
                self.transpose_children(sv, g as usize);
            }
        }
        for i in first..self.built {
            self.transpose_children(sv, i);
        }
        self.check(sv);
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
        moved(&mut self.exhausted);
        moved(&mut self.parent);
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
            c.exhausted.push(t.exhausted as u32);
            c.parent.push(sv.parent[i]);
            c.nc.push(sv.nc[i]);
            c.roff.push(sv.roff[i]);
            c.voff.push(sv.voff[i]);
            c.soff.push(sv.soff[i]);
            c.util.push(t.util);
            c.child_at.push(c.child.len() as u32);
            c.child_n.push(t.child.len() as u32);
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
    /// legal row, so a thread per (node, config) gathers the same products as
    /// `Solver::backprop`. The reduction order differs between the flat and
    /// action-major layouts, so callers compare the f32 results within a small
    /// error. Leaf values must already be in `vals`; this fills every interior
    /// node and leaves `cur` holding the fresh regret-matching iterate.
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
        qval: &mut [f32],
    ) {
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
                // The expansion phase reads these as PUCT's Q. A sweep that
                // computes the values but drops them leaves selection blind,
                // and the tree it grows is a different tree.
                let cells = self.legal_off[lb + nc] as usize;
                qval[so..so + cells].fill(0.0);
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
                        let av = vals[cv + self.legal_trans[so + cell] as usize];
                        qval[so + cell] = av;
                        base += av * cur[so + cell];
                    }
                    vals[vi + c] = base;
                    let mut total = 0.0f32;
                    for cell in a..b {
                        let delta = qval[so + cell] - base;
                        let old = regret[so + cell];
                        let r = old * if old > 0.0 { da } else { db } + delta;
                        regret[so + cell] = r;
                        let v = (r + cfr.predict * delta).max(0.0);
                        cur[so + cell] = v;
                        total += v;
                    }
                    if total > 0.0 {
                        let inv = 1.0 / total;
                        for cell in a..b {
                            cur[so + cell] *= inv;
                        }
                    } else {
                        cur[so + a..so + b].fill(1.0 / (b - a) as f32);
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

    use crate::search::Cfg;
    use std::sync::Arc;
    use crate::selfplay::collect_roots;
    use crate::board::N_HEXES;
    use crate::pbs::NSLOT;

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
        let nets = Arc::new(random_net(0x5EED));
        let cfg = Cfg { s: 32, c: 4.0, ..Default::default() };
        let roots = collect_roots(4, 7);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::pbs::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, Arc::clone(&nets), cfg, belief.clone(), Rng::new(0xD12E));
            for t in 0..cfg.iters() {
                sv.catch_up();
                sv.step();
                for _ in 0..4 {
                    if !sv.expand_once() {
                        break;
                    }
                }
                // `step` leaves the reaches consistent with `cur`; growth
                // appends rows that only the next propagate fills, so compare
                // straight after re-propagating the tree as it now stands.
                sv.precompute_reaches();
                let c = Contract::of(&sv);
                let mut got = vec![0.0f32; sv.cfr().reach.len()];
                let root = [&sv.root_belief[0].p[..], &sv.root_belief[1].p[..]];
                c.reach(root, &sv.cur, &mut got);
                assert_eq!(
                    got.len(),
                    sv.cfr().reach.len(),
                    "reach arena length disagrees at iteration {t}"
                );
                for (i, (a, b)) in got.iter().zip(sv.cfr().reach.iter()).enumerate() {
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

    /// The action words a solve sends must rebuild the policy head's own input.
    ///
    /// The card runs the action encoder now, and the one thing it cannot know
    /// is what an action *is*. The five words a node sends are a
    /// `Net::action_feats` one-hot in the making, and a column out of place is
    /// a policy prior that is wrong everywhere and finite everywhere.
    #[test]
    fn the_action_words_a_solve_sends_rebuild_its_one_hot() {
        use crate::net::{Net, AFEAT};
        let nets = Arc::new(random_net(0x5EED));
        let cfg = Cfg { s: 24, c: 2.0, ..Default::default() };
        let roots = collect_roots(2, 0x51E5);
        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::pbs::Ctx::new(s);
            let mut sv =
                crate::search::Solver::new(s, ctx, Arc::clone(&nets), cfg, belief.clone(), Rng::new(0xAC75));
            // A prior is only owed once the batch has reached the node's row,
            // so the tree has to be grown and its calls answered first.
            for _ in 0..cfg.iters() {
                sv.catch_up();
                sv.step();
                sv.expand_once();
            }
            sv.catch_up();
            let (prime, acts, cells) = sv.prime();
            assert!(!prime.is_empty(), "no node was ready for a prior");
            // The blocks `Net::action_feats` writes, in order: the kind, the
            // coin slot with a column for "spends nothing", and three hexes
            // each with a column for "names none".
            let widths = [crate::actions::N_KINDS, NSLOT + 1, N_HEXES + 1, N_HEXES + 1, N_HEXES + 1];
            for q in &prime {
                let n = &sv.nodes[q.node as usize];
                assert_eq!(q.na as usize, n.na(), "action count");
                assert_eq!(q.nc as usize, n.nc(n.player as usize), "config count");
                for a in 0..n.na() {
                    let d = &acts[5 * (q.at as usize + a)..][..5];
                    let mut got = vec![0.0f32; AFEAT];
                    let mut at = 0;
                    for (k, w) in widths.iter().enumerate() {
                        got[at + d[k] as usize] = 1.0;
                        at += w;
                    }
                    let mut want = vec![0.0f32; AFEAT];
                    Net::action_feats(n.acts[a].kind(), n.aslot[a], n.acts[a].hexes(), &mut want);
                    assert_eq!(got, want, "action {a} of node {}", q.node);
                }
                let at = q.cell_at as usize;
                assert_eq!(
                    &cells[at..at + n.legal_action.len()],
                    &n.legal_action[..],
                    "the cells of node {}",
                    q.node
                );
                checked += 1;
            }
        }
        assert!(checked > 4, "only {checked} nodes described");
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
        let nets = Arc::new(random_net(0x5EED));
        let cfg = Cfg {
            s: 32,
            c: 4.0,
            ..Default::default()
        };
        let roots = collect_roots(2, 31);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut deepest = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::pbs::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, Arc::clone(&nets), cfg, belief.clone(), Rng::new(0xD12E));
            for _ in 0..cfg.iters() {
                sv.catch_up();
                sv.step();
                for _ in 0..4 {
                    if !sv.expand_once() {
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
        let nets = Arc::new(random_net(0x5EED));
        let cfg = Cfg { s: 40, c: 4.0, ..Default::default() };
        let roots = collect_roots(3, 23);
        assert!(!roots.is_empty(), "no roots to test against");
        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::pbs::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, Arc::clone(&nets), cfg, belief.clone(), Rng::new(0xD12E));
            let mut inc = Contract::of(&sv);
            sv.grown.clear();
            // What the card holds, one vector an array.
            let mut card: Vec<Vec<u32>> = vec![Vec::new(); Dst::Rootb as usize + 1];
            let mut sent = Sent::default();
            let mut from = 0usize;
            let mut rewrite: Vec<u32> = Vec::new();
            // `Contract::of` already describes the root's own expansion.
            let mut resealed = sv.take_resealed();
            for _ in 0..cfg.iters() {
                let mut w = Writes::default();
                inc.write_into(&mut w, &mut sent, from, &rewrite, &resealed);
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
                // The one row that changes without the node being regrown, so
                // the one the incremental write can silently leave behind.
                assert_eq!(got(Dst::Exhausted), &inc.exhausted, "exhausted");
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
                assert_eq!(got(Dst::LegalTrans), &inc.legal_trans, "legal_trans");
                assert_eq!(got(Dst::CellRow), &inc.cell_row, "cell_row");
                assert_eq!(got(Dst::CellVal), &inc.cell_val, "cell_val");
                assert_eq!(got(Dst::RevBase), &inc.rev_base, "rev_base");
                assert_eq!(got(Dst::RevStart), &inc.rev_start, "rev_start");
                assert_eq!(got(Dst::RevSrc), &inc.rev_src, "rev_src");
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

                sv.catch_up();

                sv.step();
                for _ in 0..4 {
                    if !sv.expand_once() {
                        break;
                    }
                }
                let grown = std::mem::take(&mut sv.grown);
                from = inc.built;
                rewrite = grown.iter().copied().filter(|&g| (g as usize) < from).collect();
                resealed = sv.take_resealed();
                inc.extend(&sv, &grown, &resealed);
            }
        }
        assert!(checked > 10, "only {checked} descriptions compared");
    }

    #[test]

    fn extending_a_contract_equals_rebuilding_it() {
        let nets = Arc::new(random_net(0x5EED));
        let cfg = Cfg {
            s: 40,
            c: 4.0,
            ..Default::default()
        };
        let roots = collect_roots(3, 23);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut checked = 0usize;
        for (s, belief) in &roots {
            let ctx = crate::pbs::Ctx::new(s);
            let mut sv = crate::search::Solver::new(s, ctx, Arc::clone(&nets), cfg, belief.clone(), Rng::new(0xD12E));
            let mut inc = Contract::of(&sv);
            sv.grown.clear();
            for t in 0..cfg.iters() {
                sv.catch_up();
                sv.step();
                for _ in 0..4 {
                    if !sv.expand_once() {
                        break;
                    }
                }
                let grown = std::mem::take(&mut sv.grown);
                let resealed = sv.take_resealed();
                inc.extend(&sv, &grown, &resealed);
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
                let (mut ra, mut rb) = (vec![0.0; sv.cfr().reach.len()], vec![0.0; sv.cfr().reach.len()]);
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
                        (sv.cfr().vals.clone(), sv.cur.clone(), sv.cfr().regret.clone(), sum.clone());
                    let mut qv = vec![0.0f32; sv.ncells];
                    c.backprop(0, k, fs, &mut v, &mut cu, &mut rg, &mut sm, &mut qv);
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

}
