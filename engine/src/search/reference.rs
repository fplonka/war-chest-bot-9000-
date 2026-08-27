use super::*;
mod growth;
mod sampling;
pub use sampling::Arenas;

/// Host-oracle work summed over the iterations that ran on it.
#[derive(Default, Clone, Copy, Debug)]
pub struct Trace {
    pub iters: u64,
    pub row_iters: u64,
    pub cidx_iters: u64,
    pub cell_iters: u64,
    pub join_rows: u64,
    pub readout_cfgs: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Conv {
    /// Exploitability of the reference average strategy.
    pub nash: f32,
    /// Sum of both root values; nonzero network leaves need not be antisymmetric.
    pub zero_sum: f32,
}

/// The operation performed by an exact host backward pass.
#[derive(Clone, Copy, PartialEq)]
pub enum Back {
    Regret,
    Value,
    BestResponse,
}

#[derive(Default)]
pub struct HostCfr {
    /// Accumulated regret, laid out exactly like `Solver::cur`.
    pub regret: Vec<f32>,
    /// The expansion phase's own statistics, in the same layout.
    ///
    /// `prior` is the policy head's `softmax(logit(c, a) / prior_temp)` over a
    /// config's legal row, filled once when the node is expanded. `visits` are
    /// PUCT's counts, accumulated over every expansion phase of the search and
    /// incremented as a trajectory passes — which is also the virtual loss,
    /// since later simulations of the same phase then see the earlier ones.
    /// `qval` is the action value the last backprop formed, before it was
    /// turned into a regret. The device keeps no such array -- it holds a
    /// value arena per traverser, so `k_expand` re-forms Q out of `cell_val`
    /// where it selects. Here there is one arena and each traverser's pass
    /// overwrites it, so the number has to be kept as it is made.
    pub prior: Vec<f32>,
    pub visits: Vec<f32>,
    pub qval: Vec<f32>,
    /// The reach-weighted running strategy sum, per node. Per node rather than
    /// flat, because a node is given its cells when it is expanded and a
    /// ragged vector grows there without disturbing anything already summed.
    pub sum_strat: Vec<Vec<f32>>,
    /// Reach per config, flat: node `i`'s two players occupy
    /// `reach[roff[i] .. roff[i] + nc0 + nc1]`, player 0 first. One arena
    /// rather than `Vec<Vec<f32>>` — the CFR passes touch every node, and two
    /// pointer hops per node is what they were spending their time on.
    pub reach: Vec<f32>,
    /// The traverser's counterfactual value per config, flat the same way:
    /// `vals[voff[i] .. voff[i] + max(nc0, nc1)]`.
    pub vals: Vec<f32>,
    /// The network's value per config, per traverser, before the opponent's
    /// reach mass scales it — laid out like `vals`, one arena a seat. This is
    /// the only part of a leaf value that costs a network query, so it is the
    /// part `Cfg::refresh` keeps between iterations.
    pub vcache: [Vec<f32>; 2],
}

#[derive(Default)]
pub struct ReferenceState {
    pub cfr: HostCfr,
    /// Per traverser, leaf rows already held in `cfr.vcache`.
    pub(super) cached: [usize; 2],
    pub trace: Trace,
    /// Cached config readouts and pooling vectors.
    pub cf: Vec<f32>,
    pub cg: Vec<f32>,
    pub cp: Vec<f32>,
    /// Cached board vectors and their join projection.
    pub pb: Vec<f32>,
    pub jp: Vec<f32>,
    /// Belief pooling and join scratch.
    pub xb: Vec<f32>,
    pub h: Vec<f32>,
    pub(super) wbuf: Vec<f32>,
}

impl Solver {
    pub fn oracle(&self) -> &ReferenceState {
        &self.oracle
    }

    /// Activate the reference lazily for an integration test. Device solves
    /// keep this state empty, so the runtime path does not mirror its arenas.
    fn init_reference(&mut self) {
        if self.reference {
            return;
        }
        self.reference = true;
        let cfr = &mut self.oracle.cfr;
        cfr.reach.resize(self.nreach, 0.0);
        cfr.vals.resize(self.nvals, 0.0);
        cfr.vcache[0].resize(self.nvals, 0.0);
        cfr.vcache[1].resize(self.nvals, 0.0);
        cfr.regret.resize(self.ncells, 0.0);
        cfr.prior.resize(self.ncells, 0.0);
        cfr.prior.copy_from_slice(&self.cur);
        cfr.visits.resize(self.ncells, 0.0);
        cfr.qval.resize(self.ncells, 0.0);
        cfr.sum_strat = self
            .nodes
            .iter()
            .map(|n| vec![0.0; n.legal_action.len()])
            .collect();
        self.precompute_reaches();
    }

    /// The exact host arenas used by the parity reference.
    ///
    /// The production device path never reads them; this accessor exists for
    /// the oracle and its tests.
    pub fn cfr(&self) -> &HostCfr {
        &self.oracle.cfr
    }

    /// The same arenas, to write. Only the oracles want this: they run a
    /// contract's arithmetic beside the solver's and put the result back.
    #[doc(hidden)]
    pub fn cfr_mut(&mut self) -> &mut HostCfr {
        &mut self.oracle.cfr
    }

    /// Run `f` with the expansion's own stream, which is the stream the card
    /// runs when the CFR loop is there. Both backends draw a trajectory from
    /// the same state of the same generator, so both take the same turns.
    fn with_expand_rng<T>(&mut self, f: impl FnOnce(&mut Self, &mut Rng) -> T) -> T {
        let mut rng = Rng(self.seed);
        let out = f(self, &mut rng);
        self.seed = rng.0;
        out
    }

    /// Materialise the reference strategy: the normalised CFR average, laid
    /// out exactly like `cur`.
    ///
    /// It is built once, when the tree has stopped growing and the iterations
    /// are done, because that is the only moment at which one flat array can
    /// describe the whole tree. Everything that acts, filters a belief or
    /// values a node reads it afterwards.
    pub fn finish(&mut self) {
        // `cur` still holds the literal initial policy for a player that has
        // not traversed yet, so start there and overwrite every player whose
        // running sum has moved. Their historical average is then byte-exact
        // rather than a multiply and divide that need not round back.
        self.avg.clear();
        self.avg.extend_from_slice(&self.cur);
        let sum_strat = &self.oracle.cfr.sum_strat;
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance || !self.avg_touched[n.player as usize] {
                continue;
            }
            let so = self.soff[i] as usize;
            let nc = n.nc(n.player as usize);
            for c in 0..nc {
                let row = n.legal_row(c);
                let sum: f32 = sum_strat[i][row.clone()].iter().sum();
                let k = row.len().max(1) as f32;
                // A tiny positive mass cannot be inverted without making zero cells NaN.
                for cell in row {
                    self.avg[so + cell] = if sum > SMOOTH {
                        sum_strat[i][cell] / sum
                    } else {
                        1.0 / k
                    };
                }
            }
        }
    }

    /// Push reach probabilities down the tree under the current strategies.
    ///
    /// Children are always built after their parent, so `child > parent` and
    /// the parent's row can be borrowed alongside the child's through one
    /// `split_at_mut` — no copy of the parent's reach, which used to be two
    /// heap allocations per node per pass.
    pub fn precompute_reaches(&mut self) {
        let cur = std::mem::take(&mut self.cur);
        self.propagate(&cur);
        self.cur = cur;
    }

    /// Push reach probabilities down the tree under `strat`, from the root
    /// beliefs.
    fn propagate(&mut self, strat: &[f32]) {
        let _t = timed!(REACH);
        let reach = &mut self.oracle.cfr.reach;
        reach.fill(0.0);
        for p in 0..2 {
            let at = self.roff[0] as usize + if p == 1 { self.nc[0][0] as usize } else { 0 };
            let n = self.nc[0][p] as usize;
            reach[at..at + n].copy_from_slice(&self.root_belief[p].p);
        }
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf {
                continue;
            }
            let me = n.player as usize;
            let op = 1 - me;
            // Offsets of each player's block inside a node's reach region.
            let blk = |cnt: [u32; 2], p: usize| -> (usize, usize) {
                (if p == 0 { 0 } else { cnt[0] as usize }, cnt[p] as usize)
            };
            let (pme, nme) = blk(self.nc[i], me);
            let (pop, nop) = blk(self.nc[i], op);
            let base = self.roff[i] as usize;
            if n.chance {
                // Draw: one public child. The idle player's reach passes
                // through unchanged; the drawing player's configs transition
                // through the chance matrix, so the chance factor lives in
                // the drawing player's reach and is discarded with it when
                // the leaf values take the counterfactual convention.
                let c = n.child[0];
                debug_assert!(c > i);
                let cbase = self.roff[c] as usize;
                let (cme, _) = blk(self.nc[c], me);
                let (cop, _) = blk(self.nc[c], op);
                let (lo, hi) = reach.split_at_mut(cbase);
                let (src, dst) = (&lo[base..], &mut hi[..]);
                dst[cop..cop + nop].copy_from_slice(&src[pop..pop + nop]);
                for ci in 0..nme {
                    let w = src[pme + ci];
                    if w == 0.0 {
                        continue;
                    }
                    let (to, pr) = n.draw.row(ci);
                    for k in 0..to.len() {
                        dst[cme + to[k] as usize] += w * pr[k];
                    }
                }
                continue;
            }
            let cur = &strat[self.soff[i] as usize..];
            for ch in 0..n.child.len() {
                let c = n.child[ch];
                debug_assert!(c > i);
                let cbase = self.roff[c] as usize;
                let (cme, _) = blk(self.nc[c], me);
                let (cop, _) = blk(self.nc[c], op);
                let (lo, hi) = reach.split_at_mut(cbase);
                let (src, dst) = (&lo[base..], &mut hi[..]);
                // The idle player's information state is untouched, and the
                // child's support for them is the same list.
                dst[cop..cop + nop].copy_from_slice(&src[pop..pop + nop]);
                let (s0, s1) = (n.obs_start[ch] as usize, n.obs_start[ch + 1] as usize);
                for &au in &n.obs_act[s0..s1] {
                    let a = au as usize;
                    for &cell_u in
                        &n.action_cell[n.action_off[a] as usize..n.action_off[a + 1] as usize]
                    {
                        let cell = cell_u as usize;
                        debug_assert_eq!(n.legal_child[cell] as usize, c);
                        let t = n.legal_trans[cell];
                        if t == NO_TRANS {
                            continue;
                        }
                        let ci = n.cell_row[cell] as usize;
                        dst[cme + t as usize] += src[pme + ci] * cur[cell];
                    }
                }
            }
        }
    }

    /// Node `i`'s reach vector for player `p`.
    #[inline]
    fn reach_of(&self, i: usize, p: usize) -> &[f32] {
        let at = self.roff[i] as usize + if p == 1 { self.nc[i][0] as usize } else { 0 };
        &self.cfr().reach[at..at + self.nc[i][p] as usize]
    }

    /// Drive this solve to its end with the reference evaluator.
    ///
    /// The farm gathers calls across solves and answers them as one batch. A
    /// single test or tool answers them here instead.
    pub fn run_alone(&mut self) -> Option<Solved> {
        let mut replies: Vec<Reply> = Vec::new();
        loop {
            match self.advance_on_host(&replies) {
                Step::Calls(calls) => {
                    replies = calls.iter().map(|c| c.run(&self.net)).collect();
                }
                Step::Done(solved) => return solved,
            }
        }
    }

    /// Run the calls the last growth raised on this solve's own CPU network.
    ///
    /// The farm gathers these across every solve in flight and answers them in
    /// one batch. This is the same work for a solve driven on its own, which
    /// is what the single-position tools and the tests want.
    pub fn catch_up(&mut self) {
        let calls = self.growth_calls();
        let replies: Vec<Reply> = calls.iter().map(|c| c.run(&self.net)).collect();
        self.absorb(&replies);
    }

    /// Rewrite the pooled belief block the join reads, per row per player.
    ///
    /// Both players, every time. This used to refresh only the player whose
    /// strategy had just moved, which was sound while CFR alternated
    /// traversers. Student of Games updates both players against one reach
    /// profile, so both blocks go stale together and the shortcut silently
    /// pooled one of them under last iteration's belief.
    ///
    /// The belief the network reads is the normalised reach, as in the
    /// reference, pooled over the same `g(c)` the readout's `f(c)` comes from,
    /// so a config is described to the network exactly one way. `g` has a
    /// linear card-weighted half, which is what makes this pooled vector carry
    /// the belief's exact expected holding of each card rather than an average
    /// of nonlinearities.
    ///
    /// Rows below `from` are ones whose join output is being reused, so their
    /// block is not read and is not written.
    fn belief_blocks(&mut self, from: usize) {
        let _t = timed!(BELFEAT);
        // Sized where it is written. Growth used to do it, which fitted a
        // megabyte of pooled belief per solve on the device path -- where the
        // card pools its own and nothing here ever reads a row of it.
        crate::net::fit(
            &mut self.oracle.xb,
            2 * self.leaf_rows.len() * crate::net::POOL,
        );
        let (reach, roff, nc, coff, cidx, cg, wbuf, xb) = (
            &self.oracle.cfr.reach,
            &self.roff,
            &self.nc,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.oracle.cg,
            &mut self.oracle.wbuf,
            &mut self.oracle.xb,
        );
        let pool = crate::net::POOL;
        for (r, &i) in self.leaf_rows.iter().enumerate().skip(from) {
            for p in 0..2 {
                let n = nc[i][p] as usize;
                let ra = roff[i] as usize + if p == 1 { nc[i][0] as usize } else { 0 };
                if wbuf.len() < n {
                    wbuf.resize(n, 0.0);
                }
                normalize_weights(&reach[ra..ra + n], &mut wbuf[..n]);
                let q = 2 * r + p;
                let cs = coff[q] as usize;
                crate::net::accumulate(
                    cg,
                    &cidx[cs..cs + n],
                    &wbuf[..n],
                    pool,
                    &mut xb[q * pool..(q + 1) * pool],
                );
            }
        }
    }

    /// The two belief rows at one expanded node, at the instant its prior is
    /// formed. This is the same normalised reach pooling as `belief_blocks`,
    /// restricted to the one row the action encoder needs.
    fn belief_pair(&mut self, node: usize, row: usize, out: &mut [f32]) {
        let pool = crate::net::POOL;
        debug_assert_eq!(out.len(), 2 * pool);
        for p in 0..2 {
            let n = self.nc[node][p] as usize;
            let ra = self.roff[node] as usize + if p == 1 { self.nc[node][0] as usize } else { 0 };
            if self.oracle.wbuf.len() < n {
                self.oracle.wbuf.resize(n, 0.0);
            }
            normalize_weights(
                &self.oracle.cfr.reach[ra..ra + n],
                &mut self.oracle.wbuf[..n],
            );
            let cs = self.leaf_coff[2 * row + p] as usize;
            crate::net::accumulate(
                &self.oracle.cg,
                &self.leaf_cidx[cs..cs + n],
                &self.oracle.wbuf[..n],
                pool,
                &mut out[p * pool..(p + 1) * pool],
            );
        }
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values,
    /// querying the network at every row.
    pub fn leaf_values(&mut self, traverser: usize) {
        self.leaf_values_from(traverser, 0);
    }

    /// The same, querying the network only from row `from` on and re-scaling
    /// every earlier row's cached `v(c)` by the opponent's current reach mass.
    ///
    /// A leaf's counterfactual value is `v(c)` times that mass. The mass moves
    /// every iteration and costs a sum over a support; `v(c)` is the network,
    /// and is the whole of what the join and the readout are for. So the split
    /// here is exactly the split `Cfg::refresh` trades in.
    fn leaf_values_from(&mut self, traverser: usize, from: usize) {
        if !self.net.is_empty() {
            self.belief_blocks(from);
        }
        self.pbs_head(traverser, from);
        self.readout_from(traverser, from);
        self.oracle.cached[traverser] = self.leaf_rows.len();
    }

    /// The one path CFR pays for on every iteration.
    ///
    /// Writes `h` for rows `from ..` at its front, so row `r` of the tree is
    /// row `r - from` of `h`. The join takes a contiguous batch and this one is
    /// a suffix of the leaves, because growth only ever appends.
    fn pbs_head(&mut self, traverser: usize, from: usize) {
        let net = &self.net;
        if net.is_empty() {
            return;
        }
        let _t = timed!(NET);
        let rows = self.leaf_rows.len();
        let n = rows - from;
        if n == 0 {
            return;
        }
        crate::prof::work(0, 0, n, 0);
        self.oracle.trace.join_rows += n as u64;
        let pool = crate::net::POOL;
        // `xb` is grown by `fit` and never shrinks, so a subgame smaller than
        // an earlier one would otherwise hand the batch a trailing tail.
        self.net.join(
            &self.oracle.pb[..self.nboards * crate::net::D],
            &self.oracle.jp[..self.nboards * crate::net::JW],
            &self.board_of[from..rows],
            &self.oracle.xb[2 * from * pool..2 * rows * pool],
            n,
            traverser,
            &mut self.oracle.h,
        );
    }

    /// Per-config leaf values for player `p` — counterfactual: the network's
    /// value for that exact config times the opponent's unnormalised reach
    /// into the leaf. Runs off the `h` left by the last `pbs_head` query, and
    /// is one dot product per config.
    ///
    /// Rows below `from` take their `v(c)` from `vcache` instead of the
    /// network; every row is scaled by the reach mass it has now.
    fn readout_from(&mut self, p: usize, from: usize) {
        let _t = timed!(LEAFPOST);
        let empty = self.net.is_empty();
        let queried: usize = self.leaf_rows[from..]
            .iter()
            .map(|&i| self.nc[i][p] as usize)
            .sum();
        crate::prof::work(0, 0, 0, queried);
        self.oracle.trace.readout_cfgs += queried as u64;
        let opp = 1 - p;
        for k in 0..self.term_leaves.len() {
            let i = self.term_leaves[k];
            let opp_reach: f32 = self.reach_of(i, opp).iter().sum();
            // Zero-sum by construction (`state::horizon_tests`), so one
            // stored value serves both seats.
            let u = if p == self.nodes[i].player as usize {
                self.nodes[i].util
            } else {
                -self.nodes[i].util
            };
            let n = self.nc[i][p] as usize;
            let vo = self.voff[i] as usize;
            // A terminal leaf's value is the game's, not the network's, but
            // it travels the same arithmetic afterwards.
            self.oracle.cfr.vals[vo..vo + n].fill(u * opp_reach);
        }
        let d = crate::net::D;
        let cfr = &mut self.oracle.cfr;
        let (reach, vals, vcache) = (&cfr.reach, &mut cfr.vals, &mut cfr.vcache[p]);
        let (roff, ncs, voff, coff, cidx, cf) = (
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.oracle.cf,
        );
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            let n = ncs[i][p] as usize;
            let vo = voff[i] as usize;
            if empty {
                vals[vo..vo + n].fill(0.0);
                continue;
            }
            // `pbs_head` wrote the queried rows at the front of `h`, so the
            // tree's row `r` is `h`'s row `r - from`.
            if r >= from {
                let cs = coff[2 * r + p] as usize;
                self.net.values(
                    &self.oracle.h[(r - from) * d..(r - from + 1) * d],
                    cf,
                    &cidx[cs..cs + n],
                    &mut vcache[vo..vo + n],
                );
            }
            let ra = roff[i] as usize + if opp == 1 { ncs[i][0] as usize } else { 0 };
            let opp_reach: f32 = reach[ra..ra + ncs[i][opp] as usize].iter().sum();
            for (value, &v) in vals[vo..vo + n].iter_mut().zip(&vcache[vo..vo + n]) {
                *value = v * opp_reach;
            }
        }
    }
    fn update_regrets(&mut self, traverser: usize) {
        // Reaches are already consistent with `cur`: `new` establishes that,
        // every `step` re-establishes it after regret matching, and the
        // fixed-policy passes restore it before returning, so recomputing them
        // here would repeat the previous pass exactly.
        // `Cfg::refresh` says how often the network runs. Rows growth has added
        // since the last query have nothing to reuse and are always queried.
        let from = if self.cfg.refresh_due(self.steps[traverser]) {
            0
        } else {
            self.oracle.cached[traverser]
        };
        self.leaf_values_from(traverser, from);
        self.backprop(traverser, &[], Back::Regret);
    }

    /// One value backpropagation over the tree for `traverser`. `mode` chooses
    /// whether the traverser's decision nodes average under `strat`, average
    /// and update regret matching, or take the max. Regret mode uses
    /// `self.cur`; fixed-policy modes read `strat`.
    pub fn backprop(&mut self, traverser: usize, strat: &[f32], mode: Back) {
        // Standard positive-part regret matching keeps zero-total rows uniform.
        // The factors are constant for this whole traversal.
        let rm = if mode == Back::Regret {
            let k = self.cfg.cfr;
            let m = self.steps[traverser] as f32 + 1.0;
            Some((
                k,
                Cfr::factor(m, k.alpha),
                Cfr::factor(m, k.beta),
                (m / (m + 1.0)).powf(k.gamma),
            ))
        } else {
            None
        };
        let _t = timed!(BACK);
        let cfr = &mut self.oracle.cfr;
        for i in (0..self.nodes.len()).rev() {
            if self.nodes[i].leaf {
                continue;
            }
            let (na, me) = (self.nodes[i].na(), self.nodes[i].player as usize);
            let nc = self.nodes[i].nc(traverser);
            if self.nodes[i].chance {
                // Draw pass-through: no regrets, no strategy. If the traverser
                // is the one drawing, their per-config values are pushed
                // through the chance matrix (the probability is a real factor
                // of the value, unlike the traverser's own strategy, which the
                // counterfactual convention discards). If the idle player
                // draws, the traverser's configs are untouched and the
                // opponent's chance factor is already in their reach, which
                // the leaf values carry.
                let ch = self.nodes[i].child[0];
                debug_assert!(ch > i);
                let (vi, vc) = (self.voff[i] as usize, self.voff[ch] as usize);
                let (lo, hi) = cfr.vals.split_at_mut(vc);
                let (dst, src) = (&mut lo[vi..], &hi[..]);
                if me == traverser {
                    let n = &self.nodes[i];
                    for c in 0..nc {
                        let (to, pr) = n.draw.row(c);
                        let mut v = 0.0;
                        for k in 0..to.len() {
                            v += pr[k] * src[to[k] as usize];
                        }
                        dst[c] = v;
                    }
                } else {
                    dst[..nc].copy_from_slice(&src[..nc]);
                }
                continue;
            }
            let vbase = self.voff[i] as usize;
            // A best response takes a max at the traverser's own nodes, so
            // those start below every candidate; a config with no legal action
            // there is put back to zero below. Every other node accumulates.
            let br = mode == Back::BestResponse && me == traverser;
            cfr.vals[vbase..vbase + nc].fill(if br { f32::NEG_INFINITY } else { 0.0 });
            if mode == Back::Regret && me == traverser {
                // A cell whose action has no successor information state is
                // never visited by the pass that fills these, and must read
                // zero when the regret pass gets to it.
                let so = self.soff[i] as usize;
                let cells = self.nodes[i].legal_action.len();
                cfr.qval[so..so + cells].fill(0.0);
            }
            if me == traverser {
                let n = &self.nodes[i];
                let so = self.soff[i] as usize;
                // Children are built after their parent, so the parent's value
                // row and every child's are disjoint slices of one arena.
                let (lo, hi) = cfr.vals.split_at_mut(self.voff[i + 1] as usize);
                let vi = &mut lo[vbase..];
                for a in 0..na {
                    let ch = n.child[n.obs_child[a]];
                    let cv = &hi[self.voff[ch] as usize - self.voff[i + 1] as usize..];
                    for &cell_u in
                        &n.action_cell[n.action_off[a] as usize..n.action_off[a + 1] as usize]
                    {
                        let cell = cell_u as usize;
                        debug_assert_eq!(n.legal_child[cell] as usize, ch);
                        let t = n.legal_trans[cell];
                        if t == NO_TRANS {
                            continue;
                        }
                        let c = n.cell_row[cell] as usize;
                        let av = cv[t as usize];
                        match mode {
                            Back::Regret => {
                                // Kept, not re-gathered. The regret pass below
                                // needs this same number, and finding it again
                                // means another random hop into a child's value
                                // row -- the cache-hostile part of the sweep,
                                // paid twice per cell for nothing. The
                                // expansion phase reads it as PUCT's Q.
                                cfr.qval[so + cell] = av;
                                vi[c] += av * self.cur[so + cell];
                            }
                            Back::Value => vi[c] += av * strat[so + cell],
                            Back::BestResponse => vi[c] = vi[c].max(av),
                        }
                    }
                }
                match mode {
                    Back::Regret => {
                        let (k, da, db, dg) = rm.expect("regret factors");
                        for c in 0..nc {
                            let base = vi[c];
                            let row = n.legal_row(c);
                            let (row_start, row_end) = (row.start, row.end);
                            let mut sum = 0.0;
                            for cell in row.clone() {
                                // Re-form this row-local action value rather
                                // than retain an arena of them between phases.
                                // Starting at +0 and adding preserves the old
                                // `inst[cell] += av` FP32 operation exactly,
                                // including an explicit no-successor cell.
                                // The action value the pass above kept. A cell
                                // with no successor was skipped there and
                                // still reads the zero this node's cells were
                                // cleared to, which is what re-forming it from
                                // +0 used to produce.
                                let delta = cfr.qval[so + cell] - base;
                                let at = so + cell;
                                let old = cfr.regret[at];
                                let r = old * if old > 0.0 { da } else { db } + delta;
                                cfr.regret[at] = r;
                                let v = (r + k.predict * delta).max(0.0);
                                self.cur[at] = v;
                                sum += v;
                            }
                            // A tiny positive mass cannot be inverted without making zero cells NaN.
                            normalize_strategy(&mut self.cur[so + row_start..so + row_end], sum);
                        }
                        for x in cfr.sum_strat[i].iter_mut() {
                            *x *= dg;
                        }
                    }
                    Back::BestResponse => {
                        for c in 0..nc {
                            if vi[c] == f32::NEG_INFINITY {
                                vi[c] = 0.0;
                            }
                        }
                    }
                    Back::Value => {}
                }
            } else {
                // The traverser's information state is unchanged across an
                // opponent decision, and the opponent's strategy is already
                // baked into the reach probabilities at the children.
                for ch in 0..self.nodes[i].child.len() {
                    let c_id = self.nodes[i].child[ch];
                    let cv = self.voff[c_id] as usize;
                    for c in 0..nc {
                        cfr.vals[vbase + c] += cfr.vals[cv + c];
                    }
                }
            }
        }
    }

    /// One iteration of the regret update phase: **simultaneous updates**, as
    /// Student of Games specifies.
    ///
    /// Both players are traversed against the same reach profile, so each of
    /// them best-responds to the strategy the other held at the start of the
    /// iteration rather than to a strategy the same iteration already moved.
    /// The two traversals do not interfere: values are per traverser, and a
    /// player's regret matching writes only its own decision nodes.
    ///
    /// This is twice the work of an alternating half-iteration and twice the
    /// updates, so a solve of `iters` iterations now gives each player `iters`
    /// updates rather than `iters / 2`.
    pub fn step(&mut self) {
        self.oracle.trace.iters += 1;
        self.oracle.trace.row_iters += self.leaf_rows.len() as u64;
        self.oracle.trace.cidx_iters += self.leaf_cidx.len() as u64;
        self.oracle.trace.cell_iters += self.ncells as u64;
        let query_from = |p| {
            if self.cfg.refresh_due(self.steps[p]) {
                0
            } else {
                self.oracle.cached[p]
            }
        };
        let from = query_from(0).min(query_from(1));
        self.record_host_queries(from);
        self.update_regrets(0);
        self.update_regrets(1);
        self.precompute_reaches();
        self.avg_block();
        self.steps[0] += 1;
        self.steps[1] += 1;
    }

    /// Add the fresh reach-weighted iterate to the running strategy sum.
    /// Normalisation is deferred to `finish`.
    ///
    /// Both players in one walk. A decision node belongs to exactly one of
    /// them, so a per-player call skipped half the nodes and paid the whole
    /// traversal twice -- which was right while CFR alternated traversers and
    /// only one player's sum moved per iteration, and is not now that `step`
    /// updates both.
    pub fn avg_block(&mut self) {
        let _t = timed!(AVG);
        self.avg_touched = [true; 2];
        let cfr = &mut self.oracle.cfr;
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance {
                continue;
            }
            let me = n.player as usize;
            let nc = n.nc(me);
            let so = self.soff[i] as usize;
            let ra = self.roff[i] as usize + if me == 1 { self.nc[i][0] as usize } else { 0 };
            for c in 0..nc {
                let r = cfr.reach[ra + c];
                for cell in n.legal_row(c) {
                    cfr.sum_strat[i][cell] += r * self.cur[so + cell];
                }
            }
        }
    }
    pub fn multistep(&mut self, iters: usize) {
        for _ in 0..iters {
            self.step();
        }
    }

    /// Student of Games' GT-CFR, with the CFR loop on this host.
    ///
    /// `SoG(s, c)`: `s` expansions in total, `c` of them after each regret
    /// update, so the solve runs `ceil(s / c)` updates. Growing and
    /// solving interleave rather than staging, which is the point: the strategy
    /// decides where the tree goes, and the tree decides what the strategy is
    /// worth.
    ///
    /// The only calls this path raises are the trunk over fresh leaves and the
    /// encoder over fresh configs. Both are properties of the subgame rather
    /// than of an iteration, so they are asked for once per growth. The join
    /// that every iteration pays for, and the policy head the expansion phase
    /// reads, run inline on the core the solve is already on.
    pub fn advance_on_host(&mut self, replies: &[Reply]) -> Step {
        self.init_reference();
        if self.phase == Phase::Iterating {
            self.absorb(replies);
        }
        let iters = self.cfg.iters();
        loop {
            // Whatever the last growth added, before the iteration reads it.
            let calls = self.growth_calls();
            if !calls.is_empty() {
                self.phase = Phase::Iterating;
                return Step::Calls(calls);
            }
            if self.at == iters {
                self.finish();
                self.phase = Phase::Done;
                return Step::Done(self.collect.map(|q| self.harvest(q)));
            }
            // The same round the device runs, on this core: `done` regret
            // updates against a frozen tree, each sampling `want` trajectories,
            // and one growth at the end from all of them.
            let (done, want) = self.round_shape();
            // The expansion phase reads the prior at every node it walks
            // through, and growth has just run the batch that the nodes it
            // added were waiting for. Once a round, which is where the card's
            // policy-head stage sits.
            self.refresh_priors();
            // Every phase of a round runs before any of its leaves is grown,
            // so the round's leaves are collected in one place and each phase
            // draws until it has `want` the round has not taken yet.
            let mut taken = Vec::new();
            for _ in 0..done {
                self.at += 1;
                self.step();
                self.expansion_phase(want, &mut taken);
            }
            let grew = !taken.is_empty();
            for leaf in taken {
                if self.budget_hit() {
                    break;
                }
                self.expand(leaf);
            }
            if grew {
                // Growth appended reach rows after `step` propagated the old
                // tree. The next regret update must see the new leaves.
                self.precompute_reaches();
            }
        }
    }

    /// The root's per-config values under the reference strategy — the target
    /// a solve at this position produces for itself.
    pub fn root_values(&mut self) -> [Vec<f32>; 2] {
        let out = self.value_pass();
        self.restore();
        out
    }

    /// Value every node under the reference strategy and return the root's
    /// slice. Leaves the reference reaches in place so a caller can read
    /// beliefs off the tree; it must `restore` when it is done.
    fn value_pass(&mut self) -> [Vec<f32>; 2] {
        // A root that stayed a leaf has no average strategy. The target is the
        // network's own answer at this position, which is what the leaf pass
        // already is.
        if self.nodes[0].leaf {
            let mut out = [Vec::new(), Vec::new()];
            for p in 0..2usize {
                self.leaf_values(p);
                let n = self.nc[0][p] as usize;
                let vo = self.voff[0] as usize;
                out[p] = self.cfr().vals[vo..vo + n].to_vec();
            }
            return out;
        }
        let reference = self.reference();
        self.propagate(&reference);
        let mut out = [Vec::new(), Vec::new()];
        for p in 0..2usize {
            // One entry point for a leaf query, so a batched backend -- which
            // holds this solve's board and config vectors and therefore leaves
            // the host's copies empty -- is not bypassed here.
            self.leaf_values(p);
            // `backprop` for the second player overwrites the first's values,
            // so the root slice is taken before the next pass runs.
            self.backprop(p, &reference, Back::Value);
            let n = self.nc[0][p] as usize;
            let vo = self.voff[0] as usize;
            out[p] = self.cfr().vals[vo..vo + n].to_vec();
        }
        out
    }

    /// The interior search queries this solve produced.
    ///
    /// Student of Games trains on the public belief states whose trees supplied
    /// a better value than the network leaf they replaced.
    ///
    /// Only interior coin plays are taken. A leaf's value would be the
    /// network's own answer, so training on it would teach the network what it
    /// already said; an interior node's value comes from the subtree beneath
    /// it, which is the bootstrap the whole method rests on.
    fn harvest(&mut self, queries: usize) -> Solved {
        debug_assert_eq!(self.collect, Some(queries));
        let value = self.value_pass();
        let queries = std::mem::take(&mut self.queries);
        let policy = self.root_policy();
        self.restore();
        Solved {
            value,
            queries,
            policy,
        }
    }

    /// Record selected host-network calls with the beliefs used by that call.
    fn record_host_queries(&mut self, from: usize) {
        if self.net.is_empty() {
            return;
        }
        let rows = self.leaf_query_rows(from);
        let selected = self.plan_query_events(rows.len());
        for event in selected {
            let node = rows[event];
            self.queries
                .push((self.states[node].clone(), self.belief_at(node)));
        }
    }

    /// Node `i`'s belief for each player, under whichever reaches are
    /// currently propagated.
    fn belief_at(&self, i: usize) -> [Belief; 2] {
        std::array::from_fn(|p| {
            let mut w = vec![0.0; self.nc[i][p] as usize];
            normalize_weights(self.reach_of(i, p), &mut w);
            Belief {
                cfg: self.nodes[i].cfgs[p].to_vec(),
                p: w,
            }
        })
    }

    /// How well the solve came out, for the reference strategy — the CFR
    /// average at
    /// the end of the solve.
    ///
    /// **The leaf values are frozen** at the ones the reference strategy
    /// induces. They are a function of the beliefs at the leaf, so a real
    /// deviation would move them, so this measures exploitability of the
    /// finite search game, not of the true War Chest continuation.
    pub fn nash_conv(&mut self) -> Conv {
        let reference = self.reference();
        let root = [self.root_belief[0].p.clone(), self.root_belief[1].p.clone()];
        self.propagate(&reference);
        let (mut nash, mut zero_sum) = (0.0, 0.0);
        for p in 0..2usize {
            // One query serves both walks below: `backprop` skips leaves, so
            // the leaf values it left are still there for the second.
            self.leaf_values(p);
            let vo = self.voff[0] as usize;
            let nc = self.nc[0][p] as usize;
            let expect = |v: &[f32]| -> f32 { (0..nc).map(|c| root[p][c] * v[vo + c]).sum() };
            self.backprop(p, &reference, Back::Value);
            let v = expect(&self.cfr().vals);
            self.backprop(p, &reference, Back::BestResponse);
            nash += expect(&self.cfr().vals) - v;
            zero_sum += v;
        }
        self.restore();
        Conv { nash, zero_sum }
    }

    /// The strategy the fixed-policy passes run under.
    fn reference(&self) -> Vec<f32> {
        assert!(
            !self.avg.is_empty(),
            "a fixed-policy pass needs `finish` to have materialised the average"
        );
        self.avg.clone()
    }

    /// Put the reaches back under `cur` after a fixed-policy pass has
    /// propagated something else through them. `update_regrets` assumes they
    /// are consistent with `cur` and does not recompute them, so without this a
    /// solve that is read mid-flight — which is exactly what the solver-error
    /// harness does — would resume from another strategy's reaches.
    fn restore(&mut self) {
        self.precompute_reaches();
    }
}
