use super::*;

#[cfg(test)]
const HOST_PATH: &str = "the CFR arenas belong to the reference solver";

#[cfg(test)]
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

impl Solver {
    /// The CFR arenas this solve works in.
    ///
    /// A solve on the device path has none. Nothing there reads one — the card
    /// runs the loop — so reaching for them is a mistake about which backend is
    /// driving, and it says so here rather than returning the zeroes an
    /// unallocated arena would.
    #[cfg(test)]
    pub fn cfr(&self) -> &HostCfr {
        self.host.as_ref().expect(HOST_PATH)
    }

    /// The same arenas, to write. Only the oracles want this: they run a
    /// contract's arithmetic beside the solver's and put the result back.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn cfr_mut(&mut self) -> &mut HostCfr {
        self.host.as_mut().expect(HOST_PATH)
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
    #[cfg(test)]
    pub fn finish(&mut self) {
        // `cur` still holds the literal initial policy for a player that has
        // not traversed yet, so start there and overwrite every player whose
        // running sum has moved. Their historical average is then byte-exact
        // rather than a multiply and divide that need not round back.
        self.avg.clear();
        self.avg.extend_from_slice(&self.cur);
        let sum_strat = &self.host.as_ref().expect(HOST_PATH).sum_strat;
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
                for cell in row {
                    self.avg[so + cell] = if sum > 0.0 {
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
    #[cfg(test)]
    pub fn precompute_reaches(&mut self) {
        let cur = std::mem::take(&mut self.cur);
        self.propagate(&cur);
        self.cur = cur;
    }

    /// Push reach probabilities down the tree under `strat`, from the root
    /// beliefs.
    #[cfg(test)]
    fn propagate(&mut self, strat: &[f32]) {
        let _t = timed!(REACH);
        let reach = &mut self.host.as_mut().expect(HOST_PATH).reach;
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
    #[cfg(test)]
    fn reach_of(&self, i: usize, p: usize) -> &[f32] {
        let at = self.roff[i] as usize + if p == 1 { self.nc[i][0] as usize } else { 0 };
        &self.cfr().reach[at..at + self.nc[i][p] as usize]
    }

    /// Drive this solve to its end on this host, answering its own calls.
    ///
    /// The farm gathers those calls across every solve in flight and answers
    /// them as one batch; a single game, a tool or a test wants exactly one
    /// solve, so it answers them where they are raised. Only the host path can
    /// do this: a device keeps the solve, and there is no device here.
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
    fn belief_blocks(&mut self, from: usize) {
        let _t = timed!(BELFEAT);
        // Sized where it is written. Growth used to do it, which fitted a
        // megabyte of pooled belief per solve on the device path -- where the
        // card pools its own and nothing here ever reads a row of it.
        crate::net::fit(&mut self.xb, 2 * self.leaf_rows.len() * crate::net::POOL);
        let (reach, roff, nc, coff, cidx, cg, wbuf, xb) = (
            &self.host.as_ref().expect(HOST_PATH).reach,
            &self.roff,
            &self.nc,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cg,
            &mut self.wbuf,
            &mut self.xb,
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
    #[cfg(test)]
    fn belief_pair(&mut self, node: usize, row: usize, out: &mut [f32]) {
        let pool = crate::net::POOL;
        debug_assert_eq!(out.len(), 2 * pool);
        for p in 0..2 {
            let n = self.nc[node][p] as usize;
            let ra = self.roff[node] as usize + if p == 1 { self.nc[node][0] as usize } else { 0 };
            if self.wbuf.len() < n {
                self.wbuf.resize(n, 0.0);
            }
            normalize_weights(
                &self.host.as_ref().expect(HOST_PATH).reach[ra..ra + n],
                &mut self.wbuf[..n],
            );
            let cs = self.leaf_coff[2 * row + p] as usize;
            crate::net::accumulate(
                &self.cg,
                &self.leaf_cidx[cs..cs + n],
                &self.wbuf[..n],
                pool,
                &mut out[p * pool..(p + 1) * pool],
            );
        }
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values,
    /// querying the network at every row.
    #[cfg(test)]
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
    #[cfg(test)]
    fn leaf_values_from(&mut self, traverser: usize, from: usize) {
        if !self.net.is_empty() {
            self.belief_blocks(from);
        }
        self.pbs_head(traverser, from);
        self.readout_from(traverser, from);
        self.cached[traverser] = self.leaf_rows.len();
    }

    /// The one path CFR pays for on every iteration.
    ///
    /// Writes `h` for rows `from ..` at its front, so row `r` of the tree is
    /// row `r - from` of `h`. The join takes a contiguous batch and this one is
    /// a suffix of the leaves, because growth only ever appends.
    #[cfg(test)]
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
        self.trace.join_rows += n as u64;
        let pool = crate::net::POOL;
        // `xb` is grown by `fit` and never shrinks, so a subgame smaller than
        // an earlier one would otherwise hand the batch a trailing tail.
        self.net.join(
            &self.pb[..self.nboards * crate::net::D],
            &self.jp[..self.nboards * crate::net::JW],
            &self.board_of[from..rows],
            &self.xb[2 * from * pool..2 * rows * pool],
            n,
            traverser,
            &mut self.h,
        );
    }

    /// Per-config leaf values for player `p` — counterfactual: the network's
    /// value for that exact config times the opponent's unnormalised reach
    /// into the leaf. Runs off the `h` left by the last `pbs_head` query, and
    /// is one dot product per config.
    ///
    /// Rows below `from` take their `v(c)` from `vcache` instead of the
    /// network; every row is scaled by the reach mass it has now.
    #[cfg(test)]
    fn readout_from(&mut self, p: usize, from: usize) {
        let _t = timed!(LEAFPOST);
        let empty = self.net.is_empty();
        let queried: usize = self.leaf_rows[from..]
            .iter()
            .map(|&i| self.nc[i][p] as usize)
            .sum();
        crate::prof::work(0, 0, 0, queried);
        self.trace.readout_cfgs += queried as u64;
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
            self.host.as_mut().expect(HOST_PATH).vals[vo..vo + n].fill(u * opp_reach);
        }
        let d = crate::net::D;
        let cfr = self.host.as_mut().expect(HOST_PATH);
        let (reach, vals, vcache) = (&cfr.reach, &mut cfr.vals, &mut cfr.vcache[p]);
        let (roff, ncs, voff, coff, cidx, cf) = (
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cf,
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
                    &self.h[(r - from) * d..(r - from + 1) * d],
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

    #[cfg(test)]
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
            self.cached[traverser]
        };
        self.leaf_values_from(traverser, from);
        self.backprop(traverser, &[], Back::Regret);
    }

    /// One value backpropagation over the tree for `traverser`. `mode` chooses
    /// whether the traverser's decision nodes average under `strat`, average
    /// and update regret matching, or take the max. Regret mode uses
    /// `self.cur`; fixed-policy modes read `strat`.
    #[cfg(test)]
    pub fn backprop(&mut self, traverser: usize, strat: &[f32], mode: Back) {
        // Regret matching floors at EPS rather than at zero, so every legal
        // action keeps positive probability and carried beliefs keep their
        // full support. The factors are constant for this whole traversal.
        const EPS: f32 = 1e-6;
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
        let cfr = self.host.as_mut().expect(HOST_PATH);
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
                                let v = (r + k.predict * delta).max(EPS);
                                self.cur[at] = v;
                                sum += v;
                            }
                            if sum > 0.0 {
                                let inv = 1.0 / sum;
                                for cell in row {
                                    self.cur[so + cell] *= inv;
                                }
                            }
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
    #[cfg(test)]
    pub fn step(&mut self) {
        self.trace.iters += 1;
        self.trace.row_iters += self.leaf_rows.len() as u64;
        self.trace.cidx_iters += self.leaf_cidx.len() as u64;
        self.trace.cell_iters += self.ncells as u64;
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
    #[cfg(test)]
    pub fn avg_block(&mut self) {
        let _t = timed!(AVG);
        self.avg_touched = [true; 2];
        let cfr = self.host.as_mut().expect(HOST_PATH);
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

    #[cfg(test)]
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
    #[cfg(test)]
    pub(crate) fn advance_on_host(&mut self, replies: &[Reply]) -> Step {
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

    /// The cell PUCT would take from one config's legal row.
    ///
    /// `Q + c_puct * P * sqrt(sum N) / (1 + N)`, with `Q` the counterfactual
    /// action value divided by the opponent's reach mass at this node. That
    /// division is what Student of Games means by "normalized by the sum of
    /// the opponent's reach probability at `s_i` to resemble state-conditional
    /// action values": the raw value carries the opponent's reach as a factor,
    /// so without it a node deep behind an unlikely opponent line would look
    /// worthless next to its own siblings rather than being compared with
    /// them.
    #[cfg(test)]
    fn puct_choice(&self, node: usize, row: std::ops::Range<usize>, opp: usize) -> Option<usize> {
        let so = self.soff[node] as usize;
        let reach = self.reach_of(node, opp);
        let [mass] = warp32_sum(reach.len(), |i| [reach[i]]);
        let scale = if mass > 1e-30 { 1.0 / mass } else { 0.0 };
        let cfr = self.cfr();
        let [total] = warp32_sum(row.len(), |i| [cfr.visits[so + row.start + i]]);
        let explore = self.cfg.puct * total.max(0.0).sqrt();
        let mut best = None;
        let mut best_score = f32::NEG_INFINITY;
        for cell in row {
            if !self.live_cell(node, cell) {
                continue;
            }
            let at = so + cell;
            let score = cfr.qval[at] * scale
                + explore * cfr.prior[at] / (1.0 + cfr.visits[at]);
            if score > best_score {
                best_score = score;
                best = Some(cell);
            }
        }
        best
    }

    /// Whether the expansion phase may descend through one legal cell: the
    /// acting config has a successor there, and the subtree behind it still
    /// has somewhere to grow. A trajectory into either kind of dead end can
    /// only end on a leaf growth may not touch, and the simulation is then
    /// spent for nothing -- which is what a mature tree does to the whole of
    /// its remaining budget once its frontier stops being expandable.
    #[cfg(test)]
    fn live_cell(&self, node: usize, cell: usize) -> bool {
        let n = &self.nodes[node];
        n.legal_trans[cell] != NO_TRANS
            && !self.nodes[n.legal_child[cell] as usize].exhausted
    }

    /// Fill the policy prior of every decision node that is ready for one, on
    /// this host. The device path sends `prime` instead.
    #[cfg(test)]
    fn refresh_priors(&mut self) {
        if self.net.is_empty() {
            return;
        }
        let _t = timed!(PRIOR);
        let want = self.ready_for_prior();
        if want.is_empty() {
            return;
        }
        // One description per (node, action), and the board each is played on.
        //
        // Only the boards these nodes stand on, packed. `Net::actions`
        // projects every board row it is handed, and handing it the whole leaf
        // batch meant projecting a couple of thousand of them to reach the
        // handful just expanded -- which measured at thirty-one cpu-ms an
        // iteration per thread, more than every other host phase together.
        let d = crate::net::D;
        let mut boards = Vec::with_capacity(want.len() * d);
        let mut heads = Vec::with_capacity(want.len() * d);
        let mut feat = Vec::new();
        let mut board_of: Vec<u32> = Vec::new();
        let mut base = Vec::with_capacity(want.len());
        for &i in &want {
            base.push(board_of.len() as u32);
            let row = self.row_of[i] as usize;
            let board = self.board_of[row] as usize;
            let at = board * d;
            let mine = (boards.len() / d) as u32;
            boards.extend_from_slice(&self.pb[at..at + d]);
            let mut pooled = vec![0.0; 2 * crate::net::POOL];
            self.belief_pair(i, row, &mut pooled);
            let mut h = Vec::new();
            self.net.join(
                &self.pb[at..at + d],
                &self.jp[board * crate::net::JW..(board + 1) * crate::net::JW],
                &[0],
                &pooled,
                1,
                self.nodes[i].player as usize,
                &mut h,
            );
            heads.extend_from_slice(&h);
            let n = &self.nodes[i];
            for a in 0..n.na() {
                let at = feat.len();
                feat.resize(at + crate::net::AFEAT, 0.0);
                Net::action_feats(
                    n.acts[a].kind(),
                    n.aslot[a],
                    n.acts[a].hexes(),
                    &mut feat[at..],
                );
                board_of.push(mine);
            }
        }
        let na = board_of.len();
        let mut e = Vec::new();
        self.net.actions(&feat, &boards, &heads, &board_of, &board_of, na, &mut e);

        // `logit(c, a) = <f_p(c), e(a)>` over the node's own legal cells, then
        // a softmax across each config's row.
        let mut logit = Vec::new();
        let prior = &mut self.host.as_mut().expect(HOST_PATH).prior;
        for (k, &i) in want.iter().enumerate() {
            let me = self.nodes[i].player as usize;
            let q = 2 * self.row_of[i] as usize + me;
            let cs = self.leaf_coff[q] as usize;
            let n = &self.nodes[i];
            let cells = n.legal_action.len();
            logit.clear();
            logit.resize(cells, 0.0);
            let cfg: Vec<u32> = (0..cells)
                .map(|cell| self.leaf_cidx[cs + n.cell_row[cell] as usize])
                .collect();
            let act: Vec<u32> = (0..cells)
                .map(|cell| base[k] + n.legal_action[cell])
                .collect();
            self.net.policy(&self.cp, &e, &cfg, &act, &mut logit);
            let so = self.soff[i] as usize;
            let inv_t = 1.0 / self.cfg.prior_temp.max(1e-6);
            for c in 0..n.nc(me) {
                let row = n.legal_row(c);
                let top = row
                    .clone()
                    .map(|cell| logit[cell])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for cell in row.clone() {
                    let v = ((logit[cell] - top) * inv_t).exp();
                    prior[so + cell] = v;
                    total += v;
                }
                let scale = if total > 0.0 {
                    1.0 / total
                } else {
                    1.0 / row.len().max(1) as f32
                };
                for cell in row {
                    prior[so + cell] *= scale;
                }
            }
            self.primed[i] = true;
        }
    }

    /// One expansion simulation, and the growth it produces: sample a world
    /// from the root beliefs, walk down under the average strategy, and grow
    /// the leaf it reaches. The counterpart to `step` — a GT-CFR iteration is
    /// one `step` followed by `expand` of these.
    ///
    /// False when nothing grew, which is a spent budget, a trajectory that ran
    /// into a terminal, or a config with no legal action there.
    #[cfg(test)]
    pub fn expand_once(&mut self) -> bool {
        let Some(leaf) = self.with_expand_rng(|sv, rng| sv.sample_leaf(rng)) else {
            return false;
        };
        self.expand(leaf);
        !self.nodes[leaf].leaf
    }

    /// Grow the whole subgame, and say whether it fitted in `cap` nodes.
    ///
    /// Production never does this — the point of growing is to *not* build the
    /// whole tree. It exists for the tests and sizing tools that need the
    /// complete subgame of a small endgame. A real mid-game position is not
    /// one of those and its subgame runs to millions of nodes, so the bound is
    /// the caller's way of asking for the subgame only if it is small: a
    /// `false` means the tree it now holds is a partial one.
    #[cfg(test)]
    pub fn grow_full(&mut self, cap: usize) -> bool {
        let mut at = 0usize;
        while at < self.nodes.len() {
            if self.nodes.len() > cap {
                return false;
            }
            if self.nodes[at].leaf && self.nodes[at].expandable {
                self.expand(at);
            }
            at += 1;
        }
        true
    }

    /// The root's per-config values under the reference strategy — the target
    /// a solve at this position produces for itself.
    #[cfg(test)]
    pub fn root_values(&mut self) -> [Vec<f32>; 2] {
        let out = self.value_pass();
        self.restore();
        out
    }

    /// Value every node under the reference strategy and return the root's
    /// slice. Leaves the reference reaches in place so a caller can read
    /// beliefs off the tree; it must `restore` when it is done.
    #[cfg(test)]
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

    /// One expansion phase: draw trajectories until `want` leaves the round has
    /// not already taken have been found, and append them to `taken`.
    ///
    /// The tree is frozen for a whole round, so a trajectory that ends on a
    /// leaf an earlier trajectory of the round took would grow nothing and the
    /// phase draws again. That is what makes `s` a count of *distinct*
    /// expansions, rather than a count of trajectories that mostly land where
    /// an earlier one of the same round did.
    ///
    /// A draw that runs into a dead end costs that draw and no more. Either
    /// way the visits the trajectory left along its path stand -- that is
    /// Student of Games' virtual loss, and it is the thing that sends the next
    /// draw somewhere else.
    ///
    /// `want * TRIES` draws is the bound, and it is why the loop terminates: a
    /// tree whose every reachable leaf has already been taken would otherwise
    /// draw for ever. A phase that spends it stops short of `want`.
    #[cfg(test)]
    fn expansion_phase(&mut self, want: usize, taken: &mut Vec<usize>) {
        let (mut got, mut draws) = (0usize, 0usize);
        self.with_expand_rng(|sv, rng| {
            while got < want && draws < want * TRIES {
                draws += 1;
                if let Some(leaf) = sv.sample_leaf(rng) {
                    if !taken.contains(&leaf) {
                        taken.push(leaf);
                        got += 1;
                    }
                }
            }
        });
    }

    /// One expansion simulation: sample a world from the root beliefs, walk
    /// down under the current average strategy, and return the leaf it reaches.
    ///
    /// Sampling rather than taking the most-reached leaf is what the paper does
    /// and what its convergence result wants: an optimal policy here is often
    /// mixed, and a greedy rule can starve a line the average strategy still
    /// gives weight to. Their selection rule is half PUCT and half the CFR
    /// average; with no prior to compute PUCT from, this is the half that
    /// exists.
    #[cfg(test)]
    fn sample_leaf(&mut self, rng: &mut Rng) -> Option<usize> {
        // A tree with nothing left to grow gets no trajectory at all, not even
        // the draws one would spend.
        if self.nodes[0].exhausted {
            return None;
        }
        // One private config per player forms the sampled world.
        let mut c = [
            pick(&self.root_belief[0].p, rng),
            pick(&self.root_belief[1].p, rng),
        ];
        let mut node = 0usize;
        loop {
            if self.nodes[node].leaf {
                debug_assert!(
                    self.nodes[node].expandable,
                    "the descent skips subtrees with nothing to grow"
                );
                return Some(node);
            }
            let me = self.nodes[node].player as usize;
            if self.nodes[node].chance {
                let (idx, prob) = self.nodes[node].draw.row(c[me]);
                let k = pick(prob, rng);
                c[me] = idx[k] as usize;
                node = self.nodes[node].child[0];
                continue;
            }
            let row = self.nodes[node].legal_row(c[me]);
            // Student of Games selects by half PUCT and half the search's own
            // average: `pi_select = 1/2 pi_PUCT + 1/2 pi_CFR`. PUCT is a
            // maximisation, so its half is a point mass on the argmax, and
            // sampling the mixture is a coin flip between the two.
            //
            // Both halves are restricted to the cells this world can still
            // grow through. A config whose every legal action is a dead end
            // ends the trajectory here; that is a property of the sampled
            // world, not of the tree, so it does not seal anything.
            let so = self.soff[node] as usize;
            let live = |cell: usize| self.live_cell(node, cell);
            let cell = if rng.unit_f64() < 0.5 {
                self.puct_choice(node, row.clone(), 1 - me)
            } else if self.cfr().sum_strat[node][row.clone()].iter().any(|&x| x > 0.0) {
                pick_live(&self.cfr().sum_strat[node][row.clone()], |i| live(row.start + i), rng)
                    .map(|i| row.start + i)
            } else {
                pick_live(&self.cur[so + row.start..so + row.end], |i| live(row.start + i), rng)
                    .map(|i| row.start + i)
            };
            let cell = cell?;
            // Counted as the trajectory passes, which is also the virtual loss
            // Student of Games adds across the simulations of one iteration:
            // a later simulation of the same phase sees this one's visit.
            self.host.as_mut().expect(HOST_PATH).visits[so + cell] += 1.0;
            c[me] = self.nodes[node].legal_trans[cell] as usize;
            node = self.nodes[node].legal_child[cell] as usize;
        }
    }

    /// Run one expansion phase against arenas some other backend left behind,
    /// appending the leaves it took to `taken`.
    ///
    /// The CFR loop runs on the card in production, so on that path the host's
    /// own copies of these arenas stay at their uniform start. Given numbers
    /// of its own the host would drift a few ulps from the card's and take a
    /// different turn at the first close call, which measures the loop rather
    /// than the growth rule. Given the card's own numbers the two must agree
    /// simulation for simulation, and that is what the parity test asks.
    ///
    /// `visits` is the one arena the phase writes, so a caller comparing a
    /// phase must hand over the state as it stood before that phase ran.
    /// `taken` is the round's leaves so far, which is the other state a phase
    /// reads: the card keeps it in the round's own output buffer.
    ///
    /// Not part of the engine's interface: it gives this solve the arenas in
    /// `a` and advances `seed` by the draws the phase makes.
    #[doc(hidden)]
    #[cfg(test)]
    pub fn replay_expansion(&mut self, a: &Arenas, want: usize, taken: &mut Vec<usize>) {
        let cells = self.ncells;
        self.cur.copy_from_slice(&a.cur[..cells]);
        // A device solve has no arenas of its own, which is the whole point:
        // the rule is being run on the card's numbers, not on numbers the host
        // made. So the arenas it reads are built here, out of `a`.
        self.host = Some(HostCfr {
            regret: Vec::new(),
            prior: a.prior[..cells].to_vec(),
            visits: a.visits[..cells].to_vec(),
            qval: a.qval[..cells].to_vec(),
            sum_strat: (0..self.nodes.len())
                .map(|i| {
                    let (so, n) = (self.soff[i] as usize, self.nodes[i].legal_action.len());
                    a.sum[so..so + n].to_vec()
                })
                .collect(),
            reach: a.reach[..self.nreach].to_vec(),
            vals: Vec::new(),
            vcache: [Vec::new(), Vec::new()],
        });
        self.expansion_phase(want, taken)
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
    #[cfg(test)]
    fn harvest(&mut self, queries: usize) -> Solved {
        let value = self.value_pass();
        let queries = self.with_rng(|sv, rng| sv.sample_queries(rng, queries));
        let policy = self.root_policy();
        self.restore();
        Solved { value, queries, policy }
    }

    /// Uniform draws from the leaves this solve queried the network at.
    ///
    /// Those leaves are where the value function's error enters the solve, so
    /// they are the belief states worth solving in their own right. Every one
    /// of them is a valued decision, which is both what the network is defined
    /// on and what a training row can carry, so no filtering is needed here.
    #[cfg(test)]
    fn sample_queries(&self, rng: &mut Rng, want: usize) -> Vec<(State, [Belief; 2])> {
        if self.leaf_rows.is_empty() {
            return Vec::new();
        }
        (0..want)
            .map(|_| {
                let i = self.leaf_rows[rng.below(self.leaf_rows.len())];
                (self.states[i].clone(), self.belief_at(i))
            })
            .collect()
    }

    /// Node `i`'s belief for each player, under whichever reaches are
    /// currently propagated.
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
    fn restore(&mut self) {
        self.precompute_reaches();
    }

}
