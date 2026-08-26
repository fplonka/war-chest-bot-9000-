use super::*;

impl Solver {
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
            let score = cfr.qval[at] * scale + explore * cfr.prior[at] / (1.0 + cfr.visits[at]);
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
        n.legal_trans[cell] != NO_TRANS && !self.nodes[n.legal_child[cell] as usize].exhausted
    }

    /// Fill the policy prior of every decision node that is ready for one, on
    /// this host. The device path sends `prime` instead.
    #[cfg(test)]
    pub(super) fn refresh_priors(&mut self) {
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
        self.net
            .actions(&feat, &boards, &heads, &board_of, &board_of, na, &mut e);

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
    pub(super) fn expansion_phase(&mut self, want: usize, taken: &mut Vec<usize>) {
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
            } else if self.cfr().sum_strat[node][row.clone()]
                .iter()
                .any(|&x| x > 0.0)
            {
                pick_live(
                    &self.cfr().sum_strat[node][row.clone()],
                    |i| live(row.start + i),
                    rng,
                )
                .map(|i| row.start + i)
            } else {
                pick_live(
                    &self.cur[so + row.start..so + row.end],
                    |i| live(row.start + i),
                    rng,
                )
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
}
