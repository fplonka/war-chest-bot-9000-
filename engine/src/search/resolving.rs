use super::*;
use crate::resolve::{Continuation, Solved, Values};

impl Solver {
    #[allow(clippy::too_many_arguments)]
    pub fn play(
        continuation: Option<&Continuation>,
        live: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
        actual: Config,
    ) -> Result<Solver, String> {
        Self::continual(continuation, live, ctx, net, cfg, belief, rng, Finish::Play(actual))
    }

    pub fn refresh(
        continuation: Option<&Continuation>,
        live: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
    ) -> Result<Solver, String> {
        Self::continual(continuation, live, ctx, net, cfg, belief, rng, Finish::Refresh)
    }

    pub fn target(
        root: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
    ) -> Result<Solver, String> {
        if !root.is_valued() {
            return Err("a target solve requires a valued state".into());
        }
        let mut sv = Self::build(root, ctx, net, cfg, belief, rng, Finish::Target)?;
        sv.prepare_focus()?;
        Ok(sv)
    }

    #[allow(clippy::too_many_arguments)]
    fn continual(
        continuation: Option<&Continuation>,
        live: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
        finish: Finish,
    ) -> Result<Solver, String> {
        let mut sv = match continuation {
            Some(continuation) => Self::resolved(continuation, live, net, cfg, rng, finish)?,
            None => Self::build(live, ctx, net, cfg, belief, rng, finish)?,
        };
        sv.prepare_focus()?;
        Ok(sv)
    }

    fn build(
        root: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
        finish: Finish,
    ) -> Result<Solver, String> {
        let root_configs: usize = belief.iter().map(Belief::len).sum();
        if root_configs > 2 * crate::pbs::MAX_CONFIG_SUPPORT || root_configs > cfg.budget.cap(Ent::Config) {
            return Err(format!("root support of {root_configs} configurations exceeds solve capacity"));
        }
        for p in 0..2 {
            crate::resolve::validate_belief(root, &ctx, p, &belief[p])?;
        }
        let cfgs: [Arc<[Config]>; 2] = [belief[0].cfg.as_slice().into(), belief[1].cfg.as_slice().into()];
        let mut sv = Solver::default();
        sv.ctx = ctx;
        sv.net = net;
        sv.rng = rng;
        sv.cfg = cfg;
        sv.cfg.budget = cfg.budget.storage();
        sv.root_belief = belief;
        sv.horizon = root.round + cfg.rounds as u16;
        sv.finish = finish;
        sv.nodes = NODES.with(Pool::take);
        sv.cphi = CONFIGS.with(Pool::take);
        sv.cmap = std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash);
        sv.bmap = std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash);
        sv.nodes.reserve(640);
        sv.cur.reserve(640);
        sv.seed = Rng::new(sv.rng.next_u64()).0;
        let root = sv.push_node(crate::contract::NO_ROW, *root, cfgs);
        sv.expand(root);
        if let Some(error) = sv.failure.take() {
            return Err(error);
        }
        Ok(sv)
    }

    fn resolved(
        continuation: &Continuation,
        live: &State,
        net: Arc<Net>,
        cfg: Cfg,
        rng: Rng,
        finish: Finish,
    ) -> Result<Solver, String> {
        let root = continuation.state;
        let resolver = live.to_act();
        let opponent = 1 - resolver as usize;
        let mut sv = Self::build(&root, Ctx::new(&root), net, cfg, continuation.range.clone(), rng, finish)?;
        sv.gadget = Some(Gadget { resolver, terminate: continuation.cfv[opponent].clone() });
        sv.follow_draws(continuation.draws)?;
        if !PublicState::from_state(sv.nodes[sv.focus].state).same_public(live) {
            return Err("the retained boundary does not reach the live state".into());
        }
        if sv.nodes[sv.focus].player != resolver {
            return Err("the retained boundary reaches the wrong actor".into());
        }
        Ok(sv)
    }

    fn follow_draws(&mut self, draws: usize) -> Result<(), String> {
        let mut node = 0usize;
        for _ in 0..draws {
            if self.nodes[node].leaf {
                if !self.nodes[node].expandable {
                    return Err("the retained chance transition exceeds solve capacity".into());
                }
                self.expand_required(node)?;
            }
            if !self.nodes[node].chance || self.nodes[node].child.len() != 1 {
                return Err("the retained chance transition is missing".into());
            }
            node = self.nodes[node].child[0];
            self.horizon = self.nodes[node].state.round + self.cfg.rounds as u16;
            if !self.nodes[node].state.is_terminal() {
                self.nodes[node].expandable = true;
            }
        }
        self.focus = node;
        Ok(())
    }

    fn expand_required(&mut self, node: usize) -> Result<(), String> {
        self.expand(node);
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if self.nodes[node].leaf {
            return Err(format!(
                "required decision exceeds solve capacity: stop={} used={:?} pending={:?}",
                self.stop_reason(),
                Ent::ALL.map(|e| (e.name(), self.used(e), self.cfg.budget.cap(e))),
                self.nodes[node].state.pending(),
            ));
        }
        Ok(())
    }

    fn prepare_focus(&mut self) -> Result<(), String> {
        if self.nodes[self.focus].leaf && self.nodes[self.focus].expandable {
            self.expand_required(self.focus)?;
        }
        if self.nodes[self.focus].leaf || self.nodes[self.focus].chance {
            return Err("solve focus is not a decision".into());
        }
        for node in &mut self.nodes {
            node.carry = false;
        }
        self.nodes[self.focus].carry = true;
        if !matches!(self.finish, Finish::Target) {
            for child in self.nodes[self.focus].child.clone() {
                self.nodes[child].carry = true;
            }
        }
        if self.focus != 0 {
            for i in 0..self.nodes.len() {
                if self.nodes[i].leaf && !self.is_below(i, self.focus) {
                    self.nodes[i].expandable = false;
                }
            }
            for i in (0..self.nodes.len()).rev() {
                self.set_exhausted(i);
            }
        }
        Ok(())
    }

    fn is_below(&self, mut node: usize, ancestor: usize) -> bool {
        loop {
            if node == ancestor {
                return true;
            }
            let parent = self.nodes[node].parent;
            if parent == crate::contract::NO_ROW {
                return false;
            }
            node = parent as usize;
        }
    }

    fn carry_len(&self) -> usize {
        self.nodes.iter().filter(|node| node.carry)
            .map(|node| 2 * node.nc.iter().sum::<u32>() as usize).sum()
    }

    pub(super) fn read_round(&mut self) -> Vec<Call> {
        let mut calls = self.opening_calls();
        let node = &self.nodes[self.focus];
        let common = (self.slot, self.avg_touched, self.focus as u32, self.carry_len() as u32, node.legal_action.len() as u32);
        match &self.finish {
            Finish::Play(actual) => {
                let actor = node.player as usize;
                let index = node.cfgs[actor]
                    .binary_search(actual)
                    .expect("the acting configuration is in the focus support") as u32;
                calls.push(Call::ReadPlay {
                    solve: common.0,
                    touched: common.1,
                    focus: common.2,
                    carry: common.3,
                    cells: common.4,
                    actual: index,
                    explore: self.explore,
                });
            }
            Finish::Refresh => calls.push(Call::ReadRefresh {
                solve: common.0,
                touched: common.1,
                focus: common.2,
                carry: common.3,
                cells: common.4,
            }),
            Finish::Target => calls.push(Call::ReadTarget {
                solve: common.0,
                touched: common.1,
                focus: common.2,
                carry: common.3,
                cells: common.4,
            }),
        }
        calls
    }

    pub(super) fn read_back(&mut self, r: &Reply) -> Result<Solved, String> {
        let focus = self.focus;
        let node = &self.nodes[focus];
        let policy_at = 1 + self.carry_len();
        let expected = policy_at + node.legal_action.len();
        if r.a.len() != expected {
            return Err(format!("GPU read returned {} values, expected {expected}", r.a.len()));
        }
        let mut at = 1;
        let mut focus_values = None;
        let mut children = Vec::new();
        for i in (0..self.nodes.len()).filter(|&i| self.nodes[i].carry) {
            let len = 2 * self.nodes[i].nc.iter().sum::<u32>() as usize;
            let values = self.values_at(i, &r.a[at..at + len])?;
            at += len;
            if i == focus {
                focus_values = values;
                continue;
            }
            let ch = node.child.iter().position(|&c| c == i).ok_or("a carried node is not a focus child")?;
            let key = obs_key(&node.acts[node.obs_act[node.obs_start[ch] as usize] as usize]);
            children.extend(values.map(|values| (key, values)));
        }
        let action = match self.finish {
            Finish::Play(_) => {
                let cell = r.a[0].to_bits() as usize;
                if cell >= node.legal_action.len() {
                    return Err("the GPU selected a cell outside the focus row".into());
                }
                Some(node.acts[node.legal_action[cell] as usize])
            }
            _ => None,
        };
        let policy = self.policy_at(focus, &r.a[policy_at..]);
        Ok(Solved {
            action,
            policy,
            focus: focus_values.ok_or("the solve focus is unreached")?,
            children,
            queries: std::mem::take(&mut self.queries),
        })
    }

    fn values_at(&self, node: usize, carried: &[f32]) -> Result<Option<Values>, String> {
        let n = &self.nodes[node];
        if n.state.is_terminal() {
            return Ok(None);
        }
        let [n0, n1] = n.nc.map(|x| x as usize);
        let reach = [&carried[..n0], &carried[n0..n0 + n1]];
        let vals = [&carried[n0 + n1..2 * n0 + n1], &carried[2 * n0 + n1..]];
        if carried.iter().any(|x| !x.is_finite()) || reach.iter().any(|r| r.iter().any(|x| *x < 0.0)) {
            return Err(format!("carry node {node} has invalid values"));
        }
        let mass = [reach[0].iter().sum::<f32>(), reach[1].iter().sum::<f32>()];
        if mass.iter().any(|m| *m <= 0.0) {
            return Ok(None);
        }
        Ok(Some(Values {
            state: n.state,
            cfgs: [n.cfgs[0].to_vec(), n.cfgs[1].to_vec()],
            cfv: std::array::from_fn(|p| vals[p].iter().map(|x| x / mass[1 - p]).collect()),
        }))
    }

    fn policy_at(&self, node: usize, probs: &[f32]) -> Policy {
        let n = &self.nodes[node];
        let me = n.player as usize;
        let mut out = Policy {
            acts: (0..n.na()).map(|a| action_desc(&n.acts[a], n.player, &self.ctx, n.aslot[a])).collect(),
            ..Default::default()
        };
        out.off.push(0);
        for c in 0..n.nc[me] as usize {
            for cell in n.legal_row(c) {
                out.act.push(n.legal_action[cell] as u16);
                out.p.push(probs[cell]);
            }
            out.off.push(out.act.len() as u32);
        }
        out
    }
}
