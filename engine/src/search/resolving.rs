use super::*;
use crate::resolve::Continuation;

impl Solver {
    pub fn initial_play(
        root: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
        actual: Config,
    ) -> Result<Solver, String> {
        let mut sv = Self::build(root, ctx, net, cfg, belief, rng, Finish::Play(actual))?;
        sv.prepare_focus(true)?;
        Ok(sv)
    }

    pub fn play(
        continuation: &Continuation,
        live: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        mut rng: Rng,
        actual: Config,
    ) -> Result<Solver, String> {
        Self::continual(continuation, live, ctx, net, cfg, belief, &mut rng, Finish::Play(actual), true)
    }

    pub fn refresh(
        continuation: &Continuation,
        live: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        mut rng: Rng,
    ) -> Result<Solver, String> {
        Self::continual(continuation, live, ctx, net, cfg, belief, &mut rng, Finish::Refresh, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn continual(
        continuation: &Continuation,
        live: &State,
        ctx: Ctx,
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: &mut Rng,
        finish: Finish,
        successors: bool,
    ) -> Result<Solver, String> {
        if let Continuation::Solved { boundary, path } = continuation {
            return Self::resolved(
                boundary.as_ref().clone(), path.clone(), live,
                net, cfg, Rng::new(rng.next_u64()), finish, successors);
        }
        let mut sv = Self::build(live, ctx, net, cfg, belief, Rng::new(rng.next_u64()), finish)?;
        sv.prepare_focus(successors)?;
        Ok(sv)
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
        sv.prepare_focus(false)?;
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
        boundary: Boundary,
        path: ResolvePath,
        live: &State,
        net: Arc<Net>,
        cfg: Cfg,
        rng: Rng,
        finish: Finish,
        successors: bool,
    ) -> Result<Solver, String> {
        let ctx = Ctx::new(&boundary.public.state());
        let resolver = live.to_act();
        let opponent = 1 - resolver;
        let previous = boundary.range[opponent as usize].p.clone();
        let terminate = boundary.cfv[opponent as usize].clone();
        let root = boundary.public.state();
        let belief = boundary.range.clone();
        let mut sv = Self::build(&root, ctx, net, cfg, belief, rng, finish)?;
        sv.gadget = Some(Gadget { resolver, previous, terminate });
        sv.follow_path(&path)?;
        if !boundary.public.same_public(&sv.nodes[0].state) {
            return Err("the retained boundary does not match its canonical root".into());
        }
        if !PublicState::from_state(sv.nodes[sv.focus].state).same_public(live) {
            return Err("the forced public prefix does not reach the live state".into());
        }
        if sv.nodes[sv.focus].player != resolver {
            return Err("the forced public prefix reaches the wrong actor".into());
        }
        sv.prepare_focus(successors)?;
        Ok(sv)
    }

    fn follow_path(&mut self, path: &ResolvePath) -> Result<(), String> {
        let mut node = 0usize;
        for step in &path.steps {
            if self.nodes[node].leaf {
                if !self.nodes[node].expandable {
                    return Err("the mandatory public prefix exceeds solve capacity".into());
                }
                self.expand(node);
                if self.nodes[node].leaf {
                    return Err("the mandatory public prefix exceeds solve capacity".into());                }
            }
            let next = match *step {
                PublicStep::Chance => {
                    if !self.nodes[node].chance || self.nodes[node].child.len() != 1 {
                        return Err("the mandatory chance transition is missing".into());
                    }
                    self.nodes[node].child[0]
                }
                PublicStep::Act(key) => {
                    if self.nodes[node].chance {
                        return Err("an action observation reaches a chance node".into());
                    }
                    let mut matches = self.nodes[node]
                        .acts
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| obs_key(a) == key)
                        .map(|(a, _)| self.nodes[node].child[self.nodes[node].obs_child[a]]);
                    let first = matches.next().ok_or_else(|| format!("observation {key} is absent from the mandatory prefix"))?;
                    if matches.any(|other| other != first) {
                        return Err(format!("observation {key} has ambiguous public children"));
                    }
                    first
                }
            };
            node = next;
            self.horizon = self.nodes[node].state.round + self.cfg.rounds as u16;
            if !self.nodes[node].state.is_terminal() {
                self.nodes[node].expandable = true;
            }
        }
        self.focus = node;
        Ok(())
    }

    fn prepare_focus(&mut self, successors: bool) -> Result<(), String> {
        if self.nodes[self.focus].leaf && self.nodes[self.focus].expandable {
            self.expand(self.focus);
        }
        if self.nodes[self.focus].leaf || self.nodes[self.focus].chance {
            return Err("a solve focus must be a decision".into());        }
        for node in &mut self.nodes {
            node.carry = false;
        }
        self.nodes[self.focus].carry = true;
        if successors {
            let children = self.nodes[self.focus].child.clone();
            for child in children {
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

    pub(super) fn read_round(&mut self) -> Vec<Call> {
        let mut calls = self.opening_calls();
        let node = &self.nodes[self.focus];
        let common = (self.slot, self.avg_touched, self.focus as u32, node.nc, node.legal_action.len() as u32);
        match &self.finish {
            Finish::Play(actual) => {
                let actor = node.player as usize;
                let index = node.cfgs[actor]
                    .binary_search(actual)
                    .expect("the acting configuration is in the focus support") as u32;
                let mut next_cap = [0u32; 2];
                for &child in &node.child {
                    for p in 0..2 {
                        next_cap[p] = next_cap[p].max(self.nodes[child].nc[p]);
                    }
                }
                calls.push(Call::ReadPlay {
                    solve: common.0,
                    touched: common.1,
                    focus: common.2,
                    focus_n: common.3,
                    cells: common.4,
                    actual: index,
                    next_cap,
                });
            }
            Finish::Refresh => calls.push(Call::ReadRefresh {
                solve: common.0,
                touched: common.1,
                focus: common.2,
                focus_n: common.3,
                cells: common.4,
            }),
            Finish::Target => calls.push(Call::ReadTarget {
                solve: common.0,
                touched: common.1,
                focus: common.2,
                focus_n: common.3,
                cells: common.4,
            }),
        }
        calls
    }

    pub(super) fn read_back(&mut self, r: &Reply) -> Result<SolveOutput, String> {
        let focus = self.focus;
        let node = &self.nodes[focus];
        let n = node.nc.map(|x| x as usize);
        let caps = if matches!(self.finish, Finish::Play(_)) {
            [
                node.child.iter().map(|&c| self.nodes[c].nc[0]).max().unwrap_or(0) as usize,
                node.child.iter().map(|&c| self.nodes[c].nc[1]).max().unwrap_or(0) as usize,
            ]
        } else {
            [0, 0]
        };
        let expected = 1 + 2 * (n[0] + n[1] + caps[0] + caps[1]) + node.legal_action.len();
        if r.a.len() != expected {
            return Err(format!("GPU read returned {} values, expected {expected}", r.a.len()));
        }
        let mut at = 1usize;
        let focus_reach = [&r.a[at..at + n[0]], &r.a[at + n[0]..at + n[0] + n[1]]];
        at += n[0] + n[1];
        let focus_value = [&r.a[at..at + n[0]], &r.a[at + n[0]..at + n[0] + n[1]]];
        at += n[0] + n[1];
        let focus_boundary = self.boundary(focus, focus_reach, focus_value)?;
        let queries = std::mem::take(&mut self.queries);
        match self.finish {
            Finish::Play(_) => {
                let cell = r.a[0].to_bits() as usize;
                if cell >= self.nodes[focus].legal_action.len() {
                    return Err("the GPU selected a cell outside the focus row".into());
                }
                let child = self.nodes[focus].legal_child[cell] as usize;
                if child >= self.nodes.len() || !self.nodes[child].carry {
                    return Err("the GPU selected an invalid continuation child".into());
                }
                let cn = self.nodes[child].nc.map(|x| x as usize);
                let next_reach = [&r.a[at..at + cn[0]], &r.a[at + caps[0]..at + caps[0] + cn[1]]];
                at += caps[0] + caps[1];
                let next_value = [&r.a[at..at + cn[0]], &r.a[at + caps[0]..at + caps[0] + cn[1]]];
                at += caps[0] + caps[1];
                let policy = self.policy_at(focus, &r.a[at..]);
                let action = self.nodes[focus].acts[self.nodes[focus].legal_action[cell] as usize];
                if self.nodes[child].state.is_terminal() {
                    Ok(SolveOutput::Play(Box::new(PlaySolved::Terminal(Box::new(PlayTerminal {
                        action,
                        policy,
                        focus: focus_boundary,
                        queries,
                    })))))
                } else {
                    let next = self.boundary(child, next_reach, next_value)?;
                    Ok(SolveOutput::Play(Box::new(PlaySolved::Continue(Box::new(PlayContinue {
                        action,
                        policy,
                        focus: focus_boundary,
                        next,
                        queries,
                    })))))
                }
            }
            Finish::Refresh => Ok(SolveOutput::Refresh(Box::new(RefreshSolved { focus: focus_boundary, queries }))),
            Finish::Target => {
                let policy = self.policy_at(focus, &r.a[at..]);
                Ok(SolveOutput::Target(Box::new(TargetSolved {
                    policy,
                    values: focus_boundary.cfv,
                    queries,
                })))
            }
        }
    }

    fn boundary(&self, node: usize, reach: [&[f32]; 2], values: [&[f32]; 2]) -> Result<Boundary, String> {
        let mut range = [Belief::default(), Belief::default()];
        let mut cfv = [Vec::new(), Vec::new()];
        let mass = [reach[0].iter().sum::<f32>(), reach[1].iter().sum::<f32>()];
        if mass.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(format!("carry node {node} has invalid mass {mass:?}"));
        }
        for p in 0..2 {
            if reach[p].iter().any(|x| !x.is_finite() || *x < 0.0)
                || values[p].iter().any(|x| !x.is_finite())
            {
                return Err(format!("carry node {node} has invalid values"));
            }
            range[p] = Belief {
                cfg: self.nodes[node].cfgs[p].to_vec(),
                p: reach[p].iter().map(|x| x / mass[p]).collect(),
            };
            cfv[p] = values[p].iter().map(|x| x / mass[1 - p]).collect();
        }
        Boundary::new(self.nodes[node].state, range, cfv)
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
