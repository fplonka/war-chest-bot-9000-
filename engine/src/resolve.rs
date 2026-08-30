use crate::actions::Action;
use crate::pbs::{obs_key, pack_row, reserve, set_config, true_config, Belief, Config, Ctx, NSLOT, ROW_BYTES};
use crate::search::Policy;
use crate::state::State;

#[derive(Clone, Debug)]
pub struct PublicState {
    state: State,
    row: [u8; ROW_BYTES],
}

impl PublicState {
    pub fn new(mut state: State, ranges: &[Belief; 2]) -> Result<Self, String> {
        let ctx = Ctx::new(&state);
        for p in 0..2 {
            let c = ranges[p].cfg.first().ok_or_else(|| format!("player {p} has empty structural support"))?;
            set_config(&mut state, p as u8, &ctx, c);
        }
        let mut row = [0; ROW_BYTES];
        pack_row(&state, &ctx, &mut row);
        Ok(Self { state, row })
    }

    pub fn from_state(state: State) -> Self {
        let ctx = Ctx::new(&state);
        let ranges = [
            Belief::point(canonical_config(&state, &ctx, 0)),
            Belief::point(canonical_config(&state, &ctx, 1)),
        ];
        Self::new(state, &ranges).expect("a live state has structural support")
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn same_public(&self, state: &State) -> bool {
        let mut row = [0; ROW_BYTES];
        pack_row(state, &Ctx::new(state), &mut row);
        self.row == row
    }
}

fn canonical_config(state: &State, ctx: &Ctx, player: u8) -> Config {
    let truth = true_config(state, player, ctx);
    let mut room = reserve(state, player, ctx);
    let inflight = truth.inflight.map(|_| {
        let slot = (0..NSLOT).find(|&k| room[k] > 0).expect("an in-flight coin exists");
        room[slot] -= 1;
        slot as u8
    });
    let mut fill = |mut left: u8| {
        let mut out = [0; NSLOT];
        for k in 0..NSLOT {
            out[k] = room[k].min(left);
            room[k] -= out[k];
            left -= out[k];
        }
        assert_eq!(left, 0, "public private-zone sizes fit the reserve");
        out
    };
    let hand = fill(truth.hand_size());
    let fd = fill(truth.fd_size());
    Config { hand, fd, inflight }
}

impl PartialEq for PublicState {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}

impl Eq for PublicState {}

#[derive(Clone, Debug)]
pub struct Boundary {
    pub public: PublicState,
    pub range: [Belief; 2],
    pub cfv: [Vec<f32>; 2],
}

impl Boundary {
    pub fn new(public: PublicState, range: [Belief; 2], cfv: [Vec<f32>; 2]) -> Result<Self, String> {
        for p in 0..2 {
            if range[p].cfg.len() != range[p].p.len() || range[p].cfg.len() != cfv[p].len() {
                return Err(format!("player {p} boundary support is misaligned"));
            }
            if range[p].cfg.is_empty() || range[p].cfg.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("player {p} boundary support is empty, duplicated, or unsorted"));
            }
            if range[p].p.iter().chain(cfv[p].iter()).any(|x| !x.is_finite()) {
                return Err(format!("player {p} boundary contains a non-finite number"));
            }
            if range[p].p.iter().any(|&x| x < 0.0) {
                return Err(format!("player {p} boundary contains a negative probability"));
            }
            let mass: f32 = range[p].p.iter().sum();
            if mass <= 0.0 || (mass - 1.0).abs() > 2e-4 {
                return Err(format!("player {p} boundary mass is {mass}"));
            }
        }
        Ok(Self { public, range, cfv })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicStep {
    Act(u32),
    Chance,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvePath {
    pub steps: Vec<PublicStep>,
}

#[derive(Clone, Debug)]
pub enum Continuation {
    Unsolved([Belief; 2]),
    Solved { boundary: Box<Boundary>, path: ResolvePath },
}

pub struct PlayContinue {
    pub action: Action,
    pub policy: Policy,
    pub focus: Boundary,
    pub next: Boundary,
    pub queries: Vec<(State, [Belief; 2])>,
}

pub struct PlayTerminal {
    pub action: Action,
    pub policy: Policy,
    pub focus: Boundary,
    pub queries: Vec<(State, [Belief; 2])>,
}

pub enum PlaySolved {
    Continue(Box<PlayContinue>),
    Terminal(Box<PlayTerminal>),
}

pub struct RefreshSolved {
    pub focus: Boundary,
    pub queries: Vec<(State, [Belief; 2])>,
}

pub struct TargetSolved {
    pub policy: Policy,
    pub values: [Vec<f32>; 2],
    pub queries: Vec<(State, [Belief; 2])>,
}

pub enum SolveOutput {
    Play(Box<PlaySolved>),
    Refresh(Box<RefreshSolved>),
    Target(Box<TargetSolved>),
}

pub fn gadget_iteration(
    q: &[f32],
    terminate: &[f32],
    follow: &[f32],
    regret: &mut [[f32; 2]],
    strategy: &mut [[f32; 2]],
    sum: &mut [[f32; 2]],
    step: usize,
    cfr: crate::search::Cfr,
) {
    fn factor(t: f32, p: f32) -> f32 {
        if p.is_infinite() {
            return if p.is_sign_positive() { 1.0 } else { 0.0 };
        }
        let x = t.powf(p);
        x / (x + 1.0)
    }
    let m = step as f32 + 1.0;
    let da = factor(m, cfr.alpha);
    let db = factor(m, cfr.beta);
    let dg = ((m - 1.0) / m).powf(cfr.gamma);
    for k in 0..q.len() {
        let value = strategy[k][0] * terminate[k] + strategy[k][1] * follow[k];
        for a in 0..2 {
            let action = if a == 0 { terminate[k] } else { follow[k] };
            let old = regret[k][a];
            regret[k][a] = old * if old > 0.0 { da } else { db } + q[k] * (action - value);
            sum[k][a] = sum[k][a] * dg + q[k] * strategy[k][a];
        }
        let positive = [regret[k][0].max(0.0), regret[k][1].max(0.0)];
        let total = positive[0] + positive[1];
        strategy[k] = if total > 1e-30 {
            [positive[0] / total, positive[1] / total]
        } else {
            [0.5, 0.5]
        };
    }
}

pub fn apply_public_observation(public: &PublicState, support: &[Config], key: u32) -> Result<PublicState, String> {
    let state = public.state();
    if state.is_terminal() || state.is_chance() {
        return Err("an action observation requires a live decision".into());
    }
    let ctx = Ctx::new(&state);
    let actor = state.to_act();
    if support.is_empty() || support.windows(2).any(|w| w[0] >= w[1]) {
        return Err("action observation support is empty, duplicated, or unsorted".into());
    }
    let (acts, slots, _) = crate::search::node_actions(&state, actor, &ctx, support);
    let mut found: Option<PublicState> = None;
    for (a, slot) in acts.into_iter().zip(slots) {
        if obs_key(&a) != key {
            continue;
        }
        for c in support.iter().filter(|c| crate::pbs::action_legal(c, slot)) {
            let mut next = state;
            set_config(&mut next, actor, &ctx, c);
            next.apply_inplace(a);
            let candidate = PublicState::from_state(next);
            if found.as_ref().is_some_and(|old| old != &candidate) {
                return Err(format!("observation {key} has ambiguous public effects"));
            }
            found = Some(candidate);
        }
    }
    found.ok_or_else(|| format!("observation {key} is structurally unreachable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Cfr;

    #[test]
    fn terminate_protects_each_private_state() {
        let q = [0.45, 0.55];
        let term = [0.8, -0.4];
        let follow = [-0.2, 0.6];
        let mut regret = [[0.0; 2]; 2];
        let mut strategy = [[0.5; 2]; 2];
        let mut sum = [[0.0; 2]; 2];
        for step in 0..128 {
            gadget_iteration(&q, &term, &follow, &mut regret, &mut strategy, &mut sum, step, Cfr::DISCOUNTED);
        }
        assert!(strategy[0][0] > 0.99);
        assert!(strategy[1][1] > 0.99);
        for k in 0..2 {
            let average = (sum[k][0] * term[k] + sum[k][1] * follow[k]) / (sum[k][0] + sum[k][1]);
            assert!(average + 0.02 >= term[k]);
        }
    }
}
