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
            validate_belief(&state, &ctx, p, &ranges[p])?;
            set_config(&mut state, p as u8, &ctx, &ranges[p].cfg[0]);
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
        self.state.round == state.round && self.row == row
    }
}

pub(crate) fn validate_belief(state: &State, ctx: &Ctx, player: usize, belief: &Belief) -> Result<(), String> {
    if belief.cfg.len() != belief.p.len()
        || belief.cfg.is_empty()
        || belief.cfg.windows(2).any(|w| w[0] >= w[1])
        || belief.p.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(format!("player {player} belief is invalid"));
    }
    let mass: f32 = belief.p.iter().sum();
    if !mass.is_finite() || mass <= 0.0 || (mass - 1.0).abs() > 2e-4 {
        return Err(format!("player {player} belief mass is {mass}"));
    }
    let truth = true_config(state, player as u8, ctx);
    let room = reserve(state, player as u8, ctx);
    if belief.cfg.iter().any(|c| {
        c.hand_size() != truth.hand_size()
            || c.fd_size() != truth.fd_size()
            || c.inflight.is_some() != truth.inflight.is_some()
            || (0..NSLOT).any(|k| c.hand[k] + c.fd[k] + u8::from(c.inflight == Some(k as u8)) > room[k])
    }) {
        return Err(format!("player {player} belief contains an impossible configuration"));
    }
    Ok(())
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
        self.state.round == other.state.round && self.row == other.row
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
    pub fn new(state: State, range: [Belief; 2], cfv: [Vec<f32>; 2]) -> Result<Self, String> {
        let public = PublicState::new(state, &range)?;
        for p in 0..2 {
            if range[p].cfg.len() != cfv[p].len() || cfv[p].iter().any(|x| !x.is_finite()) {
                return Err(format!("player {p} boundary values are invalid"));
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

pub struct PlaySolved {
    pub action: Action,
    pub policy: Policy,
    pub focus: Boundary,
    pub queries: Vec<(State, [Belief; 2])>,
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
    for k in 0..terminate.len() {
        let value = strategy[k][0] * terminate[k] + strategy[k][1] * follow[k];
        for a in 0..2 {
            let action = if a == 0 { terminate[k] } else { follow[k] };
            let old = regret[k][a];
            regret[k][a] = old * if old > 0.0 { da } else { db } + action - value;
            sum[k][a] = sum[k][a] * dg + strategy[k][a];
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

pub fn apply_public_observation(public: &PublicState, prior: &Belief, key: u32) -> Result<(PublicState, Belief), String> {
    let state = public.state();
    if state.is_terminal() || state.is_chance() {
        return Err("an action observation requires a live decision".into());
    }
    let ctx = Ctx::new(&state);
    let actor = state.to_act();
    let support = &prior.cfg;
    if support.is_empty() || support.windows(2).any(|w| w[0] >= w[1]) {
        return Err("action observation support is empty, duplicated, or unsorted".into());
    }
    let (acts, slots, facedown) = crate::search::node_actions(&state, actor, &ctx, support);
    let mut found: Option<PublicState> = None;
    let mut reached: Vec<(Config, f32)> = Vec::new();
    for ((a, slot), facedown) in acts.into_iter().zip(slots).zip(facedown) {
        if obs_key(&a) != key {
            continue;
        }
        for (c, mass) in support.iter().zip(&prior.p) {
            let Some(config) = crate::pbs::advance_config(c, slot, facedown) else { continue };
            let mut next = state;
            set_config(&mut next, actor, &ctx, c);
            next.apply_inplace(a);
            let candidate = PublicState::from_state(next);
            if found.as_ref().is_some_and(|old| old != &candidate) {
                return Err(format!("observation {key} has ambiguous public effects"));
            }
            found = Some(candidate);
            reached.push((config, *mass));
        }
    }
    let posterior = Belief::from_pairs(reached);
    found.map(|public| (public, posterior))
        .ok_or_else(|| format!("observation {key} is structurally unreachable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Cfr;

    #[test]
    fn public_identity_includes_the_round() {
        let mut rng = crate::rng::Rng::new(3);
        let state = crate::selfplay::make_game(&mut rng, false);
        let public = PublicState::from_state(state);
        let mut later = state;
        later.round += 1;
        assert!(!public.same_public(&later));
    }

    #[test]
    fn malformed_beliefs_are_rejected() {
        let mut rng = crate::rng::Rng::new(4);
        let state = crate::selfplay::make_game(&mut rng, false);
        let mut ranges = [Belief::point(Config::default()), Belief::point(Config::default())];
        ranges[0].p[0] = f32::NAN;
        assert!(PublicState::new(state, &ranges).is_err());
    }

    #[test]
    fn terminate_protects_each_private_state() {
        let term = [0.8, -0.4];
        let follow = [-0.2, 0.6];
        let mut regret = [[0.0; 2]; 2];
        let mut strategy = [[0.5; 2]; 2];
        let mut sum = [[0.0; 2]; 2];
        for step in 0..128 {
            gadget_iteration(&term, &follow, &mut regret, &mut strategy, &mut sum, step, Cfr::DISCOUNTED);
        }
        assert!(strategy[0][0] > 0.99);
        assert!(strategy[1][1] > 0.99);
        for k in 0..2 {
            let average = (sum[k][0] * term[k] + sum[k][1] * follow[k]) / (sum[k][0] + sum[k][1]);
            assert!(average + 0.02 >= term[k]);
        }
    }
}
