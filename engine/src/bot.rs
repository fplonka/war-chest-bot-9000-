use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::pbs::{belief_after_draw, faceup_counts, reserve, set_config, true_config, Belief, Config, Ctx};
use crate::resolve::{apply_public_observation, Continuation, PublicState, PublicStep};
use crate::state::{Cont, State};

#[cfg(feature = "gpu")]
use crate::resolve::{PlaySolved, ResolvePath, SolveOutput};
#[cfg(feature = "gpu")]
use crate::rng::Rng;
#[cfg(feature = "gpu")]
use crate::search::Cfg;
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::farm::Cards;
#[cfg(feature = "gpu")]
use crate::net::Net;
#[cfg(feature = "gpu")]
use crate::search::{Solver, Step};

#[cfg(feature = "gpu")]
pub struct Brain {
    pub cards: Arc<Cards>,
    pub net: Arc<Net>,
    pub cfg: Cfg,
}

#[cfg(feature = "gpu")]
impl Brain {
    fn solve(&self, mut solver: Solver) -> Result<SolveOutput, String> {
        let seat = self.cards.seat();
        solver.pin(seat.slot);
        let mut replies = Vec::new();
        loop {
            match solver.advance(&replies) {
                Step::Calls(calls) => {
                    replies = self.cards.round(seat.lane, calls).ok_or("the GPU solve failed")?;
                }
                Step::Done(solved) => return solved,
            }
        }
    }
}

pub struct Session {
    s: State,
    ctx: Ctx,
    seat: u8,
    support: [Vec<Config>; 2],
    continuation: Option<Continuation>,
    #[cfg(feature = "gpu")]
    rng: Rng,
}

impl Session {
    pub fn new(draft: &Draft, seat: u8, _seed: u64) -> Result<Session, String> {
        let s = draft.state()?;
        let ctx = Ctx::new(&s);
        let belief = [Belief::point(Config::default()), Belief::point(Config::default())];
        PublicState::new(s, &belief)?;
        Ok(Session {
            s,
            ctx,
            seat,
            support: [vec![Config::default()], vec![Config::default()]],
            continuation: Some(Continuation::Unsolved(belief)),
            #[cfg(feature = "gpu")]
            rng: Rng::new(_seed),
        })
    }

    pub fn observe(&mut self, obs: &Obs) -> Result<(), String> {
        match *obs {
            Obs::Draw { player, code } => self.drew(player, code),
            Obs::Act { player, key } => self.acted(player, key),
        }
    }

    fn drew(&mut self, player: u8, code: Option<u32>) -> Result<(), String> {
        if !self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {player} is not drawing"));
        }
        self.continuation.as_ref().ok_or("the session has ended")?;
        let legal = self.s.legal_actions();
        let drawn = match code {
            Some(code) => Action::decode(code).ok_or_else(|| format!("draw {code} does not decode"))?,
            None => legal[0],
        };
        if !legal.contains(&drawn) {
            return Err(format!("draw {} is illegal", drawn.encode()));
        }
        let p = player as usize;
        let res = reserve(&self.s, player, &self.ctx);
        let faceup = faceup_counts(&self.s, player, &self.ctx);
        let in_flight = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
        match self.continuation.as_mut().unwrap() {
            Continuation::Unsolved(belief) => {
                belief[p] = belief_after_draw(&belief[p], &res, &faceup, in_flight);
                self.support[p] = belief[p].cfg.clone();
            }
            Continuation::Solved { .. } => {
                let n = self.support[p].len();
                let prior = Belief { cfg: self.support[p].clone(), p: vec![1.0 / n as f32; n] };
                self.support[p] = belief_after_draw(&prior, &res, &faceup, in_flight).cfg;
            }
        }
        let mut state = self.s;
        state.apply_inplace(drawn);
        let more = matches!(state.pending(), Cont::Draw { .. }) && state.to_act() == player;
        if !more {
            if let Some(Continuation::Solved { path, .. }) = self.continuation.as_mut() {
                path.steps.push(PublicStep::Chance);
            }
        }
        self.s = state;
        Ok(())
    }

    fn acted(&mut self, player: u8, key: u32) -> Result<(), String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {player} is not to act"));
        }
        if player == self.seat {
            return Err("a bot is told its own move back".into());
        }
        let Continuation::Solved { .. } = self.continuation.as_ref().ok_or("the session has ended")? else {
            return Err("an opponent action arrived before the initial refresh".into());
        };
        let own = true_config(&self.s, self.seat, &self.ctx);
        let (public, support) = apply_public_observation(
            &PublicState::from_state(self.s), &self.support[player as usize], key)?;
        let mut state = public.state();
        set_config(&mut state, self.seat, &self.ctx, &own);
        if state.is_terminal() {
            self.continuation = None;
        } else if let Some(Continuation::Solved { path, .. }) = self.continuation.as_mut() {
            path.steps.push(PublicStep::Act(key));
        }
        self.support[player as usize] = support;
        self.s = state;
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn watch(&mut self, brain: &Brain) -> Result<(), String> {
        if self.s.is_terminal() {
            self.continuation = None;
            return Ok(());
        }
        if self.s.is_chance() || self.s.to_act() == self.seat {
            return Ok(());
        }
        if let Some(Continuation::Solved { boundary, path }) = &self.continuation {
            if path.steps.is_empty() && boundary.public.same_public(&self.s) {
                return Ok(());
            }
        }
        let solver = match self.continuation.as_ref().ok_or("the session has ended")? {
            Continuation::Unsolved(belief) => Solver::initial_refresh(
                &self.s,
                self.ctx,
                Arc::clone(&brain.net),
                brain.cfg,
                belief.clone(),
                Rng::new(self.rng.next_u64()),
            )?,
            Continuation::Solved { boundary, path } => Solver::resolve_refresh(
                boundary.as_ref().clone(),
                path.clone(),
                &self.s,
                Arc::clone(&brain.net),
                brain.cfg,
                Rng::new(self.rng.next_u64()),
            )?,
        };
        let SolveOutput::Refresh(solved) = brain.solve(solver)? else {
            return Err("refresh solve returned the wrong result type".into());
        };
        self.support = std::array::from_fn(|p| solved.focus.range[p].cfg.clone());
        self.continuation = Some(Continuation::Solved {
            boundary: Box::new(solved.focus),
            path: ResolvePath::default(),
        });
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn decide(&mut self, brain: &Brain) -> Result<u32, String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != self.seat {
            return Err("the bot was asked to move out of turn".into());
        }
        let actual = true_config(&self.s, self.seat, &self.ctx);
        let solver = match self.continuation.as_ref().ok_or("the session has ended")? {
            Continuation::Unsolved(belief) => Solver::initial_play(
                &self.s,
                self.ctx,
                Arc::clone(&brain.net),
                brain.cfg,
                belief.clone(),
                Rng::new(self.rng.next_u64()),
                actual,
            )?,
            Continuation::Solved { boundary, path } => Solver::resolve_play(
                boundary.as_ref().clone(),
                path.clone(),
                &self.s,
                Arc::clone(&brain.net),
                brain.cfg,
                Rng::new(self.rng.next_u64()),
                actual,
            )?,
        };
        let SolveOutput::Play(solved) = brain.solve(solver)? else {
            return Err("play solve returned the wrong result type".into());
        };
        let (action, next) = match *solved {
            PlaySolved::Continue(s) => (s.action, Some(s.next)),
            PlaySolved::Terminal(s) => (s.action, None),
        };
        self.s.apply_inplace(action);
        if let Some(boundary) = &next {
            self.support = std::array::from_fn(|p| boundary.range[p].cfg.clone());
        }
        self.continuation = next.map(|boundary| Continuation::Solved {
            boundary: Box::new(boundary),
            path: ResolvePath::default(),
        });
        Ok(action.encode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbs::{obs_key, uniform_belief};
    use crate::resolve::Boundary;

    fn draft() -> Draft {
        Draft {
            white: vec![17, 12, 4, 9],
            black: vec![1, 3, 8, 16],
            first: 0,
        }
    }

    #[test]
    fn observation_has_no_policy_input() {
        let session = Session::new(&draft(), 0, 7).unwrap();
        let _observe: fn(&mut Session, &Obs) -> Result<(), String> = Session::observe;
        assert!(matches!(session.continuation, Some(Continuation::Unsolved(_))));
    }

    #[test]
    fn lifecycle_error_does_not_advance_the_session() {
        let mut session = Session::new(&draft(), 1, 9).unwrap();
        while session.s.is_chance() {
            let player = session.s.to_act();
            session.drew(player, None).unwrap();
        }
        session.seat = 1 - session.s.to_act();
        let state = session.s;
        let support = session.support.clone();
        let player = session.s.to_act();
        let key = obs_key(&session.s.legal_actions()[0]);
        assert!(session.acted(player, key).is_err());
        assert_eq!(session.s, state);
        assert_eq!(session.support, support);
    }

    #[test]
    fn opponent_action_is_only_a_public_path_step() {
        let mut session = Session::new(&draft(), 1, 9).unwrap();
        while session.s.is_chance() {
            let player = session.s.to_act();
            session.drew(player, None).unwrap();
        }
        session.seat = 1 - session.s.to_act();
        let ranges = [
            uniform_belief(&session.s, &session.ctx, 0),
            uniform_belief(&session.s, &session.ctx, 1),
        ];
        let boundary = Boundary::new(
            session.s,
            ranges.clone(),
            [vec![0.0; ranges[0].len()], vec![0.0; ranges[1].len()]],
        )
        .unwrap();
        session.continuation = Some(Continuation::Solved {
            boundary: Box::new(boundary),
            path: Default::default(),
        });
        let player = session.s.to_act();
        let key = obs_key(&session.s.legal_actions()[0]);
        session.observe(&Obs::Act { player, key }).unwrap();
        let Some(Continuation::Solved { boundary, path }) = &session.continuation else { unreachable!() };
        assert_eq!(path.steps, [PublicStep::Act(key)]);
        assert_eq!(boundary.range[0].p, ranges[0].p);
        assert_eq!(boundary.range[1].p, ranges[1].p);
    }
}
