use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::pbs::{belief_after_draw, faceup_counts, reserve, set_config, true_config, Belief, Config, Ctx};
use crate::resolve::{apply_public_observation, Continuation, PublicState, PublicStep};
use crate::state::{Cont, State};

#[cfg(feature = "gpu")]
use crate::resolve::{ResolvePath, SolveOutput};
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

    fn prior(&self, p: usize) -> Result<Belief, String> {
        Ok(match self.continuation.as_ref().ok_or("the session has ended")? {
            Continuation::Unsolved(belief) => belief[p].clone(),
            _ => Belief::from_pairs(self.support[p].iter().map(|&c| (c, 1.0)).collect()),
        })
    }

    fn absorb(&mut self, p: usize, posterior: Belief) {
        self.support[p] = posterior.cfg.clone();
        if let Some(Continuation::Unsolved(belief)) = self.continuation.as_mut() {
            belief[p] = posterior;
        }
    }

    fn drew(&mut self, player: u8, code: Option<u32>) -> Result<(), String> {
        if !self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {player} is not drawing"));
        }
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
        let posterior = belief_after_draw(&self.prior(p)?, &res, &faceup, in_flight);
        self.absorb(p, posterior);
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
        let p = player as usize;
        let own = true_config(&self.s, self.seat, &self.ctx);
        let (public, posterior) =
            apply_public_observation(&PublicState::from_state(self.s), &self.prior(p)?, key)?;
        let mut state = public.state();
        set_config(&mut state, self.seat, &self.ctx, &own);
        self.absorb(p, posterior);
        if state.is_terminal() {
            self.continuation = None;
        } else if let Some(Continuation::Solved { path, .. }) = self.continuation.as_mut() {
            path.steps.push(PublicStep::Act(key));
        }
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
        let solver = Solver::refresh(
            self.continuation.as_ref().ok_or("the session has ended")?,
            &self.s,
            self.ctx,
            Arc::clone(&brain.net),
            brain.cfg,
            [self.prior(0)?, self.prior(1)?],
            Rng::new(self.rng.next_u64()),
        )?;
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
        let solver = Solver::play(
            self.continuation.as_ref().ok_or("the session has ended")?,
            &self.s,
            self.ctx,
            Arc::clone(&brain.net),
            brain.cfg,
            [self.prior(0)?, self.prior(1)?],
            Rng::new(self.rng.next_u64()),
            actual,
        )?;
        let SolveOutput::Play(solved) = brain.solve(solver)? else {
            return Err("play solve returned the wrong result type".into());
        };
        let action = solved.action;
        if let Some(next) = &solved.next {
            self.support = std::array::from_fn(|p| next.range[p].cfg.clone());
        }
        self.s.apply_inplace(action);
        self.continuation = solved.next.map(|boundary| Continuation::Solved {
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

    fn facing_the_opponent() -> Session {
        let mut session = Session::new(&draft(), 1, 9).unwrap();
        while session.s.is_chance() {
            let player = session.s.to_act();
            session.drew(player, None).unwrap();
        }
        session.seat = 1 - session.s.to_act();
        session
    }

    #[test]
    fn observation_has_no_policy_input() {
        let session = Session::new(&draft(), 0, 7).unwrap();
        let _observe: fn(&mut Session, &Obs) -> Result<(), String> = Session::observe;
        assert!(matches!(session.continuation, Some(Continuation::Unsolved(_))));
    }

    #[test]
    fn lifecycle_error_does_not_advance_the_session() {
        let mut session = facing_the_opponent();
        let state = session.s;
        let support = session.support.clone();
        let player = session.s.to_act();
        assert!(session.acted(player, u32::MAX).is_err());
        assert_eq!(session.s, state);
        assert_eq!(session.support, support);
    }

    #[test]
    fn an_opponent_action_before_the_first_solve_advances_the_belief() {
        let mut session = facing_the_opponent();
        let player = session.s.to_act();
        let key = obs_key(&session.s.legal_actions()[0]);
        session.acted(player, key).unwrap();
        let Some(Continuation::Unsolved(belief)) = &session.continuation else {
            panic!("the session must still be unsolved");
        };
        assert_eq!(belief[player as usize].cfg, session.support[player as usize]);
    }

    #[test]
    fn opponent_action_is_only_a_public_path_step() {
        let mut session = facing_the_opponent();
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
