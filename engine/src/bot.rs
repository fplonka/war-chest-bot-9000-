use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::pbs::{belief_after_draw, faceup_counts, reserve, set_config, true_config, Belief, Config, Ctx};
use crate::policy;
use crate::resolve::{observed_state, Continuation, PublicState, Solved};
use crate::state::{Cont, State};

#[cfg(feature = "gpu")]
use crate::pbs::obs_key;
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
    fn solve(&self, mut solver: Solver) -> Result<Solved, String> {
        let seat = self.cards.seat();
        solver.pin(seat.slot);
        let mut replies = Vec::new();
        loop {
            match solver.advance(&replies) {
                Step::Calls(calls) => {
                    replies = self.cards.round(seat.lane, calls).ok_or("the GPU solve failed")?;
                }
                Step::Done(solved) => return solved.map(|solved| *solved),
            }
        }
    }
}

pub struct Session {
    s: State,
    ctx: Ctx,
    seat: u8,
    belief: [Belief; 2],
    continuation: Option<Continuation>,
    refreshed: Option<Solved>,
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
            belief,
            continuation: None,
            refreshed: None,
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
        self.belief[p] = belief_after_draw(&self.belief[p], &res, &faceup, in_flight);
        let mut state = self.s;
        state.apply_inplace(drawn);
        let more = matches!(state.pending(), Cont::Draw { .. }) && state.to_act() == player;
        if !more {
            if let Some(continuation) = self.continuation.as_mut() {
                continuation.draws += 1;
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
        let mut state = observed_state(&self.s, &self.belief[p].cfg, key)?;
        set_config(&mut state, self.seat, &self.ctx, &own);
        self.continuation = match self.refreshed.take() {
            Some(solved) => solved.advance(&mut self.belief, &self.ctx, 0.0, key)?,
            None => {
                let model = policy::uniform(&self.s, &self.ctx, player, &self.belief[p].cfg);
                self.belief[p] = model.posterior(&self.belief[p], key);
                None
            }
        };
        self.s = state;
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn watch(&mut self, brain: &Brain) -> Result<(), String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() == self.seat || self.refreshed.is_some() {
            return Ok(());
        }
        let solver = Solver::refresh(
            self.continuation.as_ref(),
            &self.s,
            self.ctx,
            Arc::clone(&brain.net),
            brain.cfg,
            self.belief.clone(),
            Rng::new(self.rng.next_u64()),
        )?;
        self.refreshed = Some(brain.solve(solver)?);
        self.continuation = None;
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn decide(&mut self, brain: &Brain) -> Result<u32, String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != self.seat {
            return Err("the bot was asked to move out of turn".into());
        }
        let actual = true_config(&self.s, self.seat, &self.ctx);
        let solver = Solver::play(
            self.continuation.as_ref(),
            &self.s,
            self.ctx,
            Arc::clone(&brain.net),
            brain.cfg,
            self.belief.clone(),
            Rng::new(self.rng.next_u64()),
            actual,
        )?;
        let solved = brain.solve(solver)?;
        let action = solved.action.ok_or("a play solve chose no action")?;
        self.continuation = solved.advance(&mut self.belief, &self.ctx, 0.0, obs_key(&action))?;
        self.s.apply_inplace(action);
        Ok(action.encode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbs::{obs_key, uniform_belief};
    use crate::resolve::Values;

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
        assert!(session.continuation.is_none() && session.refreshed.is_none());
    }

    #[test]
    fn lifecycle_error_does_not_advance_the_session() {
        let mut session = facing_the_opponent();
        let state = session.s;
        let belief = session.belief.clone();
        let player = session.s.to_act();
        assert!(session.acted(player, u32::MAX).is_err());
        assert_eq!(session.s, state);
        assert_eq!(session.belief[0].cfg, belief[0].cfg);
        assert_eq!(session.belief[1].cfg, belief[1].cfg);
    }

    #[test]
    fn an_opponent_action_before_a_refresh_is_filtered_under_a_uniform_model() {
        let mut session = facing_the_opponent();
        let player = session.s.to_act();
        let prior = session.belief[player as usize].clone();
        let key = obs_key(&session.s.legal_actions()[0]);
        let want = policy::uniform(&session.s, &session.ctx, player, &prior.cfg).posterior(&prior, key);
        session.acted(player, key).unwrap();
        assert!(session.continuation.is_none());
        assert_eq!(session.belief[player as usize].cfg, want.cfg);
        assert_eq!(session.belief[player as usize].p, want.p);
    }

    #[test]
    fn a_stale_boundary_does_not_survive_an_unrefreshed_opponent_action() {
        let mut session = facing_the_opponent();
        let range = [
            uniform_belief(&session.s, &session.ctx, 0),
            uniform_belief(&session.s, &session.ctx, 1),
        ];
        let values = Values {
            state: session.s,
            cfgs: [range[0].cfg.clone(), range[1].cfg.clone()],
            cfv: [vec![0.0; range[0].len()], vec![0.0; range[1].len()]],
        };
        session.belief = range.clone();
        session.continuation = Some(Continuation::new(values, range.clone()).unwrap());
        let player = session.s.to_act();
        let key = obs_key(&session.s.legal_actions()[0]);
        session.observe(&Obs::Act { player, key }).unwrap();
        assert!(session.continuation.is_none());
        assert_eq!(session.belief[1 - player as usize].p, range[1 - player as usize].p);
    }
}
