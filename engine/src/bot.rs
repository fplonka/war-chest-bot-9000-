use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::pbs::{belief_after_draw, faceup_counts, obs_key, reserve, set_config, true_config, Belief, Config, Ctx};
use crate::policy::{self, NodePolicy};
use crate::rng::Rng;

#[cfg(feature = "gpu")]
use crate::farm::Cards;
use crate::net::Net;
#[cfg(feature = "gpu")]
use crate::search::{Solver, Step};
use crate::search::Cfg;
use crate::state::{Cont, State};
use std::sync::Arc;

#[derive(Clone)]
pub enum Mind {
    #[cfg(feature = "gpu")]
    Sog(Arc<Cards>),
    Random,
    Greedy { temp: f32 },
}

pub struct Brain {
    pub mind: Mind,
    pub net: Arc<Net>,
    pub cfg: Cfg,
}

impl Brain {
    #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
    pub fn policy(&self, s: &State, ctx: &Ctx, player: u8, bel: &[Belief; 2], rng: &mut Rng) -> NodePolicy {
        let cfgs = &bel[player as usize].cfg;
        match &self.mind {
            Mind::Random => policy::uniform(s, ctx, player, cfgs),
            Mind::Greedy { temp } => policy::greedy(s, ctx, player, cfgs, *temp),
            #[cfg(feature = "gpu")]
            Mind::Sog(cards) => self.solve(s, ctx, bel, rng, cards),
        }
    }

    #[cfg(feature = "gpu")]
    fn solve(&self, s: &State, ctx: &Ctx, bel: &[Belief; 2], rng: &mut Rng, cards: &Cards) -> NodePolicy {
        let mut sv = Solver::new(
            s,
            *ctx,
            Arc::clone(&self.net),
            self.cfg,
            bel.clone(),
            Rng::new(rng.next_u64()),
        );
        let seat = cards.seat();
        sv.pin(seat.slot);
        let mut replies = Vec::new();
        while let Step::Calls(calls) = sv.advance(&replies) {
            replies = cards
                .round(seat.lane, calls)
                .expect("a card failed while a solve was still running");
        }
        policy::root(&sv)
    }
}

pub struct Session {
    s: State,
    ctx: Ctx,
    bel: [Belief; 2],
    seat: u8,
    rng: Rng,
    modelled: Option<NodePolicy>,
}

impl Session {
    pub fn new(draft: &Draft, seat: u8, seed: u64) -> Result<Session, String> {
        let s = draft.state()?;
        let ctx = Ctx::new(&s);
        Ok(Session {
            s,
            ctx,
            bel: [Belief::point(Config::default()), Belief::point(Config::default())],
            seat,
            rng: Rng::new(seed),
            modelled: None,
        })
    }

    fn resync(&mut self, player: u8) {
        let stand_in = self.bel[player as usize].cfg[0];
        set_config(&mut self.s, player, &self.ctx, &stand_in);
    }

    pub fn watch(&mut self, brain: &Brain) {
        if !self.s.is_terminal() && !self.s.is_chance() && self.s.to_act() != self.seat {
            let player = self.s.to_act();
            self.modelled = Some(brain.policy(&self.s, &self.ctx, player, &self.bel, &mut self.rng));
        }
    }

    pub fn observe(&mut self, obs: &Obs, brain: &Brain) -> Result<(), String> {
        match *obs {
            Obs::Draw { player, code } => self.drew(player, code),
            Obs::Act { player, key } => self.acted(player, key, brain),
        }
    }

    fn drew(&mut self, player: u8, code: Option<u32>) -> Result<(), String> {
        if !self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {} is not drawing", player));
        }
        self.modelled = None;
        let res = reserve(&self.s, player, &self.ctx);
        let faceup = faceup_counts(&self.s, player, &self.ctx);
        let in_flight = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
        self.bel[player as usize] = belief_after_draw(&self.bel[player as usize], &res, &faceup, in_flight);
        let drawn = match code {
            Some(code) => Action::decode(code).ok_or_else(|| format!("draw {} does not decode", code))?,
            None => self.s.legal_actions()[0],
        };
        self.s.apply_inplace(drawn);
        if player != self.seat {
            self.resync(player);
        }
        Ok(())
    }

    fn acted(&mut self, player: u8, key: u32, brain: &Brain) -> Result<(), String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {} is not to act", player));
        }
        if player == self.seat {
            return Err("a bot is told its own moves back".into());
        }
        let np = match self.modelled.take() {
            Some(np) => np,
            None => brain.policy(&self.s, &self.ctx, player, &self.bel, &mut self.rng),
        };
        let (ci, cell) = np
            .cell_for(key)
            .ok_or_else(|| format!("observation {} is unreachable from this belief", key))?;
        let posterior = np.posterior(&self.bel[player as usize], key);
        set_config(&mut self.s, player, &self.ctx, &self.bel[player as usize].cfg[ci]);
        self.s.apply_inplace(np.acts[np.action_at(cell)]);
        self.bel[player as usize] = posterior;
        self.resync(player);
        Ok(())
    }

    pub fn decide(&mut self, brain: &Brain) -> Result<u32, String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != self.seat {
            return Err("the bot was asked to move out of turn".into());
        }
        self.modelled = None;
        let truth = true_config(&self.s, self.seat, &self.ctx);
        let ci = self.bel[self.seat as usize]
            .index_of(&truth)
            .ok_or("the belief filter dropped this seat's own config")?;
        let np = brain.policy(&self.s, &self.ctx, self.seat, &self.bel, &mut self.rng);
        if np.row(ci).is_empty() {
            return Err(format!(
                "this seat's hand has no legal action here ({} actions at the node)",
                np.acts.len()
            ));
        }
        let cell = np.sample(&mut self.rng, ci);
        let action = np.acts[np.action_at(cell)];
        self.bel[self.seat as usize] = np.posterior(&self.bel[self.seat as usize], obs_key(&action));
        self.s.apply_inplace(action);
        Ok(action.encode())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::arena::{Done, Reply, Table};

    fn draft() -> Draft {
        Draft {
            white: vec![17, 12, 4, 9],
            black: vec![1, 3, 8, 16],
            first: 0,
        }
    }

    fn brain() -> Brain {
        Brain {
            mind: Mind::Random,
            net: Arc::new(Net::default()),
            cfg: Cfg::default(),
        }
    }

    fn play(seed: u64, mut inspect: impl FnMut(&Session)) -> f32 {
        let brain = brain();
        let mut table = Table::new();
        table.start(1, &draft(), [0, 1], seed).unwrap();
        let mut seats = [
            Session::new(&draft(), 0, seed ^ 11).unwrap(),
            Session::new(&draft(), 1, seed ^ 22).unwrap(),
        ];
        let mut waiting: [HashSet<u32>; 2] = [HashSet::new(), HashSet::new()];
        loop {
            table.settle();
            if let Some(&(_, _, z)) = table.reap().first() {
                return z;
            }
            for bot in 0..2 {
                let request = table.request(bot);
                for id in &request.drop {
                    assert!(
                        !waiting[bot].contains(id),
                        "game {} dropped while bot {} was still answering about it",
                        id,
                        bot
                    );
                }
                let mut done = Vec::new();
                for ask in request.watch {
                    waiting[bot].insert(ask.id);
                    for obs in &ask.obs {
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    seats[bot].watch(&brain);
                    done.push(Done {
                        id: ask.id,
                        action: None,
                    });
                }
                for ask in request.go {
                    waiting[bot].insert(ask.id);
                    for obs in &ask.obs {
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    inspect(&seats[bot]);
                    done.push(Done {
                        id: ask.id,
                        action: Some(seats[bot].decide(&brain).unwrap()),
                    });
                }
                for entry in &done {
                    waiting[bot].remove(&entry.id);
                }
                table.accept(bot, Reply { done, error: None }).unwrap();
            }
        }
    }

    #[test]
    fn a_seat_keeps_its_own_config_in_support() {
        play(77, |sess| {
            let truth = true_config(&sess.s, sess.seat, &sess.ctx);
            assert!(
                sess.bel[sess.seat as usize].index_of(&truth).is_some(),
                "seat {} lost its own config",
                sess.seat
            );
        });
    }

    #[test]
    fn a_seat_is_told_nothing_private() {
        let brain = brain();
        let mut table = Table::new();
        table.start(1, &draft(), [0, 1], 8191).unwrap();
        let mut seats = [
            Session::new(&draft(), 0, 1).unwrap(),
            Session::new(&draft(), 1, 2).unwrap(),
        ];
        let mut checked = 0;
        for _ in 0..60 {
            table.settle();
            if !table.reap().is_empty() {
                break;
            }
            for bot in 0..2 {
                let request = table.request(bot);
                let mut done = Vec::new();
                for (ask, acting) in request
                    .watch
                    .into_iter()
                    .map(|a| (a, false))
                    .chain(request.go.into_iter().map(|a| (a, true)))
                {
                    for obs in &ask.obs {
                        match *obs {
                            Obs::Draw { player, code } => {
                                if player != bot as u8 {
                                    assert!(code.is_none(), "seat {} was shown the other's draw", bot);
                                    checked += 1;
                                }
                            }
                            Obs::Act { player, key } => {
                                assert_ne!(player, bot as u8);
                                if let Some(action) = Action::decode(key) {
                                    assert_eq!(obs_key(&action), key, "an observation carried more than was seen");
                                }
                                checked += 1;
                            }
                        }
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    if acting {
                        let action = seats[bot].decide(&brain).unwrap();
                        done.push(Done {
                            id: ask.id,
                            action: Some(action),
                        });
                    } else {
                        seats[bot].watch(&brain);
                        done.push(Done {
                            id: ask.id,
                            action: None,
                        });
                    }
                }
                table.accept(bot, Reply { done, error: None }).unwrap();
            }
        }
        assert!(checked > 20, "the game was too short to have tested anything");
    }

    #[test]
    fn the_opponents_stand_in_stays_in_support() {
        play(1009, |sess| {
            let other = 1 - sess.seat;
            let stand_in = true_config(&sess.s, other, &sess.ctx);
            assert!(
                sess.bel[other as usize].index_of(&stand_in).is_some(),
                "the stand-in for seat {} left the belief",
                other
            );
        });
    }
}
