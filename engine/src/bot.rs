//! One bot's side of an arena game.
//!
//! A bot is told only what its seat may know: its own draws, and the public
//! observation of everything else. From that it keeps a public belief over
//! each player's private config — the pair of ranges a subgame solver takes as
//! its root — and its own true config, which it knows exactly because it drew
//! the coins itself.
//!
//! It therefore has to model its opponent. A belief moves on an opponent's
//! action only under some assumption about how the opponent chooses, and the
//! bot's assumption is that the opponent thinks like it does: it solves the
//! opponent's node with its own network and filters the result on what it saw.
//! That is the same update self-play makes, and against a copy of itself it
//! reproduces self-play exactly. Against a different opponent it is a model,
//! and being wrong about the opponent is part of what the ladder measures.
//!
//! The bot's `State` carries the opponent's private zones because the rules
//! engine has nowhere else to put them, but nothing reads them: the solver
//! overwrites them from the belief for every config it evaluates, and the only
//! public quantity taken from them — the reserve — is invariant across the
//! support. `resync` keeps them at some member of the support so the
//! invariant is easy to state and cheap to check.

use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::policy::{self, NodePolicy};
use crate::rebel::{
    belief_after_draw, faceup_counts, obs_key, reserve, set_config, true_config, Belief, Config,
    Ctx,
};
use crate::rng::Rng;
use crate::search::{Cfg, Nets, Solver};
use crate::state::{Cont, State};

/// How a bot picks its move.
#[derive(Clone, Copy, Debug)]
pub enum Mind {
    /// Solve the depth-limited subgame, act on the CFR average strategy.
    Rebel,
    /// One-ply search on the handcrafted static evaluation.
    Greedy { temp: f32 },
    /// Uniform over legal actions.
    Random,
    /// The exploitability probe. Plays ReBeL, but takes its opponent's
    /// strategy from the referee instead of guessing at it, and answers with
    /// the best reply rather than an equilibrium one.
    ///
    /// What it wins above a plain bot is a lower bound on how exploitable the
    /// opponent is: a real best response would take at least as much. It is a
    /// measuring instrument, not a player, and never belongs in a ladder.
    Lbr,
}

/// Everything a bot brings to every game it plays: how it thinks, what it
/// thinks with, and the device it thinks on. Shared across all live games, and
/// read-only, so games can be stepped in parallel.
pub struct Brain {
    pub mind: Mind,
    pub nets: Nets,
    pub cfg: Cfg,
    #[cfg(feature = "gpu")]
    pub gpu: Option<crate::gpu::GpuClient>,
}

impl Brain {
    /// The acting player's policy at this node, over the configs they might
    /// hold. Used both to choose the bot's own move and to model its
    /// opponent's, which is why it takes the player rather than assuming the
    /// bot's own seat.
    pub fn policy(&self, s: &State, ctx: &Ctx, player: u8, bel: &[Belief; 2]) -> NodePolicy {
        let cfgs = &bel[player as usize].cfg;
        match self.mind {
            Mind::Greedy { temp } => return policy::greedy(s, ctx, player, cfgs, temp),
            Mind::Random => return policy::uniform(s, ctx, player, cfgs),
            Mind::Rebel | Mind::Lbr => {}
        }
        #[cfg(feature = "gpu")]
        let cfg = Cfg {
            snapshots: false,
            gpu_build: self.gpu.is_some(),
            ..self.cfg
        };
        #[cfg(not(feature = "gpu"))]
        let cfg = Cfg {
            snapshots: false,
            ..self.cfg
        };
        let mut sv = Solver::new(s, *ctx, &self.nets, cfg, bel.clone());
        // A subgame too large to build is played uniformly. The tail of the
        // tree-size distribution is fat enough that an unbounded build would
        // stall a whole batch on one decision.
        if sv.capped() {
            return policy::uniform(s, ctx, player, cfgs);
        }
        #[cfg(feature = "gpu")]
        if let Some(client) = self.gpu.as_ref() {
            let roots = vec![[bel[0].p.clone(), bel[1].p.clone()]];
            let (job, tree) = crate::serialize::PackedJob::from_solver_with_walk(&sv, &roots);
            match client.submit(job).and_then(|handle| handle.wait()) {
                Ok(result) => return policy::from_wave(&tree, &result, player as usize),
                Err(error) => eprintln!("wave solve failed, falling back to the CPU: {error}"),
            }
            sv = Solver::new(
                s,
                *ctx,
                &self.nets,
                Cfg {
                    gpu_build: false,
                    ..cfg
                },
                bel.clone(),
            );
        }
        policy::solved(&mut sv, cfg.iters, cfgs.len())
    }
}

/// One game, from one seat.
pub struct Session {
    s: State,
    ctx: Ctx,
    bel: [Belief; 2],
    seat: u8,
    rng: Rng,
    /// The opponent's policy at the node they are sitting at, solved before
    /// they moved. See `watch`.
    modelled: Option<NodePolicy>,
}

impl Session {
    pub fn new(draft: &Draft, seat: u8, seed: u64) -> Result<Session, String> {
        let s = draft.state()?;
        let ctx = Ctx::new(&s);
        // Both players start with an empty hand and nothing discarded, so the
        // opening belief is a point mass and every later one is reached by the
        // same filter.
        Ok(Session {
            s,
            ctx,
            bel: [
                Belief::point(Config::default()),
                Belief::point(Config::default()),
            ],
            seat,
            rng: Rng::new(seed),
            modelled: None,
        })
    }

    /// Begin at a benchmark position rather than at the start of a game.
    ///
    /// The ranges come with it, because the question and the proof behind it
    /// have to be about the same ranges. This is the one place a belief
    /// arrives from outside; in a game a bot builds its own, and nothing else
    /// would be honest.
    pub fn at(state: State, belief: [Belief; 2], seat: u8, seed: u64) -> Result<Session, String> {
        if belief[0].is_empty() || belief[1].is_empty() {
            return Err("a position needs a range for each seat".into());
        }
        Ok(Session {
            s: state,
            ctx: Ctx::new(&state),
            bel: belief,
            seat,
            rng: Rng::new(seed),
            modelled: None,
        })
    }

    /// Put a member of the belief's support into the opponent's private zones,
    /// so the position stays a legal one to reason from.
    fn resync(&mut self, player: u8) {
        let stand_in = self.bel[player as usize].cfg[0];
        set_config(&mut self.s, player, &self.ctx, &stand_in);
    }

    /// Model the opponent at the node they are about to act in.
    ///
    /// This is half of a bot's work and none of its move: the model is a solve
    /// of the opponent's current position, which does not depend on the move
    /// they are about to make. Doing it here, while they are still thinking,
    /// is what keeps both bots' devices busy at once.
    pub fn watch(&mut self, brain: &Brain) {
        if !self.s.is_terminal() && !self.s.is_chance() && self.s.to_act() != self.seat {
            let player = self.s.to_act();
            self.modelled = Some(brain.policy(&self.s, &self.ctx, player, &self.bel));
        }
    }

    pub fn observe(&mut self, obs: &Obs, brain: &Brain) -> Result<(), String> {
        match *obs {
            Obs::Draw { player, code } => self.drew(player, code),
            Obs::Act {
                player,
                key,
                ref policy,
            } => self.acted(player, key, policy.as_deref(), brain),
        }
    }

    /// A coin left `player`'s bag. The belief update is public arithmetic over
    /// the reserve, so both seats make it; only the drawer learns which coin.
    fn drew(&mut self, player: u8, code: Option<u32>) -> Result<(), String> {
        if !self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {} is not drawing", player));
        }
        self.modelled = None;
        let res = reserve(&self.s, player, &self.ctx);
        let faceup = faceup_counts(&self.s, player, &self.ctx);
        let in_flight = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
        self.bel[player as usize] =
            belief_after_draw(&self.bel[player as usize], &res, &faceup, in_flight);
        let drawn = match code {
            Some(code) => {
                Action::decode(code).ok_or_else(|| format!("draw {} does not decode", code))?
            }
            // Someone else's draw: any coin their stand-in config could have
            // taken moves the public counts the same way.
            None => self.s.legal_actions()[0],
        };
        self.s.apply_inplace(drawn);
        if player != self.seat {
            self.resync(player);
        }
        Ok(())
    }

    /// The opponent was seen to make an observation. Model their strategy at
    /// the node, filter it on what was seen, and step the public position with
    /// a private action consistent with it.
    fn acted(
        &mut self,
        player: u8,
        key: u32,
        revealed: Option<&[f32]>,
        brain: &Brain,
    ) -> Result<(), String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {} is not to act", player));
        }
        if player == self.seat {
            return Err("a bot is told its own moves back".into());
        }
        // A probe is handed the strategy that was actually played, so its
        // belief becomes the truth rather than a model of it.
        let mut np = match self.modelled.take() {
            Some(np) => np,
            None => brain.policy(&self.s, &self.ctx, player, &self.bel),
        };
        if let Some(revealed) = revealed {
            if revealed.len() != np.probs.len() {
                return Err(format!(
                    "revealed strategy has {} cells, this node has {}",
                    revealed.len(),
                    np.probs.len()
                ));
            }
            np.probs.copy_from_slice(revealed);
        }
        let np = np;
        let (ci, cell) = np
            .cell_for(key)
            .ok_or_else(|| format!("observation {} is unreachable from this belief", key))?;
        let posterior = np.posterior(&self.bel[player as usize], key);
        set_config(
            &mut self.s,
            player,
            &self.ctx,
            &self.bel[player as usize].cfg[ci],
        );
        self.s.apply_inplace(np.acts[np.action_at(cell)]);
        self.bel[player as usize] = posterior;
        self.resync(player);
        Ok(())
    }

    /// Choose a move, update what the opponent now knows about this seat, and
    /// return the action's encoding — with the whole strategy it sampled from
    /// when `report` asks for it.
    pub fn decide(
        &mut self,
        brain: &Brain,
        report: bool,
    ) -> Result<(u32, Option<Vec<f32>>), String> {
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != self.seat {
            return Err("the bot was asked to move out of turn".into());
        }
        self.modelled = None;
        let truth = true_config(&self.s, self.seat, &self.ctx);
        let ci = self.bel[self.seat as usize]
            .index_of(&truth)
            .ok_or("the belief filter dropped this seat's own config")?;
        let np = brain.policy(&self.s, &self.ctx, self.seat, &self.bel);
        if np.row(ci).is_empty() {
            // The node's actions are enumerated over the public reserve and
            // then filtered per hand. A hand with no row cannot move, which
            // means the position and the range disagree about what is possible.
            return Err(format!(
                "this seat's hand has no legal action here ({} actions at the node)",
                np.acts.len()
            ));
        }
        // A probe answers with the best reply it can see, not a mixed one: the
        // question it exists to ask is how much the opponent's strategy gives
        // away, and an equilibrium answer would not collect it.
        let cell = match brain.mind {
            Mind::Lbr => np.best(ci),
            _ => np.sample(&mut self.rng, ci),
        };
        let action = np.acts[np.action_at(cell)];
        let policy = report.then(|| np.probs.clone());
        self.bel[self.seat as usize] =
            np.posterior(&self.bel[self.seat as usize], obs_key(&action));
        self.s.apply_inplace(action);
        Ok((action.encode(), policy))
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
            nets: Nets::default(),
            cfg: Cfg::default(),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Play one refereed game to the end, checking `inspect` at every decision
    /// the seat about to move takes.
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
                        policy: None,
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
                        action: Some(seats[bot].decide(&brain, false).unwrap().0),
                        policy: None,
                    });
                }
                for entry in &done {
                    waiting[bot].remove(&entry.id);
                }
                table.accept(bot, Reply { done, error: None }).unwrap();
            }
        }
    }

    /// The referee and two blind seats play a whole game, neither seat ever
    /// having been handed the other's private state.
    #[test]
    fn a_refereed_game_runs_to_the_end() {
        assert!(play(4242, |_| {}).abs() <= 1.0);
    }

    /// A seat's own config is exact and stays in its own belief's support: the
    /// solver indexes its strategy rows by position in that support, so losing
    /// it would silently play someone else's hand.
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

    /// A benchmark position survives the wire and seats a bot exactly where it
    /// was. Everything the tablebase claims rests on this: a question is only a
    /// question if the bot answering it stands in the position that was proven,
    /// holding the ranges it was proven against.
    #[test]
    fn a_position_survives_the_wire() {
        let brain = brain();
        let mut table = Table::new();
        table.start(1, &draft(), [0, 1], 6001).unwrap();
        let mut seats = [
            Session::new(&draft(), 0, 41).unwrap(),
            Session::new(&draft(), 1, 42).unwrap(),
        ];
        // Play a while, so the ranges are something other than a point mass.
        for _ in 0..14 {
            table.settle();
            if !table.reap().is_empty() {
                break;
            }
            for bot in 0..2 {
                let request = table.request(bot);
                let mut done = Vec::new();
                for ask in request.watch {
                    for obs in &ask.obs {
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    seats[bot].watch(&brain);
                    done.push(Done { id: ask.id, action: None, policy: None });
                }
                for ask in request.go {
                    for obs in &ask.obs {
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    let (action, _) = seats[bot].decide(&brain, false).unwrap();
                    done.push(Done { id: ask.id, action: Some(action), policy: None });
                }
                table.accept(bot, Reply { done, error: None }).unwrap();
            }
        }
        let seat = seats[0].seat;
        let wire = crate::arena::encode_position(&seats[0].s, &seats[0].bel);
        let (state, belief) = crate::arena::decode_position(&wire).unwrap();
        let placed = Session::at(state, belief, seat, 7).unwrap();
        assert_eq!(placed.s, seats[0].s, "the position moved");
        for p in 0..2 {
            assert_eq!(placed.bel[p].cfg, seats[0].bel[p].cfg, "range {} moved", p);
        }
    }

    /// The referee never tells a seat what the other seat drew, and never
    /// names the coin behind a face-down play. This is the property the whole
    /// arrangement rests on: if it failed, every rating taken here would be a
    /// rating of a bot that could see its opponent's hand.
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
                                    assert!(
                                        code.is_none(),
                                        "seat {} was shown the other's draw",
                                        bot
                                    );
                                    checked += 1;
                                }
                            }
                            Obs::Act { player, key, .. } => {
                                // A play that hides its coin must not arrive
                                // with one: the key has to differ from the
                                // action that produced it.
                                assert_ne!(player, bot as u8);
                                if let Some(action) = Action::decode(key) {
                                    assert_eq!(
                                        obs_key(&action),
                                        key,
                                        "an observation carried more than was seen"
                                    );
                                }
                                checked += 1;
                            }
                        }
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    if acting {
                        let (action, _) = seats[bot].decide(&brain, false).unwrap();
                        done.push(Done {
                            id: ask.id,
                            action: Some(action),
                            policy: None,
                        });
                    } else {
                        seats[bot].watch(&brain);
                        done.push(Done {
                            id: ask.id,
                            action: None,
                            policy: None,
                        });
                    }
                }
                table.accept(bot, Reply { done, error: None }).unwrap();
            }
        }
        assert!(
            checked > 20,
            "the game was too short to have tested anything"
        );
    }

    /// The stand-in kept in the opponent's private zones is always a member of
    /// the belief over them, so the position is one the rules engine can
    /// reason from and the public reserve is right.
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
