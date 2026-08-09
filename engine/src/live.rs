//! Human-vs-agent live play session (the browser UI's engine side).
//!
//! Wraps a `State` together with the two public beliefs and mirrors
//! `selfplay::play_game` decision for decision, except that the human's
//! decisions are handed back to Python instead of being sampled:
//!
//!   * chance nodes (round-start draws) update the drawing player's belief
//!     and resolve from the true bag — exactly as in self-play;
//!   * the agent's decisions solve the depth-limited subgame at the current
//!     PBS and act on the CFR *average* strategy — the configuration
//!     `eval_match` measures (full solve, average strategy, sample the true
//!     config's row);
//!   * the human's decisions are validated against the true state and their
//!     belief is updated on the **public observation** (a face-down play
//!     hides the coin), under a *uniform* behaviour model: every legal action
//!     of the human is treated as equally likely.
//!
//! The uniform model matters for soundness, not strength: the belief must
//! keep the true config in its support or the solver's config indexing breaks.
//! A behaviour model copied from the agent's own policy would drop configs
//! that only a human would reach ("the agent would never Pass the Royal
//! Coin"), so it risks silently discarding the real world. Uniform is the
//! model that cannot.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::actions::Action;
use crate::board::NONE;
use crate::py::{action_from_dict, action_to_dict, nets, state_to_dict};
use crate::rebel::*;
use crate::rng::Rng;
use crate::search::{node_actions, Cfg, Solver};
use crate::selfplay::effective_bag_count;
use crate::state::{Cont, State, BLACK, WHITE};
use crate::units::{def, index_of_id};

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn sample_row(rng: &mut Rng, row: &[f32]) -> usize {
    let w: Vec<f64> = row.iter().map(|&x| x.max(0.0) as f64).collect();
    if w.iter().sum::<f64>() > 0.0 {
        rng.weighted_index(&w)
    } else {
        rng.below(row.len().max(1))
    }
}

/// A live game against one trained agent. The human is always the other seat.
#[pyclass]
pub struct LiveGame {
    s: State,
    ctx: Ctx,
    bel: [Belief; 2],
    rng: Rng,
    /// Which seat the agent plays (0 = white, 1 = black).
    agent: u8,
    /// Weight slot the agent solves with.
    slot: usize,
    depth: usize,
    iters: usize,
    log: Vec<String>,
}

#[pymethods]
impl LiveGame {
    /// Construct from a draft dict (`{white_units, black_units, first_player}`,
    /// as `Game.new`) plus the agent's seat. Resolves the opening draws and any
    /// early agent decisions, so the first snapshot is already a human
    /// decision or a terminal state.
    #[new]
    #[pyo3(signature = (draft, agent, slot=0, depth=2, iters=16, seed=0))]
    fn new(
        draft: &Bound<'_, PyDict>,
        agent: u8,
        slot: usize,
        depth: usize,
        iters: usize,
        seed: u64,
    ) -> PyResult<LiveGame> {
        if agent != WHITE && agent != BLACK {
            return err("agent must be 0 (white) or 1 (black)");
        }
        let get_units = |key: &str| -> PyResult<Vec<u16>> {
            let v = draft.get_item(key)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("draft needs '{}'", key))
            })?;
            let ids: Vec<u16> = v.extract()?;
            if ids.len() != 4 {
                return err(format!("'{}' must have 4 unitTypeIds", key));
            }
            for &id in &ids {
                if index_of_id(id).is_none() {
                    return err(format!("unitTypeId {} is out of scope", id));
                }
            }
            Ok(ids)
        };
        let white = get_units("white_units")?;
        let black = get_units("black_units")?;
        let fp: String = draft
            .get_item("first_player")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("draft needs 'first_player'"))?
            .extract()?;
        let first = match fp.as_str() {
            "white" => WHITE,
            "black" => BLACK,
            _ => return err("first_player must be 'white' or 'black'"),
        };
        let s = State::from_draft(&white, &black, first);
        let ctx = Ctx::new(&s);
        // The engine starts everyone with an empty hand and an empty face-down
        // discard; every later belief is reached by the same filter the
        // self-play loop applies.
        let mut g = LiveGame {
            s,
            ctx,
            bel: [
                Belief::point(Config::default()),
                Belief::point(Config::default()),
            ],
            rng: Rng::new(if seed == 0 {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x9E3779B97F4A7C15)
            } else {
                seed
            }),
            agent,
            slot,
            depth,
            iters,
            log: Vec::new(),
        };
        g.auto_advance()?;
        Ok(g)
    }

    /// The full UI state: `state_dict` plus the event log, the seats, and the
    /// legal actions of whoever is to act.
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let obj = state_to_dict(py, &self.s)?;
        let d = obj.downcast::<PyDict>(py)?;
        let log = PyList::empty_bound(py);
        for l in &self.log {
            log.append(l)?;
        }
        d.set_item("log", log)?;
        d.set_item("human", 1 - self.agent)?;
        d.set_item("agent", self.agent)?;
        let acts = PyList::empty_bound(py);
        for a in self.s.legal_actions() {
            acts.append(action_to_dict(py, &self.s, &a)?)?;
        }
        d.set_item("actions", acts)?;
        Ok(d.into())
    }

    /// Apply the human's action, then resolve draws and the agent's replies
    /// until the next decision belongs to the human (or the game is over).
    /// Returns the snapshot.
    fn human_move(&mut self, py: Python<'_>, action: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let a = action_from_dict(action)?;
        self.human_decide(a)?;
        self.auto_advance()?;
        self.snapshot(py)
    }
}

impl LiveGame {
    /// Validate the human's action, update their belief on the public
    /// observation, and apply the action to the true state.
    fn human_decide(&mut self, a: Action) -> PyResult<()> {
        if self.s.is_terminal() {
            return err("the game is over");
        }
        if self.s.is_chance() {
            return err("draws resolve automatically");
        }
        let player = 1 - self.agent;
        if self.s.to_act() != player {
            return err("it is not your turn");
        }
        let code = a.encode();
        if !self.s.legal_actions().iter().any(|x| x.encode() == code) {
            return err(format!("illegal action {}", a));
        }
        // Bayes update on the public observation, with a uniform behaviour
        // model for the human (see the module doc). `node_actions` gives the
        // per-config legal set exactly as the solver sees it.
        let cfgs = self.bel[player as usize].cfg.clone();
        let (acts, aslot, fdown) = node_actions(&self.s, player, &self.ctx, &cfgs);
        let na = acts.len();
        let obs = obs_key(&a);
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in cfgs.iter().enumerate() {
            let mut legal_n = 0usize;
            for k in 0..na {
                if action_legal(c, aslot[k]) {
                    legal_n += 1;
                }
            }
            if legal_n == 0 {
                continue;
            }
            let w = self.bel[player as usize].p[ci] / legal_n as f32;
            for k in 0..na {
                if !action_legal(c, aslot[k]) {
                    continue;
                }
                if obs_key(&acts[k]) != obs {
                    continue;
                }
                if let Some(n) = advance_config(c, aslot[k], fdown[k]) {
                    pairs.push((n, w));
                }
            }
        }
        self.bel[player as usize] = Belief::from_pairs(pairs);
        self.s.apply_inplace(a);
        self.log.push(format!("You: {}", a));
        Ok(())
    }

    /// Solve the subgame at the current PBS and act with the CFR average
    /// strategy, exactly as evaluation does in `selfplay::play_game`.
    fn agent_decide(&mut self) -> PyResult<()> {
        let player = self.agent;
        let cfgs = self.bel[player as usize].cfg.clone();
        let truth = true_config(&self.s, player, &self.ctx);
        let true_ci = self.bel[player as usize].index_of(&truth).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "agent belief dropped the true config — this should not happen",
            )
        })?;
        let scfg = Cfg {
            depth: self.depth,
            iters: self.iters,
            snapshots: false,
            ..Default::default()
        };
        // Work on copies so the solver's borrows do not tie up `self`; the
        // state is `Copy` and the belief only needs a clone per decision.
        let ctx = self.ctx;
        let mut s = self.s;
        let mut bel = self.bel.clone();
        let guard = nets().read().unwrap();
        let mut sv = Solver::new(&s, ctx, &guard[self.slot], scfg, bel.clone());
        if sv.capped() {
            // Pathological root: play uniformly instead of solving. The
            // belief update below uses the true state's actions, which stays
            // sound for the human opponent.
            let acts = self.s.legal_actions();
            let chosen = self.rng.below(acts.len());
            let act = acts[chosen];
            let label = format!("{}", act);
            self.s.apply_inplace(act);
            self.log.push(format!("Agent: {}", label));
            return Ok(());
        }
        sv.multistep(self.iters);
        let nid = 0usize;
        let na = sv.nodes[nid].na();
        let mut probs = vec![0.0f32; cfgs.len() * na];
        for ci in 0..cfgs.len() {
            probs[ci * na..(ci + 1) * na].copy_from_slice(sv.average_strategy(nid, ci));
        }
        let chosen = sample_row(&mut self.rng, &probs[true_ci * na..(true_ci + 1) * na]);
        let act = sv.nodes[nid].acts[chosen];
        let label = format!("{}", act);
        // Belief update on the public observation, weighted by the policy the
        // solver actually acts on (same formula as self-play).
        let obs = obs_key(&act);
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in cfgs.iter().enumerate() {
            for a in 0..na {
                if !sv.nodes[nid].legal[ci * na + a] || obs_key(&sv.nodes[nid].acts[a]) != obs {
                    continue;
                }
                if let Some(n) = advance_config(c, sv.nodes[nid].aslot[a], sv.nodes[nid].fdown[a]) {
                    pairs.push((n, bel[player as usize].p[ci] * probs[ci * na + a]));
                }
            }
        }
        drop(sv);
        drop(guard);
        bel[player as usize] = Belief::from_pairs(pairs);
        s.apply_inplace(act);
        self.s = s;
        self.bel = bel;
        self.log.push(format!("Agent: {}", label));
        Ok(())
    }

    /// Resolve pending draws and play the agent's decisions until the next
    /// decision belongs to the human, or the game ends.
    fn auto_advance(&mut self) -> PyResult<()> {
        loop {
            if self.s.is_terminal() {
                return Ok(());
            }
            if self.s.is_chance() {
                let player = self.s.to_act();
                // The belief update reads the *pre-draw* reserve and face-up
                // counts, exactly like the self-play loop.
                let res = reserve(&self.s, player, &self.ctx);
                let fu = faceup_counts(&self.s, player, &self.ctx);
                let wp = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
                self.bel[player as usize] =
                    belief_after_draw(&self.bel[player as usize], &res, &fu, wp);
                let acts = self.s.legal_actions();
                let mut w: Vec<f64> = Vec::with_capacity(acts.len());
                let mut any = false;
                for a in &acts {
                    let c = match a {
                        Action::DrawCoin { unit } if *unit != NONE => {
                            effective_bag_count(&self.s, player, *unit)
                        }
                        _ => 1,
                    };
                    any |= c > 0;
                    w.push(c as f64);
                }
                if !any {
                    w.iter_mut().for_each(|x| *x = 1.0);
                }
                let ai = self.rng.weighted_index(&w);
                let drawn = acts[ai];
                self.s.apply_inplace(drawn);
                self.log.push(if player == 1 - self.agent {
                    if let Action::DrawCoin { unit } = drawn {
                        format!("You draw a {}", def(unit).name)
                    } else {
                        "You draw".to_string()
                    }
                } else {
                    "Agent draws (hidden)".to_string()
                });
                continue;
            }
            if self.s.to_act() == self.agent {
                self.agent_decide()?;
                continue;
            }
            return Ok(());
        }
    }
}
