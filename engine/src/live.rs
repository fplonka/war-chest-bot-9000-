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

struct RootPolicy {
    acts: Vec<Action>,
    aslot: Vec<i8>,
    fdown: Vec<bool>,
    legal_off: Vec<u32>,
    legal_action: Vec<u32>,
    probs: Vec<f32>,
}

impl RootPolicy {
    fn from_cpu(sv: &mut Solver<'_>, iters: usize, configs: usize) -> RootPolicy {
        sv.multistep(iters);
        let node = &sv.nodes[0];
        let mut probs = vec![0.0; node.legal_action.len()];
        for config in 0..configs {
            let row = node.legal_row(config);
            probs[row].copy_from_slice(sv.average_strategy(0, config));
        }
        RootPolicy {
            acts: node.acts.clone(),
            aslot: node.aslot.clone(),
            fdown: node.fdown.clone(),
            legal_off: node.legal_off.clone(),
            legal_action: node.legal_action.clone(),
            probs,
        }
    }

    #[cfg(feature = "gpu")]
    fn from_gpu(
        tree: crate::serialize::WalkTree,
        result: crate::gpu::SolveResult,
        player: usize,
    ) -> RootPolicy {
        let actions = tree.action_range(0);
        let row0 = tree.legal_row_of[0] as usize;
        let configs = tree.supports[0][player].len();
        let cell0 = tree.legal_off[row0];
        let cell1 = tree.legal_off[row0 + configs];
        RootPolicy {
            acts: tree.actions[actions.clone()].to_vec(),
            aslot: tree.aslot[actions.clone()].to_vec(),
            fdown: tree.fdown[actions].to_vec(),
            legal_off: tree.legal_off[row0..=row0 + configs]
                .iter()
                .map(|&offset| offset - cell0)
                .collect(),
            legal_action: tree.legal_action[cell0 as usize..cell1 as usize].to_vec(),
            probs: result.strategy[cell0 as usize..cell1 as usize].to_vec(),
        }
    }

    fn legal_row(&self, config: usize) -> std::ops::Range<usize> {
        self.legal_off[config] as usize..self.legal_off[config + 1] as usize
    }
}

type BeliefRow = (Vec<u8>, Vec<u8>, i16, f32);

fn belief_rows(b: &Belief) -> Vec<BeliefRow> {
    b.cfg
        .iter()
        .zip(&b.p)
        .map(|(c, &p)| {
            (
                c.hand.to_vec(),
                c.fd.to_vec(),
                c.inflight.map_or(-1, i16::from),
                p,
            )
        })
        .collect()
}

fn belief_from_rows(rows: Vec<BeliefRow>) -> PyResult<Belief> {
    let mut cfg = Vec::with_capacity(rows.len());
    let mut p = Vec::with_capacity(rows.len());
    for (hand, fd, inflight, weight) in rows {
        let hand: [u8; NSLOT] = hand
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("bad hand width"))?;
        let fd: [u8; NSLOT] = fd
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("bad facedown width"))?;
        let inflight = if inflight == -1 {
            None
        } else if (0..NSLOT as i16).contains(&inflight) {
            Some(inflight as u8)
        } else {
            return err("bad in-flight slot");
        };
        if !weight.is_finite() || weight < 0.0 {
            return err("bad belief weight");
        }
        cfg.push(Config { hand, fd, inflight });
        p.push(weight);
    }
    if cfg.is_empty() {
        return err("empty belief");
    }
    Ok(Belief { cfg, p })
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
    relay: bool,
}

#[pymethods]
impl LiveGame {
    /// Construct from a draft dict (`{white_units, black_units, first_player}`)
    /// plus the agent's seat. Normal play resolves through the first human
    /// decision. Relay play exposes one transition at a time so evaluators
    /// built from different engine revisions can share an exact game.
    #[new]
    #[pyo3(signature = (draft, agent, slot=0, depth=2, iters=16, seed=0, relay=false))]
    fn new(
        draft: &Bound<'_, PyDict>,
        agent: u8,
        slot: usize,
        depth: usize,
        iters: usize,
        seed: u64,
        relay: bool,
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
            relay,
        };
        if !relay {
            g.auto_advance(draft.py())?;
        }
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
        self.auto_advance(py)?;
        self.snapshot(py)
    }

    /// Resolve exactly one chance transition in relay mode.
    fn relay_chance(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.require_relay()?;
        if !self.s.is_chance() {
            return err("the relay is not at chance");
        }
        self.chance_decide()?;
        self.snapshot(py)
    }

    /// Let this relay's agent make exactly one decision.
    fn relay_agent_move(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.require_relay()?;
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() != self.agent {
            return err("the relay is not at its agent decision");
        }
        let action = self.agent_decide(py)?;
        let out = PyDict::new_bound(py);
        out.set_item("code", action.encode())?;
        out.set_item("belief", belief_rows(&self.bel[self.agent as usize]))?;
        Ok(out.into())
    }

    /// Apply the other engine's decision and its exact posterior belief.
    fn relay_apply(
        &mut self,
        py: Python<'_>,
        action: &Bound<'_, PyDict>,
        belief: Vec<BeliefRow>,
    ) -> PyResult<PyObject> {
        self.require_relay()?;
        if self.s.is_terminal() || self.s.is_chance() || self.s.to_act() == self.agent {
            return err("the relay is not at its peer decision");
        }
        let action = action_from_dict(action)?;
        if !self
            .s
            .legal_actions()
            .iter()
            .any(|candidate| candidate.encode() == action.encode())
        {
            return err(format!("illegal relay action {}", action));
        }
        let player = self.s.to_act() as usize;
        self.bel[player] = belief_from_rows(belief)?;
        self.rng.next_u64();
        self.s.apply_inplace(action);
        self.log.push(format!("Peer: {}", action));
        self.snapshot(py)
    }
}

impl LiveGame {
    fn require_relay(&self) -> PyResult<()> {
        if self.relay {
            Ok(())
        } else {
            err("relay mode was not enabled")
        }
    }

    fn apply_uniform_decision(&mut self, player: u8, a: Action, label: &str) {
        let cfgs = self.bel[player as usize].cfg.clone();
        let (acts, aslot, fdown) = node_actions(&self.s, player, &self.ctx, &cfgs);
        let obs = obs_key(&a);
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in cfgs.iter().enumerate() {
            let legal_n = aslot.iter().filter(|&&slot| action_legal(c, slot)).count();
            if legal_n == 0 {
                continue;
            }
            let weight = self.bel[player as usize].p[ci] / legal_n as f32;
            for k in 0..acts.len() {
                if action_legal(c, aslot[k]) && obs_key(&acts[k]) == obs {
                    if let Some(next) = advance_config(c, aslot[k], fdown[k]) {
                        pairs.push((next, weight));
                    }
                }
            }
        }
        self.bel[player as usize] = Belief::from_pairs(pairs);
        self.s.apply_inplace(a);
        self.log.push(format!("{label}: {a}"));
    }

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
        // Bayes update on the public observation under the uniform human model.
        self.apply_uniform_decision(player, a, "You");
        Ok(())
    }

    /// Solve the subgame at the current PBS and act with the CFR average
    /// strategy, exactly as evaluation does in `selfplay::play_game`.
    fn agent_decide(&mut self, py: Python<'_>) -> PyResult<Action> {
        let player = self.agent;
        let cfgs = self.bel[player as usize].cfg.clone();
        let truth = true_config(&self.s, player, &self.ctx);
        let true_ci = self.bel[player as usize].index_of(&truth).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "agent belief dropped the true config — this should not happen",
            )
        })?;
        #[cfg(feature = "gpu")]
        let client = if self.relay {
            crate::py::gpu_clients().lock().first().cloned()
        } else {
            None
        };
        #[cfg(feature = "gpu")]
        let gpu_build = client.is_some();
        #[cfg(not(feature = "gpu"))]
        let gpu_build = false;
        let scfg = Cfg {
            depth: self.depth,
            iters: self.iters,
            snapshots: false,
            node_cap: 200_000,
            gpu_build,
            ..Default::default()
        };
        let mut s = self.s;
        let mut bel = self.bel.clone();
        let guard = nets().read();
        let mut sv = Solver::new(&s, self.ctx, &guard[self.slot], scfg, bel.clone());
        if sv.capped() {
            let acts = self.s.legal_actions();
            let act = acts[self.rng.below(acts.len())];
            self.apply_uniform_decision(player, act, "Agent");
            return Ok(act);
        }
        #[cfg(feature = "gpu")]
        let policy = if let Some(client) = client {
            let roots = vec![[bel[0].p.clone(), bel[1].p.clone()]];
            let (job, tree) = crate::serialize::PackedJob::from_solver_with_walk(&sv, &roots);
            // The service batches every other relay game that submits while
            // this one waits, so the GIL must be released: a hundred games in
            // flight is what turns a relayed ladder from serial into a wave.
            match py.allow_threads(|| client.submit(job).and_then(|handle| handle.wait())) {
                Ok(result) => RootPolicy::from_gpu(tree, result, player as usize),
                Err(error) => {
                    eprintln!("GPU relay solve failed; retrying on CPU: {error}");
                    let mut cpu = Solver::new(
                        &s,
                        self.ctx,
                        &guard[self.slot],
                        Cfg {
                            gpu_build: false,
                            ..scfg
                        },
                        bel.clone(),
                    );
                    RootPolicy::from_cpu(&mut cpu, self.iters, cfgs.len())
                }
            }
        } else {
            RootPolicy::from_cpu(&mut sv, self.iters, cfgs.len())
        };
        #[cfg(not(feature = "gpu"))]
        let policy = RootPolicy::from_cpu(&mut sv, self.iters, cfgs.len());

        let true_row = policy.legal_row(true_ci);
        let chosen_cell = true_row.start + sample_row(&mut self.rng, &policy.probs[true_row]);
        let chosen = policy.legal_action[chosen_cell] as usize;
        let act = policy.acts[chosen];
        let obs = obs_key(&act);
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in cfgs.iter().enumerate() {
            for cell in policy.legal_row(ci) {
                let action = policy.legal_action[cell] as usize;
                if obs_key(&policy.acts[action]) != obs {
                    continue;
                }
                if let Some(next) = advance_config(c, policy.aslot[action], policy.fdown[action]) {
                    pairs.push((next, bel[player as usize].p[ci] * policy.probs[cell]));
                }
            }
        }
        bel[player as usize] = Belief::from_pairs(pairs);
        s.apply_inplace(act);
        self.s = s;
        self.bel = bel;
        self.log.push(format!("Agent: {}", act));
        Ok(act)
    }

    fn chance_decide(&mut self) -> PyResult<Action> {
        let player = self.s.to_act();
        let res = reserve(&self.s, player, &self.ctx);
        let fu = faceup_counts(&self.s, player, &self.ctx);
        let wp = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
        self.bel[player as usize] = belief_after_draw(&self.bel[player as usize], &res, &fu, wp);
        let acts = self.s.legal_actions();
        let mut weights: Vec<f64> = Vec::with_capacity(acts.len());
        let mut any = false;
        for action in &acts {
            let count = match action {
                Action::DrawCoin { unit } if *unit != NONE => {
                    effective_bag_count(&self.s, player, *unit)
                }
                _ => 1,
            };
            any |= count > 0;
            weights.push(count as f64);
        }
        if !any {
            weights.fill(1.0);
        }
        let drawn = acts[self.rng.weighted_index(&weights)];
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
        Ok(drawn)
    }

    /// Resolve pending draws and play the agent's decisions until the next
    /// decision belongs to the human, or the game ends.
    fn auto_advance(&mut self, py: Python<'_>) -> PyResult<()> {
        loop {
            if self.s.is_terminal() {
                return Ok(());
            }
            if self.s.is_chance() {
                self.chance_decide()?;
                continue;
            }
            if self.s.to_act() == self.agent {
                self.agent_decide(py)?;
                continue;
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relay_game(agent: u8, seed: u64) -> LiveGame {
        let state = State::from_draft(&[17, 12, 4, 9], &[1, 3, 8, 16], WHITE);
        LiveGame {
            s: state,
            ctx: Ctx::new(&state),
            bel: [
                Belief::point(Config::default()),
                Belief::point(Config::default()),
            ],
            rng: Rng::new(seed),
            agent,
            slot: 0,
            depth: 2,
            iters: 16,
            log: Vec::new(),
            relay: true,
        }
    }

    #[test]
    fn opposite_relays_resolve_identical_chance() {
        let mut white = relay_game(WHITE, 17);
        let mut black = relay_game(BLACK, 17);
        while white.s.is_chance() {
            assert!(black.s.is_chance());
            assert_eq!(
                white.chance_decide().unwrap().encode(),
                black.chance_decide().unwrap().encode()
            );
            assert_eq!(white.s, black.s);
            assert_eq!(belief_rows(&white.bel[0]), belief_rows(&black.bel[0]));
            assert_eq!(belief_rows(&white.bel[1]), belief_rows(&black.bel[1]));
        }
        assert_eq!(white.rng.0, black.rng.0);
    }

    #[test]
    fn uniform_fallback_preserves_the_true_config() {
        let mut game = relay_game(WHITE, 29);
        while game.s.is_chance() {
            game.chance_decide().unwrap();
        }
        let player = game.s.to_act();
        let action = game.s.legal_actions()[0];
        game.apply_uniform_decision(player, action, "Agent");
        let truth = true_config(&game.s, player, &game.ctx);
        assert!(game.bel[player as usize].index_of(&truth).is_some());
    }

    #[test]
    fn relay_belief_wire_round_trips() {
        let game = relay_game(WHITE, 23);
        for belief in &game.bel {
            let rows = belief_rows(belief);
            let decoded = belief_from_rows(rows.clone()).unwrap();
            assert_eq!(belief_rows(&decoded), rows);
        }
    }
}
