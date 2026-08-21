//! A decision node's policy, and the belief filter that follows from it.
//!
//! Every agent answers the same question at a decision node: for each private
//! config the acting player might hold, how likely is each of that config's
//! legal actions? `NodePolicy` is that answer. The solver, the one-ply greedy
//! reference and uniform random differ only in how they fill `probs`.
//!
//! `posterior` is what the rest of the game does with it. An action is
//! observed by its *public* projection — a face-down play hides the coin
//! behind it — so the belief over the actor's private config sums over every
//! private action that could have produced the same observation, weighted by
//! how likely the actor was to take it. That is the only place a policy leaks
//! into what anyone else knows, and it is why an agent needs a model of its
//! opponent as well as of itself.

use std::ops::Range;

use crate::actions::Action;
use crate::pbs::{action_legal, advance_config, obs_key, set_config, Belief, Config, Ctx};
use crate::rng::Rng;
use crate::search::{node_actions, Solver};
use crate::state::State;

/// Private actions plus one probability per legal config/action cell, in
/// config-major CSR order.
pub struct NodePolicy {
    /// The node's distinct actions, deduplicated by encoding.
    pub acts: Vec<Action>,
    /// The coin slot each action spends, or -1 if it spends none.
    pub aslot: Vec<i8>,
    /// Whether each action's coin goes face down.
    pub fdown: Vec<bool>,
    legal_off: Vec<u32>,
    legal_action: Vec<u32>,
    pub probs: Vec<f32>,
}

impl NodePolicy {
    /// The node's shape: which actions each config may legally take. `probs`
    /// comes back zeroed for an agent to fill.
    pub fn frame(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
        let (acts, aslot, fdown) = node_actions(s, player, ctx, cfgs);
        let na = acts.len();
        let mut legal_off = Vec::with_capacity(cfgs.len() + 1);
        let mut legal_action = Vec::new();
        legal_off.push(0);
        for c in cfgs {
            for a in 0..na {
                if action_legal(c, aslot[a]) {
                    legal_action.push(a as u32);
                }
            }
            legal_off.push(legal_action.len() as u32);
        }
        let cells = legal_action.len();
        NodePolicy {
            acts,
            aslot,
            fdown,
            legal_off,
            legal_action,
            probs: vec![0.0; cells],
        }
    }

    /// The cells belonging to config `c`.
    #[inline]
    pub fn row(&self, c: usize) -> Range<usize> {
        self.legal_off[c] as usize..self.legal_off[c + 1] as usize
    }

    /// The action a cell selects, as an index into `acts`.
    #[inline]
    pub fn action_at(&self, cell: usize) -> usize {
        self.legal_action[cell] as usize
    }

    /// Draw one cell from config `c`'s row. A row whose probabilities have all
    /// underflowed is played uniformly rather than dropped.
    pub fn sample(&self, rng: &mut Rng, c: usize) -> usize {
        let row = self.row(c);
        let weights = &self.probs[row.clone()];
        let total: f64 = weights.iter().map(|&x| x.max(0.0) as f64).sum();
        if total == 0.0 {
            return row.start + rng.below(row.len().max(1));
        }
        let mut needle = rng.unit_f64() * total;
        for (i, &weight) in weights.iter().enumerate() {
            needle -= weight.max(0.0) as f64;
            if needle < 0.0 {
                return row.start + i;
            }
        }
        row.end - 1
    }


    /// The belief over the actor's private config after they were seen to make
    /// the observation `obs`. `prior` must be the belief this node was framed
    /// with.
    pub fn posterior(&self, prior: &Belief, obs: u32) -> Belief {
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in prior.cfg.iter().enumerate() {
            for cell in self.row(ci) {
                let a = self.action_at(cell);
                if obs_key(&self.acts[a]) != obs {
                    continue;
                }
                if let Some(next) = advance_config(c, self.aslot[a], self.fdown[a]) {
                    pairs.push((next, prior.p[ci] * self.probs[cell]));
                }
            }
        }
        Belief::from_pairs(pairs)
    }

    /// Some cell whose action carries the observation `obs`, if any config in
    /// the framing belief could have produced it. An observer that holds no
    /// true private state for the actor uses this to pick a stand-in.
    pub fn cell_for(&self, obs: u32) -> Option<(usize, usize)> {
        (0..self.legal_off.len() - 1).find_map(|ci| {
            self.row(ci)
                .find(|&cell| obs_key(&self.acts[self.action_at(cell)]) == obs)
                .map(|cell| (ci, cell))
        })
    }
}

/// Uniform over each config's legal actions.
pub fn uniform(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
    let mut np = NodePolicy::frame(s, ctx, player, cfgs);
    for ci in 0..cfgs.len() {
        let row = np.row(ci);
        let k = row.len() as f32;
        for cell in row {
            np.probs[cell] = 1.0 / k;
        }
    }
    np
}

/// One-ply greedy, softmaxed at `temp`. An action's score is a property of the
/// successor's public state, so it is evaluated once per action and shared
/// across configs; only the legal set differs between them.
pub fn greedy(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config], temp: f32) -> NodePolicy {
    let mut np = NodePolicy::frame(s, ctx, player, cfgs);
    let na = np.acts.len();
    let mut score = vec![f32::NEG_INFINITY; na];
    for a in 0..na {
        let Some(rep) = cfgs.iter().find(|c| action_legal(c, np.aslot[a])) else {
            continue;
        };
        let mut probe = *s;
        set_config(&mut probe, player, ctx, rep);
        probe.apply_inplace(np.acts[a]);
        score[a] = crate::selfplay::eval_static(&probe, player) / temp;
    }
    for ci in 0..cfgs.len() {
        let cells = np.row(ci);
        let best = cells.clone().fold(f32::NEG_INFINITY, |best, cell| {
            best.max(score[np.action_at(cell)])
        });
        let mut sum = 0.0;
        for cell in cells.clone() {
            let e = (score[np.action_at(cell)] - best).exp();
            np.probs[cell] = e;
            sum += e;
        }
        // A little uniform mass keeps the belief filter from collapsing and
        // keeps warm-start games diverse.
        let k = cells.len() as f32;
        for cell in cells {
            np.probs[cell] = 0.95 * np.probs[cell] / sum + 0.05 / k;
        }
    }
    np
}

/// A node's shape, read off a solved CPU tree.
pub fn cpu_frame(sv: &Solver<'_>, node: usize) -> NodePolicy {
    let n = &sv.nodes[node];
    NodePolicy {
        acts: n.acts.clone(),
        aslot: n.aslot.clone(),
        fdown: n.fdown.clone(),
        legal_off: n.legal_off.clone(),
        legal_action: n.legal_action.clone(),
        probs: vec![0.0; n.legal_action.len()],
    }
}

/// The CFR average strategy at a node of a finished solve.
pub fn at_node(sv: &Solver<'_>, node: usize, configs: usize) -> NodePolicy {
    let mut np = cpu_frame(sv, node);
    for config in 0..configs {
        let row = np.row(config);
        np.probs[row].copy_from_slice(sv.average_strategy(node, config));
    }
    np
}
