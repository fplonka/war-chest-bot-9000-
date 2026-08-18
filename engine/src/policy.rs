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
use crate::rebel::{action_legal, advance_config, obs_key, set_config, Belief, Config, Ctx};
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
        let w: Vec<f64> = self.probs[row.clone()]
            .iter()
            .map(|&x| x.max(0.0) as f64)
            .collect();
        row.start
            + if w.iter().sum::<f64>() > 0.0 {
                rng.weighted_index(&w)
            } else {
                rng.below(w.len().max(1))
            }
    }

    /// The most likely cell of config `c`'s row. Ties go to the first, so the
    /// choice is a property of the position rather than of the run.
    pub fn best(&self, c: usize) -> usize {
        let row = self.row(c);
        row.clone()
            .max_by(|&a, &b| self.probs[a].total_cmp(&self.probs[b]))
            .unwrap_or(row.start)
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

/// Local best response: pick the action worth most, with the value of each
/// found by playing the rest of the game out rather than by asking a network.
///
/// This is the whole point of the probe and the reason it borrows nothing. A
/// best response that leant on a trained value network would measure that
/// network as much as the bot under test — and would make the answer depend
/// on which network the probe happened to be carrying. Rollouts under a fixed
/// simple policy are crude, but they are crude in the same way for every
/// opponent, which is what a measuring instrument needs.
///
/// `belief` is the range the probe holds over the *other* seat, kept current
/// from the strategy the referee reveals to it. Every hand in it is tried and
/// weighted by its probability, rather than sampled: the ranges here run to
/// tens of hands, so enumerating them costs little and removes a whole source
/// of noise from the estimate.
pub fn lbr(
    s: &State,
    ctx: &Ctx,
    player: u8,
    cfgs: &[Config],
    mine: usize,
    belief: &Belief,
    rounds: usize,
    rng: &mut Rng,
) -> NodePolicy {
    let mut np = uniform(s, ctx, player, cfgs);
    let them = 1 - player;
    // Only the hand the probe is actually holding. The rest of the rows are
    // what its opponent would need to model it, and nothing models a probe.
    {
        let ci = mine;
        let cells = np.row(ci);
        let mut best = (cells.start, f32::NEG_INFINITY);
        for cell in cells.clone() {
            let (mut total, mut mass) = (0.0, 0.0);
            for (k, hand) in belief.cfg.iter().enumerate() {
                let w = belief.p.get(k).copied().unwrap_or(1.0);
                if w <= 0.0 {
                    continue;
                }
                for _ in 0..rounds {
                    let mut probe = *s;
                    set_config(&mut probe, player, ctx, &cfgs[ci]);
                    set_config(&mut probe, them, ctx, hand);
                    probe.apply_inplace(np.acts[np.action_at(cell)]);
                    total += w * playout(&mut probe, player, rng);
                    mass += w;
                }
            }
            let mean = if mass > 0.0 { total / mass } else { 0.0 };
            if mean > best.1 {
                best = (cell, mean);
            }
        }
        // A best response is not a mixed strategy: it plays the one action it
        // believes is worth most.
        for cell in cells {
            np.probs[cell] = f32::from(cell == best.0);
        }
    }
    np
}

/// How far a rollout runs before the static evaluation takes over.
const HORIZON: usize = 24;

/// Play forward a little and report who stands better.
///
/// Both sides play the handcrafted static evaluation greedily, with a little
/// noise so the estimate is not one line repeated. Uniform-random play was
/// tried first and is far too weak to be useful — a probe that cannot beat
/// Greedy will not find a leak in anything, and reports zero everywhere. A
/// heavy playout costs more per ply and is worth it, which is the same trade
/// every rollout searcher makes. It is still a handcrafted evaluation and
/// still no trained network.
fn playout(s: &mut State, player: u8, rng: &mut Rng) -> f32 {
    let mut acts = Vec::new();
    // Truncated, and that is the whole trick. Played to the end, a rollout
    // carries the noise of a hundred more plies, and eight of them cannot see
    // which action was better through it -- measured, a probe scoring that way
    // loses to uniform random play. Stopping early and reading the static
    // evaluation trades a little bias for far less variance, which is the
    // trade every rollout searcher makes.
    for _ in 0..HORIZON {
        if s.is_terminal() {
            return s.utility(player as usize);
        }
        if s.is_chance() {
            let who = s.to_act();
            crate::selfplay::resolve_chance(s, who, rng);
            continue;
        }
        s.legal_actions_into(&mut acts);
        if acts.is_empty() {
            break;
        }
        let a = if rng.below(8) == 0 {
            acts[rng.below(acts.len())]
        } else {
            let who = s.to_act();
            let mut best = (acts[0], f32::NEG_INFINITY);
            for &a in &acts {
                let mut next = *s;
                next.apply_inplace(a);
                let v = crate::selfplay::eval_static(&next, who);
                if v > best.1 {
                    best = (a, v);
                }
            }
            best.0
        };
        s.apply_inplace(a);
    }
    (crate::selfplay::eval_static(s, player) / 100.0).clamp(-0.99, 0.99)
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

/// A node's shape, read off a packed tree. The packed offsets span the whole
/// tree, so they are rebased onto this node's own cells.
pub fn wave_frame(tree: &crate::serialize::WalkTree, node: usize, player: usize) -> NodePolicy {
    let actions = tree.action_range(node);
    let row0 = tree.legal_row_of[node] as usize;
    let configs = tree.supports[node][player].len();
    let cell0 = tree.legal_off[row0];
    let cell1 = tree.legal_off[row0 + configs];
    NodePolicy {
        acts: tree.actions[actions.clone()].to_vec(),
        aslot: tree.aslot[actions.clone()].to_vec(),
        fdown: tree.fdown[actions].to_vec(),
        legal_off: tree.legal_off[row0..=row0 + configs]
            .iter()
            .map(|&offset| offset - cell0)
            .collect(),
        legal_action: tree.legal_action[cell0 as usize..cell1 as usize].to_vec(),
        probs: vec![0.0; (cell1 - cell0) as usize],
    }
}

/// The CFR average strategy at a solved subgame's root.
pub fn solved(sv: &mut Solver<'_>, iters: usize, configs: usize) -> NodePolicy {
    sv.multistep(iters);
    let mut np = cpu_frame(sv, 0);
    for config in 0..configs {
        let row = np.row(config);
        np.probs[row].copy_from_slice(sv.average_strategy(0, config));
    }
    np
}

/// The same strategy, read off a wave solve's downloaded root rows.
pub fn from_wave(
    tree: &crate::serialize::WalkTree,
    result: &crate::gpu::SolveResult,
    player: usize,
) -> NodePolicy {
    let mut np = wave_frame(tree, 0, player);
    let cell0 = tree.legal_off[tree.legal_row_of[0] as usize] as usize;
    let cells = np.probs.len();
    np.probs
        .copy_from_slice(&result.strategy[cell0..cell0 + cells]);
    np
}
