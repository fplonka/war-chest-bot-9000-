//! A decision node's policy, and the belief filter that follows from it.
//!
//! Every agent answers the same question at a decision node: for each private
//! config the acting player might hold, how likely is each of that config's
//! legal actions? `NodePolicy` is that answer. The solver, one-ply greedy, and
//! uniform random differ only in how they fill `probs`.
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
use crate::board::{board, NONE, N_HEXES, N_LOCATIONS};
use crate::pbs::{action_legal, advance_config, obs_key, set_config, Belief, Config, Ctx};
use crate::rng::Rng;
use crate::search::{node_actions, Policy, Solver};
use crate::state::{State, Z_ELIM};

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

    /// This node's policy in the replay-row layout a SoG root stores, so a
    /// warm row is an ordinary row.
    pub fn to_replay(&self) -> Policy {
        let ncfg = self.legal_off.len().saturating_sub(1);
        let mut out = Policy {
            acts: self
                .acts
                .iter()
                .enumerate()
                .map(|(a, act)| {
                    let h = act.hexes();
                    [
                        act.kind() as u8,
                        crate::net::slot_column(self.aslot[a]) as u8,
                        h[0],
                        h[1],
                        h[2],
                    ]
                })
                .collect(),
            off: Vec::with_capacity(ncfg + 1),
            act: Vec::new(),
            p: Vec::new(),
        };
        out.off.push(0);
        for ci in 0..ncfg {
            for cell in self.row(ci) {
                out.act.push(self.action_at(cell) as u16);
                out.p.push(self.probs[cell]);
            }
            out.off.push(out.act.len() as u32);
        }
        out
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

    /// Mix `eps` of the uniform-over-legal into every config's row.
    ///
    /// The mixture is the acting policy: public knowledge. Sampling and the
    /// belief update both read it.
    pub fn mix_uniform(&mut self, eps: f32) {
        if eps == 0.0 {
            return;
        }
        let n = self.legal_off.len().saturating_sub(1);
        for ci in 0..n {
            let row = self.row(ci);
            let k = row.len();
            if k == 0 {
                continue;
            }
            let u = 1.0 / k as f32;
            let keep = 1.0 - eps;
            for cell in row {
                self.probs[cell] = keep * self.probs[cell] + eps * u;
            }
        }
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

/// A node's shape, read off a solved CPU tree.
pub fn cpu_frame(sv: &Solver, node: usize) -> NodePolicy {
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

/// The acting policy at a finished solve's root.
///
/// A first expansion that did not fit the slot left the root a leaf, with an
/// empty `legal_off`. That root plays uniformly over legal actions.
pub fn root(sv: &Solver) -> NodePolicy {
    let n = &sv.nodes[0];
    let cfgs = n.cfgs[n.player as usize].as_ref();
    if n.legal_off.is_empty() {
        uniform(&sv.states[0], &sv.ctx, n.player, cfgs)
    } else {
        at_node(sv, 0, cfgs.len())
    }
}

/// The CFR average strategy at a node of a finished solve.
pub fn at_node(sv: &Solver, node: usize, configs: usize) -> NodePolicy {
    let mut np = cpu_frame(sv, node);
    for config in 0..configs {
        let row = np.row(config);
        np.probs[row].copy_from_slice(sv.average_strategy(node, config));
    }
    np
}

/// A handwritten positional score from `p`'s seat, over public facts only:
/// markers, hex owners and heights, eliminated coins, and how far each side's
/// nearest unit is from the locations it does not yet hold.
pub fn eval_static(s: &State, p: u8) -> f32 {
    if s.is_terminal() {
        return s.utility(p as usize) * 1e6;
    }
    let b = board();
    let o = 1 - p;
    let mut sc = 12.0 * (s.markers_on_board(p) as f32 - s.markers_on_board(o) as f32);
    let (mut coins_p, mut coins_o) = (0.0f32, 0.0f32);
    for h in 0..N_HEXES {
        match s.hex_owner[h] {
            x if x == p => coins_p += s.hex_height[h] as f32,
            x if x == o => coins_o += s.hex_height[h] as f32,
            _ => {}
        }
    }
    sc += 1.5 * (coins_p - coins_o);
    let elim = |q: u8| -> f32 {
        s.zones[q as usize][Z_ELIM]
            .iter()
            .map(|&x| x as f32)
            .sum()
    };
    sc += elim(o) - elim(p);
    let (mut cover_p, mut cover_o) = (0.0f32, 0.0f32);
    for li in 0..N_LOCATIONS {
        let l = b.location_hexes[li] as usize;
        let (mut bp, mut bo) = (7.0f32, 7.0f32);
        for h in 0..N_HEXES {
            let owner = s.hex_owner[h];
            if owner == NONE {
                continue;
            }
            let d = b.dist[h][l] as f32;
            if owner == p {
                bp = bp.min(d);
            } else {
                bo = bo.min(d);
            }
        }
        if s.loc_marker[l] != p {
            cover_p += bp;
        }
        if s.loc_marker[l] != o {
            cover_o += bo;
        }
    }
    sc += 0.5 * (cover_o - cover_p);
    sc
}

/// `eval_static` squashed into (-1, 1) so it can label the value head.
pub fn eval_squashed(s: &State, p: u8) -> f32 {
    (eval_static(s, p) / 25.0).tanh()
}

/// One-ply greedy, softmaxed at `temp`. An action's score is a fact of the
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
        score[a] = eval_static(&probe, player) / temp;
    }
    for ci in 0..cfgs.len() {
        let cells = np.row(ci);
        let best = cells
            .clone()
            .fold(f32::NEG_INFINITY, |best, cell| best.max(score[np.action_at(cell)]));
        let mut sum = 0.0f32;
        for cell in cells.clone() {
            let e = (score[np.action_at(cell)] - best).exp();
            np.probs[cell] = e;
            sum += e;
        }
        if sum > 0.0 {
            for cell in cells {
                np.probs[cell] /= sum;
            }
        }
    }
    np
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbs::{true_config, Belief, Ctx};
    use crate::search::{Budget, Cfg, Nets};
    use crate::selfplay::make_game;
    use crate::state::Cont;
    use std::sync::Arc;

    #[test]
    fn a_leaf_root_plays_uniform() {
        let mut rng = Rng::new(0x69);
        let mut s = make_game(&mut rng, true);
        while !s.is_terminal() && (s.is_chance() || !matches!(s.pending(), Cont::MainPlay)) {
            let a = s.legal_actions();
            s.apply_inplace(a[rng.below(a.len())]);
        }
        assert!(!s.is_terminal() && matches!(s.pending(), Cont::MainPlay));
        let ctx = Ctx::new(&s);
        let bel = [
            Belief::point(true_config(&s, 0, &ctx)),
            Belief::point(true_config(&s, 1, &ctx)),
        ];
        let sv = Solver::new(
            &s,
            ctx,
            Arc::new(Nets::default()),
            Cfg {
                s: 0,
                budget: Budget {
                    nodes: 1,
                    ..Budget::default()
                },
                ..Default::default()
            },
            bel,
            Rng::new(1),
        );
        assert!(sv.nodes[0].legal_off.is_empty());
        let np = root(&sv);
        let _ = np.sample(&mut Rng::new(2), 0);
    }
}
