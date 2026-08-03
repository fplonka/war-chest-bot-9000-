//! Depth-limited CFR over public belief states — the search half of ReBeL
//! (Brown, Bakhtin, Lerer & Gong 2020), specialised to War Chest.
//!
//! The subgame rooted at a PBS is unrolled over **public observations**. A node
//! is a leaf when it is terminal, when it is a chance node (a draw: the outcome
//! is private, so clamping there keeps the subgame chance-free), or when the
//! depth limit is reached. Leaf values come from the value network.
//!
//! Conventions follow the reference implementation (`csrc/liars_dice` of
//! `facebookresearch/rebel`):
//!   * alternating-traverser linear CFR,
//!   * leaf values are *counterfactual* — the network's per-hand value scaled by
//!     the opponent's unnormalised reach into that leaf,
//!   * the network is queried with normalised reaches as the beliefs,
//!   * the root value target is the running mean of per-config root values,
//!   * acting and belief propagation use the current regret-matching iterate.
//!
//! Three things differ from poker, all consequences of War Chest's observation
//! structure:
//!
//! 1. **Action sets depend on the config**: you can only play a coin you hold.
//! 2. **An action moves the information state**: hand minus that coin, and for a
//!    face-down play the coin lands in the face-down discard, so the config
//!    changes in both components. `trans` carries that map.
//! 3. **Actions are partially private**: Pass, Claim Initiative and a Recruit
//!    payment announce the event but not the coin, so several private actions
//!    share one public child. `obs_child` carries that many-to-one map, and the
//!    belief update sums over the private actions consistent with what was seen.

use crate::actions::Action;
use crate::board::NONE;
use crate::net::Mlp;
use crate::rebel::*;
use crate::state::{Cont, State};
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};

#[derive(Clone, Copy)]
pub struct Cfg {
    /// Public-tree depth. 1 means "the root's children are leaves".
    pub depth: usize,
    /// CFR iterations (alternating, so each player is traversed iters/2 times).
    pub iters: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg { depth: 1, iters: 16 }
    }
}

/// The value network: PBS -> one value per hand key per player.
#[derive(Clone, Default)]
pub struct Nets {
    pub value: Mlp,
}

impl Nets {
    pub fn ready(&self) -> bool {
        !self.value.is_empty()
    }
}

/// Which coin leaves the acting player's hand when `a` is played. Derived from
/// the rules rather than from the engine's action listing: the Royal Guard
/// tactic is offered whenever a Royal Guard is deployed, but it always spends
/// the Royal Coin.
pub fn action_coin(a: &Action, s: &State) -> u8 {
    use Action::*;
    match *a {
        Deploy { unit, .. } | Bolster { unit, .. } => unit,
        ClaimInitiative { coin } | Recruit { coin, .. } | Pass { coin } | TacFootman { coin } => {
            coin
        }
        TacRoyalGuard { .. } => ROYAL_COIN,
        TacEnsign { .. } => ENSIGN,
        TacMarshal { .. } => MARSHAL,
        Move { from, .. }
        | Control { from }
        | Attack { from, .. }
        | TacArcher { from, .. }
        | TacCavalryMove { from, .. }
        | TacCrossbow { from, .. }
        | TacLancer { from, .. }
        | TacLightCav { from, .. } => s.hex_type[from as usize],
        // Micro-decisions (berserker chain, swordsman step, ...) spend no coin.
        _ => NONE,
    }
}

/// The private actions available at a node, each tagged with the coin slot it
/// spends and whether that coin goes face down. Only `MainPlay` nodes have
/// config-dependent action sets; every other decision is public, so one call to
/// `legal_actions` suffices there.
///
/// Enumeration runs over the public *reserve*, which can offer a coin no config
/// in `cfgs` actually holds; those actions are unreachable and are dropped, so
/// every caller sees the same list and no child is built that nobody can enter.
pub fn node_actions(
    s: &State,
    player: u8,
    ctx: &Ctx,
    cfgs: &[Config],
) -> (Vec<Action>, Vec<i8>, Vec<bool>) {
    let mut acts: Vec<Action> = Vec::new();
    let mut aslot: Vec<i8> = Vec::new();
    let mut fdown: Vec<bool> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    if matches!(s.pending(), Cont::MainPlay) {
        let res = reserve(s, player, ctx);
        for k in 0..NSLOT {
            if res[k] == 0 {
                continue;
            }
            let mut probe = s.clone();
            let mut one = Config::default();
            one.hand[k] = 1;
            set_config(&mut probe, player, ctx, &one);
            if !cfgs.is_empty() && !cfgs.iter().any(|c| c.hand[k] > 0) {
                continue;
            }
            for a in probe.legal_actions() {
                if !seen.insert(a.encode()) {
                    continue;
                }
                let coin = action_coin(&a, &probe);
                let slot = if coin == NONE {
                    -1
                } else {
                    ctx.slot_of[player as usize][coin as usize]
                };
                if slot >= 0
                    && !cfgs.is_empty()
                    && !cfgs.iter().any(|c| c.hand[slot as usize] > 0)
                {
                    continue;
                }
                aslot.push(slot);
                fdown.push(is_facedown_play(&a));
                acts.push(a);
            }
        }
    } else {
        // Micro-decisions spend nothing out of hand: the config is untouched.
        for a in s.legal_actions() {
            if seen.insert(a.encode()) {
                acts.push(a);
                aslot.push(-1);
                fdown.push(false);
            }
        }
    }
    (acts, aslot, fdown)
}

pub struct TNode {
    pub s: State,
    pub player: u8,
    pub leaf: bool,
    pub acts: Vec<Action>,
    pub aslot: Vec<i8>,
    pub fdown: Vec<bool>,
    /// Action index -> position in `child` (many private actions, one public
    /// observation).
    pub obs_child: Vec<usize>,
    pub child: Vec<usize>,
    /// Config support per player at this node.
    pub cfgs: [Vec<Config>; 2],
    /// `[config * na + action]`, for the acting player.
    pub legal: Vec<bool>,
    /// `[config * na + action]` -> the successor's config index in the child.
    pub trans: Vec<i32>,
}

impl TNode {
    #[inline]
    pub fn na(&self) -> usize {
        self.acts.len()
    }
    #[inline]
    pub fn nc(&self, p: usize) -> usize {
        self.cfgs[p].len()
    }
}

pub struct Solver<'a> {
    ctx: &'a Ctx,
    nets: &'a Nets,
    cfg: Cfg,
    pub nodes: Vec<TNode>,
    root_belief: [Belief; 2],
    regret: Vec<Vec<f32>>,
    sum_strat: Vec<Vec<f32>>,
    cur: Vec<Vec<f32>>,
    avg: Vec<Vec<f32>>,
    /// `[node][player]` -> reach per config.
    reach: Vec<[Vec<f32>; 2]>,
    /// `[node]` -> the traverser's counterfactual value per config.
    vals: Vec<Vec<f32>>,
    root_mean: [Vec<f32>; 2],
    steps: [usize; 2],
    leaves: Vec<usize>,
    xb: Vec<f32>,
    ob: Vec<f32>,
    sb: Vec<f32>,
}

impl<'a> Solver<'a> {
    pub fn new(
        root: &State,
        ctx: &'a Ctx,
        nets: &'a Nets,
        cfg: Cfg,
        belief: [Belief; 2],
    ) -> Solver<'a> {
        let cfgs = [belief[0].cfg.clone(), belief[1].cfg.clone()];
        let mut sv = Solver {
            ctx,
            nets,
            cfg,
            nodes: Vec::new(),
            root_mean: [vec![0.0; cfgs[0].len()], vec![0.0; cfgs[1].len()]],
            root_belief: belief,
            regret: Vec::new(),
            sum_strat: Vec::new(),
            cur: Vec::new(),
            avg: Vec::new(),
            reach: Vec::new(),
            vals: Vec::new(),
            steps: [0, 0],
            leaves: Vec::new(),
            xb: Vec::new(),
            ob: Vec::new(),
            sb: Vec::new(),
        };
        sv.build(root.clone(), cfg.depth.max(1), cfgs);
        for i in 0..sv.nodes.len() {
            let n = &sv.nodes[i];
            let (na, p) = (n.na(), n.player as usize);
            let nc = n.nc(p);
            sv.reach
                .push([vec![0.0; n.nc(0)], vec![0.0; n.nc(1)]]);
            sv.vals.push(vec![0.0; n.nc(0).max(n.nc(1))]);
            sv.regret.push(vec![0.0; nc * na]);
            sv.sum_strat.push(vec![0.0; nc * na]);
            // CFR starts from a uniform strategy over the legal actions, as in
            // the reference. No heuristic prior is injected here: the greedy
            // knowledge enters through the pretrained value network, which is
            // what CFR actually consumes.
            let mut u = vec![0.0f32; nc * na];
            for c in 0..nc {
                let k = (0..na).filter(|&a| n.legal[c * na + a]).count() as f32;
                for a in 0..na {
                    if n.legal[c * na + a] {
                        u[c * na + a] = 1.0 / k;
                    }
                }
            }
            sv.cur.push(u.clone());
            sv.avg.push(u);
        }
        // Seed the strategy sums with one reach-weighted uniform strategy, as
        // `get_uniform_reach_weigted_strategy` does in the reference.
        sv.precompute_reaches();
        for i in 0..sv.nodes.len() {
            if sv.nodes[i].leaf {
                continue;
            }
            let (na, p) = (sv.nodes[i].na(), sv.nodes[i].player as usize);
            for c in 0..sv.nodes[i].nc(p) {
                let r = sv.reach[i][p][c];
                for a in 0..na {
                    sv.sum_strat[i][c * na + a] += r * sv.cur[i][c * na + a];
                }
            }
        }
        sv
    }

    // ------------------------------------------------------------ tree build

    fn build(&mut self, s: State, depth: usize, cfgs: [Vec<Config>; 2]) -> usize {
        let player = s.to_act();
        let leaf = s.is_terminal() || s.is_chance() || depth == 0;
        let id = self.nodes.len();
        self.nodes.push(TNode {
            s: s.clone(),
            player,
            leaf,
            acts: Vec::new(),
            aslot: Vec::new(),
            fdown: Vec::new(),
            obs_child: Vec::new(),
            child: Vec::new(),
            cfgs: cfgs.clone(),
            legal: Vec::new(),
            trans: Vec::new(),
        });
        if leaf {
            return id;
        }

        let me = player as usize;
        let mine = cfgs[me].clone();
        let nc = mine.len();
        let (acts, aslot, fdown) = node_actions(&s, player, self.ctx, &mine);
        let na = acts.len();
        debug_assert!(na > 0, "a decision node must offer a reachable action");

        let mut legal = vec![false; nc * na];
        for (ci, c) in mine.iter().enumerate() {
            for a in 0..na {
                legal[ci * na + a] = aslot[a] < 0 || c.hand[aslot[a] as usize] > 0;
            }
        }

        // Group private actions by what the opponent actually observes.
        let mut obs_keys: Vec<u32> = Vec::new();
        let mut obs_child = vec![0usize; na];
        for a in 0..na {
            let k = obs_key(&acts[a]);
            obs_child[a] = match obs_keys.iter().position(|&x| x == k) {
                Some(i) => i,
                None => {
                    obs_keys.push(k);
                    obs_keys.len() - 1
                }
            };
        }

        // Config support of each public child: the union over the private
        // actions that produce that observation.
        let mut child_cfgs: Vec<Vec<Config>> = vec![Vec::new(); obs_keys.len()];
        for (ci, c) in mine.iter().enumerate() {
            for a in 0..na {
                if !legal[ci * na + a] {
                    continue;
                }
                if let Some(n) = advance_config(c, aslot[a], fdown[a]) {
                    child_cfgs[obs_child[a]].push(n);
                }
            }
        }
        for v in child_cfgs.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }

        let mut trans = vec![-1i32; nc * na];
        for (ci, c) in mine.iter().enumerate() {
            for a in 0..na {
                if !legal[ci * na + a] {
                    continue;
                }
                if let Some(n) = advance_config(c, aslot[a], fdown[a]) {
                    let ch = obs_child[a];
                    trans[ci * na + a] = child_cfgs[ch]
                        .binary_search(&n)
                        .map(|x| x as i32)
                        .unwrap_or(-1);
                }
            }
        }

        // One world per public child, built from any config that can produce
        // it: the public projection of the successor is the same either way.
        let mut child = Vec::with_capacity(obs_keys.len());
        for ch in 0..obs_keys.len() {
            let a = (0..na).find(|&a| obs_child[a] == ch).unwrap();
            let rep = *mine
                .iter()
                .find(|c| aslot[a] < 0 || c.hand[aslot[a] as usize] > 0)
                .expect("a kept action is playable by some config in the support");
            let mut cs = s.clone();
            set_config(&mut cs, player, self.ctx, &rep);
            cs.apply_inplace(acts[a]);
            let mut cc = cfgs.clone();
            cc[me] = child_cfgs[ch].clone();
            child.push(self.build(cs, depth - 1, cc));
        }

        let n = &mut self.nodes[id];
        n.acts = acts;
        n.aslot = aslot;
        n.fdown = fdown;
        n.obs_child = obs_child;
        n.child = child;
        n.legal = legal;
        n.trans = trans;
        id
    }

    // -------------------------------------------------------------- CFR core

    fn precompute_reaches(&mut self) {
        for r in self.reach.iter_mut() {
            r[0].iter_mut().for_each(|v| *v = 0.0);
            r[1].iter_mut().for_each(|v| *v = 0.0);
        }
        for p in 0..2 {
            self.reach[0][p].copy_from_slice(&self.root_belief[p].p);
        }
        for i in 0..self.nodes.len() {
            if self.nodes[i].leaf {
                continue;
            }
            let (na, me) = (self.nodes[i].na(), self.nodes[i].player as usize);
            let op = 1 - me;
            let src_op = self.reach[i][op].clone();
            let src_me = self.reach[i][me].clone();
            for ch in 0..self.nodes[i].child.len() {
                let c = self.nodes[i].child[ch];
                // The idle player's information state is untouched, and the
                // child's support for them is the same list.
                self.reach[c][op].copy_from_slice(&src_op);
            }
            for a in 0..na {
                let c = self.nodes[i].child[self.nodes[i].obs_child[a]];
                for ci in 0..src_me.len() {
                    if !self.nodes[i].legal[ci * na + a] {
                        continue;
                    }
                    let t = self.nodes[i].trans[ci * na + a];
                    if t < 0 {
                        continue;
                    }
                    self.reach[c][me][t as usize] += src_me[ci] * self.cur[i][ci * na + a];
                }
            }
        }
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values.
    fn leaf_values(&mut self, traverser: usize) {
        if self.leaves.is_empty() {
            self.leaves = (0..self.nodes.len())
                .filter(|&i| self.nodes[i].leaf)
                .collect();
        }
        let leaves = std::mem::take(&mut self.leaves);
        let mut rows = 0usize;
        self.xb.clear();
        for &i in &leaves {
            if self.nodes[i].s.is_terminal() {
                continue;
            }
            rows += 1;
            let b = [self.leaf_belief(i, 0), self.leaf_belief(i, 1)];
            let base = self.xb.len();
            self.xb.resize(base + FEAT, 0.0);
            write_features(
                &self.nodes[i].s,
                self.ctx,
                &b,
                &mut self.xb[base..base + FEAT],
            );
        }
        if rows > 0 && !self.nets.value.is_empty() {
            self.nets
                .value
                .forward(&self.xb, rows, &mut self.sb, &mut self.ob);
        } else {
            self.ob.clear();
            self.ob.resize(rows * 2 * NHAND, 0.0);
        }
        let mut row = 0usize;
        for &i in &leaves {
            let opp = 1 - traverser;
            let opp_reach: f32 = self.reach[i][opp].iter().sum();
            let nc = self.nodes[i].nc(traverser);
            if self.nodes[i].s.is_terminal() {
                let u = self.nodes[i].s.utility(traverser);
                for c in 0..nc {
                    self.vals[i][c] = u * opp_reach;
                }
            } else {
                let off = row * 2 * NHAND + traverser * NHAND;
                for c in 0..nc {
                    let h = self.nodes[i].cfgs[traverser][c].hand_index();
                    self.vals[i][c] = self.ob[off + h] * opp_reach;
                }
                row += 1;
            }
        }
        self.leaves = leaves;
    }

    /// The normalised reach at a leaf, as a belief the network can consume.
    fn leaf_belief(&self, node: usize, p: usize) -> Belief {
        let mut b = Belief {
            cfg: self.nodes[node].cfgs[p].clone(),
            p: self.reach[node][p].clone(),
        };
        b.normalize();
        b
    }

    fn update_regrets(&mut self, traverser: usize) {
        self.precompute_reaches();
        self.leaf_values(traverser);
        for i in (0..self.nodes.len()).rev() {
            if self.nodes[i].leaf {
                continue;
            }
            let (na, me) = (self.nodes[i].na(), self.nodes[i].player as usize);
            let nc = self.nodes[i].nc(traverser);
            for c in 0..nc {
                self.vals[i][c] = 0.0;
            }
            if me == traverser {
                for a in 0..na {
                    let ch = self.nodes[i].child[self.nodes[i].obs_child[a]];
                    for c in 0..nc {
                        if !self.nodes[i].legal[c * na + a] {
                            continue;
                        }
                        let t = self.nodes[i].trans[c * na + a];
                        if t < 0 {
                            continue;
                        }
                        let av = self.vals[ch][t as usize];
                        self.regret[i][c * na + a] += av;
                        self.vals[i][c] += av * self.cur[i][c * na + a];
                    }
                }
                for c in 0..nc {
                    let base = self.vals[i][c];
                    for a in 0..na {
                        if self.nodes[i].legal[c * na + a] {
                            self.regret[i][c * na + a] -= base;
                        }
                    }
                }
            } else {
                // The traverser's information state is unchanged across an
                // opponent decision, and the opponent's strategy is already
                // baked into the reach probabilities at the children.
                for ch in 0..self.nodes[i].child.len() {
                    let c_id = self.nodes[i].child[ch];
                    for c in 0..nc {
                        self.vals[i][c] += self.vals[c_id][c];
                    }
                }
            }
        }
    }

    pub fn step(&mut self, traverser: usize) {
        self.update_regrets(traverser);
        let alpha = 2.0 / (self.steps[traverser] as f32 + 2.0);
        for c in 0..self.root_mean[traverser].len() {
            self.root_mean[traverser][c] +=
                (self.vals[0][c] - self.root_mean[traverser][c]) * alpha;
        }
        // Linear CFR: discount by t/(t+1) after each update.
        let m = self.steps[traverser] as f32 + 1.0;
        let disc = m / (m + 1.0);
        for i in 0..self.nodes.len() {
            if self.nodes[i].leaf || self.nodes[i].player as usize != traverser {
                continue;
            }
            let (na, nc) = (self.nodes[i].na(), self.nodes[i].nc(traverser));
            for c in 0..nc {
                let mut sum = 0.0;
                for a in 0..na {
                    let v = if self.nodes[i].legal[c * na + a] {
                        self.regret[i][c * na + a].max(1e-6)
                    } else {
                        0.0
                    };
                    self.cur[i][c * na + a] = v;
                    sum += v;
                }
                if sum > 0.0 {
                    for a in 0..na {
                        self.cur[i][c * na + a] /= sum;
                    }
                }
                for a in 0..na {
                    self.regret[i][c * na + a] *= disc;
                    self.sum_strat[i][c * na + a] *= disc;
                }
            }
        }
        // Accumulate the average strategy, weighted by the traverser's reach
        // under the strategy just computed.
        self.precompute_reaches();
        for i in 0..self.nodes.len() {
            if self.nodes[i].leaf || self.nodes[i].player as usize != traverser {
                continue;
            }
            let (na, nc) = (self.nodes[i].na(), self.nodes[i].nc(traverser));
            for c in 0..nc {
                let r = self.reach[i][traverser][c];
                let mut sum = 0.0;
                for a in 0..na {
                    self.sum_strat[i][c * na + a] += r * self.cur[i][c * na + a];
                    sum += self.sum_strat[i][c * na + a];
                }
                if sum > 0.0 {
                    for a in 0..na {
                        self.avg[i][c * na + a] = self.sum_strat[i][c * na + a] / sum;
                    }
                } else {
                    let k = (0..na).filter(|&a| self.nodes[i].legal[c * na + a]).count() as f32;
                    for a in 0..na {
                        self.avg[i][c * na + a] = if self.nodes[i].legal[c * na + a] {
                            1.0 / k
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        self.steps[traverser] += 1;
    }

    pub fn multistep(&mut self, iters: usize) {
        for i in 0..iters {
            self.step(i % 2);
        }
    }

    /// Run the remaining CFR iterations up to the configured total
    /// (`self.cfg.iters`), preserving the alternating-traverser schedule of
    /// `step(i % 2)`. Used to finish a solve whose first `stop` steps were
    /// run when the subgame was built, after the walk acted on the
    /// strategies at that iterate.
    pub fn complete(&mut self) {
        let iters = self.cfg.iters;
        let done = self.steps[0] + self.steps[1];
        for i in done..iters {
            self.step(i % 2);
        }
    }

    pub fn solved(&self) -> bool {
        self.steps[0] > 0 && self.steps[1] > 0
    }

    /// Per-config root values for both players: the ReBeL value target, before
    /// projection onto the network's hand-key basis.
    pub fn root_values(&self, p: usize) -> &[f32] {
        &self.root_mean[p]
    }

    /// The strategy used for acting and for belief propagation. As in the
    /// reference that is the current regret-matching iterate, not the average.
    pub fn sampling_strategy(&self, node: usize, c: usize) -> &[f32] {
        let na = self.nodes[node].na();
        &self.cur[node][c * na..(c + 1) * na]
    }

    /// The CFR average strategy: the approximate equilibrium of the subgame.
    pub fn average_strategy(&self, node: usize, c: usize) -> &[f32] {
        let na = self.nodes[node].na();
        &self.avg[node][c * na..(c + 1) * na]
    }
}
