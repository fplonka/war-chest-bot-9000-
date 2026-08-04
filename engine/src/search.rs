//! Depth-limited CFR over public belief states — the search half of ReBeL
//! (Brown, Bakhtin, Lerer & Gong 2020), specialised to War Chest.
//!
//! The subgame rooted at a PBS is unrolled over **public observations**. A node
//! is a leaf when it is terminal or when the depth limit is reached. A round-
//! start draw is *walked through*: its outcome is private, so the public tree
//! does not branch — the one child is the post-draw state, and the drawing
//! player's configs are convolved through the draw distribution. Leaf values
//! come from the value network.
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
use crate::{shape, timed};
use std::rc::Rc;
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
    /// Maintain the CFR *average* strategy. Only evaluation reads it —
    /// self-play acts on the current regret-matching iterate — and keeping it
    /// costs a second reach pass plus two sweeps over every strategy cell per
    /// iteration, so generation turns it off.
    pub average: bool,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            depth: 1,
            iters: 16,
            average: true,
        }
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
    /// Draw pass-through node: the public tree does not branch, there is one
    /// public child, and the drawing player's configs transition through the
    /// `draw` chance map.
    pub chance: bool,
    /// The drawing player's chance transition, composed over the whole run of
    /// consecutive draws this node stands for. Empty for decision nodes.
    pub draw: DrawMap,
    /// How many of the game's draws this node covers (0 for a decision node).
    pub draw_steps: u8,
    pub acts: Vec<Action>,
    pub aslot: Vec<i8>,
    pub fdown: Vec<bool>,
    /// Action index -> position in `child` (many private actions, one public
    /// observation).
    pub obs_child: Vec<usize>,
    /// The same map inverted, CSR-style: the actions leading to public child
    /// `ch` are `obs_act[obs_start[ch]..obs_start[ch + 1]]`. Reach propagation
    /// walks children on the outside so it can borrow parent and child reach
    /// vectors disjointly instead of copying the parent's.
    pub obs_start: Vec<u32>,
    pub obs_act: Vec<u32>,
    pub child: Vec<usize>,
    /// Config support per player at this node.
    ///
    /// Shared rather than owned: every public child of a decision node has the
    /// *same* support for the idle player, and a draw leaves the idle player's
    /// support untouched, so a subgame that copied these per node spent most of
    /// its build time duplicating lists nobody edits.
    pub cfgs: [Rc<[Config]>; 2],
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

/// The belief blocks are the tail of the encoding, and they are the only part
/// that moves between CFR iterations at a fixed leaf.
const SPLIT: usize = FEAT - OFF_BELIEF;

// A subgame's leaf batch runs to a couple of megabytes — the public encodings,
// the cached first layer, the belief blocks, the network scratch — and a game
// builds a fresh solver every couple of decisions. Allocating those buffers per
// solve meant faulting in megabytes of fresh zero pages thousands of times a
// second; the arithmetic that followed was the cheap part. They are handed back
// on drop and reused, capacity and all.
/// The five per-solve buffers, pooled *by role*: they differ in size by 5x, so
/// a single shared pool handed each one somebody else's buffer and made it grow
/// — and growth is the one thing that zeroes.
const N_ROLES: usize = 5;
const R_H0: usize = 0;
const R_XPUB: usize = 1;
const R_XB0: usize = 2;
const R_OB: usize = 3;
const R_SB: usize = 4;

thread_local! {
    static BUFS: std::cell::RefCell<[Vec<Vec<f32>>; N_ROLES]> = const {
        std::cell::RefCell::new([
            Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        ])
    };
}

fn take_buf(role: usize) -> Vec<f32> {
    BUFS.with(|b| b.borrow_mut()[role].pop().unwrap_or_default())
}

fn give_buf(role: usize, v: Vec<f32>) {
    if v.capacity() == 0 {
        return;
    }
    // Kept at length, not cleared: every user grows on demand and overwrites
    // what it reads, so a cleared buffer would just have to be re-zeroed.
    BUFS.with(|b| {
        let mut b = b.borrow_mut();
        if b[role].len() < 2 {
            b[role].push(v);
        }
    });
}

pub struct Solver<'a> {
    ctx: &'a Ctx,
    nets: &'a Nets,
    cfg: Cfg,
    pub nodes: Vec<TNode>,
    root_belief: [Belief; 2],
    /// Regrets and the current regret-matching iterate, flat by node:
    /// `[soff[i] .. soff[i] + nc(player) * na]`, laid out `[config * na + a]`.
    regret: Vec<f32>,
    cur: Vec<f32>,
    soff: Vec<u32>,
    /// The average strategy and its running sum. Only evaluation reads these,
    /// so generation leaves them empty and they stay per-node.
    sum_strat: Vec<Vec<f32>>,
    avg: Vec<Vec<f32>>,
    /// Reach per config, flat: node `i`'s two players occupy
    /// `reach[roff[i] .. roff[i] + nc0 + nc1]`, player 0 first. One arena
    /// rather than `Vec<Vec<f32>>` — the CFR passes touch every node, and two
    /// pointer hops per node is what they were spending their time on.
    reach: Vec<f32>,
    roff: Vec<u32>,
    /// The traverser's counterfactual value per config, flat the same way:
    /// `vals[voff[i] .. voff[i] + max(nc0, nc1)]`.
    vals: Vec<f32>,
    voff: Vec<u32>,
    /// `[node]` -> config counts per player, so the hot loops never chase the
    /// `Rc` to ask how long a support is.
    nc: Vec<[u32; 2]>,
    root_mean: [Vec<f32>; 2],
    steps: [usize; 2],

    // ---------------------------------------------------------- leaf batch
    // Built once per solve. Everything here is a property of the leaf's public
    // state or its config support, so it survives every CFR iteration; only
    // `xb` (the belief blocks) is rewritten per iteration.
    /// Non-terminal leaves in node order — the rows of the network batch.
    leaf_rows: Vec<usize>,
    /// Terminal leaves, scored from the game instead of the network.
    term_leaves: Vec<usize>,
    /// Per row, per player: the public reserve.
    leaf_res: Vec<[[u8; NSLOT]; 2]>,
    /// Per row, per player: the hand key of every config in support, packed
    /// back to back and indexed through `leaf_hoff`.
    leaf_hand: Vec<u16>,
    leaf_hoff: Vec<u32>,
    /// `rows * hidden`: the public half of the first layer.
    h0: Vec<f32>,
    /// `rows * OFF_BELIEF`: the public half, filled during the build.
    xpub: Vec<f32>,
    /// `rows * SPLIT`: the belief half of the encoding.
    xb: Vec<f32>,
    ob: Vec<f32>,
    sb: Vec<f32>,
    batch_ready: bool,
    /// Working memory for the chance transitions, reused across the tree.
    draw_scratch: DrawScratch,
    /// `(config key, cell)` scratch for ordering a public child's support.
    cell_order: Vec<(u64, u32)>,
    dm: [DrawMap; 3],
}

impl Drop for Solver<'_> {
    fn drop(&mut self) {
        for (role, v) in [
            (R_H0, &mut self.h0),
            (R_XPUB, &mut self.xpub),
            (R_XB0, &mut self.xb),
            (R_OB, &mut self.ob),
            (R_SB, &mut self.sb),
        ] {
            give_buf(role, std::mem::take(v));
        }
    }
}

impl<'a> Solver<'a> {
    pub fn new(
        root: &State,
        ctx: &'a Ctx,
        nets: &'a Nets,
        cfg: Cfg,
        belief: [Belief; 2],
    ) -> Solver<'a> {
        let cfgs: [Rc<[Config]>; 2] = [
            belief[0].cfg.as_slice().into(),
            belief[1].cfg.as_slice().into(),
        ];
        let mut sv = Solver {
            ctx,
            nets,
            cfg,
            nodes: Vec::new(),
            root_mean: [vec![0.0; cfgs[0].len()], vec![0.0; cfgs[1].len()]],
            root_belief: belief,
            regret: Vec::new(),
            cur: Vec::new(),
            soff: Vec::new(),
            sum_strat: Vec::new(),
            avg: Vec::new(),
            reach: Vec::new(),
            roff: Vec::new(),
            vals: Vec::new(),
            voff: Vec::new(),
            nc: Vec::new(),
            steps: [0, 0],
            leaf_rows: Vec::new(),
            term_leaves: Vec::new(),
            leaf_res: Vec::new(),
            leaf_hand: Vec::new(),
            leaf_hoff: Vec::new(),
            h0: take_buf(R_H0),
            xpub: take_buf(R_XPUB),
            xb: take_buf(R_XB0),
            ob: take_buf(R_OB),
            sb: take_buf(R_SB),
            batch_ready: false,
            draw_scratch: DrawScratch::default(),
            cell_order: Vec::new(),
            dm: Default::default(),
        };
        {
            let _t = timed!(BUILD);
            // A depth-2 subgame runs to roughly 800 nodes and a `TNode` is
            // about a kilobyte, so growing the vector by doubling copied
            // hundreds of kilobytes per solve.
            sv.nodes.reserve(1024);
            sv.reach.reserve(1024);
            sv.vals.reserve(1024);
            sv.regret.reserve(1024);
            sv.cur.reserve(1024);
            sv.build(root.clone(), cfg.depth.max(1), cfgs);
        }
        let _t = timed!(ALLOC);
        let keep_avg = cfg.average;
        for i in 0..sv.nodes.len() {
            let n = &sv.nodes[i];
            let (na, p) = (n.na(), n.player as usize);
            let nc = n.nc(p);
            let (c0, c1) = (n.nc(0), n.nc(1));
            sv.nc.push([c0 as u32, c1 as u32]);
            sv.roff.push(sv.reach.len() as u32);
            sv.reach.resize(sv.reach.len() + c0 + c1, 0.0);
            sv.voff.push(sv.vals.len() as u32);
            sv.vals.resize(sv.vals.len() + c0.max(c1), 0.0);
            sv.soff.push(sv.regret.len() as u32);
            sv.regret.resize(sv.regret.len() + nc * na, 0.0);
            sv.sum_strat.push(if keep_avg {
                vec![0.0; nc * na]
            } else {
                Vec::new()
            });
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
            sv.cur.extend_from_slice(&u);
            sv.avg.push(if keep_avg { u } else { Vec::new() });
        }
        sv.soff.push(sv.regret.len() as u32);
        drop(_t);
        shape!(SOLVES, 1);
        shape!(NODES, sv.nodes.len());
        #[cfg(feature = "prof")]
        for n in &sv.nodes {
            if n.leaf {
                shape!(LEAVES, 1);
            } else {
                shape!(INNER_CA, n.na() * n.nc(n.player as usize));
                if n.chance {
                    shape!(CHANCE, 1);
                }
            }
            shape!(CFGSUM, n.nc(0) + n.nc(1));
        }
        sv.precompute_reaches();
        if keep_avg {
            // Seed the strategy sums with one reach-weighted uniform strategy,
            // as `get_uniform_reach_weigted_strategy` does in the reference.
            for i in 0..sv.nodes.len() {
                if sv.nodes[i].leaf {
                    continue;
                }
                let (na, p) = (sv.nodes[i].na(), sv.nodes[i].player as usize);
                let so = sv.soff[i] as usize;
                for c in 0..sv.nodes[i].nc(p) {
                    let r = sv.reach_of(i, p)[c];
                    for a in 0..na {
                        sv.sum_strat[i][c * na + a] += r * sv.cur[so + c * na + a];
                    }
                }
            }
        }
        sv
    }

    // ------------------------------------------------------------ tree build

    fn build(&mut self, s: State, depth: usize, cfgs: [Rc<[Config]>; 2]) -> usize {
        let player = s.to_act();
        // A plain round-start draw is walked through (one public child, no
        // depth cost). Other chance nodes (Warrior Priest draws — excluded
        // from every draft) stay leaves, as do depth-0 nodes and terminals.
        let draw_pass = matches!(s.pending(), Cont::Draw { .. });
        let leaf = s.is_terminal() || (!draw_pass && (depth == 0 || s.is_chance()));
        let id = self.nodes.len();
        let _tp = timed!(BPUSH);
        self.nodes.push(TNode {
            s: s.clone(),
            player,
            leaf,
            chance: false,
            draw: DrawMap::default(),
            draw_steps: 0,
            acts: Vec::new(),
            aslot: Vec::new(),
            fdown: Vec::new(),
            obs_child: Vec::new(),
            obs_start: Vec::new(),
            obs_act: Vec::new(),
            child: Vec::new(),
            cfgs: cfgs.clone(),
            legal: Vec::new(),
            trans: Vec::new(),
        });
        drop(_tp);
        if leaf {
            self.push_leaf(id, &s, &cfgs);
            return id;
        }

        if draw_pass {
            // The draw's outcome is private, so the public tree does not
            // branch: there is exactly one child, the state after the draw.
            // Which coin is drawn changes nothing public, so any legal
            // DrawCoin produces the same child. The drawing player's configs
            // are convolved through the draw distribution — the chance
            // factor stays separate from both players' strategies: it enters
            // the drawing player's reach as a transition, and the idle
            // player's reach passes through untouched. The depth is not
            // consumed: a draw is not a decision, and spending depth here is
            // what stops subgames from spanning a round boundary.
            //
            // A round start queues up to three draws in a row for the same
            // player. None of them branches and none of them is a decision, so
            // the whole run collapses into this one node with the composed
            // transition; `steps` is how many of the game's draws it stands
            // for, which is what the self-play walk counts off.
            let td = timed!(BDRAW);
            let me = player as usize;
            let mut cs = s;
            let mut cur: Vec<Config> = cfgs[me].to_vec();
            let mut next: Vec<Config> = Vec::new();
            let (mut draw, mut step, mut acc) = (
                std::mem::take(&mut self.dm[0]),
                std::mem::take(&mut self.dm[1]),
                std::mem::take(&mut self.dm[2]),
            );
            let mut steps = 0u8;
            loop {
                let acts = cs.legal_actions();
                debug_assert!(matches!(acts.first(), Some(Action::DrawCoin { .. })));
                let res = reserve(&cs, player, self.ctx);
                let fu = faceup_counts(&cs, player, self.ctx);
                self.draw_scratch
                    .transition(&cur, &res, &fu, &mut next, &mut step);
                if steps == 0 {
                    std::mem::swap(&mut draw, &mut step);
                } else {
                    self.draw_scratch.compose(&draw, &step, next.len(), &mut acc);
                    std::mem::swap(&mut draw, &mut acc);
                }
                std::mem::swap(&mut cur, &mut next);
                cs.apply_inplace(acts[0]);
                steps += 1;
                if !(matches!(cs.pending(), Cont::Draw { .. }) && cs.to_act() == player) {
                    break;
                }
            }
            drop(td);
            let mut cc = cfgs;
            cc[me] = cur.as_slice().into();
            let ch = self.build(cs, depth, cc);
            let n = &mut self.nodes[id];
            n.chance = true;
            n.child = vec![ch];
            // The node keeps its own map; the other two buffers go back to the
            // scratch set for the next chance node.
            n.draw = draw;
            n.draw_steps = steps;
            self.dm = [DrawMap::default(), step, acc];
            return id;
        }

        let me = player as usize;
        let mine = cfgs[me].clone();
        let nc = mine.len();
        let ta = timed!(BACTS);
        let (acts, aslot, fdown) = node_actions(&s, player, self.ctx, &mine);
        drop(ta);
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
        // The inverse map, CSR by public child.
        let nch = obs_keys.len();
        let mut obs_start = vec![0u32; nch + 1];
        for a in 0..na {
            obs_start[obs_child[a] + 1] += 1;
        }
        for ch in 0..nch {
            obs_start[ch + 1] += obs_start[ch];
        }
        let mut fill = obs_start.clone();
        let mut obs_act = vec![0u32; na];
        for a in 0..na {
            let ch = obs_child[a];
            obs_act[fill[ch] as usize] = a as u32;
            fill[ch] += 1;
        }

        // Config support of each public child — the union over the private
        // actions that produce that observation — and, in the same pass, where
        // each (config, action) cell lands in it.
        //
        // Ordering by integer key and reading the support off that single
        // ordering is what the draw transitions do, and for the same reason:
        // the obvious version sorts `Config`s and then binary-searches one per
        // cell, which at ~800 cells per decision node is most of the build.
        let mut child_cfgs: Vec<Vec<Config>> = vec![Vec::new(); nch];
        let mut trans = vec![-1i32; nc * na];
        let mut ent = std::mem::take(&mut self.cell_order);
        for ch in 0..nch {
            ent.clear();
            for &au in &obs_act[obs_start[ch] as usize..obs_start[ch + 1] as usize] {
                let a = au as usize;
                for ci in 0..nc {
                    if !legal[ci * na + a] {
                        continue;
                    }
                    if let Some(n) = advance_config(&mine[ci], aslot[a], fdown[a]) {
                        ent.push((n.key(), (ci * na + a) as u32));
                    }
                }
            }
            ent.sort_unstable();
            let sup = &mut child_cfgs[ch];
            let mut prev = u64::MAX;
            for &(k, cell) in ent.iter() {
                if k != prev {
                    prev = k;
                    let (ci, a) = (cell as usize / na, cell as usize % na);
                    sup.push(advance_config(&mine[ci], aslot[a], fdown[a]).unwrap());
                }
                trans[cell as usize] = (sup.len() - 1) as i32;
            }
        }
        self.cell_order = ent;

        // One world per public child, built from any config that can produce
        // it: the public projection of the successor is the same either way.
        let mut child = Vec::with_capacity(nch);
        for ch in 0..nch {
            let a = obs_act[obs_start[ch] as usize] as usize;
            let rep = *mine
                .iter()
                .find(|c| aslot[a] < 0 || c.hand[aslot[a] as usize] > 0)
                .expect("a kept action is playable by some config in the support");
            let tb = timed!(BAPPLY);
            let mut cs = s.clone();
            set_config(&mut cs, player, self.ctx, &rep);
            cs.apply_inplace(acts[a]);
            drop(tb);
            let mut cc = cfgs.clone();
            cc[me] = std::mem::take(&mut child_cfgs[ch]).into();
            child.push(self.build(cs, depth - 1, cc));
        }

        let n = &mut self.nodes[id];
        n.acts = acts;
        n.aslot = aslot;
        n.fdown = fdown;
        n.obs_child = obs_child;
        n.obs_start = obs_start;
        n.obs_act = obs_act;
        n.child = child;
        n.legal = legal;
        n.trans = trans;
        id
    }

    // -------------------------------------------------------------- CFR core

    /// Push reach probabilities down the tree under the current strategies.
    ///
    /// Children are always built after their parent, so `child > parent` and
    /// the parent's row can be borrowed alongside the child's through one
    /// `split_at_mut` — no copy of the parent's reach, which used to be two
    /// heap allocations per node per pass.
    fn precompute_reaches(&mut self) {
        let _t = timed!(REACH);
        self.reach.fill(0.0);
        for p in 0..2 {
            let at = self.roff[0] as usize + if p == 1 { self.nc[0][0] as usize } else { 0 };
            let n = self.nc[0][p] as usize;
            self.reach[at..at + n].copy_from_slice(&self.root_belief[p].p);
        }
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf {
                continue;
            }
            let (na, me) = (n.na(), n.player as usize);
            let op = 1 - me;
            // Offsets of each player's block inside a node's reach region.
            let blk = |cnt: [u32; 2], p: usize| -> (usize, usize) {
                (
                    if p == 0 { 0 } else { cnt[0] as usize },
                    cnt[p] as usize,
                )
            };
            let (pme, nme) = blk(self.nc[i], me);
            let (pop, nop) = blk(self.nc[i], op);
            let base = self.roff[i] as usize;
            if n.chance {
                // Draw: one public child. The idle player's reach passes
                // through unchanged; the drawing player's configs transition
                // through the chance matrix, so the chance factor lives in
                // the drawing player's reach and is discarded with it when
                // the leaf values take the counterfactual convention.
                let c = n.child[0];
                debug_assert!(c > i);
                let cbase = self.roff[c] as usize;
                let (cme, _) = blk(self.nc[c], me);
                let (cop, _) = blk(self.nc[c], op);
                let (lo, hi) = self.reach.split_at_mut(cbase);
                let (src, dst) = (&lo[base..], &mut hi[..]);
                dst[cop..cop + nop].copy_from_slice(&src[pop..pop + nop]);
                for ci in 0..nme {
                    let w = src[pme + ci];
                    if w == 0.0 {
                        continue;
                    }
                    let (to, pr) = n.draw.row(ci);
                    for k in 0..to.len() {
                        dst[cme + to[k] as usize] += w * pr[k];
                    }
                }
                continue;
            }
            let cur = &self.cur[self.soff[i] as usize..];
            for ch in 0..n.child.len() {
                let c = n.child[ch];
                debug_assert!(c > i);
                let cbase = self.roff[c] as usize;
                let (cme, _) = blk(self.nc[c], me);
                let (cop, _) = blk(self.nc[c], op);
                let (lo, hi) = self.reach.split_at_mut(cbase);
                let (src, dst) = (&lo[base..], &mut hi[..]);
                // The idle player's information state is untouched, and the
                // child's support for them is the same list.
                dst[cop..cop + nop].copy_from_slice(&src[pop..pop + nop]);
                let (s0, s1) = (n.obs_start[ch] as usize, n.obs_start[ch + 1] as usize);
                for &au in &n.obs_act[s0..s1] {
                    let a = au as usize;
                    for ci in 0..nme {
                        if !n.legal[ci * na + a] {
                            continue;
                        }
                        let t = n.trans[ci * na + a];
                        if t < 0 {
                            continue;
                        }
                        dst[cme + t as usize] += src[pme + ci] * cur[ci * na + a];
                    }
                }
            }
        }
    }

    /// Node `i`'s reach vector for player `p`.
    #[inline]
    fn reach_of(&self, i: usize, p: usize) -> &[f32] {
        let at = self.roff[i] as usize + if p == 1 { self.nc[i][0] as usize } else { 0 };
        &self.reach[at..at + self.nc[i][p] as usize]
    }

    /// Record a leaf in the network batch. Called from `build`, while the
    /// leaf's state is still the one just constructed and therefore still in
    /// cache — walking the finished node array to do this instead meant
    /// re-reading a 700-byte state per leaf out of a half-megabyte tree.
    fn push_leaf(&mut self, id: usize, s: &State, cfgs: &[Rc<[Config]>; 2]) {
        let _t = timed!(PUBFEAT);
        if s.is_terminal() {
            self.term_leaves.push(id);
            return;
        }
        let at = self.leaf_rows.len() * OFF_BELIEF;
        if self.xpub.len() < at + OFF_BELIEF {
            // Grow in chunks so the zero-fill happens a handful of times per
            // solve, and not at all once the pooled buffer is warm.
            self.xpub.resize(at + 64 * OFF_BELIEF, 0.0);
        }
        write_public_features(s, self.ctx, &mut self.xpub[at..at + OFF_BELIEF]);
        self.leaf_res
            .push([reserve(s, 0, self.ctx), reserve(s, 1, self.ctx)]);
        let tab = hands();
        for p in 0..2 {
            self.leaf_hoff.push(self.leaf_hand.len() as u32);
            for c in cfgs[p].iter() {
                self.leaf_hand.push(hand_index_in(tab, &c.hand) as u16);
            }
        }
        self.leaf_rows.push(id);
    }

    /// Push the leaf batch's public half through the network's public tower.
    /// Everything else about the batch was assembled during `build`.
    fn ensure_leaf_batch(&mut self) {
        if self.batch_ready {
            return;
        }
        self.batch_ready = true;
        self.leaf_hoff.push(self.leaf_hand.len() as u32);
        let rows = self.leaf_rows.len();
        if self.xb.len() < rows * SPLIT {
            self.xb.resize(rows * SPLIT, 0.0);
        }
        if !self.nets.value.is_empty() {
            debug_assert_eq!(self.nets.value.in_dim(), FEAT);
            let _t = timed!(PUBNET);
            let xpub = std::mem::take(&mut self.xpub);
            self.nets
                .value
                .trunk(&xpub, rows, OFF_BELIEF, &mut self.sb, &mut self.h0);
            self.xpub = xpub;
        }
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values.
    fn leaf_values(&mut self, traverser: usize) {
        self.ensure_leaf_batch();
        let rows = self.leaf_rows.len();
        {
            let _t = timed!(BELFEAT);
            let (nodes, reach, roff, nc, hoff, hand, res, xb) = (
                &self.nodes,
                &self.reach,
                &self.roff,
                &self.nc,
                &self.leaf_hoff,
                &self.leaf_hand,
                &self.leaf_res,
                &mut self.xb,
            );
            for (r, &i) in self.leaf_rows.iter().enumerate() {
                for p in 0..2 {
                    let (h0, h1) = (hoff[2 * r + p] as usize, hoff[2 * r + p + 1] as usize);
                    let at = r * SPLIT + p * BELIEF_DIM;
                    let ra = roff[i] as usize + if p == 1 { nc[i][0] as usize } else { 0 };
                    write_belief_block(
                        &nodes[i].cfgs[p],
                        Some(&hand[h0..h1]),
                        &reach[ra..ra + nc[i][p] as usize],
                        &res[r][p],
                        &mut xb[at..at + BELIEF_DIM],
                    );
                }
            }
        }
        {
            let _t = timed!(NET);
            let net = &self.nets.value;
            if !net.is_empty() {
                net.forward_split(
                    &self.xb[..rows * SPLIT],
                    rows,
                    &self.h0,
                    traverser * NHAND..(traverser + 1) * NHAND,
                    &mut self.sb,
                    &mut self.ob,
                );
            } else {
                if self.ob.len() < rows * NHAND {
                    self.ob.resize(rows * NHAND, 0.0);
                }
                self.ob[..rows * NHAND].fill(0.0);
            }
        }

        let _t = timed!(LEAFPOST);
        let opp = 1 - traverser;
        for k in 0..self.term_leaves.len() {
            let i = self.term_leaves[k];
            let opp_reach: f32 = self.reach_of(i, opp).iter().sum();
            let u = self.nodes[i].s.utility(traverser);
            let n = self.nc[i][traverser] as usize;
            let vo = self.voff[i] as usize;
            self.vals[vo..vo + n].fill(u * opp_reach);
        }
        let (reach, roff, ncs, voff, hoff, hand, ob, vals) = (
            &self.reach,
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_hoff,
            &self.leaf_hand,
            &self.ob,
            &mut self.vals,
        );
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            let ra = roff[i] as usize + if opp == 1 { ncs[i][0] as usize } else { 0 };
            let opp_reach: f32 = reach[ra..ra + ncs[i][opp] as usize].iter().sum();
            let n = ncs[i][traverser] as usize;
            let off = r * NHAND;
            let hs = hoff[2 * r + traverser] as usize;
            let vo = voff[i] as usize;
            for c in 0..n {
                vals[vo + c] = ob[off + hand[hs + c] as usize] * opp_reach;
            }
        }
    }

    fn update_regrets(&mut self, traverser: usize) {
        // Reaches are already consistent with `cur`: `new` establishes that and
        // every `step` re-establishes it after regret matching, so recomputing
        // them here would repeat the previous pass exactly.
        self.leaf_values(traverser);
        let _t = timed!(BACK);
        for i in (0..self.nodes.len()).rev() {
            if self.nodes[i].leaf {
                continue;
            }
            let (na, me) = (self.nodes[i].na(), self.nodes[i].player as usize);
            let nc = self.nodes[i].nc(traverser);
            if self.nodes[i].chance {
                // Draw pass-through: no regrets, no strategy. If the traverser
                // is the one drawing, their per-config values are pushed
                // through the chance matrix (the probability is a real factor
                // of the value, unlike the traverser's own strategy, which the
                // counterfactual convention discards). If the idle player
                // draws, the traverser's configs are untouched and the
                // opponent's chance factor is already in their reach, which
                // the leaf values carry.
                let ch = self.nodes[i].child[0];
                debug_assert!(ch > i);
                let (vi, vc) = (self.voff[i] as usize, self.voff[ch] as usize);
                let (lo, hi) = self.vals.split_at_mut(vc);
                let (dst, src) = (&mut lo[vi..], &hi[..]);
                if me == traverser {
                    let n = &self.nodes[i];
                    for c in 0..nc {
                        let (to, pr) = n.draw.row(c);
                        let mut v = 0.0;
                        for k in 0..to.len() {
                            v += pr[k] * src[to[k] as usize];
                        }
                        dst[c] = v;
                    }
                } else {
                    dst[..nc].copy_from_slice(&src[..nc]);
                }
                continue;
            }
            let vbase = self.voff[i] as usize;
            self.vals[vbase..vbase + nc].fill(0.0);
            if me == traverser {
                let n = &self.nodes[i];
                let so = self.soff[i] as usize;
                let (regret, cur) = (
                    &mut self.regret[so..],
                    &self.cur[so..],
                );
                // Children are built after their parent, so the parent's value
                // row and every child's are disjoint slices of one arena.
                let (lo, hi) = self.vals.split_at_mut(self.voff[i + 1] as usize);
                let vi = &mut lo[vbase..];
                for a in 0..na {
                    let ch = n.child[n.obs_child[a]];
                    let cv = &hi[self.voff[ch] as usize - self.voff[i + 1] as usize..];
                    for c in 0..nc {
                        if !n.legal[c * na + a] {
                            continue;
                        }
                        let t = n.trans[c * na + a];
                        if t < 0 {
                            continue;
                        }
                        let av = cv[t as usize];
                        regret[c * na + a] += av;
                        vi[c] += av * cur[c * na + a];
                    }
                }
                for c in 0..nc {
                    let base = vi[c];
                    for a in 0..na {
                        if n.legal[c * na + a] {
                            regret[c * na + a] -= base;
                        }
                    }
                }
            } else {
                // The traverser's information state is unchanged across an
                // opponent decision, and the opponent's strategy is already
                // baked into the reach probabilities at the children.
                for ch in 0..self.nodes[i].child.len() {
                    let c_id = self.nodes[i].child[ch];
                    let cv = self.voff[c_id] as usize;
                    for c in 0..nc {
                        self.vals[vbase + c] += self.vals[cv + c];
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
                (self.vals[self.voff[0] as usize + c] - self.root_mean[traverser][c]) * alpha;
        }
        // Linear CFR: discount by t/(t+1) after each update.
        {
            let _t = timed!(RM);
            let m = self.steps[traverser] as f32 + 1.0;
            let disc = m / (m + 1.0);
            let keep_avg = self.cfg.average;
            for i in 0..self.nodes.len() {
                let n = &self.nodes[i];
                if n.leaf || n.chance || n.player as usize != traverser {
                    continue;
                }
                let (na, nc) = (n.na(), n.nc(traverser));
                let so = self.soff[i] as usize;
                let regret = &mut self.regret[so..];
                let cur = &mut self.cur[so..];
                for c in 0..nc {
                    let mut sum = 0.0;
                    for a in 0..na {
                        let v = if n.legal[c * na + a] {
                            regret[c * na + a].max(1e-6)
                        } else {
                            0.0
                        };
                        cur[c * na + a] = v;
                        sum += v;
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for a in 0..na {
                            cur[c * na + a] *= inv;
                        }
                    }
                    for a in 0..na {
                        regret[c * na + a] *= disc;
                    }
                }
                if keep_avg {
                    for x in self.sum_strat[i].iter_mut() {
                        *x *= disc;
                    }
                }
            }
        }
        // Restore the reach probabilities under the strategy just computed:
        // the next iteration's traversal reads them, and so does the average
        // strategy accumulation below.
        self.precompute_reaches();
        if self.cfg.average {
            let _t = timed!(AVG);
            for i in 0..self.nodes.len() {
                let n = &self.nodes[i];
                if n.leaf || n.chance || n.player as usize != traverser {
                    continue;
                }
                let (na, nc) = (n.na(), n.nc(traverser));
                let so = self.soff[i] as usize;
                for c in 0..nc {
                    let r = self.reach_of(i, traverser)[c];
                    let mut sum = 0.0;
                    for a in 0..na {
                        self.sum_strat[i][c * na + a] += r * self.cur[so + c * na + a];
                        sum += self.sum_strat[i][c * na + a];
                    }
                    if sum > 0.0 {
                        for a in 0..na {
                            self.avg[i][c * na + a] = self.sum_strat[i][c * na + a] / sum;
                        }
                    } else {
                        let k = (0..na).filter(|&a| n.legal[c * na + a]).count() as f32;
                        for a in 0..na {
                            self.avg[i][c * na + a] = if n.legal[c * na + a] { 1.0 / k } else { 0.0 };
                        }
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
        let so = self.soff[node] as usize;
        &self.cur[so + c * na..so + (c + 1) * na]
    }

    /// The CFR average strategy: the approximate equilibrium of the subgame.
    /// Only available when the solver was configured to maintain it.
    pub fn average_strategy(&self, node: usize, c: usize) -> &[f32] {
        debug_assert!(self.cfg.average, "solver was built without the average strategy");
        let na = self.nodes[node].na();
        &self.avg[node][c * na..(c + 1) * na]
    }
}
