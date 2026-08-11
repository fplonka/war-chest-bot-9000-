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
//! `facebookresearch/rebel`), with TurboReBeL's reorganisation of the data
//! generation:
//!   * alternating-traverser linear CFR,
//!   * leaf values are *counterfactual* — the network's value for that exact
//!     config, scaled by the opponent's unnormalised reach into that leaf,
//!   * the network is queried with normalised reaches as the beliefs,
//!   * the value target is the root value under the **fixed reference
//!     strategy** — the CFR average at the end of the solve — computed by a
//!     fixed-policy pass (no regrets) per root belief (`value_under`), one for
//!     every belief the walk carries in (`carried_beliefs`);
//!   * acting and belief propagation use the CFR average strategy.
//!
//! That is TurboReBeL's single-sample multi-iteration generation (ICLR 2026,
//! "Turbo ReBeL"): one solve yields T+1 training rows instead of one, all
//! valued under the same reference strategy, so a higher iteration count stops
//! costing data rate. See docs/REBEL.md.
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
use crate::{shape, timed};
use std::rc::Rc;

#[derive(Clone, Copy)]
pub struct Cfg {
    /// Public-tree depth. 1 means "the root's children are leaves".
    pub depth: usize,
    /// CFR iterations (alternating, so each player is traversed iters/2 times).
    pub iters: usize,
    /// Snapshot the CFR average strategy at every iteration. Generation needs
    /// the per-iterate averages for TurboReBeL's carried beliefs (`value_under`
    /// and `carried_beliefs`); evaluation acts on the solved tree and never
    /// looks at an intermediate iterate, so it turns them off.
    pub snapshots: bool,
    /// The regret-update rule.
    pub cfr: Cfr,
    /// Iterations the policy head's strategy is worth when a solve is seeded
    /// from it. 0 starts uniform, which is the default until the measurement
    /// says otherwise.
    pub warm: f32,
    /// Max tree nodes a solve may build. 0 = unlimited. A solve that hits the
    /// cap is flagged `capped` and its caller falls back to a non-search
    /// policy for that decision: the tail of the tree-size distribution
    /// (random-draft roots with broad beliefs at round boundaries) is fat
    /// enough that an unbounded build hangs training for minutes on one
    /// decision.
    pub node_cap: usize,
    /// Build for the GPU service: the tree, features and offsets only. The
    /// CFR arenas (regrets, strategies, reaches, values, snapshots) are
    /// neither allocated nor initialised — the device builds its own from
    /// the job, so doing it here too was pure allocation traffic.
    pub gpu_build: bool,
    /// Keep each node's `State` in `Solver::states`, for tests that assert on
    /// the tree's shape (every leaf terminal, this node is a Warrior Priest
    /// draw, that leaf's hand has the right size). The tree itself dropped the
    /// field because it was 688 of a node's 1,136 bytes across 2,039 nodes for
    /// four read sites, none of them in a hot loop — so this is off in every
    /// production path and the vector stays empty there.
    pub keep_states: bool,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            depth: 1,
            iters: 64,
            snapshots: true,
            cfr: Cfr::LINEAR,
            warm: 0.0,
            node_cap: 0,
            gpu_build: false,
            keep_states: false,
        }
    }
}

/// The exact iterations the per-iterate average strategy is kept at:
/// log-spaced early (0, 1, 2, 4, 8, ...) plus the final one. This list is
/// the runtime metadata of the tree contract — the GPU must not assume
/// powers of two, and any list including 0 and the final iteration is a
/// legal request.
pub fn snapshot_iters(iters: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for t in 0..=iters {
        if t == 0 || t.is_power_of_two() || t == iters {
            out.push(t);
        }
    }
    out
}

/// Which CFR the solver runs.
///
/// Every variant worth comparing is one formula with four numbers. Discounted
/// CFR (Brown & Sandholm 2019) multiplies accumulated *positive* regrets by
/// `t^alpha / (t^alpha + 1)` and negative ones by `t^beta / (t^beta + 1)` each
/// iteration, and contributions to the average strategy by
/// `(t / (t + 1))^gamma`; `predict` adds Predictive CFR+'s optimism, which
/// does regret matching on `R + predict * r` — the regret just observed
/// standing in for the one about to be. So:
///
/// | | alpha | beta | gamma | predict |
/// |---|---|---|---|---|
/// | linear CFR (the reference implementation's) | 1 | 1 | 1 | 0 |
/// | CFR+ (Tammelin 2014) | inf | -inf | 2 | 0 |
/// | DCFR (what TurboReBeL itself runs) | 1.5 | 0 | 2 | 0 |
/// | PCFR+ (Farina et al. 2021) | inf | -inf | 2 | 1 |
/// | SAPCFR+ (Meng et al. 2026) | inf | -inf | 2 | 1/3 |
///
/// `beta = -inf` zeroes negative accumulated regret, which is regret matching+;
/// `alpha = inf` leaves positive regret undiscounted.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Cfr {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub predict: f32,
}

impl Cfr {
    pub const LINEAR: Cfr = Cfr {
        alpha: 1.0,
        beta: 1.0,
        gamma: 1.0,
        predict: 0.0,
    };
    pub const PLUS: Cfr = Cfr {
        alpha: f32::INFINITY,
        beta: f32::NEG_INFINITY,
        gamma: 2.0,
        predict: 0.0,
    };
    pub const DISCOUNTED: Cfr = Cfr {
        alpha: 1.5,
        beta: 0.0,
        gamma: 2.0,
        predict: 0.0,
    };
    pub const PREDICTIVE: Cfr = Cfr {
        alpha: f32::INFINITY,
        beta: f32::NEG_INFINITY,
        gamma: 2.0,
        predict: 1.0,
    };
    pub const SIMPLE_ASYM: Cfr = Cfr {
        alpha: f32::INFINITY,
        beta: f32::NEG_INFINITY,
        gamma: 2.0,
        predict: 1.0 / 3.0,
    };

    /// The five named variants, for the tools that sweep them.
    pub const NAMED: [(&'static str, Cfr); 5] = [
        ("linear", Cfr::LINEAR),
        ("plus", Cfr::PLUS),
        ("dcfr", Cfr::DISCOUNTED),
        ("pcfr", Cfr::PREDICTIVE),
        ("sapcfr", Cfr::SIMPLE_ASYM),
    ];

    pub fn named(name: &str) -> Option<Cfr> {
        Cfr::NAMED.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
    }

    /// `t^p / (t^p + 1)`, with the infinities that name "do not discount" and
    /// "discard entirely" evaluated rather than computed.
    fn factor(t: f32, p: f32) -> f32 {
        if p.is_infinite() {
            return if p > 0.0 { 1.0 } else { 0.0 };
        }
        let x = t.powf(p);
        x / (x + 1.0)
    }
}

/// How well a solve came out, in two numbers that are read together.
#[derive(Clone, Copy, Debug)]
pub struct Conv {
    /// `sum_p (BR_p - v_p)`: what a best response to the solve's own average
    /// strategy would gain, summed over the players. Zero means the strategy is
    /// an equilibrium of the subgame it induces — the fixed point ReBeL
    /// iterates towards — and it is what tells two regret rules apart at a
    /// given iteration count, because unlike a distance to some other solve it
    /// is absolute.
    pub nash: f32,
    /// `v_0 + v_1` at the root. **This is not zero**, and that is not a bug in
    /// the solver. The subgame's leaves are network values, and nothing makes
    /// the network's value for player 0 at a leaf the negative of its value for
    /// player 1 there. So the depth-limited game the solver is handed is only
    /// *approximately* zero-sum, by however far the value network is from
    /// antisymmetric — which is what this measures, and it is a property of the
    /// network rather than of the solve. It vanishes when every leaf is
    /// terminal, which is the case `tests/rebel_solver.rs` pins against an
    /// independent solver.
    ///
    /// How big: over 11,188 positions from 40 games, mean +0.025 and mean
    /// absolute 0.032, against a value spread of 0.416. Call it 8% of the
    /// signal, a third of the network's own error, and ~130x the target bias
    /// that stopping CFR at T=64 introduces. A randomly initialised network is
    /// off by the same amount, so nothing in training creates it and nothing
    /// removes it. See `runs/solvererr_g8/NOTES.md`.
    pub zero_sum: f32,
}

/// What a backward pass over the tree does with the values it computes.
#[derive(Clone, Copy, PartialEq)]
pub enum Back {
    /// CFR: average under the old current strategy, then immediately fold each
    /// action value less the node value into regret matching. The delta is
    /// consumed in the row where it is formed; there is no instantaneous-
    /// regret arena.
    Regret,
    /// Policy-head warm start: form the same per-action delta as CFR and seed
    /// accumulated regret with `weight * delta`, without regret matching away
    /// from the policy being seeded.
    Seed(f32),
    /// Pure value propagation under a fixed strategy — TurboReBeL's Phase 2.
    Value,
    /// The traverser maxes instead of averaging, which makes the root values a
    /// best response to whatever the opponent's reaches were built under.
    BestResponse,
}

/// The value network: `(PBS, config) -> counterfactual value`.
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
    // A node offers at most a few dozen distinct actions, so a linear scan of
    // their encodings beats hashing each one into a fresh table per node.
    let mut seen: Vec<u32> = Vec::new();
    // `set_config` recomputes each slot's reserve from the probe's own bag,
    // hand and face-down counts and redistributes it, and that total is
    // invariant, so one probe can be reconfigured for every slot instead of
    // cloning a 688-byte State per slot.
    let mut probe = *s;
    if matches!(s.pending(), Cont::MainPlay) {
        let res = reserve(s, player, ctx);
        for k in 0..NSLOT {
            if res[k] == 0 {
                continue;
            }
            let mut one = Config::default();
            one.hand[k] = 1;
            set_config(&mut probe, player, ctx, &one);
            if !cfgs.is_empty() && !cfgs.iter().any(|c| c.hand[k] > 0) {
                continue;
            }
            for a in probe.legal_actions() {
                let key = a.encode();
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                let coin = action_coin(&a, &probe);
                let slot = if coin == NONE {
                    -1
                } else {
                    ctx.slot_of[player as usize][coin as usize]
                };
                if slot >= 0 && !cfgs.is_empty() && !cfgs.iter().any(|c| c.hand[slot as usize] > 0)
                {
                    continue;
                }
                aslot.push(slot);
                fdown.push(is_facedown_play(&a));
                acts.push(a);
            }
        }
    } else if matches!(s.pending(), Cont::WarriorPriestPlay { .. }) {
        // A forced play is config-dependent the same way a main play is: the
        // legal set is a function of the config's pending coin. Probe one
        // state per pending slot present in the support; the probe's pending
        // node names the drawn unit so `legal_actions` lists exactly the plays
        // of that coin.
        for k in 0..NSLOT {
            if !cfgs.is_empty() && !cfgs.iter().any(|c| c.pending_coin == Some(k as u8)) {
                continue;
            }
            let mut one = Config::default();
            one.hand[k] = 1;
            set_config(&mut probe, player, ctx, &one);
            probe.pending = Cont::WarriorPriestPlay {
                player,
                coin: ctx.slots[player as usize][k],
            };
            for a in probe.legal_actions() {
                let key = a.encode();
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                let coin = action_coin(&a, &probe);
                let slot = if coin == NONE {
                    -1
                } else {
                    ctx.slot_of[player as usize][coin as usize]
                };
                aslot.push(slot);
                fdown.push(is_facedown_play(&a));
                acts.push(a);
            }
        }
    } else {
        // Micro-decisions spend nothing out of hand: the config is untouched.
        for a in s.legal_actions() {
            let key = a.encode();
            if !seen.contains(&key) {
                seen.push(key);
                acts.push(a);
                aslot.push(-1);
                fdown.push(false);
            }
        }
    }
    (acts, aslot, fdown)
}

pub struct TNode {
    /// Terminal leaves only: the game's utility for `player`. The tree used to
    /// keep the whole `State` here, 688 of a node's 1,136 bytes, for four read
    /// sites -- this, the warm start's decision rows, and two debug asserts.
    /// A mature subgame builds 2,039 nodes, so that was 1.4 MiB per solve
    /// written and then never looked at.
    pub util: f32,
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
    /// Legal actions for the acting player's configs, in CSR form. Config
    /// `c` owns cells `legal_off[c]..legal_off[c + 1]`; actions within a row
    /// retain the public action order.
    pub legal_off: Vec<u32>,
    pub legal_action: Vec<u32>,
    /// The public child node and successor-config index for each legal cell.
    /// `NO_TRANS` is explicit: legality and having an information-state
    /// successor are different facts and must never be inferred from a signed
    /// sentinel.
    pub legal_child: Vec<u32>,
    pub legal_trans: Vec<u32>,
    /// Sparse action-major view used by the CPU oracle. It preserves the old
    /// action-then-config FP32 accumulation order without restoring a dense
    /// config-by-action table. `cell_row` is also the direct row map the wave
    /// kernels need later.
    pub action_off: Vec<u32>,
    pub action_cell: Vec<u32>,
    pub cell_row: Vec<u32>,
}

pub const NO_TRANS: u32 = u32::MAX;

impl TNode {
    #[inline]
    pub fn na(&self) -> usize {
        self.acts.len()
    }
    #[inline]
    pub fn nc(&self, p: usize) -> usize {
        self.cfgs[p].len()
    }

    #[inline]
    pub fn legal_row(&self, c: usize) -> std::ops::Range<usize> {
        self.legal_off[c] as usize..self.legal_off[c + 1] as usize
    }
}

// A subgame's leaf batch runs to a couple of megabytes — the public encodings,
// the cached first layer, the belief blocks, the network scratch — and a game
// builds a fresh solver every couple of decisions. Allocating those buffers per
// solve meant faulting in megabytes of fresh zero pages thousands of times a
// second; the arithmetic that followed was the cheap part. They are handed back
// on drop and reused, capacity and all.
/// The per-solve buffers, pooled *by role*: they differ in size by 5x, so a
/// single shared pool handed each one somebody else's buffer and made it grow
/// — and growth is the one thing that zeroes.
const N_ROLES: usize = 8;
const R_H0: usize = 0;
const R_XPUB: usize = 1;
const R_XB0: usize = 2;
const R_OB: usize = 3;
const R_SB: usize = 4;
const R_CPHI: usize = 5;
const R_CZ: usize = 6;
const R_CG: usize = 7;

thread_local! {
    static BUFS: std::cell::RefCell<[Vec<Vec<f32>>; N_ROLES]> = const {
        std::cell::RefCell::new([
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        ])
    };
}

// The per-solve snapshot arena is a `Vec<Vec<f32>>` (one flat normalised copy
// of the running strategy sums per retained iterate), so it pools separately
// from the flat buffers above.
thread_local! {
    static SNAP_POOL: std::cell::RefCell<Vec<Vec<Vec<f32>>>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

fn take_snaps() -> Vec<Vec<f32>> {
    let mut v = SNAP_POOL.with(|p| p.borrow_mut().pop().unwrap_or_default());
    v.clear();
    v
}

fn give_snaps(v: Vec<Vec<f32>>) {
    if v.is_empty() {
        return;
    }
    SNAP_POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < 2 {
            p.push(v);
        }
    });
}

thread_local! {
    /// One retired node array per thread. A `TNode` is about six hundred
    /// bytes -- a `State`, a `DrawMap` and fourteen more vectors -- and a
    /// mature subgame holds thousands of them, so a fresh `Vec` that doubles
    /// from 640 memcpies the whole array several times per solve and
    /// first-touches megabytes of new pages. That was 6.4 of the 27.3
    /// CPU-milliseconds a mature solve costs, all of it inside `push`.
    static NODE_POOL: std::cell::RefCell<Vec<Vec<TNode>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Retired arrays above this many nodes are dropped rather than kept: one
/// pathological subgame near the 200,000-node cap should not pin hundreds of
/// megabytes per builder thread for the rest of the run.
const NODE_POOL_CAP: usize = 1 << 16;

fn take_nodes() -> Vec<TNode> {
    NODE_POOL.with(|b| b.borrow_mut().pop().unwrap_or_default())
}

fn give_nodes(mut v: Vec<TNode>) {
    if v.capacity() == 0 || v.capacity() > NODE_POOL_CAP {
        return;
    }
    // Cleared, not kept at length: the elements own heap of their own, and a
    // retired solver must not hold it. Only the outer allocation is reused.
    v.clear();
    NODE_POOL.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() < 2 {
            b.push(v);
        }
    });
}

/// The config-interning map is looked up once per leaf, per player, per config
/// in support -- about thirty thousand times per mature solve -- and it is only
/// ever `get` and `insert`, never iterated, so the hash is free to change. The
/// standard hasher is SipHash, which is overkill for a 39-bit packed count
/// vector and was most of the compact-leaf-row phase's cost. One multiply and a
/// shift is enough to spread a key that is already dense.
#[derive(Default, Clone, Copy)]
pub(crate) struct KeyHash;

#[derive(Default)]
pub(crate) struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, n: u64) {
        let x = (n ^ self.0).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = x ^ (x >> 29);
    }
}

impl std::hash::BuildHasher for KeyHash {
    type Hasher = KeyHasher;
    fn build_hasher(&self) -> KeyHasher {
        KeyHasher(0)
    }
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
    /// Owned (it is `Copy`): the walk holds a solver across a whole subgame,
    /// and the GPU path keeps one around while the solve runs on the device,
    /// so the solver cannot borrow a builder-local context.
    pub(crate) ctx: Ctx,
    nets: &'a Nets,
    pub(crate) cfg: Cfg,
    pub nodes: Vec<TNode>,
    /// Node states, parallel to `nodes`. Empty unless `Cfg::keep_states`.
    pub states: Vec<State>,
    pub(crate) root_belief: [Belief; 2],
    /// Regrets and the current regret-matching iterate, flat by node over legal
    /// cells. Node `i` occupies `soff[i] .. soff[i + 1]`; within that range its
    /// config rows are described by `TNode::legal_off`.
    pub(crate) regret: Vec<f32>,
    pub cur: Vec<f32>,
    pub(crate) soff: Vec<u32>,
    /// The reach-weighted running strategy sum, per node. The normalised
    /// average exists only in retained snapshots; keeping a second persistent
    /// arena here doubled the strategy traffic for no useful information.
    pub sum_strat: Vec<Vec<f32>>,
    /// One flat normalised copy of `sum_strat` (per-node regions in node order,
    /// aligned with `soff`) at retained iterations. Snapshot `t` is the average
    /// strategy at iterate t, and the last is the reference strategy
    /// `value_under` and the walk act on. Evaluation retains only the final
    /// copy; generation also retains the requested intermediate copies.
    pub snaps: Vec<Vec<f32>>,
    /// Which snapshot the next `snapshot()` call is (0 = the pre-iteration
    /// average). Drives the log-spaced thinning: the carried beliefs are one
    /// per *kept* iterate, and the spread is in the early ones.
    pub(crate) snap_t: usize,
    /// The kept iteration numbers (`snapshot_iters`); the GPU contract
    /// uploads this list verbatim.
    pub(crate) snap_list: Vec<usize>,
    /// Whether each player's running sum has been normalised at least once.
    /// Until then the historical average is the literal initial iterate, not
    /// a multiply-then-divide reconstruction of it.
    pub(crate) avg_touched: [bool; 2],
    /// Total legal strategy cells across decision nodes, so the
    /// snapshot arenas are reserved to size instead of grown.
    pub ncells: usize,
    /// Reach per config, flat: node `i`'s two players occupy
    /// `reach[roff[i] .. roff[i] + nc0 + nc1]`, player 0 first. One arena
    /// rather than `Vec<Vec<f32>>` — the CFR passes touch every node, and two
    /// pointer hops per node is what they were spending their time on.
    pub reach: Vec<f32>,
    pub(crate) roff: Vec<u32>,
    /// The traverser's counterfactual value per config, flat the same way:
    /// `vals[voff[i] .. voff[i] + max(nc0, nc1)]`.
    pub vals: Vec<f32>,
    pub(crate) voff: Vec<u32>,
    /// `[node]` -> config counts per player, so the hot loops never chase the
    /// `Rc` to ask how long a support is.
    pub(crate) nc: Vec<[u32; 2]>,
    pub(crate) steps: [usize; 2],

    // ---------------------------------------------------------- leaf batch
    // Built once per solve. Everything here is a property of the leaf's public
    // state or its config support, so it survives every CFR iteration; only
    // `xb` (the belief blocks) is rewritten per iteration.
    /// Non-terminal leaves in node order — the rows of the network batch.
    pub leaf_rows: Vec<usize>,
    /// Terminal leaves, scored from the game instead of the network.
    pub(crate) term_leaves: Vec<usize>,
    /// Per row, per player: an index into `cphi` for every config in support,
    /// packed back to back and indexed through `leaf_coff`.
    pub(crate) leaf_cidx: Vec<u32>,
    pub(crate) leaf_coff: Vec<u32>,
    /// The subgame's distinct config vectors, `[n * CFEAT]`, and the map that
    /// deduplicates them. The same config recurs at hundreds of leaves — a
    /// depth-2 subgame has a few hundred leaves over a few dozen distinct
    /// configs — and the config tower is the one part of the network whose cost
    /// scales with the support, so it runs once per distinct config per solve.
    pub(crate) cphi: Vec<f32>,
    pub(crate) cmap: std::collections::HashMap<u64, u32, KeyHash>,
    /// Decision-node states, kept only for a warm start's policy rows.
    warm_states: Vec<(usize, State)>,
    /// Whether the root is a normal coin-play choice, which is all the policy
    /// label's assertion needed the root's state for.
    pub root_mainplay: bool,
    /// How many distinct configs `cphi` actually holds. Pooled buffers keep
    /// their length across solves, so the count cannot be read off `cphi.len()`.
    pub ncfg: usize,
    /// `embed` output for `cphi`: the belief embedding and the readout
    /// embedding. Both survive every CFR iteration.
    pub cz: Vec<f32>,
    pub cg: Vec<f32>,
    /// The card table `[NTYPE, de]`. The draft is fixed for the game, so this is
    /// built once per solve and read by every tower that names a card.
    pub ce: Vec<f32>,
    /// The draft's unit ids in player-major slot order, for the describer's
    /// learned id embedding. Constant per solve.
    pub(crate) ids: [u8; NTYPE],
    /// Decision nodes in the network batch, after the leaves. Only populated
    /// when a warm start needs the policy head at them.
    pub inner_rows: Vec<usize>,
    /// `[rows, ncfg]`: every leaf's PBS vector dotted with every interned config
    /// embedding, rebuilt per readout.
    vt: Vec<f32>,
    /// `rows * hidden`: the public half of the hidden layer.
    pub h0: Vec<f32>,
    /// Width of one public row: the *net's*, not the current encoding's. A
    /// pre-describer checkpoint reads a 972-wide row written by `v1`'s frozen
    /// encoder, so that a gate can play the new architecture against the pool.
    pub(crate) pubfeat: usize,
    /// `rows * pubfeat`: the public encoding, filled during the build.
    pub(crate) xpub: Vec<f32>,
    /// Compact public rows for GPU admission. The device expands these into
    /// the same trunk input; a GPU-built solver never materialises `xpub`.
    pub(crate) gpu_rows: Vec<u8>,
    /// `rows * 2 * dg`: both players' belief embeddings.
    pub xb: Vec<f32>,
    /// `rows * hidden`: the hidden layer, rebuilt per iteration.
    pub ob: Vec<f32>,
    sb: Vec<f32>,
    /// Normalised belief weights for one leaf's support.
    wbuf: Vec<f32>,
    batch_ready: bool,
    /// Traverser of the previous leaf query, i.e. whose beliefs have moved
    /// since. `None` before the first query of a solve.
    last_traverser: Option<usize>,
    /// The build hit `Cfg::node_cap`: the tree is incomplete and the caller
    /// must not solve or walk it.
    capped: bool,
    /// Working memory for the chance transitions, reused across the tree.
    draw_scratch: DrawScratch,
    /// `(config key, legal cell)` scratch for ordering a public child's
    /// support. Keeping the index separate removes the old 24-bit width
    /// assertion; every valid wide job remains representable.
    cell_order: Vec<(u64, u32)>,
}

impl Drop for Solver<'_> {
    fn drop(&mut self) {
        for (role, v) in [
            (R_H0, &mut self.h0),
            (R_XPUB, &mut self.xpub),
            (R_XB0, &mut self.xb),
            (R_OB, &mut self.ob),
            (R_SB, &mut self.sb),
            (R_CPHI, &mut self.cphi),
            (R_CZ, &mut self.cz),
            (R_CG, &mut self.cg),
        ] {
            give_buf(role, std::mem::take(v));
        }
        give_snaps(std::mem::take(&mut self.snaps));
        give_nodes(std::mem::take(&mut self.nodes));
    }
}

impl<'a> Solver<'a> {
    pub(crate) fn network_dims(&self) -> &[usize] {
        &self.nets.value.dims
    }

    pub fn new(
        root: &State,
        ctx: Ctx,
        nets: &'a Nets,
        cfg: Cfg,
        belief: [Belief; 2],
    ) -> Solver<'a> {
        let cfgs: [Rc<[Config]>; 2] = [
            belief[0].cfg.as_slice().into(),
            belief[1].cfg.as_slice().into(),
        ];
        let mut ids = [0u8; NTYPE];
        for t in 0..NTYPE {
            ids[t] = ctx.slots[t / NSLOT][t % NSLOT];
        }
        let mut sv = Solver {
            ctx,
            nets,
            cfg,
            nodes: take_nodes(),
            states: Vec::new(),
            root_belief: belief,
            regret: Vec::new(),
            cur: Vec::new(),
            soff: Vec::new(),
            sum_strat: Vec::new(),
            snaps: take_snaps(),
            snap_t: 0,
            snap_list: snapshot_iters(cfg.iters),
            avg_touched: [false; 2],
            ncells: 0,
            reach: Vec::new(),
            roff: Vec::new(),
            vals: Vec::new(),
            voff: Vec::new(),
            nc: Vec::new(),
            steps: [0, 0],
            leaf_rows: Vec::new(),
            term_leaves: Vec::new(),
            leaf_cidx: Vec::new(),
            leaf_coff: Vec::new(),
            cphi: take_buf(R_CPHI),
            cmap: std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash),
            warm_states: Vec::new(),
            root_mainplay: false,
            ncfg: 0,
            cz: take_buf(R_CZ),
            cg: take_buf(R_CG),
            ce: Vec::new(),
            ids,
            inner_rows: Vec::new(),
            vt: Vec::new(),
            pubfeat: if nets.value.is_empty() {
                PUBFEAT
            } else {
                nets.value.pub_dim()
            },
            h0: take_buf(R_H0),
            xpub: take_buf(R_XPUB),
            gpu_rows: Vec::new(),
            xb: take_buf(R_XB0),
            ob: take_buf(R_OB),
            sb: take_buf(R_SB),
            wbuf: Vec::new(),
            batch_ready: false,
            last_traverser: None,
            capped: false,
            draw_scratch: DrawScratch::default(),
            cell_order: Vec::new(),
        };
        {
            let _t = timed!(BUILD);
            // A depth-2 subgame runs to roughly 550 nodes, so this is one
            // allocation each instead of a doubling sequence. Sizing it larger
            // measures no better: the allocator handles the churn.
            sv.nodes.reserve(640);
            sv.reach.reserve(640);
            sv.vals.reserve(640);
            sv.regret.reserve(640);
            sv.cur.reserve(640);
            sv.root_mainplay = matches!(root.pending(), Cont::MainPlay);
            sv.build(root.clone(), cfg.depth.max(1), cfgs);
        }
        let _t = timed!(ALLOC);
        for i in 0..sv.nodes.len() {
            let n = &sv.nodes[i];
            let p = n.player as usize;
            let nc = n.nc(p);
            let (c0, c1) = (n.nc(0), n.nc(1));
            sv.nc.push([c0 as u32, c1 as u32]);
            sv.soff.push(sv.ncells as u32);
            let cells = n.legal_action.len();
            sv.ncells += cells;
            if cfg.gpu_build {
                // The device owns the arenas; the walk after trip 1 reads
                // only the tree, the features and `soff`.
                continue;
            }
            sv.roff.push(sv.reach.len() as u32);
            sv.reach.resize(sv.reach.len() + c0 + c1, 0.0);
            sv.voff.push(sv.vals.len() as u32);
            sv.vals.resize(sv.vals.len() + c0.max(c1), 0.0);
            sv.regret.resize(sv.ncells, 0.0);
            sv.sum_strat.push(vec![0.0; cells]);
            // CFR starts from a uniform strategy over the legal actions, as in
            // the reference. No heuristic prior is injected here: the greedy
            // knowledge enters through the pretrained value network, which is
            // what CFR actually consumes.
            let mut u = vec![0.0f32; cells];
            if cells > 0 {
                for c in 0..nc {
                    let row = n.legal_row(c);
                    let k = row.len() as f32;
                    for cell in row {
                        u[cell] = 1.0 / k;
                    }
                }
            }
            sv.cur.extend_from_slice(&u);
        }
        sv.soff.push(sv.ncells as u32);
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
        if !cfg.gpu_build {
            sv.precompute_reaches();
            // Seed the strategy sums with one reach-weighted uniform strategy,
            // as `get_uniform_reach_weigted_strategy` does in the reference.
            for i in 0..sv.nodes.len() {
                if sv.nodes[i].leaf || sv.nodes[i].chance || sv.nodes[i].legal_action.is_empty() {
                    continue;
                }
                let p = sv.nodes[i].player as usize;
                let so = sv.soff[i] as usize;
                for c in 0..sv.nodes[i].nc(p) {
                    let r = sv.reach_of(i, p)[c];
                    for cell in sv.nodes[i].legal_row(c) {
                        sv.sum_strat[i][cell] += r * sv.cur[so + cell];
                    }
                }
            }
            // Snapshot 0: the average before any iteration, i.e. the uniform
            // policy — the t = 0 member of the carried-belief set.
            sv.snapshot_initial();
        }
        sv
    }

    /// Retain the literal initial iterate. Its running sum is reach weighted;
    /// normalising that sum is mathematically uniform too, but the multiply
    /// and divide need not round back to the exact same `f32`. The frozen CPU
    /// trajectory contract uses the literal iterate at t = 0.
    fn snapshot_initial(&mut self) {
        debug_assert_eq!(self.snap_t, 0);
        self.snap_t = 1;
        if self.cfg.snapshots || self.cfg.iters == 0 {
            self.snaps.push(self.cur.clone());
        }
    }

    /// Materialise one flat normalised copy of `sum_strat`, aligned with
    /// `soff`: snapshot `t` is the average strategy at iterate t.
    ///
    /// Thinning: the carried beliefs are one per *kept* iterate, and the
    /// spread is in the early iterations — the late ones all repeat the final
    /// average — so only the log-spaced iterates (0, 1, 2, 4, 8, ...) plus the
    /// final one are stored. The final one is the reference strategy Phase 2
    /// and the walk act on (`value_under` reads `snaps.last()`), so it is kept
    /// however many iterations actually run.
    fn snapshot(&mut self) {
        let t = self.snap_t;
        self.snap_t += 1;
        let keep = if self.cfg.snapshots {
            self.snap_list.contains(&t)
        } else {
            // Evaluation does not carry intermediate beliefs, but acting still
            // needs the final average. No persistent `avg` arena is kept just
            // to bridge the iterations before it exists.
            t == self.cfg.iters
        };
        if !keep {
            return;
        }
        // `cur` still contains the literal initial policy for a player that has
        // not traversed yet. Start there so their historical average stays
        // byte-identical; overwrite every player whose sum has been updated.
        let mut snap = self.cur.clone();
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance || !self.avg_touched[n.player as usize] {
                continue;
            }
            let so = self.soff[i] as usize;
            let nc = n.nc(n.player as usize);
            for c in 0..nc {
                let row = n.legal_row(c);
                let sum: f32 = self.sum_strat[i][row.clone()].iter().sum();
                let k = row.len().max(1) as f32;
                for cell in row {
                    snap[so + cell] = if sum > 0.0 {
                        self.sum_strat[i][cell] / sum
                    } else {
                        1.0 / k
                    };
                }
            }
        }
        self.snaps.push(snap);
    }

    // ------------------------------------------------------------ tree build

    fn build(&mut self, s: State, depth: usize, cfgs: [Rc<[Config]>; 2]) -> usize {
        let player = s.to_act();
        // A draw is walked through (one public child, no depth cost): the
        // outcome is private, so the public tree does not branch. Round-start
        // draws collapse over a whole run; a Warrior Priest draw is a single
        // chance node whose children carry the pending forced-play coin.
        // Depth-0 nodes and terminals are leaves.
        let draw_pass = matches!(
            s.pending(),
            Cont::Draw { .. } | Cont::WarriorPriestDraw { .. }
        );
        // Depth counts completed coin plays. A main-play node spends exactly
        // one coin per legal action and every micro node's actions spend
        // nothing, so "is this a main play?" is the whole story: a micro
        // node at depth 0 (cavalry: move, then choose the attack) rides free,
        // a main-play node at depth 0 is a leaf. Without this, "depth 2"
        // would sometimes contain zero opponent moves, because a compound
        // tactic is several decision nodes for one coin.
        let mainplay = matches!(s.pending(), Cont::MainPlay);
        let mut leaf = s.is_terminal() || (!draw_pass && s.is_chance());
        if !leaf && !draw_pass && depth == 0 {
            leaf = mainplay;
        }
        let id = self.nodes.len();
        if self.cfg.keep_states {
            self.states.push(s.clone());
        }
        let _tp = timed!(BPUSH);
        self.nodes.push(TNode {
            util: if leaf && s.is_terminal() {
                s.utility(player as usize)
            } else {
                0.0
            },
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
            legal_off: Vec::new(),
            legal_action: Vec::new(),
            legal_child: Vec::new(),
            legal_trans: Vec::new(),
            action_off: Vec::new(),
            action_cell: Vec::new(),
            cell_row: Vec::new(),
        });
        drop(_tp);
        if self.cfg.node_cap > 0 && self.nodes.len() >= self.cfg.node_cap {
            // Pathological root: stop expanding. The node stays a stub (no
            // actions, no children); the caller checks `capped` and falls
            // back to a non-search policy, so nothing here is ever walked.
            self.capped = true;
            return id;
        }
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
            let mut support: Vec<Config> = Vec::new();
            let mut draw = DrawMap::default();
            // The reserve and the face-up pile are what the draws read, and a
            // draw changes neither (a refill does, and `run` accounts for it
            // internally), so both come from the state at the head of the run.
            let res = reserve(&cs, player, &self.ctx);
            let fu = faceup_counts(&cs, player, &self.ctx);
            // A Warrior Priest draw's children carry the forced-play coin.
            let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
            let mut steps = 0u8;
            loop {
                // `drawable` is order-preserving and this is its first entry;
                // `None` is the Warrior Priest fizzle, which `legal_actions`
                // represents the same way.
                let unit = cs.first_drawable(player).unwrap_or(crate::board::NONE);
                cs.apply_inplace(Action::DrawCoin { unit });
                steps += 1;
                if !(matches!(cs.pending(), Cont::Draw { .. }) && cs.to_act() == player) {
                    break;
                }
            }
            if wp {
                debug_assert_eq!(steps, 1, "a WP draw is a single forced draw");
                self.draw_scratch
                    .transition(&cfgs[me], &res, &fu, &mut support, &mut draw, true);
            } else {
                self.draw_scratch
                    .run(&cfgs[me], &res, &fu, steps, &mut support, &mut draw);
            }
            drop(td);
            let mut cc = cfgs;
            cc[me] = support.as_slice().into();
            let ch = self.build(cs, depth, cc);
            let n = &mut self.nodes[id];
            n.chance = true;
            n.child = vec![ch];
            n.draw = draw;
            n.draw_steps = steps;
            return id;
        }

        // Only a warm start needs a decision node's state after the build, and
        // it needs them in node order, which is push order.
        if self.cfg.warm > 0.0 {
            self.warm_states.push((id, s));
        }
        let me = player as usize;
        let mine = cfgs[me].clone();
        let nc = mine.len();
        let ta = timed!(BACTS);
        let (acts, aslot, fdown) = node_actions(&s, player, &self.ctx, &mine);
        drop(ta);
        let na = acts.len();
        debug_assert!(na > 0, "a decision node must offer a reachable action");

        // A Warrior Priest forced play may only spend the pending coin, so the
        // per-config mask is the pending match rather than the hand check.
        let wp_play = matches!(s.pending(), Cont::WarriorPriestPlay { .. });
        let tcells = timed!(BCELLS);
        let mut legal_off = Vec::with_capacity(nc + 1);
        let mut legal_action = Vec::new();
        let mut legal_child = Vec::new();
        let mut legal_trans = Vec::new();
        let mut cell_row = Vec::new();
        legal_off.push(0);
        for (ci, c) in mine.iter().enumerate() {
            for a in 0..na {
                let legal = if wp_play {
                    c.pending_coin == Some(aslot[a] as u8)
                } else {
                    action_legal(c, aslot[a])
                };
                if legal {
                    legal_action.push(a as u32);
                    legal_child.push(0);
                    legal_trans.push(NO_TRANS);
                    cell_row.push(ci as u32);
                }
            }
            legal_off.push(legal_action.len() as u32);
        }

        drop(tcells);

        // Group private actions by what the opponent actually observes.
        let tobs = timed!(BOBS);
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

        // Patch each legal cell to its observation-local child and build a
        // sparse action-major view. The primary representation remains
        // config-major CSR; this second ordering preserves the CPU oracle's
        // action-then-config FP32 accumulation order.
        for (cell, &au) in legal_action.iter().enumerate() {
            legal_child[cell] = obs_child[au as usize] as u32;
        }
        let mut action_off = vec![0u32; na + 1];
        for &a in &legal_action {
            action_off[a as usize + 1] += 1;
        }
        for a in 0..na {
            action_off[a + 1] += action_off[a];
        }
        let mut action_fill = action_off.clone();
        let mut action_cell = vec![0u32; legal_action.len()];
        for (cell, &a) in legal_action.iter().enumerate() {
            let at = &mut action_fill[a as usize];
            action_cell[*at as usize] = cell as u32;
            *at += 1;
        }

        drop(tobs);

        // Config support of each public child — the union over the private
        // actions that produce that observation — and, in the same pass, where
        // each (config, action) cell lands in it.
        //
        // Ordering by integer key and reading the support off that single
        // ordering is what the draw transitions do, and for the same reason:
        // the obvious version sorts `Config`s and then binary-searches one per
        // cell, which at ~800 cells per decision node is most of the build.
        let tsup = timed!(BSUP);
        let mut child_cfgs: Vec<Vec<Config>> = vec![Vec::new(); nch];
        let mut ent = std::mem::take(&mut self.cell_order);
        for ch in 0..nch {
            ent.clear();
            for &au in &obs_act[obs_start[ch] as usize..obs_start[ch + 1] as usize] {
                let a = au as usize;
                for &cell_u in &action_cell[action_off[a] as usize..action_off[a + 1] as usize] {
                    let cell = cell_u as usize;
                    if legal_child[cell] as usize != ch {
                        continue;
                    }
                    let ci = cell_row[cell] as usize;
                    if let Some(n) = advance_config(&mine[ci], aslot[a], fdown[a]) {
                        ent.push((n.key(), cell_u));
                    }
                }
            }
            ent.sort_unstable_by_key(|&(key, cell)| (key, cell));
            let sup = &mut child_cfgs[ch];
            let mut prev = u64::MAX;
            for &(k, cell_u) in ent.iter() {
                let cell = cell_u as usize;
                if k != prev {
                    prev = k;
                    let ci = cell_row[cell] as usize;
                    let a = legal_action[cell] as usize;
                    sup.push(advance_config(&mine[ci], aslot[a], fdown[a]).unwrap());
                }
                legal_trans[cell] = (sup.len() - 1) as u32;
            }
        }
        self.cell_order = ent;
        drop(tsup);

        // One world per public child, built from any config that can produce
        // it: the public projection of the successor is the same either way.
        let mut child = Vec::with_capacity(nch);
        for ch in 0..nch {
            let a = obs_act[obs_start[ch] as usize] as usize;
            let rep = *mine
                .iter()
                .find(|c| action_legal(c, aslot[a]))
                .expect("a kept action is playable by some config in the support");
            let tb = timed!(BAPPLY);
            let mut cs = s.clone();
            set_config(&mut cs, player, &self.ctx, &rep);
            cs.apply_inplace(acts[a]);
            drop(tb);
            let mut cc = cfgs.clone();
            cc[me] = std::mem::take(&mut child_cfgs[ch]).into();
            // One depth unit per *completed coin play*, not per decision node.
            // A main-play node spends exactly one coin and consumes a depth
            // unit; a Warrior Priest forced play also spends a coin but rides
            // free (the chain belongs to the coin play that triggered it);
            // every micro node spends nothing. The node-level structural test
            // decides the whole observation group, and the debug assertion
            // keeps it honest against `action_coin`.
            let spends = matches!(s.pending(), Cont::MainPlay);
            debug_assert_eq!(
                matches!(s.pending(), Cont::MainPlay)
                    || matches!(s.pending(), Cont::WarriorPriestPlay { .. }),
                obs_act[obs_start[ch] as usize..obs_start[ch + 1] as usize]
                    .iter()
                    .any(|&au| action_coin(&acts[au as usize], &s) != NONE),
                "structural mainplay test diverged from action_coin"
            );
            child.push(self.build(cs, depth - usize::from(spends), cc));
        }

        let n = &mut self.nodes[id];
        n.acts = acts;
        n.aslot = aslot;
        n.fdown = fdown;
        n.obs_child = obs_child;
        n.obs_start = obs_start;
        n.obs_act = obs_act;
        n.child = child;
        for c in &mut legal_child {
            *c = n.child[*c as usize] as u32;
        }
        n.legal_off = legal_off;
        n.legal_action = legal_action;
        n.legal_child = legal_child;
        n.legal_trans = legal_trans;
        n.action_off = action_off;
        n.action_cell = action_cell;
        n.cell_row = cell_row;
        id
    }

    // -------------------------------------------------------------- CFR core

    /// Push reach probabilities down the tree under the current strategies.
    ///
    /// Children are always built after their parent, so `child > parent` and
    /// the parent's row can be borrowed alongside the child's through one
    /// `split_at_mut` — no copy of the parent's reach, which used to be two
    /// heap allocations per node per pass.
    pub fn precompute_reaches(&mut self) {
        let cur = std::mem::take(&mut self.cur);
        let root = [self.root_belief[0].p.clone(), self.root_belief[1].p.clone()];
        self.propagate(&cur, [&root[0], &root[1]]);
        self.cur = cur;
    }

    /// Push reach probabilities down the tree under `strat` (a flat arena in
    /// the same per-node layout as `cur` and the snapshots, aligned with
    /// `soff`), from the given root beliefs.
    fn propagate(&mut self, strat: &[f32], root: [&[f32]; 2]) {
        let _t = timed!(REACH);
        self.reach.fill(0.0);
        for p in 0..2 {
            let at = self.roff[0] as usize + if p == 1 { self.nc[0][0] as usize } else { 0 };
            let n = self.nc[0][p] as usize;
            self.reach[at..at + n].copy_from_slice(root[p]);
        }
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf {
                continue;
            }
            let me = n.player as usize;
            let op = 1 - me;
            // Offsets of each player's block inside a node's reach region.
            let blk = |cnt: [u32; 2], p: usize| -> (usize, usize) {
                (if p == 0 { 0 } else { cnt[0] as usize }, cnt[p] as usize)
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
            let cur = &strat[self.soff[i] as usize..];
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
                    for &cell_u in
                        &n.action_cell[n.action_off[a] as usize..n.action_off[a + 1] as usize]
                    {
                        let cell = cell_u as usize;
                        debug_assert_eq!(n.legal_child[cell] as usize, c);
                        let t = n.legal_trans[cell];
                        if t == NO_TRANS {
                            continue;
                        }
                        let ci = n.cell_row[cell] as usize;
                        dst[cme + t as usize] += src[pme + ci] * cur[cell];
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

    /// One public row.
    fn encode(&mut self, s: &State, at: usize) {
        let pf = self.pubfeat;
        write_public_features(s, &self.ctx, &mut self.xpub[at..at + pf]);
    }

    /// Record a leaf in the network batch. Called from `build`, while the
    /// leaf's state is still the one just constructed and therefore still in
    /// cache — walking the finished node array to do this instead meant
    /// re-reading a 700-byte state per leaf out of a half-megabyte tree.
    fn push_leaf(&mut self, id: usize, s: &State, cfgs: &[Rc<[Config]>; 2]) {
        if s.is_terminal() {
            self.term_leaves.push(id);
            return;
        }
        self.push_row(id, s, cfgs);
        self.leaf_rows.push(id);
    }

    /// One row of the network batch: its public encoding, and its configs
    /// interned into the shared table. Leaves and — when a warm start needs
    /// them — decision nodes go through here alike.
    fn push_row(&mut self, _id: usize, s: &State, cfgs: &[Rc<[Config]>; 2]) {
        // The network is queried only at normal coin-play states: a subgame
        // finishes every tactic, trigger and forced play before a leaf.
        debug_assert!(
            matches!(s.pending(), Cont::MainPlay),
            "a network row must be a MainPlay state"
        );
        let _t = timed!(PUBFEAT);
        let row = self.leaf_coff.len() / 2;
        let raw_at = row * crate::rebel::GPU_ROW_BYTES;
        if self.gpu_rows.len() < raw_at + crate::rebel::GPU_ROW_BYTES {
            self.gpu_rows
                .resize(raw_at + 64 * crate::rebel::GPU_ROW_BYTES, 0);
        }
        crate::rebel::pack_gpu_row(
            s,
            &self.ctx,
            &mut self.gpu_rows[raw_at..raw_at + crate::rebel::GPU_ROW_BYTES],
        );
        // The CPU solver needs the expanded public feature batch. A GPU solve
        // does not: expanding 897 floats here only to upload and immediately
        // rearrange them was its largest host and PCIe cost.
        if !self.cfg.gpu_build {
            let pf = self.pubfeat;
            let at = row * pf;
            if self.xpub.len() < at + pf {
                // Grow in chunks so the zero-fill happens a handful of times
                // per solve, and not at all once the pooled buffer is warm.
                self.xpub.resize(at + 64 * pf, 0.0);
            }
            self.encode(s, at);
        }
        for p in 0..2 {
            let res = reserve(s, p as u8, &self.ctx);
            self.leaf_coff.push(self.leaf_cidx.len() as u32);
            for c in cfgs[p].iter() {
                let idx = self.intern_config(c, &res, p);
                self.leaf_cidx.push(idx);
            }
        }
    }

    /// Index of one config's feature vector in `cphi`, adding it if new.
    ///
    /// The key is the raw counts the vector is built from — hand, face-down and
    /// bag, four bits each, plus the seat — so two configs share a row exactly
    /// when their feature vectors are identical. Keying on the `Config` alone
    /// would be wrong: the bag depends on the node's reserve, which changes as
    /// coins leave it.
    fn intern_config(&mut self, c: &Config, res: &[u8; NSLOT], p: usize) -> u32 {
        let mut cnt = [0u8; CCOUNTS];
        config_counts(c, res, &mut cnt);
        let mut key = p as u64;
        for x in cnt.iter() {
            debug_assert!(*x < 16, "count over the key width");
            key = (key << 4) | *x as u64;
        }
        if let Some(&i) = self.cmap.get(&key) {
            return i;
        }
        let i = self.ncfg as u32;
        self.ncfg += 1;
        let at = i as usize * CFEAT;
        if self.cphi.len() < at + CFEAT {
            self.cphi.resize(at + 64 * CFEAT, 0.0);
        }
        for k in 0..CCOUNTS {
            self.cphi[at + k] = cnt[k] as f32 / CNORM;
        }
        self.cphi[at + CCOUNTS] = p as f32;
        self.cmap.insert(key, i);
        i
    }

    /// Everything about the batch that does not move between CFR iterations:
    /// the public tower, and the config tower over every distinct config in the
    /// tree. Both are pure functions of the subgame, so this runs once.
    fn ensure_leaf_batch(&mut self) {
        if self.batch_ready {
            return;
        }
        self.batch_ready = true;
        if self.nets.value.is_empty() {
            self.leaf_coff.push(self.leaf_cidx.len() as u32);
            return;
        }
        // A warm start reads the policy head at every decision node, so those
        // nodes join the same batch as the leaves — appended after them, so a
        // leaf's row index is still its position in `leaf_rows` and nothing on
        // the value path moves. They then share the card table, the one public
        // tower pass and the one config-tower pass, instead of running their own
        // of each. Nothing is pushed when the warm start is off.
        if self.cfg.warm > 0.0 {
            let states = std::mem::take(&mut self.warm_states);
            for &(i, ref st) in &states {
                // The policy head is read only at normal coin-play choices.
                if !matches!(st.pending(), Cont::MainPlay) {
                    continue;
                }
                self.push_row(i, st, &self.nodes[i].cfgs.clone());
                self.inner_rows.push(i);
            }
            self.warm_states = states;
        }
        self.leaf_coff.push(self.leaf_cidx.len() as u32);
        let (leaves, rows) = (
            self.leaf_rows.len(),
            self.leaf_rows.len() + self.inner_rows.len(),
        );
        let net = &self.nets.value;
        debug_assert_eq!(net.pub_dim(), self.pubfeat);
        debug_assert_eq!(net.cfeat(), CFEAT);
        if self.xb.len() < leaves * net.belief_dim() {
            self.xb.resize(leaves * net.belief_dim(), 0.0);
        }
        shape!(NCFG, self.ncfg);
        let _t = timed!(PUBNET);
        let xpub = std::mem::take(&mut self.xpub);
        // The cards in play are fixed at the draft, so every row of the subgame
        // carries the same card block and the table is built once. Everything
        // downstream — the hex block, the pile summary, the holding tower, the
        // action tower — reads a row of it by coin-type index.
        if rows > 0 {
            net.cards(&xpub[..self.pubfeat], &self.ids, &mut self.ce);
        }
        net.trunk(
            &xpub,
            rows,
            self.pubfeat,
            &self.ce,
            &mut self.sb,
            &mut self.h0,
        );
        self.xpub = xpub;
        let cphi = std::mem::take(&mut self.cphi);
        net.embed(
            &cphi[..self.ncfg * CFEAT],
            self.ncfg,
            &self.ce,
            &mut self.cz,
            &mut self.cg,
        );
        self.cphi = cphi;
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values.
    pub fn leaf_values(&mut self, traverser: usize) {
        self.ensure_leaf_batch();
        let rows = self.leaf_rows.len();
        let empty = self.nets.value.is_empty();
        let dg = if empty { 0 } else { self.nets.value.dg() };
        // Only one player's beliefs have moved since the previous query: the
        // one whose strategy regret matching updated at the end of the last
        // step. The other player's embedding is still exactly what it was, so
        // it is not rewritten.
        let redo = self.last_traverser;
        self.last_traverser = Some(traverser);
        if !empty {
            let _t = timed!(BELFEAT);
            let (reach, roff, nc, coff, cidx, cz, wbuf, xb) = (
                &self.reach,
                &self.roff,
                &self.nc,
                &self.leaf_coff,
                &self.leaf_cidx,
                &self.cz,
                &mut self.wbuf,
                &mut self.xb,
            );
            for (r, &i) in self.leaf_rows.iter().enumerate() {
                for p in 0..2 {
                    if redo.is_some_and(|l| l != p) {
                        continue;
                    }
                    let n = nc[i][p] as usize;
                    let ra = roff[i] as usize + if p == 1 { nc[i][0] as usize } else { 0 };
                    // The belief the network reads is the normalised reach, as
                    // in the reference -- but as a weighted sum of the same
                    // config embeddings the value readout uses, so a config is
                    // described to the network exactly one way.
                    if wbuf.len() < n {
                        wbuf.resize(n, 0.0);
                    }
                    normalize_weights(&reach[ra..ra + n], &mut wbuf[..n]);
                    let at = r * 2 * dg + p * dg;
                    let cs = coff[2 * r + p] as usize;
                    crate::net::accumulate(
                        cz,
                        &cidx[cs..cs + n],
                        &wbuf[..n],
                        dg,
                        &mut xb[at..at + dg],
                    );
                }
            }
        }
        if !empty {
            let _t = timed!(NET);
            let net = &self.nets.value;
            net.pbs_head(
                &self.xb[..rows * 2 * dg],
                rows,
                &self.h0,
                &mut self.sb,
                &mut self.ob,
            );
        }
        self.readout(traverser);
    }

    /// Refresh *both* players' belief blocks and run the PBS head once. The
    /// fixed-policy passes of TurboReBeL's Phase 2 seed a different root
    /// belief per pass, so both blocks move and the alternating-traverser
    /// cache of `leaf_values` does not apply. The per-config readout is left
    /// to `readout`, which may run twice off the same `ob`.
    fn leaf_values_both(&mut self) {
        self.ensure_leaf_batch();
        let rows = self.leaf_rows.len();
        let empty = self.nets.value.is_empty();
        let dg = if empty { 0 } else { self.nets.value.dg() };
        self.last_traverser = None;
        if empty {
            return;
        }
        let _t = timed!(BELFEAT);
        let (reach, roff, nc, coff, cidx, cz, wbuf, xb) = (
            &self.reach,
            &self.roff,
            &self.nc,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cz,
            &mut self.wbuf,
            &mut self.xb,
        );
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            for p in 0..2 {
                let n = nc[i][p] as usize;
                let ra = roff[i] as usize + if p == 1 { nc[i][0] as usize } else { 0 };
                if wbuf.len() < n {
                    wbuf.resize(n, 0.0);
                }
                normalize_weights(&reach[ra..ra + n], &mut wbuf[..n]);
                let at = r * 2 * dg + p * dg;
                let cs = coff[2 * r + p] as usize;
                crate::net::accumulate(cz, &cidx[cs..cs + n], &wbuf[..n], dg, &mut xb[at..at + dg]);
            }
        }
        let _t = timed!(NET);
        let net = &self.nets.value;
        net.pbs_head(
            &self.xb[..rows * 2 * dg],
            rows,
            &self.h0,
            &mut self.sb,
            &mut self.ob,
        );
    }

    /// Per-config leaf values for player `p` — counterfactual: the network's
    /// value for that exact config times the opponent's unnormalised reach
    /// into the leaf. Runs off the `ob` left by the last `leaf_values` /
    /// `leaf_values_both`, so two players can be read off one PBS-head pass.
    pub fn readout(&mut self, p: usize) {
        let _t = timed!(LEAFPOST);
        let empty = self.nets.value.is_empty();
        let opp = 1 - p;
        for k in 0..self.term_leaves.len() {
            let i = self.term_leaves[k];
            let opp_reach: f32 = self.reach_of(i, opp).iter().sum();
            // Zero-sum by construction (`state::horizon_tests`), so one
            // stored value serves both seats.
            let u = if p == self.nodes[i].player as usize {
                self.nodes[i].util
            } else {
                -self.nodes[i].util
            };
            let n = self.nc[i][p] as usize;
            let vo = self.voff[i] as usize;
            self.vals[vo..vo + n].fill(u * opp_reach);
        }
        let rk = if empty { 0 } else { self.nets.value.rank() };
        // The readout is a dot product of the leaf's PBS vector with each of its
        // configs' embeddings. Done leaf by leaf it is a few thousand short
        // vectorised dots per iteration; done as one matmul against the whole
        // interned table it is ~7x the arithmetic but runs on the matrix
        // coprocessor, which is worth far more than 7x. A solve interns ~160
        // distinct configs against ~17k slots, so the table is small and the
        // result stays in cache.
        if !empty {
            let _t = timed!(LEAFDOT);
            let rows = self.leaf_rows.len();
            crate::net::fit(&mut self.vt, rows * self.ncfg);
            crate::net::dots(
                &self.ob[..rows * rk],
                rk,
                &self.cg[..self.ncfg * (rk + 1)],
                rk + 1,
                rows,
                self.ncfg,
                &mut self.vt,
            );
        }
        let (reach, roff, ncs, voff, coff, cidx, cg, vt, vals) = (
            &self.reach,
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cg,
            &self.vt,
            &mut self.vals,
        );
        let ncfg = self.ncfg;
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            let ra = roff[i] as usize + if opp == 1 { ncs[i][0] as usize } else { 0 };
            let opp_reach: f32 = reach[ra..ra + ncs[i][opp] as usize].iter().sum();
            let n = ncs[i][p] as usize;
            let vo = voff[i] as usize;
            if empty {
                vals[vo..vo + n].fill(0.0);
                continue;
            }
            // Only the player's own configs are ever looked up here. The
            // opponent's private state reaches this value solely through the
            // belief embedding, which is what keeps the query leak-free.
            let cs = coff[2 * r + p] as usize;
            let row = &vt[r * ncfg..(r + 1) * ncfg];
            for (v, &c) in vals[vo..vo + n].iter_mut().zip(&cidx[cs..cs + n]) {
                // The trailing column of `g` is the per-config bias term.
                *v = (row[c as usize] + cg[c as usize * (rk + 1) + rk]) * opp_reach;
            }
        }
    }

    fn update_regrets(&mut self, traverser: usize) {
        // Reaches are already consistent with `cur`: `new` establishes that,
        // every `step` re-establishes it after regret matching, and the
        // fixed-policy passes restore it before returning, so recomputing them
        // here would repeat the previous pass exactly.
        self.leaf_values(traverser);
        self.backprop(traverser, &[], Back::Regret);
    }

    /// One value backpropagation over the tree for `traverser`: the shared walk
    /// behind CFR (`update_regrets`), TurboReBeL's fixed-policy passes
    /// (`value_under`) and the best response (`nash_conv`). `mode` picks what
    /// the traverser's own decision nodes do with their children's values —
    /// average under `strat`, average and immediately update regret matching,
    /// seed warm-start regret, or take the max. Regret modes use `self.cur` in
    /// place; fixed-policy modes read `strat`.
    pub fn backprop(&mut self, traverser: usize, strat: &[f32], mode: Back) {
        // Regret matching floors at EPS rather than at zero, so every legal
        // action keeps positive probability and carried beliefs keep their
        // full support. The factors are constant for this whole traversal.
        const EPS: f32 = 1e-6;
        let rm = if mode == Back::Regret {
            let k = self.cfg.cfr;
            let m = self.steps[traverser] as f32 + 1.0;
            Some((
                k,
                Cfr::factor(m, k.alpha),
                Cfr::factor(m, k.beta),
                (m / (m + 1.0)).powf(k.gamma),
            ))
        } else {
            None
        };
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
            // A best response takes a max at the traverser's own nodes, so
            // those start below every candidate; a config with no legal action
            // there is put back to zero below. Every other node accumulates.
            let br = mode == Back::BestResponse && me == traverser;
            self.vals[vbase..vbase + nc].fill(if br { f32::NEG_INFINITY } else { 0.0 });
            if me == traverser {
                let n = &self.nodes[i];
                let so = self.soff[i] as usize;
                // Children are built after their parent, so the parent's value
                // row and every child's are disjoint slices of one arena.
                let (lo, hi) = self.vals.split_at_mut(self.voff[i + 1] as usize);
                let vi = &mut lo[vbase..];
                for a in 0..na {
                    let ch = n.child[n.obs_child[a]];
                    let cv = &hi[self.voff[ch] as usize - self.voff[i + 1] as usize..];
                    for &cell_u in
                        &n.action_cell[n.action_off[a] as usize..n.action_off[a + 1] as usize]
                    {
                        let cell = cell_u as usize;
                        debug_assert_eq!(n.legal_child[cell] as usize, ch);
                        let t = n.legal_trans[cell];
                        if t == NO_TRANS {
                            continue;
                        }
                        let c = n.cell_row[cell] as usize;
                        let av = cv[t as usize];
                        match mode {
                            Back::Regret | Back::Seed(_) => vi[c] += av * self.cur[so + cell],
                            Back::Value => vi[c] += av * strat[so + cell],
                            Back::BestResponse => vi[c] = vi[c].max(av),
                        }
                    }
                }
                match mode {
                    Back::Regret => {
                        let (k, da, db, dg) = rm.expect("regret factors");
                        for c in 0..nc {
                            let base = vi[c];
                            let row = n.legal_row(c);
                            let mut sum = 0.0;
                            for cell in row.clone() {
                                // Re-form this row-local action value rather
                                // than retain an arena of them between phases.
                                // Starting at +0 and adding preserves the old
                                // `inst[cell] += av` FP32 operation exactly,
                                // including an explicit no-successor cell.
                                let mut delta = 0.0f32;
                                let t = n.legal_trans[cell];
                                if t != NO_TRANS {
                                    let ch = n.legal_child[cell] as usize;
                                    let cv = self.voff[ch] as usize - self.voff[i + 1] as usize;
                                    delta += hi[cv + t as usize];
                                }
                                delta -= base;
                                let at = so + cell;
                                let old = self.regret[at];
                                let r = old * if old > 0.0 { da } else { db } + delta;
                                self.regret[at] = r;
                                let v = (r + k.predict * delta).max(EPS);
                                self.cur[at] = v;
                                sum += v;
                            }
                            if sum > 0.0 {
                                let inv = 1.0 / sum;
                                for cell in row {
                                    self.cur[so + cell] *= inv;
                                }
                            }
                        }
                        for x in self.sum_strat[i].iter_mut() {
                            *x *= dg;
                        }
                    }
                    Back::Seed(weight) => {
                        for c in 0..nc {
                            let base = vi[c];
                            for cell in n.legal_row(c) {
                                let mut delta = 0.0f32;
                                let t = n.legal_trans[cell];
                                if t != NO_TRANS {
                                    let ch = n.legal_child[cell] as usize;
                                    let cv = self.voff[ch] as usize - self.voff[i + 1] as usize;
                                    delta += hi[cv + t as usize];
                                }
                                delta -= base;
                                self.regret[so + cell] = weight * delta;
                            }
                        }
                    }
                    Back::BestResponse => {
                        for c in 0..nc {
                            if vi[c] == f32::NEG_INFINITY {
                                vi[c] = 0.0;
                            }
                        }
                    }
                    Back::Value => {}
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
        // Restore the reach probabilities under the strategy just computed:
        // the next iteration's traversal reads them, and so does the average
        // strategy accumulation below.
        self.precompute_reaches();
        self.avg_block(traverser);
        self.snapshot();
        self.steps[traverser] += 1;
    }

    /// Add the fresh reach-weighted iterate to the running strategy sum.
    /// Normalisation is deferred until a retained snapshot is materialised.
    pub fn avg_block(&mut self, traverser: usize) {
        let _t = timed!(AVG);
        self.avg_touched[traverser] = true;
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance || n.player as usize != traverser {
                continue;
            }
            let nc = n.nc(traverser);
            let so = self.soff[i] as usize;
            for c in 0..nc {
                let r = self.reach_of(i, traverser)[c];
                for cell in n.legal_row(c) {
                    self.sum_strat[i][cell] += r * self.cur[so + cell];
                }
            }
        }
    }

    pub fn multistep(&mut self, iters: usize) {
        for i in 0..iters {
            self.step(i % 2);
        }
    }

    /// TurboReBeL Phase 2: for each root belief, the per-config root values
    /// under the fixed reference strategy — the CFR average at the end of the
    /// solve (the last snapshot). Reach is propagated under the reference and
    /// values backed up under it; no regrets move and nothing is learned.
    ///
    /// `roots` are probability vectors over the root support, per player, in
    /// the root belief's config order. Returns the per-player per-config root
    /// values for each member, in the same order.
    pub fn value_under(&mut self, roots: &[[Vec<f32>; 2]]) -> Vec<[Vec<f32>; 2]> {
        let reference = self.reference();
        let mut out = Vec::with_capacity(roots.len());
        for root in roots {
            let _t = timed!(P2);
            self.propagate(&reference, [&root[0], &root[1]]);
            self.leaf_values_both();
            let mut pair = [Vec::new(), Vec::new()];
            for p in 0..2usize {
                self.readout(p);
                self.backprop(p, &reference, Back::Value);
                let n = self.nc[0][p] as usize;
                let vo = self.voff[0] as usize;
                pair[p] = self.vals[vo..vo + n].to_vec();
            }
            out.push(pair);
        }
        self.restore();
        out
    }

    /// How well the solve came out, for the reference strategy — the CFR
    /// average at
    /// the end of the solve.
    ///
    /// **The leaf values are frozen** at the ones the reference strategy
    /// induces. They are a function of the beliefs at the leaf, so a real
    /// deviation would move them, and this is therefore the exploitability of
    /// the depth-limited game the reference *defines* rather than of the true
    /// continuation. That is the usual convention in depth-limited solving, and
    /// it is exactly the fixed-point question ReBeL iterates — but it is not
    /// the exploitability of War Chest and must not be reported as if it were.
    pub fn nash_conv(&mut self) -> Conv {
        let reference = self.reference();
        let root = [self.root_belief[0].p.clone(), self.root_belief[1].p.clone()];
        self.propagate(&reference, [&root[0], &root[1]]);
        self.leaf_values_both();
        let (mut nash, mut zero_sum) = (0.0, 0.0);
        for p in 0..2usize {
            // One `readout` serves both passes: `backprop` skips leaves, so the
            // leaf values it left are still there for the second walk.
            self.readout(p);
            let vo = self.voff[0] as usize;
            let nc = self.nc[0][p] as usize;
            let expect = |v: &[f32]| -> f32 { (0..nc).map(|c| root[p][c] * v[vo + c]).sum() };
            self.backprop(p, &reference, Back::Value);
            let v = expect(&self.vals);
            self.backprop(p, &reference, Back::BestResponse);
            nash += expect(&self.vals) - v;
            zero_sum += v;
        }
        self.restore();
        Conv { nash, zero_sum }
    }

    /// The strategy the fixed-policy passes run under: the CFR average at the
    /// end of the solve.
    fn reference(&self) -> Vec<f32> {
        self.snaps
            .last()
            .cloned()
            .expect("a fixed-policy pass needs per-iterate snapshots (Cfg::snapshots)")
    }

    /// Put the reaches back under `cur` after a fixed-policy pass has
    /// propagated something else through them. `update_regrets` assumes they
    /// are consistent with `cur` and does not recompute them, so without this a
    /// solve that is read mid-flight — which is exactly what the solver-error
    /// harness does — would resume from another strategy's reaches.
    fn restore(&mut self) {
        self.precompute_reaches();
        self.last_traverser = None;
    }

    /// The beliefs at tree node `leaf` under each per-iterate average strategy
    /// (t = 0..T-1), from the solve's root belief — TurboReBeL's intermediate
    /// PBSs. The caller appends the walk's live belief as the t = T member.
    ///
    /// Every iterate's average gives every legal action positive probability
    /// (regret matching clamps at 1e-6), so each belief has the same support
    /// as the node's own — the caller asserts that against the live belief
    /// before consuming the set.
    pub fn carried_beliefs(&mut self, leaf: usize) -> Vec<[Vec<f32>; 2]> {
        let root = [self.root_belief[0].p.clone(), self.root_belief[1].p.clone()];
        let n = self.snaps.len().saturating_sub(1);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let snap = self.snaps[i].clone();
            self.propagate(&snap, [&root[0], &root[1]]);
            let mut pair = [Vec::new(), Vec::new()];
            for p in 0..2usize {
                let ra =
                    self.roff[leaf] as usize + if p == 1 { self.nc[leaf][0] as usize } else { 0 };
                let n = self.nc[leaf][p] as usize;
                let mut w = vec![0.0; n];
                normalize_weights(&self.reach[ra..ra + n], &mut w);
                pair[p] = w;
            }
            out.push(pair);
        }
        out
    }

    /// Seed the solve from the policy head instead of from a uniform strategy:
    /// start CFR as though the policy had already been played for `weight`
    /// iterations. One CFR traversal under the policy gives the instantaneous
    /// regret it accrues, `r(a) = v(a) - sum_a pi(a) v(a)`; scaling that by
    /// `weight` is the whole of the warm start, and seeding the average
    /// strategy the same way is what keeps the two consistent.
    ///
    /// The baseline has to be the value of *playing the policy*. An earlier
    /// version used the best-response value, `v(a) - max_a v(a)`, which is
    /// non-positive everywhere and zero at the best action — so regret matching
    /// clamped every action to the floor and handed back a uniform strategy,
    /// destroying exactly the information the seed exists to inject.
    ///
    /// This is not a handcrafted prior. It is the network's own summary of what
    /// the solves before it converged to, and `nash_conv` measures what it does
    /// to the answer with no noise in the measurement.
    pub fn warm_start(&mut self, weight: f32) {
        if weight <= 0.0 || self.nets.value.is_empty() {
            return;
        }
        self.ensure_leaf_batch();
        if !self.policy_into_cur() {
            return;
        }
        self.precompute_reaches();
        for p in 0..2usize {
            self.leaf_values(p);
            self.backprop(p, &[], Back::Seed(weight));
            for i in 0..self.nodes.len() {
                let n = &self.nodes[i];
                if n.leaf || n.chance || n.player as usize != p {
                    continue;
                }
                let nc = n.nc(p);
                let so = self.soff[i] as usize;
                // The average strategy starts as though the policy had been
                // played for those iterations, which is what makes the seeded
                // regrets and the average consistent with each other.
                let r: Vec<f32> = self.reach_of(i, p).to_vec();
                for c in 0..nc {
                    for cell in n.legal_row(c) {
                        self.sum_strat[i][cell] = weight * r[c] * self.cur[so + cell];
                    }
                }
            }
            self.steps[p] = weight as usize;
        }
        // The iterate-0 snapshot was taken from the uniform strategy in `new`;
        // the solve now starts somewhere else, so it is retaken.
        self.snaps.clear();
        self.snap_t = 0;
        self.avg_touched = [true; 2];
        self.snapshot();
    }

    /// Write the policy head's distribution into `cur` at every decision node.
    /// Returns false if the network has no usable policy.
    fn policy_into_cur(&mut self) -> bool {
        self.ensure_leaf_batch();
        if self.inner_rows.is_empty() {
            return false;
        }
        // Everything the head needs is already in the solve's batch: the public
        // tower ran over these rows alongside the leaves, and their configs were
        // interned into the same table. What is left per node is its action
        // descriptions, its belief block, and the readout.
        let net = &self.nets.value;
        let (dg, h) = (net.dg(), net.head());
        let base = self.leaf_rows.len();
        let (mut q, mut logit, mut w, mut psi, mut sb) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut xbel = vec![0.0f32; 2 * dg];
        for k in 0..self.inner_rows.len() {
            let (r, i) = (base + k, self.inner_rows[k]);
            let (na, me) = (self.nodes[i].na(), self.nodes[i].player as usize);
            // The belief the head reads is the normalised reach, as everywhere
            // else. At this point the reaches are the uniform ones `new` left.
            xbel.iter_mut().for_each(|x| *x = 0.0);
            for p in 0..2usize {
                let (lo, hi) = (
                    self.leaf_coff[2 * r + p] as usize,
                    self.leaf_coff[2 * r + p + 1] as usize,
                );
                w.resize(hi - lo, 0.0);
                normalize_weights(self.reach_of(i, p), &mut w);
                crate::net::accumulate(
                    &self.cz,
                    &self.leaf_cidx[lo..hi],
                    &w,
                    dg,
                    &mut xbel[p * dg..(p + 1) * dg],
                );
            }
            psi.resize(na * AFEAT, 0.0);
            for a in 0..na {
                let n = &self.nodes[i];
                write_action_feats(
                    &n.acts[a],
                    &self.ctx,
                    me,
                    n.aslot[a],
                    n.fdown[a],
                    &mut psi[a * AFEAT..(a + 1) * AFEAT],
                );
            }
            net.embed_actions(&psi, na, &self.ce, &mut q);
            let (lo, hi) = (
                self.leaf_coff[2 * r + me] as usize,
                self.leaf_coff[2 * r + me + 1] as usize,
            );
            logit.resize((hi - lo) * na, 0.0);
            net.policy(
                &xbel,
                &self.h0[r * h..],
                &self.cz,
                &self.leaf_cidx[lo..hi],
                &q,
                na,
                &mut sb,
                &mut logit,
            );
            let idx = lo..hi;
            // Softmax over the *legal* actions of each config, with the same
            // floor regret matching uses, so every legal action keeps positive
            // probability and the carried beliefs keep full support.
            let so = self.soff[i] as usize;
            let n = &self.nodes[i];
            for c in 0..idx.len() {
                let logits = &logit[c * na..(c + 1) * na];
                let cells = n.legal_row(c);
                let m = cells.clone().fold(f32::NEG_INFINITY, |m, cell| {
                    m.max(logits[n.legal_action[cell] as usize])
                });
                let mut sum = 0.0;
                for cell in cells.clone() {
                    let a = n.legal_action[cell] as usize;
                    let v = (logits[a] - m).exp();
                    self.cur[so + cell] = v;
                    sum += v;
                }
                if sum > 0.0 {
                    for cell in cells {
                        let x = &mut self.cur[so + cell];
                        *x = (*x / sum).max(1e-6);
                    }
                }
            }
        }
        true
    }

    /// True when the build hit the node cap and the solve must not be used.
    pub fn capped(&self) -> bool {
        self.capped
    }

    /// How many per-iterate snapshots the solve kept. Part of the tree-size
    /// contract.
    pub fn snapshot_count(&self) -> usize {
        self.snaps.len()
    }

    /// The exact kept iteration numbers, in order — the contract's
    /// `snap_iters` array.
    pub fn snapshot_iterations(&self) -> &[usize] {
        &self.snap_list
    }

    /// The CFR average strategy: the approximate equilibrium of the subgame.
    /// Acting and belief propagation use it — the reference strategy of
    /// TurboReBeL's Phase 2 and of the walk through the solved tree.
    pub fn average_strategy(&self, node: usize, c: usize) -> &[f32] {
        let so = self.soff[node] as usize;
        let row = self.nodes[node].legal_row(c);
        &self
            .snaps
            .last()
            .expect("the configured solve must finish before its average is read")
            [so + row.start..so + row.end]
    }
}
