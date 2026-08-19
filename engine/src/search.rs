//! Growing-tree CFR over public belief states, specialised to War Chest.
//!
//! The subgame rooted at a PBS is unrolled over **public observations**. A node
//! is a leaf when it is terminal or when it is a coin play the tree has not
//! grown through yet. A round-start draw is passed through: its private outcome
//! does not branch the public tree. The one child is the post-draw state, and the
//! drawing player's configs are convolved through the draw distribution. Leaf values
//! come from the value network.
//!
//! Conventions follow the reference implementation (`csrc/liars_dice` of
//! `facebookresearch/rebel`), with Student of Games' growing tree
//! (Schmid et al. 2023) in place of a fixed depth limit:
//!   * alternating-traverser linear CFR,
//!   * leaf values are *counterfactual* — the network's value for that exact
//!     config, scaled by the opponent's unnormalised reach into that leaf,
//!   * the network is queried with normalised reaches as the beliefs,
//!   * acting and belief propagation use the final CFR average;
//!   * training rows are solved interior search queries.
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
use crate::farm::{self, Call, Gate, Reply};
use crate::net::Net;
use crate::rng::Rng;
use crate::rebel::*;
use crate::state::{Cont, State};
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};
use crate::timed;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct Cfg {
    /// CFR iterations (alternating, so each player is traversed iters/2 times).
    pub iters: usize,
    /// Expansions per regret update — Student of Games' `c`. Their total
    /// expansion count `s` is `iters * expand`, and one expansion adds one
    /// public state together with all of its children, which is their
    /// `k = infinity` for imperfect-information games.
    pub expand: usize,
    /// Ceiling on public-tree nodes. The tree is grown towards it while CFR
    /// runs, rather than built to a uniform depth up front: a fixed depth
    /// spends the budget evenly over a tree that branches about twenty ways a
    /// ply, so one more ply costs twenty times as much everywhere, including
    /// down lines nobody plays. Growing puts the nodes where the strategy
    /// actually goes.
    pub nodes: usize,
    /// The regret-update rule.
    pub cfr: Cfr,
    /// Max tree nodes a solve may build. 0 = unlimited. A solve that hits the
    /// cap is flagged `capped` and its caller falls back to a non-search
    /// policy for that decision: the tail of the tree-size distribution
    /// (random-draft roots with broad beliefs at round boundaries) is fat
    /// enough that an unbounded build hangs training for minutes on one
    /// decision.
    pub node_cap: usize,
    /// Maximum combined belief support at a self-play root. Broad round-start
    /// beliefs have a fat cost tail before the first tree node can be grown;
    /// callers skip those solves instead of stalling every other worker.
    /// Zero disables the limit.
    pub config_cap: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            iters: 64,
            expand: 1,
            // A depth-2 uniform tree ran to about a thousand nodes, which is
            // the budget this replaces, spent by growth instead of evenly.
            nodes: 1024,
            cfr: Cfr::LINEAR,
            node_cap: 200_000,
            config_cap: 256,
        }
    }
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
    /// player 1 there. So the finite search game is only *approximately*
    /// zero-sum, by however far the value network is from
    /// antisymmetric — which is what this measures, and it is a property of the
    /// network rather than of the solve. It vanishes when every leaf is
    /// terminal, which is the case `tests/rebel_solver.rs` pins against an
    /// independent solver.
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
    /// Pure value propagation under a fixed strategy.
    Value,
    /// The traverser maxes instead of averaging, which makes the root values a
    /// best response to whatever the opponent's reaches were built under.
    BestResponse,
}

/// The value network: `(PBS, config) -> counterfactual value`.
#[derive(Clone, Default)]
pub struct Nets {
    pub value: Net,
    /// When set, network calls are gathered with every other solver thread's
    /// and run as one batch. Without it a solve evaluates its own rows alone,
    /// which is what the tests and the single-game tools want.
    pub gate: Option<Arc<Gate>>,
}

impl Nets {
    pub fn ready(&self) -> bool {
        !self.value.is_empty()
    }

    /// The trunk over fresh leaves: board vectors and the join cache.
    fn trunk(&self, xpub: &[f32], cards: &[f32], rows: usize, pb: &mut Vec<f32>, jp: &mut Vec<f32>) {
        match &self.gate {
            None => {
                self.value.board(xpub, cards, rows, farm::CARD_ROWS, pb);
                self.value.join_cache(pb, rows, jp);
            }
            Some(g) => {
                let r = self.submit(
                    g,
                    Call::Trunk {
                        xpub: xpub.to_vec(),
                        cards: cards.to_vec(),
                        rows,
                    },
                );
                (*pb, *jp) = (r.a, r.b);
            }
        }
    }

    /// The config encoder over fresh configs: `f(c)` and `g(c)`.
    fn encode(
        &self,
        phi: &[f32],
        owner: &[u32],
        n: usize,
        cards: &[f32],
        f: &mut Vec<f32>,
        g_out: &mut Vec<f32>,
    ) {
        match &self.gate {
            None => self.value.configs(phi, owner, n, cards, f, g_out),
            Some(g) => {
                let r = self.submit(
                    g,
                    Call::Configs {
                        phi: phi.to_vec(),
                        owner: owner.to_vec(),
                        cards: cards.to_vec(),
                        n,
                    },
                );
                (*f, *g_out) = (r.a, r.b);
            }
        }
    }

    /// The belief-conditioned head, which every CFR iteration pays for.
    fn head(
        &self,
        p: &[f32],
        jp: &[f32],
        pooled: &[f32],
        rows: usize,
        player: usize,
        h: &mut Vec<f32>,
    ) {
        match &self.gate {
            None => self.value.join(p, jp, pooled, rows, player, h),
            Some(g) => {
                let r = self.submit(
                    g,
                    Call::Join {
                        p: p.to_vec(),
                        jp: jp.to_vec(),
                        pooled: pooled.to_vec(),
                        rows,
                        player,
                    },
                );
                *h = r.a;
            }
        }
    }

    /// A closed gate means the farm is shutting down under a running solve.
    /// There is nothing sensible to return, and the farm joins its threads, so
    /// unwinding here is how the thread stops.
    fn submit(&self, gate: &Gate, call: Call) -> Reply {
        gate.submit(call)
            .expect("the inference gate closed while a solve was still running")
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
/// spends and whether that coin goes face down. `MainPlay` and
/// `WarriorPriestPlay` have config-dependent action sets; every other decision
/// is public, so one call to `legal_actions` suffices there.
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
    let forced = matches!(s.pending(), Cont::WarriorPriestPlay { .. });
    if matches!(s.pending(), Cont::MainPlay) || forced {
        let res = reserve(s, player, ctx);
        for k in 0..NSLOT {
            if res[k] == 0 {
                continue;
            }
            if !cfgs.is_empty()
                && !cfgs.iter().any(|c| {
                    if forced {
                        c.inflight == Some(k as u8)
                    } else {
                        c.hand[k] > 0
                    }
                })
            {
                continue;
            }
            let mut one = Config::default();
            if forced {
                one.inflight = Some(k as u8);
            } else {
                one.hand[k] = 1;
            }
            set_config(&mut probe, player, ctx, &one);
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
                if slot >= 0
                    && !cfgs.is_empty()
                    && !cfgs.iter().any(|c| {
                        if forced {
                            c.inflight == Some(slot as u8)
                        } else {
                            c.hand[slot as usize] > 0
                        }
                    })
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
    /// sites -- this and two debug asserts.
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

/// What a finished solve gives back.
///
/// `value` is the root's counterfactual value per config, and only the root's.
/// The opposing range is normalised there, so the number is on the game's own
/// scale. An interior node's range is not normalised — it carries the reach
/// that led to it, so its value shrinks with depth and shrinks again as the
/// strategy sharpens — and a node beside the frontier would hand back little
/// more than the network's own leaf output. ReBeL adds `{beta_r, v(beta_r)}`
/// and nothing else to its value set; Student of Games re-solves every query
/// from scratch and stores that solve's root.
///
/// `queries` are public belief states this search asked the network about.
/// They are not targets. They are roots for later solves, which is how the
/// value function becomes accurate away from the line of play.
pub struct Solved {
    pub value: [Vec<f32>; 2],
    pub queries: Vec<(State, [Belief; 2])>,
}

/// Draw an index from non-negative weights without allocating. A row whose
/// weights have all underflowed is drawn uniformly rather than dropped.
fn pick(w: &[f32], rng: &mut Rng) -> usize {
    let total: f64 = w.iter().map(|&x| x.max(0.0) as f64).sum();
    if total == 0.0 {
        return rng.below(w.len().max(1));
    }
    let mut needle = rng.unit_f64() * total;
    for (i, &weight) in w.iter().enumerate() {
        needle -= weight.max(0.0) as f64;
        if needle < 0.0 {
            return i;
        }
    }
    w.len() - 1
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
const R_PB: usize = 0;
const R_XPUB: usize = 1;
const R_XB: usize = 2;
const R_H: usize = 3;
const R_CPHI: usize = 4;
const R_CF: usize = 5;
const R_CG: usize = 6;
const R_JP: usize = 7;

thread_local! {
    static BUFS: std::cell::RefCell<[Vec<Vec<f32>>; N_ROLES]> = const {
        std::cell::RefCell::new([
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        ])
    };
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
    // Point-written workspaces retain their length. Append-only cache arenas
    // are cleared by `Solver::new` before their first extension.
    BUFS.with(|b| {
        let mut b = b.borrow_mut();
        if b[role].len() < 2 {
            b[role].push(v);
        }
    });
}

pub struct Solver<'a> {
    /// Owned because a solver retains its context for its full solve.
    pub(crate) ctx: Ctx,
    nets: &'a Nets,
    pub(crate) cfg: Cfg,
    pub nodes: Vec<TNode>,
    /// Node states, parallel to `nodes`. A leaf can only be expanded later if
    /// the solver still knows the position it stands for, so a growing tree
    /// keeps every one of them.
    pub states: Vec<State>,
    pub(crate) root_belief: [Belief; 2],
    /// Regrets and the current regret-matching iterate, flat by node over legal
    /// cells. Node `i` occupies `soff[i] ..` for as many cells as its
    /// `legal_action` holds; within that range its config rows are described by
    /// `TNode::legal_off`. `soff` is *not* sorted: a node is born a leaf with
    /// no cells and is given its region when it is expanded, which is what lets
    /// the arena grow by appending while every regret already accumulated stays
    /// where it is.
    pub(crate) regret: Vec<f32>,
    pub cur: Vec<f32>,
    pub(crate) soff: Vec<u32>,
    /// The reach-weighted running strategy sum, per node. Per node rather than
    /// flat, because a node is given its cells when it is expanded and a
    /// ragged vector grows there without disturbing anything already summed.
    pub sum_strat: Vec<Vec<f32>>,
    /// The reference strategy: one flat normalised copy of `sum_strat`, laid
    /// out exactly like `cur`. Materialised by `finish` once the tree has
    /// stopped growing, and read by everything that acts, filters a belief or
    /// values a node.
    pub avg: Vec<f32>,
    /// Whether each player's running sum has been normalised at least once.
    /// Until then the historical average is the literal initial iterate, not
    /// a multiply-then-divide reconstruction of it.
    pub(crate) avg_touched: [bool; 2],
    /// Total legal strategy cells across decision nodes: the length of `cur`,
    /// `regret` and `avg`, and where the next expanded node's region starts.
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
    // the belief block `xb` and the join output `h` are rewritten
    // per iteration. That split is the whole architecture: the trunk runs
    // ~2,000 times a solve and the join ~158,000.
    /// Non-terminal leaves in node order — the rows of the network batch.
    pub leaf_rows: Vec<usize>,
    /// Terminal leaves, scored from the game instead of the network.
    pub(crate) term_leaves: Vec<usize>,
    /// Per row, per player: an index into `cphi` for every config in support,
    /// packed back to back and indexed through `leaf_coff`.
    pub(crate) leaf_cidx: Vec<u32>,
    pub(crate) leaf_coff: Vec<u32>,
    /// Distinct config vectors and their canonical owner query.
    pub(crate) cphi: Vec<f32>,
    pub(crate) cplayer: Vec<u8>,
    pub(crate) cmap: std::collections::HashMap<u64, u32, KeyHash>,
    /// How many distinct configs `cphi` actually holds. Pooled buffers keep
    /// their length across solves, so the count cannot be read off `cphi.len()`.
    pub ncfg: usize,
    /// How many leaf rows and how many distinct configs the batch below has
    /// already been built for. Everything in it is a pure function of the
    /// subgame and none of it moves when the tree grows, so a growth round
    /// only ever runs the network on what it just added.
    batch_rows: usize,
    batch_cfgs: usize,
    /// `[ncfg, D]` readout rows `f(c)` and `[ncfg, POOL]` pooling vectors
    /// `g(c)`. Both survive every CFR iteration.
    pub cf: Vec<f32>,
    pub cg: Vec<f32>,
    /// `[2, NTYPE, TYPE]`: the printed-card tokens, one table per player view.
    /// The draft is fixed for the solve, so this is built once.
    pub cards: Vec<f32>,
    /// `[2 * rows, D]` board vectors, and their `[2 * rows, JW]` projection
    /// into the join's first layer. Neither moves between CFR iterations.
    pub pb: Vec<f32>,
    pub jp: Vec<f32>,
    /// Expanded public encoding, filled during the build.
    pub(crate) xpub: Vec<f32>,
    /// `[2 * rows, POOL]` pooled belief embeddings — the one thing the join
    /// reads that moves between CFR iterations.
    pub xb: Vec<f32>,
    /// `[rows, D]`: the join output for the last traverser queried.
    pub h: Vec<f32>,
    /// Normalised belief weights for one leaf's support.
    wbuf: Vec<f32>,
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
            (R_PB, &mut self.pb),
            (R_JP, &mut self.jp),
            (R_XPUB, &mut self.xpub),
            (R_XB, &mut self.xb),
            (R_H, &mut self.h),
            (R_CPHI, &mut self.cphi),
            (R_CF, &mut self.cf),
            (R_CG, &mut self.cg),
        ] {
            give_buf(role, std::mem::take(v));
        }
        give_nodes(std::mem::take(&mut self.nodes));
    }
}

impl<'a> Solver<'a> {

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
            avg: Vec::new(),
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
            cplayer: Vec::new(),
            cmap: std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash),
            ncfg: 0,
            batch_rows: 0,
            batch_cfgs: 0,
            cf: take_buf(R_CF),
            cg: take_buf(R_CG),
            cards: Vec::new(),
            pb: take_buf(R_PB),
            jp: take_buf(R_JP),
            xpub: take_buf(R_XPUB),
            xb: take_buf(R_XB),
            h: take_buf(R_H),
            wbuf: Vec::new(),
            last_traverser: None,
            capped: false,
            draw_scratch: DrawScratch::default(),
            cell_order: Vec::new(),
        };
        sv.cf.clear();
        sv.cg.clear();
        sv.pb.clear();
        sv.jp.clear();
        {
            let _t = timed!(BUILD);
            // The root is born a leaf like every other node; `solve` grows it.
            sv.nodes.reserve(640);
            sv.reach.reserve(640);
            sv.vals.reserve(640);
            sv.regret.reserve(640);
            sv.cur.reserve(640);
            let root = sv.push_node(root.clone(), cfgs);
            // A root that is a coin play would otherwise stay a leaf with no
            // strategy to read, so the first expansion is unconditional.
            sv.grow(root);
            // The first CFR update and every expansion trajectory require
            // reaches for the tree that now exists.
            sv.precompute_reaches();
        }
        sv
    }

    /// Push a node for `s`, as a leaf, and give it its slice of every arena.
    ///
    /// Every node is born a leaf. `grow` turns one into a decision node later,
    /// which is when it is given its strategy cells — so a node's reach and
    /// value rows are laid out in node order while its cells are not, and both
    /// are found through their own offset table.
    fn push_node(&mut self, s: State, cfgs: [Rc<[Config]>; 2]) -> usize {
        let player = s.to_act();
        let terminal = s.is_terminal();
        let id = self.nodes.len();
        let _tp = timed!(BPUSH);
        self.nodes.push(TNode {
            util: if terminal { s.utility(player as usize) } else { 0.0 },
            player,
            leaf: true,
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
        let (c0, c1) = (cfgs[0].len(), cfgs[1].len());
        self.nc.push([c0 as u32, c1 as u32]);
        // No cells yet: a leaf has no strategy. `grow` appends its region.
        self.soff.push(self.ncells as u32);
        self.roff.push(self.reach.len() as u32);
        self.reach.resize(self.reach.len() + c0 + c1, 0.0);
        self.voff.push(self.vals.len() as u32);
        self.vals.resize(self.vals.len() + c0.max(c1), 0.0);
        self.sum_strat.push(Vec::new());
        // Only a coin play carries a network row. Everything between two coin
        // plays is grown through immediately and never stays a leaf.
        if terminal {
            self.term_leaves.push(id);
        } else if matches!(s.pending(), Cont::MainPlay) {
            self.push_row(id, &s, &cfgs);
            self.leaf_rows.push(id);
        }
        self.states.push(s);
        id
    }

    /// Push a child and, unless it is somewhere the search may stop, grow it.
    ///
    /// A leaf always stands for a coin play or a terminal, because the value
    /// network is defined only at a coin play. A round-start draw run, the
    /// micro-decisions inside a tactic and a Warrior Priest's forced play are
    /// none of those, so they ride free here exactly as they used to ride free
    /// inside a depth unit.
    fn push_child(&mut self, s: State, cfgs: [Rc<[Config]>; 2]) -> usize {
        let stop = s.is_terminal() || matches!(s.pending(), Cont::MainPlay);
        let ch = self.push_node(s, cfgs);
        if !stop {
            self.grow(ch);
        }
        ch
    }

    /// Give an expanded node its strategy cells, at the end of the arenas.
    ///
    /// CFR starts from a uniform strategy over the legal actions, as in the
    /// reference. No heuristic prior is injected here: what the network knows
    /// enters through the leaf values, which is what CFR actually consumes.
    fn alloc_cells(&mut self, id: usize) {
        let cells = self.nodes[id].legal_action.len();
        self.soff[id] = self.ncells as u32;
        self.ncells += cells;
        self.regret.resize(self.ncells, 0.0);
        let n = &self.nodes[id];
        let nc = n.nc(n.player as usize);
        let mut u = vec![0.0f32; cells];
        for c in 0..nc {
            let row = n.legal_row(c);
            let k = row.len() as f32;
            for cell in row {
                u[cell] = 1.0 / k;
            }
        }
        self.cur.extend_from_slice(&u);
        self.sum_strat[id] = vec![0.0; cells];
    }

    /// Materialise the reference strategy: the normalised CFR average, laid
    /// out exactly like `cur`.
    ///
    /// It is built once, when the tree has stopped growing and the iterations
    /// are done, because that is the only moment at which one flat array can
    /// describe the whole tree. Everything that acts, filters a belief or
    /// values a node reads it afterwards.
    pub fn finish(&mut self) {
        // `cur` still holds the literal initial policy for a player that has
        // not traversed yet, so start there and overwrite every player whose
        // running sum has moved. Their historical average is then byte-exact
        // rather than a multiply and divide that need not round back.
        self.avg.clear();
        self.avg.extend_from_slice(&self.cur);
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
                    self.avg[so + cell] = if sum > 0.0 {
                        self.sum_strat[i][cell] / sum
                    } else {
                        1.0 / k
                    };
                }
            }
        }
    }

    // ------------------------------------------------------------ tree build

    /// Turn leaf `id` into a decision node, its children pushed as leaves.
    ///
    /// This is the whole of tree growth. It is the old fixed-depth build with
    /// the depth counter taken out: one call adds exactly one coin play, and
    /// the search decides how many calls to make and where, rather than
    /// spending the same budget evenly over every line.
    fn grow(&mut self, id: usize) {
        debug_assert!(self.nodes[id].leaf, "only a leaf can be grown");
        let s = self.states[id].clone();
        debug_assert!(!s.is_terminal(), "a terminal has nothing to grow");
        let cfgs = self.nodes[id].cfgs.clone();
        self.nodes[id].leaf = false;
        // The node keeps its row in the network batch, which is now spent on a
        // value nobody reads: `backprop` decides leafhood from the node and
        // recomputes an interior node from its children, so the stale readout
        // is overwritten. Retiring the row would mean compacting the batch or
        // carrying a liveness mask through every per-iteration loop, to save
        // the interior-node share of the rows -- about one in twenty at this
        // branching factor. Not worth either.
        let player = s.to_act();
        // A draw is walked through: the outcome is private, so the public tree
        // does not branch. Round-start draws collapse over a whole run; a
        // Warrior Priest draw is a single chance node whose children carry the
        // pending forced-play coin.
        let draw_pass = matches!(
            s.pending(),
            Cont::Draw { .. } | Cont::WarriorPriestDraw { .. }
        );
        if draw_pass {
            // The draw's outcome is private, so the public tree does not
            // branch: there is exactly one child, the state after the draw.
            // Which coin is drawn changes nothing public, so any legal
            // DrawCoin produces the same child. The drawing player's configs
            // are convolved through the draw distribution — the chance
            // factor stays separate from both players' strategies: it enters
            // the drawing player's reach as a transition, and the idle
            //
            // A round start queues up to three draws in a row for the same
            // player. None of them branches and none of them is a decision, so
            // the whole run collapses into this one node with the composed
            // transition; `steps` is how many of the game's draws it stands
            // for, which is what the self-play walk counts off.
            let td = timed!(BDRAW);
            let me = player as usize;
            // A Warrior Priest draw's children carry the forced-play coin.
            let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
            let mut cs = s;
            // The stored world is only a public-state carrier. Materialise a
            // belief member before any rule helper reads private zones.
            set_config(&mut cs, player, &self.ctx, &cfgs[me][0]);
            let mut support: Vec<Config> = Vec::new();
            let mut draw = DrawMap::default();
            // The reserve and the face-up pile are what the draws read, and a
            // draw changes neither (a refill does, and `run` accounts for it
            // internally), so both come from the state at the head of the run.
            let res = reserve(&cs, player, &self.ctx);
            let fu = faceup_counts(&cs, player, &self.ctx);
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
            let ch = self.push_child(cs, cc);
            let n = &mut self.nodes[id];
            n.chance = true;
            n.child = vec![ch];
            n.draw = draw;
            n.draw_steps = steps;
            return;
        }

        let me = player as usize;
        let mine = cfgs[me].clone();
        let nc = mine.len();
        let ta = timed!(BACTS);
        let (acts, aslot, fdown) = node_actions(&s, player, &self.ctx, &mine);
        drop(ta);
        let na = acts.len();
        debug_assert!(na > 0, "a decision node must offer a reachable action");

        let tcells = timed!(BCELLS);
        let mut legal_off = Vec::with_capacity(nc + 1);
        let mut legal_action = Vec::new();
        let mut legal_child = Vec::new();
        let mut legal_trans = Vec::new();
        let mut cell_row = Vec::new();
        legal_off.push(0);
        for (ci, c) in mine.iter().enumerate() {
            for a in 0..na {
                if action_legal(c, aslot[a]) {
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
            child.push(self.push_child(cs, cc));
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
        self.alloc_cells(id);
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
        self.propagate(&cur);
        self.cur = cur;
    }

    /// Push reach probabilities down the tree under `strat`, from the root
    /// beliefs.
    fn propagate(&mut self, strat: &[f32]) {
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

    fn encode(&mut self, s: &State, row: usize) {
        let at = 2 * row * PUBFEAT;
        write_public_features(s, &self.ctx, &mut self.xpub[at..at + PUBFEAT]);
        let mirrored = s.mirror();
        write_public_features(
            &mirrored,
            &self.ctx.mirrored(),
            &mut self.xpub[at + PUBFEAT..at + 2 * PUBFEAT],
        );
    }

    /// One row of the network batch: its public encoding, and its configs
    /// interned into the shared table.
    fn push_row(&mut self, _id: usize, s: &State, cfgs: &[Rc<[Config]>; 2]) {
        // The network is queried only at normal coin-play states: a subgame
        // finishes every tactic, trigger and forced play before a leaf.
        debug_assert!(
            matches!(s.pending(), Cont::MainPlay),
            "a network row must be a MainPlay state"
        );
        let _t = timed!(PUBFEAT);
        let row = self.leaf_coff.len() / 2;
        let at = 2 * row * PUBFEAT;
        if self.xpub.len() < at + 2 * PUBFEAT {
            // Grow in chunks so the zero-fill happens a handful of times per
            // solve, and not at all once the pooled buffer is warm.
            self.xpub.resize(at + 128 * PUBFEAT, 0.0);
        }
        self.encode(s, row);
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
        self.cplayer.push(p as u8);
        self.cmap.insert(key, i);
        i
    }

    /// Everything about the batch that does not move between CFR iterations:
    /// the card tokens, the board trunk, the projection of its output into the
    /// join, and the config encoder over every distinct config in the tree.
    ///
    /// All of it is a pure function of the subgame, so growth never invalidates
    /// any of it — a leaf's board vector is the same board vector after the
    /// tree around it has changed. This therefore runs the trunk once per leaf
    /// *ever created*, which is what a fixed-depth solve of the final tree
    /// would have cost. Growing is free on this path; only the CFR iterations
    /// see a bigger tree, and the early ones see a smaller one.
    fn extend_leaf_batch(&mut self) {
        let rows = self.leaf_rows.len();
        if rows > self.batch_rows {
            // New rows have no cached belief block for either player.
            self.last_traverser = None;
        }
        if rows == self.batch_rows && self.ncfg == self.batch_cfgs {
            return;
        }
        if self.nets.value.is_empty() {
            self.batch_rows = rows;
            self.batch_cfgs = self.ncfg;
            return;
        }
        let net = &self.nets.value;
        crate::net::fit(&mut self.xb, 2 * rows * crate::net::POOL);
        let _t = timed!(PUBNET);
        let xpub = std::mem::take(&mut self.xpub);
        if rows > self.batch_rows {
            // The cards in play are fixed at the draft, so every row of the
            // subgame carries the same card block and the table is built once,
            // one view per seat. Everything downstream reads it by canonical
            // coin-type index.
            if self.cards.is_empty() {
                net.cards(&xpub[..2 * PUBFEAT], 2, &mut self.cards);
            }
            let fresh = rows - self.batch_rows;
            let at = 2 * self.batch_rows * PUBFEAT;
            let (mut pb, mut jp) = (Vec::new(), Vec::new());
            // Exactly the fresh rows. `xpub` is a grown scratch buffer, so an
            // open-ended slice would carry whatever the last, larger subgame
            // left behind — invisible to a solve evaluating alone, and wrong
            // the moment the farm concatenates this call with another.
            let end = at + 2 * fresh * PUBFEAT;
            self.nets
                .trunk(&xpub[at..end], &self.cards, fresh, &mut pb, &mut jp);
            crate::prof::work(fresh, 0, 0, 0);
            self.pb.extend_from_slice(&pb);
            self.jp.extend_from_slice(&jp);
            self.batch_rows = rows;
        }
        self.xpub = xpub;
        if self.ncfg > self.batch_cfgs {
            let fresh = self.ncfg - self.batch_cfgs;
            let cphi = std::mem::take(&mut self.cphi);
            let owner: Vec<u32> = self.cplayer[self.batch_cfgs..]
                .iter()
                .map(|&p| p as u32)
                .collect();
            let (mut cf, mut cg) = (Vec::new(), Vec::new());
            self.nets.encode(
                &cphi[self.batch_cfgs * CFEAT..self.ncfg * CFEAT],
                &owner,
                fresh,
                &self.cards,
                &mut cf,
                &mut cg,
            );
            crate::prof::work(0, fresh, 0, 0);
            self.cphi = cphi;
            self.cf.extend_from_slice(&cf);
            self.cg.extend_from_slice(&cg);
            self.batch_cfgs = self.ncfg;
        }
    }

    /// Rewrite the pooled belief block the join reads, per row per player.
    /// `only` restricts the work to one player — between two CFR queries just
    /// one player's strategy has moved, so the other block is still exactly
    /// what it was.
    ///
    /// The belief the network reads is the normalised reach, as in the
    /// reference, pooled over the same `g(c)` the readout's `f(c)` comes from,
    /// so a config is described to the network exactly one way. `g` has a
    /// linear card-weighted half, which is what makes this pooled vector carry
    /// the belief's exact expected holding of each card rather than an average
    /// of nonlinearities.
    fn belief_blocks(&mut self, only: Option<usize>) {
        let _t = timed!(BELFEAT);
        let (reach, roff, nc, coff, cidx, cg, wbuf, xb) = (
            &self.reach,
            &self.roff,
            &self.nc,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cg,
            &mut self.wbuf,
            &mut self.xb,
        );
        let pool = crate::net::POOL;
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            for p in 0..2 {
                if only.is_some_and(|l| l != p) {
                    continue;
                }
                let n = nc[i][p] as usize;
                let ra = roff[i] as usize + if p == 1 { nc[i][0] as usize } else { 0 };
                if wbuf.len() < n {
                    wbuf.resize(n, 0.0);
                }
                normalize_weights(&reach[ra..ra + n], &mut wbuf[..n]);
                let q = 2 * r + p;
                let cs = coff[q] as usize;
                crate::net::accumulate(
                    cg,
                    &cidx[cs..cs + n],
                    &wbuf[..n],
                    pool,
                    &mut xb[q * pool..(q + 1) * pool],
                );
            }
        }
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values.
    pub fn leaf_values(&mut self, traverser: usize) {
        self.extend_leaf_batch();
        let redo = self.last_traverser;
        self.last_traverser = Some(traverser);
        if !self.nets.value.is_empty() {
            self.belief_blocks(redo);
        }
        self.pbs_head(traverser);
        self.readout(traverser);
    }

    /// Refresh both belief blocks for a fixed-policy pass. Each traverser then
    /// gets its own PBS-head query over these same beliefs.
    fn leaf_beliefs_both(&mut self) {
        self.extend_leaf_batch();
        self.last_traverser = None;
        if self.nets.value.is_empty() {
            return;
        }
        self.belief_blocks(None);
    }

    /// The one path CFR pays for on every iteration.
    fn pbs_head(&mut self, traverser: usize) {
        let net = &self.nets.value;
        if net.is_empty() {
            return;
        }
        let _t = timed!(NET);
        let rows = self.leaf_rows.len();
        crate::prof::work(0, 0, rows, 0);
        // `xb` is grown by `fit` and never shrinks, so a subgame smaller than
        // an earlier one would otherwise hand the batch a trailing tail.
        self.nets.head(
            &self.pb[..rows * crate::net::D],
            &self.jp[..rows * crate::net::JW],
            &self.xb[..2 * rows * crate::net::POOL],
            rows,
            traverser,
            &mut self.h,
        );
    }

    /// Per-config leaf values for player `p` — counterfactual: the network's
    /// value for that exact config times the opponent's unnormalised reach
    /// into the leaf. Runs off the `h` left by the last `pbs_head` query, and
    /// is one dot product per config.
    pub fn readout(&mut self, p: usize) {
        let _t = timed!(LEAFPOST);
        let empty = self.nets.value.is_empty();
        let readout_configs = self
            .leaf_rows
            .iter()
            .map(|&i| self.nc[i][p] as usize)
            .sum();
        crate::prof::work(0, 0, 0, readout_configs);
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
        let d = crate::net::D;
        let (reach, roff, ncs, voff, coff, cidx, cf, vals) = (
            &self.reach,
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cf,
            &mut self.vals,
        );
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            let ra = roff[i] as usize + if opp == 1 { ncs[i][0] as usize } else { 0 };
            let opp_reach: f32 = reach[ra..ra + ncs[i][opp] as usize].iter().sum();
            let n = ncs[i][p] as usize;
            let vo = voff[i] as usize;
            if empty {
                vals[vo..vo + n].fill(0.0);
                continue;
            }
            let cs = coff[2 * r + p] as usize;
            self.nets.value.values(
                &self.h[r * d..(r + 1) * d],
                cf,
                &cidx[cs..cs + n],
                &mut vals[vo..vo + n],
            );
            for value in &mut vals[vo..vo + n] {
                *value *= opp_reach;
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

    /// One value backpropagation over the tree for `traverser`. `mode` chooses
    /// whether the traverser's decision nodes average under `strat`, average
    /// and update regret matching, or take the max. Regret mode uses
    /// `self.cur`; fixed-policy modes read `strat`.
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
                            Back::Regret => vi[c] += av * self.cur[so + cell],
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
        self.steps[traverser] += 1;
    }

    /// Add the fresh reach-weighted iterate to the running strategy sum.
    /// Normalisation is deferred to `finish`.
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

    /// Run the solve: Student of Games' GT-CFR.
    ///
    /// One round is one regret update followed by `expand` expansion
    /// simulations, which is their `GT-CFR(L, beta, s, c)` with `c = expand`
    /// and `s = iters * expand`. Growing and solving interleave rather than
    /// staging, which is the point: the strategy decides where the tree goes,
    /// and the tree decides what the strategy is worth.
    pub fn solve(&mut self, rng: &mut Rng) {
        for t in 0..self.cfg.iters {
            self.step(t % 2);
            let mut grew = false;
            for _ in 0..self.cfg.expand {
                if self.nodes.len() >= self.cfg.nodes {
                    break;
                }
                if self.cfg.node_cap > 0 && self.nodes.len() >= self.cfg.node_cap {
                    self.capped = true;
                    break;
                }
                let Some(leaf) = self.sample_leaf(rng) else {
                    break;
                };
                self.grow(leaf);
                grew = true;
                if self.cfg.node_cap > 0 && self.nodes.len() > self.cfg.node_cap {
                    self.capped = true;
                    break;
                }
            }
            if grew {
                // Growth appended reach rows after `step` propagated the old
                // tree. The next regret update must see the new leaves.
                self.precompute_reaches();
            }
            if self.capped {
                break;
            }
        }
        self.finish();
    }

    /// Grow the whole subgame, to the node budget or until nothing is left to
    /// expand.
    ///
    /// Production never does this — the point of growing is to *not* build the
    /// whole tree. It exists for the tests and sizing tools that need the
    /// complete subgame of a small endgame, which is what the old fixed depth
    /// limit gave them when the limit was larger than the game.
    pub fn grow_full(&mut self) {
        let mut at = 0usize;
        while at < self.nodes.len() {
            if self.nodes.len() >= self.cfg.nodes
                || (self.cfg.node_cap > 0 && self.nodes.len() >= self.cfg.node_cap)
            {
                self.capped = true;
                return;
            }
            if self.nodes[at].leaf && !self.states[at].is_terminal() {
                self.grow(at);
            }
            at += 1;
        }
    }

    /// The root's per-config values under the reference strategy — the target
    /// a solve at this position produces for itself.
    pub fn root_values(&mut self) -> [Vec<f32>; 2] {
        let out = self.value_pass();
        self.restore();
        out
    }

    /// Value every node under the reference strategy and return the root's
    /// slice. Leaves the reference reaches in place so a caller can read
    /// beliefs off the tree; it must `restore` when it is done.
    fn value_pass(&mut self) -> [Vec<f32>; 2] {
        let reference = self.reference();
        self.propagate(&reference);
        self.leaf_beliefs_both();
        let mut out = [Vec::new(), Vec::new()];
        for p in 0..2usize {
            self.pbs_head(p);
            self.readout(p);
            // `backprop` for the second player overwrites the first's values,
            // so the root slice is taken before the next pass runs.
            self.backprop(p, &reference, Back::Value);
            let n = self.nc[0][p] as usize;
            let vo = self.voff[0] as usize;
            out[p] = self.vals[vo..vo + n].to_vec();
        }
        out
    }

    /// One expansion simulation: sample a world from the root beliefs, walk
    /// down under the current average strategy, and return the leaf it reaches.
    ///
    /// Sampling rather than taking the most-reached leaf is what the paper does
    /// and what its convergence result wants: an optimal policy here is often
    /// mixed, and a greedy rule can starve a line the average strategy still
    /// gives weight to. Their selection rule is half PUCT and half the CFR
    /// average; with no prior to compute PUCT from, this is the half that
    /// exists.
    fn sample_leaf(&mut self, rng: &mut Rng) -> Option<usize> {
        // One private config per player forms the sampled world.
        let mut c = [
            pick(&self.root_belief[0].p, rng),
            pick(&self.root_belief[1].p, rng),
        ];
        let mut node = 0usize;
        loop {
            if self.nodes[node].leaf {
                return (!self.states[node].is_terminal()).then_some(node);
            }
            let me = self.nodes[node].player as usize;
            if self.nodes[node].chance {
                let (idx, prob) = self.nodes[node].draw.row(c[me]);
                let k = pick(prob, rng);
                c[me] = idx[k] as usize;
                node = self.nodes[node].child[0];
                continue;
            }
            let row = self.nodes[node].legal_row(c[me]);
            if row.is_empty() {
                return None;
            }
            let cell = if self.sum_strat[node][row.clone()].iter().any(|&x| x > 0.0) {
                row.start + pick(&self.sum_strat[node][row.clone()], rng)
            } else {
                let so = self.soff[node] as usize;
                row.start + pick(&self.cur[so + row.start..so + row.end], rng)
            };
            let trans = self.nodes[node].legal_trans[cell];
            if trans == NO_TRANS {
                return None;
            }
            c[me] = trans as usize;
            node = self.nodes[node].legal_child[cell] as usize;
        }
    }


    /// The interior search queries this solve produced.
    ///
    /// Student of Games trains on the public belief states whose trees supplied
    /// a better value than the network leaf they replaced.
    ///
    /// Only interior coin plays are taken. A leaf's value would be the
    /// network's own answer, so training on it would teach the network what it
    /// already said; an interior node's value comes from the subtree beneath
    /// it, which is the bootstrap the whole method rests on.
    pub fn harvest(&mut self, rng: &mut Rng, queries: usize) -> Solved {
        let value = self.value_pass();
        let queries = self.sample_queries(rng, queries);
        self.restore();
        Solved { value, queries }
    }

    /// Uniform draws from the leaves this solve queried the network at.
    ///
    /// Those leaves are where the value function's error enters the solve, so
    /// they are the belief states worth solving in their own right. Every one
    /// of them is a coin play, which is both what the network is defined on
    /// and what a training row can carry, so no filtering is needed here.
    fn sample_queries(&self, rng: &mut Rng, want: usize) -> Vec<(State, [Belief; 2])> {
        if self.leaf_rows.is_empty() {
            return Vec::new();
        }
        (0..want)
            .map(|_| {
                let i = self.leaf_rows[rng.below(self.leaf_rows.len())];
                (self.states[i].clone(), self.belief_at(i))
            })
            .collect()
    }

    /// Node `i`'s belief for each player, under whichever reaches are
    /// currently propagated.
    fn belief_at(&self, i: usize) -> [Belief; 2] {
        std::array::from_fn(|p| {
            let mut w = vec![0.0; self.nc[i][p] as usize];
            normalize_weights(self.reach_of(i, p), &mut w);
            Belief {
                cfg: self.nodes[i].cfgs[p].to_vec(),
                p: w,
            }
        })
    }

    /// How well the solve came out, for the reference strategy — the CFR
    /// average at
    /// the end of the solve.
    ///
    /// **The leaf values are frozen** at the ones the reference strategy
    /// induces. They are a function of the beliefs at the leaf, so a real
    /// deviation would move them, so this measures exploitability of the
    /// finite search game, not of the true War Chest continuation.
    pub fn nash_conv(&mut self) -> Conv {
        let reference = self.reference();
        let root = [self.root_belief[0].p.clone(), self.root_belief[1].p.clone()];
        self.propagate(&reference);
        self.leaf_beliefs_both();
        let (mut nash, mut zero_sum) = (0.0, 0.0);
        for p in 0..2usize {
            self.pbs_head(p);
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

    /// The strategy the fixed-policy passes run under.
    fn reference(&self) -> Vec<f32> {
        assert!(
            !self.avg.is_empty(),
            "a fixed-policy pass needs `finish` to have materialised the average"
        );
        self.avg.clone()
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

    /// True when the build hit the node cap and the solve must not be used.
    pub fn capped(&self) -> bool {
        self.capped
    }

    /// The CFR average strategy: the approximate equilibrium of the subgame.
    /// Acting and belief propagation use it.
    pub fn average_strategy(&self, node: usize, c: usize) -> &[f32] {
        let so = self.soff[node] as usize;
        let row = self.nodes[node].legal_row(c);
        assert!(
            !self.avg.is_empty(),
            "the solve must `finish` before its average is read"
        );
        &self.avg[so + row.start..so + row.end]
    }
}
