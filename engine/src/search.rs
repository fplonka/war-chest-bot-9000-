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
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            depth: 1,
            iters: 64,
            snapshots: true,
            cfr: Cfr::LINEAR,
            warm: 0.0,
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
    pub const LINEAR: Cfr = Cfr { alpha: 1.0, beta: 1.0, gamma: 1.0, predict: 0.0 };
    pub const PLUS: Cfr = Cfr { alpha: f32::INFINITY, beta: f32::NEG_INFINITY, gamma: 2.0, predict: 0.0 };
    pub const DISCOUNTED: Cfr = Cfr { alpha: 1.5, beta: 0.0, gamma: 2.0, predict: 0.0 };
    pub const PREDICTIVE: Cfr = Cfr { alpha: f32::INFINITY, beta: f32::NEG_INFINITY, gamma: 2.0, predict: 1.0 };
    pub const SIMPLE_ASYM: Cfr = Cfr { alpha: f32::INFINITY, beta: f32::NEG_INFINITY, gamma: 2.0, predict: 1.0 / 3.0 };

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
    pub zero_sum: f32,
}

/// What a backward pass over the tree does with the values it computes.
#[derive(Clone, Copy, PartialEq)]
enum Back {
    /// CFR: the traverser averages over their strategy, and the per-action
    /// values less the node value accumulate as instantaneous regret.
    Regret,
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

// The per-solve snapshot arena is a `Vec<Vec<f32>>` (one flat copy of `avg`
// per iterate), so it pools separately from the flat buffers above.
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
    /// The instantaneous counterfactual regret of the traversal just finished,
    /// same layout. Kept apart from `regret` because the accumulated regret is
    /// discounted before this iteration's is added to it, so afterwards there
    /// is no way to recover it — and Predictive CFR+ needs it a second time,
    /// as its guess at the regret the next iteration will see.
    inst: Vec<f32>,
    cur: Vec<f32>,
    soff: Vec<u32>,
    /// The average strategy and its running sum, per node. Always maintained:
    /// the walk acts on the average, and generation snapshots it per iterate.
    sum_strat: Vec<Vec<f32>>,
    avg: Vec<Vec<f32>>,
    /// One flat copy of `avg` (per-node regions in node order, aligned with
    /// `soff`) taken before the first iteration and after each one: snapshot
    /// `t` is the average strategy at iterate t, and the last is the reference
    /// strategy `value_under` and the walk act on. Pooled across solves.
    snaps: Vec<Vec<f32>>,
    /// Which snapshot the next `snapshot()` call is (0 = the pre-iteration
    /// average). Drives the log-spaced thinning: the carried beliefs are one
    /// per *kept* iterate, and the spread is in the early ones.
    snap_t: usize,
    /// Total strategy cells (sum over decision nodes of `nc * na`), so the
    /// snapshot arenas are reserved to size instead of grown.
    ncells: usize,
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
    steps: [usize; 2],

    // ---------------------------------------------------------- leaf batch
    // Built once per solve. Everything here is a property of the leaf's public
    // state or its config support, so it survives every CFR iteration; only
    // `xb` (the belief blocks) is rewritten per iteration.
    /// Non-terminal leaves in node order — the rows of the network batch.
    leaf_rows: Vec<usize>,
    /// Terminal leaves, scored from the game instead of the network.
    term_leaves: Vec<usize>,
    /// Per row, per player: an index into `cphi` for every config in support,
    /// packed back to back and indexed through `leaf_coff`.
    leaf_cidx: Vec<u32>,
    leaf_coff: Vec<u32>,
    /// The subgame's distinct config vectors, `[n * CFEAT]`, and the map that
    /// deduplicates them. The same config recurs at hundreds of leaves — a
    /// depth-2 subgame has a few hundred leaves over a few dozen distinct
    /// configs — and the config tower is the one part of the network whose cost
    /// scales with the support, so it runs once per distinct config per solve.
    cphi: Vec<f32>,
    cmap: std::collections::HashMap<u64, u32>,
    /// How many distinct configs `cphi` actually holds. Pooled buffers keep
    /// their length across solves, so the count cannot be read off `cphi.len()`.
    ncfg: usize,
    /// `embed` output for `cphi`: the belief embedding and the readout
    /// embedding. Both survive every CFR iteration.
    cz: Vec<f32>,
    cg: Vec<f32>,
    /// The card table `[NTYPE, de]`. The draft is fixed for the game, so this is
    /// built once per solve and read by every tower that names a card.
    ce: Vec<f32>,
    /// `rows * hidden`: the public half of the hidden layer.
    h0: Vec<f32>,
    /// `rows * PUBFEAT`: the public encoding, filled during the build.
    xpub: Vec<f32>,
    /// `rows * 2 * dg`: both players' belief embeddings.
    xb: Vec<f32>,
    /// `rows * hidden`: the hidden layer, rebuilt per iteration.
    ob: Vec<f32>,
    sb: Vec<f32>,
    /// Normalised belief weights for one leaf's support.
    wbuf: Vec<f32>,
    batch_ready: bool,
    /// Traverser of the previous leaf query, i.e. whose beliefs have moved
    /// since. `None` before the first query of a solve.
    last_traverser: Option<usize>,
    /// Working memory for the chance transitions, reused across the tree.
    draw_scratch: DrawScratch,
    /// `key << IDX_BITS | cell` scratch for ordering a public child's support.
    cell_order: Vec<u64>,
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
            (R_CPHI, &mut self.cphi),
            (R_CZ, &mut self.cz),
            (R_CG, &mut self.cg),
        ] {
            give_buf(role, std::mem::take(v));
        }
        give_snaps(std::mem::take(&mut self.snaps));
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
            root_belief: belief,
            regret: Vec::new(),
            inst: Vec::new(),
            cur: Vec::new(),
            soff: Vec::new(),
            sum_strat: Vec::new(),
            avg: Vec::new(),
            snaps: take_snaps(),
            snap_t: 0,
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
            cmap: std::collections::HashMap::new(),
            ncfg: 0,
            cz: take_buf(R_CZ),
            cg: take_buf(R_CG),
            ce: Vec::new(),
            h0: take_buf(R_H0),
            xpub: take_buf(R_XPUB),
            xb: take_buf(R_XB0),
            ob: take_buf(R_OB),
            sb: take_buf(R_SB),
            wbuf: Vec::new(),
            batch_ready: false,
            last_traverser: None,
            draw_scratch: DrawScratch::default(),
            cell_order: Vec::new(),
            dm: Default::default(),
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
            sv.build(root.clone(), cfg.depth.max(1), cfgs);
        }
        let _t = timed!(ALLOC);
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
            sv.inst.resize(sv.regret.len(), 0.0);
            sv.ncells += nc * na;
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
            sv.cur.extend_from_slice(&u);
            sv.avg.push(u);
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
        // Snapshot 0: the average before any iteration, i.e. the uniform
        // policy — the t = 0 member of the carried-belief set.
        sv.snapshot();
        sv
    }

    /// One flat copy of `avg`, aligned with `soff`: snapshot `t` is the
    /// average strategy at iterate t.
    ///
    /// Thinning: the carried beliefs are one per *kept* iterate, and the
    /// spread is in the early iterations — the late ones all repeat the final
    /// average — so only the log-spaced iterates (0, 1, 2, 4, 8, ...) plus the
    /// final one are stored. The final one is the reference strategy Phase 2
    /// and the walk act on (`value_under` reads `snaps.last()`), so it is kept
    /// however many iterations actually run.
    fn snapshot(&mut self) {
        if !self.cfg.snapshots {
            return;
        }
        let t = self.snap_t;
        self.snap_t += 1;
        if t != 0 && !t.is_power_of_two() && t < self.cfg.iters {
            return;
        }
        let mut snap = Vec::with_capacity(self.ncells);
        for v in self.avg.iter() {
            snap.extend_from_slice(v);
        }
        self.snaps.push(snap);
    }

    // ------------------------------------------------------------ tree build

    fn build(&mut self, s: State, depth: usize, cfgs: [Rc<[Config]>; 2]) -> usize {
        let player = s.to_act();
        // A plain round-start draw is walked through (one public child, no
        // depth cost). Other chance nodes (Warrior Priest draws — excluded
        // from every draft) stay leaves, as do depth-0 nodes and terminals.
        let draw_pass = matches!(s.pending(), Cont::Draw { .. });
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

        assert!(nc * na < 1 << IDX_BITS, "decision node over the index width");
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
                        ent.push((n.key() << IDX_BITS) | (ci * na + a) as u64);
                    }
                }
            }
            ent.sort_unstable();
            let sup = &mut child_cfgs[ch];
            let mut prev = u64::MAX;
            for &packed in ent.iter() {
                let (k, cell) = (packed >> IDX_BITS, (packed & IDX_MASK) as usize);
                if k != prev {
                    prev = k;
                    let (ci, a) = (cell / na, cell % na);
                    sup.push(advance_config(&mine[ci], aslot[a], fdown[a]).unwrap());
                }
                trans[cell] = (sup.len() - 1) as i32;
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
            // One depth unit per *completed coin play*, not per decision node.
            // Every legal action at a main-play node spends exactly one coin
            // and every action at a micro node spends none, so the node-level
            // structural test decides the whole observation group and no
            // per-action scan is needed. The known future divergence:
            // Cont::WarriorPriestPlay spends a real coin without being
            // MainPlay — when the Warrior Priest re-enters the draft pool the
            // predicate becomes `MainPlay | WarriorPriestPlay`, and the debug
            // assertion below exists so that day is a test failure, not a
            // silent depth miscount.
            let spends = matches!(s.pending(), Cont::MainPlay);
            debug_assert_eq!(
                spends,
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
        let cur = std::mem::take(&mut self.cur);
        let root = [
            self.root_belief[0].p.clone(),
            self.root_belief[1].p.clone(),
        ];
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
        let at = self.leaf_rows.len() * PUBFEAT;
        if self.xpub.len() < at + PUBFEAT {
            // Grow in chunks so the zero-fill happens a handful of times per
            // solve, and not at all once the pooled buffer is warm.
            self.xpub.resize(at + 64 * PUBFEAT, 0.0);
        }
        write_public_features(s, self.ctx, &mut self.xpub[at..at + PUBFEAT]);
        for p in 0..2 {
            let res = reserve(s, p as u8, self.ctx);
            self.leaf_coff.push(self.leaf_cidx.len() as u32);
            for c in cfgs[p].iter() {
                let idx = self.intern_config(c, &res, p);
                self.leaf_cidx.push(idx);
            }
        }
        self.leaf_rows.push(id);
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
        self.leaf_coff.push(self.leaf_cidx.len() as u32);
        let rows = self.leaf_rows.len();
        if self.nets.value.is_empty() {
            return;
        }
        let net = &self.nets.value;
        debug_assert_eq!(net.pub_dim(), PUBFEAT);
        debug_assert_eq!(net.cfeat(), CFEAT);
        if self.xb.len() < rows * net.belief_dim() {
            self.xb.resize(rows * net.belief_dim(), 0.0);
        }
        let _t = timed!(PUBNET);
        let xpub = std::mem::take(&mut self.xpub);
        // The cards in play are fixed at the draft, so every leaf of the subgame
        // carries the same card block and the table is built once. Everything
        // downstream — the hex block, the pile summary, the holding tower, the
        // action tower — reads a row of it by coin-type index.
        if rows > 0 {
            net.cards(&xpub[..PUBFEAT], &mut self.ce);
        }
        net.trunk(&xpub, rows, PUBFEAT, &self.ce, &mut self.sb, &mut self.h0);
        self.xpub = xpub;
        let cphi = std::mem::take(&mut self.cphi);
        net.embed(&cphi[..self.ncfg * CFEAT], self.ncfg, &self.ce, &mut self.cz, &mut self.cg);
        self.cphi = cphi;
    }

    /// Fill `vals` at every leaf with the traverser's counterfactual values.
    fn leaf_values(&mut self, traverser: usize) {
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
            net.pbs_head(&self.xb[..rows * 2 * dg], rows, &self.h0, &mut self.sb, &mut self.ob);
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
                crate::net::accumulate(
                    cz,
                    &cidx[cs..cs + n],
                    &wbuf[..n],
                    dg,
                    &mut xb[at..at + dg],
                );
            }
        }
        let _t = timed!(NET);
        let net = &self.nets.value;
        net.pbs_head(&self.xb[..rows * 2 * dg], rows, &self.h0, &mut self.sb, &mut self.ob);
    }

    /// Per-config leaf values for player `p` — counterfactual: the network's
    /// value for that exact config times the opponent's unnormalised reach
    /// into the leaf. Runs off the `ob` left by the last `leaf_values` /
    /// `leaf_values_both`, so two players can be read off one PBS-head pass.
    fn readout(&mut self, p: usize) {
        let _t = timed!(LEAFPOST);
        let empty = self.nets.value.is_empty();
        let opp = 1 - p;
        for k in 0..self.term_leaves.len() {
            let i = self.term_leaves[k];
            let opp_reach: f32 = self.reach_of(i, opp).iter().sum();
            let u = self.nodes[i].s.utility(p);
            let n = self.nc[i][p] as usize;
            let vo = self.voff[i] as usize;
            self.vals[vo..vo + n].fill(u * opp_reach);
        }
        let rk = if empty { 0 } else { self.nets.value.rank() };
        let (net, reach, roff, ncs, voff, coff, cidx, ob, cg, vals) = (
            &self.nets.value,
            &self.reach,
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.ob,
            &self.cg,
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
            // Only the player's own configs are ever looked up here. The
            // opponent's private state reaches this value solely through the
            // belief embedding, which is what keeps the query leak-free.
            let cs = coff[2 * r + p] as usize;
            net.values(
                &ob[r * rk..r * rk + rk],
                cg,
                &cidx[cs..cs + n],
                &mut vals[vo..vo + n],
            );
            for v in vals[vo..vo + n].iter_mut() {
                *v *= opp_reach;
            }
        }
    }

    fn update_regrets(&mut self, traverser: usize) {
        // Reaches are already consistent with `cur`: `new` establishes that,
        // every `step` re-establishes it after regret matching, and the
        // fixed-policy passes restore it before returning, so recomputing them
        // here would repeat the previous pass exactly.
        self.leaf_values(traverser);
        let cur = std::mem::take(&mut self.cur);
        self.backprop(traverser, &cur, Back::Regret);
        self.cur = cur;
    }

    /// One value backpropagation over the tree for `traverser`: the shared walk
    /// behind CFR (`update_regrets`), TurboReBeL's fixed-policy passes
    /// (`value_under`) and the best response (`nash_conv`). `mode` picks what
    /// the traverser's own decision nodes do with their children's values —
    /// average under `strat`, average and record the regret, or take the max.
    fn backprop(&mut self, traverser: usize, strat: &[f32], mode: Back) {
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
                let cur = &strat[so..];
                if mode != Back::Value {
                    self.inst[so..so + nc * na].fill(0.0);
                }
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
                        match mode {
                            Back::Regret => {
                                self.inst[so + c * na + a] += av;
                                vi[c] += av * cur[c * na + a];
                            }
                            Back::Value => vi[c] += av * cur[c * na + a],
                            // Records the regret too: `warm_start` wants the
                            // regrets a best response leaves, and nothing else
                            // reads `inst` until the next traversal fills it.
                            Back::BestResponse => {
                                self.inst[so + c * na + a] += av;
                                vi[c] = vi[c].max(av);
                            }
                        }
                    }
                }
                match mode {
                    Back::Regret => {
                        for c in 0..nc {
                            let base = vi[c];
                            for a in 0..na {
                                if n.legal[c * na + a] {
                                    self.inst[so + c * na + a] -= base;
                                }
                            }
                        }
                    }
                    Back::BestResponse => {
                        for c in 0..nc {
                            if vi[c] == f32::NEG_INFINITY {
                                vi[c] = 0.0;
                            }
                            let base = vi[c];
                            for a in 0..na {
                                if n.legal[c * na + a] {
                                    self.inst[so + c * na + a] -= base;
                                }
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
        // Fold this traversal's instantaneous regret into the accumulated one,
        // discount, and regret-match. `Cfr` says how: see its table.
        //
        // Regret matching floors at EPS rather than at zero, so every legal
        // action keeps positive probability in every iterate. That is not
        // cosmetic — `carried_beliefs` hands the self-play walk a belief per
        // iterate, and the walk asserts that each has the same support as the
        // live one. A hard zero here would drop configs and fail that assert.
        const EPS: f32 = 1e-6;
        {
            let _t = timed!(RM);
            let k = self.cfg.cfr;
            let m = self.steps[traverser] as f32 + 1.0;
            let (da, db) = (Cfr::factor(m, k.alpha), Cfr::factor(m, k.beta));
            let dg = (m / (m + 1.0)).powf(k.gamma);
            for i in 0..self.nodes.len() {
                let n = &self.nodes[i];
                if n.leaf || n.chance || n.player as usize != traverser {
                    continue;
                }
                let (na, nc) = (n.na(), n.nc(traverser));
                let so = self.soff[i] as usize;
                let regret = &mut self.regret[so..];
                let inst = &self.inst[so..];
                let cur = &mut self.cur[so..];
                for c in 0..nc {
                    let mut sum = 0.0;
                    for a in 0..na {
                        let j = c * na + a;
                        if !n.legal[j] {
                            cur[j] = 0.0;
                            continue;
                        }
                        let r = regret[j] * if regret[j] > 0.0 { da } else { db } + inst[j];
                        regret[j] = r;
                        let v = (r + k.predict * inst[j]).max(EPS);
                        cur[j] = v;
                        sum += v;
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for a in 0..na {
                            cur[c * na + a] *= inv;
                        }
                    }
                }
                for x in self.sum_strat[i].iter_mut() {
                    *x *= dg;
                }
            }
        }
        // Restore the reach probabilities under the strategy just computed:
        // the next iteration's traversal reads them, and so does the average
        // strategy accumulation below.
        self.precompute_reaches();
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
        self.snapshot();
        self.steps[traverser] += 1;
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
        let root = [
            self.root_belief[0].p.clone(),
            self.root_belief[1].p.clone(),
        ];
        self.propagate(&reference, [&root[0], &root[1]]);
        self.leaf_values_both();
        let (mut nash, mut zero_sum) = (0.0, 0.0);
        for p in 0..2usize {
            // One `readout` serves both passes: `backprop` skips leaves, so the
            // leaf values it left are still there for the second walk.
            self.readout(p);
            let vo = self.voff[0] as usize;
            let nc = self.nc[0][p] as usize;
            let expect = |v: &[f32]| -> f32 {
                (0..nc).map(|c| root[p][c] * v[vo + c]).sum()
            };
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
        let root = [
            self.root_belief[0].p.clone(),
            self.root_belief[1].p.clone(),
        ];
        let n = self.snaps.len().saturating_sub(1);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let snap = self.snaps[i].clone();
            self.propagate(&snap, [&root[0], &root[1]]);
            let mut pair = [Vec::new(), Vec::new()];
            for p in 0..2usize {
                let ra = self.roff[leaf] as usize
                    + if p == 1 { self.nc[leaf][0] as usize } else { 0 };
                let n = self.nc[leaf][p] as usize;
                let mut w = vec![0.0; n];
                normalize_weights(&self.reach[ra..ra + n], &mut w);
                pair[p] = w;
            }
            out.push(pair);
        }
        out
    }

    /// Seed the solve from the policy head instead of from a uniform strategy,
    /// as ReBeL's Appendix J does (after Brown & Sandholm 2016): take the
    /// policy, compute an exact best response to it, and start CFR as though
    /// that policy had already been played for `weight` iterations.
    ///
    /// The best response is the pass `nash_conv` already needs, run with regret
    /// recording on, so nothing new is computed — the regrets it leaves are the
    /// ones CFR would have accumulated against the policy, and scaling them is
    /// the whole of the warm start.
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
            let cur = std::mem::take(&mut self.cur);
            self.backprop(p, &cur, Back::BestResponse);
            self.cur = cur;
            for i in 0..self.nodes.len() {
                let n = &self.nodes[i];
                if n.leaf || n.chance || n.player as usize != p {
                    continue;
                }
                let (na, nc) = (n.na(), n.nc(p));
                let so = self.soff[i] as usize;
                for j in 0..nc * na {
                    self.regret[so + j] = weight * self.inst[so + j];
                }
                // The average strategy starts as though the policy had been
                // played for those iterations, which is what makes the seeded
                // regrets and the average consistent with each other.
                let r: Vec<f32> = self.reach_of(i, p).to_vec();
                for c in 0..nc {
                    for a in 0..na {
                        self.sum_strat[i][c * na + a] = weight * r[c] * self.cur[so + c * na + a];
                    }
                }
            }
            self.steps[p] = weight as usize;
        }
        // The iterate-0 snapshot was taken from the uniform strategy in `new`;
        // the solve now starts somewhere else, so it is retaken.
        self.recompute_avg();
        self.snaps.clear();
        self.snap_t = 0;
        self.snapshot();
    }

    /// Write the policy head's distribution into `cur` at every decision node.
    /// Returns false if the network has no usable policy.
    fn policy_into_cur(&mut self) -> bool {
        let net = &self.nets.value;
        let inner: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].leaf && !self.nodes[i].chance)
            .collect();
        if inner.is_empty() {
            return false;
        }
        // A depth-2 tree has a couple of dozen decision nodes against several
        // hundred leaves, so encoding them costs a few percent of the batch the
        // solve already builds.
        let (mut xpub, mut phi, mut coff) = (Vec::new(), Vec::new(), vec![0u32]);
        for &i in &inner {
            let at = xpub.len();
            xpub.resize(at + PUBFEAT, 0.0);
            write_public_features(&self.nodes[i].s, self.ctx, &mut xpub[at..at + PUBFEAT]);
            for p in 0..2usize {
                let res = reserve(&self.nodes[i].s, p as u8, self.ctx);
                for c in self.nodes[i].cfgs[p].iter() {
                    let at = phi.len();
                    phi.resize(at + CFEAT, 0.0);
                    write_config_feats(c, &res, p, &mut phi[at..at + CFEAT]);
                }
                coff.push((phi.len() / CFEAT) as u32);
            }
        }
        let rows = inner.len();
        let (dg, mut z, mut g, mut sb, mut pre) =
            (net.dg(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        net.embed(&phi, phi.len() / CFEAT, &self.ce, &mut z, &mut g);
        net.trunk(&xpub, rows, PUBFEAT, &self.ce, &mut sb, &mut pre);

        let (mut q, mut logit, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let mut psi = Vec::new();
        let mut xbel = vec![0.0f32; 2 * dg];
        for (r, &i) in inner.iter().enumerate() {
            let (na, me) = (self.nodes[i].na(), self.nodes[i].player as usize);
            // The belief the head reads is the normalised reach, as everywhere
            // else. At this point the reaches are the uniform ones `new` left.
            xbel.iter_mut().for_each(|x| *x = 0.0);
            for p in 0..2usize {
                let (lo, hi) = (coff[2 * r + p] as usize, coff[2 * r + p + 1] as usize);
                w.resize(hi - lo, 0.0);
                normalize_weights(self.reach_of(i, p), &mut w);
                let idx: Vec<u32> = (lo as u32..hi as u32).collect();
                crate::net::accumulate(&z, &idx, &w, dg, &mut xbel[p * dg..(p + 1) * dg]);
            }
            psi.resize(na * AFEAT, 0.0);
            for a in 0..na {
                let n = &self.nodes[i];
                write_action_feats(&n.acts[a], self.ctx, me, n.aslot[a], n.fdown[a],
                                   &mut psi[a * AFEAT..(a + 1) * AFEAT]);
            }
            net.embed_actions(&psi, na, &self.ce, &mut q);
            let (lo, hi) = (coff[2 * r + me] as usize, coff[2 * r + me + 1] as usize);
            let idx: Vec<u32> = (lo as u32..hi as u32).collect();
            logit.resize(idx.len() * na, 0.0);
            net.policy(&xbel, &pre[r * net.hidden()..], &z, &idx, &q, na, &mut sb, &mut logit);
            // Softmax over the *legal* actions of each config, with the same
            // floor regret matching uses, so every legal action keeps positive
            // probability and the carried beliefs keep full support.
            let so = self.soff[i] as usize;
            let n = &self.nodes[i];
            for c in 0..idx.len() {
                let row = &logit[c * na..(c + 1) * na];
                let m = (0..na)
                    .filter(|&a| n.legal[c * na + a])
                    .fold(f32::NEG_INFINITY, |m, a| m.max(row[a]));
                let mut sum = 0.0;
                for a in 0..na {
                    let v = if n.legal[c * na + a] { (row[a] - m).exp() } else { 0.0 };
                    self.cur[so + c * na + a] = v;
                    sum += v;
                }
                if sum > 0.0 {
                    for a in 0..na {
                        let x = &mut self.cur[so + c * na + a];
                        *x = (*x / sum).max(if n.legal[c * na + a] { 1e-6 } else { 0.0 });
                    }
                }
            }
        }
        true
    }

    /// Rebuild `avg` from `sum_strat`, normalised per config.
    fn recompute_avg(&mut self) {
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance {
                continue;
            }
            let (na, nc) = (n.na(), n.nc(n.player as usize));
            for c in 0..nc {
                let sum: f32 = self.sum_strat[i][c * na..(c + 1) * na].iter().sum();
                let k = (0..na).filter(|&a| n.legal[c * na + a]).count().max(1) as f32;
                for a in 0..na {
                    self.avg[i][c * na + a] = if sum > 0.0 {
                        self.sum_strat[i][c * na + a] / sum
                    } else if n.legal[c * na + a] {
                        1.0 / k
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    /// The CFR average strategy: the approximate equilibrium of the subgame.
    /// Acting and belief propagation use it — the reference strategy of
    /// TurboReBeL's Phase 2 and of the walk through the solved tree.
    pub fn average_strategy(&self, node: usize, c: usize) -> &[f32] {
        let na = self.nodes[node].na();
        &self.avg[node][c * na..(c + 1) * na]
    }
}
