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
use crate::board::{N_HEXES, NONE};
use crate::farm::{Call, Dst, Reply, Writes};
use crate::net::Net;
use crate::rng::Rng;
use crate::pbs::*;
use crate::state::{Cont, State};
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};
use crate::timed;
use std::sync::Arc;

/// Trajectories an expansion phase may draw for each distinct leaf it owes.
///
/// The tree is frozen for a whole round, so a phase draws again when it lands
/// on a leaf the round already took, and this bounds the redrawing. Measured
/// at `SoG(512, 8)`, a round of one spends 1.4 draws a leaf, a round of four
/// 1.8 and a round of eight 2.3.
///
/// Four, because raising it buys almost nothing. Half the selection rule is
/// the CFR average, which does not read the visit counts at all, so a large
/// round keeps drawing the same lines however long it is allowed to. Taking
/// the bound from four to forty at a round of eight moved 7.6 draws a leaf
/// where four had cost 2.3, and returned three percent more tree. The
/// remaining loss is the rule's, not the bound's.
///
/// Read by the device backend too: the rule is one rule, and a phase that gave
/// up at a different point on the two would leave them drawing from different
/// points of the same stream ever after.
pub const TRIES: usize = 4;

/// What one solve may hold, in the terms every arena it owns is linear in.
///
/// A slot's arenas are allocated once at this size and reused by every solve
/// that ever runs in it, so this is a *bound* and not a forecast. An expansion
/// that would take the solve past any of these is abandoned exactly as one that
/// runs away past `EXPANSION_CAP` is: the arenas are rewound, the leaf it
/// started from is marked unexpandable, and the solve carries on iterating on
/// the tree it already has. Nothing anywhere else has to ask how large a solve
/// is, because the answer is this and it is the same for every solve.
///
/// Each term is named for the arenas it sizes:
///
/// | term | what it sizes |
/// |---|---|
/// | `nodes` | the flat tree arrays; a `TNode` and a `State` each on the host |
/// | `rows` | `board_of`, `leaf_node`, `term`, `coff`, the whole leaf pass |
/// | `boards` | `p` and `jp` -- the trunk runs once per distinct public state |
/// | `configs` | `f`, `g`, `fp`, `cphi`: the config encoder's rows |
/// | `cidx` | the belief index, one entry per (row, player, config) |
/// | `reach` | `reach`, `vals` and the `nc + 1` offsets beside them |
/// | `cells` | `cur` `regret` `sum` `qval` `visits` `prior`, and the per-cell tree arrays |
/// | `draws` | the draw transition, forward and transposed |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub nodes: usize,
    pub rows: usize,
    pub boards: usize,
    pub configs: usize,
    pub cidx: usize,
    pub reach: usize,
    pub cells: usize,
    pub draws: usize,
}

impl Budget {
    /// The budget for a solve of `s` expansions.
    ///
    /// `s` counts distinct expansions and an expansion adds a bounded piece of
    /// tree, so every term is linear in it. `BUDGET_512` is measured; this is
    /// that one proportion, and it is the only arithmetic in the design.
    pub fn for_s(s: u32) -> Budget {
        let k = |at512: usize| (at512 * s as usize / 512).max(1);
        Budget {
            nodes: k(BUDGET_512.nodes),
            rows: k(BUDGET_512.rows),
            boards: k(BUDGET_512.boards),
            // Not scaled: interning keys on the node's reserve. A first
            // expansion interned 2096 against a p90 of 1074; the slot is 4096.
            configs: BUDGET_512.configs,
            cidx: k(BUDGET_512.cidx),
            reach: k(BUDGET_512.reach),
            cells: k(BUDGET_512.cells),
            draws: k(BUDGET_512.draws),
        }
    }
}

impl Default for Budget {
    fn default() -> Budget {
        Budget::for_s(512)
    }
}

impl Budget {
    /// No budget at all.
    ///
    /// `Solver::grow_full` builds a whole subgame for the CPU oracle to compare
    /// against, and a bound on that is a bound on the reference. Nothing a run
    /// uses takes this: it is the tests' way of saying they want the whole
    /// game, and saying it out loud rather than through a second growth path.
    pub fn unbounded() -> Budget {
        Budget {
            nodes: usize::MAX,
            rows: usize::MAX,
            boards: usize::MAX,
            configs: usize::MAX,
            cidx: usize::MAX,
            reach: usize::MAX,
            cells: usize::MAX,
            draws: usize::MAX,
        }
    }
}

impl Budget {
    /// Device bytes one slot holds, arena by arena.
    ///
    /// A slot allocates every one of these once, at this size, and reuses them
    /// for every solve that ever runs in it. So this is not an estimate of what
    /// a solve will take: it is what the slot *is*, and `Solve::bytes` on the
    /// device must equal it exactly -- there is a test that says so.
    ///
    /// Written as the arena list rather than a formula because the arena list
    /// is the thing that can change. Adding a device array without a line here
    /// should look like the omission it is.
    pub fn device_bytes(&self) -> usize {
        use crate::net::{D, JW, POOL};
        let f = 4;
        let u = 4;
        // The leaf pass: board vectors a board, the config encoder's rows a
        // config, and the index that joins them a row.
        let leaf = self.boards * (D + JW) * f
            + self.configs * (2 * D + POOL) * f
            + self.cidx * u
            + (2 * self.rows + 1) * u
            + 2 * self.rows * u
            + self.configs * D * f;
        // The CFR arenas: six a cell, and reach and values over the supports.
        let cfr = 6 * self.cells * f + self.reach * f + 2 * self.reach * f;
        // The tree, in the four terms the contract's arrays are indexed by.
        // Per node, all four bytes wide on the card: kind, player, exhausted,
        // both config counts, parent, roff, voff, soff, util, child_at,
        // child_n, legal_base, rev_base, rvd_base, draw_base, the child slot
        // every node but the root fills, and a level bound each way.
        // `draw_start` is a config CSR like `legal_off`, so it sits on reach.
        let tree = self.nodes * 19 * u
            + (self.reach + self.nodes) * 5 * u
            + self.cells * 6 * u
            + self.draws * 4 * u;
        leaf + cfr + tree + 8
    }

    /// Host bytes one solve holds at this budget, when the card keeps the CFR
    /// state -- which is every solve the farm runs.
    ///
    /// The `Solver`'s own CFR arenas (`regret`, `prior`, `visits`, `qval`,
    /// `sum`, `reach`, `vals`) are absent then, and they are most of a solve
    /// on the reference path. What is left is the tree, the states its nodes
    /// stand on, the description the card reads, and the leaf batch.
    pub fn host_bytes(&self) -> usize {
        use crate::net::D;
        use crate::pbs::PUBFEAT;
        let f = 4;
        let u = 4;
        let node = std::mem::size_of::<TNode>() + std::mem::size_of::<crate::state::State>();
        // A `TNode` owns per-action and per-cell vectors of its own.
        let owned = self.cells * 6 * u + self.reach * u;
        let contract = self.device_bytes() - self.boards * (D + crate::net::JW) * f;
        let batch = self.cidx * u
            + 2 * self.rows * u
            + self.boards * PUBFEAT * f * 2
            + self.configs * crate::pbs::CFEAT * f
            + self.rows * u;
        let readout = self.rows * D * f + self.configs * D * f;
        self.nodes * node + owned + contract + batch + readout + self.cells * f
    }
}

/// The shape a solve at `SoG(512, 8)` is allowed.
///
/// `nodes` `rows` `boards` are the ninetieth percentile `examples/shapes`
/// measured over a corpus of real roots. The other terms are a first
/// expansion: that one is not a budget, because a root that stayed a leaf has
/// no strategy, so the slot has to hold it. Measured on the training farm at
/// `s = 512`: a first expansion interned 2096 configs, wrote 199271 cells,
/// 393223 cidx, 724360 draws. `budget_hits` is then the later expansions that
/// would have grown past this.
const BUDGET_512: Budget = Budget {
    nodes: 24_582,
    rows: 14_516,
    boards: 8_518,
    configs: 4_096,
    cidx: 524_288,
    reach: 1_048_576,
    cells: 262_144,
    draws: 1_048_576,
};

#[derive(Clone, Copy)]
pub struct Cfg {
    /// Expansions the whole solve makes — Student of Games' `s`.
    ///
    /// Distinct ones. A phase draws trajectories until it has leaves the round
    /// has not taken yet, so `s` is a count of nodes the tree gains and not of
    /// trajectories walked; see `TRIES`.
    ///
    /// One simulation walks to a leaf and expands it, and an expansion adds one
    /// public state together with *every* one of its public children. That is
    /// their `k = infinity`, which is what both of their imperfect-information
    /// games use and is not a budget choice: CFR needs a counterfactual value
    /// at every action of a decision node, so a partially expanded node has no
    /// regret to update.
    pub s: u32,
    /// Expansions per regret update — Student of Games' `c`. Below one it is
    /// several regret updates per expansion, which is the same schedule read
    /// the other way round.
    pub c: f32,
    /// Regret updates one round carries, at least one.
    ///
    /// A round is the unit the host and the backend trade in: the host grows
    /// the tree, the backend runs the round's updates back to back and takes
    /// `c` leaves after each of them, and the host grows again from every leaf
    /// the round took. So the tree lags up to `batch - 1` updates behind the
    /// trajectories that chose it, and the per-round cost of describing a tree
    /// that did not change is paid once for `batch` updates instead of once
    /// each. What it costs is tree, because the phases of one round draw
    /// against a strategy that does not move: they take distinct leaves, but a
    /// large round runs out of lines to sample down and takes fewer than it
    /// owes. Measured at `SoG(512, 8)`, a round of four builds nine percent
    /// less tree than a round of one and a round of eight twenty-three.
    pub batch: usize,
    /// Round boundaries growth may grow *through*.
    ///
    /// Zero is DeepStack's street boundary, and what the solver has always
    /// done: the round-start draw is put into the tree, priced by the value
    /// network, and nothing under it is ever expanded. Student of Games has no
    /// such limit, which is `u8::MAX`.
    ///
    /// What it saves is tree and not belief. The draw's broadened support is
    /// already in the tree at zero, sitting there as a priced leaf, so a node
    /// past the boundary carries no more configs than one before it -- 4.5 a
    /// row against 4.6 at `SoG(512, 8)`. Taking the limit off costs 1.9x the
    /// rows and 2.1x the readouts, and leaves a harder game: `nash_conv` at 64
    /// updates goes 0.031 to 0.057.
    pub rounds: u8,
    /// Iterations between two full leaf queries of the value network.
    ///
    /// One is what the solver has always done: every CFR iteration re-runs the
    /// join and the readout at every leaf row, sixty-four times a leaf. Above
    /// one, a leaf's per-config values `v(c)` are kept from the last query and
    /// only re-scaled by the opponent's current reach mass, so beliefs and
    /// reaches still move every iteration and only the network's opinion of
    /// them is held. Zero never re-queries: a row is valued when it is created
    /// and again in the final value pass, which is the target, and nowhere in
    /// between.
    ///
    /// A row growth has just added has nothing to reuse, so it is always
    /// queried on the first iteration it exists whatever this says. That is
    /// also why `refresh` aligned with `batch` is the natural setting: the
    /// round's first iteration has to run the network for the new rows anyway.
    ///
    /// Host path only. The card holds no `h` per row between iterations.
    pub refresh: u32,
    /// The regret-update rule.
    pub cfr: Cfr,
    /// PUCT's exploration constant, weighting the prior against the search's
    /// own action values during the expansion phase.
    pub puct: f32,
    /// Softmax temperature on the policy head's logits where they are read as
    /// the PUCT prior. Above one this flattens the prior, which Student of
    /// Games notes "can decrease weight of the prior in some games and
    /// encourage more exploration in the search phase".
    pub prior_temp: f32,
    /// What one solve may hold, and so how large a slot is. Admission is a free
    /// list because of this and nothing else.
    pub budget: Budget,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            s: 512,
            c: 8.0,
            batch: 8,
            rounds: 0,
            refresh: 1,
            cfr: Cfr::SOG,
            puct: 1.5,
            prior_temp: 1.0,
            budget: Budget::for_s(512),
        }
    }
}

impl Cfg {
    /// Regret updates the solve runs: `ceil(s / c)`, never stored.
    ///
    /// `c = 0` is the degenerate schedule that never expands, and then `s` is
    /// read as the update count. Nothing in a run sets it; the device parity
    /// tests do, because comparing two backends needs both to solve the same
    /// tree and a sampled expansion is where they would part company.
    pub fn iters(&self) -> usize {
        if self.c <= 0.0 {
            return self.s as usize;
        }
        (self.s as f32 / self.c).ceil() as usize
    }

    /// Whether the iteration after `done` of them re-queries the network at
    /// every leaf.
    ///
    /// The first iteration of every `refresh`, so `refresh = batch` puts the
    /// query at the start of a round and nowhere else in it.
    fn refresh_due(&self, done: usize) -> bool {
        self.refresh > 0 && done % self.refresh as usize == 0
    }

    /// Expansions the solve owes after its `i`-th regret update, one-based.
    ///
    /// `floor(i * c)` expansions have been earned by then, capped at `s`, so
    /// this is the difference between two of those. It spreads `c` over the
    /// iterations whether `c` is above one or below it.
    fn expansions_at(&self, i: usize) -> usize {
        if self.c <= 0.0 {
            return 0;
        }
        let earned = |k: usize| (self.s as usize).min((k as f32 * self.c).floor() as usize);
        earned(i) - earned(i - 1)
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

    /// What Student of Games runs in its regret update phase, verbatim:
    /// "simultaneous updates, regret-matching+, and linearly-weighted policy
    /// averaging".
    ///
    /// Regret matching+ is `alpha = inf, beta = -inf` — accumulated positive
    /// regret is undiscounted and negative accumulated regret is floored at
    /// zero, which is `Q_t(s,a) = (Q_{t-1}(s,a) + r_t(s,a))^+`.
    ///
    /// The averaging weight needs care. A running sum multiplied by
    /// `(t / (t + 1))^gamma` before each iterate is added gives iterate `j` a
    /// weight proportional to `(j + 1)^gamma` at the end, so *linear*
    /// weighting is `gamma = 1` and not the 2 that `PLUS` carries.
    ///
    /// Simultaneous updates are not in this constant — they are `Solver::step`
    /// traversing both players against one reach profile.
    pub const SOG: Cfr = Cfr {
        alpha: f32::INFINITY,
        beta: f32::NEG_INFINITY,
        gamma: 1.0,
        predict: 0.0,
    };

    /// The named variants, for the tools that sweep them.
    pub const NAMED: [(&'static str, Cfr); 6] = [
        ("linear", Cfr::LINEAR),
        ("plus", Cfr::PLUS),
        ("dcfr", Cfr::DISCOUNTED),
        ("pcfr", Cfr::PREDICTIVE),
        ("sapcfr", Cfr::SIMPLE_ASYM),
        ("sog", Cfr::SOG),
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

/// The same units, summed over the iterations that ran on them.
///
/// The device charges per iteration for the join, the readout, the pooling and
/// the two sweeps, and a growing tree is a different size at every one, so a
/// final `Shape` prices none of them. `Solver::step` adds a term.
#[derive(Default, Clone, Copy, Debug)]
pub struct Trace {
    pub iters: u64,
    pub row_iters: u64,
    pub cidx_iters: u64,
    pub cell_iters: u64,
    /// Rows the join actually ran over, summed across every query the solve
    /// made — two a regret update, plus the fixed-policy passes. `row_iters`
    /// is the nominal that assumes every iteration queries every row; under
    /// `Cfg::refresh` above one it does not, and this is what it cost.
    pub join_rows: u64,
    /// Config values the readout actually formed from the network, on the same
    /// terms. The re-scaling of a cached value is not counted: it is a
    /// multiply, where this is a dot product of width `D`.
    pub readout_cfgs: u64,
}

/// What a solve built, in the units the device charges for. See `Solver::shape`.
#[derive(Default, Clone, Copy, Debug)]
pub struct Shape {
    pub nodes: usize,
    /// Network rows: one per decision node at a coin play, ever created. The
    /// join runs over all of them every iteration, including the rows of nodes
    /// that growth has since made interior.
    pub rows: usize,
    /// Distinct public states among those rows. The trunk runs once per board,
    /// not once per row: coin plays commute, so a tree spanning one round
    /// reaches the same public state several ways.
    pub boards: usize,
    pub cells: usize,
    /// Distinct configs the solve interned. `f` is `[ncfg, D]` and the readout
    /// gathers rows of it, so this is the working set that decides whether the
    /// gather hits L2 or memory.
    pub ncfg: usize,
    /// Belief-index entries over the rows: one per (row, player, config in
    /// support). The readout and the belief pooling each run once per entry
    /// per iteration.
    pub cidx: usize,
    pub depth: usize,
    /// The widest legal-action row in the tree: actions one config of one
    /// decision node offers. `cells` is the sum of these over the tree, and
    /// this is its tail.
    pub acts: usize,
    /// The largest config support at any node, over both players. The exact
    /// maximum the game admits is 4 628 -- reserve (5,5,5,5,1), hand 3,
    /// face-down 9 -- so this says how far a real tree is from it.
    pub support: usize,
    /// Reach entries: one per (node, player, config in support). `reach` is per
    /// entry, `vals` is the larger of the two supports a node, and the `nc + 1`
    /// offset arrays beside them are per entry too.
    pub reach: usize,
    pub vals: usize,
    /// Draw-transition entries, forward and transposed: `draw_to`, `draw_p`,
    /// `rvd_src` and `rvd_p` are per entry.
    pub draws: usize,
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
    /// terminal, which is the case `tests/sog_solver.rs` pins against an
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
    /// Whether the backend runs the CFR loop itself. A device does: the arenas
    /// are tens of megabytes a round trip, so a solve that kept them on the
    /// host would spend everything it had on the bus. Without it the solver
    /// walks its own tree and only the network is batched, which is the
    /// reference every device answer is held to.
    pub device: bool,
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
    /// Whether growth may still turn this leaf into a decision node. False
    /// past the draw that starts the next round, false for a leaf whose own
    /// expansion was abandoned, and false for a terminal, which has nothing to
    /// grow. Such a leaf is priced by the value network -- or, at a terminal,
    /// by the game -- which is what every other leaf gets anyway.
    pub expandable: bool,
    /// Whether the subtree under this node holds no expandable leaf at all.
    ///
    /// Growth only ever takes it away, never gives it back, so the flag
    /// spreads towards the root and never retreats. The expansion phase reads
    /// it: a trajectory that walked into an exhausted subtree could only end
    /// on a leaf nothing may grow, and the simulation would be spent for
    /// nothing. See `Solver::seal`.
    pub exhausted: bool,
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
    pub cfgs: [Arc<[Config]>; 2],
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

/// The length of every append-only arena a `grow` extends, so an abandoned one
/// can put them all back. See `Solver::rewind`.
#[derive(Clone, Copy)]
struct Mark {
    nodes: usize,
    ncells: usize,
    ncfg: usize,
    ndraws: usize,
    leaf_rows: usize,
    nboards: usize,
    term_leaves: usize,
    leaf_coff: usize,
    leaf_cidx: usize,
}

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
    /// The root's average policy — Student of Games' policy target, "the
    /// output policies for all information states within the root public
    /// state, computed in the regret update phase".
    ///
    /// `acts` describes each of the root's actions the way the policy head
    /// reads one, so a stored row can rebuild `e(a)` without keeping a whole
    /// `State`. `off`, `act` and `p` are the acting player's configs in
    /// belief order, each with its legal cells and their probability.
    pub policy: Policy,
}

/// One public state's action list and the average policy over it.
#[derive(Default, Clone)]
pub struct Policy {
    pub acts: Vec<[u8; ACT_BYTES]>,
    pub off: Vec<u32>,
    pub act: Vec<u8>,
    pub p: Vec<f32>,
}

/// How an action is stored in a replay row: kind, the coin slot it spends
/// (offset by one so `-1` is zero), and the three squares it names.
pub const ACT_BYTES: usize = 5;

/// The arenas one expansion phase reads, as whichever backend ran the CFR
/// loop left them. `Solver::replay_expansion` is the only consumer.
pub struct Arenas<'a> {
    pub reach: &'a [f32],
    pub cur: &'a [f32],
    pub sum: &'a [f32],
    pub qval: &'a [f32],
    pub visits: &'a [f32],
    pub prior: &'a [f32],
}

/// Sum `K` running totals over `n` terms the way a warp of thirty-two does:
/// the lanes stride through the terms, then a butterfly folds the lanes.
///
/// The expansion phase runs on the card in production, and every total it
/// forms is this shape. f32 addition is not associative, so a host that summed
/// a row straight through would put the cell boundaries of a sampled draw a
/// few ulps elsewhere and take a different turn often enough to build a
/// different tree. It sums the same way instead. Growth is the one place this
/// matters: everywhere else a few ulps stay a few ulps, and here they decide
/// which node exists.
///
/// Only the lower half is folded at each step, and that is exact rather than a
/// shortcut: IEEE addition is commutative, so `a[k] + a[k^s]` and
/// `a[k^s] + a[k]` are the same bits and lanes `k` and `k^s` stay equal all the
/// way down. The card's butterfly leaves every lane holding this value.
fn warp32_sum<const K: usize>(n: usize, f: impl Fn(usize) -> [f32; K]) -> [f32; K] {
    let mut lane = [[0.0f32; K]; 32];
    for (t, acc) in lane.iter_mut().enumerate() {
        let mut i = t;
        while i < n {
            let v = f(i);
            for k in 0..K {
                acc[k] += v[k];
            }
            i += 32;
        }
    }
    let mut s = 16;
    while s > 0 {
        for j in 0..s {
            for k in 0..K {
                let other = lane[j + s][k];
                lane[j][k] += other;
            }
        }
        s >>= 1;
    }
    lane[0]
}

/// Draw an index from non-negative weights, over the entries `live` accepts
/// and no others. A row whose live weights have all underflowed is drawn
/// uniformly over them rather than dropped; a row with no live entry at all
/// gives nothing back.
fn pick_live(w: &[f32], live: impl Fn(usize) -> bool, rng: &mut Rng) -> Option<usize> {
    // Weight and count in one walk, which is also how the card takes them.
    let [total, count] =
        warp32_sum(w.len(), |i| if live(i) { [w[i].max(0.0), 1.0] } else { [0.0, 0.0] });
    let n = count as usize;
    if n == 0 {
        return None;
    }
    let mut last = None;
    if !(total > 0.0) {
        let mut k = rng.below(n);
        for i in 0..w.len() {
            if live(i) {
                if k == 0 {
                    return Some(i);
                }
                k -= 1;
            }
        }
        unreachable!("a live entry was counted");
    }
    let mut needle = rng.unit_f64() * total as f64;
    for (i, &weight) in w.iter().enumerate() {
        if !live(i) {
            continue;
        }
        last = Some(i);
        needle -= weight.max(0.0) as f64;
        if needle < 0.0 {
            return Some(i);
        }
    }
    last
}

/// Draw an index from non-negative weights without allocating. A row whose
/// weights have all underflowed is drawn uniformly rather than dropped, and an
/// empty row costs no draw at all.
fn pick(w: &[f32], rng: &mut Rng) -> usize {
    let [total] = warp32_sum(w.len(), |i| [w[i].max(0.0)]);
    if !(total > 0.0) {
        return if w.is_empty() { 0 } else { rng.below(w.len()) };
    }
    let mut needle = rng.unit_f64() * total as f64;
    for (i, &weight) in w.iter().enumerate() {
        needle -= weight.max(0.0) as f64;
        if needle < 0.0 {
            return i;
        }
    }
    w.len() - 1
}

impl TNode {
    /// Host bytes this node's own lists hold, beside the struct itself.
    ///
    /// The config supports are shared between a node and its children, so they
    /// are not counted here: charging every node for an `Arc` most of its
    /// siblings hold too would make a tree look several times its size.
    fn bytes(&self) -> usize {
        let u = |v: &Vec<u32>| v.capacity() * 4;
        self.draw.bytes()
            + self.acts.capacity() * std::mem::size_of::<Action>()
            + self.aslot.capacity()
            + self.fdown.capacity()
            + self.obs_child.capacity() * 8
            + u(&self.obs_start)
            + u(&self.obs_act)
            + self.child.capacity() * 8
            + u(&self.legal_off)
            + u(&self.legal_action)
            + u(&self.legal_child)
            + u(&self.legal_trans)
            + u(&self.action_off)
            + u(&self.action_cell)
            + u(&self.cell_row)
    }

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
const N_ROLES: usize = 9;
const R_PB: usize = 0;
const R_XPUB: usize = 1;
const R_XB: usize = 2;
const R_H: usize = 3;
const R_CPHI: usize = 4;
const R_CF: usize = 5;
const R_CG: usize = 6;
const R_JP: usize = 7;
const R_CP: usize = 8;

thread_local! {
    static BUFS: std::cell::RefCell<[Vec<Vec<f32>>; N_ROLES]> = const {
        std::cell::RefCell::new([
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
            Vec::new(),
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

/// Nodes one expansion may add before it is abandoned.
///
/// An expansion adds a coin play and everything beneath it up to the next coin
/// plays, and that walk recurses through draws, tactics and forced plays. It
/// usually costs about seventeen nodes, but one that branches through several
/// tactics at once has been measured building two hundred thousand. Abandoning
/// that single expansion through `rewind` leaves the rest of the solve intact,
/// which a ceiling on the whole tree could not do.
const EXPANSION_CAP: usize = 4096;

/// Nodes a healthy solve of `s` expansions builds, with room to spare: what a
/// retired node array may hold and still be worth pooling. Larger arrays came
/// from a pathological tree and holding one per thread costs more memory than
/// every reallocation it saves.
fn pool_budget(cfg: &Cfg) -> usize {
    32 * cfg.s as usize
}

fn take_nodes() -> Vec<TNode> {
    NODE_POOL.with(|b| b.borrow_mut().pop().unwrap_or_default())
}

fn give_nodes(mut v: Vec<TNode>, budget: usize) {
    if v.capacity() == 0 || v.capacity() > budget {
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

/// One public encoding, hashed. A word at a time, because the row is nine
/// hundred floats and a byte-at-a-time hash of it would cost more than the
/// duplicate trunk row it saves.
fn row_key(row: &[f32]) -> u64 {
    let mut h = 0u64;
    for &x in row {
        h = (x.to_bits() as u64 ^ h).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    h
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

/// Where a solve stands between two of its rounds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing in flight yet.
    Fresh,
    /// The round in flight runs iterations, and on the device path an
    /// expansion phase after them.
    Iterating,
    /// The round in flight is the last one.
    Reading,
    Done,
}

/// What one turn of a solve produced.
pub enum Step {
    /// Network work this solve is waiting on. Call `advance` again with one
    /// reply per call, in order.
    Calls(Vec<Call>),
    /// The solve is over. `Some` when it was asked to collect a row.
    Done(Option<Solved>),
}

/// Everything the CFR loop works in, when the loop runs on this host.
///
/// The device path allocates none of it. The card holds its own regret, visit,
/// Q, prior, strategy-sum, reach and value arenas, advances all of them itself,
/// and `farm::Dst` has no variant that could carry one back — so a solve driven
/// on a card that grew these here would fill tens of megabytes of host memory
/// with zeroes and read them never. `Solver::host` is `None` there, and every
/// reader below goes through `Solver::cfr`, which says so rather than handing
/// out a uniform start that looks like an answer.
const HOST_PATH: &str = "the CFR arenas belong to the host path";

#[derive(Default)]
pub struct HostCfr {
    /// Accumulated regret, laid out exactly like `Solver::cur`.
    pub regret: Vec<f32>,
    /// The expansion phase's own statistics, in the same layout.
    ///
    /// `prior` is the policy head's `softmax(logit(c, a) / prior_temp)` over a
    /// config's legal row, filled once when the node is expanded. `visits` are
    /// PUCT's counts, accumulated over every expansion phase of the search and
    /// incremented as a trajectory passes — which is also the virtual loss,
    /// since later simulations of the same phase then see the earlier ones.
    /// `qval` is the action value the last backprop formed, before it was
    /// turned into a regret. The device keeps no such array -- it holds a
    /// value arena per traverser, so `k_expand` re-forms Q out of `cell_val`
    /// where it selects. Here there is one arena and each traverser's pass
    /// overwrites it, so the number has to be kept as it is made.
    pub prior: Vec<f32>,
    pub visits: Vec<f32>,
    pub qval: Vec<f32>,
    /// The reach-weighted running strategy sum, per node. Per node rather than
    /// flat, because a node is given its cells when it is expanded and a
    /// ragged vector grows there without disturbing anything already summed.
    pub sum_strat: Vec<Vec<f32>>,
    /// Reach per config, flat: node `i`'s two players occupy
    /// `reach[roff[i] .. roff[i] + nc0 + nc1]`, player 0 first. One arena
    /// rather than `Vec<Vec<f32>>` — the CFR passes touch every node, and two
    /// pointer hops per node is what they were spending their time on.
    pub reach: Vec<f32>,
    /// The traverser's counterfactual value per config, flat the same way:
    /// `vals[voff[i] .. voff[i] + max(nc0, nc1)]`.
    pub vals: Vec<f32>,
    /// The network's value per config, per traverser, before the opponent's
    /// reach mass scales it — laid out like `vals`, one arena a seat. This is
    /// the only part of a leaf value that costs a network query, so it is the
    /// part `Cfg::refresh` keeps between iterations.
    pub vcache: [Vec<f32>; 2],
}

pub struct Solver {
    /// Owned because a solver retains its context for its full solve.
    pub(crate) ctx: Ctx,
    /// Owned, so a solve can be moved between threads between two rounds.
    nets: Arc<Nets>,
    /// The solve's own random stream: the world an expansion samples, the
    /// leaves a harvest picks. Owned for the same reason.
    rng: Rng,
    /// Which of the card's solve slots this one holds. The card keeps a
    /// solve's arenas between its rounds, so every call it raises names the
    /// same slot. Zero, and meaningless, when the backend keeps no state.
    slot: usize,
    /// How many query roots to nominate at the end, or nothing when this solve
    /// is acted on and thrown away. An uncollected solve skips the value pass
    /// under the reference strategy, which is most of a CFR iteration.
    collect: Option<usize>,
    /// Iterations run. `advance` picks up from here every time it is called.
    at: usize,
    /// Which round, if any, is in flight.
    phase: Phase,
    /// The leaves the last round was asked for the reach at, in the order the
    /// reply concatenates them.
    picks: Vec<usize>,
    pub(crate) cfg: Cfg,
    pub nodes: Vec<TNode>,
    /// Node states, parallel to `nodes`. A leaf can only be expanded later if
    /// the solver still knows the position it stands for, so a growing tree
    /// keeps every one of them.
    pub states: Vec<State>,
    /// Whoever listed each node as a child, `NO_ROW` at the root. Exhaustion
    /// travels up it, and the device's flat description takes its own parent
    /// array straight from here.
    pub parent: Vec<u32>,
    /// Nodes whose `exhausted` flag has changed since the card last saw it.
    /// Every other node property is append-only under growth; this one is not,
    /// because sealing a leaf can seal a whole chain of its ancestors.
    resealed: Vec<u32>,
    pub root_belief: [Belief; 2],
    /// The current regret-matching iterate, flat by node over legal cells.
    /// Node `i` occupies `soff[i] ..` for as many cells as its `legal_action`
    /// holds; within that range its config rows are described by
    /// `TNode::legal_off`. `soff` is *not* sorted: a node is born a leaf with
    /// no cells and is given its region when it is expanded, which is what lets
    /// the arena grow by appending while everything already accumulated stays
    /// where it is.
    pub cur: Vec<f32>,
    pub(crate) soff: Vec<u32>,
    /// The reference strategy: one flat normalised copy of the running
    /// strategy sum, laid out exactly like `cur`. The host materialises the
    /// whole of it in `finish`; the device path reads the root's row and
    /// nothing else, so `read_back` sizes it to that row alone.
    pub avg: Vec<f32>,
    /// The CFR arenas, when this solve runs its own CFR loop. `None` on the
    /// device path: the card owns equivalents of every one of them and
    /// advances them itself, and not one ever crosses back.
    host: Option<HostCfr>,
    /// Leaves that have become decision or chance nodes since a reader last
    /// looked. A flat description of the tree is append-only apart from these,
    /// so they are what an incremental update needs to be told.
    pub grown: Vec<u32>,
    /// Whether each player's running sum has been normalised at least once.
    /// Until then the historical average is the literal initial iterate, not
    /// a multiply-then-divide reconstruction of it.
    pub(crate) avg_touched: [bool; 2],
    /// Total legal strategy cells across decision nodes: the length of `cur`,
    /// `regret` and `avg`, and where the next expanded node's region starts.
    pub ncells: usize,
    /// Reach and value cells the tree has: the lengths `HostCfr::reach` and
    /// `HostCfr::vals` are held at, and the sizes the card fits its own two
    /// arenas to. Counted rather than read off a `Vec`, because on the device
    /// path there is no `Vec` to read them off.
    pub nreach: usize,
    pub nvals: usize,
    /// Draw-transition entries over the whole tree, which the budget bounds and
    /// the device's `draw_to` / `draw_p` / `rvd_src` / `rvd_p` are sized by.
    pub ndraws: usize,
    /// Whether any expansion of this solve was abandoned for want of budget
    /// rather than for running away.
    ///
    /// The run counts these. A budget is a percentile of a measured shape, and
    /// the count is the only thing that can argue with the percentile chosen:
    /// it says how often the tail is being truncated, and so whether a slot is
    /// too small for the game or four times larger than it needs to be.
    budget_hit: bool,
    pub(crate) roff: Vec<u32>,
    pub(crate) voff: Vec<u32>,
    /// `[node]` -> its row in the network batch, or `u32::MAX` for a node that
    /// carries none. The policy head needs a node's own board vector, which
    /// lives in `pb` at that row.
    pub(crate) row_of: Vec<u32>,
    /// Whether a node's `prior` has been filled. A node is expanded before the
    /// batch that holds its board vector has necessarily run, so the prior is
    /// computed at the next expansion phase rather than inside `grow`.
    pub(crate) primed: Vec<bool>,
    /// Nodes that still want a policy prior: decision nodes whose network row
    /// the batch has not reached yet, plus whatever `grow` has just made.
    ///
    /// This used to be a scan of every node an iteration, looking for the
    /// handful just grown. A solve holds eight thousand nodes and runs
    /// sixty-four iterations, so that is half a million filter tests over four
    /// scattered arrays -- and `refresh_priors` measured at two thirds of all
    /// host work a solve did. Growth knows exactly which nodes it made.
    wants_prior: Vec<u32>,
    /// `[node]` -> config counts per player, so the hot loops never chase the
    /// `Rc` to ask how long a support is.
    pub nc: Vec<[u32; 2]>,
    pub(crate) steps: [usize; 2],

    // ---------------------------------------------------------- leaf batch
    // Built once per solve. Everything here is a property of the leaf's public
    // state or its config support, so it survives every CFR iteration; only
    // the belief block `xb` and the join output `h` are rewritten
    // per iteration. That split is the whole architecture: the trunk runs
    // ~2,000 times a solve and the join ~158,000.
    /// Non-terminal leaves in node order — the rows of the network batch.
    pub leaf_rows: Vec<usize>,
    /// Per traverser: how many leaf rows `HostCfr::vcache` holds a network
    /// value for. Rows past it are the ones growth has added since the last
    /// query, and they are queried whatever `Cfg::refresh` says.
    cached: [usize; 2],
    /// Diagnostics: the per-iteration work this solve has done so far. Nothing
    /// in a run reads it; the budget study prices a search with it.
    pub trace: Trace,
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
    batch_boards: usize,
    batch_cfgs: usize,
    /// `[ncfg, D]` readout rows `f(c)` and `[ncfg, POOL]` pooling vectors
    /// `g(c)`. Both survive every CFR iteration.
    pub cf: Vec<f32>,
    pub cg: Vec<f32>,
    /// `[ncfg, D]` policy readout rows `f_p(c)`, beside the value's `f(c)`.
    pub cp: Vec<f32>,
    /// `[2, NTYPE, TYPE]`: the printed-card tokens, one table per player view.
    /// The draft is fixed for the solve, so this is built once.
    pub cards: Vec<f32>,
    /// `[boards, D]` board vectors, and their `[boards, JW]` projection into
    /// the join's first layer. Neither moves between CFR iterations.
    pub pb: Vec<f32>,
    pub jp: Vec<f32>,
    /// `[row]` -> the board vector the row reads.
    ///
    /// The trunk reads the public state and nothing else, and a tree that
    /// spans one round is full of transpositions: coin plays commute, so two
    /// orders of the same two plays reach the same public state. A sixth to a
    /// quarter of a solve's rows are duplicates of an earlier one, measured.
    /// So the public encoding is interned and the trunk runs once per distinct
    /// public state, which is one less row to encode, to marshal, to send and
    /// to multiply.
    ///
    /// The belief index stays per row. Two rows that share a board sit at
    /// different places in the tree, so their reaches and their supports are
    /// their own.
    pub(crate) board_of: Vec<u32>,
    /// Hash of a public encoding -> the board that holds it. A hit is checked
    /// against the row itself, so a collision costs a duplicate board rather
    /// than a wrong answer.
    bmap: std::collections::HashMap<u64, u32, KeyHash>,
    /// Distinct public encodings `xpub` holds. Pooled buffers keep their
    /// length across solves, so this cannot be read off `xpub.len()`.
    pub nboards: usize,
    /// Expanded public encoding, one row a distinct public state.
    pub(crate) xpub: Vec<f32>,
    /// The mirrored view of the first leaf, which is all the card table wants.
    mirror0: Vec<f32>,
    /// `[2 * rows, POOL]` pooled belief embeddings — the one thing the join
    /// reads that moves between CFR iterations.
    pub xb: Vec<f32>,
    /// `[rows, D]`: the join output for the last traverser queried.
    pub h: Vec<f32>,
    /// Normalised belief weights for one leaf's support.
    wbuf: Vec<f32>,
    /// The expansion in flight has passed its bound and is unwinding.
    abandon: bool,
    /// Node count at which the expansion in flight gives up. The bound is on
    /// one expansion, not on the tree.
    limit: usize,
    /// The budget applies. False while `new` grows the root: a root that
    /// stayed a leaf has no strategy, so that one expansion is `EXPANSION_CAP`
    /// and nothing else.
    bounded: bool,
    /// Working memory for the chance transitions, reused across the tree.
    draw_scratch: DrawScratch,
    /// `(config key, legal cell)` scratch for ordering a public child's
    /// support. Keeping the index separate removes the old 24-bit width
    /// assertion; every valid wide job remains representable.
    cell_order: Vec<(u64, u32)>,

    // ------------------------------------------------------- the device path
    /// The flat description the device reads. Shared rather than copied: the
    /// call that carries it is dropped before this solver runs again, so
    /// `make_mut` finds itself alone and extends in place.
    contract: Arc<crate::contract::Contract>,
    /// The first node whose row the card has yet to be told about, and how many
    /// strategy cells it has been given. Growth only ever appends past these.
    sent_from: usize,
    /// Rows of already-described nodes that this growth rewrote: the leaves it
    /// turned into decision nodes.
    rewrite: Vec<u32>,
    /// Rows whose `exhausted` flag moved in the growth being described, so the
    /// card is told about the ones it already holds.
    resent: Vec<u32>,
    sent_cells: usize,
    /// The expansion's random stream, which lives on the card once seeded.
    seed: u64,
    /// How much of each of the card's arrays it has already been told about.
    sent: crate::contract::Sent,
}

/// A solve moves between the worker that grows it and the card that evaluates
/// it, so it must be `Send`. The farm's queues need it anyway; this says so
/// where the type is, and fails here rather than three files away.
const _: () = {
    const fn is_send<T: Send>() {}
    is_send::<Solver>();
};

impl Drop for Solver {
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
            (R_CP, &mut self.cp),
        ] {
            give_buf(role, std::mem::take(v));
        }
        give_nodes(std::mem::take(&mut self.nodes), pool_budget(&self.cfg));
    }
}

impl Solver {
    /// A solve of `root`, ready for its first `advance`.
    ///
    /// The tree is built here so that a root which is itself a coin play has a
    /// strategy to read from the first moment.
    pub fn new(
        root: &State,
        ctx: Ctx,
        nets: Arc<Nets>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
    ) -> Solver {
        let cfgs: [Arc<[Config]>; 2] = [
            belief[0].cfg.as_slice().into(),
            belief[1].cfg.as_slice().into(),
        ];
        let device = nets.device;
        let mut sv = Solver {
            ctx,
            nets,
            rng,
            slot: 0,
            collect: None,
            at: 0,
            phase: Phase::Fresh,
            picks: Vec::new(),
            cfg,
            nodes: take_nodes(),
            states: Vec::new(),
            parent: Vec::new(),
            resealed: Vec::new(),
            root_belief: belief,
            cur: Vec::new(),
            row_of: Vec::new(),
            primed: Vec::new(),
            wants_prior: Vec::new(),
            soff: Vec::new(),
            avg: Vec::new(),
            host: (!device).then(HostCfr::default),
            grown: Vec::new(),
            avg_touched: [false; 2],
            ncells: 0,
            nreach: 0,
            nvals: 0,
            ndraws: 0,
            budget_hit: false,
            roff: Vec::new(),
            voff: Vec::new(),
            nc: Vec::new(),
            steps: [0, 0],
            leaf_rows: Vec::new(),
            cached: [0; 2],
            trace: Trace::default(),
            term_leaves: Vec::new(),
            leaf_cidx: Vec::new(),
            leaf_coff: Vec::new(),
            cphi: take_buf(R_CPHI),
            cplayer: Vec::new(),
            cmap: std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash),
            board_of: Vec::new(),
            bmap: std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash),
            nboards: 0,
            ncfg: 0,
            batch_rows: 0,
            batch_boards: 0,
            batch_cfgs: 0,
            cf: take_buf(R_CF),
            cg: take_buf(R_CG),
            cp: take_buf(R_CP),
            cards: Vec::new(),
            pb: take_buf(R_PB),
            jp: take_buf(R_JP),
            xpub: take_buf(R_XPUB),
            mirror0: Vec::new(),
            xb: take_buf(R_XB),
            h: take_buf(R_H),
            wbuf: Vec::new(),
            abandon: false,
            limit: usize::MAX,
            bounded: false,
            draw_scratch: DrawScratch::default(),
            cell_order: Vec::new(),
            contract: Arc::new(crate::contract::Contract::default()),
            sent_from: 0,
            rewrite: Vec::new(),
            resent: Vec::new(),
            sent_cells: 0,
            seed: 0,
            sent: Default::default(),
        };
        sv.cf.clear();
        sv.cg.clear();
        sv.cp.clear();
        sv.pb.clear();
        sv.jp.clear();
        {
            let _t = timed!(BUILD);
            // The root is born a leaf like every other node; `solve` grows it.
            // Its own expansion is unbounded, because a root that stayed a leaf
            // would leave the solve with no strategy to act on at all. The
            // bound below is on how much of the *expansion budget* one leaf may
            // take, and the root spends none of it.
            sv.nodes.reserve(640);
            sv.cur.reserve(640);
            if let Some(h) = &mut sv.host {
                h.reach.reserve(640);
                h.vals.reserve(640);
                h.regret.reserve(640);
            }
            // The expansion's stream lives on the card once it is seeded, so
            // it is drawn here rather than by the round that sends it.
            sv.seed = Rng::new(sv.rng.next_u64()).0;
            let root = sv.push_node(crate::contract::NO_ROW, root.clone(), cfgs);
            // A root that is a coin play would otherwise stay a leaf with no
            // strategy to read, so the first expansion is unconditional.
            // The budget does not apply yet: it would rewind this grow and
            // leave a leaf. `EXPANSION_CAP` still bounds the walk.
            sv.limit = sv.nodes.len() + EXPANSION_CAP;
            sv.grow(root);
            sv.limit = usize::MAX;
            sv.seal(root, 1);
            // The first CFR update and every expansion trajectory require
            // reaches for the tree that now exists. The card seeds and sweeps
            // its own, from the root beliefs the first tree call carries, so
            // on that path this pass would be redone before it was ever read.
            if sv.host.is_some() {
                sv.precompute_reaches();
            }
        }
        sv.bounded = true;
        sv
    }

    /// The CFR arenas this solve works in.
    ///
    /// A solve on the device path has none. Nothing there reads one — the card
    /// runs the loop — so reaching for them is a mistake about which backend is
    /// driving, and it says so here rather than returning the zeroes an
    /// unallocated arena would.
    pub fn cfr(&self) -> &HostCfr {
        self.host.as_ref().expect(HOST_PATH)
    }

    /// The same arenas, to write. Only the oracles want this: they run a
    /// contract's arithmetic beside the solver's and put the result back.
    #[doc(hidden)]
    pub fn cfr_mut(&mut self) -> &mut HostCfr {
        self.host.as_mut().expect(HOST_PATH)
    }

    /// Pin this solve to one of a card's solve slots.
    ///
    /// The card keeps a solve's board vectors, config rows and CFR arenas
    /// between its rounds, so every call the solve raises must name the same
    /// slot -- and the slot must not be handed to another solve until this one
    /// is done with it.
    pub fn pin(&mut self, slot: usize) {
        self.slot = slot;
    }

    /// Ask this solve for a training row: the root's values, its policy, and
    /// `queries` of the leaves it asked the network about, as roots for later
    /// solves. Without it the solve is acted on and thrown away.
    pub fn collect(&mut self, queries: usize) {
        self.collect = Some(queries);
    }

    /// Run `f` with the solve's own random stream.
    ///
    /// The stream is a field, so it has to come out for the duration or
    /// nothing else about the solver can be borrowed while it is in use.
    fn with_rng<T>(&mut self, f: impl FnOnce(&mut Self, &mut Rng) -> T) -> T {
        let mut rng = std::mem::replace(&mut self.rng, Rng(1));
        let out = f(self, &mut rng);
        self.rng = rng;
        out
    }

    /// Run `f` with the expansion's own stream, which is the stream the card
    /// runs when the CFR loop is there. Both backends draw a trajectory from
    /// the same state of the same generator, so both take the same turns.
    fn with_expand_rng<T>(&mut self, f: impl FnOnce(&mut Self, &mut Rng) -> T) -> T {
        let mut rng = Rng(self.seed);
        let out = f(self, &mut rng);
        self.seed = rng.0;
        out
    }

    /// Push a node for `s`, as a leaf, and give it its slice of every arena.
    ///
    /// Every node is born a leaf. `grow` turns one into a decision node later,
    /// which is when it is given its strategy cells — so a node's reach and
    /// value rows are laid out in node order while its cells are not, and both
    /// are found through their own offset table.
    fn push_node(&mut self, parent: u32, s: State, cfgs: [Arc<[Config]>; 2]) -> usize {
        let player = s.to_act();
        let terminal = s.is_terminal();
        let id = self.nodes.len();
        let _tp = timed!(BPUSH);
        self.nodes.push(TNode {
            util: if terminal { s.utility(player as usize) } else { 0.0 },
            player,
            leaf: true,
            expandable: !terminal,
            // Set for the whole fresh subtree by `seal` once the expansion
            // that made it has either finished or been abandoned.
            exhausted: false,
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
        self.parent.push(parent);
        let (c0, c1) = (cfgs[0].len(), cfgs[1].len());
        self.nc.push([c0 as u32, c1 as u32]);
        // No cells yet: a leaf has no strategy. `grow` appends its region.
        self.soff.push(self.ncells as u32);
        self.roff.push(self.nreach as u32);
        self.voff.push(self.nvals as u32);
        self.nreach += c0 + c1;
        self.nvals += c0.max(c1);
        if let Some(h) = &mut self.host {
            h.reach.resize(self.nreach, 0.0);
            h.vals.resize(self.nvals, 0.0);
            h.vcache[0].resize(self.nvals, 0.0);
            h.vcache[1].resize(self.nvals, 0.0);
            h.sum_strat.push(Vec::new());
        }
        self.primed.push(false);
        self.row_of.push(u32::MAX);
        // Only a coin play carries a network row. Everything between two coin
        // plays is grown through immediately and never stays a leaf.
        if terminal {
            self.term_leaves.push(id);
        } else if matches!(s.pending(), Cont::MainPlay) {
            self.row_of[id] = (self.leaf_coff.len() / 2) as u32;
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
    fn push_child(&mut self, parent: usize, s: State, cfgs: [Arc<[Config]>; 2]) -> usize {
        // The single funnel every node comes through, and so the only place the
        // budget can bite exactly. The head of `grow` is not enough: a decision
        // node pushes its whole fan-out of coin-play children here, and none of
        // them recurses, so up to forty rows land between two of those checks.
        //
        // The parent is handed back rather than a new node. Nothing reads it:
        // both callers test `abandon` on the next line and rewind.
        if self.runaway() {
            return parent;
        }
        let stop = s.is_terminal() || matches!(s.pending(), Cont::MainPlay);
        let ch = self.push_node(parent as u32, s, cfgs);
        // `runaway` is the start-of-step check; one row of cidx or one node's
        // reach can land past the cap between two of those. The caller rewinds.
        if self.runaway() {
            return ch;
        }
        if !stop {
            self.grow(ch);
        }
        ch
    }

    /// One expansion of leaf `id`, abandoned whole if it runs away.
    ///
    /// The bound has to bite *inside* the grow-through recursion, because one
    /// expansion grows through every draw, tactic and forced play beneath the
    /// coin play it starts from and that walk branches. An abandoned expansion
    /// leaves `id` a leaf that growth will not try again; the solve carries on
    /// with the tree it already had.
    fn expand(&mut self, id: usize) {
        debug_assert!(
            self.nodes[id].leaf && self.nodes[id].expandable,
            "growth turns an expandable leaf into a decision node, and nothing else"
        );
        let fresh = self.nodes.len();
        self.limit = fresh + EXPANSION_CAP;
        self.grow(id);
        if self.abandon {
            self.abandon = false;
            self.nodes[id].expandable = false;
        }
        self.seal(id, fresh);
    }

    /// Settle `exhausted` after an expansion of `id` that appended the nodes
    /// from `fresh` on.
    ///
    /// A leaf is exhausted when growth may not turn it into a decision node --
    /// past the round boundary, at a terminal, or where its own expansion ran
    /// away. An interior node is exhausted when every child is. The fresh
    /// subtree is settled deepest first, which one reverse pass gives because
    /// a child is always built after its parent; the flag then travels up from
    /// `id` as far as it keeps turning on. It never turns off, so the walk
    /// stops at the first ancestor that still has somewhere to grow.
    fn seal(&mut self, id: usize, fresh: usize) {
        for i in (fresh..self.nodes.len()).rev() {
            self.set_exhausted(i);
        }
        let mut at = id;
        loop {
            if !self.set_exhausted(at) {
                return;
            }
            match self.parent[at] {
                crate::contract::NO_ROW => return,
                p => at = p as usize,
            }
        }
    }

    /// Recompute one node's flag, and say whether it now holds.
    fn set_exhausted(&mut self, i: usize) -> bool {
        let n = &self.nodes[i];
        let e = if n.leaf {
            !n.expandable
        } else {
            n.child.iter().all(|&c| self.nodes[c].exhausted)
        };
        if e != self.nodes[i].exhausted {
            self.nodes[i].exhausted = e;
            self.resealed.push(i as u32);
        }
        e
    }

    /// The nodes whose `exhausted` flag has moved since the last call.
    pub(crate) fn take_resealed(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.resealed)
    }

    /// Whether the expansion in flight has spent its bound.
    fn runaway(&mut self) -> bool {
        if self.bounded && self.spent() {
            self.abandon = true;
            self.budget_hit = true;
        } else if self.nodes.len() >= self.limit {
            self.abandon = true;
        }
        self.abandon
    }

    /// Whether this solve ever ran out of budget. See `Solver::budget_hit`.
    pub fn budget_hit(&self) -> bool {
        self.budget_hit
    }

    /// Whether the solve has reached its budget in any term.
    ///
    /// Asked at the head of every `grow`, which is the same place
    /// `EXPANSION_CAP` is asked and for the same reason: the recursion is what
    /// grows the tree, so a check there bounds what one step can add to one
    /// node's worth, and `rewind` then puts even that back. `alloc_cells` is
    /// checked separately because a node's cells are appended after its whole
    /// subtree, with no `grow` between.
    fn spent(&self) -> bool {
        let b = &self.cfg.budget;
        self.nodes.len() >= b.nodes
            || self.leaf_rows.len() >= b.rows
            || self.nboards >= b.boards
            || self.ncfg >= b.configs
            || self.leaf_cidx.len() >= b.cidx
            || self.nreach >= b.reach
            || self.ncells >= b.cells
            || self.ndraws >= b.draws
    }

    /// Where every append-only arena stands.
    fn mark(&self) -> Mark {
        Mark {
            nodes: self.nodes.len(),
            ncells: self.ncells,
            ncfg: self.ncfg,
            ndraws: self.ndraws,
            leaf_rows: self.leaf_rows.len(),
            nboards: self.nboards,
            term_leaves: self.term_leaves.len(),
            leaf_coff: self.leaf_coff.len(),
            leaf_cidx: self.leaf_cidx.len(),
        }
    }

    /// Undo everything an abandoned `grow` appended, and make `id` a leaf
    /// again.
    ///
    /// Growth is all-or-nothing so that a tree is a tree at every moment a
    /// caller can see one: every non-terminal leaf stands for a coin play,
    /// which is the only place the value network is defined. Stopping halfway
    /// would leave a leaf in the middle of a tactic, whose value nothing
    /// defines and which reads as zero. An inner abandon unwinds through every
    /// enclosing `grow`, so the whole expansion is undone up to the coin play
    /// it started from.
    fn rewind(&mut self, id: usize, m: Mark) {
        self.nreach = self.roff[m.nodes] as usize;
        self.nvals = self.voff[m.nodes] as usize;
        self.nodes.truncate(m.nodes);
        self.states.truncate(m.nodes);
        self.parent.truncate(m.nodes);
        self.resealed.retain(|&i| (i as usize) < m.nodes);
        self.nc.truncate(m.nodes);
        self.soff.truncate(m.nodes);
        self.roff.truncate(m.nodes);
        self.voff.truncate(m.nodes);
        self.primed.truncate(m.nodes);
        self.wants_prior.retain(|&i| (i as usize) < m.nodes);
        self.row_of.truncate(m.nodes);
        self.leaf_rows.truncate(m.leaf_rows);
        self.cached = self.cached.map(|k| k.min(m.leaf_rows));
        // `xpub` is written by index and reused, so only the count and the
        // interning map name boards that no longer exist. A board an abandoned
        // row shared with a surviving one stays, and stays right.
        self.board_of.truncate(m.leaf_rows);
        self.nboards = m.nboards;
        self.bmap.retain(|_, &mut b| (b as usize) < m.nboards);
        self.term_leaves.truncate(m.term_leaves);
        self.leaf_coff.truncate(m.leaf_coff);
        self.leaf_cidx.truncate(m.leaf_cidx);
        self.cur.truncate(m.ncells);
        self.ncells = m.ncells;
        self.ndraws = m.ndraws;
        if let Some(h) = &mut self.host {
            h.reach.truncate(self.nreach);
            h.vals.truncate(self.nvals);
            h.vcache[0].truncate(self.nvals);
            h.vcache[1].truncate(self.nvals);
            h.sum_strat.truncate(m.nodes);
            h.regret.truncate(m.ncells);
            h.prior.truncate(m.ncells);
            h.visits.truncate(m.ncells);
            h.qval.truncate(m.ncells);
        }
        // `cphi` is written by index and reused, so only the count and the
        // interning map name configs that no longer exist.
        self.cplayer.truncate(m.ncfg);
        self.cmap.retain(|_, &mut i| (i as usize) < m.ncfg);
        self.ncfg = m.ncfg;
        self.grown.retain(|&g| (g as usize) < m.nodes && g != id as u32);
        let n = &mut self.nodes[id];
        n.leaf = true;
        n.chance = false;
        n.draw = DrawMap::default();
        n.draw_steps = 0;
        n.acts.clear();
        n.aslot.clear();
        n.fdown.clear();
        n.obs_child.clear();
        n.obs_start.clear();
        n.obs_act.clear();
        n.child.clear();
        n.legal_off.clear();
        n.legal_action.clear();
        n.legal_child.clear();
        n.legal_trans.clear();
        n.action_off.clear();
        n.action_cell.clear();
        n.cell_row.clear();
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
        let ncells = self.ncells;
        if let Some(h) = &mut self.host {
            h.regret.resize(ncells, 0.0);
            h.visits.resize(ncells, 0.0);
            h.qval.resize(ncells, 0.0);
            // Until the policy head has spoken the prior is the uniform
            // strategy, which is what CFR starts from too.
            h.prior.extend_from_slice(&u);
            h.sum_strat[id] = vec![0.0; cells];
        }
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
        let sum_strat = &self.host.as_ref().expect(HOST_PATH).sum_strat;
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance || !self.avg_touched[n.player as usize] {
                continue;
            }
            let so = self.soff[i] as usize;
            let nc = n.nc(n.player as usize);
            for c in 0..nc {
                let row = n.legal_row(c);
                let sum: f32 = sum_strat[i][row.clone()].iter().sum();
                let k = row.len().max(1) as f32;
                for cell in row {
                    self.avg[so + cell] = if sum > 0.0 {
                        sum_strat[i][cell] / sum
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
        // It is a decision node from here, so it wants a policy prior as soon
        // as the batch reaches its row.
        if self.row_of[id] != u32::MAX {
            self.wants_prior.push(id as u32);
        }
        if self.runaway() {
            return;
        }
        let mark = self.mark();
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
            let ch = self.push_child(id, cs, cc);
            if self.abandon {
                self.rewind(id, mark);
                return;
            }
            // The round boundary. Once `Cfg::rounds` of them are behind the
            // tree, everything a further one puts into it stays a leaf, priced
            // by the value network. A Warrior Priest draw is mid-round and
            // bounds nothing. `TODO.md` records why the limit is there.
            if !wp && cs.round > self.states[0].round + self.cfg.rounds as u16 {
                for n in &mut self.nodes[mark.nodes..] {
                    n.expandable = false;
                }
            }
            let n = &mut self.nodes[id];
            n.chance = true;
            n.child = vec![ch];
            if self.bounded && self.ndraws + draw.len() > self.cfg.budget.draws {
                self.abandon = true;
                self.budget_hit = true;
                self.rewind(id, mark);
                return;
            }
            self.ndraws += draw.len();
            let n = &mut self.nodes[id];
            n.draw = draw;
            n.draw_steps = steps;
            self.grown.push(id as u32);
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
            child.push(self.push_child(id, cs, cc));
            if self.abandon {
                self.rewind(id, mark);
                return;
            }
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
        // The one term a `grow` at the head of the recursion cannot bound: a
        // node's cells are appended after its whole subtree has been built, so
        // several nodes' worth can land with no check between them.
        if self.bounded && self.ncells + self.nodes[id].legal_action.len() > self.cfg.budget.cells {
            self.abandon = true;
            self.budget_hit = true;
            self.rewind(id, mark);
            return;
        }
        self.alloc_cells(id);
        self.grown.push(id as u32);
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
        let reach = &mut self.host.as_mut().expect(HOST_PATH).reach;
        reach.fill(0.0);
        for p in 0..2 {
            let at = self.roff[0] as usize + if p == 1 { self.nc[0][0] as usize } else { 0 };
            let n = self.nc[0][p] as usize;
            reach[at..at + n].copy_from_slice(&self.root_belief[p].p);
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
                let (lo, hi) = reach.split_at_mut(cbase);
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
                let (lo, hi) = reach.split_at_mut(cbase);
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
        &self.cfr().reach[at..at + self.nc[i][p] as usize]
    }

    /// One leaf's public encoding, interned: the board it reads.
    ///
    /// A row, not two. The board is public and the trunk reads the physical
    /// view only -- the mirrored one was written for every leaf, carried
    /// through the call, and gathered past by everything that read it. What
    /// still wants it is the card table, which holds a view a seat and is
    /// built once a solve off the first board; that one mirror is kept here.
    ///
    /// The encoding is written where the next board would go and kept only if
    /// no earlier board already holds it. Writing first is what makes the
    /// comparison free: the row has to be built to be hashed either way.
    fn encode(&mut self, s: &State) -> u32 {
        let at = self.nboards * PUBFEAT;
        if self.xpub.len() < at + PUBFEAT {
            // Grow in chunks so the zero-fill happens a handful of times per
            // solve, and not at all once the pooled buffer is warm.
            self.xpub.resize(at + 128 * PUBFEAT, 0.0);
        }
        write_public_features(s, &self.ctx, &mut self.xpub[at..at + PUBFEAT]);
        if self.nboards == 0 {
            self.mirror0.resize(PUBFEAT, 0.0);
            write_public_features(&s.mirror(), &self.ctx.mirrored(), &mut self.mirror0);
        }
        let key = row_key(&self.xpub[at..at + PUBFEAT]);
        if let Some(&b) = self.bmap.get(&key) {
            let old = b as usize * PUBFEAT;
            if self.xpub[old..old + PUBFEAT] == self.xpub[at..at + PUBFEAT] {
                return b;
            }
        }
        let b = self.nboards as u32;
        self.bmap.insert(key, b);
        self.nboards += 1;
        b
    }

    /// One row of the network batch: its public encoding, and its configs
    /// interned into the shared table.
    fn push_row(&mut self, _id: usize, s: &State, cfgs: &[Arc<[Config]>; 2]) {
        // The network is queried only at normal coin-play states: a subgame
        // finishes every tactic, trigger and forced play before a leaf.
        debug_assert!(
            matches!(s.pending(), Cont::MainPlay),
            "a network row must be a MainPlay state"
        );
        let _t = timed!(PUBFEAT);
        let board = self.encode(s);
        self.board_of.push(board);
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

    /// Drive this solve to its end on this host, answering its own calls.
    ///
    /// The farm gathers those calls across every solve in flight and answers
    /// them as one batch; a single game, a tool or a test wants exactly one
    /// solve, so it answers them where they are raised. Only the host path can
    /// do this: a device keeps the solve, and there is no device here.
    pub fn run_alone(&mut self) -> Option<Solved> {
        assert!(!self.nets.device, "a device solve cannot be run on the host");
        let mut replies: Vec<Reply> = Vec::new();
        loop {
            match self.advance(&replies) {
                Step::Calls(calls) => {
                    replies = calls.iter().map(|c| c.run(&self.nets.value)).collect();
                }
                Step::Done(solved) => return solved,
            }
        }
    }

    /// Run the calls the last growth raised on this solve's own CPU network.
    ///
    /// The farm gathers these across every solve in flight and answers them in
    /// one batch. This is the same work for a solve driven on its own, which
    /// is what the single-position tools and the tests want.
    pub fn catch_up(&mut self) {
        let calls = self.growth_calls();
        let replies: Vec<Reply> = calls.iter().map(|c| c.run(&self.nets.value)).collect();
        self.absorb(&replies);
    }

    /// The network calls the last growth made necessary: the trunk over fresh
    /// leaves, the encoder over fresh configs. Empty when nothing grew.
    ///
    /// Everything they produce is a pure function of the subgame, so growth
    /// never invalidates any of it — a leaf's board vector is the same board
    /// vector after the tree around it has changed. The trunk therefore runs
    /// once per leaf *ever created*, which is what a fixed-depth solve of the
    /// final tree would have cost; only the CFR iterations see a bigger tree.
    ///
    /// Handed back rather than run here, because they ride in the same round
    /// as the iteration that needs them — a backend runs a round's stages in
    /// order, so the resident state they write is there before the iteration
    /// reads it.
    fn growth_calls(&mut self) -> Vec<Call> {
        let rows = self.leaf_rows.len();
        if rows == self.batch_rows && self.ncfg == self.batch_cfgs {
            return Vec::new();
        }
        if self.nets.value.is_empty() {
            self.batch_rows = rows;
            self.batch_cfgs = self.ncfg;
            return Vec::new();
        }
        let _t = timed!(PUBNET);
        let mut calls = Vec::with_capacity(2);
        let fresh_rows = rows - self.batch_rows;
        if fresh_rows > 0 {
            // The cards in play are fixed at the draft, so every row of the
            // subgame carries the same card block and the table is built once,
            // one view per seat. Everything downstream reads it by canonical
            // coin-type index.
            if self.cards.is_empty() {
                let both = [&self.xpub[..PUBFEAT], &self.mirror0[..]].concat();
                self.nets.value.cards(&both, 2, &mut self.cards);
            }
            // Exactly the fresh boards. `xpub` is a grown scratch buffer, so
            // an open-ended slice would carry whatever the last, larger
            // subgame left behind — invisible to a solve evaluating alone, and
            // wrong the moment the farm concatenates this call with another.
            let at = self.batch_boards * PUBFEAT;
            let end = self.nboards * PUBFEAT;
            // The belief index of exactly the rows this call makes. A leaf's
            // support is fixed when the leaf is, so it travels with the trunk
            // and never again. `leaf_coff` holds a query's *start* and nothing
            // else — a length comes from `nc` — so the terminator a CSR range
            // needs is appended here rather than assumed.
            let q0 = 2 * self.batch_rows;
            let cs = self.leaf_coff[q0] as usize;
            let mut coff: Vec<u32> = self.leaf_coff[q0..].iter().map(|x| x - cs as u32).collect();
            coff.push((self.leaf_cidx.len() - cs) as u32);
            calls.push(Call::Trunk {
                solve: self.slot,
                at: self.batch_rows,
                rows: fresh_rows,
                board_of: self.board_of[self.batch_rows..].to_vec(),
                boards_at: self.batch_boards,
                boards: self.nboards - self.batch_boards,
                xpub: self.xpub[at..end].to_vec(),
                cards: self.cards.clone(),
                cidx: self.leaf_cidx[cs..].to_vec(),
                coff,
            });
            crate::prof::work(fresh_rows, 0, 0, 0);
        }
        let fresh_cfgs = self.ncfg - self.batch_cfgs;
        if fresh_cfgs > 0 {
            calls.push(Call::Configs {
                solve: self.slot,
                at: self.batch_cfgs,
                phi: self.cphi[self.batch_cfgs * CFEAT..self.ncfg * CFEAT].to_vec(),
                owner: self.cplayer[self.batch_cfgs..].iter().map(|&p| p as u32).collect(),
                cards: self.cards.clone(),
                n: fresh_cfgs,
            });
            crate::prof::work(0, fresh_cfgs, 0, 0);
        }
        calls
    }

    /// Take in what `growth_calls` asked for. The host keeps the board vectors
    /// because the policy head builds its action embeddings against them, and
    /// `f_p` for the same reason.
    fn absorb(&mut self, replies: &[Reply]) {
        let mut at = 0;
        if self.leaf_rows.len() > self.batch_rows {
            let r = &replies[at];
            self.pb.extend_from_slice(&r.a);
            self.jp.extend_from_slice(&r.b);
            self.batch_rows = self.leaf_rows.len();
            self.batch_boards = self.nboards;
            at += 1;
        }
        if self.ncfg > self.batch_cfgs {
            let r = &replies[at];
            self.cf.extend_from_slice(&r.a);
            self.cg.extend_from_slice(&r.b);
            self.cp.extend_from_slice(&r.c);
            self.batch_cfgs = self.ncfg;
        }
    }

    /// Rewrite the pooled belief block the join reads, per row per player.
    ///
    /// Both players, every time. This used to refresh only the player whose
    /// strategy had just moved, which was sound while CFR alternated
    /// traversers. Student of Games updates both players against one reach
    /// profile, so both blocks go stale together and the shortcut silently
    /// pooled one of them under last iteration's belief.
    ///
    /// The belief the network reads is the normalised reach, as in the
    /// reference, pooled over the same `g(c)` the readout's `f(c)` comes from,
    /// so a config is described to the network exactly one way. `g` has a
    /// linear card-weighted half, which is what makes this pooled vector carry
    /// the belief's exact expected holding of each card rather than an average
    /// of nonlinearities.
    ///
    /// Rows below `from` are ones whose join output is being reused, so their
    /// block is not read and is not written.
    fn belief_blocks(&mut self, from: usize) {
        let _t = timed!(BELFEAT);
        // Sized where it is written. Growth used to do it, which fitted a
        // megabyte of pooled belief per solve on the device path -- where the
        // card pools its own and nothing here ever reads a row of it.
        crate::net::fit(&mut self.xb, 2 * self.leaf_rows.len() * crate::net::POOL);
        let (reach, roff, nc, coff, cidx, cg, wbuf, xb) = (
            &self.host.as_ref().expect(HOST_PATH).reach,
            &self.roff,
            &self.nc,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cg,
            &mut self.wbuf,
            &mut self.xb,
        );
        let pool = crate::net::POOL;
        for (r, &i) in self.leaf_rows.iter().enumerate().skip(from) {
            for p in 0..2 {
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

    /// Fill `vals` at every leaf with the traverser's counterfactual values,
    /// querying the network at every row.
    pub fn leaf_values(&mut self, traverser: usize) {
        self.leaf_values_from(traverser, 0);
    }

    /// The same, querying the network only from row `from` on and re-scaling
    /// every earlier row's cached `v(c)` by the opponent's current reach mass.
    ///
    /// A leaf's counterfactual value is `v(c)` times that mass. The mass moves
    /// every iteration and costs a sum over a support; `v(c)` is the network,
    /// and is the whole of what the join and the readout are for. So the split
    /// here is exactly the split `Cfg::refresh` trades in.
    fn leaf_values_from(&mut self, traverser: usize, from: usize) {
        if !self.nets.value.is_empty() {
            self.belief_blocks(from);
        }
        self.pbs_head(traverser, from);
        self.readout_from(traverser, from);
        self.cached[traverser] = self.leaf_rows.len();
    }

    /// The one path CFR pays for on every iteration.
    ///
    /// Writes `h` for rows `from ..` at its front, so row `r` of the tree is
    /// row `r - from` of `h`. The join takes a contiguous batch and this one is
    /// a suffix of the leaves, because growth only ever appends.
    fn pbs_head(&mut self, traverser: usize, from: usize) {
        let net = &self.nets.value;
        if net.is_empty() {
            return;
        }
        let _t = timed!(NET);
        let rows = self.leaf_rows.len();
        let n = rows - from;
        if n == 0 {
            return;
        }
        crate::prof::work(0, 0, n, 0);
        self.trace.join_rows += n as u64;
        let pool = crate::net::POOL;
        // `xb` is grown by `fit` and never shrinks, so a subgame smaller than
        // an earlier one would otherwise hand the batch a trailing tail.
        self.nets.value.join(
            &self.pb[..self.nboards * crate::net::D],
            &self.jp[..self.nboards * crate::net::JW],
            &self.board_of[from..rows],
            &self.xb[2 * from * pool..2 * rows * pool],
            n,
            traverser,
            &mut self.h,
        );
    }

    /// Per-config leaf values for player `p` — counterfactual: the network's
    /// value for that exact config times the opponent's unnormalised reach
    /// into the leaf. Runs off the `h` left by the last `pbs_head` query, and
    /// is one dot product per config.
    ///
    /// Rows below `from` take their `v(c)` from `vcache` instead of the
    /// network; every row is scaled by the reach mass it has now.
    fn readout_from(&mut self, p: usize, from: usize) {
        let _t = timed!(LEAFPOST);
        let empty = self.nets.value.is_empty();
        let queried: usize = self.leaf_rows[from..]
            .iter()
            .map(|&i| self.nc[i][p] as usize)
            .sum();
        crate::prof::work(0, 0, 0, queried);
        self.trace.readout_cfgs += queried as u64;
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
            // A terminal leaf's value is the game's, not the network's, but
            // it travels the same arithmetic afterwards.
            self.host.as_mut().expect(HOST_PATH).vals[vo..vo + n].fill(u * opp_reach);
        }
        let d = crate::net::D;
        let cfr = self.host.as_mut().expect(HOST_PATH);
        let (reach, vals, vcache) = (&cfr.reach, &mut cfr.vals, &mut cfr.vcache[p]);
        let (roff, ncs, voff, coff, cidx, cf) = (
            &self.roff,
            &self.nc,
            &self.voff,
            &self.leaf_coff,
            &self.leaf_cidx,
            &self.cf,
        );
        for (r, &i) in self.leaf_rows.iter().enumerate() {
            let n = ncs[i][p] as usize;
            let vo = voff[i] as usize;
            if empty {
                vals[vo..vo + n].fill(0.0);
                continue;
            }
            // `pbs_head` wrote the queried rows at the front of `h`, so the
            // tree's row `r` is `h`'s row `r - from`.
            if r >= from {
                let cs = coff[2 * r + p] as usize;
                self.nets.value.values(
                    &self.h[(r - from) * d..(r - from + 1) * d],
                    cf,
                    &cidx[cs..cs + n],
                    &mut vcache[vo..vo + n],
                );
            }
            let ra = roff[i] as usize + if opp == 1 { ncs[i][0] as usize } else { 0 };
            let opp_reach: f32 = reach[ra..ra + ncs[i][opp] as usize].iter().sum();
            for (value, &v) in vals[vo..vo + n].iter_mut().zip(&vcache[vo..vo + n]) {
                *value = v * opp_reach;
            }
        }
    }

    fn update_regrets(&mut self, traverser: usize) {
        // Reaches are already consistent with `cur`: `new` establishes that,
        // every `step` re-establishes it after regret matching, and the
        // fixed-policy passes restore it before returning, so recomputing them
        // here would repeat the previous pass exactly.
        // `Cfg::refresh` says how often the network runs. Rows growth has added
        // since the last query have nothing to reuse and are always queried.
        let from = if self.cfg.refresh_due(self.steps[traverser]) {
            0
        } else {
            self.cached[traverser]
        };
        self.leaf_values_from(traverser, from);
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
        let cfr = self.host.as_mut().expect(HOST_PATH);
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
                let (lo, hi) = cfr.vals.split_at_mut(vc);
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
            cfr.vals[vbase..vbase + nc].fill(if br { f32::NEG_INFINITY } else { 0.0 });
            if mode == Back::Regret && me == traverser {
                // A cell whose action has no successor information state is
                // never visited by the pass that fills these, and must read
                // zero when the regret pass gets to it.
                let so = self.soff[i] as usize;
                let cells = self.nodes[i].legal_action.len();
                cfr.qval[so..so + cells].fill(0.0);
            }
            if me == traverser {
                let n = &self.nodes[i];
                let so = self.soff[i] as usize;
                // Children are built after their parent, so the parent's value
                // row and every child's are disjoint slices of one arena.
                let (lo, hi) = cfr.vals.split_at_mut(self.voff[i + 1] as usize);
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
                            Back::Regret => {
                                // Kept, not re-gathered. The regret pass below
                                // needs this same number, and finding it again
                                // means another random hop into a child's value
                                // row -- the cache-hostile part of the sweep,
                                // paid twice per cell for nothing. The
                                // expansion phase reads it as PUCT's Q.
                                cfr.qval[so + cell] = av;
                                vi[c] += av * self.cur[so + cell];
                            }
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
                                // The action value the pass above kept. A cell
                                // with no successor was skipped there and
                                // still reads the zero this node's cells were
                                // cleared to, which is what re-forming it from
                                // +0 used to produce.
                                let delta = cfr.qval[so + cell] - base;
                                let at = so + cell;
                                let old = cfr.regret[at];
                                let r = old * if old > 0.0 { da } else { db } + delta;
                                cfr.regret[at] = r;
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
                        for x in cfr.sum_strat[i].iter_mut() {
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
                        cfr.vals[vbase + c] += cfr.vals[cv + c];
                    }
                }
            }
        }
    }

    /// One iteration of the regret update phase: **simultaneous updates**, as
    /// Student of Games specifies.
    ///
    /// Both players are traversed against the same reach profile, so each of
    /// them best-responds to the strategy the other held at the start of the
    /// iteration rather than to a strategy the same iteration already moved.
    /// The two traversals do not interfere: values are per traverser, and a
    /// player's regret matching writes only its own decision nodes.
    ///
    /// This is twice the work of an alternating half-iteration and twice the
    /// updates, so a solve of `iters` iterations now gives each player `iters`
    /// updates rather than `iters / 2`.
    pub fn step(&mut self) {
        self.trace.iters += 1;
        self.trace.row_iters += self.leaf_rows.len() as u64;
        self.trace.cidx_iters += self.leaf_cidx.len() as u64;
        self.trace.cell_iters += self.ncells as u64;
        self.update_regrets(0);
        self.update_regrets(1);
        self.precompute_reaches();
        self.avg_block();
        self.steps[0] += 1;
        self.steps[1] += 1;
    }

    /// Add the fresh reach-weighted iterate to the running strategy sum.
    /// Normalisation is deferred to `finish`.
    ///
    /// Both players in one walk. A decision node belongs to exactly one of
    /// them, so a per-player call skipped half the nodes and paid the whole
    /// traversal twice -- which was right while CFR alternated traversers and
    /// only one player's sum moved per iteration, and is not now that `step`
    /// updates both.
    pub fn avg_block(&mut self) {
        let _t = timed!(AVG);
        self.avg_touched = [true; 2];
        let cfr = self.host.as_mut().expect(HOST_PATH);
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.leaf || n.chance {
                continue;
            }
            let me = n.player as usize;
            let nc = n.nc(me);
            let so = self.soff[i] as usize;
            let ra = self.roff[i] as usize + if me == 1 { self.nc[i][0] as usize } else { 0 };
            for c in 0..nc {
                let r = cfr.reach[ra + c];
                for cell in n.legal_row(c) {
                    cfr.sum_strat[i][cell] += r * self.cur[so + cell];
                }
            }
        }
    }

    pub fn multistep(&mut self, iters: usize) {
        for _ in 0..iters {
            self.step();
        }
    }

    /// Expansion simulations owed by iteration `i`, and none once the tree has
    /// nowhere left to grow.
    ///
    /// Without that second clause the rest of `s` is spent sampling
    /// trajectories that end on leaves growth may not touch, which is exactly
    /// the failure the deleted node ceiling used to cause.
    fn expansions_at(&self, i: usize) -> usize {
        if self.nodes[0].exhausted {
            0
        } else {
            self.cfg.expansions_at(i)
        }
    }

    /// Consume the replies this solve was waiting for, do the host's share of
    /// the work, and say what it wants next.
    ///
    /// The host's share is growth, which is the game's rules: it turns the
    /// leaves an expansion sampled into decision nodes. Everything else is a
    /// call. The solve owns its whole state, so nothing about it blocks a
    /// thread -- it can sit in a queue between two rounds and be picked up on
    /// whichever core comes free.
    pub fn advance(&mut self, replies: &[Reply]) -> Step {
        if self.nets.device {
            self.advance_on_device(replies)
        } else {
            self.advance_on_host(replies)
        }
    }

    /// Student of Games' GT-CFR, with the CFR loop on this host.
    ///
    /// `SoG(s, c)`: `s` expansions in total, `c` of them after each regret
    /// update, so the solve runs `ceil(s / c)` updates. Growing and
    /// solving interleave rather than staging, which is the point: the strategy
    /// decides where the tree goes, and the tree decides what the strategy is
    /// worth.
    ///
    /// The only calls this path raises are the trunk over fresh leaves and the
    /// encoder over fresh configs. Both are properties of the subgame rather
    /// than of an iteration, so they are asked for once per growth. The join
    /// that every iteration pays for, and the policy head the expansion phase
    /// reads, run inline on the core the solve is already on.
    fn advance_on_host(&mut self, replies: &[Reply]) -> Step {
        if self.phase == Phase::Iterating {
            self.absorb(replies);
        }
        let iters = self.cfg.iters();
        loop {
            // Whatever the last growth added, before the iteration reads it.
            let calls = self.growth_calls();
            if !calls.is_empty() {
                self.phase = Phase::Iterating;
                return Step::Calls(calls);
            }
            if self.at == iters {
                self.finish();
                self.phase = Phase::Done;
                return Step::Done(self.collect.map(|q| self.harvest(q)));
            }
            // The same round the device runs, on this core: `done` regret
            // updates against a frozen tree, each sampling `want` trajectories,
            // and one growth at the end from all of them.
            let (done, want) = self.round_shape();
            // The expansion phase reads the prior at every node it walks
            // through, and growth has just run the batch that the nodes it
            // added were waiting for. Once a round, which is where the card's
            // policy-head stage sits.
            self.refresh_priors();
            // Every phase of a round runs before any of its leaves is grown,
            // so the round's leaves are collected in one place and each phase
            // draws until it has `want` the round has not taken yet.
            let mut taken = Vec::new();
            for _ in 0..done {
                self.at += 1;
                self.step();
                self.expansion_phase(want, &mut taken);
            }
            let grew = !taken.is_empty();
            for leaf in taken {
                self.expand(leaf);
            }
            if grew {
                // Growth appended reach rows after `step` propagated the old
                // tree. The next regret update must see the new leaves.
                self.precompute_reaches();
            }
        }
    }

    /// The same search, with the CFR loop on the backend.
    ///
    /// The host keeps growth and nothing else: growth is the game rules, so it
    /// turns the leaves an expansion sampled into decision nodes and describes
    /// them. Everything the description feeds — the reaches, the network at
    /// every leaf, the policy prior, the regret update, the average strategy
    /// and the expansion trajectories themselves — happens on the card, because
    /// the arenas they read are tens of megabytes and a round trip an iteration
    /// is more traffic than the bus has.
    ///
    /// The round is the one `advance_on_host` runs, iteration for iteration:
    /// `Cfg::batch` regret updates against a frozen tree, each sampling its
    /// own trajectories, and one growth from all of them. A trajectory that
    /// lands on a leaf the round already took is drawn again, so what the round
    /// hands back is distinct leaves and `s` counts nodes the tree gains rather
    /// than trajectories walked.
    ///
    /// The growth rule itself is the same rule on both, and deliberately so:
    /// the same stream, the same warp-shaped sums, the same per-simulation
    /// treatment of a dead end. What the two backends cannot share is the
    /// numbers the rule reads — a cuBLAS leaf pass and a host one part company
    /// in the last bits — so two whole solves still build different trees.
    /// `cuda_parity` holds the rule to the card by giving the host the card's
    /// own arenas.
    fn advance_on_device(&mut self, replies: &[Reply]) -> Step {
        match self.phase {
            Phase::Fresh => {}
            Phase::Iterating => {
                self.absorb(replies);
                let last = replies.last().expect("a round answers every call it was given");
                // Distinct by construction: a phase draws until it has leaves
                // no phase of this round has taken. A short row reads as
                // nothing, which is a phase that spent its draws.
                for &leaf in &last.leaves.clone() {
                    if leaf == crate::contract::NO_ROW {
                        continue;
                    }
                    self.expand(leaf as usize);
                }
            }
            Phase::Reading => {
                self.absorb(replies);
                self.phase = Phase::Done;
                let last = replies.last().expect("a round answers every call it was given");
                return Step::Done(self.read_back(last));
            }
            Phase::Done => unreachable!("a finished solve is not advanced again"),
        }
        if self.at < self.cfg.iters() {
            self.phase = Phase::Iterating;
            Step::Calls(self.iterate_round())
        } else {
            self.phase = Phase::Reading;
            Step::Calls(self.read_round())
        }
    }

    /// The next round: iterations it carries, and leaves each of their
    /// expansion phases takes.
    ///
    /// Every iteration that owes the same number of expansions rides in one
    /// round, capped at `batch`. The tree is frozen for the whole round, so
    /// the host has nothing to do between those iterations and should not be
    /// woken; it grows once at the end, from every leaf the round took. The
    /// tail a solve runs once its tree can no longer grow is the same rule
    /// with `want = 0`, not a case of its own.
    ///
    /// The cap is there because each iteration issues a hundred-odd dependent
    /// launches from the one driver thread, and a round runs as many as its
    /// longest member asks for, so an unbounded run makes every other solve in
    /// the round wait for it.
    fn round_shape(&self) -> (usize, usize) {
        let (iters, at) = (self.cfg.iters(), self.at);
        let want = self.expansions_at(at + 1);
        let done = (at + 1..=iters)
            .take_while(|&k| self.expansions_at(k) == want)
            .count()
            .min(self.cfg.batch.max(1));
        (done, want)
    }

    /// The next round of iterations, and the expansion phases inside it.
    fn iterate_round(&mut self) -> Vec<Call> {
        let (done, expand) = self.round_shape();
        let mut calls = self.growth_calls();
        // The iteration's decay factors read the step count as it stands, so
        // both calls are built before it advances. The tree call also names the
        // nodes whose prior the card is to fill, which it does between the
        // scatter and the iteration that reads it.
        calls.push(self.tree_call());
        calls.push(Call::Iterate {
            solve: self.slot,
            step: self.steps[0],
            iters: done,
            expand,
            cfr: self.cfg.cfr,
            puct: self.cfg.puct,
        });
        self.steps = [self.steps[0] + done, self.steps[1] + done];
        self.avg_touched = [true; 2];
        self.at += done;
        calls
    }

    /// The last round: the reference strategy, and what a collected solve
    /// keeps.
    ///
    /// One round, not two. The read materialises the average, runs the value
    /// pass under it and slices out the root's policy, the root's values and
    /// the reach at the leaves this solve nominates — so a harvest is the round
    /// that ends the solve rather than one after it. An uncollected solve asks
    /// for the policy alone, and the value pass, which is most of a CFR
    /// iteration, does not run for it.
    fn read_round(&mut self) -> Vec<Call> {
        let mut calls = self.growth_calls();
        calls.push(self.tree_call());
        self.picks = match self.collect {
            None => Vec::new(),
            Some(q) => self.with_rng(|sv, rng| {
                (0..q)
                    .filter(|_| !sv.leaf_rows.is_empty())
                    .map(|_| sv.leaf_rows[rng.below(sv.leaf_rows.len())])
                    .collect()
            }),
        };
        // The card holds one value arena per traverser, so the second player's
        // root row sits a whole arena past the first's.
        let nvals = self.nvals as u32;
        let vals_at = match self.collect {
            None => [(0, 0); 2],
            Some(_) => [
                (self.voff[0], self.nc[0][0]),
                (nvals + self.voff[0], self.nc[0][1]),
            ],
        };
        let reach_at = self
            .picks
            .iter()
            .map(|&i| (self.roff[i], self.nc[i][0] + self.nc[i][1]))
            .collect();
        let (at, cells) = self.root_cells();
        calls.push(Call::Read {
            solve: self.slot,
            touched: self.avg_touched,
            vals_at,
            policy_at: (at as u32, cells as u32),
            reach_at,
        });
        calls
    }

    /// What the last round brought back: the root's slice of the reference
    /// strategy, and — for a collected solve — its values and the beliefs at
    /// the leaves it nominated.
    ///
    /// The arenas stay where they are. Everything else the value pass touches
    /// is tens of megabytes and has no reader here.
    fn read_back(&mut self, r: &Reply) -> Option<Solved> {
        // The root's row and nothing after it. The card holds the whole average
        // and the round carries back only this slice, so an arena of `ncells`
        // would be a megabyte of zeroes with `at + cells` real numbers in it.
        let (at, cells) = self.root_cells();
        self.avg = vec![0.0; at + cells];
        self.avg[at..at + cells].copy_from_slice(&r.b);
        self.collect?;
        let n0 = self.nc[0][0] as usize;
        let value = [r.a[..n0].to_vec(), r.a[n0..].to_vec()];
        let policy = self.root_policy();
        let mut queries = Vec::with_capacity(self.picks.len());
        let mut cut = 0;
        for &i in &self.picks {
            let beliefs = std::array::from_fn(|p| {
                let k = self.nc[i][p] as usize;
                let mut w = vec![0.0; k];
                normalize_weights(&r.c[cut..cut + k], &mut w);
                cut += k;
                Belief { cfg: self.nodes[i].cfgs[p].to_vec(), p: w }
            });
            queries.push((self.states[i].clone(), beliefs));
        }
        Some(Solved { value, queries, policy })
    }

    /// Where the root's strategy cells are, or nothing when it has none.
    fn root_cells(&self) -> (usize, usize) {
        let n = &self.nodes[0];
        if n.leaf || n.chance {
            (0, 0)
        } else {
            (self.soff[0] as usize, n.legal_action.len())
        }
    }

    /// Everything the card has yet to be told about the tree: the description
    /// since `sent_from`, the strategy cells this growth appended, and — the
    /// first time — the root beliefs and the seed of the expansion's stream.
    fn tree_call(&mut self) -> Call {
        self.contract_extend();
        let sent = self.sent_cells;
        self.sent_cells = self.ncells;
        let first = self.steps[0] == 0 && self.sent_from == 0;
        if first {
            self.sent = Default::default();
        }
        // Built here rather than in the backend. A card has one driver thread
        // and it is the round's bottleneck, so a solve's marshalling belongs on
        // the worker that grew the tree.
        let mut w = Writes::default();
        let resent = std::mem::take(&mut self.resent);
        self.contract
            .write_into(&mut w, &mut self.sent, self.sent_from, &self.rewrite, &resent);
        w.u32s(Dst::LeafNode, 0, &self.leaf_rows.iter().map(|&i| i as u32).collect::<Vec<_>>());
        w.u32s(Dst::Term, 0, &self.term_leaves.iter().map(|&i| i as u32).collect::<Vec<_>>());
        if first {
            let b = [&self.root_belief[0].p[..], &self.root_belief[1].p[..]].concat();
            w.f32s(Dst::Rootb, 0, &b);
        }
        // The tail this growth appended, which `cur` and `prior` both start at.
        // The prior of a node the card primes is written there, by the policy
        // head, after this scatter has laid the uniform start down.
        w.f32s_both(Dst::Cur, Dst::Prior, sent, &self.cur[sent..]);
        let (prime, acts, cells) = self.prime();
        let call = Call::Tree {
            solve: self.slot,
            writes: w,
            fresh: first,
            ncells: self.ncells,
            nreach: self.nreach,
            nvals: self.nvals,
            levels: self.contract.level_start.clone(),
            nterm: self.term_leaves.len(),
            seed: first.then_some(self.seed),
            prime,
            acts,
            cells,
            prior_temp: self.cfg.prior_temp,
        };
        self.sent_from = self.nodes.len();
        call
    }

    /// Describe whatever the tree has grown since the last call.
    ///
    /// `Contract` is append-only apart from the leaves this growth turned into
    /// decision nodes, so what moved is exactly those rows and the tail. Taking
    /// the lowest of them and sending everything after it sends most of the
    /// tree whenever one shallow leaf is expanded, which is a few hundred
    /// kilobytes a solve an iteration for a handful of rows.
    fn contract_extend(&mut self) {
        let _t = timed!(CONTRACT);
        let grown = std::mem::take(&mut self.grown);
        let resealed = self.take_resealed();
        self.sent_from = self.contract.built;
        self.rewrite.clear();
        self.rewrite
            .extend(grown.iter().copied().filter(|&g| (g as usize) < self.sent_from));
        let mut c = std::mem::take(&mut self.contract);
        Arc::make_mut(&mut c).extend(self, &grown, &resealed);
        self.contract = c;
        self.resent = resealed;
    }

    /// The cell PUCT would take from one config's legal row.
    ///
    /// `Q + c_puct * P * sqrt(sum N) / (1 + N)`, with `Q` the counterfactual
    /// action value divided by the opponent's reach mass at this node. That
    /// division is what Student of Games means by "normalized by the sum of
    /// the opponent's reach probability at `s_i` to resemble state-conditional
    /// action values": the raw value carries the opponent's reach as a factor,
    /// so without it a node deep behind an unlikely opponent line would look
    /// worthless next to its own siblings rather than being compared with
    /// them.
    fn puct_choice(&self, node: usize, row: std::ops::Range<usize>, opp: usize) -> Option<usize> {
        let so = self.soff[node] as usize;
        let reach = self.reach_of(node, opp);
        let [mass] = warp32_sum(reach.len(), |i| [reach[i]]);
        let scale = if mass > 1e-30 { 1.0 / mass } else { 0.0 };
        let cfr = self.cfr();
        let [total] = warp32_sum(row.len(), |i| [cfr.visits[so + row.start + i]]);
        let explore = self.cfg.puct * total.max(0.0).sqrt();
        let mut best = None;
        let mut best_score = f32::NEG_INFINITY;
        for cell in row {
            if !self.live_cell(node, cell) {
                continue;
            }
            let at = so + cell;
            let score = cfr.qval[at] * scale
                + explore * cfr.prior[at] / (1.0 + cfr.visits[at]);
            if score > best_score {
                best_score = score;
                best = Some(cell);
            }
        }
        best
    }

    /// Whether the expansion phase may descend through one legal cell: the
    /// acting config has a successor there, and the subtree behind it still
    /// has somewhere to grow. A trajectory into either kind of dead end can
    /// only end on a leaf growth may not touch, and the simulation is then
    /// spent for nothing -- which is what a mature tree does to the whole of
    /// its remaining budget once its frontier stops being expandable.
    fn live_cell(&self, node: usize, cell: usize) -> bool {
        let n = &self.nodes[node];
        n.legal_trans[cell] != NO_TRANS
            && !self.nodes[n.legal_child[cell] as usize].exhausted
    }

    /// The decision nodes that are ready for a policy prior: the batch has
    /// reached their board vector and nothing has given them one yet.
    ///
    /// Deferred rather than done inside `grow` because a node is expanded
    /// before the batch carrying its board vector has necessarily run — the
    /// root most of all, which `Solver::new` expands before any network call.
    /// Deferring also makes this one batched pass per expansion phase instead
    /// of one small call per node.
    ///
    /// Only nodes that get expanded need a prior. A leaf has no action list and
    /// an expansion trajectory stops there, so this is exactly Student of
    /// Games' "the prior policy `p` obtained from the queries", asked for at
    /// the moment it first has a use.
    fn ready_for_prior(&mut self) -> Vec<usize> {
        let mut want: Vec<usize> = Vec::new();
        let mut queue = std::mem::take(&mut self.wants_prior);
        queue.retain(|&i| {
            let i = i as usize;
            if self.primed[i] || self.nodes[i].leaf || self.nodes[i].chance {
                return false;
            }
            if self.row_of[i] == u32::MAX {
                return false;
            }
            if (self.row_of[i] as usize) < self.batch_rows {
                want.push(i);
                return false;
            }
            true
        });
        self.wants_prior = queue;
        want
    }

    /// The same nodes, described for the card, which runs the policy head
    /// itself. See `farm::Prime`.
    pub(crate) fn prime(&mut self) -> (Vec<crate::farm::Prime>, Vec<u32>, Vec<u32>) {
        let want = self.ready_for_prior();
        let (mut prime, mut acts, mut cells) = (Vec::new(), Vec::new(), Vec::new());
        for i in want {
            let n = &self.nodes[i];
            prime.push(crate::farm::Prime {
                node: i as u32,
                row: self.row_of[i],
                at: (acts.len() / 5) as u32,
                na: n.na() as u32,
                cell_at: cells.len() as u32,
                nc: n.nc(n.player as usize) as u32,
            });
            // Already the column each block of `Net::action_feats` sets, so
            // "spends nothing" and "names no hex" cross as the column past the
            // last rather than as a sentinel the card would have to fold.
            for a in 0..n.na() {
                let slot = n.aslot[a];
                let hex = |h: u8| if h == NONE { N_HEXES as u32 } else { h as u32 };
                let h = n.acts[a].hexes();
                acts.extend([
                    n.acts[a].kind() as u32,
                    if slot < 0 { NSLOT as u32 } else { slot as u32 },
                    hex(h[0]),
                    hex(h[1]),
                    hex(h[2]),
                ]);
            }
            cells.extend_from_slice(&n.legal_action);
            self.primed[i] = true;
        }
        (prime, acts, cells)
    }

    /// Fill the policy prior of every decision node that is ready for one, on
    /// this host. The device path sends `prime` instead.
    fn refresh_priors(&mut self) {
        if self.nets.value.is_empty() {
            return;
        }
        let _t = timed!(PRIOR);
        let want = self.ready_for_prior();
        if want.is_empty() {
            return;
        }
        // One description per (node, action), and the board each is played on.
        //
        // Only the boards these nodes stand on, packed. `Net::actions`
        // projects every board row it is handed, and handing it the whole leaf
        // batch meant projecting a couple of thousand of them to reach the
        // handful just expanded -- which measured at thirty-one cpu-ms an
        // iteration per thread, more than every other host phase together.
        let d = crate::net::D;
        let mut boards = Vec::with_capacity(want.len() * d);
        let mut feat = Vec::new();
        let mut board_of: Vec<u32> = Vec::new();
        let mut base = Vec::with_capacity(want.len());
        for &i in &want {
            base.push(board_of.len() as u32);
            let at = self.board_of[self.row_of[i] as usize] as usize * d;
            let mine = (boards.len() / d) as u32;
            boards.extend_from_slice(&self.pb[at..at + d]);
            let n = &self.nodes[i];
            for a in 0..n.na() {
                let at = feat.len();
                feat.resize(at + crate::net::AFEAT, 0.0);
                Net::action_feats(
                    n.acts[a].kind(),
                    n.aslot[a],
                    n.acts[a].hexes(),
                    &mut feat[at..],
                );
                board_of.push(mine);
            }
        }
        let na = board_of.len();
        let mut e = Vec::new();
        self.nets.value.actions(&feat, &boards, &board_of, na, &mut e);

        // `logit(c, a) = <f_p(c), e(a)>` over the node's own legal cells, then
        // a softmax across each config's row.
        let mut logit = Vec::new();
        let prior = &mut self.host.as_mut().expect(HOST_PATH).prior;
        for (k, &i) in want.iter().enumerate() {
            let me = self.nodes[i].player as usize;
            let q = 2 * self.row_of[i] as usize + me;
            let cs = self.leaf_coff[q] as usize;
            let n = &self.nodes[i];
            let cells = n.legal_action.len();
            logit.clear();
            logit.resize(cells, 0.0);
            let cfg: Vec<u32> = (0..cells)
                .map(|cell| self.leaf_cidx[cs + n.cell_row[cell] as usize])
                .collect();
            let act: Vec<u32> = (0..cells)
                .map(|cell| base[k] + n.legal_action[cell])
                .collect();
            self.nets.value.policy(&self.cp, &e, &cfg, &act, &mut logit);
            let so = self.soff[i] as usize;
            let inv_t = 1.0 / self.cfg.prior_temp.max(1e-6);
            for c in 0..n.nc(me) {
                let row = n.legal_row(c);
                let top = row
                    .clone()
                    .map(|cell| logit[cell])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for cell in row.clone() {
                    let v = ((logit[cell] - top) * inv_t).exp();
                    prior[so + cell] = v;
                    total += v;
                }
                let scale = if total > 0.0 {
                    1.0 / total
                } else {
                    1.0 / row.len().max(1) as f32
                };
                for cell in row {
                    prior[so + cell] *= scale;
                }
            }
            self.primed[i] = true;
        }
    }

    /// One expansion simulation, and the growth it produces: sample a world
    /// from the root beliefs, walk down under the average strategy, and grow
    /// the leaf it reaches. The counterpart to `step` — a GT-CFR iteration is
    /// one `step` followed by `expand` of these.
    ///
    /// False when nothing grew, which is a spent budget, a trajectory that ran
    /// into a terminal, or a config with no legal action there.
    pub fn expand_once(&mut self) -> bool {
        let Some(leaf) = self.with_expand_rng(|sv, rng| sv.sample_leaf(rng)) else {
            return false;
        };
        self.expand(leaf);
        !self.nodes[leaf].leaf
    }

    /// Grow the whole subgame, and say whether it fitted in `cap` nodes.
    ///
    /// Production never does this — the point of growing is to *not* build the
    /// whole tree. It exists for the tests and sizing tools that need the
    /// complete subgame of a small endgame. A real mid-game position is not
    /// one of those and its subgame runs to millions of nodes, so the bound is
    /// the caller's way of asking for the subgame only if it is small: a
    /// `false` means the tree it now holds is a partial one.
    pub fn grow_full(&mut self, cap: usize) -> bool {
        let mut at = 0usize;
        while at < self.nodes.len() {
            if self.nodes.len() > cap {
                return false;
            }
            if self.nodes[at].leaf && self.nodes[at].expandable {
                self.expand(at);
            }
            at += 1;
        }
        true
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
        let mut out = [Vec::new(), Vec::new()];
        for p in 0..2usize {
            // One entry point for a leaf query, so a batched backend -- which
            // holds this solve's board and config vectors and therefore leaves
            // the host's copies empty -- is not bypassed here.
            self.leaf_values(p);
            // `backprop` for the second player overwrites the first's values,
            // so the root slice is taken before the next pass runs.
            self.backprop(p, &reference, Back::Value);
            let n = self.nc[0][p] as usize;
            let vo = self.voff[0] as usize;
            out[p] = self.cfr().vals[vo..vo + n].to_vec();
        }
        out
    }

    /// One expansion phase: draw trajectories until `want` leaves the round has
    /// not already taken have been found, and append them to `taken`.
    ///
    /// The tree is frozen for a whole round, so a trajectory that ends on a
    /// leaf an earlier trajectory of the round took would grow nothing and the
    /// phase draws again. That is what makes `s` a count of *distinct*
    /// expansions, rather than a count of trajectories that mostly land where
    /// an earlier one of the same round did.
    ///
    /// A draw that runs into a dead end costs that draw and no more. Either
    /// way the visits the trajectory left along its path stand -- that is
    /// Student of Games' virtual loss, and it is the thing that sends the next
    /// draw somewhere else.
    ///
    /// `want * TRIES` draws is the bound, and it is why the loop terminates: a
    /// tree whose every reachable leaf has already been taken would otherwise
    /// draw for ever. A phase that spends it stops short of `want`.
    fn expansion_phase(&mut self, want: usize, taken: &mut Vec<usize>) {
        let (mut got, mut draws) = (0usize, 0usize);
        self.with_expand_rng(|sv, rng| {
            while got < want && draws < want * TRIES {
                draws += 1;
                if let Some(leaf) = sv.sample_leaf(rng) {
                    if !taken.contains(&leaf) {
                        taken.push(leaf);
                        got += 1;
                    }
                }
            }
        });
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
        // A tree with nothing left to grow gets no trajectory at all, not even
        // the draws one would spend.
        if self.nodes[0].exhausted {
            return None;
        }
        // One private config per player forms the sampled world.
        let mut c = [
            pick(&self.root_belief[0].p, rng),
            pick(&self.root_belief[1].p, rng),
        ];
        let mut node = 0usize;
        loop {
            if self.nodes[node].leaf {
                debug_assert!(
                    self.nodes[node].expandable,
                    "the descent skips subtrees with nothing to grow"
                );
                return Some(node);
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
            // Student of Games selects by half PUCT and half the search's own
            // average: `pi_select = 1/2 pi_PUCT + 1/2 pi_CFR`. PUCT is a
            // maximisation, so its half is a point mass on the argmax, and
            // sampling the mixture is a coin flip between the two.
            //
            // Both halves are restricted to the cells this world can still
            // grow through. A config whose every legal action is a dead end
            // ends the trajectory here; that is a property of the sampled
            // world, not of the tree, so it does not seal anything.
            let so = self.soff[node] as usize;
            let live = |cell: usize| self.live_cell(node, cell);
            let cell = if rng.unit_f64() < 0.5 {
                self.puct_choice(node, row.clone(), 1 - me)
            } else if self.cfr().sum_strat[node][row.clone()].iter().any(|&x| x > 0.0) {
                pick_live(&self.cfr().sum_strat[node][row.clone()], |i| live(row.start + i), rng)
                    .map(|i| row.start + i)
            } else {
                pick_live(&self.cur[so + row.start..so + row.end], |i| live(row.start + i), rng)
                    .map(|i| row.start + i)
            };
            let cell = cell?;
            // Counted as the trajectory passes, which is also the virtual loss
            // Student of Games adds across the simulations of one iteration:
            // a later simulation of the same phase sees this one's visit.
            self.host.as_mut().expect(HOST_PATH).visits[so + cell] += 1.0;
            c[me] = self.nodes[node].legal_trans[cell] as usize;
            node = self.nodes[node].legal_child[cell] as usize;
        }
    }


    /// Run one expansion phase against arenas some other backend left behind,
    /// appending the leaves it took to `taken`.
    ///
    /// The CFR loop runs on the card in production, so on that path the host's
    /// own copies of these arenas stay at their uniform start. Given numbers
    /// of its own the host would drift a few ulps from the card's and take a
    /// different turn at the first close call, which measures the loop rather
    /// than the growth rule. Given the card's own numbers the two must agree
    /// simulation for simulation, and that is what the parity test asks.
    ///
    /// `visits` is the one arena the phase writes, so a caller comparing a
    /// phase must hand over the state as it stood before that phase ran.
    /// `taken` is the round's leaves so far, which is the other state a phase
    /// reads: the card keeps it in the round's own output buffer.
    ///
    /// Not part of the engine's interface: it gives this solve the arenas in
    /// `a` and advances `seed` by the draws the phase makes.
    #[doc(hidden)]
    pub fn replay_expansion(&mut self, a: &Arenas, want: usize, taken: &mut Vec<usize>) {
        let cells = self.ncells;
        self.cur.copy_from_slice(&a.cur[..cells]);
        // A device solve has no arenas of its own, which is the whole point:
        // the rule is being run on the card's numbers, not on numbers the host
        // made. So the arenas it reads are built here, out of `a`.
        self.host = Some(HostCfr {
            regret: Vec::new(),
            prior: a.prior[..cells].to_vec(),
            visits: a.visits[..cells].to_vec(),
            qval: a.qval[..cells].to_vec(),
            sum_strat: (0..self.nodes.len())
                .map(|i| {
                    let (so, n) = (self.soff[i] as usize, self.nodes[i].legal_action.len());
                    a.sum[so..so + n].to_vec()
                })
                .collect(),
            reach: a.reach[..self.nreach].to_vec(),
            vals: Vec::new(),
            vcache: [Vec::new(), Vec::new()],
        });
        self.expansion_phase(want, taken)
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
    fn harvest(&mut self, queries: usize) -> Solved {
        let value = self.value_pass();
        let queries = self.with_rng(|sv, rng| sv.sample_queries(rng, queries));
        let policy = self.root_policy();
        self.restore();
        Solved { value, queries, policy }
    }

    /// The root's action list and its average policy, per acting config.
    ///
    /// Read off `avg`, which `finish` materialised — so this is the reference
    /// strategy the solve acts under, not the last iterate. An unexpanded root
    /// has no cells and gives an empty policy, which a caller stores as "no
    /// target here" rather than as a uniform one.
    pub fn root_policy(&self) -> Policy {
        let n = &self.nodes[0];
        if n.leaf || n.chance || self.avg.is_empty() {
            return Policy::default();
        }
        let me = n.player as usize;
        let mut out = Policy {
            acts: (0..n.na())
                .map(|a| {
                    let h = n.acts[a].hexes();
                    [
                        n.acts[a].kind() as u8,
                        (n.aslot[a] + 1) as u8,
                        h[0],
                        h[1],
                        h[2],
                    ]
                })
                .collect(),
            ..Default::default()
        };
        let so = self.soff[0] as usize;
        out.off.push(0);
        for c in 0..n.nc(me) {
            for cell in n.legal_row(c) {
                out.act.push(n.legal_action[cell] as u8);
                out.p.push(self.avg[so + cell]);
            }
            out.off.push(out.act.len() as u32);
        }
        out
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
        let (mut nash, mut zero_sum) = (0.0, 0.0);
        for p in 0..2usize {
            // One query serves both walks below: `backprop` skips leaves, so
            // the leaf values it left are still there for the second.
            self.leaf_values(p);
            let vo = self.voff[0] as usize;
            let nc = self.nc[0][p] as usize;
            let expect = |v: &[f32]| -> f32 { (0..nc).map(|c| root[p][c] * v[vo + c]).sum() };
            self.backprop(p, &reference, Back::Value);
            let v = expect(&self.cfr().vals);
            self.backprop(p, &reference, Back::BestResponse);
            nash += expect(&self.cfr().vals) - v;
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
    }

    /// How concentrated the expansion phase's visits are, as the share of them
    /// that went to each config's most-visited action, averaged over the
    /// decision nodes that were visited at all.
    ///
    /// The counters a solve's device cost is proportional to.
    ///
    /// Diagnostics: nothing in a run reads this. The kernel table is a handful
    /// of terms in these -- the trunk runs once per row, the join twice per
    /// row per iteration, the readout and the pooling once per belief-index
    /// entry per iteration, and the two sweeps once per cell per iteration --
    /// so a search budget can be priced without running the farm.
    /// What this solve holds in host memory, group by group.
    ///
    /// The mirror of `cuda::Solve::census`, and it exists for the same reason:
    /// the farm admits the next solve against what the population *will* hold,
    /// so it needs a figure for one solve that is a projection and not a level.
    /// Every term is a capacity already recorded, so the walk is over nodes and
    /// nothing deeper.
    pub fn host_census(&self) -> Vec<(&'static str, usize)> {
        let f = |v: &Vec<f32>| v.capacity() * 4;
        let u = |v: &Vec<u32>| v.capacity() * 4;
        let z = |v: &Vec<usize>| v.capacity() * 8;
        let cfg_bytes = std::mem::size_of::<Config>();
        let mut v = vec![
            (
                "nodes",
                self.nodes.capacity() * std::mem::size_of::<TNode>()
                    + self.nodes.iter().map(TNode::bytes).sum::<usize>(),
            ),
            ("states", self.states.capacity() * std::mem::size_of::<State>()),
            ("contract", self.contract.bytes()),
            (
                "tree",
                u(&self.parent)
                    + u(&self.soff)
                    + u(&self.roff)
                    + u(&self.voff)
                    + u(&self.row_of)
                    + u(&self.grown)
                    + u(&self.resealed)
                    + u(&self.wants_prior)
                    + u(&self.rewrite)
                    + u(&self.resent)
                    + self.nc.capacity() * 8
                    + self.primed.capacity()
                    + z(&self.leaf_rows)
                    + z(&self.term_leaves)
                    + z(&self.picks),
            ),
            ("cur", f(&self.cur)),
            ("avg", f(&self.avg)),
            (
                "batch",
                u(&self.leaf_cidx)
                    + u(&self.leaf_coff)
                    + u(&self.board_of)
                    + f(&self.cphi)
                    + f(&self.xpub)
                    + f(&self.mirror0)
                    + f(&self.cards)
                    + self.cplayer.capacity(),
            ),
            (
                "readout",
                f(&self.pb) + f(&self.jp) + f(&self.cf) + f(&self.cg) + f(&self.cp)
                    + f(&self.xb)
                    + f(&self.h)
                    + f(&self.wbuf),
            ),
            (
                "interning",
                (self.cmap.capacity() + self.bmap.capacity()) * 16,
            ),
            (
                "scratch",
                self.draw_scratch.bytes() + self.cell_order.capacity() * 16,
            ),
            (
                "beliefs",
                self.root_belief
                    .iter()
                    .map(|b| b.cfg.capacity() * cfg_bytes + b.p.capacity() * 4)
                    .sum(),
            ),
        ];
        if let Some(h) = &self.host {
            v.extend([
                ("regret", f(&h.regret)),
                ("prior", f(&h.prior)),
                ("visits", f(&h.visits)),
                ("qval", f(&h.qval)),
                (
                    "sum",
                    h.sum_strat.capacity() * 24
                        + h.sum_strat.iter().map(|r| r.capacity() * 4).sum::<usize>(),
                ),
                ("reach", f(&h.reach)),
                ("vals", f(&h.vals) + f(&h.vcache[0]) + f(&h.vcache[1])),
            ]);
        }
        v.sort_by_key(|&(_, b)| std::cmp::Reverse(b));
        v
    }

    /// Host bytes this solve holds, all of them together.
    pub fn host_bytes(&self) -> usize {
        self.host_census().iter().map(|&(_, b)| b).sum()
    }

    pub fn shape(&self) -> Shape {
        let mut depth = vec![0u32; self.nodes.len()];
        let mut worst = 0;
        for i in 0..self.nodes.len() {
            for &ch in &self.nodes[i].child {
                if ch > i {
                    depth[ch] = depth[i] + 1;
                    worst = worst.max(depth[ch]);
                }
            }
        }
        Shape {
            nodes: self.nodes.len(),
            rows: self.leaf_rows.len(),
            boards: self.nboards,
            cells: self.ncells,
            ncfg: self.ncfg,
            cidx: self.leaf_cidx.len(),
            depth: worst as usize,
            acts: self
                .nodes
                .iter()
                .flat_map(|n| n.legal_off.windows(2).map(|w| (w[1] - w[0]) as usize))
                .max()
                .unwrap_or(0),
            support: self
                .nodes
                .iter()
                .flat_map(|n| n.cfgs.iter().map(|c| c.len()))
                .max()
                .unwrap_or(0),
            reach: self.nreach,
            vals: self.nvals,
            draws: self.nodes.iter().map(|n| n.draw.len()).sum(),
        }
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
