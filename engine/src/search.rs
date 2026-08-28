//! Growing-tree CFR over public belief states, specialised to War Chest.
//!
//! The subgame rooted at a PBS is unrolled over **public observations**. A node
//! is a leaf when it is terminal or when it is a decision the tree has not
//! grown through yet. Chance (a round-start draw, a Warrior Priest draw) is
//! walked through: its private outcome does not branch the public tree. A forced
//! Warrior Priest play is walked through too, so the in-flight coin never
//! reaches the net. Leaf values come from the value network.
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
use crate::farm::{Call, Dst, QueryPick, Reply, Writes};
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

/// One of the eight things a solve grows. A slot is eight allocations, one
/// per entity, each `cap × nfields × 4` bytes. Host and device both ask
/// `Budget::reserve` before an append.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Ent {
    Node = 0,
    Cell = 1,
    Reach = 2,
    Draw = 3,
    Row = 4,
    Board = 5,
    Config = 6,
    Cidx = 7,
}

impl Ent {
    pub const ALL: [Ent; 8] = [
        Ent::Node,
        Ent::Cell,
        Ent::Reach,
        Ent::Draw,
        Ent::Row,
        Ent::Board,
        Ent::Config,
        Ent::Cidx,
    ];
    pub const NAME: [&'static str; 8] = [
        "node", "cell", "reach", "draw", "row", "board", "config", "cidx",
    ];
    pub fn name(self) -> &'static str {
        Self::NAME[self as usize]
    }
}

/// What one solve may hold, in the eight entities it owns.
///
/// A slot is allocated once at this size and reused by every solve that ever
/// runs in it, so this is a *bound* and not a forecast. An expansion that
/// would take the solve past any of these is abandoned.
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
            // A percentile may bound growth, but it may not reject a root: a
            // root without every information state has no valid strategy.
            configs: k(BUDGET_512.configs).max(2 * crate::pbs::MAX_CONFIG_SUPPORT),
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
    /// No budget at all for a complete test-oracle subgame.
    #[cfg(test)]
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
    pub fn cap(&self, e: Ent) -> usize {
        match e {
            Ent::Node => self.nodes,
            Ent::Cell => self.cells,
            Ent::Reach => self.reach,
            Ent::Draw => self.draws,
            Ent::Row => self.rows,
            Ent::Board => self.boards,
            Ent::Config => self.configs,
            Ent::Cidx => self.cidx,
        }
    }

    /// The one guard. False when `n` does not fit this entity.
    pub fn reserve(&self, e: Ent, n: usize) -> bool {
        n <= self.cap(e)
    }

    /// Host bytes a device-resident solve holds at this budget, from the Vec
    /// capacities it would reserve. The card keeps the CFR arenas.
    pub fn host_slot_bytes(&self) -> usize {
        fn cap<T>(n: usize) -> usize {
            Vec::<T>::with_capacity(n).capacity() * std::mem::size_of::<T>()
        }
        cap::<TNode>(self.nodes)
            + cap::<crate::state::State>(self.nodes)
            + cap::<u32>(self.nodes) * 8
            + cap::<u32>(self.cidx)
            + cap::<u32>(self.rows) * 4
            + cap::<f32>(self.cells)
            + cap::<u32>(self.cells) * 6
            + cap::<u32>(self.reach)
            + cap::<u32>(self.draws) * 3
            + cap::<f32>(self.boards * crate::pbs::PUBFEAT) * 2
            + cap::<f32>(self.configs * crate::pbs::CFEAT)
            + cap::<f32>(self.rows * crate::net::D)
            + cap::<f32>(self.configs * crate::net::D)
    }
}

/// The shape a solve at `SoG(512, 8)` is allowed: p99 of each entity over
/// finished solves in a run that trains, after the warm phase. Source:
/// `runs/prof_exp`, SoG window t ∈ [210, 330] (early SoG; warm is 3 min).
const BUDGET_512: Budget = Budget {
    nodes: 16_595,
    rows: 10_090,
    boards: 8_219,
    configs: 921,
    cidx: 259_756,
    reach: 346_018,
    cells: 136_283,
    draws: 174_834,
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
    /// The test oracle's interval between full value-network queries.
    #[cfg(test)]
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
            #[cfg(test)]
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
    #[cfg(any(test, feature = "gpu"))]
    fn refresh_due(&self, done: usize) -> bool {
        #[cfg(test)]
        let refresh = self.refresh;
        #[cfg(not(test))]
        let refresh = 1;
        refresh > 0 && done % refresh as usize == 0
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
    #[cfg(any(test, feature = "gpu"))]
    fn factor(t: f32, p: f32) -> f32 {
        if p.is_infinite() {
            return if p > 0.0 { 1.0 } else { 0.0 };
        }
        let x = t.powf(p);
        x / (x + 1.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum StopReason {
    Complete,
    Exhausted,
    BudgetNode,
    BudgetCell,
    BudgetReach,
    BudgetDraw,
    BudgetRow,
    BudgetBoard,
    BudgetConfig,
    BudgetCidx,
    Other,
}

impl StopReason {
    pub const NAMES: [&'static str; 11] = [
        "complete", "exhausted",
        "budget_node", "budget_cell", "budget_reach", "budget_draw",
        "budget_row", "budget_board", "budget_config", "budget_cidx",
        "other",
    ];
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
    nlegal_off: usize,
    nrev_start: usize,
    nrvd_start: usize,
    ndraw_start: usize,
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
    pub act: Vec<u16>,
    pub p: Vec<f32>,
}

/// How an action points into a position: kind, paying and recruited physical
/// coin types, then source, destination and target hexes. Optional entities use
/// `NONE`. The policy gathers their learned tokens directly.
pub const ACT_BYTES: usize = 6;

pub(crate) fn action_desc(a: &Action, player: u8, ctx: &Ctx, slot: i8) -> [u8; ACT_BYTES] {
    let coin = |k: i8| {
        if k < 0 { NONE } else { player * NSLOT as u8 + k as u8 }
    };
    let recruited = a.recruited();
    let rslot = if recruited == NONE {
        -1
    } else {
        ctx.slot_of[player as usize][recruited as usize]
    };
    let h = a.hexes();
    [a.kind() as u8, coin(slot), coin(rslot), h[0], h[1], h[2]]
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

thread_local! {
    static CONFIG_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {
        std::cell::RefCell::new(Vec::new())
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

/// Nodes a healthy solve of `s` expansions builds, with room to spare: what a
/// retired node array may hold and still be worth pooling.
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

/// One packed public row, hashed.
fn row_key(row: &[u8]) -> u64 {
    let mut h = 0u64;
    for &x in row {
        h = (x as u64 ^ h).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    h
}

/// A uniform `k`-subset of `0..n`, in O(k) space and random draws.
fn sample_indices(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    debug_assert!(k <= n);
    let mut out = Vec::with_capacity(k);
    for j in n - k..n {
        let pick = rng.below(j + 1);
        out.push(if out.contains(&pick) { j } else { pick });
    }
    out
}

fn take_config_buf() -> Vec<f32> {
    CONFIG_POOL.with(|pool| pool.borrow_mut().pop().unwrap_or_default())
}

fn give_config_buf(v: Vec<f32>) {
    if v.capacity() == 0 {
        return;
    }
    CONFIG_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 2 {
            pool.push(v);
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

/// Exact host arithmetic used only by the CUDA parity oracle.
#[cfg(any(test, feature = "gpu"))]
mod reference;
#[cfg(any(test, feature = "gpu"))]
pub use reference::{Arenas, Back, Conv, HostCfr, ReferenceState, Trace};

pub struct Solver {
    /// Owned because a solver retains its context for its full solve.
    pub(crate) ctx: Ctx,
    /// Owned, so a solve can be moved between threads between two rounds.
    net: Arc<Net>,
    /// The solve's own random stream: the world an expansion samples and the
    /// value-query reservoir. Owned for the same reason.
    rng: Rng,
    /// Which of the card's solve slots this one holds. The card keeps a
    /// solve's arenas between its rounds, so every call it raises names the
    /// same slot. Zero, and meaningless, when the backend keeps no state.
    slot: usize,
    /// How many value queries to retain, or nothing when this solve is acted on
    /// and thrown away. An uncollected solve skips the value pass under the
    /// reference strategy, which is most of a CFR iteration.
    collect: Option<usize>,
    /// Iterations run. `advance` picks up from here every time it is called.
    at: usize,
    /// Successful expansion phases. Unlike nodes, this is exactly Student of
    /// Games' `s`: one sampled leaf grown, regardless of its branching factor.
    expansions: u32,
    /// Which round, if any, is in flight.
    phase: Phase,
    /// Uniform reservoir of the value-network queries made during CFR.
    queries: Vec<(State, [Belief; 2])>,
    /// Query events considered by the reservoir.
    query_seen: usize,
    /// Nodes selected from the current device round, in reply order.
    query_nodes: Vec<usize>,
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
    /// Whether the host reference is active. Device solves leave it off, so
    /// they keep no host copies of the card's arenas.
    #[cfg(any(test, feature = "gpu"))]
    reference: bool,
    /// Exact host-reference state for tests and the parity target. The runtime
    /// CUDA path never reads it; the card owns the corresponding arenas.
    #[cfg(any(test, feature = "gpu"))]
    oracle: ReferenceState,
    /// Leaves that have become decision or chance nodes since a reader last
    /// looked. A flat description of the tree is append-only apart from these,
    /// so they are what an incremental update needs to be told.
    pub grown: Vec<u32>,
    /// Whether each player's running sum has been normalised at least once.
    /// Until then the historical average is the literal initial iterate, not
    /// a multiply-then-divide reconstruction of it.
    pub(crate) avg_touched: [bool; 2],
    /// Total legal strategy cells across decision nodes: the length of `cur`,
    /// `avg`, and the card's strategy arenas.
    pub ncells: usize,
    /// Reach and value cells the tree has. The card fits its arenas to these
    /// counts; no host vector exists in production.
    pub nreach: usize,
    pub nvals: usize,
    /// Draw-transition entries over the whole tree, which the budget bounds and
    /// the device's `draw_to` / `draw_p` / `rvd_src` / `rvd_p` are sized by.
    pub ndraws: usize,
    /// Concatenated CSR lengths the contract writes into Reach, besides `nreach`
    /// and `nvals`. Each is one Reach column, so the slot must hold the max.
    nlegal_off: usize,
    nrev_start: usize,
    nrvd_start: usize,
    ndraw_start: usize,
    /// Which entities this solve ran out of, bit `1 << Ent`. The run counts
    /// them. A budget is a percentile of a measured shape, and the count is
    /// the only thing that can argue with the percentile chosen: it says how
    /// often the tail is being truncated, and which term is doing it.
    budget_hit: u8,
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
    // the device's belief block and join output are rewritten per iteration.
    // That split is the whole architecture: the trunk runs ~2,000 times a
    // solve and the join ~158,000.
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
    batch_boards: usize,
    batch_cfgs: usize,
    /// `[2, NTYPE, TYPE]`: the printed-card tokens, one table per player view.
    /// The draft is fixed for the solve, so this is built once.
    pub cards: Vec<f32>,
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
    /// Distinct public encodings `packed` holds.
    pub nboards: usize,
    /// Packed public encoding, one row a distinct public state.
    pub(crate) packed: Vec<u8>,
    /// The mirrored packed first leaf, which is all the card table wants.
    mirror0: Vec<u8>,
    /// The expansion in flight has passed its bound and is unwinding.
    abandon: bool,
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
        give_config_buf(std::mem::take(&mut self.cphi));
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
        net: Arc<Net>,
        cfg: Cfg,
        belief: [Belief; 2],
        rng: Rng,
    ) -> Solver {
        let root_configs: usize = belief.iter().map(Belief::len).sum();
        assert!(
            root_configs <= 2 * crate::pbs::MAX_CONFIG_SUPPORT,
            "root has {root_configs} configs, above the game's support bound"
        );
        assert!(
            root_configs <= cfg.budget.configs,
            "device slot holds {} configs but this root needs {root_configs}",
            cfg.budget.configs
        );
        let cfgs: [Arc<[Config]>; 2] = [
            belief[0].cfg.as_slice().into(),
            belief[1].cfg.as_slice().into(),
        ];
        let mut sv = Solver {
            ctx,
            net,
            rng,
            slot: 0,
            collect: None,
            at: 0,
            expansions: 0,
            phase: Phase::Fresh,
            queries: Vec::new(),
            query_seen: 0,
            query_nodes: Vec::new(),
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
            #[cfg(any(test, feature = "gpu"))]
            reference: cfg!(test),
            #[cfg(any(test, feature = "gpu"))]
            oracle: ReferenceState::default(),
            grown: Vec::new(),
            avg_touched: [false; 2],
            ncells: 0,
            nreach: 0,
            nvals: 0,
            ndraws: 0,
            nlegal_off: 0,
            nrev_start: 0,
            nrvd_start: 0,
            ndraw_start: 0,
            budget_hit: 0,
            roff: Vec::new(),
            voff: Vec::new(),
            nc: Vec::new(),
            steps: [0, 0],
            leaf_rows: Vec::new(),
            term_leaves: Vec::new(),
            leaf_cidx: Vec::new(),
            leaf_coff: Vec::new(),
            cphi: take_config_buf(),
            cplayer: Vec::new(),
            cmap: std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash),
            board_of: Vec::new(),
            bmap: std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash),
            nboards: 0,
            ncfg: 0,
            batch_rows: 0,
            batch_boards: 0,
            batch_cfgs: 0,
            cards: Vec::new(),
            packed: Vec::new(),
            mirror0: Vec::new(),
            abandon: false,
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
        {
            let _t = timed!(BUILD);
            // The root is born a leaf like every other node; `expand` grows it.
            // The budget applies: a first expansion that would not fit the slot
            // is abandoned, and the root stays a leaf. That is rare at this
            // budget; a root that stayed a leaf has no strategy, and the farm
            // still holds a tree that fits.
            sv.nodes.reserve(640);
            sv.cur.reserve(640);
            #[cfg(any(test, feature = "gpu"))]
            if sv.reference {
                let h = &mut sv.oracle.cfr;
                h.reach.reserve(640);
                h.vals.reserve(640);
                h.regret.reserve(640);
            }
            // The expansion's stream lives on the card once it is seeded, so
            // it is drawn here rather than by the round that sends it.
            sv.seed = Rng::new(sv.rng.next_u64()).0;
            let root = sv.push_node(crate::contract::NO_ROW, root.clone(), cfgs);
            sv.expand(root);
            // The first CFR update and every expansion trajectory require
            // reaches for the tree that now exists. The card seeds and sweeps
            // its own, from the root beliefs the first tree call carries, so
            // on that path this pass would be redone before it was ever read.
            #[cfg(any(test, feature = "gpu"))]
            if sv.reference {
                sv.precompute_reaches();
            }
        }
        sv
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

    /// Ask this solve for a training row: the root's values, its policy, and a
    /// reservoir of `queries` value-network calls as roots for later solves.
    /// Without it the solve is acted on and thrown away.
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

    /// Update the query reservoir and return selected indices among new events.
    fn plan_query_events(&mut self, events: usize) -> Vec<usize> {
        let keep = self.collect.unwrap_or(0);
        if keep == 0 || events == 0 {
            self.query_seen += events;
            return Vec::new();
        }
        self.with_rng(|sv, rng| {
            let total = sv.query_seen + events;
            sv.query_seen = total;
            if total <= keep {
                return (0..events).collect();
            }

            let mut new_left = events;
            let mut all_left = total;
            let mut new_count = 0;
            for _ in 0..keep {
                if rng.below(all_left) < new_left {
                    new_count += 1;
                    new_left -= 1;
                }
                all_left -= 1;
            }
            let old_count = keep - new_count;
            let old = sample_indices(rng, sv.queries.len(), old_count);
            sv.queries = std::mem::take(&mut sv.queries)
                .into_iter()
                .enumerate()
                .filter_map(|(i, row)| old.contains(&i).then_some(row))
                .collect();
            sample_indices(rng, events, new_count)
        })
    }

    /// The network batch keeps rows after growth for policy work, but only
    /// live leaves are query roots.
    fn leaf_query_rows(&self, from: usize) -> Vec<usize> {
        self.leaf_rows[from..]
            .iter()
            .copied()
            .filter(|&node| self.nodes[node].leaf)
            .collect()
    }

    /// Attach query-time beliefs to the device round's selected nodes.
    fn absorb_queries(&mut self, reach: &[f32]) {
        let mut cut = 0;
        for node in std::mem::take(&mut self.query_nodes) {
            let beliefs = std::array::from_fn(|p| {
                let n = self.nc[node][p] as usize;
                let mut w = vec![0.0; n];
                normalize_weights(&reach[cut..cut + n], &mut w);
                cut += n;
                Belief { cfg: self.nodes[node].cfgs[p].to_vec(), p: w }
            });
            self.queries.push((self.states[node].clone(), beliefs));
        }
        assert_eq!(cut, reach.len(), "query reach reply has a trailing tail");
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
        let coin = !terminal && s.is_valued();
        let (c0, c1) = (cfgs[0].len(), cfgs[1].len());
        let next_row = if terminal {
            self.leaf_rows.len().max(self.term_leaves.len() + 1)
        } else if coin {
            (self.leaf_rows.len() + 1).max(self.term_leaves.len())
        } else {
            self.leaf_rows.len().max(self.term_leaves.len())
        };
        // A non-root node is one more child pointer, which is a Cell column.
        let next_cell = if parent == crate::contract::NO_ROW {
            self.ncells
        } else {
            self.ncells.max(self.nodes.len())
        };
        if !self.reserve(Ent::Node, self.nodes.len() + 1)
            || !self.reserve(
                Ent::Reach,
                (self.nreach + c0 + c1)
                    .max(self.nvals + c0.max(c1))
                    .max(self.reach_aux()),
            )
            || !self.reserve(Ent::Cell, next_cell)
            || !self.reserve(Ent::Row, next_row)
            || (parent == crate::contract::NO_ROW
                && !self.reserve(Ent::Config, (c0 + c1).div_ceil(2)))
        {
            return parent as usize;
        }
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
        self.nc.push([c0 as u32, c1 as u32]);
        // No cells yet: a leaf has no strategy. `grow` appends its region.
        self.soff.push(self.ncells as u32);
        self.roff.push(self.nreach as u32);
        self.voff.push(self.nvals as u32);
        self.nreach += c0 + c1;
        self.nvals += c0.max(c1);
        #[cfg(any(test, feature = "gpu"))]
        if self.reference {
            let h = &mut self.oracle.cfr;
            h.reach.resize(self.nreach, 0.0);
            h.vals.resize(self.nvals, 0.0);
            h.vcache[0].resize(self.nvals, 0.0);
            h.vcache[1].resize(self.nvals, 0.0);
            h.sum_strat.push(Vec::new());
        }
        self.primed.push(false);
        self.row_of.push(u32::MAX);
        // A valued decision carries a network row. Chance and the forced
        // Warrior Priest play do not. The Row slot also holds terminals, so
        // both appends go through the same reserve.
        if terminal {
            self.term_leaves.push(id);
        } else if coin {
            self.row_of[id] = (self.leaf_coff.len() / 2) as u32;
            self.push_row(id, &s, &cfgs);
            if !self.abandon {
                self.leaf_rows.push(id);
            }
        }
        self.states.push(s);
        id
    }

    /// Push a child and, unless it is somewhere the search may stop, grow it.
    ///
    /// A leaf is a valued decision or a terminal. Chance is walked through: it
    /// does not branch the public tree. A forced Warrior Priest play is walked
    /// through too, so the in-flight coin never reaches the net.
    fn push_child(&mut self, parent: usize, s: State, cfgs: [Arc<[Config]>; 2]) -> usize {
        if self.abandon {
            return parent;
        }
        let stop = s.is_terminal() || s.is_valued();
        let ch = self.push_node(parent as u32, s, cfgs);
        if self.abandon {
            return ch;
        }
        if !stop {
            self.grow(ch);
        }
        ch
    }

    /// One expansion of leaf `id`, abandoned whole if it does not fit the budget.
    fn expand(&mut self, id: usize) {
        debug_assert!(
            self.nodes[id].leaf && self.nodes[id].expandable,
            "growth turns an expandable leaf into a decision node, and nothing else"
        );
        let fresh = self.nodes.len();
        self.grow(id);
        if self.abandon {
            self.abandon = false;
            self.nodes[id].expandable = false;
        }
        if !self.nodes[id].leaf {
            self.expansions += 1;
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

    /// Whether this solve ever ran out of budget. See `Solver::budget_hit`.
    pub fn budget_hit(&self) -> bool {
        self.budget_hit != 0
    }

    /// Bit `1 << Ent` for each entity this solve ran out of.
    pub fn hit_mask(&self) -> u8 {
        self.budget_hit
    }

    /// How many of this entity the solve holds, as the card would reserve it.
    ///
    /// Each entity is one slot, shared by every column of that entity. `used`
    /// is the max of those columns: a terminal is a Row, a child pointer is a
    /// Cell, `nvals` and the CSR starts are Reach, and the root belief is a
    /// Config. The contract debug-asserts its columns against this.
    pub fn used(&self, e: Ent) -> usize {
        match e {
            Ent::Node => self.nodes.len(),
            Ent::Cell => self.ncells.max(self.nodes.len().saturating_sub(1)),
            Ent::Reach => self.nreach.max(self.nvals).max(self.reach_aux()),
            Ent::Draw => self.ndraws,
            Ent::Row => self.leaf_rows.len().max(self.term_leaves.len()),
            Ent::Board => self.nboards,
            Ent::Config => self.ncfg.max(self.rootb_len()),
            Ent::Cidx => self.leaf_cidx.len(),
        }
    }

    fn reach_aux(&self) -> usize {
        self.nlegal_off
            .max(self.nrev_start)
            .max(self.nrvd_start)
            .max(self.ndraw_start)
    }

    fn rootb_len(&self) -> usize {
        match self.nc.first() {
            Some(&[a, b]) => (a as usize + b as usize).div_ceil(2),
            None => 0,
        }
    }

    pub fn stop_reason(&self) -> StopReason {
        if self.budget_hit != 0 {
            debug_assert!(self.budget_hit.is_power_of_two());
            return match self.budget_hit.trailing_zeros() {
                0 => StopReason::BudgetNode,
                1 => StopReason::BudgetCell,
                2 => StopReason::BudgetReach,
                3 => StopReason::BudgetDraw,
                4 => StopReason::BudgetRow,
                5 => StopReason::BudgetBoard,
                6 => StopReason::BudgetConfig,
                7 => StopReason::BudgetCidx,
                _ => unreachable!(),
            };
        }
        if self.expansions >= self.cfg.s {
            StopReason::Complete
        } else if self.nodes[0].exhausted {
            StopReason::Exhausted
        } else {
            StopReason::Other
        }
    }

    /// Entity counts followed by why growth stopped.
    pub fn counts(&self) -> [u32; 9] {
        let mut out = [0; 9];
        for (i, e) in Ent::ALL.into_iter().enumerate() {
            out[i] = self.used(e) as u32;
        }
        out[8] = self.stop_reason() as u32;
        out
    }

    /// Ensure this entity can hold `n` items. The one host-side guard.
    fn reserve(&mut self, e: Ent, n: usize) -> bool {
        if !self.cfg.budget.reserve(e, n) {
            self.abandon = true;
            self.budget_hit |= 1 << (e as u8);
            false
        } else {
            true
        }
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
            nlegal_off: self.nlegal_off,
            nrev_start: self.nrev_start,
            nrvd_start: self.nrvd_start,
            ndraw_start: self.ndraw_start,
        }
    }

    /// Undo everything an abandoned `grow` appended, and make `id` a leaf
    /// again.
    ///
    /// Growth is all-or-nothing so that a tree is a tree at every moment a
    /// caller can see one. An inner abandon unwinds through every enclosing
    /// `grow`, so the whole expansion is undone.
    fn rewind(&mut self, id: usize, m: Mark) {
        if m.nodes < self.roff.len() {
            self.nreach = self.roff[m.nodes] as usize;
            self.nvals = self.voff[m.nodes] as usize;
        }
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
        #[cfg(any(test, feature = "gpu"))]
        if self.reference {
            self.oracle.cached = self.oracle.cached.map(|k| k.min(m.leaf_rows));
        }
        // `packed` is written by index and reused, so only the count and the
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
        #[cfg(any(test, feature = "gpu"))]
        if self.reference {
            let h = &mut self.oracle.cfr;
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
        self.nlegal_off = m.nlegal_off;
        self.nrev_start = m.nrev_start;
        self.nrvd_start = m.nrvd_start;
        self.ndraw_start = m.ndraw_start;
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
        #[cfg(any(test, feature = "gpu"))]
        if self.reference {
            let ncells = self.ncells;
            let h = &mut self.oracle.cfr;
            h.regret.resize(ncells, 0.0);
            h.visits.resize(ncells, 0.0);
            h.qval.resize(ncells, 0.0);
            // Until the policy head has spoken the prior is the uniform
            // strategy, which is what CFR starts from too.
            h.prior.extend_from_slice(&u);
            h.sum_strat[id] = vec![0.0; cells];
        }
    }

    // ------------------------------------------------------------ tree build

    /// Turn leaf `id` into a decision node, its children pushed as leaves.
    ///
    /// This is the whole of tree growth. One call adds this node's public
    /// children. Chance and a forced Warrior Priest play are grown through;
    /// every other decision stops as a leaf.
    fn grow(&mut self, id: usize) {
        debug_assert!(self.nodes[id].leaf, "only a leaf can be grown");
        // It is a decision node from here, so it wants a policy prior as soon
        // as the batch reaches its row.
        if self.row_of[id] != u32::MAX {
            self.wants_prior.push(id as u32);
        }
        if self.abandon {
            return;
        }
        let mark = self.mark();
        let s = self.states[id].clone();
        debug_assert!(!s.is_terminal(), "a terminal has nothing to grow");
        let cfgs = self.nodes[id].cfgs.clone();
        self.nodes[id].leaf = false;
        // The network batch is append-only because its policy head still needs
        // this row. Query sampling filters it out now that it is interior.
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
            let extra_draw_start = draw.rows() + 1;
            let extra_rvd = self.nc[ch][me] as usize + 1;
            let n = &mut self.nodes[id];
            n.chance = true;
            n.child = vec![ch];
            if !self.reserve(Ent::Draw, self.ndraws + draw.len())
                || !self.reserve(
                    Ent::Reach,
                    self.nreach
                        .max(self.nvals)
                        .max(self.nlegal_off)
                        .max(self.nrev_start)
                        .max(self.nrvd_start + extra_rvd)
                        .max(self.ndraw_start + extra_draw_start),
                )
            {
                self.rewind(id, mark);
                return;
            }
            self.ndraws += draw.len();
            self.ndraw_start += extra_draw_start;
            self.nrvd_start += extra_rvd;
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

        let extra_legal = legal_off.len();
        let extra_rev: usize = child.iter().map(|&c| self.nc[c][me] as usize + 1).sum();
        let extra_cells = legal_action.len();
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
        // several nodes' worth can land with no check between them. Child
        // pointers and the reverse CSR are Cell/Reach columns of the same slot.
        if !self.reserve(
            Ent::Cell,
            (self.ncells + extra_cells).max(self.nodes.len().saturating_sub(1)),
        ) || !self.reserve(
            Ent::Reach,
            self.nreach
                .max(self.nvals)
                .max(self.nlegal_off + extra_legal)
                .max(self.nrev_start + extra_rev)
                .max(self.nrvd_start)
                .max(self.ndraw_start),
        ) {
            self.rewind(id, mark);
            return;
        }
        self.nlegal_off += extra_legal;
        self.nrev_start += extra_rev;
        self.alloc_cells(id);
        self.grown.push(id as u32);
    }

    // -------------------------------------------------------------- CFR core

    /// One leaf's public encoding, interned: the board it reads.
    ///
    /// A row, not two. The board is public and the trunk reads the physical
    /// view only -- the mirrored one was written for every leaf, carried
    /// through the call, and gathered past by everything that read it. What
    /// still wants it is the card table, which holds a view a seat and is
    /// built once a solve off the first board; that one mirror is kept here.
    ///
    /// The packed row is written where the next board would go and kept only if
    /// no earlier board already holds it. Writing first is what makes the
    /// comparison free: the row has to be built to be hashed either way.
    fn encode(&mut self, s: &State) -> u32 {
        let at = self.nboards * ROW_BYTES;
        if self.packed.len() < at + ROW_BYTES {
            self.packed.resize(at + 128 * ROW_BYTES, 0);
        }
        pack_row(s, &self.ctx, &mut self.packed[at..at + ROW_BYTES]);
        if self.nboards == 0 {
            self.mirror0.resize(ROW_BYTES, 0);
            pack_row(&s.mirror(), &self.ctx.mirrored(), &mut self.mirror0);
        }
        let key = row_key(&self.packed[at..at + ROW_BYTES]);
        if let Some(&b) = self.bmap.get(&key) {
            let old = b as usize * ROW_BYTES;
            if self.packed[old..old + ROW_BYTES] == self.packed[at..at + ROW_BYTES] {
                return b;
            }
        }
        if !self.reserve(Ent::Board, self.nboards + 1) {
            return 0;
        }
        let b = self.nboards as u32;
        self.bmap.insert(key, b);
        self.nboards += 1;
        b
    }

    /// One row of the network batch: its public encoding, and its configs
    /// interned into the shared table.
    fn push_row(&mut self, _id: usize, s: &State, cfgs: &[Arc<[Config]>; 2]) {
        debug_assert!(
            s.is_valued(),
            "a network row must be a valued decision"
        );
        if !self.reserve(Ent::Row, (self.leaf_rows.len() + 1).max(self.term_leaves.len())) {
            return;
        }
        let _t = timed!(PUBFEAT);
        let board = self.encode(s);
        self.board_of.push(board);
        for p in 0..2 {
            let res = reserve(s, p as u8, &self.ctx);
            self.leaf_coff.push(self.leaf_cidx.len() as u32);
            for c in cfgs[p].iter() {
                if !self.reserve(Ent::Cidx, self.leaf_cidx.len() + 1) {
                    break;
                }
                // A miss returns 0 and sets `abandon`; the index is still
                // stored so the row's length matches `nc`. The enclosing
                // `grow` rewinds.
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
        if !self.reserve(Ent::Config, self.ncfg + 1) {
            return 0;
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
        if self.net.is_empty() {
            self.batch_rows = rows;
            self.batch_cfgs = self.ncfg;
            return Vec::new();
        }
        let _t = timed!(PUBNET);
        let mut calls = Vec::with_capacity(2);
        let fresh_rows = rows - self.batch_rows;
        let fresh_cfgs = self.ncfg - self.batch_cfgs;
        if self.cards.is_empty() && (fresh_rows > 0 || fresh_cfgs > 0) {
            let both = [&self.packed[..ROW_BYTES], &self.mirror0[..]].concat();
            self.net.cards_from_rows(&both, 2, &mut self.cards);
        }
        if fresh_rows > 0 {
            // The cards in play are fixed at the draft, so every row of the
            // subgame carries the same card block and the table is built once,
            // one view per seat. Everything downstream reads it by canonical
            // coin-type index.
            // Exactly the fresh boards. `packed` is a grown scratch buffer, so
            // an open-ended slice would carry whatever the last, larger
            // subgame left behind — invisible to a solve evaluating alone, and
            // wrong the moment the farm concatenates this call with another.
            let at = self.batch_boards * ROW_BYTES;
            let end = self.nboards * ROW_BYTES;
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
                queries: fresh_rows,
                board_of: self.board_of[self.batch_rows..].to_vec(),
                boards_at: self.batch_boards,
                boards: self.nboards - self.batch_boards,
                packed: self.packed[at..end].to_vec(),
                cards: self.cards.clone(),
                cidx: self.leaf_cidx[cs..].to_vec(),
                coff,
            });
            crate::prof::work(fresh_rows, 0, 0, 0);
        }
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

    /// Record what `growth_calls` made resident. The test oracle also retains
    /// the returned vectors for its host network.
    fn absorb(&mut self, replies: &[Reply]) {
        #[cfg(not(any(test, feature = "gpu")))]
        let _ = replies;
        #[cfg(any(test, feature = "gpu"))]
        let mut at = 0;
        if self.leaf_rows.len() > self.batch_rows {
            #[cfg(any(test, feature = "gpu"))]
            if self.reference {
                let r = &replies[at];
                self.oracle.pb.extend_from_slice(&r.a);
                self.oracle.jp.extend_from_slice(&r.b);
                at += 1;
            }
            self.batch_rows = self.leaf_rows.len();
            self.batch_boards = self.nboards;
        }
        if self.ncfg > self.batch_cfgs {
            #[cfg(any(test, feature = "gpu"))]
            if self.reference {
                let r = &replies[at];
                self.oracle.cf.extend_from_slice(&r.a);
                self.oracle.cg.extend_from_slice(&r.b);
                self.oracle.cp.extend_from_slice(&r.c);
            }
            self.batch_cfgs = self.ncfg;
        }
    }

    /// Expansion simulations owed by iteration `i`, and none once the tree has
    /// nowhere left to grow.
    ///
    /// Without that second clause the rest of `s` is spent sampling
    /// trajectories that end on leaves growth may not touch, which is exactly
    /// the failure the deleted node ceiling used to cause.
    fn expansions_at(&self, i: usize) -> usize {
        if self.budget_hit() || self.nodes[0].exhausted {
            0
        } else {
            self.cfg.expansions_at(i)
        }
    }

    /// Consume the replies this solve was waiting for and request its next
    /// CUDA round.
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
    pub fn advance(&mut self, replies: &[Reply]) -> Step {
        match self.phase {
            Phase::Fresh => {}
            Phase::Iterating => {
                self.absorb(replies);
                let last = replies.last().expect("a round answers every call it was given");
                self.absorb_queries(&last.c);
                // Distinct by construction: a phase draws until it has leaves
                // no phase of this round has taken. A short row reads as
                // nothing, which is a phase that spent its draws.
                for &leaf in &last.leaves.clone() {
                    if leaf == crate::contract::NO_ROW {
                        continue;
                    }
                    if self.budget_hit() {
                        break;
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
        let query_rows = self.leaf_query_rows(0);
        let rows = query_rows.len();
        let selected = self.plan_query_events(done * rows);
        self.query_nodes = selected.iter().map(|&e| query_rows[e % rows]).collect();
        let query = selected
            .into_iter()
            .map(|e| {
                let node = query_rows[e % rows];
                QueryPick {
                    iter: (e / rows) as u32,
                    reach: self.roff[node],
                    len: self.nc[node][0] + self.nc[node][1],
                }
            })
            .collect();
        calls.push(Call::Iterate {
            solve: self.slot,
            step: self.steps[0],
            iters: done,
            expand,
            query,
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
    /// pass under it and slices out the root's policy and values. Query beliefs
    /// were captured during the CFR rounds that made them.
    fn read_round(&mut self) -> Vec<Call> {
        let mut calls = self.growth_calls();
        calls.push(self.tree_call());
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
        let (at, cells) = self.root_cells();
        calls.push(Call::Read {
            solve: self.slot,
            touched: self.avg_touched,
            vals_at,
            policy_at: (at as u32, cells as u32),
        });
        calls
    }

    /// What the last round brought back: the root strategy and values. Query
    /// beliefs arrived with the CFR rounds that made them.
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
        let queries = std::mem::take(&mut self.queries);
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
                at: (acts.len() / ACT_BYTES) as u32,
                na: n.na() as u32,
                cell_at: cells.len() as u32,
                nc: n.nc(n.player as usize) as u32,
            });
            for a in 0..n.na() {
                acts.extend(action_desc(&n.acts[a], n.player, &self.ctx, n.aslot[a]).map(u32::from));
            }
            cells.extend_from_slice(&n.legal_action);
            self.primed[i] = true;
        }
        (prime, acts, cells)
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
                .map(|a| action_desc(&n.acts[a], n.player, &self.ctx, n.aslot[a]))
                .collect(),
            ..Default::default()
        };
        let so = self.soff[0] as usize;
        out.off.push(0);
        for c in 0..n.nc(me) {
            for cell in n.legal_row(c) {
                out.act.push(n.legal_action[cell] as u16);
                out.p.push(self.avg[so + cell]);
            }
            out.off.push(out.act.len() as u32);
        }
        out
    }

    /// The CFR average strategy at the root: the policy used to act.
    pub(crate) fn root_strategy(&self, config: usize) -> &[f32] {
        let row = self.nodes[0].legal_row(config);
        assert!(
            !self.avg.is_empty(),
            "the solve must finish before its average is read"
        );
        let so = self.soff[0] as usize;
        &self.avg[so + row.start..so + row.end]
    }
}
