//! Self-play, training-data generation and evaluation.
//!
//! One loop drives every agent. At each decision node an agent produces a
//! *per-config* policy `P(private action | config)`; the loop then
//!   1. samples the action from the row for the true config,
//!   2. Bayes-updates the public belief from what the opponent actually
//!      **observes** — summing over every private action consistent with that
//!      observation, which is what keeps face-down plays private,
//!   3. applies the action to the true state.
//! Chance nodes are resolved from the true bag and convolve the belief with
//! each config's own draw distribution.
//!
//! ReBeL decisions walk the solved subgame: a `Solver` is built once at a
//! subgame root and runs its full CFR solve there; the game then descends
//! through the tree taking an action at every decision on the way, acting on
//! the CFR *average* strategy (TurboReBeL's reference), and a new solver is
//! built only when the walk reaches a leaf of the tree — a draw, a terminal
//! state, or the depth limit.
//!
//! Each level yields T+1 training rows, TurboReBeL's single-sample
//! multi-iteration generation: the beliefs at the next root under this
//! solve's per-iterate average strategies (carried in `carried_beliefs`),
//! each valued by one fixed-policy pass under the reference strategy
//! (`Solver::value_under`). The first level carries just the live belief.
//!
//! Training data comes in two flavours:
//!   * `Collect::Mc` — the greedy warm start. Value targets blend the realised
//!     game outcome with a squashed handcrafted public-information evaluation.
//!     Without it the value network is noise, CFR plays without purpose, and
//!     games only ever end at the horizon.
//!   * `Collect::Rebel` — the ReBeL loop proper: value targets are the CFR
//!     subgame root values, one per config in each player's belief support.

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::gpu::GpuClient;
use crate::rebel::*;
use crate::rng::Rng;
use crate::search::{node_actions, Cfg, Nets, Solver};
use crate::serialize::{PackedJob, WalkTree};
use crate::state::{Cont, State, BLACK, WHITE, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP};
use rayon::prelude::*;

/// The rulebook's recommended starter matchup. Training on one fixed matchup
/// removes a large chunk of variance; randomised drafts are a distribution
/// extension, behind `random_draft`.
const STARTER_WHITE: [u16; 4] = [17, 12, 4, 9]; // Swordsman, Pikeman, Crossbowman, Light Cavalry
const STARTER_BLACK: [u16; 4] = [1, 3, 8, 16]; // Archer, Cavalry, Lancer, Scout

/// Draftable units. The Warrior Priest pair (ids 18 and 54) is included: their
/// private mid-round draw puts "which coin must I now play" into the private
/// state as `Config::pending_coin`, which the solver, belief filter and walk
/// all carry.
pub const DRAFT_POOL: [u16; 19] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54,
];

pub fn make_game(rng: &mut Rng, random: bool) -> State {
    let first = if rng.next_u64() & 1 == 0 {
        WHITE
    } else {
        BLACK
    };
    if !random {
        return State::from_draft(&STARTER_WHITE, &STARTER_BLACK, first);
    }
    // Eight units off one shuffled pool, four each. The pool is *shared*: a unit
    // type is one card with one set of coins, so the two players' sets are
    // necessarily disjoint. Drafting each side independently would put a second
    // copy of a card's coins into the game — with `Berserker 5` on both sides,
    // ten Berserker coins would exist where the card says five.
    let mut pool = DRAFT_POOL;
    for i in (1..pool.len()).rev() {
        pool.swap(i, rng.below(i + 1));
    }
    State::from_draft(&pool[..4], &pool[4..8], first)
}

// ------------------------------------------------------------- greedy policy

/// A hand-written positional evaluation from `p`'s point of view, over **public
/// information only** — it never inspects either player's hidden coins, so it
/// cannot teach the value network something no legal policy could achieve.
pub fn eval_static(s: &State, p: u8) -> f32 {
    if s.is_terminal() {
        return s.utility(p as usize) * 1e6;
    }
    let bd = board();
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
            .sum::<f32>()
    };
    sc += elim(o) - elim(p);

    // Coverage: for every location each side still has to take, how far its
    // nearest unit is. Per location rather than per unit, which is what pushes
    // units onto the map instead of shuffling on the spot.
    let (mut cover_p, mut cover_o) = (0.0f32, 0.0f32);
    for li in 0..10 {
        let l = bd.location_hexes[li] as usize;
        let (mut bp, mut bo) = (7.0f32, 7.0f32);
        for h in 0..N_HEXES {
            let owner = s.hex_owner[h];
            if owner == NONE {
                continue;
            }
            let d = bd.dist[h][l] as f32;
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

/// `eval_static` squashed into (-1, 1) so it can seed the value network.
pub fn eval_squashed(s: &State, p: u8) -> f32 {
    (eval_static(s, p) / 25.0).tanh()
}

// -------------------------------------------------------------------- agents

#[derive(Clone, Copy)]
pub enum Agent {
    /// Greedy one-ply search on `eval_static`, softmaxed at `temp`.
    Greedy { temp: f32 },
    /// Uniform over legal actions: the weakest reference on the Elo ladder.
    Random,
    /// ReBeL: solve the depth-limited subgame, act on the CFR strategy.
    Rebel { cfg: Cfg, slot: usize },
}

/// A decision node's policy: private actions plus one probability per legal
/// config/action cell, in config-major CSR order.
struct NodePolicy {
    acts: Vec<Action>,
    aslot: Vec<i8>,
    fdown: Vec<bool>,
    legal_off: Vec<u32>,
    legal_action: Vec<u32>,
    probs: Vec<f32>,
}

impl NodePolicy {
    fn frame(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
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

    #[inline]
    fn row(&self, c: usize) -> std::ops::Range<usize> {
        self.legal_off[c] as usize..self.legal_off[c + 1] as usize
    }
}

/// One-ply greedy. An action's score is a property of the successor's public
/// state, so it is evaluated once per action and shared across configs; only
/// the legal set differs between them.
fn greedy_policy(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config], temp: f32) -> NodePolicy {
    let mut np = NodePolicy::frame(s, ctx, player, cfgs);
    let na = np.acts.len();
    let mut score = vec![f32::NEG_INFINITY; na];
    for a in 0..na {
        let rep = cfgs.iter().find(|c| action_legal(c, np.aslot[a]));
        let Some(rep) = rep else { continue };
        let mut probe = s.clone();
        set_config(&mut probe, player, ctx, rep);
        probe.apply_inplace(np.acts[a]);
        score[a] = eval_static(&probe, player) / temp;
    }
    for ci in 0..cfgs.len() {
        let cells = np.row(ci);
        let best = cells.clone().fold(f32::NEG_INFINITY, |best, cell| {
            best.max(score[np.legal_action[cell] as usize])
        });
        let mut sum = 0.0;
        for cell in cells.clone() {
            let a = np.legal_action[cell] as usize;
            let e = (score[a] - best).exp();
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

fn random_policy(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
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

// -------------------------------------------------------------- data records

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collect {
    None,
    /// Monte-Carlo value targets from the game outcome (greedy warm start).
    Mc,
    /// ReBeL: CFR subgame root values.
    Rebel,
}

/// One training row is a public state plus, for each player, that player's
/// whole belief: the exact configs in support, their probabilities, and the
/// value the solve gave each one. Nothing is projected onto a fixed-width
/// basis, so the row carries the same information the solver had.
///
/// The config lists are ragged — a belief support runs from one config to a few
/// hundred — so they live in one flat arena indexed by `coff`, which holds
/// `2 * n + 1` offsets: row `r`, player `p` spans `coff[2*r+p] .. coff[2*r+p+1]`.
#[derive(Default)]
pub struct Data {
    /// `[n * ROW_BYTES]` packed replay rows: raw small integers plus the aux
    /// targets (see `rebel::ROW_*`). The public encoding is *not* stored — a
    /// row is expanded when a batch is made, so the stored bytes never go
    /// stale as the network changes.
    pub rows: Vec<u8>,
    /// `[total_configs, CCOUNTS]` raw counts per config, in the arena order.
    /// Raw rather than normalised: they are `u8`-valued, and storing them that
    /// way is what keeps a replay row small enough to hold millions of them.
    pub cc: Vec<u8>,
    /// `[total_configs]` belief probability of each config.
    pub cw: Vec<f32>,
    /// `[total_configs]` the solve's value for each config.
    pub cy: Vec<f32>,
    /// `[n]` the round each row was taken at, which is what the aux backfill
    /// needs and what it consumes. Not shipped to Python (the aux targets
    /// themselves live in the row).
    round: Vec<u16>,
    /// Games abandoned because the solve service refused their subgame. Zero
    /// in every healthy run; GPU capacity is not an algorithmic game filter.
    pub dropped: usize,
    /// Valid solves too large to coexist with the ordinary GPU wave lanes.
    /// They run through the serialized exact path and are never discarded.
    pub oversize_routes: usize,
    /// Oversized solves that drained and trimmed every lane on one card.
    pub card_exclusive_routes: usize,
    /// GPU submissions retried by the exact CPU solver after an unexpected
    /// device error. Oversize jobs normally complete on the serialized GPU path.
    pub exact_fallbacks: usize,
    /// Live games intentionally discarded at a wall-clock deadline. This is
    /// time-censored work, not a capacity drop and never enters replay.
    pub censored_games: usize,

    // ------------------------------------------------------- policy targets
    // One per solve, not one per row. A solve's rows share a public state and
    // differ only in belief, while the reference strategy is a single object,
    // so labelling every row would teach the head that the belief does not
    // matter. The exact label belongs to the live-belief row -- the one whose
    // belief is the solve's own root -- which `finish_solve` records.
    //
    // These are per epoch and never enter the replay buffer. A value target is
    // bootstrapped and gains from being averaged over a long history; a
    // strategy is not, and is regenerated in full every epoch.
    /// `[s]` the row each solve's label belongs to.
    pub prow: Vec<u32>,
    /// `[s]` which player was to act there.
    pub pact: Vec<u8>,
    /// `[sum na, AFEAT]` the action descriptions, and `[s + 1]` offsets in
    /// actions.
    pub pa: Vec<f32>,
    pub paoff: Vec<u32>,
    /// `[sum nc * na]` the reference strategy: one distribution per config of
    /// the acting player, in the row's config order.
    pub pp: Vec<f32>,
    /// `[2 * n + 1]` arena offsets.
    pub coff: Vec<u32>,
    /// Solve starts in row space: `soff[k]` is the row at which solve k
    /// starts (first entry 0). The Python binding appends the total row count
    /// as the trailing entry, so the buffer sees `soff[k]..soff[k+1]` as one
    /// solve's rows. TurboReBeL produces T+1 near-duplicate rows per solve,
    /// so the replay buffer treats the solve as its sampling unit — this
    /// array is what lets it. Empty when no rows were collected.
    pub soff: Vec<u32>,
    pub nv: usize,
    pub games: usize,
    pub decisions: usize,
    pub wins: [usize; 2],
    pub draws: usize,
    /// Completed games that reached `MAX_MAIN_PLAYS`. This is the game horizon,
    /// not the solver's tree-node cap.
    pub cap_hits: usize,
    /// Attempted subgame builds that hit `Cfg::node_cap` and used the uniform
    /// policy fallback instead of producing a solve.
    pub node_caps: usize,
    pub configs: usize,
    /// Seconds workers spent blocked on the GPU (idle CPU), summed.
    pub gpu_wait_s: f32,
    /// Seconds a builder spent blocked handing a finished solve to the merge
    /// thread. It is invisible in both CPU time and `gpu_wait_s`, so without
    /// it a saturated result path looks like a system with nothing saturated.
    pub merge_wait_s: f32,
}

impl Data {
    pub fn merge(&mut self, o: Data) {
        let base = self.cw.len() as u32;
        self.rows.extend(o.rows);
        let (row_base, act_base) = (self.nv as u32, (self.pa.len() / AFEAT) as u32);
        self.prow.extend(o.prow.iter().map(|r| r + row_base));
        self.pact.extend(o.pact);
        self.pa.extend(o.pa);
        // Same leading-zero rule as `coff`: the merged arena keeps exactly one.
        let phead = if self.paoff.is_empty() { 0 } else { 1 };
        self.paoff
            .extend(o.paoff.iter().skip(phead).map(|x| x + act_base));
        self.pp.extend(o.pp);
        self.cc.extend(o.cc);
        self.cw.extend(o.cw);
        self.cy.extend(o.cy);
        // Both sides carry a leading zero; the merged arena has exactly one, so
        // the other's is dropped. `coff` must stay `2 * nv + 1` long or every
        // row after the join is read with somebody else's configs.
        let tail = if self.coff.is_empty() { 0 } else { 1 };
        self.coff.extend(o.coff.iter().skip(tail).map(|x| x + base));
        let rb = self.nv as u32;
        self.soff.extend(o.soff.iter().map(|x| x + rb));
        self.nv += o.nv;
        self.games += o.games;
        self.decisions += o.decisions;
        self.dropped += o.dropped;
        self.oversize_routes += o.oversize_routes;
        self.card_exclusive_routes += o.card_exclusive_routes;
        self.exact_fallbacks += o.exact_fallbacks;
        self.censored_games += o.censored_games;
        self.wins[0] += o.wins[0];
        self.wins[1] += o.wins[1];
        self.draws += o.draws;
        self.gpu_wait_s += o.gpu_wait_s;
        self.merge_wait_s += o.merge_wait_s;
        self.cap_hits += o.cap_hits;
        self.node_caps += o.node_caps;
        self.configs += o.configs;
    }

    /// Mark the start of a solve's rows. One call per solve, before its rows
    /// are pushed; `soff[k]` is the row index where solve k starts.
    pub fn begin_solve(&mut self) {
        self.soff.push(self.nv as u32);
    }

    /// `y[p]` holds one value per *config* in `bel[p]`. Every one of them is
    /// stored: the value function is a function of the config, so there is
    /// nothing to average away.
    fn push_value(&mut self, s: &State, ctx: &Ctx, bel: &[Belief; 2], y: [&[f32]; 2]) {
        debug_assert!(
            matches!(s.pending(), Cont::MainPlay),
            "every saved value row is a normal coin-play state"
        );
        let base = self.rows.len();
        self.rows.resize(base + ROW_BYTES, 0);
        pack_row(s, ctx, &mut self.rows[base..base + ROW_BYTES]);
        self.round.push(s.round);
        if self.coff.is_empty() {
            self.coff.push(0);
        }
        for p in 0..2 {
            let res = reserve(s, p as u8, ctx);
            let mut cnt = [0u8; CCOUNTS];
            for (ci, c) in bel[p].cfg.iter().enumerate() {
                config_counts(c, &res, &mut cnt);
                self.cc.extend_from_slice(&cnt);
                self.cw.push(bel[p].p[ci]);
                self.cy.push(y[p][ci]);
            }
            self.coff.push(self.cw.len() as u32);
        }
        self.nv += 1;
    }

    /// Record the policy label for the solve just pushed: the reference
    /// strategy at its root, and the descriptions of the actions it is over.
    /// `row` is the live-belief row, whose belief is the solve's own root.
    fn push_policy(&mut self, sv: &Solver, ctx: &Ctx, row: usize, player: u8) {
        debug_assert!(
            sv.root_mainplay,
            "a policy label is the strategy at a normal coin-play root"
        );
        let n = &sv.nodes[0];
        let (na, nc) = (n.na(), n.nc(player as usize));
        if na == 0 || nc == 0 {
            return;
        }
        if self.paoff.is_empty() {
            self.paoff.push(0);
        }
        let base = self.pa.len();
        self.pa.resize(base + na * AFEAT, 0.0);
        for a in 0..na {
            write_action_feats(
                &n.acts[a],
                ctx,
                player as usize,
                n.aslot[a],
                n.fdown[a],
                &mut self.pa[base + a * AFEAT..base + (a + 1) * AFEAT],
            );
        }
        self.paoff.push((self.pa.len() / AFEAT) as u32);
        for c in 0..nc {
            let base = self.pp.len();
            self.pp.resize(base + na, 0.0);
            for (cell, &p) in n.legal_row(c).zip(sv.average_strategy(0, c).iter()) {
                self.pp[base + n.legal_action[cell] as usize] = p;
            }
        }
        self.prow.push(row as u32);
        self.pact.push(player);
    }

    /// `push_policy` with an explicit reference strategy, for the GPU path
    /// where the solve ran on the device: `strat` is the downloaded flat
    /// reference strategy (`soff`-aligned legal-cell CSR).
    fn push_policy_strat(
        &mut self,
        tree: &WalkTree,
        ctx: &Ctx,
        row: usize,
        player: u8,
        strat: &[f32],
    ) {
        let ar = tree.action_range(0);
        let (na, nc) = (ar.len(), tree.supports[0][player as usize].len());
        if na == 0 || nc == 0 {
            return;
        }
        if self.paoff.is_empty() {
            self.paoff.push(0);
        }
        let base = self.pa.len();
        self.pa.resize(base + na * AFEAT, 0.0);
        for a in 0..na {
            let aa = ar.start + a;
            write_action_feats(
                &tree.actions[aa],
                ctx,
                player as usize,
                tree.aslot[aa],
                tree.fdown[aa],
                &mut self.pa[base + a * AFEAT..base + (a + 1) * AFEAT],
            );
        }
        self.paoff.push((self.pa.len() / AFEAT) as u32);
        for c in 0..nc {
            let base = self.pp.len();
            self.pp.resize(base + na, 0.0);
            for cell in tree.legal_row(0, c) {
                self.pp[base + tree.legal_action[cell] as usize] = strat[cell];
            }
        }
        self.prow.push(row as u32);
        self.pact.push(player);
    }

    /// The config range of row `r`, player `p`, in the arena.
    #[inline]
    pub fn row_span(&self, r: usize, p: usize) -> std::ops::Range<usize> {
        self.coff[2 * r + p] as usize..self.coff[2 * r + p + 1] as usize
    }
}

// ----------------------------------------------------------------- game loop

#[derive(Clone, Copy)]
pub struct GameCfg {
    pub agents: [Agent; 2],
    pub collect: Collect,
    /// Probability that a uniformly sampled player plays a uniformly random
    /// action (ReBeL's `random_action_prob`), redrawn each decision.
    pub explore: f32,
    /// Randomise the draft instead of using the fixed starter matchup.
    pub random_draft: bool,
    /// Warm start only: how much of the value target comes from the squashed
    /// handcrafted public-information evaluation rather than the realised game
    /// outcome. The outcome is unbiased but very noisy — most of a game's
    /// states cannot predict who wins — while the handcrafted eval is biased
    /// but dense and low-variance, which is what makes one-ply differences
    /// legible to CFR from the first game. Both are only a starting point:
    /// every ReBeL-phase target comes from real solves and real outcomes, so
    /// the bias washes out.
    pub eval_mix: f32,
}

// ----------------------------------------------------------------- game loop

/// A live ReBeL walk: the solver for the current subgame, the checkpoint
/// slot it was built with, and the tree node the game is currently at.
///
enum WalkState<'a> {
    Cpu(Solver<'a>),
    Gpu(WalkTree),
}

/// On the GPU path the actor retains only a compact `WalkTree`; the full CPU
/// solver is released immediately after packing. The strategy rows and every
/// possible exit belief arrive in the one solve completion.
struct Walk<'a> {
    tree: WalkState<'a>,
    slot: usize,
    node: usize,
    /// Draws taken so far inside the current collapsed chance node.
    drawn: u8,
    /// The downloaded sparse reference strategy (GPU path), `soff`-aligned. Empty
    /// on the CPU path, where `sv.average_strategy` is the source.
    strat: Vec<f32>,
    /// Snapshot beliefs for every possible exit (GPU path).
    carries: Option<crate::gpu::CarryStore>,
}

impl<'a> Walk<'a> {
    /// The reference strategy row for config `c` of node `node`.
    fn strategy(&self, node: usize, c: usize) -> &[f32] {
        match &self.tree {
            WalkState::Cpu(sv) => sv.average_strategy(node, c),
            WalkState::Gpu(tree) => &self.strat[tree.legal_row(node, c)],
        }
    }

    fn player(&self, node: usize) -> u8 {
        match &self.tree {
            WalkState::Cpu(sv) => sv.nodes[node].player,
            WalkState::Gpu(tree) => tree.node_player[node],
        }
    }

    fn is_leaf(&self, node: usize) -> bool {
        match &self.tree {
            WalkState::Cpu(sv) => sv.nodes[node].leaf,
            WalkState::Gpu(tree) => tree.is_leaf(node),
        }
    }

    fn is_chance(&self, node: usize) -> bool {
        match &self.tree {
            WalkState::Cpu(sv) => sv.nodes[node].chance,
            WalkState::Gpu(tree) => tree.is_chance(node),
        }
    }

    fn draw_steps(&self, node: usize) -> u8 {
        match &self.tree {
            WalkState::Cpu(sv) => sv.nodes[node].draw_steps,
            WalkState::Gpu(tree) => tree.draw_steps[node],
        }
    }

    fn first_child(&self, node: usize) -> usize {
        match &self.tree {
            WalkState::Cpu(sv) => sv.nodes[node].child[0],
            WalkState::Gpu(tree) => tree.children(node)[0] as usize,
        }
    }

    fn child_for_action(&self, node: usize, action: usize) -> usize {
        match &self.tree {
            WalkState::Cpu(sv) => sv.nodes[node].child[sv.nodes[node].obs_child[action]],
            WalkState::Gpu(tree) => tree.child_for_action(node, action),
        }
    }

    fn support(&self, node: usize, player: usize) -> &[Config] {
        match &self.tree {
            WalkState::Cpu(sv) => &sv.nodes[node].cfgs[player],
            WalkState::Gpu(tree) => &tree.supports[node][player],
        }
    }

    fn policy(&self, node: usize) -> NodePolicy {
        match &self.tree {
            WalkState::Cpu(sv) => {
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
            WalkState::Gpu(tree) => {
                let ar = tree.action_range(node);
                let row0 = tree.legal_row_of[node] as usize;
                let nc = tree.supports[node][tree.node_player[node] as usize].len();
                let cell0 = tree.legal_off[row0];
                let cell1 = tree.legal_off[row0 + nc];
                NodePolicy {
                    acts: tree.actions[ar.clone()].to_vec(),
                    aslot: tree.aslot[ar.clone()].to_vec(),
                    fdown: tree.fdown[ar].to_vec(),
                    legal_off: tree.legal_off[row0..=row0 + nc]
                        .iter()
                        .map(|&x| x - cell0)
                        .collect(),
                    legal_action: tree.legal_action[cell0 as usize..cell1 as usize].to_vec(),
                    probs: vec![0.0; (cell1 - cell0) as usize],
                }
            }
        }
    }
}

/// End a walk: take TurboReBeL's intermediate PBSs off the solver — the
/// beliefs at the walk's current node under each per-iterate average strategy
/// (t = 0..T-1), from the subgame's root belief. The caller appends the live
/// belief as the t = T member; the next subgame's Phase 2 values the whole
/// set. Rows are *not* taken here: they were pushed at the solve site, which
/// is where the reference strategy lives.
///
/// On the GPU path the completion has already streamed every exit's normalised
/// reaches into a pageable `CarryStore`; selecting the actual exit is local.
fn finish_walk<'a>(w: Walk<'a>, bel: &[Belief; 2]) -> Vec<[Vec<f32>; 2]> {
    let Walk {
        tree,
        node,
        carries,
        ..
    } = w;
    let mut out = match tree {
        WalkState::Gpu(_) => carries
            .expect("GPU walk without carry store")
            .select(node as u32)
            .expect("carry-store exit"),
        WalkState::Cpu(mut sv) => sv.carried_beliefs(node),
    };
    out.push([bel[0].p.clone(), bel[1].p.clone()]);
    out
}

/// One game, as a resumable state machine.
///
/// `advance` plays until the next GPU solve is submitted (returning
/// `Step::Submitted`) or the game ends (`Step::Ended`, after which `finish`
/// returns the result). On the CPU path (`gpu` is `None`) the solve runs
/// inline and `advance` never returns `Submitted`. Two games per worker share
/// a thread: while one is blocked on the GPU, the other keeps the CPU busy.
pub struct Game<'a> {
    rng: Rng,
    s: State,
    ctx: Ctx,
    bel: [Belief; 2],
    walk: Option<Walk<'a>>,
    /// TurboReBeL's carried beliefs: the T+1 probability vectors at the next
    /// subgame's root under the previous solve's per-iterate average
    /// strategies (plus the live belief), which Phase 2 values once the next
    /// solve is done. Empty means the first level.
    carried: Vec<[Vec<f32>; 2]>,
    /// One entry per round, for the auxiliary targets.
    timeline: Timeline,
    /// The rows this game produced; merged into the caller's `Data` by
    /// `play_game` or the GPU worker when the game ends.
    data: Data,
    from_row: usize,
    gc: &'a GameCfg,
    /// Subgame roots for the tree-sizing work (None in production).
    roots: Option<Vec<(State, [Belief; 2])>>,
    /// The GPU job this game wants solved, and the state to resume with.
    /// The worker owns the actual submission, so it can tag replies onto one
    /// channel and resume whichever of its games answers first.
    pending_job: Option<crate::serialize::PackedJob>,
    pending_walk: Option<WalkTree>,
    pending_roots: Option<Vec<[Vec<f32>; 2]>>,
    pending_slot: usize,
    pending_player: u8,
    pending_oversize: bool,
}

/// What `Game::advance` returns.
pub enum Step {
    /// A solve was submitted to the GPU; call `take_pending` and wait.
    Submitted,
    /// The game ended; call `finish` for the result.
    Ended,
}

impl<'a> Game<'a> {
    pub fn new(mut rng: Rng, gc: &'a GameCfg) -> Game<'a> {
        let s = make_game(&mut rng, gc.random_draft);
        let ctx = Ctx::new(&s);
        Game {
            rng,
            s,
            ctx,
            bel: [
                Belief::point(Config::default()),
                Belief::point(Config::default()),
            ],
            walk: None,
            carried: Vec::new(),
            timeline: Vec::new(),
            data: Data::default(),
            from_row: 0,
            gc,
            roots: None,
            pending_job: None,
            pending_walk: None,
            pending_roots: None,
            pending_slot: 0,
            pending_player: 0,
            pending_oversize: false,
        }
    }

    pub fn set_roots(&mut self, r: Vec<(State, [Belief; 2])>) {
        self.roots = Some(r);
    }

    pub fn take_roots(&mut self) -> Vec<(State, [Belief; 2])> {
        self.roots.take().unwrap_or_default()
    }

    /// Take the pending job (after `Step::Submitted`); the worker submits it.
    pub fn take_job(&mut self) -> Option<crate::serialize::PackedJob> {
        self.pending_job.take()
    }

    /// The nets slot of the pending solve, for routing to a service.
    pub fn pending_slot(&self) -> usize {
        self.pending_slot
    }

    /// The rows produced so far (the worker takes them when a game ends).
    pub fn take_data(&mut self) -> Data {
        std::mem::take(&mut self.data)
    }

    /// True when the game has ended (used by the worker's idle check).
    pub fn is_terminal(&self) -> bool {
        self.s.is_terminal()
    }

    /// Play until a GPU solve is submitted or the game ends.
    pub fn advance(&mut self, gpu: Option<&[GpuClient]>, nets: &'a [Nets]) -> Step {
        let gc = self.gc;
        let Game {
            rng,
            s,
            ctx,
            bel,
            walk,
            carried,
            timeline,
            data,
            roots,
            ..
        } = self;
        let mut roots_out: Option<&mut Vec<(State, [Belief; 2])>> = roots.as_mut();
        while !s.is_terminal() {
            while timeline.len() <= s.round as usize {
                timeline.push(([s.markers_on_board(0), s.markers_on_board(1)], s.initiative));
            }
            let last = timeline.len() - 1;
            timeline[last] = ([s.markers_on_board(0), s.markers_on_board(1)], s.initiative);
            let player = s.to_act();
            if s.is_chance() {
                let res = reserve(s, player, ctx);
                let fu = faceup_counts(s, player, ctx);
                let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
                bel[player as usize] = belief_after_draw(&bel[player as usize], &res, &fu, wp);
                resolve_chance(s, player, rng);
                // The walk spans draws now: a draw is an internal node of the
                // subgame with one public child, so advance through it. The
                // post-draw belief must equal the tree's post-draw config
                // support (same list, same order) or every strategy row read
                // from here on is wrong.
                let mut walk_ended = false;
                if let Some(w) = walk.as_mut() {
                    let nid = w.node;
                    assert!(
                        w.is_chance(nid) && w.player(nid) == player,
                        "walk not at the draw"
                    );
                    // One tree node stands for a whole run of that player's
                    // draws, so it is exhausted only once the game has taken
                    // all of them.
                    w.drawn += 1;
                    if w.drawn == w.draw_steps(nid) {
                        w.drawn = 0;
                        let child = w.first_child(nid);
                        assert!(
                            w.support(child, player as usize)
                                == bel[player as usize].cfg.as_slice(),
                            "walk desync: post-draw support does not match the game belief"
                        );
                        w.node = child;
                        if w.is_leaf(child) {
                            walk_ended = true;
                        }
                    }
                }
                if walk_ended {
                    if gc.collect == Collect::Rebel {
                        *carried = finish_walk(walk.take().unwrap(), bel);
                    } else {
                        // Evaluation: the carried beliefs exist only to be
                        // valued by the next solve's Phase 2, which collection
                        // runs and evaluation does not — drop them with the
                        // walk.
                        walk.take();
                        carried.clear();
                    }
                }
                continue;
            }

            let cfgs = bel[player as usize].cfg.clone();
            let truth = true_config(s, player, ctx);
            let true_ci = bel[player as usize]
                .index_of(&truth)
                // Losing the real world would silently corrupt every target
                // taken from here on, so fail loudly instead.
                .expect("belief filter dropped the true config");
            data.decisions += 1;
            data.configs += cfgs.len();

            let np = match gc.agents[player as usize] {
                Agent::Greedy { temp } => {
                    // A non-ReBeL decision is not in the walk's tree: end any
                    // pending walk and drop the carried beliefs (they belong
                    // to the state the walk ended at).
                    walk.take();
                    carried.clear();
                    greedy_policy(s, ctx, player, &cfgs, temp)
                }
                Agent::Random => {
                    walk.take();
                    carried.clear();
                    random_policy(s, ctx, player, &cfgs)
                }
                Agent::Rebel { cfg, slot } => {
                    // A walk belongs to the checkpoint that built it. Playing
                    // a decision on another slot's solver would make that
                    // player act with the wrong network — the Elo ladder pits
                    // one snapshot's slot against another's — so end a walk
                    // built by a different slot before starting a new one.
                    if walk.as_ref().is_some_and(|w| w.slot != slot) {
                        walk.take();
                        carried.clear();
                    }
                    // A pathological root falls back to a uniform policy; the
                    // policy is produced outside the walk bookkeeping below.
                    let mut fallback: Option<NodePolicy> = None;
                    if walk.is_none() {
                        if let Some(r) = roots_out.as_deref_mut() {
                            r.push((s.clone(), bel.clone()));
                        }
                        // Start a new subgame at this decision: build the
                        // tree (Phase 1's CPU half). TurboReBeL's Phase 2 then
                        // values every carried belief — the T+1 rows of this
                        // level — under the reference strategy (the CFR
                        // average), and the walk acts on that same average.
                        let scfg = Cfg {
                            // Evaluation never reads the per-iterate
                            // snapshots or the carried beliefs, so it must
                            // not pay for either.
                            snapshots: gc.collect == Collect::Rebel,
                            // On the GPU path the device builds its own
                            // arenas; allocating them here too is waste.
                            gpu_build: gpu.is_some(),
                            ..cfg
                        };
                        let mut sv = Solver::new(s, *ctx, &nets[slot], scfg, bel.clone());
                        if sv.capped() {
                            // The tree-size tail is fat (broad random-draft
                            // beliefs at round boundaries); an unbounded build
                            // hangs a worker for minutes on one decision.
                            // Fall back to a uniform policy for this decision
                            // and drop the walk (and any carried beliefs); the
                            // next Rebel decision starts a fresh solve. No
                            // rows are collected here, so the data keeps the
                            // MainPlay-only invariant without a search that
                            // never ends.
                            walk.take();
                            carried.clear();
                            data.node_caps += 1;
                            fallback = Some(random_policy(s, ctx, player, &cfgs));
                        } else if gpu.is_some() {
                            // GPU path: package the tree as one job. The
                            // carried roots (or the live belief, for the
                            // first level) travel with it; trip 1 returns
                            // the reference strategy and the root values.
                            let roots_v: Vec<[Vec<f32>; 2]> = if carried.is_empty() {
                                vec![[bel[0].p.clone(), bel[1].p.clone()]]
                            } else {
                                std::mem::take(carried)
                            };
                            let (job, walk_tree) = PackedJob::from_solver_with_walk(&sv, &roots_v);
                            self.pending_oversize = job.work().requires_exclusive_route();
                            self.pending_job = Some(job);
                            self.pending_walk = Some(walk_tree);
                            self.pending_roots = Some(roots_v);
                            self.pending_slot = slot;
                            self.pending_player = player;
                            return Step::Submitted;
                        } else {
                            // CPU path: the full solve, then the walk.
                            sv.warm_start(scfg.warm);
                            sv.multistep(cfg.iters);
                            if gc.collect == Collect::Rebel {
                                // Phase 2: one fixed-policy value pass per
                                // carried belief. The first level carries
                                // nothing yet, so it values just the live
                                // belief — the same single row the old loop
                                // took per solve.
                                let roots_v: Vec<[Vec<f32>; 2]> = if carried.is_empty() {
                                    vec![[bel[0].p.clone(), bel[1].p.clone()]]
                                } else {
                                    std::mem::take(carried)
                                };
                                data.begin_solve();
                                for r in &roots_v {
                                    assert_eq!(r[0].len(), bel[0].cfg.len(),
                                               "carried belief does not match the root support: {} vs {} at dec {}",
                                               r[0].len(), bel[0].cfg.len(), data.decisions);
                                    assert_eq!(r[1].len(), bel[1].cfg.len(),
                                               "carried belief does not match the root support: {} vs {} at dec {}",
                                               r[1].len(), bel[1].cfg.len(), data.decisions);
                                }
                                let vals = sv.value_under(&roots_v);
                                for (r, v) in roots_v.iter().zip(vals.iter()) {
                                    assert_eq!(
                                        r[0].len(),
                                        bel[0].cfg.len(),
                                        "carried belief does not match the root support"
                                    );
                                    assert_eq!(
                                        r[1].len(),
                                        bel[1].cfg.len(),
                                        "carried belief does not match the root support"
                                    );
                                    data.push_value(
                                        s,
                                        ctx,
                                        &[
                                            Belief {
                                                cfg: bel[0].cfg.clone(),
                                                p: r[0].clone(),
                                            },
                                            Belief {
                                                cfg: bel[1].cfg.clone(),
                                                p: r[1].clone(),
                                            },
                                        ],
                                        [&v[0], &v[1]],
                                    );
                                }
                                // `roots` ends with the live belief —
                                // `finish_walk` appends it last, and the first
                                // level carries only it — so the row just
                                // pushed is the one whose belief is this
                                // solve's own root, and the only one the
                                // reference strategy is the exact answer for.
                                data.push_policy(&sv, ctx, data.nv - 1, player);
                            }
                            *walk = Some(Walk {
                                tree: WalkState::Cpu(sv),
                                slot,
                                node: 0,
                                drawn: 0,
                                strat: Vec::new(),
                                carries: None,
                            });
                        }
                    }
                    if let Some(np) = fallback {
                        np
                    } else {
                        let w = walk.as_mut().unwrap();
                        let nid = w.node;
                        // The tree was built from the belief at the subgame
                        // root and advanced in lockstep with the Bayes
                        // filter: the acting player's config support must be
                        // the same list *in order*, because the strategy rows
                        // are indexed by it. A silent desync would read the
                        // wrong row for the true config and corrupt every
                        // target from here on, so fail loudly.
                        assert!(
                            w.player(nid) == player
                                && w.support(nid, player as usize)
                                    == bel[player as usize].cfg.as_slice(),
                            "walk desync: subgame tree no longer matches the game belief"
                        );
                        let mut np = w.policy(nid);
                        for ci in 0..cfgs.len() {
                            // Act on the CFR average — the reference strategy
                            // of the solve. Evaluation and generation are the
                            // same walk now.
                            let row = w.strategy(nid, ci);
                            let cells = np.row(ci);
                            np.probs[cells].copy_from_slice(row);
                        }
                        np
                    }
                }
            };

            if gc.collect == Collect::Mc && matches!(s.pending(), Cont::MainPlay) {
                // Park the handcrafted evaluation in the target now; the
                // realised outcome is blended in once the game ends.
                // `eval_static` is exactly antisymmetric, so this stays
                // zero-sum. Rows are taken only at MainPlay states, like
                // every other training row.
                let e = eval_squashed(s, 0);
                let (a, b) = (vec![e; bel[0].len()], vec![-e; bel[1].len()]);
                data.begin_solve();
                data.push_value(s, ctx, bel, [&a, &b]);
            }

            let true_row = np.row(true_ci);
            let mut chosen_cell = true_row.start + sample_row(rng, &np.probs[true_row.clone()]);
            if gc.explore > 0.0
                && player as u64 == (rng.next_u64() & 1)
                && rng.unit_f64() < gc.explore as f64
            {
                if !true_row.is_empty() {
                    chosen_cell = true_row.start + rng.below(true_row.len());
                }
            }
            let chosen = np.legal_action[chosen_cell] as usize;

            // Bayes update on the *public observation*: several private
            // actions can produce it, and the belief must sum over all of
            // them.
            let obs = obs_key(&np.acts[chosen]);
            let mut pairs: Vec<(Config, f32)> = Vec::new();
            for (ci, c) in cfgs.iter().enumerate() {
                for cell in np.row(ci) {
                    let a = np.legal_action[cell] as usize;
                    if obs_key(&np.acts[a]) != obs {
                        continue;
                    }
                    if let Some(n) = advance_config(c, np.aslot[a], np.fdown[a]) {
                        pairs.push((n, bel[player as usize].p[ci] * np.probs[cell]));
                    }
                }
            }
            bel[player as usize] = Belief::from_pairs(pairs);
            s.apply_inplace(np.acts[chosen]);

            // Advance the walk along the solved tree. The public observation
            // of the chosen action selects the child; if that child is a leaf
            // (depth exhausted, terminal, or a draw), the walk ends and the
            // next subgame takes over — with this solve's carried beliefs.
            let mut walk_ended = false;
            if let Some(w) = walk.as_mut() {
                let nid = w.node;
                let child = w.child_for_action(nid, chosen);
                // Advance regardless: if the child is a leaf, the walk ends
                // *at* it, and the carried beliefs must be read off that
                // node's reach.
                w.node = child;
                if w.is_leaf(child) {
                    walk_ended = true;
                }
            }
            if walk_ended {
                if gc.collect == Collect::Rebel {
                    *carried = finish_walk(walk.take().unwrap(), bel);
                } else {
                    walk.take();
                    carried.clear();
                }
            }
        }
        Step::Ended
    }

    /// Resume after a wave completion: push the Phase-2 rows and start the
    /// walk with the downloaded reference strategy and carry store.
    pub fn resume(&mut self, result: crate::gpu::SolveResult) {
        let gc = self.gc;
        let tree = self.pending_walk.take().expect("pending walk tree");
        let slot = self.pending_slot;
        let player = self.pending_player;
        let roots_v = self.pending_roots.take().expect("pending roots");
        self.pending_oversize = false;
        if gc.collect == Collect::Rebel {
            self.data.begin_solve();
            for (r, v) in roots_v.iter().zip(result.root_values.iter()) {
                self.data.push_value(
                    &self.s,
                    &self.ctx,
                    &[
                        Belief {
                            cfg: self.bel[0].cfg.clone(),
                            p: r[0].clone(),
                        },
                        Belief {
                            cfg: self.bel[1].cfg.clone(),
                            p: r[1].clone(),
                        },
                    ],
                    [&v[0], &v[1]],
                );
            }
            // The policy label comes from the downloaded reference strategy.
            self.data.push_policy_strat(
                &tree,
                &self.ctx,
                self.data.nv - 1,
                player,
                &result.strategy,
            );
        }
        self.carried = Vec::new();
        self.walk = Some(Walk {
            tree: WalkState::Gpu(tree),
            slot,
            node: 0,
            drawn: 0,
            strat: result.strategy,
            carries: Some(result.carries),
        });
        self.data.oversize_routes += result.oversize_route as usize;
        self.data.card_exclusive_routes += result.card_exclusive_route as usize;
    }

    /// Rebuild and solve a pending GPU job on the verified CPU path. This is
    /// deliberately serialized by the caller: an oversize solve is rare, but
    /// allowing several of its multi-gigabyte arenas at once would merely move
    /// the capacity failure from device memory to host memory.
    #[cfg(feature = "gpu")]
    pub fn retry_cpu(&mut self, nets: &'a [Nets]) {
        // Release the packed GPU representation before allocating the CPU CFR
        // arenas. The worker has normally taken `pending_job` already; keep the
        // take here so direct callers cannot accidentally retain it.
        self.pending_job.take();
        self.pending_walk.take().expect("pending walk tree");
        let roots_v = self.pending_roots.take().expect("pending roots");
        let oversize = std::mem::take(&mut self.pending_oversize);
        let player = self.pending_player;
        let Agent::Rebel { cfg, slot } = self.gc.agents[player as usize] else {
            panic!("GPU retry requested for a non-ReBeL agent");
        };
        let scfg = Cfg {
            snapshots: self.gc.collect == Collect::Rebel,
            gpu_build: false,
            ..cfg
        };
        let mut sv = Solver::new(&self.s, self.ctx, &nets[slot], scfg, self.bel.clone());
        assert!(
            !sv.capped(),
            "a GPU job that passed the node cap capped on its exact CPU retry"
        );
        sv.warm_start(scfg.warm);
        sv.multistep(cfg.iters);
        if self.gc.collect == Collect::Rebel {
            self.data.begin_solve();
            let vals = sv.value_under(&roots_v);
            for (r, v) in roots_v.iter().zip(&vals) {
                self.data.push_value(
                    &self.s,
                    &self.ctx,
                    &[
                        Belief {
                            cfg: self.bel[0].cfg.clone(),
                            p: r[0].clone(),
                        },
                        Belief {
                            cfg: self.bel[1].cfg.clone(),
                            p: r[1].clone(),
                        },
                    ],
                    [&v[0], &v[1]],
                );
            }
            self.data
                .push_policy(&sv, &self.ctx, self.data.nv - 1, player);
        }
        self.carried.clear();
        self.walk = Some(Walk {
            tree: WalkState::Cpu(sv),
            slot,
            node: 0,
            drawn: 0,
            strat: Vec::new(),
            carries: None,
        });
        self.data.exact_fallbacks += 1;
        self.data.oversize_routes += oversize as usize;
    }

    /// The game ended: blend the outcome into the parked targets, fill the
    /// aux heads, and return the result from White's point of view.
    pub fn finish(&mut self) -> f32 {
        let z = self.s.utility(WHITE as usize);
        if self.gc.collect == Collect::Mc {
            // The parked value is the handcrafted evaluation; blend in the
            // realised outcome. This is the warm start only — ReBeL-phase
            // targets come entirely from the subgame solve.
            let m = self.gc.eval_mix.clamp(0.0, 1.0);
            blend_outcome(&mut self.data, self.from_row, m, 1.0 - m, z);
        }
        fill_aux(&mut self.data, self.from_row, &self.timeline, z);
        self.data.games += 1;
        if self.s.main_plays >= crate::state::MAX_MAIN_PLAYS {
            self.data.cap_hits += 1;
        }
        match self.s.winner() {
            Some(w) => self.data.wins[w as usize] += 1,
            None => self.data.draws += 1,
        }
        z
    }
}

/// Play one game to the end. Returns the result from White's point of view.
///
/// When `roots` is given, every subgame root — the public state and both
/// beliefs at each ReBeL decision — is appended to it. The GPU tree-sizing
/// work collects these during a training run; they cost a clone per solve
/// site and nothing when `None`.
pub fn play_game(
    rng: Rng,
    nets: &[Nets],
    gc: &GameCfg,
    data: &mut Data,
    mut roots: Option<&mut Vec<(State, [Belief; 2])>>,
) -> f32 {
    let mut g = Game::new(rng, gc);
    if roots.is_some() {
        g.set_roots(Vec::new());
    }
    let z = loop {
        match g.advance(None, nets) {
            Step::Ended => break g.finish(),
            Step::Submitted => unreachable!("cpu path never submits"),
        }
    };
    if let Some(r) = roots.as_deref_mut() {
        r.extend(g.take_roots());
    }
    let d = g.take_data();
    data.merge(d);
    z
}
/// How many auxiliary targets a row carries. Defined with the frozen row
/// format; see `rebel::AUX`.
pub use crate::rebel::AUX;
/// How far ahead the marker target looks.
const AUX_ROUNDS: u16 = 3;

/// One entry per round the game reached: each player's markers on the board,
/// and who held the initiative. The aux targets are read off this.
type Timeline = Vec<([u8; 2], u8)>;

/// Fill in the auxiliary targets now that the game is over and the future they
/// ask about has happened. A row taken in round `r` wants the board three rounds
/// later; a game that ends first is asked about its final position instead,
/// which is the true answer to "how many markers are down later on".
/// Round-to-nearest-even f32 -> IEEE-754 binary16 bit pattern. The aux
/// targets are stored as float16 in the frozen row; numpy reads them back
/// with the same rounding.
pub(crate) fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        // Inf / NaN: keep the top bits (never produced by the aux targets).
        return (sign as u32 | 0x7c00 | man >> 13) as u16;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return (sign as u32 | 0x7c00) as u16; // overflow to inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow to zero
        }
        let m = man | 0x800_000;
        let shift = 14 - e;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let round = if rem > (1 << (shift - 1)) || (rem == (1 << (shift - 1)) && (half & 1) == 1) {
            half + 1
        } else {
            half
        };
        return (sign as u32 | round) as u16;
    }
    // Round-to-nearest-even; when the mantissa carries into the exponent the
    // f16 value must become the next binade, not wrap to a subnormal.
    let mut half = (man >> 13) as u32;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
        if half == 0x400 {
            return (sign as u32 | ((e as u32 + 1) << 10)) as u16;
        }
    }
    (sign as u32 | ((e as u32) << 10) | half) as u16
}

fn fill_aux(data: &mut Data, from_row: usize, tl: &Timeline, z: f32) {
    if tl.is_empty() {
        return;
    }
    let result = if z > 0.0 {
        0.0
    } else if z < 0.0 {
        2.0
    } else {
        1.0
    };
    for r in from_row..data.nv {
        let now = (data.round[r] as usize).min(tl.len() - 1);
        let then = (now + AUX_ROUNDS as usize).min(tl.len() - 1);
        let at = r * ROW_BYTES + ROW_AUX;
        let row = &mut data.rows[at..at + 2 * AUX];
        let vals = [
            tl[then].0[0] as f32 / 6.0,
            tl[then].0[1] as f32 / 6.0,
            // Does the initiative change hands by the start of the next round?
            (tl[(now + 1).min(tl.len() - 1)].1 != tl[now].1) as u8 as f32,
            result,
        ];
        for (k, v) in vals.iter().enumerate() {
            row[2 * k..2 * k + 2].copy_from_slice(&f32_to_f16(*v).to_le_bytes());
        }
    }
}

/// `y <- keep * y + mix * (+-z)` over every config of every row this game
/// produced. The sign flips for player 1: `z` is White's outcome and the
/// targets are per-player utilities of a zero-sum game.
fn blend_outcome(data: &mut Data, from_row: usize, keep: f32, mix: f32, z: f32) {
    for r in from_row..data.nv {
        for p in 0..2 {
            let sign = if p == 0 { 1.0 } else { -1.0 };
            for i in data.row_span(r, p) {
                data.cy[i] = keep * data.cy[i] + mix * sign * z;
            }
        }
    }
}

fn resolve_chance(s: &mut State, player: u8, rng: &mut Rng) {
    debug_assert!(matches!(
        s.pending(),
        Cont::Draw { .. } | Cont::WarriorPriestDraw { .. }
    ));
    let acts = s.legal_actions();
    let mut w: Vec<f64> = Vec::with_capacity(acts.len());
    let mut any = false;
    for a in &acts {
        let c = match a {
            Action::DrawCoin { unit } if *unit != NONE => effective_bag_count(s, player, *unit),
            _ => 1,
        };
        any |= c > 0;
        w.push(c as f64);
    }
    if !any {
        w.iter_mut().for_each(|x| *x = 1.0);
    }
    let ai = rng.weighted_index(&w);
    s.apply_inplace(acts[ai]);
}

pub(crate) fn effective_bag_count(s: &State, p: u8, unit: u8) -> u8 {
    let bag_total: u8 = s.zones[p as usize][Z_BAG].iter().sum();
    if bag_total > 0 {
        s.zones[p as usize][Z_BAG][unit as usize]
    } else {
        s.zones[p as usize][Z_FACEUP][unit as usize]
            + s.zones[p as usize][Z_FACEDOWN][unit as usize]
    }
}

fn sample_row(rng: &mut Rng, row: &[f32]) -> usize {
    let w: Vec<f64> = row.iter().map(|&x| x.max(0.0) as f64).collect();
    if w.iter().sum::<f64>() > 0.0 {
        rng.weighted_index(&w)
    } else {
        rng.below(row.len().max(1))
    }
}

// ------------------------------------------------------------- batch drivers

fn worker_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) ^ (i as u64).wrapping_mul(0xD1B54A32D192ED03)
}

/// Collect subgame roots from `games` random-draft games: `(state, belief)`
/// pairs at every solve site, for GPU tree sizing.
pub fn collect_roots(
    games: usize,
    seed: u64,
    nets: &[Nets],
    gc: &GameCfg,
    cap: usize,
) -> Vec<(State, [Belief; 2])> {
    let mut out: Vec<(State, [Belief; 2])> = Vec::new();
    for i in 0..games {
        if out.len() >= cap {
            break;
        }
        let rng = Rng::new(worker_seed(seed, i));
        let mut d = Data::default();
        let mut roots: Vec<(State, [Belief; 2])> = Vec::new();
        play_game(rng, nets, gc, &mut d, Some(&mut roots));
        for r in roots {
            if out.len() >= cap {
                break;
            }
            out.push(r);
        }
    }
    out
}

/// Play `games` games in parallel, returning merged data and statistics.

/// The GPU generation loop.
///
/// Games each generation worker keeps in flight. Two is enough to hide one
/// CPU tree build behind one GPU solve; more is what fills the service's live
/// set when the service, not the CPU, is the bottleneck.
#[cfg(feature = "gpu")]
pub fn gen_workers_per() -> usize {
    std::env::var("WARCHEST_GEN_PER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
        .clamp(1, 64)
}

/// Worker count follows the box, not the game count (the old rule
/// `games / 2` left a 64-core machine at 24 workers when the trainer asked
/// for 48 games). Each worker interleaves a few games and resumes whichever
/// game's solve answers first: trip-1 replies are tagged onto one channel
/// per worker, so a slow solve never blocks a finished one behind it.
///
/// `gpus` is one client per service and `route` maps a solve's nets slot to
/// a service index: training routes everything to service 0, a GPU ladder
/// splits the two checkpoints between two devices.
#[cfg(feature = "gpu")]
pub fn run_games_gpu(
    games: usize,
    seed: u64,
    nets: &[Nets],
    gc: &GameCfg,
    gpus: &[crate::gpu::GpuClient],
    route: &(dyn Fn(usize) -> usize + Sync),
) -> Data {
    run_games_gpu_until(games, seed, nets, gc, gpus, route, None)
}

/// Deadline-aware form used by the streaming trainer. At the deadline no new
/// game or solve is admitted; already-submitted waves drain, and unfinished
/// games are reported as time-censored rather than silently counted as drops.
#[cfg(feature = "gpu")]
pub fn run_games_gpu_until(
    games: usize,
    seed: u64,
    nets: &[Nets],
    gc: &GameCfg,
    gpus: &[crate::gpu::GpuClient],
    route: &(dyn Fn(usize) -> usize + Sync),
    deadline: Option<std::time::Instant>,
) -> Data {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static EXACT_FALLBACK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Live games per worker: one solving on the GPU while another builds its
    // tree on the CPU. More than two is worth it whenever the service is the
    // bottleneck — the resident live set is `workers * per`, and a bigger live
    // set is what fills the tick's grids — so it is a knob, not a constant.
    let per = gen_workers_per();
    // Workers spend most of their time blocked on a solve, not running, so
    // the useful count is not the core count: it is whatever keeps the
    // service's live set full. `WARCHEST_GEN_WORKERS` overrides the
    // one-per-core default for exactly that reason.
    let workers = std::env::var("WARCHEST_GEN_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2).max(1))
                .unwrap_or(8)
        })
        .min(games.div_ceil(per).max(1));
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let (gc, nets, gpus, next, route) = (gc, nets, gpus, &next, route);
                scope.spawn(move || {
                    let mut out = Data::default();
                    let mut game: Vec<Option<Game>> = (0..per).map(|_| None).collect();
                    let mut busy = vec![false; per];
                    let (tx, rx) = std::sync::mpsc::channel();
                    let mut live = 0usize;
                    loop {
                        let expired = deadline.is_some_and(|x| std::time::Instant::now() >= x);
                        // Advance every idle game to its next solve (or its
                        // end), starting fresh games while any remain.
                        for k in 0..per {
                            if busy[k] {
                                continue;
                            }
                            if expired {
                                if game[k].take().is_some() {
                                    live -= 1;
                                    out.censored_games += 1;
                                }
                                continue;
                            }
                            loop {
                                if game[k].is_none() {
                                    let i = next.fetch_add(1, Ordering::Relaxed);
                                    if i >= games {
                                        break;
                                    }
                                    game[k] = Some(Game::new(Rng::new(worker_seed(seed, i)), gc));
                                    live += 1;
                                }
                                let g = game[k].as_mut().unwrap();
                                let step = {
                                    let _t = crate::timed!(ADVANCE);
                                    g.advance(Some(gpus), nets)
                                };
                                match step {
                                    Step::Submitted => {
                                        if deadline.is_some_and(|x| std::time::Instant::now() >= x)
                                        {
                                            game[k] = None;
                                            live -= 1;
                                            out.censored_games += 1;
                                            break;
                                        }
                                        let job = g.take_job().expect("submitted job");
                                        let dev = route(g.pending_slot()) % gpus.len();
                                        gpus[dev]
                                            .submit_tagged(job, k, tx.clone())
                                            .expect("gpu submit");
                                        busy[k] = true;
                                        break;
                                    }
                                    Step::Ended => {
                                        if deadline.is_some_and(|x| std::time::Instant::now() >= x)
                                        {
                                            game[k] = None;
                                            live -= 1;
                                            out.censored_games += 1;
                                            break;
                                        }
                                        let _ = g.finish();
                                        out.merge(g.take_data());
                                        game[k] = None;
                                        live -= 1;
                                    }
                                }
                            }
                        }
                        if live == 0 {
                            break;
                        }
                        if !busy.iter().any(|&b| b) {
                            continue;
                        }
                        // Resume whichever game answered first.
                        let t0 = std::time::Instant::now();
                        let (k, res) = {
                            let _t = crate::timed!(TRIP1);
                            rx.recv().expect("gpu trip 1")
                        };
                        out.gpu_wait_s += t0.elapsed().as_secs_f32();
                        busy[k] = false;
                        if deadline.is_some_and(|x| std::time::Instant::now() >= x) {
                            game[k] = None;
                            live -= 1;
                            out.censored_games += 1;
                            continue;
                        }
                        match res {
                            Ok(trip1) => game[k].as_mut().expect("pending game").resume(trip1),
                            Err(e) => {
                                eprintln!("gen: exact CPU retry after GPU error: {e}");
                                let _exclusive = EXACT_FALLBACK.lock().unwrap();
                                game[k].as_mut().expect("pending game").retry_cpu(nets);
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        let mut merged = Data::default();
        for h in handles {
            merged.merge(h.join().expect("gpu worker"));
        }
        merged
    })
}

/// Continuous ReBeL generation for the trainer. A fixed number of CPU builder
/// threads each owns many lightweight game actors; completed solves are
/// detached immediately and merged into bounded chunks while the actors keep
/// playing. This is valid because ReBeL targets are pure bootstrap: a value
/// target is final the moment its solve completes and never depends on how the
/// game later ended. The auxiliary heads do depend on that, which is why they
/// are not available on this path.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn run_games_gpu_stream(
    seed: u64,
    nets: &[Nets],
    gc: &GameCfg,
    gpus: &[crate::gpu::GpuClient],
    workers: usize,
    actors_per_worker: usize,
    inflight_per_worker: usize,
    chunk_solves: usize,
    stop: &std::sync::atomic::AtomicBool,
    output: std::sync::mpsc::SyncSender<Result<Option<Data>, String>>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    assert_eq!(gc.collect, Collect::Rebel);
    assert_eq!(
        gc.mc_mix, 0.0,
        "eager stream requires pure bootstrap targets"
    );
    let workers = workers.max(1);
    let per = actors_per_worker.max(1);
    let max_inflight = inflight_per_worker.max(1).min(per);
    let chunk_solves = chunk_solves.max(1);
    let next = AtomicUsize::new(0);
    let (data_tx, data_rx) = std::sync::mpsc::sync_channel(workers * 4);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let data_tx = data_tx.clone();
            let next = &next;
            scope.spawn(move || {
                let mut game: Vec<Option<Game>> = (0..per).map(|_| None).collect();
                let mut busy = vec![false; per];
                let (tx, rx) = std::sync::mpsc::channel();
                let mut live = 0usize;
                let mut inflight = 0usize;
                let mut cursor = 0usize;
                let mut merge_wait = 0.0f32;
                loop {
                    let stopping = stop.load(Ordering::Acquire);
                    if stopping {
                        for k in 0..per {
                            if busy[k] {
                                continue;
                            }
                            if let Some(mut g) = game[k].take() {
                                let mut d = g.take_data();
                                d.censored_games += 1;
                                let _ = data_tx.send(Ok(d));
                                live -= 1;
                            }
                        }
                    }
                    let mut scanned = 0usize;
                    while !stopping && scanned < per && inflight < max_inflight {
                        let k = cursor;
                        cursor = (cursor + 1) % per;
                        scanned += 1;
                        if busy[k] {
                            continue;
                        }
                        loop {
                            if game[k].is_none() {
                                let i = next.fetch_add(1, Ordering::Relaxed);
                                game[k] = Some(Game::new(Rng::new(worker_seed(seed, i)), gc));
                                live += 1;
                            }
                            let g = game[k].as_mut().expect("live stream game");
                            let step = {
                                let _t = crate::timed!(ADVANCE);
                                g.advance(Some(gpus), nets)
                            };
                            match step {
                                Step::Submitted => {
                                    if stop.load(Ordering::Acquire) {
                                        let mut g = game[k].take().unwrap();
                                        let mut d = g.take_data();
                                        d.censored_games += 1;
                                        let _ = data_tx.send(Ok(d));
                                        live -= 1;
                                        break;
                                    }
                                    let job = g.take_job().expect("submitted stream job");
                                    let dev = gpus
                                        .iter()
                                        .enumerate()
                                        .min_by_key(|(_, gpu)| gpu.queued_work())
                                        .map_or(0, |(i, _)| i);
                                    if let Err(e) = gpus[dev].submit_tagged(job, k, tx.clone()) {
                                        stop.store(true, Ordering::Release);
                                        let _ = data_tx.send(Err(e));
                                    } else {
                                        busy[k] = true;
                                        inflight += 1;
                                    }
                                    break;
                                }
                                Step::Ended => {
                                    let _ = g.finish();
                                    let mut d = g.take_data();
                                    d.merge_wait_s += std::mem::take(&mut merge_wait);
                                    if data_tx.send(Ok(d)).is_err() {
                                        stop.store(true, Ordering::Release);
                                    }
                                    game[k] = None;
                                    live -= 1;
                                }
                            }
                        }
                    }
                    if live == 0 && stop.load(Ordering::Acquire) {
                        break;
                    }
                    if !busy.iter().any(|&x| x) {
                        continue;
                    }
                    let waited = std::time::Instant::now();
                    let Ok((k, result)) = rx.recv() else {
                        stop.store(true, Ordering::Release);
                        let _ = data_tx.send(Err("GPU stream completion channel closed".into()));
                        continue;
                    };
                    busy[k] = false;
                    inflight -= 1;
                    if stop.load(Ordering::Acquire) {
                        match result {
                            Ok(value) => {
                                // Stop closes admission, not accounting. This
                                // solve completed while the final waves drained;
                                // keep its final bootstrap target, then censor
                                // only the unfinished public game around it.
                                let mut g = game[k].take().expect("draining stream game");
                                g.resume(value);
                                let mut d = g.take_data();
                                d.gpu_wait_s += waited.elapsed().as_secs_f32();
                                d.censored_games += 1;
                                let _ = data_tx.send(Ok(d));
                                live -= 1;
                            }
                            Err(e) => {
                                let _ = game[k].take();
                                live -= 1;
                                let _ = data_tx.send(Err(format!(
                                    "GPU stream solve failed while draining: {e}"
                                )));
                            }
                        }
                        continue;
                    }
                    match result {
                        Ok(value) => {
                            let g = game[k].as_mut().expect("pending stream game");
                            g.resume(value);
                            let mut d = g.take_data();
                            d.gpu_wait_s += waited.elapsed().as_secs_f32();
                            // Carries the previous handoff's block time: this
                            // one is not known until the send returns.
                            d.merge_wait_s += std::mem::take(&mut merge_wait);
                            let handoff = std::time::Instant::now();
                            let sent = data_tx.send(Ok(d));
                            merge_wait += handoff.elapsed().as_secs_f32();
                            if sent.is_err() {
                                stop.store(true, Ordering::Release);
                            }
                        }
                        Err(e) => {
                            stop.store(true, Ordering::Release);
                            let _ = data_tx.send(Err(format!("GPU stream solve failed: {e}")));
                        }
                    }
                }
            });
        }
        drop(data_tx);

        let mut chunk = Data::default();
        let mut failed = false;
        while let Ok(item) = data_rx.recv() {
            match item {
                Ok(d) if !failed => {
                    chunk.merge(d);
                    if chunk.soff.len() >= chunk_solves {
                        if output.send(Ok(Some(std::mem::take(&mut chunk)))).is_err() {
                            stop.store(true, Ordering::Release);
                            failed = true;
                        }
                    }
                }
                Ok(_) => {}
                Err(e) if !failed => {
                    failed = true;
                    stop.store(true, Ordering::Release);
                    let _ = output.send(Err(e));
                }
                Err(_) => {}
            }
        }
        if !failed
            && (chunk.soff.len() > 0
                || chunk.games > 0
                || chunk.censored_games > 0
                || chunk.decisions > 0)
        {
            let _ = output.send(Ok(Some(chunk)));
        }
        let _ = output.send(Ok(None));
    });
}

/// A GPU evaluation match: the same paired-seating scheme as `eval_match`,
/// as two batches (seats swapped in the second) over the same seed stream.
/// Weights live on the services; `a` and `b` route by their nets slots.
#[cfg(feature = "gpu")]
pub fn eval_match_gpu(
    games: usize,
    seed: u64,
    nets: &[Nets],
    a: Agent,
    b: Agent,
    random_draft: bool,
    gpus: &[crate::gpu::GpuClient],
) -> (usize, usize, usize) {
    let mk = |agents: [Agent; 2]| GameCfg {
        agents,
        collect: Collect::None,
        explore: 0.0,
        random_draft,
        eval_mix: 0.0,
    };
    // Route: side A's checkpoint sits on service 0, side B's on service 1.
    let slot_of = |ag: &Agent| match ag {
        Agent::Rebel { slot, .. } => *slot,
        _ => usize::MAX,
    };
    let (sa, _sb) = (slot_of(&a), slot_of(&b));
    let route = move |slot: usize| usize::from(slot != sa);
    let pairs = games / 2;
    let d1 = run_games_gpu(pairs, seed.wrapping_add(7), nets, &mk([a, b]), gpus, &route);
    let d2 = run_games_gpu(pairs, seed.wrapping_add(7), nets, &mk([b, a]), gpus, &route);
    (
        d1.wins[0] + d2.wins[1],
        d1.wins[1] + d2.wins[0],
        d1.draws + d2.draws,
    )
}

pub fn run_games(games: usize, seed: u64, nets: &[Nets], gc: &GameCfg) -> Data {
    (0..games)
        .into_par_iter()
        .fold(Data::default, |mut acc, i| {
            let rng = Rng::new(worker_seed(seed, i));
            play_game(rng, nets, gc, &mut acc, None);
            acc
        })
        .reduce(Data::default, |mut a, b| {
            a.merge(b);
            a
        })
}

/// Head-to-head match, colours alternating on paired seeds (same draft and the
/// same random stream for both seatings — a large variance reduction).
/// Returns `(wins_for_a, wins_for_b, draws)`.
pub fn eval_match(
    games: usize,
    seed: u64,
    nets: &[Nets],
    a: Agent,
    b: Agent,
    random_draft: bool,
) -> (usize, usize, usize) {
    (0..games)
        .into_par_iter()
        .map(|i| {
            let rng = Rng::new(worker_seed(seed.wrapping_add(7), i / 2));
            let swap = i % 2 == 1;
            let gc = GameCfg {
                agents: if swap { [b, a] } else { [a, b] },
                collect: Collect::None,
                explore: 0.0,
                random_draft,
                eval_mix: 0.0,
            };
            let mut d = Data::default();
            let z = play_game(rng, nets, &gc, &mut d, None);
            let za = if swap { -z } else { z };
            if za > 1e-6 {
                (1, 0, 0)
            } else if za < -1e-6 {
                (0, 1, 0)
            } else {
                (0, 0, 1)
            }
        })
        .reduce(|| (0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2))
}
