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
use crate::rebel::*;
use crate::rng::Rng;
use crate::gpu::GpuClient;
use crate::search::{node_actions, Cfg, Nets, Solver};
use crate::serialize::Job;
use crate::state::{Cont, State, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, BLACK, WHITE};
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
pub const DRAFT_POOL: [u16; 19] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54];

pub fn make_game(rng: &mut Rng, random: bool) -> State {
    let first = if rng.next_u64() & 1 == 0 { WHITE } else { BLACK };
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

/// A decision node's policy: private actions plus `P(action | config)` laid out
/// as `[config * na + action]`.
struct NodePolicy {
    acts: Vec<Action>,
    aslot: Vec<i8>,
    fdown: Vec<bool>,
    legal: Vec<bool>,
    probs: Vec<f32>,
}

impl NodePolicy {
    fn frame(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
        let (acts, aslot, fdown) = node_actions(s, player, ctx, cfgs);
        let na = acts.len();
        let mut legal = vec![false; cfgs.len() * na];
        for (ci, c) in cfgs.iter().enumerate() {
            for a in 0..na {
                legal[ci * na + a] = action_legal(c, aslot[a]);
            }
        }
        NodePolicy {
            acts,
            aslot,
            fdown,
            legal,
            probs: vec![0.0; cfgs.len() * na],
        }
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
        let mut best = f32::NEG_INFINITY;
        for a in 0..na {
            if np.legal[ci * na + a] {
                best = best.max(score[a]);
            }
        }
        let mut sum = 0.0;
        for a in 0..na {
            if np.legal[ci * na + a] {
                let e = (score[a] - best).exp();
                np.probs[ci * na + a] = e;
                sum += e;
            }
        }
        // A little uniform mass keeps the belief filter from collapsing and
        // keeps warm-start games diverse.
        let k = (0..na).filter(|&a| np.legal[ci * na + a]).count() as f32;
        for a in 0..na {
            if np.legal[ci * na + a] {
                np.probs[ci * na + a] = 0.95 * np.probs[ci * na + a] / sum + 0.05 / k;
            }
        }
    }
    np
}

fn random_policy(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
    let mut np = NodePolicy::frame(s, ctx, player, cfgs);
    let na = np.acts.len();
    for ci in 0..cfgs.len() {
        let k = (0..na).filter(|&a| np.legal[ci * na + a]).count() as f32;
        for a in 0..na {
            if np.legal[ci * na + a] {
                np.probs[ci * na + a] = 1.0 / k;
            }
        }
    }
    np
}

// -------------------------------------------------------------- data records

#[derive(Clone, Copy, PartialEq)]
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
    pub cap_hits: usize,
    pub configs: usize,
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
        self.coff
            .extend(o.coff.iter().skip(tail).map(|x| x + base));
        let rb = self.nv as u32;
        self.soff.extend(o.soff.iter().map(|x| x + rb));
        self.nv += o.nv;
        self.games += o.games;
        self.decisions += o.decisions;
        self.wins[0] += o.wins[0];
        self.wins[1] += o.wins[1];
        self.draws += o.draws;
        self.cap_hits += o.cap_hits;
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
            matches!(sv.nodes[0].s.pending(), Cont::MainPlay),
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
            write_action_feats(&n.acts[a], ctx, player as usize, n.aslot[a], n.fdown[a],
                               &mut self.pa[base + a * AFEAT..base + (a + 1) * AFEAT]);
        }
        self.paoff.push((self.pa.len() / AFEAT) as u32);
        for c in 0..nc {
            self.pp.extend_from_slice(sv.average_strategy(0, c));
        }
        self.prow.push(row as u32);
        self.pact.push(player);
    }

    /// `push_policy` with an explicit reference strategy, for the GPU path
    /// where the solve ran on the device: `strat` is the downloaded flat
    /// reference strategy (`soff`-aligned), so the root's rows are
    /// `strat[soff[0] + c * na..]` with `soff[0] == 0`.
    fn push_policy_strat(&mut self, sv: &Solver, ctx: &Ctx, row: usize, player: u8,
                         strat: &[f32]) {
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
            write_action_feats(&n.acts[a], ctx, player as usize, n.aslot[a], n.fdown[a],
                               &mut self.pa[base + a * AFEAT..base + (a + 1) * AFEAT]);
        }
        self.paoff.push((self.pa.len() / AFEAT) as u32);
        for c in 0..nc {
            self.pp.extend_from_slice(&strat[c * na..(c + 1) * na]);
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
    /// ReBeL phase only: how much of the value target comes from the realised
    /// game outcome instead of the pure CFR bootstrap (MuZero-style n-step /
    /// TD(λ) anchor). 0.0 is plain ReBeL: pure bootstrap targets.
    pub mc_mix: f32,
}


// ----------------------------------------------------------------- game loop

/// A live ReBeL walk: the solver for the current subgame, the checkpoint
/// slot it was built with, and the tree node the game is currently at.
///
/// On the GPU path (`strat` non-empty) the solve runs on the service: the
/// solver object is the walk's tree holder, the strategy rows come from the
/// downloaded reference strategy, and `gpu_id` names the resident solve for
/// the trip-2 carried beliefs.
struct Walk<'a> {
    sv: Solver<'a>,
    slot: usize,
    node: usize,
    /// Draws taken so far inside the current collapsed chance node.
    drawn: u8,
    /// The downloaded reference strategy (GPU path), `soff`-aligned. Empty
    /// on the CPU path, where `sv.average_strategy` is the source.
    strat: Vec<f32>,
    /// The resident solve's id (GPU path), for trip 2.
    gpu_id: u64,
}

impl<'a> Walk<'a> {
    /// The reference strategy row for config `c` of node `node`.
    fn strategy(&self, node: usize, c: usize) -> &[f32] {
        if self.strat.is_empty() {
            self.sv.average_strategy(node, c)
        } else {
            let na = self.sv.nodes[node].na();
            let so = self.sv.soff[node] as usize;
            &self.strat[so + c * na..so + (c + 1) * na]
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
/// On the GPU path the beliefs come from the service (trip 2): it propagates
/// each kept snapshot to the exit leaf and returns the normalised reaches.
fn finish_walk<'a>(
    w: Walk<'a>,
    bel: &[Belief; 2],
    gpu: Option<&GpuClient>,
) -> Vec<[Vec<f32>; 2]> {
    let Walk { mut sv, node, gpu_id, .. } = w;
    let mut out = if let Some(gpu) = gpu {
        gpu.carried_beliefs(gpu_id, node as u32)
            .expect("gpu trip 2")
    } else {
        sv.carried_beliefs(node)
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
    /// The GPU solve this game is waiting on, and the state to resume with.
    pending: Option<crate::gpu::SolveHandle>,
    pending_sv: Option<Solver<'a>>,
    pending_roots: Option<Vec<[Vec<f32>; 2]>>,
    pending_slot: usize,
    pending_player: u8,
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
            pending: None,
            pending_sv: None,
            pending_roots: None,
            pending_slot: 0,
            pending_player: 0,
        }
    }

    pub fn set_roots(&mut self, r: Vec<(State, [Belief; 2])>) {
        self.roots = Some(r);
    }

    pub fn take_roots(&mut self) -> Vec<(State, [Belief; 2])> {
        self.roots.take().unwrap_or_default()
    }

    /// Take the pending solve handle (after `Step::Submitted`).
    pub fn take_pending(&mut self) -> Option<crate::gpu::SolveHandle> {
        self.pending.take()
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
    pub fn advance(&mut self, gpu: Option<&GpuClient>, nets: &'a [Nets]) -> Step {
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
                bel[player as usize] =
                    belief_after_draw(&bel[player as usize], &res, &fu, wp);
                resolve_chance(s, player, rng);
                // The walk spans draws now: a draw is an internal node of the
                // subgame with one public child, so advance through it. The
                // post-draw belief must equal the tree's post-draw config
                // support (same list, same order) or every strategy row read
                // from here on is wrong.
                let mut walk_ended = false;
                if let Some(w) = walk.as_mut() {
                    let nid = w.node;
                    let n = &w.sv.nodes[nid];
                    assert!(n.chance && n.player == player, "walk not at the draw");
                    // One tree node stands for a whole run of that player's
                    // draws, so it is exhausted only once the game has taken
                    // all of them.
                    w.drawn += 1;
                    if w.drawn == n.draw_steps {
                        w.drawn = 0;
                        let child = n.child[0];
                        assert!(
                            *w.sv.nodes[child].cfgs[player as usize] == bel[player as usize].cfg[..],
                            "walk desync: post-draw support does not match the game belief"
                        );
                        w.node = child;
                        if w.sv.nodes[child].leaf {
                            walk_ended = true;
                        }
                    }
                }
                if walk_ended {
                    if gc.collect == Collect::Rebel {
                        *carried = finish_walk(walk.take().unwrap(), bel, gpu);
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
                            fallback = Some(random_policy(s, ctx, player, &cfgs));
                        } else if let Some(gpu) = gpu {
                            // GPU path: submit the tree as one job. The
                            // carried roots (or the live belief, for the
                            // first level) travel with it; trip 1 returns
                            // the reference strategy and the root values.
                            let roots_v: Vec<[Vec<f32>; 2]> = if carried.is_empty() {
                                vec![[bel[0].p.clone(), bel[1].p.clone()]]
                            } else {
                                std::mem::take(carried)
                            };
                            let job = Job::from_solver(&sv, &roots_v);
                            let handle = gpu.submit(job).expect("gpu submit");
                            self.pending = Some(handle);
                            self.pending_sv = Some(sv);
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
                                    assert_eq!(r[0].len(), bel[0].cfg.len(),
                                               "carried belief does not match the root support");
                                    assert_eq!(r[1].len(), bel[1].cfg.len(),
                                               "carried belief does not match the root support");
                                    data.push_value(
                                        s,
                                        ctx,
                                        &[
                                            Belief { cfg: bel[0].cfg.clone(), p: r[0].clone() },
                                            Belief { cfg: bel[1].cfg.clone(), p: r[1].clone() },
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
                                sv,
                                slot,
                                node: 0,
                                drawn: 0,
                                strat: Vec::new(),
                                gpu_id: 0,
                            });
                        }
                    }
                    if let Some(np) = fallback {
                        np
                    } else {
                        let w = walk.as_mut().unwrap();
                        let nid = w.node;
                        let n = &w.sv.nodes[nid];
                        // The tree was built from the belief at the subgame
                        // root and advanced in lockstep with the Bayes
                        // filter: the acting player's config support must be
                        // the same list *in order*, because the strategy rows
                        // are indexed by it. A silent desync would read the
                        // wrong row for the true config and corrupt every
                        // target from here on, so fail loudly.
                        assert!(
                            n.player == player
                                && *n.cfgs[player as usize] == bel[player as usize].cfg[..],
                            "walk desync: subgame tree no longer matches the game belief"
                        );
                        let na = n.na();
                        let mut np = NodePolicy {
                            acts: n.acts.clone(),
                            aslot: n.aslot.clone(),
                            fdown: n.fdown.clone(),
                            legal: n.legal.clone(),
                            probs: vec![0.0; cfgs.len() * na],
                        };
                        for ci in 0..cfgs.len() {
                            // Act on the CFR average — the reference strategy
                            // of the solve. Evaluation and generation are the
                            // same walk now.
                            let row = w.strategy(nid, ci);
                            np.probs[ci * na..(ci + 1) * na].copy_from_slice(row);
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

            let na = np.acts.len();
            let mut chosen = sample_row(rng, &np.probs[true_ci * na..(true_ci + 1) * na]);
            if gc.explore > 0.0
                && player as u64 == (rng.next_u64() & 1)
                && rng.unit_f64() < gc.explore as f64
            {
                let legal: Vec<usize> = (0..na).filter(|&a| np.legal[true_ci * na + a]).collect();
                if !legal.is_empty() {
                    chosen = legal[rng.below(legal.len())];
                }
            }

            // Bayes update on the *public observation*: several private
            // actions can produce it, and the belief must sum over all of
            // them.
            let obs = obs_key(&np.acts[chosen]);
            let mut pairs: Vec<(Config, f32)> = Vec::new();
            for (ci, c) in cfgs.iter().enumerate() {
                for a in 0..na {
                    if !np.legal[ci * na + a] || obs_key(&np.acts[a]) != obs {
                        continue;
                    }
                    if let Some(n) = advance_config(c, np.aslot[a], np.fdown[a]) {
                        pairs.push((n, bel[player as usize].p[ci] * np.probs[ci * na + a]));
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
                let child = w.sv.nodes[nid].child[w.sv.nodes[nid].obs_child[chosen]];
                // Advance regardless: if the child is a leaf, the walk ends
                // *at* it, and the carried beliefs must be read off that
                // node's reach.
                w.node = child;
                if w.sv.nodes[child].leaf {
                    walk_ended = true;
                }
            }
            if walk_ended {
                if gc.collect == Collect::Rebel {
                    *carried = finish_walk(walk.take().unwrap(), bel, gpu);
                } else {
                    walk.take();
                    carried.clear();
                }
            }
        }
        Step::Ended
    }

    /// Resume after trip 1: push the Phase-2 rows and start the walk with
    /// the downloaded reference strategy.
    pub fn resume(&mut self, trip1: crate::gpu::Trip1) {
        let gc = self.gc;
        let sv = self.pending_sv.take().expect("pending solver");
        let slot = self.pending_slot;
        let player = self.pending_player;
        let roots_v = self.pending_roots.take().expect("pending roots");
        if gc.collect == Collect::Rebel {
            self.data.begin_solve();
            for (r, v) in roots_v.iter().zip(trip1.root_values.iter()) {
                self.data.push_value(
                    &self.s,
                    &self.ctx,
                    &[
                        Belief { cfg: self.bel[0].cfg.clone(), p: r[0].clone() },
                        Belief { cfg: self.bel[1].cfg.clone(), p: r[1].clone() },
                    ],
                    [&v[0], &v[1]],
                );
            }
            // The policy label comes from the downloaded reference strategy.
            self.data
                .push_policy_strat(&sv, &self.ctx, self.data.nv - 1, player, &trip1.strategy);
        }
        self.carried = Vec::new();
        self.walk = Some(Walk {
            sv,
            slot,
            node: 0,
            drawn: 0,
            strat: trip1.strategy,
            gpu_id: trip1.id,
        });
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
        if self.gc.collect == Collect::Rebel && self.gc.mc_mix > 0.0 {
            // Anchor the pure bootstrap target to the realised outcome
            // (TD(lambda)-style), blended in once per game.
            let m = self.gc.mc_mix.clamp(0.0, 1.0);
            blend_outcome(&mut self.data, self.from_row, 1.0 - m, m, z);
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
fn f32_to_f16(x: f32) -> u16 {
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
        let round = if rem > (1 << (shift - 1))
            || (rem == (1 << (shift - 1)) && (half & 1) == 1)
        {
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
    let result = if z > 0.0 { 0.0 } else if z < 0.0 { 2.0 } else { 1.0 };
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
        s.zones[p as usize][Z_FACEUP][unit as usize] + s.zones[p as usize][Z_FACEDOWN][unit as usize]
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
        let mut rng = Rng::new(worker_seed(seed, i));
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

/// The GPU generation loop: `workers` threads, each playing two games so the
/// CPU has work while a game's solve runs on the GPU (docs/arch plan B1).
/// Each game submits its solve as one job and blocks on trip 1; the worker
/// alternates between its two games, waiting on whichever solve is ready.
/// When a game ends its rows are merged and a fresh game takes its place.
#[cfg(feature = "gpu")]
pub fn run_games_gpu(
    games: usize,
    seed: u64,
    nets: &[Nets],
    gc: &GameCfg,
    gpu: &crate::gpu::GpuClient,
) -> Data {
    let workers = (games / 2).max(1).min(64);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let gc = gc;
                let nets = nets;
                let gpu = gpu;
                scope.spawn(move || {
                    let mut out = Data::default();
                    let mut slot = w * 2;
                    // Two interleaved games; each holds at most one pending
                    // solve (it cannot progress past a solve site without
                    // trip 1).
                    let mut games_a: Option<Game> = None;
                    let mut games_b: Option<Game> = None;
                    let mut ha: Option<crate::gpu::SolveHandle> = None;
                    let mut hb: Option<crate::gpu::SolveHandle> = None;

                    // (Re)start a game in `g` if it ended, and advance it as
                    // far as it can go without blocking. Returns the pending
                    // handle if the game is now waiting on the GPU.
                    fn advance_slot<'a>(
                        g: &mut Option<Game<'a>>,
                        h: &mut Option<crate::gpu::SolveHandle>,
                        slot: &mut usize,
                        games: usize,
                        seed: u64,
                        gc: &'a GameCfg,
                        nets: &'a [Nets],
                        gpu: &crate::gpu::GpuClient,
                        out: &mut Data,
                    ) {
                        if g.is_none() {
                            if *slot >= games {
                                return;
                            }
                            let rng = Rng::new(worker_seed(seed, *slot));
                            *slot += 1;
                            *g = Some(Game::new(rng, gc));
                        }
                        if h.is_some() {
                            return;
                        }
                        let game = g.as_mut().unwrap();
                        match game.advance(Some(gpu), nets) {
                            Step::Submitted => *h = game.take_pending(),
                            Step::Ended => {
                                let _ = game.finish();
                                out.merge(game.take_data());
                                *g = None;
                            }
                        }
                    }

                    loop {
                        advance_slot(&mut games_a, &mut ha, &mut slot, games, seed,
                                     gc, nets, gpu, &mut out);
                        advance_slot(&mut games_b, &mut hb, &mut slot, games, seed,
                                     gc, nets, gpu, &mut out);
                        if games_a.is_none() && games_b.is_none() {
                            break;
                        }
                        // Wait on the first pending solve, then advance the
                        // game whose turn it is.
                        if let Some(h) = ha.take() {
                            let trip1 = h.wait().expect("gpu trip 1");
                            games_a.as_mut().expect("pending game").resume(trip1);
                        } else if let Some(h) = hb.take() {
                            let trip1 = h.wait().expect("gpu trip 1");
                            games_b.as_mut().expect("pending game").resume(trip1);
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

pub fn run_games(games: usize, seed: u64, nets: &[Nets], gc: &GameCfg) -> Data {
    (0..games)
        .into_par_iter()
        .fold(Data::default, |mut acc, i| {
            let mut rng = Rng::new(worker_seed(seed, i));
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
            let mut rng = Rng::new(worker_seed(seed.wrapping_add(7), i / 2));
            let swap = i % 2 == 1;
            let gc = GameCfg {
                agents: if swap { [b, a] } else { [a, b] },
                collect: Collect::None,
                explore: 0.0,
                random_draft,
                eval_mix: 0.0,
                mc_mix: 0.0,
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
