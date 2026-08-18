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

use crate::actions::{Action, Play};
use crate::board::{board, NONE, N_HEXES, N_LOCATIONS};
use crate::gpu::GpuClient;
use crate::policy::{self, NodePolicy};
use crate::rebel::*;
use crate::rng::Rng;
use crate::search::{Cfg, Nets, Solver};
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
/// state as `Config::inflight`, which the solver, belief filter and walk
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
    Rebel { cfg: Cfg },
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
    /// `[n * ROW_BYTES]` packed replay rows (see `rebel::ROW_*`). The public encoding is *not* stored — a
    /// row is expanded when a batch is made, so the stored bytes never go
    /// stale as the network changes.
    pub rows: Vec<u8>,
    /// `[n * N_LOCATIONS]` the auxiliary ownership target: who owns each
    /// control location when the game the row came from ends, `0`/`1` for the
    /// player and `2` for neither. `push_value` reserves the space and
    /// `backfill_owners` writes it at `finish`, which is why no row leaves a
    /// worker before its game has ended: the ownership at the solve site is a
    /// raw input feature of the same row, so a row shipped early would teach
    /// this head to copy its own input. Training-only: the engine never
    /// predicts it.
    pub aux: Vec<u8>,
    /// `[total_configs, CCOUNTS]` raw counts per config, in the arena order.
    /// Raw rather than normalised: they are `u8`-valued, and storing them that
    /// way is what keeps a replay row small enough to hold millions of them.
    pub cc: Vec<u8>,
    /// `[total_configs]` belief probability of each config.
    pub cw: Vec<f32>,
    /// `[total_configs]` the solve's value for each config.
    pub cy: Vec<f32>,
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
    /// Decisions by coarse move class, for the run report's strategy mix.
    pub plays: [usize; 6],
    /// Live games intentionally discarded at a wall-clock deadline. This is
    /// time-censored work, not a capacity drop and never enters replay.
    pub censored_games: usize,

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
        self.aux.extend(o.aux);
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
        for (a, b) in self.plays.iter_mut().zip(o.plays) {
            *a += b;
        }
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
        // Reserved, not written: the auxiliary target is a property of the
        // finished game and `backfill_owners` fills it at `finish`. The
        // sentinel is not a valid class, so a row that ever reached training
        // without its label would fail the cross-entropy instead of quietly
        // teaching the head to copy an input feature -- which is exactly what
        // happened for as long as rows left a worker mid-game.
        self.aux.resize(self.aux.len() + N_LOCATIONS, AUX_UNSET);
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
    /// Probability the walk's explorer plays uniform. Without a walk, a new
    /// explorer is drawn each decision.
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

/// A live ReBeL walk: the solver for the current subgame, and the tree node
/// the game is currently at.
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
    /// Fixed for the walk: this player may play uniform, the other does not.
    explorer: u8,
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
            WalkState::Cpu(sv) => policy::cpu_frame(sv, node),
            WalkState::Gpu(tree) => policy::wave_frame(tree, node, tree.node_player[node] as usize),
        }
    }
}

/// Whether the subgame rooted at `s` may be collected. A row's public encoding
/// is frozen at normal coin-play states, so a subgame rooted mid-coin-play is
/// solved for play only. A walk that ends of its own accord ends at a leaf, and
/// a leaf is a coin play; a walk that is *dropped* — by a node cap, by the
/// other seat playing Greedy or Random, by a slot change — leaves the next
/// decision wherever the game stands, which can be a forced play or a soak.
fn collects_rows(gc: &GameCfg, s: &State) -> bool {
    gc.collect == Collect::Rebel && matches!(s.pending(), Cont::MainPlay)
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
            data: Data::default(),
            from_row: 0,
            gc,
            roots: None,
            pending_job: None,
            pending_walk: None,
            pending_roots: None,
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

    /// This game's rows, which it gives up only once it has ended.
    ///
    /// The guard is the whole reason the auxiliary target works: its label is
    /// the finished game's location ownership, written by `finish`, and a row
    /// handed over before then would instead carry the ownership at its own
    /// solve site — which is one of that row's own input features. The GPU
    /// stream used to take rows after every solve, and no test caught it
    /// because the tests drive `play_game`, which never did.
    pub fn take_data(&mut self) -> Data {
        assert!(
            self.s.is_terminal(),
            "a game gives up its rows only once it has ended"
        );
        std::mem::take(&mut self.data)
    }

    /// True when the game has ended (used by the worker's idle check).
    pub fn is_terminal(&self) -> bool {
        self.s.is_terminal()
    }

    /// Play until a GPU solve is submitted or the game ends.
    pub fn advance(&mut self, gpu: Option<&[GpuClient]>, nets: &'a Nets) -> Step {
        let gc = self.gc;
        let Game {
            rng,
            s,
            ctx,
            bel,
            walk,
            carried,
            data,
            roots,
            ..
        } = self;
        let mut roots_out: Option<&mut Vec<(State, [Belief; 2])>> = roots.as_mut();
        while !s.is_terminal() {
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
                    policy::greedy(s, ctx, player, &cfgs, temp)
                }
                Agent::Random => {
                    walk.take();
                    carried.clear();
                    policy::uniform(s, ctx, player, &cfgs)
                }
                Agent::Rebel { cfg } => {
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
                        let mut sv = Solver::new(s, *ctx, nets, scfg, bel.clone());
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
                            fallback = Some(policy::uniform(s, ctx, player, &cfgs));
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
                            self.pending_player = player;
                            return Step::Submitted;
                        } else {
                            // CPU path: the full solve, then the walk.
                            sv.multistep(cfg.iters);
                            if collects_rows(gc, s) {
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
                            } else {
                                // Nothing will ever value them: this solve is
                                // the only one whose root they belong to.
                                carried.clear();
                            }
                            *walk = Some(Walk {
                                tree: WalkState::Cpu(sv),
                                explorer: sample_explorer(rng, gc.explore),
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
            let mut chosen_cell = np.sample(rng, true_ci);
            // One explorer per walk. No walk: this decision is its own walk.
            let explorer = walk
                .as_ref()
                .map(|w| w.explorer)
                .unwrap_or_else(|| sample_explorer(rng, gc.explore));
            if gc.explore > 0.0
                && player == explorer
                && rng.unit_f64() < gc.explore as f64
                && !true_row.is_empty()
            {
                chosen_cell = true_row.start + rng.below(true_row.len());
            }
            let chosen = np.action_at(chosen_cell);

            // Bayes update on the *public observation*: several private
            // actions can produce it, and the belief must sum over all of
            // them.
            bel[player as usize] = np.posterior(&bel[player as usize], obs_key(&np.acts[chosen]));
            if let Some(slot) = match np.acts[chosen].play() {
                Play::Attack => Some(0),
                Play::Pass => Some(1),
                Play::Deploy => Some(2),
                Play::Bolster => Some(3),
                Play::Maneuver => Some(4),
                Play::Recruit => Some(5),
                Play::Other => None,
            } {
                data.plays[slot] += 1;
            }
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
        let roots_v = self.pending_roots.take().expect("pending roots");
        self.pending_oversize = false;
        if collects_rows(gc, &self.s) {
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
        }
        self.carried = Vec::new();
        self.walk = Some(Walk {
            tree: WalkState::Gpu(tree),
            explorer: sample_explorer(&mut self.rng, self.gc.explore),
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
    pub fn retry_cpu(&mut self, nets: &'a Nets) {
        // Release the packed GPU representation before allocating the CPU CFR
        // arenas. The worker has normally taken `pending_job` already; keep the
        // take here so direct callers cannot accidentally retain it.
        self.pending_job.take();
        self.pending_walk.take().expect("pending walk tree");
        let roots_v = self.pending_roots.take().expect("pending roots");
        let oversize = std::mem::take(&mut self.pending_oversize);
        let player = self.pending_player;
        let Agent::Rebel { cfg } = self.gc.agents[player as usize] else {
            panic!("GPU retry requested for a non-ReBeL agent");
        };
        let scfg = Cfg {
            snapshots: self.gc.collect == Collect::Rebel,
            gpu_build: false,
            ..cfg
        };
        let mut sv = Solver::new(&self.s, self.ctx, nets, scfg, self.bel.clone());
        assert!(
            !sv.capped(),
            "a GPU job that passed the node cap capped on its exact CPU retry"
        );
        sv.multistep(cfg.iters);
        if collects_rows(self.gc, &self.s) {
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
        }
        self.carried.clear();
        self.walk = Some(Walk {
            tree: WalkState::Cpu(sv),
            explorer: sample_explorer(&mut self.rng, self.gc.explore),
            node: 0,
            drawn: 0,
            strat: Vec::new(),
            carries: None,
        });
        self.data.exact_fallbacks += 1;
        self.data.oversize_routes += oversize as usize;
    }

    /// The game ended: blend the outcome into parked targets, stamp the final
    /// location ownership onto them, and return White's result.
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
        // The auxiliary target is a property of the finished game, not of the
        // row's own state, so it is backfilled here exactly as the outcome is.
        backfill_owners(&mut self.data, self.from_row, &self.s);
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
    nets: &Nets,
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

/// A game cut off when the run stops contributes its census and nothing else.
///
/// Its solves were real and its bootstrapped values are sound, but the targets
/// that are properties of the finished game — the blended outcome anchor and
/// the final location ownership — were never written, and the ownership a row
/// would otherwise carry is one of that row's own input features. Rather than
/// mark such rows and mask them in two languages, they are dropped: censoring
/// happens only while a run drains, and `censored_games` says how many.
#[cfg(feature = "gpu")]
fn censored() -> Data {
    Data {
        censored_games: 1,
        ..Default::default()
    }
}

/// Stamp the final owner of every control location onto every row this game
/// produced. Reaching the horizon counts as an ending: the ownership then is
/// the ownership the game got to.
fn backfill_owners(data: &mut Data, from_row: usize, s: &State) {
    let owners = location_owners(s);
    for r in from_row..data.nv {
        data.aux[r * N_LOCATIONS..(r + 1) * N_LOCATIONS].copy_from_slice(&owners);
    }
}

/// Not one of the three ownership classes: the value an auxiliary target holds
/// between the row being written and its game ending.
const AUX_UNSET: u8 = 3;

/// Who owns each control location: `0`/`1` for the player holding its control
/// marker, `2` for neither. The auxiliary head's target, in `location_hexes`
/// order.
fn location_owners(s: &State) -> [u8; N_LOCATIONS] {
    let loc = &board().location_hexes;
    std::array::from_fn(|i| match s.loc_marker[loc[i] as usize] {
        NONE => 2,
        p => p,
    })
}

pub(crate) fn resolve_chance(s: &mut State, player: u8, rng: &mut Rng) -> Action {
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
    let drawn = acts[rng.weighted_index(&w)];
    s.apply_inplace(drawn);
    drawn
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

fn sample_explorer(rng: &mut Rng, explore: f32) -> u8 {
    if explore > 0.0 {
        (rng.next_u64() & 1) as u8
    } else {
        0
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
    nets: &Nets,
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
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn run_games_gpu_stream(
    seed: u64,
    nets: &Nets,
    gc: &GameCfg,
    gpus: &[crate::gpu::GpuClient],
    workers: usize,
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
    // A worker holds exactly as many live games as it can have solves in
    // flight. Holding more only delays how long a game takes to finish, and a
    // bootstrap that never finishes a game has no terminal outcome to anchor
    // its value targets against.
    let per = inflight_per_worker.max(1);
    let max_inflight = per;
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
                            if game[k].take().is_some() {
                                let _ = data_tx.send(Ok(censored()));
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
                                        game[k] = None;
                                        let _ = data_tx.send(Ok(censored()));
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
                                    // The only place a finished game's rows
                                    // leave the worker. Their outcome-shaped
                                    // targets -- the blended value anchor and
                                    // the final location ownership -- exist
                                    // only once `finish` has run, so a row
                                    // shipped before this point could not
                                    // carry them.
                                    let _ = g.finish();
                                    let mut d = g.take_data();
                                    d.merge_wait_s += std::mem::take(&mut merge_wait);
                                    let handoff = std::time::Instant::now();
                                    let sent = data_tx.send(Ok(d));
                                    merge_wait += handoff.elapsed().as_secs_f32();
                                    if sent.is_err() {
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
                        // Stop closes admission, and this solve finished while
                        // the last waves drained. Its row would still be a row
                        // of an unfinished game, so it goes with the rest of
                        // them; only the census survives.
                        game[k] = None;
                        live -= 1;
                        match result {
                            Ok(_) => {
                                let _ = data_tx.send(Ok(censored()));
                            }
                            Err(e) => {
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
                            g.data.gpu_wait_s += waited.elapsed().as_secs_f32();
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

pub fn run_games(games: usize, seed: u64, nets: &Nets, gc: &GameCfg) -> Data {
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
