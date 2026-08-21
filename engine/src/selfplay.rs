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
//! A SoG decision solves its own subgame with GT-CFR and acts on the CFR
//! average. The tree grows towards its node budget while the solve runs.
//!
//! **A solve yields exactly one value row: its root's.** The opposing range is
//! normalised there, so the value is on the game's own scale and stays on it.
//! Interior nodes are not targets — their range carries the reach that led to
//! them, and a node beside the frontier would only return the network's own
//! leaf output, which trains the network on itself. Both references agree:
//! ReBeL adds `{beta_r, v(beta_r)}` alone to its value set, and Student of
//! Games re-solves each query from scratch and stores that solve's root.
//!
//! Coverage away from the line of play comes from the query solver instead.
//! Each search nominates a few of the leaves it asked the network about; each
//! of those is later solved as a root in its own right and yields its own row.
//!
//! There is no warm start. A game that reaches the play cap scores
//! `cap_value * delta_markers`, so the first solves against an untrained
//! network still see a graded outcome -- the win condition, graded -- rather
//! than a flat zero. `state::cap_marker_value` anneals it away.

use crate::actions::{Action, Play};
use crate::board::NONE;
use crate::policy;
use crate::pbs::*;
use crate::rng::Rng;
use crate::search::{Cfg, Nets, Solver};
use crate::state::{Cont, State, BLACK, WHITE, Z_BAG, Z_FACEDOWN, Z_FACEUP};
use rayon::prelude::*;
use std::collections::VecDeque;

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

// -------------------------------------------------------------------- agents

#[derive(Clone, Copy)]
pub enum Agent {
    /// Uniform over legal actions: the weakest reference on the Elo ladder.
    Random,
    /// Grow and solve a GT-CFR tree, then act on its average strategy.
    Sog { cfg: Cfg },
}

// -------------------------------------------------------------- data records

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collect {
    None,
    /// Student of Games: CFR subgame root values.
    Sog,
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
    /// `[n * ROW_BYTES]` packed replay rows (see `pbs::ROW_*`). The public encoding is *not* stored — a
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

    // ------------------------------------------------- the policy target
    // Student of Games trains the policy head on the root's average policy.
    // It is per (config, legal action), so it is a second ragged level under
    // the config arena, and it exists only for the acting player's configs --
    // the other player's rows are empty, which is also what a row with no
    // policy at all looks like.
    /// `[n * ACT_BYTES]` per row: how many actions the root offered, and each
    /// described the way the policy head reads one. `paoff` bounds a row.
    pub pa: Vec<u8>,
    pub paoff: Vec<u32>,
    /// `[n + 1]` offsets into the cell arrays below, by row.
    pub pcoff: Vec<u32>,
    /// Per cell: which config it belongs to, as an index into the row's own
    /// configs *across both players*, which of the row's actions it is, and the
    /// target probability. Indexed within the row rather than within the acting
    /// support so a batch never has to know which seat acted.
    pub pci: Vec<u16>,
    pub pcell: Vec<u8>,
    pub pprob: Vec<f32>,
    /// Decisions by coarse move class, for the run report's strategy mix.
    pub plays: [usize; 6],

    /// `[2 * n + 1]` arena offsets.
    pub coff: Vec<u32>,
    /// Solve starts in row space: `soff[k]` is the row at which solve k
    /// starts (first entry 0). The Python binding appends the total row count
    /// as the trailing entry. Empty when no rows were collected.
    pub soff: Vec<u32>,
    pub nv: usize,
    pub games: usize,
    pub decisions: usize,
    pub wins: [usize; 2],
    pub draws: usize,
    /// Completed games that reached `MAX_MAIN_PLAYS`. This is the game horizon,
    /// not the solver's tree-node cap.
    pub cap_hits: usize,
    pub configs: usize,
    /// Rows that came from the query solver rather than from a self-play
    /// decision. Off the line of play, so this is the coverage term.
    pub queries: usize,
}

impl Data {
    pub fn merge(&mut self, o: Data) {
        let base = self.cw.len() as u32;
        self.rows.extend(o.rows);
        self.cc.extend(o.cc);
        self.cw.extend(o.cw);
        self.cy.extend(o.cy);
        // Both sides carry a leading zero; the merged arena has exactly one, so
        // the other's is dropped. `coff` must stay `2 * nv + 1` long or every
        // row after the join is read with somebody else's configs.
        let tail = if self.coff.is_empty() { 0 } else { 1 };
        self.coff.extend(o.coff.iter().skip(tail).map(|x| x + base));
        // The policy arenas join the same way: one leading zero between them,
        // and each side's offsets shifted onto the merged arrays.
        let (ab, cb) = (
            (self.pa.len() / crate::search::ACT_BYTES) as u32,
            self.pcell.len() as u32,
        );
        self.pa.extend(o.pa);
        self.pci.extend(o.pci);
        self.pcell.extend(o.pcell);
        self.pprob.extend(o.pprob);
        self.paoff.extend(o.paoff.iter().skip(tail).map(|x| x + ab));
        self.pcoff.extend(o.pcoff.iter().skip(tail).map(|x| x + cb));
        let rb = self.nv as u32;
        self.soff.extend(o.soff.iter().map(|x| x + rb));
        self.nv += o.nv;
        self.games += o.games;
        self.decisions += o.decisions;
        for (a, b) in self.plays.iter_mut().zip(o.plays) {
            *a += b;
        }
        self.wins[0] += o.wins[0];
        self.wins[1] += o.wins[1];
        self.draws += o.draws;
        self.cap_hits += o.cap_hits;
        self.configs += o.configs;
        self.queries += o.queries;
    }

    /// Mark the start of a solve's rows. One call per solve, before its rows
    /// are pushed; `soff[k]` is the row index where solve k starts.
    pub fn begin_solve(&mut self) {
        self.soff.push(self.nv as u32);
    }

    /// `y[p]` holds one value per *config* in `bel[p]`. Every one of them is
    /// stored: the value function is a function of the config, so there is
    /// nothing to average away.
    fn push_value(
        &mut self,
        s: &State,
        ctx: &Ctx,
        bel: &[Belief; 2],
        y: [&[f32]; 2],
        policy: &crate::search::Policy,
    ) {
        debug_assert!(
            matches!(s.pending(), Cont::MainPlay),
            "every saved value row is a normal coin-play state"
        );
        let base = self.rows.len();
        self.rows.resize(base + ROW_BYTES, 0);
        pack_row(s, ctx, &mut self.rows[base..base + ROW_BYTES]);
        if self.coff.is_empty() {
            self.coff.push(0);
            self.paoff.push(0);
            self.pcoff.push(0);
        }
        // The policy belongs to whoever acts at the root; the other player's
        // configs carry an empty row, as does every config of a row that has
        // no policy at all.
        let actor = s.to_act() as usize;
        let usable = !policy.acts.is_empty() && policy.off.len() == bel[actor].len() + 1;
        for a in policy.acts.iter().take(if usable { usize::MAX } else { 0 }) {
            self.pa.extend_from_slice(a);
        }
        self.paoff.push((self.pa.len() / crate::search::ACT_BYTES) as u32);
        self.pcoff.push(0); // rewritten below, once the cells are known
        for p in 0..2 {
            let res = reserve(s, p as u8, ctx);
            let mut cnt = [0u8; CCOUNTS];
            for (ci, c) in bel[p].cfg.iter().enumerate() {
                config_counts(c, &res, &mut cnt);
                self.cc.extend_from_slice(&cnt);
                self.cw.push(bel[p].p[ci]);
                self.cy.push(y[p][ci]);
                if usable && p == actor {
                    let row = policy.off[ci] as usize..policy.off[ci + 1] as usize;
                    let within = if actor == 0 { ci } else { bel[0].len() + ci };
                    self.pci
                        .extend(std::iter::repeat(within as u16).take(row.len()));
                    self.pcell.extend_from_slice(&policy.act[row.clone()]);
                    self.pprob.extend_from_slice(&policy.p[row]);
                }
            }
            self.coff.push(self.cw.len() as u32);
        }
        *self.pcoff.last_mut().expect("row offset") = self.pcell.len() as u32;
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
    /// Probability that the designated explorer plays uniformly this decision.
    pub explore: f32,
    /// Randomise the draft instead of using the fixed starter matchup.
    pub random_draft: bool,
    /// How much of the value target comes from the realised game outcome
    /// rather than from the CFR bootstrap -- Student of Games' `p_td1`. Zero is
    /// the pure bootstrap, which is what the paper runs for poker.
    pub p_td1: f32,
    /// Belief states drawn from each self-play search and queued to be solved
    /// as roots of their own — Student of Games' `q_search`. A search only
    /// tells us the value at the state it was rooted at, so this is what makes
    /// the network accurate at the leaves it is actually asked about, rather
    /// than only along the line of play.
    pub query_rate: f32,
    /// The same, drawn from a query's own solve — `q_recursive`. Kept well
    /// below `query_rate` so the queue cannot run away from self-play.
    pub recursive_rate: f32,
}

// ----------------------------------------------------------------- game loop

pub struct Game {
    rng: Rng,
    s: State,
    ctx: Ctx,
    bel: [Belief; 2],
    data: Data,
    gc: GameCfg,
    /// Belief states this game's searches asked the network about, waiting to
    /// be solved as roots of their own.
    queries: Vec<(State, [Belief; 2])>,
}

/// A rate like "0.9 queries per search", drawn into a whole number.
/// Solve one belief state as a root and keep its row. Returns the roots the
/// harvest sampled, for whoever wants to queue them.
///
/// Value only. Student of Games assembles policy targets from "the searches
/// started at public states along the main line of episodes"; a query solve is
/// off that line, and its job is coverage of the value function. Its policy
/// would also be trained against, and so reinforce, states self-play never
/// actually reaches.
pub fn solve_root(
    nets: &Nets,
    cfg: Cfg,
    recursive_rate: f32,
    s: &State,
    bel: &[Belief; 2],
    rng: &mut Rng,
    out: &mut Data,
) -> Vec<(State, [Belief; 2])> {
    let ctx = Ctx::new(s);
    let mut sv = Solver::new(s, ctx, nets, cfg, bel.clone());
    sv.solve(rng);
    let want = draw_count(rng, recursive_rate);
    let solved = sv.harvest(rng, want);
    out.begin_solve();
    out.push_value(
        s,
        &ctx,
        bel,
        [&solved.value[0], &solved.value[1]],
        &Default::default(),
    );
    out.queries += 1;
    solved.queries
}

fn draw_count(rng: &mut Rng, rate: f32) -> usize {
    let whole = rate.max(0.0).floor();
    whole as usize + (rng.unit_f64() < (rate - whole) as f64) as usize
}

/// Whether a row is collected here. Only coin plays carry one, and only the
/// SoG collector takes them from a solve.
fn collects_rows(gc: &GameCfg, s: &State) -> bool {
    gc.collect == Collect::Sog && matches!(s.pending(), Cont::MainPlay)
}

impl Game {
    pub fn new(mut rng: Rng, gc: &GameCfg) -> Game {
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
            data: Data::default(),
            gc: *gc,
            queries: Vec::new(),
        }
    }

    /// The belief states this game's searches queued, handed to whoever will
    /// solve them.
    pub fn take_queries(&mut self) -> Vec<(State, [Belief; 2])> {
        std::mem::take(&mut self.queries)
    }

    /// This game's rows, which it gives up only once it has ended and final
    /// ownership targets have been backfilled.
    pub fn take_data(&mut self) -> Data {
        assert!(
            self.s.is_terminal(),
            "a game gives up its rows only once it has ended"
        );
        std::mem::take(&mut self.data)
    }

    /// Yield solved SoG rows without ending the game. Pure bootstrap targets
    /// are complete when their subgame solve ends; no game outcome backfill is
    /// pending.
    pub fn take_rows(&mut self) -> Data {
        assert_eq!(self.gc.collect, Collect::Sog);
        assert_eq!(self.gc.p_td1, 0.0);
        std::mem::take(&mut self.data)
    }

    /// True when the game has ended (used by the worker's idle check).
    pub fn is_terminal(&self) -> bool {
        self.s.is_terminal()
    }

    /// Play until `max_solves` fresh training solves are available or the game
    /// ends. This gives streaming generation a bounded unit of work.
    pub fn advance_solves(&mut self, nets: &Nets, max_solves: usize) {
        assert!(max_solves > 0);
        let stop = self.data.soff.len().saturating_add(max_solves);
        let gc = &self.gc;
        let Game {
            rng,
            s,
            ctx,
            bel,
            data,
            queries,
            ..
        } = self;
        while !s.is_terminal() && data.soff.len() < stop {
            let player = s.to_act();
            if s.is_chance() {
                let res = reserve(s, player, ctx);
                let fu = faceup_counts(s, player, ctx);
                let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
                bel[player as usize] = belief_after_draw(&bel[player as usize], &res, &fu, wp);
                resolve_chance(s, player, rng);
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
                Agent::Random => policy::uniform(s, ctx, player, &cfgs),
                Agent::Sog { cfg } => {
                    // Student of Games re-solves from scratch at every
                    // decision. The solve gives this state its value and
                    // nominates leaves to be solved in their own right.
                    let mut sv = Solver::new(s, *ctx, nets, cfg, bel.clone());
                    sv.solve(rng);
                    if collects_rows(gc, s) {
                        let want = draw_count(rng, gc.query_rate);
                        let solved = sv.harvest(rng, want);
                        data.begin_solve();
                        data.push_value(
                            s,
                            ctx,
                            bel,
                            [&solved.value[0], &solved.value[1]],
                            &solved.policy,
                        );
                        queries.extend(solved.queries);
                    }
                    policy::at_node(&sv, 0, cfgs.len())
                }
            };


            let true_row = np.row(true_ci);
            let mut chosen_cell = np.sample(rng, true_ci);
            // A fresh explorer each decision, as the reference does.
            let explorer = sample_explorer(rng, gc.explore);
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

        }
    }

    /// Play the game to the end.
    pub fn advance(&mut self, nets: &Nets) {
        while !self.is_terminal() {
            self.advance_solves(nets, usize::MAX);
        }
    }


    /// The game ended: blend the outcome into parked targets and return White's
    /// result.
    pub fn finish(&mut self) -> f32 {
        let z = self.s.utility(WHITE as usize);
        if self.gc.p_td1 > 0.0 {
            // Anchor the bootstrap target to the realised outcome, blended in
            // once per game.
            let m = self.gc.p_td1.clamp(0.0, 1.0);
            blend_outcome(&mut self.data, 1.0 - m, m, z);
        }
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

/// A resumable SoG game stream. Each chunk ends at a solved decision rather
/// than at a game boundary, so one long game cannot stall the trainer.
pub struct GameStream {
    seed: u64,
    game_index: usize,
    gc: GameCfg,
    game: Game,
    /// Student of Games' query solver, inlined into the actor: belief states
    /// nominated by earlier searches, each waiting to become the root of its
    /// own solve and its own training row.
    pending: VecDeque<(State, [Belief; 2])>,
    rng: Rng,
}

/// How many queued queries one stream will hold. At the intended rates the
/// queue is stationary, so this only bounds the damage if a run is configured
/// with `query_rate` above one.
const QUEUE_CAP: usize = 4096;

impl GameStream {
    pub fn new(seed: u64, gc: GameCfg) -> GameStream {
        let game = Game::new(Rng::new(worker_seed(seed, 0)), &gc);
        GameStream {
            seed,
            game_index: 1,
            gc,
            game,
            pending: VecDeque::new(),
            rng: Rng::new(worker_seed(seed, usize::MAX)),
        }
    }

    pub fn generate(&mut self, nets: &Nets, solves: usize) -> Data {
        assert!(solves > 0);
        let mut out = Data::default();
        // Self-play and the query solver take turns, one solve each. Each
        // self-play search queues `query_rate` queries and each query solve
        // queues `recursive_rate`, so at the reference rates a turn each
        // leaves the queue exactly where it started. An empty queue simply
        // gives its turn back to self-play.
        while out.soff.len() < solves {
            self.advance_game(nets, &mut out);
            if out.soff.len() < solves {
                self.solve_query(nets, &mut out);
            }
        }
        out
    }

    fn advance_game(&mut self, nets: &Nets, out: &mut Data) {
        self.game.advance_solves(nets, 1);
        let ended = self.game.is_terminal();
        if ended {
            self.game.finish();
        }
        let queued = self.game.take_queries();
        self.enqueue(queued);
        out.merge(self.game.take_rows());
        if ended {
            self.game = Game::new(
                Rng::new(worker_seed(self.seed, self.game_index)),
                &self.gc,
            );
            self.game_index += 1;
        }
    }

    /// Solve one queued belief state as a root and keep its row.
    fn solve_query(&mut self, nets: &Nets, out: &mut Data) {
        let Some((s, bel)) = self.pending.pop_front() else {
            return;
        };
        let Agent::Sog { cfg } = self.gc.agents[s.to_act() as usize] else {
            return;
        };
        let more = solve_root(nets, cfg, self.gc.recursive_rate, &s, &bel, &mut self.rng, out);
        self.enqueue(more);
    }

    fn enqueue(&mut self, qs: Vec<(State, [Belief; 2])>) {
        for q in qs {
            if self.pending.len() >= QUEUE_CAP {
                break;
            }
            self.pending.push_back(q);
        }
    }
}

/// Play one game to the end. Returns the result from White's point of view.
///
/// Any belief states the searches queued are appended to `queries`.
pub fn play_game(
    rng: Rng,
    nets: &Nets,
    gc: &GameCfg,
    data: &mut Data,
    queries: Option<&mut Vec<(State, [Belief; 2])>>,
) -> f32 {
    let mut g = Game::new(rng, gc);
    g.advance(nets);
    let z = g.finish();
    if let Some(q) = queries {
        q.extend(g.take_queries());
    }
    let d = g.take_data();
    data.merge(d);
    z
}
/// `y <- keep * y + mix * (+-z)` over every config of every row this game
/// produced. The sign flips for player 1: `z` is White's outcome and the
/// targets are per-player utilities of a zero-sum game.
fn blend_outcome(data: &mut Data, keep: f32, mix: f32, z: f32) {
    for r in 0..data.nv {
        for p in 0..2 {
            let sign = if p == 0 { 1.0 } else { -1.0 };
            for i in data.row_span(r, p) {
                data.cy[i] = keep * data.cy[i] + mix * sign * z;
            }
        }
    }
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

/// Collect subgame roots from random-draft games, for the tree-sizing tools.
///
/// These are the belief states the searches queued, so they are drawn from the
/// leaf distribution the value network is actually queried on.
pub fn collect_roots(
    games: usize,
    seed: u64,
    nets: &Nets,
    gc: &GameCfg,
    cap: usize,
) -> Vec<(State, [Belief; 2])> {
    let mut out: Vec<(State, [Belief; 2])> = (0..games)
        .into_par_iter()
        .fold(Vec::new, |mut acc, i| {
            let rng = Rng::new(worker_seed(seed, i));
            let mut d = Data::default();
            play_game(rng, nets, gc, &mut d, Some(&mut acc));
            acc
        })
        .reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            a
        });
    out.truncate(cap);
    assert!(!out.is_empty(), "no roots: a game collected none");
    out
}

/// Play `games` games in parallel, returning merged data and statistics.


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

#[cfg(test)]
mod policy_target_tests {
    use super::*;
    use crate::search::{Cfg, Solver};

    fn random_net(seed: u64) -> crate::net::Net {
        let mut r = Rng::new(seed);
        let l = crate::net::NetLayout::new();
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect()
        };
        let (w, b) = (draw(l.w_len), draw(l.b_len));
        let mut ln = vec![0.0; l.ln_len];
        for n in &l.norms {
            ln[n.g..n.g + n.width].fill(1.0);
        }
        crate::net::Net::from_flat(&w, &b, &ln).expect("random net")
    }

    /// The uniform belief over every config a seat could be holding.
    fn uniform_belief(s: &State, ctx: &Ctx, p: u8) -> Belief {
        let truth = true_config(s, p, ctx);
        let cfg = enumerate_configs(
            &reserve(s, p, ctx),
            truth.hand_size(),
            truth.fd_size(),
            truth.inflight.is_some(),
        );
        let n = cfg.len() as f32;
        Belief { p: vec![1.0 / n; cfg.len()], cfg }
    }

    /// A few real coin plays, reached by random play, each with the uniform
    /// belief over both seats.
    ///
    /// Taking roots from self-play instead would play whole games to keep three
    /// of their queries, and a game is up to 256 plies with a solve at each.
    /// What this test checks is how a solved root is stored, so any real root
    /// will do.
    fn positions(seed: u64, want: usize) -> Vec<(State, [Belief; 2])> {
        let mut out = Vec::new();
        for i in 0..64 {
            if out.len() == want {
                break;
            }
            let mut rng = Rng::new(seed + i);
            let mut s = make_game(&mut rng, true);
            for _ in 0..8 {
                if s.is_terminal() {
                    break;
                }
                let acts = s.legal_actions();
                s.apply_inplace(acts[rng.below(acts.len())]);
            }
            if s.is_terminal() || s.is_chance() || !matches!(s.pending(), Cont::MainPlay) {
                continue;
            }
            let ctx = Ctx::new(&s);
            let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
            out.push((s, bel));
        }
        out
    }

    /// A stored row's policy target must be the solve's own root average: one
    /// row per acting config, summing to one, over the actions that config can
    /// actually play.
    ///
    /// The arena is ragged twice over — configs under rows, cells under
    /// configs — so an off-by-one in either offset array silently pairs a
    /// config with somebody else's policy. That is invisible in training
    /// except as a loss that will not fall, which is why it is pinned here.
    #[test]
    fn a_stored_row_carries_the_root_average_policy() {
        let nets = Nets {
            value: random_net(0x5EED),
            device: false,
            gate: None,
        };
        let cfg = Cfg { s: 32, c: 4.0, ..Default::default() };
        let roots = positions(0x5EED, 3);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut rng = Rng::new(0x9017);
        let mut checked = 0usize;
        for (s, bel) in &roots {
            let ctx = Ctx::new(s);
            let mut sv = Solver::new(s, ctx, &nets, cfg, bel.clone());
            sv.solve(&mut rng);
            let solved = sv.harvest(&mut rng, 0);
            let mut d = Data::default();
            d.begin_solve();
            d.push_value(
                s,
                &ctx,
                bel,
                [&solved.value[0], &solved.value[1]],
                &solved.policy,
            );

            let actor = s.to_act() as usize;
            let span = d.row_span(0, actor);
            assert_eq!(span.len(), bel[actor].len(), "acting config count");
            let na = (d.paoff[1] - d.paoff[0]) as usize;
            assert!(na > 0, "a solved root must offer actions");
            assert_eq!(na, sv.nodes[0].na(), "action count");

            let cells = d.pcoff[0] as usize..d.pcoff[1] as usize;
            let shift = if actor == 0 { 0 } else { bel[0].len() };
            for k in 0..span.len() {
                let mine: Vec<usize> = cells
                    .clone()
                    .filter(|&cell| d.pci[cell] as usize == k + shift)
                    .collect();
                let want = sv.average_strategy(0, k);
                assert_eq!(mine.len(), want.len(), "config {k}: cell count");
                let total: f32 = mine.iter().map(|&cell| d.pprob[cell]).sum();
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "config {k}: policy sums to {total}"
                );
                for (j, &cell) in mine.iter().enumerate() {
                    assert_eq!(
                        d.pprob[cell], want[j],
                        "config {k} cell {j}: stored policy differs"
                    );
                    assert!((d.pcell[cell] as usize) < na, "action index out of range");
                }
            }
            let total = bel[0].len() + bel[1].len();
            assert_eq!(
                d.pci.iter().filter(|&&c| c as usize >= total).count(),
                0,
                "a cell names a config outside the row"
            );
            checked += 1;
        }
        assert!(checked > 0, "no solve to check");
    }
}
