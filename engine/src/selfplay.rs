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
//! Training starts with a short greedy warm phase: rows labelled by the
//! deterministic public evaluation `policy::eval_squashed`, with the greedy
//! policy as the policy target. No solve runs in a warm game. The following SoG
//! phase treats a game that reaches the play cap as a draw with zero utility.
//!
//! **Grounding: `p_td1`.** A bootstrap target is the network's own answer
//! propagated one subgame back, so a run whose trees never reach a terminal
//! trains on nothing but itself. Student of Games' remedy is the TD(1) target:
//! with probability `p_td1` a self-play row's value is the realised outcome of
//! the game it came from instead of that solve's counterfactual values. The
//! outcome is known for the configs the players actually held, so it is written
//! there and nowhere else; every other config keeps the search value, which is
//! the best estimate there is for a hand nobody was dealt. A row that is owed
//! an outcome stays with its game until the game ends, so `p_td1 > 0` is also
//! what makes a run's rows arrive a game at a time.

use crate::actions::{Action, Play};
use crate::board::NONE;
use crate::net::Net;
use crate::policy;
use crate::pbs::*;
use crate::rng::Rng;
use crate::search::{Cfg, Solved, Solver};
use crate::state::{Cont, State, BLACK, WHITE, Z_BAG, Z_FACEDOWN, Z_FACEUP};
#[cfg(feature = "python")]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

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
    /// One-ply greedy on the public static evaluation, softmaxed at `temp`.
    Greedy { temp: f32 },
    /// Grow and solve a GT-CFR tree, then act on its average strategy.
    Sog { cfg: Cfg },
}

// -------------------------------------------------------------- data records

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collect {
    None,
    /// Greedy warm start: the public static evaluation at the current state,
    /// and the greedy policy, for every config of both seats.
    Static,
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
    pub pcell: Vec<u16>,
    pub pprob: Vec<f32>,

    /// Per row and seat: realised config within that seat's support, and final
    /// game outcome. Query rows use `u32::MAX` and NaN because no game owns them.
    pub truth: Vec<u32>,
    pub outcome: Vec<f32>,
    /// Unix time when the solve produced its rows. Main-line rows can wait for
    /// their game's outcome before Python receives them.
    pub created: Vec<f64>,
    solve_created: f64,
    /// One for a query row and one for a main-line row selected for TD(1).
    pub query: Vec<u8>,
    pub td1: Vec<u8>,
    /// Decisions by coarse move class, for the run report's strategy mix.
    pub plays: [usize; 7],

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
    /// Query nominations discarded because the stream queue was full.
    pub dropped: usize,
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
        self.truth.extend(o.truth);
        self.outcome.extend(o.outcome);
        self.created.extend(o.created);
        self.query.extend(o.query);
        self.td1.extend(o.td1);
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
        self.dropped += o.dropped;
    }

    /// Mark the start of a solve's rows. One call per solve, before its rows
    /// are pushed; `soff[k]` is the row index where solve k starts.
    pub fn begin_solve(&mut self) {
        self.soff.push(self.nv as u32);
        self.solve_created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before Unix epoch")
            .as_secs_f64();
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
        truth: [u32; 2],
        policy: &crate::search::Policy,
    ) {
        debug_assert!(
            s.is_valued(),
            "every saved value row is a valued decision"
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
        self.truth.extend(truth);
        self.outcome.extend([f32::NAN; 2]);
        self.created.push(self.solve_created);
        self.query.push(0);
        self.td1.push(0);
        self.nv += 1;
    }

    /// Move the running counters out, leaving the rows behind. See
    /// `Game::take_ready`.
    fn take_counters(&mut self) -> Data {
        Data {
            decisions: std::mem::take(&mut self.decisions),
            configs: std::mem::take(&mut self.configs),
            plays: std::mem::take(&mut self.plays),
            ..Default::default()
        }
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
    /// Weight of the uniform-over-legal mixed into each seat's acting policy.
    /// The mixture is public: sampling and the belief update both read it.
    pub explore: f32,
    /// Randomise the draft instead of using the fixed starter matchup.
    pub random_draft: bool,
    /// Probability that a self-play row's value target is the realised game
    /// outcome rather than this solve's CFR bootstrap -- Student of Games'
    /// `p_td1`, drawn once per row. Zero is the pure bootstrap, which is what
    /// the paper runs for poker; Go runs 0.2.
    pub p_td1: f32,
    /// Belief states drawn from each self-play search and queued to be solved
    /// as roots of their own — Student of Games' `q_search`. A search only
    /// tells us the value at the state it was rooted at, so this is what makes
    /// the network accurate at the leaves it is actually asked about, rather
    /// than only along the line of play. It must be in `[0, 1]`.
    pub query_rate: f32,
    /// The same, drawn from a query's own solve — `q_recursive`. Kept well
    /// below `query_rate` so the queue cannot run away from self-play. It must
    /// be in `[0, 1]`.
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
fn draw_count(rng: &mut Rng, rate: f32) -> usize {
    assert!(
        rate.is_finite() && (0.0..=1.0).contains(&rate),
        "query rate must be finite and in [0, 1], got {rate}"
    );
    (rng.unit_f64() < rate as f64) as usize
}

/// A solve of one queued belief state, as a root in its own right.
pub fn query_solver(
    nets: &Arc<Net>,
    cfg: Cfg,
    recursive_rate: f32,
    s: &State,
    bel: &[Belief; 2],
    rng: &mut Rng,
) -> Solver {
    let mut sv = Solver::new(
        s,
        Ctx::new(s),
        Arc::clone(nets),
        cfg,
        bel.clone(),
        Rng::new(rng.next_u64()),
    );
    sv.collect(draw_count(rng, recursive_rate));
    sv
}

/// Keep a query solve's root value as a training row, and hand back the roots
/// its own harvest nominated.
///
/// Value only. Student of Games assembles policy targets from "the searches
/// started at public states along the main line of episodes"; a query solve is
/// off that line, and its job is coverage of the value function. Its policy
/// would also be trained against, and so reinforce, states self-play never
/// actually reaches.
pub fn keep_query(
    sv: &Solver,
    solved: Option<Solved>,
    out: &mut Data,
) -> Vec<(State, [Belief; 2])> {
    let Some(solved) = solved else {
        return Vec::new();
    };
    out.begin_solve();
    out.push_value(
        &sv.states[0],
        &sv.ctx,
        &sv.root_belief,
        [&solved.value[0], &solved.value[1]],
        [u32::MAX; 2],
        &Default::default(),
    );
    *out.query.last_mut().expect("query row") = 1;
    out.queries += 1;
    solved.queries
}

/// Whether a row is collected here. Every valued decision carries one, and
/// only the SoG collector takes them from a solve.
fn collects_rows(gc: &GameCfg, s: &State) -> bool {
    gc.collect == Collect::Sog && s.is_valued()
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

    /// This game's rows, which it gives up only once it has ended and `finish`
    /// has written the outcome into the rows that drew it.
    pub fn take_data(&mut self) -> Data {
        assert!(
            self.s.is_terminal(),
            "a game gives up its rows only once it has ended"
        );
        std::mem::take(&mut self.data)
    }

    /// Everything whose value target is already final.
    ///
    /// A pure bootstrap row is finished the moment its own solve is: the target
    /// is that solve's own answer, and nothing later changes it. A run with
    /// `p_td1 > 0` still owes some of its rows the game's outcome, so those wait
    /// for `finish` and only the counters come out here -- otherwise a run that
    /// has yet to end a game would report no decisions and no play mix either,
    /// which is exactly the run whose diagnostics matter most.
    pub fn take_ready(&mut self) -> Data {
        if self.gc.p_td1 > 0.0 {
            return self.data.take_counters();
        }
        std::mem::take(&mut self.data)
    }

    /// True when the game has ended (used by the worker's idle check).
    pub fn is_terminal(&self) -> bool {
        self.s.is_terminal()
    }

    /// Play forward until a solve is wanted, and hand it over. `None` once the
    /// game has ended.
    ///
    /// A chance node and a random agent's decision need no search, so they are
    /// resolved here. Everything else about a decision waits for the solve,
    /// which is why the game is a state machine: the solve belongs to the farm
    /// and may be run on any core, or on a card, long after this returns.
    pub fn next_solve(&mut self, nets: &Arc<Net>) -> Option<Solver> {
        loop {
            if self.s.is_terminal() {
                return None;
            }
            let player = self.s.to_act();
            if self.s.is_chance() {
                let res = reserve(&self.s, player, &self.ctx);
                let fu = faceup_counts(&self.s, player, &self.ctx);
                let wp = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
                let me = player as usize;
                self.bel[me] = belief_after_draw(&self.bel[me], &res, &fu, wp);
                resolve_chance(&mut self.s, player, &mut self.rng);
                continue;
            }
            self.data.decisions += 1;
            self.data.configs += self.bel[player as usize].cfg.len();
            match self.gc.agents[player as usize] {
                Agent::Random => {
                    let cfgs = self.bel[player as usize].cfg.clone();
                    let np = policy::uniform(&self.s, &self.ctx, player, &cfgs);
                    self.play(np);
                }
                Agent::Greedy { temp } => {
                    let cfgs = self.bel[player as usize].cfg.clone();
                    let np = policy::greedy(&self.s, &self.ctx, player, &cfgs, temp);
                    if self.gc.collect == Collect::Static
                        && matches!(self.s.pending(), Cont::MainPlay)
                    {
                        let y0 = vec![policy::eval_squashed(&self.s, 0); self.bel[0].cfg.len()];
                        let y1 = vec![policy::eval_squashed(&self.s, 1); self.bel[1].cfg.len()];
                        let truth = [self.true_index(0) as u32, self.true_index(1) as u32];
                        self.data.begin_solve();
                        self.data.push_value(
                            &self.s,
                            &self.ctx,
                            &self.bel,
                            [&y0, &y1],
                            truth,
                            &np.to_replay(),
                        );
                    }
                    self.play(np);
                }
                Agent::Sog { cfg } => {
                    // Student of Games re-solves from scratch at every
                    // decision. The solve gives this state its value and
                    // nominates leaves to be solved in their own right.
                    let mut sv = Solver::new(
                        &self.s,
                        self.ctx,
                        Arc::clone(nets),
                        cfg,
                        self.bel.clone(),
                        Rng::new(self.rng.next_u64()),
                    );
                    if collects_rows(&self.gc, &self.s) {
                        sv.collect(draw_count(&mut self.rng, self.gc.query_rate));
                    }
                    return Some(sv);
                }
            }
        }
    }

    /// Act on a finished solve: keep the row it produced, then play its move.
    pub fn play_solved(&mut self, sv: &Solver, solved: Option<Solved>) {
        if let Some(solved) = solved {
            // The row is stored under the belief that is about to be updated,
            // so the seats' realised configs are read here and not at `finish`.
            let truth = [self.true_index(0) as u32, self.true_index(1) as u32];
            self.data.begin_solve();
            self.data.push_value(
                &self.s,
                &self.ctx,
                &self.bel,
                [&solved.value[0], &solved.value[1]],
                truth,
                &solved.policy,
            );
            self.queries.extend(solved.queries);
        }
        self.play(policy::root(sv));
    }

    /// Where the config a seat is really holding sits in its own belief
    /// support. Losing the real world would silently corrupt every target taken
    /// from here on, so this fails loudly instead.
    fn true_index(&self, p: usize) -> usize {
        self.bel[p]
            .index_of(&true_config(&self.s, p as u8, &self.ctx))
            .expect("belief filter dropped the true config")
    }

    /// Sample the acting player's move, update the public belief from what the
    /// opponent observes, and apply it.
    ///
    /// The acting policy is `(1-eps) π + eps` uniform over each config's
    /// legal set; sampling and the belief update both read that mixture.
    fn play(&mut self, mut np: policy::NodePolicy) {
        let me = self.s.to_act() as usize;
        np.mix_uniform(self.gc.explore);
        let true_ci = self.true_index(me);
        let chosen_cell = np.sample(&mut self.rng, true_ci);
        let chosen = np.action_at(chosen_cell);
        // Bayes update on the *public observation*: several private actions can
        // produce it, and the belief must sum over all of them.
        self.bel[me] = np.posterior(&self.bel[me], obs_key(&np.acts[chosen]));
        if let Some(slot) = match np.acts[chosen].play() {
            Play::Attack => Some(0),
            Play::Pass => Some(1),
            Play::Deploy => Some(2),
            Play::Bolster => Some(3),
            Play::Maneuver => Some(4),
            Play::Recruit => Some(5),
            Play::ClaimInitiative => Some(6),
            Play::Other => None,
        } {
            self.data.plays[slot] += 1;
        }
        self.s.apply_inplace(np.acts[chosen]);
    }

    /// The game ended: write the outcome into the rows that drew a TD(1)
    /// target, and return White's result.
    pub fn finish(&mut self) -> f32 {
        // A game cut at the play cap is a draw, so its utility is zero.
        let z = [self.s.utility(0), self.s.utility(1)];
        for r in 0..self.data.nv {
            for (p, &outcome) in z.iter().enumerate() {
                self.data.outcome[2 * r + p] = outcome;
            }
            if self.gc.p_td1 <= 0.0 || self.rng.unit_f64() >= self.gc.p_td1 as f64 {
                continue;
            }
            self.data.td1[r] = 1;
            for p in 0..2 {
                let at = self.data.row_span(r, p).start + self.data.truth[2 * r + p] as usize;
                self.data.cy[at] = z[p];
            }
        }
        self.data.games += 1;
        if self.s.main_plays >= crate::state::MAX_MAIN_PLAYS {
            self.data.cap_hits += 1;
        }
        match self.s.winner() {
            Some(w) => self.data.wins[w as usize] += 1,
            None => self.data.draws += 1,
        }
        self.s.utility(WHITE as usize)
    }
}

/// A resumable SoG game stream: one solve at a time, and the game around it.
///
/// The solve it hands out belongs to whoever runs it. That is what makes the
/// stream a state machine — a run keeps hundreds of these in flight against a
/// few dozen cores, so a stream cannot be a thread.
pub struct GameStream {
    seed: u64,
    game_index: usize,
    gc: GameCfg,
    game: Game,
    /// Student of Games' query solver, inlined into the actor: belief states
    /// nominated by earlier searches, each waiting to become the root of its
    /// own solve and its own training row.
    pending: VecDeque<(State, [Belief; 2])>,
    /// What the solve in flight is for.
    kind: SolveKind,
    /// Whether the query solver has the next turn.
    query_turn: bool,
    rng: Rng,
}

/// What the solve in flight is for: the line of play, or coverage away from it.
#[derive(Clone, Copy)]
#[repr(u32)]
pub(crate) enum SolveKind {
    Play,
    Query,
}

impl SolveKind {
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub const NAMES: [&'static str; 2] = ["play", "query"];
}

/// How many queued queries one stream will hold. A bounded queue keeps a
/// random-walk backlog from becoming an unbounded host allocation.
const QUEUE_CAP: usize = 64;

impl GameStream {
    pub fn new(seed: u64, gc: GameCfg) -> GameStream {
        let game = Game::new(Rng::new(worker_seed(seed, 0)), &gc);
        GameStream {
            seed,
            game_index: 1,
            gc,
            game,
            pending: VecDeque::new(),
            kind: SolveKind::Play,
            query_turn: false,
            rng: Rng::new(worker_seed(seed, usize::MAX)),
        }
    }

    /// The next solve this stream wants run.
    ///
    /// Self-play and the query solver take turns, one solve each. Each
    /// self-play search queues `query_rate` queries and each query solve queues
    /// `recursive_rate`, so at the reference rates a turn each leaves the queue
    /// exactly where it started. An empty queue simply gives its turn back to
    /// self-play.
    pub fn next_solve(&mut self, nets: &Arc<Net>, out: &mut Data) -> Solver {
        if self.query_turn {
            self.query_turn = false;
            if let Some(sv) = self.next_query(nets) {
                self.kind = SolveKind::Query;
                return sv;
            }
        }
        self.query_turn = true;
        self.kind = SolveKind::Play;
        loop {
            if let Some(sv) = self.game.next_solve(nets) {
                return sv;
            }
            self.end_game(out);
        }
    }

    pub(crate) fn solve_kind(&self) -> SolveKind {
        self.kind
    }

    /// Run this stream on this host until it has produced `solves` rows.
    ///
    /// The farm runs a stream one solve at a time and answers its calls in a
    /// round shared with every other solve in flight. This is the same stream
    /// driven alone, which is what the single-process tools and the tests
    /// want.
    #[cfg(test)]
    pub fn generate(&mut self, nets: &Arc<Net>, solves: usize) -> Data {
        assert!(solves > 0);
        let mut out = Data::default();
        while out.soff.len() < solves {
            let mut sv = self.next_solve(nets, &mut out);
            let solved = sv.run_alone();
            self.keep(&sv, solved, &mut out);
        }
        out
    }

    /// Take the finished solve back: keep its row, and let the game act on it.
    pub fn keep(&mut self, sv: &Solver, solved: Option<Solved>, out: &mut Data) {
        match self.kind {
            SolveKind::Play => {
                self.game.play_solved(sv, solved);
                let queued = self.game.take_queries();
                out.dropped += self.enqueue(queued);
                out.merge(self.game.take_ready());
            }
            SolveKind::Query => {
                let more = keep_query(sv, solved, out);
                out.dropped += self.enqueue(more);
            }
        }
    }

    fn next_query(&mut self, nets: &Arc<Net>) -> Option<Solver> {
        let (s, bel) = self.pending.pop_front()?;
        let Agent::Sog { cfg } = self.gc.agents[s.to_act() as usize] else {
            return None;
        };
        Some(query_solver(nets, cfg, self.gc.recursive_rate, &s, &bel, &mut self.rng))
    }

    /// The game ended: score it, keep what it left, and start the next one.
    fn end_game(&mut self, out: &mut Data) {
        self.game.finish();
        let queued = self.game.take_queries();
        out.dropped += self.enqueue(queued);
        out.merge(self.game.take_data());
        self.game = Game::new(Rng::new(worker_seed(self.seed, self.game_index)), &self.gc);
        self.game_index += 1;
    }

    fn enqueue(&mut self, qs: Vec<(State, [Belief; 2])>) -> usize {
        let n = qs.len();
        let room = QUEUE_CAP.saturating_sub(self.pending.len());
        self.pending.extend(qs.into_iter().take(room));
        n.saturating_sub(room)
    }
}

#[cfg(feature = "python")]
fn play_static_game(rng: Rng, net: &Arc<Net>, gc: &GameCfg, data: &mut Data) -> f32 {
    let mut game = Game::new(rng, gc);
    assert!(
        game.next_solve(net).is_none(),
        "SoG self-play runs through SolveFarm"
    );
    let value = game.finish();
    data.merge(game.take_data());
    value
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

// ------------------------------------------------------------- batch drivers

fn worker_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) ^ (i as u64).wrapping_mul(0xD1B54A32D192ED03)
}

/// Make deterministic public roots without playing a complete self-play game.
///
/// The oracle tests need a valid decision state, not a complete game stream.
/// Resolving the opening draw gives each test a deterministic, cheap fixture.
#[cfg(test)]
fn resolve_fixture_chance(game: &mut Game) {
    while game.s.is_chance() {
        let player = game.s.to_act();
        let res = reserve(&game.s, player, &game.ctx);
        let fu = faceup_counts(&game.s, player, &game.ctx);
        let wp = matches!(game.s.pending(), Cont::WarriorPriestDraw { .. });
        game.bel[player as usize] =
            belief_after_draw(&game.bel[player as usize], &res, &fu, wp);
        resolve_chance(&mut game.s, player, &mut game.rng);
    }
}

#[cfg(test)]
pub(crate) fn collect_roots(count: usize, seed: u64) -> Vec<(State, [Belief; 2])> {
    let gc = GameCfg {
        agents: [Agent::Random; 2],
        collect: Collect::None,
        explore: 0.0,
        random_draft: false,
        p_td1: 0.0,
        query_rate: 0.0,
        recursive_rate: 0.0,
    };
    let mut out = Vec::with_capacity(count);
    for attempt in 0..count.saturating_mul(64).max(1) {
        if out.len() == count {
            break;
        }
        let mut game = Game::new(Rng::new(seed.wrapping_add(attempt as u64)), &gc);
        resolve_fixture_chance(&mut game);
        if game.s.is_terminal() || !matches!(game.s.pending(), Cont::MainPlay) {
            continue;
        }
        out.push((game.s, game.bel));
    }
    assert_eq!(out.len(), count, "not enough random-walk roots");
    out
}

/// Play `games` static-agent games in parallel, returning merged data and statistics.
#[cfg(feature = "python")]
pub(crate) fn run_static_games(games: usize, seed: u64, nets: &Arc<Net>, gc: &GameCfg) -> Data {
    (0..games)
        .into_par_iter()
        .fold(Data::default, |mut acc, i| {
            let rng = Rng::new(worker_seed(seed, i));
            play_static_game(rng, nets, gc, &mut acc);
            acc
        })
        .reduce(Data::default, |mut a, b| {
            a.merge(b);
            a
        })
}

#[cfg(test)]
mod target_tests {
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

    #[test]
    fn a_full_query_queue_reports_every_dropped_nomination() {
        let gc = GameCfg {
            agents: [Agent::Sog { cfg: Cfg::default() }; 2],
            collect: Collect::Sog,
            explore: 0.0,
            random_draft: true,
            p_td1: 0.0,
            query_rate: 0.0,
            recursive_rate: 0.0,
        };
        let mut stream = GameStream::new(1, gc);
        let q = (stream.game.s, stream.game.bel.clone());
        let total = QUEUE_CAP + 3;
        let mut nominations = Vec::with_capacity(total);
        for _ in 0..total {
            nominations.push((q.0, q.1.clone()));
        }
        let mut out = Data::default();
        out.dropped += stream.enqueue(nominations);
        assert_eq!(stream.pending.len(), QUEUE_CAP);
        assert_eq!(out.dropped, 3);
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
        let nets = Arc::new(random_net(0x5EED));
        let cfg = Cfg { s: 32, c: 4.0, ..Default::default() };
        let roots = collect_roots(3, 0x5EED);
        assert!(!roots.is_empty(), "no roots to test against");

        let mut rng = Rng::new(0x9017);
        let mut checked = 0usize;
        for (s, bel) in &roots {
            let ctx = Ctx::new(s);
            let mut sv = Solver::new(
                s,
                ctx,
                Arc::clone(&nets),
                cfg,
                bel.clone(),
                Rng::new(rng.next_u64()),
            );
            sv.collect(0);
            let solved = sv.run_alone().expect("a collected solve");
            let mut d = Data::default();
            d.begin_solve();
            d.push_value(
                s,
                &ctx,
                bel,
                [&solved.value[0], &solved.value[1]],
                [u32::MAX; 2],
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
                let want = sv.root_strategy(k);
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

    /// A finished game writes its TD(1) outcome only at the configs the seats
    /// really held. A query row in the same buffer must remain unchanged.
    #[test]
    fn a_finished_game_writes_its_outcome_where_the_seats_were() {
        let nets = Arc::new(random_net(0x7D1));
        let cfg = Cfg { s: 4, c: 1.0, ..Default::default() };

        // A query row, in the buffer the game will merge into.
        let mut out = Data::default();
        let (s, bel) = collect_roots(1, 0x7D1).pop().expect("a root");
        let mut qv = Solver::new(
            &s,
            Ctx::new(&s),
            Arc::clone(&nets),
            cfg,
            bel.clone(),
            Rng::new(0x9E4),
        );
        qv.collect(0);
        let solved = qv.run_alone();
        keep_query(&qv, solved, &mut out);
        let query_cy = out.cy.clone();
        assert!(!query_cy.is_empty(), "the query solve stored no values");

        let gc = GameCfg {
            agents: [Agent::Sog { cfg }; 2],
            collect: Collect::Sog,
            explore: 0.0,
            random_draft: false,
            p_td1: 1.0,
            query_rate: 0.0,
            recursive_rate: 0.0,
        };
        let mut g = Game::new(Rng::new(6), &gc);
        g.s = s;
        g.ctx = Ctx::new(&s);
        g.bel = bel;
        let values = [vec![0.25; g.bel[0].len()], vec![-0.25; g.bel[1].len()]];
        let truth = [g.true_index(0) as u32, g.true_index(1) as u32];
        g.data.begin_solve();
        g.data.push_value(
            &g.s,
            &g.ctx,
            &g.bel,
            [&values[0], &values[1]],
            truth,
            &Default::default(),
        );
        g.s.winner = WHITE;
        let before = g.data.cy.clone();
        let truth = g.data.truth.clone();
        let z = [g.s.utility(0), g.s.utility(1)];
        g.finish();
        let d = g.take_data();
        assert_eq!(d.nv, 1, "fixture stored the wrong number of rows");
        assert_eq!(d.queries, 0, "a query row reached a game");

        let mut written = 0;
        for p in 0..2 {
            let span = d.row_span(0, p);
            let at = span.start + truth[p] as usize;
            for i in span {
                if i == at {
                    assert_eq!(d.cy[i], z[p], "seat {p}: outcome target");
                    written += 1;
                } else {
                    assert_eq!(d.cy[i], before[i], "seat {p}: unrelated config moved");
                }
            }
        }
        assert_eq!(written, 2);
        out.merge(d);
        assert_eq!(out.cy[..query_cy.len()], query_cy[..]);

        let mut st = GameStream::new(5, GameCfg { p_td1: 0.2, ..gc });
        let mut live = Data::default();
        let mut sv = st.next_solve(&nets, &mut live);
        let solved = sv.run_alone();
        st.keep(&sv, solved, &mut live);
        assert_eq!(live.nv, 0, "a row left its game before the game ended");
        assert!(live.decisions > 0, "the counters did not come out");
    }

    /// A static row stores one public evaluation per seat and the greedy policy.
    #[test]
    fn a_static_row_is_antisymmetric_and_constant_across_configs() {
        let (s, bel) = collect_roots(1, 0x2E57).pop().expect("a root");
        let ctx = Ctx::new(&s);
        let actor = s.to_act() as usize;
        let cfgs = bel[actor].cfg.clone();
        let policy = policy::greedy(&s, &ctx, actor as u8, &cfgs, 2.0).to_replay();
        let values = [vec![0.25; bel[0].len()], vec![-0.25; bel[1].len()]];
        let truth = [
            bel[0].index_of(&true_config(&s, 0, &ctx)).expect("white config") as u32,
            bel[1].index_of(&true_config(&s, 1, &ctx)).expect("black config") as u32,
        ];
        let mut data = Data::default();
        data.begin_solve();
        data.push_value(
            &s,
            &ctx,
            &bel,
            [&values[0], &values[1]],
            truth,
            &policy,
        );
        assert_eq!(data.nv, 1);
        let a = data.row_span(0, 0);
        let b = data.row_span(0, 1);
        assert!(a.len() > 1 && b.len() > 1);
        for i in a.clone() {
            assert_eq!(data.cy[i], 0.25, "white configs disagree");
        }
        for i in b.clone() {
            assert_eq!(data.cy[i], -0.25, "black configs disagree");
        }
        assert!(data.paoff[1] > data.paoff[0], "no policy actions");
    }

    /// With explore = 1 the acting policy is uniform over legal, so the public
    /// belief after a move is the Bayes update under that uniform, not under
    /// the unmixed search policy.
    #[test]
    fn exploration_is_the_policy_the_belief_is_updated_with() {
        let gc = GameCfg {
            agents: [Agent::Random; 2],
            collect: Collect::None,
            explore: 1.0,
            random_draft: true,
            p_td1: 0.0,
            query_rate: 0.0,
            recursive_rate: 0.0,
        };
        let (s, bel) = collect_roots(8, 0xE1)
            .into_iter()
            .find(|(s, bel)| bel[s.to_act() as usize].cfg.len() > 1)
            .expect("a MainPlay whose actor has more than one config");
        let me = s.to_act() as usize;
        let ctx = Ctx::new(&s);
        let mut g = Game::new(Rng::new(0xE1), &gc);
        g.s = s;
        g.ctx = ctx;
        g.bel = bel;

        let prior = g.bel[me].clone();
        let uni = policy::uniform(&g.s, &g.ctx, me as u8, &prior.cfg);
        let mut peaked = policy::NodePolicy::frame(&g.s, &g.ctx, me as u8, &prior.cfg);
        for ci in 0..prior.cfg.len() {
            let row = peaked.row(ci);
            if !row.is_empty() {
                peaked.probs[row.start] = 1.0;
            }
        }
        let before = g.s;
        g.play(peaked);
        let obs = before
            .legal_actions()
            .into_iter()
            .find_map(|a| {
                let mut t = before;
                t.apply_inplace(a);
                (t == g.s).then_some(obs_key(&a))
            })
            .expect("the played action");
        let want = uni.posterior(&prior, obs);
        assert_eq!(g.bel[me].cfg, want.cfg, "posterior support");
        for (got, exp) in g.bel[me].p.iter().zip(&want.p) {
            assert!(
                (got - exp).abs() < 1e-5,
                "posterior mass {got} vs uniform-Bayes {exp}"
            );
        }
    }
}
