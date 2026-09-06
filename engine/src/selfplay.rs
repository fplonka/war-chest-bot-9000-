use crate::actions::{Action, N_PLAYS};
use crate::board::NONE;
use crate::net::Net;
use crate::pbs::*;
use crate::policy;
use crate::rng::Rng;
use crate::search::{Budget, Cfg, Solved, Solver};
use crate::state::{Cont, State, BLACK, WHITE};
#[cfg(feature = "python")]
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::Arc;

const STARTER_WHITE: [u16; 4] = [17, 12, 4, 9];
const STARTER_BLACK: [u16; 4] = [1, 3, 8, 16];

pub const DRAFT_POOL: [u16; 19] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54];

pub fn make_game(rng: &mut Rng, random: bool) -> State {
    let first = if rng.next_u64() & 1 == 0 { WHITE } else { BLACK };
    if !random {
        return State::from_draft(&STARTER_WHITE, &STARTER_BLACK, first);
    }
    let mut pool = DRAFT_POOL;
    for i in (1..pool.len()).rev() {
        pool.swap(i, rng.below(i + 1));
    }
    State::from_draft(&pool[..4], &pool[4..8], first)
}

#[derive(Clone, Copy)]
pub enum Agent {
    Random,
    Greedy { temp: f32 },
    Sog { cfg: Cfg },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collect {
    None,
    Static,
    Sog,
}

#[derive(Default)]
pub struct Data {
    pub rows: Vec<u8>,
    pub cc: Vec<u8>,
    pub cw: Vec<f32>,
    pub cy: Vec<f32>,

    pub pa: Vec<u8>,
    pub paoff: Vec<u32>,
    pub pcoff: Vec<u32>,
    pub pci: Vec<u16>,
    pub pcell: Vec<u16>,
    pub pprob: Vec<f32>,

    pub truth: Vec<u32>,
    pub outcome: Vec<f32>,
    pub created: Vec<f64>,
    solve_created: f64,
    pub query: Vec<u8>,
    pub terminal: Option<Box<Data>>,
    pub plays: [usize; N_PLAYS],

    pub coff: Vec<u32>,
    pub soff: Vec<u32>,
    pub nv: usize,
    pub games: usize,
    pub decisions: usize,
    pub wins: [usize; 2],
    pub draws: usize,
    pub timeouts: usize,
    pub configs: usize,
    pub queries: usize,
    pub dropped: usize,
    pub completed: usize,
}

impl Data {
    pub fn merge(&mut self, o: Data) {
        let base = self.cw.len() as u32;
        let tail = if self.coff.is_empty() { 0 } else { 1 };
        self.coff.extend(o.coff.iter().skip(tail).map(|x| x + base));
        let (ab, cb) = (
            (self.pa.len() / crate::search::ACT_BYTES) as u32,
            self.pcell.len() as u32,
        );
        macro_rules! append {
            ($($name:ident),* $(,)?) => { $( self.$name.extend(o.$name); )* };
        }
        append!(rows, cc, cw, cy, pa, pci, pcell, pprob, truth, outcome, created, query);
        if let Some(t) = o.terminal {
            self.terminal.get_or_insert_with(Default::default).merge(*t);
        }
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
        self.timeouts += o.timeouts;
        self.configs += o.configs;
        self.queries += o.queries;
        self.dropped += o.dropped;
        self.completed += o.completed;
    }

    pub fn begin_solve(&mut self) {
        self.soff.push(self.nv as u32);
        self.solve_created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before Unix epoch")
            .as_secs_f64();
    }

    fn push_value(
        &mut self,
        s: &State,
        ctx: &Ctx,
        bel: &[Belief; 2],
        y: [&[f32]; 2],
        truth: [u32; 2],
        policy: &crate::search::Policy,
    ) {
        debug_assert!(s.is_valued(), "every saved value row is a valued decision");
        let base = self.rows.len();
        self.rows.resize(base + ROW_BYTES, 0);
        pack_row(s, ctx, &mut self.rows[base..base + ROW_BYTES]);
        if self.coff.is_empty() {
            self.coff.push(0);
            self.paoff.push(0);
            self.pcoff.push(0);
        }
        let actor = s.to_act() as usize;
        let usable = !policy.acts.is_empty() && policy.off.len() == bel[actor].len() + 1;
        for a in policy.acts.iter().take(if usable { usize::MAX } else { 0 }) {
            self.pa.extend_from_slice(a);
        }
        self.paoff.push((self.pa.len() / crate::search::ACT_BYTES) as u32);
        self.pcoff.push(0);
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
                    self.pci.extend(std::iter::repeat_n(within as u16, row.len()));
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
        self.nv += 1;
    }

}

#[derive(Clone, Copy)]
pub struct GameCfg {
    pub agents: [Agent; 2],
    pub collect: Collect,
    pub explore: f32,
    pub random_draft: bool,
    pub query_rate: f32,
    pub recursive_rate: f32,
}

const CHEAP_S: u32 = 32;
const TEACH_RATE: f64 = 0.1;

pub struct Game {
    rng: Rng,
    budget: Rng,
    s: State,
    ctx: Ctx,
    bel: [Belief; 2],
    ready: Data,
    pending: Data,
    gc: GameCfg,
    queries: Vec<(State, [Belief; 2])>,
    main_plays: u16,
    teaches: bool,
}

fn draw_count(rng: &mut Rng, rate: f32) -> usize {
    assert!(
        rate.is_finite() && (0.0..=1.0).contains(&rate),
        "query rate must be finite and in [0, 1], got {rate}"
    );
    (rng.unit_f64() < rate as f64) as usize
}

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

pub fn keep_query(sv: &Solver, solved: Option<Solved>, out: &mut Data) -> Vec<(State, [Belief; 2])> {
    let Some(solved) = solved else {
        return Vec::new();
    };
    out.begin_solve();
    out.push_value(
        &sv.nodes[0].state,
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

fn collects_rows(gc: &GameCfg, s: &State) -> bool {
    gc.collect == Collect::Sog && s.is_valued()
}

impl Game {
    pub fn new(mut rng: Rng, gc: &GameCfg) -> Game {
        let budget = Rng::new(rng.next_u64());
        let s = make_game(&mut rng, gc.random_draft);
        let ctx = Ctx::new(&s);
        Game {
            rng,
            budget,
            s,
            ctx,
            bel: [Belief::point(Config::default()), Belief::point(Config::default())],
            ready: Data::default(),
            pending: Data::default(),
            gc: *gc,
            queries: Vec::new(),
            main_plays: 0,
            teaches: false,
        }
    }

    pub fn take_queries(&mut self) -> Vec<(State, [Belief; 2])> {
        std::mem::take(&mut self.queries)
    }

    pub fn take_ready(&mut self) -> Data {
        std::mem::take(&mut self.ready)
    }

    pub fn next_solve(&mut self, nets: &Arc<Net>) -> Option<Solver> {
        loop {
            if crate::finished(&self.s, self.main_plays) {
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
            self.ready.decisions += 1;
            self.ready.configs += self.bel[player as usize].cfg.len();
            self.main_plays += matches!(self.s.pending(), Cont::MainPlay) as u16;
            match self.gc.agents[player as usize] {
                Agent::Random => {
                    let cfgs = self.bel[player as usize].cfg.clone();
                    let np = policy::uniform(&self.s, &self.ctx, player, &cfgs);
                    self.play(np);
                }
                Agent::Greedy { temp } => {
                    let cfgs = self.bel[player as usize].cfg.clone();
                    let np = policy::greedy(&self.s, &self.ctx, player, &cfgs, temp);
                    if self.gc.collect == Collect::Static && matches!(self.s.pending(), Cont::MainPlay) {
                        let y0 = vec![policy::eval_squashed(&self.s, 0); self.bel[0].cfg.len()];
                        let y1 = vec![policy::eval_squashed(&self.s, 1); self.bel[1].cfg.len()];
                        self.record([&y0, &y1], &np.to_replay(player, &self.ctx));
                    }
                    self.play(np);
                }
                Agent::Sog { mut cfg } => {
                    let valued = collects_rows(&self.gc, &self.s);
                    self.teaches = valued && self.budget.unit_f64() < TEACH_RATE;
                    if valued {
                        self.record_terminal();
                    }
                    if valued && !self.teaches {
                        cfg.s = CHEAP_S;
                        cfg.budget = Budget::for_s(CHEAP_S);
                    }
                    let mut sv = Solver::new(
                        &self.s,
                        self.ctx,
                        Arc::clone(nets),
                        cfg,
                        self.bel.clone(),
                        Rng::new(self.rng.next_u64()),
                    );
                    if self.teaches {
                        sv.collect(draw_count(&mut self.rng, self.gc.query_rate));
                    }
                    return Some(sv);
                }
            }
        }
    }

    pub fn play_solved(&mut self, sv: &Solver, solved: Option<Solved>) {
        if let (true, Some(solved)) = (self.teaches, solved) {
            self.record([&solved.value[0], &solved.value[1]], &solved.policy);
            self.queries.extend(solved.queries);
        }
        self.play(policy::root(sv));
    }

    fn record_terminal(&mut self) {
        let zeros = vec![0.0; self.bel.iter().map(Belief::len).max().unwrap()];
        self.pending.begin_solve();
        self.pending.push_value(
            &self.s,
            &self.ctx,
            &self.bel,
            [zeros.as_slice(); 2],
            [self.true_index(0) as u32, self.true_index(1) as u32],
            &Default::default(),
        );
    }

    fn record(&mut self, value: [&[f32]; 2], policy: &crate::search::Policy) {
        let truth = [self.true_index(0) as u32, self.true_index(1) as u32];
        self.ready.begin_solve();
        self.ready.push_value(&self.s, &self.ctx, &self.bel, value, truth, policy);
    }

    fn true_index(&self, p: usize) -> usize {
        self.bel[p]
            .index_of(&true_config(&self.s, p as u8, &self.ctx))
            .expect("belief filter dropped the true config")
    }

    fn play(&mut self, mut np: policy::NodePolicy) {
        if crate::finished(&self.s, self.main_plays) {
            return;
        }
        let me = self.s.to_act() as usize;
        np.mix_uniform(self.gc.explore);
        let true_ci = self.true_index(me);
        let chosen_cell = np.sample(&mut self.rng, true_ci);
        let chosen = np.action_at(chosen_cell);
        self.bel[me] = np.posterior(&self.bel[me], obs_key(&np.acts[chosen]));
        if let Some(slot) = self.ready.plays.get_mut(np.acts[chosen].play() as usize) {
            *slot += 1;
        }
        self.s.apply_inplace(np.acts[chosen]);
    }

    pub fn finish(&mut self) {
        let ended = self.s.is_terminal();
        let mut pending = std::mem::take(&mut self.pending);
        if ended {
            let outcome = [self.s.utility(0), self.s.utility(1)];
            pending.outcome = outcome.repeat(pending.nv);
            self.ready.terminal.get_or_insert_with(Default::default).merge(pending);
        }
        self.ready.games += 1;
        match self.s.winner() {
            Some(w) => self.ready.wins[w as usize] += 1,
            None if ended => self.ready.draws += 1,
            None => self.ready.timeouts += 1,
        }
    }
}

pub struct GameStream {
    seed: u64,
    game_index: usize,
    gc: GameCfg,
    game: Game,
    pending: VecDeque<(State, [Belief; 2])>,
    kind: SolveKind,
    query_turn: bool,
    rng: Rng,
}

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

    #[cfg(feature = "gpu")]
    pub(crate) fn solve_kind(&self) -> SolveKind {
        self.kind
    }

    pub fn keep(&mut self, sv: &Solver, solved: Option<Solved>, out: &mut Data) {
        out.completed += 1;
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

    fn end_game(&mut self, out: &mut Data) {
        self.game.finish();
        let queued = self.game.take_queries();
        out.dropped += self.enqueue(queued);
        out.merge(self.game.take_ready());
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
fn play_static_game(rng: Rng, net: &Arc<Net>, gc: &GameCfg, data: &mut Data) {
    let mut game = Game::new(rng, gc);
    assert!(game.next_solve(net).is_none(), "SoG self-play runs through SolveFarm");
    game.finish();
    data.merge(game.take_ready());
}
pub(crate) fn resolve_chance(s: &mut State, player: u8, rng: &mut Rng) -> Action {
    debug_assert!(matches!(
        s.pending(),
        Cont::Draw { .. } | Cont::WarriorPriestDraw { .. }
    ));
    let acts = s.legal_actions();
    let pool = s.draw_pool(player);
    let mut w: Vec<f64> = acts
        .iter()
        .map(|a| match a {
            Action::DrawCoin { unit } if *unit != NONE => pool[*unit as usize] as f64,
            _ => 1.0,
        })
        .collect();
    if w.iter().all(|&x| x == 0.0) {
        w.iter_mut().for_each(|x| *x = 1.0);
    }
    let drawn = acts[rng.weighted_index(&w)];
    s.apply_inplace(drawn);
    drawn
}

fn worker_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) ^ (i as u64).wrapping_mul(0xD1B54A32D192ED03)
}

#[cfg(test)]
fn resolve_fixture_chance(game: &mut Game) {
    while game.s.is_chance() {
        let player = game.s.to_act();
        let res = reserve(&game.s, player, &game.ctx);
        let fu = faceup_counts(&game.s, player, &game.ctx);
        let wp = matches!(game.s.pending(), Cont::WarriorPriestDraw { .. });
        game.bel[player as usize] = belief_after_draw(&game.bel[player as usize], &res, &fu, wp);
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
    use crate::search::Cfg;

    #[test]
    fn a_full_query_queue_reports_every_dropped_nomination() {
        let gc = GameCfg {
            agents: [Agent::Sog { cfg: Cfg::default() }; 2],
            collect: Collect::Sog,
            explore: 0.0,
            random_draft: true,
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

    #[test]
    fn returns_label_copies_of_naturally_ended_games_only() {
        let gc = GameCfg {
            agents: [Agent::Random; 2],
            collect: Collect::None,
            explore: 0.0,
            random_draft: true,
            query_rate: 0.0,
            recursive_rate: 0.0,
        };
        for winner in [None, Some(WHITE)] {
            let mut game = Game::new(Rng::new(17), &gc);
            resolve_fixture_chance(&mut game);
            let values = [vec![0.25; game.bel[0].len()], vec![-0.5; game.bel[1].len()]];
            game.record([&values[0], &values[1]], &Default::default());
            game.record_terminal();
            let targets = game.ready.cy.clone();
            if let Some(w) = winner {
                game.s.winner = w;
            }
            game.finish();
            assert_eq!(game.ready.cy, targets);
            let labels = game.ready.terminal.as_ref().map(|t| t.outcome.clone());
            assert_eq!(labels, winner.map(|_| vec![1.0, -1.0]));
        }
    }

    #[test]
    fn valued_play_uses_cheap_search_nine_times_in_ten() {
        let full = Cfg::default();
        let gc = GameCfg {
            agents: [Agent::Sog { cfg: full }; 2],
            collect: Collect::Sog,
            explore: 0.0,
            random_draft: true,
            query_rate: 0.0,
            recursive_rate: 0.0,
        };
        let net = Arc::new(Net::default());
        let mut teachers = 0;
        for seed in 0..1000 {
            let mut game = Game::new(Rng::new(seed), &gc);
            let solver = game.next_solve(&net).expect("a valued play");
            assert_eq!(game.pending.nv, 1);
            assert_eq!(game.ready.nv, 0);
            if game.teaches {
                teachers += 1;
                assert_eq!(solver.cfg.s, full.s);
            } else {
                assert_eq!(solver.cfg.s, CHEAP_S);
                assert_eq!(solver.cfg.budget, Budget::for_s(CHEAP_S));
            }
        }
        assert!((70..=130).contains(&teachers), "full searches: {teachers}");
    }

    #[test]
    fn exploration_is_the_policy_the_belief_is_updated_with() {
        let gc = GameCfg {
            agents: [Agent::Random; 2],
            collect: Collect::None,
            explore: 1.0,
            random_draft: true,
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
            assert!((got - exp).abs() < 1e-5, "posterior mass {got} vs uniform-Bayes {exp}");
        }
    }
}
