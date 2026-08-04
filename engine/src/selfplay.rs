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
//! subgame root, the game then descends through its tree taking an action at
//! every decision on the way (the reference's `sample_state_to_leaf`), and a
//! new solver is built only when the walk reaches a leaf of the tree — a draw,
//! a terminal state, or the depth limit. The value target is taken once per
//! subgame, at its root, exactly as in the reference (`RlRunner::step`).
//!
//! Training data comes in two flavours:
//!   * `Collect::Mc` — the greedy warm start. Value targets blend the realised
//!     game outcome with a squashed handcrafted public-information evaluation.
//!     Without it the value network is noise, CFR plays without purpose, and
//!     games only ever end at the horizon.
//!   * `Collect::Rebel` — the ReBeL loop proper: value targets are the CFR
//!     subgame root values, projected onto the network's hand-key basis.

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::rebel::*;
use crate::rng::Rng;
use crate::search::{node_actions, Cfg, Nets, Solver};
use crate::state::{Cont, State, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, BLACK, WHITE};
use rayon::prelude::*;

/// The rulebook's recommended starter matchup. Training on one fixed matchup
/// removes a large chunk of variance; randomised drafts are a distribution
/// extension, behind `random_draft`.
const STARTER_WHITE: [u16; 4] = [17, 12, 4, 9]; // Swordsman, Pikeman, Crossbowman, Light Cavalry
const STARTER_BLACK: [u16; 4] = [1, 3, 8, 16]; // Archer, Cavalry, Lancer, Scout

/// Draftable units, excluding the Warrior Priest and Warrior Priest V2 (ids 18
/// and 54). Their attribute triggers a *private* mid-round draw, which would
/// put "which coin must I now play" into the private state; the paper's own
/// advice for such a case is to clamp or exclude, and excluding keeps the
/// config space exactly `(hand, facedown)`.
const DRAFT_POOL: [u16; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 19, 52, 53];

pub fn make_game(rng: &mut Rng, random: bool) -> State {
    let first = if rng.next_u64() & 1 == 0 { WHITE } else { BLACK };
    if !random {
        return State::from_draft(&STARTER_WHITE, &STARTER_BLACK, first);
    }
    let mut pick = |rng: &mut Rng| -> Vec<u16> {
        let mut c: Vec<u16> = Vec::new();
        while c.len() < 4 {
            let id = DRAFT_POOL[rng.below(DRAFT_POOL.len())];
            if !c.contains(&id) {
                c.push(id);
            }
        }
        c
    };
    let w = pick(rng);
    let b = pick(rng);
    State::from_draft(&w, &b, first)
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
    /// Uniform over legal actions.
    Uniform,
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
                legal[ci * na + a] = aslot[a] < 0 || c.hand[aslot[a] as usize] > 0;
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
        let rep = cfgs
            .iter()
            .find(|c| np.aslot[a] < 0 || c.hand[np.aslot[a] as usize] > 0);
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

fn uniform_policy(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
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

#[derive(Default)]
pub struct Data {
    /// `[n, FEAT]` PBS encodings.
    pub vx: Vec<f32>,
    /// `[n, 2 * NHAND]` per-hand values, player 0 then player 1.
    pub vy: Vec<f32>,
    /// `[n, 2 * NHAND]` mask: which hand keys the belief actually supports.
    pub vm: Vec<f32>,
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
        self.vx.extend(o.vx);
        self.vy.extend(o.vy);
        self.vm.extend(o.vm);
        self.nv += o.nv;
        self.games += o.games;
        self.decisions += o.decisions;
        self.wins[0] += o.wins[0];
        self.wins[1] += o.wins[1];
        self.draws += o.draws;
        self.cap_hits += o.cap_hits;
        self.configs += o.configs;
    }

    /// `y[p]` holds one value per *config* in `bel[p]`; project it onto the
    /// network's hand-key basis by belief-weighted averaging, which is what the
    /// query encoding's information bottleneck requires.
    fn push_value(&mut self, s: &State, ctx: &Ctx, bel: &[Belief; 2], y: [&[f32]; 2]) {
        let base = self.vx.len();
        self.vx.resize(base + FEAT, 0.0);
        write_features(s, ctx, bel, &mut self.vx[base..base + FEAT]);
        for p in 0..2 {
            let (mut num, mut den) = ([0.0f32; NHAND], [0.0f32; NHAND]);
            for (ci, c) in bel[p].cfg.iter().enumerate() {
                let h = c.hand_index();
                num[h] += bel[p].p[ci] * y[p][ci];
                den[h] += bel[p].p[ci];
            }
            for h in 0..NHAND {
                self.vy
                    .push(if den[h] > 0.0 { num[h] / den[h] } else { 0.0 });
                self.vm.push(if den[h] > 0.0 { 1.0 } else { 0.0 });
            }
        }
        self.nv += 1;
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
    /// Evaluation mode: solve fully and act on the CFR *average* strategy.
    /// Self-play instead stops at a uniformly random iterate and acts on that,
    /// which is what makes ReBeL's value targets sound (Theorem 3).
    pub eval: bool,
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

/// A live ReBeL walk: the solver for the current subgame, the checkpoint
/// slot it was built with, the tree node the game is currently at, and the
/// (state, belief) snapshot of the subgame root — the state the value target
/// belongs to.
struct Walk<'a> {
    sv: Solver<'a>,
    slot: usize,
    node: usize,
    root_s: State,
    root_bel: [Belief; 2],
}

/// End a walk: run the solver out to its full iteration count (the partial
/// run happened at build time), then take the value target off the root of
/// the subgame — the state the walk started at.
fn finish_walk<'a>(w: Walk<'a>, gc: &GameCfg, ctx: &Ctx, data: &mut Data) {
    let Walk {
        mut sv,
        root_s,
        root_bel,
        ..
    } = w;
    sv.complete();
    if gc.collect == Collect::Rebel && sv.solved() {
        data.push_value(&root_s, ctx, &root_bel, [sv.root_values(0), sv.root_values(1)]);
    }
}

/// Play one game to the end. Returns the result from White's point of view.
pub fn play_game(rng: &mut Rng, nets: &[Nets], gc: &GameCfg, data: &mut Data) -> f32 {
    let mut s = make_game(rng, gc.random_draft);
    let ctx = Ctx::new(&s);
    let mut bel = [
        Belief::point(Config::default()),
        Belief::point(Config::default()),
    ];
    let from_row = data.nv;
    // The live ReBeL walk, if the game is inside a solved subgame. Rebuilt
    // only when the previous walk ended at a leaf of its tree.
    let mut walk: Option<Walk> = None;

    while !s.is_terminal() {
        let player = s.to_act();
        if s.is_chance() {
            let res = reserve(&s, player, &ctx);
            let fu = faceup_counts(&s, player, &ctx);
            bel[player as usize] = belief_after_draw(&bel[player as usize], &res, &fu);
            resolve_chance(&mut s, player, rng);
            // The walk spans draws now: a draw is an internal node of the
            // subgame with one public child, so advance through it. The
            // post-draw belief must equal the tree's post-draw config support
            // (same list, same order) or every strategy row read from here on
            // is wrong.
            let mut walk_ended = false;
            if let Some(w) = walk.as_mut() {
                let nid = w.node;
                let n = &w.sv.nodes[nid];
                assert!(n.chance && n.player == player, "walk not at the draw");
                let child = n.child[0];
                assert!(
                    w.sv.nodes[child].cfgs[player as usize] == bel[player as usize].cfg,
                    "walk desync: post-draw support does not match the game belief"
                );
                if w.sv.nodes[child].leaf {
                    walk_ended = true;
                } else {
                    w.node = child;
                }
            }
            if walk_ended {
                finish_walk(walk.take().unwrap(), gc, &ctx, data);
            }
            continue;
        }

        let cfgs = bel[player as usize].cfg.clone();
        let truth = true_config(&s, player, &ctx);
        let true_ci = bel[player as usize]
            .index_of(&truth)
            // Losing the real world would silently corrupt every target taken
            // from here on, so fail loudly instead.
            .expect("belief filter dropped the true config");
        data.decisions += 1;
        data.configs += cfgs.len();

        let np = match gc.agents[player as usize] {
            Agent::Greedy { temp } => {
                // A non-ReBeL decision is not in the walk's tree: end any
                // pending walk (its subgame is still solved, its target
                // still collected).
                if let Some(w) = walk.take() {
                    finish_walk(w, gc, &ctx, data);
                }
                greedy_policy(&s, &ctx, player, &cfgs, temp)
            }
            Agent::Uniform => {
                if let Some(w) = walk.take() {
                    finish_walk(w, gc, &ctx, data);
                }
                uniform_policy(&s, &ctx, player, &cfgs)
            }
            Agent::Rebel { cfg, slot } => {
                // A walk belongs to the checkpoint that built it. Playing a
                // decision on another slot's solver would make that player
                // act with the wrong network — `final_vs_init` pairs slot 0
                // against slot 1 — so end a walk built by a different slot
                // before starting a new one.
                if walk.as_ref().is_some_and(|w| w.slot != slot) {
                    finish_walk(walk.take().unwrap(), gc, &ctx, data);
                }
                if walk.is_none() {
                    // Start a new subgame at this decision: build the tree,
                    // run the partial CFR solve up to a uniformly random
                    // iterate, and snapshot the root for the value target.
                    // Acting on a random iterate keeps the targets unbiased
                    // (Theorem 3); in eval mode the full solve runs up front
                    // and the walk acts on the average strategy instead.
                    // The average strategy is only read in evaluation mode;
                    // maintaining it during generation is pure overhead.
                    let scfg = Cfg {
                        average: gc.eval,
                        ..cfg
                    };
                    let mut sv = Solver::new(&s, &ctx, &nets[slot], scfg, bel.clone());
                    let stop = if gc.eval {
                        cfg.iters
                    } else {
                        rng.below(cfg.iters + 1)
                    };
                    for i in 0..stop {
                        sv.step(i % 2);
                    }
                    walk = Some(Walk {
                        sv,
                        slot,
                        node: 0,
                        root_s: s.clone(),
                        root_bel: bel.clone(),
                    });
                }
                let w = walk.as_mut().unwrap();
                let nid = w.node;
                let n = &w.sv.nodes[nid];
                // The tree was built from the belief at the subgame root and
                // advanced in lockstep with the Bayes filter: the acting
                // player's config support must be the same list *in order*,
                // because the strategy rows are indexed by it. A silent
                // desync would read the wrong row for the true config and
                // corrupt every target from here on, so fail loudly.
                assert!(
                    n.player == player
                        && n.cfgs[player as usize] == bel[player as usize].cfg,
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
                    let row = if gc.eval {
                        w.sv.average_strategy(nid, ci)
                    } else {
                        w.sv.sampling_strategy(nid, ci)
                    };
                    np.probs[ci * na..(ci + 1) * na].copy_from_slice(row);
                }
                np
            }
        };

        if gc.collect == Collect::Mc {
            // Park the handcrafted evaluation in the target now; the realised
            // outcome is blended in once the game ends. `eval_static` is exactly
            // antisymmetric, so this stays zero-sum.
            let e = eval_squashed(&s, 0);
            let (a, b) = (vec![e; bel[0].len()], vec![-e; bel[1].len()]);
            data.push_value(&s, &ctx, &bel, [&a, &b]);
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

        // Bayes update on the *public observation*: several private actions can
        // produce it, and the belief must sum over all of them.
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

        // Advance the walk along the solved tree. The public observation of
        // the chosen action selects the child; if that child is a leaf
        // (depth exhausted, terminal, or a draw), the walk ends and the
        // subgame gets its full solve and its value target now.
        let mut walk_ended = false;
        if let Some(w) = walk.as_mut() {
            let nid = w.node;
            let child = w.sv.nodes[nid].child[w.sv.nodes[nid].obs_child[chosen]];
            if w.sv.nodes[child].leaf {
                walk_ended = true;
            } else {
                w.node = child;
            }
        }
        if walk_ended {
            finish_walk(walk.take().unwrap(), gc, &ctx, data);
        }
    }

    let z = s.utility(WHITE as usize);
    if gc.collect == Collect::Mc {
        // The parked value is the handcrafted evaluation; blend in the realised
        // outcome. This is the warm start only — ReBeL-phase targets come
        // entirely from the subgame solve.
        let m = gc.eval_mix.clamp(0.0, 1.0);
        for r in from_row..data.nv {
            let base = r * 2 * NHAND;
            for h in 0..NHAND {
                data.vy[base + h] = m * data.vy[base + h] + (1.0 - m) * z;
                data.vy[base + NHAND + h] = m * data.vy[base + NHAND + h] - (1.0 - m) * z;
            }
        }
    }
    data.games += 1;
    if s.main_plays >= crate::state::MAX_MAIN_PLAYS {
        data.cap_hits += 1;
    }
    match s.winner() {
        Some(w) => data.wins[w as usize] += 1,
        None => data.draws += 1,
    }
    z
}

fn resolve_chance(s: &mut State, player: u8, rng: &mut Rng) {
    debug_assert!(matches!(s.pending(), Cont::Draw { .. }));
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

fn effective_bag_count(s: &State, p: u8, unit: u8) -> u8 {
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

/// Play `games` games in parallel, returning merged data and statistics.
pub fn run_games(games: usize, seed: u64, nets: &[Nets], gc: &GameCfg) -> Data {
    (0..games)
        .into_par_iter()
        .fold(Data::default, |mut acc, i| {
            let mut rng = Rng::new(worker_seed(seed, i));
            play_game(&mut rng, nets, gc, &mut acc);
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
                eval: true,
                random_draft,
                eval_mix: 0.0,
            };
            let mut d = Data::default();
            let z = play_game(&mut rng, nets, &gc, &mut d);
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
