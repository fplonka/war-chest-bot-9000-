//! The ReBeL trajectory walk: a solver built once at a subgame root must
//! serve every decision inside its tree, with the tree's config support
//! staying in lockstep with the game's Bayes-filtered belief. A desync trips
//! the hard assertion in `play_game` and fails the test.

use std::collections::HashSet;
use warchest::board::N_LOCATIONS;
use warchest::rng::Rng;
use warchest::search::{Cfg, Nets};
use warchest::selfplay::{play_game, Agent, Collect, Data, GameCfg};

fn cfg() -> Cfg {
    Cfg {
        iters: 2,
        snapshots: false,
        ..Default::default()
    }
}

fn play(seed: u64, nets: &Nets, gc: GameCfg) -> Data {
    let mut d = Data::default();
    let z = play_game(Rng::new(seed), nets, &gc, &mut d, None);
    assert!(z.is_finite());
    d
}

fn rebel(explore: f32, random_draft: bool) -> GameCfg {
    GameCfg {
        agents: [Agent::Rebel { cfg: cfg() }; 2],
        collect: Collect::Rebel,
        explore,
        random_draft,
        eval_mix: 0.0,
        mc_mix: 0.0,
    }
}

/// Empty nets: the walk (build, act, advance, finish at a leaf) still runs.
/// TurboReBeL: each solve yields T+1 rows while the walk serves a couple of
/// decisions, so rows must exceed decisions overall.
#[test]
fn walk_serves_multiple_decisions_per_solve() {
    let nets = Nets::default();
    let mut dec = 0;
    let mut rows = 0;
    for (i, explore) in [0.0, 0.25].into_iter().enumerate() {
        let d = play(7919 + i as u64, &nets, rebel(explore, false));
        assert!(d.nv > 0, "no targets collected");
        dec += d.decisions;
        rows += d.nv;
    }
    assert!(
        rows > dec,
        "the T+1 multiplier is missing: {rows} rows for {dec} decisions"
    );
}

/// Eval mode: full solve up front, walk acts on the average strategy.
#[test]
fn walk_in_eval_mode() {
    let nets = Nets::default();
    let d = play(
        0xE7A1,
        &nets,
        GameCfg {
            agents: [Agent::Rebel { cfg: cfg() }, Agent::Rebel { cfg: cfg() }],
            collect: Collect::None,
            explore: 0.0,
            random_draft: false,
            eval_mix: 0.0,
            mc_mix: 0.0,
        },
    );
    assert_eq!(d.nv, 0, "eval must not collect targets");
}

/// A non-ReBeL decision drops any pending walk, wherever the game stands — with
/// random drafts, that includes a Warrior Priest forced play. The pending
/// target is kept.
#[test]
fn walk_interrupted_by_non_rebel_agent() {
    let nets = Nets::default();
    let d = play(
        0x1DEF,
        &nets,
        GameCfg {
            agents: [Agent::Rebel { cfg: cfg() }, Agent::Greedy { temp: 1.0 }],
            collect: Collect::Rebel,
            explore: 0.1,
            random_draft: true,
            eval_mix: 0.0,
            mc_mix: 0.0,
        },
    );
    assert!(d.nv > 0, "interrupted walks must still yield their target");
}

/// Random drafts (Warrior Priest included) must stay in lockstep. The hard
/// desync asserts in `play_game` are the test.
#[test]
fn walk_with_random_drafts() {
    let nets = Nets::default();
    let mut dec = 0;
    let mut rows = 0;
    for i in 0..4u64 {
        let d = play(104729 + i, &nets, rebel(0.25, true));
        assert!(d.nv > 0);
        dec += d.decisions;
        rows += d.nv;
    }
    assert!(
        rows > dec,
        "the T+1 multiplier is missing: {rows} rows for {dec} decisions"
    );
}

/// A capped build falls back to a uniform policy and drops the walk, so the
/// next subgame can root mid-coin-play, where a row has no frozen encoding.
/// A tiny node cap makes that common; the games must still finish, and the
/// `push_value` assertion on every saved row is the oracle for the rows.
#[test]
fn capped_solves_fall_back_and_games_finish() {
    let nets = Nets::default();
    let scfg = Cfg {
        node_cap: 40,
        ..cfg()
    };
    let gc = GameCfg {
        agents: [Agent::Rebel { cfg: scfg }; 2],
        ..rebel(0.25, true)
    };
    let mut dec = 0;
    let mut caps = 0;
    for i in 0..8u64 {
        let d = play(7919 + i, &nets, gc);
        dec += d.decisions;
        caps += d.node_caps;
    }
    assert!(dec > 0);
    assert!(caps > 0, "the real solver-cap counter stayed zero");
}

/// The auxiliary ownership target. It is a fact about the *finished* game, so
/// it is backfilled onto every row the game produced when the game ends: the
/// rows of one game must therefore all carry the same ten owners. Values are
/// `0`, `1` or `2` — a `NONE` marker leaking through as 255 would index past
/// the head's three classes.
#[test]
fn every_row_carries_the_finished_games_location_owners() {
    let nets = Nets::default();
    let mut seen = HashSet::new();
    for i in 0..8u64 {
        let d = play(0x5EED + i, &nets, rebel(0.25, true));
        assert!(d.nv > 0, "no rows collected");
        assert_eq!(
            d.aux.len(),
            d.nv * N_LOCATIONS,
            "aux is not in lockstep with rows"
        );
        let owners = &d.aux[..N_LOCATIONS];
        for (r, row) in d.aux.chunks_exact(N_LOCATIONS).enumerate() {
            assert!(
                row.iter().all(|&o| o <= 2),
                "row {r}: {row:?} is not an owner label"
            );
            assert_eq!(row, owners, "row {r} kept its solve-site ownership");
        }
        seen.insert(owners.to_vec());
    }
    assert!(
        seen.len() > 1,
        "the ownership label is the same in every game"
    );
}
