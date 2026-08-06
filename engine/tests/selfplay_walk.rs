//! The ReBeL trajectory walk: a solver built once at a subgame root must
//! serve every decision inside its tree, with the tree's config support
//! staying in lockstep with the game's Bayes-filtered belief. A desync trips
//! the hard assertion in `play_game` and fails the test.

use warchest::rng::Rng;
use warchest::search::Cfg;
use warchest::selfplay::{play_game, Agent, Collect, Data, GameCfg};

fn cfg() -> Cfg {
    Cfg { depth: 2, iters: 8, snapshots: false, ..Default::default() }
}

/// Self-play with empty nets: the walk mechanics (build, act on the sampled
/// iterate, advance, finish at a leaf) run with zero leaf values.
#[test]
fn walk_serves_multiple_decisions_per_solve() {
    let nets = [warchest::search::Nets::default()];
    let mut total_dec = 0usize;
    let mut total_rows = 0usize;
    for seed in 0..12u64 {
        for explore in [0.0, 0.25] {
            let mut rng = Rng::new(seed * 7919 + 1);
            let mut d = Data::default();
            let gc = GameCfg {
                agents: [
                    Agent::Rebel { cfg: cfg(), slot: 0 },
                    Agent::Rebel { cfg: cfg(), slot: 0 },
                ],
                collect: Collect::Rebel,
                explore,
                random_draft: false,
                eval_mix: 0.0,
                mc_mix: 0.0,
            };
            let z = play_game(&mut rng, &nets, &gc, &mut d);
            assert!(z.is_finite());
            assert!(d.nv > 0, "no targets collected");
            total_dec += d.decisions;
            total_rows += d.nv;
        }
    }
    // TurboReBeL: each solve yields T+1 rows (the carried beliefs valued
    // under the reference strategy) while the walk serves a couple of
    // decisions, so rows must exceed decisions overall.
    assert!(
        total_rows > total_dec,
        "the T+1 multiplier is missing: {} rows for {} decisions",
        total_rows,
        total_dec
    );
}

/// Eval mode: full solve up front, walk acts on the average strategy.
#[test]
fn walk_in_eval_mode() {
    let nets = [warchest::search::Nets::default()];
    let mut rng = Rng::new(0xE7A1);
    let mut d = Data::default();
    let gc = GameCfg {
        agents: [
            Agent::Rebel { cfg: cfg(), slot: 0 },
            Agent::Rebel { cfg: cfg(), slot: 0 },
        ],
        collect: Collect::None,
        explore: 0.0,
        random_draft: false,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let z = play_game(&mut rng, &nets, &gc, &mut d);
    assert!(z.is_finite());
    assert_eq!(d.nv, 0, "eval must not collect targets");
}

/// Mixed agents: a non-ReBeL decision ends any pending walk, and the pending
/// subgame's target is still collected.
#[test]
fn walk_interrupted_by_non_rebel_agent() {
    let nets = [warchest::search::Nets::default()];
    let mut rng = Rng::new(0x1DEF);
    let mut d = Data::default();
    let gc = GameCfg {
        agents: [
            Agent::Rebel { cfg: cfg(), slot: 0 },
            Agent::Greedy { temp: 1.0 },
        ],
        collect: Collect::Rebel,
        explore: 0.1,
        random_draft: false,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let z = play_game(&mut rng, &nets, &gc, &mut d);
    assert!(z.is_finite());
    assert!(d.nv > 0, "interrupted walks must still yield their target");
    assert!(d.nv <= d.decisions);
}

/// Two different checkpoints (slot 0 vs slot 1): a walk built by one slot
/// must never serve the other player's decisions. With alternating slots,
/// only same-player micro-decision continuations (Swordsman step, berserker
/// chain, ...) may reuse a walk, so targets stay close to one per decision.
/// Without the slot check the walk survives to the other player's nodes and
/// serves ~half of all decisions from the wrong checkpoint's solver.
#[test]
fn walk_never_crosses_slots() {
    let nets = [
        warchest::search::Nets::default(),
        warchest::search::Nets::default(),
    ];
    let mut total_dec = 0usize;
    let mut total_rows = 0usize;
    for seed in 0..12u64 {
        let mut rng = Rng::new(seed * 31337 + 3);
        let mut d = Data::default();
        let gc = GameCfg {
            agents: [
                Agent::Rebel { cfg: cfg(), slot: 0 },
                Agent::Rebel { cfg: cfg(), slot: 1 },
            ],
            collect: Collect::Rebel,
            explore: 0.1,
            random_draft: false,
            eval_mix: 0.0,
            mc_mix: 0.0,
        };
        let z = play_game(&mut rng, &nets, &gc, &mut d);
        assert!(z.is_finite());
        assert!(d.nv > 0);
        total_dec += d.decisions;
        total_rows += d.nv;
    }
    assert!(
        total_rows < total_dec * 3,
        "rows/decisions = {}/{} — a walk crossed slot boundaries",
        total_rows,
        total_dec
    );
    eprintln!("slot-crossing probe: rows/decisions = {}/{}", total_rows, total_dec);
}

/// Random drafts exercise different unit sets, slot maps and action shapes;
/// the walk must stay in lockstep through all of them.
#[test]
fn walk_with_random_drafts() {
    let nets = [warchest::search::Nets::default()];
    let mut total_dec = 0usize;
    let mut total_rows = 0usize;
    for seed in 0..20u64 {
        let mut rng = Rng::new(seed * 104729 + 7);
        let mut d = Data::default();
        let gc = GameCfg {
            agents: [
                Agent::Rebel { cfg: cfg(), slot: 0 },
                Agent::Rebel { cfg: cfg(), slot: 0 },
            ],
            collect: Collect::Rebel,
            explore: 0.3,
            random_draft: true,
            eval_mix: 0.0,
            mc_mix: 0.0,
        };
        let z = play_game(&mut rng, &nets, &gc, &mut d);
        assert!(z.is_finite());
        assert!(d.nv > 0);
        total_dec += d.decisions;
        total_rows += d.nv;
    }
    assert!(
        total_rows > total_dec,
        "the T+1 multiplier is missing: {} rows for {} decisions",
        total_rows,
        total_dec
    );
}

/// The walk across private Warrior Priest draws. Random drafts include the
/// Warrior Priest pair, so a WP-heavy seed stream crosses mid-round draws and
/// forced plays often; every hard desync assert in `play_game` (post-draw
/// support equality, belief-filter support, true-config presence) is the test.
#[test]
fn walk_across_warrior_priest_draws() {
    let nets = [warchest::search::Nets::default()];
    let mut total_dec = 0usize;
    for seed in 0..16u64 {
        let mut rng = Rng::new(seed * 65537 + 11);
        let mut d = Data::default();
        let gc = GameCfg {
            agents: [
                Agent::Rebel { cfg: cfg(), slot: 0 },
                Agent::Rebel { cfg: cfg(), slot: 0 },
            ],
            collect: Collect::Rebel,
            explore: 0.25,
            random_draft: true,
            eval_mix: 0.0,
            mc_mix: 0.0,
        };
        let z = play_game(&mut rng, &nets, &gc, &mut d);
        assert!(z.is_finite());
        assert!(d.nv > 0);
        total_dec += d.decisions;
    }
    assert!(total_dec > 0);
}
