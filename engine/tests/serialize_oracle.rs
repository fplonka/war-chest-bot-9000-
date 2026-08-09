//! The tree serializer against real solvers: a solver built from real game
//! states must round-trip through the byte format identically, and every
//! contract array must be consistent with the solver it came from.

use warchest::rng::Rng;
use warchest::search::{Cfg, Nets, Solver};
use warchest::selfplay::collect_roots;
use warchest::selfplay::{Agent, Collect, GameCfg};
use warchest::serialize::Job;

fn cfg() -> Cfg {
    Cfg {
        depth: 2,
        iters: 8,
        snapshots: true,
        ..Default::default()
    }
}

/// Serialize a real solver, parse it back, and require byte-identical
/// re-serialization.
#[test]
fn real_solver_round_trips() {
    let nets = [Nets::default()];
    let gc = GameCfg {
        agents: [Agent::Rebel {
            cfg: cfg(),
            slot: 0,
        }; 2],
        collect: Collect::Rebel,
        explore: 0.0,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let roots = collect_roots(12, 0x5EED, &nets, &gc, 8);
    assert!(!roots.is_empty(), "no roots collected");
    let mut checked = 0;
    for (s, bel) in roots {
        let ctx = warchest::rebel::Ctx::new(&s);
        let sv = Solver::new(&s, ctx, &nets[0], cfg(), bel);
        if sv.capped() {
            continue;
        }
        let job = Job::from_solver(&sv, &[]);
        let bytes = job.to_bytes();
        let back = Job::from_bytes(&bytes).expect("parse");
        assert_eq!(back.to_bytes(), bytes, "byte-identical round trip");
        assert_eq!(back.tables.nodes, sv.nodes.len());
        assert_eq!(back.meta.iters, cfg().iters);
        assert_eq!(back.tables.nleaf, sv.leaf_rows.len());
        assert_eq!(back.tables.rows, sv.leaf_rows.len() + sv.inner_rows.len());
        assert_eq!(back.tables.ncells, sv.ncells);
        checked += 1;
    }
    assert!(checked >= 4, "too few uncapped solves: {checked}");
}

/// The contract arrays must be internally consistent: child CSR spans, obs
/// CSR spans, config spans, soff monotonicity, BFS order covering all nodes.
#[test]
fn tables_are_consistent() {
    let nets = [Nets::default()];
    let gc = GameCfg {
        agents: [Agent::Rebel {
            cfg: cfg(),
            slot: 0,
        }; 2],
        collect: Collect::Rebel,
        explore: 0.0,
        random_draft: true,
        eval_mix: 0.0,
        mc_mix: 0.0,
    };
    let roots = collect_roots(12, 0xABCD, &nets, &gc, 6);
    for (s, bel) in roots {
        let ctx = warchest::rebel::Ctx::new(&s);
        let sv = Solver::new(&s, ctx, &nets[0], cfg(), bel);
        if sv.capped() {
            continue;
        }
        let job = Job::from_solver(&sv, &[]);
        let t = &job.tables;
        let n = t.nodes;
        assert_eq!(t.node_child_start.len(), n + 1);
        assert_eq!(t.node_child_start[n] as usize, t.node_child.len());
        assert_eq!(t.soff.len(), n + 1);
        assert_eq!(t.soff[n] as usize, t.ncells);
        assert_eq!(t.cfg_off.len(), 2 * n + 1);
        assert_eq!(t.reach_off[n] as usize, t.reach_len);
        // Reverse tables: every non-root node has exactly one gather block,
        // in the row space matching its parent's kind.
        assert_eq!(*t.rev_start.last().unwrap() as usize, t.rev_src.len());
        assert_eq!(t.rev_src.len(), t.rev_cell.len());
        assert_eq!(*t.rvd_start.last().unwrap() as usize, t.rvd_src.len());
        assert_eq!(t.rvd_src.len(), t.rvd_p.len());
        for j in 1..n {
            let p = t.node_parent[j] as usize;
            assert!(p < j, "parent after child at {j}");
            let dec = t.node_kind[p] == 0;
            assert_eq!(t.rev_row_of[j] != u32::MAX, dec, "rev block kind at {j}");
            assert_eq!(t.rvd_row_of[j] != u32::MAX, !dec, "rvd block kind at {j}");
        }
        assert_eq!(t.node_parent[0], u32::MAX);
        assert_eq!(t.leaf_coff.len(), 2 * t.rows + 1);
        assert_eq!(t.leaf_coff[2 * t.rows] as usize, t.leaf_cidx.len());
        // BFS order is a permutation of 0..n and levels tile it.
        let mut seen = vec![false; n];
        for &i in &t.bfs_order {
            assert!(!seen[i as usize], "duplicate node in bfs order");
            seen[i as usize] = true;
        }
        assert!(seen.iter().all(|&x| x), "bfs order misses nodes");
        assert_eq!(t.level_start[0], 0);
        assert_eq!(t.level_start[t.nlevels] as usize, n);
        // Every leaf id appears in leaf_rows or term_leaves exactly once.
        let mut leaf_marks = vec![0u8; n];
        for &i in &t.leaf_rows {
            leaf_marks[i as usize] += 1;
        }
        for &i in &t.term_leaves {
            leaf_marks[i as usize] += 1;
        }
        for i in 0..n {
            assert_eq!(
                leaf_marks[i] > 0,
                t.node_leaf[i] == 1,
                "leaf flag mismatch at {i}"
            );
        }
        // terminal utilities are ±1 (or horizon-scaled), and finite.
        for &u in &t.terminal_utility {
            assert!(u.is_finite());
        }
    }
}

/// Randomly generated game states of all shapes must serialize. Run with the
/// real weights if available? No — empty nets are fine: the tree build does
/// not need weights.
#[test]
fn starter_draft_round_trips() {
    let nets = [Nets::default()];
    let mut rng = Rng::new(99);
    let mut checked = 0;
    for seed in 0..10u64 {
        let mut rng = Rng::new(seed.wrapping_mul(31) + 7);
        let gc = GameCfg {
            agents: [Agent::Rebel {
                cfg: cfg(),
                slot: 0,
            }; 2],
            collect: Collect::Rebel,
            explore: 0.5,
            random_draft: seed % 2 == 0,
            eval_mix: 0.0,
            mc_mix: 0.0,
        };
        let roots = collect_roots(3, seed, &nets, &gc, 2);
        for (s, bel) in roots {
            let ctx = warchest::rebel::Ctx::new(&s);
            let sv = Solver::new(&s, ctx, &nets[0], cfg(), bel);
            if sv.capped() {
                continue;
            }
            let job = Job::from_solver(&sv, &[]);
            let back = Job::from_bytes(&job.to_bytes()).unwrap();
            assert_eq!(back.to_bytes(), job.to_bytes());
            checked += 1;
        }
        let _ = rng;
    }
    assert!(checked >= 3);
}
