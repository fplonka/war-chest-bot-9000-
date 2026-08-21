//! What a search budget costs, in the units the cards charge for.
//!
//! `SoG(s, c)` is the paper's notation: `s` total expansion simulations at `c`
//! per regret update, so the solve runs `ceil(s / c)` regret updates.
//!
//! The device's kernel table is a handful of terms in a solve's shape:
//!
//! * the trunk runs once per row ever created,
//! * the join runs over every row, twice, every iteration,
//! * the readout and the belief pooling run once per belief-index entry per
//!   iteration,
//! * the two CFR sweeps run once per cell per iteration.
//!
//! So the totals below price a budget without running the farm. What they
//! cannot say is whether the cheaper budget searches as well; that is
//! `budgetq`.
//!
//! `cargo run --release --example budget -- <weights.bin> [roots] [games]`


use rayon::prelude::*;
use warchest::rng::Rng;
use warchest::search::{Cfg, Cfr, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};

/// One budget's cost, summed over the roots it was run on.
#[derive(Default, Clone, Copy)]
struct Cost {
    solves: f64,
    nodes: f64,
    rows: f64,
    ncfg: f64,
    depth: f64,
    /// Rows summed over iterations: the join, twice per row.
    row_iters: f64,
    /// Belief-index entries summed over iterations: the readout and the pool.
    cidx_iters: f64,
    /// Cells summed over iterations: the two sweeps.
    cell_iters: f64,
    /// Distinct public boards among the rows, against the rows themselves.
    /// A trunk that recognised a repeat would run this many times instead.
    distinct: f64,
    /// Distinct config supports among the (row, player) queries, and the
    /// largest group of queries that share one. The readout is a gather
    /// because every query is treated as its own list; queries that share a
    /// support are a matrix multiply `[queries, D] x [D, configs]` instead.
    supports: f64,
    biggest: f64,
    /// Coin plays, not public-tree levels. One coin play is a decision node
    /// plus whatever forced micro-decisions and draws follow it, so a tree
    /// twenty levels deep may be only four plays deep -- and a round is six
    /// plays, which is where beliefs reset.
    plies: f64,
    /// Share of solves whose tree reaches a round-start draw at all.
    crossed: f64,
}

impl Cost {
    fn add(&mut self, o: &Cost) {
        self.solves += o.solves;
        self.nodes += o.nodes;
        self.rows += o.rows;
        self.ncfg += o.ncfg;
        self.depth += o.depth;
        self.row_iters += o.row_iters;
        self.cidx_iters += o.cidx_iters;
        self.cell_iters += o.cell_iters;
        self.distinct += o.distinct;
        self.supports += o.supports;
        self.biggest += o.biggest;
        self.plies += o.plies;
        self.crossed += o.crossed;
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let weights = a.get(1).cloned().unwrap_or_else(|| "weights.bin".into());
    let roots: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(64);
    let games: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(24);

    let nets = std::sync::Arc::new(Nets { value: warchest::net::Net::load_bin(&weights).expect("weights file"), device: false });
    // Roots off real play under the same net, at a cheap budget. What a solve
    // costs varies twenty-six fold with how far into a game its root sits, so
    // the corpus has to be a sample of play and not of one phase.
    let small = Cfg { s: 32, c: 4.0, cfr: Cfr::SOG, ..Default::default() };
    let gc = GameCfg {
        agents: [Agent::Sog { cfg: small }; 2],
        collect: Collect::Sog,
        explore: 0.1,
        random_draft: true,
        p_td1: 0.0,
        query_rate: 0.9,
        recursive_rate: 0.1,
    };
    // The same corpus the farm bench solves, when one is given: a shape
    // measured on one set of roots and a rate measured on another cannot be
    // divided into each other.
    //
    // Otherwise roots off fresh play. `collect_roots` concatenates whole games
    // and then truncates, so asking for a small cap takes every root from the
    // first game or two -- and a solve's cost varies twenty-six fold with how
    // far into a game its root sits. Take the whole harvest and stride it.
    let all = match std::env::var("WARCHEST_ROOTS") {
        Ok(path) => {
            let f = std::fs::File::open(&path).expect("roots corpus");
            warchest::roots::read_roots(&mut std::io::BufReader::new(f)).expect("roots corpus")
        }
        Err(_) => collect_roots(games, 99, &nets, &gc, usize::MAX),
    };
    let step = (all.len() / roots.max(1)).max(1);
    let positions: Vec<_> = all.into_iter().step_by(step).take(roots).collect();
    println!("{} roots from {games} games\n", positions.len());

    // `s` total expansions and `c` per regret update, the paper's axes.
    // Student of Games trains chess and Go at (400, 1) and poker at (10, 0.01);
    // this engine's default is (512, 8).
    let budgets: Vec<(u32, f32)> = vec![
        (512, 8.0),
        (512, 4.0),
        (512, 1.0),
        (256, 2.0),
        (256, 1.0),
        (128, 1.0),
        (64, 1.0),
    ];

    println!(
        "{:>12} {:>6} {:>6} {:>7} {:>6} {:>6} {:>5} | {:>9} {:>10} {:>10} | {:>7}",
        "SoG(s,c)", "iters", "nodes", "rows", "ncfg", "dist%", "depth",
        "joinrows", "readouts", "sweepcell", "f MB"
    );
    let mut base: Option<Cost> = None;
    for &(s, c) in &budgets {
        let cfg = Cfg { s, c, cfr: Cfr::SOG, ..Default::default() };
        let iters = cfg.iters();
        let t0 = std::time::Instant::now();
        // Per solve, not only the mean: cost varies twenty-six fold with how
        // far into a game a root sits, and it is the tail that fills a card.
        let each: Vec<Cost> = positions
            .par_iter()
            .enumerate()
            .map(|(i, (st, bel))| {
                let ctx = warchest::pbs::Ctx::new(st);
                let mut sv = Solver::new(st, ctx, std::sync::Arc::clone(&nets), cfg, bel.clone(), Rng::new(0x51D5 ^ i as u64));
                let mut one = Cost { solves: 1.0, ..Default::default() };
                // The production loop, not a transcription of it: `solve` also
                // refreshes the policy prior between phases, and the prior is
                // what decides where the tree goes.
                sv.run_alone();
                let (sh, tr) = (sv.shape(), sv.trace);
                one.row_iters = tr.row_iters as f64;
                one.cidx_iters = tr.cidx_iters as f64;
                one.cell_iters = tr.cell_iters as f64;
                one.nodes = sh.nodes as f64;
                one.rows = sh.rows as f64;
                one.ncfg = sh.ncfg as f64;
                one.depth = sh.depth as f64;
                // How many of the rows stand on a public state the trunk has
                // not already seen. The solver interns them, so this is what
                // it kept.
                one.distinct = sh.boards as f64;
                // A node's children carry the *same* support for the idle
                // player, and the tree shares those lists by pointer rather
                // than copying them, so identity is the grouping.
                let mut sup: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::new();
                for &node in sv.leaf_rows.iter() {
                    for p in 0..2 {
                        let key = std::sync::Arc::as_ptr(&sv.nodes[node].cfgs[p]) as *const u8 as usize;
                        *sup.entry(key).or_default() += 1;
                    }
                }
                // Coin plays deep, and whether a round boundary is inside the
                // tree at all.
                let root_plays = st.main_plays;
                let mut deepest = 0i32;
                let mut crossed = false;
                for w in sv.states.iter() {
                    deepest = deepest.max(w.main_plays as i32 - root_plays as i32);
                    crossed |= matches!(w.pending(), warchest::state::Cont::Draw { .. });
                }
                one.plies = deepest as f64;
                one.crossed = crossed as u8 as f64;
                one.supports = sup.len() as f64;
                one.biggest = sup.values().copied().max().unwrap_or(0) as f64;
                one
            })
            .collect();
        let mut tot = Cost::default();
        for o in &each {
            tot.add(o);
        }
        let n = tot.solves.max(1.0);
        // `f` is `[ncfg, D]` in half precision: the readout's working set.
        let fmb = tot.ncfg / n * warchest::net::D as f64 * 2.0 / 1e6;
        let line = format!(
            "{:>12} {:>6} {:>6.0} {:>7.0} {:>6.0} {:>5.0}% {:>5.1} | {:>9.0} {:>10.0} {:>10.0} | {:>7.2}",
            format!("({s},{c})"),
            iters,
            tot.nodes / n,
            tot.rows / n,
            tot.ncfg / n,
            100.0 * tot.distinct / tot.rows.max(1.0),
            tot.depth / n,
            tot.row_iters / n,
            tot.cidx_iters / n,
            tot.cell_iters / n,
            fmb,
        );
        let line = format!(
            "{line}  sup {:.0} ({:.1} q each, max {:.0})  plies {:.1} cross {:.0}%",
            tot.supports / n,
            2.0 * tot.rows / tot.supports.max(1.0),
            tot.biggest / n,
            tot.plies / n,
            100.0 * tot.crossed / n,
        );
        // Wall over the whole corpus on every core: not the device's cost, but
        // the same ratios, and it says what the quality study can afford.
        let line = format!("{line}  {:.1} core-s/solve",
            t0.elapsed().as_secs_f64() * rayon::current_num_threads() as f64 / n);
        // The tail, which is what a card has to be sized for.
        let pct = |f: fn(&Cost) -> f64, q: f64| -> f64 {
            let mut v: Vec<f64> = each.iter().map(f).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((v.len() - 1) as f64 * q) as usize]
        };
        let line = format!(
            "{line}\n{:>12} tail: cells p50 {:.0} p90 {:.0} max {:.0} | ncfg p50 {:.0} p90 {:.0} max {:.0} | cfg/row p50 {:.1} max {:.1}",
            "",
            pct(|c| c.cell_iters, 0.5), pct(|c| c.cell_iters, 0.9), pct(|c| c.cell_iters, 1.0),
            pct(|c| c.ncfg, 0.5), pct(|c| c.ncfg, 0.9), pct(|c| c.ncfg, 1.0),
            pct(|c| c.cidx_iters / c.row_iters.max(1.0), 0.5),
            pct(|c| c.cidx_iters / c.row_iters.max(1.0), 1.0),
        );
        match &base {
            None => {
                println!("{line}   baseline");
                base = Some(tot);
            }
            Some(b) => {
                // A crude device price: the join and the trunk are arithmetic,
                // the readout is bytes, the sweeps are latency, and at the
                // profile in `docs/THROUGHPUT.md` they come to roughly a third
                // each. Weighted that way so one number ranks budgets.
                let f = |x: f64, y: f64| y / x.max(1.0);
                let rel = (f(b.rows / b.solves, tot.rows / n)
                    + f(b.row_iters / b.solves, tot.row_iters / n)
                    + f(b.cidx_iters / b.solves, tot.cidx_iters / n)
                    + f(b.cell_iters / b.solves, tot.cell_iters / n))
                    / 4.0;
                println!("{line}   {:.2}x cost", rel);
            }
        }
    }
}
