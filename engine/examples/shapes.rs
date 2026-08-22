//! The distribution of what a solve builds, over a corpus of real roots.
//!
//! Admission sizes a slot from this: a slot holds a solve at the budget, so the
//! budget has to be a percentile of the shape and not a mean. Prints, for each
//! `s` given, the percentiles of every term a slot's arenas are linear in.
//!
//!   cargo run --release --example shapes -- [roots] [s...]
use std::sync::Arc;
use warchest::rng::Rng;
use warchest::search::{Cfg, Cfr, Nets, Shape, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};

fn pct(v: &[usize], q: f64) -> usize {
    if v.is_empty() {
        return 0;
    }
    let i = ((v.len() - 1) as f64 * q).round() as usize;
    v[i]
}

fn row(name: &str, mut v: Vec<usize>) {
    v.sort_unstable();
    let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
    println!(
        "{name:>10} {:>10.0} {:>10} {:>10} {:>10} {:>10} {:>10}",
        mean,
        pct(&v, 0.50),
        pct(&v, 0.90),
        pct(&v, 0.99),
        pct(&v, 1.0),
        pct(&v, 1.0) * 100 / pct(&v, 0.99).max(1),
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let roots: usize = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(64);
    let sizes: Vec<u32> = if a.len() > 2 {
        a[2..].iter().filter_map(|x| x.parse().ok()).collect()
    } else {
        vec![512, 192]
    };

    let net = {
        let mut r = Rng::new(0x2E57);
        let l = warchest::net::NetLayout::new();
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect()
        };
        let (w, b) = (draw(l.w_len), draw(l.b_len));
        let mut ln = vec![0.0; l.ln_len];
        for n in &l.norms {
            ln[n.g..n.g + n.width].fill(1.0);
        }
        warchest::net::Net::from_flat(&w, &b, &ln).expect("net")
    };
    let nets = Arc::new(Nets { value: net, device: false });
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
    // A spread of the whole game, not of its openings: a solve's cost varies
    // twenty-six fold with how far into a game its root sits.
    let all = collect_roots(64, 99, &nets, &gc, usize::MAX);
    let step = (all.len() / roots.max(1)).max(1);
    let positions: Vec<_> = all.into_iter().step_by(step).take(roots).collect();
    println!("{} roots", positions.len());

    for &s in &sizes {
        let cfg = Cfg { s, c: 8.0, cfr: Cfr::SOG, ..Default::default() };
        let mut shapes: Vec<Shape> = Vec::new();
        let mut host: Vec<usize> = Vec::new();
        for (i, (st, belief)) in positions.iter().enumerate() {
            let ctx = warchest::pbs::Ctx::new(st);
            let mut sv = Solver::new(
                st,
                ctx,
                Arc::clone(&nets),
                cfg,
                belief.clone(),
                Rng::new(i as u64 * 7 + 1),
            );
            sv.collect(4);
            sv.run_alone();
            shapes.push(sv.shape());
            host.push(sv.host_bytes());
        }
        println!(
            "\n== s={s} c=8 batch=8 ==\n{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "", "mean", "p50", "p90", "p99", "max", "max/p99%"
        );
        row("nodes", shapes.iter().map(|x| x.nodes).collect());
        row("rows", shapes.iter().map(|x| x.rows).collect());
        row("boards", shapes.iter().map(|x| x.boards).collect());
        row("cells", shapes.iter().map(|x| x.cells).collect());
        row("ncfg", shapes.iter().map(|x| x.ncfg).collect());
        row("cidx", shapes.iter().map(|x| x.cidx).collect());
        row("acts", shapes.iter().map(|x| x.acts).collect());
        row("support", shapes.iter().map(|x| x.support).collect());
        row("reach", shapes.iter().map(|x| x.reach).collect());
        row("vals", shapes.iter().map(|x| x.vals).collect());
        row("draws", shapes.iter().map(|x| x.draws).collect());
        row("depth", shapes.iter().map(|x| x.depth).collect());
        row("host_KB", host.iter().map(|x| x / 1024).collect());
    }
}
