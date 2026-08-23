//! The distribution of what a solve builds, over a corpus of real roots.
//!
//!   cargo run --release --example shapes -- [roots] [s... | first]
use std::sync::Arc;
use std::time::Instant;
use warchest::pbs::{enumerate_configs, reserve, true_config, Belief, Ctx};
use warchest::rng::Rng;
use warchest::search::{Cfg, Cfr, Nets, Shape, Solver};
use warchest::selfplay::make_game;
use warchest::state::{Cont, State};

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

/// Main-play positions from random legal play of whole games.
///
/// `collect_roots` would solve every ply; that is the thing being measured, so
/// it cannot be how the corpus is gathered. Random play still sits at every
/// depth, which is what a first expansion's cost varies with.
fn random_roots(games: usize, seed: u64) -> Vec<(State, [Belief; 2])> {
    let mut out = Vec::new();
    for g in 0..games {
        let mut rng = Rng::new(seed + g as u64);
        let mut s = make_game(&mut rng, true);
        let mut ply = 0u32;
        while !s.is_terminal() && ply < 256 {
            if matches!(s.pending(), Cont::MainPlay) && !s.is_chance() {
                let ctx = Ctx::new(&s);
                let bel = [uniform_belief(&s, &ctx, 0), uniform_belief(&s, &ctx, 1)];
                out.push((s.clone(), bel));
            }
            let acts = s.legal_actions();
            if acts.is_empty() {
                break;
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
            ply += 1;
        }
    }
    out
}

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

/// Coin-play depth, round depth (draw_steps on the deepest path), then
/// chance / walked-through / decision-leaf counts.
fn mix(sv: &Solver) -> (usize, usize, usize, usize, usize) {
    let n = sv.nodes.len();
    let mut chance_n = 0usize;
    let mut walked_n = 0usize;
    let mut dleaf_n = 0usize;
    for i in 0..n {
        let nd = &sv.nodes[i];
        let st = &sv.states[i];
        if nd.chance {
            chance_n += 1;
        } else if nd.leaf && st.is_valued() {
            dleaf_n += 1;
        } else if !nd.leaf && !st.is_valued() {
            walked_n += 1;
        }
    }
    fn rec(sv: &Solver, i: usize, coin: usize, round: usize) -> (usize, usize) {
        let st = &sv.states[i];
        let nd = &sv.nodes[i];
        let coin = coin + usize::from(matches!(st.pending(), Cont::MainPlay));
        let round = round + if nd.chance { (nd.draw_steps as usize).max(1) } else { 0 };
        let mut best = (coin, round);
        for &ch in &nd.child {
            let (c, r) = rec(sv, ch, coin, round);
            best.0 = best.0.max(c);
            best.1 = best.1.max(r);
        }
        best
    }
    let (d_coin, d_round) = if n == 0 { (0, 0) } else { rec(sv, 0, 0, 0) };
    (d_coin, d_round, chance_n, walked_n, dleaf_n)
}

struct Fat {
    i: usize,
    shape: Shape,
    round: u16,
    main_plays: u16,
    turns: [u8; 2],
    pending: String,
    to_act: u8,
    hands: [u8; 2],
    support: [usize; 2],
    legal: usize,
    chance: usize,
    decision: usize,
    leaves: usize,
    root_cells: usize,
    max_node_cells: usize,
    max_draw: usize,
    max_draw_row: usize,
    exhausted: bool,
    interiors: String,
}

fn describe(i: usize, st: &warchest::state::State, belief: &[warchest::pbs::Belief; 2], sv: &Solver) -> Fat {
    let mut chance = 0usize;
    let mut decision = 0usize;
    let mut leaves = 0usize;
    let mut max_node_cells = 0usize;
    let mut max_draw = 0usize;
    let mut max_draw_row = 0usize;
    let mut interiors: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for (ni, n) in sv.nodes.iter().enumerate() {
        if n.leaf {
            leaves += 1;
        } else if n.chance {
            chance += 1;
            max_draw = max_draw.max(n.draw.len());
            for ci in 0..n.draw.rows() {
                max_draw_row = max_draw_row.max(n.draw.row(ci).0.len());
            }
            *interiors.entry(format!("{:?}", sv.states[ni].pending())).or_insert(0) += 1;
        } else {
            decision += 1;
            *interiors.entry(format!("{:?}", sv.states[ni].pending())).or_insert(0) += 1;
        }
        max_node_cells = max_node_cells.max(n.legal_action.len());
    }
    Fat {
        i,
        shape: sv.shape(),
        round: st.round,
        main_plays: st.main_plays,
        turns: st.turns_taken,
        pending: format!("{:?}", st.pending()),
        to_act: st.to_act(),
        hands: [st.hand_size(0), st.hand_size(1)],
        support: [belief[0].cfg.len(), belief[1].cfg.len()],
        legal: st.legal_actions().len(),
        chance,
        decision,
        leaves,
        root_cells: sv.nodes[0].legal_action.len(),
        max_node_cells,
        max_draw,
        max_draw_row,
        exhausted: sv.nodes[0].exhausted,
        interiors: interiors
            .iter()
            .map(|(k, n)| format!("{k}×{n}"))
            .collect::<Vec<_>>()
            .join(","),
    }
}

fn print_fat(title: &str, fats: &mut [Fat], key: impl Fn(&Fat) -> usize) {
    fats.sort_by_key(|f| std::cmp::Reverse(key(f)));
    println!("\n== {title} ==");
    for f in fats.iter().take(8) {
        println!(
            "root {:>3} cells={:<7} draws={:<7} cidx={:<7} reach={:<7} nodes={:<5} rows={:<5} ncfg={:<5} \
             round={} plays={} turns={:?} pending={} to_act={} hands={:?} support={:?} legal={} \
             chance={} decision={} leaves={} root_cells={} max_node_cells={} max_draw={} max_draw_row={} exhausted={} interiors={}",
            f.i,
            f.shape.cells,
            f.shape.draws,
            f.shape.cidx,
            f.shape.reach,
            f.shape.nodes,
            f.shape.rows,
            f.shape.ncfg,
            f.round,
            f.main_plays,
            f.turns,
            f.pending,
            f.to_act,
            f.hands,
            f.support,
            f.legal,
            f.chance,
            f.decision,
            f.leaves,
            f.root_cells,
            f.max_node_cells,
            f.max_draw,
            f.max_draw_row,
            f.exhausted,
            f.interiors,
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let roots: usize = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(96);
    let first_only = a.iter().any(|x| x == "first");
    let skip_first = a.iter().any(|x| x == "nofirst");
    let sizes: Vec<u32> = if first_only {
        vec![]
    } else if a.len() > 2 {
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
    let all = random_roots(64, 99);
    let step = (all.len() / roots.max(1)).max(1);
    let positions: Vec<_> = all.into_iter().step_by(step).take(roots).collect();
    println!("{} roots from {} random games ({} plies kept)", positions.len(), 64, positions.len() * step);

    if skip_first {
        println!("(skipping first-expansion census)");
    } else {
    let cfg = Cfg {
        s: 1,
        c: 1.0,
        cfr: Cfr::SOG,
        budget: warchest::search::Budget::unbounded(),
        ..Default::default()
    };
    let mut first: Vec<Shape> = Vec::new();
    let mut fats: Vec<Fat> = Vec::new();
    let mut pending: Vec<(String, usize)> = Vec::new();
    for (i, (st, belief)) in positions.iter().enumerate() {
        let ctx = warchest::pbs::Ctx::new(st);
        let sv = Solver::new(
            st,
            ctx,
            Arc::clone(&nets),
            cfg,
            belief.clone(),
            Rng::new(i as u64 * 7 + 1),
        );
        let f = describe(i, st, belief, &sv);
        if i % 8 == 0 || f.shape.cells > 200_000 {
            eprintln!(
                "root {i} cells={} draws={} nodes={} pending={} support={:?}",
                f.shape.cells, f.shape.draws, f.shape.nodes, f.pending, f.support
            );
        }
        pending.push((f.pending.clone(), f.shape.cells));
        first.push(f.shape);
        fats.push(f);
    }
    println!(
        "\n== first expansion ==\n{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "mean", "p50", "p90", "p99", "max", "max/p99%"
    );
    row("nodes", first.iter().map(|x| x.nodes).collect());
    row("rows", first.iter().map(|x| x.rows).collect());
    row("boards", first.iter().map(|x| x.boards).collect());
    row("cells", first.iter().map(|x| x.cells).collect());
    row("ncfg", first.iter().map(|x| x.ncfg).collect());
    row("cidx", first.iter().map(|x| x.cidx).collect());
    row("reach", first.iter().map(|x| x.reach).collect());
    row("draws", first.iter().map(|x| x.draws).collect());
    row("acts", first.iter().map(|x| x.acts).collect());
    row("support", first.iter().map(|x| x.support).collect());
    row("root_cell", fats.iter().map(|x| x.root_cells).collect());
    row("max_node", fats.iter().map(|x| x.max_node_cells).collect());
    row("decision", fats.iter().map(|x| x.decision).collect());
    row("chance", fats.iter().map(|x| x.chance).collect());
    row("leaves", fats.iter().map(|x| x.leaves).collect());

    let mut by_pending: std::collections::BTreeMap<String, (usize, usize, usize)> =
        std::collections::BTreeMap::new();
    for (p, cells) in &pending {
        let e = by_pending.entry(p.clone()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += *cells;
        e.2 = e.2.max(*cells);
    }
    println!("\n== first expansion by pending ==");
    for (p, (n, sum, max)) in &by_pending {
        println!(
            "{p:<40} n={n:<4} mean_cells={:<8.0} max_cells={max}",
            *sum as f64 / *n as f64
        );
    }

    print_fat("fattest first expansions by cells", &mut fats, |f| f.shape.cells);
    print_fat("fattest first expansions by draws", &mut fats, |f| f.shape.draws);
    }

    for &s in &sizes {
        let cfg = Cfg {
            s,
            c: 8.0,
            cfr: Cfr::SOG,
            budget: warchest::search::Budget::unbounded(),
            ..Default::default()
        };
        let mut shapes: Vec<Shape> = Vec::new();
        let mut host: Vec<usize> = Vec::new();
        let mut coin_d = Vec::new();
        let mut round_d = Vec::new();
        let mut chance_p = Vec::new();
        let mut walked_p = Vec::new();
        let mut dleaf_p = Vec::new();
        let mut ms = Vec::new();
        for (i, (st, belief)) in positions.iter().enumerate() {
            let ctx = warchest::pbs::Ctx::new(st);
            let t0 = Instant::now();
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
            ms.push(t0.elapsed().as_millis() as usize);
            let n = sv.nodes.len().max(1);
            let (dc, dr, ch, wk, dl) = mix(&sv);
            coin_d.push(dc);
            round_d.push(dr);
            chance_p.push(ch * 100 / n);
            walked_p.push(wk * 100 / n);
            dleaf_p.push(dl * 100 / n);
            shapes.push(sv.shape());
            host.push(sv.host_bytes());
            if i % 8 == 0 {
                eprintln!("s={s} root {i}/{} {}ms nodes={}", positions.len(), ms[i], n);
            }
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
        row("coin_d", coin_d);
        row("round_d", round_d);
        row("chance%", chance_p);
        row("walked%", walked_p);
        row("dleaf%", dleaf_p);
        row("ms", ms);
        row("host_KB", host.iter().map(|x| x / 1024).collect());
    }
}
