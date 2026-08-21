//! What one solve holds in host memory, group by group.
use std::sync::Arc;
use warchest::rng::Rng;
use warchest::search::{Cfg, Cfr, Nets, Solver};
use warchest::selfplay::{collect_roots, Agent, Collect, GameCfg};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let roots: usize = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(16);
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
    let all = collect_roots(24, 99, &nets, &gc, usize::MAX);
    let step = (all.len() / roots.max(1)).max(1);
    let positions: Vec<_> = all.into_iter().step_by(step).take(roots).collect();

    let s: u32 = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(512);
    let c: f32 = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(8.0);
    let cfg = Cfg { s, c, cfr: Cfr::SOG, ..Default::default() };
    let mut total: Vec<(String, f64)> = Vec::new();
    let (mut n, mut nodes) = (0.0f64, 0.0f64);
    let (mut before, mut after) = (0.0f64, 0.0f64);
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
        let census = sv.host_census();
        let group = |name: &str| -> f64 {
            census.iter().find(|e| e.0 == name).map_or(0.0, |e| e.1 as f64)
        };
        let host = sv.host_bytes() as f64;
        // What the same tree costs on the device path, before and after. The
        // readout buffers were empty there either way except `xb`, which growth
        // used to fit; the CFR arenas were grown and never read; `avg` was
        // allocated whole to hold the root's row; and the flat description of
        // the tree, which only the device path builds, is added.
        let cfr_arenas: f64 = ["regret", "prior", "visits", "qval", "sum", "reach", "vals"]
            .iter()
            .map(|k| group(k))
            .sum();
        let contract = warchest::contract::Contract::of(&sv).bytes() as f64;
        let xb = (sv.xb.capacity() * 4) as f64;
        let root_row = (sv.nodes[0].legal_action.len() * 4) as f64;
        before += host - group("readout") + xb + contract;
        after += host - group("readout") - cfr_arenas - group("avg") + root_row + contract;
        for (k, b) in census {
            match total.iter_mut().find(|e| e.0 == k) {
                Some(e) => e.1 += b as f64,
                None => total.push((k.to_string(), b as f64)),
            }
        }
        nodes += sv.nodes.len() as f64;
        n += 1.0;
    }
    total.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mb = |x: f64| x / n / 1e6;
    let sum: f64 = total.iter().map(|e| e.1).sum();
    println!("SoG({s},{c}): {n} solves, {:.0} nodes each\n", nodes / n);
    println!("host path, group by group:");
    for (k, b) in &total {
        println!("{k:>10} {:8.3} MB", mb(*b));
    }
    println!("{:>10} {:8.3} MB\n", "total", mb(sum));
    println!("device path per solve: {:8.3} MB before, {:8.3} MB after", mb(before), mb(after));
}
