//! GPU tree sizing (pre-CUDA plan section 8): build depth-2/3/4 subgame
//! trees from real collected roots with an all-zero network, and report the
//! shape statistics the GPU pool is sized from.
//!
//! Usage: `treesize <roots.bin> [depths]` — depths default to "2,3,4".
//! Roots come from `warchest.save_roots` (section 6).
//!
//! Reported per depth, over all roots: median / p95 / p99 of nodes, leaves,
//! action cells (sum of config x action per decision node), private configs
//! (sum of both supports), an estimate of the bytes a flat tree upload would
//! carry (see docs/TREE.md for the array set), and the CPU build time. The
//! live GPU pool is sized from p99, not the average.

use std::time::Instant;

use warchest::rebel::*;
use warchest::roots;
use warchest::search::{Cfg, Nets, Solver};

/// The flat upload contract of docs/TREE.md, as bytes per tree, sized from
/// the arrays a solver already builds. Rough but honest: the point is
/// p99-vs-average, not a byte-exact bill.
fn uploaded_bytes(nodes: usize, leaves: usize, cells: usize, cfgs: usize, snaps: usize) -> usize {
    // Per node: flags/kind/player (8) + child offsets (4 each) + obs maps
    // (12 per action). Per cell: legal bit + trans u32. Per config across
    // the tree: reach f32 x2. Per leaf: public row (4 x PUBFEAT) + config
    // indices (4 per config). Per snapshot: the strategy cells (4 f32 each).
    let per_node_fixed = 8usize;
    let per_action = 12usize;
    let per_cell = 5usize;
    let per_cfg = 8usize;
    let per_leaf_row = 4 * PUBFEAT;
    let per_snap_cell = 4usize;
    nodes * per_node_fixed
        + cells / 8 // legal bits
        + cells * per_cell
        + cfgs * per_cfg
        + leaves * per_leaf_row
        + snaps * cells * per_snap_cell
        + nodes * per_action
}

fn percentile(sorted: &[usize], q: f64) -> usize {
    let i = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[i]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or_else(|| "roots.bin".into());
    let depths: Vec<usize> = a
        .get(2)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3, 4]);
    // Optional root limit: depth 4 on random-play roots is heavy, so a
    // smaller sample is a reasonable preliminary. 0 = all.
    let max_roots: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    let f = std::fs::File::open(&path).expect("roots file");
    let mut r = std::io::BufReader::new(f);
    let roots = roots::read_roots(&mut r).expect("roots file");
    println!("{} roots from {}", roots.len(), path);
    assert!(!roots.is_empty(), "no roots to size");

    let nets = Nets::default(); // all-zero: identical games, full matmul work
    let iters = 64usize;
    // The tail of the tree-size distribution is fat — a handful of roots
    // explode (random-play beliefs stay broad). The tool caps the build so
    // the table is computable; the cap-hit rate is reported beside the
    // percentiles, and the trained-run roots (plan section 6) are tamer.
    let node_cap: usize = std::env::var("TREESIZE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    for depth in depths {
        // The solver's own node cap bounds the build (the tool's post-build
        // cap check below only decides what to report).
        let cfg = Cfg {
            depth,
            iters,
            snapshots: true,
            node_cap,
            ..Default::default()
        };
        let (mut ns, mut ls, mut cs, mut lcs, mut fs, mut bs, mut times) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut capped = 0usize;
        'roots: for (s, bel) in roots.iter().take(if max_roots == 0 {
            roots.len()
        } else {
            max_roots
        }) {
            let ctx = Ctx::new(s);
            let t0 = Instant::now();
            let sv = Solver::new(s, ctx, &nets, cfg, bel.clone());
            times.push(t0.elapsed().as_micros() as usize);
            if sv.capped() {
                capped += 1;
                continue 'roots;
            }
            let (mut nodes, mut leaves, mut cells, mut legal_cells, mut cfgs) = (0, 0, 0, 0, 0);
            for n in &sv.nodes {
                nodes += 1;
                if n.leaf {
                    leaves += 1;
                } else if !n.chance {
                    cells += n.na() * n.nc(n.player as usize);
                    legal_cells += n.legal_action.len();
                }
                cfgs += n.nc(0) + n.nc(1);
            }
            ns.push(nodes);
            ls.push(leaves);
            cs.push(cells);
            lcs.push(legal_cells);
            fs.push(cfgs);
            bs.push(uploaded_bytes(
                nodes,
                leaves,
                cells,
                cfgs,
                sv.snapshot_count(),
            ));
        }
        let n_done = roots.len().min(if max_roots == 0 {
            roots.len()
        } else {
            max_roots
        });
        if capped > 0 {
            println!("depth {depth}  {capped}/{n_done} roots hit the {node_cap}-node cap");
        }
        for (name, v) in [
            ("nodes", &ns),
            ("leaves", &ls),
            ("action cells", &cs),
            ("legal cells", &lcs),
            ("configs", &fs),
            ("upload MB", &bs),
            ("build ms", &times),
        ] {
            let mut v = v.clone();
            v.sort_unstable();
            let scale = if name == "upload MB" {
                1.0 / 1e6
            } else if name == "build ms" {
                1.0 / 1000.0
            } else {
                1.0
            };
            println!(
                "depth {depth}  {name:12} med {:9.1}  p95 {:9.1}  p99 {:9.1}",
                percentile(&v, 0.50) as f64 * scale,
                percentile(&v, 0.95) as f64 * scale,
                percentile(&v, 0.99) as f64 * scale,
            );
        }
    }
}
