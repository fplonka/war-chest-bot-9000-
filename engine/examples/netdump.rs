//! A deterministic dump of the network's output, for checking that a
//! performance change did not move a single float.
//!
//! `docs/PERF.md`'s rule: an optimisation not meant to change results must be
//! shown not to. Build at the old commit, run, build at the new one, run, diff.
use warchest::board::N_HEXES;
use warchest::net::Mlp;
use warchest::rebel::{AFEAT, CFEAT, HEX_CH, HEX_FACTS, LOOSE, NTYPE, PILE_COUNTS, PUBFEAT};
use warchest::rng::Rng;
use warchest::units::{CARD_FEATS, N_UNITS};

fn main() {
    let (h, dg, rk, de, dc) = (128usize, 32usize, 48usize, 16usize, 24usize);
    let dims = [PUBFEAT, h, h, CFEAT, dg, rk, AFEAT, de, dc];
    let xd = N_HEXES * (HEX_FACTS + de) + 2 * de + LOOSE;
    let nw = CARD_FEATS * dc + dc * de + N_UNITS * de + (PILE_COUNTS + de) * de
        + xd * h + h * h + 2 * dg * h + (4 + de) * dg + dg * (rk + 1) + h * rk
        + (AFEAT + de) * rk + dg * rk + h * rk;
    let mut r = Rng::new(12345);
    let mut draw = |n: usize| -> Vec<f32> {
        (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.4).collect()
    };
    let w = draw(nw);
    let b = draw(dc + de + de + h + h + dg + (rk + 1) + 4 * rk);
    let mut ln = Vec::new();
    for _ in 0..2 {
        ln.extend(std::iter::repeat(1.0).take(h));
        ln.extend(std::iter::repeat(0.0).take(h));
    }
    let net = Mlp::from_flat(&dims, &w, &b, &ln).expect("net");

    // Rows shaped like the encoder's: a genuine one-hot per hex occupant.
    let rows = 24usize;
    let mut xpub = draw(rows * PUBFEAT);
    for r0 in 0..rows {
        for hx in 0..N_HEXES {
            let at = r0 * PUBFEAT + hx * HEX_CH + HEX_FACTS;
            for k in 0..NTYPE {
                xpub[at + k] = 0.0;
            }
            let occ = (r0 * 7 + hx * 3) % (NTYPE + 1);
            if occ < NTYPE {
                xpub[at + occ] = 1.0;
            }
        }
    }
    let xbel = draw(rows * 2 * dg);
    let ids: Vec<u8> = (0..rows * NTYPE).map(|i| (i % N_UNITS) as u8).collect();
    let mut phi = draw(rows * CFEAT);
    for r0 in 0..rows {
        phi[r0 * CFEAT + CFEAT - 1] = (r0 % 2) as f32;
    }
    for (i, x) in net.forward(&xpub, &xbel, &phi, &ids, rows).iter().enumerate() {
        println!("{i:3} {x:.9}");
    }
}
