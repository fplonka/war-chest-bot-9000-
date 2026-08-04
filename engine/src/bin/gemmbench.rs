//! Where the value-network time actually goes, piece by piece.
//!
//! If the matmuls are near peak and the elementwise passes are not, shrinking
//! the network is the wrong lever; if a particular gemm shape is far off, the
//! fix is to reshape the call rather than the model.

use std::time::Instant;
use warchest::net::{gemm_probe, Mlp};

fn mk(dims: &[usize], split: usize) -> Mlp {
    let mut m = Mlp {
        dims: dims.to_vec(),
        w: Vec::new(),
        b: Vec::new(),
        ln_w: Vec::new(),
        ln_b: Vec::new(),
        split,
        wb: (0..split * dims[1]).map(|i| (i % 13) as f32 * 1e-4).collect(),
    };
    let mut x = 12345u64;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 40) as f32 / 1e6 - 0.01
    };
    for l in 0..dims.len() - 1 {
        m.w.push((0..dims[l] * dims[l + 1]).map(|_| rnd()).collect());
        m.b.push((0..dims[l + 1]).map(|_| rnd()).collect());
        if l + 2 < dims.len() {
            m.ln_w.push(vec![1.0; dims[l + 1]]);
            m.ln_b.push(vec![0.0; dims[l + 1]]);
        }
    }
    m
}

fn bench(name: &str, macs: f64, reps: usize, mut f: impl FnMut()) {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "  {:<22} {:8.1} us  {:8.1} Gflop/s",
        name,
        s / reps as f64 * 1e6,
        2.0 * macs * reps as f64 / s / 1e9
    );
}

fn main() {
    let feat = 812usize;
    let split = 132usize;
    let h: usize = std::env::args()
        .nth(1)
        .and_then(|x| x.parse().ok())
        .unwrap_or(384);
    let m = mk(&[feat - split, h, h, 112], split);
    for &rows in &[368usize, 1472] {
        println!("rows = {}", rows);
        let xpub: Vec<f32> = (0..rows * (feat - split)).map(|i| (i % 7) as f32 * 0.1).collect();
        let xbel: Vec<f32> = (0..rows * split).map(|i| (i % 5) as f32 * 0.1).collect();
        let (mut sb, mut ob, mut h0) = (vec![0.0f32; rows * h], vec![0.0f32; rows * h], Vec::new());
        m.trunk(&xpub, rows, feat - split, &mut sb, &mut h0);
        let mut a = vec![0.0f32; rows * h];
        let mut b = vec![0.0f32; rows * h];
        let reps = 400;

        bench("gemm belief k=132", (rows * split * h) as f64, reps, || {
            gemm_probe(rows, h, split, &xbel, split, &m.wb, h, &mut a, h);
        });
        bench("gemm hidden k=h", (rows * h * h) as f64, reps, || {
            gemm_probe(rows, h, h, &a, h, &m.w[1], h, &mut b, h);
        });
        bench("gemm head k=h n=56", (rows * h * 56) as f64, reps, || {
            gemm_probe(rows, 56, h, &b, h, &m.w[2], 112, &mut ob, 56);
        });
        bench("activate(LN+relu)", (rows * h * 6) as f64, reps, || {
            m.activate_probe(0, rows, Some(&h0), &mut a);
        });
        bench("full split-forward", ((split * h + h * 56) * rows) as f64, reps, || {
            m.forward_split(&xbel, rows, &h0, 0..56, &mut sb, &mut ob);
        });
        bench("trunk (public tower)", (rows * ((feat - split) * h + h * h)) as f64, 100, || {
            m.trunk(&xpub, rows, feat - split, &mut sb, &mut h0);
        });
    }
}
