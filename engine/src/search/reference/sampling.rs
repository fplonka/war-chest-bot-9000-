use crate::rng::Rng;

/// Device arenas replayed by the exact host expansion oracle.
pub struct Arenas<'a> {
    pub reach: &'a [f32],
    pub cur: &'a [f32],
    pub sum: &'a [f32],
    pub qval: &'a [f32],
    pub visits: &'a [f32],
    pub prior: &'a [f32],
}

/// Sum in the same strided butterfly order as a CUDA warp.
pub(super) fn warp32_sum<const K: usize>(n: usize, f: impl Fn(usize) -> [f32; K]) -> [f32; K] {
    let mut lane = [[0.0f32; K]; 32];
    for (thread, acc) in lane.iter_mut().enumerate() {
        let mut i = thread;
        while i < n {
            let value = f(i);
            for k in 0..K {
                acc[k] += value[k];
            }
            i += 32;
        }
    }
    let mut stride = 16;
    while stride > 0 {
        for lane_index in 0..stride {
            for k in 0..K {
                let other = lane[lane_index + stride][k];
                lane[lane_index][k] += other;
            }
        }
        stride >>= 1;
    }
    lane[0]
}

/// Draw from the accepted entries, with a uniform fallback for zero mass.
pub(super) fn pick_live(
    weights: &[f32],
    live: impl Fn(usize) -> bool,
    rng: &mut Rng,
) -> Option<usize> {
    let [total, count] = warp32_sum(weights.len(), |i| {
        if live(i) {
            [weights[i].max(0.0), 1.0]
        } else {
            [0.0, 0.0]
        }
    });
    let count = count as usize;
    if count == 0 {
        return None;
    }
    if !(total > 0.0) {
        let mut selected = rng.below(count);
        for i in 0..weights.len() {
            if live(i) {
                if selected == 0 {
                    return Some(i);
                }
                selected -= 1;
            }
        }
        unreachable!("a live entry was counted");
    }
    let mut last = None;
    let mut needle = rng.unit_f64() * total as f64;
    for (i, &weight) in weights.iter().enumerate() {
        if live(i) {
            last = Some(i);
            needle -= weight.max(0.0) as f64;
            if needle < 0.0 {
                return Some(i);
            }
        }
    }
    last
}

/// Draw from non-negative weights, with a uniform fallback for zero mass.
pub(super) fn pick(weights: &[f32], rng: &mut Rng) -> usize {
    let [total] = warp32_sum(weights.len(), |i| [weights[i].max(0.0)]);
    if !(total > 0.0) {
        return if weights.is_empty() {
            0
        } else {
            rng.below(weights.len())
        };
    }
    let mut needle = rng.unit_f64() * total as f64;
    for (i, &weight) in weights.iter().enumerate() {
        needle -= weight.max(0.0) as f64;
        if needle < 0.0 {
            return i;
        }
    }
    weights.len() - 1
}
