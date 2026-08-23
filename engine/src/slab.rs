//! Size-class slabs: one per solve, eight entity regions inside.
//!
//! Finished-solve device bytes from `runs/unc5d` (epochs after t=60; per-entity
//! percentiles, so the joint total is a lower bound). p50 ≈ 30 MiB, p90 ≈ 100
//! MiB, p99 sat on the old caps. Counts are one mix; carve scales them to the
//! memory that remains after weights and round scratch.

use crate::net::{D, JW, POOL};

pub const CLASSES: [usize; 6] = [
    16 << 20,
    32 << 20,
    64 << 20,
    128 << 20,
    256 << 20,
    512 << 20,
];

/// Relative mix. Carve scales this table to fill the card.
pub const COUNTS: [usize; 6] = [32, 24, 12, 6, 2, 1];

/// `COUNTS` scaled so slabs plus `pipes * extra` bytes of round scratch per
/// slab fill `usable` bytes.
pub fn mix(usable: u64, extra: u64, pipes: usize) -> [usize; 6] {
    let cost = |c: &[usize; 6]| -> u64 {
        let n = c.iter().copied().sum::<usize>() as u64;
        let slabs: u64 = c
            .iter()
            .zip(CLASSES.iter())
            .map(|(n, s)| *n as u64 * *s as u64)
            .sum();
        pipes as u64 * extra * n + slabs
    };
    let mut counts = COUNTS;
    if cost(&counts) <= usable {
        let scale = (usable / cost(&counts).max(1)) as usize;
        counts = COUNTS.map(|c| c.saturating_mul(scale.max(1)));
        loop {
            let mut t = counts;
            t[0] += 1;
            if cost(&t) > usable {
                break;
            }
            counts = t;
        }
    } else {
        for _ in 0..8 {
            if cost(&counts) <= usable {
                break;
            }
            let scale = usable as f64 / cost(&counts).max(1) as f64;
            counts = counts.map(|c| (c as f64 * scale).floor() as usize);
            counts[0] = counts[0].max(8);
            counts[5] = counts[5].max(2);
        }
    }
    counts[0] = counts[0].max(8);
    counts[5] = counts[5].max(2);
    counts
}

/// p50 entity counts, `Ent::ALL` order. A new solve's caps are this shape
/// scaled to the slab.
pub const SHAPE: [usize; 8] = [3_425, 65_924, 509_561, 92_124, 2_099, 1_552, 4_501, 370_906];

/// p90 entity counts, `Ent::ALL` order. Round scratch is sized from this so a
/// round of fat solves still lays.
pub const PEAK: [usize; 8] = [
    11_277, 325_067, 1_712_134, 708_330, 7_021, 5_020, 8_356, 1_118_067,
];

pub const FIELDS: [usize; 8] = [18, 13, 7, 4, 5, D + JW, 2 * D + POOL + 2, 1];

pub fn words(caps: &[usize; 8]) -> usize {
    (0..8).map(|e| caps[e] * FIELDS[e]).sum()
}

/// Caps that fill `slab` with `shape`'s proportions.
pub fn caps_for(slab: usize, shape: &[usize; 8]) -> [usize; 8] {
    let den = words(shape).max(1);
    let room = (slab / 4).max(1);
    let mut caps = [1usize; 8];
    for e in 0..8 {
        caps[e] = (room * shape[e] / den).max(1);
    }
    while words(&caps) > room {
        let e = (0..8).max_by_key(|&i| caps[i] * FIELDS[i]).unwrap();
        caps[e] = caps[e].saturating_sub(1).max(1);
        if caps.iter().all(|&c| c == 1) {
            break;
        }
    }
    caps
}

/// Next layout: the entity that grew gets room, the others keep theirs.
pub fn grow_caps(old: &[usize; 8], lens: &[usize; 8]) -> [usize; 8] {
    std::array::from_fn(|e| lens[e].saturating_mul(2).max(old[e]).max(1))
}

pub fn class_of(caps: &[usize; 8]) -> Option<usize> {
    let bytes = words(caps).saturating_mul(4);
    CLASSES.iter().position(|&c| c >= bytes)
}

/// Whether `lens`, doubled, still fit the largest class.
pub fn fits(lens: &[usize; 8]) -> bool {
    class_of(&grow_caps(&[0; 8], lens)).is_some()
}
