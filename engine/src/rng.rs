//! Tiny deterministic PRNG (xorshift64*). Used ONLY by the benchmark binary and
//! the test suite to pick uniformly among legal actions. The engine core never
//! calls it: all randomness enters the game through chance-node actions.

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E3779B97F4A7C15)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    #[inline]
    pub fn unit_f64(&mut self) -> f64 {
        // Top 53 bits map exactly into [0, 1), matching f64 mantissa width.
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn weighted_index(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().copied().sum();
        assert!(total > 0.0 && total.is_finite());
        let mut needle = self.unit_f64() * total;
        for (i, &weight) in weights.iter().enumerate() {
            assert!(weight >= 0.0 && weight.is_finite());
            if needle < weight {
                return i;
            }
            needle -= weight;
        }
        weights.len() - 1
    }
}
