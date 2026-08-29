pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let s = seed ^ 0x9E3779B97F4A7C15;
        Rng(if s == 0 { 0xDEADBEEFCAFEF00D } else { s })
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
