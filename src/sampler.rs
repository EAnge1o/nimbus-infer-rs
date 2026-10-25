//! Sampling strategies: greedy, temperature, top-k, top-p, repetition
//! penalty -- same semantics and filter order as the Python engine
//! (`sampling.py`), but with the crate's own xoshiro256++ PRNG (numpy's
//! PCG64 stream is not reproduced bit-for-bit; greedy decoding is exact).

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,   // 0 disables
    pub top_p: f32,     // 1.0 disables
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub stop_tokens: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            max_new_tokens: 64,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: None,
            stop_tokens: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

// ---------------------------------------------------------------------------
// xoshiro256++ (seeded via splitmix64)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = move || {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        };
        Rng { state: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self
            .state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Filters (operate on f64 logits like the Python engine)
// ---------------------------------------------------------------------------
pub fn apply_repetition_penalty(logits: &mut [f64], generated: &[u32], penalty: f32) {
    if penalty == 1.0 || generated.is_empty() {
        return;
    }
    let mut seen = generated.to_vec();
    seen.sort_unstable();
    seen.dedup();
    for &token in &seen {
        let idx = token as usize;
        if idx >= logits.len() {
            continue;
        }
        if logits[idx] > 0.0 {
            logits[idx] /= penalty as f64;
        } else {
            logits[idx] *= penalty as f64;
        }
    }
}

fn top_k_filter(logits: &mut Vec<f64>, k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // Find the k-th largest value, keep everything >= it.
    let mut sorted = logits.clone();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let kth = sorted[k - 1];
    for v in logits.iter_mut() {
        if *v < kth {
            *v = f64::NEG_INFINITY;
        }
    }
}

fn top_p_filter(logits: &mut [f64], p: f64) {
    if p >= 1.0 {
        return;
    }
    if p <= 0.0 {
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let keep = logits[best];
        for v in logits.iter_mut() {
            *v = f64::NEG_INFINITY;
        }
        logits[best] = keep;
        return;
    }
    // Sort descending, softmax, cumulative mass; drop a token once the mass
    // *excluding* it already exceeds p (same rule as the Python engine).
    let n = logits.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    let max = logits[order[0]];
    let mut probs: Vec<f64> = order.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f64 = probs.iter().sum();
    for v in probs.iter_mut() {
        *v /= sum;
    }
    let mut cum = 0.0f64;
    let mut remove = vec![false; n];
    for (pos, &i) in order.iter().enumerate() {
        if cum - probs[pos] > p {
            remove[i] = true;
        }
        cum += probs[pos];
    }
    for (i, r) in remove.iter().enumerate() {
        if *r {
            logits[i] = f64::NEG_INFINITY;
        }
    }
}

/// Turn a logits vector into one token id.
pub fn sample(
    logits: &[f32],
    params: &SamplingParams,
    generated: &[u32],
    rng: &mut Rng,
) -> u32 {
    let mut work: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    apply_repetition_penalty(&mut work, generated, params.repetition_penalty);

    if params.greedy() {
        // First index of the maximum, matching np.argmax.
        let mut best = 0usize;
        for (i, &v) in work.iter().enumerate() {
            if v > work[best] {
                best = i;
            }
        }
        return best as u32;
    }

    let temp = (params.temperature as f64).max(1e-6);
    for v in work.iter_mut() {
        *v /= temp;
    }
    top_k_filter(&mut work, params.top_k);
    top_p_filter(&mut work, params.top_p as f64);

    // Softmax over the filtered logits, then inverse-CDF sampling.
    let max = work.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        let mut best = 0usize;
        for (i, &v) in work.iter().enumerate() {
            if v > work[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let mut total = 0.0f64;
    for v in work.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }
    if !total.is_finite() || total <= 0.0 {
        return 0;
    }
    let mut r = rng.next_f64() * total;
    for (i, &v) in work.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (work.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_first_max() {
        let params = SamplingParams { temperature: 0.0, ..Default::default() };
        let mut rng = Rng::new(0);
        assert_eq!(sample(&[0.1, 5.0, 5.0, -1.0], &params, &[], &mut rng), 1);
    }

    #[test]
    fn rng_is_deterministic_and_seeded() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Rng::new(43);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn top_k_keeps_only_best_k() {
        let mut logits = vec![1.0f64, 5.0, 3.0, 4.0];
        top_k_filter(&mut logits, 2);
        assert_eq!(logits[0], f64::NEG_INFINITY);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], f64::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
    }

    #[test]
    fn repetition_penalty_divides_positive_multiplies_negative() {
        let mut logits = vec![2.0f64, -2.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], -4.0);
    }

    #[test]
    fn top_p_zero_keeps_argmax_only() {
        let mut logits = vec![1.0f64, 5.0, 3.0];
        top_p_filter(&mut logits, 0.0);
        assert_eq!(logits[0], f64::NEG_INFINITY);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], f64::NEG_INFINITY);
    }
}
