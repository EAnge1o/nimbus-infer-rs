//! Hand-written SIMD dot products -- the hot core of the engine.
//!
//! Three kernels, all computing `sum(a[i] * w[i])` with different weight
//! encodings:
//!
//! * [`dot_f32`]      -- plain float32 weights.
//! * [`dot_f32_i8`]   -- int8 weights (dequantized on the fly; scale is applied
//!   by the caller, so it factors out of the sum).
//! * [`dot_f32_i4`]   -- int4 weights, still nibble-packed in memory (two
//!   weights per byte), dequantized in registers.
//!
//! x86_64 builds use AVX2 + FMA when the CPU supports it (detected at runtime,
//! so a single portable binary still runs everywhere); everything else falls
//! back to a scalar loop. Order of accumulation differs between the two paths,
//! but both accumulate into one f32 lane chain, which keeps results stable for
//! a given build.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Runtime-selected float32 dot product.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

/// Dot product against int8 weights stored as biased bytes (`value + 128`),
/// converted to f32 on the fly. Taking biased bytes lets the kernel read the
/// weight buffer directly -- no per-row conversion, no allocation.
pub fn dot_f32_i8(a: &[f32], w: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), w.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_i8_avx2(a, w) };
        }
    }
    w.iter().zip(a).map(|(q, x)| (*q as i32 as f32 - 128.0) * x).sum()
}

/// Dot product against nibble-packed int4 weights.
///
/// `w` holds `numel` weights, two per byte (low nibble first, biased by +8 so
/// each nibble is stored as `value + 8` in 0..16 -- the same layout the Python
/// engine writes). The per-group scale is applied by the caller.
pub fn dot_f32_i4(a: &[f32], w: &[u8], numel: usize) -> f32 {
    debug_assert!(numel <= w.len() * 2);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_i4_avx2(a, w, numel) };
        }
    }
    let mut acc = 0.0f32;
    for i in 0..numel {
        let byte = w[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        acc += a[i] * (nibble as f32 - 8.0);
    }
    acc
}

pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
const AVX_LANES: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + AVX_LANES <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += AVX_LANES;
    }
    let mut sum = hsum256(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(a: &[f32], w: &[u8]) -> f32 {
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let bias = _mm256_set1_ps(128.0);
    let mut i = 0;
    while i + AVX_LANES <= n {
        // Load 8 biased int8 weights, widen to int32, un-bias, convert to f32.
        let w8 = _mm_loadl_epi64(w.as_ptr().add(i) as *const __m128i);
        let wi = _mm256_cvtepu8_epi32(w8);
        let wf = _mm256_sub_ps(_mm256_cvtepi32_ps(wi), bias);
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, wf, acc);
        i += AVX_LANES;
    }
    let mut sum = hsum256(acc);
    while i < n {
        sum += a[i] * (w[i] as i32 as f32 - 128.0);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i4_avx2(a: &[f32], w: &[u8], numel: usize) -> f32 {
    // 8 packed bytes hold 16 nibbles: byte k carries weight 2k in its low
    // nibble and weight 2k+1 in its high nibble. Deinterleave the two nibble
    // streams so the weight vector is back in natural order, then a plain
    // contiguous FMA against a[0..16] per 8 bytes.
    let mut acc = _mm256_setzero_ps();
    let bias = _mm256_set1_ps(8.0);
    let mut bytes_done = 0usize;
    let mut a_off = 0usize;
    let full_bytes = numel / 2;
    while bytes_done + AVX_LANES <= full_bytes {
        let raw = _mm_loadl_epi64(w.as_ptr().add(bytes_done) as *const __m128i);
        let expanded = _mm256_cvtepu8_epi32(raw); // lane j = byte j (0..255)
        let lo = _mm256_and_si256(expanded, _mm256_set1_epi32(0x0F));
        let hi = _mm256_srli_epi32::<4>(expanded);
        // Interleave back to [bKlo, bKhi, ...] order. unpacklo/hi work per
        // 128-bit lane, so each permute2f128 joins one half: the first gets
        // weights 0..7 (bytes 0..3), the second weights 8..15 (bytes 4..7).
        let t0 = _mm256_unpacklo_epi32(lo, hi);
        let t1 = _mm256_unpackhi_epi32(lo, hi);
        let q_lo = _mm256_permute2f128_si256::<0x20>(t0, t1);
        let q_hi = _mm256_permute2f128_si256::<0x31>(t0, t1);
        let q_lo_f = _mm256_sub_ps(_mm256_cvtepi32_ps(q_lo), bias);
        let q_hi_f = _mm256_sub_ps(_mm256_cvtepi32_ps(q_hi), bias);
        let va_lo = _mm256_loadu_ps(a.as_ptr().add(a_off));
        let va_hi = _mm256_loadu_ps(a.as_ptr().add(a_off + AVX_LANES));
        acc = _mm256_fmadd_ps(va_lo, q_lo_f, acc);
        acc = _mm256_fmadd_ps(va_hi, q_hi_f, acc);
        bytes_done += AVX_LANES;
        a_off += 2 * AVX_LANES;
    }
    let mut sum = hsum256(acc);
    for k in bytes_done..full_bytes {
        let byte = w[k];
        sum += a[2 * k] * ((byte & 0x0F) as f32 - 8.0);
        sum += a[2 * k + 1] * ((byte >> 4) as f32 - 8.0);
    }
    if numel % 2 == 1 {
        let byte = w[full_bytes];
        sum += a[numel - 1] * ((byte & 0x0F) as f32 - 8.0);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn hsum256(v: __m256) -> f32 {
    // Reduce 8 lanes: first fold the two 128-bit halves, then horizontal-add.
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let shuf1 = _mm_shuffle_ps(sum2, sum2, 0x55);
    let sum1 = _mm_add_ss(sum2, shuf1);
    _mm_cvtss_f32(sum1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_f32_matches_scalar() {
        let a: Vec<f32> = (0..257).map(|i| (i as f32) * 0.5 - 30.0).collect();
        let b: Vec<f32> = (0..257).map(|i| 1.0 / (i as f32 + 3.0)).collect();
        let fast = dot_f32(&a, &b);
        let slow = dot_f32_scalar(&a, &b);
        assert!((fast - slow).abs() < 1e-2, "{fast} vs {slow}");
    }

    #[test]
    fn dot_i8_matches_reference() {
        let a: Vec<f32> = (0..131).map(|i| (i as f32) * 0.125).collect();
        let raw: Vec<i32> = (0..131).map(|i| ((i * 7) % 255) - 128).collect();
        let biased: Vec<u8> = raw.iter().map(|&v| (v + 128) as u8).collect();
        let fast = dot_f32_i8(&a, &biased);
        let reference: f32 = a
            .iter()
            .zip(raw.iter())
            .map(|(x, q)| x * (*q as f32))
            .sum();
        assert!((fast - reference).abs() < 1e-2, "{fast} vs {reference}");
    }

    #[test]
    fn dot_i4_matches_reference() {
        let a: Vec<f32> = (0..255).map(|i| (i as f32) * 0.125).collect();
        let q: Vec<i8> = (0..255).map(|i| ((i * 13) % 16 - 8) as i8).collect();
        let mut packed = Vec::with_capacity(128);
        let mut chunk = 0u8;
        for (i, v) in q.iter().enumerate() {
            let nib = (v + 8) as u8 & 0x0F;
            if i % 2 == 0 {
                chunk = nib;
            } else {
                chunk |= nib << 4;
                packed.push(chunk);
            }
        }
        if q.len() % 2 == 1 {
            packed.push(chunk);
        }
        let fast = dot_f32_i4(&a, &packed, q.len());
        let reference: f32 = a
            .iter()
            .zip(q.iter())
            .map(|(x, v)| x * (*v as f32))
            .sum();
        assert!((fast - reference).abs() < 1e-2, "{fast} vs {reference}");
    }
}
