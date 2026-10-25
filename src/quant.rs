//! Weight-only int8 / int4 quantization, symmetric round-to-nearest -- the
//! same scheme as the Python engine (`quant.py`): a scale per output channel,
//! or per group of input channels when `group_size` is set. 4-bit weights stay
//! nibble-packed in memory during inference and are dequantized in registers
//! by the matmul kernel (the Python engine instead widens to f32 at load time,
//! which is exactly the overhead this module exists to remove).
//!
//! One difference to note: np.round is banker's rounding (half to even), so we
//! reimplement it here to keep quantized checkpoints bit-comparable.

use crate::simd;

#[derive(Debug, Clone)]
pub struct QuantMatrix {
    /// (out_features, in_features), i.e. the transposed layout the engine uses.
    pub out_features: usize,
    pub in_features: usize,
    pub bits: u8,
    /// None = per output channel; Some(g) = per g input channels.
    pub group_size: Option<usize>,
    /// int8: `out * in` values. int4: packed two per byte, `ceil(out*in / 2)`.
    pub q: Vec<u8>,
    /// scales: per row, or per (row, group).
    pub scales: Vec<f32>,
}

fn round_half_even(x: f32) -> f32 {
    let floor = x.floor();
    let diff = x - floor;
    if diff > 0.5 {
        floor + 1.0
    } else if diff < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

impl QuantMatrix {
    /// Quantize a (out, in) row-major matrix (the engine's transposed layout).
    pub fn quantize(w: &[f32], out_features: usize, in_features: usize, bits: u8, group_size: Option<usize>) -> QuantMatrix {
        assert!(bits == 4 || bits == 8, "only 4-bit and 8-bit are supported");
        let qmax = (1i32 << (bits - 1)) - 1;
        let qmin = -qmax - 1;

        let groups = group_size.map(|g| in_features / g).unwrap_or(1);
        let group_len = group_size.unwrap_or(in_features);
        assert!(group_size.is_none() || in_features % group_size.unwrap() == 0);

        let mut scales = vec![0.0f32; out_features * groups];
        let numel = out_features * in_features;
        let mut q = vec![0u8; if bits == 4 { numel.div_ceil(2) } else { numel }];

        for row in 0..out_features {
            for g in 0..groups {
                let start = g * group_len;
                let end = start + group_len;
                let mut amax = 0.0f32;
                for k in start..end {
                    let v = w[row * in_features + k];
                    amax = amax.max(v.abs());
                }
                let mut scale = amax / qmax as f32;
                if scale == 0.0 {
                    scale = 1e-8;
                }
                scales[row * groups + g] = scale;
                let inv = 1.0 / scale;
                for k in start..end {
                    let v = w[row * in_features + k];
                    let mut r = round_half_even(v * inv) as i32;
                    r = r.clamp(qmin, qmax);
                    let store = (r + (1 << (bits - 1))) as u8; // shift into 0..2^bits
                    let idx = row * in_features + k;
                    if bits == 8 {
                        q[idx] = store;
                    } else if idx % 2 == 0 {
                        q[idx / 2] = store & 0x0F;
                    } else {
                        q[idx / 2] |= store << 4;
                    }
                }
            }
        }

        QuantMatrix {
            out_features,
            in_features,
            bits,
            group_size,
            q,
            scales,
        }
    }

    /// Decode the stored code back to the integer domain.
    /// int8: biased byte (`value + 128`); int4: biased nibble (`value + 8`).
    #[inline]
    fn code_to_i32(&self, code: u8) -> i32 {
        match self.bits {
            8 => code as i32 - 128,
            _ => (code & 0x0F) as i32 - 8,
        }
    }

    /// y = W @ x for one row of x (GEMV), streaming dequantization.
    /// Weights are read directly from storage -- no conversion, no allocation.
    pub fn dot_row(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; self.out_features];
        self.dot_row_into(x, &mut y);
        y
    }

    pub fn dot_row_into(&self, x: &[f32], y: &mut [f32]) {
        self.dot_rows_into(x, 0, self.out_features, y);
    }

    /// Compute rows `row0..row1` of W @ x into y (length row1 - row0).
    pub fn dot_rows_into(&self, x: &[f32], row0: usize, row1: usize, y: &mut [f32]) {
        assert_eq!(y.len(), row1 - row0);
        let groups = self.group_size.map(|g| self.in_features / g).unwrap_or(1);
        match (self.bits, self.group_size) {
            (8, None) => {
                for (ri, row) in (row0..row1).enumerate() {
                    let w = &self.q[row * self.in_features..(row + 1) * self.in_features];
                    y[ri] = simd::dot_f32_i8(x, w) * self.scales[row];
                }
            }
            (4, None) => {
                for (ri, row) in (row0..row1).enumerate() {
                    let w = &self.q[row * self.in_features / 2..];
                    y[ri] = simd::dot_f32_i4(x, w, self.in_features) * self.scales[row];
                }
            }
            (_, Some(g)) => {
                for (ri, row) in (row0..row1).enumerate() {
                    let mut acc = 0.0f32;
                    for group in 0..groups {
                        let start = group * g;
                        let scale = self.scales[row * groups + group];
                        let xs = &x[start..start + g];
                        let sub = if self.bits == 8 {
                            simd::dot_f32_i8(
                                xs,
                                &self.q[row * self.in_features + start
                                    ..row * self.in_features + start + g],
                            )
                        } else {
                            let byte_off = (row * self.in_features + start) / 2;
                            simd::dot_f32_i4(xs, &self.q[byte_off..], g)
                        };
                        acc += sub * scale;
                    }
                    y[ri] = acc;
                }
            }
            _ => unreachable!(),
        }
    }

    /// Dot of x with a single weight row (used by the prefill GEMM).
    pub fn dot_row_scalar(&self, x: &[f32], row: usize) -> f32 {
        let groups = self.group_size.map(|g| self.in_features / g).unwrap_or(1);
        match (self.bits, self.group_size) {
            (8, None) => {
                let w = &self.q[row * self.in_features..(row + 1) * self.in_features];
                simd::dot_f32_i8(x, w) * self.scales[row]
            }
            (4, None) => {
                let w = &self.q[row * self.in_features / 2..];
                simd::dot_f32_i4(x, w, self.in_features) * self.scales[row]
            }
            (_, Some(g)) => {
                let mut acc = 0.0f32;
                for group in 0..groups {
                    let start = group * g;
                    let scale = self.scales[row * groups + group];
                    let xs = &x[start..start + g];
                    let sub = if self.bits == 8 {
                        simd::dot_f32_i8(
                            xs,
                            &self.q[row * self.in_features + start
                                ..row * self.in_features + start + g],
                        )
                    } else {
                        let byte_off = (row * self.in_features + start) / 2;
                        simd::dot_f32_i4(xs, &self.q[byte_off..], g)
                    };
                    acc += sub * scale;
                }
                acc
            }
            _ => unreachable!(),
        }
    }

    /// Copy one dequantized row out (embedding gather).
    pub fn row_into(&self, row: usize, y: &mut [f32]) {
        let groups = self.group_size.map(|g| self.in_features / g).unwrap_or(1);
        for k in 0..self.in_features {
            let code = if self.bits == 8 {
                self.q[row * self.in_features + k]
            } else {
                let byte = self.q[(row * self.in_features + k) / 2];
                if (row * self.in_features + k) % 2 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                }
            };
            let scale = if let Some(g) = self.group_size {
                self.scales[row * groups + k / g]
            } else {
                self.scales[row]
            };
            y[k] = (code as i32 - if self.bits == 8 { 128 } else { 8 }) as f32 * scale;
        }
    }

    /// Exact in-memory footprint (weights + scales), for size reporting.
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }
}

/// Dequantize back to f32 (used by tests to verify round-trip error bounds).
pub fn dequantize(m: &QuantMatrix) -> Vec<f32> {
    let mut w = vec![0.0f32; m.out_features * m.in_features];
    let groups = m.group_size.map(|g| m.in_features / g).unwrap_or(1);
    for row in 0..m.out_features {
        for k in 0..m.in_features {
            let code = if m.bits == 8 {
                m.q[row * m.in_features + k]
            } else {
                let byte = m.q[(row * m.in_features + k) / 2];
                if (row * m.in_features + k) % 2 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                }
            };
            let scale = if let Some(g) = m.group_size {
                m.scales[row * groups + k / g]
            } else {
                m.scales[row]
            };
            w[row * m.in_features + k] = m.code_to_i32(code) as f32 * scale;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights(out: usize, in_f: usize) -> Vec<f32> {
        // Deterministic pseudo-random weights in a realistic range.
        (0..out * in_f)
            .map(|i| {
                let x = ((i as u64 * 2654435761) % 1000) as f64 / 1000.0;
                ((x * 2.0 - 1.0) * 0.05) as f32
            })
            .collect()
    }

    #[test]
    fn round_half_even_matches_numpy() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-0.5), -0.0);
        assert_eq!(round_half_even(-1.5), -2.0);
        assert_eq!(round_half_even(3.5), 4.0);
    }

    #[test]
    fn int8_roundtrip_error_is_small() {
        let w = sample_weights(16, 64);
        let m = QuantMatrix::quantize(&w, 16, 64, 8, None);
        assert_eq!(m.bytes(), 16 * 64 + 16 * 4);
        let restored = dequantize(&m);
        let max_err = w.iter().zip(&restored).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // One step of the uniform grid: scale/2 at most.
        assert!(max_err <= m.scales[0] / 2.0 + 1e-9);
    }

    #[test]
    fn int4_grouped_roundtrip_error_bounded() {
        let w = sample_weights(8, 128);
        let m = QuantMatrix::quantize(&w, 8, 128, 4, Some(64));
        assert_eq!(m.bytes(), 8 * 128 / 2 + 8 * 2 * 4);
        let restored = dequantize(&m);
        let max_err = w.iter().zip(&restored).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err <= m.scales[0] / 2.0 + 1e-9);
    }

    #[test]
    fn dot_row_matches_dequantized_matmul() {
        let w = sample_weights(12, 96);
        let x: Vec<f32> = (0..96).map(|i| (i as f32) * 0.02 - 0.9).collect();
        for (bits, group) in [(8u8, None), (4, Some(32)), (8, Some(48)), (4, None)] {
            let m = QuantMatrix::quantize(&w, 12, 96, bits, group);
            let got = m.dot_row(&x);
            let wq = dequantize(&m);
            for row in 0..12 {
                let reference: f32 =
                    wq[row * 96..(row + 1) * 96].iter().zip(&x).map(|(a, b)| a * b).sum();
                assert!(
                    (got[row] - reference).abs() < 1e-3,
                    "bits={bits} group={group:?} row={row}: {} vs {}",
                    got[row],
                    reference
                );
            }
        }
    }
}
