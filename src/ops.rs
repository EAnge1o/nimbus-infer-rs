//! Numerical primitives: softmax, GELU, SiLU, LayerNorm, RMSNorm.
//! Direct ports of the Python engine's `backend.py` so both engines produce
//! the same numbers given the same inputs.

/// Numerically stable softmax over one row.
pub fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// GPT-2's tanh-approximation of GELU.
pub fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// SiLU / swish.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// LayerNorm over the last dimension of a row.
pub fn layer_norm(row: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = row.len();
    let mean = row.iter().sum::<f32>() / n as f32;
    let var = row.iter().map(|v| (*v - mean) * (*v - mean)).sum::<f32>() / n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        row[i] = (row[i] - mean) * inv * weight[i] + bias[i];
    }
}

/// RMSNorm over the last dimension of a row.
pub fn rms_norm(row: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for i in 0..n {
        row[i] *= inv * weight[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let mut row = vec![1000.0f32, 1001.0, 999.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(row[0] > 0.0 && row[1] > row[0]);
        assert!(row.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gelu_matches_reference_values() {
        // Reference: numpy implementation 0.5*x*(1+tanh(c*(x+0.044715x^3)))
        for (x, expected) in [(0.0f32, 0.0), (1.0, 0.841192), (-1.0, -0.158808), (3.0, 2.996363)] {
            let got = gelu(x);
            assert!((got - expected).abs() < 1e-4, "gelu({x}) = {got}, want {expected}");
        }
    }

    #[test]
    fn layer_norm_normalizes() {
        let mut row = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; 4];
        let bias = vec![0.0f32; 4];
        layer_norm(&mut row, &weight, &bias, 1e-5);
        let mean: f32 = row.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        let var: f32 = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn rms_norm_scales_by_rms() {
        let mut row = vec![3.0f32, 4.0];
        let weight = vec![1.0f32, 1.0];
        rms_norm(&mut row, &weight, 1e-6);
        // rms = sqrt((9+16)/2) = sqrt(12.5)
        assert!((row[0] - 3.0 / 12.5f32.sqrt()).abs() < 1e-5);
        assert!((row[1] - 4.0 / 12.5f32.sqrt()).abs() < 1e-5);
    }
}
