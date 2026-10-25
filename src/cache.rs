//! KV cache: per layer, per KV head, contiguous `(max_seq, head_dim)` rows so
//! attention reads are sequential SIMD dots.

#[derive(Debug)]
pub struct KVCache {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// keys[layer] = (n_kv_heads, max_seq, head_dim), row-major.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// How many positions are currently filled.
    pos: usize,
}

impl KVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> KVCache {
        let per_layer = n_kv_heads * max_seq_len * head_dim;
        KVCache {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            keys: vec![vec![0.0; per_layer]; n_layers],
            values: vec![vec![0.0; per_layer]; n_layers],
            pos: 0,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    pub fn bytes(&self) -> usize {
        self.keys.iter().chain(self.values.iter()).map(|v| v.len() * 4).sum()
    }

    /// Append k/v of shape (n_kv_heads, seq, head_dim) for one layer.
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32], seq: usize) {
        let hd = self.head_dim;
        let row = self.pos; // absolute start position
        for h in 0..self.n_kv_heads {
            let dst = &mut self.keys[layer]
                [h * self.max_seq_len * hd + row * hd..(h * self.max_seq_len + row + seq) * hd];
            let src = &k[h * seq * hd..(h + 1) * seq * hd];
            dst.copy_from_slice(src);
            let dst = &mut self.values[layer]
                [h * self.max_seq_len * hd + row * hd..(h * self.max_seq_len + row + seq) * hd];
            let src = &v[h * seq * hd..(h + 1) * seq * hd];
            dst.copy_from_slice(src);
        }
    }

    /// Read-only view of head `h`'s keys for positions `0..total`.
    pub fn head_keys(&self, layer: usize, h: usize, total: usize) -> &[f32] {
        let hd = self.head_dim;
        &self.keys[layer][h * self.max_seq_len * hd..h * self.max_seq_len * hd + total * hd]
    }

    pub fn head_values(&self, layer: usize, h: usize, total: usize) -> &[f32] {
        let hd = self.head_dim;
        &self.values[layer][h * self.max_seq_len * hd..h * self.max_seq_len * hd + total * hd]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_then_read_roundtrips() {
        let mut cache = KVCache::new(2, 3, 4, 16);
        let k: Vec<f32> = (0..3 * 2 * 4).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..3 * 2 * 4).map(|i| 100.0 + i as f32).collect();
        cache.append(0, &k, &v, 2);
        cache.advance(2);
        let head0 = cache.head_keys(0, 0, 2);
        assert_eq!(head0, &(0..8).map(|i| i as f32).collect::<Vec<_>>());
        let head2 = cache.head_values(0, 2, 2);
        assert_eq!(head2, &(0..8).map(|i| 100.0 + 16.0 + i as f32).collect::<Vec<_>>());

        // Second step lands at row 2.
        let k2: Vec<f32> = (0..3 * 4).map(|i| 50.0 + i as f32).collect();
        let v2 = k2.clone();
        cache.append(0, &k2, &v2, 1);
        cache.advance(1);
        let head1 = cache.head_keys(0, 1, 3);
        // rows 0,1 from step one (k[8..16]); row 2 from step two (k2[4..8] = 54..58)
        assert_eq!(&head1[8..12], &[54.0f32, 55.0, 56.0, 57.0]);
    }
}
