//! Decoder-only transformer: one forward path covers GPT-2 (learned
//! positions, LayerNorm, tanh-GELU, MHA) and LLaMA (RoPE, RMSNorm, SwiGLU,
//! GQA), selected by `ModelConfig` -- same design as the Python engine.
//!
//! Linear weights are stored transposed, `(out, in)` row-major, so every
//! output element is one contiguous SIMD dot product. Decode steps are
//! GEMVs parallelized over output rows; prefill uses a block-over-outputs
//! GEMM that keeps the activation rows hot in cache.

use crate::cache::KVCache;
use crate::config::ModelConfig;
use crate::ops;
use crate::quant::QuantMatrix;
use crate::simd;
use crate::threads::par_range;
use crate::threads::par_range_capped;
use crate::threads::UnsafeSlice;

/// Minimum rows before a decode-step GEMV is worth splitting across threads.
const PAR_ROWS_THRESHOLD: usize = 4096;
/// Minimum multiply-accumulates per GEMM chunk (prefill parallel split).
const PAR_WORK_MACS: usize = 4_000_000;
/// Decode GEMVs are memory-bandwidth-bound: past a handful of threads the
/// workers contend for DRAM instead of adding throughput (measured on
/// DDR4 client parts). Prefill GEMMs are compute-bound and scale further.
/// Override the default parallelism with NIMBUS_THREADS.
const GEMV_WORKERS_CAP: usize = 4;

/// A linear layer: either dense f32 (already transposed to (out, in)) or a
/// quantized matrix dequantized on the fly.
#[derive(Debug, Clone)]
pub enum Mat {
    Dense { wt: Vec<f32>, out: usize, inp: usize },
    Q(QuantMatrix),
}

impl Mat {
    pub fn dense(wt: Vec<f32>, out: usize, inp: usize) -> Mat {
        assert_eq!(wt.len(), out * inp);
        Mat::Dense { wt, out, inp }
    }

    pub fn out(&self) -> usize {
        match self {
            Mat::Dense { out, .. } => *out,
            Mat::Q(m) => m.out_features,
        }
    }

    pub fn inp(&self) -> usize {
        match self {
            Mat::Dense { inp, .. } => *inp,
            Mat::Q(m) => m.in_features,
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mat::Dense { wt, .. } => wt.len() * 4,
            Mat::Q(m) => m.bytes(),
        }
    }

    /// y[j] = dot(x, Wt[j]) for all j -- the decode-step GEMV.
    pub fn gemv_into(&self, x: &[f32], y: &mut [f32]) {
        let out = self.out();
        assert_eq!(y.len(), out);
        match self {
            Mat::Dense { wt, inp, .. } => {
                let inp = *inp;
                let y_shared = UnsafeSlice::new(y);
                par_range_capped(out, PAR_ROWS_THRESHOLD, GEMV_WORKERS_CAP, |j0, j1| {
                    for j in j0..j1 {
                        let v = simd::dot_f32(x, &wt[j * inp..(j + 1) * inp]);
                        unsafe { y_shared.write(j, v) };
                    }
                });
            }
            Mat::Q(m) => {
                let y_shared = UnsafeSlice::new(y);
                par_range_capped(out, PAR_ROWS_THRESHOLD, GEMV_WORKERS_CAP, |j0, j1| {
                    unsafe { m.dot_rows_into(x, j0, j1, y_shared.slice_mut(j0..j1)) };
                });
            }
        }
    }

    /// y[i][j] = dot(x[i], Wt[j]) -- the prefill GEMM.
    ///
    /// Parallel over output rows j, sequential over the sequence: the whole
    /// activation panel (seq x inp, a few hundred KB at prompt sizes) stays
    /// hot in L2 while each weight row streams from memory exactly once.
    /// (The naive i-major loop re-reads the weight matrix once per row.)
    pub fn gemm_into(&self, x: &[f32], seq: usize, y: &mut [f32]) {
        if seq == 1 {
            self.gemv_into(x, y);
            return;
        }
        let out = self.out();
        let inp = self.inp();
        assert_eq!(y.len(), seq * out);
        // Parallelize by total work: split rows so each chunk is roughly
        // PAR_WORK_MACS multiply-accumulates, but never so coarse that whole
        // threads idle (skinny LLaMA-style matrices have few output rows).
        let workers = crate::threads::worker_count();
        let by_work = (PAR_WORK_MACS / inp / seq).clamp(1, out);
        let min_rows = by_work.min((out / workers).max(1));
        match self {
            Mat::Dense { wt, .. } => {
                let y_shared = UnsafeSlice::new(y);
                par_range(out, min_rows, |j0, j1| {
                    for j in j0..j1 {
                        let wr = &wt[j * inp..(j + 1) * inp];
                        for i in 0..seq {
                            let v = simd::dot_f32(&x[i * inp..(i + 1) * inp], wr);
                            unsafe { y_shared.write(i * out + j, v) };
                        }
                    }
                });
            }
            Mat::Q(m) => {
                let y_shared = UnsafeSlice::new(y);
                par_range(out, min_rows, |j0, j1| {
                    for j in j0..j1 {
                        for i in 0..seq {
                            let v = m.dot_row_scalar(&x[i * inp..(i + 1) * inp], j);
                            unsafe { y_shared.write(i * out + j, v) };
                        }
                    }
                });
            }
        }
    }

    /// Copy one row out (embedding gather).
    pub fn row_into(&self, row: usize, y: &mut [f32]) {
        match self {
            Mat::Dense { wt, inp, .. } => {
                y.copy_from_slice(&wt[row * inp..(row + 1) * inp]);
            }
            Mat::Q(m) => m.row_into(row, y),
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Option<Vec<f32>>,
    pub q: Mat,
    pub q_b: Option<Vec<f32>>,
    pub k: Mat,
    pub k_b: Option<Vec<f32>>,
    pub v: Mat,
    pub v_b: Option<Vec<f32>>,
    pub o: Mat,
    pub o_b: Option<Vec<f32>>,
    // gelu: fc -> gelu -> proj; swiglu: gate/up -> silu(gate)*up -> down
    pub fc: Option<Mat>,
    pub fc_b: Option<Vec<f32>>,
    pub proj: Option<Mat>,
    pub proj_b: Option<Vec<f32>>,
    pub gate: Option<Mat>,
    pub up: Option<Mat>,
    pub down: Option<Mat>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct Model {
    pub cfg: ModelConfig,
    /// (vocab, d_model) row-major; doubles as the LM head when tied.
    pub wte: Mat,
    wpe: Option<Vec<f32>>,
    lm_head: Option<Mat>,
    layers: Vec<Layer>,
    norm_f_w: Vec<f32>,
    norm_f_b: Option<Vec<f32>>,
    rope: Option<(Vec<f32>, Vec<f32>)>, // cos, sin: (max_seq, hd/2)
}

pub fn build_rope(head_dim: usize, max_seq_len: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; max_seq_len * half];
    let mut sin = vec![0.0f32; max_seq_len * half];
    for t in 0..max_seq_len {
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * i) as f32 / half as f32);
            let ang = t as f32 * inv_freq;
            cos[t * half + i] = ang.cos();
            sin[t * half + i] = ang.sin();
        }
    }
    (cos, sin)
}

impl Model {
    pub fn new(
        cfg: ModelConfig,
        wte: Mat,
        wpe: Option<Vec<f32>>,
        lm_head: Option<Mat>,
        layers: Vec<Layer>,
        norm_f_w: Vec<f32>,
        norm_f_b: Option<Vec<f32>>,
    ) -> Model {
        let rope = if cfg.pos_encoding == crate::config::PosEncoding::Rope {
            Some(build_rope(cfg.resolved_head_dim(), cfg.max_seq_len, cfg.rope_theta))
        } else {
            None
        };
        Model { cfg, wte, wpe, lm_head, layers, norm_f_w, norm_f_b, rope }
    }

    pub fn n_params(&self) -> usize {
        let mut total = self.wte.out() * self.wte.inp();
        if let Some(wpe) = &self.wpe {
            total += wpe.len();
        }
        if let Some(head) = &self.lm_head {
            total += head.out() * head.inp();
        }
        for l in &self.layers {
            for m in [&l.q, &l.k, &l.v, &l.o] {
                total += m.out() * m.inp();
            }
            for m in [&l.fc, &l.proj, &l.gate, &l.up, &l.down].into_iter().flatten() {
                total += m.out() * m.inp();
            }
            total += l.norm1_w.len() + l.norm2_w.len();
            for b in [&l.norm1_b, &l.norm2_b, &l.q_b, &l.k_b, &l.v_b, &l.o_b, &l.fc_b, &l.proj_b]
                .into_iter()
                .flatten()
            {
                total += b.len();
            }
        }
        total += self.norm_f_w.len();
        if let Some(b) = &self.norm_f_b {
            total += b.len();
        }
        total
    }

    /// In-memory weight footprint (matrices only; norms/biases excluded).
    pub fn weight_bytes(&self) -> usize {
        let mut total = self.wte.bytes() + self.lm_head.as_ref().map_or(0, Mat::bytes);
        for l in &self.layers {
            for m in [&l.q, &l.k, &l.v, &l.o] {
                total += m.bytes();
            }
            for m in [&l.fc, &l.proj, &l.gate, &l.up, &l.down].into_iter().flatten() {
                total += m.bytes();
            }
        }
        total
    }

    pub fn new_cache(&self) -> KVCache {
        KVCache::new(
            self.cfg.n_layers,
            self.cfg.resolved_n_kv_heads(),
            self.cfg.resolved_head_dim(),
            self.cfg.max_seq_len,
        )
    }

    /// Run the transformer over `tokens` (which start at the cache's current
    /// position). Writes hidden states (seq, d_model) into `hidden`.
    pub fn forward_hidden(&self, tokens: &[u32], mut cache: Option<&mut KVCache>, hidden: &mut Vec<f32>) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let seq = tokens.len();
        let pos_offset = cache.as_ref().map_or(0, |c| c.pos());
        assert!(
            pos_offset + seq <= cfg.max_seq_len,
            "sequence length {} exceeds max_seq_len {}",
            pos_offset + seq,
            cfg.max_seq_len
        );

        hidden.clear();
        hidden.resize(seq * d, 0.0);
        let mut row = vec![0.0f32; d];
        for (i, &tok) in tokens.iter().enumerate() {
            self.wte.row_into(tok as usize, &mut row);
            if let Some(wpe) = &self.wpe {
                let w = &wpe[(pos_offset + i) * d..(pos_offset + i + 1) * d];
                for k in 0..d {
                    row[k] += w[k];
                }
            }
            hidden[i * d..(i + 1) * d].copy_from_slice(&row);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            self.attention_block(layer, li, hidden, cache.as_deref_mut(), pos_offset);
            self.mlp_block(layer, hidden);
        }

        for i in 0..seq {
            let r = &mut hidden[i * d..(i + 1) * d];
            match cfg.norm {
                crate::config::Norm::RmsNorm => ops::rms_norm(r, &self.norm_f_w, cfg.norm_eps),
                crate::config::Norm::LayerNorm => ops::layer_norm(
                    r,
                    &self.norm_f_w,
                    self.norm_f_b.as_ref().unwrap(),
                    cfg.norm_eps,
                ),
            }
        }
    }

    /// Logits for the last hidden row (weight-tied or separate LM head).
    pub fn logits_last(&self, hidden_last: &[f32], logits: &mut Vec<f32>) {
        let vocab = self.cfg.vocab_size;
        logits.clear();
        logits.resize(vocab, 0.0);
        match &self.lm_head {
            Some(head) => head.gemv_into(hidden_last, logits),
            None => self.wte.gemv_into(hidden_last, logits),
        }
    }

    fn attention_block(
        &self,
        layer: &Layer,
        layer_idx: usize,
        hidden: &mut [f32],
        mut cache: Option<&mut KVCache>,
        pos_offset: usize,
    ) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let seq = hidden.len() / d;
        let n_heads = cfg.n_heads;
        let n_kv = cfg.resolved_n_kv_heads();
        let hd = cfg.resolved_head_dim();
        let total = pos_offset + seq;
        let group = cfg.kv_group_size();

        // norm1
        let mut h = hidden.to_vec();
        for i in 0..seq {
            let r = &mut h[i * d..(i + 1) * d];
            match cfg.norm {
                crate::config::Norm::RmsNorm => ops::rms_norm(r, &layer.norm1_w, cfg.norm_eps),
                crate::config::Norm::LayerNorm => ops::layer_norm(
                    r,
                    &layer.norm1_w,
                    layer.norm1_b.as_ref().unwrap(),
                    cfg.norm_eps,
                ),
            }
        }

        // q/k/v projections, kept as (head, seq, hd) for attention/rope.
        let mut q = vec![0.0f32; n_heads * seq * hd];
        let mut k = vec![0.0f32; n_kv * seq * hd];
        let mut v = vec![0.0f32; n_kv * seq * hd];
        gemm_to_head_major(&layer.q, &h, seq, n_heads, hd, &mut q, layer.q_b.as_ref());
        gemm_to_head_major(&layer.k, &h, seq, n_kv, hd, &mut k, layer.k_b.as_ref());
        gemm_to_head_major(&layer.v, &h, seq, n_kv, hd, &mut v, layer.v_b.as_ref());

        if let Some((cos, sin)) = &self.rope {
            let half = hd / 2;
            apply_rope(&mut q, n_heads, seq, half, pos_offset, cos, sin);
            apply_rope(&mut k, n_kv, seq, half, pos_offset, cos, sin);
        }

        let use_cache = cache.is_some();
        if let Some(c) = cache.as_deref_mut() {
            c.append(layer_idx, &k, &v, seq);
        }
        // After appending, hand the cache to the workers read-only.
        let cache_ref: Option<&KVCache> = cache.as_deref();

        let mut attn = vec![0.0f32; n_heads * seq * hd];
        let attn_shared = UnsafeSlice::new(&mut attn);
        let scale = cfg.attn_scale();
        // Attention is negligible at decode sizes (seq=1): run it inline.
        // At prefill sizes it is compute-heavy and worth the full fan-out.
        let attn_workers = if n_heads * seq * total * hd < PAR_WORK_MACS { 1 } else { usize::MAX };
        par_range_capped(n_heads, 1, attn_workers, |h0, h1| {
            let mut scores = vec![0.0f32; total];
            let mut acc = vec![0.0f32; hd];
            for h in h0..h1 {
                let kv_h = h / group;
                let keys: &[f32] = if use_cache {
                    cache_ref.unwrap().head_keys(layer_idx, kv_h, total)
                } else {
                    &k[kv_h * seq * hd..]
                };
                let vals: &[f32] = if use_cache {
                    cache_ref.unwrap().head_values(layer_idx, kv_h, total)
                } else {
                    &v[kv_h * seq * hd..]
                };
                for i in 0..seq {
                    let qr = &q[(h * seq + i) * hd..(h * seq + i + 1) * hd];
                    let valid = pos_offset + i + 1; // causal: keys 0..valid
                    for j in 0..valid {
                        scores[j] = simd::dot_f32(qr, &keys[j * hd..(j + 1) * hd]) * scale;
                    }
                    for j in valid..total {
                        scores[j] = -1e9;
                    }
                    ops::softmax_row(&mut scores);
                    for a in acc.iter_mut() {
                        *a = 0.0;
                    }
                    for j in 0..valid {
                        let p = scores[j];
                        for (a, vv) in acc.iter_mut().zip(&vals[j * hd..(j + 1) * hd]) {
                            *a += p * vv;
                        }
                    }
                    unsafe { attn_shared.slice_mut((h * seq + i) * hd..(h * seq + i + 1) * hd) }
                        .copy_from_slice(&acc);
                }
            }
        });

        // (heads, seq, hd) -> (seq, heads*hd), then o projection + residual.
        let mut merged = vec![0.0f32; seq * n_heads * hd];
        for h in 0..n_heads {
            for i in 0..seq {
                merged[i * n_heads * hd + h * hd..i * n_heads * hd + (h + 1) * hd]
                    .copy_from_slice(&attn[(h * seq + i) * hd..(h * seq + i + 1) * hd]);
            }
        }
        let mut out = vec![0.0f32; seq * d];
        layer.o.gemm_into(&merged, seq, &mut out);
        add_bias(&mut out, layer.o_b.as_ref());
        for (dst, src) in hidden.iter_mut().zip(out.iter()) {
            *dst += src;
        }
    }

    fn mlp_block(&self, layer: &Layer, hidden: &mut [f32]) {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let seq = hidden.len() / d;
        let m = cfg.d_mlp;

        let mut h = hidden.to_vec();
        for i in 0..seq {
            let r = &mut h[i * d..(i + 1) * d];
            match cfg.norm {
                crate::config::Norm::RmsNorm => ops::rms_norm(r, &layer.norm2_w, cfg.norm_eps),
                crate::config::Norm::LayerNorm => ops::layer_norm(
                    r,
                    &layer.norm2_w,
                    layer.norm2_b.as_ref().unwrap(),
                    cfg.norm_eps,
                ),
            }
        }

        let mut out = vec![0.0f32; seq * d];
        match cfg.activation {
            crate::config::Activation::Gelu => {
                let mut mid = vec![0.0f32; seq * m];
                layer.fc.as_ref().unwrap().gemm_into(&h, seq, &mut mid);
                add_bias(&mut mid, layer.fc_b.as_ref());
                for v in mid.iter_mut() {
                    *v = ops::gelu(*v);
                }
                layer.proj.as_ref().unwrap().gemm_into(&mid, seq, &mut out);
                add_bias(&mut out, layer.proj_b.as_ref());
            }
            crate::config::Activation::Swiglu => {
                let mut gate = vec![0.0f32; seq * m];
                let mut up = vec![0.0f32; seq * m];
                layer.gate.as_ref().unwrap().gemm_into(&h, seq, &mut gate);
                layer.up.as_ref().unwrap().gemm_into(&h, seq, &mut up);
                for (g, u) in gate.iter_mut().zip(up.iter_mut()) {
                    *g = ops::silu(*g) * *u;
                }
                layer.down.as_ref().unwrap().gemm_into(&gate, seq, &mut out);
            }
        }
        for (dst, src) in hidden.iter_mut().zip(out.iter()) {
            *dst += src;
        }
    }
}

/// Project x (seq, inp) through a linear layer and write the result directly
/// in head-major (heads, seq, hd) layout, adding an optional bias.
fn gemm_to_head_major(
    mat: &Mat,
    x: &[f32],
    seq: usize,
    heads: usize,
    hd: usize,
    out: &mut [f32],
    bias: Option<&Vec<f32>>,
) {
    let mut flat = vec![0.0f32; seq * heads * hd];
    mat.gemm_into(x, seq, &mut flat);
    if let Some(b) = bias {
        for row in 0..seq {
            for j in 0..heads * hd {
                flat[row * heads * hd + j] += b[j];
            }
        }
    }
    for h in 0..heads {
        for i in 0..seq {
            out[(h * seq + i) * hd..(h * seq + i + 1) * hd]
                .copy_from_slice(&flat[i * heads * hd + h * hd..i * heads * hd + (h + 1) * hd]);
        }
    }
}

fn add_bias(y: &mut [f32], bias: Option<&Vec<f32>>) {
    if let Some(b) = bias {
        for (i, v) in y.iter_mut().enumerate() {
            *v += b[i % b.len()];
        }
    }
}

fn apply_rope(
    x: &mut [f32],
    n_heads: usize,
    seq: usize,
    half: usize,
    pos_offset: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let hd = half * 2;
    for h in 0..n_heads {
        for i in 0..seq {
            let row = &mut x[(h * seq + i) * hd..(h * seq + i + 1) * hd];
            let p = pos_offset + i;
            let c = &cos[p * half..(p + 1) * half];
            let s = &sin[p * half..(p + 1) * half];
            for j in 0..half {
                let x1 = row[j];
                let x2 = row[j + half];
                row[j] = x1 * c[j] - x2 * s[j];
                row[j + half] = x1 * s[j] + x2 * c[j];
            }
        }
    }
}
