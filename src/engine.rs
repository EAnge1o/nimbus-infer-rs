//! High-level engine: load a nimbus-infer checkpoint (config.json +
//! safetensors weights + tokenizer) and run the prefill/decode generation
//! loop. Checkpoints are byte-compatible with the Python engine -- the same
//! directory loads in either implementation.

use std::path::Path;

use crate::config::{Activation, ModelConfig, Norm, PosEncoding};
use crate::json;
use crate::model::{Mat, Model};
use crate::quant::QuantMatrix;
use crate::sampler::{sample, Rng, SamplingParams};
use crate::safetensors::SafeTensors;
use crate::tokenizer::BPETokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    F32,
    Int8,
    Int4,
}

impl QuantMode {
    pub fn parse(s: &str) -> Result<QuantMode, String> {
        match s {
            "fp32" | "f32" | "none" => Ok(QuantMode::F32),
            "int8" | "i8" => Ok(QuantMode::Int8),
            "int4" | "i4" => Ok(QuantMode::Int4),
            other => Err(format!(
                "unknown quant mode {other:?} (expected fp32 | int8 | int4)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            QuantMode::F32 => "fp32",
            QuantMode::Int8 => "int8",
            QuantMode::Int4 => "int4",
        }
    }
}

pub struct Engine {
    pub model: Model,
    pub tokenizer: Option<BPETokenizer>,
    pub quant: QuantMode,
}

/// Which 2-D matrices get quantized (everything except the final norm).
fn quantize_mat(mat: Mat, bits: u8, group_size: Option<usize>) -> Mat {
    match mat {
        Mat::Dense { wt, out, inp } => {
            Mat::Q(QuantMatrix::quantize(&wt, out, inp, bits, group_size))
        }
        other => other,
    }
}

/// Largest standard group size that divides `inp` evenly (int4 needs an even
/// group so packed nibble segments stay byte-aligned). d_model is not always
/// a multiple of 128 -- e.g. SmolLM2's is 576 -- so fall back to 64, 32, and
/// finally per-channel.
fn pick_group(inp: usize) -> Option<usize> {
    [128, 64, 32].into_iter().find(|&g| g % 2 == 0 && inp % g == 0)
}

impl Engine {
    /// Load a checkpoint directory produced by `Engine.save_pretrained` (or
    /// `python -m nimbus_infer convert`), optionally quantizing 2-D weights.
    pub fn load(dir: &Path, quant: QuantMode) -> Result<Engine, String> {
        let cfg = ModelConfig::load(&dir.join("config.json"))?;

        let weights_path = ["weights.safetensors", "model.safetensors"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| format!("no weights.safetensors in {}", dir.display()))?;
        let st = SafeTensors::open(&weights_path).map_err(|e| e.to_string())?;

        let take = |name: &str| -> Result<Vec<f32>, String> {
            st.take_f32(name).map_err(|e| e.to_string())
        };
        let take_mat =
            |name: &str| -> Result<Mat, String> {
                let info = st
                    .tensors
                    .get(name)
                    .ok_or_else(|| format!("missing tensor '{name}'"))?;
                if info.shape.len() != 2 {
                    return Err(format!("'{name}' is not 2-D"));
                }
                let (inp, out) = (info.shape[0], info.shape[1]); // stored (in, out)
                let flat = st.take_f32(name).map_err(|e| e.to_string())?;
                // Transpose (in, out) -> (out, in) once, at load time.
                let mut wt = vec![0.0f32; out * inp];
                for i in 0..inp {
                    for j in 0..out {
                        wt[j * inp + i] = flat[i * out + j];
                    }
                }
                let mat = Mat::dense(wt, out, inp);
                Ok(match quant {
                    QuantMode::F32 => mat,
                    QuantMode::Int8 => quantize_mat(mat, 8, None),
                    QuantMode::Int4 => quantize_mat(mat, 4, pick_group(inp)),
                })
            };

        let wte_info = st
            .tensors
            .get("wte")
            .ok_or("missing tensor 'wte'")?;
        if wte_info.shape != vec![cfg.vocab_size, cfg.d_model] {
            return Err(format!(
                "wte shape {:?} does not match config ({}, {})",
                wte_info.shape, cfg.vocab_size, cfg.d_model
            ));
        }
        let wte_flat = take("wte")?;
        let wte = match quant {
            QuantMode::F32 => Mat::dense(wte_flat, cfg.vocab_size, cfg.d_model),
            QuantMode::Int8 | QuantMode::Int4 => {
                // Embeddings must support row gather AND lm_head GEMV.
                let (out, inp) = (cfg.vocab_size, cfg.d_model);
                let bits = if quant == QuantMode::Int8 { 8 } else { 4 };
                let group = if quant == QuantMode::Int8 { None } else { pick_group(inp) };
                Mat::Q(QuantMatrix::quantize(&wte_flat, out, inp, bits, group))
            }
        };

        let wpe = if cfg.pos_encoding == PosEncoding::Learned {
            Some(take("wpe")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = move |suffix: &str| format!("h.{i}.{suffix}");
            let norm1_w = take(&p("norm1.weight"))?;
            let norm1_b = if cfg.norm == Norm::LayerNorm {
                Some(take(&p("norm1.bias"))?)
            } else {
                None
            };
            let norm2_w = take(&p("norm2.weight"))?;
            let norm2_b = if cfg.norm == Norm::LayerNorm {
                Some(take(&p("norm2.bias"))?)
            } else {
                None
            };
            // Bias tensors are optional per tensor (real GPT-2 checkpoints
            // carry them; LLaMA checkpoints and random-init checkpoints don't).
            let opt = |name: String| -> Result<Option<Vec<f32>>, String> {
                if st.tensors.contains_key(&name) {
                    Ok(Some(take(&name)?))
                } else {
                    Ok(None)
                }
            };
            let q = take_mat(&p("attn.q.weight"))?;
            let q_b = opt(p("attn.q.bias"))?;
            let k = take_mat(&p("attn.k.weight"))?;
            let k_b = opt(p("attn.k.bias"))?;
            let v = take_mat(&p("attn.v.weight"))?;
            let v_b = opt(p("attn.v.bias"))?;
            let o = take_mat(&p("attn.o.weight"))?;
            let o_b = opt(p("attn.o.bias"))?;

            let (fc, fc_b, proj, proj_b, gate, up, down) = match cfg.activation {
                Activation::Gelu => (
                    Some(take_mat(&p("mlp.fc.weight"))?),
                    opt(p("mlp.fc.bias"))?,
                    Some(take_mat(&p("mlp.proj.weight"))?),
                    opt(p("mlp.proj.bias"))?,
                    None,
                    None,
                    None,
                ),
                Activation::Swiglu => (
                    None,
                    None,
                    None,
                    None,
                    Some(take_mat(&p("mlp.gate.weight"))?),
                    Some(take_mat(&p("mlp.up.weight"))?),
                    Some(take_mat(&p("mlp.down.weight"))?),
                ),
            };

            layers.push(super::model::Layer {
                norm1_w,
                norm1_b,
                q,
                q_b,
                k,
                k_b,
                v,
                v_b,
                o,
                o_b,
                fc,
                fc_b,
                proj,
                proj_b,
                gate,
                up,
                down,
                norm2_w,
                norm2_b,
            });
        }

        let norm_f_w = take("norm_f.weight")?;
        let norm_f_b = if cfg.norm == Norm::LayerNorm {
            Some(take("norm_f.bias")?)
        } else {
            None
        };
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(take_mat("lm_head.weight")?)
        };

        let model = Model::new(cfg, wte, wpe, lm_head, layers, norm_f_w, norm_f_b);

        let tok_dir = dir.join("tokenizer");
        let tokenizer = match BPETokenizer::load(&tok_dir) {
            Ok(tok) => Some(tok),
            Err(e) => {
                eprintln!("warning: tokenizer not usable ({e}); generation by --ids still works");
                None
            }
        };

        Ok(Engine { model, tokenizer, quant })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.model.cfg
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or("this checkpoint has no tokenizer; pass --ids instead")?;
        Ok(tok.encode(text))
    }

    /// Prefill the prompt, then decode up to `params.max_new_tokens` tokens.
    /// Returns the generated ids (prompt not included).
    pub fn generate_ids(
        &self,
        prompt: &[u32],
        params: &SamplingParams,
        mut on_token: Option<&mut dyn FnMut(u32, usize)>,
    ) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err("prompt produced an empty token sequence".into());
        }
        let cfg = &self.model.cfg;
        let budget = cfg.max_seq_len.saturating_sub(prompt.len());
        let n_new = params.max_new_tokens.min(budget);

        let mut cache = self.model.new_cache();
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        let mut rng = Rng::new(params.seed.unwrap_or(0x5DEECE66D));

        self.model.forward_hidden(prompt, Some(&mut cache), &mut hidden);
        cache.advance(prompt.len());
        let last = &hidden[(prompt.len() - 1) * cfg.d_model..];
        self.model.logits_last(last, &mut logits);

        let mut generated = Vec::with_capacity(n_new);
        for step in 0..n_new {
            let token = sample(&logits, params, &generated, &mut rng);
            generated.push(token);
            if let Some(cb) = on_token.as_mut() {
                cb(token, step);
            }
            if params.stop_tokens.contains(&token) || step == n_new - 1 {
                break;
            }
            self.model.forward_hidden(&[token], Some(&mut cache), &mut hidden);
            cache.advance(1);
            self.model.logits_last(&hidden[..cfg.d_model], &mut logits);
        }
        Ok(generated)
    }

    /// Generate and stream text deltas (never splitting a character).
    pub fn generate_stream(
        &self,
        prompt: &[u32],
        params: &SamplingParams,
    ) -> Result<String, String> {
        let mut text = String::new();
        let mut stream = self
            .tokenizer
            .as_ref()
            .map(|t| t.incremental_decoder());
        let mut sink = |token: u32, _step: usize| {
            if let Some(s) = stream.as_mut() {
                text.push_str(&s.push(token));
            }
        };
        self.generate_ids(prompt, params, Some(&mut sink))?;
        if let Some(s) = stream.as_mut() {
            text.push_str(&s.flush());
        }
        Ok(text)
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.model.new_cache().bytes()
    }
}

/// Convenience: load a config.json standalone (used by tests).
pub fn load_config(path: &Path) -> Result<ModelConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let json = json::parse(&text).map_err(|e| e.to_string())?;
    ModelConfig::from_json(&json)
}
