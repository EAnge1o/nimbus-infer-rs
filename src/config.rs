//! Model configuration, mirroring the Python engine's `ModelConfig` (same
//! field names in `config.json`, same defaults, same derived properties) so a
//! checkpoint written by one engine loads in the other without conversion.

use crate::json::Json;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_mlp: usize,
    pub max_seq_len: usize,
    pub head_dim: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub norm: Norm,
    pub activation: Activation,
    pub pos_encoding: PosEncoding,
    pub tie_embeddings: bool,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub scale_attn_by_head_dim: bool,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    LayerNorm,
    RmsNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
    Swiglu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosEncoding {
    Learned,
    Rope,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            vocab_size: 50257,
            n_layers: 12,
            n_heads: 12,
            d_model: 768,
            d_mlp: 3072,
            max_seq_len: 1024,
            head_dim: None,
            n_kv_heads: None,
            norm: Norm::LayerNorm,
            activation: Activation::Gelu,
            pos_encoding: PosEncoding::Learned,
            tie_embeddings: true,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            scale_attn_by_head_dim: true,
            name: "model".to_string(),
        }
    }
}

impl ModelConfig {
    pub fn from_json(value: &Json) -> Result<ModelConfig, String> {
        let mut cfg = ModelConfig::default();
        if let Some(v) = value.get("vocab_size").and_then(Json::as_usize) {
            cfg.vocab_size = v;
        }
        if let Some(v) = value.get("n_layers").and_then(Json::as_usize) {
            cfg.n_layers = v;
        }
        if let Some(v) = value.get("n_heads").and_then(Json::as_usize) {
            cfg.n_heads = v;
        }
        if let Some(v) = value.get("d_model").and_then(Json::as_usize) {
            cfg.d_model = v;
        }
        if let Some(v) = value.get("d_mlp").and_then(Json::as_usize) {
            cfg.d_mlp = v;
        }
        if let Some(v) = value.get("max_seq_len").and_then(Json::as_usize) {
            cfg.max_seq_len = v;
        }
        if let Some(v) = value.get("head_dim").and_then(Json::as_usize) {
            cfg.head_dim = Some(v);
        }
        if let Some(v) = value.get("n_kv_heads").and_then(Json::as_usize) {
            cfg.n_kv_heads = Some(v);
        }
        if let Some(v) = value.get("norm").and_then(Json::as_str) {
            cfg.norm = match v {
                "layernorm" => Norm::LayerNorm,
                "rmsnorm" => Norm::RmsNorm,
                other => return Err(format!("unknown norm {other:?}")),
            };
        }
        if let Some(v) = value.get("activation").and_then(Json::as_str) {
            cfg.activation = match v {
                "gelu" => Activation::Gelu,
                "swiglu" => Activation::Swiglu,
                other => return Err(format!("unknown activation {other:?}")),
            };
        }
        if let Some(v) = value.get("pos_encoding").and_then(Json::as_str) {
            cfg.pos_encoding = match v {
                "learned" => PosEncoding::Learned,
                "rope" => PosEncoding::Rope,
                other => return Err(format!("unknown pos_encoding {other:?}")),
            };
        }
        if let Some(v) = value.get("tie_embeddings").and_then(Json::as_bool) {
            cfg.tie_embeddings = v;
        }
        if let Some(v) = value.get("rope_theta").and_then(Json::as_f64) {
            cfg.rope_theta = v as f32;
        }
        if let Some(v) = value.get("norm_eps").and_then(Json::as_f64) {
            cfg.norm_eps = v as f32;
        }
        if let Some(v) = value.get("scale_attn_by_head_dim").and_then(Json::as_bool) {
            cfg.scale_attn_by_head_dim = v;
        }
        if let Some(v) = value.get("name").and_then(Json::as_str) {
            cfg.name = v.to_string();
        }
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &std::path::Path) -> Result<ModelConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let json = crate::json::parse(&text).map_err(|e| e.to_string())?;
        Self::from_json(&json)
    }

    fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 && self.head_dim.is_none() {
            return Err("d_model must be divisible by n_heads when head_dim is unset".into());
        }
        if let Some(n_kv) = self.n_kv_heads {
            if self.n_heads % n_kv != 0 {
                return Err("n_heads must be divisible by n_kv_heads".into());
            }
        }
        Ok(())
    }

    pub fn resolved_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.d_model / self.n_heads)
    }

    pub fn resolved_n_kv_heads(&self) -> usize {
        self.n_kv_heads.unwrap_or(self.n_heads)
    }

    pub fn attn_scale(&self) -> f32 {
        if self.scale_attn_by_head_dim {
            (self.resolved_head_dim() as f32).powf(-0.5)
        } else {
            1.0
        }
    }

    /// How many query heads share one KV head (1 = plain multi-head attention).
    pub fn kv_group_size(&self) -> usize {
        self.n_heads / self.resolved_n_kv_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn parses_a_llama_style_config() {
        let text = r#"{
            "name": "smol", "vocab_size": 49152, "n_layers": 30, "n_heads": 9,
            "n_kv_heads": 3, "d_model": 576, "d_mlp": 1536, "max_seq_len": 2048,
            "norm": "rmsnorm", "activation": "swiglu", "pos_encoding": "rope",
            "tie_embeddings": true, "rope_theta": 100000.0, "norm_eps": 1e-5
        }"#;
        let cfg = ModelConfig::from_json(&json::parse(text).unwrap()).unwrap();
        assert_eq!(cfg.resolved_head_dim(), 64);
        assert_eq!(cfg.resolved_n_kv_heads(), 3);
        assert_eq!(cfg.kv_group_size(), 3);
        assert!((cfg.attn_scale() - 0.125).abs() < 1e-6);
        assert_eq!(cfg.norm, Norm::RmsNorm);
        assert_eq!(cfg.activation, Activation::Swiglu);
    }

    #[test]
    fn rejects_bad_head_divisibility() {
        let text = r#"{"n_heads": 12, "n_kv_heads": 5, "d_model": 768}"#;
        assert!(ModelConfig::from_json(&json::parse(text).unwrap()).is_err());
    }
}
