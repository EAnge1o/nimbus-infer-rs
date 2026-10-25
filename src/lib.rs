//! # nimbus-infer-rs
//!
//! A from-scratch, **std-only** LLM inference engine: byte-level BPE
//! tokenizer, prefill/decode with a KV cache, GPT-2 and LLaMA architectures,
//! hand-written AVX2/FMA kernels and int8/int4 weight quantization -- zero
//! external crates.
//!
//! It reads checkpoints produced by the Python engine
//! [nimbus-infer](https://github.com/EAnge1o/nimbus-infer) directly:
//! `config.json` + `weights.safetensors` + `tokenizer/{vocab.json,merges.txt}`.

pub mod cache;
pub mod config;
pub mod engine;
pub mod json;
pub mod model;
pub mod ops;
pub mod quant;
pub mod safetensors;
pub mod sampler;
pub mod simd;
pub mod threads;
pub mod tokenizer;
