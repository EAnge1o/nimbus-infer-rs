//! End-to-end tests against a real checkpoint directory (created by the
//! Python engine, see benchmarks/cross_engine.py --parity). The fixture is
//! committed so CI needs no downloads.

use nimbus_infer_rs::engine::{Engine, QuantMode};
use nimbus_infer_rs::sampler::SamplingParams;

fn fixture() -> Engine {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny-gpt2");
    Engine::load(&path, QuantMode::F32).expect("fixture checkpoint loads")
}

#[test]
fn generates_deterministically() {
    let engine = fixture();
    let prompt = engine.encode("the engine caches keys and values").unwrap();
    let params = SamplingParams { max_new_tokens: 24, temperature: 0.0, ..Default::default() };
    let a = engine.generate_ids(&prompt, &params, None).unwrap();
    let b = engine.generate_ids(&prompt, &params, None).unwrap();
    assert_eq!(a.len(), 24);
    assert_eq!(a, b, "greedy generation must be deterministic");
}

#[test]
fn kv_cache_matches_full_recompute() {
    // Decode with a cache must produce the same tokens as re-running the
    // whole sequence every step (the Python engine proves the same property).
    let engine = fixture();
    let prompt = engine.encode("the engine caches keys and values").unwrap();
    let n = 8;

    let cached = engine
        .generate_ids(&prompt, &SamplingParams { max_new_tokens: n, temperature: 0.0, ..Default::default() }, None)
        .unwrap();

    // No-cache path: greedy over growing context, reimplemented via the model.
    let mut context = prompt.clone();
    let mut nocache = Vec::new();
    let mut hidden = Vec::new();
    let mut logits = Vec::new();
    for _ in 0..n {
        engine.model.forward_hidden(&context, None, &mut hidden);
        let d = engine.config().d_model;
        let last = &hidden[(context.len() - 1) * d..context.len() * d];
        engine.model.logits_last(last, &mut logits);
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        nocache.push(best as u32);
        context.push(best as u32);
    }
    assert_eq!(cached, nocache, "cached and full-recompute paths must agree");
}

#[test]
fn tokenizer_roundtrips_any_bytes() {
    let engine = fixture();
    let tok = engine.tokenizer.as_ref().unwrap();
    for text in [
        "hello world",
        "  spaces\tand\nnewlines  ",
        "中文混排 with emoji 😀!",
        "it's the engine's 123 tokens, don't you know?",
        "\u{0}\u{1} control chars \u{7f}",
    ] {
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text, "roundtrip failed for {text:?}");
    }
}

#[test]
fn int8_tracks_fp32_and_int4_stays_deterministic() {
    // On a *random-weight* model the logit gaps are the same order as the
    // 4-bit quantization grid, so int4 may legitimately flip an argmax (it
    // does not on trained weights -- see the benchmark table in the README).
    // What must hold at any bit width: generation stays deterministic and the
    // in-memory weight footprint shrinks.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tiny-gpt2");
    let fp32 = Engine::load(&path, QuantMode::F32).unwrap();
    let prompt = fp32.encode("the engine caches").unwrap();
    let params = SamplingParams { max_new_tokens: 16, temperature: 0.0, ..Default::default() };
    let base = fp32.generate_ids(&prompt, &params, None).unwrap();
    let base_bytes = fp32.model.weight_bytes();

    let int8 = Engine::load(&path, QuantMode::Int8).unwrap();
    assert_eq!(
        int8.generate_ids(&prompt, &params, None).unwrap(),
        base,
        "int8 must keep the fp32 greedy path on the tiny model"
    );
    assert!(int8.model.weight_bytes() < base_bytes);

    let int4 = Engine::load(&path, QuantMode::Int4).unwrap();
    let a = int4.generate_ids(&prompt, &params, None).unwrap();
    let b = int4.generate_ids(&prompt, &params, None).unwrap();
    assert_eq!(a, b, "int4 generation must be deterministic");
    assert!(int4.model.weight_bytes() < int8.model.weight_bytes());
    assert!(int4.model.weight_bytes() * 2 < base_bytes, "int4 should be ~1/4 of fp32");
}
