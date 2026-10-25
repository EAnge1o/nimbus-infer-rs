//! CLI: `nimbus-rs <command> [options]`
//!
//! Commands: generate | bench | bench-kernel | tokenize | inspect | logits

use std::path::PathBuf;
use std::time::Instant;

use nimbus_infer_rs::engine::{Engine, QuantMode};
use nimbus_infer_rs::sampler::SamplingParams;
use nimbus_infer_rs::simd;
use nimbus_infer_rs::threads::par_range;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        std::process::exit(2);
    }
    let result = match args[0].as_str() {
        "generate" => cmd_generate(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "bench-kernel" => cmd_bench_kernel(),
        "tokenize" => cmd_tokenize(&args[1..]),
        "inspect" => cmd_inspect(&args[1..]),
        "logits" => cmd_logits(&args[1..]),
        "--help" | "-h" | "help" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("unknown command {other:?}")),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn print_usage() {
    println!(
        "nimbus-rs -- a from-scratch, std-only LLM inference engine

USAGE:
  nimbus-rs generate --model DIR [--prompt TEXT | --ids 1,2,3] [options]
  nimbus-rs bench    --model DIR [--prefill N] [--decode N] [--runs R]
                     [--quant fp32|int8|int4] [--json PATH]
  nimbus-rs logits   --model DIR --ids 1,2,3 [--quant fp32|int8|int4]
  nimbus-rs tokenize --model DIR --text TEXT
  nimbus-rs inspect  --model DIR [--quant fp32|int8|int4]
  nimbus-rs bench-kernel

generate options:
  --max-new-tokens N   tokens to generate (default 64)
  --temperature T      0 = greedy (default 0)
  --top-k K            0 disables (default 0)
  --top-p P            1.0 disables (default 1.0)
  --repetition-penalty R (default 1.0)
  --seed S             xoshiro256++ seed (default 0)
  --stream             print text as it is produced
  --quant MODE         quantize weights at load time (default fp32)"
    );
}

struct Flags {
    map: std::collections::HashMap<String, String>,
    #[allow(dead_code)] // kept for future subcommands that take positionals
    positional: Vec<String>,
}

fn parse_flags(args: &[String]) -> Flags {
    let mut map = std::collections::HashMap::new();
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if let Some(name) = arg.strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                map.insert(name.to_string(), args[i + 1].clone());
                i += 2;
            } else {
                map.insert(name.to_string(), String::new());
                i += 1;
            }
        } else {
            positional.push(arg.clone());
            i += 1;
        }
    }
    Flags { map, positional }
}

impl Flags {
    fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(|s| s.as_str())
    }

    fn require(&self, name: &str) -> Result<String, String> {
        self.get(name)
            .map(|s| s.to_string())
            .ok_or_else(|| format!("missing required flag --{name}"))
    }

    fn usize_of(&self, name: &str, default: usize) -> Result<usize, String> {
        match self.get(name) {
            Some(s) => s.parse::<usize>().map_err(|_| format!("--{name}: bad number {s:?}")),
            None => Ok(default),
        }
    }

    fn f32_of(&self, name: &str, default: f32) -> Result<f32, String> {
        match self.get(name) {
            Some(s) => s.parse::<f32>().map_err(|_| format!("--{name}: bad number {s:?}")),
            None => Ok(default),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}

fn load_engine(flags: &Flags) -> Result<Engine, String> {
    let model: PathBuf = flags.require("model")?.into();
    let quant = match flags.get("quant") {
        Some(s) => QuantMode::parse(s)?,
        None => QuantMode::F32,
    };
    Engine::load(&model, quant)
}

fn parse_ids(text: &str) -> Result<Vec<u32>, String> {
    text.split(',')
        .map(|p| p.trim().parse::<u32>().map_err(|_| format!("bad token id {p:?}")))
        .collect()
}

fn cmd_generate(args: &[String]) -> Result<(), String> {
    let flags = parse_flags(args);
    let engine = load_engine(&flags)?;
    let prompt: Vec<u32> = if let Some(text) = flags.get("prompt") {
        engine.encode(text)?
    } else if let Some(ids) = flags.get("ids") {
        parse_ids(ids)?
    } else {
        return Err("pass --prompt TEXT or --ids 1,2,3".into());
    };

    let params = SamplingParams {
        max_new_tokens: flags.usize_of("max-new-tokens", 64)?,
        temperature: flags.f32_of("temperature", 0.0)?,
        top_k: flags.usize_of("top-k", 0)?,
        top_p: flags.f32_of("top-p", 1.0)?,
        repetition_penalty: flags.f32_of("repetition-penalty", 1.0)?,
        seed: flags.get("seed").and_then(|s| s.parse().ok()),
        stop_tokens: Vec::new(),
    };

    let start = Instant::now();
    if flags.has("stream") {
        let text = engine.generate_stream(&prompt, &params)?;
        let dt = start.elapsed().as_secs_f64();
        println!("{text}");
        eprintln!(
            "[{} tokens in {:.2}s = {:.1} tok/s]",
            params.max_new_tokens.min(engine.config().max_seq_len - prompt.len()),
            dt,
            params.max_new_tokens.min(engine.config().max_seq_len - prompt.len()) as f64 / dt
        );
    } else {
        let ids = engine.generate_ids(&prompt, &params, None)?;
        let dt = start.elapsed().as_secs_f64();
        let text = engine
            .tokenizer
            .as_ref()
            .map(|tok| tok.decode(&ids))
            .unwrap_or_else(|| format!("{ids:?}"));
        if flags.has("json") {
            println!(
                "{{\"prompt_tokens\": {}, \"generated_ids\": {:?}, \"text\": {:?}}}",
                prompt.len(),
                ids,
                text
            );
        } else {
            println!("{text}");
        }
        eprintln!("[{} tokens in {:.2}s = {:.1} tok/s]", ids.len(), dt, ids.len() as f64 / dt);
    }
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let flags = parse_flags(args);
    let prefill_tokens = flags.usize_of("prefill", 128)?;
    let decode_tokens = flags.usize_of("decode", 64)?;
    let runs = flags.usize_of("runs", 3)?;
    let engine = load_engine(&flags)?;
    let cfg = engine.config();

    // Deterministic prompt ids in the valid vocab range.
    let vocab = cfg.vocab_size as u64;
    let prompt: Vec<u32> = (0..prefill_tokens)
        .map(|i| ((i as u64 * 2654435761 + 12345) % (vocab - 8) + 8) as u32)
        .collect();

    // Warmup.
    let _ = engine.generate_ids(&prompt, &SamplingParams { max_new_tokens: 2, ..Default::default() }, None);

    let mut best_prefill = f64::INFINITY;
    let mut best_decode = f64::INFINITY;
    let mut best_e2e = f64::INFINITY;
    let mut last_generated = Vec::new();
    for _ in 0..runs {
        let mut cache = engine.model.new_cache();
        let mut hidden = Vec::new();
        let mut logits = Vec::new();
        let t0 = Instant::now();
        engine.model.forward_hidden(&prompt, Some(&mut cache), &mut hidden);
        cache.advance(prompt.len());
        let last = &hidden[(prompt.len() - 1) * cfg.d_model..];
        engine.model.logits_last(last, &mut logits);
        let t1 = Instant::now();

        let params = SamplingParams { max_new_tokens: decode_tokens, ..Default::default() };
        let mut rng = nimbus_infer_rs::sampler::Rng::new(1);
        let mut generated = Vec::with_capacity(decode_tokens);
        for step in 0..decode_tokens {
            let token = nimbus_infer_rs::sampler::sample(&logits, &params, &generated, &mut rng);
            generated.push(token);
            if step == decode_tokens - 1 {
                break;
            }
            engine.model.forward_hidden(&[token], Some(&mut cache), &mut hidden);
            cache.advance(1);
            engine.model.logits_last(&hidden[..cfg.d_model], &mut logits);
        }
        let t2 = Instant::now();
        last_generated = generated;

        let prefill_s = (t1 - t0).as_secs_f64();
        let decode_s = (t2 - t1).as_secs_f64();
        best_prefill = best_prefill.min(prefill_tokens as f64 / prefill_s);
        best_decode = best_decode.min(decode_tokens as f64 / decode_s);
        best_e2e = best_e2e.min(decode_tokens as f64 / (t2 - t0).as_secs_f64());
    }

    let weight_mb = engine.model.weight_bytes() as f64 / (1024.0 * 1024.0);
    let kv_mb = engine.kv_cache_bytes() as f64 / (1024.0 * 1024.0);
    let result = serde_json_like(&[
        ("engine", "\"nimbus-infer-rs\"".to_string()),
        ("quant", format!("\"{}\"", engine.quant.as_str())),
        ("model", format!("\"{}\"", cfg.name)),
        ("params", format!("{}", engine.model.n_params())),
        ("prefill_tokens", format!("{prefill_tokens}")),
        ("decode_tokens", format!("{decode_tokens}")),
        ("runs", format!("{runs}")),
        ("prefill_tok_s", fmt3(best_prefill)),
        ("decode_tok_s", fmt3(best_decode)),
        ("e2e_tok_s", fmt3(best_e2e)),
        ("weight_mb", fmt2(weight_mb)),
        ("kv_cache_mb", fmt2(kv_mb)),
        ("first_generated_ids", format!("{:?}", &last_generated[..last_generated.len().min(8)])),
    ]);
    println!("{result}");

    if let Some(path) = flags.get("json") {
        std::fs::write(path, &result).map_err(|e| format!("writing {path}: {e}"))?;
    }
    Ok(())
}

fn fmt3(v: f64) -> String {
    format!("{v:.3}")
}

fn fmt2(v: f64) -> String {
    format!("{v:.2}")
}

/// Hand-rolled JSON object (keys are known-good, values are pre-formatted).
fn serde_json_like(entries: &[(&str, String)]) -> String {
    let body: Vec<String> = entries.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
    format!("{{\n  {}\n}}", body.join(",\n  "))
}

fn cmd_logits(args: &[String]) -> Result<(), String> {
    let flags = parse_flags(args);
    let engine = load_engine(&flags)?;
    let ids = parse_ids(&flags.require("ids")?)?;
    let cfg = engine.config();
    let mut hidden = Vec::new();
    let mut logits = Vec::new();
    engine.model.forward_hidden(&ids, None, &mut hidden);
    let last = &hidden[(ids.len() - 1) * cfg.d_model..];
    engine.model.logits_last(last, &mut logits);
    // First and last 4 logits + argmax: enough for cross-engine comparison.
    let head: Vec<String> = logits[..4].iter().map(|v| format!("{v:.6}")).collect();
    let tail: Vec<String> = logits[logits.len() - 4..].iter().map(|v| format!("{v:.6}")).collect();
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    println!(
        "{{\"argmax\": {best}, \"argmax_logit\": {:.6}, \"head\": [{}], \"tail\": [{}], \"sum\": {:.6}}}",
        logits[best],
        head.join(", "),
        tail.join(", "),
        logits.iter().map(|&v| v as f64).sum::<f64>()
    );
    Ok(())
}

fn cmd_tokenize(args: &[String]) -> Result<(), String> {
    let flags = parse_flags(args);
    let engine = load_engine(&flags)?;
    let text = flags.require("text")?;
    let tok = engine
        .tokenizer
        .as_ref()
        .ok_or("checkpoint has no tokenizer")?;
    let ids = tok.encode(&text);
    let roundtrip = tok.decode(&ids);
    println!(
        "{{\"ids\": {:?}, \"pieces\": {:?}, \"roundtrip_ok\": {}}}",
        ids,
        ids.iter().map(|&i| tok.decode(&[i])).collect::<Vec<_>>(),
        roundtrip == text
    );
    Ok(())
}

fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let flags = parse_flags(args);
    let engine = load_engine(&flags)?;
    let cfg = engine.config();
    println!(
        "{{\"name\": \"{}\", \"n_params\": {}, \"weight_mb\": {}, \"kv_cache_mb\": {}, \
         \"vocab\": {}, \"layers\": {}, \"heads\": {}, \"kv_heads\": {}, \"d_model\": {}, \
         \"head_dim\": {}, \"max_seq_len\": {}, \"norm\": \"{}\", \"activation\": \"{}\", \
         \"pos\": \"{}\", \"quant\": \"{}\", \"tokenizer_vocab\": {}}}",
        cfg.name,
        engine.model.n_params(),
        fmt2(engine.model.weight_bytes() as f64 / (1024.0 * 1024.0)),
        fmt2(engine.kv_cache_bytes() as f64 / (1024.0 * 1024.0)),
        cfg.vocab_size,
        cfg.n_layers,
        cfg.n_heads,
        cfg.resolved_n_kv_heads(),
        cfg.d_model,
        cfg.resolved_head_dim(),
        cfg.max_seq_len,
        match cfg.norm {
            nimbus_infer_rs::config::Norm::LayerNorm => "layernorm",
            nimbus_infer_rs::config::Norm::RmsNorm => "rmsnorm",
        },
        match cfg.activation {
            nimbus_infer_rs::config::Activation::Gelu => "gelu",
            nimbus_infer_rs::config::Activation::Swiglu => "swiglu",
        },
        match cfg.pos_encoding {
            nimbus_infer_rs::config::PosEncoding::Learned => "learned",
            nimbus_infer_rs::config::PosEncoding::Rope => "rope",
        },
        engine.quant.as_str(),
        engine.tokenizer.as_ref().map_or(0, |t| t.vocab_size()),
    );
    Ok(())
}

/// Microbenchmark the three dot-product kernels at GPT-2 lm_head scale.
fn cmd_bench_kernel() -> Result<(), String> {
    const OUT: usize = 50257;
    const IN: usize = 768;
    let x: Vec<f32> = (0..IN).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
    let w_f32: Vec<f32> = (0..OUT * IN)
        .map(|i| (((i as u64 * 2654435761) % 2000) as f32 - 1000.0) / 10000.0)
        .collect();
    let w_i8: Vec<u8> = (0..OUT * IN)
        .map(|i| ((i as u64 * 2654435761) % 256) as u8)
        .collect();
    let w_i4: Vec<u8> = (0..OUT * IN / 2)
        .map(|i| ((i as u64 * 2654435761) % 256) as u8)
        .collect();

    let mut y = vec![0.0f32; OUT];

    let f32_s = {
        let y_shared = nimbus_infer_rs::threads::UnsafeSlice::new(&mut y);
        let t0 = Instant::now();
        par_range(OUT, PAR_CHUNK, |j0, j1| {
            for j in j0..j1 {
                let v = simd::dot_f32(&x, &w_f32[j * IN..(j + 1) * IN]);
                unsafe { y_shared.write(j, v) };
            }
        });
        drop(y_shared);
        t0.elapsed().as_secs_f64()
    };
    let check_f32 = y[0];

    let i8_s = {
        let y_shared = nimbus_infer_rs::threads::UnsafeSlice::new(&mut y);
        let t0 = Instant::now();
        par_range(OUT, PAR_CHUNK, |j0, j1| {
            for j in j0..j1 {
                let v = simd::dot_f32_i8(&x, &w_i8[j * IN..(j + 1) * IN]);
                unsafe { y_shared.write(j, v) };
            }
        });
        drop(y_shared);
        t0.elapsed().as_secs_f64()
    };
    let check_i8 = y[0];

    let i4_s = {
        let y_shared = nimbus_infer_rs::threads::UnsafeSlice::new(&mut y);
        let t0 = Instant::now();
        par_range(OUT, PAR_CHUNK, |j0, j1| {
            for j in j0..j1 {
                let v = simd::dot_f32_i4(&x, &w_i4[j * IN / 2..], IN);
                unsafe { y_shared.write(j, v) };
            }
        });
        drop(y_shared);
        t0.elapsed().as_secs_f64()
    };
    let check_i4 = y[0];

    let flops = 2.0 * OUT as f64 * IN as f64;
    println!(
        "{{\"kernel\": \"dot 50257x768\", \"f32_gflops\": {:.1}, \"int8_gflops\": {:.1}, \"int4_gflops\": {:.1}, \
         \"f32_ms\": {:.1}, \"int8_ms\": {:.1}, \"int4_ms\": {:.1}, \"checks\": [{:.4}, {:.4}, {:.4}]}}",
        flops / f32_s / 1e9,
        flops / i8_s / 1e9,
        flops / i4_s / 1e9,
        f32_s * 1000.0,
        i8_s * 1000.0,
        i4_s * 1000.0,
        check_f32,
        check_i8,
        check_i4,
    );
    Ok(())
}

const PAR_CHUNK: usize = 1024;
