//! Debug helper: print intermediate tensors of the forward pass for
//! cross-engine comparison. Not part of the public CLI.
//!
//!     cargo run --release --example dump -- tests/fixtures/tiny-gpt2 283,309,308

use nimbus_infer_rs::engine::{Engine, QuantMode};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let ids: Vec<u32> = args[2]
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let engine = Engine::load(std::path::Path::new(dir), QuantMode::F32).unwrap();
    let cfg = engine.config();

    // raw wte bytes: first 4 floats of token 0
    let mut row = vec![0.0f32; cfg.d_model];
    engine.model.wte.row_into(0, &mut row);
    println!("wte[0][:4] = {:?}", &row[..4]);

    let mut hidden = Vec::new();
    engine.model.forward_hidden(&ids, None, &mut hidden);
    let d = cfg.d_model;
    println!("hidden[0][:4]  = {:?}", &hidden[0..4]);
    println!("hidden[last][:4] = {:?}", &hidden[(ids.len() - 1) * d..(ids.len() - 1) * d + 4]);

    let mut logits = Vec::new();
    let last = &hidden[(ids.len() - 1) * d..ids.len() * d];
    engine.model.logits_last(last, &mut logits);
    println!("logits[:4] = {:?}", &logits[..4]);
    println!("n_params = {}", engine.model.n_params());
}
