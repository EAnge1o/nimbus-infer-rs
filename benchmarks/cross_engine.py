#!/usr/bin/env python3
"""Cross-engine parity + benchmark harness.

Creates a tiny random checkpoint with the Python engine (so both engines read
byte-identical weights), then compares logits and greedy generation between
the two implementations, and finally runs the same benchmark protocol on both.

Usage (from nimbus-infer-rs/, with the nimbus-infer repo cloned next to it):
    python benchmarks/cross_engine.py parity          # logits + token parity
    python benchmarks/cross_engine.py bench           # speed comparison table
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make the sibling Python engine importable.
sys.path.insert(0, str(ROOT.parent / "nimbus-infer"))

import numpy as np  # noqa: E402

from nimbus_infer import BPETokenizer, Engine, ModelConfig, SamplingParams  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "tiny-gpt2"
RS_BIN = ROOT / "target" / "release" / "nimbus-rs.exe"

PROMPT = "the engine caches keys and values"

CORPUS = (
    "nimbus infer is a from scratch inference engine written in pure numpy. "
    "the engine tokenizes text with a byte level byte pair encoding vocabulary. "
    "attention is causal and keys and values are cached between decode steps. "
    "grouped query attention reduces the cache while rotary embeddings encode position. "
    "the history of computing began with mechanical calculators and punch cards. "
) * 30


def make_tiny_checkpoint() -> Path:
    """Random-weight tiny GPT-2 with a real trained tokenizer."""
    if FIXTURE.exists():
        return FIXTURE
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=512)
    config = ModelConfig(
        name="tiny-gpt2",
        vocab_size=tokenizer.vocab_size,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_mlp=512,
        max_seq_len=256,
        norm="layernorm",
        activation="gelu",
        pos_encoding="learned",
        tie_embeddings=True,
    )
    engine = Engine.random(config, tokenizer=tokenizer, seed=42)
    engine.save_pretrained(FIXTURE)
    print(f"wrote fixture checkpoint to {FIXTURE}")
    return FIXTURE


def python_logits(ckpt: Path, ids: list[int]) -> dict:
    engine = Engine.from_pretrained(str(ckpt))
    hidden = engine.model.forward(ids, None)
    last = hidden[-1]
    return {
        "argmax": int(np.argmax(last)),
        "head": [round(float(v), 6) for v in last[:4]],
        "tail": [round(float(v), 6) for v in last[-4:]],
        "sum": round(float(last.astype(np.float64).sum()), 6),
        "max_abs": round(float(np.abs(last).max()), 6),
    }


def rust_logits(ckpt: Path, ids: list[int]) -> dict:
    out = subprocess.run(
        [str(RS_BIN), "logits", "--model", str(ckpt), "--ids", ",".join(map(str, ids))],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def greedy_python(ckpt: Path, n: int = 32) -> list[int]:
    engine = Engine.from_pretrained(str(ckpt))
    params = SamplingParams(max_new_tokens=n, temperature=0.0)
    return engine.generate_ids(PROMPT, params)


def greedy_rust(ckpt: Path, n: int = 32) -> list[int]:
    """Run greedy generation in Rust; returns generated ids via --json."""
    ids = rust_tokenize(ckpt, PROMPT)
    out = subprocess.run(
        [str(RS_BIN), "generate", "--model", str(ckpt), "--ids", ",".join(map(str, ids)),
         "--max-new-tokens", str(n), "--temperature", "0", "--json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["generated_ids"]


def rust_tokenize(ckpt: Path, text: str) -> list[int]:
    out = subprocess.run(
        [str(RS_BIN), "tokenize", "--model", str(ckpt), "--text", text],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["ids"]


def cmd_parity() -> int:
    ckpt = make_tiny_checkpoint()
    py_engine = Engine.from_pretrained(str(ckpt))
    ids = py_engine.encode(PROMPT)
    print(f"prompt ids ({len(ids)}): {ids[:12]}...")

    rs_tok = rust_tokenize(ckpt, PROMPT)
    tok_ok = rs_tok == ids
    print(f"tokenizer parity: {'PASS' if tok_ok else 'FAIL'}")
    if not tok_ok:
        print(f"  python: {ids}")
        print(f"  rust:   {rs_tok}")

    py = python_logits(ckpt, ids)
    rs = rust_logits(ckpt, ids)
    # Compare the shared fields. Tolerance accounts for BLAS vs hand-rolled
    # SIMD accumulation order on fp32.
    max_logit = py["max_abs"]
    def close(a: float, b: float) -> bool:
        return abs(a - b) <= 2e-3 * max(1.0, max_logit)
    fields_ok = (
        py["argmax"] == rs["argmax"]
        and all(close(a, b) for a, b in zip(py["head"], rs["head"]))
        and all(close(a, b) for a, b in zip(py["tail"], rs["tail"]))
    )
    print(f"logits parity:   {'PASS' if fields_ok else 'FAIL'}")
    print(f"  argmax  py={py['argmax']} rs={rs['argmax']}")
    print(f"  head    py={py['head']}")
    print(f"  head    rs={rs['head']}")
    print(f"  sum     py={py['sum']} rs={rs['sum']}")

    py_greedy = greedy_python(ckpt)
    rs_ids = greedy_rust(ckpt)
    gen_ok = rs_ids == py_greedy
    print(f"greedy parity:   {'PASS' if gen_ok else 'FAIL'} ({len(py_greedy)} tokens)")
    if not gen_ok:
        print(f"  python: {py_greedy}")
        print(f"  rust:   {rs_ids}")
    else:
        decoded = Engine.from_pretrained(str(ckpt)).tokenizer.decode(rs_ids)
        print(f"  decoded: {decoded!r}")

    ok = tok_ok and fields_ok and gen_ok
    print("\nPARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# benchmark protocol (identical work for both engines)
# ---------------------------------------------------------------------------
def bench_python(ckpt: Path, prefill: int, decode: int, runs: int) -> dict:
    engine = Engine.from_pretrained(str(ckpt))
    cfg = engine.config
    vocab = cfg.vocab_size
    ids = [int((i * 2654435761 + 12345) % (vocab - 8) + 8) for i in range(prefill)]
    params = SamplingParams(max_new_tokens=decode, temperature=0.0)

    engine.generate_ids(ids, SamplingParams(max_new_tokens=2, temperature=0.0))  # warmup

    best_prefill = best_decode = best_e2e = float("inf")
    generated = []
    for _ in range(runs):
        cache = engine.model.new_cache()
        t0 = time.perf_counter()
        logits = engine.model.forward(ids, cache)
        cache.advance(len(ids))
        t1 = time.perf_counter()
        last = logits[-1]
        turn = []
        for step in range(decode):
            token = int(np.argmax(last))
            generated = turn + [token]
            turn.append(token)
            if step == decode - 1:
                break
            logits = engine.model.forward([token], cache)
            cache.advance(1)
            last = logits[-1]
        t2 = time.perf_counter()
        best_prefill = min(best_prefill, prefill / (t1 - t0))
        best_decode = min(best_decode, decode / (t2 - t1))
        best_e2e = min(best_e2e, decode / (t2 - t0))
    return {
        "engine": "nimbus-infer (Python/NumPy)",
        "quant": "fp32",
        "prefill_tok_s": round(best_prefill, 2),
        "decode_tok_s": round(best_decode, 2),
        "e2e_tok_s": round(best_e2e, 2),
        "weight_mb": round(engine.model.n_params() * 4 / 1024 / 1024, 2),
        "kv_cache_mb": round(engine.model.new_cache().bytes / 1024 / 1024, 2),
        "first_tokens": generated[:8],
    }


def bench_rust(ckpt: Path, prefill: int, decode: int, runs: int, quant: str) -> dict:
    out = subprocess.run(
        [str(RS_BIN), "bench", "--model", str(ckpt), "--prefill", str(prefill),
         "--decode", str(decode), "--runs", str(runs), "--quant", quant],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def cmd_bench(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints" / "gpt2"))
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args(argv)
    ckpt = Path(args.model)
    prefill, decode, runs = args.prefill, args.decode, args.runs

    rows = [bench_python(ckpt, prefill, decode, runs)]
    for quant in ("fp32", "int8", "int4"):
        rows.append(bench_rust(ckpt, prefill, decode, runs, quant))
    print()
    print(f"model: {ckpt.name}   prefill {prefill} tokens, decode {decode} tokens, best of {runs} runs")
    print()
    print(f"| {'engine':<26} | {'quant':>5} | {'weights':>9} | {'prefill tok/s':>13} | {'decode tok/s':>12} |")
    print(f"|{'-'*28}|{'-'*7}|{'-'*11}|{'-'*15}|{'-'*14}|")
    for r in rows:
        print(
            f"| {r['engine']:<26} | {r['quant']:>5} | {r['weight_mb']:>7.1f}MB | {r['prefill_tok_s']:>13.1f} "
            f"| {r['decode_tok_s']:>12.1f} |"
        )
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "parity"
    table = {"parity": cmd_parity, "bench": cmd_bench}
    if cmd == "bench":
        sys.exit(table[cmd](sys.argv[2:]))
    sys.exit(table[cmd]())
