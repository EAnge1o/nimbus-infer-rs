#!/usr/bin/env python3
"""Convert a HuggingFace `tokenizer.json` (GPT-2 BPE model) into the
`vocab.json` + `merges.txt` pair the nimbus engines read.

Many modern checkpoints (e.g. SmolLM2) ship only the fast-tokenizer JSON;
this extracts the byte-level BPE tables so both engines can use them.

    python scripts/tokenizer_json_to_gpt2.py tokenizer.json tokenizer_dir/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(src.read_text(encoding="utf-8"))
    model = data["model"]
    if model.get("type") != "BPE":
        print(f"unsupported tokenizer model type: {model.get('type')!r}")
        return 1

    vocab = model["vocab"]           # token string -> id
    merges = model.get("merges", [])  # list of [a, b] pairs, ranked

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "merges.txt").open("w", encoding="utf-8") as fh:
        fh.write("#version: 0.2\n")
        for pair in merges:
            if isinstance(pair, list):
                fh.write(f"{pair[0]} {pair[1]}\n")
            else:
                fh.write(f"{pair}\n")  # some versions store "a b" strings
    print(f"wrote {out_dir / 'vocab.json'} ({len(vocab)} tokens) "
          f"and {out_dir / 'merges.txt'} ({len(merges)} merges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
