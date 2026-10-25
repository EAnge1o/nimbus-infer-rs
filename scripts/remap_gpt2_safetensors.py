#!/usr/bin/env python3
"""Remap the gpt2 repo's newer safetensors key layout onto the classic
`transformer.*` names the nimbus-infer converter expects, and drop the
per-layer causal-mask buffers (`h.*.attn.bias`, `attn.masked_bias`) that are
torch buffers, not weights.

    python scripts/remap_gpt2_safetensors.py in.safetensors out.safetensors
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nimbus-infer"))

import numpy as np  # noqa: E402

from nimbus_infer.weights import read_safetensors, save_safetensors  # noqa: E402


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    tensors = read_safetensors(str(src))
    out = {}
    dropped = []
    for name, value in tensors.items():
        # Drop the causal-mask buffers, but not the real c_attn.bias weights
        # ("h.0.attn.bias" vs "h.0.attn.c_attn.bias").
        if name.endswith("masked_bias") or (
            name.endswith("attn.bias") and not name.endswith("c_attn.bias")
        ):
            dropped.append(name)
            continue
        new = re.sub(r"^(h\.|wte|wpe|ln_f)", r"transformer.\1", name)
        out[new] = np.asarray(value, dtype=np.float32)
    save_safetensors(str(dst), out)
    print(f"remapped {len(out)} tensors, dropped {len(dropped)} mask buffers -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
