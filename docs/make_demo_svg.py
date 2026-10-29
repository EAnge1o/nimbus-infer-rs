#!/usr/bin/env python3
"""Render the README terminal screenshot from real command output.

Same principle as nimbus-infer's make_demo_gif.py: the image in the README
is rendered from actual captured output of the commands shown. Re-capture by
rerunning the generate commands and rerunning this script:

    python docs/make_demo_svg.py
"""
from __future__ import annotations

from pathlib import Path

# --- captured from real runs (gpt2-124M, i7-12700H, greedy decoding) ------
FLOW = [
    # (kind, text) kind: cmd | out | dim | ok | bad
    ("dim",  "# same checkpoint, three precision modes -- the fp32 text is the"),
    ("dim",  "# reference; int8 tracks it; int4 drifts but stays coherent"),
    ("cmd",  'nimbus-rs generate --model checkpoints/gpt2 --prompt "The quick brown fox" \\'),
    ("cont", "    --max-new-tokens 24 --temperature 0"),
    ("out",  "es are a great way to get a little bit of a kick out of your dog."),
    ("out",  "The quick brown fox"),
    ("dim",  "[24 tokens in 1.39s = 17.3 tok/s]"),
    ("cmd",  "nimbus-rs generate --model checkpoints/gpt2 --prompt \"The quick brown fox\" --quant int8"),
    ("out",  "es were the most common species of foxes in the area, and they were found in the area's forests and in"),
    ("dim",  "[24 tokens in 0.94s = 25.5 tok/s]"),
    ("cmd",  "nimbus-rs generate --model checkpoints/gpt2 --prompt \"The quick brown fox\" --quant int4"),
    ("out",  "'s favorite favorite. I swear. I swear. I swear. I swear. I swear. I swear"),
    ("dim",  "[24 tokens in 0.91s = 26.2 tok/s]"),
    ("cmd",  "nimbus-rs inspect --model checkpoints/gpt2 --quant int4"),
    ("ok",   '{"n_params": 124439808, "weight_mb": 62.59, "quant": "int4", "vocab": 50257, ...}'),
]

COLORS = {
    "cmd": "#79c0ff",
    "cont": "#79c0ff",
    "out": "#c9d1d9",
    "dim": "#8b949e",
    "ok": "#7ee787",
    "bad": "#ff7b72",
}

FONT = "ui-monospace,'Cascadia Code','JetBrains Mono',Menlo,Consolas,monospace"
FS = 13.0
LH = 21.0
PAD_X = 20.0
CHAR_W = 7.9


def esc(text: str) -> str:
    # XML whitespace collapsing would destroy alignment; NBSP has the same
    # advance width in monospace.
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "\u00a0")
    )


def render(lines, title: str) -> str:
    width = max(int(max(len(t) for _, t in lines) * CHAR_W) + 2 * PAD_X + 8, 560)
    header_h = 38.0
    height = header_h + PAD_X + len(lines) * LH + 14

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{int(height)}" '
        f'viewBox="0 0 {width} {int(height)}" role="img" aria-label="{esc(title)}">',
        "<!-- rendered from real command output by docs/make_demo_svg.py -->",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" '
        f'fill="#0d1117" stroke="#30363d"/>',
        f'<path d="M0.5 10.5 a10 10 0 0 1 10 -10 h{width - 21} a10 10 0 0 1 10 10 v28 h-{width - 1} z" '
        f'fill="#161b22" stroke="none"/>',
        f'<line x1="0.5" y1="{header_h}" x2="{width - 0.5}" y2="{header_h}" stroke="#30363d"/>',
    ]
    for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        parts.append(f'<circle cx="{24 + i * 20}" cy="19" r="6" fill="{color}"/>')
    parts.append(
        f'<text x="{width / 2}" y="23.5" text-anchor="middle" font-family="{FONT}" '
        f'font-size="12" fill="#8b949e">{esc(title)}</text>'
    )
    y = header_h + PAD_X + 2
    for kind, text in lines:
        if kind == "cmd":
            parts.append(
                f'<text x="{PAD_X}" y="{y}" font-family="{FONT}" font-size="{FS}">'
                f'<tspan fill="#3fb950" font-weight="bold">$</tspan>'
                f'<tspan fill="{COLORS[kind]}" dx="8">{esc(text)}</tspan></text>'
            )
        else:
            parts.append(
                f'<text x="{PAD_X}" y="{y}" font-family="{FONT}" font-size="{FS}" '
                f'fill="{COLORS[kind]}">{esc(text)}</text>'
            )
        y += LH
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    out = Path(__file__).resolve().parent
    (out / "demo.svg").write_text(
        render(FLOW, "nimbus-infer-rs — gpt2-124M, fp32 / int8 / int4 (real output)"),
        encoding="utf-8",
    )
    print(f"wrote {out / 'demo.svg'}")


if __name__ == "__main__":
    main()
