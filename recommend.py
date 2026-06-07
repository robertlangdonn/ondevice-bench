#!/usr/bin/env python3
"""
recommend.py — model selection CLI for on-device LLMs on Apple Silicon.

Recommends the best model + flags based on your RAM, task, and priority,
using real benchmark data measured on M3 16 GB MacBook Air.

Usage:
    python3 recommend.py --ram 16 --task coding
    python3 recommend.py --ram 8 --task general --priority speed
    python3 recommend.py --list
    python3 recommend.py --compare
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark data — real measurements on M3 16 GB MacBook Air
# Updated: 2026-06-03
# ---------------------------------------------------------------------------

MODELS = [
    {
        "id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "name": "Llama-3.1-8B",
        "family": "llama",
        "params_b": 8,
        "quant": "4bit",
        "ram_gb": 4.94,
        "gen_tps": 7.7,
        "quality_pct": 100,   # 21/21
        "coding_pct": 100,    # 8/8
        "math_pct": 100,      # 8/8
        "factual_pct": 100,   # 5/5
        "flags": [],
        "notes": "Solid all-rounder. Highest confidence for general tasks.",
        "min_ram_gb": 6.5,    # model + OS headroom
    },
    {
        "id": "mlx-community/Qwen3-4B-4bit",
        "name": "Qwen3-4B",
        "family": "qwen3",
        "params_b": 4,
        "quant": "4bit",
        "ram_gb": 2.74,
        "gen_tps": 12.0,
        "quality_pct": 100,   # 21/21 with thinking off
        "coding_pct": 100,
        "math_pct": 100,
        "factual_pct": 100,
        "flags": ["--thinking-off"],   # enable_thinking=False in chat template
        "notes": "Best RAM/quality tradeoff. REQUIRES thinking mode off (--thinking-off flag). "
                 "With thinking on: 1/8 coding. With thinking off: 8/8 coding, 100% overall.",
        "min_ram_gb": 4.0,
        "thinking_warning": True,
    },
    {
        "id": "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
        "name": "LFM2.5-8B-A1B",
        "family": "lfm2",
        "params_b": 8,
        "quant": "4bit",
        "ram_gb": 4.85,
        "gen_tps": 18.2,
        "quality_pct": 81,    # 17/21
        "coding_pct": 63,     # 5/8
        "math_pct": 100,      # 8/8
        "factual_pct": 80,    # 4/5
        "flags": [],
        "notes": "Fastest by far (MoE A1B: only 1B active params/token). "
                 "Excellent for math and factual. Coding gaps: off-by-one errors, "
                 "thinking-mode overflow on complex functions. Avoid for critical code.",
        "min_ram_gb": 6.5,
    },
    {
        "id": "mlx-community/gemma-3-12b-it-qat-3bit",
        "name": "Gemma-3-12B-QAT",
        "family": "gemma3",
        "params_b": 12,
        "quant": "3bit-qat",
        "ram_gb": 5.90,
        "gen_tps": 5.4,
        "quality_pct": 90,    # 19/21
        "coding_pct": 100,    # 8/8
        "math_pct": 75,       # 6/8 — fails prime + compound interest
        "factual_pct": 100,   # 5/5
        "flags": [],
        "notes": "Best coding quality. Slowest. Specific math weaknesses: "
                 "incorrectly identifies 97 as non-prime; wrong compound interest. "
                 "QAT 3-bit keeps quality close to higher precision.",
        "min_ram_gb": 7.5,
    },
]

# Placeholder — will be populated after benchmarking
GEMMA4_PLACEHOLDER = {
    "id": "mlx-community/gemma-4-12B-it-4bit",
    "name": "Gemma-4-12B",
    "family": "gemma4",
    "params_b": 12,
    "quant": "4bit",
    "ram_gb": None,        # TBD
    "gen_tps": None,       # TBD
    "quality_pct": None,
    "coding_pct": None,
    "math_pct": None,
    "factual_pct": None,
    "flags": [],
    "notes": "NEW (2026-06-03). Encoder-free multimodal, MTP drafters, Apache 2.0. "
             "Benchmarking in progress.",
    "min_ram_gb": 8.0,
}


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

TASK_WEIGHTS = {
    "coding":   {"coding_pct": 0.7, "math_pct": 0.1, "factual_pct": 0.2},
    "math":     {"coding_pct": 0.1, "math_pct": 0.7, "factual_pct": 0.2},
    "writing":  {"coding_pct": 0.1, "math_pct": 0.1, "factual_pct": 0.8},
    "general":  {"coding_pct": 0.33, "math_pct": 0.33, "factual_pct": 0.34},
    "chat":     {"coding_pct": 0.2, "math_pct": 0.1, "factual_pct": 0.7},
    "agents":   {"coding_pct": 0.5, "math_pct": 0.3, "factual_pct": 0.2},
}


def weighted_quality(model: dict, task: str) -> float:
    w = TASK_WEIGHTS.get(task, TASK_WEIGHTS["general"])
    q = 0.0
    for key, weight in w.items():
        val = model.get(key)
        if val is None:
            return -1.0   # skip models without data
        q += val * weight
    return q


def score_model(model: dict, ram_gb: float, task: str, priority: str) -> float:
    if model.get("ram_gb") is None:
        return -1.0
    if model["min_ram_gb"] > ram_gb:
        return -1.0   # doesn't fit

    quality = weighted_quality(model, task)
    speed = model.get("gen_tps") or 0
    ram_efficiency = 1.0 - (model["ram_gb"] / ram_gb)  # lower RAM = higher score

    if priority == "quality":
        return quality * 0.7 + speed * 0.1 + ram_efficiency * 0.2
    elif priority == "speed":
        return speed * 0.6 + quality * 0.3 + ram_efficiency * 0.1
    elif priority == "ram":
        return ram_efficiency * 0.5 + quality * 0.4 + speed * 0.1
    else:  # balanced
        return quality * 0.5 + speed * 0.3 + ram_efficiency * 0.2


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def bar(val, max_val, width=20, char="█"):
    if val is None:
        return "?" * width
    filled = int(width * val / max_val)
    return char * filled + "░" * (width - filled)


def print_recommendation(model: dict, rank: int, task: str):
    tps = f"{model['gen_tps']:.1f} tok/s" if model.get('gen_tps') else "?"
    ram = f"{model['ram_gb']:.2f} GB" if model.get('ram_gb') else "?"
    q = f"{model['quality_pct']}%" if model.get('quality_pct') is not None else "?"
    flags = " ".join(model["flags"]) if model["flags"] else "(no extra flags)"

    prefix = "★ " if rank == 1 else f"{rank}. "
    print(f"\n{prefix}{model['name']}  [{model['quant']}]")
    print(f"   {model['id']}")
    print(f"   Speed: {tps}  │  RAM: {ram}  │  Quality: {q}")
    print(f"   Flags: {flags}")
    if model.get("thinking_warning"):
        print(f"   ⚠  IMPORTANT: run with thinking mode OFF or quality drops to ~12%")
    print(f"   {model['notes']}")


def print_comparison_table(models: list, task: str):
    available = [m for m in models if m.get('ram_gb') and m.get('quality_pct') is not None]
    if not available:
        print("No benchmark data available yet.")
        return

    max_tps = max(m['gen_tps'] for m in available if m.get('gen_tps'))

    print(f"\n{'Model':<22} {'RAM':>7} {'tok/s':>7} {'Quality':>8} {'Coding':>8} {'Math':>6}")
    print(f"{'─'*22} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*6}")
    for m in sorted(available, key=lambda x: -weighted_quality(x, task)):
        tps = f"{m['gen_tps']:.1f}" if m.get('gen_tps') else "?"
        q = f"{m['quality_pct']}%" if m.get('quality_pct') is not None else "?"
        c = f"{m['coding_pct']}%" if m.get('coding_pct') is not None else "?"
        ma = f"{m['math_pct']}%" if m.get('math_pct') is not None else "?"
        note = " ← thinking off!" if m.get('thinking_warning') else ""
        print(f"  {m['name']:<20} {m['ram_gb']:>6.2f}G {tps:>7} {q:>8} {c:>8} {ma:>6}{note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Model recommendation for on-device LLMs — based on real M3 16 GB benchmarks"
    )
    ap.add_argument("--ram", type=float, default=16.0,
                    help="Available RAM in GB (default: 16)")
    ap.add_argument("--task", choices=list(TASK_WEIGHTS), default="general",
                    help="Primary use case (default: general)")
    ap.add_argument("--priority", choices=["balanced", "quality", "speed", "ram"],
                    default="balanced", help="What to optimize for (default: balanced)")
    ap.add_argument("--list", action="store_true", help="List all benchmarked models")
    ap.add_argument("--compare", action="store_true", help="Show full comparison table")
    ap.add_argument("--top", type=int, default=3, help="How many recommendations (default: 3)")
    args = ap.parse_args()

    all_models = MODELS[:]  # excludes placeholder until benchmarked

    if args.list:
        print(f"\nBenchmarked models (M3 16 GB MacBook Air, as of 2026-06-03):\n")
        for m in all_models:
            tps = f"{m['gen_tps']:.1f} tok/s" if m.get('gen_tps') else "?"
            q = f"{m['quality_pct']}%" if m.get('quality_pct') is not None else "?"
            print(f"  {m['name']:<24} {m['ram_gb']:.2f} GB  {tps:>8}  quality {q}")
        print(f"\n  (Gemma-4-12B: benchmarking in progress)")
        return

    if args.compare:
        print(f"\nFull comparison — task: {args.task}")
        print_comparison_table(all_models, args.task)
        return

    # Filter and score
    candidates = []
    for m in all_models:
        s = score_model(m, args.ram, args.task, args.priority)
        if s >= 0:
            candidates.append((s, m))

    candidates.sort(reverse=True)

    print(f"\n{'='*60}")
    print(f"  Recommendations for {args.ram:.0f} GB RAM  ·  task: {args.task}  ·  priority: {args.priority}")
    print(f"{'='*60}")

    if not candidates:
        print(f"\n  No models fit in {args.ram:.0f} GB with current data.")
        print(f"  Try --ram with a higher value or check --list for model requirements.")
        return

    for rank, (score, model) in enumerate(candidates[:args.top], 1):
        print_recommendation(model, rank, args.task)

    print(f"\n{'─'*60}")
    print(f"  Benchmarks: github.com/robertlangdonn/ondevice-bench")
    print(f"  Hardware: M3 MacBook Air 15\", 16 GB, macOS 26.x")
    print(f"  Quality: 21-task suite (coding/math/factual), execution-verified")


if __name__ == "__main__":
    main()
