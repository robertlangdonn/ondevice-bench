#!/usr/bin/env python3
"""On-device LLM benchmark for Apple Silicon (MLX).

Design notes (learned the hard way on a 16 GB M3):
  - Each model runs in a FRESH subprocess. MLX does not fully release memory
    in-process between models, so a single-process loop lets pressure accumulate
    and silently depresses the later (bigger) models. Isolation fixes that.
  - Per-model TIMEOUT: on a memory-constrained Mac a too-big model thrashes swap
    and crawls. The timeout turns that into a graceful "DNF" row instead of a hang.
  - WARMUP: a few throwaway tokens first, so the timed run is steady-state, not
    first-run Metal-kernel-compile cost.
  - Run with the machine awake (use `caffeinate -i python bench.py`) — wall-clock
    timing otherwise absorbs sleep time.

Usage:
    caffeinate -i python bench.py                 # default set
    python bench.py --timeout 300                 # per-model budget (s)
    python bench.py --models <repo> ...
    python bench.py --single <repo>               # internal: one model -> JSON
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

DEFAULT_MODELS = [
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3.5-4B-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Falcon3-7B-Instruct-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
]

PROMPT = (
    "Explain, in about 200 words, how a transformer language model generates "
    "text one token at a time, and why the first token usually takes longer "
    "than the rest."
)
RESULT_MARKER = "__RESULT__"


def run_single(repo: str, max_tokens: int):
    """Bench one model in this (fresh) process; print one JSON line."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    def reset_peak():
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

    try:
        reset_peak()
        t0 = time.time()
        model, tok = load(repo)
        load_s = time.time() - t0

        msgs = [{"role": "user", "content": PROMPT}]
        try:
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            prompt = PROMPT

        # warmup (untimed) — trigger kernel compilation
        for _ in stream_generate(model, tok, prompt, max_tokens=4):
            pass

        last = None
        for resp in stream_generate(model, tok, prompt, max_tokens=max_tokens):
            last = resp

        r = {
            "model": repo,
            "load_s": round(load_s, 1),
            "prompt_tokens": getattr(last, "prompt_tokens", None),
            "prompt_tps": round(getattr(last, "prompt_tps", 0) or 0, 1),
            "gen_tokens": getattr(last, "generation_tokens", None),
            "gen_tps": round(getattr(last, "generation_tps", 0) or 0, 1),
            "peak_gb": round(getattr(last, "peak_memory", 0) or 0, 2),
        }
    except Exception as e:
        r = {"model": repo, "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
    print(RESULT_MARKER + json.dumps(r))


def orchestrate(models, max_tokens, timeout, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo in models:
        print(f"\n=== {repo} ===", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--single", repo, "--max-tokens", str(max_tokens)],
                capture_output=True, text=True, timeout=timeout,
            )
            line = next((l for l in p.stdout.splitlines() if l.startswith(RESULT_MARKER)), None)
            if line:
                r = json.loads(line[len(RESULT_MARKER):])
            else:
                tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
                r = {"model": repo, "error": f"no result (exit {p.returncode}): {tail}"}
        except subprocess.TimeoutExpired:
            r = {"model": repo, "error": f"DNF — exceeded {timeout}s (memory/swap pressure on this machine)"}
        print(r, flush=True)
        rows.append(r)
        json.dump(rows, open(out, "w"), indent=2)

    print("\n\n## Results — Apple Silicon, MLX, 4-bit (each model in a fresh process)\n")
    print("| Model | Load (s) | Prompt t/s | Gen t/s | Peak RAM (GB) |")
    print("|---|--:|--:|--:|--:|")
    for r in rows:
        name = r["model"].split("/")[-1]
        if "error" in r:
            print(f"| {name} | — | — | — | _{r['error']}_ |")
        else:
            print(f"| {name} | {r['load_s']} | {r['prompt_tps']} | {r['gen_tps']} | {r['peak_gb']} |")
    print(f"\nRaw: {out}")


def make_card(results_path, out):
    """Render a shareable scorecard PNG from a results.json — `--card`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(results_path))
    rows = [(r["model"].split("/")[-1], None if "error" in r else r.get("gen_tps")) for r in d][::-1]
    names = [n for n, _ in rows]
    vals = [v or 0 for _, v in rows]
    labels = [f"{v} t/s" if v else "DNF" for _, v in rows]
    PAPER, INK, RED, MUTED, BAR = "#f2ede4", "#1a1814", "#c0392b", "#7a7468", "#d8d0c2"

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    bars = ax.barh(names, vals, height=0.62, color=[RED if v else BAR for v in vals])
    top = max(vals) or 1
    for i, b in enumerate(bars):
        ax.text(b.get_width() + top * 0.02, b.get_y() + b.get_height() / 2, labels[i],
                va="center", fontsize=11, color=(INK if vals[i] else RED))
    ax.set_title("Local LLM speed — generation tokens/sec (4-bit, MLX)",
                 loc="left", color=INK, fontsize=15, pad=14)
    ax.set_xticks([])
    ax.tick_params(colors=INK, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.18)
    fig.text(0.99, 0.02, "prasadkhake.com", ha="right", color=MUTED, fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, facecolor=PAPER, bbox_inches="tight", dpi=140)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", help="internal: bench one model and print JSON")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=300, help="per-model wall-clock budget (s)")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--card", action="store_true", help="render a shareable scorecard from --out")
    ap.add_argument("--card-out", default="results/card.png")
    args = ap.parse_args()

    if args.single:
        run_single(args.single, args.max_tokens)
    elif args.card:
        make_card(args.out, args.card_out)
    else:
        orchestrate(args.models, args.max_tokens, args.timeout, args.out)


if __name__ == "__main__":
    main()
