#!/usr/bin/env python3
"""Quality spot-check: same prompts through each model (fresh subprocess each).
Does a bigger-but-lower-bit model actually answer better, or did quantization wreck it?
"""
import argparse, json, subprocess, sys
from pathlib import Path

MODELS = [
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/gemma-3-12b-it-qat-3bit",
    "mlx-community/Qwen3-14B-3bit",
]
PROMPTS = [
    ("reasoning (answer: $0.05)",
     "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost? Give the number and one short sentence of reasoning."),
    ("code",
     "Write a Python function fib(n) that returns the nth Fibonacci number iteratively. Code only, no explanation."),
]


def run_single(model):
    from mlx_lm import load, stream_generate
    m, t = load(model)
    out = {}
    for key, p in PROMPTS:
        prompt = t.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        txt = ""
        for r in stream_generate(m, t, prompt, max_tokens=768):  # room for reasoning models to finish
            txt += r.text
        out[key] = txt.strip()
    print("__Q__" + json.dumps({"model": model, "out": out}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single")
    a = ap.parse_args()
    if a.single:
        run_single(a.single)
        return
    Path("results").mkdir(exist_ok=True)
    rows = []
    for m in MODELS:
        print("===", m, "===", flush=True)
        try:
            p = subprocess.run([sys.executable, __file__, "--single", m], capture_output=True, text=True, timeout=400)
            line = next((l for l in p.stdout.splitlines() if l.startswith("__Q__")), None)
            rows.append(json.loads(line[5:]) if line else {"model": m, "error": "no output"})
        except subprocess.TimeoutExpired:
            rows.append({"model": m, "error": "DNF"})
        json.dump(rows, open("results/quality.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
