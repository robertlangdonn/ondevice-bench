#!/usr/bin/env python3
"""
kv_memory.py — measure actual KV cache RAM growth vs context length on Apple Silicon.

Runs each measurement in a fresh subprocess (clean memory state).
Compares full KVCache vs RotatingKVCache (--max-kv-size) vs QuantizedKVCache (--kv-bits).

Usage:
    python3 kv_memory.py --model mlx-community/Llama-3.1-8B-Instruct-4bit
    python3 kv_memory.py --model mlx-community/Llama-3.1-8B-Instruct-4bit --max-kv-size 4096
    python3 kv_memory.py --model mlx-community/Llama-3.1-8B-Instruct-4bit --kv-bits 4
"""

import argparse
import json
import subprocess
import sys


_MEASURE = """
import json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache

model_path = sys.argv[1]
prompt_len = int(sys.argv[2])
max_kv_size = int(sys.argv[3]) if sys.argv[3] != "None" else None
kv_bits     = int(sys.argv[4]) if sys.argv[4] != "None" else None

mx.reset_peak_memory()
model, tokenizer = load(model_path)
mx.eval(model.parameters())
load_mem_gb = mx.get_peak_memory() / 1e9

# Build cache
cache = make_prompt_cache(model, max_kv_size=max_kv_size)
if kv_bits is not None:
    cache = [
        c.to_quantized(bits=kv_bits) if isinstance(c, KVCache) else c
        for c in cache
    ]

# Feed a synthetic prompt of the given length
tokens = mx.zeros((1, prompt_len), dtype=mx.int32)
mx.reset_peak_memory()
logits = model(tokens, cache=cache)
mx.eval(logits)

cache_gb = sum(c.nbytes for c in cache) / 1e9
peak_gb  = mx.get_peak_memory() / 1e9

# Try to extract KV shape from the first attention layer
n_kv = head_dim = None
for layer in model.model.layers:
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
    if attn is None:
        continue
    n_kv    = getattr(attn, "n_kv_heads", None) or getattr(attn, "num_key_value_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if n_kv and head_dim:
        break

print(json.dumps({
    "prompt_len":   prompt_len,
    "cache_gb":     round(cache_gb, 4),
    "peak_gb":      round(peak_gb, 4),
    "load_mem_gb":  round(load_mem_gb, 4),
    "n_kv_heads":   n_kv,
    "head_dim":     head_dim,
    "cache_type":   type(cache[0]).__name__,
    "num_layers":   len(cache),
}))
"""


def measure(model_path, prompt_len, max_kv_size, kv_bits, python):
    result = subprocess.run(
        [python, "-c", _MEASURE,
         model_path, str(prompt_len), str(max_kv_size), str(kv_bits)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  [error len={prompt_len}] {result.stderr[-300:]}", file=sys.stderr)
        return None
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return None


def theory_gb(n_layers, n_kv, head_dim, seq_len, bits=16):
    return n_layers * 2 * n_kv * head_dim * seq_len * (bits / 8) / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--kv-bits", type=int, default=None,
                        help="Quantize KV cache to N bits (4 or 8)")
    parser.add_argument("--lengths", default="256,512,1024,2048,4096,8192",
                        help="Comma-separated prompt lengths")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    label = args.model.split("/")[-1]
    if args.max_kv_size:
        label += f"  [RotatingKV max={args.max_kv_size}]"
    if args.kv_bits:
        label += f"  [KV {args.kv_bits}-bit]"

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"{'Tokens':>8}  {'Cache GB':>9}  {'Peak GB':>9}  {'Theory GB':>10}  Cache type")
    print(f"{'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}  ----------")

    n_layers = n_kv = head_dim = None

    for length in lengths:
        r = measure(args.model, length, args.max_kv_size, args.kv_bits, args.python)
        if r is None:
            print(f"{length:>8}  {'OOM/error':>32}")
            continue

        if n_layers is None:
            n_layers = r.get("num_layers")
            n_kv     = r.get("n_kv_heads")
            head_dim = r.get("head_dim")

        theory = ""
        if n_layers and n_kv and head_dim:
            eff = min(length, args.max_kv_size) if args.max_kv_size else length
            bits = args.kv_bits or 16
            theory = f"{theory_gb(n_layers, n_kv, head_dim, eff, bits):.4f}"

        print(f"{length:>8}  {r['cache_gb']:>9.4f}  {r['peak_gb']:>9.4f}  {theory:>10}  {r['cache_type']}")

    if n_layers and n_kv and head_dim:
        kb_per_tok = theory_gb(n_layers, n_kv, head_dim, 1) * 1e6
        tokens_to_oom = int(16.0 / theory_gb(n_layers, n_kv, head_dim, 1))
        print(f"\n  Layers {n_layers} × 2 × {n_kv} KV heads × {head_dim} head_dim")
        print(f"  {kb_per_tok:.0f} KB per token  |  {tokens_to_oom:,} tokens fills 16 GB (bf16)")
        if args.kv_bits:
            kb_q = theory_gb(n_layers, n_kv, head_dim, 1, args.kv_bits) * 1e6
            tokens_q = int(16.0 / theory_gb(n_layers, n_kv, head_dim, 1, args.kv_bits))
            print(f"  {kb_q:.0f} KB per token at {args.kv_bits}-bit  |  {tokens_q:,} tokens fills 16 GB")


if __name__ == "__main__":
    main()
