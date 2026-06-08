"""Clean per-model peak RAM UNDER A ~2048-TOKEN LOAD — one model per process
(fresh process peak, no cumulative contamination). Unlike measure_ram.py (40-tok
'ocean' gen ≈ load RAM only), this fills the KV cache to ~2048 context so the
reported peak matches the conditions the capability quality/speed numbers were
taken under. Long prompt → near-max KV during prefill, then a short generation,
so it's fast (~30s) instead of generating 2048 tokens one at a time.

Usage: measure_ram_loaded.py <model_id>   →  appends to results/clean_ram_loaded.json
"""
import sys, json, os
import mlx.core as mx
from mlx_lm import load, generate

model_id = sys.argv[1]
TARGET_PROMPT_TOKENS = 1850   # + ~200 generated ≈ 2048 context

model, tok = load(model_id)

# Build a ~1850-token prompt: a long passage + a recall question, so prefill
# allocates KV for the full context (that's where peak RAM lives).
passage = (
    "The unified memory architecture on Apple Silicon shares a single pool between "
    "CPU and GPU, so every byte the model weights occupy is a byte unavailable to "
    "the KV cache, the activations, and the operating system. On a 16 GB machine "
    "this makes the memory budget the binding constraint long before compute does. "
)
chunk = tok.encode(passage)
reps = max(1, TARGET_PROMPT_TOKENS // len(chunk))
long_text = passage * reps
question = ("\n\nQuestion: in one sentence, what is the binding constraint on a "
            "16 GB Apple Silicon machine, and why?")
content = long_text + question

p = tok.apply_chat_template([{"role": "user", "content": content}],
                            add_generation_prompt=True, tokenize=False)
n_prompt = len(tok.encode(p))

generate(model, tok, prompt=p, max_tokens=200, verbose=False)

peak = mx.get_peak_memory() / 1e9
print(f"PEAK_LOADED {model_id} prompt_tok={n_prompt} peak_gb={peak:.2f}")

rec = {}
path = "results/clean_ram_loaded.json"
if os.path.exists(path):
    rec = json.load(open(path))
rec[model_id] = {"peak_gb": round(peak, 2), "prompt_tok": n_prompt}
json.dump(rec, open(path, "w"), indent=2)
