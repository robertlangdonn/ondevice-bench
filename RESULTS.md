# Results — local LLMs on a 16 GB MacBook (Apple M3, base)

**Setup:** MacBook M3 (base), 16 GB unified memory · MLX (mlx-lm 0.31.3) · 4-bit models ·
256 max tokens · per-model warmup · each model in a fresh subprocess · run under `caffeinate`
· 300 s per-model timeout.

**Caveat:** measured with the system under memory pressure (~7 GB swap in use from concurrent
apps), so throughput here is *conservative* — a quiet machine runs ~10–30% faster. The relative
shape (and the 9B wall) is stable across runs.

| Model | Gen t/s | Prompt t/s | Peak RAM (GB) | Load (s) |
|---|--:|--:|--:|--:|
| Llama-3.2-1B-Instruct-4bit | 38.7 | 535.8 | 0.81 | 2.4 |
| Phi-3.5-mini-instruct-4bit (3.8B) | 10.6 | 71.4 | 2.47 | 1.8 |
| Qwen3-4B-4bit | 10.8 | 83.5 | 2.42 | 2.1 |
| Qwen3.5-4B-4bit | 9.8 | 63.6 | 2.56 | 4.0 |
| Falcon3-7B-Instruct-4bit | 5.7 | 58.7 | 4.33 | 3.2 |
| Meta-Llama-3.1-8B-Instruct-4bit | 5.1 | 41.9 | 4.68 | 3.5 |
| Qwen3.5-9B-MLX-4bit | DNF | — | — | exceeded 300 s — swap thrashing |

## Takeaways
- **1B flies** (~40 t/s), **4B-class is comfortable** (~10–13 t/s), **7–8B is the practical edge** (~5–7 t/s).
- **9B is the wall** on 16 GB: model + OS + browser pushes past real RAM, swap kicks in, generation crawls. With apps closed it may *just* fit — but it's the ceiling.
- **Peak RAM ≈ model size at 4-bit** + headroom; an 8B sits ~4.7 GB, leaving little slack on 16 GB once the OS and a browser are accounted for.

## Methodology notes (things that bit us)
1. **Warmup** — the first generation pays a one-time Metal kernel-compile cost; warm up before timing.
2. **`caffeinate`** — if the Mac sleeps mid-run, wall-clock timing absorbs the nap (a "460 s load" was the machine asleep).
3. **Fresh subprocess per model** — MLX doesn't fully release memory in-process; a single-process loop lets pressure accumulate and depresses later models.
