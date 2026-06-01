# on-device LLM benchmark — Apple Silicon

Honest numbers for running LLMs **locally on a Mac**, measured on the machine itself.
Most published benchmarks run on datacenter GPUs; this one targets the laptop in front of you.

![Local LLM speed on a 16 GB MacBook Air (M3): 1B ~39 t/s, 4B-class ~10 t/s, 7–8B ~5 t/s, 9B did not finish](results/chart.png)

Full write-up: **[What actually runs well on a 16 GB MacBook](https://prasadkhake.com/blog/16gb-mac-llm)**.

Measured per model (MLX, 4-bit unless noted):

| Metric | What it means |
|---|---|
| **Load (s)** | time to load weights into memory |
| **Prompt t/s** | prefill speed — processing your input ("first token is slow") |
| **Gen t/s** | decode speed — how fast it writes the answer |
| **Peak RAM (GB)** | high-water memory — what actually fits in 16 GB |

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bench.py                      # default 6-model set
# or pick your own:
python bench.py --models mlx-community/Qwen3-4B-4bit mlx-community/Qwen3-8B-4bit
python bench.py --max-tokens 256
python bench.py --card                # render a shareable scorecard PNG → results/card.png
```

First run downloads each model (cached afterward in `~/.cache/huggingface`). Results stream to
`results/results.json` and a Markdown table prints at the end.

## Hardware

Record yours here so the numbers mean something:

- **Machine:** MacBook Air 15-inch (M3, 8-core), 16 GB unified memory, macOS 26.5
- **mlx-lm:** (printed by `pip show mlx-lm`)

## Roadmap (next data points)

- **4-bit vs 8-bit** for one model — the speed/memory/quality tradeoff.
- **Max context before OOM** — push prompt length until it fails; record the ceiling per model.
- **Quality spot-check** — a couple of fixed prompts, eyeballed, so "fast" isn't the only axis.
- **Quantization deep-dive** — group size, what degrades first.

## Why this exists

Part of an "On Device" series — getting LLMs to run well on real, consumer hardware.
Writeups: [prasadkhake.com](https://prasadkhake.com).
