#!/usr/bin/env python3
"""
21-task verifiable quality benchmark for on-device LLMs.
Tasks: 8 coding (execute output), 8 math (parse answer), 5 factual (keyword check).

Usage:
    python quality.py --model mlx-community/Qwen3-4B-4bit
    python quality.py --model mlx-community/gemma-4-12B-it-4bit [--max-tokens 512]
"""

import argparse
import subprocess
import sys
import re
import time
import textwrap
import tempfile
import os

MLX_LM_PATH = os.path.join(os.path.dirname(__file__), "..", "mlx-lm")
sys.path.insert(0, MLX_LM_PATH)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

CODING_TASKS = [
    {
        "id": "fizzbuzz",
        "prompt": "Write a Python function fizzbuzz(n) that returns a list of strings from 1 to n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number as a string. Print fizzbuzz(15).",
        "check": lambda out: _fizzbuzz_ok(out),
    },
    {
        "id": "fibonacci",
        "prompt": "Write a Python function fib(n) that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1). Print fib(10).",
        "check": lambda out: "55" in out,
    },
    {
        "id": "palindrome",
        "prompt": "Write a Python function is_palindrome(s) that returns True if s is a palindrome (ignoring case and non-alphanumeric chars), False otherwise. Print is_palindrome('A man a plan a canal Panama') and is_palindrome('hello').",
        "check": lambda out: "True" in out and "False" in out,
    },
    {
        "id": "flatten",
        "prompt": "Write a Python function flatten(lst) that recursively flattens a nested list. Print flatten([1, [2, [3, 4], 5], [6, 7]]).",
        "check": lambda out: bool("1" in out and "7" in out and re.search(r"\[1,\s*2,\s*3,\s*4,\s*5,\s*6,\s*7\]", out)),
    },
    {
        "id": "reverse_string",
        "prompt": "Write a Python one-liner that reverses the string 'OpenSource' and prints it.",
        "check": lambda out: "ecruoSnepO" in out,
    },
    {
        "id": "count_vowels",
        "prompt": "Write a Python function count_vowels(s) that counts vowels (a,e,i,o,u, case-insensitive). Print count_vowels('Hello World').",
        "check": lambda out: "3" in out,
    },
    {
        "id": "binary_search",
        "prompt": "Write a Python function binary_search(arr, target) that returns the index of target in sorted arr, or -1 if not found. Print binary_search([1,3,5,7,9,11,13], 7).",
        "check": lambda out: "3" in out,
    },
    {
        "id": "word_count",
        "prompt": "Write a Python function word_count(text) that returns a dict of word frequencies (lowercase). Print word_count('the cat sat on the mat the cat')['the'].",
        "check": lambda out: "3" in out,
    },
]

MATH_TASKS = [
    {
        "id": "compound_interest",
        "prompt": "What is the total amount (principal + interest) when $1000 is invested at 10% annual compound interest for 5 years? Give the answer rounded to 2 decimal places.",
        "answer": "1610.51",
        "tolerance": 0.02,
    },
    {
        "id": "prime_97",
        "prompt": "Is 97 a prime number? Answer only Yes or No.",
        "answer": "yes",
        "tolerance": None,
    },
    {
        "id": "quadratic",
        "prompt": "Solve x^2 - 5x + 6 = 0. What are the two roots? List them separated by a comma.",
        "answer": "2,3",
        "tolerance": None,
    },
    {
        "id": "percentage",
        "prompt": "What is 17.5% of 240? Give only the number.",
        "answer": "42",
        "tolerance": 0.01,
    },
    {
        "id": "log_base2",
        "prompt": "What is log base 2 of 1024? Give only the number.",
        "answer": "10",
        "tolerance": 0.01,
    },
    {
        "id": "pythagorean",
        "prompt": "A right triangle has legs of length 3 and 4. What is the length of the hypotenuse? Give only the number.",
        "answer": "5",
        "tolerance": 0.01,
    },
    {
        "id": "factorial",
        "prompt": "What is 7! (7 factorial)? Give only the number.",
        "answer": "5040",
        "tolerance": 0.5,
    },
    {
        "id": "series_sum",
        "prompt": "What is the sum of the arithmetic series 1 + 2 + 3 + ... + 100? Give only the number.",
        "answer": "5050",
        "tolerance": 0.5,
    },
]

FACTUAL_TASKS = [
    {
        "id": "python_creator",
        "prompt": "Who created the Python programming language? Give only the person's name.",
        "keywords": ["guido", "van rossum"],
    },
    {
        "id": "transformer_paper",
        "prompt": "What is the title of the 2017 Google paper that introduced the Transformer architecture? Give only the title.",
        "keywords": ["attention is all you need"],
    },
    {
        "id": "apple_silicon",
        "prompt": "What is the name of Apple's first in-house chip for Mac computers (released 2020)? Give only the chip name.",
        "keywords": ["m1"],
    },
    {
        "id": "git_creator",
        "prompt": "Who created the Git version control system? Give only the person's name.",
        "keywords": ["linus", "torvalds"],
    },
    {
        "id": "tcp_ip",
        "prompt": "What does TCP stand for in TCP/IP networking? Give only the full expansion.",
        "keywords": ["transmission control protocol"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fizzbuzz_ok(out: str) -> bool:
    """Correctness of fizzbuzz(15) output, INDEPENDENT of print style.

    The old check required the last line to endswith("FizzBuzz"), which fails a
    correct `print(fizzbuzz(15))` (a list repr ends with `']`) while passing a
    line-per-element print — i.e. it scored output FORMAT, not correctness. This
    extracts the token stream and looks for the exact expected sequence as a
    contiguous run, so list-repr, line-per-element, and label-prefixed output
    ("fizzbuzz(15): [...]") all score on whether the answer is right.
    """
    exp = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8",
           "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
    toks = re.findall(r"FizzBuzz|Fizz|Buzz|\d+", out)
    return any(toks[i:i + len(exp)] == exp for i in range(len(toks) - len(exp) + 1))


def _clean_math(text: str) -> str:
    """Strip LaTeX boxes, bold markdown, trailing punctuation."""
    text = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = text.strip().rstrip(".,;:")
    return text


def _extract_number(text: str) -> float | None:
    # Take the LAST number in the response, not the first. Models that show
    # their work put the final answer last ("7! = 5040"); the first number is
    # part of the working ("7! ...", "17.5% of 240 ..."). Extracting the first
    # number marks correct show-work answers as wrong.
    text = _clean_math(text).replace(",", "")
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    for tok in reversed(nums):
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def _extract_code(response: str) -> str:
    """Pull out the first Python code block, or everything if no block.

    KNOWN LIMITATION: this assumes the model emits a single, self-contained,
    runnable code block (definition + the invocation/print the task asks for).
    That holds for most instruct models (Qwen, Llama). It does NOT hold for
    models whose default output is exploratory chain-of-thought (e.g. Gemma 4's
    `<|channel>thought` format): they emit multiple candidate blocks and
    describe the invocation in prose, so the extracted block defines a function
    but never calls it -> empty output -> false FAIL. Such models' coding
    scores from this harness are not meaningful; report N/A, not a number.
    """
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: lines that look like code (start with def/import/print or indented)
    lines = []
    for line in response.splitlines():
        if line.startswith(("def ", "import ", "print(", "    ", "\t", "for ", "if ", "return ", "class ")):
            lines.append(line)
    return "\n".join(lines).strip()


def _run_code(code: str, timeout: int = 5) -> tuple[bool, str]:
    """Execute Python code in a subprocess, return (success, stdout)."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(fname)


def _check_math(response: str, task: dict) -> bool:
    answer = task["answer"]
    tol = task.get("tolerance")

    if tol is None:
        # Keyword/string match (yes/no, or a comma list like "2,3"). Normalize
        # whitespace so "2, 3" in the response matches the answer "2,3".
        clean = _clean_math(response).lower().replace(" ", "")
        return answer.lower().replace(" ", "") in clean

    # Numeric match within tolerance
    got = _extract_number(response)
    expected = float(answer)
    if got is None:
        return False
    return abs(got - expected) <= tol


def _check_factual(response: str, task: dict) -> bool:
    text = response.lower()
    return all(kw in text for kw in task["keywords"])


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_id: str, max_tokens: int = 512, verbose: bool = False,
                  no_think: bool = False):
    import mlx.core as mx
    from mlx_lm import load, generate

    print(f"\n{'='*60}")
    print(f"Model: {model_id}{'  [thinking OFF]' if no_think else ''}")
    print(f"Tasks: {len(CODING_TASKS)} coding + {len(MATH_TASKS)} math + {len(FACTUAL_TASKS)} factual = {len(CODING_TASKS)+len(MATH_TASKS)+len(FACTUAL_TASKS)} total")
    print(f"{'='*60}\n")

    # Reset peak memory so peak_gb reflects THIS model, not a cumulative peak
    # from a previous model in the same process (a batch runner reuses one
    # process). Without this, every model after the largest reports the same
    # inflated peak.
    mx.reset_peak_memory()

    print("Loading model...", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_id)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s\n")

    results = {"coding": [], "math": [], "factual": []}
    total_tokens = 0
    total_gen_time = 0.0

    def ask(prompt: str) -> tuple[str, float, int]:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False is the Qwen3 "one flag" that disables the
        # <think> trace. Pass it only when requested; templates that don't
        # support the kwarg (e.g. Llama) raise TypeError -> fall back.
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if no_think:
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(messages, **kwargs)
        else:
            formatted = tokenizer.apply_chat_template(messages, **kwargs)
        t_start = time.time()
        response = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
        elapsed = time.time() - t_start
        n_tokens = len(tokenizer.encode(response))
        return response, elapsed, n_tokens

    # --- Coding ---
    print("CODING TASKS")
    print("-" * 40)
    for task in CODING_TASKS:
        response, elapsed, n_tok = ask(task["prompt"])
        total_tokens += n_tok
        total_gen_time += elapsed
        code = _extract_code(response)
        ok, exec_out = _run_code(code)
        if ok:
            passed = task["check"](exec_out)
        else:
            passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {task['id']:20s}  ({n_tok} tok, {elapsed:.1f}s)")
        if verbose or not passed:
            print(f"         exec_out: {exec_out.strip()[:120]}")
        results["coding"].append(passed)

    # --- Math ---
    print("\nMATH TASKS")
    print("-" * 40)
    for task in MATH_TASKS:
        response, elapsed, n_tok = ask(task["prompt"])
        total_tokens += n_tok
        total_gen_time += elapsed
        passed = _check_math(response, task)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {task['id']:20s}  ({n_tok} tok, {elapsed:.1f}s)")
        if verbose or not passed:
            clean = _clean_math(response).strip()[:100]
            print(f"         got: {clean!r}  expected: {task['answer']}")
        results["math"].append(passed)

    # --- Factual ---
    print("\nFACTUAL TASKS")
    print("-" * 40)
    for task in FACTUAL_TASKS:
        response, elapsed, n_tok = ask(task["prompt"])
        total_tokens += n_tok
        total_gen_time += elapsed
        passed = _check_factual(response, task)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {task['id']:20s}  ({n_tok} tok, {elapsed:.1f}s)")
        if verbose or not passed:
            print(f"         got: {response.strip()[:100]!r}")
        results["factual"].append(passed)

    # --- Summary ---
    c = sum(results["coding"])
    m = sum(results["math"])
    f = sum(results["factual"])
    total = c + m + f
    n_total = len(CODING_TASKS) + len(MATH_TASKS) + len(FACTUAL_TASKS)
    avg_tps = total_tokens / total_gen_time if total_gen_time > 0 else 0

    peak_gb = mx.get_peak_memory() / 1e9

    print(f"\n{'='*60}")
    print(f"RESULTS: {total}/{n_total}  ({100*total//n_total}%)")
    print(f"  Coding:  {c}/{len(CODING_TASKS)}")
    print(f"  Math:    {m}/{len(MATH_TASKS)}")
    print(f"  Factual: {f}/{len(FACTUAL_TASKS)}")
    print(f"  Speed:   {avg_tps:.1f} tok/s (avg over all tasks)")
    print(f"  Peak RAM: {peak_gb:.2f} GB (mx.get_peak_memory)")
    print(f"{'='*60}\n")

    return {"coding": c, "math": m, "factual": f, "total": total,
            "n_total": n_total, "tok_s": avg_tps, "peak_gb": peak_gb}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking trace (Qwen3: enable_thinking=False)")
    args = parser.parse_args()
    run_benchmark(args.model, max_tokens=args.max_tokens, verbose=args.verbose,
                  no_think=args.no_think)
