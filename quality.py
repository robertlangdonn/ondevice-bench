#!/usr/bin/env python3
"""
quality.py — task-based quality benchmark for on-device LLMs on Apple Silicon.

Measures pass rate on concrete tasks where correctness is verifiable:
  - coding: generate Python, execute it, check output
  - math:   generate answer, check against known result
  - factual: check key facts present in response

Each model runs in a fresh subprocess (isolated memory).

Usage:
    python3 quality.py --models mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
    python3 quality.py --suite coding --models mlx-community/Qwen3-4B-4bit
    python3 quality.py --list-tasks
    python3 quality.py --models A B C  # compare multiple models
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(response: str) -> str:
    """Remove <think>...</think> blocks emitted by thinking models (Qwen3, etc.)."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _extract_code(response: str) -> str:
    """Pull first Python code block from a response, stripping thinking tags first."""
    response = _strip_thinking(response)
    lines = response.split("\n")
    in_block, code_lines = False, []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```python") or stripped == "```python":
            in_block = True
            continue
        if in_block and stripped.startswith("```"):
            break
        if in_block:
            code_lines.append(line)
    if not code_lines:
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith(("#", "```"))]
    return "\n".join(code_lines)


def _run_code(code: str, expected: str, timeout: int = 5) -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() == expected.strip()
    except Exception:
        return False


def _contains(response: str, *words) -> bool:
    r = response.lower()
    return all(w.lower() in r for w in words)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    suite: str
    prompt: str
    check: object  # callable(str) -> bool
    description: str


def _chk_fib(r):
    code = _extract_code(r)
    return _run_code(code + "\nprint(fibonacci(10))", "55")

def _chk_fizzbuzz(r):
    code = _extract_code(r)
    out = subprocess.run([sys.executable, "-c", code + "\nfizzbuzz(15)"],
                         capture_output=True, text=True, timeout=5).stdout
    return "FizzBuzz" in out and "Fizz" in out and "Buzz" in out

def _chk_reverse(r):
    code = _extract_code(r)
    return _run_code(code + "\nprint(reverse_string('hello'))", "olleh")

def _chk_palindrome(r):
    code = _extract_code(r)
    return _run_code(
        code + "\nprint(is_palindrome('racecar'))\nprint(is_palindrome('hello'))",
        "True\nFalse"
    )

def _chk_vowels(r):
    code = _extract_code(r)
    return _run_code(code + "\nprint(count_vowels('hello world'))", "3")

def _chk_flatten(r):
    code = _extract_code(r)
    return _run_code(code + "\nprint(flatten([[1, 2], [3, [4, 5]]]))", "[1, 2, 3, 4, 5]")

def _chk_binary(r):
    code = _extract_code(r)
    return _run_code(
        code + "\nprint(binary_search([1,3,5,7,9,11], 7))\nprint(binary_search([1,3,5], 4))",
        "3\n-1"
    )

def _chk_anagram(r):
    code = _extract_code(r)
    return _run_code(
        code + "\nprint(is_anagram('listen', 'silent'))\nprint(is_anagram('hello', 'world'))",
        "True\nFalse"
    )

def _clean_math(r: str) -> str:
    """Strip LaTeX/markdown formatting for numeric checking."""
    import re
    r = re.sub(r"\\boxed\{([^}]+)\}", r"\1", r)   # \boxed{N} → N
    r = re.sub(r"\*\*([^*]+)\*\*", r"\1", r)       # **bold** → bold
    r = re.sub(r"__([^_]+)__", r"\1", r)           # __bold__ → bold
    r = r.replace("$", "").replace(",", "")
    return r

def _chk_mul(r):    return "221" in _clean_math(r)
def _chk_pct(r):    return "36" in _clean_math(r)
def _chk_prime(r):
    r = r.lower().strip()
    if "not prime" in r or "isn't prime" in r or "is not prime" in r or r.startswith("no"):
        return False
    return r.startswith("yes") or "is prime" in r or "97 is a prime" in r or r == "yes"
def _chk_sum(r):    return "55" in _clean_math(r)
def _chk_sqrt(r):   return "12" in _clean_math(r)
def _chk_compound(r):
    c = _clean_math(r)
    return "1610.51" in c or "1610.5" in c
def _chk_gcd(r):
    c = _clean_math(r).lower()
    # Accept "12" anywhere as long as the answer isn't clearly wrong
    return bool(re.search(r'\b12\b', c)) and "not 12" not in c
def _chk_log(r):
    c = _clean_math(r).strip()
    return c.startswith("3") or " 3" in c[:10] or c == "3"

def _chk_python(r): return _contains(r, "guido", "rossum")
def _chk_attn(r):   return _contains(r, "attention")
def _chk_sort(r):
    import re as _re
    r2 = r.lower().replace("\\log", "log")
    # normalise all "n log n", "n·log n", "nlogn" variants to "nlogn"
    r2 = _re.sub(r"n\s*[·*]?\s*log\s*[₂2]?\s*n", "nlogn", r2)
    return "nlogn" in r2
def _chk_os(r):     return _contains(r, "unix")
def _chk_git(r):    return _contains(r, "linus")


TASKS: list[Task] = [
    # Coding — 8 tasks
    Task("fib",        "coding", "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number iteratively (not recursively). Only output the Python function, no explanation.", _chk_fib, "fibonacci(10)==55"),
    Task("fizzbuzz",   "coding", "Write a Python function called `fizzbuzz(n)` that prints 1 to n, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', multiples of both with 'FizzBuzz'. Only output the function.", _chk_fizzbuzz, "fizzbuzz(15) prints Fizz/Buzz/FizzBuzz"),
    Task("reverse",    "coding", "Write a Python function called `reverse_string(s)` that returns the string reversed. Only output the function.", _chk_reverse, "reverse_string('hello')=='olleh'"),
    Task("palindrome", "coding", "Write a Python function called `is_palindrome(s)` returning True if the string is a palindrome, False otherwise. Only output the function.", _chk_palindrome, "racecar→True, hello→False"),
    Task("vowels",     "coding", "Write a Python function called `count_vowels(s)` returning the count of vowels (a,e,i,o,u), case-insensitive. Only output the function.", _chk_vowels, "count_vowels('hello world')==3"),
    Task("flatten",    "coding", "Write a Python function called `flatten(lst)` that flattens a nested list of any depth. Only output the function.", _chk_flatten, "flatten([[1,2],[3,[4,5]]])==[1,2,3,4,5]"),
    Task("binary_search", "coding", "Write a Python function called `binary_search(arr, target)` that returns the index of target in sorted arr, or -1 if not found. Only output the function.", _chk_binary, "binary_search([1,3,5,7,9,11],7)==3"),
    Task("anagram",    "coding", "Write a Python function called `is_anagram(s1, s2)` returning True if the strings are anagrams of each other, False otherwise. Only output the function.", _chk_anagram, "is_anagram('listen','silent')==True"),
    # Math — 5 tasks
    Task("multiply",   "math", "What is 17 multiplied by 13? Answer with just the number.", _chk_mul, "17×13=221"),
    Task("percentage", "math", "What is 15% of 240? Answer with just the number.", _chk_pct, "15% of 240=36"),
    Task("prime",      "math", "Is 97 a prime number? Answer with just Yes or No.", _chk_prime, "97 is prime"),
    Task("series",     "math", "What is the sum of the first 10 natural numbers (1 through 10)? Answer with just the number.", _chk_sum, "1+…+10=55"),
    Task("sqrt",       "math", "What is the square root of 144? Answer with just the number.", _chk_sqrt, "√144=12"),
    Task("compound",   "math", "If you invest $1,000 at 10% annual compound interest for 5 years, what is the final amount rounded to 2 decimal places? Answer with just the number.", _chk_compound, "$1000×1.1^5=1610.51"),
    Task("gcd",        "math", "What is the greatest common divisor (GCD) of 84 and 36? Answer with just the number.", _chk_gcd, "gcd(84,36)=12"),
    Task("log2",       "math", "What is log base 2 of 8? Answer with just the number.", _chk_log, "log2(8)=3"),
    # Factual — 5 tasks
    Task("python_creator", "factual", "Who created the Python programming language? One sentence.", _chk_python, "Guido van Rossum"),
    Task("attention",      "factual", "What mechanism makes transformers different from earlier RNN-based sequence models? One sentence.", _chk_attn, "attention"),
    Task("quicksort",      "factual", "What is the average-case time complexity of quicksort? Big-O notation only.", _chk_sort, "O(n log n)"),
    Task("unix_origin",    "factual", "What operating system is Linux based on or inspired by? One word.", _chk_os, "Unix/Linux"),
    Task("git_creator",    "factual", "Who created the Git version control system? One sentence.", _chk_git, "Linus Torvalds"),
]


# ---------------------------------------------------------------------------
# Per-model runner — loads model once, runs all tasks, returns JSON lines
# ---------------------------------------------------------------------------

_RUNNER = '''
import sys, json
from mlx_lm import load, generate

model_path = sys.argv[1]
max_tokens = int(sys.argv[2])
tasks_json = sys.argv[3]   # JSON: list of {id, prompt}

model, tokenizer = load(model_path)

import re

def _strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

tasks = json.loads(tasks_json)
for task in tasks:
    messages = [{"role": "user", "content": task["prompt"]}]
    # Disable thinking mode for models that support it (Qwen3, etc.)
    # to avoid multi-minute chain-of-thought for simple tasks
    formatted = None
    for kwargs in [{"enable_thinking": False}, {}]:
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            break
        except TypeError:
            continue
    if formatted is None:
        formatted = task["prompt"]
    response = generate(model, tokenizer, formatted, max_tokens=max_tokens, verbose=False)
    response = _strip_thinking(response)
    print(json.dumps({"id": task["id"], "response": response}), flush=True)
'''


def run_suite(model_path: str, tasks: list, max_tokens: int, python: str, timeout: int) -> list[dict]:
    """Load model once, run all tasks, return list of result dicts."""
    tasks_arg = json.dumps([{"id": t.id, "prompt": t.prompt} for t in tasks])
    task_map = {t.id: t for t in tasks}

    try:
        r = subprocess.run(
            [python, "-c", _RUNNER, model_path, str(max_tokens), tasks_arg],
            capture_output=True, text=True,
            timeout=timeout * len(tasks) + 120,  # model load + N tasks
        )
        results = []
        seen = set()
        for line in r.stdout.splitlines():
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                tid = data.get("id")
                if tid not in task_map or tid in seen:
                    continue
                seen.add(tid)
                response = data.get("response", "")
                passed = bool(task_map[tid].check(response))
                results.append({"task": tid, "passed": passed, "response": response[:300]})
            except Exception:
                continue

        # Any tasks that didn't produce output
        for t in tasks:
            if t.id not in seen:
                results.append({"task": t.id, "passed": False, "error": "no output", "response": ""})

        return results

    except subprocess.TimeoutExpired:
        return [{"task": t.id, "passed": False, "error": "suite timeout"} for t in tasks]
    except Exception as e:
        return [{"task": t.id, "passed": False, "error": str(e)} for t in tasks]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task-based quality benchmark for on-device LLMs")
    ap.add_argument("--models", nargs="+",
                    default=["mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"])
    ap.add_argument("--suite", choices=["coding", "math", "factual", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=120, help="seconds per task")
    ap.add_argument("--out", default="results/quality.json")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--list-tasks", action="store_true")
    args = ap.parse_args()

    if args.list_tasks:
        for t in TASKS:
            print(f"  [{t.suite:8}] {t.id:20} — {t.description}")
        return

    suite_tasks = TASKS if args.suite == "all" else [t for t in TASKS if t.suite == args.suite]
    all_results = {}

    for model_path in args.models:
        name = model_path.split("/")[-1]
        print(f"\n{'='*62}")
        print(f"  {name}")
        print(f"{'='*62}")
        print(f"  {'Task':<22} {'Suite':<10} Result")
        print(f"  {'-'*22} {'-'*10} ------")

        print(f"  Loading model and running {len(suite_tasks)} tasks…", flush=True)
        raw_results = run_suite(model_path, suite_tasks, args.max_tokens, args.python, args.timeout)
        task_lookup = {t.id: t for t in suite_tasks}
        results, passed = [], 0
        for r in raw_results:
            task = task_lookup.get(r["task"])
            suite = task.suite if task else "?"
            icon = "✓" if r["passed"] else "✗"
            note = f"  [{r.get('error','')[:35]}]" if not r["passed"] and r.get("error") else ""
            print(f"  {r['task']:<22} {suite:<10} {icon}{note}")
            if r["passed"]:
                passed += 1
            results.append(r)

        total = len(suite_tasks)
        pct = round(100 * passed / total) if total else 0
        print(f"\n  {passed}/{total} ({pct}%)  ──  {name}")
        all_results[model_path] = {"passed": passed, "total": total, "pct": pct, "tasks": results}

    if len(args.models) > 1:
        print(f"\n{'='*62}")
        print("  Summary")
        print(f"{'='*62}")
        print(f"  {'Model':<42} Score")
        print(f"  {'-'*42} -----")
        for mp, r in all_results.items():
            n = mp.split("/")[-1][:42]
            print(f"  {n:<42} {r['passed']}/{r['total']} ({r['pct']}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
