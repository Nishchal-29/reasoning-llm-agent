"""
agent/data_prep.py

Utilities to prepare training data for planner fine-tuning.

Two main modes:
 - "solution": create (input, target) pairs where target is the textual solution.
 - "planner" : attempt to convert an available multi-step trace into a JSON "plan"
               matching your executor schema:
               [ {"step":1,"tool":"python","input":"...", "comment":"..."}, ... ]

Usage examples:
  # 1) prepare textual-solution SFT data from OpenR1-Math:
  python -m agent.data_prep --mode solution --out data/openr1_solution_train.jsonl --max 50000

  # 2) prepare planner-format data (best-effort automatic conversion)
  python -m agent.data_prep --mode planner --out data/openr1_planner_train.jsonl --max 20000

Notes:
 - The HuggingFace dataset: open-r1/OpenR1-Math-220k is used if no local CSV provided.
 - Planner-mode uses heuristics to split a trace into steps (split on newlines/sentences).
 - You MUST inspect and curate planner outputs before using them for supervised planner training,
   because automatic conversion from free-form CoT -> executable plan is noisy.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datasets import load_dataset
import re
import html
import random

# Prompt prefix used at training time. Should match your inference prompt (minus {few_shot_examples}).
# Keep it strict and identical to prompts/planner_prompt.txt instructions except few-shot block.
DEFAULT_PROMPT_PREFIX = """You are a step-by-step REASONING PLANNER.

Your job: transform a math or logic problem into a short, deterministic, executable PLAN.
OUTPUT FORMAT (must follow exactly):
 - Produce EXACTLY one valid JSON array. No words, no commentary, nothing else.
 - The array must start with '[' and end with ']'.
 - Each element must be an object with ONLY these keys:
     "step"  : integer
     "tool"  : string   (one of: "calc","python","sympy","z3","bruteforce")
     "input" : string   (the exact command the executor should run)
     "comment": string  (one short sentence description)

Hard rules (follow exactly):
1) Never compute numbers in the plan text — use tools. For arithmetic use "calc" or "python", for algebra use "sympy", for constraints use "z3".
2) The "input" must be a concise, runnable snippet the executor can call.
3) Keep plans short (1–6 steps). Use step numbers starting at 1.

Plan:"""


def choose_best_trace(record: Dict[str, Any]) -> Optional[str]:
    """
    Heuristic to choose the best available reasoning trace / solution for a dataset row.
    Priority:
      1) 'solution' (if present and non-empty)
      2) 'generations' entries that pass verification flags if available
      3) first element of 'generations'
      4) 'answer' fallback (wrapped as a short solution)
    Returns a text trace or None.
    """
    # 1) solution field (often ready-made)
    sol = record.get("solution")
    if sol and isinstance(sol, str) and sol.strip():
        return sol.strip()

    # 2) generations: a list of generated traces (sometimes with metadata)
    gens = record.get("generations") or record.get("generation") or None
    if isinstance(gens, list) and len(gens) > 0:
        # If generations are dicts with 'text' or similar, extract
        # Also prefer traces with correctness flags if present
        # Try to find a generation with positive verification flags
        # common keys in generation dicts: 'text', 'solution', 'reasoning'
        best_candidate = None
        for g in gens:
            # if generation is a dict with text-like content
            if isinstance(g, dict):
                text = g.get("text") or g.get("solution") or g.get("reasoning") or None
                # check for verification flag inside this generation dict (if exists)
                verified = g.get("math_verify") or g.get("verified") or g.get("correctness") or None
                if text and verified:
                    return text.strip()
                if text and not best_candidate:
                    best_candidate = text.strip()
            elif isinstance(g, str):
                if len(g.strip()) > 10 and not best_candidate:
                    best_candidate = g.strip()
        if best_candidate:
            return best_candidate

    # 3) answer fallback: if there's an 'answer' field, turn into a small textual solution
    ans = record.get("answer")
    if ans is not None:
        # convert to string carefully (list/tuple -> comma join)
        if isinstance(ans, (list, tuple)):
            ans_str = ", ".join(map(str, ans))
        else:
            ans_str = str(ans)
        return f"Answer: {ans_str}"

    # no usable content
    return None


def split_solution_into_sentences(sol_text: str) -> List[str]:
    """
    Split a solution (coT) text into reasonable step-like sentences.
    Heuristic: split on newlines first; if none, split on sentence punctuation.
    Returns non-empty trimmed sentences.
    """
    # Unescape HTML entities if any
    sol_text = html.unescape(sol_text)
    # Normalize whitespace
    sol_text = re.sub(r"\s+", " ", sol_text).strip()
    # First try splitting by explicit newlines (many CoT traces use them)
    lines = [ln.strip() for ln in sol_text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines
    # Else, split by sentence end punctuation
    sents = re.split(r'(?<=[.!?])\s+', sol_text)
    sents = [s.strip() for s in sents if s.strip()]
    # If still too long, consider splitting by semicolons or commas heuristically
    return sents


def coerce_step_tool(sentence: str) -> Dict[str, str]:
    """
    Heuristic mapping from a natural-language sentence (step) to a tool + input + comment.
    This is necessarily approximate. We aim for safe defaults so executor can run them:
      - If sentence contains algebraic keywords (solve, equation, x=), prefer "sympy"
      - If sentence is arithmetic-like (digits, + - * /, %, **), prefer "calc"
      - Else default to "python" and put the sentence as a comment and a placeholder python return
    The function returns a dict: {'tool':..., 'input':..., 'comment':...}
    """
    s = sentence.strip()
    s_low = s.lower()
    # quick checks
    if re.search(r'\bsolve\b|\bsolve for\b|=', s_low):
        # try to extract a Pythonic sympy call: naive
        # We won't attempt perfect parsing here; request human curation later.
        return {
            "tool": "sympy",
            "input": s.replace("solve for", "solve(").replace("Solve for", "solve(")  # best-effort
                      + ("" if s.endswith(")") else ""),
            "comment": s[:200]
        }
    # arithmetic detection: presence of digits and arithmetic operators
    if re.search(r'\d', s) and re.search(r'[\+\-\*\/\^%]', s):
        # convert caret to ** if present
        s_calc = s.replace("^", "**")
        # Remove words that would break evaluation, keep expression-like characters
        expr = re.sub(r'[^0-9\.\+\-\*\/\%\(\)\s\*]', ' ', s_calc)
        expr = re.sub(r'\s+', ' ', expr).strip()
        if len(expr) > 0:
            return {"tool": "calc", "input": f"compute {expr}", "comment": s[:200]}
    # default: python placeholder; executor can treat as comment or run as-is if safe
    # Put the sentence in comment and use a no-op python that returns nothing (must be curated)
    py_input = f"# STEP (needs curation): {s}\nreturn {{'note':'needs_curation'}}"
    return {"tool": "python", "input": py_input, "comment": s[:200]}


def solution_to_plan_json(sol_text: str, max_steps: int = 8) -> List[Dict[str, Any]]:
    """
    Convert a free-form solution string into a list of plan steps (best-effort).
    Returns the list of dict steps (not serialized JSON).
    """

    sentences = split_solution_into_sentences(sol_text)
    if not sentences:
        return []

    steps = []
    # cap number of steps to avoid extremely long plans
    for i, sent in enumerate(sentences[:max_steps]):
        mapping = coerce_step_tool(sent)
        step_obj = {
            "step": i + 1,
            "tool": mapping["tool"],
            "input": mapping["input"],
            "comment": mapping["comment"]
        }
        steps.append(step_obj)
    # If the solution contains an explicit final numeric answer, try to append a final python return step
    # Try to extract last numeric token
    m = re.search(r'(-?\d+(\.\d+)?)\s*$', sol_text.strip())
    if m:
        last_num = m.group(1)
        steps.append({
            "step": len(steps) + 1,
            "tool": "python",
            "input": f"return {{'answer': {last_num}}}",
            "comment": "return numeric answer (extracted from solution)"
        })
    return steps


def build_input_from_prompt(problem_text: str, prompt_prefix: str = DEFAULT_PROMPT_PREFIX, few_shot_examples: str = "") -> str:
    """
    Construct the model input string for supervised fine-tuning:
      input = prompt_prefix (without {few_shot_examples}) + optional few_shot_examples + 'Problem: {problem_text}\nPlan:'
    We purposely avoid injecting a huge few-shot block during SFT; you may choose to add a small curated block.
    """
    p = prompt_prefix
    # If the prefix contains placeholder {few_shot_examples} we remove it in SFT (we don't want to bake in many examples).
    p = p.replace("{few_shot_examples}", few_shot_examples or "")
    # Ensure the prompt ends with 'Plan:' so the model produces the plan JSON
    if not p.rstrip().endswith("Plan:"):
        p = p.rstrip() + "\n\nPlan:"
    # Now append problem
    full = f"{p}\nProblem: {problem_text}\n"
    # If you're training the model to continue inside a JSON array, you may append '[' to the prompt:
    # full += "\nPlan: ["
    return full


def record_to_example(record: Dict[str, Any], mode: str = "planner", prompt_prefix: str = DEFAULT_PROMPT_PREFIX) -> Optional[Dict[str, str]]:
    """
    Convert a dataset record to a training example dict: {'input':..., 'target':...}
    mode: 'solution' or 'planner'
    """
    problem = record.get("problem") or record.get("question") or record.get("problem_statement")
    if not problem:
        return None
    chosen = choose_best_trace(record)
    if mode == "solution":
        if not chosen:
            return None
        input_text = build_input_from_prompt(problem, prompt_prefix=prompt_prefix, few_shot_examples="")
        target_text = chosen
        return {"input": input_text, "target": target_text}
    elif mode == "planner":
        # If 'record' includes a pre-structured plan in a field, use it directly
        existing_plan = record.get("plan") or None
        if existing_plan and isinstance(existing_plan, (list, dict, str)):
            # If it's already a list/dict, serialize to compact JSON
            if isinstance(existing_plan, (list, dict)):
                plan_json = json.dumps(existing_plan, ensure_ascii=False)
            else:
                plan_json = str(existing_plan)
            input_text = build_input_from_prompt(problem, prompt_prefix=prompt_prefix, few_shot_examples="")
            return {"input": input_text, "target": plan_json}

        # else try to convert textual solution -> plan
        if not chosen:
            return None
        steps = solution_to_plan_json(chosen, max_steps=8)
        if not steps:
            return None
        plan_json = json.dumps(steps, ensure_ascii=False)
        input_text = build_input_from_prompt(problem, prompt_prefix=prompt_prefix, few_shot_examples="")
        return {"input": input_text, "target": plan_json}
    else:
        raise ValueError("mode must be 'solution' or 'planner'")


def prepare_from_hf_dataset(split: str = "train", max_examples: Optional[int] = None, mode: str = "planner") -> List[Dict[str, str]]:
    """
    Load open-r1/OpenR1-Math-220k from Hugging Face and build examples.
    Returns a list of example dicts.
    """
    print("Loading HuggingFace dataset open-r1/OpenR1-Math-220k ...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split=split)
    print("Dataset loaded, iterating...")
    out = []
    for i, rec in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        ex = record_to_example(rec, mode=mode)
        if ex:
            out.append(ex)
        if (i + 1) % 1000 == 0:
            print(f"processed {i+1} rows, produced {len(out)} examples")
    print("Finished. produced", len(out), "examples")
    return out


def load_local_csv(csv_path: str, problem_col: str = "problem", solution_col: str = "solution", mode: str = "planner", max_examples: Optional[int] = None):
    """
    Load a local CSV and map rows to examples. Expects problem and solution columns (or adjust).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    rows = []
    for idx, r in df.iterrows():
        if max_examples and idx >= max_examples:
            break
        rec = {"problem": r.get(problem_col), "solution": r.get(solution_col), "answer": r.get("answer")}
        ex = record_to_example(rec, mode=mode)
        if ex:
            rows.append(ex)
    return rows


def save_jsonl(out_path: str, examples: List[Dict[str, str]]):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")

def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["planner", "solution"], default="planner")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--split", default="train", help="HF dataset split (train/validation/test)")
    ap.add_argument("--max", type=int, default=None, help="max examples to prepare")
    ap.add_argument("--local_csv", default=None, help="optional local CSV path to load instead of HF dataset")
    ap.add_argument("--prompt_prefix", default=None, help="optional custom prompt prefix file")
    args = ap.parse_args()

    prompt_prefix_text = DEFAULT_PROMPT_PREFIX
    if args.prompt_prefix:
        prompt_prefix_text = Path(args.prompt_prefix).read_text(encoding="utf-8")

    if args.local_csv:
        examples = load_local_csv(args.local_csv, mode=args.mode, max_examples=args.max)
    else:
        examples = prepare_from_hf_dataset(split=args.split, max_examples=args.max, mode=args.mode)

    # Optionally shuffle to get a spread
    random.shuffle(examples)

    save_jsonl(args.out, examples)

    # write a small sample for quick inspection
    sample_path = Path(args.out).with_suffix(".sample.json")
    sample = examples[:100]
    with sample_path.open("w", encoding="utf-8") as sf:
        json.dump(sample, sf, ensure_ascii=False, indent=2)
    print("Sample saved to", sample_path)

if __name__ == "__main__":
    main_cli()
