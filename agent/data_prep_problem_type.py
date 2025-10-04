"""
agent/data_prep_problem_type.py

Prepare a JSONL file for conditional LM training to map:
  Input (prompt) -> Target (problem_type string)

Usage:
  python -m agent.data_prep_problem_type \
    --out data/openr1_problem_type_train.jsonl \
    --max_examples 50000

The script loads open-r1/OpenR1-Math-220k by default; optionally pass --local_csv to read a CSV.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset
import random
import csv

DEFAULT_PROMPT = (
    "You are a classifier. Given a math problem, output a single token label describing the PROBLEM TYPE.\n"
    "Output FORMAT: a single short label (e.g., 'algebra', 'geometry', 'probability', 'combinatorics', 'number_theory').\n\n"
    "Problem:\n{problem}\n\nType:"
)

def load_hf_openr1(split: str = "train"):
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split=split)
    return ds

def csv_to_examples(csv_path: str, problem_col: str = "problem", type_col: str = "problem_type", max_examples: Optional[int] = None):
    rows = []
    import pandas as pd
    df = pd.read_csv(csv_path)
    for i, r in df.iterrows():
        if max_examples and i >= max_examples:
            break
        problem = str(r.get(problem_col) or "").strip()
        ptype = r.get(type_col)
        if ptype is None or str(ptype).strip()=="":
            continue
        rows.append({"problem": problem, "problem_type": str(ptype).strip()})
    return rows

def hf_to_examples(split: str = "train", max_examples: Optional[int] = None):
    ds = load_hf_openr1(split)
    examples = []
    for i, rec in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        problem = rec.get("problem") or ""
        ptype = rec.get("problem_type") or None
        if not ptype or str(ptype).strip() == "":
            continue
        examples.append({"problem": str(problem).strip(), "problem_type": str(ptype).strip()})
    return examples

def build_io(problem: str, problem_type: str, prompt_template: str = DEFAULT_PROMPT):
    # Build input prompt and target label
    # Keep target minimal -- a single token label (we'll rely on tokenizer)
    input_text = prompt_template.format(problem=problem.strip())
    target_text = problem_type.strip()
    return {"input": input_text, "target": target_text}

def save_jsonl(out_path: str, examples: list):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output jsonl file")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--local_csv", default=None, help="Optional local CSV path")
    ap.add_argument("--prompt_template", default=None, help="Optional prompt template file (must include {problem})")
    args = ap.parse_args()

    prompt_template = DEFAULT_PROMPT
    if args.prompt_template:
        prompt_template = Path(args.prompt_template).read_text(encoding="utf-8")

    if args.local_csv:
        raw = csv_to_examples(args.local_csv, max_examples=args.max_examples)
    else:
        raw = hf_to_examples(split=args.split, max_examples=args.max_examples)

    examples = []
    for r in raw:
        try:
            io = build_io(r["problem"], r["problem_type"], prompt_template)
            examples.append(io)
        except Exception as e:
            continue

    random.shuffle(examples)
    save_jsonl(args.out, examples)

if __name__ == "__main__":
    main()
