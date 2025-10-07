"""
Create a stratified pilot sampler JSONL from train.csv, using your fine-tuned
classifier to produce problem_type labels.

Usage:
  python scripts/create_pilot_sampler.py \
      --input /mnt/data/train.csv \
      --out data/pilot_problems.jsonl \
      --n 5000 \
      --model_dir outputs/flan_t5_problem_type \
      --batch_size 64
"""
import argparse
import os
import json
import math
import random
import pandas as pd
from collections import Counter
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/mnt/data/train.csv", help="Path to train.csv")
    p.add_argument("--out", default="data/pilot_problems.jsonl", help="Path to output jsonl")
    p.add_argument("--n", type=int, default=5000, help="Number of samples to draw (max)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_dir", default="outputs/flan_t5_problem_type", help="PEFT adapter directory for classifier")
    p.add_argument("--base_model", default="google/flan-t5-small", help="base model name used with PEFT")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for classifier inference")
    p.add_argument("--max_new_tokens", type=int, default=8, help="Max tokens to generate for class label")
    return p.parse_args()

def stratified_sample(df, group_col, target_n, seed=42):
    total = len(df)
    if total <= target_n:
        return df.copy()
    groups = df.groupby(group_col)
    quotas = {}
    fractional = []
    for name, g in groups:
        prop = len(g) / total
        exact = prop * target_n
        q = math.floor(exact)
        quotas[name] = q
        fractional.append((name, exact - q))
    remainder = target_n - sum(quotas.values())
    fractional.sort(key=lambda x: x[1], reverse=True)
    idx = 0
    while remainder > 0 and idx < len(fractional):
        quotas[fractional[idx][0]] += 1
        remainder -= 1
        idx += 1
    for name, g in groups:
        quotas[name] = min(quotas[name], len(g))
    assigned = sum(quotas.values())
    if assigned < target_n:
        deficit = target_n - assigned
        remaining_caps = sorted([(name, len(g) - quotas[name]) for name,g in groups], key=lambda x: x[1], reverse=True)
        i = 0
        while deficit > 0 and i < len(remaining_caps):
            name, cap = remaining_caps[i]
            add = min(cap, deficit)
            quotas[name] += add
            deficit -= add
            i += 1
    sampled_indices = []
    for name, g in groups:
        q = quotas.get(name, 0)
        if q > 0:
            chosen = g.sample(n=q, random_state=seed).index.tolist()
            sampled_indices.extend(chosen)
    if len(sampled_indices) < target_n:
        need = target_n - len(sampled_indices)
        remaining = list(set(df.index.tolist()) - set(sampled_indices))
        if need > len(remaining):
            need = len(remaining)
        sampled_indices.extend(random.sample(remaining, need))
    elif len(sampled_indices) > target_n:
        sampled_indices = random.sample(sampled_indices, target_n)
    return df.loc[sampled_indices].reset_index(drop=True)

def load_classifier(model_dir: str, base_model: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True) 
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, model_dir)
    model = model.to(device)
    model.eval()
    return tokenizer, model

def batch_predict_labels(texts: List[str], tokenizer, model, device: str, max_new_tokens: int = 8):
    """Predict labels for a list of texts using seq2seq PEFT model. Returns list[str]."""
    labels = []
    model_device = device
    bs = 32  
    with torch.no_grad():
        enc = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(model_device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                             early_stopping=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        for out in gen:
            text = tokenizer.decode(out, skip_special_tokens=True).strip()
            if len(text) == 0:
                labels.append("unknown")
            else:
                lab = text.split()[0].strip().lower()
                labels.append(lab)
    return labels

def main():
    args = parse_args()
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    print("Loading CSV:", args.input)
    df = pd.read_csv(args.input)

    problem_col = "problem_statement"
    if problem_col not in df.columns:
        text_lengths = {c: df[c].astype(str).map(len).mean() for c in df.columns}
        problem_col = max(text_lengths.items(), key=lambda x: x[1])[0]
        print(f"Warning: 'problem_statement' not found. Using '{problem_col}' as problem column.")

    df['__problem__'] = df[problem_col].astype(str).str.strip()
    df = df[df['__problem__'].str.len() > 0].reset_index(drop=True)
    total = len(df)
    if total == 0:
        raise ValueError("No non-empty problems found in dataset.")
    print(f"Found {total} problems. Using device={device}")
    print("Loading classifier adapter from:", args.model_dir)
    tokenizer, classifier = load_classifier(args.model_dir, args.base_model, device)
    batch_size = args.batch_size
    predicted = []
    n = len(df)
    print(f"Running classifier inference in batches of {batch_size} ...")
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_texts = [  
            ("You are a classifier. Given a math problem, output a single token label describing the PROBLEM TYPE.\n"
             "Output FORMAT: a single short label (e.g., 'algebra', 'geometry', 'probability', 'combinatorics', 'number_theory').\n\n"
             "Problem:\n" + p + "\n\nType:")
            for p in df['__problem__'].iloc[start:end].tolist()
        ]
        try:
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                gen = classifier.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                          early_stopping=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            for out in gen:
                text = tokenizer.decode(out, skip_special_tokens=True).strip()
                if len(text) == 0:
                    predicted.append("unknown")
                else:
                    lab = text.split()[0].lower()
                    predicted.append(lab)
        except Exception as e:
            print(f"Warning: classifier batch failed at {start}-{end}: {e}")
            predicted.extend(["unknown"] * (end - start))

        if (start // batch_size) % 10 == 0:
            print(f"  processed {end}/{n}")

    df['__ptype__'] = predicted
    print("Classifier done. Top predicted types:")
    for k,v in Counter(df['__ptype__'].tolist()).most_common(20):
        print(f"  {k}: {v}")

    sampled_df = stratified_sample(df, '__ptype__', args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for _, row in sampled_df.iterrows():
            rec = {"problem": row['__problem__'], "problem_type": row['__ptype__']}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sampled_df)} examples to {args.out}")

    print("Example outputs (first 3):")
    for _, r in sampled_df.head(3).iterrows():
        print("----")
        print("type:", r['__ptype__'])
        print(r['__problem__'][:400].replace("\n"," "))

if __name__ == "__main__":
    main()
