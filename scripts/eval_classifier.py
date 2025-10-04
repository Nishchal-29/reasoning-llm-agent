# scripts/eval_classifier.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json
import torch
from datasets import load_dataset

MODEL_DIR = "outputs/llama7b_problem_type_lora"  # or the dir where you saved LoRA adapter
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load base and adapter if necessary
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = model.to(DEVICE)
model.eval()

def make_prompt(problem):
    template = open("prompts/problem_type_prompt.txt").read() if Path("prompts/problem_type_prompt.txt").exists() else None
    if template:
        return template.replace("{problem}", problem)
    else:
        return "Problem:\n" + problem + "\n\nType:"

def predict(problem, max_new_tokens=16):
    prompt = make_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # extract continuation after prompt
    cont = text[len(prompt):].strip()
    # take first token/word as label (split on whitespace or newline)
    label = cont.split()[0] if cont else ""
    return label

# Evaluate on a small validation split (use your JSONL or dataset)
ds = load_dataset("json", data_files="data/openr1_problem_type_train.jsonl", split="train[:2000]")
correct = 0
total = 0
for ex in ds:
    pred = predict(ex["input"].split("\n\nType:")[-1].strip(), max_new_tokens=8)
    # gold is ex["target"]
    gold = ex["target"].strip()
    total += 1
    if pred.lower() == gold.lower():
        correct += 1

print("Exact-match accuracy:", correct/total, f"({correct}/{total})")