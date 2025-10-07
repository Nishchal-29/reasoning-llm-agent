from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
from pathlib import Path

CLASSIFIER_DIR = "outputs/flan_t5_problem_type"  
PLANNER_DIR = "outputs/flan_t5_planner_lora"     
BASE = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
base.resize_token_embeddings(len(tokenizer))
classifier = PeftModel.from_pretrained(base, CLASSIFIER_DIR).to(DEVICE)
classifier.eval()

DEFAULT_PROMPT = (
    "You are a classifier. Given a math problem, output a single token label describing the PROBLEM TYPE.\n"
    "Output FORMAT: a single short label (e.g., 'algebra', 'geometry', 'probability', 'combinatorics', 'number_theory').\n\n"
    "Problem:\n{problem}\n\nType:"
)

def classify(problem, max_new_tokens=8):
    prompt = DEFAULT_PROMPT.replace("{problem}", problem)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = classifier.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    label = txt.strip().split()[0].lower()
    return label

if __name__ == "__main__":
    p = "A coffee shop sold 45 lattes on Monday and 20% more on Tuesday. How many in total?"
    print("pred:", classify(p))
