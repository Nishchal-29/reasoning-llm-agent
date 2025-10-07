from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
import torch
from datasets import load_dataset

MODEL_DIR = "outputs/flan_t5_problem_type"  
BASE_MODEL = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPT = (
    "You are a classifier. Given a math problem, output a single token label describing the PROBLEM TYPE.\n"
    "Output FORMAT: a single short label (e.g., 'algebra', 'geometry', 'probability', 'combinatorics', 'number_theory').\n\n"
    "Problem:\n{problem}\n\nType:"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
base.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base, MODEL_DIR)

model = model.to(DEVICE)
model.eval()

def make_prompt(problem):
    return DEFAULT_PROMPT.replace("{problem}", problem)

def predict(problem, max_new_tokens=16):
    prompt = make_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, early_stopping=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    cont = text.strip()
    return cont

# q = "What are the conditions for a triangle's two vertices, its orthocenter, and the center of its inscribed circle to lie on a circle?"
# q2 = "How many chocolates are left if I have 10 and I gives 2 to my brother?"
# model2 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
# tokenizer2 = AutoTokenizer.from_pretrained("google/flan-t5-small")
# def predict2(problem, max_new_tokens=16):
#     prompt = make_prompt(problem)
#     inputs = tokenizer2(prompt, return_tensors="pt", truncation=True).to(DEVICE)
#     out = model2.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, early_stopping=True, eos_token_id=tokenizer2.eos_token_id, pad_token_id=tokenizer2.pad_token_id)
#     text = tokenizer2.decode(out[0], skip_special_tokens=True)
#     cont = text.strip()
#     # label = cont.split()[0] if cont else ""
#     return cont

# print("A:", predict(q, max_new_tokens=8))
# print("A2:", predict2(q, max_new_tokens=8))

# print("A:", predict(q2, max_new_tokens=8))
# print("A2:", predict2(q2, max_new_tokens=8))