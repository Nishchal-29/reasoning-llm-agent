# scripts/train_lora_llama_classifier.py
"""
LoRA training for decoder-only LM (Llama-family) to map prompt -> short label (target).
Training format: each example is {"input": prompt_str, "target": target_str}

Important: This script trains causal LM; it masks prompt tokens in labels so loss focuses on target.
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import math
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--target_modules", type=str, default="q_proj,v_proj")  # comma separated; may need to adjust
    return ap.parse_args()

def load_jsonl_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    return ds

def preprocess_tokenize(example, tokenizer, max_length):
    # example: {"input":..., "target":...}
    inp = example["input"]
    tgt = example["target"]
    # tokenization
    input_ids = tokenizer(inp, truncation=True, max_length=max_length, padding=False)["input_ids"]
    target_ids = tokenizer(" " + tgt, truncation=True, max_length=64, padding=False)["input_ids"]
    # combine
    ids = input_ids + target_ids
    # construct labels with -100 for prompt tokens
    labels = [-100] * len(input_ids) + target_ids
    # pad to max_length (right padding)
    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]
    else:
        pad_len = max_length - len(ids)
        ids = ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
    return {"input_ids": ids, "labels": labels, "attention_mask": [1 if x != tokenizer.pad_token_id else 0 for x in ids]}

def main():
    args = parse_args()
    # Load dataset
    print("Loading dataset:", args.train_file)
    ds = load_jsonl_dataset(args.train_file)
    print("Examples:", len(ds))

    # tokenizer & model
    print("Loading tokenizer & model:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.float16 if args.fp16 else None)
    model.resize_token_embeddings(len(tokenizer))

    # Optionally prepare for k-bit training if using bitsandbytes (not required)
    # model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("Wrapped model in PEFT (LoRA). Trainable parameters:")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print("Trainable params:", n_trainable, "Total params:", n_total)

    # Preprocess dataset (tokenize & build labels)
    max_length = args.max_length
    print("Tokenizing & preparing labels (max_length=", max_length, ") ...")
    def tokenize_fn(examples):
        return preprocess_tokenize(examples, tokenizer, max_length)
    tokenized = ds.map(lambda ex: tokenize_fn(ex), remove_columns=ds.column_names)

    # Data collator (not typical LM collator because we already padded)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_strategy="epoch",
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()
