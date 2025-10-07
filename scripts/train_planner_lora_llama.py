"""
LoRA training for Llama-2-7b as planner (seq generation).
Input: prompt (planner prompt), Target: JSON plan string (one-line or pretty JSON).
This script uses causal LM training where the model is trained to continue the prompt with the plan.
"""

import argparse, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True, help="jsonl with {input:..., target:...}")
    p.add_argument("--model_name_or_path", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_auth_token", action="store_true")
    return p.parse_args()

def load_jsonl(path):
    ds = load_dataset("json", data_files=path, split="train")
    return ds

def prepare_examples(examples, tokenizer, max_length):
    sources = examples["input"]
    targets = examples["target"]
    input_ids = []
    attention_mask = []
    labels = []
    for src, tgt in zip(sources, targets):
        # Build the full sequence: src + tgt
        src_ids = tokenizer(src, add_special_tokens=False)["input_ids"]
        tgt_ids = tokenizer(tgt, add_special_tokens=False)["input_ids"]
        ids = src_ids + tgt_ids + [tokenizer.eos_token_id]
        if len(ids) > max_length:
            ids = ids[:max_length]
        # labels: mask prompt area with -100 so loss only on target continuation OR compute full LM loss on entire sequence
        # Here we compute loss on continuation only:
        label = [-100] * len(src_ids) + tgt_ids + [tokenizer.eos_token_id]
        # pad to max_length
        pad_len = max_length - len(ids)
        ids = ids + [tokenizer.pad_token_id] * pad_len
        label = label + [-100] * pad_len
        input_ids.append(ids)
        labels.append(label)
        attention_mask.append([1 if x!=tokenizer.pad_token_id else 0 for x in ids])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

def main():
    args = parse_args()
    ds = load_jsonl(args.train_file)
    print("Examples:", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, use_auth_token=args.use_auth_token)
    # Ensure pad token
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token":"<pad>"})

    # load model with 8-bit if desired
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            load_in_8bit=True,     # requires bitsandbytes
            torch_dtype=torch.float16 if args.fp16 else None,
            low_cpu_mem_usage=True,
            use_auth_token=args.use_auth_token
        )
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print("8-bit load failed, fallback to normal load:", e)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.float16 if args.fp16 else None, use_auth_token=args.use_auth_token)

    # LoRA config; target_modules may need inspecting for your checkpoint
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","v_proj"],  # adjust if different on your checkpoint
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # prepare dataset
    def tok_map(batch):
        return prepare_examples(batch, tokenizer, args.max_length)
    ds_tok = ds.map(tok_map, batched=True, remove_columns=ds.column_names)

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=50,
        save_strategy="epoch",
        optim="adamw_torch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()