"""
LoRA fine-tuning script for Flan-T5-small on (input, target) jsonl.

Expected input data format (jsonl):
{"input": "<prompt_prefix>... Problem: <problem_text>\nPlan:", "target": "<plan_json_text>"}
(one example per line)

Usage example:
python scripts/train_lora_planner.py \
  --train_file data/openr1_planner_train.jsonl \
  --output_dir outputs/flan_t5_small_lora_planner \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Any

import transformers
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch
import math


@dataclass
class TrainConfig:
    model_name_or_path: str = "google/flan-t5-small"
    train_file: str = "data/openr1_planner_train.jsonl"
    output_dir: str = "outputs/flan_t5_small_lora_planner"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_source_length: int = 512
    max_target_length: int = 256
    logging_steps: int = 200
    save_strategy: str = "epoch"
    eval_strategy: str = "no"  # set to "epoch" if you have eval set
    gradient_accumulation_steps: int = 1
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fp16: bool = True
    optim: str = "adamw_torch"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, default="google/flan-t5-small")
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_source_length", type=int, default=512)
    ap.add_argument("--max_target_length", type=int, default=256)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--overwrite_output_dir", action="store_true")
    args = ap.parse_args()
    return args


def main():
    args = parse_args()

    cfg = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fp16=args.fp16,
    )

    transformers.logging.set_verbosity_info()
    print("Loading dataset from", cfg.train_file)

    # load newline JSON (jsonl)
    ds = load_dataset("json", data_files={"train": cfg.train_file}, split="train")
    print("Loaded", len(ds), "examples")

    # Load tokenizer and model
    print("Loading tokenizer & model:", cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)

    # Make tokenizer add EOS token for T5 if needed
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for k-bit training if you later use quantization (optional)
    # model = prepare_model_for_kbit_training(model)  # uncomment if using 8-bit

    # Setup LoRA with PEFT
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q", "v", "k", "o"] if "t5" in cfg.model_name_or_path.lower() else ["q", "v"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, peft_config)
    # Now model is wrapped for LoRA training — only LoRA params will be trained.

    # Tokenization / preprocessing
    def preprocess_function(examples: Dict[str, List[str]]):
        # examples["input"] and examples["target"] are strings
        inputs = examples["input"]
        targets = examples["target"]

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=cfg.max_source_length, truncation=True, padding="max_length")

        # Tokenize targets (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=cfg.max_target_length, truncation=True, padding="max_length")

        # Replace pad token id's in the labels by -100 so they are ignored in loss
        label_ids = labels["input_ids"]
        label_ids_masked = []
        for ids in label_ids:
            masked = [(l if l != tokenizer.pad_token_id else -100) for l in ids]
            label_ids_masked.append(masked)

        model_inputs["labels"] = label_ids_masked
        return model_inputs

    # Map dataset
    print("Tokenizing dataset...")
    tokenized = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)

    # Data collator to dynamically pad
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        predict_with_generate=True,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        remove_unused_columns=False,
        run_name="flan_t5_lora_planner",
        seed=cfg.seed,
        dataloader_num_workers=2,
        optim=cfg.optim,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training finished. Saving model to", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved.")

    # Print PEFT config / trainable params
    try:
        import peft
        print("PEFT trained modules:", [n for n, p in model.named_parameters() if p.requires_grad])
    except Exception:
        pass

if __name__ == "__main__":
    main()
