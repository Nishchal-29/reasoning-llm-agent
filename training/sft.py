from __future__ import annotations
import importlib
import logging
import os
import re
from typing import Dict, List, Optional
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth import train_on_responses_only
import argparse

_hf_datasets = importlib.import_module("datasets")
Dataset = _hf_datasets.Dataset
load_dataset = _hf_datasets.load_dataset
logger = logging.getLogger(__name__)

MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH: int = 1024
LOAD_IN_4BIT: bool = True
LORA_R: int = 32
LORA_ALPHA: int = 64
LORA_DROPOUT: float = 0.05
TARGET_MODULES: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NUM_TRAIN_EPOCHS: int = 3
PER_DEVICE_TRAIN_BATCH_SIZE: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 4
LEARNING_RATE: float = 2e-4
WARMUP_STEPS: int = 50
LOGGING_STEPS: int = 25
SAVE_STEPS: int = 200
EVAL_STEPS: int = 200
OUTPUT_DIR: str = "./outputs/sft_reasoning"
FP16: bool = True

_ANSWER_EXTRACT_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)

def _extract_gsm8k_answer(answer_text: str) -> str:
    match = _ANSWER_EXTRACT_RE.search(answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    lines = [l.strip() for l in answer_text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else answer_text.strip()

def _extract_reasoning_steps(answer_text: str) -> str:
    parts = answer_text.split("####")
    reasoning = parts[0].strip() if parts else answer_text.strip()
    return reasoning

def format_gsm8k_example(example: Dict[str, str]) -> Dict[str, str]:
    question = example["question"].strip()
    raw_answer = example["answer"]
    reasoning = _extract_reasoning_steps(raw_answer)
    final_value = _extract_gsm8k_answer(raw_answer)
    formatted = (
        f"<|im_start|>system\n"
        f"You are a precise mathematical reasoning agent. "
        f"Show your step-by-step reasoning in <think> tags, "
        f"then provide your final answer in <final_answer> tags.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{reasoning}\n</think>\n"
        f"<final_answer>\n{final_value}\n</final_answer>\n"
        f"<|im_end|>"
    )

    return {"text": formatted}

def prepare_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    logger.info("Loading GSM8K split='%s'", split)
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
        logger.info("Truncated to %d samples", len(ds))

    logger.info("Formatting %d examples into structured training prompts", len(ds))
    formatted = ds.map(format_gsm8k_example, remove_columns=ds.column_names)
    return formatted

def train(model_name: str = MODEL_NAME, output_dir: str = OUTPUT_DIR, max_samples: Optional[int] = None, merge_and_save_16bit: bool = False) -> None:
    logger.info("Loading model '%s' in 4-bit precision via Unsloth", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None, 
    )

    logger.info(
        "Attaching LoRA (r=%d, alpha=%d, dropout=%.2f) to modules: %s",
        LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = prepare_dataset(split="train", max_samples=max_samples)
    eval_dataset = prepare_dataset(split="test", max_samples=max_samples)
    logger.info(
        "Datasets ready: %d train, %d eval",
        len(train_dataset), len(eval_dataset),
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=FP16,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,  
    )

    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    logger.info("Enabled train_on_responses_only (masking system/user tokens)")
    logger.info("Starting Stage 1 Reasoning SFT …")
    train_result = trainer.train()
    logger.info("Training complete. Loss: %.4f, Runtime: %.1fs",train_result.training_loss,train_result.metrics.get("train_runtime", 0),)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA adapter saved to '%s'", output_dir)
    if merge_and_save_16bit:
        merged_dir = output_dir + "_merged_16bit"
        logger.info("Merging LoRA weights and saving 16-bit model to '%s'", merged_dir)
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        logger.info("Merged model saved.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Stage 1: Reasoning SFT on GSM8K")
    parser.add_argument("--model", default=MODEL_NAME, help="Base model name or path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--merge-16bit", action="store_true", help="Merge LoRA and save 16-bit")
    args = parser.parse_args()
    train(model_name=args.model, output_dir=args.output_dir, max_samples=args.max_samples, merge_and_save_16bit=args.merge_16bit)