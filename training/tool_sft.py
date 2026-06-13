from __future__ import annotations
import importlib
import logging
import os
from typing import Dict, List, Optional
import sys
from pathlib import Path
from unsloth import FastLanguageModel, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from peft import PeftModel

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_hf_datasets = importlib.import_module("datasets")
Dataset = _hf_datasets.Dataset
concatenate_datasets = _hf_datasets.concatenate_datasets
load_dataset = _hf_datasets.load_dataset

logger = logging.getLogger(__name__)
MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
REASONING_ADAPTER_PATH: str = "./outputs/sft_reasoning"
MAX_SEQ_LENGTH: int = 1024
LOAD_IN_4BIT: bool = True
LORA_R: int = 32
LORA_ALPHA: int = 64
LORA_DROPOUT: float = 0.05
TARGET_MODULES: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NUM_TRAIN_EPOCHS: int = 2
PER_DEVICE_TRAIN_BATCH_SIZE: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 4
LEARNING_RATE: float = 1e-4
WARMUP_STEPS: int = 100
LOGGING_STEPS: int = 25
SAVE_STEPS: int = 500
EVAL_STEPS: int = 500
OUTPUT_DIR: str = "./outputs/sft_combined"
FP16: bool = True
TRAJECTORY_DIR: str = "./datasets/tool_trajectories"

def _load_jsonl_dataset(path: str) -> Dataset:
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                import json
                record = json.loads(line)
                if "text" in record:
                    records.append({"text": record["text"]})
    return Dataset.from_list(records)

def load_curriculum(trajectory_dir: str = TRAJECTORY_DIR, include_reasoning: bool = True, max_reasoning_samples: Optional[int] = None) -> Dataset:
    parts: List[Dataset] = []
    part_names: List[str] = []
    if include_reasoning:
        from training.sft import prepare_dataset as prepare_gsm8k
        reasoning_ds = prepare_gsm8k(split="train", max_samples=max_reasoning_samples)
        parts.append(reasoning_ds)
        part_names.append(f"reasoning({len(reasoning_ds)})")

    trajectory_files = [
        "calculator_trajectories.jsonl",
        "sympy_trajectories.jsonl",
        "python_trajectories.jsonl",
        "tool_selection_trajectories.jsonl",
        "verification_trajectories.jsonl",
        "reflection_trajectories.jsonl",
    ]

    for filename in trajectory_files:
        filepath = os.path.join(trajectory_dir, filename)
        if os.path.exists(filepath):
            ds = _load_jsonl_dataset(filepath)
            parts.append(ds)
            part_names.append(f"{filename.replace('_trajectories.jsonl', '')}({len(ds)})")
            logger.info("Loaded %s: %d examples", filename, len(ds))
        else:
            logger.warning("Trajectory file not found: %s — skipping", filepath)

    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=42)
    logger.info("Combined curriculum: %d examples [%s]", len(combined), " + ".join(part_names))
    return combined

def train(model_name: str = MODEL_NAME, reasoning_adapter_path: Optional[str] = REASONING_ADAPTER_PATH, trajectory_dir: str = TRAJECTORY_DIR, output_dir: str = OUTPUT_DIR, include_reasoning: bool = True, max_reasoning_samples: Optional[int] = None) -> None:
    logger.info("Loading base model '%s' in 4-bit", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT, dtype=None)

    if reasoning_adapter_path and os.path.exists(reasoning_adapter_path):
        logger.info("Merging Stage 1 reasoning adapter from '%s'", reasoning_adapter_path)
        model = PeftModel.from_pretrained(model, reasoning_adapter_path)
        model = model.merge_and_unload()
        logger.info("Reasoning adapter merged into base weights.")
    else:
        logger.warning("Reasoning adapter not found at '%s'. Training from base model (Stage 1 skipped).", reasoning_adapter_path)

    logger.info("Attaching LoRA (r=%d, alpha=%d, dropout=%.2f)", LORA_R, LORA_ALPHA, LORA_DROPOUT)
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

    train_dataset = load_curriculum(trajectory_dir=trajectory_dir, include_reasoning=include_reasoning, max_reasoning_samples=max_reasoning_samples)

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
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, args=training_args)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    logger.info("Enabled train_on_responses_only")

    logger.info("Starting Combined Curriculum SFT (%d examples) …", len(train_dataset))
    train_result = trainer.train()
    logger.info(
        "Training complete. Loss: %.4f, Runtime: %.1fs",
        train_result.training_loss,
        train_result.metrics.get("train_runtime", 0),
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Combined curriculum adapter saved to '%s'", output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    train(model_name=MODEL_NAME, reasoning_adapter_path=REASONING_ADAPTER_PATH, trajectory_dir=TRAJECTORY_DIR, output_dir=OUTPUT_DIR, include_reasoning=True, max_reasoning_samples=None)