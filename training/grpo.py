from __future__ import annotations
import ast
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import importlib
import argparse
from peft import PeftModel
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTER_PATH: str = "./outputs/sft_combined"
MAX_SEQ_LENGTH: int = 1024
LOAD_IN_4BIT: bool = True
NUM_GENERATIONS: int = 4
PER_DEVICE_TRAIN_BATCH_SIZE: int = 1
GRADIENT_ACCUMULATION_STEPS: int = 4
NUM_TRAIN_EPOCHS: int = 1
LEARNING_RATE: float = 5e-6
MAX_COMPLETION_LENGTH: int = 512
OUTPUT_DIR: str = "./outputs/grpo_lora"
TRAJECTORY_DIR: str = "./datasets/tool_trajectories"

REWARD_FORMAT: float = 0.25
REWARD_TOOL_SYNTAX: float = 0.25
REWARD_TOOL_EXECUTION: float = 1.00
REWARD_VERIFICATION: float = 1.00
REWARD_REFLECTION: float = 1.00
REWARD_CORRECTNESS: float = 1.00

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OBSERVATION_RE = re.compile(r"<observation>\s*(.*?)\s*</observation>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL)
_FORMAT_ORDER_RE = re.compile(
    r"<think>.*?</think>"
    r"(?:\s*<tool_call>.*?</tool_call>\s*"
    r"(?:<observation>.*?</observation>\s*"
    r"<think>.*?</think>)*\s*)?"
    r"<final_answer>.*?</final_answer>",
    re.DOTALL,
)

_GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
_SAFE_TOOLS: Dict[str, Any] = {}

def _get_safe_tools() -> Dict[str, Any]:
    global _SAFE_TOOLS
    if not _SAFE_TOOLS:
        from tools.calculator import run as calc_run
        from tools.sympy_tool import run as sympy_run
        _SAFE_TOOLS = {"calculator": calc_run, "sympy": sympy_run}
    return _SAFE_TOOLS

def _normalize_numeric(value: str) -> Optional[float]:
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None

def _numeric_match(a: str, b: str, epsilon: float = 1e-6) -> bool:
    if a.strip() == b.strip():
        return True
    a_num = _normalize_numeric(a)
    b_num = _normalize_numeric(b)
    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) < epsilon
    return False

def _extract_gsm8k_ground_truth(answer_text: str) -> str:
    match = _GSM8K_ANSWER_RE.search(answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    lines = [l.strip() for l in answer_text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    results = []
    for json_str in _TOOL_CALL_RE.findall(text):
        try:
            payload = json.loads(json_str)
            if isinstance(payload, dict) and "name" in payload:
                results.append(payload)
        except json.JSONDecodeError:
            pass
    return results

def _build_tool_input(arguments: Dict[str, Any]) -> str:
    for key in ("expression", "code", "equation", "query", "input"):
        if key in arguments:
            return str(arguments[key])
    return json.dumps(arguments)

def reward_format_check(completions: List[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for text in completions:
        has_think = bool(_THINK_RE.search(text))
        has_final = bool(_FINAL_ANSWER_RE.search(text))
        correct_order = bool(_FORMAT_ORDER_RE.search(text))
        if correct_order and has_think and has_final:
            rewards.append(REWARD_FORMAT)
        elif has_think and has_final:
            rewards.append(REWARD_FORMAT * 0.5)
        elif has_think or has_final:
            rewards.append(0.0)
        else:
            rewards.append(-0.125)
    return rewards

def reward_tool_syntax(completions: List[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for text in completions:
        tool_matches = _TOOL_CALL_RE.findall(text)
        if not tool_matches:
            rewards.append(0.0)
            continue

        all_valid = True
        for json_str in tool_matches:
            try:
                payload = json.loads(json_str)
                if not isinstance(payload, dict) or "name" not in payload:
                    all_valid = False
                    break
                if "arguments" in payload and not isinstance(payload["arguments"], dict):
                    all_valid = False
                    break
            except json.JSONDecodeError:
                all_valid = False
                break

        rewards.append(REWARD_TOOL_SYNTAX if all_valid else -0.125)
    return rewards

def reward_tool_execution(completions: List[str], **kwargs) -> List[float]:
    safe_tools = _get_safe_tools()
    rewards: List[float] = []
    for text in completions:
        tool_calls = _extract_tool_calls(text)
        observations = _OBSERVATION_RE.findall(text)

        if not tool_calls:
            rewards.append(0.0)
            continue

        matched_count = 0
        total_checked = 0
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})

            if i >= len(observations):
                break

            obs_text = observations[i].strip()
            if tool_name in safe_tools:
                total_checked += 1
                try:
                    tool_input = _build_tool_input(arguments)
                    actual_result = safe_tools[tool_name](tool_input).strip()
                    if _numeric_match(actual_result, obs_text):
                        matched_count += 1
                except Exception:
                    pass  

            elif tool_name == "python_repl":
                total_checked += 1
                code = arguments.get("code", "")
                try:
                    ast.parse(code)
                    matched_count += 0.5 
                except SyntaxError:
                    pass

        if total_checked > 0:
            score = (matched_count / total_checked) * REWARD_TOOL_EXECUTION
            rewards.append(score)
        else:
            rewards.append(0.0)

    return rewards

def reward_verification_match(completions: List[str], **kwargs) -> List[float]:
    safe_tools = _get_safe_tools()
    rewards: List[float] = []
    for text in completions:
        tool_calls = _extract_tool_calls(text)
        observations = _OBSERVATION_RE.findall(text)
        if len(tool_calls) < 2 or len(observations) < 2:
            rewards.append(0.0)
            continue

        executed_results: List[Optional[str]] = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            if tool_name in safe_tools:
                try:
                    tool_input = _build_tool_input(arguments)
                    result = safe_tools[tool_name](tool_input).strip()
                    executed_results.append(result)
                except Exception:
                    executed_results.append(None)
            else:
                executed_results.append(None)

        valid_results = [r for r in executed_results if r is not None]
        if len(valid_results) >= 2:
            first = valid_results[0]
            if all(_numeric_match(first, r) for r in valid_results[1:]):
                rewards.append(REWARD_VERIFICATION)
            else:
                rewards.append(0.0)
        elif len(valid_results) >= 1 and len(observations) >= 2:
            obs_values = [o.strip() for o in observations]
            if len(obs_values) >= 2 and _numeric_match(obs_values[0], obs_values[1]):
                rewards.append(REWARD_VERIFICATION * 0.5) 
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards

def reward_reflection(completions: List[str], **kwargs) -> List[float]:
    _ERROR_INDICATORS = re.compile(
        r"\[(?:CalcError|SymPyError|ExecutionError|Timeout|ToolError)\]|"
        r"(?:SyntaxError|ZeroDivisionError|ValueError|TypeError|Error:)",
        re.IGNORECASE,
    )
    _CORRECTION_KEYWORDS = re.compile(
        r"\b(correct|fix|retry|try again|mistake|wrong|invalid|error)\b",
        re.IGNORECASE,
    )

    rewards: List[float] = []
    for text in completions:
        observations = _OBSERVATION_RE.findall(text)
        think_blocks = _THINK_RE.findall(text)
        tool_calls = _extract_tool_calls(text)
        if len(observations) < 2 or len(tool_calls) < 2:
            rewards.append(0.0)
            continue

        score = 0.0
        for i in range(len(observations) - 1):
            obs_text = observations[i].strip()
            if _ERROR_INDICATORS.search(obs_text):
                subsequent_thinks = think_blocks[i + 1:] if i + 1 < len(think_blocks) else []
                acknowledged = any(
                    _CORRECTION_KEYWORDS.search(t) for t in subsequent_thinks
                )
                subsequent_obs = observations[i + 1:]
                recovered = any(
                    not _ERROR_INDICATORS.search(o) for o in subsequent_obs
                )
                if acknowledged and recovered:
                    score = REWARD_REFLECTION
                    break
                elif recovered:
                    score = REWARD_REFLECTION * 0.5
                    break

        rewards.append(score)
    return rewards

def reward_correctness(completions: List[str], answer: Optional[List[str]] = None, **kwargs) -> List[float]:
    rewards: List[float] = []
    answers = answer or [""] * len(completions)
    for text, gt in zip(completions, answers):
        match = _FINAL_ANSWER_RE.search(text)
        if not match:
            rewards.append(0.0)
            continue

        predicted = match.group(1).strip()
        ground_truth = _extract_gsm8k_ground_truth(gt) if "####" in gt else gt.strip()
        if _numeric_match(predicted, ground_truth):
            rewards.append(REWARD_CORRECTNESS)
        else:
            rewards.append(0.0)

    return rewards

SYSTEM_PROMPT_GRPO = (
    "You are a precise reasoning agent with access to tools. "
    "Show your step-by-step reasoning in <think> tags. "
    "Use tools when needed via <tool_call> tags. "
    "Provide your final answer in <final_answer> tags."
)

def _format_grpo_prompt(question: str, answer: str = "") -> Dict[str, str]:
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT_GRPO}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "answer": answer}

def prepare_grpo_dataset(trajectory_dir: str = TRAJECTORY_DIR, max_gsm8k: int = 1000, max_tool: int = 400, max_verification: int = 300, max_reflection: int = 300, seed: int = 42):
    hf_datasets = importlib.import_module("datasets")
    load_dataset = hf_datasets.load_dataset
    HFDataset = hf_datasets.Dataset
    rng = random.Random(seed)
    parts: List[Dict[str, str]] = []

    logger.info("Loading GSM8K for GRPO prompts")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    indices = rng.sample(range(len(gsm8k)), min(max_gsm8k, len(gsm8k)))
    for idx in indices:
        ex = gsm8k[idx]
        parts.append(_format_grpo_prompt(ex["question"], ex["answer"]))

    tool_files = ["calculator_trajectories.jsonl", "sympy_trajectories.jsonl", "python_trajectories.jsonl", "tool_selection_trajectories.jsonl"]
    tool_questions = _extract_questions_from_trajectories(trajectory_dir, tool_files)
    rng.shuffle(tool_questions)
    for q in tool_questions[:max_tool]:
        parts.append(_format_grpo_prompt(q))

    verif_questions = _extract_questions_from_trajectories(trajectory_dir, ["verification_trajectories.jsonl"])
    rng.shuffle(verif_questions)
    for q in verif_questions[:max_verification]:
        parts.append(_format_grpo_prompt(q))

    reflect_questions = _extract_questions_from_trajectories(trajectory_dir, ["reflection_trajectories.jsonl"])
    rng.shuffle(reflect_questions)
    for q in reflect_questions[:max_reflection]:
        parts.append(_format_grpo_prompt(q))

    rng.shuffle(parts)
    ds = HFDataset.from_list(parts)
    logger.info("GRPO dataset ready: %d prompts", len(ds))
    return ds

def _extract_questions_from_trajectories(trajectory_dir: str, filenames: List[str]) -> List[str]:
    questions: List[str] = []
    user_re = re.compile(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", re.DOTALL)
    for filename in filenames:
        filepath = os.path.join(trajectory_dir, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    text = record.get("text", "")
                    match = user_re.search(text)
                    if match:
                        questions.append(match.group(1).strip())
                except (json.JSONDecodeError, KeyError):
                    pass
    return questions

def train(model_name: str = MODEL_NAME, sft_adapter_path: Optional[str] = SFT_ADAPTER_PATH, trajectory_dir: str = TRAJECTORY_DIR, output_dir: str = OUTPUT_DIR, max_samples: Optional[int] = None) -> None:
    if sft_adapter_path and os.path.exists(sft_adapter_path):
        logger.info("Loading SFT adapter directly for GRPO: '%s'", sft_adapter_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_adapter_path, 
            max_seq_length=MAX_SEQ_LENGTH, 
            load_in_4bit=LOAD_IN_4BIT, 
            dtype=None
        )
        FastLanguageModel.for_training(model) 
    else:
        logger.info("Loading base model '%s' in 4-bit", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=LOAD_IN_4BIT, dtype=None)
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    train_dataset = prepare_grpo_dataset(trajectory_dir=trajectory_dir)
    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100, 
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        max_prompt_length=MAX_SEQ_LENGTH - MAX_COMPLETION_LENGTH,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,  
        train_dataset=train_dataset,
        reward_funcs=[
            reward_format_check,       
            reward_tool_syntax,        
            reward_tool_execution,     
            reward_verification_match, 
            reward_reflection,         
            reward_correctness,        
        ],
    )

    logger.info("Starting GRPO training with 6 reward functions …")
    train_result = trainer.train()
    logger.info("GRPO training complete. Loss: %.4f", train_result.training_loss)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("GRPO adapter saved to '%s'", output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="GRPO RL with execution-aligned rewards")
    parser.add_argument("--model", default=MODEL_NAME, help="Base model name")
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER_PATH, help="SFT adapter path")
    parser.add_argument("--trajectory-dir", default=TRAJECTORY_DIR, help="Trajectory directory")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples")
    args = parser.parse_args()

    train(model_name=args.model, sft_adapter_path=args.sft_adapter, trajectory_dir=args.trajectory_dir, output_dir=args.output_dir, max_samples=args.max_samples)