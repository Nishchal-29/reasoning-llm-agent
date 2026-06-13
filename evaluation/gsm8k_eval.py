from __future__ import annotations
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib
from unsloth import FastLanguageModel

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.agent_loop import AgentLoop, make_hf_generate_fn, SYSTEM_PROMPT
from tools import dispatch

_hf_datasets = importlib.import_module("datasets")
load_dataset = _hf_datasets.load_dataset
logger = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OBSERVATION_RE = re.compile(r"<observation>\s*(.*?)\s*</observation>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)
_ERROR_INDICATORS = re.compile(
    r"\[(?:CalcError|SymPyError|ExecutionError|Timeout|ToolError)\]|"
    r"(?:SyntaxError|ZeroDivisionError|ValueError|TypeError)",
    re.IGNORECASE,
)
_SAFE_TOOLS: Dict[str, Any] = {}

def _get_safe_tools() -> Dict[str, Any]:
    global _SAFE_TOOLS
    if not _SAFE_TOOLS:
        from tools.calculator import run as calc_run
        from tools.sympy_tool import run as sympy_run
        _SAFE_TOOLS = {"calculator": calc_run, "sympy": sympy_run}
    return _SAFE_TOOLS

def _extract_ground_truth(answer_text: str) -> str:
    match = _GSM8K_ANSWER_RE.search(answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    lines = [l.strip() for l in answer_text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else answer_text.strip()

def _normalize_numeric(value: str) -> Optional[float]:
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
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

def _build_tool_input(arguments: Dict[str, Any]) -> str:
    for key in ("expression", "code", "equation", "query", "input"):
        if key in arguments:
            return str(arguments[key])
    return json.dumps(arguments)

def _infer_expected_tool(question: str) -> Optional[str]:
    q_lower = question.lower()
    if any(kw in q_lower for kw in ["solve for", "solve the equation", "find x", "find y"]):
        return "sympy"
    if re.search(r"\b\d+\s*\*?\s*[xynt]\s*[\+\-]", q_lower):
        return "sympy"

    if any(kw in q_lower for kw in ["prime", "fibonacci", "factorial", "sum of", "all integers"]):
        return "python_repl"
    if any(kw in q_lower for kw in ["factors of", "convert", "binary", "digits of"]):
        return "python_repl"

    if re.search(r"\b\d+\s*[\*\+\-/]\s*\d+", question):
        return "calculator"
    if any(kw in q_lower for kw in ["calculate", "compute", "what is", "how much"]):
        return "calculator"

    return None

@dataclass
class ExampleResult:
    index: int
    question: str
    ground_truth: str
    predicted: Optional[str]
    exact_match: bool = False
    numeric_match: bool = False
    num_tool_calls: int = 0
    num_think_steps: int = 0
    had_error: bool = False
    elapsed_ms: float = 0.0
    raw_output: str = ""
    tool_calls_successful: int = 0
    tool_calls_total: int = 0
    uses_multiple_tools: bool = False
    tool_errors_total: int = 0
    tool_errors_recovered: int = 0
    expected_tool: Optional[str] = None
    chosen_tool: Optional[str] = None
    tool_selection_correct: Optional[bool] = None

@dataclass
class EvalReport:
    total: int = 0
    exact_matches: int = 0
    numeric_matches: int = 0
    exact_accuracy: float = 0.0
    numeric_accuracy: float = 0.0
    avg_tool_calls: float = 0.0
    avg_think_steps: float = 0.0
    tool_usage_rate: float = 0.0
    error_rate: float = 0.0
    total_elapsed_s: float = 0.0
    tool_correctness_rate: float = 0.0
    verification_rate: float = 0.0
    reflection_success_rate: float = 0.0
    tool_selection_accuracy: float = 0.0
    results: List[ExampleResult] = field(default_factory=list)

def evaluate_single(question: str, ground_truth_answer: str, model_output: str, index: int = 0) -> ExampleResult:
    safe_tools = _get_safe_tools()
    gt = _extract_ground_truth(ground_truth_answer)
    result = ExampleResult(index=index, question=question, ground_truth=gt, predicted=None, raw_output=model_output)

    result.num_think_steps = len(_THINK_RE.findall(model_output))
    tool_call_jsons = _TOOL_CALL_RE.findall(model_output)
    observations = _OBSERVATION_RE.findall(model_output)
    result.num_tool_calls = len(tool_call_jsons)
    tool_names_used: List[str] = []
    error_obs_indices: List[int] = []
    for i, json_str in enumerate(tool_call_jsons):
        result.tool_calls_total += 1
        try:
            payload = json.loads(json_str)
            tool_name = payload.get("name", "")
            arguments = payload.get("arguments", {})
            tool_names_used.append(tool_name)

            if i < len(observations):
                obs_text = observations[i].strip()
                if _ERROR_INDICATORS.search(obs_text):
                    result.tool_errors_total += 1
                    error_obs_indices.append(i)

            if tool_name in safe_tools and i < len(observations):
                obs_text = observations[i].strip()
                try:
                    tool_input = _build_tool_input(arguments)
                    actual = safe_tools[tool_name](tool_input).strip()
                    if _numeric_match(actual, obs_text):
                        result.tool_calls_successful += 1
                except Exception:
                    pass
            elif tool_name not in safe_tools and i < len(observations):
                obs_text = observations[i].strip()
                if not _ERROR_INDICATORS.search(obs_text) and obs_text != "(no output)":
                    result.tool_calls_successful += 1

        except json.JSONDecodeError:
            pass

    result.uses_multiple_tools = len(set(tool_names_used)) >= 2 or len(tool_names_used) >= 2
    for err_idx in error_obs_indices:
        subsequent_obs = observations[err_idx + 1:]
        if any(not _ERROR_INDICATORS.search(o) for o in subsequent_obs):
            result.tool_errors_recovered += 1

    result.expected_tool = _infer_expected_tool(question)
    result.chosen_tool = tool_names_used[0] if tool_names_used else None
    if result.expected_tool is not None and result.chosen_tool is not None:
        result.tool_selection_correct = (result.chosen_tool == result.expected_tool)

    match = _FINAL_ANSWER_RE.search(model_output)
    if match:
        result.predicted = match.group(1).strip()
    else:
        result.had_error = True
        return result

    if result.predicted == gt:
        result.exact_match = True
        result.numeric_match = True
        return result

    if _numeric_match(result.predicted, gt):
        result.numeric_match = True

    return result

def evaluate_batch(questions: List[str], answers: List[str], model_outputs: List[str]) -> EvalReport:
    report = EvalReport(total=len(questions))
    for i, (q, a, out) in enumerate(zip(questions, answers, model_outputs)):
        result = evaluate_single(q, a, out, index=i)
        report.results.append(result)

        if result.exact_match:
            report.exact_matches += 1
        if result.numeric_match:
            report.numeric_matches += 1

    if report.total > 0:
        report.exact_accuracy = report.exact_matches / report.total
        report.numeric_accuracy = report.numeric_matches / report.total

        total_tools = sum(r.num_tool_calls for r in report.results)
        total_thinks = sum(r.num_think_steps for r in report.results)
        total_errors = sum(1 for r in report.results if r.had_error)

        report.avg_tool_calls = total_tools / report.total
        report.avg_think_steps = total_thinks / report.total
        report.tool_usage_rate = sum(1 for r in report.results if r.num_tool_calls > 0) / report.total
        report.error_rate = total_errors / report.total

        total_tc = sum(r.tool_calls_total for r in report.results)
        total_tc_success = sum(r.tool_calls_successful for r in report.results)
        report.tool_correctness_rate = total_tc_success / total_tc if total_tc > 0 else 0.0
        report.verification_rate = sum(1 for r in report.results if r.uses_multiple_tools) / report.total

        total_tool_errors = sum(r.tool_errors_total for r in report.results)
        total_recovered = sum(r.tool_errors_recovered for r in report.results)
        report.reflection_success_rate = (total_recovered / total_tool_errors if total_tool_errors > 0 else 0.0)
        selection_evaluated = [r for r in report.results if r.tool_selection_correct is not None]
        if selection_evaluated:
            report.tool_selection_accuracy = sum(1 for r in selection_evaluated if r.tool_selection_correct) / len(selection_evaluated)

    return report

def run_evaluation(generate_fn, max_samples: int = 100, output_path: Optional[str] = None) -> EvalReport:
    logger.info("Loading GSM8K test split (max_samples=%d)", max_samples)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(max_samples, len(ds))))
    questions: List[str] = []
    answers: List[str] = []
    outputs: List[str] = []

    t0 = time.perf_counter()
    for i, example in enumerate(ds):
        question = example["question"]
        answer = example["answer"]
        questions.append(question)
        answers.append(answer)

        logger.info("Evaluating example %d/%d", i + 1, len(ds))
        try:
            output = generate_fn(question)
        except Exception as exc:
            logger.exception("Generation failed for example %d", i)
            output = f"[GenerationError] {exc}"
        outputs.append(output)

    elapsed = time.perf_counter() - t0
    report = evaluate_batch(questions, answers, outputs)
    report.total_elapsed_s = elapsed
    logger.info(
        "Evaluation complete: exact=%.2f%%, numeric=%.2f%%, "
        "tool_correct=%.2f%%, verification=%.2f%%, "
        "reflection=%.2f%%, tool_selection=%.2f%%",
        report.exact_accuracy * 100,
        report.numeric_accuracy * 100,
        report.tool_correctness_rate * 100,
        report.verification_rate * 100,
        report.reflection_success_rate * 100,
        report.tool_selection_accuracy * 100,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        report_dict: Dict[str, Any] = {
            "total": report.total,
            "exact_matches": report.exact_matches,
            "numeric_matches": report.numeric_matches,
            "exact_accuracy": round(report.exact_accuracy, 4),
            "numeric_accuracy": round(report.numeric_accuracy, 4),
            "avg_tool_calls": round(report.avg_tool_calls, 2),
            "avg_think_steps": round(report.avg_think_steps, 2),
            "tool_usage_rate": round(report.tool_usage_rate, 4),
            "error_rate": round(report.error_rate, 4),
            "tool_correctness_rate": round(report.tool_correctness_rate, 4),
            "verification_rate": round(report.verification_rate, 4),
            "reflection_success_rate": round(report.reflection_success_rate, 4),
            "tool_selection_accuracy": round(report.tool_selection_accuracy, 4),
            "total_elapsed_seconds": round(report.total_elapsed_s, 2),
            "per_example": [
                {
                    "index": r.index,
                    "question": r.question[:100] + "…" if len(r.question) > 100 else r.question,
                    "ground_truth": r.ground_truth,
                    "predicted": r.predicted,
                    "exact_match": r.exact_match,
                    "numeric_match": r.numeric_match,
                    "num_tool_calls": r.num_tool_calls,
                    "num_think_steps": r.num_think_steps,
                    "had_error": r.had_error,
                    "tool_calls_successful": r.tool_calls_successful,
                    "tool_calls_total": r.tool_calls_total,
                    "uses_multiple_tools": r.uses_multiple_tools,
                    "tool_errors_total": r.tool_errors_total,
                    "tool_errors_recovered": r.tool_errors_recovered,
                    "expected_tool": r.expected_tool,
                    "chosen_tool": r.chosen_tool,
                    "tool_selection_correct": r.tool_selection_correct,
                }
                for r in report.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to '%s'", output_path)

    return report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./outputs/grpo_lora",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    gen_fn = make_hf_generate_fn(model, tokenizer)
    agent = AgentLoop(generate_fn=gen_fn, tool_dispatch_fn=dispatch)
    def evaluate_question(question: str) -> str:
        result = agent.run(question)
        return result.full_trajectory

    report = run_evaluation(generate_fn=evaluate_question, max_samples=100, output_path="./outputs/eval_report.json")

    print(f"Exact Accuracy: {report.exact_accuracy:.2%}")
    print(f"Numeric Accuracy: {report.numeric_accuracy:.2%}")
    print(f"Tool Usage Rate: {report.tool_usage_rate:.2%}")
    print(f"Tool Correctness: {report.tool_correctness_rate:.2%}")
    print(f"Verification Rate: {report.verification_rate:.2%}")
    print(f"Reflection Success: {report.reflection_success_rate:.2%}")
    print(f"Tool Selection Acc: {report.tool_selection_accuracy:.2%}")
    print(f"Error Rate: {report.error_rate:.2%}")
    print(f"Avg Tool Calls: {report.avg_tool_calls:.1f}")
    print(f"Avg Think Steps: {report.avg_think_steps:.1f}")