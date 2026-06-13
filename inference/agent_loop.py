from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_OBSERVATION_OPEN = "<observation>"
_OBSERVATION_CLOSE = "</observation>"
_FINAL_ANSWER_OPEN = "<final_answer>"
_FINAL_ANSWER_CLOSE = "</final_answer>"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

class AgentState(Enum):
    REASONING = auto()
    TOOL_CALL = auto()
    OBSERVATION = auto()
    FINAL_ANSWER = auto()
    ERROR = auto()

@dataclass
class StepRecord:
    step_index: int
    state: AgentState
    think: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    raw_output: str = ""
    elapsed_ms: float = 0.0

@dataclass
class AgentResult:
    answer: Optional[str] = None
    steps: List[StepRecord] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    loop_count: int = 0
    terminated_reason: str = ""
    full_trajectory: str = ""

SYSTEM_PROMPT = """\
You are a precise, step-by-step reasoning agent. You must solve problems \
by thinking carefully, using tools when needed, and providing a final answer.

## Available Tools
- **calculator**: Evaluate arithmetic expressions. Input: math expression string.
- **python_repl**: Execute Python code. Input: Python source code string.
- **sympy**: Solve algebraic equations or simplify expressions. Input: equation string or JSON.
- **websearch**: Search the web for factual information, definitions, or current data. Input: search query string.

## Response Format
Always structure your response using these XML tags in order:

<think>
[Your step-by-step reasoning. Break the problem down. Identify what needs calculation.]
</think>

If you need a tool:
<tool_call>
{"name": "tool_name", "arguments": {"expression": "value"}}
</tool_call>

After receiving a tool result in <observation> tags, continue reasoning or provide your final answer.

<final_answer>
[Your final numeric or text answer — just the value, no explanation]
</final_answer>

## Rules
1. Always show your reasoning in <think> tags before making tool calls.
2. Use tools for any non-trivial arithmetic — do NOT attempt mental math for complex calculations.
3. Verify your answer by reviewing the reasoning chain before giving <final_answer>.
4. You may use multiple tool calls (up to 5 rounds).
"""

def _parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse tool_call JSON: %s", exc)
        return None

    tool_name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not tool_name:
        logger.warning("tool_call JSON missing 'name' field: %s", payload)
        return None

    return tool_name, arguments

def _extract_final_answer(text: str) -> Optional[str]:
    match = _FINAL_ANSWER_RE.search(text)
    return match.group(1).strip() if match else None

def _extract_think(text: str) -> Optional[str]:
    matches = _THINK_RE.findall(text)
    return matches[-1].strip() if matches else None

def _build_tool_input(arguments: Dict[str, Any]) -> str:
    for key in ("expression", "code", "equation", "query", "input"):
        if key in arguments:
            return str(arguments[key])

    return json.dumps(arguments)

class AgentLoop:
    def __init__(self, generate_fn, tool_dispatch_fn, max_iterations: int = 5, system_prompt: str = SYSTEM_PROMPT):
        self.generate_fn = generate_fn
        self.tool_dispatch_fn = tool_dispatch_fn
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    def run(self, user_query: str) -> AgentResult:
        result = AgentResult()
        t0 = time.perf_counter()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        trajectory_parts: List[str] = []
        for iteration in range(1, self.max_iterations + 1):
            result.loop_count = iteration
            step = StepRecord(step_index=iteration, state=AgentState.REASONING)
            step_t0 = time.perf_counter()
            logger.info("── Agent loop iteration %d/%d ──", iteration, self.max_iterations)
            try:
                raw_output = self.generate_fn(messages)
            except Exception as exc:
                step.state = AgentState.ERROR
                step.error = f"Generation failed: {exc}"
                step.elapsed_ms = (time.perf_counter() - step_t0) * 1000
                result.steps.append(step)
                result.terminated_reason = "generation_error"
                logger.exception("LLM generation error on iteration %d", iteration)
                break

            step.raw_output = raw_output
            trajectory_parts.append(raw_output)
            step.think = _extract_think(raw_output)
            final = _extract_final_answer(raw_output)
            if final is not None:
                step.state = AgentState.FINAL_ANSWER
                step.final_answer = final
                step.elapsed_ms = (time.perf_counter() - step_t0) * 1000
                result.steps.append(step)
                result.answer = final
                result.terminated_reason = "final_answer"
                logger.info("Final answer extracted: %s", final)
                break

            parsed = _parse_tool_call(raw_output)
            if parsed is not None:
                tool_name, tool_args = parsed
                step.state = AgentState.TOOL_CALL
                step.tool_name = tool_name
                step.tool_args = tool_args
                logger.info("Tool call detected: %s(%s)", tool_name, tool_args)

                tool_input = _build_tool_input(tool_args)
                try:
                    tool_output = self.tool_dispatch_fn(tool_name, tool_input)
                except Exception as exc:
                    tool_output = f"[ToolError] {exc}"
                    logger.exception("Tool dispatch error for '%s'", tool_name)

                step.tool_result = tool_output
                step.elapsed_ms = (time.perf_counter() - step_t0) * 1000
                result.steps.append(step)
                observation_block = f"{_OBSERVATION_OPEN}\n{tool_output}\n{_OBSERVATION_CLOSE}"
                trajectory_parts.append(observation_block)
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": observation_block})
                logger.info("Observation injected (%d chars), continuing loop", len(tool_output))
                continue

            step.state = AgentState.ERROR
            step.error = ("Model output contains neither <tool_call> nor <final_answer>. Attempting recovery by re-prompting.")
            step.elapsed_ms = (time.perf_counter() - step_t0) * 1000
            result.steps.append(step)
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": (
                    "You did not produce a <tool_call> or <final_answer>. "
                    "Please provide your <final_answer> now."
                ),
            })
            logger.warning("No structured output detected — injecting recovery prompt")

        else:
            result.terminated_reason = "max_iterations_exceeded"
            logger.warning("Agent loop exceeded %d iterations", self.max_iterations)

        result.full_trajectory = "\n".join(trajectory_parts)
        result.total_elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

def make_hf_generate_fn(model, tokenizer, max_new_tokens: int = 1024, temperature: float = 0.7, stop_strings: Optional[List[str]] = None):
    import torch
    effective_stop = stop_strings or [_TOOL_CALL_CLOSE, _FINAL_ANSWER_CLOSE]
    def generate_fn(messages: List[Dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop_str in effective_stop:
            idx = decoded.find(stop_str)
            if idx != -1:
                decoded = decoded[: idx + len(stop_str)]
                break

        return decoded

    return generate_fn