from __future__ import annotations
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.calculator import run as calc_run
from tools.python_repl import run as python_run
from tools.sympy_tool import run as sympy_run

SYSTEM_MESSAGE = (
    "You are a precise reasoning agent with access to tools. "
    "Show your step-by-step reasoning in <think> tags. "
    "Use tools when needed via <tool_call> tags. "
    "Provide your final answer in <final_answer> tags."
)

def _wrap_trajectory(question: str, assistant_content: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}\n<|im_end|>"
    )

_CALC_TEMPLATES = [
    ("What is {a} * {b}?", "{a} * {b}"),
    ("Calculate {a} + {b} * {c}.", "{a} + {b} * {c}"),
    ("What is ({a} + {b}) * {c}?", "({a} + {b}) * {c}"),
    ("Compute {a} - {b} + {c}.", "{a} - {b} + {c}"),
    ("What is {a} * {b} + {c} * {d}?", "{a} * {b} + {c} * {d}"),
    ("Calculate {a} ** {e}.", "{a} ** {e}"),
    ("What is {a} * {b} - {c}?", "{a} * {b} - {c}"),
    ("Compute ({a} - {b}) * ({c} + {d}).", "({a} - {b}) * ({c} + {d})"),
    ("What is {a} / {b}?", "{a} / {b}"),
    ("Calculate {a} * {b} / {c}.", "{a} * {b} / {c}"),
]

def generate_calculator_dataset(count: int = 3000, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    rng = rng or random.Random(42)
    results: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        c = rng.randint(2, 99)
        d = rng.randint(2, 99)
        e = rng.randint(2, 5)

        template_q, template_expr = rng.choice(_CALC_TEMPLATES)
        question = template_q.format(a=a, b=b, c=c, d=d, e=e)
        expression = template_expr.format(a=a, b=b, c=c, d=d, e=e)
        observation = calc_run(expression)
        if observation.startswith("[CalcError]"):
            continue
        final_answer = observation.rstrip("0").rstrip(".") if "." in observation else observation

        assistant = (
            f"<think>\nThis requires numerical computation.\n"
            f"I need to calculate: {expression}\n"
            f"This is a large or multi-step calculation, so I should use the calculator.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "calculator", "arguments": {{"expression": "{expression}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{observation}\n</observation>\n"
            f"<think>\nThe calculator returned {observation}.\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})

    logger.info("Generated %d calculator trajectories", len(results))
    return results

def _make_linear_equation(rng: random.Random) -> Tuple[str, str, str]:
    a = rng.randint(2, 15)
    b = rng.randint(1, 30)
    c = rng.randint(1, 100)
    var = rng.choice(["x", "y", "n", "t"])
    question = f"Solve for {var}: {a}*{var} + {b} = {c}"
    equation = f"{a}*{var} + {b} = {c}"
    return question, equation, var

def _make_quadratic_equation(rng: random.Random) -> Tuple[str, str, str]:
    r1 = rng.randint(-10, 10)
    r2 = rng.randint(-10, 10)
    b = -(r1 + r2)
    c = r1 * r2
    var = rng.choice(["x", "y"])
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    equation = f"{var}**2 {b_str}*{var} {c_str} = 0"
    question = f"Solve the equation: {var}² {b_str}{var} {c_str} = 0"
    return question, equation, var


def generate_sympy_dataset(count: int = 2000, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    rng = rng or random.Random(43)
    results: List[Dict[str, str]] = []

    for i in range(count):
        if rng.random() < 0.6:
            question, equation, var = _make_linear_equation(rng)
        else:
            question, equation, var = _make_quadratic_equation(rng)

        sympy_input = json.dumps({"action": "solve", "equation": equation, "variable": var})
        observation = sympy_run(sympy_input)
        if observation.startswith("[SymPyError]"):
            continue

        try:
            solutions = json.loads(observation)
            if isinstance(solutions, list) and solutions:
                final_answer = ", ".join(str(s) for s in solutions)
            else:
                final_answer = observation
        except json.JSONDecodeError:
            final_answer = observation

        assistant = (
            f"<think>\nThis requires solving an algebraic equation.\n"
            f"The equation is: {equation}\n"
            f"I need to solve for {var}. Let me use sympy.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "sympy", "arguments": {{"equation": "{equation}", "variable": "{var}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{observation}\n</observation>\n"
            f"<think>\nSymPy gives the solution: {observation}.\n"
            f"The answer is {final_answer}.\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})
    logger.info("Generated %d sympy trajectories", len(results))
    return results

_PYTHON_PROBLEMS = [
    ("What is the sum of all integers from 1 to {n}?", "print(sum(range(1, {n}+1)))"),
    ("How many even numbers are between 1 and {n}?", "print(len([x for x in range(1, {n}+1) if x % 2 == 0]))"),
    ("What is {n} factorial?", "import math; print(math.factorial({n}))"),
    ("What is the sum of squares from 1 to {n}?", "print(sum(x**2 for x in range(1, {n}+1)))"),
    ("How many prime numbers are there below {n}?",
     "primes = [x for x in range(2, {n}) if all(x%i!=0 for i in range(2, int(x**0.5)+1))]; print(len(primes))"),
    ("What is the greatest common divisor of {a} and {b}?", "import math; print(math.gcd({a}, {b}))"),
    ("What is the least common multiple of {a} and {b}?", "import math; print(math.lcm({a}, {b}))"),
    ("What is the {n}th Fibonacci number?",
     "a,b=0,1\nfor _ in range({n}-1): a,b=b,a+b\nprint(b)"),
    ("What is the sum of digits of {big}?", "print(sum(int(d) for d in str({big})))"),
    ("Convert {n} from decimal to binary.", "print(bin({n}))"),
]

def generate_python_dataset(count: int = 2000, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    rng = rng or random.Random(44)
    results: List[Dict[str, str]] = []
    for _ in range(count):
        template_q, template_code = rng.choice(_PYTHON_PROBLEMS)
        n = rng.randint(5, 50)
        a = rng.randint(10, 500)
        b = rng.randint(10, 500)
        big = rng.randint(10000, 999999)

        question = template_q.format(n=n, a=a, b=b, big=big)
        code = template_code.format(n=n, a=a, b=b, big=big)

        observation = python_run(code, timeout=5)
        if "[ExecutionError]" in observation or "[Timeout]" in observation:
            continue

        final_answer = observation.strip()
        if not final_answer or final_answer == "(no output)":
            continue
        code_escaped = code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        assistant = (
            f"<think>\nThis problem requires computation that's best done with Python.\n"
            f"I need to write code to solve: {question}\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "python_repl", "arguments": {{"code": "{code_escaped}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{observation}\n</observation>\n"
            f"<think>\nThe Python code returned: {final_answer}\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})
    logger.info("Generated %d python trajectories", len(results))
    return results

_TOOL_SELECTION_EXAMPLES = [
    ("What is {a} * {b}?", "calculator", "This is straightforward arithmetic."),
    ("Calculate ({a} + {b}) * {c}.", "calculator", "This is a multi-step arithmetic problem."),
    ("Solve for x: {a}*x + {b} = {c}", "sympy", "This requires solving an algebraic equation."),
    ("Solve the equation: x**2 - {a} = 0", "sympy", "This is a quadratic equation."),
    ("Sum all integers from 1 to {n}.", "python_repl", "This involves iteration, best done in Python."),
    ("How many prime numbers are below {n}?", "python_repl", "Primality testing requires code."),
    ("Find all factors of {big}.", "python_repl", "Factorization requires iteration."),
    ("What is {n} factorial?", "python_repl", "Factorial of large numbers needs code."),
    ("Simplify (x+{a})*(x-{a}).", "sympy", "This is algebraic simplification."),
    ("What is {a}**{e}?", "calculator", "This is exponentiation — arithmetic."),
]

def generate_tool_selection_dataset(count: int = 1000, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    rng = rng or random.Random(45)
    results: List[Dict[str, str]] = []
    for _ in range(count):
        a = rng.randint(10, 200)
        b = rng.randint(10, 200)
        c = rng.randint(50, 500)
        n = rng.randint(10, 80)
        e = rng.randint(2, 5)
        big = rng.randint(100, 9999)

        template = rng.choice(_TOOL_SELECTION_EXAMPLES)
        question_tmpl, tool_name, reasoning_hint = template
        question = question_tmpl.format(a=a, b=b, c=c, n=n, e=e, big=big)
        if tool_name == "calculator":
            if "*" in question:
                expr = f"{a} * {b}"
            elif "**" in question:
                expr = f"{a} ** {e}"
            else:
                expr = f"({a} + {b}) * {c}"
            observation = calc_run(expr)
            if observation.startswith("[CalcError]"):
                continue
            tool_call_json = f'{{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}'

        elif tool_name == "sympy":
            if "Simplify" in question:
                sympy_input = json.dumps({"action": "simplify", "expression": f"(x+{a})*(x-{a})"})
            elif "x**2" in question:
                sympy_input = json.dumps({"action": "solve", "equation": f"x**2 - {a} = 0"})
            else:
                sympy_input = json.dumps({"action": "solve", "equation": f"{a}*x + {b} = {c}"})
            observation = sympy_run(sympy_input)
            if observation.startswith("[SymPyError]"):
                continue
            tool_call_json = f'{{"name": "sympy", "arguments": {{"equation": "{a}*x + {b} = {c}"}}}}'

        elif tool_name == "python_repl":
            if "prime" in question.lower():
                code = f"primes = [x for x in range(2, {n}) if all(x%i!=0 for i in range(2, int(x**0.5)+1))]; print(len(primes))"
            elif "factorial" in question.lower():
                code = f"import math; print(math.factorial({n}))"
            elif "factors" in question.lower():
                code = f"print([i for i in range(1, {big}+1) if {big} % i == 0])"
            else:
                code = f"print(sum(range(1, {n}+1)))"
            observation = python_run(code, timeout=5)
            if "[ExecutionError]" in observation or "[Timeout]" in observation:
                continue
            code_escaped = code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            tool_call_json = f'{{"name": "python_repl", "arguments": {{"code": "{code_escaped}"}}}}'
        else:
            continue

        final_answer = observation.strip()
        if not final_answer or final_answer == "(no output)":
            continue

        assistant = (
            f"<think>\n{reasoning_hint}\n"
            f"I should use the {tool_name} tool for this.\n</think>\n"
            f"<tool_call>\n{tool_call_json}\n</tool_call>\n"
            f"<observation>\n{observation}\n</observation>\n"
            f"<think>\nThe {tool_name} returned: {final_answer}\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})
    logger.info("Generated %d tool selection trajectories", len(results))
    return results

def generate_verification_dataset(count: int = 2000, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    rng = rng or random.Random(46)
    results: List[Dict[str, str]] = []

    for _ in range(count):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        op = rng.choice(["*", "+", "-"])
        expression = f"{a} {op} {b}"
        question = f"What is {expression}?"
        calc_obs = calc_run(expression)
        if calc_obs.startswith("[CalcError]"):
            continue
        python_code = f"print({expression})"
        python_obs = python_run(python_code, timeout=5)
        if "[ExecutionError]" in python_obs or "[Timeout]" in python_obs:
            continue

        python_obs_clean = python_obs.strip()
        code_escaped = python_code.replace('"', '\\"')

        try:
            calc_val = float(calc_obs.strip())
            py_val = float(python_obs_clean)
            values_agree = abs(calc_val - py_val) < 1e-6
        except (ValueError, TypeError):
            values_agree = calc_obs.strip() == python_obs_clean

        if not values_agree:
            continue  

        final_answer = calc_obs.rstrip("0").rstrip(".") if "." in calc_obs else calc_obs
        assistant = (
            f"<think>\nI need to compute {expression}.\n"
            f"Let me use the calculator first.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "calculator", "arguments": {{"expression": "{expression}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{calc_obs}\n</observation>\n"
            f"<think>\nThe calculator returned {calc_obs}.\n"
            f"Let me verify this with Python to make sure.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "python_repl", "arguments": {{"code": "{code_escaped}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{python_obs}\n</observation>\n"
            f"<think>\nBoth tools agree: calculator returned {calc_obs} "
            f"and Python returned {python_obs_clean}.\n"
            f"I can confidently report the answer.\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})
    logger.info("Generated %d verification trajectories", len(results))
    return results

def generate_reflection_dataset(count: int = 300, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    rng = rng or random.Random(47)
    results: List[Dict[str, str]] = []
    _MALFORMED_EXPRESSIONS = [
        ("{a} ** ", "{a} ** {e}", "What is {a} to the power of {e}?"),
        ("{a} * * {b}", "{a} * {b}", "What is {a} times {b}?"),
        ("({a} + {b} * {c}", "({a} + {b}) * {c}", "What is ({a} + {b}) times {c}?"),
        ("{a} / 0", "{a} / {b}", "What is {a} divided by {b}?"),
        ("{a} +* {b}", "{a} + {b}", "What is {a} plus {b}?"),
    ]

    pattern_a_count = int(count * 0.6)
    pattern_b_count = count - pattern_a_count
    for _ in range(pattern_a_count):
        a = rng.randint(10, 500)
        b = rng.randint(2, 500)
        c = rng.randint(2, 50)
        e = rng.randint(2, 4)

        broken_tmpl, fixed_tmpl, question_tmpl = rng.choice(_MALFORMED_EXPRESSIONS)
        broken_expr = broken_tmpl.format(a=a, b=b, c=c, e=e)
        fixed_expr = fixed_tmpl.format(a=a, b=b, c=c, e=e)
        question = question_tmpl.format(a=a, b=b, c=c, e=e)
        error_obs = calc_run(broken_expr)
        if not error_obs.startswith("[CalcError]"):
            continue  

        success_obs = calc_run(fixed_expr)
        if success_obs.startswith("[CalcError]"):
            continue

        final_answer = success_obs.rstrip("0").rstrip(".") if "." in success_obs else success_obs
        assistant = (
            f"<think>\nI need to calculate: {question}\n"
            f"Let me use the calculator.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "calculator", "arguments": {{"expression": "{broken_expr}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{error_obs}\n</observation>\n"
            f"<think>\nThe expression caused an error: {error_obs}\n"
            f"I made a syntax mistake. The correct expression should be: {fixed_expr}\n"
            f"Let me try again.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "calculator", "arguments": {{"expression": "{fixed_expr}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{success_obs}\n</observation>\n"
            f"<think>\nThe corrected expression gives {success_obs}.\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})

    for _ in range(pattern_b_count):
        a = rng.randint(10, 200)
        b = rng.randint(10, 200)
        expression = f"{a} * {b}"
        question = f"What is {a} times {b}?"
        correct_result = calc_run(expression)
        if correct_result.startswith("[CalcError]"):
            continue

        wrong_value = str(int(float(correct_result)) + rng.choice([-7, -3, 5, 11, -13, 8]))
        py_code = f"print({a} * {b})"
        py_obs = python_run(py_code, timeout=5)
        if "[ExecutionError]" in py_obs or "[Timeout]" in py_obs:
            continue

        code_escaped = py_code.replace('"', '\\"')
        final_answer = correct_result.rstrip("0").rstrip(".") if "." in correct_result else correct_result

        assistant = (
            f"<think>\nI need to compute {a} × {b}.\n"
            f"Let me try: I think it's roughly {wrong_value}.\n"
            f"Actually, let me verify with the calculator to be sure.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "calculator", "arguments": {{"expression": "{expression}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{correct_result}\n</observation>\n"
            f"<think>\nThe calculator says {correct_result}, not {wrong_value} as I initially thought.\n"
            f"Let me double-check with Python.\n</think>\n"
            f"<tool_call>\n"
            f'{{"name": "python_repl", "arguments": {{"code": "{code_escaped}"}}}}\n'
            f"</tool_call>\n"
            f"<observation>\n{py_obs}\n</observation>\n"
            f"<think>\nBoth tools confirm the result is {correct_result}.\n"
            f"My initial mental math was wrong. The correct answer is {final_answer}.\n</think>\n"
            f"<final_answer>\n{final_answer}\n</final_answer>"
        )

        results.append({"text": _wrap_trajectory(question, assistant)})
    logger.info("Generated %d reflection trajectories", len(results))
    return results

DEFAULT_COUNTS = {
    "calculator": 3000,
    "sympy": 2000,
    "python": 2000,
    "tool_selection": 1000,
    "verification": 2000,
    "reflection": 300,
}

def generate_all(output_dir: str = "./datasets/tool_trajectories", counts: Optional[Dict[str, int]] = None, seed: int = 42) -> Dict[str, int]:
    effective_counts = {**DEFAULT_COUNTS, **(counts or {})}
    os.makedirs(output_dir, exist_ok=True)
    generators = {
        "calculator": (generate_calculator_dataset, effective_counts["calculator"], random.Random(seed)),
        "sympy": (generate_sympy_dataset, effective_counts["sympy"], random.Random(seed + 1)),
        "python": (generate_python_dataset, effective_counts["python"], random.Random(seed + 2)),
        "tool_selection": (generate_tool_selection_dataset, effective_counts["tool_selection"], random.Random(seed + 3)),
        "verification": (generate_verification_dataset, effective_counts["verification"], random.Random(seed + 4)),
        "reflection": (generate_reflection_dataset, effective_counts["reflection"], random.Random(seed + 5)),
    }

    actual_counts: Dict[str, int] = {}
    for name, (gen_fn, count, rng) in generators.items():
        logger.info("Generating '%s' dataset (target=%d) …", name, count)
        data = gen_fn(count=count, rng=rng)
        actual_counts[name] = len(data)
        output_path = os.path.join(output_dir, f"{name}_trajectories.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, record in enumerate(data):
                record["id"] = f"{name}_{i}"
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("  → %s: %d examples → %s", name, len(data), output_path)

    combined_path = os.path.join(output_dir, "combined_tool_trajectories.jsonl")
    combined_count = 0
    with open(combined_path, "w", encoding="utf-8") as f:
        for name in generators:
            part_path = os.path.join(output_dir, f"{name}_trajectories.jsonl")
            if os.path.exists(part_path):
                with open(part_path, "r", encoding="utf-8") as part_f:
                    for line in part_f:
                        f.write(line)
                        combined_count += 1

    logger.info("Combined dataset: %d examples → %s", combined_count, combined_path)
    actual_counts["combined"] = combined_count
    return actual_counts

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Generate live-execution ReAct trajectories")
    parser.add_argument("--output-dir", default="./datasets/tool_trajectories", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calculator", type=int, default=3000)
    parser.add_argument("--sympy", type=int, default=2000)
    parser.add_argument("--python", type=int, default=2000)
    parser.add_argument("--tool-selection", type=int, default=1000)
    parser.add_argument("--verification", type=int, default=2000)
    parser.add_argument("--reflection", type=int, default=300)
    args = parser.parse_args()

    counts = {
        "calculator": args.calculator,
        "sympy": args.sympy,
        "python": args.python,
        "tool_selection": args.tool_selection,
        "verification": args.verification,
        "reflection": args.reflection,
    }

    result = generate_all(output_dir=args.output_dir, counts=counts, seed=args.seed)
    for name, cnt in result.items():
        print(f"  {name}: {cnt}")