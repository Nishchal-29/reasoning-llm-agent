from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import sympy
from sympy import Eq, Symbol, solve, simplify, sympify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

logger = logging.getLogger(__name__)
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_SOLVE_FOR_RE = re.compile(r"solve\s+for\s+([a-zA-Z_]\w*)", re.IGNORECASE)
_EQUATION_RE = re.compile(r"^(.+?)\s*=\s*(.+)$")

def _safe_parse(expr_str: str) -> sympy.Expr:
    return parse_expr(
        expr_str.strip(),
        transformations=_TRANSFORMATIONS,
        evaluate=False,
    )


def _detect_variable(expr: sympy.Expr, hint: Optional[str] = None) -> Symbol:
    if hint:
        return Symbol(hint)

    free = expr.free_symbols if hasattr(expr, "free_symbols") else set()
    if len(free) == 1:
        return free.pop()

    named = {s.name: s for s in free}
    if "x" in named:
        return named["x"]
    if named:
        return named[sorted(named.keys())[0]]

    return Symbol("x")

def solve_equation(equation_str: str, variable_hint: Optional[str] = None) -> str:
    logger.info("Solving equation: '%s' (hint=%s)", equation_str, variable_hint)
    match = _EQUATION_RE.match(equation_str.strip())
    if match:
        lhs = _safe_parse(match.group(1))
        rhs = _safe_parse(match.group(2))
        equation = Eq(lhs, rhs)
        combined = lhs - rhs
    else:
        combined = _safe_parse(equation_str)
        equation = Eq(combined, 0)

    target = _detect_variable(combined, variable_hint)
    solutions = solve(equation, target)
    results = [str(s) for s in solutions]
    logger.info("Solutions for %s: %s", target, results)
    return json.dumps(results)

def simplify_expression(expression_str: str) -> str:
    logger.info("Simplifying: '%s'", expression_str)
    expr = _safe_parse(expression_str)
    simplified = simplify(expr)
    result = str(simplified)
    logger.info("Simplified result: %s", result)
    return result


def verify_equivalence(expr_a: str, expr_b: str) -> bool:
    a = _safe_parse(expr_a)
    b = _safe_parse(expr_b)
    diff = simplify(a - b)
    is_equiv = diff == 0
    logger.info(
        "Equivalence check: '%s' vs '%s' → %s (diff=%s)",
        expr_a, expr_b, is_equiv, diff,
    )
    return is_equiv

def run(input_str: str) -> str:
    input_str = input_str.strip()
    try:
        payload: Dict[str, Any] = json.loads(input_str)
        action = payload.get("action", "solve")

        if action == "solve":
            equation = payload.get("equation", payload.get("expression", ""))
            hint = payload.get("variable", None)
            return solve_equation(equation, variable_hint=hint)
        elif action == "simplify":
            expression = payload.get("expression", "")
            return simplify_expression(expression)
        elif action == "verify":
            return str(verify_equivalence(payload["expr_a"], payload["expr_b"]))
        else:
            return f"[SymPyError] Unknown action '{action}'. Use: solve, simplify, verify."
    except (json.JSONDecodeError, KeyError):
        pass  

    try:
        var_match = _SOLVE_FOR_RE.search(input_str)
        variable_hint = var_match.group(1) if var_match else None
        clean = _SOLVE_FOR_RE.sub("", input_str).strip().rstrip(".")

        if "=" in clean:
            return solve_equation(clean, variable_hint=variable_hint)
        else:
            return simplify_expression(clean)
    except Exception as exc:
        error_msg = f"[SymPyError] {exc}"
        logger.warning(error_msg)
        return error_msg