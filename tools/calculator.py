from __future__ import annotations
import ast
import logging
import operator
from typing import Union

logger = logging.getLogger(__name__)

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_MAGNITUDE: float = 1e15
NumericResult = Union[int, float]

class _SafeEvaluator(ast.NodeVisitor):
    def visit_Constant(self, node: ast.Constant) -> NumericResult:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_Num(self, node: ast.Num) -> NumericResult: 
        return node.n  # type: ignore[attr-defined]

    def visit_BinOp(self, node: ast.BinOp) -> NumericResult:
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 100:
                raise ValueError(
                    f"Exponent too large ({right}); max allowed is ±100"
                )

        result = op_func(left, right)
        self._check_magnitude(result)
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> NumericResult:
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = self.visit(node.operand)
        return op_func(operand)

    def visit_Expression(self, node: ast.Expression) -> NumericResult:
        return self.visit(node.body)

    def generic_visit(self, node: ast.AST) -> NumericResult:
        raise ValueError(
            f"Disallowed AST node: {type(node).__name__}. "
            "Only numeric literals and arithmetic operators are permitted."
        )

    @staticmethod
    def _check_magnitude(value: NumericResult) -> None:
        if isinstance(value, float) and abs(value) > _MAX_MAGNITUDE:
            raise ValueError(
                f"Result magnitude ({abs(value):.2e}) exceeds safety "
                f"ceiling ({_MAX_MAGNITUDE:.2e})"
            )

def safe_eval(expression: str) -> NumericResult:
    tree = ast.parse(expression.strip(), mode="eval")
    return _SafeEvaluator().visit(tree)

def run(input_str: str) -> str:
    logger.info("Calculator invoked with: %s", input_str.strip())
    try:
        result = safe_eval(input_str)
        output = str(result)
        logger.info("Calculator result: %s", output)
        return output
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        error_msg = f"[CalcError] {exc}"
        logger.warning(error_msg)
        return error_msg