"""
Reasoning Agent — Tool Registry
================================
Clean, importable exports for all agent-callable tools.
Each tool exposes a unified ``run(input_str) -> str`` interface so the
agentic loop can route dynamically without per-tool boilerplate.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

from tools.calculator import run as calculator_run
from tools.python_repl import run as python_repl_run
from tools.sympy_tool import run as sympy_run
from tools.websearch import run as websearch_run

logger = logging.getLogger(__name__)

# ── Canonical tool registry ────────────────────────────────────────────
# Maps the exact ``name`` string models emit inside <tool_call> payloads
# to the corresponding execution function.

TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {
    "calculator": calculator_run,
    "python_repl": python_repl_run,
    "sympy": sympy_run,
    "websearch": websearch_run,
}


def list_tools() -> list[str]:
    """Return sorted names of all registered tools."""
    return sorted(TOOL_REGISTRY.keys())


def dispatch(tool_name: str, arguments: str) -> str:
    """
    Route a tool call to the correct backend.

    Parameters
    ----------
    tool_name : str
        Must match a key in ``TOOL_REGISTRY``.
    arguments : str
        Raw argument string forwarded verbatim to the tool's ``run()`` function.

    Returns
    -------
    str
        The tool's plain-text output (stdout, result, or error message).

    Raises
    ------
    ValueError
        If *tool_name* is not found in the registry.
    """
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(list_tools())
        error_msg = (
            f"Unknown tool '{tool_name}'. Available tools: [{available}]"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Dispatching tool '%s' with arguments: %s", tool_name, arguments[:200])
    try:
        result = TOOL_REGISTRY[tool_name](arguments)
        logger.info("Tool '%s' returned %d chars", tool_name, len(result))
        return result
    except Exception as exc:
        error_msg = f"Tool '{tool_name}' raised an exception: {exc}"
        logger.exception(error_msg)
        return error_msg


__all__ = [
    "TOOL_REGISTRY",
    "dispatch",
    "list_tools",
    "calculator_run",
    "python_repl_run",
    "sympy_run",
    "websearch_run",
]
