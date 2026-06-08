from __future__ import annotations
import logging
import subprocess
import sys
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: int = 5
MAX_OUTPUT_CHARS: int = 8_000
PYTHON_EXECUTABLE: str = sys.executable

def execute_code(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, max_output: int = MAX_OUTPUT_CHARS) -> dict[str, str]:
    code = textwrap.dedent(code).strip()
    logger.info(
        "Executing Python code (%d chars, timeout=%ds):\n%s",
        len(code),
        timeout,
        code[:300],
    )

    try:
        proc = subprocess.run(
            [PYTHON_EXECUTABLE, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "en_US.UTF-8",
            },
        )
        stdout = proc.stdout[:max_output] if proc.stdout else ""
        stderr = proc.stderr[:max_output] if proc.stderr else ""
        status = "ok" if proc.returncode == 0 else "error"
        logger.info(
            "Code execution finished (status=%s, rc=%d, stdout=%d chars, stderr=%d chars)",
            status,
            proc.returncode,
            len(stdout),
            len(stderr),
        )

        return {"stdout": stdout, "stderr": stderr, "status": status}

    except subprocess.TimeoutExpired:
        logger.warning("Code execution timed out after %ds", timeout)
        return {
            "stdout": "",
            "stderr": f"[TimeoutError] Execution exceeded {timeout}s limit and was killed.",
            "status": "timeout",
        }
    except OSError as exc:
        logger.exception("OS error during code execution")
        return {
            "stdout": "",
            "stderr": f"[OSError] {exc}",
            "status": "error",
        }

def run(input_str: str, timeout: Optional[int] = None) -> str:
    effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
    result = execute_code(input_str, timeout=effective_timeout)
    parts: list[str] = []
    if result["stdout"]:
        parts.append(result["stdout"].rstrip())
    if result["stderr"]:
        parts.append(f"STDERR: {result['stderr'].rstrip()}")
    if not parts:
        parts.append("(no output)")

    combined = "\n".join(parts)
    if result["status"] == "ok":
        return combined
    elif result["status"] == "timeout":
        return f"[Timeout] {combined}"
    else:
        return f"[ExecutionError]\n{combined}"