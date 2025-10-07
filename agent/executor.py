import subprocess, json, shlex, tempfile, os, signal, time
from sympy import sympify, symbols, solve
import math

def run_calc(cmd: str):
    assert cmd.startswith("compute ")
    expr = cmd[len("compute "):].strip()
    val = sympify(expr).evalf()
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return float(f)
    except Exception:
        return str(val)

def run_sympy(cmd: str):
    allowed_globals = {"solve": solve, "symbols": symbols, "sympify": sympify}
    return eval(cmd, {"__builtins__": None}, allowed_globals)

def run_python_safe(code: str, timeout=3):
    if code.startswith('    '):
        code_block = code
    else:
        code_block = '    ' + code.replace('\n', '\n    ')
    wrapper = (
        "import json, math\n"
        "def _run():\n"
        f"{code_block}\n"
        "try:\n"
        "    res = _run()\n"
        "    print(json.dumps({'ok': True, 'result': res}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'error': str(e)}))\n"
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tf:
        tf.write(wrapper)
        fname = tf.name
    try:
        proc = subprocess.run(["python", fname], capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip()
        try:
            return json.loads(out)
        except Exception:
            return {"ok": False, "error": "Bad JSON from subprocess", "stdout": out, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass

def execute_plan(plan):
    ctx = {}
    last_result = None
    for step in plan:
        tool = step.get("tool")
        inp = step.get("input")
        if tool == "calc":
            try:
                last_result = run_calc(inp)
            except Exception as e:
                return {"ok": False, "error": f"calc failed: {e}"}
        elif tool == "sympy":
            try:
                last_result = run_sympy(inp)
            except Exception as e:
                return {"ok": False, "error": f"sympy failed: {e}"}
        elif tool == "python":
            code = inp.replace("result_from_step1", "ctx.get('step1')")
            res = run_python_safe(code)
            if not res.get("ok", False):
                return {"ok": False, "error": f"python step failed: {res.get('error')}"}
            last_result = res.get("result")
        elif tool == "bruteforce":
            res = run_python_safe(inp)
            if not res.get("ok", False):
                return {"ok": False, "error": f"bruteforce failed: {res.get('error')}"}
            last_result = res.get("result")
        elif tool == "z3":
            return {"ok": False, "error": "z3 not implemented in this executor skeleton"}
        else:
            return {"ok": False, "error": f"Unknown tool {tool}"}
        step_name = f"step{step.get('step')}"
        ctx[step_name] = last_result
    return {"ok": True, "result": last_result, "ctx": ctx}