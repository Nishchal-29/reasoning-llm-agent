import re
import sympy as sp
import numpy as np
from scipy import stats, integrate, optimize
from fractions import Fraction
from decimal import Decimal, getcontext

getcontext().prec = 28  # high precision decimal

class ToolExecutor:
    def __init__(self):
        self.trace = []

    # Calculator
    def execute_calculator(self, expr):
        try:
            result = eval(expr)
        except Exception as e:
            result = str(e)
        self.trace.append(f"{expr} -> Result: {result}")
        return result

    # Symbolic math using sympy
    def execute_symbolic(self, expr):
        try:
            result = sp.collect(expr)
            result = sp.simplify(sp.sympify(expr))
        except Exception as e:
            result = str(e)
        self.trace.append(f"symbolic({expr}) -> Result: {result}")
        return result

    # Numerical computations using numpy
    def execute_numeric(self, expr):
        try:
            result = eval(expr, {"np": np})
        except Exception as e:
            result = str(e)
        self.trace.append(f"numeric({expr}) -> Result: {result}")
        return result

    # Probability / optimization using scipy
    def execute_scipy(self, expr_type, *args):
        try:
            if expr_type == "integrate":
                result = integrate.quad(*args)[0]  # integrate.quad(func, a, b)
            elif expr_type == "optimize":
                result = optimize.minimize(*args).fun
            elif expr_type == "probability":
                dist, params, x = args
                result = getattr(stats, dist).pmf(x, *params)
            else:
                result = "Unsupported scipy operation"
        except Exception as e:
            result = str(e)
        self.trace.append(f"scipy({expr_type}, {args}) -> Result: {result}")
        return result

    # Exact arithmetic
    def execute_exact(self, expr):
        try:
            result = eval(expr, {"Fraction": Fraction, "Decimal": Decimal})
        except Exception as e:
            result = str(e)
        self.trace.append(f"exact({expr}) -> Result: {result}")
        return result

    # Main dispatcher
    def run_plan(self, plan):
        results = []
        for step in plan:
            if step.startswith("calculator("):
                expr = re.search(r'calculator\((.*)\)', step).group(1)
                results.append(self.execute_calculator(expr))
            elif step.startswith("symbolic("):
                expr = re.search(r'symbolic\((.*)\)', step).group(1)
                results.append(self.execute_symbolic(expr))
            elif step.startswith("numeric("):
                expr = re.search(r'numeric\((.*)\)', step).group(1)
                results.append(self.execute_numeric(expr))
            elif step.startswith("scipy("):
                parts = re.findall(r'scipy\((.*?)\)', step)[0].split(',')
                expr_type = parts[0].strip()
                args = tuple(eval(x.strip()) for x in parts[1:])
                results.append(self.execute_scipy(expr_type, *args))
            elif step.startswith("exact("):
                expr = re.search(r'exact\((.*)\)', step).group(1)
                results.append(self.execute_exact(expr))
            else:
                results.append("Unsupported Tool")
        return results


# -----------------------------
# Example test
# -----------------------------
if __name__ == "__main__":
    # Sample plan containing all types of math tools
    plan = [
        "calculator(150*0.2)",
        "symbolic(x**2 + 2*x + 1)",
        "numeric(np.array([1,2,3]) * 2)",
        "exact(Fraction(1,3) + Fraction(2,3))",
        "scipy(integrate, lambda x: x**2, 0, 1)"
    ]

    executor = ToolExecutor()
    results = executor.run_plan(plan)

    print("\n--- Execution Trace ---")
    for step in executor.trace:
        print(step)

    print("\n--- Final Results ---")
    print(results)
