from z3 import Solver, Int, Real, Bool, And, Or, Not, Implies, sat

class Z3FullReasoner:
    def __init__(self):
        self.solver = Solver()

    def reset(self):
        """Reset the solver state."""
        self.solver = Solver()

    # ------------------------------
    # Arithmetic / Puzzle Solvers
    # ------------------------------
    def solve_linear_equation(self, equation_str: str):
        """
        Solve single variable linear equations, e.g. 'x + 2 = 5'.
        """
        try:
            var_name = ''.join(filter(str.isalpha, equation_str))
            x = Real(var_name)
            eq = equation_str.replace('=', '==')
            context = {var_name: x, 'And': And, 'Or': Or, 'Not': Not, 'Implies': Implies}
            parsed_eq = eval(eq, {}, context)
            
            self.solver.add(parsed_eq)
            if self.solver.check() == sat:
                model = self.solver.model()
                return {str(var_name): model[x]}
            else:
                return {"error": "No solution found"}
        except Exception as e:
            return {"error": str(e)}

    def solve_puzzle(self, constraints: list):
        """
        Solve puzzles with multiple constraints. Example: ["x + y == 10", "x > 2"]
        """
        self.reset()
        try:
            vars_set = set()
            for c in constraints:
                for ch in c:
                    if ch.isalpha():
                        vars_set.add(ch)

            z3_vars = {v: Real(v) for v in vars_set}

            for c in constraints:
                eq = c.replace('=', '==')
                parsed = eval(eq, {}, z3_vars)
                self.solver.add(parsed)

            if self.solver.check() == sat:
                model = self.solver.model()
                return {str(v): model[z3_vars[v]] for v in z3_vars}
            else:
                return {"error": "No solution found"}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------
    # Logical Reasoning
    # ------------------------------
    def solve_logical(self, statements: list):
        """
        Solve logical statements. Example: ["A or B", "not A"]
        """
        self.reset()
        try:
            vars_set = set()
            for stmt in statements:
                for ch in stmt:
                    if ch.isalpha():
                        vars_set.add(ch)
            z3_vars = {v: Bool(v) for v in vars_set}
            context = {**z3_vars, 'And': And, 'Or': Or, 'Not': Not, 'Implies': Implies}
            for stmt in statements:
                expr = stmt.replace("and", "And").replace("or", "Or").replace("not", "Not").replace("implies", "Implies")
                parsed = eval(expr, {}, context)
                self.solver.add(parsed)

            if self.solver.check() == sat:
                model = self.solver.model()
                return {str(v): model[z3_vars[v]] for v in z3_vars}
            else:
                return {"error": "Statements are unsatisfiable"}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------
    # Unified Executor
    # ------------------------------
    def execute(self, query):
        """
        Detect type of query:
        - str → single equation
        - list → either puzzle constraints (with '=') or logical statements
        """
        if isinstance(query, str):
            return self.solve_linear_equation(query)
        elif isinstance(query, list):
            if any(op in ''.join(query) for op in ['and', 'or', 'not', 'implies']):
                return self.solve_logical(query)
            elif any('=' in c or '>' in c or '<' in c for c in query):
                return self.solve_puzzle(query)
            else:
                return {"error": "Cannot determine query type"}
        else:
            return {"error": "Unsupported query format"}
