"""
Safe symbolic execution sandbox.

Uses restricted eval to execute generated expression strings.
No file IO, no subprocesses, isolated namespace.
"""
from typing import Dict, Any, Optional

# Math functions that are safe to expose
import math

SAFE_BUILTINS = {
    'abs': abs,
    'max': max,
    'min': min,
    'pow': pow,
    'round': round,
    'int': int,
    'float': float,
    'bool': bool,
    'math': math,
    'True': True,
    'False': False,
}

def safe_eval(expr_str: str, variables: Dict[str, Any]) -> Any:
    """
    Safely evaluate a mathematical expression string.

    Args:
        expr_str: string like "(x * x) + 2" or "(x > 0 if (x > 0) else -x)"
        variables: dict like {"x": 5}

    Returns:
        Evaluation result or None if error.
    """
    # Create isolated environment
    env = {"__builtins__": SAFE_BUILTINS}
    env.update(variables)

    try:
        # Compile first for safety
        compiled = compile(expr_str, "<string>", "eval")
        result = eval(compiled, env)
        # Normalize booleans to int for fitness evaluation consistency
        if isinstance(result, bool):
            result = int(result)
        return result
    except Exception:
        # Any error (division by zero, overflow, invalid syntax) means failed execution
        return None

def execute_individual(ind: 'Individual', variables: Dict[str, Any]) -> Any:
    """Execute an individual's expression tree with given variable bindings."""
    expr = ind.to_expr()
    return safe_eval(expr, variables)
