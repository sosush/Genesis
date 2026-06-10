"""
Safe symbolic execution sandbox.

Uses restricted eval to execute generated expression strings.
No file IO, no subprocesses, isolated namespace.
"""
from typing import Dict, Any, Callable, Optional

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
    'math': math
}

def safe_eval(expr_str: str, variables: Dict[str, Any]) -> Any:
    """
    Safely evaluate a mathematical expression string.
    
    Args:
        expr_str: string like "(x * x) + 2"
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
        return result
    except Exception:
        # Any error (division by zero, overflow, invalid syntax) means failed execution
        return None

def execute_individual(ind: 'Individual', variables: Dict[str, Any]) -> Any:
    """Execute an individual's expression tree with given variable bindings."""
    expr = ind.to_expr()
    return safe_eval(expr, variables)
