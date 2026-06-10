"""
Simplified SyGuS string manipulation benchmarks.
Because our grammar is numeric, we simulate these with integers 
(e.g., ascii shifts) or skip them for the numeric engine.
"""

# Note: The current genetic engine operates on arithmetic expression trees.
# String manipulation would require a typed grammar (String, Int) and string operators.
# For demonstration purposes, we include a simple shift benchmark.

# Shift by 1: simulate string shift
SHIFT = [({"x": i}, i + 1) for i in range(10)]

PROBLEMS = {
    "shift_string": (SHIFT, ["x"])
}
