"""
Arithmetic benchmarks for program synthesis.
Each problem provides a list of input-output dictionaries.
"""

IDENTITY = [({"x": i}, i) for i in range(-5, 6)]

SQUARE = [({"x": i}, i*i) for i in range(-5, 6)]

ADD = [({"x": i, "y": j}, i+j) for i in range(-3, 4) for j in range(-3, 4)]

PROBLEMS = {
    "identity": (IDENTITY, ["x"]),
    "square": (SQUARE, ["x"]),
    "add": (ADD, ["x", "y"])
}
