"""
Tier-1 Problem Library for Genesis.

All problems are solvable within the extended grammar:
  arithmetic ops (+, -, *, /, %), comparisons (cmp), and ternary conditionals.

Each problem dict has:
  name        — human-readable label (used in the UI gallery)
  slug        — URL-safe identifier
  description — plain-English explanation for a non-technical visitor
  variables   — list of variable names the program takes
  test_cases  — List[({var: val, ...}, expected_output)]
  hint        — what the "aha" formula looks like (shown after solve)
  difficulty  — "easy" | "medium" | "hard" (for gallery ordering)
"""
from typing import Any, Dict, List, Tuple

TestCase = Tuple[Dict[str, Any], Any]


def _tc(variables: Dict[str, Any], expected: Any) -> TestCase:
    return (variables, expected)


# ------------------------------------------------------------------
# Problem definitions
# ------------------------------------------------------------------

ABSOLUTE_VALUE: List[TestCase] = [
    _tc({'x': x}, abs(x))
    for x in [-5, -3, -1, 0, 1, 3, 5, 7, -7, 2, -2]
]

SIGN_FUNCTION: List[TestCase] = [
    _tc({'x': x}, (1 if x > 0 else (-1 if x < 0 else 0)))
    for x in [-4, -2, -1, 0, 1, 2, 4, 6, -6]
]

MAX_OF_TWO: List[TestCase] = [
    _tc({'a': a, 'b': b}, max(a, b))
    for a, b in [(-3, 2), (5, 1), (0, 0), (-1, -2), (4, 4), (3, -3), (7, 9), (-5, -1)]
]

MIN_OF_TWO: List[TestCase] = [
    _tc({'a': a, 'b': b}, min(a, b))
    for a, b in [(-3, 2), (5, 1), (0, 0), (-1, -2), (4, 4), (3, -3), (7, 9), (-5, -1)]
]

EVEN_OR_ODD: List[TestCase] = [
    _tc({'x': x}, 1 if x % 2 == 0 else 0)
    for x in range(-6, 10)
]

FIZZBUZZ_CLASSIFIER: List[TestCase] = [
    # 0 = normal, 1 = Fizz (div by 3), 2 = Buzz (div by 5), 3 = FizzBuzz (div by 15)
    _tc({'x': x}, (3 if x % 15 == 0 else (1 if x % 3 == 0 else (2 if x % 5 == 0 else 0))))
    for x in [1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 7, 11]
]

LEAP_YEAR: List[TestCase] = [
    # Simplified: year divisible by 4 (ignoring century rule for grammar simplicity)
    _tc({'y': y}, 1 if y % 4 == 0 else 0)
    for y in [2000, 2001, 2004, 1900, 1996, 2100, 2024, 2023, 2020, 2019, 1984]
]

CLAMP: List[TestCase] = [
    _tc({'x': x, 'lo': -3, 'hi': 3}, max(-3, min(3, x)))
    for x in [-10, -5, -3, -1, 0, 1, 3, 5, 10, -4, 4]
]

QUADRATIC_HAS_REAL_ROOTS: List[TestCase] = [
    # Returns 1 if b*b - 4*a*c >= 0, else 0
    _tc({'a': a, 'b': b, 'c': c}, 1 if (b * b - 4 * a * c) >= 0 else 0)
    for a, b, c in [
        (1, 2, 1), (1, 0, 1), (1, -3, 2), (2, 0, -1), (1, 1, 1),
        (1, 4, 3), (3, 1, 3), (1, 2, -3), (1, 0, 0),
    ]
]

SQUARE: List[TestCase] = [
    _tc({'x': x}, x * x)
    for x in range(-5, 6)
]

ADDITION: List[TestCase] = [
    _tc({'x': x, 'y': y}, x + y)
    for x in range(-3, 4) for y in range(-3, 4)
]


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

PROBLEMS = {
    'absolute_value': {
        'name': 'Absolute Value',
        'slug': 'absolute_value',
        'description': 'f(x) = |x| — return the distance from zero. If x is negative, flip the sign.',
        'variables': ['x'],
        'test_cases': ABSOLUTE_VALUE,
        'hint': '(x if x >= 0 else -x)',
        'difficulty': 'easy',
        'category': 'classic',
    },
    'sign_function': {
        'name': 'Sign Function',
        'slug': 'sign_function',
        'description': 'f(x) → -1, 0, or 1 depending on whether x is negative, zero, or positive.',
        'variables': ['x'],
        'test_cases': SIGN_FUNCTION,
        'hint': '(1 if x > 0 else (-1 if x < 0 else 0))',
        'difficulty': 'easy',
        'category': 'classic',
    },
    'max_of_two': {
        'name': 'Max of Two',
        'slug': 'max_of_two',
        'description': 'f(a, b) — return whichever of a or b is larger.',
        'variables': ['a', 'b'],
        'test_cases': MAX_OF_TWO,
        'hint': '(a if a > b else b)',
        'difficulty': 'easy',
        'category': 'classic',
    },
    'min_of_two': {
        'name': 'Min of Two',
        'slug': 'min_of_two',
        'description': 'f(a, b) — return whichever of a or b is smaller.',
        'variables': ['a', 'b'],
        'test_cases': MIN_OF_TWO,
        'hint': '(a if a < b else b)',
        'difficulty': 'easy',
        'category': 'classic',
    },
    'even_or_odd': {
        'name': 'Even or Odd',
        'slug': 'even_or_odd',
        'description': 'The classic warm-up: return 1 if x is even, 0 if odd. Needs modulo (%).',
        'variables': ['x'],
        'test_cases': EVEN_OR_ODD,
        'hint': '(1 if x % 2 == 0 else 0)',
        'difficulty': 'easy',
        'category': 'interview',
    },
    'fizzbuzz_classifier': {
        'name': 'FizzBuzz Classifier',
        'slug': 'fizzbuzz_classifier',
        'description': 'Output a category: 0=normal, 1=Fizz (div by 3), 2=Buzz (div by 5), 3=FizzBuzz (div by 15). Numeric output — same logic as the classic interview question.',
        'variables': ['x'],
        'test_cases': FIZZBUZZ_CLASSIFIER,
        'hint': '(3 if x%15==0 else (1 if x%3==0 else (2 if x%5==0 else 0)))',
        'difficulty': 'hard',
        'category': 'interview',
    },
    'leap_year': {
        'name': 'Leap Year Check',
        'slug': 'leap_year',
        'description': 'Return 1 if the year is a leap year (divisible by 4), else 0. Needs % and comparison.',
        'variables': ['y'],
        'test_cases': LEAP_YEAR,
        'hint': '(1 if y % 4 == 0 else 0)',
        'difficulty': 'medium',
        'category': 'interview',
    },
    'clamp': {
        'name': 'Clamp',
        'slug': 'clamp',
        'description': 'f(x, lo=-3, hi=3) — bound x between lo and hi. If x < lo return lo; if x > hi return hi; else x.',
        'variables': ['x', 'lo', 'hi'],
        'test_cases': CLAMP,
        'hint': '(lo if x < lo else (hi if x > hi else x))',
        'difficulty': 'medium',
        'category': 'classic',
    },
    'quadratic_roots': {
        'name': 'Quadratic Has Real Roots',
        'slug': 'quadratic_roots',
        'description': 'Given a, b, c — return 1 if the equation ax² + bx + c has real solutions (discriminant b²−4ac ≥ 0). A nice one for anyone with a math background.',
        'variables': ['a', 'b', 'c'],
        'test_cases': QUADRATIC_HAS_REAL_ROOTS,
        'hint': '(1 if (b*b - 4*a*c) >= 0 else 0)',
        'difficulty': 'medium',
        'category': 'math',
    },
    'square': {
        'name': 'Square',
        'slug': 'square',
        'description': 'f(x) = x² — the benchmark problem. Watch Genesis discover it in ~23 generations.',
        'variables': ['x'],
        'test_cases': SQUARE,
        'hint': '(x * x)',
        'difficulty': 'easy',
        'category': 'benchmark',
    },
    'addition': {
        'name': 'Addition',
        'slug': 'addition',
        'description': 'f(x, y) = x + y — simple two-variable arithmetic. Good for watching population diversity.',
        'variables': ['x', 'y'],
        'test_cases': ADDITION,
        'hint': '(x + y)',
        'difficulty': 'easy',
        'category': 'benchmark',
    },
}


def get_problem(slug: str) -> dict:
    if slug not in PROBLEMS:
        raise KeyError(f"Unknown problem slug: {slug!r}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[slug]


def list_problems() -> List[dict]:
    """Return problems without the full test_case lists (for the gallery endpoint)."""
    return [
        {k: v for k, v in p.items() if k != 'test_cases'}
        for p in PROBLEMS.values()
    ]
