import pytest
from src.symbolic.executor import safe_eval
from src.symbolic.evaluator import evaluate_fitness
from src.evolution.population import Individual, Node

def test_safe_eval():
    assert safe_eval("(x * 2) + 1", {"x": 3}) == 7
    assert safe_eval("abs(x - 10)", {"x": 5}) == 5
    assert safe_eval("x / 0", {"x": 1}) is None # Should catch ZeroDivisionError

def test_evaluate_fitness():
    tree = Node('op', '+', [Node('var', 'x'), Node('const', 1)])
    ind = Individual(tree=tree)
    test_cases = [({"x": 1}, 2), ({"x": 2}, 3)]
    fitness = evaluate_fitness(ind, test_cases)
    # Slightly less than 1.0 due to parsimony pressure
    assert 0.9 < fitness <= 1.0
