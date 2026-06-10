"""
Fitness evaluator for symbolic execution.

Compares individual outputs against a set of test cases.
"""
import math
from typing import List, Dict, Any, Tuple

from .executor import execute_individual
from ..evolution.population import Individual

TestCase = Tuple[Dict[str, Any], Any]  # e.g., ({"x": 2}, 4)

def evaluate_fitness(ind: Individual, test_cases: List[TestCase]) -> float:
    """
    Evaluate fitness of an individual against test cases.
    Fitness is [0.0, 1.0].
    We use partial credit: exact match = 1.0, close numeric match = partial.
    """
    if not test_cases:
        return 0.0
        
    total_score = 0.0
    
    for inputs, expected in test_cases:
        output = execute_individual(ind, inputs)
        
        if output is None:
            # Execution failed
            continue
            
        if isinstance(expected, (int, float)) and isinstance(output, (int, float)):
            # Numeric evaluation with partial credit
            if math.isinf(output) or math.isnan(output):
                continue
                
            diff = abs(expected - output)
            if diff < 1e-6:
                total_score += 1.0
            else:
                # Partial credit: decays as diff grows
                # 1 / (1 + diff) gives 0.5 for diff=1, 0.1 for diff=9
                total_score += 1.0 / (1.0 + diff)
        else:
            # Exact match evaluation (e.g. for strings/booleans)
            if output == expected:
                total_score += 1.0
                
    # Normalize fitness to [0, 1]
    base_fitness = total_score / len(test_cases)
    
    # Parsimony pressure: slight penalty for very large trees to encourage elegance
    size = ind.tree.size()
    penalty = 0.001 * size
    
    # Ensure fitness stays in [0, 1]
    final_fitness = max(0.0, min(1.0, base_fitness - penalty))
    ind.fitness = final_fitness
    return final_fitness

def is_solved(ind: Individual) -> bool:
    """Check if an individual has perfectly solved the problem."""
    # Since we apply parsimony penalty, a perfect score might be slightly less than 1.0
    return ind.fitness > 0.99
