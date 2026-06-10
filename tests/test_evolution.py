import pytest
from src.evolution.population import create_population, Node, Individual
from src.evolution.operators import point_mutation, subtree_crossover

def test_create_population():
    pop = create_population(10, ["x"])
    assert len(pop) == 10
    assert isinstance(pop[0], Individual)

def test_point_mutation():
    ind = Individual(tree=Node('const', 5))
    mutated = point_mutation(ind, mutation_rate=1.0)
    # Could be same if random delta is 0, but usually different
    assert mutated is not ind # deep copy

def test_crossover():
    p1 = Individual(tree=Node('op', '+', [Node('const', 1), Node('const', 2)]))
    p2 = Individual(tree=Node('op', '*', [Node('const', 3), Node('const', 4)]))
    c1, c2 = subtree_crossover(p1, p2)
    assert c1 is not p1
    assert c2 is not p2
