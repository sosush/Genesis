import pytest
from src.evolution.population import Node, Individual, create_population
from src.synthesis.events import node_to_snapshot, individual_to_snapshot, build_fitness_histogram, build_population_snapshot

def test_node_id_stable_copy():
    # Make sure copying a node preserves its ID for animation match
    root = Node('op', '+', [Node('const', 1), Node('const', 2)])
    assert len(root.node_id) == 8
    
    clone = root.copy()
    assert clone.node_id == root.node_id
    assert clone.children[0].node_id == root.children[0].node_id

def test_events_snapshots():
    # Verify snapshot serializations function correctly
    ind = Individual(tree=Node('const', 5))
    ind.fitness = 0.95
    
    snap = individual_to_snapshot(ind, rank=0, include_tree=True)
    assert snap.fitness == 0.95
    assert snap.tree is not None
    assert snap.tree.value == 5
    
    dict_repr = snap.to_dict()
    assert dict_repr['fitness'] == 0.95
    assert dict_repr['tree']['value'] == 5

def test_fitness_histogram():
    fitnesses = [0.05, 0.15, 0.18, 0.45, 0.99]
    hist = build_fitness_histogram(fitnesses, bins=10)
    assert len(hist) == 10
    # 0.05 in [0, 0.1) -> bin 0
    assert hist[0] == 1
    # 0.15, 0.18 in [0.1, 0.2) -> bin 1
    assert hist[1] == 2
    # 0.99 in [0.9, 1.0] -> bin 9
    assert hist[9] == 1
