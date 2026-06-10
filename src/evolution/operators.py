"""
Crossover and mutation operators for expression tree evolution.

Operators:
  - subtree_crossover: swap random subtrees between two parents
  - point_mutation: randomly modify individual nodes
  - subtree_mutation: replace a random subtree with a fresh random one
"""
import random
from typing import List, Optional, Tuple

from .population import Individual, Node, OPERATORS, random_node, random_terminal


# ------------------------------------------------------------------
# Crossover
# ------------------------------------------------------------------

def subtree_crossover(
    parent1: Individual,
    parent2: Individual,
) -> Tuple[Individual, Individual]:
    """
    Subtree crossover: pick a random (non-root) node in each parent
    and swap the entire subtrees rooted at those nodes.

    Returns two new individuals (children). Parents are not modified
    (we deep-copy first).
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    nodes1 = child1.tree.all_nodes()
    nodes2 = child2.tree.all_nodes()

    # Need at least one non-root node to do a meaningful swap
    interior1 = nodes1[1:] if len(nodes1) > 1 else nodes1
    interior2 = nodes2[1:] if len(nodes2) > 1 else nodes2

    n1 = random.choice(interior1)
    n2 = random.choice(interior2)

    # Swap node content (effectively swaps subtrees in-place)
    n1_snapshot = (n1.ntype, n1.value, n1.children)
    n1.ntype, n1.value, n1.children = n2.ntype, n2.value, n2.children
    n2.ntype, n2.value, n2.children = n1_snapshot

    return child1, child2


# ------------------------------------------------------------------
# Mutation
# ------------------------------------------------------------------

def point_mutation(
    individual: Individual,
    mutation_rate: float = 0.3,
    variables: Optional[List[str]] = None,
) -> Individual:
    """
    Point mutation: walk every node and randomly perturb it.
    - Operator nodes: swap to a different operator
    - Constant nodes: ± small integer perturbation
    - Variable nodes: swap to another variable
    """
    if variables is None:
        variables = ['x']

    mutant = individual.copy()

    for node in mutant.tree.all_nodes():
        if random.random() > mutation_rate:
            continue

        if node.ntype == 'op':
            # Pick a different operator
            alternatives = [op for op in OPERATORS if op != node.value]
            if alternatives:
                node.value = random.choice(alternatives)

        elif node.ntype == 'const':
            delta = random.choice([-2, -1, 1, 2])
            node.value = max(-10, min(10, node.value + delta))

        elif node.ntype == 'var' and len(variables) > 1:
            node.value = random.choice([v for v in variables if v != node.value] or variables)

    return mutant


def subtree_mutation(
    individual: Individual,
    variables: Optional[List[str]] = None,
) -> Individual:
    """
    Subtree mutation: replace a random (non-root) subtree with a
    freshly generated random subtree. Preserves the root.
    """
    if variables is None:
        variables = ['x']

    mutant = individual.copy()
    nodes = mutant.tree.all_nodes()

    if len(nodes) < 2:
        # Only root exists — replace it entirely
        new_tree = random_node(0, variables)
        mutant.tree.ntype = new_tree.ntype
        mutant.tree.value = new_tree.value
        mutant.tree.children = new_tree.children
        return mutant

    target = random.choice(nodes[1:])
    replacement = random_node(0, variables)

    target.ntype = replacement.ntype
    target.value = replacement.value
    target.children = replacement.children

    return mutant


def constant_perturbation(
    individual: Individual,
    std: float = 1.0,
) -> Individual:
    """
    Gaussian perturbation of all constant nodes.
    Useful for fine-tuning numeric coefficients late in evolution.
    """
    mutant = individual.copy()
    import random as _r
    for node in mutant.tree.all_nodes():
        if node.ntype == 'const':
            node.value = int(round(node.value + _r.gauss(0, std)))
    return mutant
