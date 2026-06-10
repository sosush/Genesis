"""
Individual representation and population initialization for Genesis.

Each individual is an expression tree over arithmetic operators.
Leaves are variables (x, y) or integer constants.
Internal nodes are binary operators: +, -, *, /
"""
import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional

# Grammar constants
OPERATORS = ['+', '-', '*', '/']
MAX_DEPTH = 5


class Node:
    """A node in an expression tree."""

    __slots__ = ('ntype', 'value', 'children')

    def __init__(self, ntype: str, value, children: Optional[List['Node']] = None):
        """
        Args:
            ntype: 'op' | 'var' | 'const'
            value: operator string, variable name, or integer constant
            children: child nodes (2 for binary ops, 0 for terminals)
        """
        self.ntype = ntype
        self.value = value
        self.children: List['Node'] = children if children is not None else []

    # ------------------------------------------------------------------
    # Tree properties
    # ------------------------------------------------------------------

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def all_nodes(self) -> List['Node']:
        """BFS traversal returning all nodes (including self)."""
        result = [self]
        for child in self.children:
            result.extend(child.all_nodes())
        return result

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def to_expr(self) -> str:
        """Convert tree to a Python expression string."""
        if self.ntype == 'var':
            return str(self.value)
        if self.ntype == 'const':
            return str(self.value)
        # Binary operator
        left = self.children[0].to_expr()
        right = self.children[1].to_expr()
        return f"({left} {self.value} {right})"

    def to_lambda(self, variables: List[str]) -> str:
        """Return a lambda string, e.g. 'lambda x: (x * x)'."""
        args = ', '.join(variables)
        return f"lambda {args}: {self.to_expr()}"

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Node({self.ntype}, {self.value!r})"

    def copy(self) -> 'Node':
        return copy.deepcopy(self)


# ------------------------------------------------------------------
# Random tree generation
# ------------------------------------------------------------------

def random_terminal(variables: List[str]) -> Node:
    """Generate a random terminal node (variable or constant)."""
    if variables and random.random() < 0.65:
        return Node('var', random.choice(variables))
    return Node('const', random.randint(-5, 5))


def random_node(depth: int = 0, variables: Optional[List[str]] = None) -> Node:
    """
    Recursively build a random expression tree using ramped half-and-half.
    At max depth, always returns a terminal.
    """
    if variables is None:
        variables = ['x']

    if depth >= MAX_DEPTH - 1:
        return random_terminal(variables)

    # Probability of being an operator node decreases with depth
    p_operator = max(0.3, 0.85 - depth * 0.15)
    if random.random() < p_operator:
        op = random.choice(OPERATORS)
        left = random_node(depth + 1, variables)
        right = random_node(depth + 1, variables)
        return Node('op', op, [left, right])

    return random_terminal(variables)


# ------------------------------------------------------------------
# Individual
# ------------------------------------------------------------------

@dataclass
class Individual:
    """A candidate program (expression tree) with its fitness score."""
    tree: Node
    fitness: float = 0.0

    def to_expr(self) -> str:
        return self.tree.to_expr()

    def to_lambda(self, variables: List[str]) -> str:
        return self.tree.to_lambda(variables)

    def copy(self) -> 'Individual':
        return Individual(tree=self.tree.copy(), fitness=self.fitness)

    def __repr__(self):
        return f"Individual(expr={self.to_expr()!r}, fitness={self.fitness:.4f})"


# ------------------------------------------------------------------
# Population
# ------------------------------------------------------------------

def create_population(size: int, variables: Optional[List[str]] = None) -> List[Individual]:
    """
    Create an initial population of random individuals.
    Uses ramped half-and-half to ensure variety in tree depth/shape.
    """
    if variables is None:
        variables = ['x']
    return [Individual(tree=random_node(0, variables)) for _ in range(size)]
