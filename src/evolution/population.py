"""
Individual representation and population initialization for Genesis.

Each individual is an expression tree over arithmetic operators.
Leaves are variables (x, y) or integer constants.
Internal nodes are:
  - Binary arithmetic operators: +, -, *, /, %
  - Comparison operators (cmp): <, >, <=, >=, ==
  - Ternary conditional (ternary): (cond, if_true, if_false)
"""
import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from uuid import uuid4

# Grammar constants
OPERATORS = ['+', '-', '*', '/', '%']
CMP_OPS = ['<', '>', '<=', '>=', '==']
MAX_DEPTH = 5


def _short_id() -> str:
    """Generate a short (8-char) hex UUID."""
    return uuid4().hex[:8]


class Node:
    """A node in an expression tree."""

    __slots__ = ('ntype', 'value', 'children', 'node_id')

    def __init__(self, ntype: str, value: Any, children: Optional[List['Node']] = None,
                 node_id: Optional[str] = None):
        """
        Args:
            ntype: 'op' | 'var' | 'const' | 'cmp' | 'ternary'
            value: operator string, cmp string, variable name, or numeric constant
            children: child nodes (2 for binary ops/cmp, 3 for ternary, 0 for terminals)
            node_id: stable identifier for animation/lineage tracking; auto-generated if None
        """
        self.ntype = ntype
        self.value = value
        self.children: List['Node'] = children if children is not None else []
        self.node_id: str = node_id if node_id is not None else _short_id()

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
        if self.ntype in ('op', 'cmp'):
            left = self.children[0].to_expr()
            right = self.children[1].to_expr()
            if self.ntype == 'op' and self.value == '/':
                # Guard against division by zero at expression level
                return f"({left} / ({right} if ({right}) != 0 else 1))"
            if self.ntype == 'op' and self.value == '%':
                return f"({left} % ({right} if ({right}) != 0 else 1))"
            return f"({left} {self.value} {right})"
        if self.ntype == 'ternary':
            cond = self.children[0].to_expr()
            if_true = self.children[1].to_expr()
            if_false = self.children[2].to_expr()
            return f"({if_true} if ({cond}) else {if_false})"
        # Fallback
        return str(self.value)

    def to_lambda(self, variables: List[str]) -> str:
        """Return a lambda string, e.g. 'lambda x: (x * x)'."""
        args = ', '.join(variables)
        return f"lambda {args}: {self.to_expr()}"

    # ------------------------------------------------------------------
    # Serialization (for WebSocket streaming to frontend)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialize node to nested JSON-compatible dict."""
        return {
            'node_id': self.node_id,
            'ntype': self.ntype,
            'value': self.value,
            'children': [c.to_dict() for c in self.children],
        }

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


def random_cmp_node(depth: int, variables: List[str]) -> Node:
    """Generate a comparison node (cmp op): (left CMP right) → bool/0-1."""
    op = random.choice(CMP_OPS)
    left = random_node(depth + 1, variables, allow_cmp=False, allow_ternary=False)
    right = random_node(depth + 1, variables, allow_cmp=False, allow_ternary=False)
    return Node('cmp', op, [left, right])


def random_ternary_node(depth: int, variables: List[str]) -> Node:
    """Generate a ternary node: (cond ? if_true : if_false)."""
    cond = random_cmp_node(depth, variables)
    if_true = random_node(depth + 1, variables, allow_ternary=False)
    if_false = random_node(depth + 1, variables, allow_ternary=False)
    return Node('ternary', 'ternary', [cond, if_true, if_false])


def random_node(
    depth: int = 0,
    variables: Optional[List[str]] = None,
    allow_cmp: bool = True,
    allow_ternary: bool = True,
) -> Node:
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
        roll = random.random()
        # ~5% chance of ternary at depth 0-2, ~3% cmp as standalone
        if allow_ternary and depth <= 2 and roll < 0.08:
            return random_ternary_node(depth, variables)
        if allow_cmp and depth <= 3 and roll < 0.13:
            return random_cmp_node(depth, variables)
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
    individual_id: str = field(default_factory=_short_id)
    parent_ids: List[str] = field(default_factory=list)

    def to_expr(self) -> str:
        return self.tree.to_expr()

    def to_lambda(self, variables: List[str]) -> str:
        return self.tree.to_lambda(variables)

    def copy(self) -> 'Individual':
        new = Individual(
            tree=self.tree.copy(),
            fitness=self.fitness,
            individual_id=self.individual_id,  # preserve ID for lineage matching
            parent_ids=list(self.parent_ids),
        )
        return new

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
