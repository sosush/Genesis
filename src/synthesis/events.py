"""
Event dataclasses for streaming synthesis progress to the frontend.

GenerationEvent is yielded by run_stream() after every generation and
serialized as JSON over the WebSocket connection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


@dataclass
class NodeSnapshot:
    """JSON-serializable snapshot of a single AST node (recursive)."""
    node_id: str
    ntype: str
    value: Any
    children: List['NodeSnapshot'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'ntype': self.ntype,
            'value': self.value,
            'children': [c.to_dict() for c in self.children],
        }


@dataclass
class IndividualSnapshot:
    """
    Snapshot of one individual for streaming.

    For the best individual and the top-10 + 5 random sample:
      tree is populated (full NodeSnapshot tree).
    For the remaining population (fitness-only):
      tree is None.
    """
    individual_id: str
    expr: str
    fitness: float
    parent_ids: List[str]
    rank: int                             # rank in population (0 = best)
    is_pruned_by_scorer: bool = False     # True if neural scorer cut this individual
    tree: Optional[NodeSnapshot] = None  # None for non-featured individuals

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'individual_id': self.individual_id,
            'expr': self.expr,
            'fitness': self.fitness,
            'parent_ids': self.parent_ids,
            'rank': self.rank,
            'is_pruned_by_scorer': self.is_pruned_by_scorer,
            'tree': self.tree.to_dict() if self.tree is not None else None,
        }
        return d


@dataclass
class GenerationEvent:
    """
    Emitted by run_stream() after every generation of evolution.

    population_snapshot contains:
      - Full IndividualSnapshots (with tree) for: best + top-10 + 5 random
      - Fitness-only IndividualSnapshots (tree=None) for the rest

    fitness_histogram: 10 bucketed counts of the population fitness
      distribution [0.0–0.1, 0.1–0.2, ..., 0.9–1.0]. Compact representation
      for the population scatter view.
    """
    run_id: str
    generation: int
    population_snapshot: List[IndividualSnapshot]
    best_individual: IndividualSnapshot
    fitness_curve: List[float]           # best fitness per generation so far
    fitness_histogram: List[int]         # 10 bins: counts per 0.1-width bucket
    neural_scorer_active: bool
    pruned_count: int                    # individuals cut by neural scorer this gen
    scorer_training_loss: Optional[float]
    event_type: Literal['evaluation', 'scorer_training', 'solved', 'timeout']
    engine_type: Literal['genesis', 'pure_evolutionary', 'random']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'generation': self.generation,
            'population_snapshot': [s.to_dict() for s in self.population_snapshot],
            'best_individual': self.best_individual.to_dict(),
            'fitness_curve': self.fitness_curve,
            'fitness_histogram': self.fitness_histogram,
            'neural_scorer_active': self.neural_scorer_active,
            'pruned_count': self.pruned_count,
            'scorer_training_loss': self.scorer_training_loss,
            'event_type': self.event_type,
            'engine_type': self.engine_type,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def node_to_snapshot(node: 'Node') -> NodeSnapshot:
    """Convert a Node to a NodeSnapshot recursively."""
    return NodeSnapshot(
        node_id=node.node_id,
        ntype=node.ntype,
        value=node.value,
        children=[node_to_snapshot(c) for c in node.children],
    )


def individual_to_snapshot(
    ind: 'Individual',
    rank: int,
    include_tree: bool = False,
    is_pruned: bool = False,
) -> IndividualSnapshot:
    """Convert an Individual to an IndividualSnapshot."""
    return IndividualSnapshot(
        individual_id=ind.individual_id,
        expr=ind.to_expr(),
        fitness=ind.fitness,
        parent_ids=ind.parent_ids,
        rank=rank,
        is_pruned_by_scorer=is_pruned,
        tree=node_to_snapshot(ind.tree) if include_tree else None,
    )


def build_fitness_histogram(fitnesses: List[float], bins: int = 10) -> List[int]:
    """Bucket fitness values into `bins` equal-width buckets over [0, 1]."""
    counts = [0] * bins
    for f in fitnesses:
        bucket = min(int(f * bins), bins - 1)
        counts[bucket] += 1
    return counts


def build_population_snapshot(
    population: List['Individual'],
    pruned_individuals: Optional[List['Individual']] = None,
    full_tree_count: int = 10,
    random_sample_count: int = 5,
) -> List[IndividualSnapshot]:
    """
    Build a mixed snapshot:
    - Best individual always gets a full tree (handled separately in GenerationEvent.best_individual)
    - top `full_tree_count` get full trees
    - `random_sample_count` random others get full trees
    - Everyone else: fitness-only (tree=None)
    """
    import random
    pruned_ids = set()
    if pruned_individuals:
        pruned_ids = {ind.individual_id for ind in pruned_individuals}

    # Population is already sorted by fitness descending (caller's responsibility)
    snapshots = []
    featured_indices = set(range(min(full_tree_count, len(population))))

    remaining = [i for i in range(len(population)) if i not in featured_indices]
    if remaining and random_sample_count > 0:
        sampled = random.sample(remaining, min(random_sample_count, len(remaining)))
        featured_indices.update(sampled)

    for rank, ind in enumerate(population):
        include_tree = rank in featured_indices
        snapshots.append(individual_to_snapshot(
            ind,
            rank=rank,
            include_tree=include_tree,
            is_pruned=ind.individual_id in pruned_ids,
        ))

    return snapshots
