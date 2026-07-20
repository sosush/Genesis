"""
Orchestration engine for program synthesis.

Combines evolution, neural pre-filtering, and symbolic execution.

Three engine types:
  - SynthesisEngine:        Full neuro-symbolic (Genesis)
  - PureEvolutionaryEngine: Genetic programming, no neural scorer
  - RandomSearchEngine:     Baseline random generation per generation

All three expose:
  - run(test_cases) -> SynthesisResult      (blocking, backward-compatible)
  - run_stream(test_cases, run_id) -> Iterator[GenerationEvent]  (streaming)
"""
import time
import random
import uuid
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict, Any, Literal

from ..evolution.population import Individual, create_population
from ..evolution.operators import subtree_crossover, point_mutation, subtree_mutation
from ..evolution.selection import select_parents, elitism
from ..neural.scorer import NeuralScorer
from ..neural.train_scorer import train_scorer
from ..symbolic.evaluator import evaluate_fitness, is_solved, TestCase
from .events import (
    GenerationEvent, IndividualSnapshot,
    individual_to_snapshot, build_fitness_histogram, build_population_snapshot,
)


@dataclass
class SynthesisResult:
    program: Optional[str]
    generations_taken: int
    best_fitness: float
    fitness_curve: List[float]
    engine_type: str = 'genesis'
    run_id: Optional[str] = None


# ------------------------------------------------------------------
# Genesis (Neuro-Symbolic) Engine
# ------------------------------------------------------------------

class SynthesisEngine:
    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 50,
        mutation_rate: float = 0.3,
        use_neural_scorer: bool = True,
        variables: Optional[List[str]] = None,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.use_neural_scorer = use_neural_scorer
        self.variables = variables or ['x']

        self.scorer = NeuralScorer() if use_neural_scorer else None

    def run_stream(
        self,
        test_cases: List[TestCase],
        run_id: Optional[str] = None,
    ) -> Iterator[GenerationEvent]:
        """
        Streaming generator: yields a GenerationEvent after every generation.
        Safe to run in a ThreadPoolExecutor (no asyncio inside).
        """
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]

        population = create_population(self.pop_size, self.variables)
        fitness_curve: List[float] = []
        training_buffer: List[Individual] = []
        best_overall: Optional[Individual] = None
        scorer_training_loss: Optional[float] = None

        for gen in range(self.max_gen):
            pruned_individuals: List[Individual] = []
            neural_scorer_active = False
            event_type: Literal['evaluation', 'scorer_training', 'solved', 'timeout'] = 'evaluation'

            # 1. Neural pre-filter (if scorer has been trained)
            if self.use_neural_scorer and self.scorer is not None and len(training_buffer) > 0:
                neural_scorer_active = True
                predicted_fitness = self.scorer.predict(population)
                paired = list(zip(population, predicted_fitness))
                paired.sort(key=lambda x: x[1], reverse=True)
                keep_n = self.pop_size // 2
                eval_pool = [p[0] for p in paired[:keep_n]]
                pruned_individuals = [p[0] for p in paired[keep_n:]]
                # Fill with fresh random to maintain diversity
                eval_pool.extend(create_population(self.pop_size - len(eval_pool), self.variables))
                population = eval_pool

            # 2. Symbolic evaluation
            for ind in population:
                evaluate_fitness(ind, test_cases)
                training_buffer.append(ind.copy())

            # Track best
            best_in_gen = max(population, key=lambda i: i.fitness)
            if best_overall is None or best_in_gen.fitness > best_overall.fitness:
                best_overall = best_in_gen.copy()

            fitness_curve.append(best_in_gen.fitness)

            # 3. Train neural scorer periodically
            if self.use_neural_scorer and self.scorer is not None:
                if gen % 5 == 0 and len(training_buffer) >= 100:
                    scorer_training_loss = train_scorer(self.scorer, training_buffer, epochs=3)
                    training_buffer = training_buffer[-200:]
                    event_type = 'scorer_training'

            # 4. Check solved
            if is_solved(best_overall):
                event_type = 'solved'

            # Build and yield event
            sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
            fitnesses = [ind.fitness for ind in sorted_pop]
            population_snapshot = build_population_snapshot(sorted_pop, pruned_individuals)

            yield GenerationEvent(
                run_id=run_id,
                generation=gen,
                population_snapshot=population_snapshot,
                best_individual=individual_to_snapshot(best_overall, rank=0, include_tree=True),
                fitness_curve=list(fitness_curve),
                fitness_histogram=build_fitness_histogram(fitnesses),
                neural_scorer_active=neural_scorer_active,
                pruned_count=len(pruned_individuals),
                scorer_training_loss=scorer_training_loss,
                event_type=event_type,
                engine_type='genesis',
            )

            if event_type == 'solved':
                return

            # 5. Selection & Reproduction
            next_pop = elitism(population, n_elite=2)
            parents = select_parents(population, n_pairs=(self.pop_size - len(next_pop)) // 2)
            for p1, p2 in parents:
                c1, c2 = subtree_crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    c1 = point_mutation(c1, self.mutation_rate, self.variables)
                if random.random() < self.mutation_rate:
                    c2 = point_mutation(c2, self.mutation_rate, self.variables)
                if random.random() < 0.1:
                    c1 = subtree_mutation(c1, self.variables)
                if random.random() < 0.1:
                    c2 = subtree_mutation(c2, self.variables)
                next_pop.extend([c1, c2])
            population = next_pop[:self.pop_size]

        # Timeout event
        if best_overall:
            sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
            fitnesses = [ind.fitness for ind in sorted_pop]
            yield GenerationEvent(
                run_id=run_id,
                generation=self.max_gen - 1,
                population_snapshot=build_population_snapshot(sorted_pop),
                best_individual=individual_to_snapshot(best_overall, rank=0, include_tree=True),
                fitness_curve=list(fitness_curve),
                fitness_histogram=build_fitness_histogram(fitnesses),
                neural_scorer_active=neural_scorer_active,
                pruned_count=0,
                scorer_training_loss=scorer_training_loss,
                event_type='timeout',
                engine_type='genesis',
            )

    def run(self, test_cases: List[TestCase]) -> SynthesisResult:
        """Blocking wrapper around run_stream() for backward compatibility."""
        run_id = uuid.uuid4().hex[:12]
        last_event = None
        for event in self.run_stream(test_cases, run_id):
            last_event = event

        if last_event is None:
            return SynthesisResult(
                program=None, generations_taken=0,
                best_fitness=0.0, fitness_curve=[],
                engine_type='genesis', run_id=run_id
            )

        best = last_event.best_individual
        return SynthesisResult(
            program=best.expr,
            generations_taken=last_event.generation + 1,
            best_fitness=best.fitness,
            fitness_curve=last_event.fitness_curve,
            engine_type='genesis',
            run_id=run_id,
        )


# ------------------------------------------------------------------
# Pure Evolutionary Engine (no neural scorer)
# ------------------------------------------------------------------

class PureEvolutionaryEngine:
    """Genetic programming without neural pre-filtering."""

    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 50,
        mutation_rate: float = 0.3,
        variables: Optional[List[str]] = None,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.variables = variables or ['x']

    def run_stream(
        self,
        test_cases: List[TestCase],
        run_id: Optional[str] = None,
    ) -> Iterator[GenerationEvent]:
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]

        population = create_population(self.pop_size, self.variables)
        fitness_curve: List[float] = []
        best_overall: Optional[Individual] = None
        event_type: Literal['evaluation', 'scorer_training', 'solved', 'timeout'] = 'evaluation'

        for gen in range(self.max_gen):
            event_type = 'evaluation'

            for ind in population:
                evaluate_fitness(ind, test_cases)

            best_in_gen = max(population, key=lambda i: i.fitness)
            if best_overall is None or best_in_gen.fitness > best_overall.fitness:
                best_overall = best_in_gen.copy()

            fitness_curve.append(best_in_gen.fitness)

            if is_solved(best_overall):
                event_type = 'solved'

            sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
            fitnesses = [ind.fitness for ind in sorted_pop]

            yield GenerationEvent(
                run_id=run_id,
                generation=gen,
                population_snapshot=build_population_snapshot(sorted_pop),
                best_individual=individual_to_snapshot(best_overall, rank=0, include_tree=True),
                fitness_curve=list(fitness_curve),
                fitness_histogram=build_fitness_histogram(fitnesses),
                neural_scorer_active=False,
                pruned_count=0,
                scorer_training_loss=None,
                event_type=event_type,
                engine_type='pure_evolutionary',
            )

            if event_type == 'solved':
                return

            next_pop = elitism(population, n_elite=2)
            parents = select_parents(population, n_pairs=(self.pop_size - len(next_pop)) // 2)
            for p1, p2 in parents:
                c1, c2 = subtree_crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    c1 = point_mutation(c1, self.mutation_rate, self.variables)
                if random.random() < self.mutation_rate:
                    c2 = point_mutation(c2, self.mutation_rate, self.variables)
                if random.random() < 0.1:
                    c1 = subtree_mutation(c1, self.variables)
                if random.random() < 0.1:
                    c2 = subtree_mutation(c2, self.variables)
                next_pop.extend([c1, c2])
            population = next_pop[:self.pop_size]

        # Timeout
        sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
        fitnesses = [ind.fitness for ind in sorted_pop]
        yield GenerationEvent(
            run_id=run_id,
            generation=self.max_gen - 1,
            population_snapshot=build_population_snapshot(sorted_pop),
            best_individual=individual_to_snapshot(best_overall, rank=0, include_tree=True),
            fitness_curve=list(fitness_curve),
            fitness_histogram=build_fitness_histogram(fitnesses),
            neural_scorer_active=False,
            pruned_count=0,
            scorer_training_loss=None,
            event_type='timeout',
            engine_type='pure_evolutionary',
        )

    def run(self, test_cases: List[TestCase]) -> SynthesisResult:
        run_id = uuid.uuid4().hex[:12]
        last_event = None
        for event in self.run_stream(test_cases, run_id):
            last_event = event
        if last_event is None:
            return SynthesisResult(
                program=None, generations_taken=0, best_fitness=0.0,
                fitness_curve=[], engine_type='pure_evolutionary', run_id=run_id
            )
        best = last_event.best_individual
        return SynthesisResult(
            program=best.expr,
            generations_taken=last_event.generation + 1,
            best_fitness=best.fitness,
            fitness_curve=last_event.fitness_curve,
            engine_type='pure_evolutionary',
            run_id=run_id,
        )


# ------------------------------------------------------------------
# Random Search Engine
# ------------------------------------------------------------------

class RandomSearchEngine:
    """Baseline: generate a fresh random population every generation."""

    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 50,
        variables: Optional[List[str]] = None,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.variables = variables or ['x']

    def run_stream(
        self,
        test_cases: List[TestCase],
        run_id: Optional[str] = None,
    ) -> Iterator[GenerationEvent]:
        if run_id is None:
            run_id = uuid.uuid4().hex[:12]

        best_overall: Optional[Individual] = None
        fitness_curve: List[float] = []
        event_type: Literal['evaluation', 'scorer_training', 'solved', 'timeout'] = 'evaluation'

        for gen in range(self.max_gen):
            event_type = 'evaluation'
            population = create_population(self.pop_size, self.variables)

            for ind in population:
                evaluate_fitness(ind, test_cases)
                if best_overall is None or ind.fitness > best_overall.fitness:
                    best_overall = ind.copy()

            fitness_curve.append(best_overall.fitness)

            if is_solved(best_overall):
                event_type = 'solved'

            sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
            fitnesses = [ind.fitness for ind in sorted_pop]

            yield GenerationEvent(
                run_id=run_id,
                generation=gen,
                population_snapshot=build_population_snapshot(sorted_pop),
                best_individual=individual_to_snapshot(best_overall, rank=0, include_tree=True),
                fitness_curve=list(fitness_curve),
                fitness_histogram=build_fitness_histogram(fitnesses),
                neural_scorer_active=False,
                pruned_count=0,
                scorer_training_loss=None,
                event_type=event_type,
                engine_type='random',
            )

            if event_type == 'solved':
                return

        # Timeout
        sorted_pop = sorted(population, key=lambda i: i.fitness, reverse=True)
        fitnesses = [ind.fitness for ind in sorted_pop]
        yield GenerationEvent(
            run_id=run_id,
            generation=self.max_gen - 1,
            population_snapshot=build_population_snapshot(sorted_pop),
            best_individual=individual_to_snapshot(best_overall, rank=0, include_tree=True),
            fitness_curve=list(fitness_curve),
            fitness_histogram=build_fitness_histogram(fitnesses),
            neural_scorer_active=False,
            pruned_count=0,
            scorer_training_loss=None,
            event_type='timeout',
            engine_type='random',
        )

    def run(self, test_cases: List[TestCase]) -> SynthesisResult:
        run_id = uuid.uuid4().hex[:12]
        last_event = None
        for event in self.run_stream(test_cases, run_id):
            last_event = event
        if last_event is None:
            return SynthesisResult(
                program=None, generations_taken=0, best_fitness=0.0,
                fitness_curve=[], engine_type='random', run_id=run_id
            )
        best = last_event.best_individual
        return SynthesisResult(
            program=best.expr,
            generations_taken=last_event.generation + 1,
            best_fitness=best.fitness,
            fitness_curve=last_event.fitness_curve,
            engine_type='random',
            run_id=run_id,
        )
