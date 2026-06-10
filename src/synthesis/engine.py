"""
Orchestration engine for program synthesis.

Combines evolution, neural pre-filtering, and symbolic execution.
"""
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from ..evolution.population import Individual, create_population
from ..evolution.operators import subtree_crossover, point_mutation, subtree_mutation
from ..evolution.selection import select_parents, elitism
from ..neural.scorer import NeuralScorer
from ..neural.train_scorer import train_scorer
from ..symbolic.evaluator import evaluate_fitness, is_solved, TestCase

@dataclass
class SynthesisResult:
    program: Optional[str]
    generations_taken: int
    best_fitness: float
    fitness_curve: List[float]


class SynthesisEngine:
    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 50,
        mutation_rate: float = 0.3,
        use_neural_scorer: bool = True,
        variables: Optional[List[str]] = None
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.mutation_rate = mutation_rate
        self.use_neural_scorer = use_neural_scorer
        self.variables = variables or ['x']
        
        self.scorer = NeuralScorer() if use_neural_scorer else None
        
    def run(self, test_cases: List[TestCase]) -> SynthesisResult:
        """Run the synthesis loop."""
        population = create_population(self.pop_size, self.variables)
        fitness_curve = []
        
        # Buffer for training the neural scorer
        training_buffer = []
        
        best_overall = None
        
        for gen in range(self.max_gen):
            # 1. If using neural scorer, predict fitness to pre-filter
            if self.use_neural_scorer and len(training_buffer) > 0:
                predicted_fitness = self.scorer.predict(population)
                # Pair and sort by predicted fitness
                paired = list(zip(population, predicted_fitness))
                paired.sort(key=lambda x: x[1], reverse=True)
                # Keep top 50% for actual symbolic evaluation to save time
                eval_pool = [p[0] for p in paired[:self.pop_size // 2]]
                # Fill the rest with random to maintain diversity
                eval_pool.extend(create_population(self.pop_size - len(eval_pool), self.variables))
                population = eval_pool
            
            # 2. Symbolic Evaluation
            for ind in population:
                evaluate_fitness(ind, test_cases)
                training_buffer.append(ind.copy())
                
            # Track best
            best_in_gen = max(population, key=lambda i: i.fitness)
            if best_overall is None or best_in_gen.fitness > best_overall.fitness:
                best_overall = best_in_gen.copy()
                
            fitness_curve.append(best_in_gen.fitness)
            
            if is_solved(best_overall):
                return SynthesisResult(
                    program=best_overall.to_lambda(self.variables),
                    generations_taken=gen + 1,
                    best_fitness=best_overall.fitness,
                    fitness_curve=fitness_curve
                )
                
            # 3. Train Neural Scorer
            if self.use_neural_scorer and gen % 5 == 0 and len(training_buffer) >= 100:
                train_scorer(self.scorer, training_buffer, epochs=3)
                training_buffer = training_buffer[-200:] # Keep recent
                
            # 4. Selection & Reproduction
            next_pop = elitism(population, n_elite=2)
            
            parents = select_parents(population, n_pairs=(self.pop_size - len(next_pop)) // 2)
            for p1, p2 in parents:
                c1, c2 = subtree_crossover(p1, p2)
                
                if random.random() < self.mutation_rate:
                    c1 = point_mutation(c1, self.mutation_rate, self.variables)
                if random.random() < self.mutation_rate:
                    c2 = point_mutation(c2, self.mutation_rate, self.variables)
                    
                if random.random() < 0.1: # 10% chance for radical subtree mutation
                    c1 = subtree_mutation(c1, self.variables)
                if random.random() < 0.1:
                    c2 = subtree_mutation(c2, self.variables)
                    
                next_pop.extend([c1, c2])
                
            # Ensure pop size
            population = next_pop[:self.pop_size]

        return SynthesisResult(
            program=best_overall.to_lambda(self.variables) if best_overall else None,
            generations_taken=self.max_gen,
            best_fitness=best_overall.fitness if best_overall else 0.0,
            fitness_curve=fitness_curve
        )


class RandomSearchEngine:
    """Baseline for comparison."""
    def __init__(self, pop_size: int = 100, max_gen: int = 50, variables: Optional[List[str]] = None):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.variables = variables or ['x']
        
    def run(self, test_cases: List[TestCase]) -> SynthesisResult:
        best_overall = None
        fitness_curve = []
        
        for gen in range(self.max_gen):
            population = create_population(self.pop_size, self.variables)
            
            for ind in population:
                evaluate_fitness(ind, test_cases)
                if best_overall is None or ind.fitness > best_overall.fitness:
                    best_overall = ind.copy()
                    
            fitness_curve.append(best_overall.fitness)
            
            if is_solved(best_overall):
                return SynthesisResult(
                    program=best_overall.to_lambda(self.variables),
                    generations_taken=gen + 1,
                    best_fitness=best_overall.fitness,
                    fitness_curve=fitness_curve
                )
                
        return SynthesisResult(
            program=best_overall.to_lambda(self.variables) if best_overall else None,
            generations_taken=self.max_gen,
            best_fitness=best_overall.fitness if best_overall else 0.0,
            fitness_curve=fitness_curve
        )
