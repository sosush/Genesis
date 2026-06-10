"""
Selection strategies for evolutionary search.

Strategies:
  - tournament_selection: select the fittest from a random subset
  - elitism: carry forward top-k individuals unchanged
  - select_parents: generate pairs of parents for crossover
"""
import random
from typing import List, Tuple

from .population import Individual


def tournament_selection(
    population: List[Individual],
    tournament_size: int = 3,
) -> Individual:
    """
    Tournament selection: sample `tournament_size` individuals at random
    and return the one with the highest fitness.
    """
    contestants = random.sample(population, min(tournament_size, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


def elitism(
    population: List[Individual],
    n_elite: int = 2,
) -> List[Individual]:
    """
    Return deep copies of the top `n_elite` individuals by fitness.
    These are inserted into the next generation unchanged.
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return [ind.copy() for ind in sorted_pop[:n_elite]]


def roulette_selection(population: List[Individual]) -> Individual:
    """
    Fitness-proportionate (roulette wheel) selection.
    Falls back to tournament if all fitnesses are zero.
    """
    total_fitness = sum(ind.fitness for ind in population)
    if total_fitness == 0:
        return tournament_selection(population)

    pick = random.uniform(0, total_fitness)
    cumulative = 0.0
    for ind in population:
        cumulative += ind.fitness
        if cumulative >= pick:
            return ind
    return population[-1]


def select_parents(
    population: List[Individual],
    n_pairs: int = 10,
    tournament_size: int = 3,
) -> List[Tuple[Individual, Individual]]:
    """
    Generate `n_pairs` (parent1, parent2) tuples using tournament selection.
    """
    pairs = []
    for _ in range(n_pairs):
        p1 = tournament_selection(population, tournament_size)
        p2 = tournament_selection(population, tournament_size)
        pairs.append((p1, p2))
    return pairs
