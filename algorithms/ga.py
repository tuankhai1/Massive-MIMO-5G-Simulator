"""Genetic Algorithm for power allocation."""

from __future__ import annotations

import numpy as np

from algorithms.power_allocation import fair_utility


def ga_power_allocation(
    gains: np.ndarray,
    total_power: float,
    noise: float,
    rng: np.random.Generator,
    generations: int = 50,
    population: int = 30,
    crossover_rate: float = 0.8,
    mutation_std: float = 0.05,
) -> tuple[float, np.ndarray]:
    """GA optimizer for the two-user fair-utility power fraction.

    Uses tournament selection, single-point crossover on the 1-D fraction
    (blend crossover), and Gaussian mutation.

    Returns (best_fraction, utility_history).
    """
    # Initialize population as power fractions in [0.02, 0.98]
    pop = rng.uniform(0.02, 0.98, population)
    fitness = np.array([fair_utility(v, gains, total_power, noise) for v in pop])
    history = []

    for _ in range(generations):
        # Tournament selection (size 3)
        new_pop = np.empty(population)
        for i in range(population):
            contestants = rng.choice(population, 3, replace=False)
            winner = contestants[int(np.argmax(fitness[contestants]))]
            new_pop[i] = pop[winner]

        # Crossover (blend)
        for i in range(0, population - 1, 2):
            if rng.random() < crossover_rate:
                alpha = rng.uniform(-0.25, 1.25)
                p1, p2 = new_pop[i], new_pop[i + 1]
                new_pop[i] = alpha * p1 + (1 - alpha) * p2
                new_pop[i + 1] = (1 - alpha) * p1 + alpha * p2

        # Mutation
        mutation = rng.normal(0, mutation_std, population)
        new_pop = np.clip(new_pop + mutation, 0.02, 0.98)

        # Evaluate
        new_fitness = np.array(
            [fair_utility(v, gains, total_power, noise) for v in new_pop]
        )

        # Elitism: keep the global best
        best_old = int(np.argmax(fitness))
        worst_new = int(np.argmin(new_fitness))
        if fitness[best_old] > new_fitness[worst_new]:
            new_pop[worst_new] = pop[best_old]
            new_fitness[worst_new] = fitness[best_old]

        pop = new_pop
        fitness = new_fitness
        history.append(float(fitness.max()))

    best_idx = int(np.argmax(fitness))
    return float(pop[best_idx]), np.asarray(history)
