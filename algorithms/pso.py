"""Particle Swarm Optimization for power allocation."""

from __future__ import annotations

import numpy as np

from algorithms.power_allocation import fair_utility


def pso_power(
    gains: np.ndarray,
    total_power: float,
    noise: float,
    rng: np.random.Generator,
    iterations: int = 50,
    particles: int = 24,
) -> tuple[float, np.ndarray]:
    """PSO optimizer for the two-user fair-utility power fraction.

    Returns (best_fraction, utility_history).
    """
    positions = rng.uniform(0.02, 0.98, particles)
    velocity = np.zeros(particles)
    personal_best = positions.copy()
    personal_value = np.array(
        [fair_utility(v, gains, total_power, noise) for v in positions]
    )
    global_index = int(np.argmax(personal_value))
    global_best = personal_best[global_index]
    history = []

    for _ in range(iterations):
        r1, r2 = rng.random((2, particles))
        velocity = (
            0.7 * velocity
            + 1.35 * r1 * (personal_best - positions)
            + 1.35 * r2 * (global_best - positions)
        )
        positions = np.clip(positions + velocity, 0.02, 0.98)
        values = np.array(
            [fair_utility(v, gains, total_power, noise) for v in positions]
        )
        update = values > personal_value
        personal_best[update] = positions[update]
        personal_value[update] = values[update]
        global_index = int(np.argmax(personal_value))
        global_best = personal_best[global_index]
        history.append(personal_value[global_index])

    return float(global_best), np.asarray(history)
