"""Experiment: optimization convergence — greedy vs PSO vs GA."""

from __future__ import annotations

import numpy as np

from algorithms.power_allocation import fair_utility, grid_search_power
from algorithms.pso import pso_power
from algorithms.ga import ga_power_allocation
from algorithms.greedy import greedy_power_allocation
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Compare optimizers on the two-user power-allocation problem."""
    rng = np.random.default_rng(cfg.seed + 90)
    gains = np.array([1.1e-9, 2.8e-10])
    iterations = 50

    # Baselines
    equal_utility = fair_utility(0.5, gains, cfg.tx_power_w, cfg.noise_power_w)
    grid_fraction, grid_utility = grid_search_power(
        gains, cfg.tx_power_w, cfg.noise_power_w
    )

    # PSO
    pso_fraction, pso_history = pso_power(
        gains, cfg.tx_power_w, cfg.noise_power_w, rng, iterations=iterations
    )

    # GA
    ga_fraction, ga_history = ga_power_allocation(
        gains, cfg.tx_power_w, cfg.noise_power_w, rng, generations=iterations
    )

    # Greedy (power allocation adapted for comparison)
    greedy_powers, greedy_history = greedy_power_allocation(
        gains, cfg.tx_power_w, cfg.noise_power_w, steps=iterations
    )
    # Convert greedy sum-rate to fair-utility for comparison
    greedy_utilities = []
    for step in range(len(greedy_history)):
        frac = greedy_powers[0] * (step + 1) / (iterations * cfg.tx_power_w)
        frac = max(0.02, min(0.98, frac))
        greedy_utilities.append(fair_utility(frac, gains, cfg.tx_power_w, cfg.noise_power_w))

    return {
        "iterations": np.arange(1, iterations + 1),
        "pso_utility": pso_history,
        "ga_utility": ga_history,
        "greedy_utility": np.array(greedy_utilities),
        "equal_utility": np.array(equal_utility),
        "grid_utility": np.array(grid_utility),
        "grid_fraction": np.array(grid_fraction),
        "pso_fraction": np.array(pso_fraction),
        "ga_fraction": np.array(ga_fraction),
    }
