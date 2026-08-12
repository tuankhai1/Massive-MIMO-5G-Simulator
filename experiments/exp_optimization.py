"""Experiment: optimization convergence — greedy vs PSO vs GA."""

from __future__ import annotations

import numpy as np

from algorithms.power_allocation import fair_utility, grid_search_power
from algorithms.pso import pso_power
from algorithms.ga import ga_power_allocation
from algorithms.greedy import greedy_fair_power_allocation
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Compare optimizers on the two-user power-allocation problem.

    All three optimizers (PSO, GA, Greedy) directly optimise the same
    proportional-fair (log-sum-rate) utility, enabling fair comparison.
    """
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

    # Greedy — directly optimises fair_utility (same objective as PSO/GA)
    greedy_fraction, greedy_history = greedy_fair_power_allocation(
        gains, cfg.tx_power_w, cfg.noise_power_w, steps=iterations
    )

    return {
        "iterations": np.arange(1, iterations + 1),
        "pso_utility": pso_history,
        "ga_utility": ga_history,
        "greedy_utility": greedy_history,
        "equal_utility": np.array(equal_utility),
        "grid_utility": np.array(grid_utility),
        "grid_fraction": np.array(grid_fraction),
        "pso_fraction": np.array(pso_fraction),
        "ga_fraction": np.array(ga_fraction),
        "greedy_fraction": np.array(greedy_fraction),
    }
