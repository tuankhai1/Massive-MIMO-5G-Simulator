"""Power-allocation algorithms: grid search, fair utility, water filling."""

from __future__ import annotations

import numpy as np


def fair_utility(
    power_fraction: float,
    gains: np.ndarray,
    total_power: float,
    noise: float,
) -> float:
    """Proportional-fair (log-sum-rate) utility for a two-user power split."""
    powers = np.array([power_fraction, 1.0 - power_fraction]) * total_power
    rates = np.log2(1.0 + powers * gains / noise)
    return float(np.log(rates + 1e-8).sum())


def grid_search_power(
    gains: np.ndarray,
    total_power: float,
    noise: float,
) -> tuple[float, float]:
    """Brute-force grid search for the best two-user power fraction.

    Returns (best_fraction, best_utility).
    """
    fractions = np.linspace(0.02, 0.98, 97)
    utilities = np.array(
        [fair_utility(f, gains, total_power, noise) for f in fractions]
    )
    idx = int(np.argmax(utilities))
    return float(fractions[idx]), float(utilities[idx])


def water_filling(
    gains: np.ndarray,
    total_power: float,
    noise: float,
) -> np.ndarray:
    """Classic water-filling power allocation for K parallel channels.

    Parameters
    ----------
    gains : (K,) channel power gains.
    total_power : total transmit power budget.
    noise : noise power per channel.

    Returns
    -------
    powers : (K,) optimal power allocation.
    """
    K = len(gains)
    snr_inv = noise / (np.asarray(gains, dtype=float) + 1e-30)
    # Sort channels by quality (best first)
    order = np.argsort(snr_inv)
    sorted_inv = snr_inv[order]

    # Iterative water-filling
    powers = np.zeros(K)
    active = K
    for _ in range(K):
        water_level = (total_power + np.sum(sorted_inv[:active])) / active
        alloc = water_level - sorted_inv[:active]
        if alloc.min() < 0:
            active -= 1
        else:
            powers[order[:active]] = alloc
            break
    else:
        # Allocate everything to the best channel
        powers[order[0]] = total_power

    return powers
