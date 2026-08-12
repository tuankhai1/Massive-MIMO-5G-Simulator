"""Greedy beam selection and power allocation."""

from __future__ import annotations

import numpy as np


def greedy_beam_selection(
    snr_matrix: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively select beams by marginal rate gain.

    Parameters
    ----------
    snr_matrix : (num_users, num_beams) — per-user per-beam SNR.
    budget : max number of beams to select.

    Returns
    -------
    selected_beams : (budget,) — chosen beam indices.
    cumulative_rate : (budget,) — sum-rate after each beam addition.
    """
    num_users, num_beams = snr_matrix.shape
    available = set(range(num_beams))
    selected: list[int] = []
    cum_rate: list[float] = []

    for _ in range(min(budget, num_beams)):
        best_gain = -np.inf
        best_beam = -1
        # Current best per-user SNR from already-selected beams
        if selected:
            current_best = np.max(snr_matrix[:, selected], axis=1)
        else:
            current_best = np.zeros(num_users)

        for b in available:
            new_best = np.maximum(current_best, snr_matrix[:, b])
            gain = float(np.sum(np.log2(1.0 + new_best)))
            if gain > best_gain:
                best_gain = gain
                best_beam = b

        selected.append(best_beam)
        available.discard(best_beam)
        cum_rate.append(best_gain)

    return np.array(selected), np.array(cum_rate)


def greedy_power_allocation(
    gains: np.ndarray,
    total_power: float,
    noise: float,
    steps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy water-filling-like power allocation for K users.

    At each step, allocate a power increment to the user with the largest
    marginal rate increase.

    Returns
    -------
    powers : (K,) final power allocation.
    history : (steps,) sum-rate at each step.
    """
    K = len(gains)
    powers = np.zeros(K)
    increment = total_power / steps
    history = []

    for _ in range(steps):
        marginal = gains / (noise + powers * gains)  # derivative of log2(1+P·g/n)
        best_user = int(np.argmax(marginal))
        powers[best_user] += increment
        rate = float(np.sum(np.log2(1.0 + powers * gains / noise)))
        history.append(rate)

    return powers, np.array(history)


def greedy_fair_power_allocation(
    gains: np.ndarray,
    total_power: float,
    noise: float,
    steps: int = 50,
) -> tuple[float, np.ndarray]:
    """Greedy power allocation optimising proportional-fair (log-sum-rate) utility.

    Unlike :func:`greedy_power_allocation` which maximises sum-rate,
    this function maximises ``fair_utility`` — the same objective used
    by PSO and GA — so that all optimizers can be compared on the same
    objective.

    Designed for the two-user power-fraction problem: at each step try
    shifting the fraction by ±delta and keep the direction that improves
    the fair utility.

    Returns
    -------
    best_fraction : float
        Final power fraction for user 0.
    history : (steps,) array
        Fair-utility value at each step.
    """
    from algorithms.power_allocation import fair_utility

    fraction = 0.5  # start at equal power
    delta = 0.4 / steps  # step size shrinks with resolution
    history = []

    for i in range(steps):
        current = fair_utility(fraction, gains, total_power, noise)
        # Try both directions
        f_up = min(0.98, fraction + delta)
        f_down = max(0.02, fraction - delta)
        u_up = fair_utility(f_up, gains, total_power, noise)
        u_down = fair_utility(f_down, gains, total_power, noise)

        if u_up > current and u_up >= u_down:
            fraction = f_up
            history.append(u_up)
        elif u_down > current:
            fraction = f_down
            history.append(u_down)
        else:
            history.append(current)

    return fraction, np.array(history)
