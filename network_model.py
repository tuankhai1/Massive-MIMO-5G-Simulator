"""Multi-cell network model with scheduling, interference, and SINR.

Provides cell association, round-robin / proportional-fair scheduling,
and multi-user SINR computation.
"""

from __future__ import annotations

import numpy as np


# ===================================================================
# Cell snapshot & association (legacy, preserved)
# ===================================================================

def multi_cell_snapshot(
    rng: np.random.Generator,
    users: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random multi-cell snapshot.

    Returns (base_stations, user_positions, gain_matrix).
    """
    base_stations = np.array([[0.0, 0.0], [220.0, 0.0], [110.0, 180.0]])
    user_positions = rng.uniform([-20.0, -20.0], [240.0, 200.0], size=(users, 2))
    distance = np.linalg.norm(
        user_positions[:, None, :] - base_stations[None, :, :], axis=2
    )
    gain = 1.0 / np.maximum(distance, 10.0) ** 3.1
    return base_stations, user_positions, gain


def association_metrics(
    gain: np.ndarray,
    noise: float = 2e-12,
) -> dict[str, float]:
    """Compute mean rate, outage, and Jain fairness for a gain matrix."""
    serving = np.argmax(gain, axis=1)
    desired = gain[np.arange(gain.shape[0]), serving]
    interference = gain.sum(axis=1) - desired
    sinr = desired / (noise + interference)
    rates = np.log2(1.0 + sinr)
    load = np.bincount(serving, minlength=gain.shape[1])
    user_rates = rates / np.maximum(load[serving], 1)
    fairness = float(
        user_rates.sum() ** 2 / (len(user_rates) * np.sum(user_rates ** 2) + 1e-30)
    )
    return {
        "mean_rate": float(user_rates.mean()),
        "outage": float(np.mean(sinr < 1.0)),
        "fairness": fairness,
    }


# ===================================================================
# Scheduling
# ===================================================================

def round_robin_schedule(
    num_users: int,
    num_slots: int,
) -> np.ndarray:
    """Return a (num_slots,) array of user indices scheduled per slot."""
    return np.arange(num_slots) % num_users


def proportional_fair_schedule(
    average_rates: np.ndarray,
    instantaneous_rates: np.ndarray,
) -> int:
    """Return the user index with the highest PF metric.

    PF metric = instantaneous_rate / average_rate.
    """
    metric = instantaneous_rates / (average_rates + 1e-12)
    return int(np.argmax(metric))


def run_proportional_fair(
    instantaneous_rate_matrix: np.ndarray,
    alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Run proportional-fair scheduling over T slots for K users.

    Parameters
    ----------
    instantaneous_rate_matrix : (T, K) — per-slot per-user achievable rates.
    alpha : exponential moving average factor.

    Returns
    -------
    schedule : (T,) — user scheduled per slot.
    avg_rates : (K,) — final average rates.
    """
    T, K = instantaneous_rate_matrix.shape
    avg = np.ones(K) * 0.1
    schedule = np.empty(T, dtype=int)
    for t in range(T):
        user = proportional_fair_schedule(avg, instantaneous_rate_matrix[t])
        schedule[t] = user
        avg[user] = (1 - alpha) * avg[user] + alpha * instantaneous_rate_matrix[t, user]
    return schedule, avg


# ===================================================================
# Multi-user SINR with inter-cell interference
# ===================================================================

def multi_user_sinr(
    user_channels: np.ndarray,
    precoder: np.ndarray,
    inter_cell_interference: np.ndarray,
    noise_power: float,
    total_power: float,
) -> np.ndarray:
    """Per-user SINR including both intra-cell and inter-cell interference.

    Parameters
    ----------
    user_channels : (K, Nt) — stacked per-user channel vectors.
    precoder : (Nt, K) — per-user precoding columns.
    inter_cell_interference : (K,) — per-user ICI power.
    noise_power : float
    total_power : float

    Returns
    -------
    sinr : (K,)
    """
    K = user_channels.shape[0]
    power_per_user = total_power / K
    effective = user_channels @ precoder  # (K, K)
    signal = power_per_user * np.abs(np.diag(effective)) ** 2
    intra_cell = power_per_user * (
        np.sum(np.abs(effective) ** 2, axis=1) - np.abs(np.diag(effective)) ** 2
    )
    return signal / (intra_cell + inter_cell_interference + noise_power)
