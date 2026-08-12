"""Centralized metric calculations for the massive-MIMO simulator."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Capacity / rate
# ---------------------------------------------------------------------------

def spectral_efficiency(sinr_linear: np.ndarray | float) -> np.ndarray:
    """Shannon capacity  C = log2(1 + SINR)  in bit/s/Hz."""
    return np.log2(1.0 + np.asarray(sinr_linear, dtype=float))


def effective_spectral_efficiency(
    sinr_linear: np.ndarray | float,
    overhead_fraction: float,
) -> np.ndarray:
    """Rate after deducting a fractional pilot / training overhead."""
    return (1.0 - overhead_fraction) * spectral_efficiency(sinr_linear)


# ---------------------------------------------------------------------------
# Fairness & outage
# ---------------------------------------------------------------------------

def jain_fairness(rates: np.ndarray) -> float:
    """Jain's fairness index ∈ [1/n, 1]."""
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    if n == 0:
        return 0.0
    return float(rates.sum() ** 2 / (n * np.sum(rates ** 2) + 1e-30))


def outage_probability(
    sinr_linear: np.ndarray,
    threshold_db: float = 0.0,
) -> float:
    """Fraction of SINR samples below *threshold_db*."""
    threshold_linear = 10.0 ** (threshold_db / 10.0)
    return float(np.mean(np.asarray(sinr_linear) < threshold_linear))


# ---------------------------------------------------------------------------
# Beam alignment
# ---------------------------------------------------------------------------

def beam_alignment_accuracy(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Top-1 accuracy: fraction where predicted == ground truth beam."""
    return float(np.mean(np.asarray(predicted) == np.asarray(ground_truth)))


def top_k_accuracy(
    predicted_sets: list[np.ndarray],
    ground_truth: np.ndarray,
) -> float:
    """Fraction of steps where ground-truth beam is within the predicted set."""
    hits = 0
    for candidates, label in zip(predicted_sets, ground_truth):
        if label in candidates:
            hits += 1
    return hits / max(len(ground_truth), 1)


# ---------------------------------------------------------------------------
# Handover
# ---------------------------------------------------------------------------

def handover_failure_rate(
    ho_events: np.ndarray,
    ho_failures: np.ndarray,
) -> float:
    """Fraction of handover attempts that failed."""
    total = int(np.sum(ho_events))
    if total == 0:
        return 0.0
    return float(np.sum(ho_failures) / total)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, cdf) suitable for plt.plot(*cdf(data))."""
    sorted_vals = np.sort(np.asarray(values).ravel())
    probabilities = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, probabilities
