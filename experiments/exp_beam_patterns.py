"""Experiment: beam patterns for different ULA sizes."""

from __future__ import annotations

import numpy as np

from array_model import dft_codebook, steering_vector
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Compute beam patterns for ULA sizes {8, 16, 32, 64}."""
    antenna_sizes = [8, 16, 32, 64]
    angles = np.linspace(-np.pi / 2, np.pi / 2, 600)
    patterns: dict[int, np.ndarray] = {}

    for N in antenna_sizes:
        codebook, _ = dft_codebook(N, N)
        # Pick 5 evenly spaced beams
        chosen = np.linspace(0, N - 1, min(5, N), dtype=int)
        scan = np.column_stack(
            [steering_vector(N, np.sin(a)) for a in angles]
        )
        beam_patterns = []
        for idx in chosen:
            power = np.abs(codebook[:, idx].conj() @ scan) ** 2
            db = 10.0 * np.log10(np.maximum(power / power.max(), 1e-6))
            beam_patterns.append(db)
        patterns[N] = np.array(beam_patterns)

    return {
        "angles_deg": np.degrees(angles),
        "antenna_sizes": antenna_sizes,
        "patterns": patterns,
    }
