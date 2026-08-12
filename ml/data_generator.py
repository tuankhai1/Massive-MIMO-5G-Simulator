"""Dataset generation for ML beam-prediction experiments."""

from __future__ import annotations

import numpy as np

from array_model import dft_codebook
from beamforming import snr_per_beam
from channel_model import geometric_channel, user_route
from config import SystemConfig


def generate_beam_dataset(
    cfg: SystemConfig,
    num_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate a supervised beam-prediction dataset.

    Each sample consists of:
        * noisy position (x, y)
        * velocity (vx, vy)
        * previous beam index
        * ground-truth best beam index

    Returns
    -------
    dict with keys: positions, noisy_positions, velocities,
                    previous_beams, labels
    """
    rng = np.random.default_rng(cfg.seed + 500)
    steps = num_samples or cfg.route_steps
    positions = user_route(steps)
    codebook, freqs = dft_codebook(cfg.antennas, cfg.codebook_beams)

    labels = np.empty(steps, dtype=int)
    for i, pos in enumerate(positions):
        ch = geometric_channel(pos, cfg, rng)
        snr = snr_per_beam(ch, codebook, cfg)
        labels[i] = int(np.argmax(snr))

    velocities = np.vstack([np.zeros(2), np.diff(positions, axis=0)])
    noisy_positions = positions + rng.normal(
        0, cfg.location_error_std_m, size=positions.shape
    )
    previous_beams = np.concatenate([[labels[0]], labels[:-1]])

    return {
        "positions": positions,
        "noisy_positions": noisy_positions,
        "velocities": velocities,
        "previous_beams": previous_beams,
        "labels": labels,
    }


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train / test by fraction (temporal split, not random)."""
    split = max(1, int(len(labels) * train_fraction))
    return features[:split], labels[:split], features[split:], labels[split:]


def add_noise_to_features(
    features: np.ndarray,
    noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise for data augmentation."""
    if rng is None:
        rng = np.random.default_rng(0)
    return features + rng.normal(0, noise_std, size=features.shape)
