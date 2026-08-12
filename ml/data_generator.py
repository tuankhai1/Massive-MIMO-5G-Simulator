"""Dataset generation and feature encoding for beam-prediction experiments."""

from __future__ import annotations

import numpy as np

from array_model import dft_codebook
from beamforming import snr_per_beam
from channel_model import geometric_channel
from config import SystemConfig


def generate_beam_dataset(
    cfg: SystemConfig,
    num_samples: int | None = None,
    num_episodes: int | None = None,
    steps_per_episode: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate a trajectory-based supervised beam-prediction dataset.

    Each episode is an independent mobile-UE trajectory.  Keeping episode IDs
    allows evaluation on trajectories never seen during training, avoiding the
    optimistic temporal leakage of splitting one continuous route in half.

    Each sample consists of:
        * noisy position (x, y)
        * velocity (vx, vy)
        * previous beam index
        * ground-truth best beam index

    Returns
    -------
    dict with keys: positions, noisy_positions, velocities, previous_beams,
                    labels and per-beam SNR values.
    """
    rng = np.random.default_rng(cfg.seed + 500)
    episodes = num_episodes or cfg.ml_train_episodes + cfg.ml_test_episodes
    if num_samples is not None:
        steps_per_episode = max(8, num_samples // max(episodes, 1))
    steps = steps_per_episode or cfg.ml_steps_per_episode
    total_samples = episodes * steps

    positions = np.empty((total_samples, 2))
    velocities = np.empty((total_samples, 2))
    episode_ids = np.repeat(np.arange(episodes), steps)
    is_episode_start = np.zeros(total_samples, dtype=bool)

    for episode in range(episodes):
        # Draw a UE in a 120-degree sector served by the base station at origin.
        radius = rng.uniform(30.0, 150.0)
        azimuth = rng.uniform(-np.deg2rad(60.0), np.deg2rad(60.0))
        pos = radius * np.array([np.cos(azimuth), np.sin(azimuth)])
        speed = rng.uniform(1.0, 22.0)
        heading = rng.uniform(-np.pi, np.pi)

        for step in range(steps):
            idx = episode * steps + step
            positions[idx] = pos
            velocities[idx] = speed * np.array([np.cos(heading), np.sin(heading)])
            is_episode_start[idx] = step == 0

            # Smooth motion with reflection inside the service sector bounds.
            heading += rng.normal(0.0, 0.08)
            candidate = pos + velocities[idx] * cfg.frame_s
            if not (25.0 <= candidate[0] <= 165.0 and -115.0 <= candidate[1] <= 115.0):
                heading += np.pi + rng.normal(0.0, 0.15)
                candidate = pos + speed * np.array([np.cos(heading), np.sin(heading)]) * cfg.frame_s
            pos = candidate

    codebook, _ = dft_codebook(cfg.antennas, cfg.codebook_beams)

    labels = np.empty(total_samples, dtype=int)
    beam_snr = np.empty((total_samples, cfg.codebook_beams))
    for i, pos in enumerate(positions):
        ch = geometric_channel(pos, cfg, rng)
        snr = snr_per_beam(ch, codebook, cfg)
        beam_snr[i] = snr
        labels[i] = int(np.argmax(snr))

    noisy_positions = positions + rng.normal(
        0, cfg.location_error_std_m, size=positions.shape
    )
    previous_beams = labels.copy()
    for episode in range(episodes):
        first = episode * steps
        previous_beams[first + 1:first + steps] = labels[first:first + steps - 1]

    return {
        "positions": positions,
        "noisy_positions": noisy_positions,
        "velocities": velocities,
        "previous_beams": previous_beams,
        "labels": labels,
        "beam_snr": beam_snr,
        "episode_ids": episode_ids,
        "is_episode_start": is_episode_start,
    }


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train / test by fraction (temporal split, not random)."""
    split = max(1, int(len(labels) * train_fraction))
    return features[:split], labels[:split], features[split:], labels[split:]


def split_episodes(
    episode_ids: np.ndarray,
    train_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test masks for non-overlapping randomly ordered episodes."""
    unique_ids = np.unique(episode_ids)
    if not 0 < train_episodes < len(unique_ids):
        raise ValueError("train_episodes must be between 1 and the episode count - 1")
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(unique_ids)
    train_ids = ordered[:train_episodes]
    train_mask = np.isin(episode_ids, train_ids)
    return train_mask, ~train_mask


def beam_feature_matrix(
    noisy_positions: np.ndarray,
    velocities: np.ndarray,
    previous_beams: np.ndarray,
    num_beams: int,
    *,
    include_location: bool = True,
    include_motion: bool = True,
    include_history: bool = True,
) -> np.ndarray:
    """Create scale-friendly dense features with one-hot beam history.

    A beam index is categorical rather than ordinal; one-hot encoding prevents
    the learner from treating beams 0 and 1 as intrinsically closer than, for
    example, beams 0 and 31.
    """
    parts: list[np.ndarray] = []
    if include_location:
        parts.append(np.asarray(noisy_positions, dtype=float))
    if include_motion:
        parts.append(np.asarray(velocities, dtype=float))
    if include_history:
        indices = np.clip(np.asarray(previous_beams, dtype=int), 0, num_beams - 1)
        parts.append(np.eye(num_beams, dtype=float)[indices])
    if not parts:
        raise ValueError("At least one feature group must be enabled")
    return np.column_stack(parts)


def add_noise_to_features(
    features: np.ndarray,
    noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise for data augmentation."""
    if rng is None:
        rng = np.random.default_rng(0)
    return features + rng.normal(0, noise_std, size=features.shape)
