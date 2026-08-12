"""Experiment: beam-sweeping overhead versus mobility speed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beam_management import run_beam_management
from beamforming import snr_per_beam
from channel_model import geometric_channel, mobility_route
from config import SystemConfig
from ml.beam_predictor import MLPBeamPredictor
from ml.data_generator import beam_feature_matrix, generate_beam_dataset, split_episodes


def run(cfg: SystemConfig) -> dict:
    """Compare beam-management strategies at different UE speeds.

    Speeds in km/h: {3, 10, 30, 60, 120}.
    """
    speeds_kmh = np.array([3, 10, 30, 60, 120])
    speeds_mps = speeds_kmh / 3.6

    codebook, freqs = dft_codebook(cfg.antennas, cfg.codebook_beams)
    steps = min(cfg.route_steps, 100)
    dt = cfg.frame_s

    # Train once on independent trajectories; none of the mobility traces below
    # are used to fit the model.
    ml_data = generate_beam_dataset(
        cfg,
        num_episodes=cfg.ml_train_episodes + cfg.ml_test_episodes,
        steps_per_episode=cfg.ml_steps_per_episode,
    )
    ml_train_mask, _ = split_episodes(
        ml_data["episode_ids"], cfg.ml_train_episodes, cfg.seed + 501
    )
    ml_features = beam_feature_matrix(
        ml_data["noisy_positions"], ml_data["velocities"], ml_data["previous_beams"],
        cfg.codebook_beams,
    )
    ml_predictor = MLPBeamPredictor(
        hidden_units=cfg.ml_hidden_units,
        learning_rate=cfg.ml_learning_rate,
        epochs=cfg.ml_epochs,
        seed=cfg.seed + 512,
        num_classes=cfg.codebook_beams,
    ).fit(ml_features[ml_train_mask], ml_data["labels"][ml_train_mask])

    results = {
        "speeds_kmh": speeds_kmh,
        "exhaustive_rate": [],
        "hierarchical_rate": [],
        "location_topk_rate": [],
        "ml_topk_rate": [],
    }

    for speed in speeds_mps:
        rng = np.random.default_rng(cfg.seed + 50)
        positions, _, _ = mobility_route(steps, speed, dt, rng)
        channels = [geometric_channel(pos, cfg, rng) for pos in positions]

        # Jakes' 50%-correlation coherence time: T_c \u2248 0.423 / f_D.
        # A faster UE must repeat the same beam acquisition more frequently,
        # making pilot overhead explicitly speed-dependent.
        max_doppler_hz = speed / cfg.wavelength_m
        coherence_s = max(0.423 / max(max_doppler_hz, 1e-12), cfg.pilot_s * cfg.codebook_beams)

        local_cfg = replace(cfg, route_steps=steps)
        bm = run_beam_management(
            positions, channels, codebook, freqs, snr_per_beam, local_cfg,
            retraining_period_s=coherence_s,
            ml_predictor=ml_predictor,
        )

        results["exhaustive_rate"].append(float(np.mean(bm["exhaustive_rate"])))
        results["hierarchical_rate"].append(float(np.mean(bm["hierarchical_rate"])))
        results["location_topk_rate"].append(float(np.mean(bm["location_topk_rate"])))
        results["ml_topk_rate"].append(float(np.mean(bm["ml_topk_rate"])))

    for k in list(results.keys()):
        if k != "speeds_kmh":
            results[k] = np.array(results[k])

    return results
