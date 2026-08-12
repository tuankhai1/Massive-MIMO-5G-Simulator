"""Experiment: beam-sweeping overhead versus mobility speed."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beam_management import run_beam_management
from beamforming import snr_per_beam
from channel_model import geometric_channel, mobility_route
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Compare beam-management strategies at different UE speeds.

    Speeds in km/h: {3, 10, 30, 60, 120}.
    """
    speeds_kmh = np.array([3, 10, 30, 60, 120])
    speeds_mps = speeds_kmh / 3.6

    codebook, freqs = dft_codebook(cfg.antennas, cfg.codebook_beams)
    steps = min(cfg.route_steps, 100)
    dt = cfg.frame_s

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

        local_cfg = replace(cfg, route_steps=steps)
        bm = run_beam_management(
            positions, channels, codebook, freqs, snr_per_beam, local_cfg
        )

        results["exhaustive_rate"].append(float(np.mean(bm["exhaustive_rate"])))
        results["hierarchical_rate"].append(float(np.mean(bm["hierarchical_rate"])))
        results["location_topk_rate"].append(float(np.mean(bm["location_topk_rate"])))
        results["ml_topk_rate"].append(float(np.mean(bm["ml_topk_rate"])))

    for k in list(results.keys()):
        if k != "speeds_kmh":
            results[k] = np.array(results[k])

    return results
