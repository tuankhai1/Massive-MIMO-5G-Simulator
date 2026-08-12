"""Experiment: SNR and achievable rate versus user angle and distance."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beamforming import snr_per_beam
from channel_model import geometric_channel
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Sweep user angle and distance, report best-beam SNR and rate."""
    rng = np.random.default_rng(cfg.seed + 10)
    angles_deg = np.linspace(-60, 60, 25)
    distances = np.array([30, 60, 100, 150, 200])
    codebook, _ = dft_codebook(cfg.antennas, cfg.codebook_beams)

    snr_map = np.zeros((len(distances), len(angles_deg)))
    rate_map = np.zeros_like(snr_map)

    for di, d in enumerate(distances):
        for ai, a in enumerate(angles_deg):
            theta = np.radians(a)
            pos = np.array([d * np.cos(theta), d * np.sin(theta)])
            ch = geometric_channel(pos, cfg, rng)
            snr = snr_per_beam(ch, codebook, cfg)
            best_snr = float(snr.max())
            snr_map[di, ai] = 10.0 * np.log10(max(best_snr, 1e-10))
            rate_map[di, ai] = float(np.log2(1.0 + best_snr))

    return {
        "angles_deg": angles_deg,
        "distances": distances,
        "snr_db": snr_map,
        "rate": rate_map,
    }
