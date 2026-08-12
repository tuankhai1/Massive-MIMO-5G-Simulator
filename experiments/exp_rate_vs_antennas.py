"""Experiment: achievable rate versus number of antennas."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beamforming import effective_rate, snr_per_beam
from channel_model import geometric_channel
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Mean effective rate for antenna counts {4, 8, 16, 32, 64, 128}."""
    antenna_values = np.array([4, 8, 16, 32, 64, 128])
    rng = np.random.default_rng(cfg.seed + 30)
    positions = rng.uniform([35.0, -55.0], [150.0, 55.0], size=(120, 2))

    rates = []
    peak_snr = []
    for N in antenna_values:
        local = replace(cfg, antennas=int(N), codebook_beams=int(N))
        codebook, _ = dft_codebook(local.antennas, local.codebook_beams)
        r_samples, s_samples = [], []
        for pos in positions:
            ch = geometric_channel(pos, local, rng)
            snr = snr_per_beam(ch, codebook, local)
            r_samples.append(effective_rate(float(snr.max()), local.codebook_beams, local))
            s_samples.append(10.0 * np.log10(max(float(snr.max()), 1e-10)))
        rates.append(np.mean(r_samples))
        peak_snr.append(np.mean(s_samples))

    return {
        "antennas": antenna_values,
        "rate": np.asarray(rates),
        "peak_snr_db": np.asarray(peak_snr),
    }
