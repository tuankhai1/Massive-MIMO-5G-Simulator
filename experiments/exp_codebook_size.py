"""Experiment: rate versus beam-codebook size."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beamforming import effective_rate, snr_per_beam
from channel_model import geometric_channel
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Trade-off between codebook resolution and pilot overhead."""
    sizes = np.array([8, 16, 32, 48, 64, 96])
    rng = np.random.default_rng(cfg.seed + 40)
    positions = rng.uniform([30.0, -55.0], [150.0, 55.0], size=(100, 2))

    rates, peak_snr_db = [], []
    for size in sizes:
        local = replace(cfg, codebook_beams=int(size))
        codebook, _ = dft_codebook(local.antennas, local.codebook_beams)
        r_list, s_list = [], []
        for pos in positions:
            snr = snr_per_beam(geometric_channel(pos, local, rng), codebook, local)
            r_list.append(effective_rate(float(snr.max()), local.codebook_beams, local))
            s_list.append(10.0 * np.log10(max(float(snr.max()), 1e-10)))
        rates.append(np.mean(r_list))
        peak_snr_db.append(np.mean(s_list))

    return {
        "codebook_size": sizes,
        "rate": np.asarray(rates),
        "peak_snr_db": np.asarray(peak_snr_db),
    }
