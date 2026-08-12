"""Experiment: analog vs hybrid vs digital beamforming comparison."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beamforming import compare_beamforming
from channel_model import geometric_channel
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Compare three BF architectures across antenna counts and user positions."""
    rng = np.random.default_rng(cfg.seed + 70)
    positions = rng.uniform([30.0, -50.0], [150.0, 50.0], size=(100, 2))

    # --- Part 1: fixed antennas, many user positions ---
    codebook, _ = dft_codebook(cfg.antennas, cfg.codebook_beams)
    analog_rates, digital_rates, hybrid_rates = [], [], []

    for pos in positions:
        ch = geometric_channel(pos, cfg, rng)
        result = compare_beamforming(ch, codebook, cfg)
        analog_rates.append(result["analog"])
        digital_rates.append(result["digital"])
        hybrid_rates.append(result["hybrid"])

    # --- Part 2: rate vs antenna count ---
    antenna_values = np.array([4, 8, 16, 32, 64])
    analog_vs_ant, digital_vs_ant, hybrid_vs_ant = [], [], []

    for N in antenna_values:
        local = replace(
            cfg,
            antennas=int(N),
            codebook_beams=int(N),
            num_rf_chains=min(cfg.num_rf_chains, int(N)),
        )
        cb, _ = dft_codebook(local.antennas, local.codebook_beams)
        a_list, d_list, h_list = [], [], []
        for pos in positions[:40]:
            ch = geometric_channel(pos, local, rng)
            res = compare_beamforming(ch, cb, local)
            a_list.append(res["analog"])
            d_list.append(res["digital"])
            h_list.append(res["hybrid"])
        analog_vs_ant.append(np.mean(a_list))
        digital_vs_ant.append(np.mean(d_list))
        hybrid_vs_ant.append(np.mean(h_list))

    return {
        # CDF data
        "analog_rates": np.array(analog_rates),
        "digital_rates": np.array(digital_rates),
        "hybrid_rates": np.array(hybrid_rates),
        # vs antenna count
        "antennas": antenna_values,
        "analog_vs_ant": np.array(analog_vs_ant),
        "digital_vs_ant": np.array(digital_vs_ant),
        "hybrid_vs_ant": np.array(hybrid_vs_ant),
    }
