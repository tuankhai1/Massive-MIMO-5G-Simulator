"""Experiment: BER versus SNR for OFDM with and without beamforming."""

from __future__ import annotations

import numpy as np

from config import SystemConfig
from phy import ofdm_qpsk_ber, ofdm_ber_with_beamforming


def run(cfg: SystemConfig) -> dict:
    """Compare OFDM BER curves with and without beamforming."""
    rng_no_bf = np.random.default_rng(cfg.seed + 20)
    rng_bf = np.random.default_rng(cfg.seed + 21)

    snr_db = np.arange(-2, 22, 2)
    ber_no_bf = ofdm_qpsk_ber(snr_db, rng_no_bf, frames=120, subcarriers=cfg.num_subcarriers)
    ber_bf = ofdm_ber_with_beamforming(
        snr_db, rng_bf,
        num_antennas=cfg.antennas,
        frames=80,
        subcarriers=cfg.num_subcarriers,
        cp_length=cfg.cp_length,
        pilot_spacing=cfg.pilot_spacing,
    )

    return {
        "snr_db": snr_db,
        "ber_no_bf": ber_no_bf,
        "ber_bf": ber_bf,
    }
