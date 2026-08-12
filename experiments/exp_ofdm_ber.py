"""Experiment: BER versus SNR for OFDM with and without beamforming.

Now includes both QPSK and 16-QAM modulation to exercise the full
modulation chain.
"""

from __future__ import annotations

import numpy as np

from config import SystemConfig
from phy import ofdm_qpsk_ber, ofdm_ber_with_beamforming, ofdm_16qam_ber


def run(cfg: SystemConfig) -> dict:
    """Compare OFDM BER curves: QPSK ± BF and 16-QAM baseline."""
    rng_no_bf = np.random.default_rng(cfg.seed + 20)
    rng_bf = np.random.default_rng(cfg.seed + 21)
    rng_16qam = np.random.default_rng(cfg.seed + 22)

    snr_db = np.arange(-2, 22, 2)
    ber_no_bf = ofdm_qpsk_ber(
        snr_db, rng_no_bf, frames=180, subcarriers=cfg.num_subcarriers,
        cp_length=cfg.cp_length, pilot_spacing=cfg.pilot_spacing,
    )
    ber_bf = ofdm_ber_with_beamforming(
        snr_db, rng_bf,
        num_antennas=cfg.antennas,
        frames=180,
        subcarriers=cfg.num_subcarriers,
        cp_length=cfg.cp_length,
        pilot_spacing=cfg.pilot_spacing,
    )

    # 16-QAM needs higher SNR; use wider range
    snr_db_16qam = np.arange(0, 28, 2)
    ber_16qam = ofdm_16qam_ber(
        snr_db_16qam, rng_16qam, frames=180, subcarriers=cfg.num_subcarriers,
        cp_length=cfg.cp_length, pilot_spacing=cfg.pilot_spacing,
    )

    return {
        "snr_db": snr_db,
        "ber_no_bf": ber_no_bf,
        "ber_bf": ber_bf,
        "snr_db_16qam": snr_db_16qam,
        "ber_16qam": ber_16qam,
    }
