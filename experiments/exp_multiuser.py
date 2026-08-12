"""Experiment: multi-user sum rate and Jain fairness versus number of users."""

from __future__ import annotations

import numpy as np

from beamforming import multi_user_zf_precoder, compute_sinr
from channel_model import geometric_channel
from config import SystemConfig
from metrics import jain_fairness


def run(cfg: SystemConfig) -> dict:
    """MU-MIMO ZF sum rate and fairness for {2, 4, 8, 12, 16} users."""
    user_counts = np.array([2, 4, 8, 12, 16])
    rng = np.random.default_rng(cfg.seed + 80)
    trials = 60

    sum_rates, fairness_values, per_user_rates = [], [], []

    for K in user_counts:
        trial_sr, trial_fair = [], []
        for _ in range(trials):
            positions = rng.uniform([30.0, -60.0], [150.0, 60.0], size=(K, 2))
            channels = np.vstack(
                [geometric_channel(pos, cfg, rng).reshape(1, -1) for pos in positions]
            )
            # Only use ZF when K <= Nt
            if K <= cfg.antennas:
                W = multi_user_zf_precoder(channels)
                sinr = compute_sinr(channels, W, cfg.tx_power_w, cfg.noise_power_w)
            else:
                # Fallback: random precoding for K > Nt
                W = (rng.normal(size=(cfg.antennas, K)) + 1j * rng.normal(size=(cfg.antennas, K)))
                W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
                sinr = compute_sinr(channels, W, cfg.tx_power_w, cfg.noise_power_w)

            user_rates = np.log2(1.0 + sinr)
            trial_sr.append(float(user_rates.sum()))
            trial_fair.append(jain_fairness(user_rates))

        sum_rates.append(np.mean(trial_sr))
        fairness_values.append(np.mean(trial_fair))
        per_user_rates.append(np.mean(trial_sr) / K)

    return {
        "users": user_counts,
        "sum_rate": np.array(sum_rates),
        "fairness": np.array(fairness_values),
        "per_user_rate": np.array(per_user_rates),
    }
