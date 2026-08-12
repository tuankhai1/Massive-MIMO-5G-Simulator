"""Experiment: achievable rate versus number of antennas.

Shows three curves:
1. **Raw rate** — array gain without pilot overhead (upper bound).
2. **Effective rate (exhaustive)** — rate after deducting full codebook sweep overhead.
3. **Effective rate (top-K)** — rate after deducting only top-K pilot overhead.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from array_model import dft_codebook
from beamforming import effective_rate, snr_per_beam, best_beam, top_k_around
from beam_management import location_beam
from channel_model import geometric_channel
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Mean effective rate for antenna counts {4, 8, 16, 32, 64, 128}.

    Returns three rate arrays: raw (no overhead), exhaustive sweep, and
    top-K beam management.
    """
    antenna_values = np.array([4, 8, 16, 32, 64, 128])
    rng = np.random.default_rng(cfg.seed + 30)
    positions = rng.uniform([35.0, -55.0], [150.0, 55.0], size=(120, 2))

    raw_rates = []
    exhaustive_rates = []
    topk_rates = []
    peak_snr = []

    for N in antenna_values:
        local = replace(cfg, antennas=int(N), codebook_beams=int(N))
        codebook, spatial_freqs = dft_codebook(local.antennas, local.codebook_beams)
        r_raw, r_exh, r_topk, s_samples = [], [], [], []

        for pos in positions:
            ch = geometric_channel(pos, local, rng)
            snr = snr_per_beam(ch, codebook, local)
            best_snr = float(snr.max())

            # Raw rate: no overhead deduction
            r_raw.append(float(np.log2(1.0 + best_snr)))

            # Exhaustive: overhead = codebook_beams pilots
            r_exh.append(effective_rate(best_snr, local.codebook_beams, local))

            # Top-K: locate approximate beam, refine with top_k neighbours
            loc_beam = location_beam(pos, spatial_freqs)
            topk_cands = top_k_around(loc_beam, local.codebook_beams, local.top_k)
            topk_best = best_beam(snr, topk_cands)
            topk_snr = float(snr[topk_best])
            r_topk.append(effective_rate(topk_snr, len(topk_cands), local))

            s_samples.append(10.0 * np.log10(max(best_snr, 1e-10)))

        raw_rates.append(np.mean(r_raw))
        exhaustive_rates.append(np.mean(r_exh))
        topk_rates.append(np.mean(r_topk))
        peak_snr.append(np.mean(s_samples))

    return {
        "antennas": antenna_values,
        "raw_rate": np.asarray(raw_rates),
        "rate": np.asarray(exhaustive_rates),       # backward compat key
        "topk_rate": np.asarray(topk_rates),
        "peak_snr_db": np.asarray(peak_snr),
    }
