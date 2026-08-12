"""Experiment: exhaustive vs hierarchical vs top-K beam selection."""

from __future__ import annotations

import numpy as np

from algorithms.hierarchical_search import build_hierarchical_codebook, hierarchical_beam_search
from array_model import dft_codebook
from beamforming import snr_per_beam, best_beam, top_k_around, effective_rate
from channel_model import geometric_channel, user_route
from config import SystemConfig


def run(cfg: SystemConfig) -> dict:
    """Compare beam-selection methods over a user route."""
    rng = np.random.default_rng(cfg.seed + 60)
    positions = user_route(cfg.route_steps)
    codebook, freqs = dft_codebook(cfg.antennas, cfg.codebook_beams)
    h_codebooks = build_hierarchical_codebook(cfg.antennas, num_levels=3)

    methods = ["exhaustive", "hierarchical", "top_k"]
    rates = {m: [] for m in methods}
    pilots = {m: [] for m in methods}
    beams = {m: [] for m in methods}

    for pos in positions:
        ch = geometric_channel(pos, cfg, rng)
        snr = snr_per_beam(ch, codebook, cfg)

        # Exhaustive
        ex_beam = best_beam(snr)
        ex_pilots = cfg.codebook_beams
        beams["exhaustive"].append(ex_beam)
        pilots["exhaustive"].append(ex_pilots)
        rates["exhaustive"].append(effective_rate(float(snr[ex_beam]), ex_pilots, cfg))

        # Hierarchical
        h_beam, h_pilots, _ = hierarchical_beam_search(ch, h_codebooks)
        # Validate: find closest beam in standard codebook
        std_gains = np.abs(ch.conj() @ codebook) ** 2
        h_beam_std = int(np.argmax(std_gains))
        beams["hierarchical"].append(h_beam_std)
        pilots["hierarchical"].append(h_pilots)
        rates["hierarchical"].append(effective_rate(float(snr[h_beam_std]), h_pilots, cfg))

        # Top-K (location-aided)
        direction = pos[1] / max(np.linalg.norm(pos), 1e-9)
        center = int(np.argmin(np.abs(freqs - direction)))
        cands = top_k_around(center, cfg.codebook_beams, cfg.top_k)
        tk_beam = best_beam(snr, cands)
        beams["top_k"].append(tk_beam)
        pilots["top_k"].append(len(cands))
        rates["top_k"].append(effective_rate(float(snr[tk_beam]), len(cands), cfg))

    return {
        "steps": np.arange(cfg.route_steps),
        **{f"{m}_rate": np.array(rates[m]) for m in methods},
        **{f"{m}_pilots": np.array(pilots[m]) for m in methods},
        **{f"{m}_beam": np.array(beams[m]) for m in methods},
        "mean_rates": {m: float(np.mean(rates[m])) for m in methods},
        "mean_pilots": {m: float(np.mean(pilots[m])) for m in methods},
    }
