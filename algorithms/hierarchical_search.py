"""Hierarchical (multi-resolution) beam search algorithm."""

from __future__ import annotations

import numpy as np


def build_hierarchical_codebook(
    num_antennas: int,
    num_levels: int = 3,
) -> list[np.ndarray]:
    """Build a multi-resolution DFT codebook list.

    Level 0 has the widest beams (fewest columns), and the last level
    has resolution equal to the full array.

    Returns
    -------
    codebooks : list of ndarray, each shape (num_antennas, beams_at_level)
    """
    codebooks = []
    for level in range(num_levels):
        sub_size = max(2, num_antennas // (2 ** (num_levels - 1 - level)))
        num_beams = sub_size
        freqs = -1.0 + 2.0 * np.arange(num_beams) / num_beams
        cols = []
        for f in freqs:
            # Build a sub-array steering vector, zero-pad to full size
            idx = np.arange(sub_size)
            v_sub = np.exp(1j * np.pi * idx * f) / np.sqrt(sub_size)
            v_full = np.zeros(num_antennas, dtype=complex)
            v_full[:sub_size] = v_sub
            v_full = v_full / (np.linalg.norm(v_full) + 1e-12)
            cols.append(v_full)
        codebooks.append(np.column_stack(cols))
    return codebooks


def hierarchical_beam_search(
    channel: np.ndarray,
    codebooks: list[np.ndarray],
    beams_per_level: int = 2,
) -> tuple[int, int, np.ndarray]:
    """Multi-stage beam refinement through hierarchical codebooks.

    Parameters
    ----------
    channel : (Nt,) channel vector.
    codebooks : list of codebook matrices, coarse → fine.
    beams_per_level : number of candidates to keep per stage.

    Returns
    -------
    final_beam : int — beam index in the finest codebook.
    total_pilots : int — total pilot measurements used.
    gains_per_level : (num_levels,) — best gain at each stage.
    """
    total_pilots = 0
    gains_per_level = []

    # Level 0: exhaustive over coarse codebook
    cb0 = codebooks[0]
    gains = np.abs(channel.conj() @ cb0) ** 2
    best_idx = int(np.argmax(gains))
    total_pilots += cb0.shape[1]
    gains_per_level.append(float(gains[best_idx]))

    for lev in range(1, len(codebooks)):
        cb = codebooks[lev]
        nb = cb.shape[1]
        prev_nb = codebooks[lev - 1].shape[1]
        ratio = nb / prev_nb
        center = int(best_idx * ratio)
        half = max(1, beams_per_level)
        cands = np.unique(np.clip(np.arange(center - half, center + half + 1), 0, nb - 1))
        g = np.abs(channel.conj() @ cb[:, cands]) ** 2
        winner = int(np.argmax(g))
        best_idx = int(cands[winner])
        total_pilots += len(cands)
        gains_per_level.append(float(g[winner]))

    return best_idx, total_pilots, np.array(gains_per_level)
