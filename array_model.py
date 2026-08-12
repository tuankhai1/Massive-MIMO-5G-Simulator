"""Antenna-array steering vectors, codebooks, and radiation-pattern helpers.

Supports both Uniform Linear Arrays (ULA) and Uniform Planar Arrays (UPA).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# ULA (Uniform Linear Array)
# ---------------------------------------------------------------------------

def steering_vector(num_antennas: int, spatial_frequency: float) -> np.ndarray:
    """Half-wavelength ULA response; spatial frequency equals sin(angle)."""
    index = np.arange(num_antennas)
    return np.exp(1j * np.pi * index * spatial_frequency) / np.sqrt(num_antennas)


def dft_codebook(num_antennas: int, num_beams: int) -> tuple[np.ndarray, np.ndarray]:
    """Create unit-norm DFT codebook columns and their spatial frequencies."""
    frequencies = -1.0 + 2.0 * np.arange(num_beams) / num_beams
    codebook = np.column_stack(
        [steering_vector(num_antennas, v) for v in frequencies]
    )
    return codebook, frequencies


def array_response_db(
    codebook: np.ndarray,
    num_antennas: int,
    beam_index: int,
    angles: np.ndarray,
) -> np.ndarray:
    """Normalized power pattern (dB) of a single codebook beam over *angles*."""
    scan = np.column_stack(
        [steering_vector(num_antennas, np.sin(a)) for a in angles]
    )
    power = np.abs(codebook[:, beam_index].conj() @ scan) ** 2
    return 10.0 * np.log10(np.maximum(power / power.max(), 1e-6))


# ---------------------------------------------------------------------------
# UPA (Uniform Planar Array)
# ---------------------------------------------------------------------------

def steering_vector_upa(
    rows: int,
    cols: int,
    az_freq: float,
    el_freq: float,
) -> np.ndarray:
    """Half-wavelength UPA response.

    Parameters
    ----------
    rows, cols : int
        Number of antenna rows (elevation) and columns (azimuth).
    az_freq : float
        Azimuth spatial frequency  = sin(azimuth) * cos(elevation).
    el_freq : float
        Elevation spatial frequency = sin(elevation).

    Returns
    -------
    np.ndarray
        Unit-norm steering vector of length ``rows * cols``.
    """
    az_vec = np.exp(1j * np.pi * np.arange(cols) * az_freq)
    el_vec = np.exp(1j * np.pi * np.arange(rows) * el_freq)
    vec = np.kron(el_vec, az_vec)
    return vec / np.linalg.norm(vec)


def dft_codebook_2d(
    rows: int,
    cols: int,
    az_beams: int,
    el_beams: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D DFT codebook for a UPA.

    Returns
    -------
    codebook : ndarray, shape (rows*cols, az_beams*el_beams)
    az_freqs : ndarray
    el_freqs : ndarray
    """
    az_freqs = -1.0 + 2.0 * np.arange(az_beams) / az_beams
    el_freqs = -1.0 + 2.0 * np.arange(el_beams) / el_beams
    columns = []
    for ef in el_freqs:
        for af in az_freqs:
            columns.append(steering_vector_upa(rows, cols, af, ef))
    return np.column_stack(columns), az_freqs, el_freqs


# ---------------------------------------------------------------------------
# Random codebook (baseline)
# ---------------------------------------------------------------------------

def random_codebook(
    num_antennas: int,
    num_beams: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random unit-norm beamforming vectors (useful as a weak baseline)."""
    if rng is None:
        rng = np.random.default_rng(0)
    raw = rng.standard_normal((num_antennas, num_beams)) + 1j * rng.standard_normal(
        (num_antennas, num_beams)
    )
    norms = np.linalg.norm(raw, axis=0, keepdims=True)
    return raw / norms
