"""Digital, analog, and hybrid beamforming with multi-user precoding.

This module provides three architectures — analog (DFT codebook selection),
digital (MRT / ZF), and hybrid (alternating minimisation) — plus multi-user
ZF precoding and SINR/rate calculation utilities.
"""

from __future__ import annotations

import numpy as np

from config import SystemConfig


# ===================================================================
# Analog beamforming (codebook-based — existing logic, renamed)
# ===================================================================

def snr_per_beam(
    channel: np.ndarray,
    codebook: np.ndarray,
    cfg: SystemConfig,
) -> np.ndarray:
    """Post-beamforming SNR for every codebook beam."""
    received_power = cfg.tx_power_w * np.abs(channel.conj() @ codebook) ** 2
    return received_power / cfg.noise_power_w


def effective_rate(
    snr_linear: float,
    tested_beams: int,
    cfg: SystemConfig,
    retraining_period_s: float | None = None,
) -> float:
    """Spectral efficiency after deducting beam-training pilot overhead.

    ``retraining_period_s`` is the interval between two beam acquisitions.  It
    defaults to one frame, while mobility studies can use a beam-coherence
    interval so that faster users incur more frequent training overhead.
    """
    interval = cfg.frame_s if retraining_period_s is None else retraining_period_s
    if interval <= 0:
        raise ValueError("retraining_period_s must be positive")
    data_fraction = max(0.0, 1.0 - tested_beams * cfg.pilot_s / interval)
    return float(data_fraction * np.log2(1.0 + snr_linear))


def best_beam(
    snr_values: np.ndarray,
    candidates: np.ndarray | None = None,
) -> int:
    if candidates is None:
        return int(np.argmax(snr_values))
    return int(candidates[np.argmax(snr_values[candidates])])


def top_k_around(center: int, total_beams: int, k: int) -> np.ndarray:
    """Return ``k`` contiguous candidate beams around ``center``.

    The DFT endpoints represent opposite end-fire directions, so wrapping an
    index from the first beam to the last beam is not a physically local search.
    """
    if total_beams <= 0 or k <= 0:
        raise ValueError("total_beams and k must be positive")
    k_eff = min(k, total_beams)
    start = int(np.clip(center - k_eff // 2, 0, total_beams - k_eff))
    return np.arange(start, start + k_eff, dtype=int)


def analog_beamformer(
    channel: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Select the best codebook beam (pure analog).

    Returns (weight_vector, beam_index).
    """
    gains = np.abs(channel.conj() @ codebook) ** 2
    idx = int(np.argmax(gains))
    return codebook[:, idx], idx


# ===================================================================
# Digital beamforming
# ===================================================================

def digital_mrt(channel: np.ndarray) -> np.ndarray:
    """Maximum Ratio Transmission (matched filter) for a single user.

    Under the convention y = h^H w s + n, the MRT precoder is w = h / ||h||
    which maximises |h^H w| = ||h||.

    Parameters
    ----------
    channel : (Nt,) complex array — the channel vector h.

    Returns
    -------
    w : (Nt,) unit-norm precoding vector.
    """
    w = channel.copy()
    return w / (np.linalg.norm(w) + 1e-12)


def digital_zf_single(channel: np.ndarray) -> np.ndarray:
    """For a single-user MISO channel, ZF is the same as MRT (up to scaling)."""
    return digital_mrt(channel)


# ===================================================================
# Multi-user ZF precoding
# ===================================================================

def multi_user_zf_precoder(channels: np.ndarray) -> np.ndarray:
    """Zero-forcing precoder for K users, each with a single antenna.

    Parameters
    ----------
    channels : (K, Nt) complex array — stacked channel row-vectors.

    Returns
    -------
    W : (Nt, K) precoding matrix, columns have unit norm.
    """
    K, Nt = channels.shape
    # The receive model is y = H^H W s + n.  Work with the equivalent
    # row-channel H_rx = H^H before forming the Moore-Penrose ZF precoder.
    H_rx = channels.conj()
    gram = H_rx @ H_rx.conj().T
    regularization = 1e-10 * max(float(np.trace(gram).real) / max(K, 1), 1e-30)
    try:
        W_raw = H_rx.conj().T @ np.linalg.inv(gram + regularization * np.eye(K))
    except np.linalg.LinAlgError:
        W_raw = H_rx.conj().T @ np.linalg.pinv(gram)
    # Normalize columns
    norms = np.linalg.norm(W_raw, axis=0, keepdims=True)
    return W_raw / (norms + 1e-12)


def compute_sinr(
    channels: np.ndarray,
    precoder: np.ndarray,
    total_power: float,
    noise_power: float,
) -> np.ndarray:
    """Per-user SINR under multi-user interference.

    Parameters
    ----------
    channels : (K, Nt)
    precoder : (Nt, K)  — columns are per-user precoders
    total_power : float  — total transmit power (split equally)
    noise_power : float

    Returns
    -------
    sinr : (K,) array
    """
    K = channels.shape[0]
    power_per_user = total_power / K
    effective = channels.conj() @ precoder  # (K, K)
    signal = power_per_user * np.abs(np.diag(effective)) ** 2
    interference = power_per_user * (
        np.sum(np.abs(effective) ** 2, axis=1) - np.abs(np.diag(effective)) ** 2
    )
    return signal / (interference + noise_power)


def sum_rate(
    channels: np.ndarray,
    precoder: np.ndarray,
    total_power: float,
    noise_power: float,
) -> float:
    """Sum spectral efficiency across all users."""
    sinr = compute_sinr(channels, precoder, total_power, noise_power)
    return float(np.sum(np.log2(1.0 + sinr)))


# ===================================================================
# Hybrid beamforming (alternating minimisation)
# ===================================================================

def hybrid_beamformer(
    channel: np.ndarray,
    codebook: np.ndarray,
    num_rf_chains: int,
    num_streams: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating-minimisation hybrid precoder for a single user.

    1. **Analog part** — greedily pick ``num_rf_chains`` codebook columns that
       capture the most channel energy (OMP-inspired).
    2. **Digital part** — effective-channel MRT/ZF on the reduced-dimension
       channel  ``h_eff = h^H F_RF``.

    Parameters
    ----------
    channel : (Nt,) — single-user channel vector
    codebook : (Nt, B) — analog beam codebook
    num_rf_chains : int
    num_streams : int
        Must be 1. This educational single-user model does not model spatial
        multiplexing.

    Returns
    -------
    F_RF : (Nt, num_rf_chains) — analog precoder (columns from codebook)
    F_BB : (num_rf_chains, 1) — digital baseband precoder
    """
    if num_streams != 1:
        raise ValueError("The educational hybrid-beamforming model supports one stream only.")
    if not 1 <= num_rf_chains <= codebook.shape[1]:
        raise ValueError("num_rf_chains must be between 1 and the codebook size.")

    Nt = len(channel)
    residual = channel.copy()
    selected = []

    for _ in range(num_rf_chains):
        available = np.setdiff1d(np.arange(codebook.shape[1]), selected, assume_unique=True)
        if available.size == 0:
            break
        projections = np.abs(codebook[:, available].conj().T @ residual)
        idx = int(np.argmax(projections))
        selected.append(int(available[idx]))
        # Orthogonal matching pursuit residual after refitting all selected atoms.
        F_selected = codebook[:, selected]
        coefficients, *_ = np.linalg.lstsq(F_selected, channel, rcond=None)
        residual = channel - F_selected @ coefficients

    F_RF = codebook[:, selected]  # (Nt, Nrf)

    # Effective channel through the analog stage
    h_eff = channel.conj() @ F_RF  # (Nrf,)

    # Digital baseband: MRT on effective channel
    F_BB = h_eff.conj().reshape(-1, 1)
    F_BB = F_BB / (np.linalg.norm(F_BB) + 1e-12)

    return F_RF, F_BB[:, :num_streams]


def hybrid_rate(
    channel: np.ndarray,
    F_RF: np.ndarray,
    F_BB: np.ndarray,
    total_power: float,
    noise_power: float,
) -> float:
    """Achievable rate through a hybrid precoder for one user."""
    combined = F_RF @ F_BB  # (Nt, Ns)
    combined = combined / (np.linalg.norm(combined) + 1e-12)
    effective_gain = np.abs(channel.conj() @ combined) ** 2
    snr = total_power * float(np.sum(effective_gain)) / noise_power
    return float(np.log2(1.0 + snr))


# ===================================================================
# Convenience: compare all three architectures
# ===================================================================

def compare_beamforming(
    channel: np.ndarray,
    codebook: np.ndarray,
    cfg: SystemConfig,
) -> dict[str, float]:
    """Return achievable rate for analog, digital, and hybrid BF."""
    # Analog
    w_analog, _ = analog_beamformer(channel, codebook)
    gain_analog = np.abs(channel.conj() @ w_analog) ** 2
    snr_analog = cfg.tx_power_w * gain_analog / cfg.noise_power_w
    rate_analog = float(np.log2(1.0 + snr_analog))

    # Digital (MRT)
    w_digital = digital_mrt(channel)
    gain_digital = np.abs(channel.conj() @ w_digital) ** 2
    snr_digital = cfg.tx_power_w * gain_digital / cfg.noise_power_w
    rate_digital = float(np.log2(1.0 + snr_digital))

    # Hybrid
    F_RF, F_BB = hybrid_beamformer(channel, codebook, cfg.num_rf_chains)
    rate_hybrid = hybrid_rate(channel, F_RF, F_BB, cfg.tx_power_w, cfg.noise_power_w)

    return {
        "analog": rate_analog,
        "digital": rate_digital,
        "hybrid": rate_hybrid,
    }
