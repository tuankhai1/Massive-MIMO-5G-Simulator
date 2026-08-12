"""OFDM physical-layer transceiver with optional beamforming integration.

Provides both a standalone QPSK/OFDM BER experiment and a beamforming-aware
link for the end-to-end pipeline.
"""

from __future__ import annotations

import numpy as np

from config import SystemConfig


# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map pairs of bits to unit-energy QPSK symbols."""
    symbols = (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)
    return symbols / np.sqrt(2.0)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision QPSK demodulation."""
    bits = np.empty(symbols.size * 2, dtype=int)
    bits[0::2] = symbols.real > 0
    bits[1::2] = symbols.imag > 0
    return bits


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    """Map 4-bit groups to 16-QAM symbols (unit average energy)."""
    n_sym = len(bits) // 4
    bits = bits[: n_sym * 4].reshape(n_sym, 4)
    real = 2 * bits[:, 0] - 1
    real = real * (2 * bits[:, 1] - 1 + 2)  # ±1, ±3
    imag = 2 * bits[:, 2] - 1
    imag = imag * (2 * bits[:, 3] - 1 + 2)
    # Correct constellation: {-3,-1,+1,+3} on each axis
    real_part = (2 * bits[:, 0].astype(float) - 1) * (2 * np.abs(2 * bits[:, 1].astype(float) - 1) + 1)
    imag_part = (2 * bits[:, 2].astype(float) - 1) * (2 * np.abs(2 * bits[:, 3].astype(float) - 1) + 1)
    symbols = (real_part + 1j * imag_part) / np.sqrt(10.0)
    return symbols


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision 16-QAM demodulation."""
    symbols = symbols * np.sqrt(10.0)
    bits = np.empty(symbols.size * 4, dtype=int)
    bits[0::4] = symbols.real > 0
    bits[1::4] = np.abs(symbols.real) > 2
    bits[2::4] = symbols.imag > 0
    bits[3::4] = np.abs(symbols.imag) > 2
    return bits


# ---------------------------------------------------------------------------
# OFDM transmit / receive
# ---------------------------------------------------------------------------

def ofdm_transmit(
    data_symbols: np.ndarray,
    num_subcarriers: int,
    cp_length: int,
) -> np.ndarray:
    """IFFT + cyclic prefix insertion.  Input length must equal num_subcarriers."""
    assert len(data_symbols) == num_subcarriers
    time_domain = np.fft.ifft(data_symbols, n=num_subcarriers) * np.sqrt(num_subcarriers)
    return np.concatenate([time_domain[-cp_length:], time_domain])


def ofdm_receive(
    received: np.ndarray,
    num_subcarriers: int,
    cp_length: int,
) -> np.ndarray:
    """CP removal + FFT → frequency-domain symbols."""
    stripped = received[cp_length: cp_length + num_subcarriers]
    return np.fft.fft(stripped, n=num_subcarriers) / np.sqrt(num_subcarriers)


# ---------------------------------------------------------------------------
# Channel estimation
# ---------------------------------------------------------------------------

def channel_estimate_ls(
    rx_pilots: np.ndarray,
    tx_pilots: np.ndarray,
) -> np.ndarray:
    """Least-squares channel estimate at pilot subcarriers."""
    return rx_pilots / np.where(np.abs(tx_pilots) > 1e-12, tx_pilots, 1e-12)


def interpolate_channel(
    pilot_indices: np.ndarray,
    pilot_estimates: np.ndarray,
    num_subcarriers: int,
) -> np.ndarray:
    """Linear interpolation of pilot-based channel estimates to all subcarriers."""
    all_indices = np.arange(num_subcarriers)
    real_interp = np.interp(all_indices, pilot_indices, pilot_estimates.real)
    imag_interp = np.interp(all_indices, pilot_indices, pilot_estimates.imag)
    return real_interp + 1j * imag_interp


# ---------------------------------------------------------------------------
# Standalone OFDM BER (legacy, backward compatible)
# ---------------------------------------------------------------------------

def ofdm_qpsk_ber(
    snr_db_values: np.ndarray,
    rng: np.random.Generator,
    frames: int = 150,
    subcarriers: int = 64,
) -> np.ndarray:
    """QPSK OFDM through flat-per-subcarrier Rayleigh fading with ideal pilots."""
    ber = []
    for snr_db in snr_db_values:
        bit_errors = 0
        total_bits = 0
        noise_std = np.sqrt(1.0 / (2.0 * 10 ** (snr_db / 10.0)))
        for _ in range(frames):
            bits = rng.integers(0, 2, size=2 * subcarriers)
            transmitted = qpsk_modulate(bits)
            channel = (
                rng.normal(size=subcarriers) + 1j * rng.normal(size=subcarriers)
            ) / np.sqrt(2.0)
            noise = noise_std * (
                rng.normal(size=subcarriers) + 1j * rng.normal(size=subcarriers)
            )
            received = channel * transmitted + noise
            estimated = received / np.where(np.abs(channel) > 1e-8, channel, 1e-8)
            decoded = qpsk_demodulate(estimated)
            bit_errors += np.count_nonzero(bits != decoded)
            total_bits += bits.size
        ber.append(bit_errors / total_bits)
    return np.asarray(ber)


# ---------------------------------------------------------------------------
# OFDM BER with beamforming
# ---------------------------------------------------------------------------

def ofdm_ber_with_beamforming(
    snr_db_values: np.ndarray,
    rng: np.random.Generator,
    num_antennas: int = 16,
    frames: int = 100,
    subcarriers: int = 64,
    cp_length: int = 16,
    pilot_spacing: int = 4,
) -> np.ndarray:
    """OFDM BER where the BS applies a matched-filter beamformer per subcarrier.

    The beamforming gain is the primary difference from the non-BF case.
    """
    from array_model import steering_vector  # local import to avoid circular

    ber = []
    for snr_db in snr_db_values:
        bit_errors = 0
        total_bits = 0
        noise_std = np.sqrt(1.0 / (2.0 * 10 ** (snr_db / 10.0)))
        for _ in range(frames):
            # Random dominant AoD for this frame
            angle = rng.uniform(-np.pi / 3, np.pi / 3)
            sv = steering_vector(num_antennas, np.sin(angle))

            bits = rng.integers(0, 2, size=2 * subcarriers)
            tx_symbols = qpsk_modulate(bits)

            # Per-subcarrier flat fading channel vector (Nt,) + small spread
            h_base = sv + 0.2 * (
                rng.normal(size=num_antennas) + 1j * rng.normal(size=num_antennas)
            ) / np.sqrt(num_antennas)

            # BF weight = matched filter (ideal CSI)
            w = h_base / np.linalg.norm(h_base)
            effective_channel = h_base.conj() @ w  # scalar BF gain

            noise = noise_std * (
                rng.normal(size=subcarriers) + 1j * rng.normal(size=subcarriers)
            )
            received = effective_channel * tx_symbols + noise

            # Pilot-aided LS estimation
            pilot_idx = np.arange(0, subcarriers, pilot_spacing)
            pilot_tx = tx_symbols[pilot_idx]
            pilot_rx = received[pilot_idx]
            h_est_pilots = channel_estimate_ls(pilot_rx, pilot_tx)
            h_est = interpolate_channel(pilot_idx, h_est_pilots, subcarriers)

            equalized = received / np.where(np.abs(h_est) > 1e-8, h_est, 1e-8)
            decoded = qpsk_demodulate(equalized)
            bit_errors += np.count_nonzero(bits != decoded)
            total_bits += bits.size
        ber.append(bit_errors / max(total_bits, 1))
    return np.asarray(ber)
