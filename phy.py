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
    """Map 4-bit groups to Gray-coded 16-QAM symbols (unit average energy).

    Bit mapping per axis (Gray code):
        b_sign=0 → negative,  b_sign=1 → positive
        b_amp=0  → outer (±3), b_amp=1  → inner (±1)

    Constellation points: {-3, -1, +1, +3} / sqrt(10) on each axis.
    """
    n_sym = len(bits) // 4
    bits = bits[: n_sym * 4].reshape(n_sym, 4)
    # b0 = sign_I, b1 = amplitude_I, b2 = sign_Q, b3 = amplitude_Q
    sign_i = 2.0 * bits[:, 0] - 1.0        # 0 → -1, 1 → +1
    amp_i  = 3.0 - 2.0 * bits[:, 1]        # 0 → 3 (outer), 1 → 1 (inner)
    sign_q = 2.0 * bits[:, 2] - 1.0
    amp_q  = 3.0 - 2.0 * bits[:, 3]
    symbols = (sign_i * amp_i + 1j * sign_q * amp_q) / np.sqrt(10.0)
    return symbols


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision Gray-coded 16-QAM demodulation.

    Decision boundaries:
        sign bit:      Re/Im > 0  → 1
        amplitude bit:  |Re/Im| < 2 → 1 (inner), else 0 (outer)
    """
    symbols = symbols * np.sqrt(10.0)
    bits = np.empty(symbols.size * 4, dtype=int)
    bits[0::4] = (symbols.real > 0).astype(int)
    bits[1::4] = (np.abs(symbols.real) < 2).astype(int)
    bits[2::4] = (symbols.imag > 0).astype(int)
    bits[3::4] = (np.abs(symbols.imag) < 2).astype(int)
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


def interpolate_channel_dft(
    pilot_indices: np.ndarray,
    pilot_estimates: np.ndarray,
    num_subcarriers: int,
    channel_taps: int,
) -> np.ndarray:
    """Fit a CP-bounded tapped-delay-line channel to pilot estimates.

    Unlike direct linear interpolation, this uses the OFDM channel structure
    and remains stable at high SNR for a frequency-selective multipath link.
    """
    tap_indices = np.arange(min(channel_taps, num_subcarriers))
    pilot_matrix = np.exp(
        -2j * np.pi * np.outer(np.asarray(pilot_indices), tap_indices) / num_subcarriers
    )
    taps, *_ = np.linalg.lstsq(pilot_matrix, pilot_estimates, rcond=None)
    all_matrix = np.exp(
        -2j * np.pi * np.outer(np.arange(num_subcarriers), tap_indices) / num_subcarriers
    )
    return all_matrix @ taps


def _sample_multipath_channel(
    rng: np.random.Generator,
    cp_length: int,
) -> np.ndarray:
    """Draw a normalized tapped-delay-line channel contained within the CP."""
    # Delay spread is deliberately below the pilot-sampling coherence interval
    # (one pilot every four tones) as well as the cyclic prefix.  Otherwise a
    # linear pilot interpolator develops an artificial high-SNR error floor.
    candidate_delays = np.array([0, 1, 2, 3])
    delays = candidate_delays[candidate_delays <= cp_length]
    if len(delays) == 0:
        delays = np.array([0])
    powers = np.exp(-delays / 2.8)
    gains = (rng.normal(size=len(delays)) + 1j * rng.normal(size=len(delays)))
    gains *= np.sqrt(powers / 2.0)
    gains /= np.linalg.norm(gains) + 1e-12
    impulse_response = np.zeros(int(delays.max()) + 1, dtype=complex)
    impulse_response[delays] = gains
    return impulse_response


def _ofdm_link_ber(
    snr_db_values: np.ndarray,
    rng: np.random.Generator,
    *,
    bits_per_symbol: int,
    frames: int,
    subcarriers: int,
    cp_length: int,
    pilot_spacing: int,
    beamforming_antennas: int = 1,
) -> np.ndarray:
    """BER for CP-OFDM through a multipath channel with LS pilot equalisation.

    The requested SNR is the average single-antenna receive SNR before array
    gain.  Setting ``beamforming_antennas`` above one applies ideal matched
    transmit steering, yielding the expected coherent gain while retaining the
    *same* multipath channel, pilots, CP and receiver as the no-BF baseline.
    """
    if subcarriers <= 0 or cp_length < 0 or pilot_spacing <= 0:
        raise ValueError("Invalid OFDM dimensions or pilot spacing")

    pilot_indices = np.arange(0, subcarriers, pilot_spacing, dtype=int)
    data_indices = np.setdiff1d(np.arange(subcarriers), pilot_indices)
    pilot_symbol = (1.0 + 1.0j) / np.sqrt(2.0)
    beam_amplitude = np.sqrt(max(beamforming_antennas, 1))
    ber = []

    for snr_db in snr_db_values:
        bit_errors = 0
        total_bits = 0
        noise_std = np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0)))

        for _ in range(frames):
            bits = rng.integers(0, 2, size=bits_per_symbol * len(data_indices))
            data_symbols = qpsk_modulate(bits) if bits_per_symbol == 2 else qam16_modulate(bits)
            frequency_symbols = np.empty(subcarriers, dtype=complex)
            frequency_symbols[pilot_indices] = pilot_symbol
            frequency_symbols[data_indices] = data_symbols

            tx_signal = ofdm_transmit(frequency_symbols, subcarriers, cp_length)
            impulse_response = beam_amplitude * _sample_multipath_channel(rng, cp_length)
            received = np.convolve(tx_signal, impulse_response, mode="full")[: len(tx_signal)]
            received += noise_std * (
                rng.normal(size=len(received)) + 1j * rng.normal(size=len(received))
            )

            received_frequency = ofdm_receive(received, subcarriers, cp_length)
            pilot_estimates = channel_estimate_ls(
                received_frequency[pilot_indices], frequency_symbols[pilot_indices]
            )
            channel_estimate = interpolate_channel_dft(
                pilot_indices, pilot_estimates, subcarriers,
                channel_taps=len(impulse_response),
            )
            equalized_data = received_frequency[data_indices] / np.where(
                np.abs(channel_estimate[data_indices]) > 1e-8,
                channel_estimate[data_indices],
                1e-8,
            )
            decoded = qpsk_demodulate(equalized_data) if bits_per_symbol == 2 else qam16_demodulate(equalized_data)
            bit_errors += np.count_nonzero(bits != decoded)
            total_bits += len(bits)

        ber.append(bit_errors / max(total_bits, 1))
    return np.asarray(ber)


# ---------------------------------------------------------------------------
# Standalone OFDM BER (legacy, backward compatible)
# ---------------------------------------------------------------------------

def ofdm_qpsk_ber(
    snr_db_values: np.ndarray,
    rng: np.random.Generator,
    frames: int = 150,
    subcarriers: int = 64,
    cp_length: int = 16,
    pilot_spacing: int = 4,
) -> np.ndarray:
    """QPSK CP-OFDM BER with pilot-aided LS channel estimation."""
    return _ofdm_link_ber(
        snr_db_values, rng, bits_per_symbol=2, frames=frames,
        subcarriers=subcarriers, cp_length=cp_length, pilot_spacing=pilot_spacing,
    )


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
    """QPSK CP-OFDM BER with matched-beam array gain and identical PHY setup."""
    return _ofdm_link_ber(
        snr_db_values, rng, bits_per_symbol=2, frames=frames,
        subcarriers=subcarriers, cp_length=cp_length, pilot_spacing=pilot_spacing,
        beamforming_antennas=num_antennas,
    )


# ---------------------------------------------------------------------------
# Standalone 16-QAM OFDM BER
# ---------------------------------------------------------------------------

def ofdm_16qam_ber(
    snr_db_values: np.ndarray,
    rng: np.random.Generator,
    frames: int = 150,
    subcarriers: int = 64,
    cp_length: int = 16,
    pilot_spacing: int = 4,
) -> np.ndarray:
    """16-QAM CP-OFDM BER with the same multipath and LS-estimation model."""
    return _ofdm_link_ber(
        snr_db_values, rng, bits_per_symbol=4, frames=frames,
        subcarriers=subcarriers, cp_length=cp_length, pilot_spacing=pilot_spacing,
    )
