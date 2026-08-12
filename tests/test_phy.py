"""Unit tests for the PHY-layer transceiver (phy.py).

Tests cover:
- QPSK modulation/demodulation roundtrip
- 16-QAM modulation/demodulation roundtrip (all 16 symbols)
- 16-QAM constellation point validation
- OFDM transmit/receive roundtrip (noise-free)
- Combined OFDM + modulation roundtrip
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path so imports work from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phy import (
    qpsk_modulate,
    qpsk_demodulate,
    qam16_modulate,
    qam16_demodulate,
    ofdm_transmit,
    ofdm_receive,
)


# ===================================================================
# QPSK
# ===================================================================

class TestQPSK:
    """QPSK modulation / demodulation tests."""

    def test_roundtrip_all_symbols(self):
        """All 4 possible 2-bit combinations survive mod → demod."""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk_modulate(bits)
        recovered = qpsk_demodulate(symbols)
        np.testing.assert_array_equal(bits, recovered)

    def test_roundtrip_random(self):
        """Random bit stream survives mod → demod (noise-free)."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=1000)
        symbols = qpsk_modulate(bits)
        recovered = qpsk_demodulate(symbols)
        np.testing.assert_array_equal(bits, recovered)

    def test_unit_energy(self):
        """QPSK symbols have unit average energy."""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk_modulate(bits)
        mean_energy = np.mean(np.abs(symbols) ** 2)
        np.testing.assert_allclose(mean_energy, 1.0, atol=1e-10)


# ===================================================================
# 16-QAM
# ===================================================================

class TestQAM16:
    """16-QAM modulation / demodulation tests."""

    def test_roundtrip_all_16_symbols(self):
        """All 16 possible 4-bit combinations survive mod → demod."""
        # Generate all 16 4-bit patterns
        all_bits = np.array(
            [[b3, b2, b1, b0]
             for b3 in range(2) for b2 in range(2)
             for b1 in range(2) for b0 in range(2)]
        ).ravel()  # 64 bits total

        symbols = qam16_modulate(all_bits)
        assert len(symbols) == 16, f"Expected 16 symbols, got {len(symbols)}"

        recovered = qam16_demodulate(symbols)
        errors = np.count_nonzero(all_bits != recovered)
        assert errors == 0, (
            f"Roundtrip failed: {errors}/64 bit errors.\n"
            f"Input:     {all_bits}\n"
            f"Recovered: {recovered}"
        )

    def test_roundtrip_random(self):
        """Random bit stream survives mod → demod (noise-free)."""
        rng = np.random.default_rng(123)
        bits = rng.integers(0, 2, size=4000)
        symbols = qam16_modulate(bits)
        recovered = qam16_demodulate(symbols)
        np.testing.assert_array_equal(bits, recovered)

    def test_constellation_points(self):
        """Verify that 16-QAM generates exactly the 16-point constellation."""
        all_bits = np.array(
            [[b3, b2, b1, b0]
             for b3 in range(2) for b2 in range(2)
             for b1 in range(2) for b0 in range(2)]
        ).ravel()

        symbols = qam16_modulate(all_bits)
        # Expected constellation: {-3, -1, +1, +3} / sqrt(10) on each axis
        expected_levels = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
        real_parts = np.sort(np.unique(np.round(symbols.real, 10)))
        imag_parts = np.sort(np.unique(np.round(symbols.imag, 10)))
        np.testing.assert_allclose(real_parts, expected_levels, atol=1e-10)
        np.testing.assert_allclose(imag_parts, expected_levels, atol=1e-10)

    def test_unit_average_energy(self):
        """16-QAM has unit average symbol energy."""
        all_bits = np.array(
            [[b3, b2, b1, b0]
             for b3 in range(2) for b2 in range(2)
             for b1 in range(2) for b0 in range(2)]
        ).ravel()

        symbols = qam16_modulate(all_bits)
        mean_energy = np.mean(np.abs(symbols) ** 2)
        np.testing.assert_allclose(mean_energy, 1.0, atol=1e-10)

    def test_gray_coding_property(self):
        """Adjacent constellation points differ by exactly 1 bit."""
        all_bits = np.array(
            [[b3, b2, b1, b0]
             for b3 in range(2) for b2 in range(2)
             for b1 in range(2) for b0 in range(2)]
        )
        symbols = qam16_modulate(all_bits.ravel())

        # Check that symbols adjacent on the I-axis (same Q) differ by 1 bit
        # Group by Q value
        for q_bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            group_mask = (all_bits[:, 2] == q_bits[0]) & (all_bits[:, 3] == q_bits[1])
            group_syms = symbols[group_mask]
            group_bits_arr = all_bits[group_mask]
            # Sort by real part
            order = np.argsort(group_syms.real)
            sorted_bits = group_bits_arr[order]
            for i in range(len(sorted_bits) - 1):
                hamming = np.count_nonzero(sorted_bits[i] != sorted_bits[i + 1])
                assert hamming == 1, (
                    f"Adjacent symbols {sorted_bits[i]} and {sorted_bits[i+1]} "
                    f"differ by {hamming} bits (expected 1)"
                )


# ===================================================================
# OFDM
# ===================================================================

class TestOFDM:
    """OFDM transmit / receive roundtrip tests."""

    def test_roundtrip_noise_free(self):
        """OFDM Tx → Rx recovers original symbols without channel/noise."""
        rng = np.random.default_rng(99)
        num_sc = 64
        cp_len = 16
        data = (rng.normal(size=num_sc) + 1j * rng.normal(size=num_sc)) / np.sqrt(2)

        tx_signal = ofdm_transmit(data, num_sc, cp_len)
        rx_symbols = ofdm_receive(tx_signal, num_sc, cp_len)
        np.testing.assert_allclose(rx_symbols, data, atol=1e-10)

    def test_qpsk_ofdm_roundtrip(self):
        """QPSK → OFDM Tx → Rx → demod recovers original bits."""
        rng = np.random.default_rng(77)
        num_sc = 64
        cp_len = 16
        bits = rng.integers(0, 2, size=2 * num_sc)

        symbols = qpsk_modulate(bits)
        tx_signal = ofdm_transmit(symbols, num_sc, cp_len)
        rx_symbols = ofdm_receive(tx_signal, num_sc, cp_len)
        recovered = qpsk_demodulate(rx_symbols)
        np.testing.assert_array_equal(bits, recovered)

    def test_qam16_ofdm_roundtrip(self):
        """16-QAM → OFDM Tx → Rx → demod recovers original bits."""
        rng = np.random.default_rng(55)
        num_sc = 64
        cp_len = 16
        bits = rng.integers(0, 2, size=4 * num_sc)

        symbols = qam16_modulate(bits)
        tx_signal = ofdm_transmit(symbols, num_sc, cp_len)
        rx_symbols = ofdm_receive(tx_signal, num_sc, cp_len)
        recovered = qam16_demodulate(rx_symbols)
        np.testing.assert_array_equal(bits, recovered)

    def test_cp_length(self):
        """Transmitted signal has correct length: num_subcarriers + cp_length."""
        num_sc = 64
        cp_len = 16
        data = np.ones(num_sc, dtype=complex)
        tx = ofdm_transmit(data, num_sc, cp_len)
        assert len(tx) == num_sc + cp_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
