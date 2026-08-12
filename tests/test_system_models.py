"""Regression tests for beam management, channel scaling and ML utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.hierarchical_search import build_hierarchical_codebook, hierarchical_beam_search
from array_model import steering_vector
from beamforming import multi_user_zf_precoder, top_k_around
from channel_model import geometric_channel
from config import SystemConfig
from ml.beam_predictor import MLPBeamPredictor
from ml.data_generator import split_episodes
from phy import interpolate_channel_dft


def test_top_k_candidates_do_not_wrap_end_fire_directions():
    np.testing.assert_array_equal(top_k_around(0, 32, 3), np.array([0, 1, 2]))
    np.testing.assert_array_equal(top_k_around(31, 32, 3), np.array([29, 30, 31]))


def test_hierarchical_search_returns_its_own_fine_codebook_beam():
    num_antennas = 32
    channel = steering_vector(num_antennas, 0.25)
    codebooks = build_hierarchical_codebook(num_antennas, num_levels=3)
    beam, pilots, gains = hierarchical_beam_search(channel, codebooks)

    assert 0 <= beam < num_antennas
    assert 0 < pilots < num_antennas
    assert len(gains) == 3


def test_geometric_channel_includes_coherent_array_gain():
    position = np.array([90.0, 25.0])
    cfg_16 = SystemConfig(antennas=16, codebook_beams=16)
    cfg_64 = SystemConfig(antennas=64, codebook_beams=64)
    h_16 = geometric_channel(position, cfg_16, np.random.default_rng(7))
    h_64 = geometric_channel(position, cfg_64, np.random.default_rng(7))

    gain_ratio = np.linalg.norm(h_64) ** 2 / np.linalg.norm(h_16) ** 2
    # Multiple paths combine differently across arrays, so the scaling is
    # approximately (rather than algebraically exactly) proportional to Nt.
    np.testing.assert_allclose(gain_ratio, 4.0, rtol=0.05)


def test_zf_precoder_suppresses_multi_user_interference():
    rng = np.random.default_rng(11)
    channels = (rng.normal(size=(4, 8)) + 1j * rng.normal(size=(4, 8))) / np.sqrt(2.0)
    precoder = multi_user_zf_precoder(channels)
    effective_channel = channels.conj() @ precoder
    off_diagonal = effective_channel - np.diag(np.diag(effective_channel))

    assert np.linalg.norm(off_diagonal) / np.linalg.norm(effective_channel) < 1e-7


def test_dft_channel_interpolation_recovers_cp_bounded_channel():
    num_subcarriers = 64
    impulse_response = np.array([0.8 + 0.1j, -0.2 + 0.3j, 0.08 - 0.05j])
    channel = np.fft.fft(impulse_response, n=num_subcarriers)
    pilots = np.arange(0, num_subcarriers, 4)
    estimated = interpolate_channel_dft(pilots, channel[pilots], num_subcarriers, 3)

    np.testing.assert_allclose(estimated, channel, atol=1e-10)


def test_episode_split_has_no_trajectory_overlap():
    episode_ids = np.repeat(np.arange(8), 3)
    train_mask, test_mask = split_episodes(episode_ids, train_episodes=5, seed=12)

    assert not set(episode_ids[train_mask]).intersection(episode_ids[test_mask])


def test_mlp_beam_predictor_learns_a_nonlinear_decision_boundary():
    rng = np.random.default_rng(21)
    features = rng.uniform(-1.0, 1.0, size=(400, 2))
    labels = ((features[:, 0] * features[:, 1]) > 0.0).astype(int)
    predictor = MLPBeamPredictor(
        hidden_units=20, learning_rate=0.03, epochs=160, seed=22, num_classes=2
    ).fit(features, labels)

    accuracy = np.mean(np.array([predictor.predict(row) for row in features]) == labels)
    assert accuracy > 0.95
