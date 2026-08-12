"""Shared parameters for the 5G-inspired mmWave massive-MIMO simulator."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    # ---------- RF / carrier ----------
    carrier_hz: float = 28e9
    bandwidth_hz: float = 100e6
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 7.0

    # ---------- Antenna array ----------
    antennas: int = 32
    upa_rows: int = 1          # 1 → ULA; >1 → UPA (rows × cols = antennas)
    upa_cols: int = 32

    # ---------- Codebook ----------
    codebook_beams: int = 32

    # ---------- OFDM ----------
    num_subcarriers: int = 64
    cp_length: int = 16
    num_ofdm_symbols: int = 14
    pilot_spacing: int = 4     # one pilot every N subcarriers

    # ---------- Frame timing ----------
    frame_s: float = 1e-3
    pilot_s: float = 2e-6

    # ---------- Multi-user / MIMO ----------
    num_users: int = 4
    num_rf_chains: int = 4
    num_streams: int = 1

    # ---------- Beam management ----------
    top_k: int = 3

    # ---------- Mobility ----------
    user_speed_mps: float = 5.0
    handover_margin_db: float = 3.0
    handover_ttt_ms: float = 40.0
    mobility_step_s: float = 10e-3
    mobility_duration_s: float = 20.0
    mobility_trials: int = 30

    # ---------- Multi-cell ----------
    num_cells: int = 3
    isd_m: float = 200.0

    # ---------- ML ----------
    ml_train_fraction: float = 0.5
    knn_neighbors: int = 9
    location_error_std_m: float = 8.0
    ml_hidden_units: int = 64
    ml_epochs: int = 180
    ml_learning_rate: float = 0.025
    gbt_max_iter: int = 180
    gbt_learning_rate: float = 0.08
    gbt_max_leaf_nodes: int = 15
    gbt_min_samples_leaf: int = 12
    ml_train_episodes: int = 72
    ml_test_episodes: int = 24
    ml_steps_per_episode: int = 48

    # ---------- Simulation ----------
    route_steps: int = 150
    seed: int = 2026

    # ---------- derived ----------
    @property
    def wavelength_m(self) -> float:
        return 3e8 / self.carrier_hz

    @property
    def noise_power_w(self) -> float:
        noise_dbm = -174.0 + 10.0 * math.log10(self.bandwidth_hz) + self.noise_figure_db
        return 10.0 ** ((noise_dbm - 30.0) / 10.0)

    @property
    def tx_power_w(self) -> float:
        return 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)

    @property
    def is_upa(self) -> bool:
        return self.upa_rows > 1

    @property
    def array_gain_linear(self) -> float:
        """Ideal coherent transmit-array gain for a unit-norm codeword."""
        return float(self.antennas)
