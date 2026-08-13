"""Shared parameters for the 5G-inspired mmWave massive-MIMO simulator."""

import math
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class SystemConfig:
    # ---------- RF / carrier ----------
    carrier_hz: float = 28e9
    bandwidth_hz: float = 100e6
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 7.0

    # ---------- Antenna array ----------
    antennas: int = 32

    # ---------- Codebook ----------
    codebook_beams: int = 32

    # ---------- OFDM ----------
    num_subcarriers: int = 64
    cp_length: int = 16
    pilot_spacing: int = 4     # one pilot every N subcarriers

    # ---------- Frame timing ----------
    frame_s: float = 1e-3
    pilot_s: float = 2e-6

    # ---------- Multi-user / MIMO ----------
    num_rf_chains: int = 4

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
    isd_m: float = 200.0

    # ---------- ML ----------
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

    def __post_init__(self) -> None:
        """Reject unsupported or physically invalid teaching configurations.

        The default suite intentionally models a single-stream ULA link.  UPA
        and multi-stream helpers remain available as standalone extensions,
        but are not configuration options for the core educational pipeline.
        """
        integer_values = {
            "antennas": self.antennas,
            "codebook_beams": self.codebook_beams,
            "num_subcarriers": self.num_subcarriers,
            "cp_length": self.cp_length,
            "pilot_spacing": self.pilot_spacing,
            "num_rf_chains": self.num_rf_chains,
            "top_k": self.top_k,
            "mobility_trials": self.mobility_trials,
            "ml_hidden_units": self.ml_hidden_units,
            "ml_epochs": self.ml_epochs,
            "gbt_max_iter": self.gbt_max_iter,
            "gbt_max_leaf_nodes": self.gbt_max_leaf_nodes,
            "gbt_min_samples_leaf": self.gbt_min_samples_leaf,
            "ml_train_episodes": self.ml_train_episodes,
            "ml_test_episodes": self.ml_test_episodes,
            "ml_steps_per_episode": self.ml_steps_per_episode,
            "route_steps": self.route_steps,
            "seed": self.seed,
        }
        non_integer = [
            name for name, value in integer_values.items()
            if isinstance(value, bool) or not isinstance(value, Integral)
        ]
        if non_integer:
            raise ValueError(
                f"Configuration values must be integers: {', '.join(non_integer)}"
            )
        non_positive_integers = [
            name for name, value in integer_values.items()
            if name not in {"cp_length", "seed"} and value <= 0
        ]
        if non_positive_integers:
            raise ValueError(
                "Configuration values must be positive: "
                f"{', '.join(non_positive_integers)}"
            )
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")

        positive_values = {
            "carrier_hz": self.carrier_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "antennas": self.antennas,
            "codebook_beams": self.codebook_beams,
            "num_subcarriers": self.num_subcarriers,
            "frame_s": self.frame_s,
            "pilot_s": self.pilot_s,
            "num_rf_chains": self.num_rf_chains,
            "top_k": self.top_k,
            "mobility_step_s": self.mobility_step_s,
            "mobility_duration_s": self.mobility_duration_s,
            "isd_m": self.isd_m,
            "ml_learning_rate": self.ml_learning_rate,
            "gbt_learning_rate": self.gbt_learning_rate,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {', '.join(invalid)}")
        if not 0 <= self.cp_length < self.num_subcarriers:
            raise ValueError("cp_length must be in [0, num_subcarriers).")
        if not 1 <= self.pilot_spacing <= self.num_subcarriers:
            raise ValueError("pilot_spacing must be in [1, num_subcarriers].")
        if self.num_rf_chains > self.antennas:
            raise ValueError("num_rf_chains cannot exceed antennas.")
        if self.top_k > self.codebook_beams:
            raise ValueError("top_k cannot exceed codebook_beams.")
        if not 0 < self.ml_train_episodes < self.ml_train_episodes + self.ml_test_episodes:
            raise ValueError("ML training episodes must leave at least one test episode.")
        if self.location_error_std_m < 0:
            raise ValueError("location_error_std_m cannot be negative.")
        if self.handover_ttt_ms < 0:
            raise ValueError("handover_ttt_ms cannot be negative.")

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
    def array_gain_linear(self) -> float:
        """Ideal coherent transmit-array gain for a unit-norm codeword."""
        return float(self.antennas)
