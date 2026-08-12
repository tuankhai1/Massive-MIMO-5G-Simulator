"""Geometric mmWave channel models with LoS/NLoS, Doppler, and wideband support.

All models are 5G-*inspired* — intentionally simpler than full 3GPP TR 38.901
so that every line of physics is transparent and teachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from array_model import steering_vector
from config import SystemConfig


# ---------------------------------------------------------------------------
# Path-loss models
# ---------------------------------------------------------------------------

def path_loss_db(
    distance_m: float,
    carrier_hz: float,
    model: str = "free_space",
) -> float:
    """Large-scale path loss in dB.

    Supported *model* values:
        ``"free_space"`` — Friis free-space.
        ``"uma_los"``    — 3GPP-inspired UMa LoS (simplified).
        ``"uma_nlos"``   — 3GPP-inspired UMa NLoS (simplified).
    """
    d = max(distance_m, 1.0)
    fc_ghz = carrier_hz / 1e9
    if model == "free_space":
        return 20.0 * np.log10(4.0 * np.pi * d * carrier_hz / 3e8)
    if model == "uma_los":
        # Simplified UMa LoS: PL = 28.0 + 22·log10(d) + 20·log10(fc)
        return 28.0 + 22.0 * np.log10(d) + 20.0 * np.log10(fc_ghz)
    if model == "uma_nlos":
        # Simplified UMa NLoS: PL = 13.54 + 39.08·log10(d) + 20·log10(fc)
        return 13.54 + 39.08 * np.log10(d) + 20.0 * np.log10(fc_ghz)
    raise ValueError(f"Unknown path-loss model: {model!r}")


def los_probability(distance_m: float) -> float:
    """LoS probability as a function of 2-D distance (UMa-inspired)."""
    d = max(distance_m, 1.0)
    if d <= 18.0:
        return 1.0
    return 18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d)


# ---------------------------------------------------------------------------
# Free-space amplitude (legacy helper)
# ---------------------------------------------------------------------------

def free_space_amplitude(distance_m: float, cfg: SystemConfig) -> float:
    """Complex-amplitude gain for Friis free-space propagation."""
    return cfg.wavelength_m / (4.0 * np.pi * max(distance_m, 1.0))


# ---------------------------------------------------------------------------
# Channel realization dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChannelRealization:
    """One snapshot of a cluster-based geometric channel."""
    aod: np.ndarray          # angles of departure (radians), shape (L,)
    aoa: np.ndarray          # angles of arrival (radians),  shape (L,)
    delays: np.ndarray       # propagation delays (seconds), shape (L,)
    gains: np.ndarray        # complex path gains,           shape (L,)
    doppler_hz: np.ndarray   # per-path Doppler shifts,      shape (L,)
    num_paths: int = 0

    def __post_init__(self):
        self.num_paths = len(self.gains)


# ---------------------------------------------------------------------------
# Cluster-based channel generator
# ---------------------------------------------------------------------------

def generate_cluster_channel(
    user_position: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
    base_station: np.ndarray | None = None,
    speed_mps: float = 0.0,
    num_clusters: int = 3,
    paths_per_cluster: int = 4,
) -> ChannelRealization:
    """Clustered geometric channel with per-path delay, AoD, gain, and Doppler.

    Parameters
    ----------
    user_position : (2,) array
    cfg : SystemConfig
    rng : Generator
    base_station : (2,) array or None (defaults to origin)
    speed_mps : user speed in m/s
    num_clusters : number of scattering clusters
    paths_per_cluster : sub-paths within each cluster
    """
    bs = np.zeros(2) if base_station is None else np.asarray(base_station, dtype=float)
    rel = user_position - bs
    d_los = float(np.linalg.norm(rel))
    angle_los = float(np.arctan2(rel[1], rel[0]))

    is_los = rng.random() < los_probability(d_los)

    aods, aoas, delays, gains_list, dopplers = [], [], [], [], []

    # LoS path (if present)
    if is_los:
        pl = path_loss_db(d_los, cfg.carrier_hz, "uma_los")
        amplitude = 10.0 ** (-pl / 20.0)
        phase = np.exp(-1j * 2.0 * np.pi * d_los / cfg.wavelength_m)
        aods.append(angle_los)
        aoas.append(angle_los + np.pi)
        delays.append(d_los / 3e8)
        gains_list.append(amplitude * phase)
        # Doppler: f_d = v/λ · cos(angle between motion and AoA)
        motion_angle = rng.uniform(0, 2.0 * np.pi)
        dopplers.append(speed_mps / cfg.wavelength_m * np.cos(motion_angle - angle_los))
    else:
        motion_angle = rng.uniform(0, 2.0 * np.pi)

    # NLoS scattered clusters
    pl_model = "uma_los" if is_los else "uma_nlos"
    for _ in range(num_clusters):
        cluster_angle = angle_los + rng.uniform(-np.pi / 2, np.pi / 2)
        cluster_delay_extra = rng.exponential(50e-9)  # excess delay
        cluster_distance = d_los + cluster_delay_extra * 3e8
        pl = path_loss_db(cluster_distance, cfg.carrier_hz, pl_model)
        cluster_power = 10.0 ** (-pl / 20.0) * rng.uniform(0.15, 0.55)

        for _ in range(paths_per_cluster):
            sub_angle = cluster_angle + rng.normal(0, 0.15)
            sub_delay = d_los / 3e8 + cluster_delay_extra + rng.exponential(5e-9)
            sub_phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
            aods.append(sub_angle)
            aoas.append(sub_angle + np.pi + rng.normal(0, 0.1))
            delays.append(sub_delay)
            gains_list.append(cluster_power / np.sqrt(paths_per_cluster) * sub_phase)
            dopplers.append(
                speed_mps / cfg.wavelength_m * np.cos(motion_angle - sub_angle)
            )

    return ChannelRealization(
        aod=np.array(aods),
        aoa=np.array(aoas),
        delays=np.array(delays),
        gains=np.array(gains_list, dtype=complex),
        doppler_hz=np.array(dopplers),
    )


# ---------------------------------------------------------------------------
# Apply Doppler
# ---------------------------------------------------------------------------

def apply_doppler(
    cr: ChannelRealization,
    time_s: float,
) -> np.ndarray:
    """Return complex gains with Doppler phase rotation at *time_s*."""
    return cr.gains * np.exp(1j * 2.0 * np.pi * cr.doppler_hz * time_s)


# ---------------------------------------------------------------------------
# Narrowband spatial channel vector
# ---------------------------------------------------------------------------

def narrowband_channel(
    cr: ChannelRealization,
    num_antennas: int,
    time_s: float = 0.0,
) -> np.ndarray:
    """Collapse a channel realization into a single narrowband Nt-vector."""
    gains = apply_doppler(cr, time_s)
    h = np.zeros(num_antennas, dtype=complex)
    for g, aod in zip(gains, cr.aod):
        h += np.sqrt(num_antennas) * g * steering_vector(num_antennas, np.sin(aod))
    return h


# ---------------------------------------------------------------------------
# Wideband (OFDM subcarrier) channel
# ---------------------------------------------------------------------------

def wideband_channel_matrix(
    cr: ChannelRealization,
    num_antennas: int,
    num_subcarriers: int,
    bandwidth_hz: float,
    time_s: float = 0.0,
) -> np.ndarray:
    """Frequency-domain channel H[k] for each OFDM subcarrier.

    Returns
    -------
    H : ndarray, shape (num_subcarriers, num_antennas)
        H[k, :] is the channel vector at subcarrier k.
    """
    gains = apply_doppler(cr, time_s)
    subcarrier_spacing = bandwidth_hz / num_subcarriers
    k = np.arange(num_subcarriers)
    H = np.zeros((num_subcarriers, num_antennas), dtype=complex)
    for g, aod, tau in zip(gains, cr.aod, cr.delays):
        sv = steering_vector(num_antennas, np.sin(aod))
        freq_response = np.exp(-1j * 2 * np.pi * k * subcarrier_spacing * tau)
        H += np.sqrt(num_antennas) * g * np.outer(freq_response, sv)
    return H


# ---------------------------------------------------------------------------
# Legacy geometric channel (backward compatibility)
# ---------------------------------------------------------------------------

def geometric_channel(
    user_position: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
    base_station: np.ndarray | None = None,
) -> np.ndarray:
    """Narrowband channel with LoS and two weak single-bounce components.

    Kept for backward compatibility with existing experiments.
    """
    base_station = np.zeros(2) if base_station is None else np.asarray(base_station, dtype=float)
    relative = user_position - base_station
    distance = float(np.linalg.norm(relative))
    spatial_frequency = relative[1] / max(distance, 1e-9)
    phase = np.exp(-1j * 2.0 * np.pi * distance / cfg.wavelength_m)
    channel = (
        np.sqrt(cfg.antennas)
        * free_space_amplitude(distance, cfg)
        * phase
        * steering_vector(cfg.antennas, spatial_frequency)
    )

    reflectors = np.array([[52.0, -38.0], [96.0, 64.0]])
    reflection_loss = (0.42, 0.25)
    for reflector, loss in zip(reflectors, reflection_loss):
        first_leg = float(np.linalg.norm(reflector - base_station))
        second_leg = float(np.linalg.norm(user_position - reflector))
        total = first_leg + second_leg
        departure = (reflector - base_station)[1] / max(first_leg, 1e-9)
        random_phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))
        channel += (
            np.sqrt(cfg.antennas)
            * loss
            * free_space_amplitude(total, cfg)
            * random_phase
            * np.exp(-1j * 2.0 * np.pi * total / cfg.wavelength_m)
            * steering_vector(cfg.antennas, departure)
        )
    return channel


# ---------------------------------------------------------------------------
# Routes / mobility traces
# ---------------------------------------------------------------------------

def user_route(steps: int) -> np.ndarray:
    """A curved route that naturally causes beam changes."""
    x = np.linspace(25.0, 145.0, steps)
    y = 15.0 + 30.0 * np.sin(np.linspace(-0.9, 1.2, steps))
    return np.column_stack([x, y])


def mobility_route(
    steps: int,
    speed_mps: float = 5.0,
    dt: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-walk route with given speed.

    Returns
    -------
    positions : (steps, 2)
    velocities : (steps, 2)
    timestamps : (steps,)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    timestamps = np.arange(steps) * dt
    angles = np.cumsum(rng.normal(0, 0.15, steps))
    vx = speed_mps * np.cos(angles)
    vy = speed_mps * np.sin(angles)
    velocities = np.column_stack([vx, vy])
    positions = np.cumsum(velocities * dt, axis=0) + np.array([50.0, 50.0])
    return positions, velocities, timestamps
