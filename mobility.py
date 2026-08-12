"""Mobility model and A3-event handover simulation for multi-cell networks.

Provides random-waypoint mobility traces, RSRP computation per cell,
3GPP-inspired A3 handover triggering, and a full mobility simulation loop.
"""

from __future__ import annotations

import numpy as np

from array_model import dft_codebook, steering_vector
from channel_model import free_space_amplitude, path_loss_db
from config import SystemConfig


# ===================================================================
# Mobility trace generators
# ===================================================================

def generate_mobility_trace(
    num_steps: int,
    speed_mps: float,
    dt: float,
    area_bounds: tuple[float, float, float, float] = (-20.0, -20.0, 220.0, 200.0),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk mobility within *area_bounds*.

    Returns
    -------
    positions : (num_steps, 2)
    velocities : (num_steps, 2)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    xmin, ymin, xmax, ymax = area_bounds
    positions = np.empty((num_steps, 2))
    velocities = np.empty((num_steps, 2))
    # Start near center
    pos = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    angle = rng.uniform(0, 2 * np.pi)

    for t in range(num_steps):
        vx = speed_mps * np.cos(angle)
        vy = speed_mps * np.sin(angle)
        positions[t] = pos
        velocities[t] = [vx, vy]
        pos = pos + np.array([vx, vy]) * dt
        # Reflect at boundaries
        if pos[0] < xmin or pos[0] > xmax:
            angle = np.pi - angle
            pos[0] = np.clip(pos[0], xmin, xmax)
        if pos[1] < ymin or pos[1] > ymax:
            angle = -angle
            pos[1] = np.clip(pos[1], ymin, ymax)
        # Slight random direction change
        angle += rng.normal(0, 0.08)

    return positions, velocities


# ===================================================================
# RSRP computation
# ===================================================================

def compute_rsrp(
    position: np.ndarray,
    bs_positions: np.ndarray,
    carrier_hz: float,
    tx_power_w: float,
) -> np.ndarray:
    """Reference signal received power from each base station (linear watts).

    Uses simplified UMa path-loss. Returns shape (num_cells,).
    """
    num_cells = len(bs_positions)
    rsrp = np.empty(num_cells)
    for i, bs in enumerate(bs_positions):
        d = float(np.linalg.norm(position - bs))
        pl = path_loss_db(d, carrier_hz, "uma_los")
        rsrp[i] = tx_power_w * 10.0 ** (-pl / 10.0)
    return rsrp


def compute_rsrp_with_beamforming(
    position: np.ndarray,
    bs_positions: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """RSRP including beamforming gain from the best DFT beam per cell."""
    num_cells = len(bs_positions)
    codebook, _ = dft_codebook(cfg.antennas, cfg.codebook_beams)
    rsrp = np.empty(num_cells)

    for i, bs in enumerate(bs_positions):
        rel = position - bs
        d = float(np.linalg.norm(rel))
        amp = free_space_amplitude(d, cfg)
        spatial_freq = rel[1] / max(d, 1e-9)
        h = amp * np.exp(-1j * 2 * np.pi * d / cfg.wavelength_m) * steering_vector(
            cfg.antennas, spatial_freq
        )
        bf_gain = np.max(np.abs(h.conj() @ codebook) ** 2)
        rsrp[i] = cfg.tx_power_w * bf_gain
    return rsrp


# ===================================================================
# A3 handover event
# ===================================================================

def handover_a3_event(
    rsrp_history: np.ndarray,
    serving_cell: int,
    margin_db: float = 3.0,
    ttt_steps: int = 5,
) -> tuple[bool, int]:
    """Check the 3GPP A3 event condition.

    A3: neighbour RSRP − serving RSRP > margin  for ``ttt_steps`` consecutive steps.

    Parameters
    ----------
    rsrp_history : (T, num_cells) — recent RSRP measurements (linear).
    serving_cell : current serving cell index.
    margin_db : hysteresis margin.
    ttt_steps : time-to-trigger in number of steps.

    Returns
    -------
    triggered : bool
    target_cell : int — the best neighbour if triggered, else serving_cell.
    """
    if len(rsrp_history) < ttt_steps:
        return False, serving_cell

    margin_linear = 10.0 ** (margin_db / 10.0)
    recent = rsrp_history[-ttt_steps:]
    num_cells = recent.shape[1]

    for target in range(num_cells):
        if target == serving_cell:
            continue
        if np.all(recent[:, target] > margin_linear * recent[:, serving_cell]):
            return True, target
    return False, serving_cell


# ===================================================================
# Full mobility + handover simulation
# ===================================================================

def simulate_mobility(cfg: SystemConfig) -> dict[str, np.ndarray]:
    """Run a multi-cell mobility simulation with A3 handover.

    Returns
    -------
    dict with keys:
        speeds_kmh, outage_prob, ho_count, ho_failure_rate, mean_sinr_db
    """
    rng = np.random.default_rng(cfg.seed + 200)
    dt = cfg.frame_s

    # Hexagonal-ish BS layout
    bs_positions = np.array([
        [0.0, 0.0],
        [cfg.isd_m, 0.0],
        [cfg.isd_m / 2.0, cfg.isd_m * np.sqrt(3) / 2.0],
    ])

    speeds_mps = np.array([1.0, 3.0, 8.3, 16.7, 33.3])  # ~3.6, 11, 30, 60, 120 km/h

    all_results: dict[str, list] = {
        "speeds_kmh": [],
        "outage_prob": [],
        "ho_count": [],
        "ho_failure_rate": [],
        "mean_sinr_db": [],
    }

    ttt_steps = max(1, int(cfg.handover_ttt_ms / (dt * 1e3)))
    # Use enough steps so even slow users traverse a meaningful distance
    sim_steps = 2000

    for speed in speeds_mps:
        positions, _ = generate_mobility_trace(
            sim_steps, speed, dt,
            area_bounds=(-20, -20, cfg.isd_m + 20, cfg.isd_m * 0.87 + 20),
            rng=rng,
        )

        serving = 0
        rsrp_buffer: list[np.ndarray] = []
        sinr_values = []
        ho_events = 0
        ho_failures = 0

        # Correlated shadow fading per BS (dB) — changes faster at higher speed
        shadow_db = rng.normal(0, 4.0, size=len(bs_positions))

        for t in range(len(positions)):
            rsrp = compute_rsrp(positions[t], bs_positions, cfg.carrier_hz, cfg.tx_power_w)
            # Apply shadow fading that decorrelates over time
            decorr_dist = 50.0  # decorrelation distance in metres
            step_dist = speed * dt
            alpha = min(1.0, step_dist / decorr_dist)
            shadow_db = (1 - alpha) * shadow_db + alpha * rng.normal(0, 4.0, size=len(bs_positions))
            rsrp = rsrp * 10.0 ** (shadow_db / 10.0)

            rsrp_buffer.append(rsrp)
            if len(rsrp_buffer) > ttt_steps + 5:
                rsrp_buffer.pop(0)

            # Check handover
            rsrp_arr = np.array(rsrp_buffer)
            triggered, target = handover_a3_event(
                rsrp_arr, serving, cfg.handover_margin_db, ttt_steps
            )
            if triggered:
                ho_events += 1
                # Handover failure: too-late HO when user already moved away
                # Higher speed → more likely the channel changed during HO execution
                ho_execution_delay_s = 0.05  # 50 ms HO execution time
                distance_during_ho = speed * ho_execution_delay_s
                failure_prob = min(0.8, distance_during_ho / 20.0)
                if rng.random() < failure_prob:
                    ho_failures += 1
                else:
                    serving = target

            # SINR: serving signal / (interference + noise)
            signal = rsrp[serving]
            interference = np.sum(rsrp) - signal
            sinr_linear = signal / (interference + cfg.noise_power_w)
            sinr_values.append(10.0 * np.log10(max(sinr_linear, 1e-10)))

        sinr_arr = np.array(sinr_values)
        all_results["speeds_kmh"].append(speed * 3.6)
        all_results["outage_prob"].append(float(np.mean(sinr_arr < 0)))
        all_results["ho_count"].append(ho_events)
        all_results["ho_failure_rate"].append(
            ho_failures / max(ho_events, 1)
        )
        all_results["mean_sinr_db"].append(float(np.mean(sinr_arr)))

    return {k: np.array(v) for k, v in all_results.items()}
