"""Mobility model and A3-event handover simulation for multi-cell networks.

Provides random-waypoint mobility traces, RSRP computation per cell,
3GPP-inspired A3 handover triggering, and a full mobility simulation loop
with Monte Carlo trials and confidence intervals.
"""

from __future__ import annotations

import numpy as np

from array_model import dft_codebook, steering_vector
from channel_model import path_loss_db
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
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """RSRP including DFT-codebook array gain from each candidate cell.

    This is a reference-signal measurement: each cell can sound its best beam
    towards the UE.  Interference leakage is accounted for separately in the
    SINR calculation, because neighbouring cells normally steer to other UEs.
    """
    num_cells = len(bs_positions)
    if codebook is None:
        codebook, _ = dft_codebook(cfg.antennas, cfg.codebook_beams)
    rsrp = np.empty(num_cells)

    for i, bs in enumerate(bs_positions):
        rel = position - bs
        d = float(np.linalg.norm(rel))
        path_gain = 10.0 ** (-path_loss_db(d, cfg.carrier_hz, "uma_los") / 10.0)
        spatial_freq = rel[1] / max(d, 1e-9)
        h = np.sqrt(cfg.antennas * path_gain) * steering_vector(
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
# Single-trial mobility + handover simulation
# ===================================================================

def _run_single_trial(
    speed_mps: float,
    bs_positions: np.ndarray,
    cfg: SystemConfig,
    rng: np.random.Generator,
    sim_steps: int,
    ttt_steps: int,
    dt: float,
) -> dict[str, float]:
    """Run one trial of mobility + handover at a given speed.

    Returns per-trial scalar metrics.
    """
    positions, _ = generate_mobility_trace(
        sim_steps, speed_mps, dt,
        area_bounds=(-20, -20, cfg.isd_m + 20, cfg.isd_m * 0.87 + 20),
        rng=rng,
    )

    serving: int | None = None
    rsrp_buffer: list[np.ndarray] = []
    sinr_values = []
    ho_events = 0
    ho_failures = 0

    # Correlated shadow fading per BS (dB)
    shadow_db = rng.normal(0, 4.0, size=len(bs_positions))
    # Blockage is especially important at mmWave.  A short correlated blockage
    # process makes outage a meaningful mobility metric without assuming that
    # every neighbouring base station illuminates the UE with a main lobe.
    blocked = np.zeros(len(bs_positions), dtype=bool)
    codebook, _ = dft_codebook(cfg.antennas, cfg.codebook_beams)

    for t in range(len(positions)):
        rsrp = compute_rsrp_with_beamforming(positions[t], bs_positions, cfg, rng, codebook)
        # Apply shadow fading that decorrelates over time
        decorr_dist = 50.0  # decorrelation distance in metres
        step_dist = speed_mps * dt
        alpha = min(1.0, step_dist / decorr_dist)
        shadow_db = (1 - alpha) * shadow_db + alpha * rng.normal(0, 4.0, size=len(bs_positions))
        rsrp = rsrp * 10.0 ** (shadow_db / 10.0)
        blockage_enter = min(0.02, 0.002 + 0.00025 * speed_mps)
        blockage_exit = 0.15
        blocked = np.where(
            blocked,
            rng.random(len(bs_positions)) >= blockage_exit,
            rng.random(len(bs_positions)) < blockage_enter,
        )
        rsrp = rsrp * np.where(blocked, 10.0 ** (-25.0 / 10.0), 1.0)

        # A UE performs initial access on the strongest measured reference
        # signal instead of being artificially pinned to base station zero.
        if serving is None:
            serving = int(np.argmax(rsrp))

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
            ho_execution_delay_s = 0.05  # 50 ms HO execution time
            distance_during_ho = speed_mps * ho_execution_delay_s
            failure_prob = min(0.8, distance_during_ho / 20.0)
            if rng.random() < failure_prob:
                ho_failures += 1
            else:
                serving = target

        # SINR: serving signal / (interference + noise)
        signal = rsrp[serving]
        # Non-serving cells steer their main lobes towards their scheduled UEs;
        # this UE sees average sidelobe leakage rather than every interferer's
        # full main-lobe gain.
        sidelobe_leakage = 10.0 ** (-18.0 / 10.0)
        interference = sidelobe_leakage * (np.sum(rsrp) - signal)
        sinr_linear = signal / (interference + cfg.noise_power_w)
        sinr_values.append(10.0 * np.log10(max(sinr_linear, 1e-10)))

    sinr_arr = np.array(sinr_values)
    return {
        "outage_prob": float(np.mean(sinr_arr < 0)),
        "ho_count": ho_events,
        "ho_failure_rate": ho_failures / max(ho_events, 1),
        "mean_sinr_db": float(np.mean(sinr_arr)),
    }


# ===================================================================
# Full Monte Carlo mobility + handover simulation
# ===================================================================

def simulate_mobility(cfg: SystemConfig) -> dict[str, np.ndarray]:
    """Run a multi-cell Monte Carlo mobility simulation with A3 handover.

    Runs ``n_trials`` independent trials per speed to produce statistically
    meaningful metrics with 95 % confidence intervals.

    Returns
    -------
    dict with keys:
        speeds_kmh, outage_prob, ho_count, ho_failure_rate, mean_sinr_db,
        and *_ci_low / *_ci_high variants for confidence intervals.
    """
    dt = cfg.mobility_step_s

    # Hexagonal-ish BS layout
    bs_positions = np.array([
        [0.0, 0.0],
        [cfg.isd_m, 0.0],
        [cfg.isd_m / 2.0, cfg.isd_m * np.sqrt(3) / 2.0],
    ])

    speeds_mps = np.array([1.0, 3.0, 8.3, 16.7, 33.3])  # ~3.6, 11, 30, 60, 120 km/h
    ttt_steps = max(1, int(round(cfg.handover_ttt_ms / (dt * 1e3))))

    n_trials = cfg.mobility_trials
    sim_steps = max(1, int(round(cfg.mobility_duration_s / dt)))

    metric_keys = ["outage_prob", "ho_count", "ho_failure_rate", "mean_sinr_db"]

    all_results: dict[str, list] = {
        "speeds_kmh": [],
    }
    for mk in metric_keys:
        all_results[mk] = []
        all_results[f"{mk}_ci_low"] = []
        all_results[f"{mk}_ci_high"] = []

    for speed in speeds_mps:
        trial_metrics = {mk: [] for mk in metric_keys}

        for trial in range(n_trials):
            # Each trial gets a unique seed derived from speed + trial index
            trial_rng = np.random.default_rng(
                cfg.seed + 200 + int(speed * 100) + trial * 7
            )
            result = _run_single_trial(
                speed, bs_positions, cfg, trial_rng, sim_steps, ttt_steps, dt,
            )
            for mk in metric_keys:
                trial_metrics[mk].append(result[mk])

        all_results["speeds_kmh"].append(speed * 3.6)
        for mk in metric_keys:
            values = np.array(trial_metrics[mk])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            ci_half = 1.96 * std / np.sqrt(n_trials)
            all_results[mk].append(mean)
            all_results[f"{mk}_ci_low"].append(mean - ci_half)
            all_results[f"{mk}_ci_high"].append(mean + ci_half)

    return {k: np.array(v) for k, v in all_results.items()}
