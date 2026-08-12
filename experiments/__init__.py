"""Experiment suite: run individual or all experiments."""

from experiments.exp_beam_patterns import run as beam_patterns
from experiments.exp_snr_vs_angle import run as snr_vs_angle
from experiments.exp_ofdm_ber import run as ofdm_ber
from experiments.exp_rate_vs_antennas import run as rate_vs_antennas
from experiments.exp_codebook_size import run as codebook_size
from experiments.exp_overhead_vs_speed import run as overhead_vs_speed
from experiments.exp_beam_selection import run as beam_selection
from experiments.exp_bf_comparison import run as bf_comparison
from experiments.exp_multiuser import run as multiuser
from experiments.exp_handover import run as handover
from experiments.exp_optimization import run as optimization
from experiments.exp_ml_ablation import run as ml_ablation

from config import SystemConfig


EXPERIMENTS = {
    "beam_patterns": beam_patterns,
    "snr_vs_angle": snr_vs_angle,
    "ofdm_ber": ofdm_ber,
    "rate_vs_antennas": rate_vs_antennas,
    "codebook_size": codebook_size,
    "overhead_vs_speed": overhead_vs_speed,
    "beam_selection": beam_selection,
    "bf_comparison": bf_comparison,
    "multiuser": multiuser,
    "handover": handover,
    "optimization": optimization,
    "ml_ablation": ml_ablation,
}


def run_all(cfg: SystemConfig) -> dict[str, dict]:
    """Run every experiment and return results keyed by name."""
    results = {}
    for name, func in EXPERIMENTS.items():
        print(f"  Running: {name} ...", end=" ", flush=True)
        results[name] = func(cfg)
        print("done")
    return results
