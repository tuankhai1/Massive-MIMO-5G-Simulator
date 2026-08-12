"""Figures for all 12 experiments in the massive-MIMO pipeline.

Each plot function takes experiment-result dict + output Path and saves a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import cdf


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def _finish(fig_or_none, output: Path) -> None:
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close("all")


# -----------------------------------------------------------------------
# 1. Beam patterns (multi-panel for different array sizes)
# -----------------------------------------------------------------------

def plot_beam_patterns(data: dict, output: Path) -> None:
    sizes = data["antenna_sizes"]
    angles = data["angles_deg"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    for idx, N in enumerate(sizes):
        ax = axes[idx]
        for beam_db in data["patterns"][N]:
            ax.plot(angles, beam_db, linewidth=0.8)
        ax.set_ylim(-35, 1)
        ax.set_title(f"ULA  N = {N}")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Normalized gain (dB)")
        ax.grid(alpha=0.3)
    fig.suptitle("DFT-codebook beam patterns vs. array size", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 2. SNR vs angle and distance (heatmap)
# -----------------------------------------------------------------------

def plot_snr_vs_angle(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, label, cmap in [
        (axes[0], "snr_db", "Best-beam SNR (dB)", "viridis"),
        (axes[1], "rate",   "Achievable rate (bit/s/Hz)", "plasma"),
    ]:
        im = ax.imshow(
            data[key], aspect="auto", origin="lower",
            extent=[data["angles_deg"][0], data["angles_deg"][-1],
                    data["distances"][0], data["distances"][-1]],
            cmap=cmap,
        )
        ax.set_xlabel("User angle (°)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
    fig.suptitle("SNR and rate vs. user angle and distance", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 3. OFDM BER with / without beamforming
# -----------------------------------------------------------------------

def plot_ofdm_ber(data: dict, output: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(data["snr_db"], data["ber_no_bf"], "o-", label="QPSK OFDM (no BF)")
    plt.semilogy(data["snr_db"], data["ber_bf"], "s--", label="QPSK OFDM + Beamforming")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("OFDM link: effect of beamforming on BER")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    _finish(None, output)


# -----------------------------------------------------------------------
# 4. Rate vs antennas
# -----------------------------------------------------------------------

def plot_rate_vs_antennas(data: dict, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(data["antennas"], data["rate"], "o-", color="tab:blue", label="Effective rate")
    ax2.plot(data["antennas"], data["peak_snr_db"], "s--", color="tab:orange", label="Peak SNR")
    ax1.set_xlabel("Number of antenna elements")
    ax1.set_ylabel("Mean effective rate (bit/s/Hz)", color="tab:blue")
    ax2.set_ylabel("Mean peak SNR (dB)", color="tab:orange")
    ax1.set_title("Massive-MIMO array-size gain")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 5. Codebook size trade-off
# -----------------------------------------------------------------------

def plot_codebook_size(data: dict, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(data["codebook_size"], data["rate"], "o-", label="Effective rate")
    ax2.plot(data["codebook_size"], data["peak_snr_db"], "s--",
             color="tab:orange", label="Peak SNR")
    ax1.set_xlabel("Codebook beams swept")
    ax1.set_ylabel("Effective spectral efficiency (bit/s/Hz)")
    ax2.set_ylabel("Peak post-BF SNR (dB)")
    ax1.set_title("Beam quality vs. training overhead")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 6. Overhead vs speed
# -----------------------------------------------------------------------

def plot_overhead_vs_speed(data: dict, output: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    for key, label in [
        ("exhaustive_rate", "Exhaustive"),
        ("hierarchical_rate", "Hierarchical"),
        ("location_topk_rate", "Location top-K"),
        ("ml_topk_rate", "ML top-K"),
    ]:
        plt.plot(data["speeds_kmh"], data[key], "o-", label=label)
    plt.xlabel("User speed (km/h)")
    plt.ylabel("Mean effective rate (bit/s/Hz)")
    plt.title("Beam-sweeping overhead vs. mobility speed")
    plt.grid(alpha=0.3)
    plt.legend()
    _finish(None, output)


# -----------------------------------------------------------------------
# 7. Beam-selection comparison
# -----------------------------------------------------------------------

def plot_beam_selection(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Bar chart: mean rate
    methods = ["exhaustive", "hierarchical", "top_k"]
    labels = ["Exhaustive", "Hierarchical", "Top-K"]
    mean_r = [data["mean_rates"][m] for m in methods]
    mean_p = [data["mean_pilots"][m] for m in methods]

    x = np.arange(len(methods))
    axes[0].bar(x, mean_r, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean effective rate (bit/s/Hz)")
    axes[0].set_title("Rate comparison")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, mean_p, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean pilot measurements")
    axes[1].set_title("Pilot overhead comparison")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Beam-selection methods: rate vs. pilot cost", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 8. Beamforming architecture comparison
# -----------------------------------------------------------------------

def plot_bf_comparison(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # CDF
    for key, label in [
        ("analog_rates", "Analog"),
        ("hybrid_rates", "Hybrid"),
        ("digital_rates", "Digital"),
    ]:
        vals, prob = cdf(data[key])
        axes[0].plot(vals, prob, label=label)
    axes[0].set_xlabel("Spectral efficiency (bit/s/Hz)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Rate CDF by beamforming architecture")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # vs antenna count
    for key, label in [
        ("analog_vs_ant", "Analog"),
        ("hybrid_vs_ant", "Hybrid"),
        ("digital_vs_ant", "Digital"),
    ]:
        axes[1].plot(data["antennas"], data[key], "o-", label=label)
    axes[1].set_xlabel("Antenna elements")
    axes[1].set_ylabel("Mean rate (bit/s/Hz)")
    axes[1].set_title("Rate vs. array size")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Analog vs. Hybrid vs. Digital Beamforming", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 9. Multi-user
# -----------------------------------------------------------------------

def plot_multiuser(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = [
        ("sum_rate", "Sum rate (bit/s/Hz)"),
        ("per_user_rate", "Per-user rate (bit/s/Hz)"),
        ("fairness", "Jain fairness index"),
    ]
    for ax, (key, ylabel) in zip(axes, specs):
        ax.plot(data["users"], data[key], "o-")
        ax.set_xlabel("Number of users")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle("Multi-user ZF beamforming", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 10. Handover / outage vs speed
# -----------------------------------------------------------------------

def plot_handover(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(data["speeds_kmh"], data["outage_prob"], "o-", color="tab:red")
    axes[0].set_xlabel("Speed (km/h)")
    axes[0].set_ylabel("Outage probability")
    axes[0].set_title("Outage vs. speed")
    axes[0].grid(alpha=0.3)

    axes[1].plot(data["speeds_kmh"], data["ho_failure_rate"], "s-", color="tab:orange")
    axes[1].set_xlabel("Speed (km/h)")
    axes[1].set_ylabel("Handover failure rate")
    axes[1].set_title("HO failure vs. speed")
    axes[1].grid(alpha=0.3)

    axes[2].plot(data["speeds_kmh"], data["mean_sinr_db"], "^-", color="tab:blue")
    axes[2].set_xlabel("Speed (km/h)")
    axes[2].set_ylabel("Mean SINR (dB)")
    axes[2].set_title("Mean SINR vs. speed")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Mobility and handover performance", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 11. Optimization convergence
# -----------------------------------------------------------------------

def plot_optimization(data: dict, output: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(data["iterations"], data["pso_utility"], label="PSO")
    plt.plot(data["iterations"], data["ga_utility"], label="GA")
    plt.plot(data["iterations"], data["greedy_utility"], label="Greedy")
    plt.axhline(float(data["equal_utility"]), color="tab:gray",
                linestyle="--", label="Equal power")
    plt.axhline(float(data["grid_utility"]), color="tab:green",
                linestyle=":", label="Grid search")
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Sum log-rate utility")
    plt.title("Power-allocation optimization convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    _finish(None, output)


# -----------------------------------------------------------------------
# 12. ML ablation
# -----------------------------------------------------------------------

def plot_ml_ablation(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    methods = list(data["methods"])
    x = np.arange(len(methods))
    width = 0.35

    axes[0].bar(x, data["top1_accuracy"] * 100, width,
                color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Beam prediction — top-1")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, data["topk_accuracy"] * 100, width,
                color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel(f"Top-{data['k']} accuracy (%)")
    axes[1].set_title(f"Beam prediction — top-{data['k']}")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("ML ablation: feature-set comparison", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# Master plot dispatcher
# -----------------------------------------------------------------------

PLOT_FUNCTIONS = {
    "beam_patterns": plot_beam_patterns,
    "snr_vs_angle": plot_snr_vs_angle,
    "ofdm_ber": plot_ofdm_ber,
    "rate_vs_antennas": plot_rate_vs_antennas,
    "codebook_size": plot_codebook_size,
    "overhead_vs_speed": plot_overhead_vs_speed,
    "beam_selection": plot_beam_selection,
    "bf_comparison": plot_bf_comparison,
    "multiuser": plot_multiuser,
    "handover": plot_handover,
    "optimization": plot_optimization,
    "ml_ablation": plot_ml_ablation,
}


def plot_all(results: dict, output_dir: Path) -> None:
    """Generate all plots into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, plot_fn in PLOT_FUNCTIONS.items():
        if name in results:
            plot_fn(results[name], output_dir / f"{name}.png")
