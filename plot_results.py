"""Figures for all 12 experiments in the massive-MIMO pipeline.

Each plot function takes experiment-result dict + output Path and saves a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np

from metrics import cdf


COLORS = {
    "blue": "#2563EB",
    "orange": "#EA580C",
    "green": "#16A34A",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "slate": "#475569",
}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.labelpad": 7,
    "figure.facecolor": "white",
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CBD5E1",
})


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def _finish(fig_or_none, output: Path) -> None:
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close("all")


def _style_axis(ax) -> None:
    """Keep common axis layering without drawing a background grid."""
    ax.set_axisbelow(True)


def _style_line_axis(ax) -> None:
    """Apply a consistent readable treatment to line-based result charts."""
    _style_axis(ax)
    ax.tick_params(axis="both", which="major", labelsize=9)


def _numeric_x_axis(ax, values: np.ndarray, padding_step: float) -> np.ndarray:
    """Keep physical x positions and label only the simulated values."""
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    padding = max(0.04 * span, 0.5 * padding_step)
    x_min, x_max = values.min() - padding, values.max() + padding

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(values, [f"{value:g}" for value in values])
    ax.xaxis.set_minor_locator(NullLocator())
    return values


def _combined_legend(ax, extra_ax, **kwargs) -> None:
    """Create a single legend for a paired-axis plot."""
    lines, labels = ax.get_legend_handles_labels()
    extra_lines, extra_labels = extra_ax.get_legend_handles_labels()
    ax.legend(lines + extra_lines, labels + extra_labels, **kwargs)


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
            ax.plot(angles, beam_db, linewidth=1.15, alpha=0.9)
        ax.set_ylim(-35, 1)
        ax.set_title(f"ULA  N = {N}")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Normalized gain (dB)")
        _style_line_axis(ax)
    fig.suptitle("DFT-codebook beam patterns vs. array size", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 2. SNR vs angle and distance (heatmap)
# -----------------------------------------------------------------------

def plot_snr_vs_angle(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, label, cmap in [
        (axes[0], "snr_db", "Best-beam SNR (dB)", "viridis"),
        (axes[1], "rate", "Spectral efficiency (bit/s/Hz)", "plasma"),
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
        colorbar = plt.colorbar(im, ax=ax, pad=0.025)
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("SNR and rate vs. user angle and distance", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 3. OFDM BER with / without beamforming
# -----------------------------------------------------------------------

def plot_ofdm_ber(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    floor = 1e-5

    def _safe(values):
        return np.maximum(np.asarray(values, dtype=float), floor)

    ax.semilogy(data["snr_db"], _safe(data["ber_no_bf"]), "o-", lw=2.1,
                color=COLORS["blue"], label="QPSK, single antenna")
    ax.semilogy(data["snr_db"], _safe(data["ber_bf"]), "s--", lw=2.1,
                color=COLORS["orange"], label="QPSK, matched beamforming")
    # 16-QAM curve (may use a different SNR range)
    if "ber_16qam" in data:
        snr_16 = data.get("snr_db_16qam", data["snr_db"])
        ax.semilogy(snr_16, _safe(data["ber_16qam"]), "^-.", lw=2.1,
                    color=COLORS["green"], label="16-QAM, single antenna")
    ax.set_xlabel("Pre-beamforming SNR (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title("CP-OFDM over multipath channel with LS pilot estimation")
    ax.set_xlim(float(np.min(data["snr_db"])) - 0.8, float(np.max(snr_16)) + 0.8)
    ax.set_xticks(np.arange(0, int(np.max(snr_16)) + 1, 5))
    ax.set_ylim(bottom=floor / 1.8, top=0.6)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.text(0.01, 0.02, "Values at 10⁻⁵ indicate zero observed errors.",
            transform=ax.transAxes, fontsize=8, color=COLORS["slate"])
    _finish(fig, output)


# -----------------------------------------------------------------------
# 4. Rate vs antennas
# -----------------------------------------------------------------------

def plot_rate_vs_antennas(data: dict, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    antennas = np.asarray(data["antennas"])
    # Three rate curves
    if "raw_rate" in data:
        ax1.plot(antennas, data["raw_rate"], "D-", color=COLORS["green"],
                 lw=2, label="Raw rate (no training overhead)")
    ax1.plot(antennas, data["rate"], "o-", color=COLORS["blue"], lw=2,
             label="Effective rate (exhaustive sweep)")
    if "topk_rate" in data:
        ax1.plot(antennas, data["topk_rate"], "s-", color=COLORS["purple"], lw=2,
                 label="Effective rate (location top-K)")
    ax2.plot(antennas, data["peak_snr_db"], "^--", color=COLORS["orange"],
             alpha=0.85, lw=1.8, label="Peak SNR")
    ax1.set_xlabel("Number of antenna elements")
    ax1.set_ylabel("Mean rate (bit/s/Hz)")
    ax2.set_ylabel("Mean peak SNR (dB)")
    ax1.set_title("Array Gain vs. Training Overhead")
    _style_line_axis(ax1)
    _numeric_x_axis(ax1, antennas, padding_step=4)
    ax2.tick_params(axis="y", colors="black", labelsize=9)
    _combined_legend(ax1, ax2, fontsize=9, loc="upper left")
    _finish(fig, output)


# -----------------------------------------------------------------------
# 5. Codebook size trade-off
# -----------------------------------------------------------------------

def plot_codebook_size(data: dict, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.8, 4.9))
    ax2 = ax1.twinx()
    sizes = np.asarray(data["codebook_size"])
    ax1.plot(sizes, data["rate"], "o-", color=COLORS["blue"],
             lw=2.1, label="Effective rate")
    ax2.plot(sizes, data["peak_snr_db"], "s--",
             color=COLORS["orange"], lw=2.0, label="Peak SNR")
    ax1.set_xlabel("Codebook beams swept")
    ax1.set_ylabel("Effective spectral efficiency (bit/s/Hz)")
    ax2.set_ylabel("Peak post-BF SNR (dB)")
    ax1.set_title("Beam quality vs. training overhead")
    _style_line_axis(ax1)
    _numeric_x_axis(ax1, sizes, padding_step=8)
    ax2.tick_params(axis="y", colors="black", labelsize=9)
    _combined_legend(
        ax1, ax2,
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
        frameon=False, fontsize=9,
    )
    _finish(fig, output)


# -----------------------------------------------------------------------
# 6. Overhead vs speed
# -----------------------------------------------------------------------

def plot_overhead_vs_speed(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    speeds = np.asarray(data["speeds_kmh"])
    for key, label, marker, color, linestyle in [
        ("exhaustive_rate", "Exhaustive", "o", COLORS["blue"], "-"),
        ("hierarchical_rate", "Hierarchical", "s", COLORS["orange"], "-"),
        ("location_topk_rate", "Location top-K", "^", COLORS["green"], "-"),
        ("fusion_mlp_rate", "Fusion MLP top-K", "D", COLORS["purple"], "-"),
        ("gradient_boosted_rate", "Gradient-boosted top-K", "P", COLORS["red"], "--"),
    ]:
        ax.plot(speeds, data[key], marker=marker, linestyle=linestyle, lw=2.2, ms=6,
                color=color, label=label)
    ax.set_xlabel("User speed (km/h)")
    ax.set_ylabel("Mean effective rate (bit/s/Hz)")
    ax.set_title("Doppler-limited beam-training overhead under mobility")
    _style_line_axis(ax)
    _numeric_x_axis(ax, speeds, padding_step=10)
    ax.legend(ncol=2, fontsize=8.5, frameon=True)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 7. Beam-selection comparison
# -----------------------------------------------------------------------

def plot_beam_selection(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # Bar chart: mean rate
    methods = ["exhaustive", "hierarchical", "top_k"]
    labels = ["Exhaustive", "Hierarchical", "Top-K"]
    mean_r = [data["mean_rates"][m] for m in methods]
    mean_p = [data["mean_pilots"][m] for m in methods]

    x = np.arange(len(methods))
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    bars = axes[0].bar(x, mean_r, color=colors, width=0.66)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean effective rate (bit/s/Hz)")
    axes[0].set_title("Rate comparison")
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    _style_axis(axes[0])

    bars = axes[1].bar(x, mean_p, color=colors, width=0.66)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean pilot measurements")
    axes[1].set_title("Pilot overhead comparison")
    axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    _style_axis(axes[1])

    agreement = [100.0 * data.get("beam_accuracy", {}).get(m, np.nan) for m in methods]
    bars = axes[2].bar(x, agreement, color=colors, width=0.66)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Agreement with exhaustive beam (%)")
    axes[2].set_title("Selected-beam quality")
    axes[2].bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)
    _style_axis(axes[2])

    fig.suptitle("Beam selection: throughput, pilot cost and selected-beam quality", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 8. Beamforming architecture comparison
# -----------------------------------------------------------------------

def plot_bf_comparison(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # CDF
    for key, label, color in [
        ("analog_rates", "Analog", COLORS["blue"]),
        ("hybrid_rates", "Hybrid", COLORS["orange"]),
        ("digital_rates", "Digital", COLORS["green"]),
    ]:
        vals, prob = cdf(data[key])
        axes[0].plot(vals, prob, lw=2.0, color=color, label=label)
    axes[0].set_xlabel("Spectral efficiency (bit/s/Hz)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Rate CDF by beamforming architecture")
    _style_line_axis(axes[0])
    axes[0].legend()

    # vs antenna count
    antennas = np.asarray(data["antennas"])
    for key, label, color, marker in [
        ("analog_vs_ant", "Analog", COLORS["blue"], "o"),
        ("hybrid_vs_ant", "Hybrid", COLORS["orange"], "s"),
        ("digital_vs_ant", "Digital", COLORS["green"], "D"),
    ]:
        axes[1].plot(antennas, data[key], marker=marker, lw=2.0,
                     color=color, label=label)
    axes[1].set_xlabel("Antenna elements")
    axes[1].set_ylabel("Mean rate (bit/s/Hz)")
    axes[1].set_title("Rate vs. array size")
    _style_line_axis(axes[1])
    _numeric_x_axis(axes[1], antennas, padding_step=4)
    axes[1].legend()

    fig.suptitle("Analog vs. Hybrid vs. Digital Beamforming", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 9. Multi-user
# -----------------------------------------------------------------------

def plot_multiuser(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    users = np.asarray(data["users"])
    specs = [
        ("sum_rate", "Sum rate (bit/s/Hz)"),
        ("per_user_rate", "Per-user rate (bit/s/Hz)"),
        ("fairness", "Jain fairness index"),
    ]
    for ax, (key, ylabel) in zip(axes, specs):
        ax.plot(users, data[key], "o-", lw=2.1, ms=6, color=COLORS["blue"])
        ax.set_xlabel("Number of users")
        ax.set_ylabel(ylabel)
        _style_line_axis(ax)
        _numeric_x_axis(ax, users, padding_step=2)
    fig.suptitle("Multi-user ZF beamforming", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 10. Handover / outage vs speed
# -----------------------------------------------------------------------

def plot_handover(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    speeds = np.asarray(data["speeds_kmh"])

    # Helper: plot line with optional 95% CI shading
    def _plot_ci(ax, key, marker, color, ylabel, title):
        ax.plot(speeds, data[key], f"{marker}-", color=color, lw=2.1, ms=6)
        ci_lo = f"{key}_ci_low"
        ci_hi = f"{key}_ci_high"
        if ci_lo in data and ci_hi in data:
            lower = np.clip(data[ci_lo], 0.0, 1.0) if "prob" in key or "rate" in key else data[ci_lo]
            upper = np.clip(data[ci_hi], 0.0, 1.0) if "prob" in key or "rate" in key else data[ci_hi]
            ax.fill_between(speeds, lower, upper,
                            color=color, alpha=0.18, label="95% CI")
            ax.legend(fontsize=8)
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if "prob" in key or "rate" in key:
            upper_bound = max(float(np.max(upper)), float(np.max(data[key])))
            ax.set_ylim(0.0, min(1.0, max(0.02, 1.18 * upper_bound)))
        _style_line_axis(ax)
        _numeric_x_axis(ax, speeds, padding_step=10)

    _plot_ci(axes[0], "outage_prob", "o", "tab:red",
             "Outage probability", "Outage vs. speed")
    _plot_ci(axes[1], "ho_failure_rate", "s", "tab:orange",
             "Handover failure rate", "HO failure vs. speed")
    _plot_ci(axes[2], "mean_sinr_db", "^", "tab:blue",
             "Mean SINR (dB)", "Mean SINR vs. speed")

    fig.suptitle("Mobility and handover performance (30 Monte Carlo trials)", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 11. Optimization convergence
# -----------------------------------------------------------------------

def plot_optimization(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["iterations"], data["pso_utility"], lw=2, color=COLORS["blue"], label="PSO")
    ax.plot(data["iterations"], data["ga_utility"], lw=2, color=COLORS["orange"], label="GA")
    ax.plot(data["iterations"], data["greedy_utility"], lw=2, color=COLORS["green"], label="Greedy")
    ax.axhline(float(data["equal_utility"]), color=COLORS["slate"],
                linestyle="--", label="Equal power")
    ax.axhline(float(data["grid_utility"]), color=COLORS["red"],
                linestyle=":", label="Grid search")
    ax.set_xlabel("Iteration / generation")
    ax.set_ylabel("Proportional-fair utility")
    ax.set_title("Power-allocation convergence against grid-search reference")
    _style_line_axis(ax)
    ax.legend()
    _finish(fig, output)


# -----------------------------------------------------------------------
# 12. ML ablation
# -----------------------------------------------------------------------

def plot_ml_ablation(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    methods = list(data["methods"])
    display_labels = ["Location\nMLP", "History\nMarkov", "Fusion\nMLP", "Gradient\nBoosting"]
    x = np.arange(len(methods))
    width = 0.35

    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    bars = axes[0].bar(x, data["top1_accuracy"] * 100, width, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_labels, fontsize=8.5)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Beam prediction — top-1")
    axes[0].set_ylim(0, 105)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    _style_axis(axes[0])

    bars = axes[1].bar(x, data["topk_accuracy"] * 100, width, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_labels, fontsize=8.5)
    axes[1].set_ylabel(f"Top-{data['k']} accuracy (%)")
    axes[1].set_title(f"Beam prediction — top-{data['k']}")
    axes[1].set_ylim(0, 105)
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    _style_axis(axes[1])

    bars = axes[2].bar(x, data["topk_effective_rate"], width, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(display_labels, fontsize=8.5)
    axes[2].set_ylabel(f"Top-{data['k']} effective rate (bit/s/Hz)")
    axes[2].set_title("Rate after pilot overhead")
    axes[2].set_ylim(0, 1.15 * float(np.max(data["topk_effective_rate"])))
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    _style_axis(axes[2])

    fig.suptitle("Trajectory-disjoint ML beam-prediction ablation", fontsize=13)
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
