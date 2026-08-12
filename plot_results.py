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
    "figure.facecolor": "white",
})


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def _finish(fig_or_none, output: Path) -> None:
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close("all")


def _style_axis(ax) -> None:
    ax.grid(alpha=0.24, linewidth=0.8)
    ax.set_axisbelow(True)


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
        _style_axis(ax)
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
        plt.colorbar(im, ax=ax)
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
    ax.set_ylim(bottom=floor / 1.8, top=0.6)
    ax.grid(alpha=0.24, which="both")
    ax.legend(frameon=True, fontsize=9)
    ax.text(0.01, 0.02, "Values at 10⁻⁵ indicate zero observed errors.",
            transform=ax.transAxes, fontsize=8, color=COLORS["slate"])
    _finish(fig, output)


# -----------------------------------------------------------------------
# 4. Rate vs antennas
# -----------------------------------------------------------------------

def plot_rate_vs_antennas(data: dict, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    # Three rate curves
    if "raw_rate" in data:
        ax1.plot(data["antennas"], data["raw_rate"], "D-", color=COLORS["green"],
                 lw=2, label="Raw rate (no training overhead)")
    ax1.plot(data["antennas"], data["rate"], "o-", color=COLORS["blue"], lw=2,
             label="Effective rate (exhaustive sweep)")
    if "topk_rate" in data:
        ax1.plot(data["antennas"], data["topk_rate"], "s-", color=COLORS["purple"], lw=2,
                 label="Effective rate (location top-K)")
    ax2.plot(data["antennas"], data["peak_snr_db"], "^--", color=COLORS["orange"],
             alpha=0.85, lw=1.8, label="Peak SNR")
    ax1.set_xlabel("Number of antenna elements")
    ax1.set_ylabel("Mean rate (bit/s/Hz)")
    ax2.set_ylabel("Mean peak SNR (dB)", color="tab:orange")
    ax1.set_title("Array Gain vs. Training Overhead")
    _style_axis(ax1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 5. Codebook size trade-off
# -----------------------------------------------------------------------

def plot_codebook_size(data: dict, output: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(data["codebook_size"], data["rate"], "o-", color=COLORS["blue"],
             lw=2.1, label="Effective rate")
    ax2.plot(data["codebook_size"], data["peak_snr_db"], "s--",
             color=COLORS["orange"], lw=2.0, label="Peak SNR")
    ax1.set_xlabel("Codebook beams swept")
    ax1.set_ylabel("Effective spectral efficiency (bit/s/Hz)")
    ax2.set_ylabel("Peak post-BF SNR (dB)")
    ax1.set_title("Beam quality vs. training overhead")
    _style_axis(ax1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 6. Overhead vs speed
# -----------------------------------------------------------------------

def plot_overhead_vs_speed(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for key, label, marker, color in [
        ("exhaustive_rate", "Exhaustive", "o", COLORS["blue"]),
        ("hierarchical_rate", "Hierarchical", "s", COLORS["orange"]),
        ("location_topk_rate", "Location top-K", "^", COLORS["green"]),
        ("ml_topk_rate", "Fusion MLP top-K", "D", COLORS["purple"]),
    ]:
        ax.plot(data["speeds_kmh"], data[key], marker=marker, lw=2.2, ms=6,
                color=color, label=label)
    ax.set_xlabel("User speed (km/h)")
    ax.set_ylabel("Mean effective rate (bit/s/Hz)")
    ax.set_title("Doppler-limited beam-training overhead under mobility")
    _style_axis(ax)
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
    _style_axis(axes[0])
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
    _style_axis(axes[1])
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
        _style_axis(ax)
    fig.suptitle("Multi-user ZF beamforming", fontsize=13)
    _finish(fig, output)


# -----------------------------------------------------------------------
# 10. Handover / outage vs speed
# -----------------------------------------------------------------------

def plot_handover(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    speeds = data["speeds_kmh"]

    # Helper: plot line with optional 95% CI shading
    def _plot_ci(ax, key, marker, color, ylabel, title):
        ax.plot(speeds, data[key], f"{marker}-", color=color)
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
        _style_axis(ax)

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
    _style_axis(ax)
    ax.legend()
    _finish(fig, output)


# -----------------------------------------------------------------------
# 12. ML ablation
# -----------------------------------------------------------------------

def plot_ml_ablation(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    methods = list(data["methods"])
    x = np.arange(len(methods))
    width = 0.35

    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    bars = axes[0].bar(x, data["top1_accuracy"] * 100, width, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Beam prediction — top-1")
    axes[0].set_ylim(0, 105)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    _style_axis(axes[0])

    bars = axes[1].bar(x, data["topk_accuracy"] * 100, width, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel(f"Top-{data['k']} accuracy (%)")
    axes[1].set_title(f"Beam prediction — top-{data['k']}")
    axes[1].set_ylim(0, 105)
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    _style_axis(axes[1])

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
