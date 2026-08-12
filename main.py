"""End-to-end mmWave massive-MIMO 5G simulator — unified entry point.

Usage
-----
    python main.py                      # run ALL experiments
    python main.py --experiment ofdm_ber  # run one experiment
    python main.py --list               # list available experiments
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from config import SystemConfig
from experiments import EXPERIMENTS, run_all
from plot_results import PLOT_FUNCTIONS, plot_all


def _json_default(value):
    """Serialize NumPy scalars/arrays when writing result metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_raw_results(results: dict, output_dir: Path) -> None:
    """Persist arrays and lightweight metadata for reproducible plot/report runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment, data in results.items():
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, object] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif np.isscalar(value):
                arrays[key] = np.asarray(value)
            else:
                metadata[key] = value
        np.savez_compressed(output_dir / f"{experiment}.npz", **arrays)
        (output_dir / f"{experiment}.json").write_text(
            json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
        )


def build_summary(results: dict, cfg: SystemConfig) -> str:
    """Produce a human-readable text summary of all experiment results."""
    lines = [
        "=" * 68,
        " mmWave Massive-MIMO 5G Simulator — Results Summary",
        "=" * 68,
        f"Carrier: {cfg.carrier_hz / 1e9:.1f} GHz   |   BW: {cfg.bandwidth_hz / 1e6:.0f} MHz   |   Tx: {cfg.tx_power_dbm:.0f} dBm",
        f"Antennas: {cfg.antennas}   |   Codebook beams: {cfg.codebook_beams}   |   RF chains: {cfg.num_rf_chains}",
        f"Users: {cfg.num_users}   |   Subcarriers: {cfg.num_subcarriers}   |   Top-K: {cfg.top_k}",
        "-" * 68,
    ]

    if "rate_vs_antennas" in results:
        d = results["rate_vs_antennas"]
        lines.append("[Rate vs. Antennas — Array Gain vs. Training Overhead]")
        has_raw = "raw_rate" in d
        has_topk = "topk_rate" in d
        header = "  {:>4s}  {:>8s}".format("Nt", "Exh.")
        if has_raw:
            header = "  {:>4s}  {:>8s}  {:>8s}".format("Nt", "Raw", "Exh.")
        if has_topk:
            header += "  {:>8s}".format("Top-K")
        lines.append(header)
        for i, n in enumerate(d["antennas"]):
            row = f"  {n:>4d}"
            if has_raw:
                row += f"  {d['raw_rate'][i]:>8.3f}"
            row += f"  {d['rate'][i]:>8.3f}"
            if has_topk:
                row += f"  {d['topk_rate'][i]:>8.3f}"
            lines.append(row)

    if "codebook_size" in results:
        d = results["codebook_size"]
        lines.append("[Codebook Trade-off]")
        for s, r in zip(d["codebook_size"], d["rate"]):
            lines.append(f"  {s:>3d} beams -> {r:.3f} bit/s/Hz")

    if "bf_comparison" in results:
        d = results["bf_comparison"]
        lines.append("[Beamforming Comparison (mean rate)]")
        for arch, key in [("Analog", "analog_rates"), ("Hybrid", "hybrid_rates"), ("Digital", "digital_rates")]:
            lines.append(f"  {arch:<10s}: {float(d[key].mean()):.3f} bit/s/Hz")

    if "beam_selection" in results:
        d = results["beam_selection"]
        lines.append("[Beam Selection]")
        for m in ["exhaustive", "hierarchical", "top_k"]:
            agreement = d.get("beam_accuracy", {}).get(m, float("nan"))
            lines.append(
                f"  {m:<15s}: rate={d['mean_rates'][m]:.3f}  "
                f"pilots={d['mean_pilots'][m]:.1f}  agreement={agreement:.1%}"
            )

    if "multiuser" in results:
        d = results["multiuser"]
        lines.append("[Multi-User ZF]")
        for u, sr, f in zip(d["users"], d["sum_rate"], d["fairness"]):
            lines.append(f"  {u:>2d} users -> sum-rate={sr:.2f}  fairness={f:.3f}")

    if "handover" in results:
        d = results["handover"]
        lines.append("[Handover / Mobility]")
        for s, o, hf in zip(d["speeds_kmh"], d["outage_prob"], d["ho_failure_rate"]):
            lines.append(f"  {s:>5.0f} km/h -> outage={o:.3f}  HO-fail={hf:.3f}")

    if "optimization" in results:
        d = results["optimization"]
        lines.append("[Optimization — all optimizers use fair_utility objective]")
        lines.append(f"  Equal-power utility:  {float(d['equal_utility']):.4f}")
        lines.append(f"  Grid-search utility:  {float(d['grid_utility']):.4f}")
        lines.append(f"  PSO best fraction:    {float(d['pso_fraction']):.3f}")
        lines.append(f"  GA  best fraction:    {float(d['ga_fraction']):.3f}")
        if 'greedy_fraction' in d:
            lines.append(f"  Greedy fraction:      {float(d['greedy_fraction']):.3f}")

    if "ml_ablation" in results:
        d = results["ml_ablation"]
        lines.append("[ML Ablation]")
        rates = d.get("topk_effective_rate")
        for index, (method, t1, tk) in enumerate(zip(d["methods"], d["top1_accuracy"], d["topk_accuracy"])):
            line = f"  {method:<18s}: top-1={t1:.1%}  top-{d['k']}={tk:.1%}"
            if rates is not None:
                line += f"  rate={rates[index]:.2f} bit/s/Hz"
            lines.append(line)
        if "exhaustive_effective_rate" in d:
            lines.append(
                f"  Exhaustive reference: rate={float(d['exhaustive_effective_rate']):.2f} bit/s/Hz"
            )
        if "train_samples" in d and "test_samples" in d:
            lines.append(
                f"  Trajectory-disjoint split: train={int(d['train_samples'])}  "
                f"test={int(d['test_samples'])}"
            )

    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mmWave Massive-MIMO 5G Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py                          # run all experiments
              python main.py --experiment ofdm_ber    # run one experiment
              python main.py --list                   # list experiments
        """),
    )
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        help="Name of a single experiment to run.")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available experiments and exit.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed.")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  {name}")
        sys.exit(0)

    # --- Config ---
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = SystemConfig(**overrides) if overrides else SystemConfig()

    # --- Output dirs ---
    output_base = Path("outputs")
    plot_dir = Path("plots")
    report_dir = Path("report")
    for d in (output_base, plot_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Run ---
    t0 = time.perf_counter()
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS)}")
            sys.exit(1)
        print(f"Running experiment: {args.experiment}")
        results = {args.experiment: EXPERIMENTS[args.experiment](cfg)}
    else:
        print("Running all experiments ...")
        results = run_all(cfg)
    elapsed = time.perf_counter() - t0
    print(f"\nExperiments completed in {elapsed:.1f} s")

    # --- Plots ---
    print("Generating plots ...")
    plot_all(results, plot_dir)
    print(f"Plots saved to {plot_dir}/")

    save_raw_results(results, output_base)
    print(f"Raw results saved to {output_base}/")

    # --- Summary ---
    summary = build_summary(results, cfg)
    (report_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nFull summary saved to {report_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
