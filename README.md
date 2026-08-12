<p align="center">
  <h1 align="center"> mmWave Massive-MIMO 5G Simulator</h1>
  <p align="center">
    <em>End-to-end beamforming and beam management for 5G NR millimeter-wave networks</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/numpy-≥1.26-013243?logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/matplotlib-≥3.8-11557c" alt="Matplotlib">
  </p>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Motivation & Aims](#motivation--aims)
- [System Pipeline](#system-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Experiment Suite](#experiment-suite)
- [Results & Analysis](#results--analysis)
- [Key Design Decisions](#key-design-decisions)
- [References](#references)

---

## Overview

A **modular, reproducible Python simulator** for a 5G-inspired millimeter-wave (mmWave) massive-MIMO link and network. It includes propagation, CP-OFDM, analog/hybrid/digital beamforming, codebook-based beam management, multi-user zero-forcing, mobility-aware handover, power allocation, and ML-assisted beam prediction. The simulator uses **NumPy + Matplotlib**, with a lightweight NumPy MLP and a scikit-learn histogram gradient-boosted-tree classifier for tabular beam prediction; no deep-learning framework is required.

The project ships **12 self-contained experiments**, each generating publication-ready plots and quantitative summaries, enabling rapid prototyping and benchmarking of beamforming and beam management strategies under realistic 5G-inspired conditions.

### Default System Configuration

| Parameter | Value | Parameter | Value |
| --- | --- | --- | --- |
| Carrier frequency | 28 GHz | Bandwidth | 100 MHz |
| Tx power | 30 dBm (1 W) | Noise figure | 7 dB |
| Antenna elements | 32 (ULA) | Codebook beams | 32 (DFT) |
| OFDM subcarriers | 64 | Cyclic prefix | 16 |
| RF chains | 4 | Streams / user | 1 |
| Users | 4 | Cells | 3 |
| Mobility sampling | 10 ms | Handover trials | 30 |

---

## Motivation & Aims

Millimeter-wave (mmWave) communication at 28 GHz and above is a cornerstone of 5G New Radio (NR), offering wide bandwidths but suffering severe free-space path loss. **Massive-MIMO beamforming** compensates for this loss by focusing energy into narrow beams — but this raises several intertwined challenges:

1. **How to design and select beams efficiently?** Exhaustive sweeping over large codebooks wastes valuable pilot overhead.
2. **Which beamforming architecture to use?** Analog, digital, and hybrid architectures trade off hardware cost against achievable rate.
3. **How does mobility affect beam alignment?** Fast-moving users cause frequent beam misalignment and handover failures.
4. **Can machine learning reduce beam-search overhead?** Location-aware and history-aware predictors can narrow the search space.
5. **How to allocate power fairly across users?** Metaheuristic optimizers (PSO, GA) can outperform naive equal-power allocation.

### Project Aims

- **Build a transparent, end-to-end mmWave simulator** where every line of physics is readable and teachable — deliberately simpler than a full 3GPP TR 38.901 stack.
- **Compare beamforming architectures** (analog, hybrid, digital) in terms of achievable spectral efficiency.
- **Evaluate beam management strategies** — exhaustive search, hierarchical multi-resolution search, location-aided top-K, and ML-assisted prediction — measuring both rate and pilot overhead.
- **Study multi-user MIMO** sum-rate scaling and fairness under zero-forcing precoding.
- **Analyze mobility and handover** performance across speeds from pedestrian (4 km/h) to highway (120 km/h).
- **Benchmark optimization algorithms** (Greedy, PSO, GA) for fair power allocation.
- **Provide reproducible, publication-ready plots** for each experiment.

---

## System Pipeline

The simulator is organized as a layered pipeline, where each layer feeds into the next:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM CONFIGURATION                         │
│  carrier=28 GHz, BW=100 MHz, Nt=32, Nbeams=32, K=4 users        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              CHANNEL & PROPAGATION LAYER                        │
│  • Geometric cluster-based channel model (LoS/NLoS)             │
│  • 3GPP-inspired UMa path loss (simplified)                     │
│  • Per-path Doppler shifts, wideband delay taps                 │
│  • ULA / UPA steering vectors                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHY / OFDM LAYER                               │
│  • QPSK / 16-QAM modulation                                     │
│  • OFDM: IFFT → CP insertion → channel → CP removal → FFT       │
│  • Pilot-aided LS channel estimation + interpolation            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              BEAMFORMING & PRECODING LAYER                      │
│  • Analog: DFT codebook beam selection                          │
│  • Digital: MRT (matched filter) and ZF precoding               │
│  • Hybrid: OMP-inspired analog + effective-channel MRT          │
│  • Multi-user: ZF precoder with per-user SINR computation       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              BEAM MANAGEMENT LAYER                              │
│  • Exhaustive sweep (full codebook)                             │
│  • Hierarchical multi-resolution search                         │
│  • Location-aided top-K refinement                              │
│  • ML-assisted beam prediction (MLP, Markov baseline)           │
│  • Adjacent-beam tracking                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              NETWORK & MOBILITY LAYER                           │
│  • Multi-cell layout (3 BSs, hexagonal)                         │
│  • Random-walk mobility with boundary reflection                │
│  • RSRP computation + correlated shadow fading                  │
│  • 3GPP A3 handover event triggering (margin + TTT)             │
│  • Round-robin and proportional-fair scheduling                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              OPTIMIZATION & ML LAYER                            │
│  • Grid-search and water-filling power allocation               │
│  • Particle Swarm Optimization (PSO)                            │
│  • Genetic Algorithm (GA) with tournament selection             │
│  • Greedy marginal-rate power allocation                        │
│  • NumPy MLP and Markov beam predictors                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXPERIMENTS & VISUALIZATION                        │
│  • 12 plug-and-play experiments via registry                    │
│  • Automated plot generation (12 figures)                       │
│  • Text summary report                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mimo-beamforming-beam-management/
├── config.py                       # SystemConfig dataclass (all parameters)
├── main.py                         # CLI entry point — runs experiments, plots, summary
├── channel_model.py                # Geometric channel: LoS/NLoS path loss, Doppler,
│                                   #   cluster-based generation, wideband OFDM channel,
│                                   #   user route / mobility trace generation
├── array_model.py                  # Antenna array: ULA & UPA steering vectors,
│                                   #   DFT codebooks (1D & 2D), random codebooks
├── phy.py                          # OFDM transceiver: QPSK/16-QAM mod/demod,
│                                   #   OFDM Tx/Rx, LS channel estimation, BER
├── beamforming.py                  # Beamforming: analog (codebook), digital (MRT/ZF),
│                                   #   hybrid (alternating minimisation), MU-MIMO ZF
├── beam_management.py              # Beam management: exhaustive, hierarchical search,
│                                   #   location-aided, MLP top-K, beam tracking
├── mobility.py                     # Mobility: random-walk traces, RSRP, A3 handover,
│                                   #   multi-speed simulation loop
├── network_model.py                # Network: multi-cell layout, scheduling (RR, PF),
│                                   #   association, inter-cell SINR
├── metrics.py                      # Metrics: spectral efficiency, Jain fairness,
│                                   #   outage probability, beam alignment accuracy, CDF
├── plot_results.py                 # 12 figure generators + plot_all() dispatcher
├── algorithms/                     # Optimization algorithms
│   ├── __init__.py
│   ├── greedy.py                   #   Greedy beam selection & power allocation
│   ├── hierarchical_search.py      #   Multi-resolution beam search
│   ├── power_allocation.py         #   Grid search, water filling, fair utility
│   ├── pso.py                      #   Particle Swarm Optimization
│   └── ga.py                       #   Genetic Algorithm (tournament + blend crossover)
├── ml/                             # Machine learning beam prediction
│   ├── __init__.py
│   ├── data_generator.py           #   Supervised beam dataset from user route
│   └── beam_predictor.py           #   KNN baselines + regularized NumPy MLP
├── experiments/                    # 12 self-contained experiments
│   ├── __init__.py                 #   EXPERIMENTS registry + run_all()
│   ├── exp_beam_patterns.py        #   (1) Beam patterns vs. ULA size
│   ├── exp_snr_vs_angle.py         #   (2) SNR/rate vs. angle & distance
│   ├── exp_ofdm_ber.py             #   (3) OFDM BER with/without beamforming
│   ├── exp_rate_vs_antennas.py     #   (4) Rate vs. antenna count
│   ├── exp_codebook_size.py        #   (5) Codebook size trade-off
│   ├── exp_overhead_vs_speed.py    #   (6) Overhead vs. user speed
│   ├── exp_beam_selection.py       #   (7) Beam selection comparison
│   ├── exp_bf_comparison.py        #   (8) Analog vs. hybrid vs. digital
│   ├── exp_multiuser.py            #   (9) Multi-user ZF sum rate & fairness
│   ├── exp_handover.py             #   (10) Handover / outage vs. speed
│   ├── exp_optimization.py         #   (11) PSO vs. GA vs. greedy convergence
│   └── exp_ml_ablation.py          #   (12) ML ablation: feature-set comparison
├── plots/                          # Generated figures (12 PNGs)
├── report/                         # summary.txt + LaTeX report
├── results/                        # Legacy baseline results
├── outputs/                        # Raw experiment outputs
├── requirements.txt                # numpy, matplotlib, scipy
```

### Module Dependency Graph

```mermaid
graph TD
    A[config.py] --> B[array_model.py]
    A --> C[channel_model.py]
    A --> D[phy.py]
    A --> E[beamforming.py]
    A --> F[beam_management.py]
    A --> G[mobility.py]
    A --> H[network_model.py]
    B --> C
    B --> D
    B --> F
    B --> G
    C --> I[ml/data_generator.py]
    E --> F
    E --> I
    J[metrics.py] --> K[experiments/]
    K --> L[plot_results.py]
    L --> M[main.py]
    K --> M
    A --> M
```

---

## Getting Started

### Prerequisites

- Python **3.10+**
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/mimo-beamforming-beam-management.git
cd mimo-beamforming-beam-management

# Create a virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run all 12 experiments
python main.py

# List available experiments
python main.py --list

# Run a single experiment
python main.py -e ofdm_ber
python main.py -e bf_comparison
python main.py -e ml_ablation

# Override random seed for reproducibility studies
python main.py --seed 42

# Run the automated regression tests (does not write pytest cache files)
python -m pytest -q -p no:cacheprovider
```

All outputs are saved automatically:

| Directory | Contents |
| --- | --- |
| `plots/` | 12 publication-ready PNG figures |
| `report/` | `summary.txt` — console-friendly results summary |
| `outputs/` | Per-experiment `.npz` arrays and `.json` metadata |

---

## Experiment Suite

### Overview Table

| # | Experiment | Key Question | Metrics |
| :---: | --- | --- | --- |
| 1 | **Beam Patterns** | How do beam widths change with array size? | Normalized gain (dB) vs. angle |
| 2 | **SNR vs. Angle** | How do SNR and rate vary with user direction and distance? | SNR (dB), rate (bit/s/Hz) heatmaps |
| 3 | **OFDM BER** | How much does beamforming improve link reliability? | BER vs. SNR |
| 4 | **Rate vs. Antennas** | What is the array-size gain in spectral efficiency? | Effective rate, peak SNR |
| 5 | **Codebook Size** | What is the optimal codebook resolution? | Rate vs. pilot overhead |
| 6 | **Overhead vs. Speed** | How does beam-sweep cost scale with mobility? | Rate at 4–120 km/h |
| 7 | **Beam Selection** | Exhaustive vs. hierarchical vs. top-K? | Rate, pilot count |
| 8 | **BF Comparison** | Analog vs. hybrid vs. digital performance? | Rate CDF, rate vs. Nt |
| 9 | **Multi-User** | How does MU-MIMO ZF scale with users? | Sum rate, per-user rate, Jain fairness |
| 10 | **Handover** | How does speed affect outage and HO failure? | Outage prob., HO failure rate, mean SINR |
| 11 | **Optimization** | Which optimizer converges fastest for power allocation? | Log-sum-rate utility vs. iteration |
| 12 | **ML Ablation** | Which features help ML beam prediction most? | Top-1 and top-K accuracy |

---

## Results & Analysis

Below are the key findings from each experiment, using the default system configuration (28 GHz, 32 antennas, 32 DFT beams, 4 users).

### 1. DFT Beam Patterns vs. Array Size

<p align="center">
  <img src="plots/beam_patterns.png" width="85%" alt="Beam patterns for ULA with 8, 16, 32, and 64 elements">
</p>

**What it shows:** DFT-codebook beam patterns for Uniform Linear Arrays (ULA) with N = 8, 16, 32, and 64 antenna elements.

**Key observations:**

- As the array size doubles, the **main-lobe half-power beamwidth halves** (from ~25° at N=8 to ~3° at N=64), consistent with the theoretical relation θ₃ₐ𝐵 ≈ 0.886λ/(Nd).
- Larger arrays provide significantly deeper **sidelobe suppression** (>30 dB for N=64), reducing inter-beam interference.
- The narrower beams of massive arrays necessitate **more precise beam alignment**, motivating the beam management strategies evaluated in later experiments.

---

### 2. SNR & Rate vs. Angle and Distance

<p align="center">
  <img src="plots/snr_vs_angle.png" width="85%" alt="SNR and rate heatmaps vs. user angle and distance">
</p>

**What it shows:** Heatmaps of best-beam SNR (dB) and achievable spectral efficiency as a function of user angular position and distance from the base station.

**Key observations:**

- SNR degrades predictably with distance, following the simplified UMa LoS/NLoS path-loss models.
- Angular coverage is near-uniform thanks to the DFT codebook spanning the full ±90° sector.
- The rate map reveals that even moderate SNR (10–15 dB) yields 3–5 bit/s/Hz of spectral efficiency.

---

### 3. OFDM BER — Effect of Beamforming

<p align="center">
  <img src="plots/ofdm_ber.png" width="65%" alt="BER vs. SNR with and without beamforming">
</p>

**What it shows:** BER for a CP-OFDM link through a CP-bounded multipath tapped-delay-line channel. Pilots occupy dedicated subcarriers, the receiver estimates the channel with a DFT-domain LS fit, and the QPSK no-BF and matched-BF cases use the same channel and receiver assumptions. A 16-QAM no-BF reference is also shown.

**Key observations:**

- Matched beamforming shifts the QPSK curve strongly left: at 14 dB pre-beamforming SNR the simulated BF BER is about 3×10⁻⁴, while the single-antenna link remains near 2×10⁻².
- The 16-QAM curve makes the modulation-order trade-off explicit rather than conflating it with a different channel model.
- This remains an ideal-CSI transmit-steering benchmark; codebook quantisation, RF impairments and pilot contamination are suitable next extensions.

---

### 4. Rate vs. Number of Antennas

<p align="center">
  <img src="plots/rate_vs_antennas.png" width="65%" alt="Effective rate and peak SNR vs. antenna count">
</p>

**What it shows:** Raw rate, exhaustive-sweep effective rate, location top-K effective rate, and peak post-beamforming SNR as the ULA scales from 4 to 128 elements.

**Key observations:**

- Peak SNR increases with array size due to coherent combining gain (~3 dB per doubling).
- However, **effective rate can decrease** for very large arrays because the exhaustive beam sweep requires testing more beams, consuming more pilot overhead within the frame. This highlights the overhead–performance trade-off at the heart of beam management.

| Antennas | Raw Rate | Exhaustive | Location Top-K |
| :---: | :---: | :---: | :---: |
| 4 | 6.93 | 6.87 | 6.88 |
| 8 | 7.88 | 7.75 | 7.83 |
| 16 | 8.86 | 8.58 | 8.81 |
| 32 | 9.88 | 9.24 | 9.82 |
| 64 | 10.89 | 9.50 | 10.83 |
| 128 | 11.86 | 8.83 | 11.79 |

---

### 5. Codebook Size Trade-off

<p align="center">
  <img src="plots/codebook_size.png" width="65%" alt="Rate vs. codebook beams">
</p>

**What it shows:** Effective spectral efficiency and peak SNR as the number of codebook beams varies from 8 to 96.

**Key observations:**

- Too few beams (8) under-resolve the angular domain → low beamforming gain → 1.9 bit/s/Hz.
- The **sweet spot is around 32–48 beams** (~4.7 bit/s/Hz), balancing angular resolution against pilot overhead.
- Beyond 48 beams, the marginal SNR gain is outweighed by the increasing pilot cost, and effective rate starts declining.

---

### 6. Beam-Sweep Overhead vs. Mobility Speed

<p align="center">
  <img src="plots/overhead_vs_speed.png" width="65%" alt="Rate vs. user speed for different beam management strategies">
</p>

**What it shows:** Mean effective rate for exhaustive, hierarchical, location top-K, Fusion-MLP top-K and gradient-boosted Top-K selection across user speeds. The training cost is repeated once per Jakes Doppler coherence interval, so speed directly changes the pilot budget.

**Key observations:**

- Exhaustive sweeping degrades sharply at higher speed because a 32-pilot sweep must be repeated within a much shorter coherence time.
- Hierarchical search is a robust middle ground: it needs 18 measurements but avoids the full-sweep Doppler penalty.
- Fusion-MLP and gradient-boosted Top-K both maintain the highest effective rate in this scenario. Their curves coincide because both recover a candidate set containing the best beam over these slowly changing evaluation routes.
- The gradient-boosted model is trained on the same noisy location, velocity and one-hot prior-beam features as the Fusion MLP, keeping the comparison fair. Location-only top-K remains a deliberately weaker, localisation-limited baseline.

---

### 7. Beam Selection Methods Comparison

<p align="center">
  <img src="plots/beam_selection.png" width="85%" alt="Rate and pilot overhead comparison for beam selection methods">
</p>

**What it shows:** Side-by-side comparison of mean effective rate, pilot measurements and selected-beam agreement with exhaustive search. Each method is evaluated with the beam it actually selected; no method receives an exhaustive-search oracle beam.

**Key observations:**

- Hierarchical search selects the same beam as exhaustive in this high-SNR geometric route while using 18 instead of 32 pilots. This is a result of the selected codebook/channel scenario, not an oracle substitution.
- Location top-K needs only three pilots but sees a 36% beam-selection miss rate under 8 m location error; its lower rate quantifies the cost of imperfect localisation.
- The third panel makes this trade-off visible rather than implying that lower pilot cost always improves throughput.

| Method | Mean Rate (bit/s/Hz) | Mean Pilots | Beam Agreement |
| :---: | :---: | :---: | :---: |
| Exhaustive | 9.67 | 32 | 100% |
| Hierarchical | 9.96 | 18 | 100% |
| Location Top-K | 8.32 | 3 | 64% |

---

### 8. Beamforming Architecture Comparison

<p align="center">
  <img src="plots/bf_comparison.png" width="95%" alt="Analog vs. hybrid vs. digital beamforming comparison">
</p>

**What it shows:** (Left) CDF of achievable spectral efficiency for 100 random user positions. (Right) Mean rate vs. array size for each architecture.

**Key observations:**

- **Digital beamforming** (full MRT) achieves the highest rate (10.84 bit/s/Hz mean) by exploiting full-dimensional channel knowledge.
- **Hybrid beamforming** (four RF chains plus analog codebook) reaches 10.70 bit/s/Hz, or about 99% of the digital rate in this sparse single-user channel.
- **Analog beamforming** is simplest but lower-performing (10.31 bit/s/Hz), limited to a single codeword.
- The gap between architectures widens slightly with more antennas, where digital beamforming can exploit additional spatial degrees of freedom.

| Architecture | Mean Rate (bit/s/Hz) |
| :---: | :---: |
| Analog | 10.31 |
| Hybrid | 10.70 |
| Digital | 10.84 |

---

### 9. Multi-User MIMO (ZF Precoding)

<p align="center">
  <img src="plots/multiuser.png" width="95%" alt="Multi-user ZF beamforming: sum rate, per-user rate, and fairness">
</p>

**What it shows:** Zero-Forcing precoder performance as the number of simultaneous users scales from 2 to 16 (with 32 transmit antennas).

**Key observations:**

- **Sum rate scales with multiplexing order** — from 19.0 (2 users) to 68.7 bit/s/Hz (16 users) — after fixing the ZF channel convention and scale-aware regularisation.
- **Per-user rate decreases** as the power budget is split across more users (9.5 to 4.3 bit/s/Hz).
- **Jain's fairness index** degrades from 0.98 (near-perfect fairness with 2 users) to 0.79 with 16 users, reflecting increasing variance in channel conditions across users.

| Users | Sum Rate | Per-User Rate | Jain Fairness |
| :---: | :---: | :---: | :---: |
| 2 | 19.01 | 9.50 | 0.994 |
| 4 | 33.02 | 8.25 | 0.978 |
| 8 | 53.78 | 6.72 | 0.943 |
| 12 | 63.68 | 5.31 | 0.870 |
| 16 | 68.74 | 4.30 | 0.774 |

---

### 10. Mobility & Handover Performance

<p align="center">
  <img src="plots/handover.png" width="95%" alt="Outage, handover failure, and mean SINR vs. speed">
</p>

**What it shows:** Multi-cell mobility simulation with strongest-cell initial access, A3-event handover, correlated shadowing, correlated 25 dB mmWave blockage, realistic sidelobe interference leakage, and 30 Monte Carlo trials per speed.

**Key observations:**

- Outage rises from about 0.6% at 4 km/h to 1.8% at 120 km/h. Confidence intervals are plotted and probability intervals are clipped to the valid [0, 1] range.
- Handover failure is non-zero and increases from about 0.5% to 7.7%, reflecting the configured execution-delay model rather than an artificially perfect handover.
- The interference model applies sidelobe leakage to non-serving cells; treating every neighbouring base station as an aligned main-lobe interferer is unnecessarily pessimistic for a scheduled mmWave network.
- The 3-cell hexagonal layout with 200 m inter-site distance provides adequate overlap for seamless mobility.

---

### 11. Power-Allocation Optimization

<p align="center">
  <img src="plots/optimization.png" width="65%" alt="Optimization convergence: PSO vs GA vs Greedy">
</p>

**What it shows:** Convergence of three optimizers (PSO, GA, Greedy) towards the optimal proportional-fair log-sum-rate utility for a two-user power allocation problem.

**Key observations:**

- **PSO and GA converge within 1–2 iterations** to the near-optimal utility of 3.913, matching the grid-search optimum. Both metaheuristic methods find the optimal power fraction of ~0.445 (slightly favoring the weaker channel).
- **Greedy allocation converges slowly** (~40 iterations) because it incrementally assigns power in fixed steps, but eventually reaches the same optimum.
- The equal-power baseline (3.910) is already close to optimal because the channel gains are within one order of magnitude — the optimization gain is small but consistent.

| Method | Utility | Power Fraction |
| :---: | :---: | :---: |
| Equal power | 3.910 | 0.500 |
| Grid search | 3.913 | 0.445 |
| PSO | 3.913 | 0.445 |
| GA | 3.913 | 0.445 |

---

### 12. ML Ablation — Feature Set Comparison

<p align="center">
  <img src="plots/ml_ablation.png" width="75%" alt="ML ablation: beam prediction accuracy by feature set">
</p>

**What it shows:** Top-1, top-3 and pilot-overhead-aware Top-3 effective rate for a location-only NumPy MLP, a Markov history baseline, a fusion NumPy MLP and histogram gradient-boosted trees. Training and test data are independent trajectories; the first sample of each test trajectory is excluded because it has no prior measured beam.

**Key observations:**

- The **Markov history baseline** achieves 87.5% top-1 because a previously measured beam is highly informative over a 1 ms update interval.
- The **Fusion MLP** attains 73.4% top-1 and 79.1% top-3 using noisy location, velocity and one-hot beam feedback, and is retained as the mobility-overhead baseline.
- **Gradient Boosting** reaches 86.1% top-1 and 87.5% top-3. Its predicted Top-3 candidates yield 10.30 bit/s/Hz after pilot overhead, exceeding the 10.03 bit/s/Hz exhaustive-sweep reference on the held-out trajectories.
- Location-only MLP reaches 18.4% top-1 and 48.2% top-3 under 8 m localisation error. This makes the benefit of temporal feedback explicit rather than hiding it through an oracle feature.
- Future work can replace the feed-forward fusion MLP with a GRU/TCN or Transformer using RSRP/CSI histories, and can train a contextual bandit to choose K and the re-sweep interval dynamically.

| Predictor | Top-1 Accuracy | Top-3 Accuracy | Top-3 Effective Rate |
| :--- | :---: | :---: | :---: |
| Location MLP | 18.4% | 48.2% | 8.49 bit/s/Hz |
| History Markov | 87.5% | 87.5% | 9.62 bit/s/Hz |
| Fusion MLP | 73.4% | 79.1% | 9.65 bit/s/Hz |
| Gradient Boosting | 86.1% | 87.5% | 10.30 bit/s/Hz |
| Exhaustive reference | 100.0% | 100.0% | 10.03 bit/s/Hz |

---

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| **5G-inspired, not 3GPP-compliant** | Every formula is readable and teachable — no multi-hundred-parameter standards stack. Simplified UMa path-loss (28 + 22·log₁₀(d) + 20·log₁₀(f)) captures the essential physics. |
| **Lightweight ML stack** | The regularised softmax MLP is implemented in NumPy; histogram gradient boosting uses scikit-learn. No TensorFlow or PyTorch dependency is required. |
| **Modular single-responsibility files** | Each `.py` file owns one layer of the pipeline. Experiments are plug-and-play via the registry in `experiments/__init__.py`. |
| **Cluster-based geometric channel** | More physically meaningful than i.i.d. Rayleigh for mmWave. Models LoS/NLoS with per-path delay, AoD, and Doppler. |
| **Dataclass-based configuration** | `SystemConfig` is a frozen dataclass — immutable, hashable, and all parameters have descriptive names with defaults. |
| **Deterministic seeding** | Every experiment derives its RNG from `cfg.seed + offset`, ensuring full reproducibility. |
| **Raw-result persistence** | Every run writes arrays to `outputs/<experiment>.npz` and non-array metadata to JSON, so plots can be independently audited or regenerated. |

---

## References

1. 3GPP TR 38.901 — *Study on channel model for frequencies from 0.5 to 100 GHz*
2. Heath, R.W. & Lozano, A. — *Foundations of MIMO Communication* (Cambridge, 2019)
3. Alkhateeb, A. et al. — *Channel Estimation and Hybrid Precoding for Millimeter Wave Cellular Systems* (IEEE JSTSP, 2014)
4. va Ahmed, I. et al. — *A Survey on Hybrid Beamforming Techniques in 5G* (IEEE Comm. Surveys, 2018)
5. Giordani, M. et al. — *A Tutorial on Beam Management for 3GPP NR at mmWave Frequencies* (IEEE Comm. Surveys, 2019)
6. Kennedy, J. & Eberhart, R. — *Particle Swarm Optimization* (ICNN, 1995)

---
