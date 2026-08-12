"""Experiment: ML ablation — location only, history only, location + history."""

from __future__ import annotations

import numpy as np

from config import SystemConfig
from metrics import beam_alignment_accuracy
from ml.beam_predictor import (
    CombinedPredictor,
    HistoryOnlyPredictor,
    LocationOnlyPredictor,
)
from ml.data_generator import generate_beam_dataset, split_dataset


def run(cfg: SystemConfig) -> dict:
    """Train and evaluate three ML predictors with different feature sets."""
    data = generate_beam_dataset(cfg)
    labels = data["labels"]
    noisy_pos = data["noisy_positions"]
    velocities = data["velocities"]
    prev_beams = data["previous_beams"]

    split_frac = cfg.ml_train_fraction
    split = max(1, int(len(labels) * split_frac))

    # --- Location only ---
    loc_pred = LocationOnlyPredictor(num_classes=cfg.codebook_beams)
    loc_pred.fit(noisy_pos[:split], labels[:split])
    loc_preds = np.array([loc_pred.predict(noisy_pos[i]) for i in range(split, len(labels))])
    loc_acc = beam_alignment_accuracy(loc_preds, labels[split:])

    # --- History only ---
    hist_pred = HistoryOnlyPredictor(num_classes=cfg.codebook_beams)
    hist_pred.fit(prev_beams[:split], labels[:split])
    hist_preds = np.array([hist_pred.predict(int(prev_beams[i])) for i in range(split, len(labels))])
    hist_acc = beam_alignment_accuracy(hist_preds, labels[split:])

    # --- Combined ---
    combined_features = np.column_stack([noisy_pos, velocities, prev_beams])
    comb_pred = CombinedPredictor(num_classes=cfg.codebook_beams)
    comb_pred.fit(combined_features[:split], labels[:split])
    comb_preds = np.array(
        [comb_pred.predict(combined_features[i]) for i in range(split, len(labels))]
    )
    comb_acc = beam_alignment_accuracy(comb_preds, labels[split:])

    # --- Top-K accuracy ---
    loc_topk = [loc_pred.predict_top_k(noisy_pos[i], cfg.top_k) for i in range(split, len(labels))]
    hist_topk = [hist_pred.predict_top_k(int(prev_beams[i]), cfg.top_k) for i in range(split, len(labels))]
    comb_topk = [comb_pred.predict_top_k(combined_features[i], cfg.top_k) for i in range(split, len(labels))]

    from metrics import top_k_accuracy
    loc_topk_acc = top_k_accuracy(loc_topk, labels[split:])
    hist_topk_acc = top_k_accuracy(hist_topk, labels[split:])
    comb_topk_acc = top_k_accuracy(comb_topk, labels[split:])

    return {
        "methods": np.array(["Location", "History", "Combined"]),
        "top1_accuracy": np.array([loc_acc, hist_acc, comb_acc]),
        "topk_accuracy": np.array([loc_topk_acc, hist_topk_acc, comb_topk_acc]),
        "k": cfg.top_k,
        "test_labels": labels[split:],
        "location_preds": loc_preds,
        "history_preds": hist_preds,
        "combined_preds": comb_preds,
    }
