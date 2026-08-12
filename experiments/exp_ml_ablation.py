"""Experiment: trajectory-disjoint ML ablation for beam prediction."""

from __future__ import annotations

import numpy as np

from config import SystemConfig
from metrics import beam_alignment_accuracy
from ml.beam_predictor import (
    HistoryOnlyPredictor,
    MLPBeamPredictor,
)
from ml.data_generator import beam_feature_matrix, generate_beam_dataset, split_episodes


def run(cfg: SystemConfig) -> dict:
    """Evaluate predictors on complete trajectories withheld from training.

    The fusion MLP receives noisy position, velocity and a *previously
    measured* beam.  Test trajectories are disjoint from training trajectories,
    so the reported values measure spatial generalisation rather than the
    ability to interpolate a single path.
    """
    data = generate_beam_dataset(
        cfg,
        num_episodes=cfg.ml_train_episodes + cfg.ml_test_episodes,
        steps_per_episode=cfg.ml_steps_per_episode,
    )
    labels = data["labels"]
    noisy_pos = data["noisy_positions"]
    velocities = data["velocities"]
    prev_beams = data["previous_beams"]
    train_mask, test_mask = split_episodes(
        data["episode_ids"], cfg.ml_train_episodes, cfg.seed + 501
    )
    evaluation_mask = test_mask & ~data["is_episode_start"]

    # --- Location MLP ---
    location_features = beam_feature_matrix(
        noisy_pos, velocities, prev_beams, cfg.codebook_beams,
        include_location=True, include_motion=False, include_history=False,
    )
    loc_pred = MLPBeamPredictor(
        hidden_units=cfg.ml_hidden_units,
        learning_rate=cfg.ml_learning_rate,
        epochs=cfg.ml_epochs,
        seed=cfg.seed + 510,
        num_classes=cfg.codebook_beams,
    ).fit(location_features[train_mask], labels[train_mask])
    loc_probabilities = loc_pred.predict_proba(location_features[evaluation_mask])
    loc_preds = np.argmax(loc_probabilities, axis=1)
    loc_acc = beam_alignment_accuracy(loc_preds, labels[evaluation_mask])

    # --- History only ---
    hist_pred = HistoryOnlyPredictor(num_classes=cfg.codebook_beams)
    hist_pred.fit(prev_beams[train_mask], labels[train_mask])
    hist_preds = np.array(
        [hist_pred.predict(int(previous)) for previous in prev_beams[evaluation_mask]]
    )
    hist_acc = beam_alignment_accuracy(hist_preds, labels[evaluation_mask])

    # --- Combined ---
    combined_features = beam_feature_matrix(
        noisy_pos, velocities, prev_beams, cfg.codebook_beams,
        include_location=True, include_motion=True, include_history=True,
    )
    comb_pred = MLPBeamPredictor(
        hidden_units=cfg.ml_hidden_units,
        learning_rate=cfg.ml_learning_rate,
        epochs=cfg.ml_epochs,
        seed=cfg.seed + 511,
        num_classes=cfg.codebook_beams,
    ).fit(combined_features[train_mask], labels[train_mask])
    comb_probabilities = comb_pred.predict_proba(combined_features[evaluation_mask])
    comb_preds = np.argmax(comb_probabilities, axis=1)
    comb_acc = beam_alignment_accuracy(comb_preds, labels[evaluation_mask])

    # --- Top-K accuracy ---
    loc_topk = [np.argsort(probabilities)[-cfg.top_k:][::-1] for probabilities in loc_probabilities]
    hist_topk = [
        hist_pred.predict_top_k(int(previous), cfg.top_k)
        for previous in prev_beams[evaluation_mask]
    ]
    comb_topk = [np.argsort(probabilities)[-cfg.top_k:][::-1] for probabilities in comb_probabilities]

    from metrics import top_k_accuracy
    loc_topk_acc = top_k_accuracy(loc_topk, labels[evaluation_mask])
    hist_topk_acc = top_k_accuracy(hist_topk, labels[evaluation_mask])
    comb_topk_acc = top_k_accuracy(comb_topk, labels[evaluation_mask])

    return {
        "methods": np.array(["Location MLP", "History Markov", "Fusion MLP"]),
        "top1_accuracy": np.array([loc_acc, hist_acc, comb_acc]),
        "topk_accuracy": np.array([loc_topk_acc, hist_topk_acc, comb_topk_acc]),
        "k": cfg.top_k,
        "test_labels": labels[evaluation_mask],
        "location_preds": loc_preds,
        "history_preds": hist_preds,
        "combined_preds": comb_preds,
        "location_loss": loc_pred.loss_history,
        "combined_loss": comb_pred.loss_history,
        "train_samples": np.array(int(train_mask.sum())),
        "test_samples": np.array(int(evaluation_mask.sum())),
    }
