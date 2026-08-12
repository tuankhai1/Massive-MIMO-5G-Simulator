"""Experiment: trajectory-disjoint ML ablation for beam prediction."""

from __future__ import annotations

import numpy as np

from beamforming import effective_rate
from config import SystemConfig
from metrics import beam_alignment_accuracy, top_k_accuracy
from ml.beam_predictor import (
    GradientBoostedBeamPredictor,
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
    beam_snr = data["beam_snr"]
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

    # --- Gradient-boosted trees ---
    tree_pred = GradientBoostedBeamPredictor(
        max_iter=cfg.gbt_max_iter,
        learning_rate=cfg.gbt_learning_rate,
        max_leaf_nodes=cfg.gbt_max_leaf_nodes,
        min_samples_leaf=cfg.gbt_min_samples_leaf,
        seed=cfg.seed + 513,
        num_classes=cfg.codebook_beams,
    ).fit(combined_features[train_mask], labels[train_mask])
    tree_probabilities = tree_pred.predict_proba(combined_features[evaluation_mask])
    tree_preds = np.argmax(tree_probabilities, axis=1)
    tree_acc = beam_alignment_accuracy(tree_preds, labels[evaluation_mask])

    # --- Top-K accuracy ---
    loc_topk = [np.argsort(probabilities)[-cfg.top_k:][::-1] for probabilities in loc_probabilities]
    hist_topk = [
        hist_pred.predict_top_k(int(previous), cfg.top_k)
        for previous in prev_beams[evaluation_mask]
    ]
    comb_topk = [np.argsort(probabilities)[-cfg.top_k:][::-1] for probabilities in comb_probabilities]
    tree_topk = [np.argsort(probabilities)[-cfg.top_k:][::-1] for probabilities in tree_probabilities]

    loc_topk_acc = top_k_accuracy(loc_topk, labels[evaluation_mask])
    hist_topk_acc = top_k_accuracy(hist_topk, labels[evaluation_mask])
    comb_topk_acc = top_k_accuracy(comb_topk, labels[evaluation_mask])
    tree_topk_acc = top_k_accuracy(tree_topk, labels[evaluation_mask])

    # Evaluate the rate after selecting the strongest of the predicted Top-K
    # candidates.  The rate includes exactly K pilot measurements, so this is
    # directly comparable across learnt predictors rather than accuracy alone.
    test_snr = beam_snr[evaluation_mask]

    def _topk_effective_rate(candidate_sets: list[np.ndarray]) -> float:
        selected_snr = np.array([
            np.max(snr_values[np.asarray(candidates, dtype=int)])
            for snr_values, candidates in zip(test_snr, candidate_sets)
        ])
        return float(np.mean([
            effective_rate(float(snr), cfg.top_k, cfg) for snr in selected_snr
        ]))

    location_rate = _topk_effective_rate(loc_topk)
    history_rate = _topk_effective_rate(hist_topk)
    fusion_rate = _topk_effective_rate(comb_topk)
    tree_rate = _topk_effective_rate(tree_topk)
    exhaustive_rate = float(np.mean([
        effective_rate(float(np.max(snr_values)), cfg.codebook_beams, cfg)
        for snr_values in test_snr
    ]))

    return {
        "methods": np.array([
            "Location MLP", "History Markov", "Fusion MLP", "Gradient Boosting",
        ]),
        "top1_accuracy": np.array([loc_acc, hist_acc, comb_acc, tree_acc]),
        "topk_accuracy": np.array([loc_topk_acc, hist_topk_acc, comb_topk_acc, tree_topk_acc]),
        "topk_effective_rate": np.array([location_rate, history_rate, fusion_rate, tree_rate]),
        "exhaustive_effective_rate": np.array(exhaustive_rate),
        "k": cfg.top_k,
        "test_labels": labels[evaluation_mask],
        "location_preds": loc_preds,
        "history_preds": hist_preds,
        "combined_preds": comb_preds,
        "tree_preds": tree_preds,
        "location_loss": loc_pred.loss_history,
        "combined_loss": comb_pred.loss_history,
        "train_samples": np.array(int(train_mask.sum())),
        "test_samples": np.array(int(evaluation_mask.sum())),
    }
