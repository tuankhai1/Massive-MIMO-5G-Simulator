"""Dependency-free beam predictors for the ML ablation study.

Three predictors with a common interface:
    .fit(features, labels)
    .predict(feature) -> beam_index
    .predict_top_k(feature, k) -> array of k beam indices

* LocationOnlyPredictor — uses only (noisy) x, y.
* HistoryOnlyPredictor — uses only previous beam index (Markov chain).
* CombinedPredictor — uses (noisy) x, y, vx, vy, prev_beam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ===================================================================
# Base KNN helper
# ===================================================================

def _knn_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query: np.ndarray,
    k: int,
    num_classes: int,
) -> np.ndarray:
    """Return class vote counts from *k* nearest neighbors."""
    dists = np.sum((train_features - query) ** 2, axis=1)
    k_eff = min(k, len(dists))
    nearest = np.argpartition(dists, k_eff - 1)[:k_eff]
    votes = np.bincount(train_labels[nearest], minlength=num_classes)
    return votes


# ===================================================================
# Location-only predictor
# ===================================================================

@dataclass
class LocationOnlyPredictor:
    """KNN predictor using only (noisy) user position [x, y]."""

    neighbors: int = 9
    num_classes: int = 32
    _features: np.ndarray | None = field(default=None, repr=False)
    _labels: np.ndarray | None = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _std: np.ndarray | None = field(default=None, repr=False)

    def fit(self, positions: np.ndarray, labels: np.ndarray) -> "LocationOnlyPredictor":
        self.num_classes = max(self.num_classes, int(labels.max()) + 1)
        self._mean = positions.mean(axis=0)
        self._std = np.maximum(positions.std(axis=0), 1e-6)
        self._features = (positions - self._mean) / self._std
        self._labels = labels.astype(int)
        return self

    def predict(self, position: np.ndarray) -> int:
        q = (position - self._mean) / self._std
        votes = _knn_predict(self._features, self._labels, q, self.neighbors, self.num_classes)
        return int(np.argmax(votes))

    def predict_top_k(self, position: np.ndarray, k: int) -> np.ndarray:
        q = (position - self._mean) / self._std
        votes = _knn_predict(self._features, self._labels, q, self.neighbors, self.num_classes)
        return np.argsort(votes)[-k:][::-1]


# ===================================================================
# History-only predictor
# ===================================================================

@dataclass
class HistoryOnlyPredictor:
    """Markov-chain predictor using only the previous beam index."""

    num_classes: int = 32
    _transition: np.ndarray | None = field(default=None, repr=False)

    def fit(self, previous_beams: np.ndarray, labels: np.ndarray) -> "HistoryOnlyPredictor":
        self.num_classes = max(self.num_classes, int(labels.max()) + 1, int(previous_beams.max()) + 1)
        # Build transition count matrix
        T = np.zeros((self.num_classes, self.num_classes))
        for prev, cur in zip(previous_beams, labels):
            T[int(prev), int(cur)] += 1
        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        self._transition = T / np.maximum(row_sums, 1)
        return self

    def predict(self, previous_beam: int) -> int:
        return int(np.argmax(self._transition[previous_beam]))

    def predict_top_k(self, previous_beam: int, k: int) -> np.ndarray:
        probs = self._transition[previous_beam]
        return np.argsort(probs)[-k:][::-1]


# ===================================================================
# Combined predictor
# ===================================================================

@dataclass
class CombinedPredictor:
    """KNN predictor using [noisy_x, noisy_y, vx, vy, prev_beam]."""

    neighbors: int = 9
    num_classes: int = 32
    _features: np.ndarray | None = field(default=None, repr=False)
    _labels: np.ndarray | None = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _std: np.ndarray | None = field(default=None, repr=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "CombinedPredictor":
        """features columns: [noisy_x, noisy_y, vx, vy, prev_beam]."""
        self.num_classes = max(self.num_classes, int(labels.max()) + 1)
        self._mean = features.mean(axis=0)
        self._std = np.maximum(features.std(axis=0), 1e-6)
        self._features = (features - self._mean) / self._std
        self._labels = labels.astype(int)
        return self

    def predict(self, feature: np.ndarray) -> int:
        q = (feature - self._mean) / self._std
        votes = _knn_predict(self._features, self._labels, q, self.neighbors, self.num_classes)
        return int(np.argmax(votes))

    def predict_top_k(self, feature: np.ndarray, k: int) -> np.ndarray:
        q = (feature - self._mean) / self._std
        votes = _knn_predict(self._features, self._labels, q, self.neighbors, self.num_classes)
        return np.argsort(votes)[-k:][::-1]
