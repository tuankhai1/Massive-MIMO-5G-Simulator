"""Beam predictors for the ML ablation study.

Three predictors with a common interface:
    .fit(features, labels)
    .predict(feature) -> beam_index
    .predict_top_k(feature, k) -> array of k beam indices

* LocationOnlyPredictor — uses only (noisy) x, y.
* HistoryOnlyPredictor — uses only previous beam index (Markov chain).
* CombinedPredictor — uses (noisy) x, y, vx, vy, prev_beam.
* MLPBeamPredictor — a regularised softmax multi-layer perceptron implemented
  in NumPy for non-linear, probabilistic top-K beam prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # pragma: no cover - installation is enforced by requirements.txt
    HistGradientBoostingClassifier = None


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


# ===================================================================
# Regularised MLP classifier
# ===================================================================

@dataclass
class MLPBeamPredictor:
    """Small NumPy MLP for non-linear beam classification and top-K ranking.

    The predictor uses two ReLU layers and a softmax output.  It is purposely
    compact enough to keep the project dependency-free while providing a much
    stronger baseline than hard KNN voting for noisy location and mobility
    features.
    """

    hidden_units: int = 64
    learning_rate: float = 0.025
    epochs: int = 180
    batch_size: int = 128
    l2: float = 1e-4
    seed: int = 0
    num_classes: int = 32
    _mean: np.ndarray | None = field(default=None, repr=False)
    _std: np.ndarray | None = field(default=None, repr=False)
    _weights: tuple[np.ndarray, ...] | None = field(
        default=None, repr=False
    )
    loss_history: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MLPBeamPredictor":
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("features must be (N, D) and labels must be (N,)")
        if len(labels) == 0:
            raise ValueError("Cannot fit MLPBeamPredictor with no samples")

        self.num_classes = max(self.num_classes, int(labels.max()) + 1)
        self._mean = features.mean(axis=0)
        self._std = np.maximum(features.std(axis=0), 1e-6)
        x = (features - self._mean) / self._std
        n_samples, n_features = x.shape
        rng = np.random.default_rng(self.seed)

        # Xavier initialisation for the two hidden layers and the output head.
        scale1 = np.sqrt(2.0 / (n_features + self.hidden_units))
        scale2 = np.sqrt(2.0 / (2 * self.hidden_units))
        scale3 = np.sqrt(2.0 / (self.hidden_units + self.num_classes))
        W1 = rng.normal(0.0, scale1, size=(n_features, self.hidden_units))
        b1 = np.zeros(self.hidden_units)
        W2 = rng.normal(0.0, scale2, size=(self.hidden_units, self.hidden_units))
        b2 = np.zeros(self.hidden_units)
        W3 = rng.normal(0.0, scale3, size=(self.hidden_units, self.num_classes))
        b3 = np.zeros(self.num_classes)

        # Adam optimiser state.
        parameters = [W1, b1, W2, b2, W3, b3]
        first_moment = [np.zeros_like(p) for p in parameters]
        second_moment = [np.zeros_like(p) for p in parameters]
        beta1, beta2 = 0.9, 0.999
        step = 0
        losses = []

        for _ in range(self.epochs):
            batches = np.array_split(
                rng.permutation(n_samples),
                max(1, int(np.ceil(n_samples / self.batch_size))),
            )
            for batch_indices in batches:
                xb = x[batch_indices]
                yb = labels[batch_indices]

                z1 = xb @ W1 + b1
                h1 = np.maximum(z1, 0.0)
                z2 = h1 @ W2 + b2
                h2 = np.maximum(z2, 0.0)
                probabilities = self._softmax(h2 @ W3 + b3)

                target = np.zeros_like(probabilities)
                target[np.arange(len(yb)), yb] = 1.0
                grad_logits = (probabilities - target) / len(yb)
                grad_W3 = h2.T @ grad_logits + self.l2 * W3
                grad_b3 = grad_logits.sum(axis=0)
                grad_h2 = grad_logits @ W3.T
                grad_z2 = grad_h2 * (z2 > 0.0)
                grad_W2 = h1.T @ grad_z2 + self.l2 * W2
                grad_b2 = grad_z2.sum(axis=0)
                grad_h1 = grad_z2 @ W2.T
                grad_z1 = grad_h1 * (z1 > 0.0)
                grad_W1 = xb.T @ grad_z1 + self.l2 * W1
                grad_b1 = grad_z1.sum(axis=0)

                gradients = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
                step += 1
                for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                    first_moment[index] = beta1 * first_moment[index] + (1.0 - beta1) * gradient
                    second_moment[index] = beta2 * second_moment[index] + (1.0 - beta2) * gradient**2
                    m_hat = first_moment[index] / (1.0 - beta1**step)
                    v_hat = second_moment[index] / (1.0 - beta2**step)
                    parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            full_probabilities = self._softmax(
                np.maximum(np.maximum(x @ W1 + b1, 0.0) @ W2 + b2, 0.0) @ W3 + b3
            )
            losses.append(float(-np.mean(np.log(full_probabilities[np.arange(n_samples), labels] + 1e-12))))

        self._weights = (W1, b1, W2, b2, W3, b3)
        self.loss_history = np.asarray(losses)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None or self._mean is None or self._std is None:
            raise RuntimeError("Fit before predicting.")
        x = np.asarray(features, dtype=float)
        was_vector = x.ndim == 1
        x = np.atleast_2d(x)
        x = (x - self._mean) / self._std
        W1, b1, W2, b2, W3, b3 = self._weights
        h1 = np.maximum(x @ W1 + b1, 0.0)
        h2 = np.maximum(h1 @ W2 + b2, 0.0)
        probabilities = self._softmax(h2 @ W3 + b3)
        return probabilities[0] if was_vector else probabilities

    def predict(self, feature: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(feature)))

    def predict_top_k(self, feature: np.ndarray, k: int) -> np.ndarray:
        probabilities = self.predict_proba(feature)
        k_eff = min(k, len(probabilities))
        return np.argsort(probabilities)[-k_eff:][::-1]


# ===================================================================
# Histogram gradient-boosted trees
# ===================================================================

@dataclass
class GradientBoostedBeamPredictor:
    """Probabilistic gradient-boosted-tree beam classifier.

    Histogram gradient boosting is a CPU-friendly model for the simulator's
    tabular location, motion and beam-history features.  It exposes class
    scores for the same Top-K pilot-selection interface as the MLP.
    """

    max_iter: int = 180
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 12
    l2_regularization: float = 1e-3
    seed: int = 0
    num_classes: int = 32
    _model: object | None = field(default=None, repr=False)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "GradientBoostedBeamPredictor":
        if HistGradientBoostingClassifier is None:
            raise ImportError(
                "GradientBoostedBeamPredictor requires scikit-learn. "
                "Install the dependencies from requirements.txt."
            )
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("features must be (N, D) and labels must be (N,)")
        if len(labels) == 0:
            raise ValueError("Cannot fit GradientBoostedBeamPredictor with no samples")

        self.num_classes = max(self.num_classes, int(labels.max()) + 1)
        self._model = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=18,
            random_state=self.seed,
        ).fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Fit before predicting.")
        values = np.asarray(features, dtype=float)
        was_vector = values.ndim == 1
        probabilities = self._model.predict_proba(np.atleast_2d(values))

        # A trajectory-disjoint train split may omit a rare beam.  Expand
        # observed classes back to the full codebook index space.
        full_probabilities = np.zeros((len(probabilities), self.num_classes))
        full_probabilities[:, self._model.classes_.astype(int)] = probabilities
        return full_probabilities[0] if was_vector else full_probabilities

    def predict(self, feature: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(feature)))

    def predict_top_k(self, feature: np.ndarray, k: int) -> np.ndarray:
        probabilities = self.predict_proba(feature)
        k_eff = min(k, len(probabilities))
        return np.argsort(probabilities)[-k_eff:][::-1]
