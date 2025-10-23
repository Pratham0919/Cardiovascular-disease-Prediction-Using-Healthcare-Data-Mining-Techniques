# Models/logistic_regression.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticRegressionScratch:
    # Conservative defaults for stable convergence on this dataset.
    lr: float = 0.001          # learning rate
    n_iter: int = 8000         # max gradient steps
    l2: float = 0.0            # L2 regularization strength (set >0 after it's stable)
    random_state: int | None = 42
    print_every: int = 400     # how often to print loss

    # Learned params (set in fit)
    w: np.ndarray = field(init=False)
    b: float = field(init=False, default=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train logistic regression with full-batch gradient descent.
        X: (n, d) float features (ideally standardized); y: (n,) in {0,1}
        """
        n, d = X.shape

        # Bias init to prevalence logit so we start near baseline probability.
        p0 = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
        self.b = np.log(p0 / (1.0 - p0))

        # Zero-weight init is very stable with standardized features.
        self.w = np.zeros(d, dtype=float)
        assert self.w is not None

        prev_loss = np.inf
        for i in range(self.n_iter):
            z = X @ self.w + self.b
            p = _sigmoid(z)
            err = p - y  # (n,)

            # Gradients (L2 on weights only)
            grad_w = (X.T @ err) / n + self.l2 * self.w
            grad_b = err.mean()

            # Optional gradient clipping for extra stability
            gnorm = float(np.linalg.norm(grad_w))
            if gnorm > 5.0:
                grad_w *= 5.0 / (gnorm + 1e-12)

            # Parameter update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            # Monitor loss
            if i % self.print_every == 0 or i == self.n_iter - 1:
                eps = 1e-8
                loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
                loss += (self.l2 / 2.0) * float(np.sum(self.w ** 2))
                print(f"Iter {i}: loss={loss:.6f} (grad_norm={gnorm:.4f})")

                # Tiny-improvement early stop after warm-up
                if prev_loss - loss < 1e-5 and i > 800:
                    print(f"Early stop at iter {i} (loss plateau).")
                    break
                prev_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1|x) for each row in X."""
        assert self.w is not None, "Model not fitted."
        return _sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return hard labels using the given decision threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
