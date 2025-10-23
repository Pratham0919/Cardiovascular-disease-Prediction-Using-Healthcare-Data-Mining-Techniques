# Training/training_logis.py
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# ---- ensure project root on path (so Models/ imports work) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------

from Models.logistic_regression import LogisticRegressionScratch  # our scratch model

# ---------------------------- config ----------------------------
DATA = Path("Dataset/cardio_clean.csv")
OUT = Path("Outputs/logreg_baseline")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "cardio"
VAL_RATIO = 0.20
SEED = 42
THRESHOLD = 0.5
# ----------------------------------------------------------------


def stratified_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    (train_idx, val_idx), = sss.split(X, y)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }


def plot_threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, outpath: Path):
    ts = np.linspace(0.1, 0.9, 17)
    f1s, precs, recs = [], [], []
    for t in ts:
        yp = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, yp, zero_division=0))
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))

    plt.figure()
    plt.plot(ts, f1s, marker="o", label="F1")
    plt.plot(ts, precs, marker=".", label="Precision")
    plt.plot(ts, recs, marker="x", label="Recall")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Threshold sweep (validation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    # ---------- load & quick sanity ----------
    df = pd.read_csv(DATA)
    print(df.describe(include='all'))
    print(df.dtypes)

    print("Unique target values:", sorted(df[TARGET].unique()))
    print("Target mean (prevalence):", df[TARGET].mean())

    # ---------- features ----------
    # drop label, any categorical bucket like 'age_group', and identifier 'id'
    features = [c for c in df.columns if c not in [TARGET, "age_group", "id"]]
    print("Using features:", features)
    print("Any NaN in features?", df[features].isna().any().any())

    X = df[features].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int).reshape(-1)
    print("RAW shapes:", X.shape, y.shape)

    # ---------- stratified split ----------
    X_tr, X_va, y_tr, y_va = stratified_split(X, y, val_ratio=VAL_RATIO, seed=SEED)
    print("Split shapes:", X_tr.shape, X_va.shape)

    # ---------- SCALE (fit on train, apply to val) ----------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    print("Train means (first 5 feats):", np.round(X_tr.mean(axis=0)[:5], 3))
    print("Train stds  (first 5 feats):", np.round(X_tr.std(axis=0, ddof=0)[:5], 3))

    # ---------- train (from-scratch LR) ----------
    model = LogisticRegressionScratch(
        lr=0.001,          # start conservatively for stability
        n_iter=8000,
        l2=0.0,
        random_state=SEED,
        print_every=400
    )
    model.fit(X_tr, y_tr)

    # ---------- evaluate ----------
    prob_va = model.predict_proba(X_va)
    print("Proba stats (val):", float(prob_va.min()), float(prob_va.max()), float(prob_va.mean()))
    metrics = evaluate(y_va, prob_va, threshold=THRESHOLD)

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"{k:>15}: {v}")

    # save artifacts
    with open(OUT / "metrics_valid.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        OUT / "model_params.npz",
        w=np.array(getattr(model, "w", [])),
        b=np.array([getattr(model, "b", 0.0)]),
        features=np.array(features, dtype=object)
    )

    plot_threshold_sweep(y_va, prob_va, OUT / "threshold_sweep.png")

    print(f"\nSaved metrics to {OUT / 'metrics_valid.json'}")
    print(f"Saved params to  {OUT / 'model_params.npz'}")
    print(f"Saved plot to    {OUT / 'threshold_sweep.png'}")


if __name__ == "__main__":
    main()
