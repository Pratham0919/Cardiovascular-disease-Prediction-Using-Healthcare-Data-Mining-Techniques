from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def preprocess(
    raw_path: str | Path,
    clean_path: str | Path,
    figs_dir: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load the raw cardio dataset, perform basic cleaning/feature engineering, save a
    cleaned CSV, and produce a few diagnostic plots.
    """
    raw_path = Path(raw_path)
    clean_path = Path(clean_path)
    figs_dir = Path(figs_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, sep=";")

    df["age"] = (df["age"] / 365.25).round().astype(int)
    df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(1)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    valid_height = df["height"].between(120, 210)
    valid_weight = df["weight"].between(40, 200)
    valid_ap_hi = df["ap_hi"].between(80, 250)
    valid_ap_lo = df["ap_lo"].between(40, 200)
    valid_ap_relation = df["ap_hi"] >= df["ap_lo"]
    valid_pulse = df["pulse_pressure"] >= 0

    df = df[
        valid_height
        & valid_weight
        & valid_ap_hi
        & valid_ap_lo
        & valid_ap_relation
        & valid_pulse
    ].copy()

    df = df.drop_duplicates()

    df.to_csv(clean_path, index=False)

    stats = _class_balance(df)

    _render_figures(df, figs_dir)

    return df, stats


def _class_balance(df: pd.DataFrame) -> Dict[str, float]:
    pos = int(df["cardio"].sum())
    total = int(len(df))
    neg = total - pos
    pos_pct = round((pos / total) * 100, 2) if total else 0.0
    return {"pos": pos, "neg": neg, "pos_pct": pos_pct}


def _render_figures(df: pd.DataFrame, figs_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["age"], bins=25, ax=axes[0], kde=True, color="#1f77b4")
    axes[0].set_title("Age Distribution (years)")
    axes[0].set_xlabel("Age")

    sns.histplot(df["bmi"], bins=25, ax=axes[1], kde=True, color="#ff7f0e")
    axes[1].set_title("BMI Distribution")
    axes[1].set_xlabel("BMI")

    fig.tight_layout()
    fig.savefig(figs_dir / "age_bmi_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.countplot(data=df, x="cardio", ax=ax, palette="Set2")
    ax.set_title("Cardio Label Counts")
    ax.set_xlabel("cardio (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figs_dir / "cardio_label_counts.png", dpi=150)
    plt.close(fig)
