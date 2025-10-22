from __future__ import annotations                 
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



def load_raw(path: Path) -> pd.DataFrame:                               # main cleaning functions                 
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]        # normalize column names    
    return df


def basic_sanity_fixes(df: pd.DataFrame) -> pd.DataFrame:
    if "age" in df.columns:                                  # Convert age from days to years
        df["age"] = (df["age"] / 365.25).astype(int)

    if {"ap_hi", "ap_lo"}.issubset(df.columns):          # Some rows have swapped BP (ap_hi < ap_lo) -> swap them back
        swapped = df["ap_hi"] < df["ap_lo"]
        df.loc[swapped, ["ap_hi", "ap_lo"]] = df.loc[swapped, ["ap_lo", "ap_hi"]].values


    rules = [                                             # Remove blatantly impossible entries (keep ranges a bit generous)
        ("age", 30, 80),
        ("height", 130, 210),                                       # cm
        ("weight", 40, 200),                                       # kg
        ("ap_hi", 80, 240),
        ("ap_lo", 40, 150),
    ]
    for col, lo, hi in rules:
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]


    df = df.drop_duplicates(ignore_index=True)            # Drop duplicates
    return df




"""feature engineering for the dataset."""
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"height", "weight"}.issubset(df.columns):           # BMI
        h_m = df["height"] / 100.0
        df["bmi"] = (df["weight"] / (h_m ** 2)).round(3)

    if {"ap_hi", "ap_lo"}.issubset(df.columns):           # Pulse pressure
        df["pulse_pressure"] = (df["ap_hi"] - df["ap_lo"]).astype(int)

    if "age" in df.columns:                 # Age groups
        df["age_group"] = pd.cut(
            df["age"],
            bins=[29, 39, 49, 59, 69, 120],
            labels=["30s", "40s", "50s", "60s", "70+"],
        )

    for c in ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]:                  # Ensure expected categoricals are categorical dtype
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df


def impute_and_scale(df: pd.DataFrame, columns_to_scale: list[str]) -> tuple[pd.DataFrame, StandardScaler]:  #    (Light) imputation + standardization for numeric columns.

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()        # any NaNs left, fill numerics with median
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    scaler = StandardScaler()
    cols = [c for c in columns_to_scale if c in df.columns]
    if cols:
        df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


# Exploratory Data Analysis

def save_class_balance(df: pd.DataFrame, figdir: Path, target: str = "cardio") -> None:          # plot and save class balance for target variable                           
    if target not in df.columns:
        warnings.warn(f"Target column '{target}' not found; skipping class balance plot.")                                          
        return
    figdir.mkdir(parents=True, exist_ok=True)
    ax = df[target].value_counts().sort_index().plot(kind="bar")
    ax.set_title("Target distribution (cardio)")
    ax.set_xlabel("cardio (0 = no CVD, 1 = CVD)")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(figdir / "01_target_distribution.png", dpi=160)
    plt.close()


def class_ratio(df: pd.DataFrame, target: str = "cardio") -> dict:                      # return class counts and positive-class percentage for target
    if target not in df.columns:
        return {"total": 0, "pos": 0, "neg": 0, "pos_pct": np.nan}
    vc = df[target].value_counts()
    pos = int(vc.get(1, 0))
    neg = int(vc.get(0, 0))
    total = pos + neg
    pos_pct = round(100.0 * pos / total, 2) if total else np.nan
    return {"total": total, "pos": pos, "neg": neg, "pos_pct": pos_pct}


def save_histograms(df: pd.DataFrame, figdir: Path, cols: list[str]) -> None:               # made histograms for specified columns
    figdir.mkdir(parents=True, exist_ok=True)
    for c in cols:
        if c in df.columns:
            df[c].hist(bins=30)
            plt.title(f"Distribution: {c}")
            plt.xlabel(c)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figdir / f"hist_{c}.png", dpi=160)
            plt.close()


def save_corr_heatmap(df: pd.DataFrame, figdir: Path, target_first: bool = True) -> None:            # correlation heatmap for numeric columns
    figdir.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    if target_first and "cardio" in corr.columns:
        # Reorder to show target correlations first
        cols = ["cardio"] + [c for c in corr.columns if c != "cardio"]
        corr = corr.loc[cols, cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(figdir / "02_correlation_heatmap.png", dpi=160)
    plt.close()



def preprocess(
    input_csv: Path,
    output_csv: Path,
    figdir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = load_raw(input_csv)
    df = basic_sanity_fixes(df)
    df = add_features(df)

    
    if figdir is not None:                     #  EDA before scaling        
        pre_dir = figdir / "pre_scale"
        pre_dir.mkdir(parents=True, exist_ok=True)
        save_class_balance(df, pre_dir)
        save_histograms(df, pre_dir, ["age", "bmi", "ap_hi", "ap_lo", "pulse_pressure"])
        save_corr_heatmap(df, pre_dir)

    stats = class_ratio(df, target="cardio")               # class stats (before scaling)
    if figdir is not None:
        summary_path = figdir / "00_summary.txt"
        with open(summary_path, "w") as f:
            f.write(
                f"Rows after cleaning: {len(df)}\n"
                f"Class counts -> 0: {stats['neg']}, 1: {stats['pos']}, Total: {stats['total']}\n"
                f"Positive class %: {stats['pos_pct']}%\n"
            )

   
    cols_to_scale = ["age", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure"]   # columns post EDA 
    df, _ = impute_and_scale(df, cols_to_scale)

    # heatmap after scaling:
    # if figdir is not None:
    #     post_dir = figdir / "post_scale"
    #     post_dir.mkdir(parents=True, exist_ok=True)
    #     save_corr_heatmap(df, post_dir)

    
    output_csv.parent.mkdir(parents=True, exist_ok=True)                     # save cleaned data
    df.to_csv(output_csv, index=False)

    print(                                                                  # Print a one-liner for the report log
        f"[CLEAN] rows={len(df)} | class 0={stats['neg']} | class 1={stats['pos']} "
        f"({stats['pos_pct']}% positive)"
    )
    return df, stats


def cli():
    parser = argparse.ArgumentParser(description="Clean & explore the cardio dataset.")
    parser.add_argument("--input", type=Path, default=Path("Dataset/cardio_train.csv"))
    parser.add_argument("--output", type=Path, default=Path("Dataset/cardio_clean.csv"))
    parser.add_argument("--figdir", type=Path, default=Path("Dataset/figures"))
    args = parser.parse_args()

    preprocess(args.input, args.output, args.figdir)
    print(f"✔ Saved cleaned data to: {args.output}")
    print(f"✔ Figures (EDA) saved to: {args.figdir}")


if __name__ == "__main__":
    cli()
