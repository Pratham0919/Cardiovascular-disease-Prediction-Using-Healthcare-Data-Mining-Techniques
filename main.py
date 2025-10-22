from pathlib import Path
from Preprocessing.data_cleanup import preprocess

RAW = Path("Dataset/cardio_train.csv")
CLEAN = Path("Dataset/cardio_clean.csv")
FIGS = Path("Dataset/figures")

if __name__ == "__main__":
    df, stats = preprocess(RAW, CLEAN, FIGS)

    print("Shape after cleaning:", df.shape)
    print("Columns:", list(df.columns)[:12], "...")
    print(
        f"Class balance -> 0: {stats['neg']}, 1: {stats['pos']}, "
        f"Positive %: {stats['pos_pct']}%"
    )
    print(df[["age", "bmi", "ap_hi", "ap_lo", "pulse_pressure", "cardio"]].head())     
