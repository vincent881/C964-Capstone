import argparse
from pathlib import Path
import pandas as pd

from src.utils import find_target, coerce_binary, basic_clean

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=str, default="data/raw")
    p.add_argument("--out", type=str, default="data/clean/earthquakes_clean.csv")
    args = p.parse_args()

    raw_dir = Path(args.raw)
    # Attempt to find a single CSV
    csvs = list(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}.")
    # Choose the largest CSV (likely main file)
    csv_path = max(csvs, key=lambda p: p.stat().st_size)

    df = pd.read_csv(csv_path)
    df = basic_clean(df)

    target_col = find_target(df)
    y = coerce_binary(df[target_col])
    X = df.drop(columns=[target_col])

    # Drop non-numeric columns after cleaning
    X = X.select_dtypes(include=["number"]).copy()
    # Drop columns with too many missing
    missing_ratio = X.isna().mean()
    X = X.loc[:, missing_ratio <= 0.4]
    # Simple impute with median
    X = X.fillna(X.median(numeric_only=True))

    clean = X.copy()
    clean[target_col] = y

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path} with shape {clean.shape} and target={target_col}")

if __name__ == "__main__":
    main()
