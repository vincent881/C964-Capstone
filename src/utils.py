import json
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_CANDIDATES = ["tsunami", "has_tsunami", "tsunami_flag", "tsunami_yn"]

DROP_LIKE = {"id", "event_id", "catalog", "source", "place", "title", "region", "status"}

def find_target(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for t in TARGET_CANDIDATES:
        if t in cols_lower:
            return cols_lower[t]
    # heuristic fallback: any binary column named like 'tsunami'
    for c in df.columns:
        if "tsunami" in c.lower():
            uniq = df[c].dropna().unique()
            if len(uniq) <= 3:
                return c
    raise ValueError("Could not find target column. Set it manually in utils.py.")

def coerce_binary(y: pd.Series) -> pd.Series:
    # Normalize to {0,1}
    y = y.copy()
    if y.dtype.kind in "biu":
        return (y > 0).astype(int)
    y = y.astype(str).str.strip().str.lower()
    return y.isin({"1","true","yes","y","t"}).astype(int)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop obvious non-features (IDs/text-heavy)
    drop_cols = [c for c in df.columns if c.lower() in DROP_LIKE]
    df = df.drop(columns=drop_cols, errors="ignore")
    # Coerce numeric where possible
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")
    # Remove duplicate rows
    df = df.drop_duplicates()
    return df

def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split train/val
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, stratify=y_trainval, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
