"""
data_preprocessing.py
=====================
Hospital Readmission Dataset — Data Cleaning & Feature Engineering
Run: python src/data_preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "hospital_readmission_dataset.csv"
OUT_PATH   = BASE_DIR / "data" / "processed_data.csv"


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Shape: {df.shape}")
    print(f"[load]  Columns: {df.columns.tolist()}")
    return df


# ── 2. Inspect ────────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    print("\n── Data Types ────────────────────────────────")
    print(df.dtypes)

    print("\n── Missing Values ────────────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values ✓")

    print("\n── Duplicate Rows ────────────────────────────")
    dups = df.duplicated().sum()
    print(f"{dups} duplicate row(s)")

    print("\n── Target Distribution ───────────────────────")
    vc = df["label"].value_counts(normalize=True) * 100
    print(f"  Readmitted (1): {vc.get(1, 0):.1f}%")
    print(f"  Not Readmitted (0): {vc.get(0, 0):.1f}%")


# ── 3. Clean ──────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop patient_id (not predictive)
    df.drop(columns=["patient_id"], inplace=True)

    # Parse admission_date → extract useful temporal features
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    df["admission_month"]     = df["admission_date"].dt.month
    df["admission_dayofweek"] = df["admission_date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["admission_year"]      = df["admission_date"].dt.year
    df.drop(columns=["admission_date"], inplace=True)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    print(f"[clean] Shape after cleaning: {df.shape}")
    return df


# ── 4. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age buckets
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 75, 100],
        labels=["<40", "40-60", "60-75", "75+"]
    )

    # Interaction: clinical burden score
    df["clinical_burden"] = (
        df["comorbidities_count"] +
        df["medications_count"] +
        df["prev_readmissions"]
    )

    # High-risk flag: previous readmissions > median
    median_prev = df["prev_readmissions"].median()
    df["high_prev_readmission"] = (df["prev_readmissions"] > median_prev).astype(int)

    # Long stay flag: length_of_stay above 75th percentile
    p75_los = df["length_of_stay"].quantile(0.75)
    df["long_stay"] = (df["length_of_stay"] > p75_los).astype(int)

    # Weekend admission flag
    df["is_weekend_admission"] = (df["admission_dayofweek"] >= 5).astype(int)

    print(f"[feat]  New features added. Shape: {df.shape}")
    return df


# ── 5. Encode Categoricals ────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary encode gender
    df["gender_encoded"] = (df["gender"].str.lower() == "male").astype(int)
    df.drop(columns=["gender"], inplace=True)

    # One-hot encode remaining categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[enc]   One-hot encoding: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    print(f"[enc]   Shape after encoding: {df.shape}")
    return df


# ── 6. Save ───────────────────────────────────────────────────────────────────
def save_data(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"[save]  Processed data saved → {path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def preprocess_pipeline() -> pd.DataFrame:
    print("=" * 55)
    print("  HOSPITAL READMISSION — Data Preprocessing Pipeline")
    print("=" * 55)

    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_data(df)

    print("\n✅ Preprocessing complete!")
    print(f"   Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    preprocess_pipeline()
