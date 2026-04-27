"""
predict.py
==========
Hospital Readmission — Load saved model & predict on new patients
Run: python src/predict.py

Can also be imported as a module for API usage.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ── Feature Engineering (mirrors training) ────────────────────────────────────
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "patient_id" in df.columns:
        df.drop(columns=["patient_id"], inplace=True)

    if "admission_date" in df.columns:
        df["admission_date"]      = pd.to_datetime(df["admission_date"], errors="coerce")
        df["admission_month"]     = df["admission_date"].dt.month
        df["admission_dayofweek"] = df["admission_date"].dt.dayofweek
        df["admission_year"]      = df["admission_date"].dt.year
        df.drop(columns=["admission_date"], inplace=True)

    if "gender" in df.columns:
        df["gender_encoded"] = (df["gender"].str.lower() == "male").astype(int)
        df.drop(columns=["gender"], inplace=True)

    if all(c in df.columns for c in ["comorbidities_count", "medications_count", "prev_readmissions"]):
        df["clinical_burden"]       = df["comorbidities_count"] + df["medications_count"] + df["prev_readmissions"]
        df["high_prev_readmission"] = (df["prev_readmissions"] > 1).astype(int)

    if "length_of_stay" in df.columns:
        df["long_stay"] = (df["length_of_stay"] > 7).astype(int)

    if "admission_dayofweek" in df.columns:
        df["is_weekend_admission"] = (df["admission_dayofweek"] >= 5).astype(int)

    # Encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Drop label if accidentally included
    if "label" in df.columns:
        df.drop(columns=["label"], inplace=True)

    return df


# ── Find Latest Saved Model ───────────────────────────────────────────────────
def load_best_model():
    models = sorted(MODEL_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError(
            f"No saved model found in {MODEL_DIR}. Run model_training.py first."
        )
    model_path = models[0]
    print(f"[predict] Loading model: {model_path.name}")
    return joblib.load(model_path), model_path.name


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(raw_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Predict readmission for new patients.

    Parameters
    ----------
    raw_df    : DataFrame with the same columns as the training dataset
    threshold : Classification probability threshold (default 0.5)

    Returns
    -------
    DataFrame with columns:
      readmission_probability, readmission_predicted, risk_level
    """
    model, model_name = load_best_model()
    X = preprocess_input(raw_df)

    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    pred  = (proba >= threshold).astype(int)
    risk  = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                   labels=["Low", "Medium", "High"])

    result = pd.DataFrame({
        "readmission_probability": proba.round(4),
        "readmission_predicted"  : pred,
        "risk_level"             : risk,
    }, index=raw_df.index)

    return result


# ── Demo: Synthetic Patients ──────────────────────────────────────────────────
def demo_prediction() -> None:
    sample_patients = pd.DataFrame({
        "patient_id"               : ["NEW001", "NEW002", "NEW003"],
        "admission_date"           : ["2024-03-15", "2024-06-01", "2024-11-20"],
        "season"                   : ["Spring", "Summer", "Fall"],
        "age"                      : [72, 45, 85],
        "gender"                   : ["Male", "Female", "Male"],
        "region"                   : ["North", "South", "West"],
        "primary_diagnosis"        : ["Heart Failure", "Diabetes", "COPD"],
        "comorbidities_count"      : [6, 2, 8],
        "length_of_stay"           : [10, 3, 14],
        "treatment_type"           : ["Medical", "Surgical", "Medical"],
        "medications_count"        : [8, 3, 11],
        "followup_visits_last_year": [1, 4, 0],
        "prev_readmissions"        : [3, 0, 5],
        "insurance_type"           : ["Medicare", "Private", "Medicaid"],
        "discharge_disposition"    : ["Home", "Home", "SNF"],
        "readmission_risk_score"   : [0.91, 0.32, 0.97],
    })

    print("=" * 55)
    print("  HOSPITAL READMISSION — Prediction Demo")
    print("=" * 55)
    print("\nInput patients:")
    print(sample_patients[["patient_id", "age", "primary_diagnosis",
                            "comorbidities_count", "prev_readmissions"]].to_string(index=False))

    predictions = predict(sample_patients)

    print("\nPrediction results:")
    output = pd.concat([
        sample_patients[["patient_id", "age", "primary_diagnosis"]].reset_index(drop=True),
        predictions.reset_index(drop=True)
    ], axis=1)
    print(output.to_string(index=False))

    print("\nRisk level summary:")
    print(predictions["risk_level"].value_counts().to_string())


if __name__ == "__main__":
    demo_prediction()
