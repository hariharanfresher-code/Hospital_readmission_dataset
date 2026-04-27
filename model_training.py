"""
model_training.py
=================
Hospital Readmission — ML Model Training, Evaluation & Comparison
Run: python src/model_training.py

Models trained:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.pipeline           import Pipeline
from imblearn.over_sampling     import SMOTE

try:
    from xgboost  import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed — skipping XGBoost model")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not installed — skipping LightGBM model")


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "hospital_readmission_dataset.csv"
OUT_DIR    = BASE_DIR / "outputs"
MODEL_DIR  = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ── 1. Load & Prepare ─────────────────────────────────────────────────────────
def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, parse_dates=["admission_date"])

    # Drop ID, parse date
    df.drop(columns=["patient_id"], inplace=True)
    df["admission_month"]     = df["admission_date"].dt.month
    df["admission_dayofweek"] = df["admission_date"].dt.dayofweek
    df["admission_year"]      = df["admission_date"].dt.year
    df.drop(columns=["admission_date"], inplace=True)

    # Feature engineering
    df["gender_encoded"]         = (df["gender"].str.lower() == "male").astype(int)
    df["clinical_burden"]        = df["comorbidities_count"] + df["medications_count"] + df["prev_readmissions"]
    df["high_prev_readmission"]  = (df["prev_readmissions"] > df["prev_readmissions"].median()).astype(int)
    df["long_stay"]              = (df["length_of_stay"] > df["length_of_stay"].quantile(0.75)).astype(int)
    df["is_weekend_admission"]   = (df["admission_dayofweek"] >= 5).astype(int)
    df.drop(columns=["gender"], inplace=True)

    # Label encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=["label"])
    y = df["label"]
    print(f"[prep] Feature matrix: {X.shape}, Target: {y.shape}")
    print(f"[prep] Class balance  — 0: {(y==0).sum()}, 1: {(y==1).sum()}")
    return X, y


# ── 2. Split & Oversample ─────────────────────────────────────────────────────
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[split] Train: {X_train.shape}, Test: {X_test.shape}")

    # Apply SMOTE to handle class imbalance in training set only
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(f"[split] After SMOTE — Train: {X_train_sm.shape} | "
          f"Class 0: {(y_train_sm==0).sum()}, Class 1: {(y_train_sm==1).sum()}")
    return X_train, X_train_sm, X_test, y_train, y_train_sm, y_test


# ── 3. Define Models ──────────────────────────────────────────────────────────
def get_models() -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=63, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    return models


# ── 4. Evaluate a Single Model ────────────────────────────────────────────────
def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1"       : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
        "Avg-PR"   : average_precision_score(y_test, y_proba),
        "_model"   : model,
        "_proba"   : y_proba,
        "_pred"    : y_pred,
    }
    print(f"  {name:<22} Acc={metrics['Accuracy']:.3f}  F1={metrics['F1']:.3f}  AUC={metrics['ROC-AUC']:.3f}")
    return metrics


# ── 5. Cross-Validation ───────────────────────────────────────────────────────
def cross_validate_models(models: dict, X_train, y_train) -> pd.DataFrame:
    print("\n[CV] 5-fold Stratified Cross-Validation (ROC-AUC):")
    cv_results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_results.append({
            "Model": name,
            "CV Mean AUC": scores.mean(),
            "CV Std AUC" : scores.std(),
        })
        print(f"  {name:<22} {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(cv_results)


# ── 6. Plot — ROC Curves ──────────────────────────────────────────────────────
def plot_roc_curves(results: list, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    for (res, color) in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["_proba"])
        ax.plot(fpr, tpr, color=color,
                label=f"{res['Model']} (AUC={res['ROC-AUC']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("[plot] Saved: 07_roc_curves.png")


# ── 7. Plot — Confusion Matrix ────────────────────────────────────────────────
def plot_confusion_matrix(best_res: dict, y_test) -> None:
    cm = confusion_matrix(y_test, best_res["_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"],
                ax=ax, linewidths=0.5, cbar=False)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {best_res['Model']}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("[plot] Saved: 08_confusion_matrix.png")


# ── 8. Plot — Feature Importance ──────────────────────────────────────────────
def plot_feature_importance(best_model, feature_names: list) -> None:
    clf = best_model
    # Unwrap pipeline if needed
    if hasattr(clf, "named_steps"):
        clf = clf.named_steps.get("clf", clf)

    if not hasattr(clf, "feature_importances_"):
        print("[plot] Model has no feature_importances_ — skipping")
        return

    importance = pd.Series(clf.feature_importances_, index=feature_names)
    top20 = importance.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top20.index, top20.values,
                   color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top20))),
                   edgecolor="none")
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Top 20 Feature Importances", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("[plot] Saved: 09_feature_importance.png")


# ── 9. Plot — Model Comparison Bar Chart ─────────────────────────────────────
def plot_model_comparison(results: list) -> None:
    metrics_df = pd.DataFrame([{
        "Model": r["Model"],
        "Accuracy": r["Accuracy"],
        "Precision": r["Precision"],
        "Recall": r["Recall"],
        "F1": r["F1"],
        "ROC-AUC": r["ROC-AUC"],
    } for r in results])

    metrics_df_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    width = 0.15
    colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#E53935"]

    for i, (metric, color) in enumerate(zip(metric_names, colors)):
        vals = metrics_df[metric].values
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=color, alpha=0.85, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("[plot] Saved: 10_model_comparison.png")


# ── 10. Save Best Model ───────────────────────────────────────────────────────
def save_best_model(best_res: dict) -> None:
    path = MODEL_DIR / f"best_model_{best_res['Model'].lower().replace(' ', '_')}.pkl"
    joblib.dump(best_res["_model"], path)
    print(f"[save] Best model saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_training() -> None:
    print("=" * 55)
    print("  HOSPITAL READMISSION — Model Training Pipeline")
    print("=" * 55)

    X, y = load_and_prepare()
    X_train, X_train_sm, X_test, y_train, y_train_sm, y_test = split_data(X, y)

    models = get_models()

    # Cross-validation (on original train, before SMOTE)
    cross_validate_models(models, X_train, y_train)

    # Train & evaluate on SMOTE-resampled train set
    print("\n[eval] Test set evaluation (SMOTE training):")
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train_sm, y_train_sm, X_test, y_test)
        results.append(res)

    # Best model by ROC-AUC
    best_res = max(results, key=lambda r: r["ROC-AUC"])
    print(f"\n🏆 Best model: {best_res['Model']}  (ROC-AUC={best_res['ROC-AUC']:.4f})")

    # Detailed report for best model
    print(f"\n── Classification Report: {best_res['Model']} ────────────────")
    print(classification_report(y_test, best_res["_pred"],
                                target_names=["Not Readmitted", "Readmitted"]))

    # Summary table
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                for r in results])
    summary_df = summary_df.set_index("Model").round(4)
    print("\n── Model Summary ─────────────────────────────────────────")
    print(summary_df.to_string())

    # Save summary
    summary_df.to_csv(OUT_DIR / "model_summary.csv")
    print(f"\n[save] Model summary → {OUT_DIR / 'model_summary.csv'}")

    # Plots
    print("\n[plot] Generating evaluation plots...")
    plot_roc_curves(results, y_test)
    plot_confusion_matrix(best_res, y_test)
    plot_feature_importance(best_res["_model"], X.columns.tolist())
    plot_model_comparison(results)

    # Save best model
    save_best_model(best_res)

    print(f"\n✅ Training complete! All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    run_training()
