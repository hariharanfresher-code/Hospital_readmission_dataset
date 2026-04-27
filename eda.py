"""
eda.py
======
Hospital Readmission Dataset — Exploratory Data Analysis
Run: python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hospital_readmission_dataset.csv"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE   = {"Not Readmitted": "#2196F3", "Readmitted": "#F44336"}
BLUE      = "#2196F3"
RED       = "#F44336"
GRAY      = "#78909C"
plt.rcParams.update({
    "figure.dpi":      120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family":     "DejaVu Sans",
})


# ── Load ──────────────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["admission_date"])
    df["readmission_label"] = df["label"].map({1: "Readmitted", 0: "Not Readmitted"})
    return df


# ── Plot 1 — Target Distribution ──────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Target Variable — Hospital Readmission", fontsize=14, fontweight="bold")

    vc = df["readmission_label"].value_counts()
    colors = [RED, BLUE]

    # Bar chart
    axes[0].bar(vc.index, vc.values, color=colors, edgecolor="white", width=0.5)
    for i, (lbl, v) in enumerate(vc.items()):
        axes[0].text(i, v + 50, f"{v:,}\n({v/len(df)*100:.1f}%)",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_title("Count by Class")
    axes[0].set_ylim(0, vc.max() * 1.15)

    # Pie chart
    axes[1].pie(vc.values, labels=vc.index, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close()
    print("[EDA]  Saved: 01_target_distribution.png")


# ── Plot 2 — Numeric Feature Distributions ────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame) -> None:
    num_cols = ["age", "comorbidities_count", "length_of_stay",
                "medications_count", "followup_visits_last_year",
                "prev_readmissions", "readmission_risk_score"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Numeric Feature Distributions by Readmission Status",
                 fontsize=14, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for lbl, color in [("Readmitted", RED), ("Not Readmitted", BLUE)]:
            subset = df[df["readmission_label"] == lbl][col]
            axes[i].hist(subset, bins=25, alpha=0.6, color=color,
                         label=lbl, edgecolor="none", density=True)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=8)

    # Turn off unused axes
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_numeric_distributions.png", bbox_inches="tight")
    plt.close()
    print("[EDA]  Saved: 02_numeric_distributions.png")


# ── Plot 3 — Categorical Features ────────────────────────────────────────────
def plot_categorical_features(df: pd.DataFrame) -> None:
    cat_cols = ["gender", "season", "region", "primary_diagnosis",
                "treatment_type", "insurance_type", "discharge_disposition"]

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.suptitle("Readmission Rate by Categorical Feature",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        rate = df.groupby(col)["label"].mean().sort_values(ascending=False)
        bars = axes[i].barh(rate.index, rate.values * 100,
                            color=[RED if v > df["label"].mean() else BLUE
                                   for v in rate.values],
                            edgecolor="none")
        axes[i].axvline(df["label"].mean() * 100, color=GRAY,
                        linestyle="--", linewidth=1.2, label="Overall avg")
        axes[i].set_xlabel("Readmission Rate (%)")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].legend(fontsize=8)
        for bar, val in zip(bars, rate.values):
            axes[i].text(val * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{val*100:.1f}%", va="center", fontsize=8)

    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_categorical_readmission_rates.png", bbox_inches="tight")
    plt.close()
    print("[EDA]  Saved: 03_categorical_readmission_rates.png")


# ── Plot 4 — Correlation Heatmap ──────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_df = df[["age", "comorbidities_count", "length_of_stay",
                 "medications_count", "followup_visits_last_year",
                 "prev_readmissions", "readmission_risk_score", "label"]]

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("[EDA]  Saved: 04_correlation_heatmap.png")


# ── Plot 5 — Age × Risk Score Scatter ─────────────────────────────────────────
def plot_age_risk_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for lbl, color in [("Readmitted", RED), ("Not Readmitted", BLUE)]:
        subset = df[df["readmission_label"] == lbl]
        ax.scatter(subset["age"], subset["readmission_risk_score"],
                   alpha=0.3, s=15, color=color, label=lbl)

    ax.set_xlabel("Age")
    ax.set_ylabel("Readmission Risk Score")
    ax.set_title("Age vs. Readmission Risk Score", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_age_vs_risk_score.png", bbox_inches="tight")
    plt.close()
    print("[EDA]  Saved: 05_age_vs_risk_score.png")


# ── Plot 6 — Monthly Admission Trend ─────────────────────────────────────────
def plot_monthly_trend(df: pd.DataFrame) -> None:
    df = df.copy()
    df["month"] = df["admission_date"].dt.to_period("M")
    monthly = (df.groupby(["month", "readmission_label"])
               .size().unstack(fill_value=0))
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(13, 5))
    monthly.plot(kind="bar", ax=ax, color=[BLUE, RED],
                 edgecolor="none", width=0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Patient Count")
    ax.set_title("Monthly Admission Volume by Readmission Status",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_monthly_trend.png", bbox_inches="tight")
    plt.close()
    print("[EDA]  Saved: 06_monthly_trend.png")


# ── Summary Stats ─────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS — SUMMARY")
    print("=" * 55)
    print(f"  Total patients:     {len(df):,}")
    print(f"  Features:           {df.shape[1]}")
    print(f"  Readmitted:         {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"  Not Readmitted:     {(1-df['label']).sum():,} ({(1-df['label'].mean())*100:.1f}%)")
    print(f"  Age range:          {df['age'].min()} – {df['age'].max()}")
    print(f"  Avg length of stay: {df['length_of_stay'].mean():.1f} days")
    print(f"  Avg risk score:     {df['readmission_risk_score'].mean():.3f}")

    corr_with_label = (
        df[["age", "comorbidities_count", "length_of_stay",
            "medications_count", "prev_readmissions",
            "readmission_risk_score"]]
        .corrwith(df["label"])
        .abs()
        .sort_values(ascending=False)
    )
    print("\n  Top correlations with readmission:")
    for feat, val in corr_with_label.items():
        print(f"    {feat:<35} {val:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_eda() -> None:
    print("=" * 55)
    print("  HOSPITAL READMISSION — EDA Pipeline")
    print("=" * 55)

    df = load_raw()
    print_summary(df)

    print("\n[EDA]  Generating plots...")
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_features(df)
    plot_correlation_heatmap(df)
    plot_age_risk_scatter(df)
    plot_monthly_trend(df)

    print(f"\n✅ All EDA plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_eda()
