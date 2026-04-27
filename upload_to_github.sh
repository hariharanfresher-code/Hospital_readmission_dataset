#!/bin/bash
# ============================================================
# upload_to_github.sh
# Hospital Readmission Project — GitHub Setup & Upload Guide
# ============================================================
# Usage: bash upload_to_github.sh
# Make sure you have git installed and GitHub CLI (gh) OR
# create the repo manually at https://github.com/new

set -e

REPO_NAME="hospital-readmission-prediction"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Hospital Readmission — GitHub Upload Script"
echo "============================================================"
echo ""

# ── Step 1: Initialize git ─────────────────────────────────────────────────────
echo "[1/5] Initializing git repository..."
cd "$PROJECT_DIR"
git init
git add .
git commit -m "🏥 Initial commit: Hospital Readmission Prediction project

- Full data science pipeline (preprocessing, EDA, training, prediction)
- 4 ML models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Jupyter notebook with end-to-end analysis
- GitHub Actions CI workflow
- SMOTE for class imbalance handling"

echo "✅ Git repository initialized with initial commit"
echo ""

# ── Step 2: Create GitHub repo (requires GitHub CLI) ──────────────────────────
echo "[2/5] Creating GitHub repository..."

if command -v gh &> /dev/null; then
    gh repo create "$REPO_NAME" --public --description \
        "End-to-end ML pipeline for predicting hospital readmissions (8,000 patients, 4 models)" \
        --push --source=.
    echo "✅ Repository created and pushed via GitHub CLI"
else
    echo "⚠️  GitHub CLI (gh) not found."
    echo ""
    echo "  Option A — Install GitHub CLI:"
    echo "    macOS:   brew install gh && gh auth login"
    echo "    Ubuntu:  sudo apt install gh && gh auth login"
    echo "    Windows: winget install GitHub.cli"
    echo ""
    echo "  Option B — Create repo manually:"
    echo "    1. Go to https://github.com/new"
    echo "    2. Name it: $REPO_NAME"
    echo "    3. Click 'Create repository'"
    echo "    4. Then run these commands:"
    echo ""
    echo "       git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
    echo "       git branch -M main"
    echo "       git push -u origin main"
fi

echo ""

# ── Step 3: Verify structure ───────────────────────────────────────────────────
echo "[3/5] Project structure:"
find "$PROJECT_DIR" -not -path "*/\.*" -not -path "*/__pycache__/*" \
     -not -name "*.pyc" | head -40
echo ""

# ── Step 4: Quick test ─────────────────────────────────────────────────────────
echo "[4/5] Running quick pipeline test..."
python src/data_preprocessing.py && echo "✅ Preprocessing OK"
python src/eda.py                && echo "✅ EDA OK"
python src/model_training.py     && echo "✅ Model training OK"
echo ""

# ── Step 5: Done ───────────────────────────────────────────────────────────────
echo "[5/5] 🎉 All done!"
echo ""
echo "To open the Jupyter notebook:"
echo "  jupyter notebook notebooks/hospital_readmission_analysis.ipynb"
echo ""
echo "To open in VS Code:"
echo "  code ."
echo ""
