# 🏥 Hospital Readmission Prediction — Data Science Project

A full end-to-end data science pipeline to **predict 30-day hospital readmission** using patient clinical data.

---

## 📁 Project Structure

```
hospital_readmission/
├── data/
│   └── hospital_readmission_dataset.csv   # Raw dataset (8,000 patients, 17 features)
├── notebooks/
│   └── hospital_readmission_analysis.ipynb  # Jupyter notebook (EDA + ML)
├── src/
│   ├── data_preprocessing.py              # Data cleaning & feature engineering
│   ├── eda.py                             # Exploratory Data Analysis
│   ├── model_training.py                  # ML model training & evaluation
│   └── predict.py                         # Prediction on new data
├── outputs/                               # Saved plots & reports
├── models/                                # Saved trained models
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Overview

| Feature | Description |
|---|---|
| `patient_id` | Unique patient identifier |
| `admission_date` | Date of hospital admission |
| `season` | Season of admission |
| `age` | Patient age (18–95) |
| `gender` | Patient gender |
| `region` | Geographic region |
| `primary_diagnosis` | Main diagnosis category |
| `comorbidities_count` | Number of co-existing conditions |
| `length_of_stay` | Days admitted |
| `treatment_type` | Type of treatment received |
| `medications_count` | Number of medications prescribed |
| `followup_visits_last_year` | Follow-up visits in past year |
| `prev_readmissions` | Previous readmission count |
| `insurance_type` | Insurance coverage type |
| `discharge_disposition` | Where discharged to |
| `readmission_risk_score` | Clinical risk score (0–1) |
| `label` | **Target**: 1 = Readmitted, 0 = Not Readmitted |

- **Total records**: 8,000 patients  
- **Class distribution**: 77.3% readmitted (1), 22.7% not readmitted (0)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/hospital-readmission.git
cd hospital-readmission
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis Scripts
```bash
# Preprocessing
python src/data_preprocessing.py

# EDA
python src/eda.py

# Train models
python src/model_training.py
```

### 4. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/hospital_readmission_analysis.ipynb
```

---

## 🤖 Models Used

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree model |
| XGBoost | Gradient boosted trees |
| LightGBM | Fast gradient boosting |

---

## 📈 Key Findings

- `readmission_risk_score`, `prev_readmissions`, and `comorbidities_count` are the strongest predictors
- Patients aged 65+ have significantly higher readmission rates
- Surgical treatment type is associated with lower readmission risk
- Follow-up visits reduce readmission probability

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — ML models & evaluation
- **xgboost**, **lightgbm** — advanced boosting
- **imbalanced-learn** — class imbalance handling
- **joblib** — model serialization
- **jupyter** — interactive notebooks

---

## 📄 License

MIT License © 2025
