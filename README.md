# Fraud Detection — Adey Innovations Inc.

![CI](https://github.com/RigbeWeleslasie/fraud-detection/actions/workflows/unittests.yml/badge.svg)

End-to-end fraud detection pipeline for e-commerce and bank credit card transactions — covering data preprocessing, feature engineering, model training, and SHAP explainability.

---

## Project Overview

Adey Innovations Inc. serves both e-commerce and banking clients. This project builds independent fraud detection pipelines for two very different transaction datasets:

| Dataset | Records | Fraud Rate | Features |
|---|---|---|---|
| `Fraud_Data.csv` | 151,112 | 9.37% | User, device, behavioural, geolocation |
| `creditcard.csv` | 284,807 | 0.173% | PCA-anonymised V1–V28 + Amount, Time |

Both datasets are highly imbalanced, which shapes every decision: metric choice (AUC-PR over accuracy), resampling strategy (SMOTE on training set only), and model selection (XGBoost with `scale_pos_weight`).

---

## Repository Structure

```
fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml         # CI/CD — runs pytest on every push & PR
├── .vscode/
│   └── settings.json
├── data/                         # Gitignored — add your own data files
│   ├── raw/                      # Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv
│   └── processed/                # Train/test splits, scalers, plots
├── notebooks/
│   ├── eda-fraud-data.ipynb      # Task 1 — E-commerce EDA, geo, SMOTE
│   ├── eda-creditcard.ipynb      # Task 1 — Credit card EDA, scaling, SMOTE
│   ├── feature-engineering.ipynb # Task 1 — Feature deep-dive & correlation
│   ├── modeling.ipynb            # Task 2 — LR + XGBoost, CV, metrics
│   └── shap-explainability.ipynb # Task 3 — SHAP plots + business recommendations
├── src/
│   ├── __init__.py
│   └── data_preprocessing.py     # Validated, testable preprocessing functions
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py     # 24 unit tests — all passing
├── models/                       # Saved .pkl model artifacts (gitignored)
├── scripts/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/RigbeWeleslasie/fraud-detection.git
cd fraud-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place raw data files in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

Run notebooks **in order**:

```bash
cd notebooks
jupyter notebook
```

| # | Notebook | Task | What it does |
|---|---|---|---|
| 1 | `eda-fraud-data.ipynb` | Task 1 | Cleaning, EDA, IP to country, SMOTE |
| 2 | `eda-creditcard.ipynb` | Task 1 | Cleaning, EDA, scaling, SMOTE |
| 3 | `feature-engineering.ipynb` | Task 1 | Feature deep-dive, correlation matrix |
| 4 | `modeling.ipynb` | Task 2 | LR baseline + XGBoost, CV, metrics |
| 5 | `shap-explainability.ipynb` | Task 3 | SHAP + business recommendations |

Run unit tests:

```bash
pytest tests/ -v
```

---

## Task 1 — Data Analysis & Preprocessing

### Data Cleaning
| Dataset | Rows | Missing | Duplicates | Action |
|---|---|---|---|---|
| Fraud_Data.csv | 151,112 | None | 0 | Parse timestamps, convert IP to int64 |
| creditcard.csv | 284,807 | None | 1,081 | Drop duplicates |

### Geolocation Integration
IP addresses were converted from float to int64 and merged with `IpAddress_to_Country.csv` using `pandas.merge_asof` (range-based lookup). Over 99% of IPs were successfully mapped to a country.

### Feature Engineering (Fraud_Data.csv)
| Feature | Description | Fraud Signal |
|---|---|---|
| `time_since_signup` | Hours between signup and purchase | Highest fraud rate within first hour |
| `hour_of_day` | Purchase hour (0-23) | Late-night peaks |
| `day_of_week` | 0=Mon to 6=Sun | Weekend variation |
| `user_tx_count` | Total transactions per user | Volume anomaly |
| `user_tx_velocity` | Transactions per day active | Speed anomaly — top SHAP feature |

### Class Imbalance — SMOTE
SMOTE applied **on training set only**:

| Dataset | Before SMOTE (train) | After SMOTE (train) |
|---|---|---|
| Fraud_Data | 109,569 legit / 11,321 fraud | 109,569 / 109,569 |
| creditcard | 227,452 legit / 394 fraud | 227,452 / 227,452 |

---

## Task 2 — Model Building & Training

### 5-Fold Stratified Cross-Validation Results

| Dataset | Model | CV AUC-PR | CV F1 |
|---|---|---|---|
| Fraud_Data | Logistic Regression | 0.9470 +/- 0.0004 | 0.8723 +/- 0.0011 |
| Fraud_Data | **XGBoost** | **0.9780 +/- 0.0010** | **0.9292 +/- 0.0017** |
| CreditCard | Logistic Regression | 0.9920 +/- 0.0002 | 0.9464 +/- 0.0011 |
| CreditCard | **XGBoost** | **1.0000 +/- 0.0000** | **0.9996 +/- 0.0000** |

### Test Set Evaluation

| Dataset | Model | AUC-PR | F1 | ROC-AUC |
|---|---|---|---|---|
| Fraud_Data | Logistic Regression | 0.6201 | 0.6814 | 0.7706 |
| Fraud_Data | XGBoost | 0.5875 | 0.6686 | 0.7653 |
| CreditCard | Logistic Regression | 0.6768 | 0.1002 | 0.9619 |
| CreditCard | **XGBoost** | **0.8120** | **0.7212** | **0.9704** |

### Model Selection — XGBoost

XGBoost was selected because:
1. **CV AUC-PR dominates** on both datasets — most reliable performance estimate
2. **CreditCard F1 gap is decisive**: 0.72 vs LR's 0.10 — LR fails on extreme imbalance
3. **Non-linear fraud patterns** require tree-based models
4. **SHAP compatible** for Task 3 explainability
5. **Native imbalance handling** via `scale_pos_weight`

---

## Task 3 — Model Explainability (SHAP)

### Top 5 Fraud Drivers

**Fraud_Data.csv:**
1. `user_tx_velocity` — high transaction speed is the strongest fraud signal
2. `day_of_week` — certain days correlate with fraudulent activity
3. `country_United States` — geolocation is a key discriminator
4. `country_Unknown` — unmapped IPs are disproportionately fraudulent
5. `browser_chrome` — browser type carries a measurable fraud signal

**creditcard.csv:**
1. `V14` — strongest PCA component (spending pattern deviation)
2. `V4` — second most important anonymised feature
3. `V12` — third key PCA driver
4. `V17` — temporal/amount pattern component
5. `V8` — complementary fraud signal

### Business Recommendations

| # | Recommendation | SHAP Evidence |
|---|---|---|
| 1 | **Step-up verification for new accounts** — require OTP for purchases within 1 hour of signup | `time_since_signup` spikes in first hour |
| 2 | **Real-time velocity monitoring** — flag >3 transactions/day in first 24h | `user_tx_velocity` is the #1 SHAP driver |
| 3 | **Hour-of-day risk scoring** — tighten limits between 00:00-05:00 | `hour_of_day` peaks late-night |
| 4 | **Geolocation mismatch alerts** — flag IP country vs registered country mismatch | Country features in top 5 SHAP |
| 5 | **Credit card: monitor V14 and V4 deviations** | V14 and V4 are dominant SHAP drivers for credit card fraud |

---

## Code Quality

### Error Handling in src/data_preprocessing.py
- Required column checks on every function with named missing columns in the error
- `FileNotFoundError` with exact path, `ValueError`, `TypeError`, `IOError` — all descriptive
- Graceful `logger.warning` for recoverable issues (null IPs, inverted date ranges)
- Two orchestrators: `build_fraud_pipeline()` and `build_creditcard_pipeline()`
- Scaler serialised to prevent data leakage on inference

### Unit Tests — 24/24 Passing
```bash
pytest tests/ -v
# ======================== 24 passed in 1.59s ========================
```

### CI/CD
`.github/workflows/unittests.yml` runs `pytest` on every push and PR automatically.

---

## Evaluation Metrics

| Metric | Why |
|---|---|
| **AUC-PR** (primary) | Most informative for imbalanced classes |
| **F1-Score** | Balances false negatives (missed fraud) and false positives (frustrated customers) |
| **Confusion Matrix** | Exact TP/FP/TN/FN counts for operational decisions |
| **ROC-AUC** | Secondary — useful for ranking |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Logistic Regression, metrics, scaling, CV |
| `xgboost` | Ensemble fraud classifier |
| `imbalanced-learn` | SMOTE resampling |
| `shap` | Model explainability |
| `matplotlib` / `seaborn` | Visualisation |
| `joblib` | Model serialisation |
| `pytest` | Unit testing |
| GitHub Actions | CI/CD |

---

## Project Timeline

| Milestone | Date | Status |
|---|---|---|
| Interim-1 (Task 1) | Sun 07 Jun 2026 | Done |
| Interim-2 (Task 2) | Sun 14 Jun 2026 | Done |
| **Final Submission** | **Tue 16 Jun 2026** | **Complete** |

---

## Team

**Adey Innovations Inc. — Data Science Team**
Tutors: Kerod, Mahbubah, Feven
Slack: #all-week5-and-6
