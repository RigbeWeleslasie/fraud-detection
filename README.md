# Fraud Detection — Adey Innovations Inc.

End-to-end fraud detection pipeline for e-commerce and bank credit card transactions.

---

## Project Overview

This project builds fraud detection models across two distinct transaction streams:

| Dataset | Records | Fraud Rate | Features |
|---|---|---|---|
| `Fraud_Data.csv` | 151,112 | 9.37% | User, device, behavioural, geo |
| `creditcard.csv` | 284,807 | 0.173% | PCA-anonymised V1–V28 + Amount, Time |

Both datasets exhibit severe class imbalance, requiring careful resampling strategies and imbalance-aware evaluation metrics (AUC-PR, F1-Score).

---

## Repository Structure

```
fraud-detection/
├── .github/workflows/unittests.yml   # CI pipeline
├── .vscode/settings.json
├── data/
│   ├── raw/                          # Original datasets (gitignored)
│   └── processed/                    # Train/test splits, scaler
├── notebooks/
│   ├── eda-fraud-data.ipynb          # Task 1 — E-commerce EDA
│   ├── eda-creditcard.ipynb          # Task 1 — Credit card EDA
│   ├── feature-engineering.ipynb     # Task 1 — Feature deep-dive
│   ├── modeling.ipynb                # Task 2 — Model training & eval
│   └── shap-explainability.ipynb     # Task 3 — SHAP analysis
├── src/
│   ├── __init__.py
│   └── data_preprocessing.py         # Reusable preprocessing functions
├── tests/
│   └── test_preprocessing.py         # Unit tests (6 passing)
├── models/                           # Saved model artifacts (.pkl)
├── scripts/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place raw data files in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

---

## Running the Project

Run notebooks in order:

```bash
cd notebooks
jupyter notebook
```

1. `eda-fraud-data.ipynb`   → Task 1 (e-commerce)
2. `eda-creditcard.ipynb`   → Task 1 (credit card)
3. `modeling.ipynb`         → Task 2
4. `shap-explainability.ipynb` → Task 3

Run unit tests:

```bash
pytest tests/ -v
```

---

## Task 1 — Key Findings

### Fraud_Data.csv
- **No missing values**, no duplicates in 151,112 rows
- **Class imbalance**: 90.6% legitimate / 9.4% fraud
- **Strongest fraud signal**: Purchases within 1 hour of signup — fraud rate jumps sharply
- **Geo pattern**: Fraud is distributed globally; some countries show elevated rates
- **SMOTE**: Training set balanced from 109k/11k → 109k/109k

### creditcard.csv
- **Extreme imbalance**: 0.173% fraud (492 / 284,807)
- **Key features**: V14, V12, V10, V16 most correlated with fraud
- **Amount**: Fraud transactions average $122 vs $88 for legitimate
- **SMOTE**: Training balanced to ~227k per class

---

## Evaluation Metrics

Standard accuracy is **misleading** on imbalanced data. We use:

- **AUC-PR** (Area Under Precision-Recall Curve) — primary metric
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — visualise false positives vs false negatives
- **ROC-AUC** — secondary metric

---

## Tech Stack

| Tool | Purpose |
|---|---|
| pandas / numpy | Data manipulation |
| scikit-learn | Modelling, metrics, scaling |
| imbalanced-learn | SMOTE resampling |
| XGBoost / LightGBM | Ensemble models |
| SHAP | Model explainability |
| matplotlib / seaborn | Visualisation |
| pytest | Unit testing |

---

## Team

Adey Innovations Inc. — Data Science Team  
Tutors: Kerod, Mahbubah, Feven

**Deadlines**
- Interim-1: Sun 07 Jun 2026 (Task 1)
- Interim-2: Sun 14 Jun 2026 (Task 2)
- Final: Tue 16 Jun 2026
