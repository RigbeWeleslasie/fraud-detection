# Model Artifacts

Trained models saved with `joblib`. Load with `joblib.load(path)`.

## Files

| File | Dataset | Algorithm | Test AUC-PR | CV AUC-PR (5-fold) |
|---|---|---|---|---|
| `lr_fraud.pkl` | Fraud_Data | Logistic Regression | 0.6200 | 0.9470 ± 0.0004 |
| `xgb_fraud.pkl` | Fraud_Data | XGBoost | 0.5881 | 0.9776 ± 0.0009 |
| `lr_creditcard.pkl` | CreditCard | Logistic Regression | 0.6768 | 0.9920 ± 0.0002 |
| `xgb_creditcard.pkl` | CreditCard | XGBoost | 0.8120 | 1.0000 ± 0.0000 |
| `scaler_fraud.pkl` | Fraud_Data | StandardScaler | — | — |
| `scaler_creditcard.pkl` | CreditCard | StandardScaler | — | — |

> Note: `shap_*.pkl` files (SHAP explainer artifacts, ~2-3 MB each) are excluded
> from the repository via `.gitignore` due to size. Regenerate by running
> `notebooks/shap-explainability.ipynb`.

## Usage

```python
import joblib
import pandas as pd

# Load model and scaler
model  = joblib.load("models/xgb_fraud.pkl")
scaler = joblib.load("models/scaler_fraud.pkl")

# Predict on new data (already feature-engineered)
y_prob = model.predict_proba(X_new)[:, 1]
```

## Reproducing Models

```bash
python scripts/train_models.py
```

This script performs stratified split, SMOTE, trains all models,
evaluates with AUC-PR / F1 / confusion matrix, runs 5-fold CV,
and saves all artifacts to this directory.
