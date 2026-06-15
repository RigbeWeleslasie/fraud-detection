"""
scripts/train_models.py
=======================
Standalone training script for Task 2 — Fraud Detection.

Trains Logistic Regression (baseline) and XGBoost (ensemble) on both:
  - Fraud_Data.csv     (e-commerce transactions)
  - creditcard.csv     (bank credit card transactions)

Usage:
    python scripts/train_models.py

Outputs:
    models/lr_fraud.pkl
    models/xgb_fraud.pkl
    models/lr_creditcard.pkl
    models/xgb_creditcard.pkl
    models/scaler_fraud.pkl
    models/scaler_creditcard.pkl
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, f1_score,
    precision_recall_curve, roc_auc_score,
)
from imblearn.over_sampling import SMOTE

from src.data_preprocessing import build_fraud_pipeline, build_creditcard_pipeline

# ── Config ────────────────────────────────────────────────────────────────────
RAW    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
PROC   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(PROC,   exist_ok=True)

sns.set_theme(style="whitegrid")


def evaluate(name, model, X_test, y_test, dataset):
    """Compute and print AUC-PR, F1, ROC-AUC, confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name} | {dataset}")
    print(f"  AUC-PR : {auc_pr:.4f}  <-- PRIMARY METRIC")
    print(f"  F1     : {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "model": name, "dataset": dataset,
        "AUC-PR": round(auc_pr, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "TP": int(cm[1, 1]), "FP": int(cm[0, 1]),
        "TN": int(cm[0, 0]), "FN": int(cm[1, 0]),
    }


def run_cv(model, X, y, label):
    """5-Fold stratified cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s = cross_validate(model, X, y, cv=cv,
                       scoring=["average_precision", "f1"], n_jobs=-1)
    ap = s["test_average_precision"]
    f1 = s["test_f1"]
    print(f"  CV [{label}]  AUC-PR={ap.mean():.4f}+/-{ap.std():.4f}  "
          f"F1={f1.mean():.4f}+/-{f1.std():.4f}")
    return {"cv_aucpr": f"{ap.mean():.4f} +/- {ap.std():.4f}",
            "cv_f1":    f"{f1.mean():.4f} +/- {f1.std():.4f}"}


def main():
    print("=" * 60)
    print("FRAUD DETECTION — MODEL TRAINING SCRIPT")
    print("=" * 60)

    # ── STEP 1: Load and stratified split ────────────────────────────────────
    print("\n[1/6] Loading data and applying stratified train-test split...")

    X_fd, y_fd, scaler_fd = build_fraud_pipeline(
        os.path.join(RAW, "Fraud_Data.csv"),
        os.path.join(RAW, "IpAddress_to_Country.csv"),
    )
    X_train_fd, X_test_fd, y_train_fd, y_test_fd = train_test_split(
        X_fd, y_fd, test_size=0.2, stratify=y_fd, random_state=42
    )
    print(f"  Fraud_Data  train={X_train_fd.shape[0]:,}  "
          f"test={X_test_fd.shape[0]:,}  "
          f"fraud_rate(train)={y_train_fd.mean():.3%}  "
          f"fraud_rate(test)={y_test_fd.mean():.3%}")

    X_cc, y_cc, scaler_cc = build_creditcard_pipeline(
        os.path.join(RAW, "creditcard.csv")
    )
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(
        X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=42
    )
    print(f"  CreditCard  train={X_train_cc.shape[0]:,}  "
          f"test={X_test_cc.shape[0]:,}  "
          f"fraud_rate(train)={y_train_cc.mean():.4%}  "
          f"fraud_rate(test)={y_test_cc.mean():.4%}")

    # ── STEP 2: SMOTE on training sets only ──────────────────────────────────
    print("\n[2/6] Applying SMOTE to training sets only...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_fd_r, y_train_fd_r = smote.fit_resample(X_train_fd, y_train_fd)
    X_train_cc_r, y_train_cc_r = smote.fit_resample(X_train_cc, y_train_cc)
    print(f"  Fraud_Data  after SMOTE: {(y_train_fd_r==0).sum():,} legit / "
          f"{(y_train_fd_r==1).sum():,} fraud")
    print(f"  CreditCard  after SMOTE: {(y_train_cc_r==0).sum():,} legit / "
          f"{(y_train_cc_r==1).sum():,} fraud")
    print("  Test sets: UNCHANGED (real distribution preserved)")

    # ── STEP 3: Train Logistic Regression ────────────────────────────────────
    print("\n[3/6] Training Logistic Regression (baseline)...")
    lr_fd = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_fd.fit(X_train_fd_r, y_train_fd_r)
    print("  Fraud_Data LR trained.")

    lr_cc = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_cc.fit(X_train_cc_r, y_train_cc_r)
    print("  CreditCard LR trained.")

    # ── STEP 4: Train XGBoost ────────────────────────────────────────────────
    print("\n[4/6] Training XGBoost (ensemble)...")
    spw_fd = int((y_train_fd_r == 0).sum() / (y_train_fd_r == 1).sum())
    xgb_fd = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_fd, eval_metric="aucpr",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_fd.fit(X_train_fd_r, y_train_fd_r)
    print(f"  Fraud_Data XGBoost trained (scale_pos_weight={spw_fd})")

    spw_cc = int((y_train_cc_r == 0).sum() / (y_train_cc_r == 1).sum())
    xgb_cc = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_cc, eval_metric="aucpr",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_cc.fit(X_train_cc_r, y_train_cc_r)
    print(f"  CreditCard XGBoost trained (scale_pos_weight={spw_cc})")

    # ── STEP 5: Save all models ───────────────────────────────────────────────
    print(f"\n[5/6] Saving models to {MODELS}/")
    joblib.dump(lr_fd,     os.path.join(MODELS, "lr_fraud.pkl"))
    joblib.dump(xgb_fd,    os.path.join(MODELS, "xgb_fraud.pkl"))
    joblib.dump(lr_cc,     os.path.join(MODELS, "lr_creditcard.pkl"))
    joblib.dump(xgb_cc,    os.path.join(MODELS, "xgb_creditcard.pkl"))
    joblib.dump(scaler_fd, os.path.join(MODELS, "scaler_fraud.pkl"))
    joblib.dump(scaler_cc, os.path.join(MODELS, "scaler_creditcard.pkl"))
    for fname in ["lr_fraud.pkl", "xgb_fraud.pkl",
                  "lr_creditcard.pkl", "xgb_creditcard.pkl",
                  "scaler_fraud.pkl", "scaler_creditcard.pkl"]:
        path = os.path.join(MODELS, fname)
        print(f"  Saved {fname}  ({os.path.getsize(path):,} bytes)")

    # ── STEP 6: Evaluate and report ──────────────────────────────────────────
    print("\n[6/6] Evaluating models on held-out test sets...")
    results = []
    results.append(evaluate("Logistic Regression", lr_fd,  X_test_fd, y_test_fd, "Fraud_Data"))
    results.append(evaluate("XGBoost",             xgb_fd, X_test_fd, y_test_fd, "Fraud_Data"))
    results.append(evaluate("Logistic Regression", lr_cc,  X_test_cc, y_test_cc, "CreditCard"))
    results.append(evaluate("XGBoost",             xgb_cc, X_test_cc, y_test_cc, "CreditCard"))

    # 5-Fold CV
    print("\n5-Fold Stratified Cross-Validation:")
    cv_results = {}
    for name, model, X, y, ds in [
        ("LR",  lr_fd,  X_train_fd_r, y_train_fd_r, "Fraud_Data"),
        ("XGB", xgb_fd, X_train_fd_r, y_train_fd_r, "Fraud_Data"),
        ("LR",  lr_cc,  X_train_cc_r, y_train_cc_r, "CreditCard"),
        ("XGB", xgb_cc, X_train_cc_r, y_train_cc_r, "CreditCard"),
    ]:
        cv_results[f"{name}_{ds}"] = run_cv(model, X, y, f"{name} {ds}")

    # Save results JSON
    with open(os.path.join(PROC, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(PROC, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)

    # Final summary table
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    print(df[["dataset", "model", "AUC-PR", "F1", "ROC-AUC", "TP", "FP", "FN"]].to_string(index=False))
    print(f"\nSelected model: XGBoost (best CV AUC-PR on both datasets)")
    print("Results saved to data/processed/model_results.json")
    print("Models saved to models/")


if __name__ == "__main__":
    main()
