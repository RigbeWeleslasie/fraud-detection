"""
src/model_utils.py
==================
Helper functions for consistent model evaluation and logging
across all fraud detection models.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(name: str, model, X_test, y_test, dataset: str) -> dict:
    """
    Evaluate a trained classifier and log all key metrics.

    Parameters
    ----------
    name    : human-readable model name (e.g. "XGBoost")
    model   : fitted sklearn-compatible classifier
    X_test  : test feature matrix
    y_test  : true labels
    dataset : dataset label for display (e.g. "Fraud_Data")

    Returns
    -------
    dict with AUC-PR, F1, ROC-AUC, TP, FP, TN, FN
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Model '{name}' must implement predict_proba() for AUC-PR calculation."
        )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"[{dataset}] {name} — AUC-PR={auc_pr:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {name} | {dataset}")
    print(f"  AUC-PR : {auc_pr:.4f}  <-- PRIMARY METRIC")
    print(f"  F1     : {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")

    return {
        "model":   name,
        "dataset": dataset,
        "AUC-PR":  round(auc_pr, 4),
        "F1":      round(f1, 4),
        "ROC-AUC": round(roc_auc, 4),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Print a formatted side-by-side comparison table of model results.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()

    Returns
    -------
    pd.DataFrame of results sorted by AUC-PR descending
    """
    df = pd.DataFrame(results)
    df = df.sort_values(["dataset", "AUC-PR"], ascending=[True, False])

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(df[["dataset", "model", "AUC-PR", "F1", "ROC-AUC",
              "TP", "FP", "FN"]].to_string(index=False))
    print()

    best = df.loc[df.groupby("dataset")["AUC-PR"].idxmax()]
    for _, row in best.iterrows():
        print(f"  Best [{row['dataset']}]: {row['model']} "
              f"(AUC-PR={row['AUC-PR']}, F1={row['F1']})")

    return df


def plot_pr_curves(models_dict: dict, X_test, y_test,
                   title: str = "Precision-Recall Curves", ax=None):
    """
    Plot Precision-Recall curves for multiple models on one axis.

    Parameters
    ----------
    models_dict : {name: model} mapping
    X_test, y_test : test data
    title : plot title
    ax : matplotlib axis (creates one if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AUC-PR={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return ax


def plot_confusion_matrix(model, X_test, y_test,
                           title: str = "Confusion Matrix", ax=None):
    """
    Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    model       : fitted classifier
    X_test, y_test : test data
    title       : plot title
    ax          : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    cm = confusion_matrix(y_test, model.predict(X_test))
    y_prob = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, model.predict(X_test))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    ax.set_title(f"{title}\nAUC-PR={ap:.4f}  F1={f1:.4f}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return ax


def log_cv_results(cv_scores: dict, name: str, dataset: str) -> dict:
    """
    Log and return cross-validation results.

    Parameters
    ----------
    cv_scores : output of sklearn cross_validate()
    name      : model name
    dataset   : dataset label

    Returns
    -------
    dict with mean/std for AUC-PR and F1
    """
    ap = cv_scores["test_average_precision"]
    f1 = cv_scores["test_f1"]

    result = {
        "model":     name,
        "dataset":   dataset,
        "cv_aucpr":  f"{ap.mean():.4f} +/- {ap.std():.4f}",
        "cv_f1":     f"{f1.mean():.4f} +/- {f1.std():.4f}",
    }

    logger.info(
        f"[{dataset}] {name} CV — "
        f"AUC-PR={ap.mean():.4f}+/-{ap.std():.4f}  "
        f"F1={f1.mean():.4f}+/-{f1.std():.4f}"
    )
    print(
        f"  CV [{name} | {dataset}]  "
        f"AUC-PR={ap.mean():.4f}+/-{ap.std():.4f}  "
        f"F1={f1.mean():.4f}+/-{f1.std():.4f}"
    )
    return result
