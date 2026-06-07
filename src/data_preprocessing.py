"""
src/data_preprocessing.py
Utility functions for loading, cleaning, and merging the fraud datasets.
"""

import pandas as pd
import numpy as np


# ── 1. Loaders ──────────────────────────────────────────────────────────────

def load_fraud_data(path: str) -> pd.DataFrame:
    """Load Fraud_Data.csv with correct dtypes."""
    df = pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])
    return df


def load_ip_country(path: str) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv."""
    return pd.read_csv(path)


def load_creditcard(path: str) -> pd.DataFrame:
    """Load creditcard.csv."""
    return pd.read_csv(path)


# ── 2. Cleaning ──────────────────────────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Fraud_Data:
    - Drop exact duplicates
    - Ensure ip_address is integer (already float-encoded)
    - Strip whitespace from string columns
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Standardise string columns
    str_cols = ["source", "browser", "sex"]
    for col in str_cols:
        df[col] = df[col].str.strip().str.lower()

    # ip_address already numeric; convert to int for range lookup
    df["ip_int"] = df["ip_address"].astype(np.int64)

    return df


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates; no missing values expected."""
    return df.drop_duplicates()


# ── 3. Geolocation merge ─────────────────────────────────────────────────────

def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based IP → country lookup using merge_asof.

    Approach:
    - Sort both frames on the IP integer.
    - merge_asof finds the last lower_bound <= ip_int for each transaction.
    - Filter out rows where ip_int > upper_bound (outside the matched range).
    """
    fraud_sorted = fraud_df.sort_values("ip_int").reset_index(drop=False)
    ip_sorted = ip_df.rename(columns={
        "lower_bound_ip_address": "lower_bound",
        "upper_bound_ip_address": "upper_bound",
    }).copy()
    ip_sorted["lower_bound"] = ip_sorted["lower_bound"].astype(np.int64)
    ip_sorted["upper_bound"] = ip_sorted["upper_bound"].astype(np.int64)
    ip_sorted = ip_sorted.sort_values("lower_bound")

    merged = pd.merge_asof(
        fraud_sorted,
        ip_sorted[["lower_bound", "upper_bound", "country"]],
        left_on="ip_int",
        right_on="lower_bound",
        direction="backward",
    )

    # Keep only rows where ip_int falls within the matched range
    merged.loc[merged["ip_int"] > merged["upper_bound"], "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")

    # Restore original order
    merged = merged.sort_values("index").drop(columns=["index", "lower_bound", "upper_bound"])
    return merged


# ── 4. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add behavioural and temporal features to Fraud_Data.

    New columns:
    - time_since_signup   : hours between signup and purchase
    - hour_of_day         : hour of the purchase (0-23)
    - day_of_week         : day of week (0=Mon … 6=Sun)
    - user_tx_count       : total transactions per user_id
    - user_tx_velocity    : transactions per day for that user
    """
    df = df.copy()

    # Temporal
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600  # hours

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    # Transaction frequency per user
    tx_count = df.groupby("user_id")["user_id"].transform("count")
    df["user_tx_count"] = tx_count

    # Velocity: transactions per day active (signup → purchase)
    days_active = df["time_since_signup"] / 24
    days_active = days_active.clip(lower=1/24)  # minimum 1-hour window
    df["user_tx_velocity"] = df["user_tx_count"] / days_active

    return df


# ── 5. Encoding & Scaling ────────────────────────────────────────────────────

def encode_and_scale(df: pd.DataFrame, scaler=None, fit: bool = True):
    """
    One-hot encode categoricals; scale numerics.
    Returns (transformed_df, fitted_scaler).
    """
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    cat_cols = ["source", "browser", "sex", "country"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    num_cols = [
        "purchase_value", "age", "time_since_signup",
        "hour_of_day", "day_of_week", "user_tx_count", "user_tx_velocity",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler
