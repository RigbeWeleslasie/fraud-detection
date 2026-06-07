"""
src/data_preprocessing.py
Utility functions for loading, cleaning, merging, and encoding fraud datasets.

Includes:
- Input validation and required-column checks
- File-read error handling with clear messages
- Malformed data detection
- Orchestrator functions that chain the full pipeline
"""

import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Constants ────────────────────────────────────────────────────────────────

FRAUD_REQUIRED_COLS = {
    "user_id", "signup_time", "purchase_time", "purchase_value",
    "device_id", "source", "browser", "sex", "age", "ip_address", "class",
}

IP_REQUIRED_COLS = {
    "lower_bound_ip_address", "upper_bound_ip_address", "country",
}

CC_REQUIRED_COLS = {"Time", "Amount", "Class"}


# ── Validation helpers ───────────────────────────────────────────────────────

def _check_file_exists(path: str) -> None:
    """Raise FileNotFoundError with a clear message if path does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: '{path}'. "
            "Please check the path and ensure the file is in data/raw/."
        )


def _check_required_columns(df: pd.DataFrame, required: set, source: str) -> None:
    """Raise ValueError listing any missing columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{source}] Missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )


def _check_not_empty(df: pd.DataFrame, source: str) -> None:
    """Raise ValueError if the DataFrame has no rows."""
    if len(df) == 0:
        raise ValueError(
            f"[{source}] File loaded but contains zero rows. "
            "The file may be empty or all rows were filtered out."
        )


def _validate_numeric_range(df: pd.DataFrame, col: str, source: str,
                             min_val=None, max_val=None) -> None:
    """Warn if a numeric column contains out-of-range values."""
    if col not in df.columns:
        return
    if min_val is not None and (df[col] < min_val).any():
        n = (df[col] < min_val).sum()
        logger.warning(
            f"[{source}] Column '{col}' has {n} values below {min_val}. "
            "These may indicate malformed data."
        )
    if max_val is not None and (df[col] > max_val).any():
        n = (df[col] > max_val).sum()
        logger.warning(
            f"[{source}] Column '{col}' has {n} values above {max_val}. "
            "These may indicate malformed data."
        )


def _check_date_column(df: pd.DataFrame, col: str, source: str) -> None:
    """Warn if a datetime column contains nulls after parsing."""
    if col not in df.columns:
        return
    null_count = df[col].isnull().sum()
    if null_count > 0:
        logger.warning(
            f"[{source}] Column '{col}' has {null_count} unparseable "
            "datetime values (set to NaT). Check for malformed timestamps."
        )


# ── 1. Loaders ───────────────────────────────────────────────────────────────

def load_fraud_data(path: str) -> pd.DataFrame:
    """
    Load Fraud_Data.csv with correct dtypes and validation.

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if required columns are missing or file is empty
    """
    _check_file_exists(path)
    try:
        df = pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])
    except Exception as e:
        raise IOError(
            f"Failed to read Fraud_Data from '{path}'. "
            f"Ensure it is a valid CSV file. Original error: {e}"
        ) from e

    _check_required_columns(df, FRAUD_REQUIRED_COLS, "Fraud_Data")
    _check_not_empty(df, "Fraud_Data")
    _check_date_column(df, "signup_time", "Fraud_Data")
    _check_date_column(df, "purchase_time", "Fraud_Data")
    _validate_numeric_range(df, "age", "Fraud_Data", min_val=0, max_val=120)
    _validate_numeric_range(df, "purchase_value", "Fraud_Data", min_val=0)

    invalid_class = ~df["class"].isin([0, 1])
    if invalid_class.any():
        raise ValueError(
            f"[Fraud_Data] 'class' column contains values other than 0/1: "
            f"{df.loc[invalid_class, 'class'].unique().tolist()}"
        )

    logger.info(f"Loaded Fraud_Data: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def load_ip_country(path: str) -> pd.DataFrame:
    """
    Load IpAddress_to_Country.csv with validation.

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if required columns are missing or file is empty
    """
    _check_file_exists(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise IOError(
            f"Failed to read IP-Country mapping from '{path}'. "
            f"Original error: {e}"
        ) from e

    _check_required_columns(df, IP_REQUIRED_COLS, "IpAddress_to_Country")
    _check_not_empty(df, "IpAddress_to_Country")
    _validate_numeric_range(df, "lower_bound_ip_address", "IpAddress_to_Country", min_val=0)
    _validate_numeric_range(df, "upper_bound_ip_address", "IpAddress_to_Country", min_val=0)

    # Warn if any range is inverted (lower > upper)
    inverted = (df["lower_bound_ip_address"] > df["upper_bound_ip_address"]).sum()
    if inverted > 0:
        logger.warning(
            f"[IpAddress_to_Country] {inverted} rows have "
            "lower_bound > upper_bound. These ranges will not match any IP."
        )

    logger.info(f"Loaded IP-Country map: {df.shape[0]:,} ranges.")
    return df


def load_creditcard(path: str) -> pd.DataFrame:
    """
    Load creditcard.csv with validation.

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if required columns are missing or file is empty
    """
    _check_file_exists(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise IOError(
            f"Failed to read creditcard data from '{path}'. "
            f"Original error: {e}"
        ) from e

    _check_required_columns(df, CC_REQUIRED_COLS, "creditcard")
    _check_not_empty(df, "creditcard")
    _validate_numeric_range(df, "Amount", "creditcard", min_val=0)

    invalid_class = ~df["Class"].isin([0, 1])
    if invalid_class.any():
        raise ValueError(
            f"[creditcard] 'Class' column contains values other than 0/1: "
            f"{df.loc[invalid_class, 'Class'].unique().tolist()}"
        )

    logger.info(f"Loaded creditcard: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


# ── 2. Cleaning ──────────────────────────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Fraud_Data:
    - Validates input is a non-empty DataFrame
    - Drops exact duplicates
    - Standardises string columns to lowercase/stripped
    - Converts ip_address float to int64 for range lookup
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"clean_fraud_data expects a pandas DataFrame, got {type(df).__name__}."
        )
    _check_not_empty(df, "clean_fraud_data input")
    _check_required_columns(df, {"ip_address", "source", "browser", "sex"}, "clean_fraud_data")

    df = df.copy()

    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} duplicate rows.")

    # Standardise string columns
    for col in ["source", "browser", "sex"]:
        if df[col].isnull().any():
            logger.warning(
                f"Column '{col}' has {df[col].isnull().sum()} null values. "
                "These will remain null after standardisation."
            )
        df[col] = df[col].str.strip().str.lower()

    # Convert ip_address to integer (it is stored as float in the CSV)
    null_ips = df["ip_address"].isnull().sum()
    if null_ips > 0:
        logger.warning(
            f"ip_address has {null_ips} null values. "
            "These rows will have ip_int = -1 and map to 'Unknown' country."
        )
    df["ip_int"] = df["ip_address"].fillna(-1).astype(np.int64)

    logger.info(f"Cleaned Fraud_Data: {len(df):,} rows remain.")
    return df


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates from creditcard data with validation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"clean_creditcard expects a pandas DataFrame, got {type(df).__name__}."
        )
    _check_not_empty(df, "clean_creditcard input")

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"[creditcard] Dropped {dropped:,} duplicate rows.")

    return df


# ── 3. Geolocation merge ─────────────────────────────────────────────────────

def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based IP → country lookup using merge_asof.

    Validates both inputs before merging.
    Rows whose IP falls outside all ranges are labelled 'Unknown'.
    """
    if not isinstance(fraud_df, pd.DataFrame):
        raise TypeError("merge_ip_country: fraud_df must be a DataFrame.")
    if not isinstance(ip_df, pd.DataFrame):
        raise TypeError("merge_ip_country: ip_df must be a DataFrame.")

    _check_required_columns(fraud_df, {"ip_int"}, "merge_ip_country (fraud_df)")
    _check_required_columns(
        ip_df,
        {"lower_bound_ip_address", "upper_bound_ip_address", "country"},
        "merge_ip_country (ip_df)",
    )

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

    # Nullify country where ip_int is outside the matched range
    out_of_range = (
        merged["ip_int"] > merged["upper_bound"]
    ) | merged["upper_bound"].isnull()
    merged.loc[out_of_range, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")

    unknown_count = (merged["country"] == "Unknown").sum()
    if unknown_count > 0:
        logger.warning(
            f"{unknown_count:,} transactions could not be mapped to a country "
            "and are labelled 'Unknown'."
        )

    merged = merged.sort_values("index").drop(
        columns=["index", "lower_bound", "upper_bound"]
    )
    logger.info("IP-to-country merge complete.")
    return merged


# ── 4. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add behavioural and temporal features to Fraud_Data.

    Required columns: signup_time, purchase_time, user_id
    New columns:
        time_since_signup  — hours between signup and purchase
        hour_of_day        — purchase hour (0–23)
        day_of_week        — 0=Mon … 6=Sun
        user_tx_count      — total transactions per user_id
        user_tx_velocity   — transactions per day active
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"engineer_features expects a DataFrame, got {type(df).__name__}."
        )
    _check_required_columns(
        df, {"signup_time", "purchase_time", "user_id"}, "engineer_features"
    )

    df = df.copy()

    # Ensure datetime types
    for col in ["signup_time", "purchase_time"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.warning(
                f"Column '{col}' is not datetime — attempting conversion."
            )
            df[col] = pd.to_datetime(df[col], errors="coerce")
            _check_date_column(df, col, "engineer_features")

    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600

    # Warn on negative time_since_signup (purchase before signup)
    negative = (df["time_since_signup"] < 0).sum()
    if negative > 0:
        logger.warning(
            f"{negative} rows have purchase_time < signup_time "
            "(negative time_since_signup). These may be data entry errors."
        )

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    tx_count = df.groupby("user_id")["user_id"].transform("count")
    df["user_tx_count"] = tx_count

    days_active = (df["time_since_signup"] / 24).clip(lower=1 / 24)
    df["user_tx_velocity"] = df["user_tx_count"] / days_active

    logger.info("Feature engineering complete. New columns: "
                "time_since_signup, hour_of_day, day_of_week, "
                "user_tx_count, user_tx_velocity.")
    return df


# ── 5. Encoding & Scaling ────────────────────────────────────────────────────

def encode_and_scale(df: pd.DataFrame, scaler=None, fit: bool = True):
    """
    One-hot encode categoricals; StandardScale numerics.

    Parameters
    ----------
    df     : feature DataFrame (no target column)
    scaler : fitted StandardScaler or None (creates new one if None)
    fit    : if True, fit the scaler on df; if False, transform only

    Returns
    -------
    (transformed_df, scaler)

    Raises
    ------
    ValueError : if fit=False but no scaler is provided
    """
    from sklearn.preprocessing import StandardScaler

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"encode_and_scale expects a DataFrame, got {type(df).__name__}."
        )
    _check_not_empty(df, "encode_and_scale")

    if not fit and scaler is None:
        raise ValueError(
            "encode_and_scale called with fit=False but no scaler provided. "
            "Pass the scaler fitted on the training set."
        )

    df = df.copy()

    cat_cols = [c for c in ["source", "browser", "sex", "country"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    num_cols = [
        c for c in [
            "purchase_value", "age", "time_since_signup",
            "hour_of_day", "day_of_week", "user_tx_count", "user_tx_velocity",
        ] if c in df.columns
    ]

    if not num_cols:
        logger.warning("No numeric columns found to scale.")
    else:
        if scaler is None:
            scaler = StandardScaler()
        if fit:
            df[num_cols] = scaler.fit_transform(df[num_cols])
        else:
            # Validate that all expected features are present
            missing = set(scaler.feature_names_in_) - set(df.columns) \
                if hasattr(scaler, "feature_names_in_") else set()
            if missing:
                raise ValueError(
                    f"encode_and_scale (transform): columns present during "
                    f"fit are missing: {sorted(missing)}"
                )
            df[num_cols] = scaler.transform(df[num_cols])

    logger.info(f"Encoding/scaling complete. Output shape: {df.shape}")
    return df, scaler


# ── 6. Orchestrators ─────────────────────────────────────────────────────────

def build_fraud_pipeline(
    fraud_path: str,
    ip_path: str,
    scaler=None,
    fit_scaler: bool = True,
):
    """
    Full preprocessing pipeline for Fraud_Data.csv.

    Steps: load → validate → clean → geo-merge → feature engineering
           → drop unused cols → encode & scale

    Parameters
    ----------
    fraud_path  : path to Fraud_Data.csv
    ip_path     : path to IpAddress_to_Country.csv
    scaler      : pre-fitted StandardScaler (for inference); None to fit new
    fit_scaler  : True when building training set; False for test/inference

    Returns
    -------
    X           : encoded feature DataFrame
    y           : target Series
    scaler      : fitted StandardScaler
    """
    logger.info("=== Fraud pipeline START ===")

    fraud = load_fraud_data(fraud_path)
    ip_df = load_ip_country(ip_path)

    fraud = clean_fraud_data(fraud)
    fraud = merge_ip_country(fraud, ip_df)
    fraud = engineer_features(fraud)

    drop_cols = [
        "user_id", "device_id", "signup_time", "purchase_time",
        "ip_address", "ip_int",
    ]
    drop_cols = [c for c in drop_cols if c in fraud.columns]

    y = fraud["class"]
    X = fraud.drop(columns=drop_cols + ["class"])

    X, scaler = encode_and_scale(X, scaler=scaler, fit=fit_scaler)

    logger.info(f"=== Fraud pipeline END — X:{X.shape}, fraud rate:{y.mean():.3%} ===")
    return X, y, scaler


def build_creditcard_pipeline(
    cc_path: str,
    scaler=None,
    fit_scaler: bool = True,
):
    """
    Full preprocessing pipeline for creditcard.csv.

    Steps: load → validate → clean → scale Amount & Time

    Returns
    -------
    X           : scaled feature DataFrame
    y           : target Series
    scaler      : fitted StandardScaler
    """
    from sklearn.preprocessing import StandardScaler

    logger.info("=== Credit card pipeline START ===")

    df = load_creditcard(cc_path)
    df = clean_creditcard(df)

    y = df["Class"]
    X = df.drop(columns=["Class"])

    if scaler is None and fit_scaler:
        scaler = StandardScaler()

    if fit_scaler:
        X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    else:
        if scaler is None:
            raise ValueError(
                "build_creditcard_pipeline called with fit_scaler=False "
                "but no scaler was provided."
            )
        X[["Amount", "Time"]] = scaler.transform(X[["Amount", "Time"]])

    logger.info(f"=== Credit card pipeline END — X:{X.shape}, fraud rate:{y.mean():.4%} ===")
    return X, y, scaler
