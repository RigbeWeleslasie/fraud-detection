"""
tests/test_preprocessing.py
Unit tests for src/data_preprocessing.py — covers validation,
error handling, and core transformation logic.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import (
    load_fraud_data, load_ip_country, load_creditcard,
    clean_fraud_data, clean_creditcard,
    merge_ip_country, engineer_features, encode_and_scale,
    _check_required_columns, _check_not_empty,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_fraud():
    return pd.DataFrame({
        "user_id":       [1, 2, 2],
        "signup_time":   pd.to_datetime(["2023-01-01 10:00", "2023-01-01 08:00", "2023-01-01 08:00"]),
        "purchase_time": pd.to_datetime(["2023-01-01 12:00", "2023-01-02 09:00", "2023-01-03 10:00"]),
        "purchase_value":[100, 200, 150],
        "device_id":     ["d1", "d2", "d2"],
        "source":        ["SEO", "Ads", "Ads"],
        "browser":       ["Chrome", "Firefox", "Firefox"],
        "sex":           ["M", "F", "F"],
        "age":           [25, 35, 35],
        "ip_address":    [3.5e8, 1.6e8, 1.6e8],
        "class":         [0, 1, 0],
    })


@pytest.fixture
def sample_ip():
    return pd.DataFrame({
        "lower_bound_ip_address": [100_000_000, 300_000_000],
        "upper_bound_ip_address": [200_000_000, 400_000_000],
        "country": ["Germany", "United States"],
    })


@pytest.fixture
def sample_cc():
    rng = np.random.default_rng(42)
    n = 20
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = np.arange(n, dtype=float)
    data["Amount"] = rng.uniform(1, 500, n)
    data["Class"] = [0] * 18 + [1, 1]
    return pd.DataFrame(data)


# ── Validation helper tests ───────────────────────────────────────────────────

def test_check_required_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        _check_required_columns(df, {"a", "b", "c"}, "test")


def test_check_required_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    _check_required_columns(df, {"a", "b"}, "test")  # should not raise


def test_check_not_empty_raises_on_empty_df():
    df = pd.DataFrame({"a": []})
    with pytest.raises(ValueError, match="zero rows"):
        _check_not_empty(df, "test")


# ── Loader error handling ─────────────────────────────────────────────────────

def test_load_fraud_data_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_fraud_data("/nonexistent/path/Fraud_Data.csv")


def test_load_ip_country_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_ip_country("/nonexistent/IpAddress_to_Country.csv")


def test_load_creditcard_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_creditcard("/nonexistent/creditcard.csv")


def test_load_fraud_invalid_class_values(tmp_path, sample_fraud):
    bad = sample_fraud.copy()
    bad.loc[0, "class"] = 99
    p = tmp_path / "bad_fraud.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="class.*0/1"):
        load_fraud_data(str(p))


def test_load_creditcard_invalid_class_values(tmp_path, sample_cc):
    bad = sample_cc.copy()
    bad.loc[0, "Class"] = 5
    p = tmp_path / "bad_cc.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Class.*0/1"):
        load_creditcard(str(p))


# ── Cleaning tests ────────────────────────────────────────────────────────────

def test_clean_fraud_adds_ip_int(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    assert "ip_int" in cleaned.columns
    assert cleaned["ip_int"].dtype == np.int64


def test_clean_fraud_lowercases_strings(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    assert cleaned["source"].iloc[0] == "seo"
    assert cleaned["browser"].iloc[0] == "chrome"


def test_clean_fraud_drops_duplicates():
    df = pd.DataFrame({
        "user_id": [1, 1], "signup_time": ["2023-01-01", "2023-01-01"],
        "purchase_time": ["2023-01-02", "2023-01-02"],
        "purchase_value": [100, 100], "device_id": ["d1", "d1"],
        "source": ["SEO", "SEO"], "browser": ["Chrome", "Chrome"],
        "sex": ["M", "M"], "age": [25, 25],
        "ip_address": [1e8, 1e8], "class": [0, 0],
    })
    df[["signup_time", "purchase_time"]] = df[["signup_time", "purchase_time"]].apply(pd.to_datetime)
    cleaned = clean_fraud_data(df)
    assert len(cleaned) == 1


def test_clean_fraud_raises_on_wrong_type():
    with pytest.raises(TypeError, match="DataFrame"):
        clean_fraud_data("not a dataframe")


def test_clean_fraud_raises_on_empty():
    df = pd.DataFrame(columns=["user_id","signup_time","purchase_time",
                                "purchase_value","device_id","source",
                                "browser","sex","age","ip_address","class"])
    with pytest.raises(ValueError, match="zero rows"):
        clean_fraud_data(df)


# ── Geo merge tests ───────────────────────────────────────────────────────────

def test_merge_ip_country_correct_lookup(sample_fraud, sample_ip):
    cleaned = clean_fraud_data(sample_fraud)
    merged = merge_ip_country(cleaned, sample_ip)
    assert "country" in merged.columns
    row = merged[merged["user_id"] == 1].iloc[0]
    assert row["country"] == "United States"  # ip 3.5e8 in 300M-400M range


def test_merge_ip_country_unknown_for_unmatched(sample_fraud, sample_ip):
    cleaned = clean_fraud_data(sample_fraud)
    # Give a row an IP outside all ranges
    cleaned.loc[0, "ip_int"] = 999_999_999
    merged = merge_ip_country(cleaned, sample_ip)
    assert merged[merged["user_id"] == 1].iloc[0]["country"] == "Unknown"


def test_merge_ip_raises_on_missing_ip_int(sample_fraud, sample_ip):
    with pytest.raises(ValueError, match="ip_int"):
        merge_ip_country(sample_fraud, sample_ip)  # not cleaned yet


# ── Feature engineering tests ─────────────────────────────────────────────────

def test_engineer_features_creates_all_columns(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    for col in ["time_since_signup", "hour_of_day", "day_of_week",
                "user_tx_count", "user_tx_velocity"]:
        assert col in fe.columns, f"Missing column: {col}"


def test_time_since_signup_positive(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    assert (fe["time_since_signup"] >= 0).all()


def test_user_tx_count_correct(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    assert fe[fe["user_id"] == 2]["user_tx_count"].iloc[0] == 2
    assert fe[fe["user_id"] == 1]["user_tx_count"].iloc[0] == 1


def test_engineer_features_raises_on_wrong_type():
    with pytest.raises(TypeError):
        engineer_features([1, 2, 3])


def test_engineer_features_raises_on_missing_cols(sample_fraud):
    with pytest.raises(ValueError, match="Missing required columns"):
        engineer_features(sample_fraud.drop(columns=["signup_time"]))


# ── Encode & scale tests ──────────────────────────────────────────────────────

def test_encode_and_scale_removes_cat_cols(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    fe["country"] = "Germany"
    X = fe.drop(columns=["user_id","device_id","signup_time","purchase_time",
                          "ip_address","ip_int","class"])
    encoded, _ = encode_and_scale(X)
    for col in ["source", "browser", "sex", "country"]:
        assert col not in encoded.columns


def test_encode_and_scale_raises_without_scaler_on_transform(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    X = fe.drop(columns=["user_id","device_id","signup_time","purchase_time",
                          "ip_address","ip_int","class"])
    with pytest.raises(ValueError, match="no scaler provided"):
        encode_and_scale(X, scaler=None, fit=False)


def test_encode_and_scale_raises_on_wrong_type():
    with pytest.raises(TypeError, match="DataFrame"):
        encode_and_scale("not a df")
