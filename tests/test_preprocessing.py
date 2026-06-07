"""
tests/test_preprocessing.py
Unit tests for src/data_preprocessing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import (
    clean_fraud_data,
    merge_ip_country,
    engineer_features,
    encode_and_scale,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

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


# ── Tests ────────────────────────────────────────────────────────────────────

def test_clean_adds_ip_int(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    assert "ip_int" in cleaned.columns
    assert cleaned["ip_int"].dtype == np.int64


def test_clean_lowercases_strings(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    assert cleaned["source"].iloc[0] == "seo"
    assert cleaned["browser"].iloc[0] == "chrome"


def test_merge_ip_country(sample_fraud, sample_ip):
    cleaned = clean_fraud_data(sample_fraud)
    merged = merge_ip_country(cleaned, sample_ip)
    assert "country" in merged.columns
    # ip 3.5e8 should fall in United States range (300M-400M)
    row = merged[merged["user_id"] == 1].iloc[0]
    assert row["country"] == "United States"


def test_engineer_features(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    assert "time_since_signup" in fe.columns
    assert "hour_of_day" in fe.columns
    assert "day_of_week" in fe.columns
    assert "user_tx_count" in fe.columns
    # user_id 2 has 2 transactions
    assert fe[fe["user_id"] == 2]["user_tx_count"].iloc[0] == 2


def test_time_since_signup_positive(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    assert (fe["time_since_signup"] >= 0).all()


def test_encode_and_scale_no_cat_cols_remain(sample_fraud):
    cleaned = clean_fraud_data(sample_fraud)
    fe = engineer_features(cleaned)
    fe["country"] = "Germany"
    encoded, _ = encode_and_scale(fe)
    for col in ["source", "browser", "sex", "country"]:
        assert col not in encoded.columns
