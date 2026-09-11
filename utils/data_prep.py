"""
data_prep.py
Loads, cleans, fills missing dates, and returns
a complete daily time-series ready for forecasting.
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/HHS_Unaccompanied_Alien_Children_Program.csv"

COL_MAP = {
    "Children apprehended and placed in CBP custody*": "cbp_intake",
    "Children in CBP custody":                         "cbp_load",
    "Children transferred out of CBP custody":         "cbp_transfers",
    "Children in HHS Care":                            "hhs_load",
    "Children discharged from HHS Care":               "hhs_discharges",
}


def load_and_prepare(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Steps:
    1. Load CSV and drop blank rows
    2. Fix HHS Care comma formatting
    3. Parse dates and rename columns
    4. Fill 355 missing days using time interpolation
    5. Add time features for ML models
    Returns a clean daily DataFrame with DatetimeIndex.
    """
    # ── Load ────────────────────────────────────────────────────────────────
    df = pd.read_csv(path)
    df = df.dropna(subset=["Date"]).copy()

    # Fix comma-formatted numbers
    df["Children in HHS Care"] = (
        df["Children in HHS Care"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns=COL_MAP)
    df = df.sort_values("Date").set_index("Date")

    # ── Fill missing dates via time interpolation ────────────────────────────
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df = df.interpolate(method="time")
    df.index.name = "Date"

    # ── Time features for ML ─────────────────────────────────────────────────
    df["dayofweek"]  = df.index.dayofweek       # 0=Mon … 6=Sun
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["day_of_year"]= df.index.dayofyear

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 7, 14]:
        df[f"hhs_load_lag{lag}"]       = df["hhs_load"].shift(lag)
        df[f"hhs_discharges_lag{lag}"] = df["hhs_discharges"].shift(lag)

    # ── Rolling averages ─────────────────────────────────────────────────────
    for window in [7, 14]:
        df[f"hhs_load_roll{window}"]       = df["hhs_load"].shift(1).rolling(window).mean()
        df[f"hhs_discharges_roll{window}"] = df["hhs_discharges"].shift(1).rolling(window).mean()

    # ── Net pressure signal ──────────────────────────────────────────────────
    df["net_pressure"] = df["cbp_transfers"] - df["hhs_discharges"]

    return df


def train_test_split(df: pd.DataFrame, target: str, split: float = 0.8):
    """
    Strict time-based split — NO random sampling.
    Returns (train_df, test_df, X_train, X_test, y_train, y_test)
    """
    feature_cols = [
        "cbp_intake", "cbp_load", "cbp_transfers",
        "dayofweek", "month", "quarter", "is_weekend", "day_of_year",
        "net_pressure",
        f"{target}_lag1", f"{target}_lag7", f"{target}_lag14",
        f"{target}_roll7", f"{target}_roll14",
    ]

    # Drop rows with NaN (caused by lag features at the start)
    df_model = df[feature_cols + [target]].dropna()

    split_idx = int(len(df_model) * split)
    train = df_model.iloc[:split_idx]
    test  = df_model.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test  = test[feature_cols]
    y_test  = test[target]

    return train, test, X_train, X_test, y_train, y_test
