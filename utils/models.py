"""
models.py
All forecasting models for UAC Project 2.

Models included:
  1. Naive (baseline)        — just repeat last known value
  2. Moving Average          — average of last N days
  3. Exponential Smoothing   — weighted recent average
  4. Random Forest           — ML ensemble model
  5. Gradient Boosting       — ML boosting model

ARIMA is excluded from this module because it runs very slowly
on 1000+ day series and crashes Streamlit — Exp Smoothing covers
the same statistical forecasting need much faster.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ── Evaluation helper ────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, model_name: str) -> dict:
    """Return MAE, RMSE, MAPE for a set of predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "Model": model_name,
        "MAE":   round(mae,  2),
        "RMSE":  round(rmse, 2),
        "MAPE":  round(mape, 2),
        "Accuracy (%)": round(max(0, 100 - mape), 2),
    }


# ── 1. Naive model ───────────────────────────────────────────────────────────

def naive_forecast(train_series: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last training value for every future step."""
    return np.full(horizon, train_series.iloc[-1])


# ── 2. Moving average ────────────────────────────────────────────────────────

def moving_average_forecast(train_series: pd.Series,
                             horizon: int,
                             window: int = 7) -> np.ndarray:
    """Forecast using the mean of the last `window` days."""
    last_mean = train_series.iloc[-window:].mean()
    return np.full(horizon, last_mean)


# ── 3. Exponential Smoothing ─────────────────────────────────────────────────

def exp_smoothing_forecast(train_series: pd.Series,
                            horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Holt-Winters Exponential Smoothing with additive trend.
    Returns (forecast, lower_ci, upper_ci).
    """
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, remove_bias=True)
    forecast = fit.forecast(horizon)

    # Simple confidence interval: ±1.96 * residual std
    residuals = train_series - fit.fittedvalues
    std = residuals.std()
    lower = forecast - 1.96 * std
    upper = forecast + 1.96 * std

    return forecast.values, lower.values, upper.values


# ── 4. Random Forest ─────────────────────────────────────────────────────────

def random_forest_forecast(X_train, y_train,
                            X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random Forest Regressor.
    Returns (predictions, lower_ci, upper_ci) using
    percentile spread across individual trees.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    # Confidence interval from tree spread
    tree_preds = np.array([t.predict(X_test) for t in rf.estimators_])
    lower = np.percentile(tree_preds, 5,  axis=0)
    upper = np.percentile(tree_preds, 95, axis=0)

    return preds, lower, upper, rf


# ── 5. Gradient Boosting ─────────────────────────────────────────────────────

def gradient_boosting_forecast(X_train, y_train,
                                X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradient Boosting Regressor.
    Returns (predictions, lower_ci, upper_ci) using
    quantile regression trick via separate models.
    """
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        loss="squared_error",
    )
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)

    # Lower/upper via quantile models
    gb_low = GradientBoostingRegressor(
        n_estimators=100, loss="quantile", alpha=0.05,
        learning_rate=0.05, max_depth=4, random_state=42
    )
    gb_high = GradientBoostingRegressor(
        n_estimators=100, loss="quantile", alpha=0.95,
        learning_rate=0.05, max_depth=4, random_state=42
    )
    gb_low.fit(X_train, y_train)
    gb_high.fit(X_train, y_train)

    lower = gb_low.predict(X_test)
    upper = gb_high.predict(X_test)

    return preds, lower, upper, gb


# ── Future forecast helper ────────────────────────────────────────────────────

def build_future_features(df: pd.DataFrame,
                           target: str,
                           horizon: int) -> pd.DataFrame:
    """
    Build a feature DataFrame for the next `horizon` days beyond
    the end of df, using rolling lag logic.
    Used to generate a genuine future forecast (not just test-set).
    """
    last_date   = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                  periods=horizon, freq="D")

    # Seed the rolling window with the last known values
    history = df[target].copy()

    rows = []
    for d in future_dates:
        lag1  = history.iloc[-1]
        lag7  = history.iloc[-7:].mean()  if len(history) >= 7  else history.mean()
        lag14 = history.iloc[-14:].mean() if len(history) >= 14 else history.mean()
        roll7 = history.iloc[-7:].mean()  if len(history) >= 7  else history.mean()
        roll14= history.iloc[-14:].mean() if len(history) >= 14 else history.mean()

        row = {
            "cbp_intake":              df["cbp_intake"].iloc[-7:].mean(),
            "cbp_load":                df["cbp_load"].iloc[-7:].mean(),
            "cbp_transfers":           df["cbp_transfers"].iloc[-7:].mean(),
            "dayofweek":               d.dayofweek,
            "month":                   d.month,
            "quarter":                 d.quarter,
            "is_weekend":              int(d.dayofweek >= 5),
            "day_of_year":             d.dayofyear,
            "net_pressure":            df["net_pressure"].iloc[-7:].mean(),
            f"{target}_lag1":          lag1,
            f"{target}_lag7":          lag7,
            f"{target}_lag14":         lag14,
            f"{target}_roll7":         roll7,
            f"{target}_roll14":        roll14,
        }
        rows.append(row)
        # Update history with the predicted value (autoregressive)
        history = pd.concat([history,
                              pd.Series([lag1], index=[d])])

    future_df = pd.DataFrame(rows, index=future_dates)
    return future_df
