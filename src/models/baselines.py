from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    denom = y_true.replace(0, np.nan)
    mape = float((np.abs((y_true - y_pred) / denom)).dropna().mean() * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def _time_split(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = feature_df.drop(columns=["load_mw"])
    y = feature_df["load_mw"]

    split_idx = int(len(feature_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def compare_models(feature_df: pd.DataFrame) -> pd.DataFrame:
    if "load_mw" not in feature_df.columns:
        raise ValueError("feature_df must contain 'load_mw'.")

    X_train, X_test, y_train, y_test = _time_split(feature_df)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
    }

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = _metrics(y_test, preds)
        rows.append(
            {
                "model": name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values("rmse", ascending=True).reset_index(drop=True)
    return results_df
