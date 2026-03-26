from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
RESIDUAL_STATS_PATH = MODEL_DIR / "residual_stats.json"


def train_and_save_model(feature_df: pd.DataFrame) -> dict:
    if "load_mw" not in feature_df.columns:
        raise ValueError("Feature dataframe must contain 'load_mw'.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = feature_df.drop(columns=["load_mw"])
    y = feature_df["load_mw"]

    split_idx = int(len(feature_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    residuals = y_test - preds

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan)).dropna()) * 100)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    residual_stats = {
        "q05": float(np.quantile(residuals, 0.05)),
        "q25": float(np.quantile(residuals, 0.25)),
        "q75": float(np.quantile(residuals, 0.75)),
        "q95": float(np.quantile(residuals, 0.95)),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(RESIDUAL_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(residual_stats, f, indent=2)

    return metrics
