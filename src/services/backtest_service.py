from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.load_data import load_energy_data
from src.features.build_features import build_features
from src.models.predict import load_feature_columns, load_model
from src.utils.metrics import regression_metrics


def _predict_one_step_from_history(history_df: pd.DataFrame) -> float:
    feature_df = build_features(history_df)

    if feature_df.empty:
        raise ValueError("Not enough history to build features for backtesting.")

    model = load_model()
    feature_columns = load_feature_columns()

    latest_features = feature_df.drop(columns=["load_mw"]).iloc[[-1]].copy()
    latest_features = latest_features.reindex(columns=feature_columns)

    if latest_features.isnull().any().any():
        missing = latest_features.columns[latest_features.isnull().any()].tolist()
        raise ValueError(f"Missing feature values in backtest: {missing}")

    prediction = float(model.predict(latest_features)[0])
    return prediction


def backtest_one_step(csv_path: str, test_hours: int = 168) -> Dict[str, object]:
    df = load_energy_data(Path(csv_path))

    if len(df) < 400:
        raise ValueError("Dataset is too short for backtesting. Need at least ~400 rows.")

    if test_hours <= 0:
        raise ValueError("test_hours must be greater than 0.")

    if len(df) <= test_hours + 200:
        raise ValueError("Not enough rows for requested backtest window.")

    results: List[Dict[str, float | str]] = []

    start_idx = len(df) - test_hours

    for i in range(start_idx, len(df)):
        history_df = df.iloc[:i].copy()
        actual_timestamp = df.index[i]
        actual_value = float(df.iloc[i]["load_mw"])

        predicted_value = _predict_one_step_from_history(history_df)

        results.append(
            {
                "timestamp": str(actual_timestamp),
                "actual_load_mw": actual_value,
                "predicted_load_mw": predicted_value,
                "abs_error": abs(actual_value - predicted_value),
            }
        )

    results_df = pd.DataFrame(results)
    metrics = regression_metrics(
        results_df["actual_load_mw"],
        results_df["predicted_load_mw"],
    )

    return {
        "metrics": metrics,
        "results": results_df.to_dict(orient="records"),
    }
