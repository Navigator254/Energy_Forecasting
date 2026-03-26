from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.load_data import load_energy_data
from src.features.build_features import build_features
from src.models.predict import load_feature_columns, load_model, load_residual_stats
from src.services.storage_service import save_forecast_run


def _prepare_history(csv_path: str) -> pd.DataFrame:
    df = load_energy_data(Path(csv_path))
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    return df.copy()


def _build_feature_row(history_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = build_features(history_df)
    if feature_df.empty:
        raise ValueError("Not enough history to build forecast features.")
    return feature_df.drop(columns=["load_mw"]).iloc[[-1]].copy()


def _with_interval(prediction: float, residual_stats: dict | None) -> tuple[float | None, float | None]:
    if not residual_stats:
        return None, None
    return (
        prediction + float(residual_stats["q05"]),
        prediction + float(residual_stats["q95"]),
    )


def forecast_hours_from_csv(
    csv_path: str,
    hours: int,
    persist: bool = False,
    source: str = "api",
) -> List[Dict[str, float | str]]:
    history_df = _prepare_history(csv_path)
    model = load_model()
    feature_columns = load_feature_columns()
    residual_stats = load_residual_stats()

    forecasts: List[Dict[str, float | str]] = []

    for _ in range(hours):
        latest_features = _build_feature_row(history_df)
        latest_features = latest_features.reindex(columns=feature_columns)

        if latest_features.isnull().any().any():
            missing = latest_features.columns[latest_features.isnull().any()].tolist()
            raise ValueError(f"Missing forecast feature values: {missing}")

        next_prediction = float(model.predict(latest_features)[0])
        lower_bound, upper_bound = _with_interval(next_prediction, residual_stats)
        next_timestamp = history_df.index[-1] + pd.Timedelta(hours=1)

        record = {
            "timestamp": str(next_timestamp),
            "predicted_load_mw": next_prediction,
            "lower_bound_mw": lower_bound,
            "upper_bound_mw": upper_bound,
        }
        forecasts.append(record)

        new_row = pd.DataFrame({"load_mw": [next_prediction]}, index=[next_timestamp])
        history_df = pd.concat([history_df, new_row])

    if persist:
        save_forecast_run(
            records=forecasts,
            horizon_hours=hours,
            source=source,
            csv_path=csv_path,
            model_name="RandomForest",
        )

    return forecasts


def forecast_next_24_hours_from_csv(csv_path: str, persist: bool = False, source: str = "api") -> List[Dict[str, float | str]]:
    return forecast_hours_from_csv(csv_path, 24, persist=persist, source=source)


def forecast_next_7_days_from_csv(csv_path: str, persist: bool = False, source: str = "api") -> List[Dict[str, float | str]]:
    return forecast_hours_from_csv(csv_path, 168, persist=persist, source=source)
