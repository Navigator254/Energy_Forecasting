from pathlib import Path

import joblib
import pandas as pd

from src.services.forecast_service import (
    forecast_next_24_hours_from_csv,
    forecast_next_7_days_from_csv,
)


class DummyModel:
    def predict(self, X):
        return [float(X["lag_1"].iloc[0])]


FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "month",
    "quarter",
    "dayofyear",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "rolling_mean_6",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
]


def test_forecast_next_24_hours_from_csv(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    df = pd.DataFrame({"Datetime": idx, "PJME_MW": range(500)})
    csv_path = data_dir / "sample.csv"
    df.to_csv(csv_path, index=False)

    joblib.dump(DummyModel(), model_dir / "rf_model.pkl")
    joblib.dump(FEATURE_COLUMNS, model_dir / "feature_columns.pkl")

    result = forecast_next_24_hours_from_csv(str(csv_path))
    assert len(result) == 24
    assert "timestamp" in result[0]
    assert "predicted_load_mw" in result[0]


def test_forecast_next_7_days_from_csv(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = Path("models")
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    df = pd.DataFrame({"Datetime": idx, "PJME_MW": range(500)})
    csv_path = data_dir / "sample.csv"
    df.to_csv(csv_path, index=False)

    joblib.dump(DummyModel(), model_dir / "rf_model.pkl")
    joblib.dump(FEATURE_COLUMNS, model_dir / "feature_columns.pkl")

    result = forecast_next_7_days_from_csv(str(csv_path))
    assert len(result) == 168
    assert "timestamp" in result[0]
    assert "predicted_load_mw" in result[0]
