from pathlib import Path

import joblib
import pandas as pd

from src.services.backtest_service import backtest_one_step


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


def test_backtest_one_step(tmp_path):
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

    result = backtest_one_step(str(csv_path), test_hours=48)

    assert "metrics" in result
    assert "results" in result
    assert len(result["results"]) == 48
