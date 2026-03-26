import pandas as pd

from src.data.load_data import load_energy_data
from src.features.build_features import build_features
from src.models.baselines import compare_models


def test_compare_models_returns_expected_models(tmp_path):
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    df = pd.DataFrame({"Datetime": idx, "PJME_MW": range(500)})
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_energy_data(csv_path)
    feature_df = build_features(loaded)
    results = compare_models(feature_df)

    assert not results.empty
    assert "model" in results.columns
    assert "rmse" in results.columns
    assert set(results["model"]) == {
        "LinearRegression",
        "RandomForest",
        "HistGradientBoosting",
    }
