import pandas as pd

from src.features.build_features import build_features


def test_build_features_creates_expected_columns():
    idx = pd.date_range("2024-01-01", periods=300, freq="h")
    df = pd.DataFrame({"load_mw": range(300)}, index=idx)

    result = build_features(df)

    expected_columns = {
        "load_mw",
        "hour",
        "dayofweek",
        "month",
        "quarter",
        "dayofyear",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
    }

    assert expected_columns.issubset(set(result.columns))
    assert not result.empty
