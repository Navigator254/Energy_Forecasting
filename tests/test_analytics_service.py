import pandas as pd

from src.services.analytics_service import compute_usage_insights


def test_compute_usage_insights(tmp_path):
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    df = pd.DataFrame({"Datetime": idx, "PJME_MW": range(len(idx))})
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    result = compute_usage_insights(str(csv_path))

    assert "summary" in result
    assert "hourly_avg" in result
    assert "weekday_avg" in result
    assert "monthly_avg" in result
    assert "daily_peak" in result
    assert "hourly_heatmap" in result
    assert "top_10_peaks" in result
