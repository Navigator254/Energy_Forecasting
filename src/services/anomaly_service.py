import pandas as pd


def detect_and_clean_anomalies(
    df: pd.DataFrame,
    value_col: str,
    window: int = 24,
    z_threshold: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    series = pd.to_numeric(out[value_col], errors="coerce")

    rolling_median = series.rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    abs_dev = (series - rolling_median).abs()
    rolling_mad = abs_dev.rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    threshold = z_threshold * 1.4826 * rolling_mad

    anomaly_mask = ((abs_dev > threshold) & threshold.notna()) | (series <= 0)
    anomaly_mask = anomaly_mask.fillna(False)

    cleaned = series.copy()
    cleaned.loc[anomaly_mask] = rolling_median.loc[anomaly_mask]
    cleaned = cleaned.interpolate(method="linear").ffill().bfill()

    anomalies = out.loc[anomaly_mask].copy()
    anomalies["original_value"] = series.loc[anomaly_mask]
    anomalies["cleaned_value"] = cleaned.loc[anomaly_mask]

    out[value_col] = cleaned
    return out, anomalies
