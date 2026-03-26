import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if "load_mw" not in df.columns:
        raise ValueError("Input dataframe must contain 'load_mw' column.")

    feature_df = df.copy()

    feature_df["hour"] = feature_df.index.hour
    feature_df["dayofweek"] = feature_df.index.dayofweek
    feature_df["month"] = feature_df.index.month
    feature_df["quarter"] = feature_df.index.quarter
    feature_df["dayofyear"] = feature_df.index.dayofyear
    feature_df["is_weekend"] = feature_df["dayofweek"].isin([5, 6]).astype(int)

    feature_df["hour_sin"] = np.sin(2 * np.pi * feature_df["hour"] / 24)
    feature_df["hour_cos"] = np.cos(2 * np.pi * feature_df["hour"] / 24)
    feature_df["dayofweek_sin"] = np.sin(2 * np.pi * feature_df["dayofweek"] / 7)
    feature_df["dayofweek_cos"] = np.cos(2 * np.pi * feature_df["dayofweek"] / 7)
    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)

    feature_df["lag_1"] = feature_df["load_mw"].shift(1)
    feature_df["lag_2"] = feature_df["load_mw"].shift(2)
    feature_df["lag_3"] = feature_df["load_mw"].shift(3)
    feature_df["lag_24"] = feature_df["load_mw"].shift(24)
    feature_df["lag_48"] = feature_df["load_mw"].shift(48)
    feature_df["lag_72"] = feature_df["load_mw"].shift(72)
    feature_df["lag_168"] = feature_df["load_mw"].shift(168)

    feature_df["rolling_mean_6"] = feature_df["load_mw"].shift(1).rolling(6).mean()
    feature_df["rolling_mean_24"] = feature_df["load_mw"].shift(1).rolling(24).mean()
    feature_df["rolling_std_24"] = feature_df["load_mw"].shift(1).rolling(24).std()
    feature_df["rolling_mean_168"] = feature_df["load_mw"].shift(1).rolling(168).mean()
    feature_df["rolling_std_168"] = feature_df["load_mw"].shift(1).rolling(168).std()

    feature_df = feature_df.dropna().copy()
    return feature_df
