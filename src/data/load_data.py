from pathlib import Path

import pandas as pd


def load_energy_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)

    df = df.sort_index()

    value_columns = list(df.columns)
    if not value_columns:
        raise ValueError("No load column found in dataset.")

    target_col = value_columns[0]
    df = df[[target_col]].copy()
    df.columns = ["load_mw"]

    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")

    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    df["load_mw"] = df["load_mw"].interpolate(method="time")
    df["load_mw"] = df["load_mw"].ffill().bfill()

    if df["load_mw"].isna().any():
        raise ValueError("Missing values remain after fill.")

    # Final mild clipping
    q01 = df["load_mw"].quantile(0.01)
    q99 = df["load_mw"].quantile(0.99)
    df["load_mw"] = df["load_mw"].clip(lower=q01, upper=q99)

    return df