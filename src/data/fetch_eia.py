from pathlib import Path
import os
import sys

import pandas as pd
import requests

from src.services.anomaly_service import detect_and_clean_anomalies
from src.utils.env_loader import load_project_env


load_project_env()

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
RAW_OUTPUT = Path("data/EIA_hourly_raw.csv")
CLEAN_OUTPUT = Path("data/PJME_hourly.csv")
ANOMALY_OUTPUT = Path("data/anomalies_detected.csv")


def get_api_key() -> str:
    api_key = os.getenv("EIA_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st
        api_key = st.secrets.get("EIA_API_KEY")
        if api_key:
            return api_key
    except Exception:
        pass

    raise EnvironmentError("EIA_API_KEY not found in environment or Streamlit secrets.")


def fetch_page(
    api_key: str,
    parent: str = "CISO",
    subba: str | None = None,
    offset: int = 0,
    length: int = 5000,
) -> dict:
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[parent][0]": parent,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": length,
    }
    if subba:
        params["facets[subba][0]"] = subba

    response = requests.get(BASE_URL, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def fetch_all_eia_data(
    parent: str = "CISO",
    subba: str | None = None,
    page_size: int = 5000,
) -> pd.DataFrame:
    api_key = get_api_key()
    offset = 0
    frames = []

    while True:
        payload = fetch_page(api_key=api_key, parent=parent, subba=subba, offset=offset, length=page_size)

        if "response" not in payload or "data" not in payload["response"]:
            raise ValueError(f"Unexpected EIA response format: {payload}")

        rows = payload["response"]["data"]
        if not rows:
            break

        frames.append(pd.DataFrame(rows))

        if len(rows) < page_size:
            break

        offset += page_size

    if not frames:
        raise ValueError("No data returned from EIA API.")

    return pd.concat(frames, ignore_index=True)


def clean_eia_data(
    df: pd.DataFrame,
    min_valid_mw: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"period", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["period"] = pd.to_datetime(out["period"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["period", "value"]).copy()
    out = out[out["value"] >= min_valid_mw].copy()

    grouped = (
        out.groupby("period", as_index=False)["value"]
        .sum()
        .sort_values("period")
        .rename(columns={"period": "Datetime", "value": "PJME_MW"})
    )

    cleaned, anomalies = detect_and_clean_anomalies(grouped, value_col="PJME_MW", window=24, z_threshold=6.0)

    q01 = cleaned["PJME_MW"].quantile(0.01)
    q99 = cleaned["PJME_MW"].quantile(0.99)
    cleaned["PJME_MW"] = cleaned["PJME_MW"].clip(lower=q01, upper=q99)

    return cleaned, anomalies


def main() -> None:
    parent = os.getenv("EIA_PARENT_REGION", "CISO")
    subba = os.getenv("EIA_SUBBA") or None
    min_valid_mw = float(os.getenv("EIA_MIN_VALID_MW", "1000"))

    raw_df = fetch_all_eia_data(parent=parent, subba=subba, page_size=5000)

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_OUTPUT, index=False)
    print(f"Saved raw EIA data to {RAW_OUTPUT}")

    clean_df, anomalies_df = clean_eia_data(raw_df, min_valid_mw=min_valid_mw)
    clean_df.to_csv(CLEAN_OUTPUT, index=False)
    print(f"Saved cleaned training data to {CLEAN_OUTPUT}")

    anomalies_df.to_csv(ANOMALY_OUTPUT, index=False)
    print(f"Saved detected anomalies to {ANOMALY_OUTPUT}")
    print(clean_df.head())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
