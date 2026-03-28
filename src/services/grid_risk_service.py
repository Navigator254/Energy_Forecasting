from typing import Dict, List

import pandas as pd

from src.services.forecast_service import (
    forecast_next_24_hours_from_csv,
    forecast_next_7_days_from_csv,
)


def _classify_stress(
    predicted_load: float,
    baseline_load: float,
    elevated_ratio: float,
    high_ratio: float,
    critical_ratio: float,
) -> tuple[str, float]:
    if baseline_load <= 0:
        return "Unknown", 0.0

    stress_ratio = predicted_load / baseline_load

    if stress_ratio >= critical_ratio:
        return "Critical", stress_ratio
    if stress_ratio >= high_ratio:
        return "High", stress_ratio
    if stress_ratio >= elevated_ratio:
        return "Elevated", stress_ratio
    return "Normal", stress_ratio


def _classify_shedding_risk(
    stress_level: str,
    stress_ratio: float,
    lower_bound: float | None,
    upper_bound: float | None,
    baseline_load: float,
) -> str:
    interval_width_ratio = 0.0
    if (
        lower_bound is not None
        and upper_bound is not None
        and baseline_load > 0
    ):
        interval_width_ratio = (upper_bound - lower_bound) / baseline_load

    if stress_level == "Critical":
        return "High"
    if stress_level == "High" and interval_width_ratio >= 0.20:
        return "High"
    if stress_level == "High":
        return "Moderate"
    if stress_level == "Elevated" and interval_width_ratio >= 0.25:
        return "Moderate"
    if stress_ratio >= 0.90:
        return "Moderate"
    return "Low"


def _balancing_recommendation(stress_level: str) -> str:
    if stress_level == "Critical":
        return "Immediate balancing action recommended"
    if stress_level == "High":
        return "Prepare balancing resources"
    if stress_level == "Elevated":
        return "Monitor closely"
    return "No immediate action"


def _build_risk_output(
    forecast_records: List[Dict[str, float | str]],
) -> Dict[str, object]:
    df = pd.DataFrame(forecast_records).copy()

    if df.empty:
        raise ValueError("Forecast records are empty.")

    df["predicted_load_mw"] = pd.to_numeric(df["predicted_load_mw"], errors="coerce")
    if "lower_bound_mw" in df.columns:
        df["lower_bound_mw"] = pd.to_numeric(df["lower_bound_mw"], errors="coerce")
    else:
        df["lower_bound_mw"] = None
    if "upper_bound_mw" in df.columns:
        df["upper_bound_mw"] = pd.to_numeric(df["upper_bound_mw"], errors="coerce")
    else:
        df["upper_bound_mw"] = None

    baseline_load = float(df["predicted_load_mw"].mean())
    recent_peak = float(df["predicted_load_mw"].max())
    reference_load = max(baseline_load, recent_peak * 0.85)

    elevated_ratio = 0.90
    high_ratio = 1.00
    critical_ratio = 1.10

    rows = []
    for _, row in df.iterrows():
        predicted = float(row["predicted_load_mw"])
        lower_bound = None if pd.isna(row["lower_bound_mw"]) else float(row["lower_bound_mw"])
        upper_bound = None if pd.isna(row["upper_bound_mw"]) else float(row["upper_bound_mw"])

        stress_level, stress_ratio = _classify_stress(
            predicted_load=predicted,
            baseline_load=reference_load,
            elevated_ratio=elevated_ratio,
            high_ratio=high_ratio,
            critical_ratio=critical_ratio,
        )

        shedding_risk = _classify_shedding_risk(
            stress_level=stress_level,
            stress_ratio=stress_ratio,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            baseline_load=reference_load,
        )

        rows.append(
            {
                "timestamp": str(row["timestamp"]),
                "predicted_load_mw": predicted,
                "lower_bound_mw": lower_bound,
                "upper_bound_mw": upper_bound,
                "stress_ratio": round(stress_ratio, 4),
                "stress_level": stress_level,
                "load_shedding_risk": shedding_risk,
                "balancing_recommendation": _balancing_recommendation(stress_level),
            }
        )

    risk_df = pd.DataFrame(rows)

    summary = {
        "reference_load_mw": reference_load,
        "average_forecast_load_mw": baseline_load,
        "peak_forecast_load_mw": recent_peak,
        "critical_hours": int((risk_df["stress_level"] == "Critical").sum()),
        "high_hours": int((risk_df["stress_level"] == "High").sum()),
        "elevated_hours": int((risk_df["stress_level"] == "Elevated").sum()),
        "high_shedding_risk_hours": int((risk_df["load_shedding_risk"] == "High").sum()),
        "moderate_shedding_risk_hours": int((risk_df["load_shedding_risk"] == "Moderate").sum()),
    }

    return {
        "summary": summary,
        "risk_table": risk_df,
    }


def grid_risk_24h_from_csv(csv_path: str) -> Dict[str, object]:
    forecast_records = forecast_next_24_hours_from_csv(csv_path)
    return _build_risk_output(forecast_records)


def grid_risk_7d_from_csv(csv_path: str) -> Dict[str, object]:
    forecast_records = forecast_next_7_days_from_csv(csv_path)
    return _build_risk_output(forecast_records)
