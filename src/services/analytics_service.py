from pathlib import Path
from typing import Dict

import pandas as pd

from src.data.load_data import load_energy_data


DAY_NAME_MAP = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

MONTH_NAME_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def compute_usage_insights(csv_path: str) -> Dict[str, object]:
    df = load_energy_data(Path(csv_path)).copy()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["day_name"] = df["dayofweek"].map(DAY_NAME_MAP)
    df["month"] = df.index.month
    df["month_name"] = df["month"].map(MONTH_NAME_MAP)
    df["date"] = df.index.date

    hourly_avg = (
        df.groupby("hour")["load_mw"]
        .mean()
        .reset_index()
        .rename(columns={"load_mw": "avg_load_mw"})
    )

    weekday_avg = (
        df.groupby(["dayofweek", "day_name"])["load_mw"]
        .mean()
        .reset_index()
        .sort_values("dayofweek")
        .drop(columns=["dayofweek"])
        .rename(columns={"load_mw": "avg_load_mw"})
    )

    monthly_avg = (
        df.groupby(["month", "month_name"])["load_mw"]
        .mean()
        .reset_index()
        .sort_values("month")
        .drop(columns=["month"])
        .rename(columns={"load_mw": "avg_load_mw"})
    )

    daily_peak = (
        df.groupby("date")["load_mw"]
        .max()
        .reset_index()
        .rename(columns={"load_mw": "daily_peak_load_mw"})
    )

    hourly_heatmap = (
        df.assign(hour=df.index.hour, day_name=df["day_name"])
        .pivot_table(
            index="day_name",
            columns="hour",
            values="load_mw",
            aggfunc="mean",
        )
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )

    top_10_peaks = (
        df[["load_mw"]]
        .sort_values("load_mw", ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )

    peak_hour_row = hourly_avg.sort_values("avg_load_mw", ascending=False).iloc[0]
    quiet_hour_row = hourly_avg.sort_values("avg_load_mw", ascending=True).iloc[0]
    busiest_day_row = weekday_avg.sort_values("avg_load_mw", ascending=False).iloc[0]
    quietest_day_row = weekday_avg.sort_values("avg_load_mw", ascending=True).iloc[0]
    busiest_month_row = monthly_avg.sort_values("avg_load_mw", ascending=False).iloc[0]
    quietest_month_row = monthly_avg.sort_values("avg_load_mw", ascending=True).iloc[0]

    summary = {
        "peak_hour": int(peak_hour_row["hour"]),
        "peak_hour_avg_load_mw": float(peak_hour_row["avg_load_mw"]),
        "quiet_hour": int(quiet_hour_row["hour"]),
        "quiet_hour_avg_load_mw": float(quiet_hour_row["avg_load_mw"]),
        "busiest_day": str(busiest_day_row["day_name"]),
        "busiest_day_avg_load_mw": float(busiest_day_row["avg_load_mw"]),
        "quietest_day": str(quietest_day_row["day_name"]),
        "quietest_day_avg_load_mw": float(quietest_day_row["avg_load_mw"]),
        "busiest_month": str(busiest_month_row["month_name"]),
        "busiest_month_avg_load_mw": float(busiest_month_row["avg_load_mw"]),
        "quietest_month": str(quietest_month_row["month_name"]),
        "quietest_month_avg_load_mw": float(quietest_month_row["avg_load_mw"]),
        "overall_mean_load_mw": float(df["load_mw"].mean()),
        "overall_max_load_mw": float(df["load_mw"].max()),
        "overall_min_load_mw": float(df["load_mw"].min()),
    }

    return {
        "summary": summary,
        "hourly_avg": hourly_avg,
        "weekday_avg": weekday_avg,
        "monthly_avg": monthly_avg,
        "daily_peak": daily_peak,
        "hourly_heatmap": hourly_heatmap,
        "top_10_peaks": top_10_peaks,
    }
