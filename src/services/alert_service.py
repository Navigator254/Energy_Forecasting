from typing import Dict, List

import pandas as pd

from src.services.grid_risk_service import (
    balancing_alerts_24h_from_csv,
    load_shedding_risk_24h_from_csv,
)


def build_threshold_alerts_from_csv(
    csv_path: str,
    elevated_ratio: float = 0.90,
    high_ratio: float = 1.00,
    critical_ratio: float = 1.10,
    high_shedding_only: bool = False,
) -> Dict[str, object]:
    balancing = balancing_alerts_24h_from_csv(
        csv_path,
        elevated_ratio=elevated_ratio,
        high_ratio=high_ratio,
        critical_ratio=critical_ratio,
    )
    shedding = load_shedding_risk_24h_from_csv(
        csv_path,
        elevated_ratio=elevated_ratio,
        high_ratio=high_ratio,
        critical_ratio=critical_ratio,
    )

    alerts: List[dict] = []

    balancing_df = balancing["alerts"].copy()
    if not balancing_df.empty:
        for _, row in balancing_df.iterrows():
            alerts.append(
                {
                    "timestamp": row["timestamp"],
                    "alert_type": "balancing",
                    "severity": row["stress_level"],
                    "message": row["balancing_recommendation"],
                    "predicted_load_mw": float(row["predicted_load_mw"]),
                    "stress_ratio": float(row["stress_ratio"]),
                }
            )

    shedding_df = shedding["risk_table"].copy()
    if not shedding_df.empty:
        if high_shedding_only:
            shedding_df = shedding_df[shedding_df["load_shedding_risk"] == "High"]
        else:
            shedding_df = shedding_df[
                shedding_df["load_shedding_risk"].isin(["High", "Moderate"])
            ]

        for _, row in shedding_df.iterrows():
            alerts.append(
                {
                    "timestamp": row["timestamp"],
                    "alert_type": "load_shedding_risk",
                    "severity": row["load_shedding_risk"],
                    "message": f"Potential load shedding risk: {row['load_shedding_risk']}",
                    "predicted_load_mw": float(row["predicted_load_mw"]),
                    "stress_ratio": float(row["stress_ratio"]),
                }
            )

    alerts_df = pd.DataFrame(alerts)
    if not alerts_df.empty:
        severity_order = {"Critical": 4, "High": 3, "Moderate": 2, "Elevated": 1}
        alerts_df["severity_rank"] = alerts_df["severity"].map(severity_order).fillna(0)
        alerts_df = alerts_df.sort_values(
            ["severity_rank", "stress_ratio", "predicted_load_mw"],
            ascending=[False, False, False],
        ).drop(columns=["severity_rank"])

    summary = {
        "total_alerts": int(len(alerts_df)),
        "balancing_alerts": int((alerts_df["alert_type"] == "balancing").sum()) if not alerts_df.empty else 0,
        "load_shedding_alerts": int((alerts_df["alert_type"] == "load_shedding_risk").sum()) if not alerts_df.empty else 0,
    }

    return {
        "summary": summary,
        "alerts": alerts_df,
    }
