from pathlib import Path
from typing import Dict

from src.data.fetch_eia import RAW_OUTPUT, CLEAN_OUTPUT, ANOMALY_OUTPUT, clean_eia_data, fetch_all_eia_data
from src.data.load_data import load_energy_data
from src.features.build_features import build_features
from src.models.train import train_and_save_model
from src.services.storage_service import init_db, save_retraining_run


def run_retraining_pipeline(parent_region: str = "CISO", subba: str | None = None) -> Dict[str, object]:
    raw_df = fetch_all_eia_data(parent=parent_region, subba=subba, page_size=5000)
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_OUTPUT, index=False)

    clean_df, anomalies_df = clean_eia_data(raw_df)
    clean_df.to_csv(CLEAN_OUTPUT, index=False)
    anomalies_df.to_csv(ANOMALY_OUTPUT, index=False)

    df = load_energy_data(Path(CLEAN_OUTPUT))
    feature_df = build_features(df)
    metrics = train_and_save_model(feature_df)

    init_db()
    save_retraining_run(
        metrics=metrics,
        parent_region=parent_region,
        subba=subba,
        rows_in_dataset=len(df),
        anomaly_count=len(anomalies_df),
        model_name="RandomForest",
    )

    return {
        "parent_region": parent_region,
        "subba": subba,
        "rows_in_dataset": len(df),
        "anomaly_count": len(anomalies_df),
        "metrics": metrics,
        "clean_csv": str(CLEAN_OUTPUT),
        "raw_csv": str(RAW_OUTPUT),
        "anomaly_csv": str(ANOMALY_OUTPUT),
    }
