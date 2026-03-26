from pathlib import Path
from typing import Dict, List
import json

import joblib
import pandas as pd

from src.data.load_data import load_energy_data
from src.features.build_features import build_features


MODEL_PATH = Path("models/rf_model.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")
RESIDUAL_STATS_PATH = Path("models/residual_stats.json")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)


def load_feature_columns() -> List[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature columns file not found. Train the model first.")
    return joblib.load(FEATURES_PATH)


def load_residual_stats() -> dict | None:
    if not RESIDUAL_STATS_PATH.exists():
        return None
    with open(RESIDUAL_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_next_from_csv(csv_path: str) -> Dict[str, float | str]:
    df = load_energy_data(Path(csv_path))
    feature_df = build_features(df)

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty. Not enough history.")

    model = load_model()
    feature_columns = load_feature_columns()
    residual_stats = load_residual_stats()

    latest_row = feature_df.drop(columns=["load_mw"]).iloc[[-1]].copy()
    latest_row = latest_row.reindex(columns=feature_columns)

    if latest_row.isnull().any().any():
        missing = latest_row.columns[latest_row.isnull().any()].tolist()
        raise ValueError(f"Missing required feature values: {missing}")

    prediction = float(model.predict(latest_row)[0])
    result: Dict[str, float | str] = {
        "timestamp": str(feature_df.index[-1]),
        "predicted_load_mw": prediction,
    }

    if residual_stats:
        result["lower_bound_mw"] = prediction + float(residual_stats["q05"])
        result["upper_bound_mw"] = prediction + float(residual_stats["q95"])

    return result


def predict_from_feature_payload(payload: dict) -> float:
    model = load_model()
    feature_columns = load_feature_columns()

    X = pd.DataFrame([payload])
    X = X.reindex(columns=feature_columns)

    if X.isnull().any().any():
        missing = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Missing required feature values: {missing}")

    prediction = float(model.predict(X)[0])
    return prediction
