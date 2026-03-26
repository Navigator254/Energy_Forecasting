from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.load_data import load_energy_data
from src.features.build_features import build_features
from src.models.baselines import compare_models


def compare_models_from_csv(csv_path: str) -> Dict[str, List[Dict[str, float | str]]]:
    df = load_energy_data(Path(csv_path))
    feature_df = build_features(df)

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty. Not enough history for comparison.")

    results_df = compare_models(feature_df)

    return {
        "comparison": results_df.to_dict(orient="records")
    }
