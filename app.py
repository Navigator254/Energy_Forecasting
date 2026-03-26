from pathlib import Path

from src.data.load_data import load_energy_data
from src.features.build_features import build_features
from src.models.train import train_and_save_model


def main() -> None:
    data_path = Path("data/PJME_hourly.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run fetch first."
        )

    df = load_energy_data(data_path)
    feature_df = build_features(df)

    if feature_df.empty:
        raise ValueError("Feature dataframe is empty after feature engineering.")

    metrics = train_and_save_model(feature_df)

    print("\nTraining complete.")
    print(f"MAE  : {metrics['mae']:.3f}")
    print(f"RMSE : {metrics['rmse']:.3f}")
    print(f"R2   : {metrics['r2']:.3f}")
    print(f"MAPE : {metrics['mape']:.3f}")


if __name__ == "__main__":
    main()