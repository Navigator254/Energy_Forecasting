from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from src.utils.env_loader import load_project_env

from src.models.predict import predict_from_feature_payload, predict_next_from_csv
from src.services.alert_service import build_threshold_alerts_from_csv
from src.services.analytics_service import compute_usage_insights
from src.services.backtest_service import backtest_one_step
from src.services.forecast_service import (
    forecast_next_24_hours_from_csv,
    forecast_next_7_days_from_csv,
)
from src.services.grid_risk_service import (
    balancing_alerts_24h_from_csv,
    grid_risk_24h_from_csv,
    grid_risk_7d_from_csv,
    load_shedding_risk_24h_from_csv,
)
from src.services.model_comparison_service import compare_models_from_csv
from src.services.retrain_service import run_retraining_pipeline
from src.services.storage_service import get_forecast_runs, get_retraining_runs, init_db


load_project_env()

app = FastAPI(
    title="Energy Forecasting API",
    description="Hourly electricity load prediction and grid risk API",
    version="2.0.0",
)


class FeaturePayload(BaseModel):
    hour: int
    dayofweek: int
    month: int
    quarter: int
    dayofyear: int
    is_weekend: int
    hour_sin: float
    hour_cos: float
    dayofweek_sin: float
    dayofweek_cos: float
    month_sin: float
    month_cos: float
    lag_1: float
    lag_2: float
    lag_3: float
    lag_24: float
    lag_48: float
    lag_72: float
    lag_168: float
    rolling_mean_6: float
    rolling_mean_24: float
    rolling_std_24: float
    rolling_mean_168: float
    rolling_std_168: float


class CsvPathPayload(BaseModel):
    csv_path: str = "data/PJME_hourly.csv"


class BacktestPayload(BaseModel):
    csv_path: str = "data/PJME_hourly.csv"
    test_hours: int = 168


class RetrainPayload(BaseModel):
    parent_region: str = "CISO"
    subba: str | None = None


class GridRiskPayload(BaseModel):
    csv_path: str = "data/PJME_hourly.csv"
    elevated_ratio: float = 0.90
    high_ratio: float = 1.00
    critical_ratio: float = 1.10


class AlertPayload(BaseModel):
    csv_path: str = "data/PJME_hourly.csv"
    elevated_ratio: float = 0.90
    high_ratio: float = 1.00
    critical_ratio: float = 1.10
    high_shedding_only: bool = False


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Energy Forecasting API is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: FeaturePayload) -> Dict[str, float]:
    try:
        prediction = predict_from_feature_payload(payload.model_dump())
        return {"predicted_load_mw": prediction}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-from-csv")
def predict_from_csv(payload: CsvPathPayload) -> Dict[str, float | str]:
    try:
        return predict_next_from_csv(payload.csv_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast-24h")
def forecast_24h(payload: CsvPathPayload) -> Dict[str, List[Dict[str, float | str]]]:
    try:
        forecast = forecast_next_24_hours_from_csv(payload.csv_path, persist=True, source="api")
        return {"forecast_24h": forecast}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast-7d")
def forecast_7d(payload: CsvPathPayload) -> Dict[str, List[Dict[str, float | str]]]:
    try:
        forecast = forecast_next_7_days_from_csv(payload.csv_path, persist=True, source="api")
        return {"forecast_7d": forecast}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/grid-risk-24h")
def grid_risk_24h(payload: GridRiskPayload) -> Dict[str, object]:
    try:
        result = grid_risk_24h_from_csv(
            payload.csv_path,
            elevated_ratio=payload.elevated_ratio,
            high_ratio=payload.high_ratio,
            critical_ratio=payload.critical_ratio,
        )
        return {
            "summary": result["summary"],
            "risk_table": result["risk_table"].to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/grid-risk-7d")
def grid_risk_7d(payload: GridRiskPayload) -> Dict[str, object]:
    try:
        result = grid_risk_7d_from_csv(
            payload.csv_path,
            elevated_ratio=payload.elevated_ratio,
            high_ratio=payload.high_ratio,
            critical_ratio=payload.critical_ratio,
        )
        return {
            "summary": result["summary"],
            "risk_table": result["risk_table"].to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/load-shedding-risk")
def load_shedding_risk(payload: GridRiskPayload) -> Dict[str, object]:
    try:
        result = load_shedding_risk_24h_from_csv(
            payload.csv_path,
            elevated_ratio=payload.elevated_ratio,
            high_ratio=payload.high_ratio,
            critical_ratio=payload.critical_ratio,
        )
        return {
            "summary": result["summary"],
            "risk_table": result["risk_table"].to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/balancing-alerts")
def balancing_alerts(payload: GridRiskPayload) -> Dict[str, object]:
    try:
        result = balancing_alerts_24h_from_csv(
            payload.csv_path,
            elevated_ratio=payload.elevated_ratio,
            high_ratio=payload.high_ratio,
            critical_ratio=payload.critical_ratio,
        )
        return {
            "summary": result["summary"],
            "alerts": result["alerts"].to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/alerts")
def alerts(payload: AlertPayload) -> Dict[str, object]:
    try:
        result = build_threshold_alerts_from_csv(
            csv_path=payload.csv_path,
            elevated_ratio=payload.elevated_ratio,
            high_ratio=payload.high_ratio,
            critical_ratio=payload.critical_ratio,
            high_shedding_only=payload.high_shedding_only,
        )
        return {
            "summary": result["summary"],
            "alerts": result["alerts"].to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/backtest")
def backtest(payload: BacktestPayload) -> Dict[str, object]:
    try:
        return backtest_one_step(csv_path=payload.csv_path, test_hours=payload.test_hours)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compare-models")
def compare_models(payload: CsvPathPayload) -> Dict[str, object]:
    try:
        return compare_models_from_csv(payload.csv_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/usage-insights")
def usage_insights(payload: CsvPathPayload) -> Dict[str, object]:
    try:
        insights = compute_usage_insights(payload.csv_path)
        return {
            "summary": insights["summary"],
            "hourly_avg": insights["hourly_avg"].to_dict(orient="records"),
            "weekday_avg": insights["weekday_avg"].to_dict(orient="records"),
            "monthly_avg": insights["monthly_avg"].to_dict(orient="records"),
            "daily_peak": insights["daily_peak"].to_dict(orient="records"),
            "top_10_peaks": insights["top_10_peaks"].to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain-now")
def retrain_now(payload: RetrainPayload) -> Dict[str, object]:
    try:
        return run_retraining_pipeline(parent_region=payload.parent_region, subba=payload.subba)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/forecast-history")
def forecast_history(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, object]:
    try:
        return {"forecast_history": get_forecast_runs(limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/retraining-history")
def retraining_history(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, object]:
    try:
        return {"retraining_history": get_retraining_runs(limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
