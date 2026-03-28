from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.utils.env_loader import load_project_env
from src.data.load_data import load_energy_data
from src.models.predict import predict_next_from_csv
from src.services.alert_service import build_threshold_alerts_from_csv
from src.services.analytics_service import compute_usage_insights
from src.services.backtest_service import backtest_one_step
from src.services.forecast_service import (
    forecast_next_24_hours_from_csv,
    forecast_next_7_days_from_csv,
)
from src.services.grid_risk_service import (
    grid_risk_24h_from_csv,
    grid_risk_7d_from_csv,
)
from src.services.model_comparison_service import compare_models_from_csv
from src.services.retrain_service import run_retraining_pipeline
from src.services.storage_service import get_forecast_runs, get_retraining_runs, init_db


load_project_env()

st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")

DATA_PATH = "data/PJME_hourly.csv"
METRICS_PATH = Path("models/metrics.json")
IMPORTANCE_PATH = Path("models/feature_importance.csv")


@st.cache_data
def load_clean_data(csv_path: str) -> pd.DataFrame:
    return load_energy_data(Path(csv_path))


def bootstrap_section(default_parent: str, default_subba: str) -> None:
    st.warning("Project data is not initialized yet on this deployment. Run the setup pipeline first.")

    st.subheader("Initialize project data and model")
    with st.form("bootstrap_form"):
        parent_region = st.text_input("Parent region", value=default_parent)
        subba = st.text_input("Sub-region (optional)", value=default_subba)
        submitted = st.form_submit_button("Run setup and retraining")

    if submitted:
        try:
            with st.spinner("Fetching data, cleaning, training model, and saving artifacts..."):
                result = run_retraining_pipeline(
                    parent_region=parent_region,
                    subba=subba or None,
                )
                st.cache_data.clear()
            st.success("Setup complete.")
            st.json(result)
            st.rerun()
        except Exception as exc:
            st.error(f"Setup failed: {exc}")

    st.stop()


def style_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "timestamp",
        "predicted_load_mw",
        "lower_bound_mw",
        "upper_bound_mw",
        "stress_ratio",
        "stress_level",
        "load_shedding_risk",
        "balancing_recommendation",
    ]
    cols = [c for c in ordered_cols if c in df.columns]
    return df[cols].copy()


def render_home(csv_path: str, df: pd.DataFrame) -> None:
    st.subheader("Overview")

    try:
        one_step = predict_next_from_csv(csv_path)
    except Exception as exc:
        st.error(f"Could not compute one-step forecast: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Latest actual load", f"{df['load_mw'].iloc[-1]:.2f}")
    c3.metric("One-step forecast", f"{one_step['predicted_load_mw']:.2f}")
    c4.metric("Last timestamp", str(df.index.max()))

    st.write("Recent actual load")
    st.line_chart(df.tail(168)["load_mw"])

    st.info("Use the sidebar to load heavier sections only when you need them.")


def render_alerts(csv_path: str, elevated_ratio: float, high_ratio: float, critical_ratio: float) -> None:
    st.subheader("Alerts")

    high_shedding_only = st.checkbox("Show only high load-shedding alerts", value=False)

    if st.button("Generate threshold alerts", use_container_width=True):
        with st.spinner("Generating alerts..."):
            result = build_threshold_alerts_from_csv(
                csv_path=csv_path,
                elevated_ratio=elevated_ratio,
                high_ratio=high_ratio,
                critical_ratio=critical_ratio,
                high_shedding_only=high_shedding_only,
            )

        summary = result["summary"]
        alerts_df = result["alerts"]

        a1, a2, a3 = st.columns(3)
        a1.metric("Total alerts", summary["total_alerts"])
        a2.metric("Balancing alerts", summary["balancing_alerts"])
        a3.metric("Load shedding alerts", summary["load_shedding_alerts"])

        if alerts_df.empty:
            st.success("No active alerts for the current thresholds.")
            return

        st.warning("Alerts generated from the next 24-hour forecast horizon.")
        st.dataframe(alerts_df, use_container_width=True)

        csv_alerts = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download alerts CSV",
            data=csv_alerts,
            file_name="alerts_24h.csv",
            mime="text/csv",
        )


def render_forecasts(csv_path: str) -> None:
    st.subheader("Forecasts")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run 24-hour forecast", use_container_width=True):
            with st.spinner("Computing 24-hour forecast..."):
                forecast_24h = pd.DataFrame(forecast_next_24_hours_from_csv(csv_path))
            forecast_24h["timestamp"] = pd.to_datetime(forecast_24h["timestamp"])
            forecast_24h = forecast_24h.set_index("timestamp")

            chart_24h = forecast_24h[["predicted_load_mw"]].copy()
            if "lower_bound_mw" in forecast_24h.columns:
                chart_24h["lower_bound_mw"] = forecast_24h["lower_bound_mw"]
                chart_24h["upper_bound_mw"] = forecast_24h["upper_bound_mw"]

            st.write("24-hour forecast")
            st.line_chart(chart_24h)
            st.dataframe(forecast_24h, use_container_width=True)

            csv_24h = forecast_24h.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download 24-hour forecast CSV",
                data=csv_24h,
                file_name="forecast_24h.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("Run 7-day forecast", use_container_width=True):
            with st.spinner("Computing 7-day forecast..."):
                forecast_7d = pd.DataFrame(forecast_next_7_days_from_csv(csv_path))
            forecast_7d["timestamp"] = pd.to_datetime(forecast_7d["timestamp"])
            forecast_7d = forecast_7d.set_index("timestamp")

            chart_7d = forecast_7d[["predicted_load_mw"]].copy()
            if "lower_bound_mw" in forecast_7d.columns:
                chart_7d["lower_bound_mw"] = forecast_7d["lower_bound_mw"]
                chart_7d["upper_bound_mw"] = forecast_7d["upper_bound_mw"]

            st.write("7-day forecast")
            st.line_chart(chart_7d)
            st.dataframe(forecast_7d.head(72), use_container_width=True)

            csv_7d = forecast_7d.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download 7-day forecast CSV",
                data=csv_7d,
                file_name="forecast_7d.csv",
                mime="text/csv",
            )


def render_grid_risk(csv_path: str, elevated_ratio: float, high_ratio: float, critical_ratio: float) -> None:
    st.subheader("Grid Risk")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Run 24-hour grid risk", use_container_width=True):
            with st.spinner("Computing 24-hour grid risk..."):
                risk_24h = grid_risk_24h_from_csv(
                    csv_path,
                    elevated_ratio=elevated_ratio,
                    high_ratio=high_ratio,
                    critical_ratio=critical_ratio,
                )

            risk_24h_df = style_risk_table(risk_24h["risk_table"].copy())
            risk_24h_df["timestamp"] = pd.to_datetime(risk_24h_df["timestamp"])
            risk_24h_df = risk_24h_df.set_index("timestamp")
            summary = risk_24h["summary"]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Critical hours", summary["critical_hours"])
            r2.metric("High stress hours", summary["high_hours"])
            r3.metric("Elevated hours", summary["elevated_hours"])
            r4.metric("High shedding risk hours", summary["high_shedding_risk_hours"])

            st.line_chart(risk_24h_df["stress_ratio"])
            st.dataframe(risk_24h_df, use_container_width=True)

    with c2:
        if st.button("Run 7-day grid risk", use_container_width=True):
            with st.spinner("Computing 7-day grid risk..."):
                risk_7d = grid_risk_7d_from_csv(
                    csv_path,
                    elevated_ratio=elevated_ratio,
                    high_ratio=high_ratio,
                    critical_ratio=critical_ratio,
                )

            risk_7d_df = style_risk_table(risk_7d["risk_table"].copy())
            risk_7d_df["timestamp"] = pd.to_datetime(risk_7d_df["timestamp"])
            risk_7d_df = risk_7d_df.set_index("timestamp")
            summary = risk_7d["summary"]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Critical hours", summary["critical_hours"])
            r2.metric("High stress hours", summary["high_hours"])
            r3.metric("Moderate shedding risk hours", summary["moderate_shedding_risk_hours"])
            r4.metric("High shedding risk hours", summary["high_shedding_risk_hours"])

            st.line_chart(risk_7d_df["stress_ratio"])
            st.dataframe(risk_7d_df.head(72), use_container_width=True)


def render_usage_patterns(csv_path: str) -> None:
    st.subheader("Usage Patterns")

    if st.button("Load usage insights", use_container_width=True):
        with st.spinner("Computing usage insights..."):
            insights = compute_usage_insights(csv_path)

        summary = insights["summary"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Peak hour", f"{summary['peak_hour']}:00", f"{summary['peak_hour_avg_load_mw']:.1f} MW avg")
        s2.metric("Quietest hour", f"{summary['quiet_hour']}:00", f"{summary['quiet_hour_avg_load_mw']:.1f} MW avg")
        s3.metric("Busiest day", summary["busiest_day"], f"{summary['busiest_day_avg_load_mw']:.1f} MW avg")
        s4.metric("Busiest month", summary["busiest_month"], f"{summary['busiest_month_avg_load_mw']:.1f} MW avg")

        st.bar_chart(insights["hourly_avg"].set_index("hour")["avg_load_mw"])
        st.bar_chart(insights["weekday_avg"].set_index("day_name")["avg_load_mw"])
        st.bar_chart(insights["monthly_avg"].set_index("month_name")["avg_load_mw"])


def render_backtesting(csv_path: str, backtest_hours: int) -> None:
    st.subheader("Backtesting")

    if st.button("Run backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            backtest = backtest_one_step(csv_path=csv_path, test_hours=backtest_hours)

        metrics = backtest["metrics"]
        backtest_df = pd.DataFrame(backtest["results"])
        backtest_df["timestamp"] = pd.to_datetime(backtest_df["timestamp"])
        backtest_df = backtest_df.set_index("timestamp")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("MAE", f"{metrics['mae']:.2f}")
        b2.metric("RMSE", f"{metrics['rmse']:.2f}")
        b3.metric("R²", f"{metrics['r2']:.3f}")
        b4.metric("MAPE", f"{metrics['mape']:.2f}%")

        st.line_chart(backtest_df[["actual_load_mw", "predicted_load_mw"]])
        st.line_chart(backtest_df["abs_error"])
        st.dataframe(backtest_df, use_container_width=True)


def render_model_comparison(csv_path: str) -> None:
    st.subheader("Model Comparison")

    if st.button("Compare models", use_container_width=True):
        with st.spinner("Comparing models..."):
            comparison = compare_models_from_csv(csv_path)
            comparison_df = pd.DataFrame(comparison["comparison"])

        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df.set_index("model")[["rmse", "mae"]])
        st.bar_chart(comparison_df.set_index("model")[["r2"]])


def render_history_ops(parent_region: str, subba: str) -> None:
    st.subheader("History & Operations")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load forecast history", use_container_width=True):
            forecast_runs = pd.DataFrame(get_forecast_runs(limit=20))
            if forecast_runs.empty:
                st.info("No saved forecast runs yet.")
            else:
                st.dataframe(forecast_runs, use_container_width=True)

    with col2:
        if st.button("Load retraining history", use_container_width=True):
            retraining_runs = pd.DataFrame(get_retraining_runs(limit=20))
            if retraining_runs.empty:
                st.info("No retraining runs yet.")
            else:
                st.dataframe(retraining_runs, use_container_width=True)

    if st.button("Run retraining pipeline now", type="primary"):
        with st.spinner("Fetching data and retraining..."):
            result = run_retraining_pipeline(parent_region=parent_region, subba=subba or None)
            st.cache_data.clear()
            st.success("Retraining complete.")
            st.json(result)
            st.rerun()


def render_model_details() -> None:
    st.subheader("Model Details")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.write("Saved training metrics")
        st.json(metrics)

    if IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(IMPORTANCE_PATH)
        st.write("Feature importance")
        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.head(15).set_index("feature")["importance"])


def main():
    init_db()

    st.title("Energy Forecasting Dashboard")
    st.write("Forecasting, grid-risk intelligence, threshold alerts, and automatic rerun after retraining.")

    with st.sidebar:
        st.header("Controls")
        csv_path = st.text_input("CSV path", DATA_PATH)
        page = st.radio(
            "Section",
            [
                "Home",
                "Alerts",
                "Forecasts",
                "Grid Risk",
                "Usage Patterns",
                "Backtesting",
                "Model Comparison",
                "History & Ops",
                "Model Details",
            ],
        )

        backtest_hours = st.selectbox("Backtest window (hours)", [24, 48, 72, 168, 336], index=3)
        parent_region = st.text_input("Retrain parent region", "CISO")
        subba = st.text_input("Retrain sub-region (optional)", "")

        st.subheader("Grid risk thresholds")
        elevated_ratio = st.slider("Elevated threshold", min_value=0.70, max_value=1.20, value=0.90, step=0.01)
        high_ratio = st.slider("High threshold", min_value=0.80, max_value=1.30, value=1.00, step=0.01)
        critical_ratio = st.slider("Critical threshold", min_value=0.90, max_value=1.50, value=1.10, step=0.01)

        if not (elevated_ratio < high_ratio < critical_ratio):
            st.error("Thresholds must satisfy: Elevated < High < Critical")
            st.stop()

    if not Path(csv_path).exists():
        bootstrap_section(parent_region, subba)

    try:
        df = load_clean_data(csv_path)
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    if page == "Home":
        render_home(csv_path, df)
    elif page == "Alerts":
        render_alerts(csv_path, elevated_ratio, high_ratio, critical_ratio)
    elif page == "Forecasts":
        render_forecasts(csv_path)
    elif page == "Grid Risk":
        render_grid_risk(csv_path, elevated_ratio, high_ratio, critical_ratio)
    elif page == "Usage Patterns":
        render_usage_patterns(csv_path)
    elif page == "Backtesting":
        render_backtesting(csv_path, backtest_hours)
    elif page == "Model Comparison":
        render_model_comparison(csv_path)
    elif page == "History & Ops":
        render_history_ops(parent_region, subba)
    elif page == "Model Details":
        render_model_details()


if __name__ == "__main__":
    main()
