from pathlib import Path
import json

import pandas as pd
import streamlit as st
from src.utils.env_loader import load_project_env

from src.data.load_data import load_energy_data
from src.models.predict import predict_next_from_csv
from src.services.analytics_service import compute_usage_insights
from src.services.backtest_service import backtest_one_step
from src.services.forecast_service import (
    forecast_next_24_hours_from_csv,
    forecast_next_7_days_from_csv,
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


@st.cache_data
def get_forecast_24h(csv_path: str) -> pd.DataFrame:
    return pd.DataFrame(forecast_next_24_hours_from_csv(csv_path))


@st.cache_data
def get_forecast_7d(csv_path: str) -> pd.DataFrame:
    return pd.DataFrame(forecast_next_7_days_from_csv(csv_path))


@st.cache_data
def get_backtest(csv_path: str, test_hours: int) -> dict:
    return backtest_one_step(csv_path=csv_path, test_hours=test_hours)


@st.cache_data
def get_model_comparison(csv_path: str) -> pd.DataFrame:
    result = compare_models_from_csv(csv_path)
    return pd.DataFrame(result["comparison"])


@st.cache_data
def get_usage_insights(csv_path: str) -> dict:
    return compute_usage_insights(csv_path)


def bootstrap_section(default_parent: str, default_subba: str) -> bool:
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
            st.info("Please click 'Rerun' in Streamlit or refresh the page.")
        except Exception as exc:
            st.error(f"Setup failed: {exc}")
        return True

    st.stop()


def main():
    init_db()

    st.title("Energy Forecasting Dashboard")
    st.write("Forecasting, backtesting, model comparison, persistence, retraining, and demand insights.")

    with st.sidebar:
        st.header("Controls")
        csv_path = st.text_input("CSV path", DATA_PATH)
        backtest_hours = st.selectbox("Backtest window (hours)", [24, 48, 72, 168, 336], index=3)
        show_7d_table_rows = st.selectbox("7-day table rows", [24, 48, 72, 168], index=1)
        parent_region = st.text_input("Retrain parent region", "CISO")
        subba = st.text_input("Retrain sub-region (optional)", "")

    if not Path(csv_path).exists():
        bootstrap_section(parent_region, subba)

    df = load_clean_data(csv_path)
    one_step = predict_next_from_csv(csv_path)
    insights = get_usage_insights(csv_path)
    summary = insights["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Latest actual load", f"{df['load_mw'].iloc[-1]:.2f}")
    c3.metric("One-step forecast", f"{one_step['predicted_load_mw']:.2f}")
    c4.metric("Last timestamp", str(df.index.max()))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Peak hour", f"{summary['peak_hour']}:00", f"{summary['peak_hour_avg_load_mw']:.1f} MW avg")
    s2.metric("Quietest hour", f"{summary['quiet_hour']}:00", f"{summary['quiet_hour_avg_load_mw']:.1f} MW avg")
    s3.metric("Busiest day", summary["busiest_day"], f"{summary['busiest_day_avg_load_mw']:.1f} MW avg")
    s4.metric("Busiest month", summary["busiest_month"], f"{summary['busiest_month_avg_load_mw']:.1f} MW avg")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Forecasts", "Usage patterns", "Backtesting", "Model comparison", "History & Ops", "Model details"]
    )

    with tab1:
        st.subheader("Recent actual load")
        st.line_chart(df.tail(168)["load_mw"])

        st.subheader("24-hour forecast with confidence interval")
        forecast_24h = get_forecast_24h(csv_path)
        forecast_24h["timestamp"] = pd.to_datetime(forecast_24h["timestamp"])
        forecast_24h = forecast_24h.set_index("timestamp")
        chart_24h = forecast_24h[["predicted_load_mw"]].copy()
        if "lower_bound_mw" in forecast_24h.columns:
            chart_24h["lower_bound_mw"] = forecast_24h["lower_bound_mw"]
            chart_24h["upper_bound_mw"] = forecast_24h["upper_bound_mw"]
        st.line_chart(chart_24h)
        st.dataframe(forecast_24h, use_container_width=True)

        csv_24h = forecast_24h.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download 24-hour forecast CSV", data=csv_24h, file_name="forecast_24h.csv", mime="text/csv")

        if st.button("Save current 24-hour forecast to history"):
            saved = forecast_next_24_hours_from_csv(csv_path, persist=True, source="dashboard")
            st.success(f"Saved {len(saved)} forecast points to history.")

        st.subheader("7-day forecast with confidence interval")
        forecast_7d = get_forecast_7d(csv_path)
        forecast_7d["timestamp"] = pd.to_datetime(forecast_7d["timestamp"])
        forecast_7d = forecast_7d.set_index("timestamp")
        chart_7d = forecast_7d[["predicted_load_mw"]].copy()
        if "lower_bound_mw" in forecast_7d.columns:
            chart_7d["lower_bound_mw"] = forecast_7d["lower_bound_mw"]
            chart_7d["upper_bound_mw"] = forecast_7d["upper_bound_mw"]
        st.line_chart(chart_7d)
        st.dataframe(forecast_7d.head(show_7d_table_rows), use_container_width=True)

        csv_7d = forecast_7d.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download 7-day forecast CSV", data=csv_7d, file_name="forecast_7d.csv", mime="text/csv")

        if st.button("Save current 7-day forecast to history"):
            saved = forecast_next_7_days_from_csv(csv_path, persist=True, source="dashboard")
            st.success(f"Saved {len(saved)} forecast points to history.")

    with tab2:
        st.subheader("Average energy use by hour")
        st.bar_chart(insights["hourly_avg"].set_index("hour")["avg_load_mw"])

        st.subheader("Average energy use by weekday")
        st.bar_chart(insights["weekday_avg"].set_index("day_name")["avg_load_mw"])

        st.subheader("Average energy use by month")
        st.bar_chart(insights["monthly_avg"].set_index("month_name")["avg_load_mw"])

        st.subheader("Daily peak load")
        daily_peak = insights["daily_peak"].copy()
        daily_peak["date"] = pd.to_datetime(daily_peak["date"])
        st.line_chart(daily_peak.set_index("date")["daily_peak_load_mw"])

        st.subheader("Day-hour usage heatmap table")
        st.dataframe(insights["hourly_heatmap"], use_container_width=True)

        st.subheader("Top 10 peak demand timestamps")
        top_peaks = insights["top_10_peaks"].copy()
        st.dataframe(top_peaks, use_container_width=True)

        csv_peaks = top_peaks.to_csv(index=False).encode("utf-8")
        st.download_button("Download top peak timestamps CSV", data=csv_peaks, file_name="top_10_peaks.csv", mime="text/csv")

    with tab3:
        st.subheader("Backtest")
        backtest = get_backtest(csv_path, backtest_hours)
        backtest_metrics = backtest["metrics"]
        backtest_df = pd.DataFrame(backtest["results"])
        backtest_df["timestamp"] = pd.to_datetime(backtest_df["timestamp"])
        backtest_df = backtest_df.set_index("timestamp")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Backtest MAE", f"{backtest_metrics['mae']:.2f}")
        b2.metric("Backtest RMSE", f"{backtest_metrics['rmse']:.2f}")
        b3.metric("Backtest R²", f"{backtest_metrics['r2']:.3f}")
        b4.metric("Backtest MAPE", f"{backtest_metrics['mape']:.2f}%")

        st.write("Actual vs predicted on backtest window")
        st.line_chart(backtest_df[["actual_load_mw", "predicted_load_mw"]])

        st.write("Absolute error on backtest window")
        st.line_chart(backtest_df["abs_error"])

        st.dataframe(backtest_df, use_container_width=True)

        csv_backtest = backtest_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button("Download backtest CSV", data=csv_backtest, file_name="backtest_results.csv", mime="text/csv")

    with tab4:
        st.subheader("Model comparison")
        comparison_df = get_model_comparison(csv_path)
        st.dataframe(comparison_df, use_container_width=True)
        st.write("Lower RMSE / MAE is better")
        st.bar_chart(comparison_df.set_index("model")[["rmse", "mae"]])
        st.write("Higher R² is better")
        st.bar_chart(comparison_df.set_index("model")[["r2"]])

        csv_comparison = comparison_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download model comparison CSV", data=csv_comparison, file_name="model_comparison.csv", mime="text/csv")

    with tab5:
        st.subheader("Forecast history")
        forecast_runs = pd.DataFrame(get_forecast_runs(limit=20))
        if not forecast_runs.empty:
            st.dataframe(forecast_runs, use_container_width=True)
        else:
            st.info("No saved forecast runs yet.")

        st.subheader("Retraining history")
        retraining_runs = pd.DataFrame(get_retraining_runs(limit=20))
        if not retraining_runs.empty:
            st.dataframe(retraining_runs, use_container_width=True)
        else:
            st.info("No retraining runs yet.")

        st.subheader("Operations")
        if st.button("Run retraining pipeline now"):
            with st.spinner("Fetching data and retraining..."):
                result = run_retraining_pipeline(parent_region=parent_region, subba=subba or None)
                st.cache_data.clear()
                st.success("Retraining complete.")
                st.json(result)

    with tab6:
        if METRICS_PATH.exists():
            st.subheader("Saved training metrics")
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.json(metrics)

        st.subheader("Series summary")
        st.json(summary)

        if IMPORTANCE_PATH.exists():
            st.subheader("Feature importance")
            importance_df = pd.read_csv(IMPORTANCE_PATH)
            st.dataframe(importance_df, use_container_width=True)
            st.bar_chart(importance_df.head(15).set_index("feature")["importance"])


if __name__ == "__main__":
    main()
