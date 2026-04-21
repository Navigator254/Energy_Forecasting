
# Energy Forecasting System

An end-to-end electricity demand forecasting project with:

- EIA data ingestion
- data cleaning and feature engineering
- model training
- one-step prediction
- 24-hour forecasting
- 7-day forecasting
- backtesting
- model comparison
- interactive Streamlit dashboard
- FastAPI inference service
- Docker support

---

## Project Highlights

This project can:

- fetch and clean hourly energy demand data
- train a forecasting model
- generate future forecasts for the next 24 hours and 7 days
- compare multiple models
- backtest forecast quality on historical data
- surface insights such as busiest hours, busiest days, monthly patterns, and peak demand timestamps
- expose all functionality through an API and a dashboard

---

## Main Features

### Forecasting
- One-step prediction
- 24-hour recursive forecast
- 7-day recursive forecast

### Evaluation
- Backtesting over configurable windows
- Metrics:
  - MAE
  - RMSE
  - R²
  - MAPE

### Model Comparison
- Random Forest
- Linear Regression
- HistGradientBoosting

### Dashboard Insights
- recent actual load
- 24-hour forecast
- 7-day forecast
- actual vs forecast overlay
- average usage by hour
- average usage by weekday
- average usage by month
- top 10 peak timestamps
- daily peak demand
- backtest charts
- model comparison charts
- feature importance

### API Endpoints
- `/predict`
- `/predict-from-csv`
- `/forecast-24h`
- `/forecast-7d`
- `/backtest`
- `/compare-models`
- `/usage-insights`

---

## Project Structure

```text
energy_forecasting_system/
├── app.py
├── dashboard.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.docker.txt
├── README.md
├── .env.example
├── .dockerignore
├── data/
├── models/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── services/
│   └── utils/
└── tests/
