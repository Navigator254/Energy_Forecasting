from pathlib import Path
import sqlite3
from datetime import datetime
from typing import Any


DB_PATH = Path("data/forecast_history.db")


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            horizon_hours INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            csv_path TEXT NOT NULL,
            point_count INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            predicted_load_mw REAL NOT NULL,
            lower_bound_mw REAL,
            upper_bound_mw REAL,
            FOREIGN KEY(run_id) REFERENCES forecast_runs(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS retraining_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            parent_region TEXT NOT NULL,
            subba TEXT,
            rows_in_dataset INTEGER NOT NULL,
            anomaly_count INTEGER NOT NULL,
            mae REAL NOT NULL,
            rmse REAL NOT NULL,
            r2 REAL NOT NULL,
            mape REAL NOT NULL,
            model_name TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_forecast_run(
    records: list[dict[str, Any]],
    horizon_hours: int,
    source: str = "api",
    csv_path: str = "data/PJME_hourly.csv",
    model_name: str = "RandomForest",
    db_path: Path = DB_PATH,
) -> int:
    init_db(db_path)
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO forecast_runs (created_at, source, horizon_hours, model_name, csv_path, point_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            source,
            horizon_hours,
            model_name,
            csv_path,
            len(records),
        ),
    )
    run_id = cur.lastrowid

    for row in records:
        cur.execute(
            """
            INSERT INTO forecast_points (run_id, timestamp, predicted_load_mw, lower_bound_mw, upper_bound_mw)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["timestamp"],
                float(row["predicted_load_mw"]),
                float(row["lower_bound_mw"]) if row.get("lower_bound_mw") is not None else None,
                float(row["upper_bound_mw"]) if row.get("upper_bound_mw") is not None else None,
            ),
        )

    conn.commit()
    conn.close()
    return int(run_id)


def get_forecast_runs(limit: int = 20, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, source, horizon_hours, model_name, csv_path, point_count
        FROM forecast_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_retraining_run(
    metrics: dict[str, float],
    parent_region: str,
    subba: str | None,
    rows_in_dataset: int,
    anomaly_count: int,
    model_name: str = "RandomForest",
    db_path: Path = DB_PATH,
) -> int:
    init_db(db_path)
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO retraining_runs
        (created_at, parent_region, subba, rows_in_dataset, anomaly_count, mae, rmse, r2, mape, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            parent_region,
            subba,
            rows_in_dataset,
            anomaly_count,
            float(metrics["mae"]),
            float(metrics["rmse"]),
            float(metrics["r2"]),
            float(metrics["mape"]),
            model_name,
        ),
    )

    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(run_id)


def get_retraining_runs(limit: int = 20, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, created_at, parent_region, subba, rows_in_dataset, anomaly_count,
               mae, rmse, r2, mape, model_name
        FROM retraining_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
