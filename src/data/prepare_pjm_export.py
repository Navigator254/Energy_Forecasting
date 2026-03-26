from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/PJME_hourly_raw.csv")
OUTPUT_PATH = Path("data/PJME_hourly.csv")


def find_column(columns, options):
    lower_map = {c.lower(): c for c in columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    for c in columns:
        for opt in options:
            if opt.lower() in c.lower():
                return c
    return None


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Raw export not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    datetime_col = find_column(
        df.columns,
        ["Datetime Beginning EPT", "datetime_beginning_ept", "Datetime"]
    )
    value_col = find_column(
        df.columns,
        ["MW", "mw", "load"]
    )

    if datetime_col is None:
        raise ValueError(f"Could not find datetime column in {list(df.columns)}")
    if value_col is None:
        raise ValueError(f"Could not find MW column in {list(df.columns)}")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    grouped = (
        df.groupby(datetime_col, as_index=False)[value_col]
        .sum()
        .sort_values(datetime_col)
    )

    grouped.columns = ["Datetime", "PJME_MW"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned training file to {OUTPUT_PATH}")
    print(grouped.head())


if __name__ == "__main__":
    main()