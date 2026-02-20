#!/usr/bin/env python3
"""
Merge and clean JMA hourly CSV downloads (Mikuni, 8 selected fields).

Input:
  - CSV files downloaded from https://www.data.jma.go.jp/risk/obsdl/
  - Expected filename pattern: data*.csv (e.g., data.csv, data (1).csv ...)
  - Encoding: CP932 (Shift-JIS family)

Output:
  - jma_hourly_cleaned_merged_8fields_<start>_<end>.csv
  - jma_hourly_cleaned_merged_8fields_with_wind_direction_<start>_<end>.csv
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnMap:
    timestamp: int
    snow_depth_cm: int
    snowfall_1h_cm: int
    temp_c: int
    precip_1h_mm: int
    sun_1h_h: int
    wind_speed_ms: int
    weather_type: int
    humidity_pct: int
    wind_dir: int


def detect_header_index(rows: list[list[str]]) -> int:
    for idx, row in enumerate(rows):
        if row and row[0] == "年月日時":
            return idx
    raise ValueError("Could not find header row (年月日時).")


def find_primary_value_col(header: list[str], quality_row: list[str], label: str) -> int:
    candidates = [
        i for i, h in enumerate(header)
        if h == label and i < len(quality_row) and quality_row[i] == ""
    ]
    if not candidates:
        raise ValueError(f"Could not find primary value column for: {label}")
    return candidates[0]


def build_column_map(rows: list[list[str]], header_idx: int) -> ColumnMap:
    header = rows[header_idx]
    supplemental = rows[header_idx + 1] if len(rows) > header_idx + 1 else [""] * len(header)
    quality_row = rows[header_idx + 2] if len(rows) > header_idx + 2 else [""] * len(header)

    timestamp = header.index("年月日時")
    snow_depth_cm = find_primary_value_col(header, quality_row, "積雪(cm)")
    snowfall_1h_cm = find_primary_value_col(header, quality_row, "降雪(cm)")
    temp_c = find_primary_value_col(header, quality_row, "気温(℃)")
    precip_1h_mm = find_primary_value_col(header, quality_row, "降水量(mm)")
    sun_1h_h = find_primary_value_col(header, quality_row, "日照時間(時間)")
    weather_type = find_primary_value_col(header, quality_row, "天気")
    humidity_pct = find_primary_value_col(header, quality_row, "相対湿度(％)")

    wind_candidates = [
        i for i, h in enumerate(header)
        if h == "風速(m/s)" and i < len(quality_row) and quality_row[i] == ""
    ]
    if not wind_candidates:
        raise ValueError("Could not find wind columns.")

    # JMA layout for this export usually has:
    # - wind speed at supplemental blank column
    # - wind direction at supplemental == '風向'
    wind_speed_ms = None
    wind_dir = None
    for idx in wind_candidates:
        sup = supplemental[idx] if idx < len(supplemental) else ""
        if sup == "風向":
            wind_dir = idx
        elif wind_speed_ms is None:
            wind_speed_ms = idx

    if wind_speed_ms is None:
        wind_speed_ms = wind_candidates[0]
    if wind_dir is None:
        wind_dir = wind_candidates[-1]

    return ColumnMap(
        timestamp=timestamp,
        snow_depth_cm=snow_depth_cm,
        snowfall_1h_cm=snowfall_1h_cm,
        temp_c=temp_c,
        precip_1h_mm=precip_1h_mm,
        sun_1h_h=sun_1h_h,
        wind_speed_ms=wind_speed_ms,
        weather_type=weather_type,
        humidity_pct=humidity_pct,
        wind_dir=wind_dir,
    )


def to_float(value: str):
    text = (value or "").strip()
    if text == "":
        return pd.NA
    try:
        return float(text)
    except ValueError:
        return pd.NA


def parse_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="cp932", errors="replace", newline="") as f:
        rows = list(csv.reader(f))

    header_idx = detect_header_index(rows)
    col_map = build_column_map(rows, header_idx)
    data_start = header_idx + 3

    records = []
    for row in rows[data_start:]:
        if not row or len(row) <= col_map.timestamp:
            continue

        ts_raw = row[col_map.timestamp].strip() if col_map.timestamp < len(row) else ""
        if not ts_raw:
            continue

        records.append(
            {
                "timestamp": ts_raw,
                "snow_depth_cm": to_float(row[col_map.snow_depth_cm] if col_map.snow_depth_cm < len(row) else ""),
                "snowfall_1h_cm": to_float(row[col_map.snowfall_1h_cm] if col_map.snowfall_1h_cm < len(row) else ""),
                "temp_c": to_float(row[col_map.temp_c] if col_map.temp_c < len(row) else ""),
                "precip_1h_mm": to_float(row[col_map.precip_1h_mm] if col_map.precip_1h_mm < len(row) else ""),
                "sun_1h_h": to_float(row[col_map.sun_1h_h] if col_map.sun_1h_h < len(row) else ""),
                "wind_speed_ms": to_float(row[col_map.wind_speed_ms] if col_map.wind_speed_ms < len(row) else ""),
                "weather_type": (row[col_map.weather_type].strip() if col_map.weather_type < len(row) else "") or pd.NA,
                "humidity_pct": to_float(row[col_map.humidity_pct] if col_map.humidity_pct < len(row) else ""),
                "wind_dir": (row[col_map.wind_dir].strip() if col_map.wind_dir < len(row) else "") or pd.NA,
                "source_file": os.path.basename(path),
            }
        )

    if not records:
        return pd.DataFrame(columns=[
            "timestamp", "snow_depth_cm", "snowfall_1h_cm", "temp_c", "precip_1h_mm",
            "sun_1h_h", "wind_speed_ms", "weather_type", "humidity_pct", "wind_dir", "source_file",
        ])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_files = sorted(glob.glob(os.path.join(script_dir, "data*.csv")))

    if not input_files:
        raise FileNotFoundError("No input files found. Expected files like data.csv / data (1).csv in this folder.")

    print(f"Found {len(input_files)} input files")

    frames = []
    for path in input_files:
        df = parse_file(path)
        print(f"  {os.path.basename(path):15s} -> {len(df):5d} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Sort and deduplicate by timestamp (keep last file encountered)
    merged = merged.sort_values(["timestamp", "source_file"]).drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if merged.empty:
        raise ValueError("Merged dataframe is empty after parsing.")

    start_date = merged["timestamp"].min().strftime("%Y-%m-%d")
    end_date = merged["timestamp"].max().strftime("%Y-%m-%d")

    out_core = merged[[
        "timestamp",
        "snow_depth_cm",
        "snowfall_1h_cm",
        "temp_c",
        "precip_1h_mm",
        "sun_1h_h",
        "wind_speed_ms",
        "weather_type",
        "humidity_pct",
    ]].copy()

    out_with_dir = merged[[
        "timestamp",
        "snow_depth_cm",
        "snowfall_1h_cm",
        "temp_c",
        "precip_1h_mm",
        "sun_1h_h",
        "wind_speed_ms",
        "wind_dir",
        "weather_type",
        "humidity_pct",
    ]].copy()

    out_core_path = os.path.join(
        script_dir,
        f"jma_hourly_cleaned_merged_8fields_{start_date}_{end_date}.csv",
    )
    out_with_dir_path = os.path.join(
        script_dir,
        f"jma_hourly_cleaned_merged_8fields_with_wind_direction_{start_date}_{end_date}.csv",
    )

    out_core.to_csv(out_core_path, index=False)
    out_with_dir.to_csv(out_with_dir_path, index=False)

    print("\nDone")
    print(f"  Rows:          {len(merged):,}")
    print(f"  Date range:    {start_date} -> {end_date}")
    print(f"  Output (8f):   {out_core_path}")
    print(f"  Output (+dir): {out_with_dir_path}")


if __name__ == "__main__":
    main()
