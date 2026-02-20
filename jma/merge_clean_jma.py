#!/usr/bin/env python3
"""
Merge and clean JMA hourly CSV downloads into the canonical dataset.

Workflow
--------
1. Place CSV files downloaded from https://www.data.jma.go.jp/risk/obsdl/
    into jma/rawdata/ using any .csv filename.
    Preferred convention for monthly files: mikuni_hourly_8_YYYY_MM.csv
    (e.g., mikuni_hourly_8_2025_06.csv)
2. Run this script from anywhere:
       python jma/merge_clean_jma.py
3. The script:
    a. Parses every .csv in jma/rawdata/.
    b. Loads the existing jma/jma_mikuni_hourly_8.csv if it is already present
      (i.e., the previously committed clean dataset).
   c. Merges old + new, deduplicates by timestamp (newly parsed data wins).
    d. Writes the result back to jma/jma_mikuni_hourly_8.csv.

You only ever need to add the *new* monthly CSVs to rawdata/ — the script
extends the committed file in-place rather than re-processing everything.

Input file expectations
-----------------------
- Encoding: CP932 (Shift-JIS family) — as downloaded from JMA.
- Filename: any .csv (preferred: mikuni_hourly_8_YYYY_MM.csv)
- Station: Mikuni (三国), 8-field export.

Output columns (jma_mikuni_hourly_8.csv)
-----------------------------------
timestamp, snow_depth_cm, snowfall_1h_cm, temp_c, precip_1h_mm,
sun_1h_h, wind_speed_ms, weather_type, humidity_pct
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory = jma/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_HERE, "rawdata")
OUTPUT_FILE = os.path.join(_HERE, "jma_mikuni_hourly_8.csv")

OUTPUT_COLS = [
    "timestamp",
    "snow_depth_cm",
    "snowfall_1h_cm",
    "temp_c",
    "precip_1h_mm",
    "sun_1h_h",
    "wind_speed_ms",
    "weather_type",
    "humidity_pct",
]


# ---------------------------------------------------------------------------
# JMA CSV parsing helpers
# ---------------------------------------------------------------------------

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


def _detect_header_index(rows: list[list[str]]) -> int:
    for idx, row in enumerate(rows):
        if row and row[0] == "年月日時":
            return idx
    raise ValueError("Could not find header row (年月日時).")


def _find_primary_col(header: list[str], quality_row: list[str], label: str) -> int:
    candidates = [
        i for i, h in enumerate(header)
        if h == label and i < len(quality_row) and quality_row[i] == ""
    ]
    if not candidates:
        raise ValueError(f"Could not find primary value column for: {label}")
    return candidates[0]


def _build_column_map(rows: list[list[str]], header_idx: int) -> ColumnMap:
    header = rows[header_idx]
    supplemental = rows[header_idx + 1] if len(rows) > header_idx + 1 else [""] * len(header)
    quality_row = rows[header_idx + 2] if len(rows) > header_idx + 2 else [""] * len(header)

    timestamp     = header.index("年月日時")
    snow_depth_cm  = _find_primary_col(header, quality_row, "積雪(cm)")
    snowfall_1h_cm = _find_primary_col(header, quality_row, "降雪(cm)")
    temp_c         = _find_primary_col(header, quality_row, "気温(℃)")
    precip_1h_mm   = _find_primary_col(header, quality_row, "降水量(mm)")
    sun_1h_h       = _find_primary_col(header, quality_row, "日照時間(時間)")
    weather_type   = _find_primary_col(header, quality_row, "天気")
    humidity_pct   = _find_primary_col(header, quality_row, "相対湿度(％)")

    wind_candidates = [
        i for i, h in enumerate(header)
        if h == "風速(m/s)" and i < len(quality_row) and quality_row[i] == ""
    ]
    if not wind_candidates:
        raise ValueError("Could not find wind speed column.")

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


def _to_float(value: str):
    text = (value or "").strip()
    if text == "":
        return pd.NA
    try:
        return float(text)
    except ValueError:
        return pd.NA


def _parse_raw_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="cp932", errors="replace", newline="") as f:
        rows = list(csv.reader(f))

    header_idx = _detect_header_index(rows)
    col_map    = _build_column_map(rows, header_idx)
    data_start = header_idx + 3

    records = []
    for row in rows[data_start:]:
        if not row or len(row) <= col_map.timestamp:
            continue
        ts_raw = row[col_map.timestamp].strip() if col_map.timestamp < len(row) else ""
        if not ts_raw:
            continue

        def _cell(idx: int) -> str:
            return row[idx] if idx < len(row) else ""

        records.append({
            "timestamp":      ts_raw,
            "snow_depth_cm":  _to_float(_cell(col_map.snow_depth_cm)),
            "snowfall_1h_cm": _to_float(_cell(col_map.snowfall_1h_cm)),
            "temp_c":         _to_float(_cell(col_map.temp_c)),
            "precip_1h_mm":   _to_float(_cell(col_map.precip_1h_mm)),
            "sun_1h_h":       _to_float(_cell(col_map.sun_1h_h)),
            "wind_speed_ms":  _to_float(_cell(col_map.wind_speed_ms)),
            "weather_type":   _cell(col_map.weather_type).strip() or pd.NA,
            "humidity_pct":   _to_float(_cell(col_map.humidity_pct)),
            "source_file":    os.path.basename(path),
        })

    if not records:
        return pd.DataFrame(columns=OUTPUT_COLS + ["source_file"])

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Parse new raw files
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not raw_files:
        print(f"No raw input files found in {RAW_DIR}/")
        print("Add any .csv files downloaded from JMA ObsDL and re-run.")
        return

    print(f"Parsing {len(raw_files)} raw file(s) from rawdata/ …")
    frames: list[pd.DataFrame] = []
    for path in raw_files:
        df = _parse_raw_file(path)
        print(f"  {os.path.basename(path):20s}  {len(df):5d} rows")
        frames.append(df)

    new_data = pd.concat(frames, ignore_index=True)
    # Within newly parsed batch: sort by (timestamp, source_file), keep last
    new_data = (
        new_data
        .sort_values(["timestamp", "source_file"])
        .drop_duplicates(subset=["timestamp"], keep="last")
    )

    # 2. Load existing committed dataset (if present) for upsert
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["timestamp"])
        print(f"\nExisting dataset: {len(existing):,} rows")
        # Drop rows whose timestamps are already covered by new data (new wins)
        existing = existing[~existing["timestamp"].isin(new_data["timestamp"])]
        combined = pd.concat(
            [existing[OUTPUT_COLS], new_data[OUTPUT_COLS]],
            ignore_index=True,
        )
    else:
        print("\nNo existing dataset found — creating from scratch.")
        combined = new_data[OUTPUT_COLS].copy()

    # 3. Sort, final dedup, write
    combined = (
        combined
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    start = combined["timestamp"].min().strftime("%Y-%m-%d")
    end   = combined["timestamp"].max().strftime("%Y-%m-%d")

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone — wrote {len(combined):,} rows  [{start} → {end}]")
    print(f"  → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
