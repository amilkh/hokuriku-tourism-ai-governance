#!/usr/bin/env python3
"""
Merge and clean JMA hourly CSV downloads into canonical station datasets.

Standardized workflow
---------------------
1. Place JMA CSV downloads into station folders under `jma/`:
    - `jma/mikuni_rawdata/`
    - `jma/fukui_rawdata/`
    - `jma/katsuyama_rawdata/`
2. Run from anywhere:
        python jma/merge_clean_jma.py
    Outputs are regenerated locally (not tracked in git).
3. For each station, the script:
    a. Parses all `.csv` files in that station's raw folders.
    b. Loads existing canonical output CSV if present.
    c. Upserts by timestamp (newly parsed data wins on duplicates).
    d. Writes station output:
        - `jma/jma_mikuni_hourly_8.csv`
        - `jma/jma_fukui_hourly_8.csv`
        - `jma/jma_katsuyama_hourly_8.csv`

Input expectations
------------------
- Encoding: CP932 (Shift-JIS family), as downloaded from JMA ObsDL.
- Filename: any `.csv` (monthly naming recommended).
- Export fields: standard hourly 8-field package.

Output schema
-------------
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


@dataclass
class StationConfig:
    key: str
    output_file: str
    raw_dirs: list[str]


STATIONS: list[StationConfig] = [
    StationConfig(
        key="mikuni",
        output_file=os.path.join(_HERE, "jma_mikuni_hourly_8.csv"),
        raw_dirs=[
            os.path.join(_HERE, "mikuni_rawdata"),
        ],
    ),
    StationConfig(
        key="fukui",
        output_file=os.path.join(_HERE, "jma_fukui_hourly_8.csv"),
        raw_dirs=[
            os.path.join(_HERE, "fukui_rawdata"),
        ],
    ),
    StationConfig(
        key="katsuyama",
        output_file=os.path.join(_HERE, "jma_katsuyama_hourly_8.csv"),
        raw_dirs=[
            os.path.join(_HERE, "katsuyama_rawdata"),
        ],
    ),
]

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


def _is_pre_cleaned_csv(path: str) -> bool:
    """Check if a file is already in canonical clean format (from fetch_jma_monthly.py)."""
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            header = f.readline().strip()
        return header.startswith("timestamp,")
    except Exception:
        return False


def _parse_clean_file(path: str) -> pd.DataFrame:
    """Parse a pre-cleaned CSV (output of fetch_jma_monthly.py)."""
    df = pd.read_csv(path, dtype=str)
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=OUTPUT_COLS + ["source_file"])

    # Convert numeric columns
    for col in ["snow_depth_cm", "snowfall_1h_cm", "temp_c", "precip_1h_mm",
                "sun_1h_h", "wind_speed_ms", "humidity_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure all output columns exist
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["source_file"] = os.path.basename(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df[OUTPUT_COLS + ["source_file"]]


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

def _collect_station_raw_files(config: StationConfig) -> list[str]:
    files: list[str] = []
    for raw_dir in config.raw_dirs:
        if os.path.isdir(raw_dir):
            files.extend(sorted(glob.glob(os.path.join(raw_dir, "*.csv"))))
    return sorted(set(files))


def _merge_station(config: StationConfig) -> bool:
    raw_files = _collect_station_raw_files(config)
    if not raw_files:
        print(f"[{config.key}] No raw input files found. Checked:")
        for d in config.raw_dirs:
            print(f"  - {d}")
        return False

    print(f"\n[{config.key}] Parsing {len(raw_files)} raw file(s) …")
    frames: list[pd.DataFrame] = []
    for path in raw_files:
        if _is_pre_cleaned_csv(path):
            df = _parse_clean_file(path)
            fmt = "clean"
        else:
            df = _parse_raw_file(path)
            fmt = "JMA"
        print(f"  {os.path.basename(path):30s}  {len(df):5d} rows  ({fmt})")
        frames.append(df)

    new_data = pd.concat(frames, ignore_index=True)
    # Within newly parsed batch: sort by (timestamp, source_file), keep last
    new_data = (
        new_data
        .sort_values(["timestamp", "source_file"])
        .drop_duplicates(subset=["timestamp"], keep="last")
    )

    if os.path.exists(config.output_file):
        existing = pd.read_csv(config.output_file, parse_dates=["timestamp"])
        print(f"[{config.key}] Existing dataset: {len(existing):,} rows")
        # Drop rows whose timestamps are already covered by new data (new wins)
        existing = existing[~existing["timestamp"].isin(new_data["timestamp"])]
        combined = pd.concat(
            [existing[OUTPUT_COLS], new_data[OUTPUT_COLS]],
            ignore_index=True,
        )
    else:
        print(f"[{config.key}] No existing dataset found — creating from scratch.")
        combined = new_data[OUTPUT_COLS].copy()

    combined = (
        combined
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    start = combined["timestamp"].min().strftime("%Y-%m-%d")
    end   = combined["timestamp"].max().strftime("%Y-%m-%d")

    combined.to_csv(config.output_file, index=False)

    print(f"[{config.key}] Done — wrote {len(combined):,} rows  [{start} → {end}]")
    print(f"[{config.key}]   → {config.output_file}")
    return True


def main() -> None:
    processed = 0
    for station in STATIONS:
        ok = _merge_station(station)
        if ok:
            processed += 1

    if processed == 0:
        print("\nNo station raw files found. Add CSVs and re-run.")
        print("Expected folders:")
        for station in STATIONS:
            for d in station.raw_dirs:
                print(f"  - {d}")


if __name__ == "__main__":
    main()
