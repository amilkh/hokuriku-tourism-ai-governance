#!/usr/bin/env python3
"""
Fetch JMA historical hourly weather data by scraping HTML table pages.

JMA publishes hourly data as HTML tables at:
  - hourly_s1.php  (官署 main stations, 17 cols)
  - hourly_a1.php  (アメダス AMeDAS stations, 11 cols)

This script fetches one page per day, extracts the data table,
and writes one CSV per month matching the merge_clean_jma.py schema.

Builtin station presets (--station shorthand):
  mikuni      prec_no=57, block_no=47616, page=hourly_s1  (main station)
  fukui       prec_no=57, block_no=47631, page=hourly_s1  (main station)
  katsuyama   prec_no=57, block_no=1226,  page=hourly_a1  (AMeDAS)

Usage:
    # Fetch Katsuyama Dec 2025
    python jma/fetch_jma_monthly.py --station katsuyama --year 2025 --month 12

    # Fetch Fukui Jan-Jun 2026
    python jma/fetch_jma_monthly.py --station fukui --year 2026 \\
           --start-month 1 --end-month 6

    # Custom station (override block/page)
    python jma/fetch_jma_monthly.py --prec-no 57 --block-no 1226 \\
           --page hourly_a1 --station-name katsuyama --year 2025 --month 12

Output:  jma/<station>_rawdata/<station>_hourly_8_YYYY_MM.csv
Schema:  年月日時,降水量mm,気温℃,湿度％,風速m/s,風向,日照時間h,積雪cm,降雪cm
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import re
import time
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd

# ---------------------------------------------------------------------------
# Station presets
# ---------------------------------------------------------------------------
PRESETS = {
    "mikuni": {
        "prec_no": "57",
        "block_no": "1071",
        "page": "hourly_a1",
    },
    "fukui": {
        "prec_no": "57",
        "block_no": "47631",
        "page": "hourly_s1",
    },
    "katsuyama": {
        "prec_no": "57",
        "block_no": "1226",
        "page": "hourly_a1",
    },
}

BASE_URL = "https://www.data.jma.go.jp/obd/stats/etrn/view"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


# ---------------------------------------------------------------------------
# Column mapping  (HTML multi-index → flat output names)
# ---------------------------------------------------------------------------
# Output schema (matches merge_clean_jma.py):
#   timestamp, snow_depth_cm, snowfall_1h_cm, temp_c, precip_1h_mm,
#   sun_1h_h, wind_speed_ms, weather_type, humidity_pct

def _find_col(cols: list, *keywords: str, level: Optional[int] = None) -> Optional[int]:
    """Find column index matching keywords.
    
    If level is specified, only check that specific level of the multi-index.
    Otherwise, check all levels joined together.
    """
    for i, c in enumerate(cols):
        tup = c if isinstance(c, tuple) else (c,)
        if level is not None:
            if level < len(tup):
                text = str(tup[level])
            else:
                continue
        else:
            text = " ".join(str(x) for x in tup)
        if all(k in text for k in keywords):
            return i
    return None


def _extract_rows(df: pd.DataFrame, page_type: str) -> list[dict]:
    """Extract standardized rows from a parsed JMA HTML table DataFrame."""
    cols = df.columns.tolist()
    rows = []

    # Locate column indices by Japanese header keywords
    i_hour = _find_col(cols, "時")
    i_precip = _find_col(cols, "降水量")
    i_temp = _find_col(cols, "気温")
    i_humidity = _find_col(cols, "湿度")
    # Wind speed/direction share a parent header; match on sub-header (level=1)
    i_wind_speed = _find_col(cols, "風速", level=1)
    i_wind_dir = _find_col(cols, "風向", level=1)
    i_sun = _find_col(cols, "日照")
    i_snowfall = _find_col(cols, "降雪", level=1)
    i_snowdepth = _find_col(cols, "積雪", level=1)
    i_weather = _find_col(cols, "天気") if page_type == "hourly_s1" else None

    for _, row in df.iterrows():
        vals = row.values.tolist()

        def _clean(idx: Optional[int]) -> str:
            if idx is None or idx >= len(vals):
                return ""
            v = str(vals[idx]).strip()
            # JMA uses these markers for missing/unavailable data
            if v in ("--", "///", "×", "nan", "NaN", "#", ""):
                return ""
            # Strip trailing quality flags like ')' or ']'
            v = re.sub(r"[)\]]+$", "", v).strip()
            return v

        hour = _clean(i_hour)
        if not hour or not hour.isdigit():
            continue

        rows.append(
            {
                "hour": int(hour),
                "precip_1h_mm": _clean(i_precip),
                "temp_c": _clean(i_temp),
                "humidity_pct": _clean(i_humidity),
                "wind_speed_ms": _clean(i_wind_speed),
                "wind_dir": _clean(i_wind_dir),
                "sun_1h_h": _clean(i_sun),
                "snowfall_1h_cm": _clean(i_snowfall),
                "snow_depth_cm": _clean(i_snowdepth),
                "weather_type": _clean(i_weather) if i_weather is not None else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# HTTP fetch with retries
# ---------------------------------------------------------------------------
def _fetch_html(url: str, *, timeout: int = 30, retries: int = 3) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def fetch_day(
    *,
    prec_no: str,
    block_no: str,
    page: str,
    year: int,
    month: int,
    day: int,
    timeout: int = 30,
    retries: int = 3,
) -> list[dict]:
    """Fetch hourly data for a single day. Returns list of row dicts."""
    url = (
        f"{BASE_URL}/{page}.php?"
        f"prec_no={prec_no}&block_no={block_no}"
        f"&year={year}&month={month}&day={day}&view="
    )
    html = _fetch_html(url, timeout=timeout, retries=retries)

    if html.count("<td") < 10:
        return []  # page has no data table (e.g., future date)

    tables = pd.read_html(StringIO(html))
    # The largest table with ≥20 rows is the hourly data table
    data_tables = [t for t in tables if t.shape[0] >= 20]
    if not data_tables:
        return []

    return _extract_rows(data_tables[0], page)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch JMA hourly data by scraping HTML table pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--station",
        choices=list(PRESETS.keys()),
        help="Station preset (auto-sets prec-no, block-no, page)",
    )
    parser.add_argument("--prec-no", help="JMA prefecture code (e.g. 57)")
    parser.add_argument("--block-no", help="JMA block code (e.g. 47631, 1226)")
    parser.add_argument(
        "--page",
        choices=["hourly_s1", "hourly_a1"],
        help="hourly_s1 for main stations, hourly_a1 for AMeDAS",
    )
    parser.add_argument("--station-name", help="Name prefix for output files")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, help="Single month (shorthand for --start/end-month)")
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--output-dir", help="Output directory (default: jma/<station>_rawdata)")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds between day requests")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    # Resolve station preset
    if args.station:
        preset = PRESETS[args.station]
        prec_no = args.prec_no or preset["prec_no"]
        block_no = args.block_no or preset["block_no"]
        page = args.page or preset["page"]
        station_name = args.station_name or args.station
    else:
        if not all([args.prec_no, args.block_no, args.page]):
            parser.error("Provide --station OR all of --prec-no, --block-no, --page")
        prec_no = args.prec_no
        block_no = args.block_no
        page = args.page
        station_name = args.station_name or "jma"

    if args.month:
        start_month = args.month
        end_month = args.month
    else:
        start_month = args.start_month
        end_month = args.end_month

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        raise ValueError("months must be in 1..12")
    if start_month > end_month:
        raise ValueError("start-month must be <= end-month")

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"jma/{station_name}_rawdata")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Station: {station_name}  (prec={prec_no}, block={block_no}, {page})")
    print(f"Range: {args.year}-{start_month:02d} .. {args.year}-{end_month:02d}")
    print(f"Output: {output_dir}/")
    print()

    fetched = 0
    skipped = 0

    for month in range(start_month, end_month + 1):
        out_name = f"{station_name}_hourly_8_{args.year}_{month:02d}.csv"
        out_path = output_dir / out_name

        if out_path.exists() and not args.force:
            print(f"[skip] {out_name} (exists)")
            skipped += 1
            continue

        n_days = calendar.monthrange(args.year, month)[1]
        print(f"[fetch] {out_name}  ({n_days} days)", end="", flush=True)

        all_rows: list[dict] = []
        failed_days = 0
        for day in range(1, n_days + 1):
            try:
                day_rows = fetch_day(
                    prec_no=prec_no,
                    block_no=block_no,
                    page=page,
                    year=args.year,
                    month=month,
                    day=day,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                for r in day_rows:
                    hour = r.pop("hour")
                    if hour == 24:
                        # Hour 24 = midnight of the next day
                        dt = datetime.datetime(args.year, month, day) + datetime.timedelta(days=1)
                        r["timestamp"] = dt.strftime("%Y-%m-%d 00:00:00")
                    else:
                        r["timestamp"] = f"{args.year}-{month:02d}-{day:02d} {hour:02d}:00:00"
                all_rows.extend(day_rows)
                print(".", end="", flush=True)
            except KeyboardInterrupt:
                print(f"\n  interrupted at day {day}; saving partial data...")
                break
            except Exception as e:
                failed_days += 1
                print(f"x", end="", flush=True)
                if failed_days >= 5:
                    print(f"\n  too many failures ({failed_days}); saving partial data...")
                    break

            if day < n_days:
                time.sleep(max(args.sleep, 0.5))

        if not all_rows:
            print(f"\n  error: no data returned for {args.year}-{month:02d}")
            continue

        # Write CSV in canonical column order
        col_order = [
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
        df = pd.DataFrame(all_rows)
        # Ensure all columns exist
        for c in col_order:
            if c not in df.columns:
                df[c] = ""
        df = df[col_order]
        df.to_csv(out_path, index=False)
        fetched += 1
        print(f"  → {len(all_rows)} rows")

    print(f"\nDone. fetched={fetched}, skipped={skipped}")


if __name__ == "__main__":
    main()
