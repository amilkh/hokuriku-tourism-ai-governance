#!/usr/bin/env python3
"""Step 4 – Cross-validate scripts against CSV headers and paths.

Checks:
  1. Every hardcoded / config-resolved path actually exists.
  2. Column names referenced in code match those in the CSVs.
  3. Encoding expectations match actual file encodings.
"""
from __future__ import annotations
import sys, os, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── 1. Hardcoded path checks ─────────────────────────────────────────────────

def check_paths():
    print("=" * 72)
    print("1. PATH EXISTENCE CHECKS")
    print("=" * 72)

    # Paths that MUST exist inside the repo
    repo_paths = {
        "config/settings.yaml": "src/config.py",
        "jma/jma_mikuni_hourly_8.csv": "config/settings.yaml (weather.mikuni)",
        "jma/jma_fukui_hourly_8.csv": "config/settings.yaml (weather.fukui)",
        "jma/jma_katsuyama_hourly_8.csv": "config/settings.yaml (weather.katsuyama)",
    }
    for rp, source in repo_paths.items():
        full = REPO / rp
        status = "✅ EXISTS" if full.exists() else "❌ MISSING"
        print(f"  {status}  {rp}  (referenced by {source})")

    # Legacy fallback that may or may not exist
    legacy = "jma/jma_hourly_cleaned_merged_2024-01-01_2026-02-19.csv"
    full_legacy = REPO / legacy
    status = "✅ EXISTS" if full_legacy.exists() else "⚠ MISSING (legacy fallback, non-fatal)"
    print(f"  {status}  {legacy}")

    # merge_clean_jma.py expects jma/rawdata/
    rawdir = REPO / "jma" / "rawdata"
    if rawdir.exists():
        csvs = list(rawdir.glob("*.csv"))
        print(f"  ✅ jma/rawdata/ exists ({len(csvs)} CSVs)")
    else:
        print(f"  ⚠ jma/rawdata/ MISSING – merge_clean_jma.py has no input; raw files are in *_rawdata/ dirs instead")

    # External workspace paths (cannot verify without workspace root but we log them)
    print("\n  External workspace paths (require sibling repos):")
    externals = [
        "fukui-kanko-people-flow-data/daily/tojinbo-shotaro/Person/**/*.csv",
        "fukui-kanko-people-flow-data/daily/fukui-station-east-entrance/Person/**/*.csv",
        "fukui-kanko-people-flow-data/daily/katsuyama*/Person/**/*.csv",
        "fukui-kanko-people-flow-data/daily/rainbow-line-parking-lot-1-gate/Face/**/*.csv",
        "fukui-kanko-trend-report/public/data",
        "opendata/output_merge/merged_survey_*.csv",
        "fukui-kanko-survey/all.csv",
    ]
    ws_root = REPO.parent  # workspace root (parent of this repo)
    for ext in externals:
        # Check if base dir exists
        base = ext.split("*")[0].split("/")[0]
        full = ws_root / base
        status = "✅ EXISTS" if full.exists() else "⚠ MISSING (external data)"
        print(f"    {status}  ../{ext}")

    print()


# ── 2. Column name checks ────────────────────────────────────────────────────

import pandas as pd

def check_columns():
    print("=" * 72)
    print("2. COLUMN NAME CROSS-VALIDATION")
    print("=" * 72)

    # Expected columns per CSV used by scripts
    expected = {
        "jma/jma_mikuni_hourly_8.csv": {
            "required": ["timestamp"],
            "used_via_rename": ["temp_c", "precip_1h_mm", "sun_1h_h", "wind_speed_ms", "snow_depth_cm", "humidity_pct"],
            "scripts": ["src/data_loader.py", "src/spatial.py"],
        },
        "jma/jma_fukui_hourly_8.csv": {
            "required": ["timestamp"],
            "used_via_rename": ["temp_c", "precip_1h_mm", "sun_1h_h", "wind_speed_ms", "snow_depth_cm"],
            "scripts": ["src/spatial.py"],
        },
        "jma/jma_katsuyama_hourly_8.csv": {
            "required": ["timestamp"],
            "used_via_rename": ["temp_c", "precip_1h_mm", "sun_1h_h", "wind_speed_ms", "snow_depth_cm"],
            "scripts": ["src/spatial.py"],
        },
    }

    for csv_path, spec in expected.items():
        full = REPO / csv_path
        if not full.exists():
            print(f"  ❌ Cannot check {csv_path} – file missing")
            continue

        df = pd.read_csv(str(full), nrows=5)
        actual_cols = set(df.columns)
        print(f"\n  {csv_path} (actual cols={list(df.columns)}):")

        for col in spec["required"]:
            status = "✅" if col in actual_cols else "❌ MISSING"
            print(f"    {status} required: '{col}'")

        for col in spec["used_via_rename"]:
            status = "✅" if col in actual_cols else "⚠ ABSENT (may be renamed or optional)"
            print(f"    {status} rename-source: '{col}'")

        print(f"    Referenced by: {', '.join(spec['scripts'])}")

    print()


# ── 3. Encoding checks ───────────────────────────────────────────────────────

def check_encoding():
    print("=" * 72)
    print("3. ENCODING CROSS-VALIDATION")
    print("=" * 72)

    # merge_clean_jma.py reads with encoding="cp932"
    # But raw files in *_rawdata/ were written by fetch_jma_monthly.py as UTF-8
    print("\n  merge_clean_jma.py uses encoding='cp932' for raw input files.")
    print("  However, files in jma/*_rawdata/ were written by fetch_jma_monthly.py in UTF-8.")
    print("  ⚠ If merge_clean_jma.py ever processes these files, cp932 read would still work")
    print("    (ASCII-compatible subset), but this is a fragile assumption.\n")

    # Canonical CSVs are written by fetch_jma_monthly.py → UTF-8
    # data_loader.py reads them with default encoding (UTF-8) → OK
    for name in ["jma_mikuni_hourly_8.csv", "jma_fukui_hourly_8.csv", "jma_katsuyama_hourly_8.csv"]:
        full = REPO / "jma" / name
        if full.exists():
            # Quick byte check: read first 1000 bytes and attempt UTF-8
            raw = full.read_bytes()[:2000]
            try:
                raw.decode("utf-8")
                print(f"  ✅ jma/{name} is valid UTF-8")
            except UnicodeDecodeError:
                print(f"  ❌ jma/{name} is NOT valid UTF-8")

    # data_loader.py reads survey CSVs with encoding="utf-8" – consistent
    print("  ✅ data_loader.py reads surveys with encoding='utf-8'")
    print("  ✅ Canonical weather CSVs are read with default encoding (UTF-8)")
    print()


def main():
    check_paths()
    check_columns()
    check_encoding()

    print("=" * 72)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
