#!/usr/bin/env python3
"""Step 3 – Deep CSV quality profiler for repository audit.

Profiles every CSV in the repo (excluding venvs) and produces a
structured report covering shape, dtypes, nulls, duplicates, outliers,
near-zero-variance columns, date-format consistency, and encoding.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", ".venv_kansei", "__pycache__", "node_modules"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_csvs(root: Path) -> list[Path]:
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(".csv"):
                results.append(Path(dirpath) / f)
    return sorted(results)


def try_read_csv(path: Path) -> tuple[pd.DataFrame | None, str]:
    """Try UTF-8 first, then CP932.  Returns (df, encoding_used)."""
    for enc in ("utf-8", "cp932", "latin-1"):
        try:
            df = pd.read_csv(str(path), encoding=enc, low_memory=False)
            return df, enc
        except Exception:
            continue
    return None, "UNREADABLE"


def detect_numeric_stored_as_string(df: pd.DataFrame) -> list[str]:
    """Return columns that look numeric but are stored as object/string."""
    suspects = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(200)
        if sample.empty:
            continue
        converted = pd.to_numeric(sample, errors="coerce")
        pct_numeric = converted.notna().sum() / len(sample)
        if pct_numeric > 0.8:
            suspects.append(col)
    return suspects


def check_date_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect date-like columns and report their format consistency."""
    results = {}
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(100)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() > len(sample) * 0.8:
                # Check format consistency
                unique_formats = set()
                for v in sample.head(20):
                    v = str(v).strip()
                    if len(v) == 10 and "-" in v:
                        unique_formats.add("YYYY-MM-DD")
                    elif len(v) == 10 and "/" in v:
                        unique_formats.add("YYYY/MM/DD")
                    elif len(v) >= 19:
                        unique_formats.add("datetime")
                    else:
                        unique_formats.add("other")
                results[col] = ", ".join(sorted(unique_formats))
        elif "datetime" in str(df[col].dtype):
            results[col] = str(df[col].dtype)
    return results


def detect_outliers_iqr(series: pd.Series) -> tuple[int, float, float]:
    """IQR outlier count, low fence, high fence for numeric series."""
    clean = series.dropna()
    if len(clean) < 10:
        return 0, float("nan"), float("nan")
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, q1, q3
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n_out = int(((clean < low) | (clean > high)).sum())
    return n_out, low, high


def near_zero_variance(df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """Columns where the most frequent value accounts for >99% of rows."""
    nzv = []
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            nzv.append(col)
        else:
            top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > (1 - threshold):
                nzv.append(col)
    return nzv


# ── Main profiler ────────────────────────────────────────────────────────────

def profile_csv(path: Path, relative_to: Path) -> dict:
    rel = path.relative_to(relative_to)
    df, enc = try_read_csv(path)
    if df is None:
        return {"file": str(rel), "error": "UNREADABLE", "encoding": enc}

    n_rows, n_cols = df.shape
    dup_rows = int(df.duplicated().sum())

    # Per-column null analysis
    null_info = {}
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        pct = round(n_null / n_rows * 100, 2) if n_rows else 0
        null_info[col] = {"null_count": n_null, "null_pct": pct, "dtype": str(df[col].dtype)}

    # Numeric columns – outlier + negative checks
    numeric_issues = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        n_out, lo, hi = detect_outliers_iqr(df[col])
        n_neg = int((df[col].dropna() < 0).sum())
        numeric_issues[col] = {
            "outliers": n_out,
            "fence_lo": round(lo, 2) if not np.isnan(lo) else None,
            "fence_hi": round(hi, 2) if not np.isnan(hi) else None,
            "min": round(float(df[col].min()), 4) if df[col].notna().any() else None,
            "max": round(float(df[col].max()), 4) if df[col].notna().any() else None,
            "negatives": n_neg,
        }

    nums_as_str = detect_numeric_stored_as_string(df)
    dates = check_date_columns(df)
    nzv = near_zero_variance(df)

    return {
        "file": str(rel),
        "encoding": enc,
        "rows": n_rows,
        "cols": n_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "duplicate_rows": dup_rows,
        "nulls": null_info,
        "numeric_issues": numeric_issues,
        "numeric_as_string": nums_as_str,
        "date_columns": dates,
        "near_zero_variance": nzv,
    }


def print_report(profiles: list[dict]) -> None:
    sep = "=" * 80
    for p in profiles:
        print(f"\n{'━' * 80}")
        print(f"FILE: {p['file']}")
        print(f"{'━' * 80}")
        if "error" in p:
            print(f"  ❌ ERROR: {p['error']}  (encoding tried: {p['encoding']})")
            continue

        print(f"  Encoding: {p['encoding']}")
        print(f"  Shape: {p['rows']} rows × {p['cols']} cols")
        print(f"  Columns: {p['columns']}")
        print(f"  Duplicate rows: {p['duplicate_rows']}")

        # Nulls
        has_nulls = any(v["null_count"] > 0 for v in p["nulls"].values())
        if has_nulls:
            print(f"  Missing values:")
            for col, info in p["nulls"].items():
                if info["null_count"] > 0:
                    print(f"    {col:30s}  {info['null_count']:>6d} ({info['null_pct']:>6.2f}%)  dtype={info['dtype']}")
        else:
            print(f"  Missing values: NONE")

        # Numeric issues
        if p["numeric_issues"]:
            print(f"  Numeric column stats:")
            for col, info in p["numeric_issues"].items():
                parts = [f"range=[{info['min']}, {info['max']}]"]
                if info["outliers"]:
                    parts.append(f"outliers={info['outliers']}")
                if info["negatives"]:
                    parts.append(f"negatives={info['negatives']}")
                print(f"    {col:30s}  {', '.join(parts)}")

        # Numbers stored as strings
        if p["numeric_as_string"]:
            print(f"  ⚠ Numbers stored as strings: {p['numeric_as_string']}")

        # Date columns
        if p["date_columns"]:
            print(f"  Date columns: {p['date_columns']}")

        # Near-zero variance
        if p["near_zero_variance"]:
            print(f"  ⚠ Near-zero variance columns: {p['near_zero_variance']}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("SUMMARY")
    print(f"{'═' * 80}")
    total = len(profiles)
    errors = sum(1 for p in profiles if "error" in p)
    with_dups = sum(1 for p in profiles if p.get("duplicate_rows", 0) > 0)
    with_nulls = sum(1 for p in profiles
                     if any(v.get("null_count", 0) > 0 for v in p.get("nulls", {}).values()))
    with_nzv = sum(1 for p in profiles if p.get("near_zero_variance"))
    with_numstr = sum(1 for p in profiles if p.get("numeric_as_string"))
    print(f"  Total CSVs profiled: {total}")
    print(f"  Unreadable:          {errors}")
    print(f"  With duplicate rows: {with_dups}")
    print(f"  With null values:    {with_nulls}")
    print(f"  With near-zero var:  {with_nzv}")
    print(f"  Nums-as-strings:     {with_numstr}")


def main():
    csvs = find_csvs(REPO)
    print(f"Found {len(csvs)} CSV files to profile.\n")
    profiles = [profile_csv(p, REPO) for p in csvs]
    print_report(profiles)


if __name__ == "__main__":
    main()
