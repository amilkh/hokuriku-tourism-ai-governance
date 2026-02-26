"""Data loading for camera, weather, RSI Intent, and survey sources.

Every loader returns a clean ``pandas.DataFrame`` with a ``date`` column
(``datetime64[ns]``, normalised to midnight).  Column names follow the
pipeline convention (``count``, ``temp``, ``precip``, ``wind``, etc.).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .report import Reporter


# ── Camera (AI people-flow) ──────────────────────────────────────────────────

def load_camera_daily(
    glob_pattern: str,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Load AI-camera CSV files and aggregate to daily visitor counts.

    Args:
        glob_pattern: Recursive glob pattern pointing to per-5-min CSVs.
        reporter: Optional ``Reporter`` for logging.

    Returns:
        DataFrame with columns ``[date, count]`` sorted by date.
        Zero-count days (sensor outage) are **removed**.
    """
    rpt = reporter.log if reporter else print

    files = sorted(glob.glob(glob_pattern, recursive=True))
    rows: list[dict[str, Any]] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "aggregate from" in df.columns and "total count" in df.columns:
                rows.append({
                    "date": os.path.basename(f).replace(".csv", ""),
                    "count": df["total count"].sum(),
                })
        except Exception:
            pass

    camera = pd.DataFrame(rows)
    if camera.empty:
        rpt("WARNING: No camera data loaded.")
        return pd.DataFrame(columns=["date", "count"])

    camera["date"] = pd.to_datetime(camera["date"])
    camera = camera.sort_values("date").reset_index(drop=True)

    zero_days = camera[camera["count"] == 0]
    rpt(f"Total camera days: {len(camera)}")
    rpt(f"Zero-count days (camera outage): {len(zero_days)}")
    if len(zero_days) > 0:
        rpt(f"  Dates: {', '.join(zero_days['date'].dt.strftime('%Y-%m-%d').tolist())}")

    camera = camera[camera["count"] > 0].reset_index(drop=True)
    rpt(f"Usable camera days after removing zeros: {len(camera)}")
    return camera


# ── JMA Weather ──────────────────────────────────────────────────────────────

def load_weather_daily(
    primary_path: str | Path,
    legacy_path: str | Path | None = None,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Load JMA hourly weather and aggregate to daily means/sums.

    Automatically normalises the 8-field schema (``temp_c`` → ``temp``,
    ``precip_1h_mm`` → ``precip``, etc.).

    Args:
        primary_path: Path to the preferred JMA hourly CSV.
        legacy_path: Fallback path if ``primary_path`` does not exist.
        reporter: Optional ``Reporter`` for logging.

    Returns:
        DataFrame ``[date, precip, temp, sun, wind]`` (and optionally
        ``snow_depth``, ``humidity`` when available).
    """
    rpt = reporter.log if reporter else print

    path = Path(primary_path)
    if not path.exists() and legacy_path:
        path = Path(legacy_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No JMA weather file found at {primary_path}"
            + (f" or {legacy_path}" if legacy_path else "")
        )

    weather = pd.read_csv(str(path), parse_dates=["timestamp"], encoding="utf-8")
    rpt(f"Using JMA weather file: {path}")

    # Normalise 8-field schema to pipeline names
    renames: dict[str, str] = {
        "temp_c": "temp",
        "precip_1h_mm": "precip",
        "sun_1h_h": "sun",
        "wind_speed_ms": "wind",
        "snow_depth_cm": "snow_depth",
        "humidity_pct": "humidity",
    }
    for old, new in renames.items():
        if old in weather.columns and new not in weather.columns:
            weather[new] = pd.to_numeric(weather[old], errors="coerce")

    weather["date"] = weather["timestamp"].dt.normalize()

    agg_spec: dict[str, tuple[str, str]] = {
        "precip": ("precip", "sum"),
        "temp": ("temp", "mean"),
    }
    for col, func in [("sun", "mean"), ("wind", "mean"),
                       ("snow_depth", "mean"), ("humidity", "mean")]:
        if col in weather.columns:
            agg_spec[col] = (col, func)

    daily = weather.groupby("date").agg(**agg_spec).reset_index()
    rpt(f"Weather daily rows: {len(daily)}")
    return daily


# ── RSI Intent ────────────────────────────────────────────────────────────

def load_rsi_intent(
    trend_root: str | Path,
    *,
    reporter: Reporter | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load Route Search Impressions (RSI) daily metrics.

    Args:
        trend_root: Path to ``fukui-kanko-trend-report/public/data``.
        reporter: Optional ``Reporter``.

    Returns:
        Tuple of ``(rsi_df, route_col_name)``:

        - ``rsi_df``: DataFrame with ``date`` + all intent columns.
        - ``route_col_name``: Name of the best route-search column found.
    """
    rpt = reporter.log if reporter else print
    trend_root = Path(trend_root)

    frames: list[pd.DataFrame] = []
    if trend_root.is_dir():
        for year_dir in sorted(os.listdir(trend_root)):
            total_path = trend_root / year_dir / "total_daily_metrics.csv"
            if total_path.exists():
                frames.append(pd.read_csv(str(total_path), encoding="utf-8"))

    if not frames:
        raise FileNotFoundError(f"No RSI data CSV found in {trend_root}.")

    rsi = pd.concat(frames, ignore_index=True)

    if "date" in rsi.columns:
        rsi["date"] = pd.to_datetime(rsi["date"]).dt.normalize()
    else:
        rsi["date"] = pd.to_datetime(rsi.iloc[:, 0]).dt.normalize()

    rsi = rsi.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    rpt(f"RSI intent rows: {len(rsi)}")

    route_col: str | None = None
    for candidate in ("directions", "ルート検索", "route_searches"):
        if candidate in rsi.columns:
            route_col = candidate
            break
    if route_col is None:
        raise ValueError("No route search column in RSI data.")
    rpt(f"Using RSI intent column: '{route_col}'")

    return rsi, route_col


# ── Survey (multiple loaders) ────────────────────────────────────────────────

def load_survey_prefectures(
    glob_pattern: str,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Load merged survey CSVs with prefecture + date columns only.

    Args:
        glob_pattern: Glob for ``merged_survey_*.csv``.
        reporter: Optional ``Reporter``.

    Returns:
        DataFrame ``[prefecture, date]``.
    """
    rpt = reporter.log if reporter else print
    frames: list[pd.DataFrame] = []

    for path in sorted(glob.glob(glob_pattern)):
        try:
            sdf = pd.read_csv(path, encoding="utf-8", low_memory=False, usecols=[0, 1])
            sdf.columns = ["prefecture", "survey_date"]
            sdf["survey_date"] = pd.to_datetime(sdf["survey_date"], errors="coerce")
            sdf = sdf.dropna(subset=["survey_date"])
            sdf["date"] = sdf["survey_date"].dt.normalize()
            frames.append(sdf[["prefecture", "date"]])
        except Exception as exc:
            rpt(f"  Warning: Could not load {path}: {exc}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        rpt(f"Loaded {len(combined)} survey responses across all years")
        return combined
    rpt("WARNING: No survey data loaded.")
    return pd.DataFrame(columns=["prefecture", "date"])


def load_survey_satisfaction(
    glob_pattern: str,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Load survey CSVs with satisfaction / NPS columns.

    Args:
        glob_pattern: Glob for ``merged_survey_*.csv``.
        reporter: Optional ``Reporter``.

    Returns:
        DataFrame ``[prefecture, date, satisfaction, satisfaction_service,
        nps_raw]``.
    """
    rpt = reporter.log if reporter else print
    frames: list[pd.DataFrame] = []

    for path in sorted(glob.glob(glob_pattern)):
        try:
            sdf = pd.read_csv(path, encoding="utf-8", low_memory=False)
            sub = pd.DataFrame()
            sub["prefecture"] = sdf[sdf.columns[0]]
            sub["date"] = pd.to_datetime(sdf[sdf.columns[1]], errors="coerce").dt.normalize()

            sat_col = "満足度（旅行全体）"
            svc_col = "満足度（商品・サービス）"
            nps_col = "おすすめ度"

            sub["satisfaction"] = (
                pd.to_numeric(sdf[sat_col], errors="coerce")
                if sat_col in sdf.columns else np.nan
            )
            sub["satisfaction_service"] = (
                pd.to_numeric(sdf[svc_col], errors="coerce")
                if svc_col in sdf.columns else np.nan
            )
            if nps_col in sdf.columns:
                sub["nps_raw"] = pd.to_numeric(
                    sdf[nps_col].astype(str).str.extract(r"(\d+)", expand=False),
                    errors="coerce",
                )
            else:
                sub["nps_raw"] = np.nan

            sub = sub.dropna(subset=["date"])
            frames.append(sub)
        except Exception as exc:
            rpt(f"  Warning: {path}: {exc}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["prefecture", "date", "satisfaction",
                                  "satisfaction_service", "nps_raw"])


def load_survey_text(
    glob_pattern: str,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Load survey CSVs with free-text fields for text mining.

    Args:
        glob_pattern: Glob for ``merged_survey_*.csv``.
        reporter: Optional ``Reporter``.

    Returns:
        DataFrame ``[prefecture, date, satisfaction, reason, inconvenience,
        freetext]``.
    """
    rpt = reporter.log if reporter else print
    frames: list[pd.DataFrame] = []

    for path in sorted(glob.glob(glob_pattern)):
        try:
            sdf = pd.read_csv(path, encoding="utf-8", low_memory=False)
            sub = pd.DataFrame()
            sub["prefecture"] = sdf[sdf.columns[0]]
            sub["date"] = pd.to_datetime(sdf[sdf.columns[1]], errors="coerce")

            sat_col = next((c for c in sdf.columns if "満足度（旅行全体）" in c), None)
            sub["satisfaction"] = (
                pd.to_numeric(sdf[sat_col], errors="coerce") if sat_col else np.nan
            )

            reason_col = next(
                (c for c in sdf.columns if c == "満足度理由" or
                 ("満足度理由" in c and "サービス" not in c)), None
            )
            inconv_col = next((c for c in sdf.columns if "不便" in c), None)
            free_col = next((c for c in sdf.columns if "自由意見" in c), None)

            sub["reason"] = sdf[reason_col].astype(str) if reason_col else ""
            sub["inconvenience"] = sdf[inconv_col].astype(str) if inconv_col else ""
            sub["freetext"] = sdf[free_col].astype(str) if free_col else ""
            sub = sub.dropna(subset=["date"])
            frames.append(sub)
        except Exception as exc:
            rpt(f"  Warning loading {path}: {exc}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["prefecture", "date", "satisfaction",
                                  "reason", "inconvenience", "freetext"])


# ── Raw Fukui Survey (all.csv) ───────────────────────────────────────────────

def load_raw_fukui_survey(
    path: str | Path,
    spending_map: dict[str, int] | None = None,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Load the raw ``fukui-kanko-survey/all.csv`` with spending midpoints.

    Args:
        path: Path to ``all.csv``.
        spending_map: Mapping of spending category strings to yen midpoints.
        reporter: Optional ``Reporter``.

    Returns:
        DataFrame with original columns plus ``spending_midpoint`` and ``date``.
    """
    rpt = reporter.log if reporter else print
    path = Path(path)
    if not path.exists():
        rpt(f"WARNING: Raw survey not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(str(path), low_memory=False, encoding="utf-8")
    if spending_map and "県内消費額" in df.columns:
        df["spending_midpoint"] = df["県内消費額"].map(spending_map)
    df["date"] = pd.to_datetime(df.get("回答日時"), errors="coerce").dt.normalize()
    rpt(f"Raw Fukui survey loaded: {len(df)} rows")
    return df


# ── Merge into master daily ──────────────────────────────────────────────────

def merge_daily(
    camera: pd.DataFrame,
    weather: pd.DataFrame,
    rsi: pd.DataFrame,
    *,
    reporter: Reporter | None = None,
) -> pd.DataFrame:
    """Merge camera, weather, and RSI intent into a single daily table.

    Args:
        camera: ``[date, count]``.
        weather: ``[date, precip, temp, …]``.
        rsi: ``[date, …intent columns…]``.
        reporter: Optional ``Reporter``.

    Returns:
        Merged DataFrame with outlier flags and ADF results logged.
    """
    rpt = reporter.log if reporter else print

    daily = camera.merge(weather, on="date", how="left")
    daily = daily.merge(rsi, on="date", how="left")
    daily = daily.dropna(subset=["count"]).reset_index(drop=True)
    rpt(f"Merged daily rows (camera ∩ weather ∩ rsi): {len(daily)}")
    rpt(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")

    # Outlier detection (IQR)
    q1 = daily["count"].quantile(0.25)
    q3 = daily["count"].quantile(0.75)
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    daily["is_outlier"] = (daily["count"] < low_fence) | (daily["count"] > high_fence)
    n_out = daily["is_outlier"].sum()
    rpt(f"\nOutlier detection (IQR): Q1={q1:.0f}  Q3={q3:.0f}  "
        f"IQR={iqr:.0f}  fences=[{low_fence:.0f}, {high_fence:.0f}]")
    rpt(f"  Outlier days: {n_out}")

    return daily


def run_adf_tests(
    daily: pd.DataFrame,
    route_col: str,
    *,
    reporter: Reporter | None = None,
) -> None:
    """Run Augmented Dickey-Fuller stationarity tests and log results.

    Args:
        daily: Master daily DataFrame.
        route_col: Name of the RSI intent column.
        reporter: Optional ``Reporter``.
    """
    rpt = reporter.log if reporter else print
    rpt("\nAugmented Dickey-Fuller tests:")

    for name, series in [("count", daily["count"]),
                          (route_col, daily[route_col].dropna())]:
        if len(series) < 20:
            rpt(f"  {name}: too few observations ({len(series)})")
            continue
        adf_stat, p_value, used_lag, _, _, _ = adfuller(series, autolag="AIC")
        status = "STATIONARY" if p_value < 0.05 else "NON-STATIONARY"
        rpt(f"  {name}: ADF={adf_stat:.3f}  p={p_value:.4f}  "
            f"→ {status}  (lag={used_lag})")


# ── Convenience wrapper ──────────────────────────────────────────────────────

def load_all_data(
    cfg: dict[str, Any],
    reporter: Reporter,
) -> dict[str, Any]:
    """Load every data source required by the pipeline.

    This is the single entry-point that ``src/run_analysis.py`` calls.

    Args:
        cfg: Loaded settings dict.
        reporter: ``Reporter`` instance.

    Returns:
        Dictionary with keys: ``daily``, ``weather_daily``, ``rsi``,
        ``route_col``, ``survey_all``, ``sat_all``, ``text_all``,
        ``raw_survey``.
    """
    from .config import resolve_repo_path, resolve_ws_path

    reporter.section(1, "Data Loading & Cleaning")
    paths = cfg["paths"]

    # Camera
    camera = load_camera_daily(
        str(resolve_ws_path(cfg, paths["camera"]["tojinbo"])),
        reporter=reporter,
    )

    # Weather
    weather_daily = load_weather_daily(
        resolve_repo_path(cfg, paths["weather"]["mikuni"]),
        resolve_repo_path(cfg, paths["weather"]["mikuni_legacy"]),
        reporter=reporter,
    )

    # RSI Intent
    rsi, route_col = load_rsi_intent(
        resolve_ws_path(cfg, paths["rsi_data"]),
        reporter=reporter,
    )

    # Merge
    daily = merge_daily(camera, weather_daily, rsi, reporter=reporter)
    run_adf_tests(daily, route_col, reporter=reporter)

    # Surveys
    survey_glob = str(resolve_ws_path(cfg, paths["survey"]["merged_glob"]))
    survey_all = load_survey_prefectures(survey_glob, reporter=reporter)
    sat_all = load_survey_satisfaction(survey_glob, reporter=reporter)
    text_all = load_survey_text(survey_glob, reporter=reporter)

    # Raw Fukui survey
    raw_survey = load_raw_fukui_survey(
        resolve_ws_path(cfg, paths["survey"]["raw_fukui"]),
        spending_map=cfg.get("survey", {}).get("spending_map"),
        reporter=reporter,
    )

    return {
        "daily": daily,
        "weather_daily": weather_daily,
        "rsi": rsi,
        "route_col": route_col,
        "survey_all": survey_all,
        "sat_all": sat_all,
        "text_all": text_all,
        "raw_survey": raw_survey,
    }
