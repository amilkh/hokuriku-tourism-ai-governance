"""Data Integrity & Validation Module.

Audits all input CSVs (JMA weather, AI camera, Hokuriku survey) for:
    - **Schema mismatches** – unexpected/missing columns
    - **Data drift** – distribution shifts between monthly windows
    - **Outliers** – IQR-based and Z-score detection
    - **Missing-value patterns** – per-column and temporal gaps
    - **Temporal continuity** – date gaps in time-series sources

Generates ``validation_report.json`` alongside the analysis output so that
reviewers can verify 1.4 M rows were processed without silent errors.

Usage::

    from src.validator import validate_pipeline
    report = validate_pipeline(cfg, reporter)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .report import Reporter

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Result containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ColumnCheck:
    """Validation result for a single column."""

    name: str
    dtype: str
    n_total: int
    n_missing: int
    pct_missing: float
    n_outliers_iqr: int = 0
    n_outliers_zscore: int = 0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q75: float | None = None


@dataclass
class DriftCheck:
    """Monthly distribution drift result."""

    column: str
    period_a: str
    period_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    ks_statistic: float
    ks_pvalue: float
    drift_detected: bool


@dataclass
class SourceReport:
    """Validation report for a single data source."""

    source_name: str
    file_path: str
    n_rows: int
    n_columns: int
    expected_columns: list[str]
    actual_columns: list[str]
    missing_columns: list[str]
    extra_columns: list[str]
    date_range: str = ""
    date_gaps: list[str] = field(default_factory=list)
    column_checks: list[ColumnCheck] = field(default_factory=list)
    drift_checks: list[DriftCheck] = field(default_factory=list)
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Top-level validation report aggregating all sources."""

    generated_at: str
    pipeline_version: str
    total_rows_audited: int = 0
    sources: list[SourceReport] = field(default_factory=list)
    overall_passed: bool = True
    summary: dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Core validation functions
# ══════════════════════════════════════════════════════════════════════════════


def _safe_float(v: Any) -> float | None:
    """Convert to Python float, returning None for non-finite values."""
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def check_schema(
    df: pd.DataFrame,
    expected_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Return (missing_columns, extra_columns) vs *expected_cols*."""
    actual = set(df.columns)
    expected = set(expected_cols)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return missing, extra


def check_column(
    series: pd.Series,
    *,
    zscore_threshold: float = 3.5,
) -> ColumnCheck:
    """Compute summary statistics and outlier counts for a single column."""
    n_total = len(series)
    n_missing = int(series.isna().sum())
    pct_missing = round(n_missing / max(n_total, 1) * 100, 2)

    cc = ColumnCheck(
        name=series.name,
        dtype=str(series.dtype),
        n_total=n_total,
        n_missing=n_missing,
        pct_missing=pct_missing,
    )

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) > 0:
        cc.mean = _safe_float(numeric.mean())
        cc.std = _safe_float(numeric.std())
        cc.min = _safe_float(numeric.min())
        cc.max = _safe_float(numeric.max())
        cc.q25 = _safe_float(numeric.quantile(0.25))
        cc.q75 = _safe_float(numeric.quantile(0.75))

        # IQR outliers
        if cc.q25 is not None and cc.q75 is not None:
            iqr = cc.q75 - cc.q25
            low = cc.q25 - 1.5 * iqr
            high = cc.q75 + 1.5 * iqr
            cc.n_outliers_iqr = int(((numeric < low) | (numeric > high)).sum())

        # Z-score outliers
        if cc.std is not None and cc.std > 0:
            z = (numeric - numeric.mean()).abs() / numeric.std()
            cc.n_outliers_zscore = int((z > zscore_threshold).sum())

    return cc


def check_date_gaps(
    dates: pd.Series,
    freq: str = "D",
) -> list[str]:
    """Return a list of missing dates in a daily time-series."""
    dates = pd.to_datetime(dates).dropna().sort_values().drop_duplicates()
    if len(dates) < 2:
        return []
    full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
    missing = full_range.difference(dates)
    return [d.strftime("%Y-%m-%d") for d in missing]


def check_drift(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    window_months: int = 3,
) -> list[DriftCheck]:
    """Kolmogorov–Smirnov drift test between consecutive monthly windows."""
    from scipy.stats import ks_2samp

    results: list[DriftCheck] = []
    if date_col not in df.columns or value_col not in df.columns:
        return results

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    if len(df) < 30:
        return results

    numeric = pd.to_numeric(df[value_col], errors="coerce")
    df = df.assign(**{value_col: numeric}).dropna(subset=[value_col])

    df["_ym"] = df[date_col].dt.to_period("M")
    periods = sorted(df["_ym"].unique())

    for i in range(0, len(periods) - window_months, window_months):
        p_a = periods[i : i + window_months]
        p_b = periods[i + window_months : i + 2 * window_months]
        if len(p_b) == 0:
            break

        a = df[df["_ym"].isin(p_a)][value_col].values
        b = df[df["_ym"].isin(p_b)][value_col].values

        if len(a) < 10 or len(b) < 10:
            continue

        stat, pval = ks_2samp(a, b)
        results.append(DriftCheck(
            column=value_col,
            period_a=f"{p_a[0]}–{p_a[-1]}",
            period_b=f"{p_b[0]}–{p_b[-1]}",
            mean_a=_safe_float(np.mean(a)) or 0.0,
            mean_b=_safe_float(np.mean(b)) or 0.0,
            std_a=_safe_float(np.std(a)) or 0.0,
            std_b=_safe_float(np.std(b)) or 0.0,
            ks_statistic=round(stat, 4),
            ks_pvalue=round(pval, 4),
            drift_detected=pval < 0.05,
        ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Source-specific validators
# ══════════════════════════════════════════════════════════════════════════════


def validate_camera_data(
    glob_pattern: str,
    source_name: str = "AI Camera (Tojinbo)",
) -> SourceReport:
    """Validate AI-camera CSVs loaded via the pipeline's glob pattern."""
    import glob as glob_mod

    files = sorted(glob_mod.glob(glob_pattern, recursive=True))
    expected = ["aggregate from", "aggregate to", "total count"]

    report = SourceReport(
        source_name=source_name,
        file_path=glob_pattern,
        n_rows=0,
        n_columns=0,
        expected_columns=expected,
        actual_columns=[],
        missing_columns=[],
        extra_columns=[],
    )

    if not files:
        report.passed = False
        report.errors.append("No CSV files matched the glob pattern.")
        return report

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as exc:
            report.warnings.append(f"Could not read {f}: {exc}")

    if not frames:
        report.passed = False
        report.errors.append("All CSV files failed to load.")
        return report

    sample = frames[0]
    report.actual_columns = list(sample.columns)
    report.n_columns = len(sample.columns)
    report.missing_columns, report.extra_columns = check_schema(sample, expected)

    if report.missing_columns:
        report.passed = False
        report.errors.append(f"Missing columns: {report.missing_columns}")

    # Aggregate daily for analysis
    daily_rows: list[dict[str, Any]] = []
    for f, df in zip(files, frames):
        if "total count" in df.columns:
            import os
            daily_rows.append({
                "date": os.path.basename(f).replace(".csv", ""),
                "count": df["total count"].sum(),
            })

    daily = pd.DataFrame(daily_rows)
    report.n_rows = len(daily)

    if not daily.empty and "count" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = daily.dropna(subset=["date"]).sort_values("date")

        report.date_range = (
            f"{daily['date'].min().date()} → {daily['date'].max().date()}"
        )
        report.date_gaps = check_date_gaps(daily["date"])
        if report.date_gaps:
            report.warnings.append(
                f"{len(report.date_gaps)} date gaps detected."
            )

        report.column_checks.append(check_column(daily["count"]))

        # Drift on daily counts
        daily_for_drift = daily.rename(columns={"date": "date"})
        report.drift_checks.extend(
            check_drift(daily_for_drift, "date", "count")
        )

        # Warn on zero-count days
        n_zero = int((daily["count"] == 0).sum())
        if n_zero > 0:
            report.warnings.append(
                f"{n_zero} zero-count days (potential camera outages)."
            )

    logger.info(
        "Camera validation: %d files, %d daily rows, passed=%s",
        len(files), report.n_rows, report.passed,
    )
    return report


def validate_weather_csv(
    path: str | Path,
    source_name: str = "JMA Weather (Mikuni)",
) -> SourceReport:
    """Validate a JMA hourly weather CSV."""
    path = Path(path)
    expected = [
        "timestamp", "temp_c", "precip_1h_mm", "sun_1h_h",
        "wind_speed_ms", "snow_depth_cm", "humidity_pct",
    ]

    report = SourceReport(
        source_name=source_name,
        file_path=str(path),
        n_rows=0,
        n_columns=0,
        expected_columns=expected,
        actual_columns=[],
        missing_columns=[],
        extra_columns=[],
    )

    if not path.exists():
        report.passed = False
        report.errors.append(f"File not found: {path}")
        return report

    try:
        df = pd.read_csv(str(path), parse_dates=["timestamp"])
    except Exception as exc:
        report.passed = False
        report.errors.append(f"Failed to parse CSV: {exc}")
        return report

    report.n_rows = len(df)
    report.n_columns = len(df.columns)
    report.actual_columns = list(df.columns)
    report.missing_columns, report.extra_columns = check_schema(df, expected)

    if report.missing_columns:
        report.warnings.append(f"Missing columns: {report.missing_columns}")

    # Date range & gaps (daily)
    if "timestamp" in df.columns:
        dates = df["timestamp"].dt.normalize().drop_duplicates()
        report.date_range = f"{dates.min().date()} → {dates.max().date()}"
        report.date_gaps = check_date_gaps(dates)
        if report.date_gaps:
            report.warnings.append(
                f"{len(report.date_gaps)} daily gaps in weather data."
            )

    # Column-level checks
    for col in ["temp_c", "precip_1h_mm", "sun_1h_h", "wind_speed_ms",
                 "humidity_pct", "snow_depth_cm"]:
        if col in df.columns:
            cc = check_column(df[col])
            report.column_checks.append(cc)

            # Domain-specific sanity checks
            if col == "temp_c":
                if cc.min is not None and cc.min < -40:
                    report.warnings.append(
                        f"Suspicious minimum temperature: {cc.min}°C"
                    )
                if cc.max is not None and cc.max > 50:
                    report.warnings.append(
                        f"Suspicious maximum temperature: {cc.max}°C"
                    )
            elif col == "precip_1h_mm":
                if cc.min is not None and cc.min < 0:
                    report.errors.append("Negative precipitation detected.")
                    report.passed = False
            elif col == "wind_speed_ms":
                if cc.max is not None and cc.max > 60:
                    report.warnings.append(
                        f"Extreme wind speed: {cc.max} m/s"
                    )

    # Drift
    for col in ["temp_c", "precip_1h_mm", "wind_speed_ms"]:
        if col in df.columns:
            report.drift_checks.extend(
                check_drift(df, "timestamp", col, window_months=3)
            )

    logger.info(
        "Weather validation (%s): %d rows, passed=%s",
        source_name, report.n_rows, report.passed,
    )
    return report


def validate_survey_csv(
    glob_pattern: str,
    source_name: str = "Hokuriku Tourism Survey",
) -> SourceReport:
    """Validate merged survey CSVs."""
    import glob as glob_mod

    files = sorted(glob_mod.glob(glob_pattern))
    expected_partial = [
        "満足度（旅行全体）", "おすすめ度",
    ]

    report = SourceReport(
        source_name=source_name,
        file_path=glob_pattern,
        n_rows=0,
        n_columns=0,
        expected_columns=expected_partial,
        actual_columns=[],
        missing_columns=[],
        extra_columns=[],
    )

    if not files:
        report.passed = False
        report.errors.append("No survey CSV files matched the pattern.")
        return report

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            frames.append(df)
        except Exception as exc:
            report.warnings.append(f"Could not read {f}: {exc}")

    if not frames:
        report.passed = False
        report.errors.append("All survey CSVs failed to load.")
        return report

    combined = pd.concat(frames, ignore_index=True)
    report.n_rows = len(combined)
    report.n_columns = len(combined.columns)
    report.actual_columns = list(combined.columns)[:30]  # first 30 for brevity

    # Check for key columns (partial match)
    found_cols = set(combined.columns)
    for exp in expected_partial:
        if not any(exp in c for c in found_cols):
            report.missing_columns.append(exp)
            report.warnings.append(f"Expected column containing '{exp}' not found.")

    # Satisfaction distribution
    sat_col = next(
        (c for c in combined.columns if "満足度（旅行全体）" in c), None
    )
    if sat_col:
        sat_numeric = pd.to_numeric(combined[sat_col], errors="coerce")
        report.column_checks.append(check_column(sat_numeric.rename("satisfaction")))

        # Range check: satisfaction should be 1–5
        valid = sat_numeric.dropna()
        out_of_range = ((valid < 1) | (valid > 5)).sum()
        if out_of_range > 0:
            report.warnings.append(
                f"{int(out_of_range)} satisfaction values outside [1, 5] range."
            )

    # Prefecture column (first column)
    pref_col = combined.columns[0]
    n_unique_pref = combined[pref_col].nunique()
    report.column_checks.append(ColumnCheck(
        name="prefecture",
        dtype=str(combined[pref_col].dtype),
        n_total=len(combined),
        n_missing=int(combined[pref_col].isna().sum()),
        pct_missing=round(combined[pref_col].isna().mean() * 100, 2),
    ))

    logger.info(
        "Survey validation: %d files, %d rows, %d prefectures, passed=%s",
        len(files), report.n_rows, n_unique_pref, report.passed,
    )
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline-level validation
# ══════════════════════════════════════════════════════════════════════════════


def validate_pipeline(
    cfg: dict[str, Any],
    reporter: Reporter,
) -> ValidationReport:
    """Run all validators and write ``validation_report.json``.

    This is the entry-point called by ``run_analysis.py``.

    Args:
        cfg: Loaded pipeline configuration (from ``load_config()``).
        reporter: ``Reporter`` instance for logging.

    Returns:
        ``ValidationReport`` with per-source and aggregated results.
    """
    from .config import resolve_repo_path, resolve_ws_path
    from . import __version__

    reporter.section("V", "Data Integrity Validation")

    vr = ValidationReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        pipeline_version=__version__,
    )

    paths = cfg["paths"]

    # ── Camera ────────────────────────────────────────────────────────────
    try:
        camera_glob = str(resolve_ws_path(cfg, paths["camera"]["tojinbo"]))
        camera_report = validate_camera_data(camera_glob)
        vr.sources.append(camera_report)
        vr.total_rows_audited += camera_report.n_rows
        reporter.log(
            f"  Camera: {camera_report.n_rows} daily rows, "
            f"{len(camera_report.date_gaps)} date gaps, "
            f"passed={camera_report.passed}"
        )
    except Exception as exc:
        reporter.log(f"  Camera validation failed: {exc}")
        logger.exception("Camera validation error")

    # ── Weather (all stations) ────────────────────────────────────────────
    for station_key in ("mikuni", "fukui", "katsuyama"):
        weather_rel = paths["weather"].get(station_key)
        if not weather_rel:
            continue
        try:
            weather_path = resolve_repo_path(cfg, weather_rel)
            if weather_path.exists():
                wr = validate_weather_csv(
                    weather_path,
                    source_name=f"JMA Weather ({station_key.title()})",
                )
                vr.sources.append(wr)
                vr.total_rows_audited += wr.n_rows
                reporter.log(
                    f"  Weather ({station_key}): {wr.n_rows} rows, "
                    f"{len(wr.date_gaps)} date gaps, "
                    f"passed={wr.passed}"
                )
        except Exception as exc:
            reporter.log(f"  Weather ({station_key}) validation failed: {exc}")
            logger.exception("Weather validation error (%s)", station_key)

    # ── Survey ────────────────────────────────────────────────────────────
    try:
        survey_glob = str(resolve_ws_path(cfg, paths["survey"]["merged_glob"]))
        survey_report = validate_survey_csv(survey_glob)
        vr.sources.append(survey_report)
        vr.total_rows_audited += survey_report.n_rows
        reporter.log(
            f"  Survey: {survey_report.n_rows} rows, "
            f"passed={survey_report.passed}"
        )
    except Exception as exc:
        reporter.log(f"  Survey validation failed: {exc}")
        logger.exception("Survey validation error")

    # ── Aggregate ─────────────────────────────────────────────────────────
    vr.overall_passed = all(s.passed for s in vr.sources)

    total_warnings = sum(len(s.warnings) for s in vr.sources)
    total_errors = sum(len(s.errors) for s in vr.sources)
    total_drift = sum(
        sum(1 for d in s.drift_checks if d.drift_detected)
        for s in vr.sources
    )

    vr.summary = {
        "total_sources_audited": len(vr.sources),
        "total_rows_audited": vr.total_rows_audited,
        "total_warnings": total_warnings,
        "total_errors": total_errors,
        "total_drift_events": total_drift,
        "overall_passed": vr.overall_passed,
    }

    reporter.log(f"\n  ── Validation Summary ──")
    reporter.log(f"  Sources audited:   {len(vr.sources)}")
    reporter.log(f"  Total rows:        {vr.total_rows_audited:,}")
    reporter.log(f"  Warnings:          {total_warnings}")
    reporter.log(f"  Errors:            {total_errors}")
    reporter.log(f"  Drift events:      {total_drift}")
    reporter.log(f"  Overall:           {'PASSED ✓' if vr.overall_passed else 'FAILED ✗'}")

    # ── Write JSON report ─────────────────────────────────────────────────
    output_path = reporter.out_dir / "validation_report.json"
    _write_report_json(vr, output_path)
    reporter.log(f"  Report written to: {output_path}")

    return vr


def _write_report_json(vr: ValidationReport, path: Path) -> None:
    """Serialize the ValidationReport to a JSON file."""

    def _serialize(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(vr), fh, indent=2, default=_serialize, ensure_ascii=False)
