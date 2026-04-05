"""Chronological benchmarking and feature-family ablation for demand models.

Compares naive baselines, OLS, and Random Forest on a held-out tail of the
daily panel, then measures RF test performance when feature families are
removed.  Intended to sit alongside the main in-sample fits in
``src.models`` without replacing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .report import Reporter


@dataclass
class BenchmarkResult:
    """Outputs from :func:`run_benchmark`.

    Attributes:
        summary_table: Test-set metrics per model / baseline (wide form).
        ablation_table: RF test metrics when feature families are dropped.
        predictions_table: Optional alignable test predictions for plotting.
    """

    summary_table: pd.DataFrame
    ablation_table: pd.DataFrame
    predictions_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (MAE, RMSE, R²); safe for edge cases."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = r2_score(yt, yp)
    return mae, rmse, r2


def chronological_split(
    model_df: pd.DataFrame,
    train_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Sort by ``date`` and split into train / test (no shuffle).

    Returns:
        train_df, test_df, train_n
    """
    df = model_df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_n = max(1, int(round(n * train_pct)))
    train_n = min(train_n, n - 1) if n > 1 else n
    if train_n < 1 or train_n >= n:
        return df.iloc[:0].copy(), df.copy(), 0
    return df.iloc[:train_n].copy(), df.iloc[train_n:].copy(), train_n


def predict_naive_lag1(full_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Test predictions: yesterday's count (last train count before test)."""
    counts = full_df["count"].values
    dates = full_df["date"].values
    test_dates = test_df["date"].values
    out = np.empty(len(test_df), dtype=float)
    idx_map = {d: i for i, d in enumerate(dates)}
    for j, d in enumerate(test_dates):
        pos = idx_map.get(d)
        if pos is None or pos == 0:
            out[j] = np.nan
        else:
            out[j] = counts[pos - 1]
    return out


def predict_rolling_mean_7(full_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Causal 7-day rolling mean of count using only prior days."""
    df = full_df.sort_values("date").reset_index(drop=True)
    counts = df["count"].values
    dates = df["date"].values
    test_dates = test_df["date"].values
    pos_by_date = {dates[i]: i for i in range(len(dates))}
    out = np.empty(len(test_df), dtype=float)
    for j, d in enumerate(test_dates):
        pos = pos_by_date.get(d)
        if pos is None or pos == 0:
            out[j] = np.nan
            continue
        start = max(0, pos - 7)
        window = counts[start:pos]
        out[j] = float(np.mean(window)) if len(window) else np.nan
    return out


def _fit_predict_ols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> np.ndarray:
    """OLS fit on train, predict test (no logging)."""
    if not cols:
        return np.full(len(test_df), np.nan)
    X_tr = train_df[cols].values
    y_tr = train_df["count"].values
    X_te = test_df[cols].values
    X_tr_c = sm.add_constant(X_tr)
    X_te_c = sm.add_constant(X_te)
    res = sm.OLS(y_tr, X_tr_c).fit()
    return np.asarray(res.predict(X_te_c))


def _fit_predict_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    rf_params: dict[str, Any] | None,
) -> np.ndarray:
    """RandomForest fit on train, predict test."""
    if not cols:
        return np.full(len(test_df), np.nan)
    params = dict(rf_params or {})
    params.setdefault("n_estimators", 200)
    params.setdefault("max_depth", 10)
    params.setdefault("min_samples_leaf", 5)
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)
    rf = RandomForestRegressor(**params)
    rf.fit(train_df[cols].values, train_df["count"].values)
    return rf.predict(test_df[cols].values)


def classify_feature_families(
    feature_cols: list[str],
    route_col: str,
) -> dict[str, list[str]]:
    """Map feature columns into weather / google_intent / calendar families."""
    weather_names = {
        "precip", "temp", "sun", "wind", "precip_lag1",
        "weather_severity", "weekend_x_severity",
    }
    calendar_names = {"is_weekend_or_holiday", "month", "dow_mean_count"}
    intent: set[str] = set()
    for c in feature_cols:
        if c == route_col:
            intent.add(c)
        elif c.startswith(f"{route_col}_"):
            intent.add(c)
        elif c == "weekend_x_intent":
            intent.add(c)
    return {
        "weather": [c for c in feature_cols if c in weather_names],
        "google_intent": [c for c in feature_cols if c in intent],
        "calendar": [c for c in feature_cols if c in calendar_names],
    }


def ablated_columns(
    feature_cols: list[str],
    family: str,
    families: dict[str, list[str]],
) -> list[str]:
    """Return ``feature_cols`` with the given family removed."""
    drop = set(families.get(family, []))
    return [c for c in feature_cols if c not in drop]


def run_benchmark(data: dict[str, Any], reporter: Reporter) -> BenchmarkResult:
    """Run chronological benchmarks and RF feature-family ablation.

    Args:
        data: Must include:

            - ``model_df``: DataFrame with ``date``, ``count``, features.
            - ``feature_cols``: list of model feature names.
            - ``route_col``: Google intent column name (for family tagging).

        Optional:

            - ``cfg``: full settings dict; reads ``benchmark`` section.

    Returns:
        :class:`BenchmarkResult` with summary, ablation, and prediction tables.
    """
    model_df: pd.DataFrame = data["model_df"].copy()
    feature_cols: list[str] = list(data["feature_cols"])
    route_col: str = str(data["route_col"])
    cfg: dict[str, Any] = data.get("cfg") or {}
    bcfg = cfg.get("benchmark", {})
    if not bcfg.get("enabled", True):
        reporter.log("Benchmark stage skipped (benchmark.enabled: false).")
        return BenchmarkResult(
            summary_table=pd.DataFrame(),
            ablation_table=pd.DataFrame(),
            predictions_table=pd.DataFrame(),
        )

    train_pct = float(bcfg.get("train_pct", 0.8))
    base_cfg = bcfg.get("baselines", {}) or {}
    use_naive = base_cfg.get("naive_lag1", True)
    use_roll7 = base_cfg.get("rolling_mean_7", True)
    ab_cfg = bcfg.get("ablation", {}) or {}

    reporter.log("\n--- Chronological benchmark (train/test) ---")
    full_df = model_df.sort_values("date").reset_index(drop=True)
    train_df, test_df, train_n = chronological_split(full_df, train_pct)

    if len(test_df) == 0 or train_n < 5:
        reporter.log(
            "Benchmark: insufficient rows for chronological split; skipping."
        )
        return BenchmarkResult(
            summary_table=pd.DataFrame(),
            ablation_table=pd.DataFrame(),
            predictions_table=pd.DataFrame(),
        )

    reporter.log(
        f"  Train: n={len(train_df)}  Test: n={len(test_df)}  "
        f"(train_pct={train_pct:.2f})"
    )

    y_test = test_df["count"].values
    rows: list[dict[str, Any]] = []
    pred_cols: dict[str, np.ndarray] = {}

    if use_naive:
        y_hat = predict_naive_lag1(full_df, test_df)
        mae, rmse, r2 = _metrics(y_test, y_hat)
        rows.append({"model": "naive_lag1", "MAE": mae, "RMSE": rmse, "R2": r2})
        pred_cols["pred_naive_lag1"] = y_hat

    if use_roll7:
        y_hat = predict_rolling_mean_7(full_df, test_df)
        mae, rmse, r2 = _metrics(y_test, y_hat)
        rows.append({"model": "rolling_mean_7", "MAE": mae, "RMSE": rmse, "R2": r2})
        pred_cols["pred_rolling_mean_7"] = y_hat

    y_hat_ols = _fit_predict_ols(train_df, test_df, feature_cols)
    mae, rmse, r2 = _metrics(y_test, y_hat_ols)
    rows.append({"model": "ols", "MAE": mae, "RMSE": rmse, "R2": r2})
    pred_cols["pred_ols"] = y_hat_ols

    rf_params = cfg.get("model", {}).get("random_forest")
    y_hat_rf = _fit_predict_rf(train_df, test_df, feature_cols, rf_params)
    mae, rmse, r2 = _metrics(y_test, y_hat_rf)
    rows.append({"model": "random_forest", "MAE": mae, "RMSE": rmse, "R2": r2})
    pred_cols["pred_random_forest"] = y_hat_rf

    summary_table = pd.DataFrame(rows)

    # Ablation (RF only), same split
    families = classify_feature_families(feature_cols, route_col)
    ablation_rows: list[dict[str, Any]] = []
    mae_full, rmse_full, r2_full = _metrics(y_test, y_hat_rf)

    ablation_rows.append({
        "scenario": "full",
        "MAE": mae_full,
        "RMSE": rmse_full,
        "R2": r2_full,
        "delta_MAE_vs_full": 0.0,
        "delta_R2_vs_full": 0.0,
    })

    for fam_key, label in [
        ("weather", "no_weather"),
        ("google_intent", "no_google_intent"),
        ("calendar", "no_calendar"),
    ]:
        if not ab_cfg.get(fam_key, True):
            continue
        cols = ablated_columns(feature_cols, fam_key, families)
        if len(cols) == 0:
            reporter.log(f"  Ablation '{label}': no remaining features; skip.")
            continue
        if len(cols) == len(feature_cols):
            reporter.log(f"  Ablation '{label}': family empty in data; skip.")
            continue
        y_ab = _fit_predict_rf(train_df, test_df, cols, rf_params)
        mae_a, rmse_a, r2_a = _metrics(y_test, y_ab)
        ablation_rows.append({
            "scenario": label,
            "MAE": mae_a,
            "RMSE": rmse_a,
            "R2": r2_a,
            "delta_MAE_vs_full": mae_a - mae_full if np.isfinite(mae_a) else np.nan,
            "delta_R2_vs_full": r2_a - r2_full if np.isfinite(r2_a) else np.nan,
        })

    ablation_table = pd.DataFrame(ablation_rows)

    pred_df = test_df[["date", "count"]].copy()
    pred_df = pred_df.rename(columns={"count": "y_true"})
    for k, v in pred_cols.items():
        pred_df[k] = v

    # Human-readable summary
    reporter.log("\nBenchmark test metrics:")
    for _, r in summary_table.iterrows():
        reporter.log(
            f"  {r['model']:<18}  MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R²={r['R2']:.4f}"
        )
    if not ablation_table.empty:
        reporter.log("\nRF ablation (test set):")
        for _, r in ablation_table.iterrows():
            reporter.log(
                f"  {r['scenario']:<20}  MAE={r['MAE']:.2f}  "
                f"ΔMAE={r['delta_MAE_vs_full']:+.2f}  R²={r['R2']:.4f}"
            )

    # Machine-readable metrics
    reporter.metrics("# benchmark_summary")
    for _, r in summary_table.iterrows():
        reporter.metrics(
            f"benchmark.{r['model']}.mae={r['MAE']:.6f}"
        )
        reporter.metrics(
            f"benchmark.{r['model']}.rmse={r['RMSE']:.6f}"
        )
        reporter.metrics(
            f"benchmark.{r['model']}.r2={r['R2']:.6f}"
        )
    reporter.metrics("# benchmark_ablation_rf")
    for _, r in ablation_table.iterrows():
        reporter.metrics(
            f"benchmark.ablation.{r['scenario']}.mae={r['MAE']:.6f}"
        )
        reporter.metrics(
            f"benchmark.ablation.{r['scenario']}.r2={r['R2']:.6f}"
        )

    return BenchmarkResult(
        summary_table=summary_table,
        ablation_table=ablation_table,
        predictions_table=pred_df,
    )
