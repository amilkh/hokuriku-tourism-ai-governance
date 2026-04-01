"""Tests for src.benchmark – splits, baselines, ablation tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmark import (
    BenchmarkResult,
    ablated_columns,
    chronological_split,
    classify_feature_families,
    predict_naive_lag1,
    run_benchmark,
)
from src.report import Reporter


def _reporter_cfg(tmp_path) -> dict:
    return {
        "_resolved": {"repo_dir": tmp_path, "workspace_root": tmp_path},
        "paths": {"output": "output", "figures": "output"},
        "visualization": {"dpi": 72, "ja_copy": False},
        "model": {
            "random_forest": {
                "n_estimators": 40,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "random_state": 0,
                "n_jobs": 1,
            },
        },
        "benchmark": {
            "enabled": True,
            "train_pct": 0.75,
            "baselines": {"naive_lag1": True, "rolling_mean_7": True},
            "ablation": {"weather": True, "rsi_intent": True, "calendar": True},
        },
    }


@pytest.fixture()
def reporter_and_cfg(tmp_path):
    cfg = _reporter_cfg(tmp_path)
    return Reporter(cfg), cfg


@pytest.fixture()
def panel_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    intent = rng.normal(4000, 400, n)
    precip = rng.normal(5, 3, n)
    temp = rng.normal(12, 5, n)
    sun = rng.normal(4, 2, n)
    wind = rng.normal(3, 1, n)
    count = 2000 + 0.4 * intent - 20 * precip + 10 * temp + rng.normal(0, 200, n)
    return pd.DataFrame({
        "date": dates,
        "count": count,
        "directions": intent,
        "directions_lag1": np.roll(intent, 1),
        "directions_lag2": np.roll(intent, 2),
        "directions_lag3": np.roll(intent, 3),
        "directions_roll7": pd.Series(intent).rolling(7, min_periods=1).mean(),
        "precip": precip,
        "temp": temp,
        "sun": sun,
        "wind": wind,
        "precip_lag1": np.roll(precip, 1),
        "is_weekend_or_holiday": (pd.Series(dates).dt.dayofweek >= 5).astype(int),
        "weather_severity": (precip > 8).astype(int),
        "dow_mean_count": 8000.0,
        "weekend_x_severity": 0.0,
        "weekend_x_intent": 0.0,
        "month": pd.Series(dates).dt.month,
    })


class TestNaiveBaseline:
    def test_naive_predictions_length(self, panel_df, reporter_and_cfg):
        reporter, cfg = reporter_and_cfg
        feature_cols = [c for c in panel_df.columns if c not in ("date", "count")]
        df = panel_df.dropna().reset_index(drop=True)
        train, test, _ = chronological_split(df, 0.8)
        pred = predict_naive_lag1(df, test)
        assert len(pred) == len(test)


class TestChronologicalSplit:
    def test_preserves_order_no_shuffle(self, panel_df):
        df = panel_df.dropna().reset_index(drop=True)
        train, test, tn = chronological_split(df, 0.8)
        assert tn == len(train)
        if len(test):
            assert train["date"].max() < test["date"].min()
        combined = pd.concat([train, test], ignore_index=True)
        assert combined["date"].is_monotonic_increasing


class TestRunBenchmark:
    def test_summary_table_columns(self, panel_df, reporter_and_cfg):
        reporter, cfg = reporter_and_cfg
        feature_cols = [
            c for c in panel_df.columns
            if c not in ("date", "count")
        ]
        df = panel_df.dropna().reset_index(drop=True)
        res = run_benchmark(
            {
                "model_df": df,
                "feature_cols": feature_cols,
                "route_col": "directions",
                "cfg": cfg,
            },
            reporter,
        )
        assert isinstance(res, BenchmarkResult)
        assert not res.summary_table.empty
        for col in ("model", "MAE", "RMSE", "R2"):
            assert col in res.summary_table.columns

    def test_ablation_table_columns(self, panel_df, reporter_and_cfg):
        reporter, cfg = reporter_and_cfg
        feature_cols = [
            c for c in panel_df.columns
            if c not in ("date", "count")
        ]
        df = panel_df.dropna().reset_index(drop=True)
        res = run_benchmark(
            {
                "model_df": df,
                "feature_cols": feature_cols,
                "route_col": "directions",
                "cfg": cfg,
            },
            reporter,
        )
        assert not res.ablation_table.empty
        for col in ("scenario", "MAE", "RMSE", "R2", "delta_MAE_vs_full", "delta_R2_vs_full"):
            assert col in res.ablation_table.columns

    def test_disabled_returns_empty(self, panel_df, reporter_and_cfg):
        reporter, cfg = reporter_and_cfg
        cfg["benchmark"]["enabled"] = False
        feature_cols = [c for c in panel_df.columns if c not in ("date", "count")]
        df = panel_df.dropna().reset_index(drop=True)
        res = run_benchmark(
            {
                "model_df": df,
                "feature_cols": feature_cols,
                "route_col": "directions",
                "cfg": cfg,
            },
            reporter,
        )
        assert res.summary_table.empty
        assert res.ablation_table.empty


class TestAblationHurtsWithoutIntent:
    """Removing dominant intent features should not improve test MAE."""

    def test_no_intent_worse_than_full(self, reporter_and_cfg):
        reporter, cfg = reporter_and_cfg
        rng = np.random.default_rng(123)
        n = 200
        dates = pd.date_range("2023-06-01", periods=n, freq="D")
        intent = rng.normal(10, 1, n)
        noise = rng.normal(0, 0.5, n)
        count = 100 + 8 * intent + noise
        df = pd.DataFrame({
            "date": dates,
            "count": count,
            "directions": intent,
            "directions_lag1": pd.Series(intent).shift(1),
            "directions_roll7": pd.Series(intent).rolling(7, min_periods=1).mean(),
            "precip": rng.normal(0, 1, n),
            "temp": rng.normal(15, 2, n),
            "sun": rng.normal(5, 1, n),
            "wind": rng.normal(2, 0.5, n),
            "precip_lag1": pd.Series(rng.normal(0, 1, n)).shift(1),
            "is_weekend_or_holiday": 0,
            "weather_severity": 0,
            "dow_mean_count": float(np.mean(count)),
            "weekend_x_severity": 0,
            "weekend_x_intent": 0,
            "month": pd.Series(dates).dt.month,
        }).dropna().reset_index(drop=True)

        feature_cols = [c for c in df.columns if c not in ("date", "count")]
        res = run_benchmark(
            {
                "model_df": df,
                "feature_cols": feature_cols,
                "route_col": "directions",
                "cfg": cfg,
            },
            reporter,
        )
        full_mae = res.ablation_table.loc[
            res.ablation_table["scenario"] == "full", "MAE"
        ].iloc[0]
        no_g_mae = res.ablation_table.loc[
            res.ablation_table["scenario"] == "no_rsi_intent", "MAE"
        ].iloc[0]
        assert no_g_mae >= full_mae - 1e-6, (
            f"expected ablation MAE >= full, got full={full_mae}, no_intent={no_g_mae}"
        )


class TestClassifyFeatures:
    def test_families_partition_overlap_intent(self):
        cols = ["directions", "directions_lag1", "precip", "month", "weekend_x_intent"]
        fam = classify_feature_families(cols, "directions")
        assert "directions" in fam["rsi_intent"]
        assert "precip" in fam["weather"]
        assert "month" in fam["calendar"]

    def test_ablated_columns(self):
        cols = ["directions", "precip", "month"]
        fam = classify_feature_families(cols, "directions")
        out = ablated_columns(cols, "weather", fam)
        assert "precip" not in out
        assert "directions" in out
