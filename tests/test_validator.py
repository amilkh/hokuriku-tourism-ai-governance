"""Tests for src.validator – Data integrity and validation module.

Verifies schema checking, outlier detection, date-gap discovery,
distribution drift detection, and the pipeline-level validation
orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validator import (
    ColumnCheck,
    DriftCheck,
    SourceReport,
    check_column,
    check_date_gaps,
    check_drift,
    check_schema,
)


# ══════════════════════════════════════════════════════════════════════════════
# Schema Checking
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaCheck:

    def test_exact_match(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        missing, extra = check_schema(df, ["a", "b", "c"])
        assert missing == []
        assert extra == []

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        missing, extra = check_schema(df, ["a", "b", "c", "d"])
        assert missing == ["c", "d"]
        assert extra == []

    def test_extra_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "x": [3]})
        missing, extra = check_schema(df, ["a", "b"])
        assert missing == []
        assert extra == ["x"]

    def test_both_missing_and_extra(self):
        df = pd.DataFrame({"a": [1], "x": [2]})
        missing, extra = check_schema(df, ["a", "b"])
        assert missing == ["b"]
        assert extra == ["x"]


# ══════════════════════════════════════════════════════════════════════════════
# Column Checks
# ══════════════════════════════════════════════════════════════════════════════


class TestColumnCheck:

    def test_basic_stats(self):
        s = pd.Series([1, 2, 3, 4, 5], name="test_col")
        cc = check_column(s)
        assert cc.name == "test_col"
        assert cc.n_total == 5
        assert cc.n_missing == 0
        assert cc.pct_missing == 0.0
        assert cc.mean == pytest.approx(3.0)
        assert cc.min == pytest.approx(1.0)
        assert cc.max == pytest.approx(5.0)

    def test_missing_values(self):
        s = pd.Series([1, np.nan, 3, np.nan, 5], name="x")
        cc = check_column(s)
        assert cc.n_missing == 2
        assert cc.pct_missing == pytest.approx(40.0)

    def test_iqr_outliers(self):
        # values 1-10 have IQR = 4.5, fences = [−4.25, 15.75]
        # 100 is an outlier
        s = pd.Series(list(range(1, 11)) + [100], name="x")
        cc = check_column(s)
        assert cc.n_outliers_iqr >= 1

    def test_zscore_outliers(self):
        rng = np.random.default_rng(42)
        data = list(rng.normal(50, 5, 100)) + [200]  # 200 is extreme
        s = pd.Series(data, name="x")
        cc = check_column(s, zscore_threshold=3.0)
        assert cc.n_outliers_zscore >= 1

    def test_all_nan(self):
        s = pd.Series([np.nan, np.nan], name="x")
        cc = check_column(s)
        assert cc.n_missing == 2
        assert cc.mean is None


# ══════════════════════════════════════════════════════════════════════════════
# Date Gap Detection
# ══════════════════════════════════════════════════════════════════════════════


class TestDateGaps:

    def test_no_gaps(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        gaps = check_date_gaps(pd.Series(dates))
        assert gaps == []

    def test_with_gaps(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
        gaps = check_date_gaps(pd.Series(dates))
        assert "2024-01-02" in gaps
        assert "2024-01-04" in gaps
        assert len(gaps) == 2

    def test_single_date(self):
        dates = pd.to_datetime(["2024-01-01"])
        gaps = check_date_gaps(pd.Series(dates))
        assert gaps == []

    def test_empty(self):
        gaps = check_date_gaps(pd.Series([], dtype="datetime64[ns]"))
        assert gaps == []


# ══════════════════════════════════════════════════════════════════════════════
# Distribution Drift
# ══════════════════════════════════════════════════════════════════════════════


class TestDriftCheck:

    def test_no_drift_in_stable_data(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=365)
        df = pd.DataFrame({
            "date": dates,
            "value": rng.normal(50, 5, 365),
        })
        results = check_drift(df, "date", "value", window_months=3)
        # Stable data → few or no drift detections
        n_drift = sum(1 for r in results if r.drift_detected)
        # With random noise, a false positive is possible but unlikely
        assert n_drift <= 2

    def test_detects_obvious_drift(self):
        # First half: mean=50, second half: mean=100
        dates = pd.date_range("2024-01-01", periods=365)
        values = [50.0] * 180 + [100.0] * 185
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 2, 365)
        df = pd.DataFrame({
            "date": dates,
            "value": np.array(values) + noise,
        })
        results = check_drift(df, "date", "value", window_months=3)
        # At least one drift should be detected around the transition
        assert any(r.drift_detected for r in results)

    def test_empty_df(self):
        df = pd.DataFrame({"date": [], "value": []})
        results = check_drift(df, "date", "value")
        assert results == []
