"""Tests for src.models – OLS, Random Forest, and robustness suite.

Verifies that the regression machinery produces correct metrics on a
controlled synthetic dataset.  Tests are deterministic (fixed seeds)
so results can be reproduced on any machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import (
    OLSResult,
    RFResult,
    RobustnessResult,
    fit_ols,
    fit_random_forest,
    robustness_suite,
)
from src.report import Reporter


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def reporter(tmp_path) -> Reporter:
    """A Reporter that writes to a temporary directory."""
    cfg = {
        "_resolved": {
            "repo_dir": tmp_path,
            "workspace_root": tmp_path,
        },
        "paths": {
            "output": "output",
            "figures": "output",
        },
        "visualization": {"dpi": 72, "ja_copy": False},
    }
    return Reporter(cfg)


@pytest.fixture()
def synthetic_daily() -> pd.DataFrame:
    """Create a deterministic synthetic dataset with known properties.

    The target (``count``) is a linear function of the features plus noise,
    so OLS R² should be close to 1.0 on this data.
    """
    rng = np.random.default_rng(42)
    n = 200

    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    x1 = rng.normal(10, 2, n)   # "intent" (positive driver)
    x2 = rng.normal(5, 3, n)    # "precip" (negative driver)
    x3 = rng.normal(20, 5, n)   # "temp"

    # True relationship: count = 100 + 15*x1 - 5*x2 + 2*x3 + noise
    noise = rng.normal(0, 10, n)
    count = 100 + 15 * x1 - 5 * x2 + 2 * x3 + noise

    return pd.DataFrame({
        "date": dates,
        "count": count,
        "intent": x1,
        "precip": x2,
        "temp": x3,
    })


@pytest.fixture()
def feature_cols() -> list[str]:
    return ["intent", "precip", "temp"]


# ══════════════════════════════════════════════════════════════════════════════
# OLS Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestOLS:
    """Verify OLS regression on the synthetic data."""

    def test_ols_returns_correct_type(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_ols(synthetic_daily, feature_cols, reporter)
        assert isinstance(result, OLSResult)

    def test_ols_high_r2_on_linear_data(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_ols(synthetic_daily, feature_cols, reporter)
        # Data is linear + small noise → R² should be > 0.9
        assert result.r2 > 0.90, f"R² = {result.r2:.4f}, expected > 0.90"

    def test_ols_adj_r2_leq_r2(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_ols(synthetic_daily, feature_cols, reporter)
        assert result.adj_r2 <= result.r2

    def test_ols_predictions_shape(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_ols(synthetic_daily, feature_cols, reporter)
        assert len(result.y_pred) == len(synthetic_daily)

    def test_ols_coefficients_sign(
        self, synthetic_daily, feature_cols, reporter
    ):
        """Verify coefficient signs match the true data-generating process."""
        result = fit_ols(synthetic_daily, feature_cols, reporter)
        params = result.model.params  # [const, intent, precip, temp]
        assert params[1] > 0, "intent coefficient should be positive"
        assert params[2] < 0, "precip coefficient should be negative"
        assert params[3] > 0, "temp coefficient should be positive"

    def test_ols_with_missing_values(self, feature_cols, reporter):
        """OLS should work on data with NaN rows dropped externally."""
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n),
            "count": rng.normal(100, 10, n),
            "intent": rng.normal(5, 1, n),
            "precip": rng.normal(3, 1, n),
            "temp": rng.normal(15, 3, n),
        })
        # Insert some NaNs and drop them
        df.loc[5, "intent"] = np.nan
        df = df.dropna()
        result = fit_ols(df, feature_cols, reporter)
        assert len(result.y_pred) == len(df)


# ══════════════════════════════════════════════════════════════════════════════
# Random Forest Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRandomForest:
    """Verify Random Forest on the synthetic data."""

    def test_rf_returns_correct_type(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_random_forest(synthetic_daily, feature_cols, reporter)
        assert isinstance(result, RFResult)

    def test_rf_train_r2_high(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_random_forest(synthetic_daily, feature_cols, reporter)
        assert result.r2_train > 0.90

    def test_rf_cv_r2_reasonable(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_random_forest(
            synthetic_daily, feature_cols, reporter, cv_folds=3
        )
        # CV R² should be decent but lower than train R²
        assert result.cv_r2_mean > 0.5
        assert result.cv_r2_mean <= result.r2_train

    def test_rf_importance_sums_to_one(
        self, synthetic_daily, feature_cols, reporter
    ):
        result = fit_random_forest(synthetic_daily, feature_cols, reporter)
        total = result.mdi_importance["importance"].sum()
        assert abs(total - 1.0) < 1e-6, f"MDI sum = {total}"

    def test_rf_top_feature_is_intent(
        self, synthetic_daily, feature_cols, reporter
    ):
        """The strongest predictor (intent, coeff=15) should rank #1."""
        result = fit_random_forest(synthetic_daily, feature_cols, reporter)
        top = result.mdi_importance.iloc[0]["feature"]
        assert top == "intent", f"Expected 'intent' as top feature, got '{top}'"

    def test_rf_custom_params(
        self, synthetic_daily, feature_cols, reporter
    ):
        params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
        result = fit_random_forest(
            synthetic_daily, feature_cols, reporter, rf_params=params
        )
        assert result.r2_train > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Robustness Suite Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRobustness:
    """Verify the Durbin–Watson, first-difference, and LDV diagnostics."""

    def test_robustness_returns_correct_type(
        self, synthetic_daily, feature_cols, reporter
    ):
        ols_result = fit_ols(synthetic_daily, feature_cols, reporter)
        result = robustness_suite(
            synthetic_daily, ols_result, feature_cols, reporter
        )
        assert isinstance(result, RobustnessResult)

    def test_dw_in_valid_range(
        self, synthetic_daily, feature_cols, reporter
    ):
        ols_result = fit_ols(synthetic_daily, feature_cols, reporter)
        result = robustness_suite(
            synthetic_daily, ols_result, feature_cols, reporter
        )
        # DW statistic is always in [0, 4]
        assert 0 <= result.dw_stat <= 4

    def test_dw_clean_for_iid_noise(
        self, synthetic_daily, feature_cols, reporter
    ):
        """Synthetic data has i.i.d. noise → DW should be near 2 (clean)."""
        ols_result = fit_ols(synthetic_daily, feature_cols, reporter)
        result = robustness_suite(
            synthetic_daily, ols_result, feature_cols, reporter
        )
        assert result.dw_clean, (
            f"DW = {result.dw_stat:.3f}; expected clean for i.i.d. noise"
        )

    def test_first_diff_r2_nonnegative(
        self, synthetic_daily, feature_cols, reporter
    ):
        ols_result = fit_ols(synthetic_daily, feature_cols, reporter)
        result = robustness_suite(
            synthetic_daily, ols_result, feature_cols, reporter
        )
        assert result.fd_r2 >= 0

    def test_ldv_r2_is_reasonable(
        self, synthetic_daily, feature_cols, reporter
    ):
        ols_result = fit_ols(synthetic_daily, feature_cols, reporter)
        result = robustness_suite(
            synthetic_daily, ols_result, feature_cols, reporter
        )
        assert result.ldv_r2 > 0

    def test_weather_value_computed(
        self, synthetic_daily, feature_cols, reporter
    ):
        ols_result = fit_ols(synthetic_daily, feature_cols, reporter)
        result = robustness_suite(
            synthetic_daily, ols_result, feature_cols, reporter
        )
        # weather_value = R²(full) - R²(no weather)
        # Since none of our features are in the weather set, weather_value ≈ 0
        assert isinstance(result.weather_value, float)


# ══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Guard against degenerate inputs."""

    def test_small_dataset(self, reporter):
        """Models should still run with a very small dataset."""
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n),
            "count": rng.normal(50, 5, n),
            "x1": rng.normal(0, 1, n),
        })
        result = fit_ols(df, ["x1"], reporter)
        assert isinstance(result, OLSResult)

    def test_constant_target_ols(self, reporter):
        """OLS with zero-variance target should still return a result."""
        n = 40
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n),
            "count": [100.0] * n,
            "x1": np.random.default_rng(42).normal(0, 1, n),
        })
        result = fit_ols(df, ["x1"], reporter)
        assert isinstance(result, OLSResult)
        # R² is undefined / 0 when target has no variance – just ensure no crash
