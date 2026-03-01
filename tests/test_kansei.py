"""Tests for src.kansei – Discomfort Index, Wind Chill, and Overtourism math.

These tests verify the *exact arithmetic* of the Kansei formulas so that
the PhD committee can certify the calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kansei import (
    compute_discomfort_index,
    compute_wind_chill,
    discomfort_index_analysis,
    overtourism_threshold,
)
from src.report import Reporter


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def reporter(tmp_path) -> Reporter:
    cfg = {
        "_resolved": {
            "repo_dir": tmp_path,
            "workspace_root": tmp_path,
        },
        "paths": {"output": "output", "figures": "output"},
        "visualization": {"dpi": 72, "ja_copy": False},
    }
    return Reporter(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Discomfort Index
# ══════════════════════════════════════════════════════════════════════════════


class TestDiscomfortIndex:
    r"""Verify DI = 0.81T + 0.01H(0.99T − 14.3) + 46.3.

    Reference values computed by hand:
        T=25°C, H=60% →
            0.81×25 + 0.01×60×(0.99×25 − 14.3) + 46.3
          = 20.25 + 0.60×(24.75 − 14.3) + 46.3
          = 20.25 + 0.60×10.45 + 46.3
          = 20.25 + 6.27 + 46.3
          = 72.82

        T=30°C, H=80% →
            0.81×30 + 0.01×80×(0.99×30 − 14.3) + 46.3
          = 24.30 + 0.80×(29.70 − 14.3) + 46.3
          = 24.30 + 0.80×15.40 + 46.3
          = 24.30 + 12.32 + 46.3
          = 82.92

        T=0°C, H=50% →
            0.81×0 + 0.01×50×(0.99×0 − 14.3) + 46.3
          = 0 + 0.50×(−14.3) + 46.3
          = −7.15 + 46.3
          = 39.15
    """

    def test_hand_calculation_25_60(self):
        T = pd.Series([25.0])
        H = pd.Series([60.0])
        result = compute_discomfort_index(T, H)
        expected = 0.81 * 25 + 0.01 * 60 * (0.99 * 25 - 14.3) + 46.3
        assert abs(result.iloc[0] - expected) < 1e-10
        assert abs(result.iloc[0] - 72.82) < 0.01

    def test_hand_calculation_30_80(self):
        T = pd.Series([30.0])
        H = pd.Series([80.0])
        result = compute_discomfort_index(T, H)
        expected = 0.81 * 30 + 0.01 * 80 * (0.99 * 30 - 14.3) + 46.3
        assert abs(result.iloc[0] - expected) < 1e-10
        assert abs(result.iloc[0] - 82.92) < 0.01

    def test_hand_calculation_0_50(self):
        T = pd.Series([0.0])
        H = pd.Series([50.0])
        result = compute_discomfort_index(T, H)
        expected = 0.81 * 0 + 0.01 * 50 * (0.99 * 0 - 14.3) + 46.3
        assert abs(result.iloc[0] - expected) < 1e-10
        assert abs(result.iloc[0] - 39.15) < 0.01

    def test_vectorized_computation(self):
        """Ensure the function works on multi-element Series."""
        T = pd.Series([0.0, 15.0, 25.0, 30.0, 35.0])
        H = pd.Series([50.0, 55.0, 60.0, 70.0, 80.0])
        result = compute_discomfort_index(T, H)
        assert len(result) == 5
        # Each value should be computable independently
        for i in range(5):
            expected = (
                0.81 * T.iloc[i]
                + 0.01 * H.iloc[i] * (0.99 * T.iloc[i] - 14.3)
                + 46.3
            )
            assert abs(result.iloc[i] - expected) < 1e-10

    def test_custom_coefficients(self):
        """Users can override the DI formula coefficients."""
        T = pd.Series([20.0])
        H = pd.Series([70.0])
        result = compute_discomfort_index(
            T, H,
            coeff_temp=0.80,
            coeff_humidity=0.02,
            inner_temp=1.00,
            inner_offset=-15.0,
            constant=45.0,
        )
        expected = 0.80 * 20 + 0.02 * 70 * (1.00 * 20 - 15.0) + 45.0
        assert abs(result.iloc[0] - expected) < 1e-10

    def test_di_monotonically_increases_with_temp(self):
        """At constant humidity, DI should increase with temperature."""
        T = pd.Series(np.arange(0, 40, 1, dtype=float))
        H = pd.Series([60.0] * len(T))
        result = compute_discomfort_index(T, H)
        diffs = result.diff().dropna()
        assert (diffs > 0).all(), "DI should increase with temperature"

    def test_di_monotonically_increases_with_humidity_at_warm_temp(self):
        """At warm temp, DI should increase with humidity."""
        T = pd.Series([25.0] * 100)
        H = pd.Series(np.arange(1, 101, dtype=float))
        result = compute_discomfort_index(T, H)
        diffs = result.diff().dropna()
        # At 25°C, inner = 0.99*25 - 14.3 = 10.45 > 0 → increasing
        assert (diffs > 0).all()

    def test_preserves_index(self):
        """Output index should match input index."""
        idx = pd.Index([10, 20, 30])
        T = pd.Series([20.0, 25.0, 30.0], index=idx)
        H = pd.Series([50.0, 60.0, 70.0], index=idx)
        result = compute_discomfort_index(T, H)
        assert list(result.index) == [10, 20, 30]


# ══════════════════════════════════════════════════════════════════════════════
# Wind Chill
# ══════════════════════════════════════════════════════════════════════════════


class TestWindChill:
    r"""Verify WC = 13.12 + 0.6215T − 11.37V^{0.16} + 0.3965TV^{0.16}.

    Reference (T=−5°C, wind=5 m/s → V_kmh=18 km/h):
        V^{0.16} = 18^{0.16} ≈ 1.6330
        WC = 13.12 + 0.6215×(−5) − 11.37×1.6330 + 0.3965×(−5)×1.6330
           = 13.12 − 3.1075 − 18.5673 − 3.2374
           = −11.7922  (approx.)
    """

    def test_hand_calculation(self):
        T = pd.Series([-5.0])
        W = pd.Series([5.0])  # m/s → 18 km/h
        result = compute_wind_chill(T, W)
        v_kmh = 18.0
        expected = (
            13.12
            + 0.6215 * (-5)
            - 11.37 * v_kmh ** 0.16
            + 0.3965 * (-5) * v_kmh ** 0.16
        )
        assert abs(result.iloc[0] - expected) < 0.01

    def test_passes_through_warm_temps(self):
        """For T > 10°C, wind chill formula is invalid → return raw T."""
        T = pd.Series([15.0, 20.0, 30.0])
        W = pd.Series([5.0, 10.0, 3.0])
        result = compute_wind_chill(T, W)
        pd.testing.assert_series_equal(result, T, check_names=False)

    def test_passes_through_low_wind(self):
        """For wind < 4.8 km/h (< 1.33 m/s), formula invalid → raw T."""
        T = pd.Series([-5.0])
        W = pd.Series([1.0])  # 3.6 km/h < 4.8
        result = compute_wind_chill(T, W)
        assert result.iloc[0] == -5.0

    def test_wind_chill_lower_than_temp(self):
        """Wind chill should make it feel colder (WC < T) when valid."""
        T = pd.Series([-10.0, -5.0, 0.0, 5.0])
        W = pd.Series([8.0, 6.0, 7.0, 5.0])
        result = compute_wind_chill(T, W)
        for i in range(len(T)):
            v_kmh = W.iloc[i] * 3.6
            if T.iloc[i] <= 10 and v_kmh > 4.8:
                assert result.iloc[i] < T.iloc[i], (
                    f"WC should be < T at T={T.iloc[i]}, W={W.iloc[i]}"
                )

    def test_stronger_wind_means_colder(self):
        """Higher wind speeds should produce lower wind chill."""
        T = pd.Series([0.0, 0.0, 0.0])
        W = pd.Series([3.0, 6.0, 10.0])
        result = compute_wind_chill(T, W)
        # All have T<=10 and V>4.8 km/h (10.8, 21.6, 36)
        assert result.iloc[0] > result.iloc[1] > result.iloc[2]


# ══════════════════════════════════════════════════════════════════════════════
# Discomfort Index Analysis (Integration)
# ══════════════════════════════════════════════════════════════════════════════


class TestDIAnalysis:
    """Integration test for the full DI analysis function."""

    def test_with_humidity(self, reporter):
        n = 60
        rng = np.random.default_rng(42)
        weather = pd.DataFrame({
            "date": pd.date_range("2024-06-01", periods=n),
            "temp": rng.uniform(15, 35, n),
            "humidity": rng.uniform(40, 90, n),
            "wind": rng.uniform(1, 10, n),
        })
        result = discomfort_index_analysis(weather, None, reporter)
        assert result["di_available"] is True
        assert "di_mean" in result
        assert "wc_mean" in result

    def test_without_humidity(self, reporter):
        """Should fall back to 60% proxy."""
        n = 50
        rng = np.random.default_rng(42)
        weather = pd.DataFrame({
            "date": pd.date_range("2024-06-01", periods=n),
            "temp": rng.uniform(10, 30, n),
        })
        result = discomfort_index_analysis(weather, None, reporter)
        assert result["di_available"] is True

    def test_with_satisfaction_data(self, reporter):
        n = 100
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-06-01", periods=n)
        weather = pd.DataFrame({
            "date": dates,
            "temp": rng.uniform(15, 35, n),
            "humidity": rng.uniform(40, 90, n),
            "wind": rng.uniform(1, 10, n),
        })
        sat = pd.DataFrame({
            "date": dates,
            "mean_satisfaction": rng.uniform(3.0, 5.0, n),
            "mean_nps": rng.uniform(6, 10, n),
        })
        result = discomfort_index_analysis(weather, sat, reporter)
        assert result["di_available"] is True
        # Correlation should have been computed
        if "di_sat_r" in result:
            assert -1 <= result["di_sat_r"] <= 1


# ══════════════════════════════════════════════════════════════════════════════
# Overtourism Threshold
# ══════════════════════════════════════════════════════════════════════════════


class TestOvertourismThreshold:

    def test_with_empty_satisfaction(self, reporter):
        daily = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30),
            "count": np.random.default_rng(42).integers(5000, 20000, 30),
        })
        sat_all = pd.DataFrame(
            columns=["prefecture", "date", "satisfaction", "nps_raw",
                      "satisfaction_service"]
        )
        result = overtourism_threshold(daily, sat_all, reporter)
        assert result["spearman_r"] == 0.0

    def test_with_synthetic_satisfaction(self, reporter):
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n)
        daily = pd.DataFrame({
            "date": dates,
            "count": rng.integers(3000, 25000, n),
        })
        sat_all = pd.DataFrame({
            "prefecture": ["福井県"] * n * 3,
            "date": list(dates) * 3,
            "satisfaction": rng.integers(1, 6, n * 3),
            "nps_raw": rng.integers(0, 11, n * 3),
            "satisfaction_service": rng.integers(1, 6, n * 3),
        })
        result = overtourism_threshold(daily, sat_all, reporter)
        # Should compute Spearman
        assert isinstance(result["spearman_r"], float)


# ══════════════════════════════════════════════════════════════════════════════
# Formula Regression Tests (Golden Values)
# ══════════════════════════════════════════════════════════════════════════════


class TestGoldenValues:
    """Ensure exact numeric outputs match known-good values.

    These 'golden' tests lock in the formulas so that any accidental
    modification to the Kansei module is caught immediately.
    """

    @pytest.mark.parametrize("temp,humidity,expected", [
        (25.0, 60.0, 72.82),
        (30.0, 80.0, 82.92),
        (35.0, 90.0, 92.965),
        (0.0,  50.0, 39.15),
        (10.0, 70.0, 51.32),
    ])
    def test_di_golden(self, temp, humidity, expected):
        result = compute_discomfort_index(
            pd.Series([temp]), pd.Series([humidity])
        ).iloc[0]
        hand = (
            0.81 * temp
            + 0.01 * humidity * (0.99 * temp - 14.3)
            + 46.3
        )
        assert abs(result - hand) < 1e-10
        assert abs(result - expected) < 0.01, (
            f"DI({temp}°C, {humidity}%) = {result:.3f}, expected ≈ {expected}"
        )
