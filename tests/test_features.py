"""Tests for src.feature_engineering – Calendar, weather, lag, and interaction features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_calendar_features,
    add_dow_mean_encoding,
    add_interaction_features,
    add_lag_features,
    add_rolling_features,
    add_weather_severity,
)


@pytest.fixture()
def daily_base() -> pd.DataFrame:
    """A minimal daily DataFrame for feature engineering tests."""
    rng = np.random.default_rng(42)
    n = 60
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "count": rng.integers(5000, 20000, n),
        "intent": rng.uniform(10, 100, n),
        "precip": rng.uniform(0, 20, n),
        "temp": rng.uniform(15, 35, n),
        "sun": rng.uniform(0, 10, n),
        "wind": rng.uniform(0, 15, n),
    })


class TestCalendar:

    def test_columns_added(self, daily_base):
        result = add_calendar_features(daily_base)
        for col in ("dow", "is_weekend", "is_holiday", "is_weekend_or_holiday", "month"):
            assert col in result.columns

    def test_dow_range(self, daily_base):
        result = add_calendar_features(daily_base)
        assert result["dow"].min() >= 0
        assert result["dow"].max() <= 6

    def test_weekend_matches_dow(self, daily_base):
        result = add_calendar_features(daily_base)
        weekends = result[result["dow"].isin([5, 6])]
        assert (weekends["is_weekend"] == 1).all()
        weekdays = result[~result["dow"].isin([5, 6])]
        assert (weekdays["is_weekend"] == 0).all()


class TestWeatherSeverity:

    def test_severity_range(self, daily_base):
        result = add_weather_severity(daily_base)
        assert result["weather_severity"].min() >= 0
        assert result["weather_severity"].max() <= 3

    def test_fine_day(self):
        df = pd.DataFrame({"precip": [0.0], "wind": [2.0]})
        result = add_weather_severity(df)
        assert result["weather_severity"].iloc[0] == 0

    def test_stormy_day(self):
        df = pd.DataFrame({"precip": [15.0], "wind": [12.0]})
        result = add_weather_severity(df)
        assert result["weather_severity"].iloc[0] == 3


class TestRollingFeatures:

    def test_default_windows(self, daily_base):
        result = add_rolling_features(daily_base, "intent")
        for w in (3, 7, 14):
            assert f"intent_roll{w}" in result.columns

    def test_custom_windows(self, daily_base):
        result = add_rolling_features(daily_base, "intent", windows=[5, 10])
        assert "intent_roll5" in result.columns
        assert "intent_roll10" in result.columns


class TestLagFeatures:

    def test_default_lags(self, daily_base):
        result = add_lag_features(daily_base, "intent")
        for lag in range(8):
            assert f"intent_lag{lag}" in result.columns
        assert "precip_lag1" in result.columns
        assert "temp_lag1" in result.columns

    def test_lag_values(self, daily_base):
        result = add_lag_features(daily_base, "intent", max_lag=1)
        # lag0 should equal the original
        pd.testing.assert_series_equal(
            result["intent_lag0"],
            result["intent"],
            check_names=False,
        )
        # lag1 should be shifted by 1
        pd.testing.assert_series_equal(
            result["intent_lag1"].iloc[1:].reset_index(drop=True),
            result["intent"].iloc[:-1].reset_index(drop=True),
            check_names=False,
        )


class TestInteraction:

    def test_interaction_columns(self, daily_base):
        df = add_calendar_features(daily_base)
        df = add_weather_severity(df)
        result = add_interaction_features(df, "intent")
        assert "weekend_x_severity" in result.columns
        assert "weekend_x_intent" in result.columns


class TestDowMeanEncoding:

    def test_encoding_matches_group_mean(self, daily_base):
        df = add_calendar_features(daily_base)
        result = add_dow_mean_encoding(df)
        expected = df.groupby("dow")["count"].mean()
        for _, row in result.iterrows():
            assert row["dow_mean_count"] == pytest.approx(
                expected[row["dow"]], rel=1e-10
            )
