"""Feature engineering for the tourism demand prediction pipeline.

All functions accept a ``daily`` DataFrame and return an augmented copy.
Calendar features, weather severity, rolling/lag features, interaction
terms, and day-of-week mean encoding are computed here.
"""

from __future__ import annotations

import pandas as pd
import jpholiday

from .report import Reporter


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, weekend, holiday, and month columns.

    Args:
        daily: Master daily DataFrame (must have ``date``).

    Returns:
        DataFrame with new columns ``dow``, ``is_weekend``, ``is_holiday``,
        ``is_weekend_or_holiday``, ``month``.
    """
    daily = daily.copy()
    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"].isin([5, 6]).astype(int)
    daily["is_holiday"] = daily["date"].apply(
        lambda d: jpholiday.is_holiday(d.date())
    ).astype(int)
    daily["is_weekend_or_holiday"] = (
        (daily["is_weekend"] == 1) | (daily["is_holiday"] == 1)
    ).astype(int)
    daily["month"] = daily["date"].dt.month
    return daily


def add_weather_severity(
    daily: pd.DataFrame,
    *,
    precip_light: float = 0,
    precip_heavy: float = 10,
    wind_strong: float = 8,
) -> pd.DataFrame:
    """Compute a weather severity score (0–3).

    Scoring:
        0 = fine, 1 = light rain, 2 = heavy rain, 3 = stormy (rain + wind).

    Args:
        daily: Must have ``precip`` and ``wind`` columns.
        precip_light: Threshold for score +1.
        precip_heavy: Threshold for score +2.
        wind_strong: Threshold for score +1.

    Returns:
        DataFrame with new column ``weather_severity``.
    """
    daily = daily.copy()

    daily["weather_severity"] = (
        (daily["precip"] > precip_light).astype(int)
        + (daily["precip"] > precip_heavy).astype(int)
        + (daily["wind"] > wind_strong).astype(int)
    ).clip(upper=3)
    return daily


def add_rolling_features(
    daily: pd.DataFrame,
    col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling-mean columns for the given intent column.

    Args:
        daily: Master daily DataFrame.
        col: Column name to compute rolling means on.
        windows: Window sizes (default ``[3, 7, 14]``).

    Returns:
        DataFrame with new columns ``{col}_roll{w}`` for each window.
    """
    daily = daily.copy()
    for w in (windows or [3, 7, 14]):
        daily[f"{col}_roll{w}"] = daily[col].rolling(w, min_periods=1).mean()
    return daily


def add_lag_features(
    daily: pd.DataFrame,
    col: str,
    max_lag: int = 7,
) -> pd.DataFrame:
    """Add lagged columns for the intent column and weather.

    Args:
        daily: Must have ``col``, ``precip``, ``temp``.
        col: Column to lag.
        max_lag: Maximum lag in days (inclusive).

    Returns:
        DataFrame with ``{col}_lag0`` … ``{col}_lag{max_lag}`` plus
        ``precip_lag1`` and ``temp_lag1``.
    """
    daily = daily.copy()
    for lag in range(0, max_lag + 1):
        daily[f"{col}_lag{lag}"] = daily[col].shift(lag)
    daily["precip_lag1"] = daily["precip"].shift(1)
    daily["temp_lag1"] = daily["temp"].shift(1)
    return daily


def add_interaction_features(
    daily: pd.DataFrame,
    route_col: str,
) -> pd.DataFrame:
    """Add interaction terms.

    Args:
        daily: Must have ``is_weekend_or_holiday``, ``weather_severity``,
            and ``route_col``.
        route_col: Name of the RSI intent column.

    Returns:
        DataFrame with ``weekend_x_severity`` and ``weekend_x_intent``.
    """
    daily = daily.copy()
    daily["weekend_x_severity"] = (
        daily["is_weekend_or_holiday"] * daily["weather_severity"]
    )
    daily["weekend_x_intent"] = (
        daily["is_weekend_or_holiday"] * daily[route_col].fillna(0)
    )
    return daily


def add_dow_mean_encoding(daily: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week mean-encoded count.

    Args:
        daily: Must have ``dow`` and ``count``.

    Returns:
        DataFrame with ``dow_mean_count``.
    """
    daily = daily.copy()
    dow_means = daily.groupby("dow")["count"].mean()
    daily["dow_mean_count"] = daily["dow"].map(dow_means)
    return daily


# ── Convenience wrapper ──────────────────────────────────────────────────────

def build_features(
    daily: pd.DataFrame,
    route_col: str,
    reporter: Reporter,
    *,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full feature-engineering pipeline.

    Args:
        daily: Raw merged daily DataFrame.
        route_col: RSI intent column name.
        reporter: ``Reporter`` instance.
        cfg: Optional config for custom thresholds.

    Returns:
        Tuple of ``(daily_with_features, feature_col_names)``.
    """
    reporter.section(2, "Feature Engineering")

    thresholds = (cfg or {}).get("thresholds", {}).get("weather_severity", {})

    daily = add_calendar_features(daily)
    reporter.log(f"Weekend/Holiday days: {daily['is_weekend_or_holiday'].sum()} / {len(daily)}")

    daily = add_weather_severity(
        daily,
        precip_light=thresholds.get("precip_light", 0),
        precip_heavy=thresholds.get("precip_heavy", 10),
        wind_strong=thresholds.get("wind_strong", 8),
    )
    reporter.log(f"Weather severity distribution:\n"
                 f"{daily['weather_severity'].value_counts().sort_index().to_string()}")

    daily = add_rolling_features(daily, route_col)
    daily = add_lag_features(daily, route_col)
    daily = add_interaction_features(daily, route_col)
    daily = add_dow_mean_encoding(daily)

    # Log DOW averages
    reporter.log("\nDay-of-week average counts:")
    dow_means = daily.groupby("dow")["count"].mean()
    for dow, v in dow_means.items():
        name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(dow)]
        reporter.log(f"  {name}: {v:.1f}")

    # Correlation matrix
    corr_cols = [
        "count", route_col, f"{route_col}_lag2", f"{route_col}_roll7",
        "precip", "temp", "sun", "wind",
        "is_weekend_or_holiday", "weather_severity", "dow_mean_count",
    ]
    corr_cols = [c for c in corr_cols if c in daily.columns]
    corr_matrix = daily[corr_cols].corr()
    reporter.log("\nCorrelation with 'count':")
    for col in corr_cols:
        if col != "count":
            r = corr_matrix.loc["count", col]
            reporter.log(f"  {col:35s}  r = {r:+.3f}")

    # Define modelling feature columns
    feature_cols = [
        route_col,
        f"{route_col}_lag1", f"{route_col}_lag2", f"{route_col}_lag3",
        f"{route_col}_roll7",
        "precip", "temp", "sun", "wind",
        "precip_lag1",
        "is_weekend_or_holiday", "weather_severity",
        "dow_mean_count",
        "weekend_x_severity", "weekend_x_intent",
        "month",
    ]
    feature_cols = [c for c in feature_cols if c in daily.columns]

    return daily, feature_cols
