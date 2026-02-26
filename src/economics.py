"""Economics module – Opportunity gap, lost population, ranking simulation.

Quantifies the economic impact of weather-driven planning friction and
simulates ranking improvements from demand recovery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .report import Reporter


# ── Opportunity Gap ──────────────────────────────────────────────────────────

def compute_opportunity_gap(
    daily: pd.DataFrame,
    route_col: str,
    reporter: Reporter,
) -> pd.DataFrame:
    """Flag days with high intent but low arrivals.

    An **Opportunity Gap** day has above-median RSI intent AND
    below-median visitor count — the demand signal exists but visitors
    did not materialise.

    Args:
        daily: Master daily DataFrame.
        route_col: RSI intent column.
        reporter: ``Reporter``.

    Returns:
        ``daily`` with new columns ``high_intent``, ``low_count``,
        ``opportunity_gap``.
    """
    reporter.section(4, "Opportunity Gap Analysis")
    daily = daily.copy()

    intent_med = daily[route_col].median()
    count_med = daily["count"].median()

    daily["high_intent"] = (daily[route_col] > intent_med).astype(int)
    daily["low_count"] = (daily["count"] < count_med).astype(int)
    daily["opportunity_gap"] = (daily["high_intent"] & daily["low_count"]).astype(int)

    gap_days = daily[daily["opportunity_gap"] == 1]
    reporter.log(f"\nOpportunity Gap days: {len(gap_days)} / {len(daily)}")
    reporter.log(f"  Intent median: {intent_med:.0f}   Count median: {count_med:.0f}")

    if len(gap_days) > 0:
        reporter.log(f"\nTop 10 gap days:")
        for _, row in gap_days.sort_values(route_col, ascending=False).head(10).iterrows():
            reporter.log(f"  {row['date'].date()!s:12s}  count={row['count']:.0f}  "
                         f"intent={row[route_col]:.0f}  precip={row['precip']:.1f}")

    # Characterise
    reporter.log("\nGap vs non-gap comparison:")
    for col in ("precip", "weather_severity", "is_weekend_or_holiday", "temp", "wind"):
        if col in daily.columns:
            g = daily.loc[daily["opportunity_gap"] == 1, col].mean()
            n = daily.loc[daily["opportunity_gap"] == 0, col].mean()
            reporter.log(f"  {col:25s}  gap={g:7.2f}  non-gap={n:7.2f}  Δ={g - n:+.2f}")

    return daily


# ── Lost Population ──────────────────────────────────────────────────────────

def compute_lost_population(
    model_df: pd.DataFrame,
    ols_y_pred: np.ndarray,
    daily: pd.DataFrame,
    reporter: Reporter,
) -> dict[str, Any]:
    """Estimate visitors lost to planning friction.

    Lost Population = OLS-predicted visitors − actual visitors on gap days.

    Args:
        model_df: Modelling DataFrame with ``date`` and ``count``.
        ols_y_pred: OLS predicted values (aligned with ``model_df``).
        daily: Must have ``opportunity_gap`` column.
        reporter: ``Reporter``.

    Returns:
        Dict with ``total_lost``, ``mean_lost``, ``gap_model`` DataFrame.
    """
    reporter.section(9, "Lost Population: Quantifying the Opportunity Gap")

    model_df = model_df.copy()
    model_df["ols_predicted"] = ols_y_pred
    model_df = model_df.merge(
        daily[["date", "opportunity_gap"]].drop_duplicates(),
        on="date", how="left",
    )
    model_df["opportunity_gap"] = model_df["opportunity_gap"].fillna(0).astype(int)

    gap_model = model_df[model_df["opportunity_gap"] == 1].copy()
    reporter.log(f"\nGap days in model: {len(gap_model)} / {len(model_df)}")

    if len(gap_model) == 0:
        reporter.log("  No opportunity gap days found.")
        return {"total_lost": 0, "gap_model": gap_model}

    gap_model["lost_population"] = gap_model["ols_predicted"] - gap_model["count"]
    total = gap_model["lost_population"].sum()
    mean = gap_model["lost_population"].mean()

    reporter.log(f"\n★ LOST POPULATION:")
    reporter.log(f"  Total lost visitors: {total:,.0f}")
    reporter.log(f"  Mean per gap day:    {mean:,.0f}")

    return {
        "total_lost": total,
        "mean_lost": mean,
        "gap_model": gap_model,
    }


# ── Ranking Simulation ──────────────────────────────────────────────────────

def ranking_simulation(
    total_lost: float,
    gap_model: pd.DataFrame,
    reporter: Reporter,
    *,
    ranking_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate national ranking improvement from recovered visitors.

    Args:
        total_lost: Total lost visitors from ``compute_lost_population``.
        gap_model: Gap days DataFrame with ``date`` and ``lost_population``.
        reporter: ``Reporter``.
        ranking_cfg: Config section with ranking baselines.

    Returns:
        Dict with ``sim_df``, ``mean_actual_rank``, ``mean_hypo_rank``,
        ``best_improvement``.
    """
    reporter.section(13, "Ranking Impact Simulation (Fukui Resurrection)")

    cfg = ranking_cfg or {}
    rank_2025 = cfg.get("fukui_rank_2025",
                         [47, 47, 47, 46, 47, 47, 45, 35, 46, 47, 47, 47])
    visitors_k = cfg.get("fukui_visitors_k",
                          [85, 78, 110, 130, 145, 95, 160, 220, 140, 120, 100, 88])
    gap_41_k = cfg.get("gap_to_rank41_k",
                        [35, 40, 30, 25, 20, 38, 15, 0, 22, 30, 35, 38])

    sim_df = pd.DataFrame({
        "month": list(range(1, 13)),
        "fukui_rank_2025": rank_2025,
        "fukui_visitors_k": visitors_k,
        "gap_to_rank41_k": gap_41_k,
    })

    # Monthly distribution of lost visitors
    if len(gap_model) > 0 and "date" in gap_model.columns:
        gm = gap_model.copy()
        gm["month"] = gm["date"].dt.month
        monthly = gm.groupby("month")["lost_population"].sum().reset_index()
        monthly.columns = ["month", "monthly_lost"]
    else:
        monthly = pd.DataFrame({
            "month": range(1, 13),
            "monthly_lost": [total_lost / 12] * 12,
        })

    sim_df = sim_df.merge(monthly, on="month", how="left")
    sim_df["monthly_lost"] = sim_df["monthly_lost"].fillna(0)

    # Estimate rank gains
    sim_df["ranks_gained"] = 0
    for idx, row in sim_df.iterrows():
        extra_k = row["monthly_lost"] / 1000
        gap_k = row["gap_to_rank41_k"]
        if gap_k > 0 and extra_k > 0:
            rank_per_k = 6.0 / gap_k
            sim_df.at[idx, "ranks_gained"] = min(int(extra_k * rank_per_k), 10)

    sim_df["hypo_rank"] = (sim_df["fukui_rank_2025"] - sim_df["ranks_gained"]).clip(lower=1)

    # Summary
    winter = sim_df[sim_df["month"].isin([1, 2, 12])]
    mean_actual = winter["fukui_rank_2025"].mean()
    mean_hypo = winter["hypo_rank"].mean()
    best = sim_df["ranks_gained"].max()

    reporter.log(f"\n★ RESURRECTION SUMMARY:")
    reporter.log(f"  Winter actual mean rank:  {mean_actual:.1f}")
    reporter.log(f"  Winter hypothetical rank: {mean_hypo:.1f}")
    reporter.log(f"  Best monthly improvement: {best} ranks")

    return {
        "sim_df": sim_df,
        "mean_actual_rank": mean_actual,
        "mean_hypo_rank": mean_hypo,
        "best_improvement": best,
    }


# ── Seasonal Sensitivity ────────────────────────────────────────────────────

def seasonal_sensitivity(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    reporter: Reporter,
) -> dict[str, Any]:
    """Compare weather sensitivity between summer and winter.

    Args:
        model_df: Modelling DataFrame (must have ``month``).
        feature_cols: Feature column names.
        reporter: ``Reporter``.

    Returns:
        Dict with ``summer_lift``, ``winter_lift``, ``ratio``.
    """
    reporter.section(14, "Seasonal Weather Sensitivity Test")

    weather_feats = {"precip", "temp", "sun", "wind", "precip_lag1",
                     "weather_severity", "weekend_x_severity"}
    non_weather = [f for f in feature_cols if f not in weather_feats]

    result: dict[str, Any] = {"summer_lift": 0, "winter_lift": 0, "ratio": 0}

    for name, months in [("SUMMER (Jun-Aug)", [6, 7, 8]),
                          ("WINTER (Dec-Feb)", [12, 1, 2])]:
        sub = model_df[model_df["month"].isin(months)].dropna(subset=feature_cols)
        reporter.log(f"\n--- {name}: {len(sub)} days ---")
        if len(sub) < 15:
            reporter.log(f"  ⚠ Too few days (n={len(sub)})")
            continue

        y = sub["count"].values
        try:
            r2_full = sm.OLS(y, sm.add_constant(sub[feature_cols].values)).fit().rsquared
            r2_nw = sm.OLS(y, sm.add_constant(sub[non_weather].values)).fit().rsquared
            lift = r2_full - r2_nw
            reporter.log(f"  R² full: {r2_full:.4f}  R² no-weather: {r2_nw:.4f}  Lift: {lift:+.4f}")

            if "SUMMER" in name:
                result["summer_lift"] = lift
            else:
                result["winter_lift"] = lift
        except Exception as exc:
            reporter.log(f"  Error: {exc}")

    s = result["summer_lift"]
    w = result["winter_lift"]
    result["ratio"] = w / s if s > 0 else float("inf")
    reporter.log(f"\n★ Ratio (Winter/Summer): {result['ratio']:.2f}x")

    return result
