#!/usr/bin/env python3
"""src.run_analysis – Modular pipeline entry-point (replaces monolith).

Delegates analysis flow to ``src/`` modules.  Run from the repo root::

    python -m src.run_analysis                 # default config
    HTAG_CONFIG=custom.yaml python -m src.run_analysis  # custom config
"""

from __future__ import annotations

import logging
import os
import shutil
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.config import load_config
from src.report import Reporter
from src.validator import validate_pipeline
from src.data_loader import load_all_data
from src.feature_engineering import build_features
from src.models import fit_ols, fit_random_forest, robustness_suite, statistical_rigor
from src.economics import (
    compute_opportunity_gap,
    compute_lost_population,
    ranking_simulation,
    seasonal_sensitivity,
)
from src.spatial import cross_prefectural_ccf, multi_node_analysis
from src.kansei import (
    discomfort_index_analysis,
    overtourism_threshold,
    text_mine_undervibrancy,
)
from src import visualizer as viz
from src.latex_export import export_all_tables

logger = logging.getLogger(__name__)


def main() -> None:
    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Pipeline starting")

    # ── Configuration & Reporter ─────────────────────────────────────────
    cfg = load_config()
    rpt = Reporter(cfg)
    fig_dir = str(rpt.fig_dir)
    dpi = cfg.get("visualization", {}).get("dpi", 150)
    fig_num = 0

    # ══════════════════════════════════════════════════════════════════════
    # 0. DATA INTEGRITY VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    validation = validate_pipeline(cfg, rpt)
    logger.info(
        "Validation complete: %d sources, %d rows, passed=%s",
        len(validation.sources),
        validation.total_rows_audited,
        validation.overall_passed,
    )

    # ══════════════════════════════════════════════════════════════════════
    # 1. DATA LOADING
    # ══════════════════════════════════════════════════════════════════════
    data = load_all_data(cfg, rpt)
    daily = data["daily"]
    weather_daily = data["weather_daily"]
    rsi = data["rsi"]
    route_col = data["route_col"]
    survey_all = data["survey_all"]
    sat_all = data["sat_all"]
    text_all = data["text_all"]
    raw_survey = data["raw_survey"]

    # ══════════════════════════════════════════════════════════════════════
    # 2. FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════
    daily, feature_cols = build_features(daily, route_col, rpt)

    # Correlation matrix (used later for heatmap)
    numeric_cols = [c for c in ["count", route_col, "precip", "temp", "sun",
                                "wind", "is_weekend_or_holiday", "weather_severity"]
                    if c in daily.columns]
    corr_matrix = daily[numeric_cols].corr()

    # ══════════════════════════════════════════════════════════════════════
    # 3. MULTI-VARIABLE MODELLING
    # ══════════════════════════════════════════════════════════════════════
    rpt.section(3, "Multi-Variable Modelling (OLS + Random Forest)")

    # Build model DataFrame (drop rows with NaN in features)
    model_df = daily[["date", "count"] + feature_cols].dropna().copy()
    rpt.log(f"Model training rows (after dropna): {len(model_df)}")

    ols_result = fit_ols(model_df, feature_cols, rpt)
    rf_result = fit_random_forest(
        model_df, feature_cols, rpt,
        rf_params=cfg.get("model", {}).get("random_forest"),
    )

    # ══════════════════════════════════════════════════════════════════════
    # 4. OPPORTUNITY GAP
    # ══════════════════════════════════════════════════════════════════════
    daily = compute_opportunity_gap(daily, route_col, rpt)
    intent_median = daily[route_col].median()
    count_median = daily["count"].median()

    # ══════════════════════════════════════════════════════════════════════
    # 5. EXPLAINING THE NEGATIVE CORRELATION
    # ══════════════════════════════════════════════════════════════════════
    rpt.section(5, "Explaining the Negative Lag-2 Correlation")
    rpt.log("\nLag correlations (full data):")
    for lag in range(0, 8):
        col = f"{route_col}_lag{lag}"
        if col in daily.columns:
            r = daily[["count", col]].dropna().corr().iloc[0, 1]
            rpt.log(f"  lag {lag}: r = {r:+.3f}")

    for label, mask in [("Weekday", daily["is_weekend_or_holiday"] == 0),
                         ("Weekend/Holiday", daily["is_weekend_or_holiday"] == 1)]:
        sub = daily.loc[mask]
        r = sub[["count", f"{route_col}_lag2"]].dropna().corr().iloc[0, 1]
        rpt.log(f"  {label:20s}: r = {r:+.3f}  (n={len(sub)})")

    rpt.log("\nDay-of-week RSI intent vs count:")
    for dow in range(7):
        sub = daily[daily["dow"] == dow]
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
        rpt.log(f"  {day_name}: intent={sub[route_col].mean():7.1f}   "
                f"count={sub['count'].mean():7.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. VISUALISATIONS (Figs 1-7)
    # ══════════════════════════════════════════════════════════════════════
    rpt.section(6, "Generating Figures")
    y = model_df["count"].values

    fig_num += 1
    viz.plot_timeseries(
        daily, route_col,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_timeseries.png"),
        rpt, dpi=dpi)

    fig_num += 1
    viz.plot_correlation_heatmap(
        corr_matrix,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_correlation.png"),
        rpt, dpi=dpi)

    fig_num += 1
    viz.plot_feature_importance(
        rf_result.mdi_importance, rf_result.perm_importance,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_feature_importance.png"),
        rpt, dpi=dpi)

    fig_num += 1
    viz.plot_dow_boxplot(
        daily,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_dow_boxplot.png"),
        rpt, dpi=dpi)

    fig_num += 1
    viz.plot_rf_prediction(
        model_df["date"], y, rf_result.y_pred,
        rf_result.r2_train, rf_result.cv_r2_mean,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_rf_prediction.png"),
        rpt, dpi=dpi)

    fig_num += 1
    viz.plot_opportunity_gap(
        daily, route_col, intent_median, count_median,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_opportunity_gap.png"),
        rpt, dpi=dpi)

    fig_num += 1
    viz.plot_lag_correlations(
        daily, route_col,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_lag_correlations.png"),
        rpt, dpi=dpi)

    # ══════════════════════════════════════════════════════════════════════
    # 7. CROSS-PREFECTURAL SIGNAL
    # ══════════════════════════════════════════════════════════════════════
    ccf_data = cross_prefectural_ccf(daily, survey_all, rpt)
    best_lag = ccf_data["best_lag"]
    best_r = ccf_data["best_r"]

    fig_num += 1
    viz.plot_ccf(
        ccf_data["ccf_results"],
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_ishikawa_ccf.png"),
        rpt, dpi=dpi)

    # ══════════════════════════════════════════════════════════════════════
    # 8. KANSEI (EMOTIONAL) FEEDBACK LOOP
    # ══════════════════════════════════════════════════════════════════════
    kansei_data = overtourism_threshold(daily, sat_all, rpt)
    spear_r = kansei_data.get("spear_r", 0.0)
    spear_p = kansei_data.get("spear_p", 1.0)
    spear_r_nps = kansei_data.get("spear_r_nps", 0.0)
    spear_p_nps = kansei_data.get("spear_p_nps", 1.0)

    fig_num += 1
    if "sat_merged" in kansei_data and not kansei_data["sat_merged"].empty:
        fig9_path = os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_kansei_scatter.png")
        viz.plot_kansei_scatter(
            kansei_data["sat_merged"],
            fig9_path,
            rpt, dpi=dpi)

        # Backward compatibility: preserve legacy figure names used by reports.
        legacy_en = os.path.join(fig_dir, "ultimate_fig2_vibrancy_threshold.png")
        legacy_ja = os.path.join(fig_dir, "ultimate_fig2_vibrancy_threshold_ja.png")
        shutil.copyfile(fig9_path, legacy_en)
        shutil.copyfile(fig9_path.replace(".png", "_ja.png"), legacy_ja)
        rpt.log(f"  Saved {legacy_en} (compat)")
        rpt.log(f"  Saved {legacy_ja} (compat)")

    # ══════════════════════════════════════════════════════════════════════
    # 9. LOST POPULATION
    # ══════════════════════════════════════════════════════════════════════
    lost_data = compute_lost_population(model_df, ols_result.y_pred, daily, rpt)
    total_lost = lost_data["total_lost"]
    gap_model = lost_data["gap_model"]

    fig_num += 1
    if not gap_model.empty:
        viz.plot_lost_population(
            gap_model, total_lost,
            os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_lost_population.png"),
            rpt, dpi=dpi)

    # ══════════════════════════════════════════════════════════════════════
    # 10 + 12. MODEL ROBUSTNESS
    # ══════════════════════════════════════════════════════════════════════
    robust = robustness_suite(model_df, ols_result, feature_cols, rpt)

    # ══════════════════════════════════════════════════════════════════════
    # 10b. STATISTICAL RIGOR (効果量 – for Prof. Takemoto review)
    # ══════════════════════════════════════════════════════════════════════
    rigor = statistical_rigor(model_df, ols_result, feature_cols, rpt)

    # ══════════════════════════════════════════════════════════════════════
    # 13. RANKING SIMULATION (FUKUI RESURRECTION)
    # ══════════════════════════════════════════════════════════════════════
    ranking_data = ranking_simulation(
        total_lost, gap_model, rpt,
        ranking_cfg=cfg.get("ranking"),
    )
    sim_df = ranking_data["sim_df"]
    mean_actual_rank = ranking_data["mean_actual_rank"]
    mean_hypo_rank = ranking_data["mean_hypo_rank"]
    best_improvement = ranking_data["best_improvement"]

    # ══════════════════════════════════════════════════════════════════════
    # 14. SEASONAL SENSITIVITY
    # ══════════════════════════════════════════════════════════════════════
    seasonal = seasonal_sensitivity(model_df, feature_cols, rpt)

    # ══════════════════════════════════════════════════════════════════════
    # 15. TEXT MINING (UNDER-VIBRANCY)
    # ══════════════════════════════════════════════════════════════════════
    text_result = text_mine_undervibrancy(
        text_all, rpt,
        keywords=cfg.get("kansei", {}).get("undervibrancy_keywords"),
    )
    undervibrancy_hits = text_result.get("undervibrancy_hits", 0)
    pct = text_result.get("pct", 0)
    ratio_vs_high = text_result.get("ratio_vs_high", 0.0)

    # ══════════════════════════════════════════════════════════════════════
    # 16. FUKUI RESURRECTION CHART
    # ══════════════════════════════════════════════════════════════════════
    fig_num += 1
    viz.plot_resurrection(
        sim_df, total_lost, mean_actual_rank, mean_hypo_rank,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_fukui_resurrection.png"),
        rpt, dpi=dpi)

    # ══════════════════════════════════════════════════════════════════════
    # 11. HOKURIKU DEMAND HEATMAP
    # ══════════════════════════════════════════════════════════════════════
    fig_num += 1
    viz.plot_hokuriku_heatmap(
        survey_all,
        os.path.join(fig_dir, f"deep_analysis_fig{fig_num}_hokuriku_heatmap.png"),
        rpt, dpi=dpi)

    # ══════════════════════════════════════════════════════════════════════
    # BOLSTERED RESULTS + EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    top3_mdi = rf_result.mdi_importance.head(3)["feature"].tolist()
    top3_perm = rf_result.perm_importance.head(3)["feature"].tolist()
    n_outliers = int(daily["is_outlier"].sum()) if "is_outlier" in daily.columns else 0
    zero_days = 0  # already removed by data_loader

    mean_spending = 0.0
    if "spending_midpoint" in raw_survey.columns:
        mean_spending = raw_survey["spending_midpoint"].mean()
    total_yen_loss = abs(total_lost) * mean_spending if mean_spending else 0

    _write_bolstered(rpt, locals())
    _write_executive(rpt, locals())

    # ══════════════════════════════════════════════════════════════════════
    # 17. ECONOMIC WEIGHTING
    # ══════════════════════════════════════════════════════════════════════
    rpt.section(17, "Socio-Technical Governance: Economic Weighting & Weather Barriers")
    if mean_spending > 0:
        rpt.log(f"Mean Spending per Visitor: ¥{mean_spending:,.0f}")
        rpt.log(f"Total Annual Economic Revenue Loss: ¥{total_yen_loss:,.0f}")

    # Discomfort Index (NEW)
    di_result = discomfort_index_analysis(weather_daily, sat_all, rpt)

    # ══════════════════════════════════════════════════════════════════════
    # 20. MULTI-NODE SPATIAL GOVERNANCE
    # ══════════════════════════════════════════════════════════════════════
    try:
        spatial = multi_node_analysis(cfg, rsi, route_col, survey_all, rpt)
    except Exception as exc:
        logger.warning("multi_node_analysis failed (%s); continuing with empty spatial data.", exc)
        spatial = {"valid_nodes": {}, "node_count": 0}

    fig_num += 1
    viz.plot_spatial_friction(
        spatial.get("spatial_heat_df"),
        os.path.join(fig_dir, "spatial_friction_heatmap.png"),
        rpt, dpi=300)

    # ★ NEW: High-impact governance visualizations for grant applications
    fig_num += 1
    viz.plot_weather_shield_network(
        spatial.get("valid_nodes", {}),
        os.path.join(fig_dir, "weather_shield_network.png"),
        rpt, dpi=300)

    fig_num += 1
    viz.plot_rank_resurrection_projection(
        spatial.get("valid_nodes", {}),
        cfg.get("ranking", {}),
        os.path.join(fig_dir, "rank_resurrection_projection.png"),
        rpt, dpi=300)

    # ══════════════════════════════════════════════════════════════════════
    # METRICS EXPORT
    # ══════════════════════════════════════════════════════════════════════
    _write_metrics(rpt, locals(), spatial, cfg)

    # ══════════════════════════════════════════════════════════════════════
    # LaTeX TABLES
    # ══════════════════════════════════════════════════════════════════════
    results_bundle = {
        "ols": ols_result,
        "rf": rf_result,
        "robust": robust,
        "rigor": rigor,
        "economics": {"total_lost": total_lost},
        "spatial": spatial,
        "ccf": ccf_data,
        "seasonal": seasonal,
    }
    tex_paths = export_all_tables(results_bundle, str(rpt.out_dir))
    for p in tex_paths:
        rpt.log(f"  Saved LaTeX table: {p}")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════
    rpt.save()
    rpt.log("\n✓ Deep analysis complete.")


# ── Report Helpers ───────────────────────────────────────────────────────────

def _write_bolstered(rpt: Reporter, ctx: dict) -> None:
    """Write the bolstered results block to metrics."""
    top3_mdi = ctx["top3_mdi"]
    top3_perm = ctx["top3_perm"]
    ols = ctx["ols_result"]
    rf = ctx["rf_result"]
    robust = ctx["robust"]
    daily = ctx["daily"]
    model_df = ctx["model_df"]
    gap_model = ctx["gap_model"]
    total_lost = ctx["total_lost"]

    rpt.metrics("=" * 80)
    rpt.metrics("BOLSTERED RESULTS – Tojinbo Demand Forecast Model")
    rpt.metrics(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    rpt.metrics("=" * 80)

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. CORE MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Training Data:         {len(model_df)} days
  Camera Data Range:     {daily['date'].min().date()} → {daily['date'].max().date()}

  OLS R²           = {ols.r2:.4f}  (Adj R² = {ols.adj_r2:.4f})
  RF Train R²      = {rf.r2_train:.4f}
  RF 5-fold CV R²  = {rf.cv_r2_mean:.4f} ± {rf.cv_r2_std:.4f}

  Top 3 Predictors (MDI): {', '.join(top3_mdi)}
  Top 3 Predictors (PI):  {', '.join(top3_perm)}
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2. CROSS-PREFECTURAL SIGNAL (Ishikawa → Fukui Pipeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Best Ishikawa → Tojinbo lag:  {ctx['best_lag']:+d} day(s)
  Correlation at best lag:      r = {ctx['best_r']:+.3f}
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3. KANSEI (EMOTIONAL) OVERTOURISM THRESHOLD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Visitors vs Satisfaction (Spearman): r = {ctx['spear_r']:+.3f}, p = {ctx['spear_p']:.4f}
  Visitors vs NPS (Spearman):         r = {ctx['spear_r_nps']:+.3f}, p = {ctx['spear_p_nps']:.4f}
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4. LOST POPULATION (The "Satake Number")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Opportunity Gap days:             {len(gap_model)}
  Total Lost Visitors:              {total_lost:,.0f}
  Mean Lost per day:                {total_lost / max(len(gap_model), 1):,.0f}
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5. MODEL ROBUSTNESS (PhD Shield)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Durbin-Watson (original):  {robust.dw_stat:.3f}  ({'CLEAN' if robust.dw_clean else 'WARNING'})
  Durbin-Watson (1st-diff):  {robust.fd_dw:.3f}  ({'CLEAN' if robust.fd_dw_clean else 'residual'})
  First-Difference R²:       {robust.fd_r2:.4f}  (original: {ols.r2:.4f})
  LDV R²:                    {robust.ldv_r2:.4f}
  LDV DW:                    {robust.ldv_dw:.3f}
  Newey-West sig. predictors: {robust.nw_sig_count}
  Weather Data Value:   +{robust.weather_value:.4f} R²
  OLS R² without JMA:   {robust.r2_no_weather:.4f}
  OLS R² with JMA:      {ols.r2:.4f}
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  6. RANKING IMPACT SIMULATION (Fukui Resurrection)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total lost visitors:        {total_lost:,.0f}
  Winter actual mean rank:    {ctx['mean_actual_rank']:.1f}
  Winter hypothetical rank:   {ctx['mean_hypo_rank']:.1f}
  Best monthly improvement:   {ctx['best_improvement']} ranks
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  7. SEASONAL WEATHER SENSITIVITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Summer weather lift (ΔR²):  {ctx['seasonal'].get('summer_lift', 0):+.4f}
  Winter weather lift (ΔR²):  {ctx['seasonal'].get('winter_lift', 0):+.4f}
  Ratio (Winter/Summer):      {ctx['seasonal'].get('ratio', 0):.2f}x
""")

    rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  8. QUALITATIVE UNDER-VIBRANCY LINK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Under-vibrancy mentions in 1-2 star reviews: {ctx['undervibrancy_hits']}
  Percentage of low-sat responses:             {ctx['pct']:.1f}%
  Ratio vs high-satisfaction visitors:         {ctx['ratio_vs_high']:.1f}x
""")

    rigor = ctx.get("rigor")
    if rigor is not None:
        top_beta = rigor.beta_coefficients.sort_values(key=abs, ascending=False)
        top_str = ", ".join(f"{f}={v:+.3f}" for f, v in top_beta.head(3).items())
        rpt.metrics(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  9. STATISTICAL RIGOR (効果量 / Prof. Takemoto Review)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Top-3 Standardised β: {top_str}
  Cohen's f²:           {rigor.cohens_f2:.4f}
  Train N:              {rigor.train_n}  |  Hold-out N: {rigor.holdout_n}
  Hold-out MAE:         {rigor.holdout_mae:.1f} visitors/day
  Hold-out RMSE:        {rigor.holdout_rmse:.1f} visitors/day
  Hold-out R²:          {rigor.holdout_r2:.4f}
""")

    rpt.metrics("=" * 80)
    rpt.metrics("END OF BOLSTERED RESULTS")
    rpt.metrics("=" * 80)


def _write_executive(rpt: Reporter, ctx: dict) -> None:
    """Write executive summary to report."""
    rpt.section("", "EXECUTIVE SUMMARY")
    rpt.log(f"""
Key Findings:

1. WHAT DRIVES TOJINBO VISITORS?
   Top 3 features (RF MDI):   {', '.join(ctx['top3_mdi'])}
   Top 3 features (Perm Imp): {', '.join(ctx['top3_perm'])}

2. MODEL PERFORMANCE
   OLS R²     = {ctx['ols_result'].r2:.3f}
   RF CV R²   = {ctx['rf_result'].cv_r2_mean:.3f} ± {ctx['rf_result'].cv_r2_std:.3f}

3. CROSS-PREFECTURAL PIPELINE
   Best Ishikawa → Tojinbo lag: {ctx['best_lag']:+d} day(s), r = {ctx['best_r']:+.3f}

4. KANSEI OVERTOURISM
   Visitors vs Satisfaction: r = {ctx['spear_r']:+.3f} (p = {ctx['spear_p']:.4f})

5. LOST POPULATION ("SATAKE NUMBER")
   {len(ctx['gap_model'])} gap days → {abs(ctx['total_lost']):,.0f} visitors lost

6. MODEL ROBUSTNESS
   DW (original): {ctx['robust'].dw_stat:.3f}  DW (1st-diff): {ctx['robust'].fd_dw:.3f}
   Weather data value: +{ctx['robust'].weather_value:.4f} R²

7. FUKUI RESURRECTION
   Winter actual rank:  ~{ctx['mean_actual_rank']:.0f}th
   Winter hypothetical: ~{ctx['mean_hypo_rank']:.0f}th

8. SEASONAL SENSITIVITY
   Ratio: {ctx['seasonal'].get('ratio', 0):.2f}x → Winter is more weather-sensitive

9. UNDER-VIBRANCY
   {ctx['undervibrancy_hits']} low-satisfaction responses ({ctx['pct']:.1f}%)
""")


def _write_metrics(rpt: Reporter, ctx: dict, spatial: dict, cfg: dict) -> None:
    """Write machine-readable metrics for multi-node section."""
    rpt.metrics("MULTI_NODE_SPATIAL_GOVERNANCE")
    rpt.metrics(f"  Node_C=Katsuyama ({spatial.get('node_c_source', 'unknown')})")

    for nm, m in spatial.get("valid_nodes", {}).items():
        rpt.metrics(f"{nm}")
        rpt.metrics(f"  n={m['n']}")
        rpt.metrics(f"  OLS_R2={m['r2']:.6f}")
        rpt.metrics(f"  OLS_Adj_R2={m['adj_r2']:.6f}")
        rpt.metrics(f"  Weather_Lift_R2={m['weather_lift']:+.6f}")
        rpt.metrics(f"  Snow_Beta_Std={m['snow_beta_std']:+.6f}")
        rpt.metrics(f"  Opportunity_Lost_Visitors={m['lost_visitors']:.2f}")
        rpt.metrics(f"  Opportunity_Lost_Yen={m['lost_yen']:.2f}")

    nudge = spatial.get("nudge", {})
    if nudge:
        rpt.metrics("Atmospheric_Nudge_MikuniWindGT10")
        rpt.metrics(f"  HighWindDays={nudge.get('n_high', 0)}")
        for key in ("tojinbo_delta_pct", "fukui_delta_pct", "katsuyama_delta_pct"):
            if key in nudge:
                rpt.metrics(f"  {key}={nudge[key]:+.4f}")

    spending = cfg["economics"]["spending_per_visitor_yen"]
    node_count = spatial.get("node_count", 3)
    
    # ★ UPDATED: 4-Node Satake Number
    rpt.metrics(f"FourNode_Satake_Number (nodes={node_count})")
    rpt.metrics(f"  Node_D_Status={spatial.get('node_d_source', 'not_available')}")
    rpt.metrics(f"  Lost_Visitors={spatial.get('satake_lost_visitors', 0):.2f}")
    rpt.metrics(f"  Spending_Per_Visitor_Yen={spending:.2f}")
    rpt.metrics(f"  Total_Lost_Yen={spatial.get('satake_yen', 0):.2f}")
    satake_billion = spatial.get('satake_yen', 0) / 1e9
    rpt.metrics(f"  Total_Lost_Billion_Yen={satake_billion:.3f}")
    if node_count >= 4:
        rpt.metrics("  ★ GEOGRAPHIC_SATURATION=ACHIEVED (North/Central/South/East)")
    
    for nm, lag, r, n in spatial.get("ishikawa_lag_results", []):
        rpt.metrics(f"  {nm}: lag={lag:+d}, r={r:+.6f}, n={n}")


if __name__ == "__main__":
    main()
