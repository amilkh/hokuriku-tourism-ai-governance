"""Predictive models and robustness checks.

Provides OLS regression, Random Forest with permutation importance, and a
full robustness suite (Durbin–Watson, Newey–West, first-difference,
lagged-dependent-variable, VIF, and weather-removal sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from .report import Reporter


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class OLSResult:
    """Container for OLS regression results."""
    model: Any  # sm.OLS RegressionResultsWrapper
    r2: float
    adj_r2: float
    feature_cols: list[str]
    y_pred: np.ndarray


@dataclass
class RFResult:
    """Container for Random Forest results."""
    model: RandomForestRegressor
    r2_train: float
    mae_train: float
    cv_r2_mean: float
    cv_r2_std: float
    y_pred: np.ndarray
    mdi_importance: pd.DataFrame
    perm_importance: pd.DataFrame


@dataclass
class StatisticalRigorResult:
    """Container for statistical rigor metrics (Prof. Takemoto 効果量 review)."""
    beta_coefficients: pd.Series   # standardised OLS betas (feature_cols only)
    cohens_f2: float               # global Cohen's f² for the full OLS model
    train_n: int
    holdout_n: int
    holdout_mae: float
    holdout_rmse: float
    holdout_r2: float


@dataclass
class RobustnessResult:
    """Container for robustness checks."""
    dw_stat: float
    dw_clean: bool
    nw_sig_count: int
    fd_r2: float
    fd_dw: float
    fd_dw_clean: bool
    ldv_r2: float
    ldv_dw: float
    weather_value: float
    r2_no_weather: float
    vif: pd.DataFrame | None = None


# ── OLS ──────────────────────────────────────────────────────────────────────

def fit_ols(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    reporter: Reporter,
) -> OLSResult:
    """Fit an OLS regression model.

    Args:
        model_df: DataFrame with ``count`` and all ``feature_cols``.
        feature_cols: List of feature column names.
        reporter: ``Reporter`` instance.

    Returns:
        ``OLSResult`` with the fitted model, metrics, and predictions.
    """
    reporter.log("\n--- OLS Regression ---")
    X = model_df[feature_cols].values
    y = model_df["count"].values
    X_ols = sm.add_constant(X)

    ols_model = sm.OLS(y, X_ols).fit()
    reporter.log(ols_model.summary().as_text())

    r2 = ols_model.rsquared
    adj_r2 = ols_model.rsquared_adj
    reporter.log(f"\nOLS R²  = {r2:.4f}")
    reporter.log(f"OLS Adj R² = {adj_r2:.4f}")

    reporter.log("\nOLS Significant predictors (p < 0.05):")
    for i, col in enumerate(["const"] + feature_cols):
        p = ols_model.pvalues[i]
        coef = ols_model.params[i]
        if p < 0.05:
            reporter.log(f"  {col:35s}  coef={coef:+10.3f}  p={p:.4f} ***")

    return OLSResult(
        model=ols_model,
        r2=r2,
        adj_r2=adj_r2,
        feature_cols=feature_cols,
        y_pred=ols_model.predict(X_ols),
    )


# ── Random Forest ────────────────────────────────────────────────────────────

def fit_random_forest(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    reporter: Reporter,
    *,
    rf_params: dict[str, Any] | None = None,
    cv_folds: int = 5,
) -> RFResult:
    """Fit a Random Forest regressor with cross-validation.

    Args:
        model_df: DataFrame with ``count`` and all ``feature_cols``.
        feature_cols: List of feature column names.
        reporter: ``Reporter`` instance.
        rf_params: Hyperparameters for ``RandomForestRegressor``.
        cv_folds: Number of CV folds.

    Returns:
        ``RFResult`` with model, metrics, and both importance DataFrames.
    """
    reporter.log("\n--- Random Forest Regressor ---")
    params = rf_params or {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    X = model_df[feature_cols].values
    y = model_df["count"].values

    rf = RandomForestRegressor(**params)
    rf.fit(X, y)
    y_pred = rf.predict(X)

    r2_train = r2_score(y, y_pred)
    mae_train = mean_absolute_error(y, y_pred)
    cv_scores = cross_val_score(rf, X, y, cv=cv_folds, scoring="r2")

    reporter.log(f"RF Train R²      = {r2_train:.4f}")
    reporter.log(f"RF Train MAE     = {mae_train:.1f}")
    reporter.log(f"RF {cv_folds}-fold CV R²  = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # MDI importance
    mdi = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    reporter.log("\nRandom Forest Feature Importance (MDI):")
    for _, row in mdi.iterrows():
        bar = "█" * int(row["importance"] * 100)
        reporter.log(f"  {row['feature']:35s}  {row['importance']:.4f}  {bar}")

    # Permutation importance
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    reporter.log("\nPermutation Importance:")
    for _, row in perm_df.iterrows():
        reporter.log(f"  {row['feature']:35s}  {row['importance_mean']:+.4f} "
                     f"± {row['importance_std']:.4f}")

    return RFResult(
        model=rf,
        r2_train=r2_train,
        mae_train=mae_train,
        cv_r2_mean=cv_scores.mean(),
        cv_r2_std=cv_scores.std(),
        y_pred=y_pred,
        mdi_importance=mdi,
        perm_importance=perm_df,
    )


# ── Robustness Suite ─────────────────────────────────────────────────────────

def robustness_suite(
    model_df: pd.DataFrame,
    ols_result: OLSResult,
    feature_cols: list[str],
    reporter: Reporter,
) -> RobustnessResult:
    """Run Durbin–Watson, Newey–West, first-diff, LDV, VIF, and sensitivity.

    Args:
        model_df: Modelling DataFrame.
        ols_result: Fitted ``OLSResult``.
        feature_cols: Feature column names.
        reporter: ``Reporter`` instance.

    Returns:
        ``RobustnessResult`` with all diagnostic metrics.
    """
    X = model_df[feature_cols].values
    y = model_df["count"].values
    X_ols = sm.add_constant(X)

    reporter.section(10, "Model Robustness (PhD Shield)")

    # Durbin-Watson
    dw_stat = durbin_watson(ols_result.model.resid)
    dw_clean = 1.5 <= dw_stat <= 2.5
    reporter.log(f"\n★ Durbin-Watson statistic: {dw_stat:.3f}")
    reporter.log(f"  → {'CLEAN' if dw_clean else 'WARNING'}: "
                 f"{'No significant autocorrelation' if dw_clean else 'Autocorrelation detected'}")

    # Sensitivity: remove weather features
    reporter.log("\n--- Sensitivity Analysis: Value of JMA Weather Data ---")
    weather_feats = {"precip", "temp", "sun", "wind", "precip_lag1",
                     "weather_severity", "weekend_x_severity"}
    non_weather = [f for f in feature_cols if f not in weather_feats]

    X_nw = model_df[non_weather].dropna().values
    y_nw = model_df.loc[model_df[non_weather].dropna().index, "count"].values
    ols_nw = sm.OLS(y_nw, sm.add_constant(X_nw)).fit()
    r2_no_weather = ols_nw.rsquared
    weather_value = ols_result.r2 - r2_no_weather

    reporter.log(f"  OLS R² with weather:    {ols_result.r2:.4f}")
    reporter.log(f"  OLS R² without weather: {r2_no_weather:.4f}")
    reporter.log(f"  ★ Weather data value:   +{weather_value:.4f} R²")

    # VIF
    reporter.log("\n--- Variance Inflation Factors (VIF) ---")
    X_vif = model_df[feature_cols].dropna()
    vif_rows = []
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(sm.add_constant(X_vif.values), i + 1)
            flag = " ⚠ HIGH" if vif > 10 else ""
            reporter.log(f"  {col:35s} {vif:8.1f}{flag}")
            vif_rows.append({"feature": col, "vif": vif})
        except Exception:
            reporter.log(f"  {col:35s}     N/A")

    vif_df = pd.DataFrame(vif_rows) if vif_rows else None

    # Newey–West
    reporter.section(12, "Autocorrelation Fix (PhD Robustness)")
    reporter.log("\n--- 12a. OLS with Newey-West (HAC) Standard Errors ---")
    nw_maxlags = max(1, int(0.75 * len(y) ** (1 / 3)))
    ols_nw_hac = sm.OLS(y, X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": nw_maxlags})
    reporter.log(f"  Newey-West bandwidth: {nw_maxlags}")

    nw_sig = 0
    for i, col in enumerate(["const"] + feature_cols):
        p_nw = ols_nw_hac.pvalues[i]
        if p_nw < 0.05:
            nw_sig += 1
            reporter.log(f"    {col:35s}  p_NW={p_nw:.4f}")
    reporter.log(f"  Total NW-significant: {nw_sig}")

    # First-Difference
    reporter.log("\n--- 12b. First-Difference Model ---")
    diff_df = model_df[["date", "count"] + feature_cols].sort_values("date").reset_index(drop=True)
    diff_df = diff_df.copy()
    diff_df["count"] = diff_df["count"].diff()
    diff_df[feature_cols] = diff_df[feature_cols].diff()
    diff_clean = diff_df.dropna(subset=["count"] + feature_cols)
    diff_y = diff_clean["count"].values
    diff_X = diff_clean[feature_cols].values

    ols_fd = sm.OLS(diff_y, sm.add_constant(diff_X)).fit()
    fd_r2 = ols_fd.rsquared
    fd_dw = durbin_watson(ols_fd.resid)
    fd_dw_clean = 1.5 <= fd_dw <= 2.5
    reporter.log(f"  First-Diff R²: {fd_r2:.4f}  DW: {fd_dw:.3f} "
                 f"({'CLEAN' if fd_dw_clean else 'residual autocorrelation'})")

    # Lagged Dependent Variable
    reporter.log("\n--- 12c. Lagged Dependent Variable (LDV) ---")
    ldv_df = model_df[["date", "count"] + feature_cols].sort_values("date").reset_index(drop=True)
    ldv_df["count_lag1"] = ldv_df["count"].shift(1)
    ldv_clean = ldv_df.dropna()
    ldv_feats = feature_cols + ["count_lag1"]
    ols_ldv = sm.OLS(
        ldv_clean["count"].values,
        sm.add_constant(ldv_clean[ldv_feats].values),
    ).fit()
    ldv_r2 = ols_ldv.rsquared
    ldv_dw = durbin_watson(ols_ldv.resid)
    reporter.log(f"  LDV R²: {ldv_r2:.4f}  DW: {ldv_dw:.3f}")
    reporter.log(f"  count_lag1 coef = {ols_ldv.params[-1]:+.4f}, "
                 f"p = {ols_ldv.pvalues[-1]:.4f}")

    return RobustnessResult(
        dw_stat=dw_stat,
        dw_clean=dw_clean,
        nw_sig_count=nw_sig,
        fd_r2=fd_r2,
        fd_dw=fd_dw,
        fd_dw_clean=fd_dw_clean,
        ldv_r2=ldv_r2,
        ldv_dw=ldv_dw,
        weather_value=weather_value,
        r2_no_weather=r2_no_weather,
        vif=vif_df,
    )


# ── Statistical Rigor (効果量 / Effect Size) ──────────────────────────────────

def statistical_rigor(
    model_df: pd.DataFrame,
    ols_result: OLSResult,
    feature_cols: list[str],
    reporter: Reporter,
    *,
    train_pct: float = 0.80,
) -> StatisticalRigorResult:
    """Compute effect sizes and out-of-sample predictive validity.

    Produces:
    - Standardised beta coefficients (β_std = coef × σ_x / σ_y)
    - Cohen's f² for the full OLS model
    - Time-series 80/20 train-test split MAE, RMSE, and R²

    Args:
        model_df: Modelling DataFrame sorted by ``date``.
        ols_result: Fitted ``OLSResult`` on the full training set.
        feature_cols: Feature column names.
        reporter: ``Reporter`` instance.
        train_pct: Fraction of rows (chronological) used for training.

    Returns:
        ``StatisticalRigorResult``.
    """
    reporter.section("SR", "Statistical Rigor – Effect Size & Predictive Validity")

    # ── 1. Standardised beta coefficients ────────────────────────────────────
    reporter.log("\n--- Standardised Coefficients (β) ---")
    X = model_df[feature_cols]
    y = model_df["count"]

    std_x = X.std()
    std_y = y.std()

    # Raw OLS coefficients (features only – skip constant at index 0)
    raw_coefs = ols_result.model.params[1:]  # excludes const
    if not isinstance(raw_coefs, pd.Series):
        raw_coefs = pd.Series(raw_coefs, index=feature_cols)

    beta = raw_coefs * (std_x / std_y)
    beta.name = "beta_std"

    reporter.log(f"  {'Feature':35s}  {'β (std)':>10}  {'|β|':>8}")
    reporter.log(f"  {'-'*35}  {'-'*10}  {'-'*8}")
    for feat, b in beta.sort_values(key=abs, ascending=False).items():
        reporter.log(f"  {feat:35s}  {b:+10.4f}  {abs(b):8.4f}")

    # ── 2. Cohen's f² ────────────────────────────────────────────────────────
    r2 = ols_result.r2
    cohens_f2 = r2 / (1.0 - r2) if r2 < 1.0 else float("inf")
    magnitude = (
        "large (≥0.35)" if cohens_f2 >= 0.35 else
        "medium (≥0.15)" if cohens_f2 >= 0.15 else
        "small (≥0.02)" if cohens_f2 >= 0.02 else
        "negligible (<0.02)"
    )
    reporter.log(f"\n--- Cohen's f² ---")
    reporter.log(f"  f² = R² / (1 − R²) = {r2:.4f} / {1 - r2:.4f} = {cohens_f2:.4f}")
    reporter.log(f"  Effect magnitude: {magnitude}")

    # ── 3. Time-series train-test split ──────────────────────────────────────
    reporter.log(f"\n--- Out-of-Sample Validation (train={train_pct:.0%} / test={1-train_pct:.0%}) ---")

    sorted_df = model_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_pct)
    train_df = sorted_df.iloc[:split_idx]
    holdout_df = sorted_df.iloc[split_idx:]

    train_n = len(train_df)
    holdout_n = len(holdout_df)

    reporter.log(f"  Train period:   {train_df['date'].min().date()} → {train_df['date'].max().date()} (n={train_n})")
    reporter.log(f"  Holdout period: {holdout_df['date'].min().date()} → {holdout_df['date'].max().date()} (n={holdout_n})")

    X_train = sm.add_constant(train_df[feature_cols].values, has_constant="add")
    y_train = train_df["count"].values
    train_model = sm.OLS(y_train, X_train).fit()

    X_hold = sm.add_constant(holdout_df[feature_cols].values, has_constant="add")
    y_hold = holdout_df["count"].values
    y_pred_hold = train_model.predict(X_hold)

    mae = mean_absolute_error(y_hold, y_pred_hold)
    rmse = float(np.sqrt(mean_squared_error(y_hold, y_pred_hold)))
    holdout_r2 = r2_score(y_hold, y_pred_hold)

    reporter.log(f"\n  Holdout MAE:   {mae:.1f} visitors/day")
    reporter.log(f"  Holdout RMSE:  {rmse:.1f} visitors/day")
    reporter.log(f"  Holdout R²:    {holdout_r2:.4f}")
    reporter.log(f"\n  Interpretation: The model predicts unseen dates within ±{mae:.0f} visitors/day on average.")

    return StatisticalRigorResult(
        beta_coefficients=beta,
        cohens_f2=cohens_f2,
        train_n=train_n,
        holdout_n=holdout_n,
        holdout_mae=mae,
        holdout_rmse=rmse,
        holdout_r2=holdout_r2,
    )
