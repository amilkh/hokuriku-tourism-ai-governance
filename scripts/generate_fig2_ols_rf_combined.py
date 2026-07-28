"""Generate thesis Fig 5.1: actual vs OLS- and RF-predicted arrivals (combined).

Extends scripts/generate_fig2_ols_holdout.py per the final-defense request
(THESIS_REVISION_TASKS.md, Task 1):

- Three series on one axis: Actual, OLS predicted, RF predicted, both models
  using the SAME chronological 80/20 split so they are directly comparable.
- Hold-out window shaded.
- The 17 sensor-outage days are made legible: all series are reindexed to the
  full daily calendar with NaN on excluded dates (visible line breaks), and
  thin grey rug ticks mark each excluded date.
- Annotates OLS vs RF hold-out R² so the generalization contrast is explicit.
- Outputs EN + JA twins at dpi=300, white background.

NOTE: reproducing the paper-era metrics (OLS 0.810/0.683, RF ~0.909/~0.512)
requires the RSI data submodule (fukui-kanko-trend-report/public/data) checked
out at bf2cfc45 (2026-02-12 data). With newer data the stats drift; this
script prints actual metrics either way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import load_config
from src.data_loader import load_all_data
from src.feature_engineering import build_features
from src.report import Reporter

OUT_NAME_EN = "paper_fig2_ols_rf_combined.png"
OUT_NAME_JA = "paper_fig2_ols_rf_combined_ja.png"
TRAIN_PCT = 0.80
RF_PARAMS = dict(n_estimators=500, max_depth=10, min_samples_leaf=5,
                 random_state=42, max_features=1.0, n_jobs=-1)

COL_ACTUAL = "#2B5C8A"
COL_OLS = "#C0392B"
COL_RF = "#1E8449"


def _reindex_daily(dates: pd.Series, values: np.ndarray,
                   full_index: pd.DatetimeIndex) -> pd.Series:
    """Place values on the full daily calendar; excluded dates become NaN."""
    return pd.Series(values, index=pd.DatetimeIndex(dates)).reindex(full_index)


def main() -> None:
    cfg = load_config()
    rpt = Reporter(cfg)

    data = load_all_data(cfg, rpt)
    daily, feature_cols = build_features(data["daily"], data["route_col"], rpt)
    model_df = (daily[["date", "count"] + feature_cols]
                .dropna().sort_values("date").reset_index(drop=True))

    full_index = pd.date_range(model_df["date"].iloc[0],
                               model_df["date"].iloc[-1], freq="D")
    present = pd.DatetimeIndex(model_df["date"])
    excluded = full_index.difference(present)
    n_cal, n_eff, n_exc = len(full_index), len(model_df), len(excluded)
    print(f"calendar days: {n_cal} | effective: {n_eff} (paper: 397) | "
          f"excluded: {n_exc}")
    print(f"  of which {17} are documented zero-count camera-outage days (paper) "
          f"and {n_exc - 17} have incomplete weather/RSI coverage")
    if n_eff != 397:
        print("  !! effective N differs from the paper's 397 — data drift, "
              "check the bf2cfc45 pin before quoting metrics")

    # ---- OLS: full-sample fit (in-sample R²) + chronological 80/20 hold-out ----
    X_full = sm.add_constant(model_df[feature_cols].values, has_constant="add")
    y_full = model_df["count"].values
    full_model = sm.OLS(y_full, X_full).fit()
    ols_in_r2 = full_model.rsquared

    split = int(len(model_df) * TRAIN_PCT)
    train_df, hold_df = model_df.iloc[:split], model_df.iloc[split:]
    X_train = sm.add_constant(train_df[feature_cols].values, has_constant="add")
    ols_train = sm.OLS(train_df["count"].values, X_train).fit()
    X_hold = sm.add_constant(hold_df[feature_cols].values, has_constant="add")
    ols_pred_hold = ols_train.predict(X_hold)
    ols_hold_r2 = r2_score(hold_df["count"].values, ols_pred_hold)
    ols_hold_mae = mean_absolute_error(hold_df["count"].values, ols_pred_hold)
    ols_pred_train_window = full_model.predict(X_full)[:split]

    # ---- RF: mirrors the OLS structure so the two traces are comparable ----
    # In-sample (paper's 0.909): full-sample fit, shown on the training window.
    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(model_df[feature_cols].values, y_full)
    rf_in_r2 = r2_score(y_full, rf_full.predict(model_df[feature_cols].values))
    rf_pred_train_window = rf_full.predict(train_df[feature_cols].values)

    # Out-of-sample: fit on the training window only, predict the hold-out.
    rf_tr = RandomForestRegressor(**RF_PARAMS)
    rf_tr.fit(train_df[feature_cols].values, train_df["count"].values)
    rf_pred_hold = rf_tr.predict(hold_df[feature_cols].values)
    rf_hold_r2 = r2_score(hold_df["count"].values, rf_pred_hold)
    rf_hold_mae = mean_absolute_error(hold_df["count"].values, rf_pred_hold)

    # The paper's RF hold-out (0.512) uses the pipeline's chronological_split,
    # which ROUNDS (318/79) instead of truncating (317/80). Report both so the
    # figure's number is traceable and the split sensitivity is explicit.
    split_r = max(1, int(round(len(model_df) * TRAIN_PCT)))
    tr_r, ho_r = model_df.iloc[:split_r], model_df.iloc[split_r:]
    rf_r = RandomForestRegressor(**RF_PARAMS)
    rf_r.fit(tr_r[feature_cols].values, tr_r["count"].values)
    rf_hold_r2_round = r2_score(ho_r["count"].values,
                                rf_r.predict(ho_r[feature_cols].values))
    ols_tr_r = sm.OLS(tr_r["count"].values,
                      sm.add_constant(tr_r[feature_cols].values,
                                      has_constant="add")).fit()
    ols_hold_r2_round = r2_score(
        ho_r["count"].values,
        ols_tr_r.predict(sm.add_constant(ho_r[feature_cols].values,
                                         has_constant="add")))

    print(f"split (truncated): {split}/{len(model_df) - split}  |  "
          f"(rounded): {split_r}/{len(model_df) - split_r}")
    print(f"OLS  in-sample R² = {ols_in_r2:.4f} (paper: 0.810)")
    print(f"OLS  hold-out  R² = {ols_hold_r2:.4f} (paper: 0.683) | "
          f"MAE = {ols_hold_mae:,.0f} (paper: 1,793)")
    print(f"RF   in-sample R² = {rf_in_r2:.4f} (paper: 0.909)")
    print(f"RF   hold-out  R² = {rf_hold_r2:.4f} | "
          f"MAE = {rf_hold_mae:,.0f}")
    print(f"  split sensitivity (rounded {split_r}/{len(model_df) - split_r}): "
          f"OLS {ols_hold_r2_round:.4f} | RF {rf_hold_r2_round:.4f} "
          f"(paper RF: 0.512)")

    # ---- calendar-reindexed series (NaN gaps at outage days) ----
    s_actual = _reindex_daily(model_df["date"], y_full, full_index)
    s_ols = pd.concat([
        _reindex_daily(train_df["date"], ols_pred_train_window, full_index).dropna(),
        _reindex_daily(hold_df["date"], np.asarray(ols_pred_hold), full_index).dropna(),
    ]).reindex(full_index)
    s_rf = pd.concat([
        _reindex_daily(train_df["date"], rf_pred_train_window, full_index).dropna(),
        _reindex_daily(hold_df["date"], rf_pred_hold, full_index).dropna(),
    ]).reindex(full_index)

    hold_start = hold_df["date"].iloc[0]

    n_outage = 17  # documented zero-count camera-outage days (pipeline log)
    n_merge = n_exc - n_outage
    labels = {
        "en": dict(actual="Actual", ols="OLS predicted", rf="RF predicted",
                   ylab="Daily visitor arrivals",
                   excl=(f"{n_exc} days excluded: {n_outage} camera-outage "
                         f"+ {n_merge} incomplete coverage (N = {n_eff} of {n_cal})"),
                   sub="OLS retains more predictive power on the unseen hold-out window than RF",
                   ols_note=f"OLS R²: {ols_in_r2:.3f} in-sample → {ols_hold_r2:.3f} hold-out",
                   rf_note=f"RF R²: {rf_in_r2:.3f} in-sample → {rf_hold_r2:.3f} hold-out"),
        "ja": dict(actual="実測値", ols="OLS予測", rf="RF予測",
                   ylab="日次来訪者数",
                   excl=(f"除外{n_exc}日：カメラ障害{n_outage}日"
                         f"＋データ欠損{n_merge}日（N = {n_eff}／{n_cal}日）"),
                   sub="未知データ（ホールドアウト）ではOLSがRFより予測力を維持",
                   ols_note=f"OLS R²: {ols_in_r2:.3f}（標本内）→ {ols_hold_r2:.3f}（ホールドアウト）",
                   rf_note=f"RF R²: {rf_in_r2:.3f}（標本内）→ {rf_hold_r2:.3f}（ホールドアウト）"),
    }

    for lang, out_name in (("en", OUT_NAME_EN), ("ja", OUT_NAME_JA)):
        if lang == "ja":
            import japanize_matplotlib  # noqa: F401
        L = labels[lang]

        fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor="white")
        ax.set_facecolor("white")

        ax.plot(full_index, s_actual.values, color=COL_ACTUAL, lw=1.0,
                label=L["actual"])
        ax.plot(full_index, s_ols.values, color=COL_OLS, lw=0.9, ls="--",
                label=L["ols"])
        ax.plot(full_index, s_rf.values, color=COL_RF, lw=0.9, ls=":",
                label=L["rf"])

        ax.axvspan(hold_start, full_index[-1], color="#E8A87C", alpha=0.22,
                   zorder=0)

        ymax = float(np.nanmax([s_actual.max(), s_ols.max(), s_rf.max()])) * 1.14
        ax.set_ylim(0, ymax)
        ax.set_xlim(full_index[0], full_index[-1])

        # rug ticks marking each excluded (sensor-outage) date
        ax.vlines(excluded, 0, ymax * 0.035, color="#888888", lw=1.0,
                  label=L["excl"])

        mid_hold = hold_df["date"].iloc[len(hold_df) // 2]
        ax.text(mid_hold, ymax * 0.97, L["ols_note"], ha="center", va="top",
                fontsize=9.5, color=COL_OLS, fontweight="bold")
        ax.text(mid_hold, ymax * 0.90, L["rf_note"], ha="center", va="top",
                fontsize=9.5, color=COL_RF, fontweight="bold")

        ax.set_ylabel(L["ylab"])
        ax.set_title(L["sub"], fontsize=10.5, pad=8)
        # legend below the axes so it never covers the series
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                  fontsize=8.5, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        out = Path(rpt.fig_dir) / out_name
        fig.savefig(out, dpi=300, facecolor="white")
        plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
