"""Generate paper Figure 2: actual vs OLS-predicted arrivals with chronological holdout.

Reproduces the pipeline's OLS specification and 80/20 chronological split so the
figure matches the paper's reported metrics (in-sample R² = 0.810, holdout
R² = 0.683, MAE = 1,793). White background, x-axis limited to the data range.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import load_config
from src.data_loader import load_all_data
from src.feature_engineering import build_features
from src.report import Reporter

OUT_NAME = "paper_fig2_ols_holdout.png"
TRAIN_PCT = 0.80


def main() -> None:
    cfg = load_config()
    rpt = Reporter(cfg)

    data = load_all_data(cfg, rpt)
    daily, feature_cols = build_features(data["daily"], data["route_col"], rpt)
    # NOTE: reproducing the paper requires the RSI data submodule
    # (fukui-kanko-trend-report/public/data) checked out at bf2cfc45
    # (2026-02-12 data) — the paper-era pin. With current data the stats drift.
    model_df = (daily[["date", "count"] + feature_cols]
                .dropna().sort_values("date").reset_index(drop=True))
    print(f"rows: {len(model_df)} (paper: 397 = 317 train + 80 holdout), "
          f"end: {model_df['date'].iloc[-1].date()}")

    # Full-sample OLS (paper's in-sample R² = 0.810)
    X_full = sm.add_constant(model_df[feature_cols].values, has_constant="add")
    y_full = model_df["count"].values
    full_model = sm.OLS(y_full, X_full).fit()
    in_sample_r2 = full_model.rsquared

    # Chronological 80/20 split (paper's holdout R² = 0.683)
    split = int(len(model_df) * TRAIN_PCT)
    train_df, hold_df = model_df.iloc[:split], model_df.iloc[split:]
    X_train = sm.add_constant(train_df[feature_cols].values, has_constant="add")
    train_model = sm.OLS(train_df["count"].values, X_train).fit()
    X_hold = sm.add_constant(hold_df[feature_cols].values, has_constant="add")
    y_pred_hold = train_model.predict(X_hold)
    holdout_r2 = r2_score(hold_df["count"].values, y_pred_hold)
    holdout_mae = mean_absolute_error(hold_df["count"].values, y_pred_hold)

    print(f"in-sample R² = {in_sample_r2:.4f} (paper: 0.810)")
    print(f"holdout   R² = {holdout_r2:.4f} (paper: 0.683)")
    print(f"holdout  MAE = {holdout_mae:,.0f} (paper: 1,793)")

    # Predicted series: full-model fit on the training window,
    # train-model out-of-sample predictions on the holdout window.
    y_pred_train_window = full_model.predict(X_full)[:split]

    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(model_df["date"], y_full, color="#2B5C8A", lw=0.9, label="Actual")
    ax.plot(train_df["date"], y_pred_train_window,
            color="#C0392B", lw=0.9, ls="--", label="OLS Predicted")
    ax.plot(hold_df["date"], y_pred_hold, color="#C0392B", lw=0.9, ls="--")

    hold_start = hold_df["date"].iloc[0]
    ax.axvspan(hold_start, model_df["date"].iloc[-1],
               color="#E8A87C", alpha=0.22, zorder=0)

    ymax = max(y_full.max(), y_pred_hold.max()) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_xlim(model_df["date"].iloc[0], model_df["date"].iloc[-1])

    mid_train = train_df["date"].iloc[len(train_df) // 2]
    mid_hold = hold_df["date"].iloc[len(hold_df) // 2]
    ax.text(mid_train, ymax * 0.96, f"IN-SAMPLE  R² = {in_sample_r2:.3f}",
            ha="center", va="top", fontsize=10, color="#C0392B", fontweight="bold")
    ax.text(mid_hold, ymax * 0.96, f"HOLDOUT  R² = {holdout_r2:.3f}",
            ha="center", va="top", fontsize=10, color="#A04000", fontweight="bold")

    ax.set_ylabel("Visitor Count")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    out = Path(rpt.fig_dir) / OUT_NAME
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
