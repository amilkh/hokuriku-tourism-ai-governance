#!/usr/bin/env python3
"""Generate a grant-ready 3-prefecture CCF figure.

Panel A: Ishikawa survey activity -> Fukui (Tojinbo camera arrivals)
Panel B: Ishikawa survey activity -> Toyama (survey activity proxy)

Usage from repo root:
    python scripts/generate_grant_ccf_3pref_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Allow direct script execution without package installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import visualizer as viz
from src.config import load_config, resolve_ws_path
from src.data_loader import load_camera_daily, load_survey_prefectures
from src.report import Reporter

LAGS = list(range(-3, 8))
FIG_WIDTH_IN = 10.0
FIG_HEIGHT_IN = 5.4
FIG_DPI = 200  # 10in x 200dpi = 2000px wide


def _ccf_by_lags(lead: pd.Series, target: pd.Series, lags: list[int]) -> list[tuple[int, float, int]]:
    """Return (lag, r, n) for each lag where at least 10 rows are valid."""
    out: list[tuple[int, float, int]] = []
    base = pd.DataFrame({"lead": lead, "target": target}).dropna()
    for lag in lags:
        shifted = base["lead"].shift(lag)
        valid = pd.DataFrame({"lead": shifted, "target": base["target"]}).dropna()
        if len(valid) > 10:
            r = float(valid.corr().iloc[0, 1])
            out.append((lag, r, int(len(valid))))
    return out


def _daily_pref_counts(survey_all: pd.DataFrame, pref_keyword: str, col_name: str) -> pd.DataFrame:
    return (
        survey_all[survey_all["prefecture"].astype(str).str.contains(pref_keyword, na=False)]
        .groupby("date")
        .size()
        .reset_index(name=col_name)
    )


def _results_to_frame(results: list[tuple[int, float, int]]) -> pd.DataFrame:
    if not results:
        raise RuntimeError("No CCF rows were computed.")
    return pd.DataFrame(results, columns=["lag", "r", "n"]).sort_values("lag")


def _lag0_r(result_df: pd.DataFrame) -> float:
    row = result_df[result_df["lag"] == 0]
    if row.empty:
        raise RuntimeError("Lag 0 result not found.")
    return float(row["r"].iloc[0])


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    y_min: float,
    y_max: float,
) -> tuple[int, float]:
    colors = ["#1f77b4" if r >= 0 else "#d62728" for r in df["r"]]
    for i, lag in enumerate(df["lag"]):
        if lag == 0:
            colors[i] = "#ff7f0e"

    ax.bar(df["lag"], df["r"], color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(-0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

    best_row = df.iloc[df["r"].idxmax()]
    best_lag = int(best_row["lag"])
    best_r = float(best_row["r"])

    # Fixed shared limits across both panels prevent clipping when sharey=True.
    ax.set_ylim(y_min, y_max)

    ax.scatter([best_lag], [best_r], color="black", s=28, zorder=3)
    text_y = min(best_r + 0.08, y_max - 0.08)
    ax.annotate(
        f"positive peak: lag {best_lag:+d}, r={best_r:+.3f}",
        xy=(best_lag, best_r),
        xytext=(best_lag + 0.5, text_y),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#2f2f2f"},
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#666", "alpha": 0.9},
    )

    lag0 = _lag0_r(df)
    ax.text(
        0.02,
        0.95,
        f"Lag 0: r={lag0:+.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "#fff7e6", "ec": "#d9a441", "alpha": 0.95},
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Lag (days, + means Ishikawa leads)")
    ax.set_ylabel("Pearson r")
    ax.set_xticks(df["lag"].tolist())
    return best_lag, best_r


def build_figure() -> tuple[Path, Path, dict[str, float | int]]:
    cfg = load_config()
    rpt = Reporter(cfg)

    camera_glob = str(resolve_ws_path(cfg, cfg["paths"]["camera"]["tojinbo"]))
    survey_glob = str(resolve_ws_path(cfg, cfg["paths"]["survey"]["merged_glob"]))

    camera = load_camera_daily(camera_glob, reporter=rpt)
    survey_all = load_survey_prefectures(survey_glob, reporter=rpt)
    if camera.empty or survey_all.empty:
        raise RuntimeError("Camera or survey data is empty.")

    # Panel A: Ishikawa survey -> Fukui(Tojinbo camera)
    ishikawa_daily = _daily_pref_counts(survey_all, "石川", "ishikawa_count")
    fukui_daily = _daily_pref_counts(survey_all, "福井", "fukui_count")
    toya_daily = _daily_pref_counts(survey_all, "富山", "toyama_count")

    # Keep alignment with existing pipeline metric logic (n=418, r=0.552).
    fukui_merged = camera[["date", "count"]].merge(ishikawa_daily, on="date", how="left")
    fukui_merged = fukui_merged.merge(fukui_daily, on="date", how="left").dropna()
    fukui_results = _ccf_by_lags(
        lead=fukui_merged["ishikawa_count"],
        target=fukui_merged["count"],
        lags=LAGS,
    )

    # Panel B: Ishikawa survey -> Toyama survey (proxy for Toyama demand)
    toyama_merged = ishikawa_daily.merge(toya_daily, on="date", how="inner").dropna()
    toyama_results = _ccf_by_lags(
        lead=toyama_merged["ishikawa_count"],
        target=toyama_merged["toyama_count"],
        lags=LAGS,
    )

    df_fukui = _results_to_frame(fukui_results)
    df_toyama = _results_to_frame(toyama_results)

    # Shared y-limits across both panels (with headroom) to avoid top clipping.
    all_r = pd.concat([df_fukui["r"], df_toyama["r"]], ignore_index=True)
    y_min_data = float(all_r.min())
    y_max_data = float(all_r.max())
    y_range = max(y_max_data - y_min_data, 0.20)
    bottom_pad = max(0.04, 0.08 * y_range)
    top_pad = max(0.12, 0.22 * y_range)
    shared_y_min = min(-0.08, y_min_data - bottom_pad)
    shared_y_max = y_max_data + top_pad

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)

    f_best_lag, f_best_r = _plot_panel(
        axes[0],
        df_fukui,
        "A. Ishikawa -> Fukui (Tojinbo)",
        "Lead signal against observed camera arrivals",
        shared_y_min,
        shared_y_max,
    )
    t_best_lag, t_best_r = _plot_panel(
        axes[1],
        df_toyama,
        "B. Ishikawa -> Toyama",
        "Lead signal against Toyama survey activity (proxy)",
        shared_y_min,
        shared_y_max,
    )

    fig.suptitle(
        "Hokuriku Cross-Correlation Evidence for 3-Prefecture Linkage",
        fontsize=12,
        fontweight="bold",
        y=0.975,
    )
    fig.subplots_adjust(left=0.04, right=0.995, top=0.86, bottom=0.14, wspace=0.22)

    out_en = REPO_ROOT / "output" / "grant_fig2_hokuriku_ccf_3pref.png"
    out_ja = REPO_ROOT / "output" / "grant_fig2_hokuriku_ccf_3pref_ja.png"
    fig.savefig(out_en, dpi=FIG_DPI)

    # Japanese labels on a copied figure for direct insertion into the grant.
    axes[0].set_title("A. 石川 -> 福井（東尋坊）\n実来訪（AIカメラ）とのCCF", fontsize=10, fontweight="bold")
    axes[1].set_title("B. 石川 -> 富山\n富山アンケート活動（代理指標）とのCCF", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("ラグ（日、+は石川先行）")
    axes[1].set_xlabel("ラグ（日、+は石川先行）")
    axes[0].set_ylabel("相関係数（Pearson r）")
    fig.suptitle("北陸3県連携を示すクロス相関エビデンス", fontsize=12, fontweight="bold", y=0.975)
    viz._apply_japanese_font(fig)
    fig.savefig(out_ja, dpi=FIG_DPI)
    plt.close(fig)

    stats = {
        "fukui_lag0_r": _lag0_r(df_fukui),
        "fukui_best_lag": f_best_lag,
        "fukui_best_r": f_best_r,
        "toyama_lag0_r": _lag0_r(df_toyama),
        "toyama_best_lag": t_best_lag,
        "toyama_best_r": t_best_r,
        "fukui_n_lag0": int(df_fukui[df_fukui["lag"] == 0]["n"].iloc[0]),
        "toyama_n_lag0": int(df_toyama[df_toyama["lag"] == 0]["n"].iloc[0]),
    }
    return out_en, out_ja, stats


def main() -> None:
    out_en, out_ja, stats = build_figure()
    print("3-prefecture grant CCF figure generated")
    print(f"  EN: {out_en}")
    print(f"  JA: {out_ja}")
    print(
        "  Fukui lag0: r={:+.3f} (n={}) | best lag={:+d}, r={:+.3f}".format(
            stats["fukui_lag0_r"],
            stats["fukui_n_lag0"],
            int(stats["fukui_best_lag"]),
            stats["fukui_best_r"],
        )
    )
    print(
        "  Toyama lag0: r={:+.3f} (n={}) | best lag={:+d}, r={:+.3f}".format(
            stats["toyama_lag0_r"],
            stats["toyama_n_lag0"],
            int(stats["toyama_best_lag"]),
            stats["toyama_best_r"],
        )
    )


if __name__ == "__main__":
    main()
