"""Visualiser module – All matplotlib/seaborn figure generation.

Each public function creates one figure, saves it via ``Reporter.save_fig``,
and returns the ``matplotlib.figure.Figure`` so callers can do further
customisation if desired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from .report import Reporter


_JP_FONT_NAME: str | None = None


def _configure_japanese_font() -> None:
    """Configure matplotlib with a Japanese-capable font if available.

    This prevents tofu/missing-glyph boxes in generated ``*_ja.png`` figures.
    """
    global _JP_FONT_NAME

    # Most reliable path: japanize_matplotlib (bundles IPAexGothic settings).
    try:
        import japanize_matplotlib  # type: ignore  # noqa: F401
        _JP_FONT_NAME = plt.rcParams.get("font.family", [None])[0] if isinstance(plt.rcParams.get("font.family"), list) else None
        plt.rcParams["axes.unicode_minus"] = False
        return
    except Exception:
        pass

    candidates = [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoPGothic",
        "Yu Gothic",
        "Hiragino Sans",
        "MS Gothic",
        "Droid Sans Fallback",
    ]

    # Register known system font files explicitly (helps on headless Linux).
    known_font_files = [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKJP-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    ]
    for path in known_font_files:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
                if _JP_FONT_NAME is None:
                    _JP_FONT_NAME = fm.FontProperties(fname=path).get_name()
            except Exception:
                pass

    installed = {f.name for f in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), None)
    if selected:
        _JP_FONT_NAME = selected
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    elif _JP_FONT_NAME:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_JP_FONT_NAME, "DejaVu Sans"]
    else:
        # Last fallback (may still miss some CJK glyphs depending on environment).
        plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False


def _apply_japanese_font(fig: plt.Figure) -> None:
    if not _JP_FONT_NAME:
        return
    for text_obj in fig.findobj(lambda obj: isinstance(obj, mpl.text.Text)):
        try:
            text_obj.set_fontfamily(_JP_FONT_NAME)
        except Exception:
            pass


_configure_japanese_font()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str, reporter: Reporter,
          dpi: int = 150, ja_copy: bool = True) -> None:
    reporter.save_fig(fig, path, dpi=dpi, ja_copy=ja_copy)
    plt.close(fig)


def _save_with_ja(
    fig: plt.Figure,
    path: str,
    reporter: Reporter,
    ja_formatter: Callable[[plt.Figure], None],
    dpi: int = 150,
) -> None:
    reporter.save_fig(fig, path, dpi=dpi, ja_copy=False)
    ja_formatter(fig)
    _apply_japanese_font(fig)
    ja_path = path.replace(".png", "_ja.png")
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig)


# ── Fig 1: Time-series ──────────────────────────────────────────────────────

def plot_timeseries(
    daily: pd.DataFrame,
    route_col: str,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Dual-axis time-series: visitor count vs RSI intent."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(daily["date"], daily["count"], color="tab:blue", alpha=0.8, label="Visitor count")
    ax1.set_ylabel("Visitor Count", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(daily["date"], daily[route_col], color="tab:orange", alpha=0.6, label=f"RSI {route_col}")
    ax2.set_ylabel(f"RSI {route_col}", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Tojinbo: Visitor Count vs RSI Intent (daily)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax1_ja = fig_ja.axes[0]
        ax2_ja = fig_ja.axes[1]
        ax1_ja.set_ylabel("来訪者数", color="tab:blue")
        ax2_ja.set_ylabel(f"RSI {route_col}", color="tab:orange")
        ax1_ja.set_title("東尋坊：来訪者数とRSI需要（⽇次）")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 2: Correlation heatmap ──────────────────────────────────────────────

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Feature correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        fig_ja.axes[0].set_title("特徴量相関行列")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 3: Feature importance comparison ────────────────────────────────────

def plot_feature_importance(
    mdi_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Side-by-side MDI & permutation importance bar charts.

    Args:
        mdi_df: DataFrame with ``feature`` and ``importance``.
        perm_df: DataFrame with ``feature``, ``importance_mean``, ``importance_std``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    imp = mdi_df.sort_values("importance", ascending=True)
    axes[0].barh(imp["feature"], imp["importance"], color="steelblue")
    axes[0].set_title("Random Forest MDI Importance")
    axes[0].set_xlabel("Importance")

    perm = perm_df.sort_values("importance_mean", ascending=True)
    axes[1].barh(perm["feature"], perm["importance_mean"],
                 xerr=perm["importance_std"], color="darkorange")
    axes[1].set_title("Permutation Importance")
    axes[1].set_xlabel("Mean decrease in R²")
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        axes_ja = fig_ja.axes
        axes_ja[0].set_title("ランダムフォレスト特徴量重要度（MDI）")
        axes_ja[0].set_xlabel("重要度")
        axes_ja[1].set_title("Permutation Importance（並べ替え重要度）")
        axes_ja[1].set_xlabel("R²低下量（平均）")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 4: Day-of-week boxplot ──────────────────────────────────────────────

def plot_dow_boxplot(
    daily: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Visitor count by day of week."""
    df = daily.copy()
    df["dow_name"] = df["dow"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="dow_name", y="count",
                order=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                ax=ax, palette="Set2")
    ax.set_title("Visitor Count by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Visitor Count")
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xticklabels(["月", "火", "水", "木", "金", "土", "日"])
        ax_ja.set_title("曜日別来訪者数")
        ax_ja.set_xlabel("曜日")
        ax_ja.set_ylabel("来訪者数")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 5: RF prediction ────────────────────────────────────────────────────

def plot_rf_prediction(
    dates: pd.Series,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    rf_r2: float,
    cv_r2: float,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Actual vs RF-predicted time-series."""
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.plot(dates, y_actual, label="Actual", color="tab:blue", alpha=0.8)
    ax.plot(dates, y_pred, label="RF Predicted", color="tab:red", alpha=0.7, linestyle="--")
    ax.set_title("")
    ax.set_ylabel("Visitor Count")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.suptitle(
        f"Random Forest: Actual vs Predicted (R²={rf_r2:.3f}, CV R²={cv_r2:.3f})",
        fontsize=13,
        y=0.998,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        handles, _ = ax_ja.get_legend_handles_labels()
        ax_ja.legend(handles, ["実測（AIカメラ）", "予測（RF）"])
        ax_ja.set_title("")
        fig_ja.suptitle(
            f"需要予測（赤）とAIカメラ実測（青）の一致\n"
            f"（R²={rf_r2:.3f}, CV R²={cv_r2:.3f}）",
            fontsize=14,
            y=0.998,
        )
        ax_ja.set_ylabel("来訪者数")
        fig_ja.tight_layout(rect=[0, 0, 1, 0.955])

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 6: Opportunity gap scatter ──────────────────────────────────────────

def plot_opportunity_gap(
    daily: pd.DataFrame,
    route_col: str,
    intent_median: float,
    count_median: float,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Scatter of intent vs count coloured by gap flag."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = daily["opportunity_gap"].map({0: "steelblue", 1: "red"})
    ax.scatter(daily[route_col], daily["count"], c=colors, alpha=0.6,
               edgecolors="none", s=40)
    ax.axhline(count_median, color="gray", linestyle="--", alpha=0.5,
               label=f"Count median={count_median:.0f}")
    ax.axvline(intent_median, color="gray", linestyle=":", alpha=0.5,
               label=f"Intent median={intent_median:.0f}")
    ax.set_xlabel(f"RSI {route_col}")
    ax.set_ylabel("Visitor Count")
    ax.set_title("Opportunity Gap (red = high intent, low arrivals)")
    ax.legend()
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xlabel(f"RSI {route_col}")
        ax_ja.set_ylabel("来訪者数")
        ax_ja.set_title("オポチュニティギャップ（赤＝高需要・低来訪）")
        leg = ax_ja.get_legend()
        if leg is not None:
            labels = [
                f"来訪者中央値={count_median:.0f}",
                f"需要中央値={intent_median:.0f}",
            ]
            for txt, lbl in zip(leg.get_texts(), labels):
                txt.set_text(lbl)

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 7: Lag correlation bar chart ────────────────────────────────────────

def plot_lag_correlations(
    daily: pd.DataFrame,
    route_col: str,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of lag-0..7 Pearson correlations."""
    lag_corrs = []
    for lag in range(0, 8):
        col = f"{route_col}_lag{lag}"
        if col in daily.columns:
            r = daily[["count", col]].dropna().corr().iloc[0, 1]
            lag_corrs.append((lag, r))
    lag_df = pd.DataFrame(lag_corrs, columns=["lag", "r"])
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["tab:red" if r < 0 else "tab:green" for r in lag_df["r"]]
    ax.bar(lag_df["lag"], lag_df["r"], color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Lag Correlation: {route_col} → Visitor Count")
    ax.set_xticks(lag_df["lag"])
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xlabel("ラグ（日）")
        ax_ja.set_ylabel("相関係数（Pearson r）")
        ax_ja.set_title(f"ラグ相関：{route_col} → 来訪者数")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 8: CCF bar chart ────────────────────────────────────────────────────

def plot_ccf(
    ccf_results: list[tuple[int, float, int]],
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure | None:
    """Cross-correlation function bar chart."""
    if not ccf_results:
        return None
    ccf_df = pd.DataFrame(ccf_results, columns=["lag", "r", "n"])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:red" if r < 0 else "steelblue" for r in ccf_df["r"]]
    ax.bar(ccf_df["lag"], ccf_df["r"], color=colors)

    best_idx = ccf_df["r"].idxmax()
    best_lag = int(ccf_df.loc[best_idx, "lag"])
    best_r = float(ccf_df.loc[best_idx, "r"])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.2, color="gray", linestyle="--", alpha=0.5, label="r=0.2 threshold")

    y_min_data = float(ccf_df["r"].min())
    y_max_data = float(ccf_df["r"].max())
    if y_min_data > -0.05:
        y_min = 0.0
    else:
        y_min = min(-0.05, y_min_data - 0.03)
    y_max = min(1.0, y_max_data + 0.08)
    ax.set_ylim(y_min, y_max)

    if y_min <= -0.15:
        ax.axhline(-0.2, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Lag (days, + means Ishikawa leads): Ishikawa survey → Tojinbo arrivals")
    ax.set_ylabel("Pearson r")
    ax.set_title("Cross-Prefectural Signal: Ishikawa → Tojinbo (CCF)")
    ax.annotate(
        f"Peak lead signal: lag {best_lag}d, r={best_r:.3f}",
        xy=(best_lag, best_r),
        xytext=(best_lag + 1, min(y_max - 0.03, best_r + 0.06)),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2),
        fontsize=9,
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#2C3E50"),
    )
    ax.legend()
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xlabel("ラグ（日、+は石川先行）：石川アンケート活動 → 東尋坊来訪")
        ax_ja.set_ylabel("相関係数（Pearson r）")
        ax_ja.set_title("越境需要シグナル：石川 → 東尋坊（CCF）")
        for txt in ax_ja.texts:
            if "Peak lead signal:" in txt.get_text():
                txt.set_text(f"先行ピーク：ラグ {best_lag}日、r={best_r:.3f}")
        leg = ax_ja.get_legend()
        if leg is not None and leg.get_texts():
            leg.get_texts()[0].set_text("しきい値 r=0.2")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 9: Kansei scatter ───────────────────────────────────────────────────

def plot_kansei_scatter(
    sat_merged: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Scatterplot: daily visitor count vs mean satisfaction."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sat_merged["count"], sat_merged["mean_satisfaction"],
               alpha=0.5, edgecolors="none", s=40)
    ax.set_xlabel("Daily Visitor Count")
    ax.set_ylabel("Mean Satisfaction")
    ax.set_title("Kansei Feedback: Visitors vs Satisfaction")
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xlabel("日次来訪者数")
        ax_ja.set_ylabel("平均満足度")
        ax_ja.set_title("図2：東尋坊の賑わいと満足度の関係（自然拠点）")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 10: Lost population waterfall ────────────────────────────────────────

def plot_lost_population(
    gap_model: pd.DataFrame,
    total_lost: float,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure | None:
    """Bar chart of per-gap-day lost population."""
    if gap_model.empty:
        return None
    gap_sorted = gap_model.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(gap_sorted)), gap_sorted["lost_population"],
           color=["tab:red" if x > 0 else "tab:green"
                  for x in gap_sorted["lost_population"]])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Opportunity Gap Day (chronological)")
    ax.set_ylabel("Lost Population (Predicted - Actual)")
    ax.set_title(f"Lost Population per Gap Day (Total: {total_lost:,.0f})")
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xlabel("オポチュニティギャップ日（時系列順）")
        ax_ja.set_ylabel("逸失来訪者数（予測−実測）")
        ax_ja.set_title(f"ギャップ日の逸失来訪者（合計: {total_lost:,.0f}人）")

    _save_with_ja(fig, out_path, reporter, _ja, dpi=dpi)
    return fig


# ── Fig 11: Fukui Resurrection chart ────────────────────────────────────────

def plot_resurrection(
    sim_df: pd.DataFrame,
    total_lost: float,
    mean_actual_rank: float,
    mean_hypo_rank: float,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure:
    """Two-panel Fukui Resurrection chart (EN + JA variant)."""
    months_str = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(12)
    gains = sim_df["ranks_gained"].clip(lower=0)
    colors_gain = ["tab:purple" if m in [1, 2, 12] else "mediumpurple"
                   for m in sim_df["month"]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 2]})

    # Top: rank gains
    axes[0].bar(x, gains, color=colors_gain, edgecolor="indigo", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(months_str)
    axes[0].set_ylabel("Ranks Gained (higher = better)")
    axes[0].set_title(
        "Fukui Resurrection: Monthly Rank Gains with AI Governance\n"
        f"(Mean winter rank: {mean_actual_rank:.1f} → {mean_hypo_rank:.1f}, "
        f"recovered visitors: {total_lost:,.0f})",
        fontsize=13, fontweight="bold")
    target = max(mean_actual_rank - 41, 0)
    if target > 0:
        axes[0].axhline(y=target, color="gold", linestyle="--", linewidth=2, alpha=0.8)
    axes[0].set_ylim(0, max(float(gains.max()) + 3, 8))

    for idx, row in sim_df.iterrows():
        if row["ranks_gained"] > 0:
            axes[0].annotate(
                f"+{int(row['ranks_gained'])}\n"
                f"{int(row['fukui_rank_2025'])}→{int(row['hypo_rank'])}",
                xy=(idx, row["ranks_gained"]),
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="darkgreen")

    # Bottom: monthly recovered visitors
    colors_bar = ["tab:blue" if m in [1, 2, 12] else "lightblue"
                  for m in sim_df["month"]]
    axes[1].bar(x, sim_df["monthly_lost"], color=colors_bar,
                edgecolor="navy", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(months_str)
    axes[1].set_ylabel("Lost Visitors Recovered")
    axes[1].set_title("Monthly Distribution of Recovered Visitors (Winter months highlighted)")
    axes[1].annotate(
        f"Total: {total_lost:,.0f} visitors",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"))

    fig.tight_layout()
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)

    # Japanese variant
    axes[0].set_xticklabels([f"{m}月" for m in range(1, 13)])
    axes[0].set_ylabel("改善順位（大きいほど効果大）")
    axes[0].set_title(
        "福井復活：AIガバナンスによる月別順位改善\n"
        f"（冬季平均順位: {mean_actual_rank:.1f} → {mean_hypo_rank:.1f}、"
        f"回復見込み来訪者数: {total_lost:,.0f}人）",
        fontsize=13, fontweight="bold")
    axes[1].set_xticklabels([f"{m}月" for m in range(1, 13)])
    axes[1].set_ylabel("回復見込み来訪者数")
    axes[1].set_title("回復見込み来訪者数の月別分布（冬季を強調）")
    for txt in axes[1].texts:
        if txt.get_text().startswith("Total:"):
            txt.set_text(f"合計: {total_lost:,.0f}人")

    ja_path = out_path.replace(".png", "_ja.png")
    _apply_japanese_font(fig)
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig)
    return fig


# ── Fig 12: Hokuriku demand heatmap ─────────────────────────────────────────

def plot_hokuriku_heatmap(
    survey_all: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 150,
) -> plt.Figure | None:
    """Two-panel heatmap: monthly demand + cross-prefecture correlation."""
    if survey_all is None or survey_all.empty:
        reporter.log("  ⚠ No survey data for Hokuriku heatmap.")
        return None

    pref_daily = survey_all.copy()
    pref_daily["pref_clean"] = pref_daily["prefecture"].apply(
        lambda x: "石川" if "石川" in str(x) else (
            "福井" if "福井" in str(x) else (
                "富山" if "富山" in str(x) else "Other")))
    pref_daily = pref_daily[pref_daily["pref_clean"] != "Other"]
    pref_daily["yearmonth"] = pref_daily["date"].dt.to_period("M").astype(str)
    hm_data = pref_daily.groupby(["yearmonth", "pref_clean"]).size().reset_index(name="survey_count")
    hm_pivot = hm_data.pivot(index="pref_clean", columns="yearmonth", values="survey_count").fillna(0)

    pref_pivot = (
        pref_daily.groupby(["date", "pref_clean"]).size().reset_index(name="count")
        .pivot(index="date", columns="pref_clean", values="count").fillna(0)
    )
    pref_corr = pref_pivot.corr()

    reporter.log("\nCross-Prefecture Daily Correlation Matrix:")
    reporter.log(pref_corr.to_string())

    pref_map = {"石川": "Ishikawa", "福井": "Fukui", "富山": "Toyama"}

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    # EN labels
    hm_en = hm_pivot.copy()
    hm_en.index = [pref_map.get(v, v) for v in hm_en.index]
    corr_en = pref_corr.copy()
    corr_en.index = [pref_map.get(v, v) for v in corr_en.index]
    corr_en.columns = [pref_map.get(v, v) for v in corr_en.columns]

    sns.heatmap(hm_en, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=axes[0], cbar_kws={"label": "Survey Responses"})
    axes[0].set_title("Hokuriku Monthly Tourism Demand Heatmap (Survey Responses)")
    axes[0].set_ylabel("Prefecture")
    axes[0].set_xlabel("Month")
    axes[0].tick_params(axis="x", labelrotation=90)

    sns.heatmap(corr_en, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[1], square=True, cbar_kws={"label": "Pearson r"})
    axes[1].set_title("Cross-Prefecture Daily Demand Correlation")

    fig.tight_layout()
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)

    # JA variant
    axes[0].clear()
    axes[1].clear()
    sns.heatmap(hm_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=axes[0], cbar_kws={"label": "回答件数"})
    axes[0].set_title("北陸月次観光需要ヒートマップ（アンケート回答数）")
    axes[0].set_ylabel("都道府県")
    axes[0].set_xlabel("月")
    axes[0].tick_params(axis="x", labelrotation=90)

    sns.heatmap(pref_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[1], square=True, cbar_kws={"label": "相関係数 (Pearson r)"})
    axes[1].set_title("都道府県間の日次需要相関")

    fig.tight_layout()
    ja_path = out_path.replace(".png", "_ja.png")
    _apply_japanese_font(fig)
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig)
    return fig


# ── Fig 13: Spatial friction heatmap ─────────────────────────────────────────

def plot_spatial_friction(
    heat_df: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 300,
) -> plt.Figure | None:
    """Heatmap of weather sensitivity per node."""
    if heat_df is None or heat_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd",
                cbar_kws={"label": "Relative Friction Intensity"}, ax=ax)
    ax.set_title("Spatial Friction Heatmap (Weather Sensitivity per Node)")
    fig.tight_layout()
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)
    ax.set_title("空間摩擦ヒートマップ（拠点別の気象感応度）")
    if fig.axes and len(fig.axes) > 1:
        fig.axes[1].set_ylabel("相対的摩擦強度")
    ja_path = out_path.replace(".png", "_ja.png")
    _apply_japanese_font(fig)
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig)
    return fig


# ── Fig 14: Weather Shield Network ───────────────────────────────────────────

def plot_weather_shield_network(
    valid_nodes: dict,
    out_path: str,
    reporter: Reporter,
    dpi: int = 300,
) -> plt.Figure | None:
    """Network diagram showing weather buffering effects across 4 nodes.

    When coastal Node A (Tojinbo) has high winds, inland nodes C (Katsuyama)
    and D (Rainbow Line) act as economic buffers via demand rerouting.
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if len(valid_nodes) < 3:
        reporter.log("Weather Shield: needs at least 3 nodes")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Node positions (geographic layout of Fukui)
    positions = {
        "Node A (Tojinbo/Mikuni)": (0.2, 0.7),      # Northwest coast
        "Node B (Fukui Station)": (0.5, 0.5),       # Central hub
        "Node C (Katsuyama/Dinosaur)": (0.8, 0.6),  # East mountains
        "Node D (Rainbow Line/Wakasa)": (0.3, 0.2), # South scenic
    }

    colors = {
        "Node A (Tojinbo/Mikuni)": "#E74C3C",
        "Node B (Fukui Station)": "#3498DB",
        "Node C (Katsuyama/Dinosaur)": "#2ECC71",
        "Node D (Rainbow Line/Wakasa)": "#9B59B6",
    }

    labels_short = {
        "Node A (Tojinbo/Mikuni)": "Tojinbo\n(Coastal)",
        "Node B (Fukui Station)": "Fukui Station\n(Hub)",
        "Node C (Katsuyama/Dinosaur)": "Katsuyama\n(Mountain)",
        "Node D (Rainbow Line/Wakasa)": "Rainbow Line\n(Scenic)",
    }

    # Scale radius proportionally to per-node lost visitors
    all_lost = [v["lost_visitors"] / 1000 for v in valid_nodes.values()]
    max_lost_k = max(all_lost) if all_lost else 1.0

    # 1. Hub-and-spoke backbone: thin gray lines from each node to Fukui Station hub
    hub_pos = positions.get("Node B (Fukui Station)")
    if hub_pos:
        for name, (x, y) in positions.items():
            if name in valid_nodes and "Fukui Station" not in name:
                ax.plot([x, hub_pos[0]], [y, hub_pos[1]],
                        color="#D5D8DC", lw=1.5, alpha=0.7, zorder=1)

    # 2. Weather rerouting arrows: Tojinbo → inland buffer nodes
    routing_targets = [
        ("Node C (Katsuyama/Dinosaur)", 0.25),
        ("Node D (Rainbow Line/Wakasa)", -0.25),
    ]
    tojinbo_pos = positions.get("Node A (Tojinbo/Mikuni)")
    if tojinbo_pos and "Node A (Tojinbo/Mikuni)" in valid_nodes:
        ax_x, ax_y = tojinbo_pos
        for target, rad in routing_targets:
            if target in positions and target in valid_nodes:
                tx, ty = positions[target]
                ax.annotate(
                    "", xy=(tx, ty), xytext=(ax_x, ax_y),
                    arrowprops=dict(
                        arrowstyle="-|>", color="#2471A3",
                        connectionstyle=f"arc3,rad={rad}",
                        lw=2.2, alpha=0.85,
                    ),
                    zorder=2,
                )
                # Midpoint label
                mid_x = (ax_x + tx) / 2 + (0.08 if rad > 0 else -0.08)
                mid_y = (ax_y + ty) / 2 + (0.06 if rad > 0 else -0.06)
                ax.text(mid_x, mid_y, "weather\nreroute",
                        ha="center", va="center", fontsize=7,
                        color="#1A5276", style="italic", zorder=3)

    # 3. Draw nodes (radius ∝ lost visitors)
    for name, (x, y) in positions.items():
        if name not in valid_nodes:
            continue
        metrics = valid_nodes[name]
        lost_k = metrics["lost_visitors"] / 1000

        radius = 0.05 + (lost_k / max_lost_k) * 0.06  # 0.05–0.11

        circle = plt.Circle((x, y), radius, color=colors.get(name, "#999"),
                            alpha=0.85, zorder=5)
        ax.add_patch(circle)

        ax.text(x, y + radius + 0.025, labels_short.get(name, name.split("(")[0]),
               ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)

        ax.text(x, y - radius - 0.025, f"{lost_k:.0f}K lost/yr",
               ha="center", va="top", fontsize=8, color="#444", zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("Hokuriku Weather Shield Network\n4-Node Spatial Governance Architecture",
                fontsize=14, fontweight="bold", pad=20)

    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Coastal (weather-exposed)"),
        mpatches.Patch(color="#3498DB", label="Hub (transit)"),
        mpatches.Patch(color="#2ECC71", label="Mountain (buffer)"),
        mpatches.Patch(color="#9B59B6", label="Scenic (buffer)"),
        Line2D([0], [0], color="#2471A3", lw=2, label="Bad-weather reroute"),
        Line2D([0], [0], color="#D5D8DC", lw=1.5, label="Hub connection"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9)

    # Bubble size note
    total_lost = sum(m["lost_visitors"] for m in valid_nodes.values()) / 1000
    ax.text(0.02, 0.97, "Bubble area ∝ lost visitors per node",
            transform=ax.transAxes, fontsize=8, color="#777",
            va="top", style="italic")

    fig.tight_layout()
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)

    # JA version
    ax.set_title("北陸天候シールドネットワーク\n4拠点空間ガバナンスアーキテクチャ",
                fontsize=14, fontweight="bold", pad=20)
    labels_short_ja = {
        "Node A (Tojinbo/Mikuni)": "東尋坊\n（沿岸）",
        "Node B (Fukui Station)": "福井駅\n（拠点）",
        "Node C (Katsuyama/Dinosaur)": "勝山\n（山間）",
        "Node D (Rainbow Line/Wakasa)": "レインボーライン\n（景観）",
    }
    # Update node labels
    for txt in ax.texts:
        for eng, ja in labels_short_ja.items():
            if txt.get_text() == labels_short.get(eng, ""):
                txt.set_text(ja)
    # Update routing arrow labels
    for txt in ax.texts:
        if txt.get_text() == "weather\nreroute":
            txt.set_text("悪天候\n誘導")
    # Update bubble note
    for txt in ax.texts:
        if "Bubble area" in txt.get_text():
            txt.set_text("バブル面積 ∝ 拠点別損失来訪者数")
    legend_elements_ja = [
        mpatches.Patch(color="#E74C3C", label="沿岸（悪天候影響大）"),
        mpatches.Patch(color="#3498DB", label="拠点（交通ハブ）"),
        mpatches.Patch(color="#2ECC71", label="山間（緩衝地帯）"),
        mpatches.Patch(color="#9B59B6", label="景観（緩衝地帯）"),
        Line2D([0], [0], color="#2471A3", lw=2, label="悪天候時誘導経路"),
        Line2D([0], [0], color="#D5D8DC", lw=1.5, label="拠点接続"),
    ]
    ax.legend(handles=legend_elements_ja, loc="lower right", fontsize=9,
              framealpha=0.9)

    ja_path = out_path.replace(".png", "_ja.png")
    _apply_japanese_font(fig)
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig)
    return fig


# ── Fig 15: Rank 47→35 Resurrection Projection ───────────────────────────────

def plot_rank_resurrection_projection(
    valid_nodes: dict,
    ranking_data: dict,
    out_path: str,
    reporter: Reporter,
    dpi: int = 300,
) -> plt.Figure | None:
    """Chart showing how recovering 4-node lost population would improve 
    Fukui's national ranking from 47th toward mid-30s.
    """
    import matplotlib.patches as mpatches
    
    if len(valid_nodes) < 3:
        reporter.log("Rank projection: needs at least 3 nodes")
        return None
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Current ranks (from config)
    current_ranks = ranking_data.get("fukui_rank_2025", [47]*12)
    visitors_k = ranking_data.get("fukui_visitors_k", [100]*12)
    gap_to_41_k = ranking_data.get("gap_to_rank41_k", [30]*12)
    
    # Calculate potential recovery
    total_lost_k = sum(m["lost_visitors"] for m in valid_nodes.values()) / 1000
    monthly_recovery = total_lost_k / 12  # Simple even distribution
    
    # Project new ranks (rough heuristic)
    projected_ranks = []
    for i, (rank, gap) in enumerate(zip(current_ranks, gap_to_41_k)):
        if monthly_recovery >= gap:
            # Recovery exceeds gap to rank 41
            projected_ranks.append(max(32, rank - 12))  # Cap at ~35th
        elif monthly_recovery >= gap * 0.5:
            projected_ranks.append(max(35, rank - 6))
        else:
            projected_ranks.append(max(38, rank - 3))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Panel 1: Rank trajectory
    x = np.arange(len(months))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, current_ranks, width, label="Current Rank (47th)", 
                   color="#E74C3C", alpha=0.8)
    bars2 = ax1.bar(x + width/2, projected_ranks, width, label="Projected Rank (With Recovery)", 
                   color="#27AE60", alpha=0.8)
    
    ax1.set_ylabel("National Ranking (Lower = Better)")
    ax1.set_title("Fukui Prefecture Tourism Ranking: 47th → Mid-30s Resurrection Path",
                 fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.legend()
    ax1.set_ylim(25, 50)
    ax1.invert_yaxis()  # Lower rank is better
    ax1.axhline(y=35, color="#666", linestyle="--", alpha=0.5, label="Target: 35th")
    ax1.axhline(y=41, color="#999", linestyle=":", alpha=0.5, label="Threshold: 41st")
    ax1.text(0.98, 0.97, "1st = best", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9, color="#2C3E50")
    ax1.text(0.98, 0.03, "47th = worst", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=9, color="#7F8C8D")
    
    # Annotate best improvement
    best_idx = np.argmax(np.array(current_ranks) - np.array(projected_ranks))
    ax1.annotate(f"+{current_ranks[best_idx] - projected_ranks[best_idx]} ranks",
                xy=(best_idx + width/2, projected_ranks[best_idx]), 
                xytext=(best_idx + 1, projected_ranks[best_idx] - 5),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green", fontweight="bold")
    
    # Panel 2: Lost visitor recovery potential
    lost_per_node = {name.split("(")[1].replace(")", ""): m["lost_visitors"]/1000 
                    for name, m in valid_nodes.items()}
    names = list(lost_per_node.keys())
    values = list(lost_per_node.values())
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"][:len(names)]
    
    bars3 = ax2.barh(names, values, color=colors, alpha=0.8)
    ax2.set_xlabel("Lost Visitors (Thousands)")
    ax2.set_title("4-Node Lost Population by Location (Recoverable Economic Potential)",
                 fontsize=12, fontweight="bold")
    
    # Add value labels
    for bar, val in zip(bars3, values):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}K", va="center", fontsize=10)
    
    # Summary annotation
    total_visitors = sum(values)
    satake_b = total_visitors * 13.811 / 1000
    ax2.text(0.98, 0.02, f"Total: {total_visitors:.0f}K visitors\n≈ ¥{satake_b:.2f}B opportunity",
            transform=ax2.transAxes, ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#F7DC6F", alpha=0.9))
    
    fig.tight_layout()
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)
    
    # JA version
    ax1.set_title("福井県観光ランキング：47位→35位圏への復活パス",
                 fontsize=12, fontweight="bold")
    ax1.set_ylabel("全国ランキング（低い=良い）")
    ax1.text(0.98, 0.97, "1位＝最良", transform=ax1.transAxes,
             ha="right", va="top", fontsize=9, color="#2C3E50")
    ax1.text(0.98, 0.03, "47位＝最下位", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=9, color="#7F8C8D")
    ax1.legend([
        "現在の順位 (47位)",
        "回復後予測順位",
        "目標：35位",
        "閾値：41位",
    ])
    
    ax2.set_title("4拠点喪失人口（回復可能な経済ポテンシャル）",
                 fontsize=12, fontweight="bold")
    ax2.set_xlabel("喪失来訪者数（千人）")
    
    ja_path = out_path.replace(".png", "_ja.png")
    _apply_japanese_font(fig)
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig)
    return fig


# ── Fig 16: DHDE Architecture diagram ────────────────────────────────────────

def plot_dhde_architecture(
    out_path: str,
    reporter: Reporter,
    dpi: int = 300,
) -> plt.Figure:
    """Conceptual architecture diagram for the Distributed Human Data Engine."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Journal-appropriate palette: muted, desaturated, print-safe
    C_BG       = "#FFFFFF"
    C_COL_S    = "#EEF3F9"   # cool grey-blue tint
    C_COL_C    = "#EEF6F0"   # cool grey-green tint
    C_COL_O    = "#F5F0EC"   # warm grey tint
    C_BORDER_S = "#2B5C8A"   # steel blue
    C_BORDER_C = "#2A6B45"   # forest green
    C_BORDER_O = "#7B4B1A"   # warm sienna
    C_CARD_S   = "#DAE8F5"
    C_CARD_C   = "#D3EDDF"
    C_CARD_O   = "#F0DFD0"
    C_TEXT     = "#1A1A2E"
    C_MUTED    = "#4A5568"
    C_ARROW    = "#4A5568"
    C_ACCENT   = "#5A3E85"   # muted violet for feedback arc

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    def rbox(x, y, w, h, fc, ec, radius=0.3, lw=1.4, alpha=1.0):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3,
        ))

    def txt(x, y, s, size=9, color=C_TEXT, weight="normal", ha="center", va="center", style="normal"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, zorder=5, style=style)

    def arr(x1, y1, x2, y2, color=C_ARROW, lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.24,head_length=0.20",
                        color=color, lw=lw, connectionstyle="arc3,rad=0.0",
                    ), zorder=4)

    # Title
    txt(10, 9.62, "Distributed Human Data Engine (DHDE) — AI Governance Architecture",
        size=15, weight="bold")
    txt(10, 9.18, "Hokuriku Tourism Demand Forecasting  |  Fukui Prefecture, Japan  |  2024-2026",
        size=9.5, color=C_MUTED)
    ax.plot([0.3, 19.7], [8.88, 8.88], color="#CCCCCC", lw=1.0)

    # Column panels — tighter gaps, lower to clear title separator
    # Col1: x=0.2..4.45  gap=0.35  Col2: x=4.8..12.55  gap=0.35  Col3: x=12.9..19.8
    for px, py, pw, ph, pc, pbc, plabel in [
        (0.20, 0.40, 4.25, 8.30, C_COL_S, C_BORDER_S, "INPUT SENSORS"),
        (4.80, 0.40, 7.75, 8.30, C_COL_C, C_BORDER_C, "DHDE PROCESSING CORE"),
        (12.9, 0.40, 6.90, 8.30, C_COL_O, C_BORDER_O, "OUTPUT GOVERNANCE"),
    ]:
        rbox(px, py, pw, ph, pc, pbc, radius=0.5, lw=1.8, alpha=0.6)
        cx = px + pw / 2
        txt(cx, py + ph - 0.30, plabel, size=10, color=pbc, weight="bold")
        ax.plot([px + 0.35, px + pw - 0.35], [py + ph - 0.55, py + ph - 0.55],
                color=pbc, lw=1.0, alpha=0.40, zorder=4)

    # Column bodies start below the underline at y ≈ 8.15
    # 4 cards h=1.58, gap=0.27 → sy[0]=6.30, bottom card bottom=0.97
    CH = 1.58
    sy_starts = [6.30, 4.45, 2.60, 0.97]

    # ── Input sensor cards ─────────────────────────────────────────────────
    sensors = [
        ("RSI Business Intent",
         "47-site direction & search queries",
         "Direction counts  ->  lag / roll features"),
        ("JMA Weather Stations",
         "Temp  |  Precip  |  Snow  |  Wind  |  Humidity",
         "Winter sensitivity  6.27x  higher than summer"),
        ("Edge-AI Cameras",
         "Human-shape detection, 5-min intervals",
         "397 usable days across 4 spatial nodes"),
        ("Visitor Surveys",
         "95,653 Hokuriku responses (NPS + satisfaction)",
         "71,288 Fukui free-text  ->  Kansei NLP"),
    ]
    for (title, line1, line2), sy in zip(sensors, sy_starts):
        rbox(0.40, sy, 3.85, CH, C_CARD_S, C_BORDER_S, radius=0.25, lw=1.2)
        txt(2.325, sy + 1.25, title, size=11, color=C_BORDER_S, weight="bold")
        txt(2.325, sy + 0.82, line1, size=9.5, color=C_MUTED)
        txt(2.325, sy + 0.38, line2, size=9.5, color=C_MUTED)

    # ── Core: Feature Engineering ──────────────────────────────────────────
    rbox(5.0, 6.25, 7.35, 1.85, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(8.675, 7.78, "Feature Engineering", size=11.5, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "Calendar (dow_mean, month, is_holiday)   Lag(1,2,3)   Roll(7d)",
        "Weekend x Intent   Weekend x Severity   interaction terms",
        "Discomfort Index   Wind Chill   Kansei under-vibrancy flags",
    ]):
        txt(8.675, 7.37 - i * 0.36, ln, size=9.0, color=C_MUTED)

    # ── Core: OLS ─────────────────────────────────────────────────────────
    rbox(5.0, 3.90, 3.55, 2.10, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(6.775, 5.70, "OLS Regression", size=11, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "R2 = 0.810  (Adj 0.802)",
        "16 predictors   N = 397",
        "Newey-West HAC   sig = 8",
        "DW (LDV) = 1.898",
        "Weather lift  +0.056 R2",
    ]):
        txt(6.775, 5.31 - i * 0.33, ln, size=9.0, color=C_MUTED)

    # ── Core: Random Forest ───────────────────────────────────────────────
    rbox(8.80, 3.90, 3.55, 2.10, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(10.575, 5.70, "Random Forest", size=11, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "Train R2 = 0.909",
        "CV R2 = 0.557  (+/- 0.131)",
        "Hold-out R2 = 0.683",
        "MAE = 1,793 visitors/day",
        "Top: directions, month",
    ]):
        txt(10.575, 5.31 - i * 0.33, ln, size=9.0, color=C_MUTED)

    # ── Core: Robustness ──────────────────────────────────────────────────
    rbox(5.0, 2.10, 7.35, 1.58, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(8.675, 3.40, "Robustness Suite", size=11, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "First-Diff R2=0.708   LDV R2=0.849   Cohen f2=4.25   Newey-West sig=8",
        "4-node spatial cross-correlation   Ishikawa -> Fukui pipeline  r=+0.552",
        "Eiheiji quietude threshold x*=42.4%   Kansei Spearman r=+0.148 (p=0.002)",
    ]):
        txt(8.675, 3.02 - i * 0.35, ln, size=8.5, color=C_MUTED)

    # ── Output governance cards ────────────────────────────────────────────
    outputs = [
        ("Supply-side Nudges",
         "865,917 lost visitors/yr recovered",
         "Rank lift: 47th  ->  ~35th nationally"),
        ("Weather-Resilient Routing",
         "Winter 6.27x more weather-sensitive",
         "Snow / wind alerts  ->  alternate nodes"),
        ("Kansei Comfort Governance",
         "Discomfort Index  +  Wind Chill alerts",
         "Quietude  <=42.4%  prevents sat. drop"),
        ("Economic Impact Dashboard",
         "Annual economic loss: 11.96B JPY",
         "4-node geographic saturation achieved"),
    ]
    for (title, line1, line2), oy in zip(outputs, sy_starts):
        rbox(13.1, oy, 6.50, CH, C_CARD_O, C_BORDER_O, radius=0.25, lw=1.2)
        txt(16.35, oy + 1.25, title, size=11, color=C_BORDER_O, weight="bold")
        txt(16.35, oy + 0.82, line1, size=9.5, color=C_MUTED)
        txt(16.35, oy + 0.38, line2, size=9.5, color=C_MUTED)

    # ── Arrows: sensors -> feature eng ────────────────────────────────────
    for sy in sy_starts:
        arr(4.25, sy + 0.79, 4.95, 7.17, color=C_BORDER_S, lw=1.4)

    # ── Arrows: feature eng -> models ─────────────────────────────────────
    arr(7.8,  6.25, 6.775, 6.00, color=C_BORDER_C, lw=1.4)
    arr(9.55, 6.25, 10.575, 6.00, color=C_BORDER_C, lw=1.4)

    # ── Arrows: models -> robustness ──────────────────────────────────────
    arr(6.775, 3.90, 7.2,  3.68, color=C_BORDER_C, lw=1.4)
    arr(10.575, 3.90, 10.15, 3.68, color=C_BORDER_C, lw=1.4)

    # ── Arrows: robustness -> outputs ─────────────────────────────────────
    for oy in sy_starts:
        arr(12.35, 2.89, 13.05, oy + 0.79, color=C_BORDER_O, lw=1.4)

    # ── Footer ────────────────────────────────────────────────────────────
    ax.plot([0.3, 19.7], [0.55, 0.55], color="#CCCCCC", lw=0.8, zorder=3)
    txt(0.4, 0.3,
        "Data: Fukui Prefecture AI cameras  |  JMA  |  Route Search Impressions (RSI)  |  Hokuriku Survey  2024-2026",
        size=8.0, color=C_MUTED, ha="left")
    txt(19.6, 0.3, "DHDE v1.0  |  Amil Khanzada  PhD Research",
        size=8.0, color=C_MUTED, ha="right")

    fig.tight_layout(pad=0.3)
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)
    plt.close(fig)
    reporter.log(f"  Saved {out_path}")
    return fig
