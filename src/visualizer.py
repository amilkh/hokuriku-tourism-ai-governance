"""Visualiser module – All matplotlib/seaborn figure generation.

Each public function creates one figure, saves it via ``Reporter.save_fig``,
and returns the ``matplotlib.figure.Figure`` so callers can do further
customisation if desired.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
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
        with contextlib.suppress(Exception):
            text_obj.set_fontfamily(_JP_FONT_NAME)


_configure_japanese_font()

# Installed-family fallback for benchmark / ablation ``lang="ja"`` figures only (no font file paths).
_JA_BENCH_FONT_CANDIDATES: tuple[str, ...] = (
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "Hiragino Sans",
)


def _resolve_japanese_benchmark_fontproperties() -> fm.FontProperties | None:
    """First installed Japanese-capable font from ``_JA_BENCH_FONT_CANDIDATES``."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _JA_BENCH_FONT_CANDIDATES:
        if name in installed:
            return fm.FontProperties(family=name)
    for want in _JA_BENCH_FONT_CANDIDATES:
        w = want.replace(" ", "").lower()
        for got in installed:
            if w in got.replace(" ", "").lower():
                return fm.FontProperties(family=got)
    return None


def _apply_benchmark_japanese_font(fig: plt.Figure) -> None:
    fp = _resolve_japanese_benchmark_fontproperties()
    if fp is None:
        return
    for text_obj in fig.findobj(lambda obj: isinstance(obj, mpl.text.Text)):
        with contextlib.suppress(Exception):
            text_obj.set_fontproperties(fp)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str, reporter: Reporter,
          dpi: int = 150, ja_copy: bool | None = None) -> None:
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
    """Dual-axis time-series: visitor count vs Google intent."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(daily["date"], daily["count"], color="tab:blue", alpha=0.8, label="Visitor count")
    ax1.set_ylabel("Visitor Count", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(daily["date"], daily[route_col], color="tab:orange", alpha=0.6, label=f"Google {route_col}")
    ax2.set_ylabel(f"Google {route_col}", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("Tojinbo: Visitor Count vs Google Intent (daily)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax1_ja = fig_ja.axes[0]
        ax2_ja = fig_ja.axes[1]
        ax1_ja.set_ylabel("来訪者数", color="tab:blue")
        ax2_ja.set_ylabel(f"Google {route_col}", color="tab:orange")
        ax1_ja.set_title("東尋坊：来訪者数とGoogle需要（⽇次）")

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
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
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
    ax.set_xlabel(f"Google {route_col}")
    ax.set_ylabel("Visitor Count")
    ax.set_title("Opportunity Gap (red = high intent, low arrivals)")
    ax.legend()
    fig.tight_layout()
    def _ja(fig_ja: plt.Figure) -> None:
        ax_ja = fig_ja.axes[0]
        ax_ja.set_xlabel(f"Google {route_col}")
        ax_ja.set_ylabel("来訪者数")
        ax_ja.set_title("オポチュニティギャップ（赤＝高需要・低来訪）")
        leg = ax_ja.get_legend()
        if leg is not None:
            labels = [
                f"来訪者中央値={count_median:.0f}",
                f"需要中央値={intent_median:.0f}",
            ]
            for txt, lbl in zip(leg.get_texts(), labels, strict=False):
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
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ["tab:red" if r < 0 else "steelblue" for r in ccf_df["r"]]
    ax.bar(ccf_df["lag"], ccf_df["r"], color=colors)

    best_idx = ccf_df["r"].idxmax()
    best_lag = int(ccf_df.loc[best_idx, "lag"])
    best_r = float(ccf_df.loc[best_idx, "r"])

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.2, color="gray", linestyle="--", alpha=0.5, label="r=0.2 threshold")

    y_min_data = float(ccf_df["r"].min())
    y_max_data = float(ccf_df["r"].max())
    y_min = 0.0 if y_min_data > -0.05 else min(-0.05, y_min_data - 0.03)
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
        fontsize=10,
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#2C3E50"),
    )
    ax.legend(fontsize=10)
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
    plt.close(fig)

    # JA variant — fresh figure to avoid orphaned colorbar axes from the EN pass.
    fig_ja, axes_ja = plt.subplots(2, 1, figsize=(16, 10),
                                   gridspec_kw={"height_ratios": [3, 1]})

    hm_ja = hm_pivot.copy()
    hm_ja.index.name = "都道府県"
    corr_ja = pref_corr.copy()
    corr_ja.index.name = "都道府県"
    corr_ja.columns.name = "都道府県"

    sns.heatmap(hm_ja, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=axes_ja[0], cbar_kws={"label": "回答件数"})
    axes_ja[0].set_title("北陸月次観光需要ヒートマップ（アンケート回答数）")
    axes_ja[0].set_ylabel("都道府県")
    axes_ja[0].set_xlabel("月")
    axes_ja[0].tick_params(axis="x", labelrotation=90)

    sns.heatmap(corr_ja, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes_ja[1], square=True, cbar_kws={"label": "相関係数 (Pearson r)"})
    axes_ja[1].set_title("都道府県間の日次需要相関")

    fig_ja.tight_layout()
    ja_path = out_path.replace(".png", "_ja.png")
    _apply_japanese_font(fig_ja)
    fig_ja.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")
    plt.close(fig_ja)
    return None


# ── Fig 13: Spatial friction heatmap ─────────────────────────────────────────

def plot_spatial_friction(
    heat_df: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    dpi: int = 300,
) -> plt.Figure | None:
    """Heatmap of weather sensitivity per node.

    Colors are column-normalized (0–1 within each metric) so that metrics on
    vastly different scales (e.g. ΔR² ≈ 0.01 vs lost_visitors_k ≈ 500) each
    show meaningful within-column variation.  Raw values are shown as text.
    """
    if heat_df is None or heat_df.empty:
        return None

    # Column-wise min-max normalisation for colour scale; keep raw for labels.
    col_max = heat_df.max()
    col_max[col_max == 0] = 1  # avoid div-by-zero for all-zero columns
    heat_norm = heat_df.div(col_max)

    # Build annotation array: use ints for large values, 3dp for small ones.
    annot = heat_df.copy().astype(object)
    for col in heat_df.columns:
        for idx in heat_df.index:
            v = heat_df.loc[idx, col]
            annot.loc[idx, col] = f"{v:.0f}" if abs(v) >= 10 else f"{v:.3f}"

    _NODE_LABELS_JA = {
        "Node A (Tojinbo/Mikuni)": "A拠点（東尋坊/三国）",
        "Node B (Fukui Station)": "B拠点（福井駅）",
        "Node C (Katsuyama/Dinosaur)": "C拠点（勝山/恐竜）",
        "Node D (Rainbow Line/Wakasa)": "D拠点（レインボーライン/若狭）",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_norm, annot=annot, fmt="", cmap="YlOrRd",
                vmin=0, vmax=1,
                cbar_kws={"label": "Relative Friction Intensity (column-normalised)"},
                ax=ax)
    ax.set_title("Spatial Friction Heatmap (Weather Sensitivity per Node)")
    ax.set_ylabel("")
    fig.tight_layout()
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)

    # JA variant: translate title, colorbar label, and y-axis tick labels.
    ax.set_title("空間摩擦ヒートマップ（拠点別の気象感応度）")
    if fig.axes and len(fig.axes) > 1:
        fig.axes[1].set_ylabel("相対的摩擦強度（列内正規化）")
    ja_labels = [_NODE_LABELS_JA.get(t.get_text(), t.get_text())
                 for t in ax.get_yticklabels()]
    ax.set_yticklabels(ja_labels, rotation=0)
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
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

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
                        ha="center", va="center", fontsize=10,
                        color="#1A5276", style="italic", zorder=3)

    # 3. Draw nodes (radius ∝ lost visitors)
    for name, (x, y) in positions.items():
        if name not in valid_nodes:
            continue
        metrics = valid_nodes[name]
        lost_k = metrics["lost_visitors"] / 1000

        radius = 0.03 + 0.08 * np.sqrt(lost_k / max_lost_k)  # sqrt-scaled 0.03–0.11

        circle = plt.Circle((x, y), radius, color=colors.get(name, "#999"),
                            alpha=0.85, zorder=5)
        ax.add_patch(circle)

        ax.text(x, y + radius + 0.025, labels_short.get(name, name.split("(")[0]),
               ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)

        ax.text(x, y - radius - 0.025, f"{lost_k:.0f}K lost/yr",
               ha="center", va="top", fontsize=10, color="#444", zorder=6)

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
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.9)

    # Bubble size note
    ax.text(0.02, 0.97, "Bubble area ∝ lost visitors per node",
            transform=ax.transAxes, fontsize=10, color="#777",
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
    ax.legend(handles=legend_elements_ja, loc="lower right", fontsize=10,
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
    _visitors_k = ranking_data.get("fukui_visitors_k", [100]*12)
    gap_to_41_k = ranking_data.get("gap_to_rank41_k", [30]*12)

    # Calculate potential recovery
    total_lost_k = sum(m["lost_visitors"] for m in valid_nodes.values()) / 1000
    monthly_recovery = total_lost_k / 12  # Simple even distribution

    # Project new ranks (rough heuristic)
    projected_ranks = []
    for _, (rank, gap) in enumerate(zip(current_ranks, gap_to_41_k, strict=False)):
        if monthly_recovery >= gap:
            # Recovery exceeds gap to rank 41
            projected_ranks.append(max(32, rank - 12))  # Cap at ~35th
        elif monthly_recovery >= gap * 0.5:
            projected_ranks.append(max(35, rank - 6))
        else:
            projected_ranks.append(max(38, rank - 3))

    # ── Colour palette: blue/teal only ───────────────────────────────────────
    C_CURRENT   = "#2B5C8A"   # deep blue  – current rank bars
    C_PROJECTED = "#1A7A6E"   # teal       – projected rank bars
    C_WINTER    = "#1A4F72"   # darkest blue – winter recovery bars
    C_TRANS     = "#2980B9"   # mid blue   – transition month bars
    C_SUMMER    = "#85C1E9"   # light blue – summer recovery bars

    # ── Monthly recovery weighted by seasonal sensitivity (6.26× winter) ─────
    _season_w = [6.26, 6.26, 2.5, 2.5, 2.5, 1.0,
                 1.0,  1.0,  1.0, 2.5, 2.5, 6.26]
    total_lost = sum(m["lost_visitors"] for m in valid_nodes.values())
    _w_total = sum(_season_w)
    monthly_rec_k = [total_lost * w / _w_total / 1000 for w in _season_w]
    _bar_colors = [
        C_WINTER if w == 6.26 else (C_TRANS if w == 2.5 else C_SUMMER)
        for w in _season_w
    ]

    # ── Projected ranks: proportional to gap coverage, scaled 0–12 rank positions
    # improvement = (rec / gap) * 12, capped at 12.  Smooth, no cliff-edges.
    projected_ranks = []
    for rank, gap_k, rec_k in zip(current_ranks, gap_to_41_k, monthly_rec_k, strict=False):
        coverage = rec_k / gap_k if gap_k > 0 else 0
        improvement = min(12, int(coverage * 12))
        projected_ranks.append(max(35, rank - improvement))


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # ── Panel A: rank trajectory ──────────────────────────────────────────────
    x = np.arange(len(months))
    width = 0.35

    ax1.bar(x - width/2, current_ranks,  width, label="Current Rank",
            color=C_CURRENT, alpha=0.85)
    ax1.bar(x + width/2, projected_ranks, width, label="Projected Rank (With Recovery)",
            color=C_PROJECTED, alpha=0.85)

    ax1.set_ylabel("National Ranking (Lower = Better)")
    ax1.set_title("Panel (A)  -  Current vs. Projected Ranking  (4-Node Recovery)\n"
                  "Four nodes close up to 66% of monthly shortfall; prefecture-wide deployment projects top-35 ranking",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.set_xlim(-0.6, 11.8)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_ylim(27, 49)
    ax1.invert_yaxis()
    ax1.set_yticks(range(30, 46, 5))
    ax1.set_yticklabels([str(i) for i in range(30, 46, 5)])
    ax1.axhline(y=35, color="#555", linestyle="--", linewidth=0.9, alpha=0.7)

    # Annotation on the best-improvement bar (first winter month)
    improvements = [c - p for c, p in zip(current_ranks, projected_ranks, strict=False)]
    best_idx = int(np.argmax(improvements))
    if improvements[best_idx] > 0:
        ax1.annotate(f"+{improvements[best_idx]} ranks",
                     xy=(best_idx + width/2, projected_ranks[best_idx]),
                     xytext=(best_idx + 1.5, projected_ranks[best_idx] - 3),
                     arrowprops=dict(arrowstyle="->", color=C_PROJECTED, lw=1.4),
                     fontsize=10, color=C_PROJECTED, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor=C_PROJECTED, linewidth=0.8, alpha=0.9))

    # ── Panel B: monthly recovered demand distribution ────────────────────────
    ax2.bar(x, monthly_rec_k, color=_bar_colors, alpha=0.88,
            edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Recovered Visitors (Thousands)")
    ax2.set_title("Panel (B)  -  Monthly Distribution of Recovered Demand  (865,917 visitors)\n"
                  "Seasonal weights applied: winter 6.26× summer  |  Source: japan-kanko-stat / JTA 2025",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months)
    # Enough headroom for value labels above tallest bars (~154K)
    ax2.set_ylim(0, max(monthly_rec_k) * 1.22)

    for i, val in enumerate(monthly_rec_k):
        ax2.text(i, val + max(monthly_rec_k) * 0.01,
                 f"{val:.0f}K", ha="center", va="bottom", fontsize=10)

    ax2.legend(handles=[
        mpatches.Patch(color=C_WINTER, label="Winter (Dec-Feb)"),
        mpatches.Patch(color=C_TRANS,  label="Transition (Mar-May, Oct-Nov)"),
        mpatches.Patch(color=C_SUMMER, label="Summer (Jun-Sep)"),
    ], fontsize=10, loc="upper center")


    fig.tight_layout(pad=1.5)
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)

    # ── JA version ────────────────────────────────────────────────────────────
    months_ja = ["1月", "2月", "3月", "4月", "5月", "6月",
                 "7月", "8月", "9月", "10月", "11月", "12月"]
    ax1.set_title("パネル（A）  -  現状 vs. 予測ランキング（4拠点回復）\n"
                  "4拠点のみで月次不足分の最大66%を補填；全県展開で全国35位圏を射程に",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("全国ランキング（低い=良い）")
    ax1.set_xticklabels(months_ja)
    ax1.legend(["現在の順位", "回復後予測順位"], loc="lower left", fontsize=10)

    ax2.set_title("パネル（B）  -  回復需要の月別分布（86.6万人）\n"
                  "季節係数適用：冬季6.26倍  |  出典：japan-kanko-stat / 観光庁2025",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("月")
    ax2.set_ylabel("回復来訪者数（千人）")
    ax2.set_xticklabels(months_ja)
    ax2.legend(handles=[
        mpatches.Patch(color=C_WINTER, label="冬季（12〜2月）"),
        mpatches.Patch(color=C_TRANS,  label="移行期（3〜5月・10〜11月）"),
        mpatches.Patch(color=C_SUMMER, label="夏季（6〜9月）"),
    ], fontsize=10, loc="upper center")

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
    from matplotlib.patches import FancyBboxPatch

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
    _C_ACCENT  = "#5A3E85"   # muted violet (reserved for future feedback arc)

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

    def txt(x, y, s, size=10, color=C_TEXT, weight="normal", ha="center", va="center", style="normal"):
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
        size=11.5, color=C_MUTED)
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
        txt(cx, py + ph - 0.30, plabel, size=13, color=pbc, weight="bold")
        ax.plot([px + 0.35, px + pw - 0.35], [py + ph - 0.55, py + ph - 0.55],
                color=pbc, lw=1.0, alpha=0.40, zorder=4)

    # Column bodies start below the underline at y ≈ 8.15
    # 4 cards h=1.75, gap=0.163 → sy[0]=6.34, bottom card bottom=0.61 (above footer at 0.35)
    CH = 1.75
    sy_starts = [6.34, 4.43, 2.52, 0.61]

    # ── Input sensor cards ─────────────────────────────────────────────────
    sensors = [
        ("Google Business Intent",
         "・47-site direction & search queries",
         "・Direction counts  ->  lag / roll features"),
        ("JMA Weather Stations",
         "・Temp | Precip | Snow | Wind | Humidity",
         "・Winter sensitivity: demand gating"),
        ("Edge-AI Cameras",
         "・Human detection, 5-min intervals",
         "・427 days, 4 spatial nodes"),
    ]
    for (title, line1, line2), sy in zip(sensors, sy_starts, strict=False):
        rbox(0.40, sy, 3.85, CH, C_CARD_S, C_BORDER_S, radius=0.25, lw=1.2)
        txt(2.325, sy + 1.42, title, size=14, color=C_BORDER_S, weight="bold")
        txt(2.325, sy + 0.97, line1, size=14.0, color=C_MUTED)
        txt(2.325, sy + 0.66, line2, size=14.0, color=C_MUTED)

    # Visitor Surveys — 3-line body to avoid overflow on the 97,719 line
    sy = sy_starts[3]
    rbox(0.40, sy, 3.85, CH, C_CARD_S, C_BORDER_S, radius=0.25, lw=1.2)
    txt(2.325, sy + 1.42, "Visitor Surveys",              size=14,   color=C_BORDER_S, weight="bold")
    txt(2.325, sy + 1.02, "・97,719 Hokuriku responses",  size=14.0, color=C_MUTED)
    txt(2.325, sy + 0.74, "  (NPS + satisfaction)",       size=12.0, color=C_MUTED)
    txt(2.325, sy + 0.46, "・71,623 free-text → Kansei NLP", size=14.0, color=C_MUTED)

    # ── Core: Feature Engineering ──────────────────────────────────────────
    rbox(5.0, 5.71, 7.35, 1.85, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(8.675, 7.24, "Feature Engineering", size=14, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "・Calendar (dow_mean, month, is_holiday)   Lag(1,2,3)   Roll(7d)",
        "・Weekend x Intent   Weekend x Severity   interaction terms",
        "・Discomfort Index   Wind Chill   Kansei under-vibrancy flags",
    ]):
        txt(8.675, 6.83 - i * 0.33, ln, size=12.5, color=C_MUTED)

    # ── Core: OLS ─────────────────────────────────────────────────────────
    rbox(5.0, 2.95, 3.55, 2.25, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(6.775, 4.96, "OLS Regression", size=14, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "・R2 = 0.810  (Adj 0.802)",
        "・16 predictors   N = 397",
        "・Newey-West HAC   sig = 8",
        "・DW (LDV) = 1.898",
        "・Weather lift  +0.056 R2",
    ]):
        txt(6.775, 4.58 - i * 0.27, ln, size=12.5, color=C_MUTED)

    # ── Core: Random Forest ───────────────────────────────────────────────
    rbox(8.80, 2.95, 3.55, 2.25, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(10.575, 4.96, "Random Forest", size=14, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "・Train R2 = 0.909",
        "・CV R2 = 0.557  (+/- 0.131)",
        "・Hold-out R2 = 0.683",
        "・MAE = 1,793 visitors/day",
        "・Top: directions, month",
    ]):
        txt(10.575, 4.58 - i * 0.27, ln, size=12.5, color=C_MUTED)

    # ── Core: Robustness ──────────────────────────────────────────────────
    rbox(5.0, 0.65, 7.35, 1.90, C_CARD_C, C_BORDER_C, radius=0.25, lw=1.2)
    txt(8.675, 2.27, "Robustness Suite", size=14, color=C_BORDER_C, weight="bold")
    for i, ln in enumerate([
        "・First-Diff R2=0.708   LDV R2=0.849   Cohen f2=4.25   Newey-West sig=8",
        "・4-node spatial cross-correlation   Ishikawa -> Fukui pipeline  r=+0.549",
        "・Kansei Spearman r=+0.150 (p=0.002)   under-vibrancy positive correlation",
    ]):
        txt(8.675, 1.89 - i * 0.34, ln, size=12.5, color=C_MUTED)

    # ── Output governance cards ────────────────────────────────────────────
    outputs = [
        ("Supply-Side Nudges",
         "・865,917 lost visitors/yr recovered",
         "・Rank lift: 47th  ->  ~35th nationally"),
        ("Weather-Resilient Routing",
         "・Winter 6.26x more weather-sensitive",
         "・Snow / wind alerts  ->  alternate nodes"),
        ("Kansei Comfort Governance",
         "・Discomfort Index  +  Wind Chill alerts",
         "・Satisfaction resilient to crowd density"),
        ("Economic Impact Dashboard",
         "・Annual loss: ¥11.96B  (~$72.6M USD)",
         "・4-node geographic saturation achieved"),
    ]
    for (title, line1, line2), oy in zip(outputs, sy_starts, strict=False):
        rbox(13.1, oy, 6.50, CH, C_CARD_O, C_BORDER_O, radius=0.25, lw=1.2)
        txt(16.35, oy + 1.42, title, size=15, color=C_BORDER_O, weight="bold")
        txt(16.35, oy + 0.97, line1, size=15.0, color=C_MUTED)
        txt(16.35, oy + 0.66, line2, size=15.0, color=C_MUTED)

    # ── Arrows: sensors -> feature eng ────────────────────────────────────
    for sy in sy_starts:
        arr(4.25, sy + 0.875, 4.95, 6.64, color=C_BORDER_S, lw=1.4)

    # ── Arrows: feature eng -> models ─────────────────────────────────────
    arr(7.8,  5.71, 6.775, 5.18, color=C_BORDER_C, lw=1.4)
    arr(9.55, 5.71, 10.575, 5.18, color=C_BORDER_C, lw=1.4)

    # ── Arrows: models -> robustness ──────────────────────────────────────
    arr(6.775, 2.95, 7.2,  2.55, color=C_BORDER_C, lw=1.4)
    arr(10.575, 2.95, 10.15, 2.55, color=C_BORDER_C, lw=1.4)

    # ── Arrows: robustness -> outputs ─────────────────────────────────────
    for oy in sy_starts:
        arr(12.35, 1.60, 13.05, oy + 0.875, color=C_BORDER_O, lw=1.4)

    # ── Footer ────────────────────────────────────────────────────────────
    ax.plot([0.3, 19.7], [0.35, 0.35], color="#CCCCCC", lw=0.8, zorder=6)
    txt(0.4, 0.20,
        "Data: Fukui Prefecture AI cameras  |  JMA  |  Google Business Profile  |  Hokuriku Survey  2024-2026",
        size=10.5, color=C_MUTED, ha="left")

    fig.tight_layout(pad=0.3)
    reporter.save_fig(fig, out_path, dpi=dpi, ja_copy=False)
    reporter.log(f"  Saved {out_path}")

    # ── Japanese variant ──────────────────────────────────────────────────
    _DHDE_JA = {
        "Distributed Human Data Engine (DHDE) — AI Governance Architecture":
            "分散型人間データエンジン（DHDE）— AIガバナンスアーキテクチャ",
        "Hokuriku Tourism Demand Forecasting  |  Fukui Prefecture, Japan  |  2024-2026":
            "北陸観光需要予測 ｜ 福井県、日本 ｜ 2024-2026",
        "INPUT SENSORS":        "入力センサー",
        "DHDE PROCESSING CORE": "DHDE処理コア",
        "OUTPUT GOVERNANCE":    "出力ガバナンス",
        # sensor card titles
        "Google Business Intent":  "Googleビジネスインテント",
        "JMA Weather Stations":    "気象庁観測所",
        "Edge-AI Cameras":         "エッジAIカメラ",
        "Visitor Surveys":         "来訪者調査",
        # sensor card lines
        "・47-site direction & search queries":            "・47サイトの経路・検索クエリ",
        "・Direction counts  ->  lag / roll features":     "・方向カウント → ラグ/ローリング特徴量",
        "・Temp | Precip | Snow | Wind | Humidity": "・気温｜降水｜積雪｜風速｜湿度",
        "・Winter sensitivity: demand gating": "・冬季感応度：需要制約",
        "・Human detection, 5-min intervals":  "・人型検知、5分間隔",
        "・427 days, 4 spatial nodes":          "・4拠点427日分の有効データ",
        "・97,719 Hokuriku responses":     "・北陸回答数 97,719件",
        "  (NPS + satisfaction)":          "  （NPS＋満足度）",
        "・71,623 free-text → Kansei NLP": "・福井自由記述71,623件 → 感性NLP",
        # core card titles
        "Feature Engineering": "特徴量エンジニアリング",
        "OLS Regression":      "OLS回帰",
        "Random Forest":       "ランダムフォレスト",
        "Robustness Suite":    "頑健性検証スイート",
        # core card lines
        "・Calendar (dow_mean, month, is_holiday)   Lag(1,2,3)   Roll(7d)":
            "・カレンダー（曜日均・月・祝日）  ラグ(1,2,3)  ローリング(7日)",
        "・Weekend x Intent   Weekend x Severity   interaction terms":
            "・週末×インテント  週末×深刻度  交互作用項",
        "・Discomfort Index   Wind Chill   Kansei under-vibrancy flags":
            "・不快指数  体感気温  感性活気不足フラグ",
        "・R2 = 0.810  (Adj 0.802)":      "・R² = 0.810（調整済 0.802）",
        "・16 predictors   N = 397":      "・16予測変数  N = 397",
        "・Newey-West HAC   sig = 8":     "・Newey-West HAC  有意 = 8",
        "・DW (LDV) = 1.898":             "・DW（LDV）= 1.898",
        "・Weather lift  +0.056 R2":      "・気象寄与  +0.056 R²",
        "・Train R2 = 0.909":             "・学習R² = 0.909",
        "・CV R2 = 0.557  (+/- 0.131)":  "・CV R² = 0.557（±0.131）",
        "・Hold-out R2 = 0.683":          "・ホールドアウトR² = 0.683",
        "・MAE = 1,793 visitors/day":     "・MAE = 1,793人/日",
        "・Top: directions, month":       "・重要特徴量：方向数・月",
        "・First-Diff R2=0.708   LDV R2=0.849   Cohen f2=4.25   Newey-West sig=8":
            "・一階差分R²=0.708  LDV R²=0.849  Cohen f²=4.25  Newey-West有意=8",
        "・4-node spatial cross-correlation   Ishikawa -> Fukui pipeline  r=+0.549":
            "・4拠点空間交差相関  石川→福井パイプライン  r=+0.549",
        "・Kansei Spearman r=+0.150 (p=0.002)   under-vibrancy positive correlation":
            "・感性スピアマン r=+0.150（p=0.002）  過少賑わい正の相関",
        # output card titles
        "Supply-Side Nudges":          "供給側ナッジ",
        "Weather-Resilient Routing":   "気象耐性ルーティング",
        "Kansei Comfort Governance":   "感性コンフォートガバナンス",
        "Economic Impact Dashboard":   "経済的影響ダッシュボード",
        # output card lines
        "・865,917 lost visitors/yr recovered":    "・年間損失来訪者865,917人の回復",
        "・Rank lift: 47th  ->  ~35th nationally": "・順位改善：全国47位→約35位",
        "・Winter 6.26x more weather-sensitive":   "・冬季は気象感応度が6.26倍",
        "・Snow / wind alerts  ->  alternate nodes": "・積雪・風速警報 → 代替拠点誘導",
        "・Discomfort Index  +  Wind Chill alerts": "・不快指数＋体感気温アラート",
        "・Satisfaction resilient to crowd density": "・来訪者密度に対して満足度は安定",
        "・Annual loss: ¥11.96B  (~$72.6M USD)":    "・年間損失：¥11.96B（約72.6M USD）",
        "・4-node geographic saturation achieved":  "・4拠点による地理的飽和達成",
        # footer
        "Data: Fukui Prefecture AI cameras  |  JMA  |  Google Business Profile  |  Hokuriku Survey  2024-2026":
            "データ：福井県AIカメラ ｜ 気象庁 ｜ Googleビジネスプロフィール ｜ 北陸調査 2024-2026",
    }
    for text_obj in ax.texts:
        s = text_obj.get_text()
        if s in _DHDE_JA:
            text_obj.set_text(_DHDE_JA[s])
    _apply_japanese_font(fig)
    ja_path = out_path.replace(".png", "_ja.png")
    fig.savefig(ja_path, dpi=dpi)
    reporter.optimize_png(ja_path)
    reporter.log(f"  Saved {ja_path}")

    plt.close(fig)
    return fig


# ── Benchmark figures ─────────────────────────────────────────────────────────

_BENCH_I18N: dict[str, dict[str, object]] = {
    "en": {
        "suptitle": "Chronological hold-out benchmark",
        "no_data": "No benchmark data",
        "metrics": [
            ("MAE", "MAE", "MAE (visitors/day)"),
            ("RMSE", "RMSE", "RMSE"),
            ("R2", "R²", "R² (test)"),
        ],
        "model_labels": {
            "naive_lag1": "naive_lag1",
            "rolling_mean_7": "rolling_mean_7",
            "ols": "ols",
            "random_forest": "random_forest",
        },
    },
    "ja": {
        "suptitle": "時系列ホールドアウト・ベンチマーク",
        "no_data": "ベンチマークデータがありません",
        "metrics": [
            ("MAE", "MAE", "MAE（来訪者数/日）"),
            ("RMSE", "RMSE", "RMSE"),
            ("R2", "R²", "R²（検証）"),
        ],
        "model_labels": {
            "naive_lag1": "1日ラグ",
            "rolling_mean_7": "7日移動平均",
            "ols": "OLS",
            "random_forest": "ランダムフォレスト",
        },
    },
}

_ABLATION_I18N: dict[str, dict[str, object]] = {
    "en": {
        "no_data": "No ablation data",
        "no_scenarios": "No ablation scenarios",
        "mae_title": "RF hold-out: cost of removing a feature family",
        "mae_legend": "Higher = worse fit",
        "mae_ylabel": "ΔMAE vs full\n(visitors/day)",
        "r2_legend": "Negative = worse R²",
        "r2_ylabel": "ΔR² vs full",
        "xlabel": "Features removed",
        "feature_labels": {
            "weather": "weather",
            "google_intent": "google_intent",
            "calendar": "calendar",
        },
    },
    "ja": {
        "no_data": "アブレーションのデータがありません",
        "no_scenarios": "アブレーションのシナリオがありません",
        "mae_title": "RFホールドアウト：特徴量群を除去した場合の影響",
        "mae_legend": "大きいほど適合が悪い",
        "mae_ylabel": "フルモデル比 ΔMAE\n（来訪者数/日）",
        "r2_legend": "負ほどR²が悪い",
        "r2_ylabel": "フルモデル比 ΔR²",
        "xlabel": "除去した特徴量",
        "feature_labels": {
            "weather": "気象",
            "google_intent": "Google検索意向",
            "calendar": "カレンダー",
        },
    },
}


def plot_benchmark_comparison(
    summary_table: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    *,
    dpi: int = 150,
    lang: Literal["en", "ja"] = "en",
    ja_copy: bool = False,
) -> plt.Figure:
    """Grouped bar chart of test MAE / RMSE / R² across baselines and models.

    Args:
        summary_table: DataFrame with columns ``model``, ``MAE``, ``RMSE``, ``R2``.
        out_path: Output PNG path (absolute or under ``fig_dir``).
        reporter: Pipeline ``Reporter``.
        dpi: Figure resolution.
        lang: ``"en"`` or ``"ja"`` for titles and axis labels.
        ja_copy: Passed to ``Reporter.save_fig`` (use ``False`` when saving
            English and Japanese figures as separate explicit paths).

    Returns:
        The matplotlib ``Figure``.
    """
    t = _BENCH_I18N[lang]
    if summary_table.empty:
        fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
        ax.text(0.5, 0.5, str(t["no_data"]), ha="center", va="center")
        ax.axis("off")
        if lang == "ja":
            _apply_benchmark_japanese_font(fig)
        _save(fig, out_path, reporter, dpi=dpi, ja_copy=ja_copy)
        return fig

    df = summary_table.copy()
    models = df["model"].astype(str).tolist()
    model_map: dict[str, str] = t["model_labels"]  # type: ignore[assignment]
    xtick_labels = [model_map.get(m, m) for m in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(11, 5.0), layout="constrained")
    metrics = t["metrics"]  # type: ignore[assignment]
    for ax, (col, title, ylab) in zip(axes, metrics):
        vals = df[col].astype(float).values
        ax.bar(x, vals, width=0.62, color="#4C72B0", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", pad=2)

    fig.suptitle(str(t["suptitle"]), fontsize=12, fontweight="bold")
    if lang == "ja":
        _apply_benchmark_japanese_font(fig)
    _save(fig, out_path, reporter, dpi=dpi, ja_copy=ja_copy)
    return fig


def plot_ablation_impact(
    ablation_table: pd.DataFrame,
    out_path: str,
    reporter: Reporter,
    *,
    dpi: int = 150,
    lang: Literal["en", "ja"] = "en",
    ja_copy: bool = False,
) -> plt.Figure:
    """Bar chart of ΔMAE and ΔR² vs full RF when feature families are removed.

    Args:
        ablation_table: DataFrame with ``scenario``, ``delta_MAE_vs_full``,
            ``delta_R2_vs_full`` (and other columns ignored).
        out_path: Output PNG path.
        reporter: Pipeline ``Reporter``.
        dpi: Figure resolution.
        lang: ``"en"`` or ``"ja"`` for titles and axis labels.
        ja_copy: Passed to ``Reporter.save_fig``.

    Returns:
        The matplotlib ``Figure``.
    """
    t = _ABLATION_I18N[lang]
    if ablation_table.empty or len(ablation_table) < 2:
        fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
        ax.text(0.5, 0.5, str(t["no_data"]), ha="center", va="center")
        ax.axis("off")
        if lang == "ja":
            _apply_benchmark_japanese_font(fig)
        _save(fig, out_path, reporter, dpi=dpi, ja_copy=ja_copy)
        return fig

    sub = ablation_table[ablation_table["scenario"] != "full"].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
        ax.text(0.5, 0.5, str(t["no_scenarios"]), ha="center", va="center")
        ax.axis("off")
        if lang == "ja":
            _apply_benchmark_japanese_font(fig)
        _save(fig, out_path, reporter, dpi=dpi, ja_copy=ja_copy)
        return fig

    feat_map: dict[str, str] = t["feature_labels"]  # type: ignore[assignment]
    raw_labels = [str(s).replace("no_", "") for s in sub["scenario"].astype(str).tolist()]
    labels = [feat_map.get(lbl, lbl) for lbl in raw_labels]
    x = np.arange(len(labels))

    # Two panels: twin y-axes mixed ΔMAE (≈10²) with ΔR² (≈10⁻¹) and broke bar
    # alignment/clipping; stack separate axes with a shared x instead.
    fig, (ax_mae, ax_r2) = plt.subplots(
        2,
        1,
        figsize=(8, 5.5),
        sharex=True,
        height_ratios=[1.05, 1.0],
        layout="constrained",
    )
    d_mae = sub["delta_MAE_vs_full"].astype(float).values
    d_r2 = sub["delta_R2_vs_full"].astype(float).values

    ax_mae.bar(
        x,
        d_mae,
        width=0.55,
        color="#DD8452",
        edgecolor="black",
        linewidth=0.5,
        label=str(t["mae_legend"]),
    )
    ax_mae.axhline(0, color="black", linewidth=0.9)
    ax_mae.set_ylabel(str(t["mae_ylabel"]))
    ax_mae.grid(axis="y", alpha=0.3)
    ax_mae.legend(loc="upper right", fontsize=7)
    ax_mae.set_title(str(t["mae_title"]))

    ax_r2.bar(
        x,
        d_r2,
        width=0.55,
        color="#55A868",
        edgecolor="black",
        linewidth=0.5,
        label=str(t["r2_legend"]),
    )
    ax_r2.axhline(0, color="black", linewidth=0.9)
    ax_r2.set_ylabel(str(t["r2_ylabel"]))
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, rotation=15, ha="center", fontsize=9)
    ax_r2.set_xlabel(str(t["xlabel"]))
    ax_r2.grid(axis="y", alpha=0.3)
    ax_r2.legend(loc="lower right", fontsize=7)

    if lang == "ja":
        _apply_benchmark_japanese_font(fig)
    _save(fig, out_path, reporter, dpi=dpi, ja_copy=ja_copy)
    return fig
