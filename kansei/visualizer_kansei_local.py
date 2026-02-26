from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


def _configure_japanese_font() -> None:
    try:
        import japanize_matplotlib  # noqa: F401
        plt.rcParams["axes.unicode_minus"] = False
        return
    except Exception:
        pass

    def _set_sans_fallback(primary: str) -> None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [primary, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    font_file_candidates = [
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for font_file in font_file_candidates:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
            family_name = font_manager.FontProperties(fname=str(font_file)).get_name()
            _set_sans_fallback(family_name)
            return

    preferred = [
        "Droid Sans Fallback",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "Hiragino Sans",
        "TakaoGothic",
        "MS Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for family in preferred:
        if family in available:
            _set_sans_fallback(family)
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False


def plot_kansei_keywords_ja(
    keyword_df: pd.DataFrame,
    ratio_1star_to_5star: float,
    out_path: str,
    dpi: int = 300,
) -> plt.Figure:
    _configure_japanese_font()
    plot_df = keyword_df.copy().sort_values("rate_1star", ascending=True)
    y = np.arange(len(plot_df))
    h = 0.35

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + h / 2, plot_df["rate_1star"], height=h,
            color="#D35454", label="1★（孤独感）")
    ax.barh(y - h / 2, plot_df["rate_5star"], height=h,
            color="#2E86C1", label="5★（活気）")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["keyword"])
    ax.set_xlabel("出現率（1,000レビューあたり）")
    ax.set_title("過少な賑わいキーワード比較（1★ vs 5★）", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    ax.text(
        0.98,
        0.97,
        f"全体比率（1-2★ / 4-5★）: {ratio_1star_to_5star:.1f}倍",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF5E7", edgecolor="#F5CBA7"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return fig


def plot_eiheiji_threshold_ja(
    scatter_df: pd.DataFrame,
    a: float,
    b: float,
    c: float,
    peak_x: float,
    peak_y: float,
    out_path: str,
    dpi: int = 300,
) -> plt.Figure:
    _configure_japanese_font()
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.scatter(
        scatter_df["relative_density"],
        scatter_df["satisfaction"],
        s=35,
        alpha=0.55,
        color="#5DADE2",
        edgecolors="none",
        label="日次観測値",
    )

    x_line = np.linspace(0, max(100.0, float(scatter_df["relative_density"].max()) + 2), 200)
    y_line = a * x_line**2 + b * x_line + c
    ax.plot(x_line, y_line, color="#C0392B", linewidth=2.5, label="2次回帰曲線")

    ax.axvline(peak_x, color="#7D3C98", linestyle="--", linewidth=1.8, alpha=0.9)
    ax.scatter([peak_x], [peak_y], color="#7D3C98", s=80, zorder=5, label="満足度ピーク")

    ax.set_xlabel("相対密度（%）")
    ax.set_ylabel("平均満足度（1〜5）")
    ax.set_title("永平寺における静寂閾値：2次回帰による推定", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    ax.text(
        0.98,
        0.97,
        f"x* = -b/(2a) = {peak_x:.1f}%\n"
        f"y* = {peak_y:.2f}\n"
        f"f(x) = {a:.6f}x² {b:+.6f}x {c:+.6f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4ECF7", edgecolor="#D2B4DE"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return fig


def plot_di_heatmap_ja(
    corr_df: pd.DataFrame,
    out_path: str,
    dpi: int = 300,
) -> plt.Figure:
    _configure_japanese_font()
    label_map = {
        "corr_di_sat": "相関(DI, 満足度)",
        "corr_wc_sat": "相関(体感温度, 満足度)",
        "corr_di_wc": "相関(DI, 体感温度)",
    }
    plot_df = corr_df.rename(columns=label_map).copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Pearson 相関r"},
        ax=ax,
    )
    ax.set_title("4ノードにおける不快指数・体感温度・満足度の相関", fontweight="bold")
    ax.set_xlabel("指標")
    ax.set_ylabel("観光ノード")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return fig
