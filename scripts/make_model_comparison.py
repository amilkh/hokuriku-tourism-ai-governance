#!/usr/bin/env python3
"""make_model_comparison.py

Produces output/paper_fig_model_comparison.png: a horizontal grouped bar
chart comparing OLS vs Random Forest in-sample and hold-out R² on the same
chronological 80/20 split used by statistical_rigor() in src/models.py.

Run from the repo root:
    python scripts/make_model_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "output" / "paper_fig_model_comparison.png"
TRAIN_PCT = 0.80

# Verified thesis numbers (N=397, 317 train / 80 hold-out, chronological split).
# Source: real_data_verification.md, SUPPLEMENT.md, eaai-v1 tag pipeline run 2026-05-05.
# The live pipeline has grown (21 extra RSI-backfilled rows) so the thesis numbers
# are hardcoded here rather than re-derived to keep the figure consistent with the paper.
THESIS = {
    "ols": {"insample_r2": 0.810, "holdout_r2": 0.683, "holdout_mae": 1793.0},
    "rf":  {"insample_r2": 0.909, "holdout_r2": 0.512, "holdout_mae": 1893.0},
}


def print_summary(ols: dict, rf: dict) -> None:
    hdr = f"{'Model':<16} {'In-sample R²':>14} {'Hold-out R²':>12} {'Hold-out MAE':>14} {'Gap':>8}"
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for name, m in [("OLS", ols), ("Random Forest", rf)]:
        gap = m["holdout_r2"] - m["insample_r2"]
        print(
            f"{name:<16} {m['insample_r2']:>14.3f} {m['holdout_r2']:>12.3f}"
            f" {m['holdout_mae']:>14.1f} {gap:>+8.3f}"
        )
    print(sep + "\n")


def make_figure(ols: dict, rf: dict) -> None:
    LIGHT_BLUE = "#7BB8D4"
    DARK_BLUE = "#1A5276"

    labels = ["OLS", "Random Forest"]
    insample = [ols["insample_r2"], rf["insample_r2"]]
    holdout = [ols["holdout_r2"], rf["holdout_r2"]]

    y = np.array([1.0, 0.0])
    height = 0.32
    gap = height + 0.06

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_in = ax.barh(y + height / 2, insample, height=height,
                      color=LIGHT_BLUE, label="In-sample R²", zorder=3)
    bars_ho = ax.barh(y - height / 2, holdout, height=height,
                      color=DARK_BLUE, label="Hold-out R²", zorder=3)

    for bar, val, ins in zip(bars_ho, holdout, insample):
        diff = val - ins
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{diff:+.3f}",
            va="center", ha="left", fontsize=9, color="#C0392B", fontweight="bold",
        )
        ax.text(
            val - 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="right", fontsize=9, color="white", fontweight="bold",
        )

    for bar, val in zip(bars_in, insample):
        ax.text(
            val - 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="right", fontsize=9, color="white", fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("R²", fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.set_title(
        "OLS vs Random Forest: In-sample vs Hold-out R²",
        fontsize=13, fontweight="bold", pad=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax.grid(axis="y", visible=False)

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.text(
        0.5, -0.04,
        "OLS hold-out R² outperforms RF hold-out R², confirming approximate linearity.",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Figure saved → {OUT_PATH}")


def main() -> None:
    ols = THESIS["ols"]
    rf = THESIS["rf"]
    print("Using verified thesis numbers (N=397, 317 train / 80 hold-out):")
    print_summary(ols, rf)
    make_figure(ols, rf)


if __name__ == "__main__":
    main()
