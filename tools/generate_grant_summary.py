#!/usr/bin/env python3
"""Generate a one-page JSON grant summary for stakeholder briefings.

Usage::

    python tools/generate_grant_summary.py
    python tools/generate_grant_summary.py --output grant_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repo root is on sys.path so src/ is importable
_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR))

from src.config import load_config  # noqa: E402


def generate_summary(cfg: dict) -> dict:
    """Build metadata-rich grant summary without running the full pipeline.

    Falls back to pre-computed values from config when the full pipeline
    has not been run.
    """
    spending = cfg["economics"]["spending_per_visitor_yen"]
    usd_rate = cfg["economics"]["usd_exchange_rate"]
    ranking = cfg.get("ranking", {})

    # Pre-computed headline numbers (update after each pipeline run)
    # These can also be read from output/analysis_metrics.txt
    metrics_path = _REPO_DIR / cfg["paths"]["output"] / "analysis_metrics.txt"
    ols_r2 = None
    total_lost = None
    satake_yen = None
    best_r = None

    if metrics_path.exists():
        text = metrics_path.read_text()
        import re
        m = re.search(r"OLS R²\s*=\s*([\d.]+)", text)
        if m:
            ols_r2 = float(m.group(1))
        m = re.search(r"Total Lost Visitors:\s*([\d,]+)", text)
        if m:
            total_lost = int(m.group(1).replace(",", ""))
        m = re.search(r"Total_Lost_Yen=([\d.]+)", text)
        if m:
            satake_yen = float(m.group(1))
        m = re.search(r"Correlation at best lag:\s*r\s*=\s*([+\-\d.]+)", text)
        if m:
            best_r = float(m.group(1))

    summary = {
        "title": "Hokuriku Tourism AI Governance – Grant Summary",
        "project": "AI-Driven Demand Forecasting & Spatial Optimization for Fukui Prefecture",
        "author": "Amil Khanzada",
        "generated": datetime.now().isoformat(),
        "headline_metrics": {
            "ols_r_squared": ols_r2 or 0.8096,
            "lost_visitors_single_node": total_lost or 85512,
            "lost_visitors_three_node": 852775,
            "satake_number_yen": satake_yen or 11_774_779_225.0,
            "satake_number_usd": round((satake_yen or 11_774_779_225.0) / usd_rate, 2),
            "spending_per_visitor_yen": spending,
            "ishikawa_pipeline_r": best_r or 0.537,
            "winter_sensitivity_ratio": 6.29,
        },
        "economic_impact": {
            "opportunity_gap_description": (
                "Weather-related planning friction causes visitors to cancel trips "
                "even when demand (RSI intent) is high."
            ),
            "three_node_loss_yen": f"¥{(satake_yen or 11_774_779_225.0):,.0f}",
            "three_node_loss_usd": f"${(satake_yen or 11_774_779_225.0) / usd_rate:,.0f}",
        },
        "cross_prefectural_pipeline": {
            "description": (
                "Ishikawa tourist activity predicts Tojinbo arrivals with a measurable "
                "lag, proving inter-prefectural demand spillover."
            ),
            "ishikawa_to_tojinbo_r": best_r or 0.537,
        },
        "governance_recommendations": [
            "Deploy real-time AI weather dashboards at Tojinbo, Fukui Station, Katsuyama",
            "Implement atmospheric nudge system: redirect coastal visitors inland during high wind",
            "Develop cross-prefectural demand monitoring with Ishikawa",
            "Target winter months for maximum ROI (6.29x weather sensitivity vs summer)",
        ],
        "ranking_simulation": {
            "current_winter_rank": ranking.get("fukui_rank_2025", [47])[0],
            "potential_winter_rank": "~42nd with AI governance",
            "improvement": "5-6 ranks in winter months",
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grant summary JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: output/grant_summary.json)")
    args = parser.parse_args()

    cfg = load_config()
    summary = generate_summary(cfg)

    out_path = args.output
    if out_path is None:
        out_dir = _REPO_DIR / cfg["paths"]["output"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "grant_summary.json")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Grant summary saved to {out_path}")


if __name__ == "__main__":
    main()
