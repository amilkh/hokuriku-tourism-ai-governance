# Hokuriku Tourism AI Governance

> **Executive Reports:**
> - [English Executive Report](output/EXECUTIVE_REPORT.md)
> - [日本語エグゼクティブレポート](output/EXECUTIVE_REPORT.ja.md)
>
> **Read in other languages:**
> - [日本語 (Japanese)](README.ja.md)

## AI-Driven Visitor Demand Forecasting & Under-vibrancy Analysis for Fukui Prefecture

**PhD Research & Hokuriku Regional Tourism AI Governance Grant**

This repository contains a reproducible analysis pipeline for predicting daily visitor counts at Tojinbo (東尋坊), Fukui Prefecture, Japan, using AI camera data, JMA weather observations, Route Search Impressions (RSI) route-search intent, and Hokuriku-wide tourism surveys.

### Key Findings

| Finding | Value |
|---|---|
| OLS R² | 0.810 (Adj R² = 0.802) |
| RF 5-fold CV R² | 0.557 ± 0.131 |
| First-Difference R² (autocorrelation-corrected) | 0.708 |
| LDV R² / DW | 0.848 / 1.893 |
| #1 Predictor | `directions` (RSI) (route search), r = +0.781 |
| Ishikawa → Tojinbo cross-prefectural signal | r = +0.537 |
| Visitors vs Satisfaction | r = +0.161 (p = 0.001) — NO overtourism |
| Lost Visitors (Opportunity Gap) | 85,512 |
| Winter weather sensitivity | 6.29× worse than summer |
| Under-vibrancy in low-satisfaction reviews | 11.4× more prevalent (6.2% vs 0.5%) |
| Fukui national visitor ranking | 47th / 47 prefectures (winter) |

### Repository Structure

```
hokuriku-tourism-ai-governance/
├── deep_analysis_tojinbo.py      # Main analysis pipeline
├── requirements.txt              # Python dependencies
├── jma/                          # JMA weather observations (included)
├── output/                       # Generated artifacts
│   ├── EXECUTIVE_REPORT.md
│   ├── EXECUTIVE_REPORT.ja.md
│   ├── *.png                     # Generated figures (EN & JA)
│   └── analysis_metrics.txt      # Machine-readable metrics summary
├── ../fukui-kanko-people-flow-data/  # AI camera daily counts (sibling repo)
├── ../fukui-kanko-trend-report/      # Route Search Impressions (RSI) data (sibling repo)
├── ../opendata/                      # Hokuriku tourism survey data (sibling repo)
├── README.md
└── README.ja.md
```

### Data Sources

| Source | Description | Coverage |
|---|---|---|
| **AI Camera** (Tojinbo-Shotaro) | Daily person counts from edge-AI camera | 2024-12 → 2026-02 |
| **JMA** (Japan Meteorological Agency) | Hourly precip, temp, sun, wind | 2024-01 → 2026-02 |
| **Route Search Impressions (RSI)** | Daily route searches, map views, reviews for 47 Fukui tourism locations | 2024-01 → 2026-02 |
| **Hokuriku Tourism Survey** | 95,653 responses (satisfaction, NPS, free text) across Fukui/Ishikawa/Toyama | 2023 → 2026 |

### Reproducing

To reproduce this analysis, clone this repository with its data dependencies under the same parent directory:

```bash
# 1) Create workspace
mkdir hokuriku-workspace && cd hokuriku-workspace

# 2) Clone sibling data repositories
git clone https://github.com/YOUR_ORG/fukui-kanko-people-flow-data.git
git clone https://github.com/YOUR_ORG/fukui-kanko-trend-report.git
git clone https://github.com/YOUR_ORG/opendata.git

# 3) Clone this repository
git clone https://github.com/YOUR_ORG/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance

# 4) Install and run
pip install -r requirements.txt
python deep_analysis_tojinbo.py
```

Outputs are written to the `output/` directory.

### Citation

If you use this work, please cite:

```
@misc{hokuriku-tourism-ai-governance-2026,
  author = {Amil Khanzada},
  title  = {AI-Driven Visitor Demand Forecasting and Under-vibrancy Analysis for Fukui Prefecture},
  year   = {2026},
  url    = {https://github.com/YOUR_ORG/hokuriku-tourism-ai-governance},
  note   = {Specially Appointed Professor, Regional Revitalization Lab, University of Fukui}
}
```
