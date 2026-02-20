# hokuriku-tourism-ai-governance

## AI-Driven Visitor Demand Forecasting & Under-vibrancy Analysis for Fukui Prefecture

**PhD Research & Hokuriku Regional Tourism AI Governance Grant**

This repository contains the reproducible analysis pipeline for predicting daily visitor counts at Tojinbo (東尋坊), Fukui Prefecture, Japan, using AI camera data, JMA weather observations, Route Search Impressions (RSI) route-search intent, and Hokuriku-wide tourism surveys.

### Key Findings

| Finding | Value |
|---|---|
| OLS R² | 0.811 (Adj R² = 0.803) |
| RF 5-fold CV R² | 0.560 ± 0.127 |
| First-Difference R² (autocorrelation-corrected) | 0.701 |
| LDV R² / DW | 0.848 / 1.918 |
| #1 Predictor | `directions` (RSI) (route search), r = +0.781 |
| Ishikawa → Tojinbo cross-prefectural signal | r = +0.537 |
| Visitors vs Satisfaction | r = +0.161 (p = 0.001) — NO overtourism |
| Lost Visitors ("Satake Number") | 85,400 |
| Winter weather sensitivity | 6.4× worse than summer |
| Under-vibrancy in low-satisfaction reviews | 11.4× more prevalent (6.2% vs 0.5%) |
| Fukui national visitor ranking | 47th / 47 prefectures (winter) |

### Repository Structure

```
hokuriku-tourism-ai-governance/
├── deep_analysis_tojinbo.py      # Main analysis pipeline (16 sections)
├── requirements.txt              # Python dependencies
├── deep_analysis_results.txt     # Full text report output
├── bolstered_results.txt         # Grant-ready metrics summary
├── figures/                      # All generated charts (12 PNGs)
│   ├── fig1  – Time series (visitors vs RSI intent)
│   ├── fig2  – Correlation heatmap
│   ├── fig3  – Feature importance (MDI + Permutation)
│   ├── fig4  – Day-of-week boxplot
│   ├── fig5  – RF predicted vs actual
│   ├── fig6  – Opportunity gap scatter
│   ├── fig7  – Lag correlation bar chart
│   ├── fig8  – Ishikawa → Tojinbo CCF
│   ├── fig9  – Kansei overtourism threshold
│   ├── fig10 – Lost population waterfall
│   ├── fig11 – Fukui Resurrection (ranking simulation)
│   └── fig12 – Hokuriku demand heatmap
├── jma/                          # JMA weather observations (included)
├── ../fukui-kanko-people-flow-data/  # AI camera daily counts (sibling repo)
├── ../fukui-kanko-trend-report/      # Route Search Impressions (RSI) data (sibling repo)
├── ../opendata/                      # Hokuriku tourism survey data (sibling repo)
└── README.md
```

### Data Sources

| Source | Description | Coverage |
|---|---|---|
| **AI Camera** (Tojinbo-Shotaro) | Daily person counts from edge-AI camera | 2024-12 → 2026-02 |
| **JMA** (Japan Meteorological Agency) | Hourly precip, temp, sun, wind | 2024-01 → 2026-02 |
| **Route Search Impressions (RSI)** | Daily route searches, map views, reviews for 47 Fukui tourism locations | 2024-01 → 2026-02 |
| **Hokuriku Tourism Survey** | 95,653 responses (satisfaction, NPS, free text) across Fukui/Ishikawa/Toyama | 2023 → 2026 |

### Data Source Attribution

> **JMA Weather Data**
> This repository utilizes weather data provided by the Japan Meteorological Agency (JMA).
> - **Source:** [JMA Past Weather Data Download](https://www.data.jma.go.jp/gmd/risk/obsdl/index.php)
> - **License:** [Terms of Use of the Japan Meteorological Agency website](https://www.jma.go.jp/jma/kishou/info/coment.html) (Compatible with CC BY 4.0 — Government of Japan Standard Terms of Use v2.0)
> - **Processing:** The raw hourly data has been merged, cleaned, and processed by the author into `jma/jma_hourly_cleaned_merged_8fields.csv` for demand forecasting analysis. Processing script: [`jma/orig-full-monthly/merge_clean_jma_8fields.py`](jma/orig-full-monthly/merge_clean_jma_8fields.py)

### Analysis Pipeline (16 Sections)

1. Data loading & cleaning (zero-day exclusion, outlier flags, ADF stationarity)
2. Feature engineering (calendar, weather severity, rolling intent, lags, interactions)
3. Multi-variable modelling (OLS + Random Forest)
4. Opportunity Gap analysis (high intent, low arrivals)
5. Negative lag-2 correlation explanation
6. Visualisations (7 core figures)
7. Cross-prefectural signal test (Ishikawa → Fukui pipeline)
8. Kansei (emotional) feedback loop — overtourism threshold
9. Lost Population quantification ("Satake Number")
10. Model robustness (Durbin-Watson, VIF, sensitivity)
11. Hokuriku demand heatmap
12. **Autocorrelation fix** (Newey-West HAC, First-Difference, LDV)
13. **Ranking impact simulation** (Fukui Resurrection)
14. **Seasonal weather sensitivity** (Winter 6.4× worse)
15. **Qualitative under-vibrancy link** (survey text mining)
16. **Fukui Resurrection chart**

### Reproducing

```bash
# Clone this repo alongside the data repos in a shared parent directory
git clone https://github.com/YOUR_ORG/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance
pip install -r requirements.txt

# Ensure sibling data repos exist at ../fukui-kanko-people-flow-data,
# ../fukui-kanko-trend-report, and ../opendata
python deep_analysis_tojinbo.py
```

Outputs are written to the repo root (text reports) and `figures/` (charts).

### Citation

If you use this work, please cite:

```
@misc{hokuriku-tourism-ai-governance-2026,
  author = {Amil},
  title  = {AI-Driven Visitor Demand Forecasting and Under-vibrancy Analysis for Fukui Prefecture},
  year   = {2026},
  url    = {https://github.com/YOUR_ORG/hokuriku-tourism-ai-governance}
}
```

### License

MIT
