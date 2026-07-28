---
documentclass: article
classoption:
  - twocolumn
geometry:
  - margin=0.45in
fontsize: 9pt
mainfont: Arial
header-includes:
  - \usepackage{graphicx}
  - \usepackage{float}
---

```{=latex}
\twocolumn[
\begin{center}
{\LARGE \textbf{HOKURIKU TOURISM AI GOVERNANCE: STRATEGIC EXECUTIVE REPORT}}\\
\vspace{6pt}
{\large Distributed Human Data Engine (DHDE) for Visitor Demand Forecasting \& Spatial Under-Vibrancy Analysis, Fukui Prefecture, Japan}\\
\vspace{6pt}
Amil Khanzada, Specially Appointed Associate Professor, Headquarters for Regional Revitalization, University of Fukui \\
March 26, 2026
\vspace{10pt}
\end{center}
]
```

## Executive Summary

- **Core problem:** Fukui ranks 47th/47 in winter tourism. Root cause is Planning Friction — digital intent fails to convert to physical visits.
- **Quantified loss:** 4 DHDE nodes: 865,917 lost visitors (4 nodes) per year; Opportunity Gap ~¥11.96B (~USD 72.6M).
- **Forecast validity:** Google intent predicts visits at R² = 0.810; JMA weather adds +5.6% accuracy.
- **Policy target:** Two AI nudges can raise Fukui's ranking from 47th to ~35th.

## Key Metrics

```{=latex}
\begin{table}[H]
\small
\begin{tabular}{p{4.1cm}l}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
OLS R² / Adj. R² & 0.8096 / 0.8016 \\
RF 5-fold CV R² & 0.557 ± 0.131 \\
Top predictor & Weekend/Holiday (β=+0.547) \\
Lost visitors (4 nodes) & 865,917/year \\
Opportunity Gap & ~¥11.96B (~USD 72.6M) \\
Winter sensitivity & 6.26× summer \\
Ishikawa → Fukui lead & r = 0.549 \\
Under-vibrancy ratio & 11.5× \\
Winter national rank & 47th / 47 \\
\hline
\end{tabular}
\end{table}
```

## 1. Problem Redefinition

Against the "lack of tourism resources" hypothesis, this study demonstrates conversion-rate suppression via planning friction: Google search/route intent is strong, but snow, rain, and wind suppress winter visits (6.26× summer sensitivity), and perceived emptiness anchors satisfaction downward.

**Policy focus:** Improve intent→visit conversion over creating new resources.

## 2. DHDE Data Infrastructure

4-node geographic saturation: Tojinbo (coastal north), Fukui Station (urban centre), Katsuyama (mountain east), Rainbow Line (scenic south).

- Digital intent: Google Business Profile (47 locations)
- Environmental filter: JMA observations
- Ground truth: AI camera counts (5-min intervals, ~170K rows)
- Behavioural sensor: Hokuriku survey (97,719) + spending records (90,317)

## 3. Key Findings

### 3.1 Forecast Accuracy & Weather Shield

Accuracy: R² = 0.810 (adj. 0.802) — 81% of daily variation explained. Top predictor: Weekend/holiday (standardised β = +0.547). Weather value: JMA variables boost accuracy by +5.6%.

```{=latex}
\begin{figure}[H]
\centering
\includegraphics[height=3.5cm]{output/paper_fig2_rf_prediction.png}
\caption{\small Predicted demand vs. AI camera actuals (R² = 0.810).}
\end{figure}
```

### 3.2 Under-vibrancy Paradox & Silence Threshold

Text mining of 71,623 free-text reviews shows Fukui's core challenge is under-vibrancy, not overtourism: visitor count and satisfaction are positively correlated (p = 0.002). Low-satisfaction visitors (1–2★) use "lonely/closed/empty" vocabulary 11.5× more than high-satisfaction visitors. Eiheiji satisfaction peaks near 42.4% relative density; Tojinbo crowding raises satisfaction.

### 3.3 Economic Opportunity Gap (¥11.96B)

Proxy gap = difference between OLS-predicted visits (Google-intent baseline) and AI camera actuals on weather-degraded days, scaled by mean spending (¥13,811, n = 90,317). 42 high-friction days at the primary node, extrapolated to 4 nodes: 865,917 lost visitors/year. Total annual revenue loss: ~¥11.96B (~USD 72.6M). Winter is 6.26× more weather-sensitive — the highest-ROI intervention window.

## 4. Regional Linkage: Ishikawa Pipeline

Cross-prefectural CCF analysis confirms Ishikawa tourism activity is a significant leading indicator of Fukui visits (r = 0.549). Ishikawa and Fukui form a single "Hokuriku impression space"; single-prefecture optimisation is structurally insufficient.

## 5. Policy: Socio-Technical Nudge Loop

Two AI nudges share a 72-hour forecast:

1. **Supply-side (Shop Activation Alert):** Forecast-triggered business-hours/staffing optimisation resolves the "closed shop" problem (11.5× under-vibrancy ratio).
2. **Demand-side (Weather Routing):** On bad-weather days, redirects Tojinbo demand to indoor sites (Katsuyama, Eiheiji), retaining spending within Hokuriku.

```{=latex}
\begin{figure}[H]
\centering
\includegraphics[height=4cm]{output/paper_fig5_weather_shield_map.png}
\caption{\small 4-node Weather Shield policy network and demand routing paths.}
\end{figure}
```

## 6. Roadmap & KPIs

```{=latex}
\begin{table}[H]
\small
\begin{tabular}{p{2.3cm}p{3.0cm}p{2.5cm}}
\hline
\textbf{Phase} & \textbf{Scope} & \textbf{Target KPI} \\
\hline
Phase 1 (0–6 mo.) & Tojinbo, Fukui Stn & MAPE <15\% \\
Phase 2 (7–12 mo.) & Supply nudge live & Open rate +20\% \\
Phase 3 (Year 2) & 4 nodes + routing & Lost visitors −100K/yr \\
Phase 4 (Year 3) & Full governance loop & Rank 47th→~35th \\
\hline
\end{tabular}
\end{table}
```

*Reproducibility: All code, pipeline logic, and metrics are version-controlled. Outputs written to analysis_metrics.txt on every pipeline run.*

## Conclusion

The 865,917 annual lost visitors and ~¥11.96B opportunity loss are empirically derived lower-bound estimates from AI camera counts, Google intent signals, JMA observations, and 97,719 survey responses. The positive visitor–satisfaction correlation (p = 0.002) means recovery interventions improve both quantity and quality of vibrancy simultaneously.