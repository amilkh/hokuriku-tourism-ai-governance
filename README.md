<div align="center">

# Hokuriku Tourism AI Governance Framework

### AI-Driven Visitor Demand Forecasting & Spatial Under-vibrancy Analysis

**Amil Khanzada** — *Specially Appointed Professor, Regional Revitalization Lab, University of Fukui*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Data Validated](https://img.shields.io/badge/rows%20audited-1.4M-brightgreen.svg)](src/validator.py)

> **Executive Reports:**
> [English](EXECUTIVE_REPORT.en.md) ·
> [日本語](EXECUTIVE_REPORT.ja.md)
>
> **他の言語で読む:** [日本語 (Japanese)](README.ja.md)

</div>

---

## Abstract

This repository implements the **Distributed Human Data Engine (DHDE)** — a research framework that fuses heterogeneous tourism data sources (AI camera people-flow, JMA meteorological observations, Route search impression (RSI) intent signals, and 95,653 Hokuriku survey responses) into a unified predictive and diagnostic pipeline.

The system quantifies Fukui Prefecture's structural tourism deficit: the **¥11.96 billion annual Opportunity Gap** — revenue lost due to weather-induced demand suppression during winter months, when Fukui ranks **47th out of 47 prefectures** nationally.

**Keywords:** Tourism Demand Forecasting · Kansei Engineering · Discomfort Index · Spatial Saturation · Under-vibrancy · Hokuriku Regional Governance

---

## 1. Theoretical Framework: The Distributed Human Data Engine (DHDE)

The DHDE integrates four sensor modalities into a single analytical pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                  DISTRIBUTED HUMAN DATA ENGINE                  │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────┐│
│  │ AI Camera  │  │ JMA Weather│  │ RSI (route)  │  │ Hokuriku ││
│  │ People-Flow│  │ 8-Variable │  │ Route Intent │  │  Survey  ││
│  │  (Edge-AI) │  │ (Hourly)   │  │  (Daily)     │  │ (95,653) ││
│  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘  └────┬─────┘│
│        │               │                │               │      │
│        └───────────┬───┴────────────────┴───────┬───────┘      │
│                    │     Feature Engineering     │              │
│                    │   Calendar · Weather Severity│              │
│                    │   Lags · Rolling · Interaction│             │
│                    └──────────────┬───────────────┘              │
│                                  │                              │
│              ┌───────────────────┴───────────────────┐          │
│              │  OLS Regression + Random Forest (RF)  │          │
│              │  Robustness: DW, NW-HAC, FD, LDV, VIF│          │
│              └───────────────────┬───────────────────┘          │
│                                  │                              │
│         ┌────────────────────────┼────────────────────────┐     │
│         │                        │                        │     │
│  ┌──────▼──────┐  ┌─────────────▼───────────┐  ┌─────────▼───┐│
│  │ Opportunity  │  │ Kansei Assessment       │  │ Spatial     ││
│  │ Gap / Lost   │  │ DI · WC · Overtourism   │  │ Saturation  ││
│  │ Population   │  │ Text Mining (NLP)        │  │ Multi-Node  ││
│  └──────────────┘  └─────────────────────────┘  └─────────────┘│
│                                                                 │
│                   ──► analysis_metrics.txt                      │
│                   ──► LaTeX tables for paper                    │
└─────────────────────────────────────────────────────────────────┘
```

**Nodes in the spatial network:**

| Node | Location | Camera Source | Weather Station |
|------|----------|--------------|-----------------|
| A | Tojinbo (東尋坊) | tojinbo-shotaro | Mikuni (JMA) |
| B | Fukui Station East | fukui-station-east | Fukui (JMA) |
| C | Katsuyama (勝山) | katsuyama | Katsuyama (JMA) |
| D | Rainbow Line (レインボーライン) | rainbow-line-parking-lot-1-gate | Fukui (proxy) |

---

## 2. Key Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **OLS R²** | 0.810 (Adj R² = 0.802) | Baseline explanatory power |
| **RF 5-fold CV R²** | 0.557 ± 0.131 | Out-of-sample predictive accuracy |
| **First-Difference R²** | 0.708 | Autocorrelation-corrected |
| **LDV R² / DW** | 0.848 / 1.899 | Dynamic model, clean residuals |
| **#1 Predictor** | `directions` (RSI) | Route-search intent, r = +0.781 |
| **Ishikawa → Tojinbo lag** | r = +0.549 | Cross-prefectural demand pipeline |
| **Visitors vs Satisfaction** | rs = +0.150 (p = 0.002) | **No overtourism** detected |
| **Lost Visitors** | 85,522 (single-node) | Annual Opportunity Gap |
| **Winter Weather Sensitivity** | 6.26× summer | Seasonal asymmetry |
| **Under-vibrancy Ratio** | 11.5× | Low-satisfaction review prevalence |
| **National Ranking (Winter)** | 47th / 47 | Fukui's structural deficit |

---

## 3. The ¥11.96 Billion Opportunity Gap

The **Opportunity Gap** measures the difference between *expected* visitors (based on RSI intent signals) and *actual* arrivals on weather-degraded days:

$$
\text{Lost Visitors}_d = \hat{y}_d^{\text{OLS}} - y_d^{\text{actual}} \quad \text{when} \quad y_d < \hat{y}_d
$$

$$
\text{Total Economic Loss} = \sum_{d \in \mathcal{G}} \text{Lost Visitors}_d \times \bar{S}
$$

where $\bar{S} = ¥13{,}811$ is the mean spending per visitor (from Fukui survey, $n = 95{,}653$), and $\mathcal{G}$ is the set of gap days.

| Component | Value |
|-----------|-------|
| Gap days | 42 (high-friction days) |
| Total lost visitors | 85,522 |
| Mean spending per visitor | ¥13,811 |
| **Total annual revenue loss** | **¥11.96 billion** |

---

## 4. Kansei Environmental Assessment

### 4.1 Discomfort Index (不快指数)

The thermal comfort metric used in this framework:

$$
DI = 0.81 \cdot T + 0.01 \cdot H \cdot (0.99 \cdot T - 14.3) + 46.3
$$

where $T$ is temperature (°C) and $H$ is relative humidity (%).

### 4.2 Wind Chill (体感温度)

$$
WC = 13.12 + 0.6215T - 11.37V^{0.16} + 0.3965TV^{0.16}
$$

where $V$ is wind speed in km/h. Valid for $T \leq 10°C$ and $V > 4.8$ km/h.

### 4.3 Overtourism Threshold

Spearman correlation between daily visitor count and mean satisfaction:

rs(visitors, satisfaction) = +0.150 (p = 0.002)

The **positive** correlation confirms Fukui's problem is *under-vibrancy*, not overtourism. More visitors → higher satisfaction.

---

## 5. Spatial Saturation Map

The multi-node analysis achieves **geographic saturation** of Fukui Prefecture:

```
              ┌──── Node C: Katsuyama (Mountain / East) ────┐
              │                                             │
   Node A: Tojinbo ─── Node B: Fukui Station ─── Node D: Rainbow Line
   (Coastal / North)   (Urban / Central)          (Scenic / South)
```

Each node is modelled independently with local JMA weather, enabling:
- **Weather Shield Network**: When Mikuni (coast) is stormy, Katsuyama (inland) may be clear
- **Demand redistribution** via real-time atmospheric nudging

---

## 6. Model Robustness (PhD-Level Diagnostics)

| Diagnostic | Statistic | Interpretation |
|-----------|-----------|---------------|
| Durbin–Watson (OLS) | 1.005 | Corrected via Newey-West HAC |
| Durbin–Watson (1st-diff) | 2.525 | **Clean** residuals |
| Newey–West HAC | 8 significant | Robust to heteroskedasticity |
| First-Difference R² | 0.708 | Controls for trend |
| LDV R² | 0.848 | Dynamic specification |
| VIF (max) | < 10 | No multicollinearity |
| Weather data value | +0.056 R² | JMA contribution quantified |

---

## 7. Repository Structure

```
hokuriku-tourism-ai-governance/
├── pyproject.toml                # PEP 517/621 package definition → pip install .
├── requirements.txt              # Runtime dependencies (minimum versions)
├── config/
│   └── settings.yaml             # Pipeline configuration (all paths & params)
├── src/
│   ├── __init__.py               # Package metadata
│   ├── config.py                 # YAML config loader & path resolver
│   ├── data_loader.py            # Camera, JMA, RSI, Survey loaders
│   ├── feature_engineering.py    # Calendar, severity, lags, interactions
│   ├── models.py                 # OLS + Random Forest + robustness suite
│   ├── kansei.py                 # Discomfort Index, Wind Chill, text mining
│   ├── economics.py              # Opportunity Gap, lost population, ranking
│   ├── spatial.py                # Cross-prefectural CCF, multi-node governance
│   ├── validator.py              # Data integrity auditing (schema, drift, outliers)
│   ├── visualizer.py             # All figure generation (12+ figures, EN & JA)
│   ├── latex_export.py           # LaTeX table generator for paper
│   ├── report.py                 # Centralized Reporter for logging & metrics
│   └── run_analysis.py           # Main pipeline entry-point
├── tests/
│   ├── test_models.py            # OLS, RF, robustness tests
│   ├── test_kansei.py            # DI & Wind Chill formula verification
│   ├── test_validator.py         # Schema, outlier, drift detection tests
│   ├── test_features.py          # Feature engineering pipeline tests
│   └── test_math.py              # Core statistical function checks
├── jma/                          # JMA weather observations (committed)
│   ├── fetch_jma_monthly.py      # Scraper for JMA hourly CSVs
│   ├── merge_clean_jma.py        # Merge rawdata into per-station CSVs
│   └── jma_*.csv                 # Merged per-station 8-field datasets
├── EXECUTIVE_REPORT.en.md        # English executive report (pandoc → PDF source)
├── EXECUTIVE_REPORT.ja.md        # Japanese executive report (pandoc → PDF source)
├── output/                       # Generated artifacts (committed for reference)
│   ├── analysis_metrics.txt      # Machine-readable key metrics
│   ├── *.png                     # 12+ publication figures (EN & JA variants)
│   ├── *.tex                     # LaTeX tables for paper submission
│   └── pdf/                      # Compiled PDF reports (EN & JA)
├── README.md
└── README.ja.md
```

---

## 8. Data Sources

| Source | Type | Coverage | Rows |
|--------|------|----------|------|
| **AI Camera** (Tojinbo-Shotaro) | Edge-AI person counts (5-min intervals) | 2024-12 → 2026-02 | ~170K |
| **JMA** (Mikuni, Fukui, Katsuyama) | Hourly: precip, temp, sun, wind, humidity, snow | 2024-01 → 2026-02 | ~140K |
| **Route Search Impressions (RSI)** | Daily: route searches, map views, reviews for 47 locations | 2024-01 → 2026-02 | ~35K |
| **Hokuriku Tourism Survey** | Satisfaction, NPS, free text (Fukui/Ishikawa/Toyama) | 2023 → 2026 | **95,653** |
| **Fukui Kanko Survey (raw)** | Spending, demographics, travel patterns | 2022 → 2025 | ~1M |

**Total rows audited by `validator.py`:** ~1.4M

---

## 9. Reproduction Steps

### Setup

```bash
# Create workspace with sibling data repos
mkdir hokuriku-workspace && cd hokuriku-workspace
git clone https://github.com/code4fukui/fukui-kanko-people-flow-data.git
git clone https://github.com/code4fukui/fukui-kanko-trend-report.git
git clone https://github.com/code4fukui/opendata.git
git clone https://github.com/code4fukui/fukui-kanko-survey.git

# Clone and install this repository
git clone https://github.com/amilkh/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance
pip install ".[dev]"
```

### Commands

| Command | What it does |
|---------|-------------|
| `python -m src.run_analysis` | Run full pipeline → figures, metrics, LaTeX tables |
| `pandoc EXECUTIVE_REPORT.en.md --pdf-engine=xelatex -o output/pdf/executive_report_en.pdf` | Build English executive PDF |
| `pandoc EXECUTIVE_REPORT.ja.md --pdf-engine=xelatex -o output/pdf/executive_report_ja.pdf` | Build Japanese executive PDF |
| `pytest` | Run test suite |
| `pytest --cov=src --cov-report=html` | Tests with coverage report |
| `ruff check src/ tests/` | Lint check |

> **PDF prerequisites:** `sudo apt-get install -y pandoc texlive-xetex texlive-lang-japanese fonts-noto-cjk`
>
> **Note:** Set `HTAG_CONFIG=/path/to/settings.yaml` to use a custom config (default: `config/settings.yaml`).

All artifacts are written to `output/`: figures (EN & JA), LaTeX tables, executive reports, and compiled PDFs.

---

## 10. Modular Architecture

The pipeline follows a strict **separation of concerns**:

```python
# Entrypoint: src/run_analysis.py
cfg = load_config()                           # config.py
rpt = Reporter(cfg)                           # report.py
validation = validate_pipeline(cfg, rpt)      # validator.py
data = load_all_data(cfg, rpt)                # data_loader.py
daily, features = build_features(daily, ..)   # feature_engineering.py
ols = fit_ols(model_df, features, rpt)        # models.py
rf  = fit_random_forest(model_df, ..)         # models.py
robust = robustness_suite(model_df, ..)       # models.py
gap = compute_opportunity_gap(daily, ..)      # economics.py
kansei = discomfort_index_analysis(..)        # kansei.py
spatial = multi_node_analysis(cfg, ..)        # spatial.py
export_all_tables(results, ..)                # latex_export.py
```

Every module accepts a `Reporter` instance for deterministic logging. No module uses `print()` directly — all output flows through the centralized reporter.

---

## 11. Testing & Validation

### Test Suite

```
tests/
├── test_models.py     # OLS R², RF importance, DW, edge cases
├── test_kansei.py     # DI hand-calculations, wind chill, golden values
├── test_validator.py  # Schema, outliers, date gaps, drift detection
├── test_features.py   # Calendar, severity, lags, encodings
└── test_math.py       # Core statistical function correctness
```

### Data Validation (`src/validator.py`)

Automatically audits every data source for:
- **Schema mismatches** — columns added/removed between data versions
- **Data drift** — Kolmogorov–Smirnov tests on 3-month sliding windows
- **Outliers** — IQR and Z-score detection per column
- **Date gaps** — Missing days in time-series continuity
- **Domain violations** — Negative precipitation, extreme temperatures

Results are included in `output/analysis_metrics.txt`.

---

## 12. Citation

```bibtex
@misc{khanzada2026hokuriku,
  author       = {Khanzada, Amil},
  title        = {Hokuriku Tourism {AI} Governance Framework:
                  Distributed Human Data Engine for Visitor Demand
                  Forecasting and Spatial Under-vibrancy Analysis},
  year         = {2026},
  institution  = {University of Fukui, Regional Revitalization Lab},
  url          = {https://github.com/amilkh/hokuriku-tourism-ai-governance},
  note         = {PhD Research -- Kanazawa University \&
                  University of Fukui Joint Program.
                  Supported by the Hokuriku Regional Tourism
                  AI Governance Grant.}
}
```

---

## License

Copyright © 2024–2026 Amil Khanzada. All rights reserved.

No license is granted at this time. Reuse, redistribution, or publication requires explicit written permission from the author.

