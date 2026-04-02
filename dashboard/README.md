# 🏯 Hokuriku Tourism AI Governance — Interactive Dashboard

Interactive Streamlit dashboard for exploring the DHDE (Distributed Human
Data Engine) framework outputs: weather–tourism correlations, the ¥11.96 B
Opportunity Gap, Kansei comfort indices, spatial saturation analysis, and
NLP-driven survey insights.

## Features

| Tab | Description | Techniques |
|-----|-------------|------------|
| 📊 Overview | KPI summary, weather distributions, satisfaction overview | Descriptive statistics |
| 🌤️ Weather & Tourism | Time series, correlations, seasonal analysis | Pearson correlation, violin plots |
| 💰 Opportunity Gap | Interactive gap calculator, severity scoring | Weather severity index, seasonal decomposition |
| 🌡️ Kansei Comfort | Discomfort Index, Wind Chill, interactive gauges | DI & WC formulas from Kansei engineering |
| 🗺️ Spatial Network | Multi-node comparison, weather shield analysis | Cross-station correlation, mapbox |
| 📝 NLP Insights | Sentiment, topics, keywords, word clouds, n-grams | TF-IDF, LDA, lexicon sentiment, Janome tokenisation |

## Quick Start

```bash
# From project root
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

## NLP Engine

The `nlp_engine.py` module provides bilingual (Japanese + English) NLP:

- **Tokenisation**: Janome morphological analyser (pure Python) with regex fallback
- **Keywords**: TF-IDF extraction with domain-aware stop-words
- **Topics**: Latent Dirichlet Allocation via scikit-learn
- **Sentiment**: Bilingual tourism-focused lexicon (36 JA + 26 EN terms)
- **N-grams**: Bigram and trigram frequency analysis
- **Word Cloud**: Weighted term frequency maps

## Data Sources

The dashboard auto-detects available data:

| Source | Location | Status |
|--------|----------|--------|
| JMA Weather | `jma/jma_*.csv` (committed) | ✅ Always available |
| AI Camera | `../fukui-kanko-people-flow-data/` | Optional (clone sibling) |
| Survey | `../fukui-kanko-survey/` | Demo data if unavailable |

When sibling repos are not cloned, the dashboard generates realistic
synthetic data marked with ⚠️ in the sidebar status panel.

## Architecture

```
dashboard/
├── app.py           # Streamlit entry point (6-tab dashboard)
├── data_bridge.py   # Data loading & Kansei enrichment
├── nlp_engine.py    # Bilingual NLP pipeline
└── README.md        # This file
```

The dashboard follows the DHDE project's design principles:
- **No `print()` calls**  (all output via Streamlit/logging)
- **Graceful degradation** (works with partial data)
- **Separation of concerns** (data, NLP, and UI in distinct modules)