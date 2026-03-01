# Step 1 — Repository Inventory

**Project:** Hokuriku Tourism AI Governance Framework
**Last updated:** 2026-03-01
**Author:** Amil Khanzada

---

## Purpose

This document provides a point-in-time inventory of all tracked files in the repository, their roles, and their current status. Update this file whenever the directory structure changes significantly.

---

## Root-Level Files

| File | Purpose | Status |
|------|---------|--------|
| `pyproject.toml` | PEP 517/621 package definition; `pip install .` entry-point | Tracked |
| `requirements.txt` | Pinned runtime dependencies | Tracked |
| `README.md` | English project overview, results, reproduction steps | Tracked |
| `README.ja.md` | Japanese translation of README.md | Tracked |
| `CONTRIBUTING.md` | Collaborator onboarding guide (setup, coding standards, PR workflow) | Tracked |
| `DATA_DICTIONARY.md` | Column-name mappings for all raw CSV inputs + output file index | Tracked |
| `DATA_INVENTORY_EXPANSION_ANALYSIS.md` | Analysis of data coverage gaps and expansion opportunities | Tracked |
| `API_REFERENCE.md` | Module-level API reference for all `src/` modules | Tracked |
| `LICENSE` | All Rights Reserved — no reuse without written permission | Tracked |
| `.gitignore` | Excludes virtualenvs, caches, rawdata dirs, and generated artifacts | Tracked |

---

## `src/` — Pipeline Source Code

| File | Role |
|------|------|
| `src/__init__.py` | Package metadata (version, author) |
| `src/config.py` | YAML config loader; resolves all data paths from `config/settings.yaml` |
| `src/data_loader.py` | Loaders for Camera, JMA, Route Search Impressions (RSI), and Survey CSVs |
| `src/feature_engineering.py` | Calendar features, weather severity, rolling/lag intent, interactions |
| `src/models.py` | OLS + Random Forest fitting; full robustness suite (DW, NW-HAC, FD, LDV, VIF) |
| `src/kansei.py` | Discomfort Index, Wind Chill, overtourism threshold, text mining (NLP) |
| `src/economics.py` | Opportunity Gap calculation, lost-visitor quantification, prefecture ranking |
| `src/spatial.py` | Cross-prefectural CCF; multi-node spatial governance analysis |
| `src/validator.py` | Data integrity auditing: schema, data drift (KS), outliers, date gaps |
| `src/visualizer.py` | Generates all 12+ publication figures in English and Japanese variants |
| `src/latex_export.py` | Exports OLS/RF/metrics tables to `.tex` for paper submission |
| `src/report.py` | Centralized `Reporter` class for deterministic logging and metric capture |
| `src/run_analysis.py` | Main pipeline entry-point; orchestrates all modules in order |
| `src/generate_grant_summary.py` | One-off script producing `output/grant_summary.json` (generated artifact, gitignored) |

---

## `config/`

| File | Role |
|------|------|
| `config/settings.yaml` | Single source of truth for all data paths, model params, and spending map |

---

## `tests/`

| File | Coverage |
|------|----------|
| `tests/__init__.py` | Package marker |
| `tests/test_models.py` | OLS R², RF importance, Durbin–Watson, edge cases |
| `tests/test_kansei.py` | Discomfort Index hand-calculations, Wind Chill golden values |
| `tests/test_validator.py` | Schema mismatch, outlier detection, date-gap identification |
| `tests/test_features.py` | Calendar encoding, severity categorisation, lag correctness |
| `tests/test_math.py` | Core statistical function numerical correctness |

---

## `jma/` — JMA Weather Data

| File | Role |
|------|------|
| `jma/jma_mikuni_hourly_8.csv` | Canonical hourly dataset — Mikuni station (coast / Tojinbo proxy) |
| `jma/jma_fukui_hourly_8.csv` | Canonical hourly dataset — Fukui city station (hub) |
| `jma/jma_katsuyama_hourly_8.csv` | Canonical hourly dataset — Katsuyama station (mountain) |
| `jma/fetch_jma_monthly.py` | Scraper: downloads JMA hourly CSVs per station per month |
| `jma/merge_clean_jma.py` | Merge & upsert script: combines rawdata fragments into canonical CSVs |

Raw download folders (`jma/*_rawdata/`) are excluded by `.gitignore`.

---

## `scripts/`

| File | Role |
|------|------|
| `scripts/generate_pdfs.py` | Builds 2-page executive PDFs using Pandoc + XeLaTeX from `output/pdf/*.md` sources |

---

## `output/` — Generated Artifacts (Committed for Reference)

### Reports
| File | Description |
|------|-------------|
| `output/EXECUTIVE_REPORT.md` | English executive summary |
| `output/EXECUTIVE_REPORT.ja.md` | Japanese executive summary |
| `output/analysis_metrics.txt` | Machine-readable headline metrics |

### Figures (12 core figures × 2 language variants = 24 PNGs + extras)
| Pattern | Description |
|---------|-------------|
| `output/deep_analysis_fig1_timeseries*.png` | Visitor count time-series with weather overlay |
| `output/deep_analysis_fig2_correlation*.png` | Feature correlation heatmap |
| `output/deep_analysis_fig3_feature_importance*.png` | Random Forest feature importance |
| `output/deep_analysis_fig4_dow_boxplot*.png` | Day-of-week distribution boxplot |
| `output/deep_analysis_fig5_rf_prediction*.png` | RF vs actual prediction |
| `output/deep_analysis_fig6_opportunity_gap*.png` | Opportunity gap visualisation |
| `output/deep_analysis_fig7_lag_correlations*.png` | RSI intent lag correlations |
| `output/deep_analysis_fig8_ishikawa_ccf*.png` | Ishikawa → Tojinbo cross-correlation |
| `output/deep_analysis_fig9_kansei_scatter*.png` | Discomfort index scatter |
| `output/deep_analysis_fig10_lost_population*.png` | Lost visitors bar chart |
| `output/deep_analysis_fig11_fukui_resurrection*.png` | Rank improvement projection |
| `output/deep_analysis_fig12_hokuriku_heatmap*.png` | Hokuriku-wide weather-sensitivity heatmap |
| `output/spatial_friction_heatmap*.png` | Multi-node spatial friction heatmap |
| `output/weather_shield_network*.png` | Weather Shield Network diagram |
| `output/rank_resurrection_projection*.png` | Prefecture rank resurrection chart |
| `output/ultimate_fig2_vibrancy_threshold*.png` | Vibrancy threshold scatter (nature vs sacred) |

### LaTeX Tables
| File | Description |
|------|-------------|
| `output/table_ols.tex` | OLS regression results |
| `output/table_model_comparison.tex` | OLS vs RF model comparison |
| `output/table_key_metrics.tex` | Headline metrics summary |

### PDF Subdirectory
| File | Description |
|------|-------------|
| `output/pdf/executive_report_pdf_en.md` | Pandoc-ready source for English PDF |
| `output/pdf/executive_report_pdf.md` | Pandoc-ready source for Japanese PDF |
| `output/pdf/EXECUTIVE_REPORT.pdf` | Compiled 2-page English executive PDF |
| `output/pdf/EXECUTIVE_REPORT.ja.pdf` | Compiled 2-page Japanese executive PDF |

---

## `audit/`

| File | Description |
|------|-------------|
| `audit/STEP1_REPO_INVENTORY.md` | This file — current repo structure and file inventory |
| `audit/STEP6_AUDIT_REPORT.md` | Executive-level audit report (problem → data → results → recommendations) |
| `audit/audit_logs/step2_script_runs.log` | Pipeline execution log |
| `audit/audit_logs/step3_csv_quality.log` | CSV quality check results |
| `audit/audit_logs/step4_cross_validate.log` | Cross-validation results |
| `audit/tools/audit_csv_quality.py` | Tool: checks CSV schema and quality against expected schema |
| `audit/tools/audit_cross_validate.py` | Tool: cross-validates pipeline outputs against baseline metrics |

---

## File Count Summary

| Category | Count |
|----------|-------|
| Source code (`src/`) | 14 |
| Tests (`tests/`) | 6 |
| JMA data + scripts (`jma/`) | 5 |
| Output figures (`output/*.png`) | 26 |
| Output reports & tables (`output/`) | 9 |
| Config, docs, scripts | 14 |
| Audit (`audit/`) | 7 |
| **Total tracked files** | **~81** |
