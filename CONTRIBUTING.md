# Contributing to hokuriku-tourism-ai-governance

This project is an active research collaboration supporting evidence-based tourism
policy for Hokuriku Prefecture. If you are joining as a data scientist, research
assistant, or policy analyst, this guide covers everything you need to get started.

---

## 1. Prerequisites

- Python 3.10 or higher
- Git
- Access to the four sibling data repositories (see Section 3)
- For PDF generation: `pandoc`, `texlive-xetex`, `fonts-noto-cjk` (Linux/WSL)

---

## 2. Project Overview

The pipeline integrates four data streams into the **Distributed Human Data Engine (DHDE)**:

| Stream | Repository | Description |
|--------|-----------|-------------|
| AI Camera people-flow | `fukui-kanko-people-flow-data` | Edge-AI visitor counts at 4 nodes |
| JMA weather observations | `jma/` (this repo) | Hourly 8-field data for 3 stations |
| Route search impression (RSI) intent | `fukui-kanko-trend-report` | Daily search & route queries |
| Hokuriku Tourism Survey | `opendata`, `fukui-kanko-survey` | 95,653 responses + spending records |

The four geographic nodes cover: Tojinbo/Mikuni (coast), Fukui Station (hub),
Katsuyama (mountain), and Rainbow Line/Wakasa (scenic).

---

## 3. Workspace Setup

All sibling data repositories must sit **next to** this repository in the same
parent directory. The pipeline resolves data paths relative to that workspace root.

```
hokuriku-workspace/               ← workspace root
├── hokuriku-tourism-ai-governance/   ← this repo
├── fukui-kanko-people-flow-data/     ← AI camera CSVs
├── fukui-kanko-trend-report/         ← Route Search Impressions (RSI) data
├── opendata/                         ← merged Hokuriku survey CSVs
└── fukui-kanko-survey/               ← raw Fukui spending survey
```

```bash
# One-time workspace setup
mkdir hokuriku-workspace && cd hokuriku-workspace
git clone https://github.com/code4fukui/fukui-kanko-people-flow-data.git
git clone https://github.com/code4fukui/fukui-kanko-trend-report.git
git clone https://github.com/code4fukui/opendata.git
git clone https://github.com/code4fukui/fukui-kanko-survey.git
git clone https://github.com/amilkh/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance
```

---

## 4. Installation

```bash
# Install as an editable package with dev tools
pip install -e ".[dev]"

# Or minimal install (no pytest/ruff)
pip install .

# For Japanese text mining (kansei module)
pip install ".[nlp]"
```

All runtime dependencies are declared in `pyproject.toml`. The `requirements.txt`
file contains pinned versions for fully reproducible environments.

---

## 5. Running the Pipeline

```bash
# Full analysis pipeline (produces all figures, tables, and metrics)
python -m src.run_analysis

# Or use the console script installed by pip
htag-run

# Custom config (e.g., different workspace layout)
HTAG_CONFIG=/path/to/custom.yaml python -m src.run_analysis
```

All outputs are written to `output/`. Key files:
- `output/analysis_metrics.txt` — machine-readable key metrics
- `output/deep_analysis_fig*.png` — 12 analysis figures (EN & JA)
- `output/table_*.tex` — LaTeX tables for the working paper

---

## 6. Module Architecture

Every module receives a `Reporter` instance for deterministic logging.
No module uses `print()` directly.

```
src/
├── config.py              # Load settings.yaml, resolve paths
├── report.py              # Reporter: centralised logging + metrics writing
├── validator.py           # Schema, drift, outlier, and date-gap checks
├── data_loader.py         # Load all four data streams into DataFrames
├── feature_engineering.py # Calendar, weather severity, lags, interactions
├── models.py              # OLS + Random Forest + full robustness suite
├── kansei.py              # Discomfort Index, Wind Chill, text mining
├── economics.py           # Opportunity gap, lost population, ranking
├── spatial.py             # Cross-prefectural CCF, multi-node analysis
├── visualizer.py          # All figure generation (12+ figures, EN & JA)
├── latex_export.py        # LaTeX table generator for paper submission
└── run_analysis.py        # Pipeline entry-point (calls all modules)
```

Data flow:
```
config.py → data_loader.py → feature_engineering.py
                                    ↓
                 models.py ← kansei.py ← economics.py ← spatial.py
                                    ↓
                 visualizer.py → latex_export.py → report.py
```

---

## 7. Adding a New Analysis Module

1. Create `src/your_module.py`. All public functions should accept `(data, reporter)`.
2. Import and call from `src/run_analysis.py` in the appropriate pipeline stage.
3. Add any new configuration keys to `config/settings.yaml` with comments.
4. Write tests in `tests/test_your_module.py` covering at least the core calculations.
5. Update `API_REFERENCE.md` with the new function signatures.

---

## 8. Updating Weather Data (JMA)

New months of JMA weather data can be added with two commands:

```bash
# 1. Fetch raw hourly CSVs from JMA (repeat for each station)
python jma/fetch_jma_monthly.py --station mikuni    --year 2026 --month 3
python jma/fetch_jma_monthly.py --station fukui     --year 2026 --month 3
python jma/fetch_jma_monthly.py --station katsuyama --year 2026 --month 3

# 2. Merge into canonical per-station CSVs (upserts; safe to re-run)
python jma/merge_clean_jma.py
```

The three merged CSVs (`jma/jma_*_hourly_8.csv`) are committed to the repository
so the pipeline can run without re-fetching. Raw monthly files are gitignored.

---

## 9. Generating PDF Reports

PDF reports are built from the LaTeX Markdown sources in `output/pdf/` using
pandoc + XeLaTeX. Requires Noto Sans CJK JP for Japanese typesetting.

```bash
# Install dependencies (Debian/Ubuntu/WSL)
sudo apt-get install -y pandoc texlive-xetex texlive-lang-japanese fonts-noto-cjk

# Build all PDF reports
python scripts/generate_pdfs.py

# Build specific reports
python scripts/generate_pdfs.py --executive     # Japanese executive report
python scripts/generate_pdfs.py --executive-en  # English executive report
```

Output: `output/pdf/EXECUTIVE_REPORT.pdf`, `output/pdf/EXECUTIVE_REPORT.ja.pdf`

---

## 10. Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run a specific module's tests
pytest tests/test_models.py -v

# Skip slow tests
pytest -m "not slow"
```

The test suite covers formula correctness (Discomfort Index, Wind Chill),
model output shapes and R² bounds, feature engineering determinism, and
validator schema/drift detection.

---

## 11. Code Style

- **Formatter / linter:** `ruff` (configured in `pyproject.toml`)
- **Type hints:** All function signatures use Python 3.10+ syntax (`X | None`)
- **Docstrings:** structured with `Args`, `Returns`, and `Raises` sections
- **Imports:** `from __future__ import annotations` at the top of every module
- **Naming:** `snake_case` for functions and variables, `PascalCase` for classes

Run the linter before committing:
```bash
ruff check src/ tests/
ruff format src/ tests/
```

---

## 12. Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When to use |
|--------|-------------|
| `feat:` | New analysis, new figure, new module |
| `fix:` | Bug fix in calculations or data loading |
| `docs:` | README, CONTRIBUTING, API_REFERENCE updates |
| `refactor:` | Code restructuring with no behaviour change |
| `test:` | Adding or fixing tests |
| `chore:` | Dependency updates, gitignore, tooling |
| `data:` | New JMA data, updated survey CSVs |

Commit body should explain *why*, not just *what* (the diff shows what).

---

## 13. Contact

**Principal Investigator:** Amil Khanzada
**Affiliation:** University of Fukui, Regional Revitalization Lab
**Repository:** https://github.com/amilkh/hokuriku-tourism-ai-governance

For questions about data access, grant applications, or research collaboration,
contact the PI directly. For code issues, open a GitHub issue.
