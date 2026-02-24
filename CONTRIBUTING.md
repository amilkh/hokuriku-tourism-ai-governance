# Contributing to hokuriku-tourism-ai-governance

Thank you for considering contributing to this project!

## Project Structure

```
hokuriku-tourism-ai-governance/
├── run_analysis.py            # Main modular pipeline entry-point
├── config/
│   └── settings.yaml          # Pipeline configuration
├── src/                       # Python package
│   ├── __init__.py
│   ├── config.py              # Config loader
│   ├── report.py              # Reporter class
│   ├── data_loader.py         # Camera, weather, RSI, survey loading
│   ├── feature_engineering.py # Calendar, lags, interactions
│   ├── models.py              # OLS, Random Forest, robustness
│   ├── kansei.py              # Discomfort Index, satisfaction, text mining
│   ├── spatial.py             # Cross-prefectural, multi-node analysis
│   ├── economics.py           # Opportunity gap, lost population, ranking
│   ├── visualizer.py          # All figure generation
│   └── latex_export.py        # LaTeX table generation
├── tools/
│   └── generate_grant_summary.py
├── jma/                       # JMA weather data (hourly CSVs)
├── output/                    # Generated reports, figures, metrics
├── figures/                   # Publication-ready figures
└── README.md / README.en.md   # Bilingual documentation
```

## Development Setup

```bash
# Clone and install dependencies
git clone https://github.com/amilkh/hokuriku-tourism-ai-governance.git
cd hokuriku-tourism-ai-governance
pip install -r requirements.txt

# Run the pipeline
python run_analysis.py

# Generate grant summary
python tools/generate_grant_summary.py
```

## Dependencies

- Python 3.10+
- numpy, pandas, matplotlib, seaborn
- scikit-learn, statsmodels, scipy
- jpholiday, PyYAML
- japanize-matplotlib (optional, for Japanese figure labels)

## Code Style

- **Type hints**: All function signatures use Python 3.10+ style (`X | None`)
- **Docstrings**: structured with Args, Returns, Raises sections
- **Imports**: `from __future__ import annotations` at top of every module
- **Naming**: snake_case for functions/variables, PascalCase for classes

## Adding a New Analysis Module

1. Create `src/your_module.py` with functions that accept `(data, reporter)` params
2. Import and call from `run_analysis.py`
3. Add config keys to `config/settings.yaml` if needed
4. Update this file and `API_REFERENCE.md`

## Data Sources

All data paths are configured in `config/settings.yaml`. The pipeline expects
sibling repos at the workspace root:

- `fukui-kanko-people-flow-data/` – AI camera people-flow CSVs
- `fukui-kanko-trend-report/` – Route Search Impressions (RSI) data
- `opendata/` – Merged Hokuriku tourism survey CSVs
- `fukui-kanko-survey/` – Raw Fukui survey data

## Commit Conventions

Follow conventional commit format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation updates
- `refactor:` code restructuring

## License

Copyright © Amil Khanzada. All rights reserved.

No license is granted at this time. Reuse, redistribution, or publication
requires explicit written permission from the author.
