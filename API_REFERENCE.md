# API Reference

This document summarizes high-traffic public entry points used in the current
pipeline, including privacy/zero-shot additions and benchmark visualization APIs.

## Pipeline entry

- **`python -m src.run_analysis`**: Loads config, validates data, runs analysis
	sections, writes figures under `output/`, and flushes
	`output/analysis_metrics.txt` via `Reporter.save()`.

## Configuration

- **`src.config.load_config`**: Loads `config/settings.yaml` (or
	`HTAG_CONFIG`), and resolves `repo_dir` plus `workspace_root` under
	`_resolved`.

## src/privacy_nlp.py

### get_nlp_model()

Lazy-loads the spaCy Japanese NER model (`ja_core_news_sm`) when installed.

### sanitize_text(text: str) -> str

Best-effort redaction for accidental PII in free text:
- email addresses
- phone numbers
- PERSON entities (spaCy)

Returns text with redaction tokens such as `[REDACTED_EMAIL]`.

### apply_privacy_layer(df: pd.DataFrame, text_columns: list[str]) -> pd.DataFrame

Applies `sanitize_text` to selected DataFrame columns and returns a sanitized
copy.

## src/kansei.py

### run_zero_shot_diagnostics(
- `survey_df: pd.DataFrame`
- `reporter: Reporter | None = None`
- `max_samples: int | None = 3000`
- `text_max_chars: int = 512`
) -> dict[str, float]

Runs zero-shot classification on detractor free text and returns percentage
distribution by category.

Execution is config-gated in `config/settings.yaml`:
- `kansei.zero_shot_enabled` (default `false`)
- `kansei.zero_shot_max_samples`
- `kansei.zero_shot_text_max_chars`

Current labels:
- `weather conditions`
- `poor transportation`
- `language barrier`
- `lack of information`
- `pricing`

## Benchmarking (`src.benchmark`)

### run_benchmark(data, reporter) -> BenchmarkResult

Runs chronological train/test benchmark evaluation on the same `model_df` and
`feature_cols` used by core model fitting.

Expected `data` keys:
- `model_df` (`pd.DataFrame` with `date`, `count`, and all `feature_cols`)
- `feature_cols` (`list[str]`)
- `route_col` (`str`)

Optional:
- `cfg`: full settings dict (reads `benchmark` and `model.random_forest`)

`BenchmarkResult` tables:
- `summary_table`: one row per model (`naive_lag1`, `rolling_mean_7`, `ols`,
	`random_forest`) with `MAE`, `RMSE`, `R2`
- `ablation_table`: RF ablation rows on same split, with `delta_MAE_vs_full`
	and `delta_R2_vs_full`
- `predictions_table`: test rows (`date`, `y_true`, optional `pred_*`)

## Visualizer (`src.visualizer`)

### plot_opportunity_gap_drivers(
- `driver_percentages: dict[str, float]`
- `out_path: str`
- `reporter: Reporter`
- `dpi: int = 300`
) -> `matplotlib.figure.Figure | None`

Plots complaint-driver percentages and writes EN/JA PNG outputs.

### plot_benchmark_comparison(summary_table, out_path, reporter, *, dpi=150, lang="en", ja_copy=False)

Saves a three-panel bar chart (MAE, RMSE, R²) for each row in
`summary_table`.

### plot_ablation_impact(ablation_table, out_path, reporter, *, dpi=150, lang="en", ja_copy=False)

Saves stacked panels for `delta_MAE_vs_full` and `delta_R2_vs_full` on
ablation scenarios (excluding `full`).

### Reporter.save_fig(..., ja_copy=False)

Pipeline convention for explicit bilingual outputs is to save EN and JA paths
explicitly and pass `ja_copy=False` in those code paths.

## Models (`src.models`)

See module docstring for `fit_ols`, `fit_random_forest`, `robustness_suite`,
and `statistical_rigor`.
