# API reference (selected modules)

This document summarizes public entry points that extension authors and reviewers most often touch.  For full behaviour, see docstrings in the source files.

## Pipeline entry

- **`python -m src.run_analysis`** — Loads config, validates data, runs analysis sections, writes figures under `output/`, and flushes `output/analysis_metrics.txt` via `Reporter.save()`.

## Configuration

- **`src.config.load_config`** — Loads `config/settings.yaml` (or `HTAG_CONFIG`), resolves `repo_dir` and `workspace_root` under `_resolved`.

## Benchmarking (`src.benchmark`)

### `run_benchmark(data, reporter) -> BenchmarkResult`

Runs a **chronological** train/test evaluation on the same `model_df` / `feature_cols` used by the main OLS and Random Forest fits.  Does **not** replace those fits; it adds an out-of-sample comparison layer.

**`data` dict keys (required):**

| Key | Type | Description |
|-----|------|-------------|
| `model_df` | `pd.DataFrame` | Must include `date`, `count`, and all `feature_cols`. |
| `feature_cols` | `list[str]` | Feature names (same list as `build_features`). |
| `route_col` | `str` | Google intent column name (used to tag intent-related features for ablation). |

**Optional:**

| Key | Description |
|-----|-------------|
| `cfg` | Full settings dict.  Reads `benchmark` and `model.random_forest` sections. |

**`BenchmarkResult` fields:**

| Field | Description |
|-------|-------------|
| `summary_table` | One row per approach: `naive_lag1`, `rolling_mean_7`, `ols`, `random_forest` with columns `model`, `MAE`, `RMSE`, `R2` (test set). |
| `ablation_table` | Random Forest on the **same split**: row `full` plus rows `no_weather`, `no_google_intent`, `no_calendar` when enabled in config.  Includes `delta_MAE_vs_full` and `delta_R2_vs_full`. |
| `predictions_table` | Test rows with `date`, `y_true`, and optional `pred_*` columns for each benchmarked predictor. |

**Assumptions:**

- Rows are ordered in time; the split is **not shuffled** (early segment = train, tail = test).
- Baselines and models use the **same** test mask for fair comparison.
- Feature families for ablation are defined in code: **weather** (`precip`, `temp`, `sun`, `wind`, `precip_lag1`, `weather_severity`, `weekend_x_severity`), **google_intent** (`route_col`, its `lag`/`roll` columns, `weekend_x_intent`), **calendar** (`is_weekend_or_holiday`, `month`, `dow_mean_count`).
- If `benchmark.enabled` is `false`, returns empty tables without error.

**YAML (`config/settings.yaml` → `benchmark`):**

- `enabled`, `train_pct`, `baselines.naive_lag1`, `baselines.rolling_mean_7`, `ablation.weather`, `ablation.google_intent`, `ablation.calendar`.

## Visualizer (`src.visualizer`)

### `plot_benchmark_comparison(summary_table, out_path, reporter, *, dpi=150, lang="en", ja_copy=False)`

Saves a three-panel bar chart (MAE, RMSE, R²) for each row in `summary_table`.

| Parameter | Description |
|-----------|-------------|
| `lang` | `"en"` or `"ja"`. Titles, axis labels, legend text, and display names for known models use the corresponding strings. |
| `ja_copy` | Passed to `Reporter.save_fig`. Use **`False`** when saving English and Japanese figures as **separate explicit paths** (as `run_analysis` does) so the reporter does not append a duplicate `_ja` file. |

Japanese figures (`lang="ja"`) apply a **Japanese-capable font** to all text on the figure when one is found among installed families (see `_resolve_japanese_benchmark_fontproperties` in source). Layout uses Matplotlib **constrained layout** so the suptitle is not clipped.

### `plot_ablation_impact(ablation_table, out_path, reporter, *, dpi=150, lang="en", ja_copy=False)`

Saves **stacked** bar panels for `delta_MAE_vs_full` and `delta_R2_vs_full` on ablated scenarios (excludes the `full` row). Same `lang` / `ja_copy` semantics as `plot_benchmark_comparison`.

### `Reporter.save_fig(..., ja_copy=None)`

| `ja_copy` | Behaviour |
|-----------|-----------|
| `None` (default) | If `visualization.ja_copy` in config is `true`, saves an additional copy with a `_ja` suffix (same pixels as the primary file). |
| `False` | Never creates the `_ja` copy, **even when** `visualization.ja_copy` is `true`. |
| `True` | Always creates the `_ja` copy. |

The pipeline saves benchmark and ablation figures twice (English path + Japanese path) with `ja_copy=False`; other figures may still use `_save_with_ja` or explicit paths. When **`visualization.ja_copy` is `false`**, `run_analysis` **does not** write `fig_benchmark_comparison_ja.png` or `fig_ablation_impact_ja.png` (see comment in `run_analysis.py`).

## Models (`src.models`)

See module docstring for `fit_ols`, `fit_random_forest`, `robustness_suite`, and `statistical_rigor` — used in-sample on the full `model_df` after `dropna`.
