# API Reference

## `src.config`

### `load_config(config_path=None) → dict`
Load and resolve the pipeline configuration from `settings.yaml`.

### `resolve_repo_path(cfg, *parts) → Path`
Join path segments relative to the repository root.

### `resolve_ws_path(cfg, *parts) → Path`
Join path segments relative to the workspace root (parent of the repo).

---

## `src.report`

### `Reporter(cfg)`
Accumulates report and metrics lines. Passed to every module.

| Method | Description |
|--------|-------------|
| `log(msg)` | Print + append to report buffer |
| `metrics(msg)` | Append to machine-readable metrics buffer |
| `section(number, title)` | Print a section header |
| `save_fig(fig, fname, dpi=, ja_copy=)` | Save figure + optional `_ja` copy |
| `save()` | Flush report and metrics to disk |

---

## `src.data_loader`

### `load_camera_daily(glob_pattern) → DataFrame`
Load AI-camera CSVs → daily `[date, count]`. Zero-count days removed.

### `load_weather_daily(primary_path, legacy_path) → DataFrame`
Load JMA hourly weather → daily aggregates `[date, temp, precip, sun, wind, snow_depth, humidity]`.

### `load_rsi_intent(trend_root) → (DataFrame, route_col)`
Load Route Search Impressions (RSI) daily CSVs. Returns the DataFrame and the auto-detected route column name.

### `load_survey_prefectures(glob) → DataFrame`
Load merged survey CSVs with `[prefecture, date]`.

### `load_survey_satisfaction(glob) → DataFrame`
Load survey CSVs with `[prefecture, date, satisfaction, satisfaction_service, nps_raw]`.

### `load_survey_text(glob) → DataFrame`
Load survey CSVs with free-text fields `[prefecture, date, satisfaction, reason, inconvenience, freetext]`.

### `load_raw_fukui_survey(path, spending_map) → DataFrame`
Load raw `all.csv` with `spending_midpoint` column.

### `merge_daily(camera, weather, rsi) → DataFrame`
Merge all sources into a single daily table with outlier flags.

### `run_adf_tests(daily, route_col)`
Run Augmented Dickey-Fuller stationarity tests.

### `load_all_data(cfg, reporter) → dict`
Single entry-point: loads all data and returns dict with keys `daily`, `weather_daily`, `rsi`, `route_col`, `survey_all`, `sat_all`, `text_all`, `raw_survey`.

---

## `src.feature_engineering`

### `build_features(daily, route_col, reporter) → (DataFrame, list[str])`
Full feature engineering pipeline. Returns enhanced DataFrame and list of feature column names.

Individual steps (called by `build_features`):
- `add_calendar_features(df)` – dow, month, is_weekend_or_holiday (jpholiday)
- `add_weather_severity(df)` – 0/1/2/3 severity score
- `add_rolling_features(df, route_col, windows)` – rolling means
- `add_lag_features(df, route_col, max_lag)` – lag1..7
- `add_interaction_features(df)` – weekend×severity, weekend×intent
- `add_dow_mean_encoding(df)` – day-of-week mean count encoding

---

## `src.models`

### `fit_ols(model_df, feature_cols, reporter) → OLSResult`
Fit OLS regression. Returns `OLSResult(model, r2, adj_r2, feature_cols, y_pred)`.

### `fit_random_forest(model_df, feature_cols, reporter, rf_params=, cv_folds=) → RFResult`
Fit RF with cross-validation. Returns `RFResult(model, r2_train, mae_train, cv_r2_mean, cv_r2_std, y_pred, mdi_importance, perm_importance)`.

### `robustness_suite(model_df, ols_result, feature_cols, reporter) → RobustnessResult`
Run Durbin-Watson, Newey-West, first-difference, LDV, VIF, and weather sensitivity.

---

## `src.kansei`

### `compute_discomfort_index(temp, humidity) → Series`
DI = 0.81×T + 0.01×H×(0.99×T − 14.3) + 46.3

### `compute_wind_chill(temp, wind_speed) → Series`
North American wind chill formula (valid for T ≤ 10°C, V > 4.8 km/h).

### `discomfort_index_analysis(weather_daily, sat_all, reporter) → dict`
Correlate daily DI with satisfaction and NPS. Returns peak DI, correlations.

### `overtourism_threshold(daily, sat_all, reporter) → dict`
Spearman correlation between visitor count and satisfaction/NPS.

### `text_mine_undervibrancy(text_all, reporter, cfg=) → dict`
Japanese keyword matching for under-vibrancy complaints.

---

## `src.economics`

### `compute_opportunity_gap(daily, route_col, reporter) → DataFrame`
Flag days with above-median intent and below-median count.

### `compute_lost_population(model_df, ols_y_pred, daily, reporter) → dict`
Estimate lost visitors (OLS predicted − actual on gap days).

### `ranking_simulation(total_lost, gap_model, reporter, ranking_cfg=) → dict`
Simulate national ranking improvement from recovered visitors.

### `seasonal_sensitivity(model_df, feature_cols, reporter) → dict`
Compare summer vs winter weather sensitivity (ΔR²).

### `compute_satake_number(node_metrics, spending, reporter) → dict`
Three-node cumulative opportunity loss.

---

## `src.spatial`

### `cross_prefectural_ccf(daily, survey_all, reporter) → dict`
Cross-correlation function: Ishikawa survey → Tojinbo arrivals.

### `build_node_metrics(name, counts, weather, rsi, route_col, spending, reporter) → dict|None`
Build OLS metrics for one DHDE node.

### `atmospheric_nudge_analysis(valid_nodes, wind_threshold, reporter) → dict`
Measure visitor redistribution when coastal wind exceeds threshold.

### `multi_node_analysis(cfg, rsi, route_col, survey_all, reporter) → dict`
Full three-node spatial governance pipeline.

---

## `src.visualizer`

| Function | Figure | Description |
|----------|--------|-------------|
| `plot_timeseries` | Fig 1 | Dual-axis visitor count vs RSI intent |
| `plot_correlation_heatmap` | Fig 2 | Feature correlation matrix |
| `plot_feature_importance` | Fig 3 | MDI + permutation importance |
| `plot_dow_boxplot` | Fig 4 | Day-of-week visitor distribution |
| `plot_rf_prediction` | Fig 5 | Actual vs RF predicted |
| `plot_opportunity_gap` | Fig 6 | Intent vs count scatter |
| `plot_lag_correlations` | Fig 7 | Lag 0-7 bar chart |
| `plot_ccf` | Fig 8 | Ishikawa CCF bar chart |
| `plot_kansei_scatter` | Fig 9 | Visitors vs satisfaction |
| `plot_lost_population` | Fig 10 | Lost population waterfall |
| `plot_resurrection` | Fig 11 | Fukui Resurrection (EN+JA) |
| `plot_hokuriku_heatmap` | Fig 12 | Hokuriku demand heatmap (EN+JA) |
| `plot_spatial_friction` | Fig 13 | Spatial friction heatmap |

---

## `src.latex_export`

### `ols_summary_to_latex(params, pvalues, r2, adj_r2, n) → str`
OLS results → LaTeX table with significance stars.

### `model_comparison_to_latex(metrics) → str`
Multi-model comparison → LaTeX.

### `key_metrics_to_latex(kv) → str`
Key-value metrics → LaTeX.

### `export_all_tables(results, output_dir) → list[str]`
Generate and save all LaTeX tables.
