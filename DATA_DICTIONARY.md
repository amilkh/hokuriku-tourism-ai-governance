# Data Dictionary

Mapping of Japanese CSV column headers to English pipeline variables.

## AI Camera People-Flow CSVs

| CSV Column | Pipeline Name | Type | Description |
|------------|--------------|------|-------------|
| `aggregate from` | – | str | 5-minute aggregation timestamp |
| `total count` | `count` | int | Total person count per interval |

Daily aggregation: sum of `total count` per file → `count` per date.

## JMA Weather (Hourly)

| CSV Column | Pipeline Name | Type | Unit | Description |
|------------|--------------|------|------|-------------|
| `timestamp` | `date` | datetime | – | Observation time (normalised to date) |
| `temp_c` | `temp` | float | °C | Air temperature |
| `precip_1h_mm` | `precip` | float | mm | Hourly precipitation |
| `sun_1h_h` | `sun` | float | hours | Sunshine duration |
| `wind_speed_ms` | `wind` | float | m/s | Wind speed |
| `snow_depth_cm` | `snow_depth` | float | cm | Snow depth *(recorded only at Node B main observatory; unavailable at AMeDAS stations)* |
| `humidity_pct` | `humidity` | float | % | Relative humidity |

Daily aggregation: `mean` for temp/wind/sun/humidity/snow_depth, `sum` for precip.

## Route Search Impressions (Daily)

| CSV Column | Pipeline Name | Type | Description |
|------------|--------------|------|-------------|
| `date` | `date` | date | Calendar date |
| `東尋坊` / route CSVs | `route_col` | int | Daily direction request count (auto-detected) |

## Merged Tourism Survey (`merged_survey_*.csv`)

| CSV Column (Japanese) | Pipeline Name | Type | Description |
|----------------------|--------------|------|-------------|
| Column 0 (`対象県`) | `prefecture` | str | Target prefecture (富山/石川/福井) |
| Column 1 (`アンケート回答日`) | `date` | date | Survey response date |
| `満足度（旅行全体）` | `satisfaction` | int 1-5 | Overall trip satisfaction |
| `満足度（商品・サービス）` | `satisfaction_service` | int 1-5 | Service satisfaction |
| `おすすめ度` | `nps_raw` | int 0-10 | Net Promoter Score (raw) |
| `満足度理由` | `reason` | str | Satisfaction reason (free text) |
| `不便だったこと` | `inconvenience` | str | Inconvenience experienced |
| `自由意見` | `freetext` | str | Free-form comments |

## Raw Fukui Survey (`all.csv`)

| CSV Column (Japanese) | Pipeline Name | Type | Description |
|----------------------|--------------|------|-------------|
| `回答日時` | `date` | datetime | Response timestamp |
| `県内消費額` | `spending_midpoint` | int (yen) | Mapped via `spending_map` in config |
| `回答エリア` | – | str | Response area |
| `回答エリア2` | – | str | Response sub-area |
| `市町村` | – | str | Municipality |

### Spending Map (`config/settings.yaml`)

| Category (Japanese) | Midpoint (¥) |
|---------------------|-------------|
| `1,000円未満` | 500 |
| `1,000円以上 3,000円未満` | 2,000 |
| `3,000円以上 5,000円未満` | 4,000 |
| `5,000円以上 10,000円未満` | 7,500 |
| `10,000円以上 20,000円未満` | 15,000 |
| `20,000円以上 30,000円未満` | 25,000 |
| `30,000円以上 40,000円未満` | 35,000 |
| `40,000円以上 50,000円未満` | 45,000 |
| `50,000円以上 100,000円未満` | 75,000 |
| `100,000円以上` | 150,000 |
| `使わない` | 0 |

## Engineered Features

| Feature | Source | Description |
|---------|--------|-------------|
| `dow` | calendar | Day of week (0=Mon, 6=Sun) |
| `month` | calendar | Month (1-12) |
| `is_weekend_or_holiday` | calendar + jpholiday | 1 if weekend or Japanese holiday |
| `weather_severity` | JMA | 0=clear, 1=light precip, 2=heavy precip, 3=heavy+wind |
| `roll7` | RSI | 7-day rolling mean of intent |
| `lag1`..`lag7` | RSI | Lagged intent (1-7 days) |
| `precip_lag1` | JMA | Previous day precipitation |
| `dow_mean_count` | camera | Mean visitor count for that DOW |
| `weekend_x_severity` | interaction | is_weekend × weather_severity |
| `weekend_x_intent` | interaction | is_weekend × lag1 |

## Kansei Engineering Indices

| Index | Formula | Description |
|-------|---------|-------------|
| Discomfort Index (DI) | `0.81×T + 0.01×H×(0.99×T − 14.3) + 46.3` | Thermal comfort (higher = more discomfort) |
| Wind Chill (WC) | `13.12 + 0.6215×T − 11.37×V^0.16 + 0.3965×T×V^0.16` | Perceived temperature (V in km/h) |

## Output Files

| File | Description |
|------|-------------|
| `output/analysis_metrics.txt` | Machine-readable key metrics (all headline numbers) |
| `output/EXECUTIVE_REPORT.md` | English executive summary report |
| `output/EXECUTIVE_REPORT.ja.md` | Japanese executive summary report |
| `output/fig01_timeseries.png` | Visitor time-series with weather overlay (EN + `_ja` variant) |
| `output/fig02_correlation.png` | Feature correlation heatmap (EN + `_ja`) |
| `output/fig03_feature_importance.png` | RF feature importance (EN + `_ja`) |
| `output/fig04_dow_boxplot.png` | Day-of-week visitor distribution (EN + `_ja`) |
| `output/fig05_opportunity_gap.png` | Opportunity gap visualisation (EN + `_ja`) |
| `output/fig06_lag_correlations.png` | RSI intent lag correlations (EN + `_ja`) |
| `output/fig07_lost_population.png` | Lost visitors bar chart (EN + `_ja`) |
| `output/fig08_spatial_friction.png` | Multi-node spatial friction heatmap (EN + `_ja`) |
| `output/fig09_rank_projection.png` | Prefecture ranking recovery chart (EN + `_ja`) |
| `output/paper_fig1_dhde_architecture.png` | Paper Fig 1: DHDE architecture diagram (EN + `_ja`) |
| `output/paper_fig2_rf_prediction.png` | Paper Fig 2: RF vs actual prediction plot (EN + `_ja`) |
| `output/paper_fig3_ranking_recovery.png` | Paper Fig 3: Rank improvement projection (EN + `_ja`) |
| `output/paper_fig5_ishikawa_ccf.png` | Paper Fig 5: Ishikawa → Tojinbo cross-correlation (EN + `_ja`) |
| `output/paper_fig6_weather_shield.png` | Paper Fig 6: Weather Shield Network diagram (EN + `_ja`) |
| `output/table1_ols.png` / `.tex` | OLS regression results table |
| `output/table2_statistical_rigor.png` / `.tex` | Statistical rigor & effect size table |
| `output/table3_key_metrics.png` / `.tex` | Key headline metrics table |
| `output/paper_appendices.md` | Full appendices (A–D) for academic paper |
| `output/pdf/executive_report_pdf_en.md` | Pandoc-ready Markdown source for English PDF |
| `output/pdf/executive_report_pdf_ja.md` | Pandoc-ready Markdown source for Japanese PDF |
| `output/pdf/executive_report_pdf_en.pdf` | Compiled English executive PDF |
| `output/pdf/executive_report_pdf_ja.pdf` | Compiled Japanese executive PDF |
