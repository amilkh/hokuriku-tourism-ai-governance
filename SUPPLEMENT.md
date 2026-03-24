# Supplementary Methodological Material

**Paper:** Engineering Distributed Governance for Regional Prosperity: A Socio-Technical
Framework for Mitigating Under-Vibrancy via Human Data Engines

**Journal:** Sustainable Cities and Society (Elsevier)

*For setup, installation, and full reproduction steps, see `README.md § 9`.*

---

## 1. Final OLS Model Parameters — Node A (Tojinbo, Primary Coastal Node)

Dependent variable: daily physical visitor arrivals (camera count).
N = 397 days (2024-12-20 → 2026-03-10, after sensor-outage exclusion).
Standard errors corrected via Newey-West HAC (lags = 8).

| Variable | Coefficient | *p*-value | Std. *β* | Rank |
|---|---|---|---|---|
| const | −722.33 | 0.4482 | — | — |
| directions | +0.9174 | < 0.001 *** | +0.456 | 2 |
| directions\_lag1 | +0.3048 | 0.0353 * | +0.151 | 7 |
| directions\_lag2 | −0.0230 | 0.8718 | −0.011 | 14 |
| directions\_lag3 | −0.2662 | 0.0482 * | −0.133 | 8 |
| directions\_roll7 | +0.3872 | 0.0435 * | +0.158 | 6 |
| precip | −24.34 | 0.1709 | −0.051 | 11 |
| temp | −29.33 | 0.0707 | −0.060 | 10 |
| sun | +1,294.20 | 0.0057 ** | +0.086 | 9 |
| wind | −12.65 | 0.8689 | −0.004 | 16 |
| precip\_lag1 | −4.16 | 0.7126 | −0.009 | 15 |
| is\_weekend\_or\_holiday | +5,374.23 | < 0.001 *** | +0.547 | 1 |
| weather\_severity | −283.40 | 0.2644 | −0.050 | 12 |
| dow\_mean\_count | −0.0585 | 0.5923 | −0.029 | 13 |
| weekend\_x\_severity | −1,194.52 | < 0.001 *** | −0.159 | 5 |
| weekend\_x\_intent | −0.2271 | 0.0367 * | −0.166 | 4 |
| month | +409.47 | < 0.001 *** | +0.331 | 3 |
| **R²** | **0.8096** | | | |
| **Adj. R²** | **0.8016** | | | |
| **Cohen's *f*²** | **4.2519** | large (≥ 0.35) | | |

\* *p* < 0.05, \*\* *p* < 0.01, \*\*\* *p* < 0.001.

---

## 2. Random Forest Regressor — Hyperparameters

| Parameter | Value |
|---|---|
| n\_estimators | 500 |
| max\_depth | 10 |
| min\_samples\_leaf | 5 |
| random\_state | 42 |
| n\_jobs | −1 (all cores) |
| cross-validation | 5-fold (sklearn `cross_val_score`, `scoring="r2"`) |
| feature importance | Permutation Importance (avoids bias toward continuous variables) |
| train / hold-out split | Chronological: 317 training days / 80 hold-out days |

---

## 3. Out-of-Sample Validation Summary

| Metric | Value |
|---|---|
| Hold-out R² | 0.6834 |
| Hold-out MAE | 1,793.2 visitors/day |
| Hold-out RMSE | 2,461.0 visitors/day |
| First-Difference R² | 0.7083 (Durbin-Watson = 2.525) |
| LDV R² | 0.8485 (Durbin-Watson = 1.899) |

---

## 4. Key Dataset Schema (Dataset 2 — merged\_survey\_\*.csv)

97,719 standardized responses, Hokuriku three-prefecture merged survey, April 2023 – March 2026.

| Column (Japanese) | Pipeline Variable | Type | Description |
|---|---|---|---|
| 対象県 | prefecture | String | Survey site prefecture (富山/石川/福井) |
| アンケート回答日 | survey\_date | Date | Date of recorded visit |
| 満足度（旅行全体） | satisfaction | Integer 1–5 | Overall trip satisfaction |
| おすすめ度 | nps\_raw | Integer 0–10 | Raw Net Promoter Score |
| 満足度（商品・サービス） | satisfaction\_service | Integer 1–5 | Service satisfaction |
| 満足度理由 | reason | String | Free-text: reason for rating |
| 不便 (partial match) | inconvenience | String | Free-text: reported inconveniences |
| 自由意見 (partial match) | freetext | String | Free-text: general commentary |
| 回答場所 | location | String | Specific tourism site of collection |
| 県内消費額 | spending\_yen | Integer | In-prefecture spending (Dataset 1 only) |

Full satisfaction scale: 1 = とても不満, 2 = 不満, 3 = どちらでもない, 4 = 満足, 5 = とても満足.

---

## 5. Software Stack

| Library | Version | Role |
|---|---|---|
| Python | 3.12.3 | Runtime |
| pandas | 3.0.1 | Data harmonization and cleaning |
| statsmodels | 0.14.6 | OLS regression, ADF tests, Durbin-Watson, Newey-West HAC |
| scikit-learn | 1.8.0 | Random Forest Regressor, Permutation Importance, cross-validation |
| numpy | 2.4.2 | Numerical computation |
| jpholiday | 1.0.3 | Japanese national holiday calendar logic |
| matplotlib | (see requirements.txt) | Figure generation |
| scipy | (see requirements.txt) | Spearman correlation, chi-square tests |

Full dependency list with lower-bound version pins: `requirements.txt`.

---

## 6. External Data Repositories (Pinned Commits)

The DHDE pipeline reads from four sibling repositories. Each was frozen at the commit SHA
below at the time of submission; clone at these SHAs to exactly reproduce the analysis dataset.

| Repository | Purpose | Commit SHA |
|---|---|---|
| `fukui-kanko-people-flow-data` | Edge-AI camera 5-min CSV files (Nodes A, B, D) | `ca79a526ed50` |
| `fukui-kanko-trend-report` | Google Business Profile directions data | `8bbab30` |
| `opendata` | Hokuriku 3-prefecture merged survey (Dataset 2) | `c782c51` |
| `fukui-kanko-survey` | Fukui raw survey (Dataset 1, spending analysis) | `30f8aa1c` |

JMA meteorological data is bundled directly in `jma/` (cleaned merged CSVs):
`jma/jma_{mikuni,fukuicity,katsuyama,mihama}_hourly_8.csv`.
