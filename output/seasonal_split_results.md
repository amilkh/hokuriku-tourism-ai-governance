# Seasonal split: winter vs summer OLS (thesis Ch 5)

Season definitions: Winter = Dec/Jan/Feb; Summer = Jun/Jul/Aug.
Same OLS feature set as the paper, except `month` (near-constant within
a season) is dropped from the seasonal refits. Standardized betas.
Data: paper-era pin bf2cfc45 required for exact reproduction; effective
total N here = 397.

| Metric | Winter (Dec-Feb) | Summer (Jun-Aug) |
|---|---|---|
| Effective days (N) | 138 | 86 |
| R² | 0.566 | 0.938 |
| Adj. R² | 0.513 | 0.924 |
| β_std directions | +0.535*** (p=0.000) | +0.536*** (p=0.000) |
| β_std precip | -0.061 (p=0.549) | -0.091 (p=0.081) |
| β_std temp | +0.315*** (p=0.000) | -0.076 (p=0.089) |
| β_std wind | -0.082 (p=0.236) | -0.013 (p=0.732) |
| β_std weather_severity | +0.027 (p=0.822) | +0.053 (p=0.417) |
| Weather joint F (precip,temp,wind) | F=6.40, p=0.0005 | F=2.52, p=0.0648 |
| Weather block ΔR² | 0.191 | 0.022 |
| High-friction days (severity ≥ 2) | 44 (32%) | 12 (14%) |

Note: * p<0.05, ** p<0.01, *** p<0.001. Subgroup Ns are small relative
to the 16-predictor specification; estimates are indicative and are
interpreted with hedged language.

Interpretation: Re-estimating the model separately by season suggests that digital search intent (directions) remains a positive predictor in both subsamples (winter β_std = +0.535, p = 0.000; summer β_std = +0.536, p = 0.000). The weather block contributes substantially more explanatory power in winter (ΔR² = 0.191) than in summer (ΔR² = 0.022), and high-friction days (severity ≥ 2) are concentrated in winter (44 days, 32% of the winter subsample, versus 12 in summer). Taken together, these results are consistent with the pooled-model finding that weather acts as a seasonally asymmetric friction: the intent signal holds across seasons, while winter conditions disproportionately suppress the conversion of intent into physical arrivals. The winter-to-summer ratio of the weather block's explanatory contribution is 8.9x here, which is consistent in direction and order of magnitude with the 6.26x seasonal sensitivity ratio reported for the pooled specification. Two further patterns are worth noting for the discussion: temperature carries a positive and significant standardized coefficient in winter (β_std = +0.315, p = 0.000) but not in summer (β_std = -0.076, p = 0.089), suggesting that milder winter days recover part of the suppressed demand; and the summer subsample attains a higher R² (0.938) than the winter subsample (0.566), which is consistent with weather-driven cancellation adding unmodelled variance in winter. Given the smaller seasonal Ns (winter N = 138, summer N = 86) relative to the predictor count, these subgroup estimates should be read as supportive evidence rather than stand-alone results.
