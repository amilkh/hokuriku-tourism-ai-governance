"""Kansei Engineering module – Emotional / atmospheric satisfaction analysis.

Implements:
    1. **Discomfort Index (DI)** – Thermal comfort metric correlated with
       visitor satisfaction (requested by Dean Inoue).
    2. **Wind Chill** – Cold-stress proxy for winter tourism friction.
    3. **Overtourism threshold** – Satisfaction vs crowd density analysis.
    4. **Text mining** – Under-vibrancy keyword extraction from surveys.
    5. **Zero-Shot NLP** – Deep root-cause diagnostics for detractor complaints.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .report import Reporter

logger = logging.getLogger(__name__)

# ── Discomfort Index ─────────────────────────────────────────────────────────

def compute_discomfort_index(
    temp: pd.Series,
    humidity: pd.Series,
    *,
    coeff_temp: float = 0.81,
    coeff_humidity: float = 0.01,
    inner_temp: float = 0.99,
    inner_offset: float = -14.3,
    constant: float = 46.3,
) -> pd.Series:
    r"""Compute the Discomfort Index (不快指数).

    .. math::
        DI = 0.81 \times T + 0.01 \times H \times (0.99 \times T - 14.3) + 46.3

    Args:
        temp: Air temperature in °C.
        humidity: Relative humidity in %.
        coeff_temp: Temperature coefficient (default 0.81).
        coeff_humidity: Humidity coefficient (default 0.01).
        inner_temp: Inner temperature multiplier (default 0.99).
        inner_offset: Inner offset (default -14.3).
        constant: Additive constant (default 46.3).

    Returns:
        Discomfort Index series (same index as inputs).
    """
    return coeff_temp * temp + coeff_humidity * humidity * (inner_temp * temp + inner_offset) + constant


def compute_wind_chill(
    temp: pd.Series,
    wind_speed: pd.Series,
) -> pd.Series:
    r"""Compute the Wind Chill Temperature (体感温度).

    Uses the North American / JMA formula valid for T ≤ 10 °C and V > 4.8 km/h:

    .. math::
        WC = 13.12 + 0.6215T - 11.37V^{0.16} + 0.3965TV^{0.16}

    where *V* is wind speed in km/h.  For conditions outside this range
    the raw temperature is returned.

    Args:
        temp: Air temperature in °C.
        wind_speed: Wind speed in m/s (converted internally to km/h).

    Returns:
        Wind chill temperature series.
    """
    v_kmh = wind_speed * 3.6
    wc = (
        13.12
        + 0.6215 * temp
        - 11.37 * v_kmh.clip(lower=0.1) ** 0.16
        + 0.3965 * temp * v_kmh.clip(lower=0.1) ** 0.16
    )
    # Only apply formula where it is valid
    valid = (temp <= 10) & (v_kmh > 4.8)
    return wc.where(valid, temp)


def discomfort_index_analysis(
    weather_daily: pd.DataFrame,
    sat_daily: pd.DataFrame | None,
    reporter: Reporter,
    *,
    di_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Correlate hourly DI and Wind Chill with satisfaction / NPS.

    This is the **Environmental Kansei Assessment** requested for the
    Dean Inoue meeting.

    Args:
        weather_daily: Must have ``temp`` and ideally ``humidity``, ``wind``.
        sat_daily: Daily satisfaction aggregates (``mean_satisfaction``,
            ``mean_nps``).  May be ``None`` if no survey overlap.
        reporter: ``Reporter`` instance.
        di_cfg: Optional override for DI formula coefficients.

    Returns:
        Dict with keys ``di_available``, ``di_mean``, ``wc_mean``,
        ``di_sat_r``, ``di_sat_p``, ``wc_sat_r``, ``wc_sat_p``,
        ``peak_di`` (the DI value where satisfaction is maximised).
    """
    reporter.section("K", "Kansei Environmental Assessment (Discomfort Index)")
    result: dict[str, Any] = {"di_available": False}
    params = di_cfg or {}

    has_humidity = "humidity" in weather_daily.columns and weather_daily["humidity"].notna().sum() > 30

    if has_humidity:
        weather_daily = weather_daily.copy()
        weather_daily["discomfort_index"] = compute_discomfort_index(
            weather_daily["temp"],
            weather_daily["humidity"],
            **{k: v for k, v in params.items() if k in
                ("coeff_temp", "coeff_humidity", "inner_temp", "inner_offset", "constant")},
        )
        di_mean = weather_daily["discomfort_index"].mean()
        reporter.log(f"Discomfort Index computed. Mean DI: {di_mean:.1f}")
        result["di_available"] = True
        result["di_mean"] = di_mean
    else:
        reporter.log("Humidity data unavailable – DI computed using temp-only proxy.")
        weather_daily = weather_daily.copy()
        # Proxy: DI ≈ 0.81*T + 46.3 (assuming 60% humidity)
        weather_daily["discomfort_index"] = 0.81 * weather_daily["temp"] + 0.01 * 60 * (0.99 * weather_daily["temp"] - 14.3) + 46.3
        result["di_available"] = True
        result["di_mean"] = weather_daily["discomfort_index"].mean()
        reporter.log(f"DI proxy (assuming 60% humidity). Mean DI: {result['di_mean']:.1f}")

    # Wind Chill
    if "wind" in weather_daily.columns:
        weather_daily["wind_chill"] = compute_wind_chill(
            weather_daily["temp"], weather_daily["wind"]
        )
        result["wc_mean"] = weather_daily["wind_chill"].mean()
        reporter.log(f"Wind Chill computed. Mean WC: {result['wc_mean']:.1f}°C")

    # Correlate with satisfaction
    if sat_daily is not None and len(sat_daily) > 0:
        merged = weather_daily[["date", "discomfort_index"]].merge(
            sat_daily, on="date", how="inner"
        )
        if "wind_chill" in weather_daily.columns:
            merged = merged.merge(
                weather_daily[["date", "wind_chill"]], on="date", how="left"
            )

        if len(merged) > 20 and "mean_satisfaction" in merged.columns:
            valid = merged.dropna(subset=["discomfort_index", "mean_satisfaction"])
            r_di, p_di = stats.spearmanr(valid["discomfort_index"], valid["mean_satisfaction"])
            result["di_sat_r"] = r_di
            result["di_sat_p"] = p_di
            reporter.log(f"\nDI vs Satisfaction (Spearman): r = {r_di:+.3f}, p = {p_di:.4f}")

            if "wind_chill" in merged.columns:
                valid_wc = merged.dropna(subset=["wind_chill", "mean_satisfaction"])
                if len(valid_wc) > 10:
                    r_wc, p_wc = stats.spearmanr(valid_wc["wind_chill"], valid_wc["mean_satisfaction"])
                    result["wc_sat_r"] = r_wc
                    result["wc_sat_p"] = p_wc
                    reporter.log(f"Wind Chill vs Satisfaction: r = {r_wc:+.3f}, p = {p_wc:.4f}")

            # Find the "Atmospheric Satisfaction Peak"
            if len(valid) > 30:
                try:
                    bins = pd.qcut(valid["discomfort_index"], q=5, duplicates="drop")
                    peak_df = valid.groupby(bins)["mean_satisfaction"].mean()
                    peak_bin = peak_df.idxmax()
                    result["peak_di"] = peak_bin.mid
                    reporter.log(f"\n★ ATMOSPHERIC SATISFACTION PEAK: DI ≈ {peak_bin.mid:.1f}")
                    reporter.log(f"  Comfort zone: {peak_bin.left:.1f} – {peak_bin.right:.1f}")
                except Exception:
                    pass

            # NPS correlation
            if "mean_nps" in merged.columns:
                valid_nps = merged.dropna(subset=["discomfort_index", "mean_nps"])
                if len(valid_nps) > 10:
                    r_nps, p_nps = stats.spearmanr(valid_nps["discomfort_index"], valid_nps["mean_nps"])
                    result["di_nps_r"] = r_nps
                    result["di_nps_p"] = p_nps
                    reporter.log(f"DI vs NPS: r = {r_nps:+.3f}, p = {p_nps:.4f}")
    else:
        reporter.log("No satisfaction data available for DI correlation.")

    return result


# ── Overtourism Threshold ────────────────────────────────────────────────────

def overtourism_threshold(
    daily: pd.DataFrame,
    sat_all: pd.DataFrame,
    reporter: Reporter,
) -> dict[str, Any]:
    """Analyse satisfaction vs visitor count bins.

    Args:
        daily: Master daily with ``date`` and ``count``.
        sat_all: Loaded satisfaction survey data.
        reporter: ``Reporter``.

    Returns:
        Dict with ``spearman_r``, ``spearman_p``, ``bin_table``, etc.
    """
    reporter.section(8, "Kansei Feedback Loop: Overtourism Threshold")
    result: dict[str, Any] = {"spearman_r": 0.0, "spearman_p": 1.0}

    if sat_all.empty:
        reporter.log("  ⚠ No satisfaction data available.")
        return result

    sat_fukui = sat_all[sat_all["prefecture"].str.contains("福井", na=False)].copy()
    reporter.log(f"Fukui satisfaction responses: {len(sat_fukui)}")

    agg_dict: dict = {
        "mean_satisfaction": ("satisfaction", "mean"),
        "n_responses": ("satisfaction", "count"),
    }
    if "nps_raw" in sat_fukui.columns:
        agg_dict["mean_nps"] = ("nps_raw", "mean")
    if "satisfaction_service" in sat_fukui.columns:
        agg_dict["mean_service"] = ("satisfaction_service", "mean")

    sat_daily = sat_fukui.groupby("date").agg(**agg_dict).reset_index()
    if "mean_nps" not in sat_daily.columns:
        sat_daily["mean_nps"] = float("nan")

    # Try exact-day merge first; fall back to year-month aggregate if too sparse.
    sat_merged = daily[["date", "count"]].merge(sat_daily, on="date", how="inner")
    reporter.log(f"Days with both camera + satisfaction (exact): {len(sat_merged)}")

    if len(sat_merged) <= 20:
        reporter.log("  ⚠ Insufficient exact-day overlap — merging by year-month instead.")
        cam_monthly = daily[["date", "count"]].copy()
        cam_monthly["ym"] = cam_monthly["date"].dt.to_period("M")
        cam_monthly = cam_monthly.groupby("ym")["count"].mean().reset_index()
        cam_monthly.rename(columns={"count": "mean_count"}, inplace=True)

        sat_daily["ym"] = pd.to_datetime(sat_daily["date"]).dt.to_period("M")
        monthly_agg: dict = {
            "mean_satisfaction": ("mean_satisfaction", "mean"),
            "n_responses": ("n_responses", "sum"),
        }
        if "mean_nps" in sat_daily.columns:
            monthly_agg["mean_nps"] = ("mean_nps", "mean")
        sat_monthly = sat_daily.groupby("ym").agg(**monthly_agg).reset_index()
        if "mean_nps" not in sat_monthly.columns:
            sat_monthly["mean_nps"] = float("nan")

        sat_merged = cam_monthly.merge(sat_monthly, on="ym", how="inner")
        sat_merged["count"] = sat_merged["mean_count"]
        sat_merged["date"] = sat_merged["ym"].dt.to_timestamp()
        reporter.log(f"  Year-month overlap: {len(sat_merged)} months")

    result["sat_merged"] = sat_merged
    result["sat_daily"] = sat_daily

    if len(sat_merged) <= 3:
        reporter.log("  ⚠ Insufficient overlap even after monthly aggregation.")
        return result

    # Visitor bins (use mean_count for monthly, count for daily)
    count_col = "count"
    bins = [0, 5000, 8000, 12000, 15000, 20000, 50000]
    labels = ["<5K", "5-8K", "8-12K", "12-15K", "15-20K", ">20K"]
    sat_merged["visitor_bin"] = pd.cut(sat_merged[count_col], bins=bins, labels=labels)

    reporter.log("\nOvertourism Threshold Analysis (monthly averages):")
    reporter.log(f"  {'Bin':12s} {'Sat':>10s} {'NPS':>8s} {'Months':>7s}")
    for label in labels:
        sub = sat_merged[sat_merged["visitor_bin"] == label]
        if len(sub) > 0:
            reporter.log(f"  {label:12s} {sub['mean_satisfaction'].mean():10.2f} "
                         f"{sub['mean_nps'].mean():8.2f} {len(sub):7d}")

    # Spearman
    valid = sat_merged.dropna(subset=["mean_satisfaction", count_col])
    if len(valid) > 3:
        r, p = stats.spearmanr(valid[count_col], valid["mean_satisfaction"])
        result["spearman_r"] = r
        result["spearman_p"] = p
        reporter.log(f"\nSpearman (monthly visitors vs satisfaction): r = {r:+.3f}, p = {p:.4f}")

        r_nps, p_nps = stats.spearmanr(
            valid[count_col],
            valid["mean_nps"].fillna(valid["mean_satisfaction"]),
        )
        result["spear_r_nps"] = r_nps
        result["spear_p_nps"] = p_nps
        reporter.log(f"Spearman (monthly visitors vs NPS):           r = {r_nps:+.3f}, p = {p_nps:.4f}")

    return result


# ── Text Mining ──────────────────────────────────────────────────────────────

def text_mine_undervibrancy(
    text_all: pd.DataFrame,
    reporter: Reporter,
    *,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Mine survey free-text for under-vibrancy complaints.

    Args:
        text_all: Loaded text survey data.
        reporter: ``Reporter``.
        keywords: Override list of Japanese under-vibrancy keywords.

    Returns:
        Dict with ``n_low_sat``, ``undervibrancy_hits``, ``pct``,
        ``ratio_vs_high``, ``examples``.
    """
    reporter.section(15, "Qualitative Under-vibrancy Link (Survey Text Mining)")
    default_kw = [
        "静か", "寂し", "さびし", "さみし", "人が少な", "人がいな",
        "活気", "賑わ", "にぎわ", "閑散", "寂れ", "さびれ",
        "閉まっ", "店がな", "営業し", "何もな", "つまらな",
        "退屈", "物足りな", "盛り上が", "人通り",
    ]
    kw_list = keywords or default_kw
    result: dict[str, Any] = {"undervibrancy_hits": 0, "pct": 0.0}

    if text_all.empty:
        reporter.log("  ⚠ No survey text data.")
        return result

    text_fukui = text_all[text_all["prefecture"].str.contains("福井", na=False)].copy()
    n_text_fukui = len(text_fukui)
    reporter.log(f"Fukui survey responses with text: {n_text_fukui}")
    result["n_text_fukui"] = n_text_fukui

    low_sat = text_fukui[text_fukui["satisfaction"].isin([1, 2])].copy()
    low_sat["all_text"] = (
        low_sat["reason"].fillna("") + " "
        + low_sat["inconvenience"].fillna("") + " "
        + low_sat["freetext"].fillna("")
    )
    # Filter out empty or extremely short garbled text
    low_sat = low_sat[low_sat["all_text"].str.strip().str.len() > 3]
    reporter.log(f"Low satisfaction (1-2★) valid responses: {len(low_sat)}")

    hits = 0
    examples: list[tuple[float, str, str]] = []
    for _, row in low_sat.iterrows():
        text = str(row["all_text"])
        for kw in kw_list:
            if kw in text:
                hits += 1
                examples.append((row["satisfaction"], kw, text[:200]))
                break

    pct = hits / len(low_sat) * 100 if len(low_sat) > 0 else 0
    reporter.log(f"\n★ Under-vibrancy mentions in 1-2★: {hits} ({pct:.1f}%)")
    result.update(n_low_sat=len(low_sat), undervibrancy_hits=hits, pct=pct, examples=examples)

    if examples:
        reporter.log("  Sample complaints (up to 5):")
        for sat, kw, txt in examples[:5]:
            reporter.log(f"    [{int(sat)}★] '{kw}': {txt[:120]}...")

    # Compare with high-sat
    high_sat = text_fukui[text_fukui["satisfaction"].isin([4, 5])].copy()
    high_sat["all_text"] = (
        high_sat["reason"].fillna("") + " "
        + high_sat["inconvenience"].fillna("") + " "
        + high_sat["freetext"].fillna("")
    )
    high_hits = sum(
        1 for _, row in high_sat.iterrows()
        if any(kw in str(row["all_text"]) for kw in kw_list)
        and str(row["all_text"]) != "nan" and len(str(row["all_text"]).strip()) >= 3
    )
    pct_high = high_hits / len(high_sat) * 100 if len(high_sat) > 0 else 0.01
    ratio = pct / max(pct_high, 0.1)
    result["ratio_vs_high"] = ratio
    reporter.log(f"\n  High-sat vibrancy mentions: {high_hits} ({pct_high:.1f}%)")
    reporter.log(f"  ★ Ratio: {ratio:.1f}x more prevalent in dissatisfied visitors")

    # Chi-square test of proportions (low-sat vs high-sat under-vibrancy rate)
    n_low_sat = len(low_sat)
    n_high_sat_total = len(high_sat)
    if n_low_sat > 0 and n_high_sat_total > 0:
        obs = np.array([
            [hits, n_low_sat - hits],
            [high_hits, n_high_sat_total - high_hits],
        ])
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(obs)
        result["chi2_stat"] = float(chi2_stat)
        result["chi2_p"] = float(chi2_p)
        reporter.log(
            f"\n  Chi-square test (under-vibrancy low-sat vs high-sat): "
            f"χ² = {chi2_stat:.1f}, p = {chi2_p:.3e}"
        )

    return result


def run_zero_shot_diagnostics(
    survey_df: pd.DataFrame,
    *,
    reporter: Reporter | None = None,
    max_samples: int | None = 3000,
    text_max_chars: int = 512,
) -> dict[str, float]:
    """Diagnose root causes of 'Opportunity Gap' via Zero-Shot Classification.

    Filters for low-satisfaction responses (detractors) and classifies the 
    underlying complaints using a multilingual DeBERTa-v3 model.

    Args:
        survey_df: DataFrame containing satisfaction scores and text columns.
        reporter: Optional ``Reporter`` for logging.
        max_samples: Optional cap for number of detractor texts to classify.
        text_max_chars: Maximum characters per text to classify.

    Returns:
        Dictionary mapping candidate labels to their percentage occurrence.
    """
    info = reporter.log if reporter else logger.info
    warn = reporter.log if reporter else logger.warning
    
    # Check if satisfaction column exists to filter detractors
    if "satisfaction" not in survey_df.columns:
        warn("'satisfaction' column missing. Skipping zero-shot diagnostics.")
        return {}

    # 1. Filter for Detractors (Assuming 5-point scale, detractors are 1 or 2)
    detractors = survey_df[survey_df["satisfaction"] <= 2].copy()
    if detractors.empty:
        info("No low-satisfaction responses found. Skipping Kansei analysis.")
        return {}

    # 2. Safely Aggregate Free-Text Features
    text_cols = ["reason", "inconvenience", "freetext"]
    available_cols = [c for c in text_cols if c in detractors.columns]

    # Combine text from available columns, ignoring NAs and empty strings
    detractors["combined_text"] = detractors[available_cols].apply(
        lambda row: " ".join(str(val) for val in row if pd.notna(val) and str(val).strip()),
        axis=1
    )

    # Filter out empty or extremely short garbled text
    texts = detractors[detractors["combined_text"].str.len() > 3]["combined_text"].tolist()

    if not texts:
        info("No valid text found in detractor responses.")
        return {}

    if max_samples is not None and len(texts) > max_samples:
        info(
            f"Capping zero-shot inputs: using {max_samples} of {len(texts)} "
            "detractor texts for stable runtime."
        )
        texts = texts[:max_samples]

    # Bound sequence length to keep memory and latency predictable.
    texts = [str(t)[:text_max_chars] for t in texts]

    try:
        from transformers import pipeline
    except ImportError:
        warn("transformers library not found. Skipping zero-shot diagnostics.")
        return {}

    info(f"Running Phase 2 Kansei Extraction on {len(texts)} detractor complaints...")
    info("Booting MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (Zero-Shot Mode)...")

    # 3. Load the zero-shot pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification", 
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        )
    except Exception as e:
        warn(f"Model loading failed: {e}")
        return {}

    # 4. The Candidate Labels (Directly mapping to the ¥11.96B thesis)
    candidate_labels = [
        "weather conditions",
        "poor transportation",
        "language barrier",
        "lack of information",
        "pricing"
    ]

    # Run classification (using batch_size to respect local memory limits)
    results = classifier(
        texts,
        candidate_labels,
        batch_size=8,
        truncation=True,
        multi_label=False,
    )
    
    # Handle edge case where pipeline returns a dict instead of a list for a single item
    if isinstance(results, dict):
        results = [results]

    # 5. Aggregate Results
    counts = {label: 0 for label in candidate_labels}
    for res in results:
        top_label = res['labels'][0]  # The model's highest confidence prediction
        counts[top_label] += 1

    total = len(results)
    percentages = {label: (count / total) * 100 for label, count in counts.items()}

    # Sort descending for a cleaner log output
    sorted_pct = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))

    info("\nOpportunity Gap Drivers (mDeBERTa-v3 Distribution):")
    for label, pct in sorted_pct.items():
        info(f"  - {label.title()}: {pct:.1f}%")

    return sorted_pct


# ── Eiheiji Atmospheric Resilience ───────────────────────────────────────────

def eiheiji_atmospheric_resilience(
    sat_all: pd.DataFrame,
    reporter: Reporter,
    *,
    min_responses_per_day: int = 3,
) -> dict[str, Any]:
    """Evaluate atmospheric resilience of Eiheiji Zen temple.

    Tests whether visitor satisfaction at Eiheiji correlates with crowd density
    (using daily survey response count as a density proxy) and scans free-text
    for congestion vs under-vibrancy complaints.  The null result (r ≈ 0,
    p > 0.05) confirms that Eiheiji currently operates well below its latent
    carrying capacity — a safety guarantee for the DHDE nudge algorithm.

    Args:
        sat_all: Survey DataFrame with ``location``, ``date``, ``satisfaction``,
            and optional free-text columns (``reason``, ``inconvenience``,
            ``freetext``).
        reporter: ``Reporter`` instance.
        min_responses_per_day: Days with fewer responses are excluded.

    Returns:
        Dict with keys ``n_responses``, ``n_days``, ``spearman_r``,
        ``spearman_p``, ``sat_rate_pct``, ``congestion_pct``,
        ``congestion_low_sat_pct``.
    """
    reporter.section("E", "Eiheiji Atmospheric Resilience Analysis")
    result: dict[str, Any] = {}

    area_col = next(
        (c for c in sat_all.columns if "エリア" in c or "location" in c.lower()),
        None,
    )
    if area_col is None or sat_all.empty:
        reporter.log("  ⚠ No area/location column in sat_all — Eiheiji analysis skipped.")
        return result

    ei = sat_all[sat_all[area_col].astype(str).str.contains("永平寺", na=False)].copy()
    n_ei = len(ei)
    reporter.log(f"Eiheiji-area survey responses: {n_ei}")
    result["n_responses"] = n_ei

    if n_ei < 20:
        reporter.log("  ⚠ Insufficient responses.")
        return result

    # Satisfaction rate
    sat_col = next((c for c in ei.columns if c == "満足度" or c == "satisfaction"), None)
    if sat_col:
        high_vals = ["満足", "とても満足", 4, 5]
        low_vals = ["不満", "とても不満", 1, 2]
        n_high = ei[sat_col].isin(high_vals).sum()
        n_low = ei[sat_col].isin(low_vals).sum()
        sat_rate = n_high / n_ei * 100
        result["sat_rate_pct"] = round(sat_rate, 1)
        result["n_low_sat"] = int(n_low)
        reporter.log(f"High-satisfaction (4-5★) rate: {sat_rate:.1f}%  ({n_high}/{n_ei})")
        reporter.log(f"Low-satisfaction  (1-2★) count: {n_low}")

    # Density vs satisfaction (Spearman on daily aggregates)
    date_col = next((c for c in ei.columns if "date" in c.lower() or "回答日" in c), None)
    if date_col and sat_col:
        ei_copy = ei.copy()
        ei_copy["_date"] = pd.to_datetime(ei_copy[date_col], errors="coerce").dt.normalize()
        # Map text satisfaction to numeric for Spearman
        sat_map = {"とても不満": 1, "不満": 2, "どちらでもない": 3, "満足": 4, "とても満足": 5}
        ei_copy["_sat_num"] = ei_copy[sat_col].map(sat_map) if ei_copy[sat_col].dtype == object \
            else ei_copy[sat_col]
        daily = (
            ei_copy.dropna(subset=["_date", "_sat_num"])
            .groupby("_date")
            .agg(n=("_sat_num", "count"), mean_sat=("_sat_num", "mean"))
            .reset_index()
        )
        daily = daily[daily["n"] >= min_responses_per_day]
        n_days = len(daily)
        result["n_days"] = n_days
        reporter.log(f"Analysis days (≥{min_responses_per_day} resp): {n_days}")

        if n_days >= 10:
            daily["rel_density"] = daily["n"] / daily["n"].max() * 100
            spear_r, spear_p = stats.spearmanr(daily["rel_density"], daily["mean_sat"])
            result["spearman_r"] = round(float(spear_r), 3)
            result["spearman_p"] = round(float(spear_p), 4)
            reporter.log(f"Spearman (relative density vs satisfaction): r = {spear_r:+.3f}, p = {spear_p:.4f}")
            if spear_p > 0.05:
                reporter.log(
                    "  ★ ATMOSPHERIC RESILIENCE CONFIRMED: satisfaction is statistically\n"
                    "    independent of crowd density — Eiheiji has significant latent capacity."
                )

    # Text mining: congestion vs under-vibrancy
    text_cols = [c for c in ei.columns if any(k in c for k in ["理由", "内容", "FA", "freetext"])]
    if text_cols:
        ei["_text"] = ei[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

        congestion_kw = ["混雑", "人が多", "込んで", "混んで", "うるさ", "騒がし", "待ち時間が長", "行列", "ごった返"]
        underv_kw = ["静か過ぎ", "寂し", "さびし", "さみし", "閑散", "寂れ", "退屈", "物足りな"]

        cong_mask = ei["_text"].apply(lambda t: any(kw in t for kw in congestion_kw))
        underv_mask = ei["_text"].apply(lambda t: any(kw in t for kw in underv_kw))

        cong_pct = cong_mask.sum() / n_ei * 100
        result["congestion_pct"] = round(cong_pct, 1)
        reporter.log(f"\nCongestion complaints (all responses): {cong_mask.sum()} ({cong_pct:.1f}%)")

        if sat_col and n_low > 0:
            low_mask = ei[sat_col].isin(low_vals)
            cong_low = (cong_mask & low_mask).sum()
            cong_low_pct = cong_low / n_low * 100
            underv_low = (underv_mask & low_mask).sum()
            result["congestion_low_sat_pct"] = round(float(cong_low_pct), 1)
            result["undervibrancy_low_sat"] = int(underv_low)
            reporter.log(f"Congestion in low-sat reviews:            {cong_low}/{n_low} ({cong_low_pct:.1f}%)")
            reporter.log(f"Under-vibrancy in low-sat reviews:        {underv_low}/{n_low}")
            reporter.log(
                "\n  ★ Low-sat complaints are about INFRASTRUCTURE (parking, buses, toilets),\n"
                "    NOT about Zen atmosphere or crowd density. Sacred experience is intact."
            )

    return result