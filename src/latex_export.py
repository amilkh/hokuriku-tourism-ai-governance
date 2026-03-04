"""LaTeX table export for paper submission.

Generates publication-ready LaTeX tables from analysis results.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


def ols_summary_to_latex(
    params: pd.Series,
    pvalues: pd.Series,
    r2: float,
    adj_r2: float,
    n: int,
    *,
    caption: str = "OLS Regression Results",
    label: str = "tab:ols",
) -> str:
    """Convert OLS results to a LaTeX table.

    Args:
        params: Coefficient series (index = feature names).
        pvalues: P-value series (aligned with ``params``).
        r2: R-squared.
        adj_r2: Adjusted R-squared.
        n: Number of observations.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX source string.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Variable & Coefficient & $p$-value \\",
        r"\midrule",
    ]
    for feat in params.index:
        coef = params[feat]
        p = pvalues[feat]
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        feat_escaped = feat.replace("_", r"\_")
        lines.append(f"  {feat_escaped} & {coef:+.4f}{stars} & {p:.4f} \\\\")
    lines += [
        r"\midrule",
        f"  $R^2$ & {r2:.4f} & \\\\",
        f"  Adj. $R^2$ & {adj_r2:.4f} & \\\\",
        f"  $N$ & {n} & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def model_comparison_to_latex(
    metrics: dict[str, dict[str, float]],
    *,
    caption: str = "Model Comparison",
    label: str = "tab:model_comparison",
) -> str:
    """Generate a LaTeX comparison table for multiple models.

    Args:
        metrics: ``{model_name: {metric_name: value}}``.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX source string.
    """
    all_metric_names: list[str] = []
    for m in metrics.values():
        for k in m:
            if k not in all_metric_names:
                all_metric_names.append(k)

    headers = " & ".join(["Model"] + [n.replace("_", r"\_") for n in all_metric_names])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "r" * len(all_metric_names) + "}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for model_name, vals in metrics.items():
        cells = [model_name.replace("_", r"\_")]
        for mn in all_metric_names:
            v = vals.get(mn)
            cells.append(f"{v:.4f}" if v is not None and not np.isnan(v) else "--")
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def key_metrics_to_latex(
    kv: dict[str, Any],
    *,
    caption: str = "Key Research Metrics",
    label: str = "tab:key_metrics",
) -> str:
    """Simple two-column key-value table.

    Args:
        kv: ``{metric_label: value}``.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for k, v in kv.items():
        k_esc = str(k).replace("_", r"\_")
        if isinstance(v, float):
            v_str = f"{v:,.4f}"
        elif isinstance(v, int):
            v_str = f"{v:,}"
        else:
            v_str = str(v)
        lines.append(f"  {k_esc} & {v_str} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def statistical_rigor_to_latex(
    rigor: Any,
    *,
    caption: str = "Statistical Rigor: Effect Size and Predictive Validity",
    label: str = "tab:statistical_rigor",
) -> str:
    """Generate table_statistical_rigor.tex from a StatisticalRigorResult.

    Args:
        rigor: ``StatisticalRigorResult`` instance.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX source string.
    """
    beta: pd.Series = rigor.beta_coefficients
    f2: float = rigor.cohens_f2

    magnitude = (
        r"large ($\geq 0.35$)" if f2 >= 0.35 else
        r"medium ($\geq 0.15$)" if f2 >= 0.15 else
        r"small ($\geq 0.02$)" if f2 >= 0.02 else
        r"negligible ($< 0.02$)"
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"\multicolumn{3}{l}{\textbf{Panel A: Standardised Coefficients ($\beta$)}} \\",
        r"\midrule",
        r"Feature & $\beta$ & $|\beta|$ rank \\",
        r"\midrule",
    ]

    sorted_beta = beta.sort_values(key=abs, ascending=False)
    for rank, (feat, b) in enumerate(sorted_beta.items(), start=1):
        feat_esc = str(feat).replace("_", r"\_")
        lines.append(f"  {feat_esc} & ${b:+.4f}$ & {rank} \\\\")

    lines += [
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Panel B: Global Effect Size}} \\",
        r"\midrule",
        f"  Cohen's $f^2$ & ${f2:.4f}$ & {magnitude} \\\\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Panel C: Out-of-Sample Predictive Validity}} \\",
        r"\midrule",
        f"  Training $N$ & {rigor.train_n} & \\\\",
        f"  Hold-out $N$ & {rigor.holdout_n} & \\\\",
        f"  Hold-out MAE & ${rigor.holdout_mae:.1f}$ visitors/day & \\\\",
        f"  Hold-out RMSE & ${rigor.holdout_rmse:.1f}$ visitors/day & \\\\",
        f"  Hold-out $R^2$ & ${rigor.holdout_r2:.4f}$ & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def export_all_tables(
    results: dict[str, Any],
    output_dir: str,
) -> list[str]:
    """Generate and save all LaTeX tables.

    Args:
        results: Full pipeline results dictionary.
        output_dir: Directory for ``.tex`` files.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    # OLS table
    ols = results.get("ols")
    if ols is not None:
        coef_names = ["const"] + list(getattr(ols, "feature_cols", []))
        params = ols.model.params
        pvalues = ols.model.pvalues

        if not isinstance(params, pd.Series):
            params = pd.Series(params, index=coef_names[:len(params)])
        if not isinstance(pvalues, pd.Series):
            pvalues = pd.Series(pvalues, index=coef_names[:len(pvalues)])

        tex = ols_summary_to_latex(
            params,
            pvalues,
            ols.r2,
            ols.adj_r2,
            int(ols.model.nobs),
        )
        p = os.path.join(output_dir, "table_ols.tex")
        with open(p, "w") as f:
            f.write(tex)
        paths.append(p)

    # Model comparison table
    rf = results.get("rf")
    robust = results.get("robust")
    if ols and rf:
        comp: dict[str, dict[str, float]] = {
            "OLS": {"R2": ols.r2, "Adj_R2": ols.adj_r2},
            "Random Forest": {"R2": rf.r2_train, "CV_R2": rf.cv_r2_mean},
        }
        if robust:
            comp["First-Difference"] = {"R2": robust.fd_r2, "DW": robust.fd_dw}
            comp["LDV"] = {"R2": robust.ldv_r2, "DW": robust.ldv_dw}
        tex = model_comparison_to_latex(comp)
        p = os.path.join(output_dir, "table_model_comparison.tex")
        with open(p, "w") as f:
            f.write(tex)
        paths.append(p)

    # Key metrics table
    kv: dict[str, Any] = {}
    economics = results.get("economics", {})
    spatial = results.get("spatial", {})
    if "total_lost" in economics:
        kv["Lost Visitors (single-node)"] = int(economics["total_lost"])
    if "satake_lost_visitors" in spatial:
        kv["Lost Visitors (3-node)"] = int(spatial["satake_lost_visitors"])
        kv["Satake Number (¥)"] = f"¥{spatial['satake_yen']:,.0f}"
    ccf_res = results.get("ccf", {})
    if "best_r" in ccf_res:
        kv["Ishikawa CCF r"] = ccf_res["best_r"]
        kv["Best lag (days)"] = ccf_res["best_lag"]
    seasonal = results.get("seasonal", {})
    if "ratio" in seasonal:
        kv["Weather Sensitivity Ratio (W/S)"] = f"{seasonal['ratio']:.2f}x"

    if kv:
        tex = key_metrics_to_latex(kv)
        p = os.path.join(output_dir, "table_key_metrics.tex")
        with open(p, "w") as f:
            f.write(tex)
        paths.append(p)

    # Statistical rigor table (effect size + predictive validity)
    rigor = results.get("rigor")
    if rigor is not None:
        tex = statistical_rigor_to_latex(rigor)
        p = os.path.join(output_dir, "table_statistical_rigor.tex")
        with open(p, "w") as f:
            f.write(tex)
        paths.append(p)

    return paths
