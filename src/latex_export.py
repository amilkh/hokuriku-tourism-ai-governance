"""LaTeX table export for paper submission.

Generates publication-ready LaTeX tables from analysis results and
companion PNG previews rendered with matplotlib.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── PNG styling ───────────────────────────────────────────────────────────────
_HDR_BG  = "#2B5C8A"
_SEC_BG  = "#D8E5F0"
_ALT_BG  = "#F5F7FA"


def _render_table_png(
    col_labels: list[str],
    body: list,          # list[str] = section header row; list[list[str]] = data row
    caption: str,
    out_path: str,
    dpi: int = 150,
) -> None:
    """Render a table to PNG via matplotlib.

    Body rows are either:
    - ``list[str]``: normal data row
    - ``str``: section header spanning all columns
    """
    ncols = len(col_labels)

    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []
    is_section: list[bool] = []

    data_row_idx = 0
    for row in body:
        if isinstance(row, str):
            cell_text.append([row] + [""] * (ncols - 1))
            cell_colors.append([_SEC_BG] * ncols)
            is_section.append(True)
        else:
            padded = list(row) + [""] * (ncols - len(row))
            cell_text.append(padded[:ncols])
            bg = _ALT_BG if data_row_idx % 2 == 0 else "white"
            cell_colors.append([bg] * ncols)
            is_section.append(False)
            data_row_idx += 1

    nrows = len(cell_text)
    row_h = 0.30
    fig_h = max(3.0, nrows * row_h + 1.0)
    fig_w = max(6.0, ncols * 2.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(caption, fontsize=9, style="italic", pad=8)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=[_HDR_BG] * ncols,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)

    for j in range(ncols):
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, sec in enumerate(is_section):
        if sec:
            tbl[i + 1, 0].set_text_props(fontweight="bold", color="#1A3A5C")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── LaTeX generators ──────────────────────────────────────────────────────────

def ols_summary_to_latex(
    params: pd.Series,
    pvalues: pd.Series,
    r2: float,
    adj_r2: float,
    n: int,
    *,
    caption: str = "OLS regression results",
    label: str = "tab:ols",
) -> str:
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
    caption: str = "Forecasting model comparison",
    label: str = "tab:model_comparison",
) -> str:
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
    caption: str = "Key governance metrics",
    label: str = "tab:key_metrics",
) -> str:
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
    caption: str = "Statistical rigor: standardised coefficients, effect size, and hold-out validity",
    label: str = "tab:statistical_rigor",
) -> str:
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


# ── Master export ─────────────────────────────────────────────────────────────

def export_all_tables(
    results: dict[str, Any],
    output_dir: str,
) -> list[str]:
    """Generate and save all LaTeX tables and PNG previews.

    Args:
        results: Full pipeline results dictionary.
        output_dir: Directory for output files.

    Returns:
        List of saved file paths (.tex and .png).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    # ── Table 1: OLS ─────────────────────────────────────────────────────────
    ols = results.get("ols")
    if ols is not None:
        coef_names = ["const"] + list(getattr(ols, "feature_cols", []))
        params = ols.model.params
        pvalues = ols.model.pvalues
        r2, adj_r2, n = ols.r2, ols.adj_r2, int(ols.model.nobs)

        if not isinstance(params, pd.Series):
            params = pd.Series(params, index=coef_names[:len(params)])
        if not isinstance(pvalues, pd.Series):
            pvalues = pd.Series(pvalues, index=coef_names[:len(pvalues)])

        tex = ols_summary_to_latex(params, pvalues, r2, adj_r2, n)
        tex_path = os.path.join(output_dir, "table1_ols.tex")
        with open(tex_path, "w") as f:
            f.write(tex)
        paths.append(tex_path)

        # PNG
        body: list = []
        for feat in params.index:
            coef = params[feat]
            p = pvalues[feat]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            body.append([feat, f"{coef:+.4f}{stars}", f"{p:.4f}"])
        body += [
            ["R\u00b2",      f"{r2:.4f}",    ""],
            ["Adj. R\u00b2", f"{adj_r2:.4f}", ""],
            ["N",            str(n),          ""],
        ]
        png_path = os.path.join(output_dir, "table1_ols.png")
        _render_table_png(["Variable", "Coefficient", "p-value"], body,
                          "OLS regression results", png_path)
        paths.append(png_path)

    # ── Table 2: Statistical rigor ────────────────────────────────────────────
    rigor = results.get("rigor")
    if rigor is not None:
        tex = statistical_rigor_to_latex(rigor)
        tex_path = os.path.join(output_dir, "table2_statistical_rigor.tex")
        with open(tex_path, "w") as f:
            f.write(tex)
        paths.append(tex_path)

        # PNG
        f2: float = rigor.cohens_f2
        magnitude_plain = (
            "large (>=0.35)"   if f2 >= 0.35 else
            "medium (>=0.15)"  if f2 >= 0.15 else
            "small (>=0.02)"   if f2 >= 0.02 else
            "negligible (<0.02)"
        )
        sorted_beta = rigor.beta_coefficients.sort_values(key=abs, ascending=False)
        body = ["Panel A: Standardised Coefficients"]
        for rank, (feat, b) in enumerate(sorted_beta.items(), start=1):
            body.append([feat, f"{b:+.4f}", str(rank)])
        body.append("Panel B: Global Effect Size")
        body.append(["Cohen's f\u00b2", f"{f2:.4f}", magnitude_plain])
        body.append("Panel C: Hold-out Predictive Validity")
        body += [
            ["Training N",    str(rigor.train_n),         ""],
            ["Hold-out N",    str(rigor.holdout_n),        ""],
            ["Hold-out MAE",  f"{rigor.holdout_mae:.1f}",  "visitors/day"],
            ["Hold-out RMSE", f"{rigor.holdout_rmse:.1f}", "visitors/day"],
            ["Hold-out R\u00b2", f"{rigor.holdout_r2:.4f}", ""],
        ]
        png_path = os.path.join(output_dir, "table2_statistical_rigor.png")
        _render_table_png(
            ["Feature / Metric", "Value", "Info"], body,
            "Statistical rigor: standardised coefficients, effect size, and hold-out validity",
            png_path,
        )
        paths.append(png_path)

    # ── Table 3: Key metrics ──────────────────────────────────────────────────
    kv: dict[str, Any] = {}
    economics = results.get("economics", {})
    spatial = results.get("spatial", {})
    if "total_lost" in economics:
        kv["Lost Visitors (single-node)"] = int(economics["total_lost"])
    if "satake_lost_visitors" in spatial:
        kv["Lost Visitors (3-node)"] = int(spatial["satake_lost_visitors"])
        kv["Satake Number (\u00a5)"] = f"\u00a5{spatial['satake_yen']:,.0f}"
    ccf_res = results.get("ccf", {})
    if "best_r" in ccf_res:
        kv["Ishikawa CCF r"] = ccf_res["best_r"]
        kv["Best lag (days)"] = ccf_res["best_lag"]
    seasonal = results.get("seasonal", {})
    if "ratio" in seasonal:
        kv["Weather Sensitivity Ratio (W/S)"] = f"{seasonal['ratio']:.2f}x"

    if kv:
        tex = key_metrics_to_latex(kv)
        tex_path = os.path.join(output_dir, "table3_key_metrics.tex")
        with open(tex_path, "w") as f:
            f.write(tex)
        paths.append(tex_path)

        # PNG
        body = []
        for k, v in kv.items():
            if isinstance(v, float):
                v_str = f"{v:,.4f}"
            elif isinstance(v, int):
                v_str = f"{v:,}"
            else:
                v_str = str(v)
            body.append([str(k), v_str])
        png_path = os.path.join(output_dir, "table3_key_metrics.png")
        _render_table_png(["Metric", "Value"], body,
                          "Key governance metrics", png_path)
        paths.append(png_path)

    # ── Table 4: Model comparison ─────────────────────────────────────────────
    rf = results.get("rf")
    robust = results.get("robust")
    if ols and rf:
        comp: dict[str, dict[str, float]] = {
            "OLS":           {"R2": ols.r2, "Adj_R2": ols.adj_r2},
            "Random Forest": {"R2": rf.r2_train, "CV_R2": rf.cv_r2_mean},
        }
        if robust:
            comp["First-Difference"] = {"R2": robust.fd_r2, "DW": robust.fd_dw}
            comp["LDV"]              = {"R2": robust.ldv_r2, "DW": robust.ldv_dw}

        tex = model_comparison_to_latex(comp)
        tex_path = os.path.join(output_dir, "table4_model_comparison.tex")
        with open(tex_path, "w") as f:
            f.write(tex)
        paths.append(tex_path)

        # PNG
        all_metric_names: list[str] = []
        for m in comp.values():
            for k in m:
                if k not in all_metric_names:
                    all_metric_names.append(k)
        body = []
        for model_name, vals in comp.items():
            row = [model_name]
            for mn in all_metric_names:
                v = vals.get(mn)
                row.append(f"{v:.4f}" if v is not None and not np.isnan(v) else "\u2014")
            body.append(row)
        png_path = os.path.join(output_dir, "table4_model_comparison.png")
        _render_table_png(["Model"] + all_metric_names, body,
                          "Forecasting model comparison", png_path)
        paths.append(png_path)

    return paths
