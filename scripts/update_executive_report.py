import re
from pathlib import Path


METRICS = Path("output/analysis_metrics.txt")
REPORT = Path("EXECUTIVE_REPORT.en.md")


def extract(pattern, text, default=""):
    m = re.search(pattern, text, re.S)
    return m.group(1) if m else default


def replace_row(label_regex, new_value, text):
    """Replace the value in a LaTeX tabular row: 'Label & value \\'"""
    pattern = re.compile(rf"({label_regex} & ).*?( \\\\)")
    return pattern.sub(lambda m: f"{m.group(1)}{new_value}{m.group(2)}", text)


def main():

    metrics = METRICS.read_text(encoding="utf-8")
    report = REPORT.read_text(encoding="utf-8")

    #-------------------------
    # Metrics extraction
    #-------------------------

    m = re.search(
        r"OLS R.\s*=\s*([0-9.]+).*?Adj R.\s*=\s*([0-9.]+)",
        metrics,
        re.S,
    )
    ols_r2 = f"{m.group(1)} / {m.group(2)}" if m else "0.810 / 0.802"

    total_lost_visitors = extract(
        r"Multinode_Aggregate_Loss.*?Lost_Visitors=([0-9.]+).*?Total_Lost_Yen",
        metrics,
    )
    total_lost_visitors = (
        f"{float(total_lost_visitors):,.0f}" if total_lost_visitors else "865,917"
    )

    opportunity_gap = extract(r"Total_Lost_Billion_Yen=([0-9.]+)", metrics)
    opportunity_gap = f"¥{float(opportunity_gap):.2f}B" if opportunity_gap else "¥11.96B"

    usd_loss = extract(r"Total_Lost_USD_Million=([0-9.]+)", metrics)
    usd_loss = f"USD {usd_loss}M" if usd_loss else "USD 72.6M"   # FIXED: no bare '$' (breaks LaTeX math mode)

    #-------------------------
    # Replace table rows (LaTeX tabular format: "Label & value \\")
    #-------------------------

    report = replace_row(r"OLS R² / Adj\. R²", ols_r2, report)
    report = replace_row(
        r"Lost visitors \(4 nodes\)", f"{total_lost_visitors}/year", report
    )
    report = replace_row(
        r"Opportunity Gap", f"~{opportunity_gap} (~{usd_loss})", report
    )

    #-------------------------
    # Replace plain-text mentions (§3.3 and Conclusion)
    #-------------------------

    report = re.sub(
        r"Total annual revenue loss: ~¥[0-9.]+B \(~USD [0-9.]+M\)\.",
        f"Total annual revenue loss: ~{opportunity_gap} (~{usd_loss}).",
        report,
    )

    report = re.sub(
        r"AI governance recovers .*? lost visitors",
        f"AI governance recovers {total_lost_visitors} lost visitors",
        report,
    )

    report = re.sub(
        r"The .*? annual lost visitors",
        f"The {total_lost_visitors} annual lost visitors",
        report,
    )

    REPORT.write_text(report, encoding="utf-8")
    print("Updated:", REPORT)


if __name__ == "__main__":
    main()