from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = ROOT / "EXECUTIVE_REPORT.en.md"
METRICS = ROOT / "output" / "analysis_metrics.txt"
OUTPUT = ROOT / "output" / "EXECUTIVE_REPORT.en.md"


def extract_metrics(text):
    metrics = {}

    patterns = {
        "ols_r2": r"OLS R²\s*=\s*([\d.]+)",
        "adj_r2": r"Adj R²\s*=\s*([\d.]+)",
        "rf_cv": r"RF 5-fold CV R²\s*=\s*([\d.]+)",
        "holdout_r2": r"Hold-out R²:\s*([\d.]+)",
        "lost_visitors": r"Total Lost Visitors:\s*([\d,]+)",
        "lost_yen": r"Total_Lost_Yen=([\d.]+)",
        "weather_ratio": r"Ratio \(Winter/Summer\):\s*([\d.]+)x",
        "ishikawa_r": r"Correlation at best lag:\s*r = ([+\-\d.]+)",
        "undervibrancy": r"Ratio vs high-satisfaction visitors:\s*([\d.]+)x",
        "pearson": r"Pearson_r=([\d.]+)",
        "rank": r"Winter hypothetical rank:\s*([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = match.group(1)

    return metrics


def update_report(report, metrics):

    replacements = {

        r"OLS RÂ² / Adjusted RÂ² \| 0\.810 / 0\.802":
            f"OLS RÂ² / Adjusted RÂ² | {metrics.get('ols_r2','')} / {metrics.get('adj_r2','')}",

        r"RF 5-fold CV RÂ² \| 0\.557 Â± 0\.131":
            f"RF 5-fold CV RÂ² | {metrics.get('rf_cv','')}",

        r"Lost visitors \(4 nodes\) \| 865,917 / year":
            f"Lost visitors (4 nodes) | {metrics.get('lost_visitors','')} / year",

        r"Winter weather sensitivity \| 6\.26Ã— summer":
            f"Winter weather sensitivity | {metrics.get('weather_ratio','')}× summer",

        r"Ishikawa â†’ Fukui leading indicator \| r = \+0\.549":
            f"Ishikawa → Fukui leading indicator | r = {metrics.get('ishikawa_r','')}",

        r"Under-vibrancy ratio \| 11\.5Ã—":
            f"Under-vibrancy ratio | {metrics.get('undervibrancy','')}×",

    }

    for old, new in replacements.items():
        report = re.sub(old, new, report)

    return report


def main():

    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)

    if not METRICS.exists():
        raise FileNotFoundError(METRICS)

    report = TEMPLATE.read_text(encoding="utf-8")
    metrics_text = METRICS.read_text(encoding="utf-8")

    metrics = extract_metrics(metrics_text)

    updated = update_report(report, metrics)

    OUTPUT.write_text(updated, encoding="utf-8")

    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()