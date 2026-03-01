#!/usr/bin/env python3
"""Regenerate PDF reports from Markdown source files.

Requirements (install once):
    sudo apt-get install -y pandoc texlive-xetex texlive-lang-japanese fonts-noto-cjk

Usage:
    python3 scripts/generate_pdfs.py                # build all
    python3 scripts/generate_pdfs.py --executive    # Japanese executive report only
    python3 scripts/generate_pdfs.py --executive-en # English executive report only
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent        # repo root
PDF_DIR = ROOT / "output" / "pdf"

DOCS = {
    "executive_en": {
        "source": PDF_DIR / "executive_report_pdf_en.md",
        "output": PDF_DIR / "EXECUTIVE_REPORT.pdf",
        "expected_pages": 2,
    },
    "executive": {
        "source": PDF_DIR / "executive_report_pdf.md",
        "output": PDF_DIR / "EXECUTIVE_REPORT.ja.pdf",
        "expected_pages": 2,
    },
}


def get_pdf_pages(pdf_path: Path) -> int | None:
    if not shutil.which("pdfinfo"):
        return None
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                try:
                    return int(parts[1].strip())
                except ValueError:
                    return None
    return None


def check_deps() -> None:
    missing = []
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        print(f"[ERROR] Missing tools: {', '.join(missing)}")
        print()
        print("Install with:")
        print("  sudo apt-get install -y pandoc texlive-xetex texlive-lang-japanese fonts-noto-cjk")
        sys.exit(1)

    result = subprocess.run(
        ["fc-list", "Noto Sans CJK JP"],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print("[ERROR] Font 'Noto Sans CJK JP' not found.")
        print("  Install with: sudo apt-get install -y fonts-noto-cjk")
        sys.exit(1)


def build(name: str, doc: dict) -> bool:
    source = doc["source"]
    output = doc["output"]

    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        return False

    print(f"  Building {output.name} ...", end=" ", flush=True)
    result = subprocess.run(
        ["pandoc", str(source), "--pdf-engine=xelatex", "-o", str(output)],
        capture_output=True, text=True,
        cwd=str(source.parent),
    )
    if result.returncode == 0:
        expected_pages = doc.get("expected_pages")
        if expected_pages is not None:
            actual_pages = get_pdf_pages(output)
            if actual_pages is None:
                print("FAILED")
                print("[ERROR] Could not verify page count (pdfinfo missing).")
                return False
            if actual_pages != expected_pages:
                print("FAILED")
                print(
                    f"[ERROR] {output.name}: expected {expected_pages} pages, got {actual_pages}"
                )
                return False
        size_kb = output.stat().st_size // 1024
        print(f"OK  ({size_kb} KB)")
        return True
    else:
        print("FAILED")
        print(result.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate PDF reports from Markdown sources.")
    parser.add_argument("--executive",    action="store_true", help="Build Japanese executive report only")
    parser.add_argument("--executive-en", action="store_true", help="Build English executive report only")
    args = parser.parse_args()

    flags = {
        "executive_en": args.executive_en,
        "executive":    args.executive,
    }
    requested = [k for k, v in flags.items() if v]
    targets = requested if requested else list(DOCS.keys())

    print("Checking dependencies...")
    check_deps()
    print("  pandoc + xelatex + Noto Sans CJK JP: OK\n")

    print("Generating PDFs:")
    results = {name: build(name, DOCS[name]) for name in targets}

    print()
    if all(results.values()):
        print("Done. Output files:")
        for name in targets:
            p = DOCS[name]["output"]
            print(f"  {p.relative_to(ROOT)}")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"[ERROR] Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
