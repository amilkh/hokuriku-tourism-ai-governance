"""
Quick debug script to inspect actual data formats.
Run: python dashboard/debug_data.py
"""
import pandas as pd
from pathlib import Path
import glob

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

print("=" * 70)
print("DEBUGGING DATA FORMATS")
print("=" * 70)

# ── 1. Checking JMA files ──
print("\n📂 JMA FILES:")
jma_dir = PROJECT_ROOT / "jma"
print(f"   Looking in: {jma_dir}")
print(f"   Exists: {jma_dir.exists()}")

if jma_dir.exists():
    csv_files = sorted(jma_dir.glob("jma_*.csv"))
    print(f"   Found {len(csv_files)} JMA CSV files")

    for fp in csv_files[:3]:  # Check first 3
        print(f"\n   ── {fp.name} ──")

        # Trying different encodings
        for enc in ["utf-8", "shift_jis", "cp932", "utf-8-sig"]:
            try:
                df = pd.read_csv(fp, encoding=enc, nrows=5)
                print(f"   Encoding: {enc} ✅")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   First 2 rows:")
                print(df.head(2).to_string())
                print(f"   Dtypes:\n{df.dtypes}")
                break
            except Exception as e:
                print(f"   Encoding {enc}: ❌ {e}")

# ── 2. Checking Survey files ──
print("\n\n📂 SURVEY FILES:")
survey_dir = WORKSPACE_ROOT / "fukui-kanko-survey"
print(f"   Looking in: {survey_dir}")
print(f"   Exists: {survey_dir.exists()}")

if survey_dir.exists():
    csv_files = list(survey_dir.rglob("*.csv"))
    print(f"   Found {len(csv_files)} CSV files")

    for fp in csv_files[:5]:
        print(f"\n   ── {fp.relative_to(survey_dir)} ──")
        for enc in ["utf-8", "shift_jis", "cp932", "utf-8-sig"]:
            try:
                df = pd.read_csv(fp, encoding=enc, nrows=5)
                print(f"   Encoding: {enc} ✅")
                print(f"   Shape: {df.shape}")
                print(f"   Columns ({len(df.columns)}): {list(df.columns)}")

                # Checking for text-like columns
                for col in df.columns:
                    if df[col].dtype == object:
                        sample = df[col].dropna().head(2).tolist()
                        if sample and any(len(str(s)) > 20 for s in sample):
                            print(f"   📝 Potential text column: '{col}'")
                            print(f"      Sample: {sample[0][:100]}...")
                break
            except Exception as e:
                continue

# ── 3. Checking all CSVs in jma/ (including non-jma_ prefixed) ──
print("\n\n📂 ALL FILES IN jma/:")
if jma_dir.exists():
    all_files = list(jma_dir.glob("*"))
    for f in all_files:
        print(f"   {f.name} ({f.stat().st_size / 1024:.1f} KB)")

print("\n" + "=" * 70)
print("DEBUG COMPLETE — Copy the output and share it")
print("=" * 70)