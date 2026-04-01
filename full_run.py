"""
full_run.py
───────────
Single script — delete cache, full pipeline, excel export, report, coverage check.
Run: python full_run.py
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

def delete_files(pattern):
    for f in glob.glob(pattern):
        try:
            os.remove(f)
            print(f"  Deleted: {f}")
        except:
            pass

print("=" * 60)
print("  FULL CLEAN RUN")
print("=" * 60)

# Step 1 — Delete old cache
print("\n[Step 1] Deleting old cache...")
delete_files("data/raw/reddit/praw_*.parquet")
delete_files("data/raw/reddit/arctic_*.parquet")
delete_files("data/raw/reddit_raw.parquet")
delete_files("data/processed/*.parquet")
delete_files("data/processed/*.version")
delete_files("outputs/*.png")
delete_files("outputs/*.csv")
delete_files("outputs/*.html")
delete_files("outputs/*.parquet")
delete_files("outputs/pipeline.log")
print("  Done.")

# Step 2 — Full pipeline
print("\n[Step 2] Running full pipeline...")
result = subprocess.run([sys.executable, "run_pipeline.py"], check=False)
if result.returncode != 0:
    print("❌ Pipeline failed. Check errors above.")
    sys.exit(1)
print("✅ Pipeline complete.")

# Step 3 — Excel export
print("\n[Step 3] Exporting to Excel...")
if Path("export_to_excel.py").exists():
    subprocess.run([sys.executable, "export_to_excel.py"], check=False)
    print("✅ Excel export done.")
else:
    print("⚠  export_to_excel.py not found — skip.")

# Step 4 — Generate report
print("\n[Step 4] Generating model report...")
if Path("generate_report.py").exists():
    subprocess.run([sys.executable, "generate_report.py"], check=False)
    print("✅ Report generated.")
else:
    print("⚠  generate_report.py not found — skip.")

# Step 5 — Coverage check
print("\n[Step 5] Coverage check...")
if Path("check_coverage.py").exists():
    subprocess.run([sys.executable, "check_coverage.py"], check=False)
else:
    print("⚠  check_coverage.py not found — skip.")

print("\n" + "=" * 60)
print("  ALL DONE ✅")
print("  outputs/        — charts, CSVs, HTML")
print("  outputs/excel/  — Excel files")
print("  outputs/model_analysis.txt — report")
print("=" * 60)
