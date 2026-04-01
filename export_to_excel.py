"""
export_to_excel.py
──────────────────
Exports all project data to Excel files.
Run: python export_to_excel.py
"""

import pandas as pd
from pathlib import Path

output_dir = Path("outputs/excel")
output_dir.mkdir(exist_ok=True)

files = {
    "data/processed/volume_ranking.parquet":    "volume_ranking.xlsx",
    "data/raw/reddit_raw.parquet":              "reddit_raw.xlsx",
    "data/processed/sentiment_daily.parquet":   "sentiment_daily.xlsx",
    "data/processed/market_features.parquet":   "market_features.xlsx",
    "outputs/model_comparison.csv":             "model_comparison.xlsx",
    "outputs/test_predictions.parquet":         "test_predictions.xlsx",
    "outputs/xgboost_feature_importance.csv":   "xgboost_features.xlsx",
    "outputs/lightgbm_feature_importance.csv":  "lightgbm_features.xlsx",
}

for src, dst in files.items():
    try:
        src_path = Path(src)
        if not src_path.exists():
            print(f"⚠  Skip (not found): {src}")
            continue

        if src.endswith(".parquet"):
            df = pd.read_parquet(src_path)
        else:
            df = pd.read_csv(src_path)

        out_path = output_dir / dst
        df.to_excel(out_path, index=False)
        print(f"✅ Saved: {out_path}  ({len(df):,} rows)")

    except Exception as e:
        print(f"❌ Error: {src} — {e}")

print()
print(f"All Excel files saved in: outputs/excel/")
