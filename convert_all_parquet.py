"""
convert_all_parquet.py
──────────────────────
Converts all project Parquet files to CSV for easy inspection in Excel / notebooks.
Run: python convert_all_parquet.py

Outputs land in:  outputs/csv/          — main pipeline files
                  outputs/csv/reddit/   — per-subreddit raw crawl files
"""

import pandas as pd
from pathlib import Path

# ── Root-relative paths so the script works from any working directory ────────
_ROOT = Path(__file__).resolve().parent

_MAIN_FILES = {
    _ROOT / 'data/raw/reddit_raw.parquet':              _ROOT / 'outputs/csv/reddit_raw.csv',
    _ROOT / 'data/processed/market_features.parquet':   _ROOT / 'outputs/csv/market_features.csv',
    _ROOT / 'data/processed/sentiment_daily.parquet':   _ROOT / 'outputs/csv/sentiment_daily.csv',
    _ROOT / 'data/processed/model_dataset.parquet':     _ROOT / 'outputs/csv/model_dataset.csv',
    _ROOT / 'data/processed/volume_ranking.parquet':    _ROOT / 'outputs/csv/volume_ranking.csv',
    _ROOT / 'outputs/test_predictions.parquet':         _ROOT / 'outputs/csv/test_predictions.csv',
}

# ── Create output directory (parents=True handles outputs/ not existing yet) ──
(_ROOT / 'outputs/csv').mkdir(parents=True, exist_ok=True)

done = 0
skipped = 0

# ── Convert main pipeline files ───────────────────────────────────────────────
for src, dst in _MAIN_FILES.items():
    if not src.exists():
        print(f'⚠  Skip (not found): {src.relative_to(_ROOT)}')
        skipped += 1
        continue
    try:
        df = pd.read_parquet(src)
        df.to_csv(dst, index=False)
        print(f'✅ {src.relative_to(_ROOT):<50}  →  {dst.relative_to(_ROOT)}  ({len(df):,} rows)')
        done += 1
    except Exception as e:
        print(f'❌ Error reading {src.relative_to(_ROOT)}: {e}')
        skipped += 1

# ── Convert per-subreddit raw crawl files ─────────────────────────────────────
reddit_raw_dir = _ROOT / 'data/raw/reddit'
if reddit_raw_dir.exists():
    out_reddit_csv = _ROOT / 'outputs/csv/reddit'
    out_reddit_csv.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(reddit_raw_dir.glob('*.parquet'))
    if parquet_files:
        print(f'\nConverting {len(parquet_files)} per-subreddit raw files …')
    for pq in parquet_files:
        dst = out_reddit_csv / (pq.stem + '.csv')
        try:
            df = pd.read_parquet(pq)
            df.to_csv(dst, index=False)
            print(f'✅ {pq.name:<50}  ({len(df):,} rows)')
            done += 1
        except Exception as e:
            print(f'❌ Error reading {pq.name}: {e}')
            skipped += 1

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f'Done: {done} file(s) converted   Skipped: {skipped} file(s)')
print(f'CSV output directory: {_ROOT / "outputs/csv"}')
