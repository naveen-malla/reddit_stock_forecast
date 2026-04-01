import sys
import pandas as pd
from pathlib import Path

# Market data coverage
mkt_path = Path('data/processed/market_features.parquet')
if mkt_path.exists():
    mkt = pd.read_parquet(mkt_path)
    coverage = mkt.groupby('ticker')['date'].agg(
        min_date='min', max_date='max', trading_days='count'
    )
    coverage['years'] = (
        pd.to_datetime(coverage['max_date']) - pd.to_datetime(coverage['min_date'])
    ).dt.days / 365.25
    print('=== MARKET DATA COVERAGE ===')
    print(coverage.to_string())
else:
    print(f'WARNING: {mkt_path} not found — run pipeline first.')

# Reddit coverage
reddit_path = Path('data/raw/reddit_raw.parquet')
if reddit_path.exists():
    reddit = pd.read_parquet(reddit_path)
    reddit['date'] = pd.to_datetime(reddit['date'])
    print()
    print('=== REDDIT COVERAGE ===')
    print(f'Total rows: {len(reddit):,}')
    print(f'Date range: {reddit["date"].min().date()} to {reddit["date"].max().date()}')
    print(f'Span: {(reddit["date"].max() - reddit["date"].min()).days / 365.25:.1f} years')
    if 'source' in reddit.columns:
        print('Source breakdown:')
        for src, cnt in reddit['source'].value_counts().items():
            print(f'  {src:<20} {cnt:>8,} rows')
else:
    print(f'\nWARNING: {reddit_path} not found — run pipeline first.')
