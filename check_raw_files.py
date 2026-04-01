import pandas as pd
from pathlib import Path

folder = Path("data/raw/reddit")

files = list(folder.glob("*.parquet"))
if not files:
    print("No parquet files found in data/raw/reddit/")
    exit()

print(f"{'File':<50} {'Rows':>8}  {'ID Dupes':>10}  {'Date Range'}")
print("-" * 95)

total_rows = 0
total_dupes = 0

for f in sorted(files):
    try:
        df = pd.read_parquet(f)
        rows = len(df)
        dupes = df.duplicated(subset=['id']).sum() if 'id' in df.columns else 0

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
        else:
            date_range = "no date col"

        status = "✅" if dupes == 0 else "❌"
        print(f"{f.name:<50} {rows:>8,}  {dupes:>9,}{status}  {date_range}")
        total_rows += rows
        total_dupes += dupes
    except Exception as e:
        print(f"{f.name:<50} ERROR: {e}")

print("-" * 95)
print(f"{'TOTAL':<50} {total_rows:>8,}  {total_dupes:>10,}")
print()
print(f"Total rows  : {total_rows:,}")
print(f"Total dupes : {total_dupes:,}")
print(f"Status      : {'✅ Clean' if total_dupes == 0 else '❌ Has duplicates'}")
