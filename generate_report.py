"""
generate_report.py
──────────────────
Generates a clean model analysis report and saves to outputs/model_analysis.txt
Run: python generate_report.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from config import cfg
from src.models import BASELINE_MODEL_NAME

_ROOT = Path(__file__).resolve().parent
outputs = _ROOT / "outputs"
data_processed = _ROOT / "data" / "processed"
data_raw = _ROOT / "data" / "raw"

lines = []

def w(text=""):
    lines.append(text)

# ── Header ────────────────────────────────────────────────────────────────────
w("=" * 70)
w("  REDDIT EQUITY FORECAST — MODEL ANALYSIS REPORT")
w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
w("=" * 70)

# ── 1. Dataset Coverage ───────────────────────────────────────────────────────
w()
w("1. DATASET COVERAGE")
w("-" * 70)

try:
    mkt = pd.read_parquet(data_processed / "market_features.parquet")
    coverage = mkt.groupby("ticker")["date"].agg(
        min_date="min", max_date="max", trading_days="count"
    )
    coverage["years"] = (
        pd.to_datetime(coverage["max_date"]) - pd.to_datetime(coverage["min_date"])
    ).dt.days / 365.25

    w(f"  {'Ticker':<8} {'From':<14} {'To':<14} {'Days':>8}  {'Years':>6}")
    w("  " + "-" * 54)
    for ticker, row in coverage.iterrows():
        w(f"  {ticker:<8} {str(row['min_date'].date()):<14} "
          f"{str(row['max_date'].date()):<14} {row['trading_days']:>8,}  {row['years']:>5.1f}yr")
    w()
    w(f"  Total tickers     : {len(coverage)}")
    w(f"  Configured window : {cfg.start_date} → {cfg.end_date}")
    w(f"  Market data span  : {coverage['min_date'].min().date()} → {coverage['max_date'].max().date()}")
    w(f"  Avg trading days  : {coverage['trading_days'].mean():.0f} per ticker")
except Exception as e:
    w(f"  Market data not found: {e}")

# ── 2. Reddit Coverage ────────────────────────────────────────────────────────
w()
w("2. REDDIT DATA COVERAGE")
w("-" * 70)

try:
    reddit = pd.read_parquet(data_raw / "reddit_raw.parquet")
    reddit["date"] = pd.to_datetime(reddit["date"])
    span = (reddit["date"].max() - reddit["date"].min()).days / 365.25
    w(f"  Total posts/comments : {len(reddit):,}")
    w(f"  Configured window    : {cfg.start_date} → {cfg.end_date}")
    w(f"  Collected range      : {reddit['date'].min().date()} → {reddit['date'].max().date()}")
    w(f"  Span                 : {span:.1f} years")
    in_window = reddit["date"].between(pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date))
    w(f"  Rows in window       : {int(in_window.sum()):,} / {len(reddit):,}")
    if "source" in reddit.columns:
        w()
        w("  Source breakdown:")
        for src, cnt in reddit["source"].value_counts().items():
            w(f"    {src:<20} {cnt:>6,} rows")
except Exception as e:
    w(f"  Reddit data not found: {e}")

# ── 3. Top-10 Tickers by Volume ───────────────────────────────────────────────
w()
w("3. TOP-10 TICKER SELECTION — BY TRADING VOLUME (Acceptance Criterion 3)")
w("-" * 70)

try:
    vol = pd.read_parquet(data_processed / "volume_ranking.parquet")
    w(f"  {'Rank':<6} {'Ticker':<10} {'Avg Daily Volume':>20}")
    w("  " + "-" * 40)
    for _, row in vol.iterrows():
        w(f"  {int(row['rank']):<6} {row['ticker']:<10} {row['avg_daily_volume']:>20,.0f}")
    w()
    w("  Selection method: 90-day average daily trading volume (Yahoo Finance)")
except Exception as e:
    w(f"  Volume ranking not found: {e}")

# ── 4. Model Performance ──────────────────────────────────────────────────────
w()
w("4. MODEL PERFORMANCE COMPARISON")
w("-" * 70)

try:
    comp = pd.read_csv(outputs / "model_comparison.csv")
    w(f"  {'Model':<20} {'MAE':>10} {'RMSE':>10} {'Dir. Acc.':>12}")
    w("  " + "-" * 56)
    for _, row in comp.iterrows():
        w(f"  {row['model']:<20} {row['MAE']:>10.5f} {row['RMSE']:>10.5f} "
          f"{row['DirectionalAccuracy']:>11.1%}")

    baseline_da = comp[comp["model"] == BASELINE_MODEL_NAME]["DirectionalAccuracy"].values[0]
    ml_comp = comp[comp["model"] != BASELINE_MODEL_NAME]
    best_ml_da_idx = ml_comp["DirectionalAccuracy"].idxmax()
    best_model = comp.loc[best_ml_da_idx, "model"]
    best_da = comp.loc[best_ml_da_idx, "DirectionalAccuracy"]
    best_gain = best_da - baseline_da

    w()
    w(f"  Persistence benchmark DA : {baseline_da:.1%}")
    w(f"  Best model              : {best_model}")
    w(f"  Best directional acc    : {best_da:.1%}")
    w(f"  Gain vs benchmark       : {best_gain:+.1%}")
except Exception as e:
    w(f"  Model comparison not found: {e}")

# ── 5. Evaluation Protocol ────────────────────────────────────────────────────
w()
w("5. EVALUATION PROTOCOL")
w("-" * 70)
w("  • Split design           : first 70% of dates for training, next 10% for validation, final 20% for testing")
w("  • Target variable        : next-day close return, expressed as a percentage change")
w("  • MAE                    : average absolute prediction error")
w("  • RMSE                   : square-root of mean squared prediction error")
w("  • Directional Accuracy   : share of test observations where the predicted sign matches the realised sign")
w(f"  • Benchmark model        : {BASELINE_MODEL_NAME}")

# ── 6. Directional Confidence Intervals ───────────────────────────────────────
w()
w("6. DIRECTIONAL ACCURACY CONFIDENCE INTERVALS")
w("-" * 70)

try:
    stats = pd.read_csv(outputs / "directional_accuracy_stats.csv")
    w(f"  {'Model':<20} {'DA':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}")
    w("  " + "-" * 62)
    for _, row in stats.iterrows():
        w(
            f"  {row['model']:<20} {row['directional_accuracy']:>9.1%} "
            f"{row['ci_lower']:>13.1%} {row['ci_upper']:>13.1%}"
        )
except Exception as e:
    w(f"  Directional confidence intervals not found: {e}")

# ── 7. Stability by Month and Ticker ──────────────────────────────────────────
w()
w("7. STABILITY BY MONTH AND TICKER")
w("-" * 70)

try:
    comp = pd.read_csv(outputs / "model_comparison.csv")
    best_model = comp[comp["model"] != BASELINE_MODEL_NAME].sort_values("DirectionalAccuracy", ascending=False).iloc[0]["model"]
    monthly = pd.read_csv(outputs / "monthly_model_metrics.csv")
    ticker = pd.read_csv(outputs / "ticker_model_metrics.csv")

    best_monthly = monthly[monthly["model"] == best_model].sort_values("month")
    best_ticker = ticker[ticker["model"] == best_model].sort_values("directional_accuracy", ascending=False)

    w(f"  Best directional model : {best_model}")
    w(
        f"  Monthly DA range       : {best_monthly['directional_accuracy'].min():.1%} "
        f"→ {best_monthly['directional_accuracy'].max():.1%}"
    )
    w(
        f"  Monthly average DA     : {best_monthly['directional_accuracy'].mean():.1%}"
    )
    w()
    w("  Strongest tickers on the test split:")
    for _, row in best_ticker.head(3).iterrows():
        w(f"    {row['ticker']:<8} DA={row['directional_accuracy']:.1%}  MAE={row['mae']:.5f}")
    w()
    w("  Weakest tickers on the test split:")
    for _, row in best_ticker.tail(3).sort_values("directional_accuracy").iterrows():
        w(f"    {row['ticker']:<8} DA={row['directional_accuracy']:.1%}  MAE={row['mae']:.5f}")
except Exception as e:
    w(f"  Stability analysis not found: {e}")

# ── 8. Interpretation of Model Results ────────────────────────────────────────
w()
w("8. INTERPRETATION OF MODEL RESULTS")
w("-" * 70)

try:
    comp = pd.read_csv(outputs / "model_comparison.csv")
    ml = comp[comp["model"] != BASELINE_MODEL_NAME]
    baseline_da = comp[comp["model"] == BASELINE_MODEL_NAME]["DirectionalAccuracy"].values[0]

    for _, row in ml.iterrows():
        w()
        w(f"  {row['model']}:")
        da = row["DirectionalAccuracy"]
        mae = row["MAE"]
        rmse = row["RMSE"]
        gain = da - baseline_da

        if gain > 0:
            w(f"    Relative to the persistence benchmark, directional accuracy improves by {gain:.1%}.")
        else:
            w(f"    Relative to the persistence benchmark, directional accuracy is lower by {abs(gain):.1%}.")

        if mae < 0.02:
            w(f"    The mean absolute error is comparatively low at {mae:.5f} relative to daily return scale.")

        if rmse / max(mae, 1e-9) > 1.5:
            w(f"    RMSE ({rmse:.5f}) is materially above MAE ({mae:.5f}), which suggests sensitivity to larger forecast errors.")

        w(f"    The model still assigns the wrong return direction on {(1-da):.1%} of test observations.")

    w()
    w("  OVERALL OBSERVATIONS:")
    best_row = ml.loc[ml["DirectionalAccuracy"].idxmax()]
    w(
        f"    • Best observed directional accuracy is {best_row['DirectionalAccuracy']:.1%} "
        f"from {best_row['model']}."
    )
    w(
        f"    • That represents a {best_row['DirectionalAccuracy'] - baseline_da:+.1%} lift "
        "over the persistence benchmark on this test split."
    )
    w("    • Treat feature importance as in-sample association, not proof of causal predictive signal.")
    w("    • Predictions remain close to zero on many days, which is typical for next-day return regression.")

except Exception as e:
    w(f"  Could not generate analysis: {e}")

# ── 9. Manual Sentiment Validation ────────────────────────────────────────────
w()
w("9. MANUAL SENTIMENT VALIDATION")
w("-" * 70)

try:
    validation = pd.read_csv(outputs / "sentiment_validation_sample.csv")
    agreement = validation["is_match"].mean()
    w(f"  Reviewed sample size : {len(validation)}")
    w(f"  VADER agreement      : {agreement:.1%}")
    w()
    w("  Manual label support:")
    for label, count in validation["manual_label"].value_counts().sort_index().items():
        w(f"    {label:<8} {count:>3}")
except Exception as e:
    w(f"  Sentiment validation not found: {e}")

# ── 10. Limitations ───────────────────────────────────────────────────────────
w()
w("10. LIMITATIONS & DISCLAIMERS")
w("-" * 70)
w("  • Directional accuracy above 50% does not, by itself, guarantee economic profitability.")
w("  • Reddit sentiment reflects correlation, not causation.")
w("  • Viral narratives and coordinated posting activity may introduce additional noise.")
w("  • Transaction costs and slippage are not modelled.")
w("  • Past predictability does not imply future predictability.")
w("  • Walk-forward backtesting recommended before live deployment.")

# ── 11. Output Files ──────────────────────────────────────────────────────────
w()
w("11. OUTPUT FILES DELIVERED")
w("-" * 70)
w("  data/raw/reddit_raw.parquet              — Raw Reddit posts & comments")
w("  data/processed/sentiment_daily.parquet   — Daily sentiment per ticker")
w("  data/processed/market_features.parquet   — OHLCV + technical indicators")
w("  data/processed/model_dataset.parquet     — Final merged dataset")
w("  models/xgboost_model.pkl                 — Trained XGBoost model")
w("  models/lightgbm_model.pkl                — Trained LightGBM model")
w("  outputs/model_comparison.csv             — Performance metrics")
w("  outputs/directional_accuracy_stats.csv   — Directional accuracy confidence intervals")
w("  outputs/ticker_model_metrics.csv         — Ticker-level test metrics")
w("  outputs/monthly_model_metrics.csv        — Monthly test-period metrics")
w("  outputs/volume_ranking.png               — Top-10 ticker ranking chart")
w("  outputs/sentiment_timeseries.png         — 5-year sentiment chart")
w("  outputs/model_comparison.png             — Model comparison chart")
w("  outputs/directional_accuracy_ci.png      — Directional accuracy with confidence intervals")
w("  outputs/ticker_directional_accuracy.png  — Ticker-level directional comparison")
w("  outputs/monthly_directional_accuracy.png — Month-by-month directional stability")
w("  outputs/feature_importance.png           — Feature importance chart")
w("  outputs/predictions_scatter.png          — Actual vs Predicted")
w("  outputs/residual_distribution.png        — Residual distribution for the best model")
w("  outputs/direction_confusion.png          — Direction confusion matrix for the best model")
w("  outputs/sentiment_validation_sample.csv  — Manual sentiment review sample")
w("  outputs/sentiment_validation_summary.txt — Manual sentiment review summary")
w("  outputs/sentiment_validation_confusion.png — Manual vs VADER confusion matrix")
w("  outputs/refresh_run.log                  — Clean thesis artifact refresh log")
w("  outputs/sentiment_interactive.html       — Interactive sentiment")
w("  outputs/predictions_interactive.html     — Interactive predictions")

w()
w("=" * 70)
w("  END OF REPORT")
w("=" * 70)

# Save
report_path = outputs / "model_analysis.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Report saved → {report_path}")
print("\nPreview:")
print("\n".join(lines[:30]))
