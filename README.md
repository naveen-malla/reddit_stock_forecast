# Reddit Equity Forecast

Forecasting next-day equity returns using Reddit sentiment and market features.

This project builds a reproducible pipeline that collects Reddit discussions, aggregates daily sentiment by ticker, joins that signal with market data, and evaluates several predictive models on a chronological holdout period.

## Overview

- Study window: `2021-01-01` to `2025-12-31`
- Universe: top 10 tickers selected by 90-day average trading volume
- Reddit sources: Arctic Shift, PullPush, and optional PRAW tail collection
- Market source: Yahoo Finance
- Sentiment: VADER by default, optional FinBERT
- Models: Persistence Benchmark, XGBoost, XGBoost Calibrated, LightGBM
- Evaluation: chronological `70% / 10% / 20%` train, validation, test split

## Results

Held-out test performance:

| Model | MAE | RMSE | Directional Accuracy |
|---|---:|---:|---:|
| Persistence Benchmark | 0.02833 | 0.04170 | 48.6% |
| XGBoost | 0.01984 | 0.03048 | 52.9% |
| XGBoost Calibrated | 0.01984 | 0.03048 | 53.2% |
| LightGBM | 0.01982 | 0.03045 | 52.2% |

The best directional result comes from `XGBoost Calibrated`, with a held-out directional accuracy of `53.2%`. The committed evaluation snapshot is available in [`docs/results/`](docs/results/).

## Visual Summary

### Model Comparison

![Model Comparison](docs/results/model_comparison.png)

### Directional Accuracy Confidence Intervals

![Directional Accuracy Confidence Intervals](docs/results/directional_accuracy_ci.png)

### Stability by Ticker and Month

![Ticker Directional Accuracy](docs/results/ticker_directional_accuracy.png)

![Monthly Directional Accuracy](docs/results/monthly_directional_accuracy.png)

## Method

The pipeline follows five steps:

1. Select the top 10 liquid tickers from a fixed candidate universe.
2. Collect Reddit posts and comments, then restrict them to the configured study window.
3. Score text sentiment and aggregate it to daily ticker-level features.
4. Join sentiment features with OHLCV and technical indicators.
5. Train and evaluate models on a strict chronological split.

Key modelling choices:

- sentiment is lagged by one day before merge
- model selection uses a separate validation period
- the calibrated XGBoost model tunes its directional threshold on validation only
- integrity tests check date clipping, sentiment lagging, and split chronology

## Repository Structure

Core files:

- `run_pipeline.py`: full end-to-end pipeline
- `refresh_thesis_outputs.py`: fast regeneration of reports, tables, and figures from cached data and saved models
- `generate_report.py`: structured written summary of the current results
- `src/reddit_collector.py`: Reddit collection and coverage reporting
- `src/sentiment_engine.py`: sentiment scoring and daily aggregation
- `src/dataset_builder.py`: merge logic and chronological dataset split
- `src/models.py`: benchmark and machine-learning models
- `src/results_analyzer.py`: ticker, monthly, and confidence-interval analysis
- `tests/test_pipeline_integrity.py`: core integrity checks

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional Reddit credentials:

```bash
cp .env.example .env
```

Run the full pipeline:

```bash
python run_pipeline.py
```

Enable FinBERT:

```bash
python run_pipeline.py --use-finbert
```

Refresh the committed reports and figures from cached data:

```bash
python refresh_thesis_outputs.py
```

Run integrity checks:

```bash
python -m unittest discover -s tests -v
```

## Output Snapshot

The main committed artifacts are stored in `docs/results/`:

- `model_analysis.txt`: written summary of coverage, model results, stability, and limitations
- `directional_accuracy_stats.csv`: directional accuracy with confidence intervals
- `ticker_model_metrics.csv`: ticker-level performance on the test split
- `monthly_model_metrics.csv`: monthly stability across the test period
- `sentiment_validation_summary.txt`: manual sentiment review summary
- `refresh_run.log`: clean regeneration log for the committed artifact set

## Notes

- The default run uses VADER sentiment. FinBERT is supported but optional.
- Reddit sentiment is treated as an explanatory signal, not causal evidence.
- The observed lift over the persistence benchmark is modest and should be interpreted accordingly.
