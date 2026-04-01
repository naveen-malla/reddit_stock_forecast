# Results Guide

This folder contains the committed snapshot of the thesis-facing result artifacts.

## Recommended Reading Order

1. `model_analysis.txt`
   - Primary written summary of dataset coverage, model performance, stability analysis, sentiment validation, and limitations.
2. `directional_accuracy_stats.csv`
   - Directional accuracy with 95% Wilson confidence intervals for each model.
3. `ticker_model_metrics.csv`
   - Ticker-level MAE, RMSE, and directional accuracy on the held-out test split.
4. `monthly_model_metrics.csv`
   - Month-by-month performance stability across the test period.
5. `sentiment_validation_summary.txt`
   - Manual review summary comparing hand-labelled sentiment against VADER classifications.
6. `refresh_run.log`
   - Clean log from the cached artifact refresh workflow.

## Figure Guide

- `model_comparison.png`
  - High-level comparison of MAE, RMSE, and directional accuracy across all models.
- `directional_accuracy_ci.png`
  - Directional accuracy with uncertainty bounds. Use this when discussing whether the lift over the benchmark is statistically meaningful.
- `ticker_directional_accuracy.png`
  - Compares the best model against the persistence benchmark at ticker level.
- `monthly_directional_accuracy.png`
  - Shows whether test-period directional performance is stable or concentrated in specific months.
- `sentiment_validation_confusion.png`
  - Summarises agreement between manual sentiment labels and VADER classifications.

## Interpretation Notes

- The baseline model is labelled `Persistence Benchmark`.
  - It predicts the next-day return using the most recent observed one-day return.
- Directional accuracy should be interpreted alongside its confidence interval.
  - A small lift over 50% may still represent only modest predictive value.
- Ticker-level and monthly breakdowns are diagnostic tools.
  - They help identify where performance is concentrated or unstable.
- The sentiment validation appendix is best used as a limitation and methodology check.
  - It shows that automatic sentiment labelling is useful but imperfect.
