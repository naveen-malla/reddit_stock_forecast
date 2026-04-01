#!/usr/bin/env python
"""
Refresh thesis-facing outputs from cached raw Reddit and market data.

This is a faster, reproducible path for regenerating:
  - scored sentiment files
  - trained model metrics
  - prediction tables
  - evaluation plots
  - manual sentiment validation appendix
  - model analysis report
"""

from __future__ import annotations

import argparse
import runpy

import joblib
import pandas as pd

from config import cfg
from src.dataset_builder import DatasetBuilder
from src.market_data import MarketDataFetcher
from src.models import ModelTrainer, NaiveBaseline, evaluate
from src.reddit_collector import RedditCollector
from src.results_analyzer import ResultsAnalyzer
from src.sentiment_engine import SentimentEngine
from src.sentiment_validation import SentimentValidationReport
from src.ticker_selector import TickerSelector
from src.visualiser import Visualiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh thesis-facing outputs from cached data.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain all models instead of reusing the saved model files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = cfg.data_raw / "reddit_raw.parquet"
    reddit_df = pd.read_parquet(raw_path)
    reddit_df = RedditCollector._clip_to_window(
        reddit_df,
        pd.Timestamp(cfg.start_date).date(),
        pd.Timestamp(cfg.end_date).date(),
    )

    sentiment_df = SentimentEngine(use_finbert=False).score_and_aggregate(reddit_df)
    top_tickers = TickerSelector().get_top_tickers(force=False)
    market_df = MarketDataFetcher().fetch_and_engineer(top_tickers, force=False)

    db = DatasetBuilder()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = db.build(
        market_df=market_df,
        sentiment_df=sentiment_df,
        force=False,
    )

    preds = db.test_meta.copy().reset_index(drop=True)
    preds["actual"] = y_test

    if args.retrain:
        mt = ModelTrainer()
        results = mt.train_and_evaluate(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            feature_cols,
        )
        mt.print_comparison(results)
        for model_name, model in mt.trained_models.items():
            preds[model_name] = model.predict(X_test)
    else:
        baseline = NaiveBaseline()
        baseline.set_ret_col_idx(feature_cols)
        saved_models = {
            "Naive Baseline": baseline,
            "XGBoost": joblib.load(cfg.models_dir / "xgboost_model.pkl"),
            "XGBoost Calibrated": joblib.load(cfg.models_dir / "xgboost_calibrated_model.pkl"),
            "LightGBM": joblib.load(cfg.models_dir / "lightgbm_model.pkl"),
        }
        rows = []
        for model_name, model in saved_models.items():
            preds[model_name] = model.predict(X_test)
            metric_row = evaluate(model_name, y_test, preds[model_name])
            if hasattr(model, "threshold"):
                metric_row["direction_threshold"] = model.threshold
            rows.append(metric_row)
        pd.DataFrame(rows).to_csv(cfg.outputs_dir / "model_comparison.csv", index=False)

    preds.to_parquet(cfg.outputs_dir / "test_predictions.parquet", index=False)

    Visualiser().plot_all()
    ResultsAnalyzer().run_all()
    SentimentValidationReport().generate()
    runpy.run_path("generate_report.py", run_name="__main__")
    print("thesis outputs refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
