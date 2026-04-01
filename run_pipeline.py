#!/usr/bin/env python
"""
run_pipeline.py
───────────────
End-to-end pipeline runner.  Execute from the project root:

    python run_pipeline.py [--force] [--skip-reddit] [--skip-sentiment] [--use-finbert]

Flags:
  --force           Re-download / re-compute everything (ignore caches).
  --skip-reddit     Use cached Reddit data (useful during model iterations).
  --skip-sentiment  Skip sentiment scoring; uses only market features.
  --use-finbert     Enable optional FinBERT scoring (downloads model if not cached).

Exit codes:
  0 — success
  1 — configuration error (missing .env credentials)
  2 — runtime error

Pipeline stages:
  1. Validate configuration
  2. Select top-10 tickers by trading volume
  3. Collect Reddit posts/comments (archive APIs + optional PRAW supplement)
  4. Score text with VADER (+ optional FinBERT)
  5. Fetch OHLCV data and engineer market features
  6. Merge, lag, and split the modelling dataset
  7. Train Naive Baseline, XGBoost, XGBoost Calibrated, LightGBM; evaluate and compare
  8. Generate plots and save outputs
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add(
    Path("outputs") / "pipeline.log",
    rotation="10 MB",
    level="DEBUG",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reddit Equity Forecast – full pipeline")
    p.add_argument("--force", action="store_true", help="Ignore all caches and recompute")
    p.add_argument("--skip-reddit", action="store_true", help="Skip Reddit collection")
    p.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment scoring")
    p.add_argument(
        "--use-finbert",
        action="store_true",
        help="Enable optional FinBERT scoring (downloads model if not cached)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── 1. Config ─────────────────────────────────────────────────────────────
    try:
        from config import cfg
        cfg.validate()
    except EnvironmentError as e:
        logger.error(str(e))
        return 1

    logger.info("=" * 60)
    logger.info("  Reddit Equity Forecast Pipeline  v2.0")
    logger.info(f"  Study window : {cfg.start_date} → {cfg.end_date}")
    logger.info(f"  Top-N tickers: {cfg.top_n_tickers}")
    logger.info(f"  FinBERT      : {'enabled' if args.use_finbert else 'disabled'}")
    logger.info("=" * 60)
    if not args.skip_reddit and not cfg.has_reddit_credentials:
        logger.info("  Reddit credentials not found — running archive-only collection without PRAW.")

    try:
        # ── 2. Ticker selection ───────────────────────────────────────────────
        logger.info("\n[Stage 2] Selecting top tickers by trading volume …")
        from src.ticker_selector import TickerSelector
        ts = TickerSelector()
        ts.print_ranking()
        top_tickers = ts.get_top_tickers(force=args.force)
        logger.success(f"  Top-{cfg.top_n_tickers} tickers: {top_tickers}")

        # ── 3. Reddit collection ──────────────────────────────────────────────
        reddit_df = None
        if not args.skip_reddit:
            logger.info("\n[Stage 3] Collecting Reddit data …")
            from src.reddit_collector import RedditCollector
            rc = RedditCollector()
            reddit_df = rc.run(tickers=top_tickers, force=args.force)
            logger.success(f"  Reddit rows collected: {len(reddit_df):,}")
        else:
            raw_path = cfg.data_raw / "reddit_raw.parquet"
            sent_path = cfg.data_processed / "sentiment_daily.parquet"
            if raw_path.exists():
                import pandas as pd
                from datetime import datetime
                from src.reddit_collector import RedditCollector

                reddit_df = pd.read_parquet(raw_path)
                start_dt = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
                end_dt = datetime.strptime(cfg.end_date, "%Y-%m-%d").date()
                raw_rows = len(reddit_df)
                reddit_df = RedditCollector._clip_to_window(reddit_df, start_dt, end_dt)
                if len(reddit_df) != raw_rows:
                    reddit_df.to_parquet(raw_path, index=False)
                logger.info(
                    f"  Loaded cached Reddit data: {len(reddit_df):,} rows "
                    f"(clipped from {raw_rows:,} to the configured window)"
                )
                if reddit_df.empty:
                    logger.warning("  Cached Reddit data has no rows inside the configured window.")
            elif sent_path.exists():
                logger.warning("  No cached Reddit data found. Will reuse cached sentiment aggregates.")
            else:
                logger.warning("  No cached Reddit or sentiment data found. Running without sentiment.")
                args.skip_sentiment = True

        # ── 4. Sentiment scoring ──────────────────────────────────────────────
        sentiment_df = None
        if args.skip_sentiment:
            logger.info("\n[Stage 4] Sentiment scoring skipped — using market features only.")
        elif reddit_df is not None and len(reddit_df):
            logger.info("\n[Stage 4] Scoring sentiment …")
            from src.sentiment_engine import SentimentEngine
            se = SentimentEngine(use_finbert=args.use_finbert)
            sentiment_df = se.score_and_aggregate(reddit_df)
            logger.success(f"  Sentiment rows: {len(sentiment_df):,}")
        elif args.skip_reddit and reddit_df is None:
            sent_path = cfg.data_processed / "sentiment_daily.parquet"
            if sent_path.exists():
                import pandas as pd
                sentiment_df = pd.read_parquet(sent_path)
                logger.info(f"  Loaded cached sentiment: {len(sentiment_df):,} rows")
        else:
            logger.warning("  No Reddit rows available for sentiment scoring. Continuing with market features only.")

        # ── 5. Market data & features ─────────────────────────────────────────
        logger.info("\n[Stage 5] Fetching OHLCV data and engineering market features …")
        from src.market_data import MarketDataFetcher
        mdf = MarketDataFetcher()
        market_df = mdf.fetch_and_engineer(top_tickers, force=args.force)
        logger.success(f"  Market rows: {len(market_df):,}")

        # ── 6. Dataset construction ───────────────────────────────────────────
        logger.info("\n[Stage 6] Merging and splitting dataset …")
        from src.dataset_builder import DatasetBuilder
        db = DatasetBuilder()
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = db.build(
            market_df=market_df,
            sentiment_df=sentiment_df,
            force=args.force,
        )

        # ── 7. Model training & evaluation ────────────────────────────────────
        logger.info("\n[Stage 7] Training models (Baseline, XGBoost, XGBoost Calibrated, LightGBM) …")
        from src.models import ModelTrainer
        mt = ModelTrainer()
        results = mt.train_and_evaluate(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_cols
        )
        mt.print_comparison(results)

        # Persist predictions for the visualiser
        import pandas as pd
        preds = pd.DataFrame({"actual": y_test})
        for model_name, model in mt.trained_models.items():
            preds[model_name] = model.predict(X_test)
        preds.to_parquet(cfg.outputs_dir / "test_predictions.parquet", index=False)

        # ── 8. Visualisations ─────────────────────────────────────────────────
        logger.info("\n[Stage 8] Generating plots …")
        from src.visualiser import Visualiser
        v = Visualiser()
        v.plot_all()

        logger.info("\n" + "=" * 60)
        logger.success("  Pipeline completed successfully!")
        logger.info(f"  Outputs: {cfg.outputs_dir}")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed at runtime: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
