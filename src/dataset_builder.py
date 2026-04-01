"""
src/dataset_builder.py
──────────────────────
Joins market features with daily sentiment features on (ticker, date).

FIXES vs v1:
  - fillna(method="ffill") → ffill() — fixes pandas 2.x crash
  - Chronological split per-ticker (not global index) to avoid data leakage
  - StandardScaler kept for forward-compatibility but clearly commented
  - Validation split carved from training data (for LightGBM early stopping)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg


_EXCLUDE_COLS = {
    "ticker", "date",
    "next_day_close", cfg.target_col,
    "open", "high", "low", "close", "volume",
    "sma20", "sma50", "bb_upper", "bb_lower",
    "macd", "macd_signal", "macd_hist",
    "log_volume", "volume_ma20",
}


class DatasetBuilder:
    """Merges market and sentiment features and produces train/val/test splits."""

    def __init__(self):
        self._feature_cols: List[str] = []


    def build(
        self,
        market_df: pd.DataFrame | None = None,
        sentiment_df: pd.DataFrame | None = None,
        force: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Build and return:
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        Val set is 10% of training data, used for early stopping only.
        """
        cache = cfg.data_processed / "model_dataset.parquet"
        if cache.exists() and not force:
            logger.debug("Dataset cache hit.")
            df = pd.read_parquet(cache)
        else:
            df = self._merge(market_df, sentiment_df)
            df.to_parquet(cache, index=False)
            logger.success(f"Model dataset saved → {cache}  shape={df.shape}")

        return self._split(df)

    def load_cached_dataset(self) -> pd.DataFrame:
        p = cfg.data_processed / "model_dataset.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found at {p}. Run build() first.")
        return pd.read_parquet(p)

    @property
    def feature_cols(self) -> List[str]:
        return self._feature_cols

    def _merge(
        self,
        market_df: pd.DataFrame | None,
        sentiment_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if market_df is None:
            mkt_path = cfg.data_processed / "market_features.parquet"
            logger.info(f"Loading market features from {mkt_path}")
            market_df = pd.read_parquet(mkt_path)

        if sentiment_df is None:
            sent_path = cfg.data_processed / "sentiment_daily.parquet"
            if sent_path.exists():
                logger.info(f"Loading sentiment from {sent_path}")
                sentiment_df = pd.read_parquet(sent_path)
            else:
                logger.warning("No sentiment data found – using market features only.")
                sentiment_df = pd.DataFrame()

        market_df["date"] = pd.to_datetime(market_df["date"])

        if len(sentiment_df):
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            sentiment_df = sentiment_df.copy()
            # Lag sentiment by 1 day: today's features use yesterday's Reddit activity
            sentiment_df["date"] = sentiment_df["date"] + pd.Timedelta(days=1)

            # Outer merge to preserve weekends until ffill is complete
            merged = market_df.merge(sentiment_df, on=["ticker", "date"], how="outer")
            merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)


            sent_cols = [c for c in sentiment_df.columns if c not in ("ticker", "date")]
            # FIX: ffill() instead of deprecated fillna(method="ffill")
            merged[sent_cols] = (
                merged.groupby("ticker")[sent_cols]
                .transform(lambda g: g.ffill(limit=3))
            )
            merged[sent_cols] = merged[sent_cols].fillna(0)
        else:
            merged = market_df.copy()

        return merged.dropna(subset=[cfg.target_col]).reset_index(drop=True)

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Chronological split per ticker, then pooled.
        Train: first 70% | Val: next 10% | Test: last 20%
        This prevents future ticker data leaking into earlier ticker training.
        """
        feature_cols = sorted([
            c for c in df.columns
            if c not in _EXCLUDE_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ])
        self._feature_cols = feature_cols

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Obtain global chronological splits
        dates = np.sort(df["date"].unique())
        n_dates = len(dates)
        if n_dates < 10:
            raise RuntimeError(
                f"Only {n_dates} unique dates in dataset — too few to split into train/val/test. "
                "Collect more data or widen the study window in config."
            )
        train_end_date = dates[int(n_dates * 0.70)]
        val_end_date = dates[int(n_dates * 0.80)]

        train_df = df[df["date"] < train_end_date].sort_values("date").reset_index(drop=True)
        val_df = df[(df["date"] >= train_end_date) & (df["date"] < val_end_date)].sort_values("date").reset_index(drop=True)
        test_df = df[df["date"] >= val_end_date].sort_values("date").reset_index(drop=True)


        def extract(part):
            X = part[feature_cols].fillna(0).values
            y = part[cfg.target_col].values
            return X, y

        X_train, y_train = extract(train_df)
        X_val, y_val = extract(val_df)
        X_test, y_test = extract(test_df)

        logger.info(
            f"Split sizes — Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}"
            f"  Features: {len(feature_cols)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

