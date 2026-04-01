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
_DATASET_CACHE_VERSION = "3"


class DatasetBuilder:
    """Merges market and sentiment features and produces train/val/test splits."""

    def __init__(self):
        self._feature_cols: List[str] = []
        self.train_meta: pd.DataFrame | None = None
        self.val_meta: pd.DataFrame | None = None
        self.test_meta: pd.DataFrame | None = None


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
        version_path = cfg.data_processed / "model_dataset.version"
        cache_version = version_path.read_text().strip() if version_path.exists() else None

        if cache.exists() and not force and cache_version == _DATASET_CACHE_VERSION:
            logger.debug("Dataset cache hit.")
            df = pd.read_parquet(cache)
        else:
            df = self._merge(market_df, sentiment_df)
            df.to_parquet(cache, index=False)
            version_path.write_text(_DATASET_CACHE_VERSION)
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

        merged = merged.dropna(subset=[cfg.target_col]).reset_index(drop=True)
        merged = self._engineer_features(merged)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        return merged.reset_index(drop=True)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
        df["date"] = pd.to_datetime(df["date"])

        df["dow"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker", dtype=np.int8)
        df = pd.concat([df, ticker_dummies], axis=1)

        close_denom = df["close"].replace(0, np.nan).abs()
        df["intraday_return"] = (df["close"] - df["open"]) / (df["open"].replace(0, np.nan).abs())
        df["range_pct"] = (df["high"] - df["low"]) / close_denom
        df["close_to_high_pct"] = (df["high"] - df["close"]) / close_denom
        df["close_to_low_pct"] = (df["close"] - df["low"]) / close_denom
        df["ret_1d_abs"] = df["ret_1d"].abs()

        ticker_groups = df.groupby("ticker", group_keys=False)
        df["ret_1d_mean_20"] = ticker_groups["ret_1d"].transform(
            lambda s: s.rolling(20, min_periods=5).mean()
        )
        df["ret_1d_std_20"] = ticker_groups["ret_1d"].transform(
            lambda s: s.rolling(20, min_periods=5).std()
        )
        df["ret_5d_mean_20"] = ticker_groups["ret_5d"].transform(
            lambda s: s.rolling(20, min_periods=5).mean()
        )

        sentiment_cols = [
            col
            for col in [
                "mention_count",
                "vader_mean",
                "vader_weighted_mean",
                "vader_pos_ratio",
                "vader_w3d_mean",
                "vader_w7d_mean",
                "finbert_mean",
                "finbert_pos_ratio",
            ]
            if col in df.columns and self._has_signal(df[col])
        ]
        for col in sentiment_cols:
            df[f"{col}_ma5"] = ticker_groups[col].transform(
                lambda s: s.rolling(5, min_periods=1).mean()
            )
            df[f"{col}_ma20"] = ticker_groups[col].transform(
                lambda s: s.rolling(20, min_periods=1).mean()
            )
            df[f"{col}_shock"] = df[col] - df[f"{col}_ma20"]

        if "mention_count_ma20" in df.columns:
            df["mention_surge"] = (df["mention_count"] - df["mention_count_ma20"]) / (
                df["mention_count_ma20"] + 1
            )

        sentiment_anchor = "vader_weighted_mean" if "vader_weighted_mean" in df.columns else "vader_mean"
        if sentiment_anchor in df.columns:
            df["sentiment_x_volume"] = df[sentiment_anchor] * df["volume_ratio"]
            df["sentiment_x_ret1d"] = df[sentiment_anchor] * df["ret_1d"]

        has_finbert_signal = "finbert_mean" in df.columns and self._has_signal(df["finbert_mean"])

        if has_finbert_signal and "vader_weighted_mean" in df.columns:
            df["sentiment_model_gap"] = df["finbert_mean"] - df["vader_weighted_mean"]
            df["sentiment_model_agreement"] = df["finbert_mean"] * df["vader_weighted_mean"]
        elif has_finbert_signal and "vader_mean" in df.columns:
            df["sentiment_model_gap"] = df["finbert_mean"] - df["vader_mean"]
            df["sentiment_model_agreement"] = df["finbert_mean"] * df["vader_mean"]

        return df

    @staticmethod
    def _has_signal(series: pd.Series) -> bool:
        values = pd.to_numeric(series, errors="coerce").fillna(0)
        return bool((values.abs() > 1e-12).any())

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Global chronological split across the pooled panel.
        Train: first 70% | Val: next 10% | Test: last 20%
        This prevents any test-period dates leaking into model fitting.
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

        self.train_meta = train_df[["ticker", "date"]].copy()
        self.val_meta = val_df[["ticker", "date"]].copy()
        self.test_meta = test_df[["ticker", "date"]].copy()


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
