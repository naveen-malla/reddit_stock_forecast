"""
src/market_data.py
──────────────────
Fetches OHLCV data from Yahoo Finance and engineers market features:
returns, volatility, RSI, MACD, Bollinger Bands.

FIXES vs v1:
  - Outlier capping on returns (Winsorize at 1st/99th percentile)
  - Coverage validation printed after fetch
  - Multi-level column handling for newer yfinance versions
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg


class MarketDataFetcher:
    """Downloads OHLCV data and computes technical features for each ticker."""

    def __init__(self):
        self.out_dir = cfg.data_processed

    def fetch_and_engineer(
        self,
        tickers: List[str],
        start: str | None = None,
        end: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        start = start or cfg.start_date
        end = end or cfg.end_date
        cache = self.out_dir / "market_features.parquet"

        if cache.exists() and not force:
            logger.debug(f"Market features cache hit: {cache}")
            return pd.read_parquet(cache)

        frames = []
        for ticker in tickers:
            logger.info(f"  Fetching {ticker} …")
            try:
                raw = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False
                )
                if raw.empty:
                    logger.warning(f"  No data for {ticker}")
                    continue

                raw = raw.reset_index()
                # Handle multi-level columns from newer yfinance versions
                # yfinance >=0.2.40 returns (Price, Ticker) multi-level; flatten to Price level only
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0]
                                   for col in raw.columns]
                raw.columns = [c.lower() for c in raw.columns]

                raw["ticker"] = ticker
                engineered = self._engineer(raw)
                frames.append(engineered)
            except Exception as e:
                logger.warning(f"  Error fetching {ticker}: {e}")

        if not frames:
            raise RuntimeError("No market data fetched.")

        df = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        self._validate_market_coverage(df)

        df.to_parquet(cache, index=False)
        logger.success(f"Market features saved → {cache}  shape={df.shape}")
        return df

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        c = df["close"]

        # ── Targets ──────────────────────────────────────────────────────────
        df["next_day_close"] = c.shift(-1)
        df["next_day_close_pct"] = (c.shift(-1) - c) / c

        # ── Returns ──────────────────────────────────────────────────────────
        for n in [1, 3, 5, 10, 20]:
            df[f"ret_{n}d"] = c.pct_change(n)

        # ── Outlier capping on returns (Winsorize) ────────────────────────────
        for col in [f"ret_{n}d" for n in [1, 3, 5, 10, 20]]:
            if col in df.columns:
                lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
                df[col] = df[col].clip(lo, hi)

        # ── Volatility ───────────────────────────────────────────────────────
        log_ret = np.log(c / c.shift(1))
        df["volatility_10d"] = log_ret.rolling(10).std() * np.sqrt(252)
        df["volatility_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

        # ── RSI ──────────────────────────────────────────────────────────────
        df["rsi_14"] = MarketDataFetcher._rsi(c, 14)

        # ── MACD ─────────────────────────────────────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ── Bollinger Bands ───────────────────────────────────────────────────
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

        # ── Volume features ───────────────────────────────────────────────────
        df["log_volume"] = np.log1p(df["volume"])
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_change_pct"] = df["volume"].pct_change(1)
        df["volume_ratio"] = df["volume"] / (df["volume_ma20"] + 1)

        # ── Price vs MAs ──────────────────────────────────────────────────────
        df["sma20"] = sma20
        df["sma50"] = c.rolling(50).mean()
        df["price_sma20_pct"] = (c - sma20) / (sma20 + 1e-9)
        df["price_sma50_pct"] = (c - df["sma50"]) / (df["sma50"] + 1e-9)

        return df.dropna(subset=["next_day_close_pct"])

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _validate_market_coverage(df: pd.DataFrame) -> None:
        """Print market data coverage — proves date span to client."""
        coverage = (
            df.groupby("ticker")["date"]
            .agg(min_date="min", max_date="max", trading_days="count")
        )
        print("\n" + "═" * 68)
        print("  MARKET DATA COVERAGE (Acceptance Criterion 1)")
        print("═" * 68)
        print(f"  {'Ticker':<8} {'From':<14} {'To':<14} {'Trading Days':>14}")
        print("─" * 68)
        for ticker, row in coverage.iterrows():
            span = (row["max_date"] - row["min_date"]).days / 365.25
            print(
                f"  {ticker:<8} {str(row['min_date'].date()):<14} "
                f"{str(row['max_date'].date()):<14} "
                f"{row['trading_days']:>14,}  ({span:.1f} yrs)"
            )
        print("═" * 68 + "\n")


if __name__ == "__main__":
    from src.ticker_selector import TickerSelector
    ts = TickerSelector()
    tickers = ts.get_top_tickers()
    mdf = MarketDataFetcher()
    df = mdf.fetch_and_engineer(tickers)
    print(df.dtypes)
    print(df.head())
