"""
src/ticker_selector.py
──────────────────────
Ranks candidate tickers by 90-day average daily trading volume.
Returns top-N tickers by volume.

FIX: yfinance 0.2.x multi-level column handling added.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg


class TickerSelector:

    def __init__(self, candidates: List[str] | None = None, top_n: int | None = None):
        self.candidates = candidates or cfg.candidate_tickers
        self.top_n = top_n or cfg.top_n_tickers
        self._ranking: pd.DataFrame | None = None

    def get_top_tickers(self, force: bool = False) -> List[str]:
        df = self.volume_ranking_df(force=force)
        return df.head(self.top_n)["ticker"].tolist()

    def volume_ranking_df(self, force: bool = False) -> pd.DataFrame:
        cache = cfg.data_processed / "volume_ranking.parquet"
        if cache.exists() and not force:
            logger.debug(f"Volume ranking cache hit: {cache}")
            self._ranking = pd.read_parquet(cache)
            return self._ranking

        logger.info(f"Fetching 90-day volume data for {len(self.candidates)} candidates …")
        records = []

        for ticker in self.candidates:
            try:
                hist = self._fetch_history(ticker)
                if hist is None or hist.empty:
                    logger.warning(f"  No data for {ticker}, skipping.")
                    continue
                avg_vol = hist["Volume"].mean()
                if pd.isna(avg_vol) or avg_vol == 0:
                    logger.warning(f"  Zero volume for {ticker}, skipping.")
                    continue
                records.append({"ticker": ticker, "avg_daily_volume": avg_vol})
                logger.debug(f"  {ticker}: avg_vol={avg_vol:,.0f}")
            except Exception as e:
                logger.warning(f"  Error fetching {ticker}: {e}")

        if not records:
            raise RuntimeError(
                "Could not fetch volume data for any candidate ticker.\n"
                "Possible causes:\n"
                "  1. No internet connection\n"
                "  2. yfinance needs update — run: pip install --upgrade yfinance\n"
                "  3. Yahoo Finance temporarily unavailable — try again in a few minutes"
            )

        df = (
            pd.DataFrame(records)
            .sort_values("avg_daily_volume", ascending=False)
            .reset_index(drop=True)
        )
        df["rank"] = df.index + 1
        df.to_parquet(cache, index=False)
        logger.success(f"Volume ranking saved → {cache}")
        self._ranking = df
        return df

    @staticmethod
    def _fetch_history(ticker: str) -> pd.DataFrame | None:
        """
        Fetch 90-day history with fallback for different yfinance versions.
        yfinance 0.2.x returns multi-level columns — this handles both.
        """
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="90d")

            if hist.empty:
                return None

            # Handle multi-level columns (newer yfinance versions)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            return hist

        except Exception as e:
            logger.debug(f"  _fetch_history error for {ticker}: {e}")
            return None

    def print_ranking(self) -> None:
        df = self.volume_ranking_df()
        top  = df.head(self.top_n)
        rest = df.iloc[self.top_n:]

        print("\n" + "═" * 52)
        print(f"  Top-{self.top_n} Tickers by 90-Day Avg Daily Volume")
        print("═" * 52)
        print(f"  {'Rank':<6} {'Ticker':<10} {'Avg Daily Volume':>18}")
        print("─" * 52)
        for _, row in top.iterrows():
            print(f"  {int(row['rank']):<6} {row['ticker']:<10} {row['avg_daily_volume']:>18,.0f}  ✓")
        print("─" * 52)
        print(f"  (Remaining {len(rest)} candidates excluded)")
        print("═" * 52 + "\n")


if __name__ == "__main__":
    ts = TickerSelector()
    ts.print_ranking()
    print("Selected:", ts.get_top_tickers())
