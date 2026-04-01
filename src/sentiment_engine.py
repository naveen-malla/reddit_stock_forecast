"""
src/sentiment_engine.py
───────────────────────
Scores every Reddit post/comment with two sentiment signals:

  1. VADER  – rule-based, fast, optimised for social-media text.
  2. FinBERT (optional) – transformer fine-tuned on financial language.
              Disabled automatically if torch/GPU is unavailable.

FIXES vs v1:
  - FinBERT uses GPU when available via torch.cuda.is_available()
  - Graceful fallback to VADER-only with clear logging
  - Sentiment windows use config.sentiment_windows properly
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

_FINBERT_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _FINBERT_AVAILABLE = True
except ImportError:
    warnings.warn("torch/transformers not found – FinBERT sentiment disabled.")


class SentimentEngine:
    """Scores Reddit text and returns per-(ticker, date) aggregated features."""

    def __init__(self, use_finbert: bool = True, batch_size: int = 64):
        self.vader = SentimentIntensityAnalyzer()
        self.use_finbert = use_finbert and _FINBERT_AVAILABLE
        self.batch_size = batch_size
        self._finbert_model = None
        self._finbert_tok = None
        # FIX: detect device for FinBERT — guard against torch not being installed
        if _FINBERT_AVAILABLE:
            import torch as _torch
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self._device = "cpu"

        if self.use_finbert:
            self._load_finbert()

    def score_and_aggregate(self, reddit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score raw reddit data and aggregate to (ticker, date) granularity.
        Input: raw_text, ticker_mentions, date, score columns
        Returns: DataFrame with (ticker, date) index and sentiment features
        """
        logger.info(f"Scoring {len(reddit_df):,} Reddit items …")
        df = reddit_df.copy()
        df = self._explode_tickers(df)

        logger.info("  Running VADER …")
        df["vader_compound"] = df["raw_text"].apply(self._vader_score)

        if self.use_finbert:
            logger.info(f"  Running FinBERT on {self._device} …")
            df["finbert_score"] = self._finbert_score_batch(df["raw_text"].tolist())
        else:
            logger.info("  FinBERT disabled — using VADER only.")
            df["finbert_score"] = np.nan

        out = cfg.data_processed / "reddit_scored.parquet"
        df.to_parquet(out, index=False)
        logger.success(f"Scored data → {out}")

        agg = self._aggregate(df)
        agg_out = cfg.data_processed / "sentiment_daily.parquet"
        agg.to_parquet(agg_out, index=False)
        logger.success(f"Daily sentiment → {agg_out}  shape={agg.shape}")
        return agg

    @staticmethod
    def _explode_tickers(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ticker"] = df["ticker_mentions"].str.split(",")
        df = df.explode("ticker").dropna(subset=["ticker"])
        df["ticker"] = df["ticker"].str.strip().str.upper()
        return df[df["ticker"] != ""]

    def _vader_score(self, text: str) -> float:
        return self.vader.polarity_scores(str(text))["compound"]

    def _load_finbert(self) -> None:
        model_name = "ProsusAI/finbert"
        logger.info(f"Loading FinBERT ({model_name}) on {self._device} …")
        try:
            self._finbert_tok = AutoTokenizer.from_pretrained(model_name)
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self._device)  # FIX: move to correct device
            self._finbert_model.eval()
            logger.success(f"FinBERT loaded on {self._device}.")
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}. Falling back to VADER only.")
            self.use_finbert = False

    def _finbert_score_batch(self, texts: list) -> list:
        import torch
        scores = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="FinBERT"):
            batch = texts[i: i + self.batch_size]
            enc = self._finbert_tok(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            # FIX: move inputs to same device as model
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._finbert_model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            signed = probs[:, 0] - probs[:, 1]  # positive (0) - negative (1)
            scores.extend(signed.tolist())
        return scores

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"])
        df["upvote_weight"] = (df["score"].clip(lower=0) + 1).apply(np.log1p)

        agg_parts = []
        for window in cfg.sentiment_windows:
            window_label = f"w{window}d"
            grp = (
                df.sort_values("date")
                .groupby("ticker")
                .apply(
                    lambda g: self._rolling_agg(
                        g.drop(columns="ticker", errors="ignore"),
                        window,
                        window_label,
                    )
                )
                .reset_index(level=0)
                .reset_index(drop=True)
            )
            agg_parts.append(grp)

        base = agg_parts[0]
        for extra in agg_parts[1:]:
            base = base.merge(extra, on=["ticker", "date"], how="outer")

        # Compute weighted mean separately to avoid index mismatch
        df = df.reset_index(drop=True)
        df["_w"] = df["upvote_weight"]

        def weighted_mean(grp):
            w = grp["_w"].values
            v = grp["vader_compound"].values
            return float(np.average(v, weights=w)) if len(v) > 0 else 0.0

        weighted = df.groupby(["ticker", "date"]).apply(weighted_mean).reset_index()
        weighted.columns = ["ticker", "date", "vader_weighted_mean"]

        daily = (
            df.groupby(["ticker", "date"])
            .agg(
                mention_count=("vader_compound", "count"),
                vader_mean=("vader_compound", "mean"),
                vader_std=("vader_compound", "std"),
                vader_pos_ratio=(
                    "vader_compound",
                    lambda x: (x > 0.05).sum() / max(len(x), 1),
                ),
                finbert_mean=("finbert_score", "mean"),
                finbert_pos_ratio=(
                    "finbert_score",
                    lambda x: (x > 0.05).sum() / max(len(x), 1),
                ),
            )
            .reset_index()
        )
        daily = daily.merge(weighted, on=["ticker", "date"], how="left")

        result = daily.merge(base, on=["ticker", "date"], how="left")
        result["date"] = pd.to_datetime(result["date"])
        return result.sort_values(["ticker", "date"]).reset_index(drop=True)

    @staticmethod
    def _rolling_agg(grp: pd.DataFrame, window: int, label: str) -> pd.DataFrame:
        grp = grp.set_index("date").sort_index()
        rolled = (
            grp["vader_compound"]
            .rolling(f"{window}D", min_periods=1)
            .agg(["mean", "std", "count"])
        )
        rolled.columns = [
            f"vader_{label}_mean",
            f"vader_{label}_std",
            f"vader_{label}_count",
        ]
        return rolled.reset_index().drop_duplicates("date", keep="last")

