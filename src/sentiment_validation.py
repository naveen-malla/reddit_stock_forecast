"""
src/sentiment_validation.py
───────────────────────────
Generates a manual sentiment-validation appendix from a reviewed label file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from src.plot_style import FIG_DPI, POSITIVE, add_title, apply_chart_style


_NEG_THRESHOLD = -0.05
_POS_THRESHOLD = 0.05


apply_chart_style(font_scale=1.0)


class SentimentValidationReport:
    def __init__(self):
        self.review_path = cfg.root_dir / "data" / "validation" / "sentiment_manual_labels.csv"
        self.scored_path = cfg.data_processed / "reddit_scored.parquet"
        self.out_dir = cfg.outputs_dir

    def generate(self) -> None:
        if not self.review_path.exists() or not self.scored_path.exists():
            logger.warning("Sentiment validation inputs not found — skipping appendix generation.")
            return

        scored = pd.read_parquet(self.scored_path)
        reviewed = pd.read_csv(self.review_path)

        scored = scored.sort_values(["id", "date"]).drop_duplicates("id").reset_index(drop=True)
        reviewed["manual_label"] = reviewed["manual_label"].str.strip().str.lower()
        scored["auto_label"] = scored["vader_compound"].apply(self._auto_label)

        merged = reviewed.merge(
            scored[["id", "date", "ticker_mentions", "raw_text", "vader_compound", "auto_label"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        missing_ids = merged[merged["raw_text"].isna()]["id"].tolist()
        if missing_ids:
            raise ValueError(f"Manual sentiment review contains IDs missing from scored data: {missing_ids}")

        merged["date"] = pd.to_datetime(merged["date"])
        merged["is_match"] = merged["manual_label"] == merged["auto_label"]
        merged = merged.sort_values(["manual_label", "date", "id"]).reset_index(drop=True)

        sample_path = self.out_dir / "sentiment_validation_sample.csv"
        merged.to_csv(sample_path, index=False)
        logger.success(f"Saved validation sample → {sample_path}")

        confusion = pd.crosstab(
            merged["manual_label"],
            merged["auto_label"],
            rownames=["Manual"],
            colnames=["VADER"],
            dropna=False,
        ).reindex(index=["negative", "neutral", "positive"], columns=["negative", "neutral", "positive"], fill_value=0)

        summary_lines = [
            "SENTIMENT VALIDATION SUMMARY",
            "=" * 48,
            f"Sample size          : {len(merged)}",
            f"Agreement            : {merged['is_match'].mean():.1%}",
            f"Negative support     : {(merged['manual_label'] == 'negative').sum()}",
            f"Neutral support      : {(merged['manual_label'] == 'neutral').sum()}",
            f"Positive support     : {(merged['manual_label'] == 'positive').sum()}",
            "",
            "Confusion Matrix (manual rows x VADER columns)",
            confusion.to_string(),
        ]
        summary_path = self.out_dir / "sentiment_validation_summary.txt"
        summary_path.write_text("\n".join(summary_lines))
        logger.success(f"Saved validation summary → {summary_path}")

        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(POSITIVE, as_cmap=True),
            cbar=False,
            linewidths=1.0,
            linecolor="white",
            ax=ax,
        )
        add_title(
            ax,
            "Manual vs VADER Sentiment Labels",
            "Counts are based on the reviewed sample committed in data/validation.",
        )
        fig.tight_layout()
        fig.savefig(self.out_dir / "sentiment_validation_confusion.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Saved plot → {self.out_dir / 'sentiment_validation_confusion.png'}")

    @staticmethod
    def _auto_label(score: float) -> str:
        if score <= _NEG_THRESHOLD:
            return "negative"
        if score >= _POS_THRESHOLD:
            return "positive"
        return "neutral"
