"""
src/results_analyzer.py
───────────────────────
Builds thesis-oriented evaluation tables and plots from the saved test predictions.

Outputs:
  - directional_accuracy_stats.csv
  - ticker_model_metrics.csv
  - monthly_model_metrics.csv
  - directional_accuracy_ci.png
  - ticker_directional_accuracy.png
  - monthly_directional_accuracy.png
  - residual_distribution.png
  - direction_confusion.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg


sns.set_theme(style="darkgrid", context="notebook", font_scale=1.0)
PALETTE = sns.color_palette("tab10")
_META_COLS = {"ticker", "date", "month", "actual"}


def _directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((np.sign(y_pred) == np.sign(y_true)).mean())


def _wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = hits / total
    denom = 1 + z ** 2 / total
    center = (phat + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _metric_row(model: str, actual: pd.Series, pred: pd.Series) -> dict:
    err = pred - actual
    return {
        "model": model,
        "n_obs": int(len(actual)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "directional_accuracy": _directional_accuracy(actual, pred),
    }


class ResultsAnalyzer:
    def __init__(self):
        self.out_dir = cfg.outputs_dir

    def run_all(self) -> None:
        pred_df = self._load_predictions()
        if pred_df is None:
            return
        comparison_df = self._load_comparison()
        stats_df = self.save_directional_accuracy_stats(pred_df)
        ticker_df = self.save_ticker_metrics(pred_df)
        monthly_df = self.save_monthly_metrics(pred_df)
        self.plot_directional_accuracy_ci(stats_df)
        self.plot_ticker_directional_accuracy(ticker_df, comparison_df)
        self.plot_monthly_directional_accuracy(monthly_df)
        self.plot_residual_distribution(pred_df, comparison_df)
        self.plot_direction_confusion(pred_df, comparison_df)

    def _load_predictions(self) -> pd.DataFrame | None:
        pred_path = self.out_dir / "test_predictions.parquet"
        if not pred_path.exists():
            logger.warning("Test predictions not found — skipping thesis evaluation artifacts.")
            return None
        df = pd.read_parquet(pred_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        required = {"actual"}
        if not required.issubset(df.columns):
            logger.warning("Test predictions are missing required columns — skipping thesis evaluation artifacts.")
            return None
        return df

    def _load_comparison(self) -> pd.DataFrame:
        path = self.out_dir / "model_comparison.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    @staticmethod
    def _model_cols(df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if col not in _META_COLS]

    @staticmethod
    def _best_model_name(comparison_df: pd.DataFrame, pred_df: pd.DataFrame) -> str:
        if not comparison_df.empty:
            ml_df = comparison_df[comparison_df["model"] != "Naive Baseline"]
            if not ml_df.empty:
                best_idx = ml_df["DirectionalAccuracy"].idxmax()
                return str(ml_df.loc[best_idx, "model"])
        model_cols = [c for c in ResultsAnalyzer._model_cols(pred_df) if c != "Naive Baseline"]
        return model_cols[0] if model_cols else "Naive Baseline"

    def save_directional_accuracy_stats(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        actual = pred_df["actual"]
        for model in self._model_cols(pred_df):
            hits = int((np.sign(pred_df[model]) == np.sign(actual)).sum())
            lower, upper = _wilson_interval(hits, len(actual))
            rows.append(
                {
                    "model": model,
                    "n_obs": int(len(actual)),
                    "hits": hits,
                    "directional_accuracy": hits / len(actual),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
        stats_df = pd.DataFrame(rows)
        stats_df.to_csv(self.out_dir / "directional_accuracy_stats.csv", index=False)
        logger.success(f"Saved evaluation table → {self.out_dir / 'directional_accuracy_stats.csv'}")
        return stats_df

    def save_ticker_metrics(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        if "ticker" not in pred_df.columns:
            return pd.DataFrame()
        rows = []
        for ticker, part in pred_df.groupby("ticker"):
            for model in self._model_cols(pred_df):
                row = _metric_row(model, part["actual"], part[model])
                row["ticker"] = ticker
                rows.append(row)
        ticker_df = pd.DataFrame(rows)
        ticker_df.to_csv(self.out_dir / "ticker_model_metrics.csv", index=False)
        logger.success(f"Saved evaluation table → {self.out_dir / 'ticker_model_metrics.csv'}")
        return ticker_df

    def save_monthly_metrics(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        if "date" not in pred_df.columns:
            return pd.DataFrame()
        rows = []
        model_cols = self._model_cols(pred_df)
        df = pred_df.copy()
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        for month, part in df.groupby("month"):
            for model in model_cols:
                row = _metric_row(model, part["actual"], part[model])
                row["month"] = month
                rows.append(row)
        monthly_df = pd.DataFrame(rows)
        monthly_df.to_csv(self.out_dir / "monthly_model_metrics.csv", index=False)
        logger.success(f"Saved evaluation table → {self.out_dir / 'monthly_model_metrics.csv'}")
        return monthly_df

    def plot_directional_accuracy_ci(self, stats_df: pd.DataFrame) -> None:
        if stats_df.empty:
            return
        df = stats_df.copy()
        order = df["model"].tolist()
        yerr = np.vstack(
            [
                df["directional_accuracy"] - df["ci_lower"],
                df["ci_upper"] - df["directional_accuracy"],
            ]
        )
        colors = [PALETTE[2] if model == "Naive Baseline" else PALETTE[0] for model in order]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(order, df["directional_accuracy"], color=colors, yerr=yerr, capsize=5)
        ax.axhline(0.50, color="red", linestyle="--", linewidth=1, label="Random (50%)")
        ax.set_ylim(0.40, max(float(df["ci_upper"].max()) + 0.03, 0.55))
        ax.set_ylabel("Directional Accuracy")
        ax.set_title("Directional Accuracy with 95% Wilson Confidence Intervals")
        ax.legend(fontsize=9)
        for bar, value in zip(bars, df["directional_accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.1%}", ha="center", va="bottom")
        plt.xticks(rotation=20)
        self._save(fig, "directional_accuracy_ci")

    def plot_ticker_directional_accuracy(self, ticker_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
        if ticker_df.empty:
            return
        best_model = self._best_model_name(comparison_df, pd.DataFrame(columns=["actual"]))
        focus = ticker_df[ticker_df["model"].isin(["Naive Baseline", best_model])].copy()
        if focus.empty:
            return
        pivot = focus.pivot(index="ticker", columns="model", values="directional_accuracy")
        order = pivot[best_model].sort_values(ascending=False).index.tolist()
        pivot = pivot.loc[order]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(pivot.index))
        width = 0.36
        ax.bar(x - width / 2, pivot["Naive Baseline"], width=width, color=PALETTE[2], label="Naive Baseline")
        ax.bar(x + width / 2, pivot[best_model], width=width, color=PALETTE[0], label=best_model)
        ax.axhline(0.50, color="red", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=30)
        ax.set_ylabel("Directional Accuracy")
        ax.set_title(f"Ticker-Level Directional Accuracy: {best_model} vs Naive Baseline")
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save(fig, "ticker_directional_accuracy")

    def plot_monthly_directional_accuracy(self, monthly_df: pd.DataFrame) -> None:
        if monthly_df.empty:
            return
        fig, ax = plt.subplots(figsize=(11, 5))
        for idx, model in enumerate(monthly_df["model"].unique()):
            part = monthly_df[monthly_df["model"] == model].sort_values("month")
            ax.plot(part["month"], part["directional_accuracy"], marker="o", linewidth=2, label=model, color=PALETTE[idx % len(PALETTE)])
        ax.axhline(0.50, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel("Directional Accuracy")
        ax.set_xlabel("Test Month")
        ax.set_title("Monthly Directional Accuracy Stability on the Test Period")
        ax.legend(fontsize=9)
        plt.xticks(rotation=30)
        plt.tight_layout()
        self._save(fig, "monthly_directional_accuracy")

    def plot_residual_distribution(self, pred_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
        best_model = self._best_model_name(comparison_df, pred_df)
        residuals = pred_df[best_model] - pred_df["actual"]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(residuals, bins=40, kde=True, color=PALETTE[0], ax=ax)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Residual Distribution for {best_model}")
        ax.set_xlabel("Prediction Error (predicted - actual)")
        self._save(fig, "residual_distribution")

    def plot_direction_confusion(self, pred_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
        best_model = self._best_model_name(comparison_df, pred_df)
        actual_sign = np.where(pred_df["actual"] >= 0, "Up/Flat", "Down")
        pred_sign = np.where(pred_df[best_model] >= 0, "Up/Flat", "Down")
        confusion = pd.crosstab(
            pd.Series(actual_sign, name="Actual"),
            pd.Series(pred_sign, name="Predicted"),
            dropna=False,
        ).reindex(index=["Down", "Up/Flat"], columns=["Down", "Up/Flat"], fill_value=0)
        confusion.to_csv(self.out_dir / "direction_confusion.csv")
        logger.success(f"Saved evaluation table → {self.out_dir / 'direction_confusion.csv'}")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"Directional Confusion Matrix: {best_model}")
        self._save(fig, "direction_confusion")

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Saved plot → {path}")
