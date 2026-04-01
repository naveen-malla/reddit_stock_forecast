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
from src.models import BASELINE_MODEL_NAME
from src.plot_style import (
    FIG_DPI,
    POSITIVE,
    add_reference_line,
    add_title,
    apply_chart_style,
    display_model_name,
    model_color,
    style_axes,
)


apply_chart_style(font_scale=1.0)
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
            ml_df = comparison_df[comparison_df["model"] != BASELINE_MODEL_NAME]
            if not ml_df.empty:
                best_idx = ml_df["DirectionalAccuracy"].idxmax()
                return str(ml_df.loc[best_idx, "model"])
        model_cols = [c for c in ResultsAnalyzer._model_cols(pred_df) if c != BASELINE_MODEL_NAME]
        return model_cols[0] if model_cols else BASELINE_MODEL_NAME

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
        display_labels = [display_model_name(model) for model in order]
        yerr = np.vstack(
            [
                df["directional_accuracy"] - df["ci_lower"],
                df["ci_upper"] - df["directional_accuracy"],
            ]
        )
        colors = [model_color(model) for model in order]
        fig, ax = plt.subplots(figsize=(9.4, 5.4))
        bars = ax.bar(
            display_labels,
            df["directional_accuracy"],
            color=colors,
            yerr=yerr,
            capsize=5,
            edgecolor="none",
        )
        style_axes(ax)
        add_title(
            ax,
            "Directional Accuracy with 95% Confidence Intervals",
            "Wilson intervals on the held-out test period.",
        )
        add_reference_line(ax, 0.50, "Random expectation (50%)")
        ax.set_ylim(0.40, max(float(df["ci_upper"].max()) + 0.03, 0.55))
        ax.set_ylabel("Directional Accuracy")
        ax.legend(frameon=False, fontsize=9)
        for bar, value in zip(bars, df["directional_accuracy"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.tick_params(axis="x", rotation=0, pad=6)
        self._save(fig, "directional_accuracy_ci")

    def plot_ticker_directional_accuracy(self, ticker_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
        if ticker_df.empty:
            return
        best_model = self._best_model_name(comparison_df, pd.DataFrame(columns=["actual"]))
        focus = ticker_df[ticker_df["model"].isin([BASELINE_MODEL_NAME, best_model])].copy()
        if focus.empty:
            return
        pivot = focus.pivot(index="ticker", columns="model", values="directional_accuracy")
        order = pivot[best_model].sort_values(ascending=False).index.tolist()
        pivot = pivot.loc[order]
        fig, ax = plt.subplots(figsize=(11.3, 5.4))
        x = np.arange(len(pivot.index))
        width = 0.36
        ax.bar(
            x - width / 2,
            pivot[BASELINE_MODEL_NAME],
            width=width,
            color=model_color(BASELINE_MODEL_NAME),
            label=BASELINE_MODEL_NAME,
            edgecolor="none",
        )
        ax.bar(
            x + width / 2,
            pivot[best_model],
            width=width,
            color=model_color(best_model),
            label=best_model,
            edgecolor="none",
        )
        style_axes(ax)
        add_title(
            ax,
            f"Ticker-Level Directional Accuracy: {best_model} vs {BASELINE_MODEL_NAME}",
            "Each bar pair is computed on the same test-period observations.",
        )
        add_reference_line(ax, 0.50)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25)
        ax.set_ylabel("Directional Accuracy")
        ax.set_ylim(0.40, max(float(pivot.max().max()) + 0.04, 0.58))
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        fig.tight_layout()
        self._save(fig, "ticker_directional_accuracy")

    def plot_monthly_directional_accuracy(self, monthly_df: pd.DataFrame) -> None:
        if monthly_df.empty:
            return
        fig, ax = plt.subplots(figsize=(11.2, 5.4))
        style_axes(ax)
        add_title(
            ax,
            "Monthly Directional Accuracy Stability",
            "Performance is tracked month by month across the test period.",
        )
        for model in monthly_df["model"].unique():
            part = monthly_df[monthly_df["model"] == model].sort_values("month")
            ax.plot(
                part["month"],
                part["directional_accuracy"],
                marker="o",
                markersize=4.5,
                linewidth=2.1,
                label=model,
                color=model_color(model),
            )
        add_reference_line(ax, 0.50, "Random expectation (50%)")
        ax.set_ylabel("Directional Accuracy")
        ax.set_xlabel("Test Month")
        ax.set_ylim(0.40, max(float(monthly_df["directional_accuracy"].max()) + 0.05, 0.60))
        ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        self._save(fig, "monthly_directional_accuracy")

    def plot_residual_distribution(self, pred_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
        best_model = self._best_model_name(comparison_df, pred_df)
        residuals = pred_df[best_model] - pred_df["actual"]
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        sns.histplot(residuals, bins=36, kde=True, color=model_color(best_model), ax=ax, edgecolor="white", alpha=0.9)
        style_axes(ax)
        add_title(
            ax,
            f"Residual Distribution for {best_model}",
            "Residuals are computed as predicted return minus realised next-day return.",
        )
        add_reference_line(ax, 0.0)
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
        add_title(ax, f"Directional Confusion Matrix: {best_model}")
        self._save(fig, "direction_confusion")

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.out_dir / f"{name}.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Saved plot → {path}")
