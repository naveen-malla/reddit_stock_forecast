"""
src/visualiser.py
─────────────────
Generates all output plots:
  1. Volume ranking bar chart
  2. Sentiment time series per ticker
  3. Model comparison (MAE / RMSE / DA) including the benchmark model
  4. Feature importance (XGBoost + LightGBM)
  5. Predicted vs actual scatter
  6. Interactive Plotly HTML charts

FIXES vs v1:
  - Plotly actually used (was imported but never called in v1)
  - Baseline model included in comparison charts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from loguru import logger
from src.models import BASELINE_MODEL_NAME

sns.set_theme(style="darkgrid", context="notebook", font_scale=1.1)
PALETTE = sns.color_palette("tab10")
FIG_DPI = 150
_PREDICTION_META_COLS = {"ticker", "date", "actual"}


class Visualiser:
    def __init__(self):
        self.out = cfg.outputs_dir

    def plot_all(self) -> None:
        self.plot_volume_ranking()
        self.plot_sentiment_timeseries()
        self.plot_model_comparison()
        self.plot_feature_importance()
        self.plot_predictions()
        self.plot_interactive_sentiment()
        self.plot_interactive_predictions()

    # ── 1. Volume ranking ─────────────────────────────────────────────────────

    def plot_volume_ranking(self) -> None:
        p = cfg.data_processed / "volume_ranking.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p).head(cfg.top_n_tickers)

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(df["ticker"][::-1], df["avg_daily_volume"][::-1] / 1e6,
                       color=PALETTE[:len(df)])
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}M"))
        ax.set_xlabel("Avg Daily Volume (millions of shares)")
        ax.set_title(f"Top-{cfg.top_n_tickers} Tickers by 90-Day Average Trading Volume", pad=12)
        for bar, val in zip(bars, df["avg_daily_volume"][::-1] / 1e6):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}M", va="center", fontsize=9)
        plt.tight_layout()
        self._save(fig, "volume_ranking")

    # ── 2. Sentiment time series ──────────────────────────────────────────────

    def plot_sentiment_timeseries(self) -> None:
        p = cfg.data_processed / "sentiment_daily.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        tickers = df["ticker"].unique()[:6]

        fig, axes = plt.subplots(len(tickers), 1, figsize=(13, 3 * len(tickers)), sharex=True)
        if len(tickers) == 1:
            axes = [axes]

        for ax, ticker in zip(axes, tickers):
            sub = df[df["ticker"] == ticker].sort_values("date")
            ax.fill_between(sub["date"], sub["vader_mean"].clip(-1, 1), alpha=0.4, color=PALETTE[0])
            ax.plot(sub["date"], sub["vader_mean"].rolling(7).mean(),
                    color=PALETTE[0], linewidth=1.4, label="VADER 7-day MA")
            if "finbert_mean" in sub.columns and sub["finbert_mean"].notna().any():
                ax.plot(sub["date"], sub["finbert_mean"].rolling(7).mean(),
                        color=PALETTE[1], linewidth=1.4, label="FinBERT 7-day MA")
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
            ax.set_ylabel("Sentiment", fontsize=9)
            ax.set_title(ticker, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")

        fig.suptitle("Reddit Sentiment Time Series by Ticker", fontsize=13, y=1.01)
        plt.tight_layout()
        self._save(fig, "sentiment_timeseries")

    # ── 3. Model comparison ───────────────────────────────────────────────────

    def plot_model_comparison(self) -> None:
        p = cfg.outputs_dir / "model_comparison.csv"
        if not p.exists():
            return
        df = pd.read_csv(p)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        metrics = [
            ("MAE", "Mean Absolute Error"),
            ("RMSE", "Root Mean Squared Error"),
            ("DirectionalAccuracy", "Directional Accuracy"),
        ]

        colors = [PALETTE[2] if m == BASELINE_MODEL_NAME else PALETTE[0]
                  for m in df["model"]]

        for ax, (col, label) in zip(axes, metrics):
            vals = df[col].tolist()
            bars = ax.bar(df["model"], vals, color=colors, edgecolor="white", linewidth=0.8)
            ax.set_title(label, fontsize=10, pad=8)
            ax.set_ylabel(col)
            for bar, val in zip(bars, vals):
                fmt = f"{val:.1%}" if col == "DirectionalAccuracy" else f"{val:.5f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        fmt, ha="center", va="bottom", fontsize=9)
            if col == "DirectionalAccuracy":
                ax.axhline(0.5, color="red", linewidth=1, linestyle="--", label="Random (50%)")
                ax.legend(fontsize=8)
            ax.set_ylim(0, max(vals) * 1.2)
            ax.tick_params(axis="x", rotation=20)

        fig.suptitle("Model Comparison — Test Set (gray = persistence benchmark)", fontsize=12, y=1.02)
        plt.tight_layout()
        self._save(fig, "model_comparison")

    # ── 4. Feature importance ─────────────────────────────────────────────────

    def plot_feature_importance(self, top_n: int = 20) -> None:
        available = [
            m for m in ["xgboost", "lightgbm"]
            if (cfg.outputs_dir / f"{m}_feature_importance.csv").exists()
        ]
        if not available:
            logger.warning("No feature importance CSVs found — skipping plot.")
            return
        fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 6))
        if len(available) == 1:
            axes = [axes]
        for ax, model_name in zip(axes, available):
            p = cfg.outputs_dir / f"{model_name}_feature_importance.csv"
            df = pd.read_csv(p).head(top_n).iloc[::-1]
            ax.barh(df["feature"], df["importance"], color=PALETTE[0])
            ax.set_title(f"{model_name.capitalize()} — Top {top_n} Features", fontsize=11)
            ax.set_xlabel("Feature Importance")

        fig.suptitle("Feature Importance Comparison", fontsize=13, y=1.01)
        plt.tight_layout()
        self._save(fig, "feature_importance")

    # ── 5. Predicted vs actual scatter ────────────────────────────────────────

    def plot_predictions(self) -> None:
        pred_path = cfg.outputs_dir / "test_predictions.parquet"
        if not pred_path.exists():
            return
        df = pd.read_parquet(pred_path)

        ml_models = [c for c in df.columns if c not in (_PREDICTION_META_COLS | {BASELINE_MODEL_NAME})]
        fig, axes = plt.subplots(1, len(ml_models), figsize=(7 * len(ml_models), 5))
        if len(ml_models) == 1:
            axes = [axes]

        for ax, model_name in zip(axes, ml_models):
            if model_name not in df.columns:
                continue
            ax.scatter(df["actual"], df[model_name], alpha=0.3, s=8,
                       color=PALETTE[0], edgecolors="none")
            lim = max(abs(df["actual"].quantile(0.01)),
                      abs(df["actual"].quantile(0.99))) * 1.1
            ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Perfect prediction")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Actual next-day return")
            ax.set_ylabel("Predicted next-day return")
            ax.set_title(model_name, fontsize=11)
            ax.legend(fontsize=8)

        fig.suptitle("Actual vs Predicted Next-Day Returns (Test Set)", fontsize=12, y=1.01)
        plt.tight_layout()
        self._save(fig, "predictions_scatter")

    # ── 6. Interactive Plotly: sentiment over time ────────────────────────────

    def plot_interactive_sentiment(self) -> None:
        p = cfg.data_processed / "sentiment_daily.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure()
        for ticker in df["ticker"].unique():
            sub = df[df["ticker"] == ticker].sort_values("date")
            fig.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["vader_mean"].rolling(7, min_periods=1).mean(),
                name=ticker,
                mode="lines",
                visible="legendonly" if ticker != df["ticker"].unique()[0] else True,
            ))

        fig.update_layout(
            title="Reddit VADER Sentiment — 7-day rolling average (Interactive)",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )
        fig.write_html(str(self.out / "sentiment_interactive.html"))
        logger.success(f"Saved interactive plot → {self.out / 'sentiment_interactive.html'}")

    # ── 7. Interactive Plotly: model predictions ──────────────────────────────

    def plot_interactive_predictions(self) -> None:
        pred_path = cfg.outputs_dir / "test_predictions.parquet"
        if not pred_path.exists():
            return
        df = pd.read_parquet(pred_path).reset_index(drop=True)

        ml_models = [c for c in df.columns if c not in _PREDICTION_META_COLS]
        fig = make_subplots(
            rows=1, cols=len(ml_models),
            subplot_titles=ml_models,
        )

        for i, model_name in enumerate(ml_models, 1):
            if model_name not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df["actual"], y=df[model_name],
                    mode="markers",
                    marker=dict(size=4, opacity=0.4),
                    name=model_name,
                ),
                row=1, col=i,
            )
            lim = float(df["actual"].abs().quantile(0.99)) * 1.1
            fig.add_trace(
                go.Scatter(
                    x=[-lim, lim], y=[-lim, lim],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Perfect",
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )

        fig.update_layout(
            title="Actual vs Predicted Returns — Test Set (Interactive)",
            height=500,
            template="plotly_white",
        )
        fig.write_html(str(self.out / "predictions_interactive.html"))
        logger.success(f"Saved interactive plot → {self.out / 'predictions_interactive.html'}")

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.out / f"{name}.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Saved plot → {path}")
