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
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from loguru import logger
from src.models import BASELINE_MODEL_NAME
from src.plot_style import (
    FIG_DPI,
    FINBERT,
    PANEL,
    REFERENCE,
    SENTIMENT,
    SENTIMENT_FILL,
    XGBOOST,
    add_reference_line,
    add_figure_heading,
    add_title,
    apply_chart_style,
    apply_plotly_layout,
    display_model_name,
    model_color,
    style_axes,
)

apply_chart_style(font_scale=1.0)
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
        df = df.sort_values("avg_daily_volume", ascending=True).reset_index(drop=True)
        volumes_m = df["avg_daily_volume"] / 1e6

        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        fig.patch.set_facecolor(PANEL)
        colors = sns.light_palette(XGBOOST, n_colors=len(df) + 2)[2:]
        bars = ax.barh(df["ticker"], volumes_m, color=colors, edgecolor="none", height=0.72)
        style_axes(ax, x_grid=True, y_grid=False)
        add_title(
            ax,
            f"Top {cfg.top_n_tickers} Tickers by 90-Day Average Trading Volume",
            None,
        )
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}M"))
        ax.set_xlabel("Average daily volume (millions of shares)")
        ax.set_ylabel("")
        max_val = float(volumes_m.max())
        ax.set_xlim(0, max_val * 1.18)
        for bar, val in zip(bars, volumes_m):
            ax.text(
                val + max_val * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}M",
                va="center",
                fontsize=9,
            )
        fig.tight_layout()
        self._save(fig, "volume_ranking")

    # ── 2. Sentiment time series ──────────────────────────────────────────────

    def plot_sentiment_timeseries(self) -> None:
        p = cfg.data_processed / "sentiment_daily.parquet"
        if not p.exists():
            return
        df = pd.read_parquet(p)
        tickers = df["ticker"].unique()[:6]

        fig, axes = plt.subplots(len(tickers), 1, figsize=(13, 2.9 * len(tickers)), sharex=True)
        if len(tickers) == 1:
            axes = [axes]

        for ax, ticker in zip(axes, tickers):
            sub = df[df["ticker"] == ticker].sort_values("date")
            style_axes(ax, x_grid=False, y_grid=True)
            ax.fill_between(
                sub["date"],
                sub["vader_mean"].clip(-1, 1),
                color=SENTIMENT_FILL,
                alpha=0.55,
                linewidth=0,
            )
            ax.plot(
                sub["date"],
                sub["vader_mean"].rolling(7).mean(),
                color=SENTIMENT,
                linewidth=2.0,
                label="VADER 7-day mean",
            )
            if "finbert_mean" in sub.columns and sub["finbert_mean"].notna().any():
                ax.plot(
                    sub["date"],
                    sub["finbert_mean"].rolling(7).mean(),
                    color=FINBERT,
                    linewidth=1.8,
                    label="FinBERT 7-day mean",
                )
            add_reference_line(ax, 0.0)
            ax.set_ylabel("Sentiment", fontsize=9)
            ax.set_title(ticker, loc="left", fontsize=10.5, fontweight="bold", pad=8)
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")

        add_figure_heading(fig, "Reddit Sentiment Time Series by Ticker", "Daily ticker sentiment with smoothed VADER scores.")
        fig.tight_layout(rect=[0, 0, 1, 0.91])
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
        display_labels = [display_model_name(model) for model in df["model"]]
        colors = [model_color(model) for model in df["model"]]

        for ax, (col, label) in zip(axes, metrics):
            vals = df[col].tolist()
            bars = ax.bar(display_labels, vals, color=colors, edgecolor="none", width=0.72)
            style_axes(ax)
            add_title(ax, label)
            ax.set_ylabel(label if col == "DirectionalAccuracy" else col)
            for bar, val in zip(bars, vals):
                fmt = f"{val:.1%}" if col == "DirectionalAccuracy" else f"{val:.5f}"
                offset = max(vals) * (0.018 if col != "DirectionalAccuracy" else 0.01)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            if col == "DirectionalAccuracy":
                add_reference_line(ax, 0.5, "Random expectation (50%)")
                ax.legend(frameon=False, fontsize=8.5, loc="upper right")
                ax.set_ylim(0.0, max(vals) * 1.18)
            else:
                ax.set_ylim(0.0, max(vals) * 1.22)
            ax.tick_params(axis="x", rotation=0, pad=6)

        add_figure_heading(fig, "Model Comparison", "MAE, RMSE, and directional accuracy across the benchmark and model set.")
        fig.tight_layout(rect=[0, 0, 1, 0.91])
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
        fig, axes = plt.subplots(1, len(available), figsize=(8.5 * len(available), 6))
        if len(available) == 1:
            axes = [axes]
        for ax, model_name in zip(axes, available):
            p = cfg.outputs_dir / f"{model_name}_feature_importance.csv"
            df = pd.read_csv(p).head(top_n).iloc[::-1]
            ax.barh(df["feature"], df["importance"], color=model_color(model_name.replace("xgboost", "XGBoost").replace("lightgbm", "LightGBM")))
            style_axes(ax, x_grid=True, y_grid=False)
            add_title(ax, f"{model_name.capitalize()} Top {top_n} Features")
            ax.set_xlabel("Feature Importance")

        add_figure_heading(fig, "Feature Importance Comparison")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        self._save(fig, "feature_importance")

    # ── 5. Predicted vs actual scatter ────────────────────────────────────────

    def plot_predictions(self) -> None:
        pred_path = cfg.outputs_dir / "test_predictions.parquet"
        if not pred_path.exists():
            return
        df = pd.read_parquet(pred_path)

        ml_models = [c for c in df.columns if c not in (_PREDICTION_META_COLS | {BASELINE_MODEL_NAME})]
        fig, axes = plt.subplots(1, len(ml_models), figsize=(7 * len(ml_models), 5.2))
        if len(ml_models) == 1:
            axes = [axes]

        for ax, model_name in zip(axes, ml_models):
            if model_name not in df.columns:
                continue
            style_axes(ax, x_grid=True, y_grid=True)
            ax.scatter(
                df["actual"],
                df[model_name],
                alpha=0.28,
                s=12,
                color=model_color(model_name),
                edgecolors="none",
            )
            lim = max(abs(df["actual"].quantile(0.01)),
                      abs(df["actual"].quantile(0.99))) * 1.1
            ax.plot(
                [-lim, lim],
                [-lim, lim],
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                color=REFERENCE,
                label="Perfect agreement",
            )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Actual next-day return")
            ax.set_ylabel("Predicted next-day return")
            add_title(ax, model_name)
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")

        add_figure_heading(fig, "Predicted vs Actual Next-Day Returns")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
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

        apply_plotly_layout(
            fig,
            "Reddit VADER Sentiment — 7-Day Rolling Mean",
            "Date",
            "Sentiment score",
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
                    marker=dict(size=4, opacity=0.42, color=model_color(model_name)),
                    name=model_name,
                ),
                row=1, col=i,
            )
            lim = float(df["actual"].abs().quantile(0.99)) * 1.1
            fig.add_trace(
                go.Scatter(
                    x=[-lim, lim], y=[-lim, lim],
                    mode="lines",
                    line=dict(color=REFERENCE, dash="dash"),
                    name="Perfect",
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )

        apply_plotly_layout(
            fig,
            "Actual vs Predicted Returns — Interactive View",
            "Actual next-day return",
            "Predicted next-day return",
            height=520,
        )
        fig.write_html(str(self.out / "predictions_interactive.html"))
        logger.success(f"Saved interactive plot → {self.out / 'predictions_interactive.html'}")

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = self.out / f"{name}.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Saved plot → {path}")
