"""
Shared plotting style helpers for a consistent project-wide visual system.
"""

from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns


BACKGROUND = "#F4F7FB"
PANEL = "#FFFFFF"
GRID = "#D9E2EC"
TEXT = "#1F2937"
MUTED = "#5B6B7C"
REFERENCE = "#B85C38"
BENCHMARK = "#8A97A8"
XGBOOST = "#2F6EA7"
XGBOOST_CALIBRATED = "#163D6B"
LIGHTGBM = "#5C8F7A"
SENTIMENT = "#2F6EA7"
SENTIMENT_FILL = "#BFD4EA"
FINBERT = "#5C8F7A"
POSITIVE = "#2F6EA7"
NEGATIVE = "#B85C38"

MODEL_COLORS = {
    "Persistence Benchmark": BENCHMARK,
    "XGBoost": XGBOOST,
    "XGBoost Calibrated": XGBOOST_CALIBRATED,
    "LightGBM": LIGHTGBM,
}

MODEL_LABELS = {
    "Persistence Benchmark": "Persistence\nBenchmark",
    "XGBoost": "XGBoost",
    "XGBoost Calibrated": "XGBoost\nCalibrated",
    "LightGBM": "LightGBM",
}

COLORWAY = [XGBOOST, XGBOOST_CALIBRATED, LIGHTGBM, BENCHMARK]
FIG_DPI = 180


def apply_chart_style(font_scale: float = 1.0) -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=font_scale,
        rc={
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def style_axes(ax: plt.Axes, *, x_grid: bool = False, y_grid: bool = True) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(False)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.9)
    if x_grid:
        ax.grid(axis="x", color=GRID, linewidth=0.9, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.9)


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=18)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=MUTED,
        )


def add_reference_line(ax: plt.Axes, y: float, label: str | None = None) -> None:
    ax.axhline(y, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.95, label=label)


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, XGBOOST)


def display_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, wrap_label(model, 14))


def wrap_label(value: str, width: int = 14) -> str:
    return fill(str(value), width=width)


def apply_plotly_layout(fig, title: str, xaxis_title: str, yaxis_title: str, height: int = 500) -> None:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        colorway=COLORWAY,
        font={"family": "Arial", "color": TEXT, "size": 13},
        height=height,
        margin={"l": 70, "r": 30, "t": 70, "b": 70},
    )
