"""
config.py  –  Central configuration for the Reddit Equity Forecast pipeline.
All paths, constants, and environment variables resolved here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    root_dir: Path = _ROOT
    data_raw: Path = _ROOT / "data" / "raw"
    data_processed: Path = _ROOT / "data" / "processed"
    models_dir: Path = _ROOT / "models"
    outputs_dir: Path = _ROOT / "outputs"

    # ── Reddit API ────────────────────────────────────────────────────────────
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    reddit_user_agent: str = field(
        default_factory=lambda: os.getenv("REDDIT_USER_AGENT", "RedditEquityForecast/2.0")
    )
    pushshift_base_url: str = field(
        default_factory=lambda: os.getenv("PUSHSHIFT_BASE_URL", "https://api.pushshift.io")
    )

    # ── Study window ──────────────────────────────────────────────────────────
    start_date: str = field(default_factory=lambda: os.getenv("START_DATE", "2021-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("END_DATE", "2025-12-31"))

    # ── Ticker selection ──────────────────────────────────────────────────────
    top_n_tickers: int = field(
        default_factory=lambda: int(os.getenv("TOP_N_TICKERS", "10"))
    )

    subreddits: list = field(
        default_factory=lambda: [
            # General market subreddits
            "wallstreetbets",
            "stocks",
            "investing",
            "StockMarket",
            "options",
            # Company-specific subreddits — direct discussion = stronger signal
            "NVDA",
            "nvidia",
            "TSLA",
            "teslainvestorsclub",
            "apple",
            "AAPL",
            "amazon",
            "AMZN",
            "netflix",
            "AMD_Stock",
            "Palantir",
            "NIO_Stock",
        ]
    )

    candidate_tickers: list = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "AMZN", "NVDA", "TSLA",
            "GOOGL", "META", "AMD", "SPY", "QQQ",
            "BABA", "NFLX", "BAC", "INTC", "GME",
            "AMC", "PLTR", "RIVN", "NIO", "LCID",
        ]
    )

    # ── Modelling ─────────────────────────────────────────────────────────────
    test_size: float = 0.2
    random_state: int = 42
    target_col: str = "next_day_close_pct"

    # Sentiment look-back windows (trading days)
    sentiment_windows: list = field(default_factory=lambda: [1, 3, 7])

    # Minimum rows required from Reddit collection before warning
    min_reddit_rows: int = 500

    def __post_init__(self):
        for p in (self.data_raw, self.data_processed, self.models_dir, self.outputs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def validate(self):
        missing = []
        if not self.reddit_client_id:
            missing.append("REDDIT_CLIENT_ID")
        if not self.reddit_client_secret:
            missing.append("REDDIT_CLIENT_SECRET")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {missing}\n"
                "Copy .env.example → .env and fill in your credentials."
            )


cfg = Config()
