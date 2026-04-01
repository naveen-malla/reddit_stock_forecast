import unittest
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from config import cfg
from src.dataset_builder import DatasetBuilder
from src.market_data import MarketDataFetcher
from src.reddit_collector import RedditCollector


class PipelineIntegrityTests(unittest.TestCase):
    def test_clip_to_window_respects_bounds(self):
        start_dt = date(2021, 1, 1)
        end_dt = date(2021, 1, 3)
        rows = [
            {"id": "before", "created_utc": int(datetime(2020, 12, 31, tzinfo=timezone.utc).timestamp())},
            {"id": "inside", "created_utc": int(datetime(2021, 1, 2, tzinfo=timezone.utc).timestamp())},
            {"id": "after", "created_utc": int(datetime(2021, 1, 4, tzinfo=timezone.utc).timestamp())},
        ]
        df = pd.DataFrame(rows)

        clipped = RedditCollector._clip_to_window(df, start_dt, end_dt)

        self.assertEqual(clipped["id"].tolist(), ["inside"])

    def test_merge_lags_sentiment_by_one_day(self):
        dates = pd.bdate_range("2021-01-04", periods=80)
        raw = pd.DataFrame(
            {
                "date": dates,
                "open": [100 + i for i in range(len(dates))],
                "high": [101 + i for i in range(len(dates))],
                "low": [99 + i for i in range(len(dates))],
                "close": [100.5 + i for i in range(len(dates))],
                "volume": [1_000_000 + 1000 * i for i in range(len(dates))],
                "ticker": "AAPL",
            }
        )
        market_df = MarketDataFetcher._engineer(raw)
        source_date = next(
            dt_value
            for dt_value in market_df["date"]
            if dt_value + pd.Timedelta(days=1) in set(market_df["date"])
        )
        target_date = source_date + pd.Timedelta(days=1)
        self.assertIn(target_date, set(market_df["date"]))

        sentiment_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [source_date],
                "mention_count": [5],
                "vader_mean": [0.75],
                "vader_weighted_mean": [0.80],
                "vader_pos_ratio": [1.0],
                "vader_w3d_mean": [0.70],
                "vader_w7d_mean": [0.68],
            }
        )

        builder = DatasetBuilder()
        merged = builder._merge(market_df=market_df, sentiment_df=sentiment_df)
        lagged_row = merged.loc[(merged["ticker"] == "AAPL") & (merged["date"] == target_date)].iloc[0]

        self.assertEqual(lagged_row["mention_count"], 5)
        self.assertAlmostEqual(lagged_row["vader_mean"], 0.75)

    def test_split_is_chronological_without_overlap(self):
        dates = pd.bdate_range("2021-01-04", periods=30)
        rows = []
        for ticker in ("AAPL", "TSLA"):
            for idx, dt_value in enumerate(dates):
                rows.append(
                    {
                        "ticker": ticker,
                        "date": dt_value,
                        "feature_a": float(idx),
                        "next_day_close": 100.0 + idx,
                        cfg.target_col: 0.01 if idx % 2 == 0 else -0.01,
                    }
                )
        df = pd.DataFrame(rows)

        builder = DatasetBuilder()
        X_train, X_val, X_test, _, _, _, feature_cols = builder._split(df)

        self.assertIn("feature_a", feature_cols)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
        self.assertLess(builder.train_meta["date"].max(), builder.val_meta["date"].min())
        self.assertLess(builder.val_meta["date"].max(), builder.test_meta["date"].min())


if __name__ == "__main__":
    unittest.main()
