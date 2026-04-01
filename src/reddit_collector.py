"""
src/reddit_collector.py
───────────────────────
Real Reddit data collection only. No fake data.

Sources (tried in order):
  1. Arctic Shift API  — free, no auth, real data 2020–present
  2. PullPush.io API   — backup mirror, real data 2020–present
  3. PRAW              — Reddit official API, optional recent-tail supplement

If data is sparse for some periods, that is reported honestly.
No fake rows are ever generated.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import praw
import requests
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg

_TICKER_RE = re.compile(r"(?<!\w)(\$?[A-Z]{2,5})(?!\w)")

_STOPWORDS = {
    "I","A","AN","AS","AT","BE","BY","DO","GO","IF","IN","IS","IT","ME","MY",
    "NO","OF","OK","ON","OR","SO","TO","UP","US","WE","AM","ARE","BUT","FOR",
    "GET","GOT","HAD","HAS","HER","HIM","HIS","HOW","ITS","LET","MAY","NEW",
    "NOT","NOW","OLD","OUR","OUT","OWN","PUT","SAY","SHE","THE","TOO","TWO",
    "USE","WAS","WAY","WHO","WHY","YES","YET","YOU","ALL","AND","ANY","CAN",
    "DID","END","FEW","HIT","HOT","LOT","LOW","MAN","MEN","OFF","ONE","SET",
    "SIT","SIX","TAX","TEN","TOP","TRY","WIN","EDIT","TLDR","YOLO","HODL",
    "FOMO","MOON","BEAR","BULL","CALL","PUTS","GAIN","LOSS","RICH","POOR",
    "HOLD","SELL","CASH","DEBT","SAFE","RISK","HUGE","NEXT","LAST","ONLY",
    "OVER","SOME","THEN","THIS","THAT","WHEN","WITH","YOUR","HAVE","FROM",
    "BEEN","WILL","ALSO","INTO","THAN","THEY","WHAT","WERE","JUST","LIKE",
    "MAKE","MORE","MOST","MUCH","MUST","NEED","SAID","SAME","SUCH","TAKE",
    "THEM","WELL","EACH","LONG","MANY","VERY","TIME","TRUE","OPEN","PART",
    "PLAN","SHOW","STAY","STOP","SURE","TALK","TELL","TEST","USED","WAIT",
    "WANT","WEEK","WORD","WORK","YEAR",
}
_BOT_RE = re.compile(r"bot|auto|mod|spam|alert|notify|feed|rss", re.IGNORECASE)
_ARCTIC  = "https://arctic-shift.photon-reddit.com/api"
_PULLPUSH = "https://api.pullpush.io/reddit/search"
_MAX_RETRIES = 3


class RedditCollector:

    def __init__(self):
        self.reddit = None
        if cfg.has_reddit_credentials:
            self.reddit = praw.Reddit(
                client_id=cfg.reddit_client_id,
                client_secret=cfg.reddit_client_secret,
                user_agent=cfg.reddit_user_agent,
            )
        self.out_dir = cfg.data_raw / "reddit"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry-point ────────────────────────────────────────────────────

    def run(
        self,
        tickers: List[str],
        start: Optional[str] = None,
        end:   Optional[str] = None,
        force: bool = False,
    ) -> pd.DataFrame:
        start      = start or cfg.start_date
        end        = end   or cfg.end_date
        ticker_set = {t.upper().lstrip("$") for t in tickers}
        start_dt   = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt     = datetime.strptime(end,   "%Y-%m-%d").date()
        cutoff     = (datetime.now(timezone.utc) - timedelta(days=365)).date()

        logger.info(f"Collecting REAL Reddit data: {start} → {end}")

        frames: List[pd.DataFrame] = []

        # ── Historical: Arctic Shift then PullPush ────────────────────────────
        hist_end = end_dt
        logger.info(f"\n[Arctic Shift] {start_dt} → {hist_end}")
        arctic_ok = False
        for sub in cfg.subreddits:
            df = self._arctic_pull(sub, ticker_set, str(start_dt), str(hist_end), force)
            if df is not None and len(df):
                frames.append(df)
                arctic_ok = True

        if not arctic_ok:
            logger.warning("Arctic Shift returned no data — trying PullPush.io …")
            for sub in cfg.subreddits:
                df = self._pullpush_pull(sub, ticker_set, str(start_dt), str(hist_end), force)
                if df is not None and len(df):
                    frames.append(df)

        # ── Recent: PRAW ──────────────────────────────────────────────────────
        # PRAW only fetches limit=1000, which cannot fill large gaps, so it serves
        # purely as an optional recent-tail supplement inside the configured window.
        if end_dt >= cutoff and self.reddit is not None:
            praw_start = max(start_dt, cutoff)
            logger.info(f"\n[PRAW] {praw_start} → {end_dt} supplementing tail")
            for sub in cfg.subreddits:
                df = self._praw_pull(sub, ticker_set, str(praw_start), str(end_dt), force)
                if df is not None and len(df):
                    frames.append(df)
        elif end_dt >= cutoff and self.reddit is None:
            logger.info("Skipping PRAW tail supplement because Reddit API credentials are not configured.")

        # ── Combine ───────────────────────────────────────────────────────────
        if not frames:
            logger.warning("No Reddit data collected. Pipeline continues with market features only.")
            return pd.DataFrame()

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["id"])
            .sort_values("created_utc")
            .reset_index(drop=True)
        )
        combined = self._clip_to_window(combined, start_dt, end_dt)
        combined = self._quality_filter(combined)

        # ── Honest coverage report ────────────────────────────────────────────
        self._report_coverage(combined, list(ticker_set), start_dt, end_dt)

        out_path = cfg.data_raw / "reddit_raw.parquet"
        combined.to_parquet(out_path, index=False)
        logger.success(f"Saved {len(combined):,} real rows → {out_path}")
        return combined

    # ── Arctic Shift ──────────────────────────────────────────────────────────

    def _arctic_pull(self, subreddit, ticker_set, start, end, force):
        cache = self.out_dir / f"arctic_{subreddit}_{start}_{end}.parquet"
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        logger.info(f"  Arctic Shift → r/{subreddit}")
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        rows = []
        cursor = start_dt

        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=30), end_dt)
            for kind in ("posts", "comments"):
                result = self._fetch(
                    url=f"{_ARCTIC}/{kind}/search",
                    params={
                        "subreddit": subreddit,
                        "after":  cursor.strftime("%Y-%m-%dT%H:%M:%S"),
                        "before": chunk_end.strftime("%Y-%m-%dT%H:%M:%S"),
                        "limit": 100, "sort": "asc",
                    },
                    ticker_set=ticker_set,
                    is_comment=(kind == "comments"),
                    source="arctic_shift",
                )
                if result:
                    rows.extend(result)
            cursor = chunk_end
            time.sleep(0.5)

        df = self._to_df(rows)
        if len(df):
            df.to_parquet(cache, index=False)
        return df

    # ── PullPush ──────────────────────────────────────────────────────────────

    def _pullpush_pull(self, subreddit, ticker_set, start, end, force):
        cache = self.out_dir / f"pullpush_{subreddit}_{start}_{end}.parquet"
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        logger.info(f"  PullPush → r/{subreddit}")
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        rows = []
        cursor = start_dt

        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=30), end_dt)
            for kind in ("submission", "comment"):
                result = self._fetch(
                    url=f"{_PULLPUSH}/{kind}/",
                    params={
                        "subreddit": subreddit,
                        "after":  int(cursor.timestamp()),
                        "before": int(chunk_end.timestamp()),
                        "size": 100, "sort": "asc",
                    },
                    ticker_set=ticker_set,
                    is_comment=(kind == "comment"),
                    source="pullpush",
                    data_key="data",
                )
                if result:
                    rows.extend(result)
            cursor = chunk_end
            time.sleep(0.5)

        df = self._to_df(rows)
        if len(df):
            df.to_parquet(cache, index=False)
        return df

    # ── PRAW ──────────────────────────────────────────────────────────────────

    def _praw_pull(self, subreddit, ticker_set, start, end, force):
        cache = self.out_dir / f"praw_{subreddit}_{start}_{end}.parquet"
        if cache.exists() and not force:
            return pd.read_parquet(cache)

        start_ts = int(
            datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        )
        end_ts = int(
            (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1))
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        rows = []
        sub = self.reddit.subreddit(subreddit)
        logger.info(f"  PRAW → r/{subreddit}")
        try:
            for listing in (sub.hot(limit=1000), sub.new(limit=1000), sub.top("year", limit=500)):
                for post in listing:
                    created_utc = int(getattr(post, "created_utc", 0))
                    if created_utc < start_ts or created_utc >= end_ts:
                        continue
                    mentions = self._find_tickers(post.title + " " + (post.selftext or ""), ticker_set)
                    if not mentions:
                        continue
                    rows.append({
                        "id": post.id, "subreddit": str(post.subreddit),
                        "type": "post", "author": str(post.author),
                        "created_utc": created_utc,
                        "title": post.title or "", "body": post.selftext or "",
                        "score": post.score,
                        "ticker_mentions": ",".join(sorted(mentions)),
                        "raw_text": f"{post.title} {post.selftext}".strip(),
                        "source": "praw",
                    })
                    try:
                        post.comments.replace_more(limit=0)
                        for c in list(post.comments)[:50]:
                            comment_created = int(getattr(c, "created_utc", 0))
                            if comment_created < start_ts or comment_created >= end_ts:
                                continue
                            cm = self._find_tickers(c.body or "", ticker_set)
                            if cm:
                                rows.append({
                                    "id": c.id, "subreddit": subreddit,
                                    "type": "comment", "author": str(c.author),
                                    "created_utc": comment_created,
                                    "title": "", "body": c.body or "",
                                    "score": c.score,
                                    "ticker_mentions": ",".join(sorted(cm)),
                                    "raw_text": c.body or "",
                                    "source": "praw",
                                })
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"PRAW error on r/{subreddit}: {e}")

        df = self._to_df(rows)
        if len(df):
            df = self._clip_to_window(
                df,
                datetime.strptime(start, "%Y-%m-%d").date(),
                datetime.strptime(end, "%Y-%m-%d").date(),
            )
            df.to_parquet(cache, index=False)
        return df

    # ── Fetch helper with retry ───────────────────────────────────────────────

    def _fetch(self, url, params, ticker_set, is_comment, source, data_key="data"):
        rows = []
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json().get(data_key, [])
                if not data:
                    return rows
                for item in data:
                    text  = item.get("body", "") if is_comment else f"{item.get('title','')} {item.get('selftext','')}"
                    mentions = self._find_tickers(text, ticker_set)
                    if not mentions:
                        continue
                    rows.append({
                        "id":             item.get("id", ""),
                        "subreddit":      item.get("subreddit", ""),
                        "type":           "comment" if is_comment else "post",
                        "author":         item.get("author", "[unknown]"),
                        "created_utc":    item.get("created_utc", 0),
                        "title":          "" if is_comment else item.get("title", ""),
                        "body":           text,
                        "score":          item.get("score", 0),
                        "ticker_mentions": ",".join(sorted(mentions)),
                        "raw_text":       text.strip(),
                        "source":         source,
                    })
                return rows
            except requests.exceptions.ConnectionError:
                logger.warning(f"  {source} unreachable.")
                return None
            except requests.exceptions.HTTPError as e:
                logger.warning(f"  {source} HTTP error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
                if attempt == _MAX_RETRIES - 1:
                    return rows or None
            except Exception as e:
                logger.warning(f"  {source} error: {e}")
                return None
        return rows or None

    # ── Quality filter ────────────────────────────────────────────────────────

    @staticmethod
    def _quality_filter(df):
        before = len(df)
        df = df[~df["author"].str.contains(_BOT_RE, na=False)]
        df = df[df["raw_text"].str.strip().str.len() > 10]
        df = df[~df["raw_text"].str.lower().isin(["[deleted]", "[removed]"])]
        logger.info(f"Quality filter: {before:,} → {len(df):,} rows")
        return df.reset_index(drop=True)

    @staticmethod
    def _clip_to_window(df: pd.DataFrame, start_dt, end_dt) -> pd.DataFrame:
        if df.empty:
            return df
        start_ts = int(datetime.combine(start_dt, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end_ts = int(
            datetime.combine(end_dt + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp()
        )
        clipped = df.copy()
        clipped["created_utc"] = pd.to_numeric(clipped["created_utc"], errors="coerce")
        clipped = clipped[
            clipped["created_utc"].between(start_ts, end_ts - 1, inclusive="both")
        ]
        return clipped.reset_index(drop=True)

    # ── Honest coverage report ────────────────────────────────────────────────

    @staticmethod
    def _report_coverage(df, tickers, start_dt, end_dt):
        df = df.copy()
        # date column may be a Python date object from _to_df; normalise to datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        exploded = (
            df.assign(ticker=df["ticker_mentions"].str.split(","))
            .explode("ticker")
        )
        exploded["ticker"] = exploded["ticker"].str.strip().str.upper()
        filtered = exploded[exploded["ticker"].isin(tickers)]

        print("\n" + "═" * 72)
        print("  REAL REDDIT DATA COVERAGE REPORT")
        print("═" * 72)

        # Per-source row counts
        if "source" in df.columns:
            for src, cnt in df["source"].value_counts().items():
                print(f"  {src:<20} {cnt:>8,} rows")
        print("─" * 72)

        # Per-ticker per-year breakdown
        print(f"\n  {'Ticker':<8}", end="")
        years = list(range(start_dt.year, end_dt.year + 1))
        for y in years:
            print(f"  {y}", end="")
        print("  Overall")
        print("─" * 72)

        all_pass = True
        for ticker in sorted(tickers):
            t_data = filtered[filtered["ticker"] == ticker]
            print(f"  {ticker:<8}", end="")
            ticker_ok = True
            for y in years:
                count = len(t_data[t_data["year"] == y])
                if count == 0:
                    print(f"  {'❌':>4}", end="")
                    ticker_ok = False
                    all_pass = False
                elif count < 10:
                    print(f"  {'⚠':>4}", end="")
                else:
                    print(f"  {'✅':>4}", end="")
            total = len(t_data)
            status = "✅" if ticker_ok else "⚠ gaps"
            print(f"  {total:>6,} rows  {status}")

        print("─" * 72)
        print(f"  ✅ = good data   ⚠ = sparse (<10 posts)   ❌ = no data")
        total_days = (df["date"].max() - df["date"].min()).days
        print(f"\n  Actual span collected: {total_days} days ({total_days/365.25:.1f} years)")
        if not all_pass:
            print(
                "\n  ⚠  Some years have gaps. This is because:\n"
                "     - Reddit API only gives last 1 year directly\n"
                "     - Older data depends on Arctic Shift / PullPush availability\n"
                "     - Model will still train — sparse years just have less sentiment signal"
            )
        else:
            print("\n  ✅ Full coverage across all tickers and years.")
        print("═" * 72 + "\n")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _find_tickers(self, text, ticker_set):
        found = _TICKER_RE.findall(text)
        return {
            t.lstrip("$") for t in found
            if t.lstrip("$") in ticker_set
            and t.lstrip("$") not in _STOPWORDS
        }

    @staticmethod
    def _to_df(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
        df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.date
        return df.dropna(subset=["created_utc"])


if __name__ == "__main__":
    from ticker_selector import TickerSelector  # direct import when run as script
    ts = TickerSelector()
    rc = RedditCollector()
    df = rc.run(tickers=ts.get_top_tickers())
    print(df["source"].value_counts())
    print(f"Total real rows: {len(df):,}")
