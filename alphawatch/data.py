from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import pandas as pd
import yfinance as yf

from alphawatch.config import BENCHMARK_TICKER


def normalize_ticker(raw_ticker: str) -> str:
    """Normalize user-entered NSE symbols for Yahoo Finance."""
    ticker = raw_ticker.strip().upper()
    ticker = re.sub(r"\s+", "", ticker)

    if not ticker:
        return ""
    if ticker.startswith("^"):
        return ticker
    if "." not in ticker:
        return f"{ticker}.NS"
    return ticker


def parse_watchlist(raw_watchlist: str, fallback_tickers: Iterable[str]) -> Tuple[str, ...]:
    """Parse comma/newline-separated ticker text while preserving order."""
    tokens = re.split(r"[,\n]+", raw_watchlist)
    normalized = [normalize_ticker(token) for token in tokens]

    unique_tickers = []
    seen = set()
    for ticker in normalized:
        if ticker and ticker not in seen:
            unique_tickers.append(ticker)
            seen.add(ticker)

    if unique_tickers:
        return tuple(unique_tickers)
    return tuple(fallback_tickers)


def flatten_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance's occasional MultiIndex columns for single tickers."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    normalized = data.copy()
    for level in range(normalized.columns.nlevels):
        if ticker in normalized.columns.get_level_values(level):
            return normalized.xs(ticker, axis=1, level=level)

    normalized.columns = normalized.columns.get_level_values(0)
    return normalized


def download_ticker_data(
    tickers: Tuple[str, ...],
    lookback_period: str,
    min_download_rows: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Download daily OHLCV data and return provider failures separately."""
    downloaded: Dict[str, pd.DataFrame] = {}
    failures: Dict[str, str] = {}

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                period=lookback_period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data.empty:
                failures[ticker] = "No data returned by yfinance."
                continue

            data = flatten_yfinance_columns(data, ticker)
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                failures[ticker] = f"Missing required columns: {', '.join(missing_columns)}"
                continue

            cleaned = data[required_columns].copy()
            cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
            cleaned = cleaned.sort_index()
            cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])
            cleaned["Volume"] = pd.to_numeric(cleaned["Volume"], errors="coerce").fillna(0)

            if len(cleaned) < min_download_rows:
                failures[ticker] = (
                    f"Only {len(cleaned)} rows returned; at least {min_download_rows} rows are needed."
                )
                continue

            downloaded[ticker] = cleaned
        except Exception as exc:  # pragma: no cover - provider/network failures vary.
            failures[ticker] = str(exc)

    return downloaded, failures


def download_benchmark_data(
    lookback_period: str,
    min_download_rows: int,
) -> Tuple[pd.DataFrame, str]:
    """Download benchmark OHLCV data without mixing it into ticker failures."""
    benchmark_data, failures = download_ticker_data(
        (BENCHMARK_TICKER,),
        lookback_period,
        min_download_rows,
    )
    if BENCHMARK_TICKER in benchmark_data:
        return benchmark_data[BENCHMARK_TICKER], ""
    return pd.DataFrame(), failures.get(BENCHMARK_TICKER, "Benchmark data was unavailable.")
