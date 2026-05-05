from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def trailing_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compare the current value with the prior trailing distribution."""
    trailing = series.shift(1)
    mean = trailing.rolling(window=window, min_periods=min_periods).mean()
    std = trailing.rolling(window=window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI from trailing average gains and losses only."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    return rsi.fillna(50).clip(0, 100)


def prepare_benchmark_features(benchmark_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create date-aligned NIFTY context columns used by stock features."""
    if benchmark_data is None or benchmark_data.empty:
        return pd.DataFrame()

    benchmark = benchmark_data[["Close"]].copy()
    benchmark = benchmark.rename(columns={"Close": "benchmark_close"})
    benchmark["benchmark_return"] = benchmark["benchmark_close"].pct_change()
    benchmark["benchmark_7d_return"] = benchmark["benchmark_close"] / benchmark["benchmark_close"].shift(7) - 1
    return benchmark


def add_features(
    data: pd.DataFrame,
    ticker: str,
    benchmark_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create leakage-safe technical and volume features for one ticker."""
    featured = data.copy()
    close = featured["Close"]
    volume = featured["Volume"]
    previous_close = close.shift(1)

    featured["ticker"] = ticker
    featured["daily_return"] = close.pct_change()
    featured["rolling_7d_return"] = close / close.shift(7) - 1
    featured["rolling_14d_volatility"] = (
        featured["daily_return"].rolling(window=14, min_periods=14).std() * np.sqrt(252)
    )
    featured["return_zscore_20"] = trailing_zscore(featured["daily_return"], window=20, min_periods=20)
    featured["volatility_zscore_60"] = trailing_zscore(
        featured["rolling_14d_volatility"],
        window=60,
        min_periods=30,
    )

    trailing_volume = volume.shift(1)
    volume_mean = trailing_volume.rolling(window=20, min_periods=20).mean()
    volume_std = trailing_volume.rolling(window=20, min_periods=20).std()
    featured["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    featured["volume_ratio_20"] = volume / volume_mean.replace(0, np.nan)

    featured["price_gap"] = (featured["Open"] - previous_close) / previous_close
    featured["rsi_14"] = calculate_rsi(close, window=14)
    featured["rsi_change_3d"] = featured["rsi_14"] - featured["rsi_14"].shift(3)

    featured["moving_average_20"] = close.rolling(window=20, min_periods=20).mean()
    featured["moving_average_50"] = close.rolling(window=50, min_periods=50).mean()
    featured["distance_from_20dma"] = close / featured["moving_average_20"] - 1
    featured["distance_from_50dma"] = close / featured["moving_average_50"] - 1

    true_range = pd.concat(
        [
            featured["High"] - featured["Low"],
            (featured["High"] - previous_close).abs(),
            (featured["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    featured["atr_14_percent"] = true_range.rolling(window=14, min_periods=14).mean() / close
    featured["intraday_range"] = (featured["High"] - featured["Low"]) / close
    range_width = (featured["High"] - featured["Low"]).replace(0, np.nan)
    featured["close_position_in_range"] = ((close - featured["Low"]) / range_width).clip(0, 1)
    rolling_60d_high = featured["High"].rolling(window=60, min_periods=20).max()
    featured["drawdown_from_60d_high"] = close / rolling_60d_high - 1

    if benchmark_features is not None and not benchmark_features.empty:
        featured = featured.join(benchmark_features, how="left")
        featured["has_benchmark_context"] = featured["benchmark_return"].notna()
    else:
        featured["benchmark_close"] = np.nan
        featured["benchmark_return"] = 0.0
        featured["benchmark_7d_return"] = 0.0
        featured["has_benchmark_context"] = False

    featured["benchmark_return"] = featured["benchmark_return"].fillna(0.0)
    featured["benchmark_7d_return"] = featured["benchmark_7d_return"].fillna(0.0)
    featured["excess_return"] = featured["daily_return"] - featured["benchmark_return"]
    featured["relative_7d_return"] = featured["rolling_7d_return"] - featured["benchmark_7d_return"]
    featured["excess_return_zscore_20"] = trailing_zscore(
        featured["excess_return"],
        window=20,
        min_periods=20,
    )
    featured["excess_return_zscore_20"] = featured["excess_return_zscore_20"].fillna(0.0)
    featured["relative_7d_return"] = featured["relative_7d_return"].fillna(0.0)

    return featured.replace([np.inf, -np.inf], np.nan)


def prepare_all_features(
    price_data: Dict[str, pd.DataFrame],
    benchmark_data: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    benchmark_features = prepare_benchmark_features(benchmark_data)
    return {
        ticker: add_features(data, ticker, benchmark_features)
        for ticker, data in price_data.items()
    }
