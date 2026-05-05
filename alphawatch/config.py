from __future__ import annotations

from pathlib import Path


APP_TITLE = "AlphaWatch: Indian Stock Market Anomaly Detector"
APP_DESCRIPTION = (
    "A machine-learning dashboard for ranking abnormal market behavior in selected NSE stocks."
)
DISCLAIMER = (
    "This tool is for educational and research purposes only. "
    "It is not financial advice."
)

DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "TATAMOTORS.NS",
    "IRFC.NS",
    "RVNL.NS",
    "SUZLON.NS",
    "YESBANK.NS",
    "ZOMATO.NS",
    "ADANIPOWER.NS",
    "NHPC.NS",
    "POWERGRID.NS",
    "COALINDIA.NS",
    "ONGC.NS",
]

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "NIFTY 50"

LOOKBACK_OPTIONS = ["6mo", "1y", "2y"]
DEFAULT_LOOKBACK = "1y"

BASE_FEATURE_COLUMNS = [
    "daily_return",
    "rolling_7d_return",
    "rolling_14d_volatility",
    "volume_zscore",
    "price_gap",
    "rsi_14",
    "distance_from_20dma",
    "distance_from_50dma",
]

ENRICHED_FEATURE_COLUMNS = [
    "return_zscore_20",
    "volatility_zscore_60",
    "volume_ratio_20",
    "atr_14_percent",
    "intraday_range",
    "close_position_in_range",
    "drawdown_from_60d_high",
    "rsi_change_3d",
    "excess_return",
    "relative_7d_return",
    "excess_return_zscore_20",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENRICHED_FEATURE_COLUMNS

PERCENT_FEATURES = [
    "daily_return",
    "rolling_7d_return",
    "rolling_14d_volatility",
    "price_gap",
    "distance_from_20dma",
    "distance_from_50dma",
    "atr_14_percent",
    "intraday_range",
    "drawdown_from_60d_high",
    "benchmark_return",
    "excess_return",
    "relative_7d_return",
]

FEATURE_LABELS = {
    "daily_return": "Daily return",
    "rolling_7d_return": "7D return",
    "rolling_14d_volatility": "14D volatility",
    "volume_zscore": "Volume z-score",
    "price_gap": "Price gap",
    "rsi_14": "RSI 14",
    "distance_from_20dma": "Distance from 20DMA",
    "distance_from_50dma": "Distance from 50DMA",
    "return_zscore_20": "Return z-score",
    "volatility_zscore_60": "Volatility z-score",
    "volume_ratio_20": "Volume ratio",
    "atr_14_percent": "ATR 14 percent",
    "intraday_range": "Intraday range",
    "close_position_in_range": "Close position in range",
    "drawdown_from_60d_high": "Drawdown from 60D high",
    "rsi_change_3d": "RSI 3D change",
    "benchmark_return": "Benchmark return",
    "excess_return": "Excess return",
    "relative_7d_return": "Relative 7D return",
    "excess_return_zscore_20": "Excess return z-score",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "anomaly_results.csv"

MIN_TRAINING_ROWS = 60
MIN_DOWNLOAD_ROWS = MIN_TRAINING_ROWS + 20
MODEL_RANDOM_STATE = 42
MODEL_CONTAMINATION = 0.08
MODEL_ESTIMATORS = 120
CACHE_TTL_SECONDS = 60 * 60

RISK_THRESHOLDS = {
    "Extreme": 95,
    "High": 80,
    "Moderate": 60,
}
