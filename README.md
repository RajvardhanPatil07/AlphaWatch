# AlphaWatch: Indian Stock Market Anomaly Detector

AlphaWatch is a Streamlit dashboard that downloads NSE stock data, engineers market-behavior features, detects abnormal activity with machine learning, and ranks stocks by anomaly score.

Strong v2 improves the signal quality by combining an Isolation Forest model, robust statistical feature deviations, and NIFTY 50-relative context. It also adds an analyst report for each stock so the result is easier to interpret.

## What the Project Does

- Downloads daily OHLCV data for selected NSE stocks using `yfinance`.
- Downloads NIFTY 50 benchmark data through `^NSEI` when available.
- Lets users edit the NSE watchlist directly in the sidebar.
- Calculates price, volume, volatility, momentum, RSI, moving-average, range, drawdown, and benchmark-relative features.
- Scores anomalies with an ensemble of ML and robust statistical signals.
- Ranks stocks by final anomaly score and classifies risk, confidence, and signal direction.
- Explains each signal with primary drivers and top feature contributors.
- Saves the full latest ranked output to `anomaly_results.csv`.
- Displays Streamlit tabs for market overview, ticker deep dive, analyst report, feature explorer, and data quality.

## Why Anomaly Detection Instead of Direct Price Prediction

Direct price prediction tries to forecast the exact future price or return of a stock, which is noisy, unstable, and highly sensitive to news, liquidity, macro events, and market regime changes.

Anomaly detection asks a more realistic research question: "Is today's behavior unusual compared with this stock's own recent history and the broader market?" That makes the model useful for surfacing stocks that deserve attention without pretending to know tomorrow's price.

## Features Used

AlphaWatch calculates leakage-safe features from historical daily data:

- `daily_return`
- `rolling_7d_return`
- `rolling_14d_volatility`
- `volume_zscore`
- `price_gap`
- `rsi_14`
- `distance_from_20dma`
- `distance_from_50dma`
- `return_zscore_20`
- `volatility_zscore_60`
- `volume_ratio_20`
- `atr_14_percent`
- `intraday_range`
- `close_position_in_range`
- `drawdown_from_60d_high`
- `rsi_change_3d`
- `benchmark_return`
- `excess_return`
- `relative_7d_return`
- `excess_return_zscore_20`

Rolling baselines use current and prior historical data only. Z-score baselines compare the current value with prior trailing windows to avoid future leakage.

## ML And Scoring Method

The main model is `IsolationForest` from scikit-learn. Strong v2 keeps Isolation Forest as the core anomaly detector but adds two more scoring layers:

- `isolation_score`: percentile score from the Isolation Forest anomaly output.
- `robust_feature_score`: median/MAD-based score from the strongest feature deviations.
- `relative_market_score`: score from NIFTY 50-relative behavior such as excess return and relative 7-day return.

The final score is:

```text
final_anomaly_score = 0.60 * isolation_score
                    + 0.25 * robust_feature_score
                    + 0.15 * relative_market_score
```

The app keeps `anomaly_score` as the final score for compatibility and also exports each component score.

## Confidence And Explanation

AlphaWatch assigns:

- `risk_level`: `Extreme`, `High`, `Moderate`, or `Normal`.
- `confidence_level`: `High`, `Medium`, or `Low`, based on agreement across score components.
- `signal_direction`: `Bullish`, `Bearish`, or `Mixed`.
- `primary_driver`: the main category behind the anomaly.
- `top_feature_drivers`: the strongest robust-z feature deviations.
- `driver_details`: a short score-component explanation.

Reasons can combine multiple drivers, such as `Momentum breakout + Market-relative strength`.

## Project Structure

```text
app.py
alphawatch/
  config.py
  data.py
  features.py
  model.py
  charts.py
requirements.txt
README.md
anomaly_results.csv
```

- `app.py` contains Streamlit layout, caching, controls, tabs, and CSV export.
- `alphawatch/data.py` handles ticker normalization, watchlist parsing, yfinance downloads, benchmark downloads, and provider failures.
- `alphawatch/features.py` calculates leakage-safe stock and benchmark-relative features.
- `alphawatch/model.py` scores anomalies, builds ensemble scores, explains drivers, and assigns risk/confidence/direction.
- `alphawatch/charts.py` builds Plotly charts.

## Default NSE Tickers

```text
RELIANCE.NS
TCS.NS
INFY.NS
HDFCBANK.NS
TATAMOTORS.NS
IRFC.NS
RVNL.NS
SUZLON.NS
YESBANK.NS
ZOMATO.NS
ADANIPOWER.NS
NHPC.NS
POWERGRID.NS
COALINDIA.NS
ONGC.NS
```

## Dashboard Controls

- Editable NSE watchlist with comma or newline support.
- Bare symbols such as `RELIANCE` are automatically converted to `RELIANCE.NS`.
- Lookback selector: `6mo`, `1y`, or `2y`.
- Top-N filter.
- Minimum anomaly score filter.
- Model-flagged anomaly toggle.
- Reason, confidence, and signal-direction filters.
- Ticker selector for deep-dive charts.
- Refresh button to clear cached market data.

## Dashboard Tabs

- `Market Overview`: summary cards, ranked anomaly table, CSV download, and reason distribution.
- `Ticker Deep Dive`: latest metrics, candlestick chart with 20DMA/50DMA, anomaly markers, volume chart, and score history.
- `Analyst Report`: diagnosis, score breakdown, top feature drivers, market-relative chart, recent context, and research checklist.
- `Feature Explorer`: latest feature snapshot chart and feature-value table.
- `Data Quality`: requested tickers, downloaded/scored status, row counts, benchmark status, provider failures, and CSV status.

## How to Install

```bash
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run app.py
```

When the app runs, it automatically writes the latest full ranked output to:

```text
anomaly_results.csv
```

## Streamlit Community Cloud Deployment

AlphaWatch is designed to run directly on Streamlit Community Cloud's free plan.

1. Push this project to GitHub.
2. Open Streamlit Community Cloud and choose **New app**.
3. Select the GitHub repository and the `main` branch.
4. Set the app file to `app.py`.
5. Deploy.

No paid market-data service or API key is required. The app downloads Yahoo Finance data with `yfinance` when it runs.

## Example Use Case

A market analyst can open AlphaWatch after the market close, paste a custom NSE watchlist, and quickly see which monitored stocks showed unusual behavior. A high anomaly score might point to a volume spike, large gap, abnormal return, momentum breakout, market-relative strength, or price movement far away from recent moving averages.

The analyst can then inspect the candlestick chart, score breakdown, top drivers, confidence level, benchmark comparison, and recent score history before deciding whether the stock deserves deeper research.

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Stock markets involve risk, and anomaly scores should not be interpreted as buy, sell, or hold recommendations.
