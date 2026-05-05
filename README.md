# AlphaWatch: Indian Stock Market Anomaly Detector

AlphaWatch is a machine-learning dashboard for researching unusual behavior in Indian NSE stocks. It downloads recent OHLCV market data, engineers leakage-safe quantitative features, compares stocks against the NIFTY 50 benchmark, scores anomalies with an ensemble model, and presents the results in a clean Streamlit analyst dashboard.

## Live Demo

Open the deployed app here:

[https://alphawatch1.streamlit.app/](https://alphawatch1.streamlit.app/)

## What AlphaWatch Does

- Downloads NSE daily stock data with `yfinance`.
- Uses `^NSEI` as a NIFTY 50 benchmark when benchmark data is available.
- Lets users edit the stock watchlist directly in the sidebar.
- Normalizes bare NSE symbols such as `RELIANCE` to `RELIANCE.NS`.
- Builds price, volume, volatility, momentum, RSI, moving-average, range, drawdown, and benchmark-relative features.
- Detects abnormal market behavior with an ensemble anomaly score.
- Ranks stocks by latest anomaly score and exports the full ranking to `anomaly_results.csv`.
- Explains signals with risk level, confidence level, signal direction, primary driver, driver details, and top feature drivers.
- Includes charts for market overview, ticker deep dive, analyst report, feature explorer, and data quality.

## Why Anomaly Detection

Direct stock-price prediction is usually brittle because prices react to noisy and fast-changing information: news, liquidity, sentiment, macro events, index movement, and market regime shifts.

AlphaWatch uses anomaly detection instead. The question is not "what will the price be tomorrow?" The question is:

```text
Is this stock behaving unusually compared with its own history and the broader market?
```

That framing makes the tool useful for research and monitoring. It surfaces stocks that may deserve a closer look without pretending to produce buy or sell recommendations.

## Signal Engine

AlphaWatch combines three scoring components:

- `isolation_score`: an Isolation Forest anomaly score trained on historical feature behavior.
- `robust_feature_score`: a median/MAD statistical score based on large feature deviations.
- `relative_market_score`: a benchmark-relative score based on excess return and relative strength versus NIFTY 50.

The final anomaly score is:

```text
final_anomaly_score = 0.60 * isolation_score
                    + 0.25 * robust_feature_score
                    + 0.15 * relative_market_score
```

The app keeps the exported `anomaly_score` column as the final score for compatibility.

## Features Used

Core stock features:

- `daily_return`
- `rolling_7d_return`
- `rolling_14d_volatility`
- `volume_zscore`
- `price_gap`
- `rsi_14`
- `distance_from_20dma`
- `distance_from_50dma`

Strong v2 feature set:

- `return_zscore_20`
- `volatility_zscore_60`
- `volume_ratio_20`
- `atr_14_percent`
- `intraday_range`
- `close_position_in_range`
- `drawdown_from_60d_high`
- `rsi_change_3d`

Benchmark-relative features:

- `benchmark_return`
- `excess_return`
- `relative_7d_return`
- `excess_return_zscore_20`

Feature calculations are trailing-only. Rolling statistics, moving averages, RSI, volatility, and z-score baselines use current and past data only, avoiding future data leakage.

## Dashboard

AlphaWatch includes five main views:

- `Market Overview`: portfolio summary, ranked anomaly table, reason distribution, and CSV download.
- `Ticker Deep Dive`: latest metrics, candlestick chart, 20DMA, 50DMA, anomaly markers, volume, and score history.
- `Analyst Report`: selected-stock diagnosis, score breakdown, top drivers, confidence, risk level, benchmark comparison, and research checklist.
- `Feature Explorer`: latest feature snapshot and raw feature values.
- `Data Quality`: successful tickers, failed tickers, row counts, benchmark status, and provider error messages.

## Controls

- Editable watchlist with comma or newline support.
- Lookback selector: `6mo`, `1y`, or `2y`.
- Top-N filter.
- Minimum anomaly score filter.
- Anomaly-only toggle.
- Reason filter.
- Confidence filter.
- Signal-direction filter.
- Ticker selector for detailed charts.
- Refresh button to clear cached market data.

## Default NSE Watchlist

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

## Project Structure

```text
app.py
alphawatch/
  __init__.py
  charts.py
  config.py
  data.py
  features.py
  model.py
requirements.txt
README.md
anomaly_results.csv
```

## Architecture

- `app.py`: Streamlit layout, sidebar controls, caching, tabs, and CSV export.
- `alphawatch/config.py`: default ticker list, constants, thresholds, and scoring configuration.
- `alphawatch/data.py`: watchlist parsing, NSE ticker normalization, yfinance downloads, benchmark download, and failure handling.
- `alphawatch/features.py`: leakage-safe stock and benchmark-relative feature engineering.
- `alphawatch/model.py`: Isolation Forest scoring, robust statistical scoring, ensemble score, risk/confidence labels, and signal explanations.
- `alphawatch/charts.py`: Plotly chart builders for price, volume, anomaly score, feature drivers, and benchmark comparison.

## Tech Stack

- Python
- Streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- plotly
- matplotlib

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

When the app runs, it automatically writes the latest ranked output to:

```text
anomaly_results.csv
```

## Example Use Case

A market analyst can open AlphaWatch after market close, paste a custom NSE watchlist, and quickly identify which stocks showed unusual price, volume, volatility, momentum, or benchmark-relative behavior.

Instead of reading every chart manually, the analyst can start with the ranked anomaly table, inspect high-scoring tickers in the deep-dive chart, review the score breakdown in the analyst report, and use the data-quality tab to confirm which tickers were successfully downloaded.

## Important Notes

- AlphaWatch depends on Yahoo Finance data through `yfinance`, so symbol availability can vary.
- Failed symbols are reported explicitly in the Data Quality tab.
- Some NSE tickers can be renamed, unavailable, or delayed depending on Yahoo Finance coverage.
- The model is designed for research triage, not automated trading.

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice. Stock markets involve risk, and anomaly scores should not be interpreted as buy, sell, or hold recommendations.
