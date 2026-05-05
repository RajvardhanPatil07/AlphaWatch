from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from alphawatch.charts import (
    build_candlestick_chart,
    build_feature_snapshot_chart,
    build_market_relative_chart,
    build_reason_distribution_chart,
    build_score_component_chart,
    build_score_history_chart,
    build_top_driver_chart,
    build_volume_chart,
)
from alphawatch.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    CACHE_TTL_SECONDS,
    DEFAULT_LOOKBACK,
    DEFAULT_TICKERS,
    DISCLAIMER,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    LOOKBACK_OPTIONS,
    MIN_DOWNLOAD_ROWS,
    PERCENT_FEATURES,
    RESULTS_PATH,
)
from alphawatch.data import download_benchmark_data, download_ticker_data, parse_watchlist
from alphawatch.features import prepare_all_features
from alphawatch.model import score_historical_anomalies, score_latest_rankings


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_market_data(
    tickers: Tuple[str, ...],
    lookback_period: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], pd.DataFrame, str]:
    price_data, failures = download_ticker_data(tickers, lookback_period, MIN_DOWNLOAD_ROWS)
    benchmark_data, benchmark_error = download_benchmark_data(lookback_period, MIN_DOWNLOAD_ROWS)
    return price_data, failures, benchmark_data, benchmark_error


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def score_selected_history(data: pd.DataFrame) -> pd.DataFrame:
    return score_historical_anomalies(data)


def save_rankings(ranked_results: pd.DataFrame) -> None:
    if not ranked_results.empty:
        ranked_results.to_csv(RESULTS_PATH, index=False)


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def render_summary_cards(
    requested_count: int,
    ranked_results: pd.DataFrame,
    failures: Dict[str, str],
) -> None:
    flagged_count = int(ranked_results["is_anomaly"].sum()) if not ranked_results.empty else 0
    highest_score = ranked_results["anomaly_score"].max() if not ranked_results.empty else float("nan")

    columns = st.columns(5)
    columns[0].metric("Requested", requested_count)
    columns[1].metric("Scored", len(ranked_results))
    columns[2].metric("Failed", len(failures))
    columns[3].metric("Flagged", flagged_count)
    columns[4].metric("Highest score", format_number(highest_score, 1))


def filter_rankings(
    ranked_results: pd.DataFrame,
    top_n: int,
    min_score: int,
    anomaly_only: bool,
    selected_reasons: list[str],
    selected_confidence: list[str],
    selected_directions: list[str],
) -> pd.DataFrame:
    filtered = ranked_results.copy()
    filtered = filtered[filtered["anomaly_score"] >= min_score]

    if anomaly_only:
        filtered = filtered[filtered["is_anomaly"] == True]
    if selected_reasons:
        filtered = filtered[filtered["reason"].isin(selected_reasons)]
    if selected_confidence:
        filtered = filtered[filtered["confidence_level"].isin(selected_confidence)]
    if selected_directions:
        filtered = filtered[filtered["signal_direction"].isin(selected_directions)]

    return filtered.head(top_n)


def render_metric_cards(latest_row: pd.Series) -> None:
    columns = st.columns(6)
    columns[0].metric("Close", f"INR {format_number(latest_row['Close'])}")
    columns[1].metric(
        "Daily return",
        format_percent(latest_row["daily_return"]),
        delta=format_percent(latest_row["daily_return"]),
    )
    columns[2].metric("Excess return", format_percent(latest_row.get("excess_return")))
    columns[3].metric("Anomaly score", format_number(latest_row["anomaly_score"], 1))
    columns[4].metric("Confidence", latest_row.get("confidence_level", "Low"))
    columns[5].metric("Risk", latest_row.get("risk_level", "Normal"))


def render_ranked_table(ranked_results: pd.DataFrame) -> None:
    display_columns = [
        "rank",
        "ticker",
        "date",
        "close",
        "daily_return",
        "rolling_7d_return",
        "excess_return",
        "volume_zscore",
        "rsi_14",
        "anomaly_score",
        "confidence_level",
        "risk_level",
        "signal_direction",
        "primary_driver",
        "isolation_score",
        "robust_feature_score",
        "relative_market_score",
        "is_anomaly",
        "reason",
        "top_feature_drivers",
    ]
    formatted = ranked_results[display_columns].copy()
    for column in ["daily_return", "rolling_7d_return", "excess_return"]:
        formatted[column] = formatted[column] * 100

    st.dataframe(
        formatted,
        width="stretch",
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "ticker": "Ticker",
            "date": "Latest date",
            "close": st.column_config.NumberColumn("Close", format="INR %.2f"),
            "daily_return": st.column_config.NumberColumn("Daily return", format="%.2f%%"),
            "rolling_7d_return": st.column_config.NumberColumn("7D return", format="%.2f%%"),
            "excess_return": st.column_config.NumberColumn("Excess return", format="%.2f%%"),
            "volume_zscore": st.column_config.NumberColumn("Volume z-score", format="%.2f"),
            "rsi_14": st.column_config.NumberColumn("RSI 14", format="%.1f"),
            "anomaly_score": st.column_config.ProgressColumn(
                "Final score",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "confidence_level": "Confidence",
            "risk_level": "Risk",
            "signal_direction": "Direction",
            "primary_driver": "Primary driver",
            "isolation_score": st.column_config.NumberColumn("Isolation", format="%.1f"),
            "robust_feature_score": st.column_config.NumberColumn("Robust", format="%.1f"),
            "relative_market_score": st.column_config.NumberColumn("Market-relative", format="%.1f"),
            "is_anomaly": "Model flag",
            "reason": "Reason",
            "top_feature_drivers": "Top drivers",
        },
    )


def render_feature_table(latest_row: pd.Series) -> None:
    rows = []
    for feature in FEATURE_COLUMNS + ["benchmark_return"]:
        value = latest_row.get(feature)
        unit = "%"
        display_value = value * 100 if feature in PERCENT_FEATURES else value
        if feature not in PERCENT_FEATURES:
            unit = "score" if "zscore" in feature or feature in {"rsi_14", "volume_ratio_20"} else "value"
        rows.append(
            {
                "feature": FEATURE_LABELS.get(feature, feature),
                "latest_value": display_value,
                "unit": unit,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "feature": "Feature",
            "latest_value": st.column_config.NumberColumn("Latest value", format="%.3f"),
            "unit": "Unit",
        },
    )


def render_data_quality(
    requested_tickers: Tuple[str, ...],
    price_data: Dict[str, pd.DataFrame],
    failures: Dict[str, str],
    ranked_results: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    benchmark_error: str,
) -> None:
    st.write("Benchmark context")
    benchmark_loaded = not benchmark_data.empty
    common_dates = pd.Index([])
    if benchmark_loaded and price_data:
        common_dates = benchmark_data.index
        for data in price_data.values():
            common_dates = common_dates.intersection(data.index)
    latest_common_date = common_dates.max().date().isoformat() if len(common_dates) else "N/A"

    benchmark_columns = st.columns(3)
    benchmark_columns[0].metric("Benchmark", f"{BENCHMARK_NAME} ({BENCHMARK_TICKER})")
    benchmark_columns[1].metric("Loaded", "Yes" if benchmark_loaded else "No")
    benchmark_columns[2].metric("Latest common date", latest_common_date)
    if benchmark_error:
        st.warning(f"Benchmark warning: {benchmark_error}")

    st.write("Requested watchlist")
    requested_rows = []
    scored_tickers = set(ranked_results["ticker"]) if not ranked_results.empty else set()
    for ticker in requested_tickers:
        requested_rows.append(
            {
                "ticker": ticker,
                "downloaded": ticker in price_data,
                "scored": ticker in scored_tickers,
                "rows": len(price_data[ticker]) if ticker in price_data else 0,
            }
        )
    st.dataframe(pd.DataFrame(requested_rows), width="stretch", hide_index=True)

    st.write("Provider failures")
    if failures:
        failure_rows = [{"ticker": ticker, "message": message} for ticker, message in failures.items()]
        st.dataframe(pd.DataFrame(failure_rows), width="stretch", hide_index=True)
    else:
        st.success("No provider failures in the latest run.")

    st.write("Generated CSV")
    if RESULTS_PATH.exists():
        st.caption(f"{RESULTS_PATH.name} was generated with {len(ranked_results)} unfiltered ranked rows.")
    else:
        st.warning("No CSV has been generated yet.")


def render_analyst_report(
    selected_ticker: str,
    latest_row: pd.Series,
    scored_selected: pd.DataFrame,
) -> None:
    st.subheader(f"{selected_ticker} Analyst Report")

    summary_columns = st.columns(4)
    summary_columns[0].metric("Final score", format_number(latest_row.get("anomaly_score"), 1))
    summary_columns[1].metric("Confidence", latest_row.get("confidence_level", "Low"))
    summary_columns[2].metric("Primary driver", latest_row.get("primary_driver", "Multi-factor"))
    summary_columns[3].metric("Direction", latest_row.get("signal_direction", "Mixed"))

    st.write("Diagnosis")
    st.markdown(
        "\n".join(
            [
                f"- **Reason:** {latest_row.get('reason', 'Multi-factor anomaly')}",
                f"- **Driver details:** {latest_row.get('driver_details', 'No dominant driver')}",
                f"- **Top feature drivers:** {latest_row.get('top_feature_drivers', 'No dominant driver')}",
                f"- **Market context:** excess return {format_percent(latest_row.get('excess_return'))}, "
                f"relative 7D return {format_percent(latest_row.get('relative_7d_return'))}.",
            ]
        )
    )

    chart_columns = st.columns(2)
    chart_columns[0].plotly_chart(
        build_score_component_chart(latest_row, selected_ticker),
        use_container_width=True,
    )
    chart_columns[1].plotly_chart(
        build_top_driver_chart(latest_row, selected_ticker),
        use_container_width=True,
    )
    st.plotly_chart(
        build_market_relative_chart(scored_selected, selected_ticker),
        use_container_width=True,
    )

    recent_history = scored_selected.dropna(subset=["anomaly_score"]).tail(20)
    anomaly_days = int(recent_history["is_anomaly"].sum()) if not recent_history.empty else 0
    average_score = recent_history["anomaly_score"].mean() if not recent_history.empty else float("nan")
    st.write("Recent context")
    context_columns = st.columns(3)
    context_columns[0].metric("20-day average score", format_number(average_score, 1))
    context_columns[1].metric("20-day anomaly days", anomaly_days)
    context_columns[2].metric("Latest RSI", format_number(latest_row.get("rsi_14"), 1))

    with st.expander("Research checklist", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- Check exchange announcements, earnings dates, and corporate actions.",
                    "- Compare the move with sector peers before interpreting it as stock-specific.",
                    "- Inspect liquidity and delivery volume if using the signal outside this dashboard.",
                    "- Treat the score as an attention signal, not a buy or sell recommendation.",
                ]
            )
        )


def main() -> None:
    st.set_page_config(page_title="AlphaWatch", layout="wide")

    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)
    st.info(DISCLAIMER)

    with st.sidebar:
        st.header("Controls")
        raw_watchlist = st.text_area(
            "NSE watchlist",
            value="\n".join(DEFAULT_TICKERS),
            height=220,
            help="Use commas or new lines. Bare symbols like RELIANCE are converted to RELIANCE.NS.",
        )
        requested_tickers = parse_watchlist(raw_watchlist, DEFAULT_TICKERS)
        lookback_period = st.selectbox(
            "Lookback period",
            LOOKBACK_OPTIONS,
            index=LOOKBACK_OPTIONS.index(DEFAULT_LOOKBACK),
        )
        if st.button("Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data source: Yahoo Finance via yfinance")

    with st.spinner("Downloading NSE price data and scoring latest anomalies..."):
        price_data, failures, benchmark_data, benchmark_error = load_market_data(
            requested_tickers,
            lookback_period,
        )
        featured_data = prepare_all_features(price_data, benchmark_data)
        ranked_results = score_latest_rankings(featured_data, lookback_period)
        save_rankings(ranked_results)

    if ranked_results.empty:
        render_summary_cards(len(requested_tickers), ranked_results, failures)
        st.error("No ranked results are available. Try a longer lookback or refresh data.")
        if failures:
            render_data_quality(
                requested_tickers,
                price_data,
                failures,
                ranked_results,
                benchmark_data,
                benchmark_error,
            )
        return

    with st.sidebar:
        st.divider()
        st.subheader("Ranking Filters")
        top_n = st.slider("Top N", min_value=1, max_value=len(ranked_results), value=len(ranked_results))
        min_score = st.slider("Minimum anomaly score", min_value=0, max_value=100, value=0)
        anomaly_only = st.toggle("Show model-flagged anomalies only", value=False)
        reason_options = sorted(ranked_results["reason"].dropna().unique().tolist())
        selected_reasons = st.multiselect("Reason filter", reason_options)
        confidence_options = ["High", "Medium", "Low"]
        selected_confidence = st.multiselect("Confidence filter", confidence_options)
        direction_options = sorted(ranked_results["signal_direction"].dropna().unique().tolist())
        selected_directions = st.multiselect("Direction filter", direction_options)

    filtered_rankings = filter_rankings(
        ranked_results,
        top_n=top_n,
        min_score=min_score,
        anomaly_only=anomaly_only,
        selected_reasons=selected_reasons,
        selected_confidence=selected_confidence,
        selected_directions=selected_directions,
    )

    selected_options = (
        filtered_rankings["ticker"].tolist()
        if not filtered_rankings.empty
        else ranked_results["ticker"].tolist()
    )
    selected_ticker = st.sidebar.selectbox("Ticker deep dive", selected_options)

    selected_data = featured_data[selected_ticker]
    with st.spinner(f"Scoring historical anomalies for {selected_ticker}..."):
        scored_selected = score_selected_history(selected_data)

    scored_history = scored_selected.dropna(subset=["anomaly_score"])
    latest_selected = scored_history.iloc[-1] if not scored_history.empty else scored_selected.iloc[-1]

    overview_tab, deep_dive_tab, analyst_tab, features_tab, quality_tab = st.tabs(
        ["Market Overview", "Ticker Deep Dive", "Analyst Report", "Feature Explorer", "Data Quality"]
    )

    with overview_tab:
        render_summary_cards(len(requested_tickers), ranked_results, failures)
        st.subheader("Ranked Anomaly Table")
        if filtered_rankings.empty:
            st.warning("No stocks match the active filters.")
        else:
            render_ranked_table(filtered_rankings)

        csv_data = ranked_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full anomaly_results.csv",
            data=csv_data,
            file_name="anomaly_results.csv",
            mime="text/csv",
            width="content",
        )
        st.plotly_chart(build_reason_distribution_chart(ranked_results), use_container_width=True)

    with deep_dive_tab:
        st.subheader(f"{selected_ticker} Latest Metrics")
        render_metric_cards(latest_selected)
        st.plotly_chart(build_candlestick_chart(scored_selected, selected_ticker), use_container_width=True)
        chart_columns = st.columns(2)
        chart_columns[0].plotly_chart(
            build_volume_chart(scored_selected, selected_ticker),
            use_container_width=True,
        )
        chart_columns[1].plotly_chart(
            build_score_history_chart(scored_selected, selected_ticker),
            use_container_width=True,
        )

    with analyst_tab:
        render_analyst_report(selected_ticker, latest_selected, scored_selected)

    with features_tab:
        chart_columns = st.columns([1.2, 1])
        chart_columns[0].plotly_chart(
            build_feature_snapshot_chart(latest_selected, selected_ticker),
            use_container_width=True,
        )
        with chart_columns[1]:
            st.subheader("Latest Feature Values")
            render_feature_table(latest_selected)

    with quality_tab:
        render_data_quality(
            requested_tickers,
            price_data,
            failures,
            ranked_results,
            benchmark_data,
            benchmark_error,
        )

    st.caption(f"Full latest ranked results are saved automatically to {RESULTS_PATH.name}.")


if __name__ == "__main__":
    main()
