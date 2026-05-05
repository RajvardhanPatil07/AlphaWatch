from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from alphawatch.config import BENCHMARK_NAME, FEATURE_COLUMNS, FEATURE_LABELS, PERCENT_FEATURES


def build_candlestick_chart(scored_data: pd.DataFrame, ticker: str) -> go.Figure:
    anomalies = scored_data[scored_data["is_anomaly"] == True]

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=scored_data.index,
            open=scored_data["Open"],
            high=scored_data["High"],
            low=scored_data["Low"],
            close=scored_data["Close"],
            name="OHLC",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scored_data.index,
            y=scored_data["moving_average_20"],
            mode="lines",
            name="20DMA",
            line=dict(color="#2563eb", width=1.5),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scored_data.index,
            y=scored_data["moving_average_50"],
            mode="lines",
            name="50DMA",
            line=dict(color="#7c3aed", width=1.5),
        )
    )

    if not anomalies.empty:
        figure.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=anomalies["Close"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#f97316", size=10, symbol="diamond", line=dict(width=1)),
                customdata=np.stack(
                    [
                        anomalies["anomaly_score"].fillna(0),
                        anomalies["reason"],
                        anomalies["risk_level"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Date=%{x}<br>"
                    "Close=%{y:.2f}<br>"
                    "Score=%{customdata[0]:.1f}<br>"
                    "Risk=%{customdata[2]}<br>"
                    "%{customdata[1]}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=f"{ticker} candlestick view",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    figure.update_xaxes(rangeslider_visible=False)
    return figure


def build_volume_chart(scored_data: pd.DataFrame, ticker: str) -> go.Figure:
    colors = np.where(scored_data["volume_zscore"].fillna(0) >= 2.0, "#dc2626", "#64748b")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=scored_data.index,
            y=scored_data["Volume"],
            name="Volume",
            marker_color=colors,
            hovertemplate="Date=%{x}<br>Volume=%{y:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{ticker} trading volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=310,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        showlegend=False,
    )
    return figure


def build_score_history_chart(scored_data: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=scored_data.index,
            y=scored_data["anomaly_score"],
            mode="lines",
            name="Anomaly score",
            line=dict(color="#0f766e", width=2),
            hovertemplate="Date=%{x}<br>Score=%{y:.1f}<extra></extra>",
        )
    )
    for column, label, color in [
        ("isolation_score", "Isolation", "#64748b"),
        ("robust_feature_score", "Robust features", "#7c3aed"),
        ("relative_market_score", "Market-relative", "#ea580c"),
    ]:
        if column in scored_data:
            figure.add_trace(
                go.Scatter(
                    x=scored_data.index,
                    y=scored_data[column],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.6,
                    hovertemplate=f"Date=%{{x}}<br>{label}=%{{y:.1f}}<extra></extra>",
                )
            )
    figure.add_hrect(y0=80, y1=100, line_width=0, fillcolor="#fee2e2", opacity=0.35)
    figure.add_hline(y=80, line_dash="dot", line_color="#dc2626")
    figure.update_layout(
        title=f"{ticker} anomaly score history",
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def build_feature_snapshot_chart(latest_row: pd.Series, ticker: str) -> go.Figure:
    values = []
    labels = []
    for feature in FEATURE_COLUMNS:
        value = latest_row.get(feature, np.nan)
        if feature in PERCENT_FEATURES:
            value = value * 100
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ")) + " (%)"
        else:
            label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
        labels.append(label)
        values.append(value)

    colors = [
        "#dc2626" if pd.notna(value) and abs(value) >= 3 and "RSI" not in label.upper() else "#2563eb"
        for label, value in zip(labels, values)
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>%{x:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{ticker} latest feature snapshot",
        xaxis_title="Feature value",
        height=max(430, len(labels) * 24),
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        showlegend=False,
    )
    return figure


def build_score_component_chart(latest_row: pd.Series, ticker: str) -> go.Figure:
    labels = ["Isolation", "Robust features", "Market-relative", "Final score"]
    values = [
        latest_row.get("isolation_score", np.nan),
        latest_row.get("robust_feature_score", np.nan),
        latest_row.get("relative_market_score", np.nan),
        latest_row.get("anomaly_score", np.nan),
    ]
    colors = ["#64748b", "#7c3aed", "#ea580c", "#0f766e"]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            hovertemplate="%{x}<br>Score=%{y:.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{ticker} ensemble score breakdown",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        height=330,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        showlegend=False,
    )
    return figure


def parse_driver_string(driver_string: str) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for part in str(driver_string).split(";"):
        match = re.search(r"(.+?)\s+\(([-+]?[0-9]*\.?[0-9]+)z\)", part.strip())
        if match:
            labels.append(match.group(1))
            values.append(float(match.group(2)))
    return labels, values


def build_top_driver_chart(latest_row: pd.Series, ticker: str) -> go.Figure:
    labels, values = parse_driver_string(latest_row.get("top_feature_drivers", ""))
    if not labels:
        labels = [latest_row.get("primary_driver", "Multi-factor")]
        values = [0.0]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#dc2626",
            hovertemplate="%{y}<br>Robust z=%{x:.1f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{ticker} top feature drivers",
        xaxis_title="Robust z-score",
        height=330,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        showlegend=False,
    )
    return figure


def build_market_relative_chart(
    scored_data: pd.DataFrame,
    ticker: str,
    benchmark_name: str = BENCHMARK_NAME,
) -> go.Figure:
    figure = go.Figure()
    comparison = scored_data[["Close", "benchmark_close"]].dropna()

    if comparison.empty:
        figure.add_annotation(
            text="Benchmark comparison unavailable",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    else:
        stock_return = comparison["Close"] / comparison["Close"].iloc[0] - 1
        benchmark_return = comparison["benchmark_close"] / comparison["benchmark_close"].iloc[0] - 1
        figure.add_trace(
            go.Scatter(
                x=comparison.index,
                y=stock_return * 100,
                mode="lines",
                name=ticker,
                line=dict(color="#2563eb", width=2),
                hovertemplate="Date=%{x}<br>Return=%{y:.2f}%<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=comparison.index,
                y=benchmark_return * 100,
                mode="lines",
                name=benchmark_name,
                line=dict(color="#0f766e", width=2),
                hovertemplate="Date=%{x}<br>Return=%{y:.2f}%<extra></extra>",
            )
        )

    figure.update_layout(
        title=f"{ticker} cumulative return vs {benchmark_name}",
        xaxis_title="Date",
        yaxis_title="Cumulative return (%)",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def build_reason_distribution_chart(ranked_results: pd.DataFrame) -> go.Figure:
    reason_counts = ranked_results["reason"].value_counts().sort_values(ascending=True)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=reason_counts.values,
            y=reason_counts.index,
            orientation="h",
            marker_color="#0f766e",
            hovertemplate="%{y}<br>Stocks=%{x}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Reason distribution",
        xaxis_title="Stocks",
        height=330,
        margin=dict(l=20, r=20, t=55, b=20),
        template="plotly_white",
        showlegend=False,
    )
    return figure
