from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from alphawatch.config import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    MIN_TRAINING_ROWS,
    MODEL_CONTAMINATION,
    MODEL_ESTIMATORS,
    MODEL_RANDOM_STATE,
    RISK_THRESHOLDS,
)


def fit_isolation_model(training_features: pd.DataFrame) -> Tuple[RobustScaler, IsolationForest, np.ndarray]:
    scaler = RobustScaler()
    scaled_training = scaler.fit_transform(training_features)
    model = IsolationForest(
        n_estimators=MODEL_ESTIMATORS,
        contamination=MODEL_CONTAMINATION,
        max_samples=min(128, len(training_features)),
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(scaled_training)
    return scaler, model, scaled_training


def percentile_anomaly_score(training_raw_scores: np.ndarray, current_raw_score: float) -> float:
    """Map a raw anomaly score to 0-100 using only the training distribution."""
    if training_raw_scores.size == 0 or not np.isfinite(current_raw_score):
        return np.nan
    return float(np.mean(training_raw_scores <= current_raw_score) * 100)


def bounded_score(value: float) -> float:
    if pd.isna(value) or not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0, 100))


def robust_zscores(training_features: pd.DataFrame, current_features: pd.Series) -> pd.Series:
    median = training_features.median()
    mad = (training_features - median).abs().median()
    scaled_mad = (1.4826 * mad).replace(0, np.nan)
    zscores = ((current_features - median) / scaled_mad).abs()
    zscores = zscores.astype(float)
    return zscores.mask(~np.isfinite(zscores), np.nan).fillna(0.0)


def robust_feature_score_from_zscores(zscores: pd.Series) -> float:
    top_zscores = zscores.sort_values(ascending=False).head(5).clip(upper=8)
    if top_zscores.empty:
        return 0.0
    intensity = float(top_zscores.mean())
    return bounded_score(100 * (1 - np.exp(-intensity / 2.75)))


def relative_market_signal(row: pd.Series) -> float:
    if not bool(row.get("has_benchmark_context", False)):
        return 0.0
    components = [
        abs(row.get("excess_return_zscore_20", 0.0)),
        abs(row.get("relative_7d_return", 0.0)) * 20,
        abs(row.get("excess_return", 0.0)) * 80,
    ]
    return float(np.nanmax(components))


def relative_market_score(training: pd.DataFrame, current_row: pd.Series) -> float:
    if not bool(current_row.get("has_benchmark_context", False)):
        return 0.0

    training_signal = pd.concat(
        [
            training["excess_return_zscore_20"].abs(),
            training["relative_7d_return"].abs() * 20,
            training["excess_return"].abs() * 80,
        ],
        axis=1,
    ).max(axis=1)
    training_signal = training_signal.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    current_signal = relative_market_signal(current_row)

    if training_signal.size == 0:
        return bounded_score(100 * (1 - np.exp(-current_signal / 2.5)))
    return percentile_anomaly_score(training_signal, current_signal)


def top_feature_drivers(zscores: pd.Series, limit: int = 5) -> List[Tuple[str, float]]:
    sorted_zscores = zscores.sort_values(ascending=False).head(limit)
    return [
        (feature, float(score))
        for feature, score in sorted_zscores.items()
        if np.isfinite(score) and score > 0
    ]


def format_top_drivers(drivers: List[Tuple[str, float]]) -> str:
    if not drivers:
        return "No dominant driver"
    return "; ".join(
        f"{FEATURE_LABELS.get(feature, feature)} ({score:.1f}z)"
        for feature, score in drivers
    )


def primary_driver_from_features(row: pd.Series, drivers: List[Tuple[str, float]]) -> str:
    if not drivers:
        return "Multi-factor"

    top_feature = drivers[0][0]
    if top_feature in {"volume_zscore", "volume_ratio_20"}:
        return "Volume expansion"
    if top_feature in {"rolling_14d_volatility", "volatility_zscore_60", "atr_14_percent", "intraday_range"}:
        return "Volatility expansion"
    if top_feature in {"daily_return", "return_zscore_20", "price_gap"}:
        return "Abnormal price move"
    if top_feature in {"rsi_14", "rsi_change_3d", "rolling_7d_return"}:
        return "Momentum shift"
    if top_feature in {"distance_from_20dma", "distance_from_50dma", "drawdown_from_60d_high"}:
        return "Trend distance"
    if top_feature in {"excess_return", "relative_7d_return", "excess_return_zscore_20"}:
        return "Market-relative move"

    if abs(row.get("excess_return_zscore_20", 0)) >= 1.5:
        return "Market-relative move"
    return "Multi-factor"


def classify_risk(anomaly_score: float) -> str:
    if pd.isna(anomaly_score):
        return "Normal"
    if anomaly_score >= RISK_THRESHOLDS["Extreme"]:
        return "Extreme"
    if anomaly_score >= RISK_THRESHOLDS["High"]:
        return "High"
    if anomaly_score >= RISK_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Normal"


def classify_signal_direction(row: pd.Series) -> str:
    bullish_votes = 0
    bearish_votes = 0

    vote_rules = [
        ("daily_return", 0.015, -0.015),
        ("price_gap", 0.015, -0.015),
        ("distance_from_20dma", 0.03, -0.03),
        ("excess_return", 0.012, -0.012),
        ("relative_7d_return", 0.035, -0.035),
    ]
    for feature, bullish_threshold, bearish_threshold in vote_rules:
        value = row.get(feature, 0)
        if value > bullish_threshold:
            bullish_votes += 1
        elif value < bearish_threshold:
            bearish_votes += 1

    if row.get("rsi_14", 50) >= 60:
        bullish_votes += 1
    elif row.get("rsi_14", 50) <= 40:
        bearish_votes += 1

    if bullish_votes >= bearish_votes + 2:
        return "Bullish"
    if bearish_votes >= bullish_votes + 2:
        return "Bearish"
    return "Mixed"


def classify_confidence(
    final_score: float,
    isolation_score: float,
    robust_feature_score: float,
    relative_score: float,
) -> str:
    components = np.array([isolation_score, robust_feature_score, relative_score], dtype=float)
    strong_components = int(np.sum(components >= 70))
    medium_components = int(np.sum(components >= 55))

    if final_score >= 80 and strong_components >= 2:
        return "High"
    if final_score >= 60 and medium_components >= 2:
        return "Medium"
    return "Low"


def explain_anomaly(row: pd.Series) -> str:
    parts: List[str] = []

    if row.get("volume_zscore", 0) >= 2.0 or row.get("volume_ratio_20", 0) >= 2.0:
        parts.append("Volume spike")
    if row.get("volatility_zscore_60", 0) >= 1.5 or row.get("atr_14_percent", 0) >= 0.04:
        parts.append("High volatility")
    if abs(row.get("price_gap", 0)) >= 0.03:
        parts.append("Large price gap")

    daily_return = row.get("daily_return", 0)
    return_zscore = row.get("return_zscore_20", 0)
    if daily_return >= 0.035 or return_zscore >= 2:
        parts.append("Abnormal upside return")
    elif daily_return <= -0.035 or return_zscore <= -2:
        parts.append("Abnormal downside return")

    weekly_return = row.get("rolling_7d_return", 0)
    rsi = row.get("rsi_14", 50)
    if weekly_return >= 0.08 and rsi >= 60:
        parts.append("Momentum breakout")
    elif weekly_return <= -0.08 and rsi <= 40:
        parts.append("Momentum breakdown")

    if max(abs(row.get("distance_from_20dma", 0)), abs(row.get("distance_from_50dma", 0))) >= 0.08:
        parts.append("Price far from moving average")
    elif row.get("drawdown_from_60d_high", 0) <= -0.12:
        parts.append("Drawdown pressure")

    excess_zscore = row.get("excess_return_zscore_20", 0)
    relative_7d = row.get("relative_7d_return", 0)
    if excess_zscore >= 1.5 or relative_7d >= 0.04:
        parts.append("Market-relative strength")
    elif excess_zscore <= -1.5 or relative_7d <= -0.04:
        parts.append("Market-relative weakness")

    if not parts:
        return "Multi-factor anomaly"
    return " + ".join(dict.fromkeys(parts).keys())


def score_current_row(
    training: pd.DataFrame,
    current: pd.DataFrame,
) -> Dict[str, object]:
    scaler, model, scaled_training = fit_isolation_model(training[FEATURE_COLUMNS])
    scaled_current = scaler.transform(current[FEATURE_COLUMNS])

    training_raw_scores = -model.decision_function(scaled_training)
    current_raw_score = float(-model.decision_function(scaled_current)[0])
    isolation_score = bounded_score(percentile_anomaly_score(training_raw_scores, current_raw_score))

    current_row = current.iloc[0]
    zscores = robust_zscores(training[FEATURE_COLUMNS], current_row[FEATURE_COLUMNS])
    robust_score = robust_feature_score_from_zscores(zscores)
    market_score = bounded_score(relative_market_score(training, current_row))
    final_score = bounded_score(0.60 * isolation_score + 0.25 * robust_score + 0.15 * market_score)

    drivers = top_feature_drivers(zscores)
    primary_driver = primary_driver_from_features(current_row, drivers)
    confidence_level = classify_confidence(final_score, isolation_score, robust_score, market_score)
    is_model_anomaly = bool(model.predict(scaled_current)[0] == -1)

    return {
        "raw_anomaly_score": current_raw_score,
        "isolation_score": isolation_score,
        "robust_feature_score": robust_score,
        "relative_market_score": market_score,
        "final_anomaly_score": final_score,
        "anomaly_score": final_score,
        "is_anomaly": bool(is_model_anomaly or final_score >= 85),
        "risk_level": classify_risk(final_score),
        "signal_direction": classify_signal_direction(current_row),
        "confidence_level": confidence_level,
        "primary_driver": primary_driver,
        "driver_details": (
            f"{primary_driver}; isolation={isolation_score:.1f}, "
            f"robust={robust_score:.1f}, market-relative={market_score:.1f}"
        ),
        "top_feature_drivers": format_top_drivers(drivers),
        "reason": explain_anomaly(current_row),
    }


def score_latest_rankings(
    featured_data: Dict[str, pd.DataFrame],
    lookback_period: str,
) -> pd.DataFrame:
    """Score the latest available row for each ticker without training on that row."""
    latest_rows: List[dict] = []
    last_updated = datetime.now().isoformat(timespec="seconds")

    for ticker, data in featured_data.items():
        valid = data.dropna(subset=FEATURE_COLUMNS).copy()
        if len(valid) <= MIN_TRAINING_ROWS:
            continue

        training = valid.iloc[:-1]
        current = valid.iloc[[-1]]
        if len(training) < MIN_TRAINING_ROWS:
            continue

        row = current.iloc[0]
        score_payload = score_current_row(training, current)

        latest_row = {
            "rank": 0,
            "ticker": ticker,
            "date": row.name.date().isoformat(),
            "close": float(row["Close"]),
            "benchmark_close": float(row["benchmark_close"]) if pd.notna(row.get("benchmark_close")) else np.nan,
            "has_benchmark_context": bool(row.get("has_benchmark_context", False)),
            "lookback_period": lookback_period,
            "data_rows": int(len(data)),
            "last_updated": last_updated,
        }
        for feature in FEATURE_COLUMNS + ["benchmark_return", "benchmark_7d_return"]:
            latest_row[feature] = float(row.get(feature, np.nan))
        latest_row.update(score_payload)
        latest_rows.append(latest_row)

    if not latest_rows:
        return pd.DataFrame()

    ranked = pd.DataFrame(latest_rows)
    ranked = ranked.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def score_historical_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """Score each historical row with an expanding, past-only training window."""
    scored = data.copy()
    for column, default in {
        "raw_anomaly_score": np.nan,
        "isolation_score": np.nan,
        "robust_feature_score": np.nan,
        "relative_market_score": np.nan,
        "final_anomaly_score": np.nan,
        "anomaly_score": np.nan,
        "is_anomaly": False,
        "risk_level": "Normal",
        "signal_direction": "Mixed",
        "confidence_level": "Low",
        "primary_driver": "Not enough history",
        "driver_details": "Not enough history",
        "top_feature_drivers": "Not enough history",
        "reason": "Not enough history",
    }.items():
        scored[column] = default

    valid = scored.dropna(subset=FEATURE_COLUMNS).copy()
    if len(valid) <= MIN_TRAINING_ROWS:
        return scored

    score_columns = [
        "raw_anomaly_score",
        "isolation_score",
        "robust_feature_score",
        "relative_market_score",
        "final_anomaly_score",
        "anomaly_score",
        "is_anomaly",
        "risk_level",
        "signal_direction",
        "confidence_level",
        "primary_driver",
        "driver_details",
        "top_feature_drivers",
        "reason",
    ]

    for position in range(MIN_TRAINING_ROWS, len(valid)):
        training = valid.iloc[:position]
        current = valid.iloc[[position]]
        current_index = current.index[0]
        score_payload = score_current_row(training, current)
        for column in score_columns:
            scored.loc[current_index, column] = score_payload[column]

    return scored
