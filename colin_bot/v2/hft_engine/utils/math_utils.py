"""
Mathematical utilities for HFT calculations.

Core mathematical functions and algorithms for high-frequency trading operations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import math


def hawkes_process(intensity_func: np.ndarray, decay_rate: float = 0.5) -> np.ndarray:
    """
    Calculate Hawkes process for order flow analysis.

    Hawkes process is a self-exciting point process used to model
    order flow clustering and intensification in financial markets.

    Args:
        intensity_func: Array of intensity values over time
        decay_rate: Rate at which intensity decays

    Returns:
        Array of Hawkes process intensity values
    """
    n = len(intensity_func)
    hawkes_intensity = np.zeros(n)

    # Base intensity
    base_intensity = np.mean(intensity_func)

    for i in range(n):
        # Add base intensity
        hawkes_intensity[i] = base_intensity

        # Add excitation from previous events
        for j in range(max(0, i - 100), i):  # Look back 100 events
            decay_factor = np.exp(-decay_rate * (i - j))
            hawkes_intensity[i] += intensity_func[j] * decay_factor

    return hawkes_intensity


def calculate_skew(bid_sizes: List[float], ask_sizes: List[float]) -> float:
    """
    Calculate order book skew using log transformation.

    Skew = log10(bid_size) - log10(ask_size)

    Args:
        bid_sizes: List of bid sizes
        ask_sizes: List of ask sizes

    Returns:
        Skew value
    """
    total_bid_size = sum(bid_sizes) if bid_sizes else 0.001
    total_ask_size = sum(ask_sizes) if ask_sizes else 0.001

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    bid_log = np.log10(total_bid_size + epsilon)
    ask_log = np.log10(total_ask_size + epsilon)

    return bid_log - ask_log


def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average with specified window.

    Args:
        data: Input data series
        window: Window size for moving average

    Returns:
        List of moving average values
    """
    if window <= 0:
        raise ValueError("Window size must be positive")

    if len(data) < window:
        return [np.mean(data)] * len(data)

    ma = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        ma.append(np.mean(data[start_idx:i + 1]))

    return ma


def exponential_moving_average(data: List[float], alpha: float = 0.2) -> List[float]:
    """
    Calculate exponential moving average.

    Args:
        data: Input data series
        alpha: Smoothing factor (0 < alpha < 1)

    Returns:
        List of EMA values
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    if not data:
        return []

    ema = [data[0]]
    for i in range(1, len(data)):
        ema_value = alpha * data[i] + (1 - alpha) * ema[i-1]
        ema.append(ema_value)

    return ema


def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """
    Calculate rolling volatility using standard deviation of returns.

    Args:
        prices: List of prices
        window: Window size for calculation

    Returns:
        Volatility value
    """
    if len(prices) < 2:
        return 0.0

    # Calculate returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

    if not returns:
        return 0.0

    # Calculate rolling volatility
    if len(returns) < window:
        return np.std(returns)

    recent_returns = returns[-window:]
    return np.std(recent_returns)


def calculate_correlation(series1: List[float], series2: List[float]) -> float:
    """
    Calculate correlation between two series.

    Args:
        series1: First data series
        series2: Second data series

    Returns:
        Correlation coefficient
    """
    if len(series1) != len(series2) or len(series1) < 2:
        return 0.0

    try:
        correlation = np.corrcoef(series1, series2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except:
        return 0.0


def calculate_zscore(data: List[float], window: int = 20) -> List[float]:
    """
    Calculate Z-scores for data series.

    Args:
        data: Input data series
        window: Window size for calculation

    Returns:
        List of Z-scores
    """
    if len(data) < 2:
        return [0.0] * len(data)

    zscores = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i + 1]

        if len(window_data) < 2:
            zscores.append(0.0)
            continue

        mean = np.mean(window_data)
        std = np.std(window_data)

        if std == 0:
            zscores.append(0.0)
        else:
            zscore = (data[i] - mean) / std
            zscores.append(zscore)

    return zscores


def normalize_signal_strength(signal: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """
    Normalize signal strength to specified range.

    Args:
        signal: Signal value to normalize
        min_val: Minimum value in output range
        max_val: Maximum value in output range

    Returns:
        Normalized signal value
    """
    # Clip signal to reasonable range
    signal = np.clip(signal, -5.0, 5.0)

    # Normalize to [0, 1]
    normalized = (signal + 5.0) / 10.0

    # Scale to desired range
    return min_val + normalized * (max_val - min_val)


def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average of values.

    Args:
        values: List of values
        weights: List of weights

    Returns:
        Weighted average
    """
    if len(values) != len(weights) or not values:
        return 0.0

    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(values)

    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight


def calculate_percentile(data: List[float], percentile: float) -> float:
    """
    Calculate percentile of data series.

    Args:
        data: Input data series
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value
    """
    if not data:
        return 0.0

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    return np.percentile(data, percentile)


def linear_regression(y: List[float], x: List[float] = None) -> Tuple[float, float]:
    """
    Calculate linear regression coefficients.

    Args:
        y: Dependent variable
        x: Independent variable (defaults to range)

    Returns:
        Tuple of (slope, intercept)
    """
    if x is None:
        x = list(range(len(y)))

    if len(y) != len(x) or len(y) < 2:
        return 0.0, 0.0

    try:
        coefficients = np.polyfit(x, y, 1)
        return coefficients[0], coefficients[1]  # slope, intercept
    except:
        return 0.0, 0.0


def calculate_rsi(prices: List[float], window: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: List of prices
        window: Window size for RSI calculation

    Returns:
        RSI value (0-100)
    """
    if len(prices) < window + 1:
        return 50.0  # Neutral

    # Calculate price changes
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    # Calculate average gains and losses
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: List of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        Dictionary with MACD, signal, and histogram values
    """
    if len(prices) < slow:
        return {
            'macd': [0.0] * len(prices),
            'signal': [0.0] * len(prices),
            'histogram': [0.0] * len(prices)
        }

    # Calculate EMAs
    fast_ema = exponential_moving_average(prices, 2.0 / (fast + 1))
    slow_ema = exponential_moving_average(prices, 2.0 / (slow + 1))

    # Calculate MACD line
    macd = [f - s for f, s in zip(fast_ema, slow_ema)]

    # Calculate signal line
    signal_ema = exponential_moving_average(macd, 2.0 / (signal + 1))

    # Calculate histogram
    histogram = [m - s for m, s in zip(macd, signal_ema)]

    return {
        'macd': macd,
        'signal': signal_ema,
        'histogram': histogram
    }


def smooth_signal(signal: List[float], method: str = 'sma', window: int = 5) -> List[float]:
    """
    Smooth signal using specified method.

    Args:
        signal: Input signal
        method: Smoothing method ('sma', 'ema', 'median')
        window: Window size for smoothing

    Returns:
        Smoothed signal
    """
    if not signal:
        return []

    if method == 'sma':
        return moving_average(signal, window)
    elif method == 'ema':
        return exponential_moving_average(signal, 2.0 / (window + 1))
    elif method == 'median':
        from collections import deque
        smoothed = []
        window_vals = deque(maxlen=window)

        for val in signal:
            window_vals.append(val)
            smoothed.append(np.median(window_vals))

        return smoothed
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def calculate_entropy(probabilities: List[float]) -> float:
    """
    Calculate Shannon entropy of probability distribution.

    Args:
        probabilities: List of probabilities (should sum to 1)

    Returns:
        Entropy value
    """
    if not probabilities:
        return 0.0

    # Normalize probabilities
    total = sum(probabilities)
    if total == 0:
        return 0.0

    probs = [p / total for p in probabilities if p > 0]

    entropy = -sum(p * np.log2(p) for p in probs)
    return entropy


def distance_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Calculate distance matrix for a set of points.

    Args:
        points: List of (x, y) coordinate tuples

    Returns:
        Distance matrix
    """
    n = len(points)
    if n == 0:
        return np.array([])

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = points[i]
                x2, y2 = points[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                matrix[i][j] = distance

    return matrix