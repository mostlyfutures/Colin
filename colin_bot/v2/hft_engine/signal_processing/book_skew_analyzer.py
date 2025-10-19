"""
Order Book Skew Analyzer

Calculates order book skew using log transformation and dynamic thresholding.
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from collections import deque
import logging

from ..utils.data_structures import OrderBook, BookSkewSignal, SignalDirection
from ..utils.math_utils import calculate_zscore, moving_average, exponential_moving_average
from ..utils.performance import profile_async_hft_operation, LatencyTracker


class BookSkewAnalyzer:
    """
    Analyze order book skew to identify trading opportunities.

    Skew is calculated as: log10(bid_size) - log10(ask_size)
    Positive skew indicates buying pressure, negative skew indicates selling pressure.
    """

    def __init__(self, window_size: int = 50, sensitivity: float = 0.1):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)

        # Data storage
        self.skew_history: deque = deque(maxlen=window_size)
        self.bid_size_history: deque = deque(maxlen=window_size)
        self.ask_size_history: deque = deque(maxlen=window_size)
        self.spread_history: deque = deque(maxlen=window_size)

        # Dynamic threshold parameters
        self.base_threshold = 0.1
        self.dynamic_threshold = self.base_threshold
        self.threshold_adjustment_factor = 0.1
        self.volatility_window = 20

        # Signal generation parameters
        self.min_threshold = 0.05
        self.max_threshold = 0.5
        self.confirmation_periods = 3
        self.ema_alpha = 0.2

        # Performance tracking
        self.calculation_count = 0
        self.signal_count = 0

    def process_order_book(self, order_book: OrderBook):
        """
        Process order book update for skew analysis.

        Args:
            order_book: Current order book
        """
        # Calculate skew
        skew = self._calculate_skew(order_book)
        self.skew_history.append(skew)

        # Store component data
        self.bid_size_history.append(order_book.total_bid_size)
        self.ask_size_history.append(order_book.total_ask_size)

        # Store spread
        if order_book.spread:
            self.spread_history.append(order_book.spread)

        # Update dynamic threshold
        self._update_dynamic_threshold()

        self.calculation_count += 1

    def _calculate_skew(self, order_book: OrderBook) -> float:
        """
        Calculate order book skew using log transformation.

        Args:
            order_book: Order book to analyze

        Returns:
            Skew value
        """
        if not order_book.bids or not order_book.asks:
            return 0.0

        total_bid_size = order_book.total_bid_size
        total_ask_size = order_book.total_ask_size

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        bid_log = math.log10(total_bid_size + epsilon)
        ask_log = math.log10(total_ask_size + epsilon)

        return bid_log - ask_log

    def _update_dynamic_threshold(self):
        """Update dynamic threshold based on recent volatility."""
        if len(self.skew_history) < self.volatility_window:
            return

        # Calculate recent volatility of skew
        recent_skew = list(self.skew_history)[-self.volatility_window:]
        if len(recent_skew) < 2:
            return

        volatility = np.std(recent_skew)

        # Adjust threshold based on volatility
        # Higher volatility = higher threshold (more selective)
        volatility_adjustment = volatility * self.threshold_adjustment_factor

        # Calculate new threshold
        new_threshold = self.base_threshold + volatility_adjustment

        # Apply smoothing (EMA)
        self.dynamic_threshold = (
            self.ema_alpha * new_threshold +
            (1 - self.ema_alpha) * self.dynamic_threshold
        )

        # Ensure threshold stays within bounds
        self.dynamic_threshold = max(
            self.min_threshold,
            min(self.max_threshold, self.dynamic_threshold)
        )

    @profile_async_hft_operation("book_skew_analysis", LatencyTracker())
    async def analyze_skew(self, symbol: str, order_book: OrderBook) -> Optional[BookSkewSignal]:
        """
        Analyze order book skew and generate trading signal.

        Args:
            symbol: Trading symbol
            order_book: Current order book

        Returns:
            Book skew signal or None if insufficient data
        """
        if len(self.skew_history) < 10:
            return None

        try:
            # Get current skew
            current_skew = self.skew_history[-1] if self.skew_history else 0.0

            # Calculate signal direction
            signal_direction = self._determine_signal_direction(current_skew)

            # Calculate confidence
            confidence = self._calculate_confidence(current_skew, order_book)

            # Create signal
            signal = BookSkewSignal(
                symbol=symbol,
                skew_value=current_skew,
                threshold=self.dynamic_threshold,
                signal_direction=signal_direction,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                bid_size=order_book.total_bid_size,
                ask_size=order_book.total_ask_size
            )

            self.signal_count += 1
            return signal

        except Exception as e:
            self.logger.error(f"Error analyzing book skew for {symbol}: {e}")
            return None

    def _determine_signal_direction(self, skew_value: float) -> SignalDirection:
        """
        Determine signal direction based on skew value.

        Args:
            skew_value: Current skew value

        Returns:
            Signal direction
        """
        if skew_value > self.dynamic_threshold:
            return SignalDirection.LONG
        elif skew_value < -self.dynamic_threshold:
            return SignalDirection.SHORT
        else:
            return SignalDirection.HOLD

    def _calculate_confidence(self, skew_value: float, order_book: OrderBook) -> float:
        """
        Calculate confidence level for skew signal.

        Args:
            skew_value: Current skew value
            order_book: Current order book

        Returns:
            Confidence level (0.0 to 1.0)
        """
        # Base confidence on skew magnitude relative to threshold
        magnitude_ratio = abs(skew_value) / self.dynamic_threshold
        base_confidence = min(magnitude_ratio, 1.0)

        # Adjust based on order book depth
        total_size = order_book.total_bid_size + order_book.total_ask_size
        depth_factor = min(total_size / 10000.0, 1.0)  # Normalize to 10k size
        base_confidence *= depth_factor

        # Adjust based on consistency (if skew is persistent)
        if len(self.skew_history) >= self.confirmation_periods:
            recent_skews = list(self.skew_history)[-self.confirmation_periods:]
            consistency = self._calculate_consistency(recent_skews, skew_value)
            base_confidence *= (0.5 + 0.5 * consistency)

        # Adjust based on spread (tighter spread = higher confidence)
        if len(self.spread_history) > 0:
            avg_spread = np.mean(list(self.spread_history)[-10:])
            if order_book.mid_price and avg_spread > 0:
                spread_bps = (avg_spread / order_book.mid_price) * 10000
                spread_factor = max(0.1, 1.0 - spread_bps / 100.0)  # Reduce confidence for wide spreads
                base_confidence *= spread_factor

        # Ensure confidence is in valid range
        return max(0.0, min(1.0, base_confidence))

    def _calculate_consistency(self, skew_values: List[float], current_skew: float) -> float:
        """
        Calculate consistency of skew direction.

        Args:
            skew_values: Recent skew values
            current_skew: Current skew value

        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not skew_values:
            return 0.0

        # Count values with same sign as current skew
        same_sign_count = 0
        for value in skew_values:
            if (current_skew > 0 and value > 0) or (current_skew < 0 and value < 0):
                same_sign_count += 1

        return same_sign_count / len(skew_values)

    def get_skew_statistics(self) -> Dict[str, float]:
        """
        Get statistics on skew calculations.

        Returns:
            Dictionary of skew statistics
        """
        if len(self.skew_history) < 2:
            return {}

        skew_values = list(self.skew_history)

        return {
            'current_skew': skew_values[-1],
            'mean_skew': np.mean(skew_values),
            'std_skew': np.std(skew_values),
            'min_skew': min(skew_values),
            'max_skew': max(skew_values),
            'dynamic_threshold': self.dynamic_threshold,
            'z_score': calculate_zscore([skew_values[-1]], np.mean(skew_values), np.std(skew_values))[0] if len(skew_values) > 1 else 0.0
        }

    def get_liquidity_metrics(self) -> Dict[str, float]:
        """
        Get liquidity-related metrics.

        Returns:
            Dictionary of liquidity metrics
        """
        if not self.bid_size_history or not self.ask_size_history:
            return {}

        recent_bid_sizes = list(self.bid_size_history)[-10:]
        recent_ask_sizes = list(self.ask_size_history)[-10:]

        return {
            'avg_bid_size': np.mean(recent_bid_sizes),
            'avg_ask_size': np.mean(recent_ask_sizes),
            'bid_size_volatility': np.std(recent_bid_sizes) if len(recent_bid_sizes) > 1 else 0.0,
            'ask_size_volatility': np.std(recent_ask_sizes) if len(recent_ask_sizes) > 1 else 0.0,
            'size_imbalance': abs(np.mean(recent_bid_sizes) - np.mean(recent_ask_sizes)) / (np.mean(recent_bid_sizes) + np.mean(recent_ask_sizes)) if (np.mean(recent_bid_sizes) + np.mean(recent_ask_sizes)) > 0 else 0.0
        }

    def get_recent_signals(self, symbol: str, count: int = 10) -> List[Dict[str, any]]:
        """
        Get recent skew signals.

        Args:
            symbol: Trading symbol
            count: Number of recent signals to return

        Returns:
            List of recent signals
        """
        signals = []
        if len(self.skew_history) < count:
            return signals

        recent_skews = list(self.skew_history)[-count:]
        recent_bid_sizes = list(self.bid_size_history)[-count:]
        recent_ask_sizes = list(self.ask_size_history)[-count:]

        for i, skew in enumerate(recent_skews):
            signal_direction = self._determine_signal_direction(skew)
            confidence = self._calculate_confidence_for_history(skew, i)

            signals.append({
                'timestamp': datetime.now(timezone.utc) - timedelta(seconds=count - i),
                'skew_value': skew,
                'direction': signal_direction.value,
                'confidence': confidence,
                'threshold': self.dynamic_threshold,
                'bid_size': recent_bid_sizes[i] if i < len(recent_bid_sizes) else 0.0,
                'ask_size': recent_ask_sizes[i] if i < len(recent_ask_sizes) else 0.0
            })

        return signals

    def _calculate_confidence_for_history(self, skew_value: float, index: int) -> float:
        """Calculate confidence for historical data point."""
        magnitude_ratio = abs(skew_value) / self.dynamic_threshold
        return min(magnitude_ratio, 1.0)

    def get_threshold_evolution(self) -> List[Tuple[datetime, float]]:
        """
        Get evolution of dynamic threshold over time.

        Returns:
            List of (timestamp, threshold) tuples
        """
        # This is a simplified version - real implementation would store timestamp history
        evolution = []
        current_time = datetime.now(timezone.utc)

        # Generate mock evolution data
        for i in range(min(100, len(self.skew_history))):
            timestamp = current_time - timedelta(seconds=i)
            # Simulate threshold changes
            threshold_variation = 0.05 * math.sin(i * 0.1)
            threshold = self.dynamic_threshold + threshold_variation
            evolution.append((timestamp, threshold))

        return evolution

    def reset(self):
        """Reset analyzer state."""
        self.skew_history.clear()
        self.bid_size_history.clear()
        self.ask_size_history.clear()
        self.spread_history.clear()
        self.dynamic_threshold = self.base_threshold
        self.calculation_count = 0
        self.signal_count = 0

    def get_status(self) -> Dict[str, any]:
        """Get analyzer status."""
        return {
            'calculations_performed': self.calculation_count,
            'signals_generated': self.signal_count,
            'window_size': self.window_size,
            'current_threshold': self.dynamic_threshold,
            'base_threshold': self.base_threshold,
            'sensitivity': self.sensitivity,
            'skew_history_length': len(self.skew_history)
        }

    def set_parameters(self, **kwargs):
        """
        Update analyzer parameters.

        Args:
            **kwargs: Parameters to update
        """
        if 'window_size' in kwargs:
            new_window_size = kwargs['window_size']
            # Resize deques
            if new_window_size != self.window_size:
                self.skew_history = deque(
                    list(self.skew_history)[-new_window_size:],
                    maxlen=new_window_size
                )
                self.bid_size_history = deque(
                    list(self.bid_size_history)[-new_window_size:],
                    maxlen=new_window_size
                )
                self.ask_size_history = deque(
                    list(self.ask_size_history)[-new_window_size:],
                    maxlen=new_window_size
                )
                self.spread_history = deque(
                    list(self.spread_history)[-new_window_size:],
                    maxlen=new_window_size
                )
                self.window_size = new_window_size

        if 'sensitivity' in kwargs:
            self.sensitivity = kwargs['sensitivity']
            self.threshold_adjustment_factor = kwargs['sensitivity']

        if 'base_threshold' in kwargs:
            self.base_threshold = kwargs['base_threshold']

        if 'ema_alpha' in kwargs:
            self.ema_alpha = kwargs['ema_alpha']