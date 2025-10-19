"""
Circuit Breaker System with Market Stress Detection

Implements comprehensive circuit breaker mechanisms to protect against
extreme market conditions, system failures, and anomalous trading activity.
"""

import asyncio
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import numpy as np

from ..utils.data_structures import TradingSignal, OrderBook, Trade
from ..utils.math_utils import calculate_volatility, calculate_zscore, exponential_moving_average
from ..utils.performance import LatencyTracker


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"             # Trading halted
    HALF_OPEN = "half_open"   # Testing market conditions
    TRIPPED = "tripped"       # Manually tripped


class StressLevel(Enum):
    """Market stress levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    SEVERE = "severe"
    CRITICAL = "critical"


class TriggerType(Enum):
    """Types of circuit breaker triggers."""
    VOLATILITY_SPIKE = "volatility_spike"
    PRICE_MOVEMENT = "price_movement"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEM_STRESS = "system_stress"
    VOLUME_ANOMALY = "volume_anomaly"
    MANUAL_OVERRIDE = "manual_override"
    CORRELATION_BREAKDOWN = "correlation_breakdown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker parameters."""
    # Volatility triggers
    volatility_threshold: float = 3.0              # 3x normal volatility
    volatility_window: int = 60                    # 60 second window

    # Price movement triggers
    price_movement_threshold: float = 0.05         # 5% price movement
    price_movement_window: int = 30                # 30 second window

    # Liquidity triggers
    liquidity_threshold: float = 0.3               # 30% of normal liquidity
    spread_threshold_bps: float = 50.0             # 50 bps spread threshold

    # System stress triggers
    latency_threshold_ms: float = 100.0            # 100ms latency threshold
    error_rate_threshold: float = 0.05             # 5% error rate threshold

    # Volume triggers
    volume_multiplier: float = 5.0                 # 5x normal volume
    volume_window: int = 60                        # 60 second window

    # Recovery parameters
    cooldown_period_seconds: int = 300             # 5 minute cooldown
    test_period_seconds: int = 60                  # 1 minute test period
    max_trips_per_hour: int = 3                    # Maximum trips per hour

    # Correlation monitoring
    correlation_threshold: float = 0.3             # Minimum correlation threshold
    correlation_window: int = 300                  # 5 minute correlation window


@dataclass
class MarketStressMetrics:
    """Market stress metrics."""
    timestamp: datetime
    volatility_score: float
    price_movement_score: float
    liquidity_score: float
    volume_score: float
    correlation_score: float
    system_health_score: float
    overall_stress_score: float
    stress_level: StressLevel
    trigger_conditions: List[str] = field(default_factory=list)


@dataclass
class CircuitBreakerEvent:
    """Circuit breaker event record."""
    timestamp: datetime
    trigger_type: TriggerType
    previous_state: CircuitBreakerState
    new_state: CircuitBreakerState
    stress_metrics: MarketStressMetrics
    reason: str
    duration_seconds: Optional[float] = None
    recovery_conditions: List[str] = field(default_factory=list)


class CircuitBreakerSystem:
    """
    Advanced circuit breaker system with market stress detection.

    Monitors market conditions, system health, and trading activity
    to automatically halt trading during extreme conditions.
    """

    def __init__(self, symbols: List[str], config: CircuitBreakerConfig = None):
        self.symbols = symbols
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(__name__)

        # Circuit breaker state
        self.state = CircuitBreakerState.CLOSED
        self.state_history: deque = deque(maxlen=1000)
        self.current_trip_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now(timezone.utc)

        # Market data storage
        self.price_history: Dict[str, deque] = {s: deque(maxlen=1000) for s in symbols}
        self.volume_history: Dict[str, deque] = {s: deque(maxlen=1000) for s in symbols}
        self.spread_history: Dict[str, deque] = {s: deque(maxlen=500) for s in symbols}
        self.order_book_depth: Dict[str, deque] = {s: deque(maxlen=200) for s in symbols}

        # System health monitoring
        self.latency_tracker = LatencyTracker()
        self.error_history: deque = deque(maxlen=1000)
        self.system_metrics: deque = deque(maxlen=500)

        # Correlation monitoring
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.correlation_history: deque = deque(maxlen=100)

        # Stress tracking
        self.stress_history: deque = deque(maxlen=200)
        self.current_stress_level = StressLevel.NORMAL

        # Circuit breaker events
        self.circuit_breaker_events: List[CircuitBreakerEvent] = []
        self.trip_count_per_hour: deque = deque(maxlen=24)  # Last 24 hours

        # Callbacks and subscribers
        self.state_change_callbacks: Set[Callable] = set()
        self.stress_alert_callbacks: Set[Callable] = set()

        # Monitoring tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    async def start_monitoring(self):
        """Start circuit breaker monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Circuit breaker monitoring started")

    async def stop_monitoring(self):
        """Stop circuit breaker monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Circuit breaker monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Calculate stress metrics
                stress_metrics = await self._calculate_stress_metrics()

                # Check for circuit breaker triggers
                await self._check_circuit_breaker_triggers(stress_metrics)

                # Update stress level
                await self._update_stress_level(stress_metrics)

                # Monitor correlations
                await self._monitor_correlations()

                # Check for recovery conditions
                if self.state != CircuitBreakerState.CLOSED:
                    await self._check_recovery_conditions(stress_metrics)

                # Sleep before next iteration
                await asyncio.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)

    async def process_order_book(self, order_book: OrderBook):
        """Process order book for circuit breaker monitoring."""
        symbol = order_book.symbol
        timestamp = order_book.timestamp

        # Store price data
        if order_book.mid_price:
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': order_book.mid_price
            })

        # Store spread data
        if order_book.spread:
            self.spread_history[symbol].append({
                'timestamp': timestamp,
                'spread': order_book.spread,
                'spread_bps': (order_book.spread / order_book.mid_price) * 10000 if order_book.mid_price else 0
            })

        # Store order book depth
        total_size = order_book.total_bid_size + order_book.total_ask_size
        self.order_book_depth[symbol].append({
            'timestamp': timestamp,
            'total_size': total_size,
            'bid_size': order_book.total_bid_size,
            'ask_size': order_book.total_ask_size
        })

        # Check for immediate triggers
        await self._check_immediate_triggers(symbol, order_book)

    async def process_trade(self, trade: Trade):
        """Process trade for circuit breaker monitoring."""
        symbol = trade.symbol
        timestamp = trade.timestamp

        # Store volume data
        self.volume_history[symbol].append({
            'timestamp': timestamp,
            'volume': trade.size,
            'price': trade.price
        })

    async def record_system_metric(self, metric_name: str, value: float, is_error: bool = False):
        """Record system metric for health monitoring."""
        timestamp = datetime.now(timezone.utc)

        self.system_metrics.append({
            'timestamp': timestamp,
            'metric': metric_name,
            'value': value,
            'is_error': is_error
        })

        if is_error:
            self.error_history.append(timestamp)

    async def _calculate_stress_metrics(self) -> MarketStressMetrics:
        """Calculate comprehensive market stress metrics."""
        timestamp = datetime.now(timezone.utc)

        # Calculate individual stress components
        volatility_score = await self._calculate_volatility_stress()
        price_movement_score = await self._calculate_price_movement_stress()
        liquidity_score = await self._calculate_liquidity_stress()
        volume_score = await self._calculate_volume_stress()
        correlation_score = await self._calculate_correlation_stress()
        system_health_score = await self._calculate_system_health_stress()

        # Calculate overall stress score
        overall_stress_score = (
            volatility_score * 0.25 +
            price_movement_score * 0.20 +
            liquidity_score * 0.20 +
            volume_score * 0.15 +
            correlation_score * 0.10 +
            system_health_score * 0.10
        )

        # Determine stress level
        stress_level = self._determine_stress_level(overall_stress_score)

        # Identify trigger conditions
        trigger_conditions = []
        if volatility_score > 0.7:
            trigger_conditions.append(f"High volatility ({volatility_score:.2f})")
        if price_movement_score > 0.7:
            trigger_conditions.append(f"Extreme price movement ({price_movement_score:.2f})")
        if liquidity_score > 0.7:
            trigger_conditions.append(f"Liquidity crisis ({liquidity_score:.2f})")
        if volume_score > 0.7:
            trigger_conditions.append(f"Volume anomaly ({volume_score:.2f})")
        if correlation_score > 0.7:
            trigger_conditions.append(f"Correlation breakdown ({correlation_score:.2f})")
        if system_health_score > 0.7:
            trigger_conditions.append(f"System stress ({system_health_score:.2f})")

        return MarketStressMetrics(
            timestamp=timestamp,
            volatility_score=volatility_score,
            price_movement_score=price_movement_score,
            liquidity_score=liquidity_score,
            volume_score=volume_score,
            correlation_score=correlation_score,
            system_health_score=system_health_score,
            overall_stress_score=overall_stress_score,
            stress_level=stress_level,
            trigger_conditions=trigger_conditions
        )

    async def _calculate_volatility_stress(self) -> float:
        """Calculate volatility-based stress score."""
        if not any(self.price_history.values()):
            return 0.0

        volatilities = []
        for symbol in self.symbols:
            if len(self.price_history[symbol]) < 10:
                continue

            prices = [entry['price'] for entry in list(self.price_history[symbol])[-60:]]  # Last 60 entries
            if len(prices) > 1:
                volatility = calculate_volatility(prices)
                volatilities.append(volatility)

        if not volatilities:
            return 0.0

        avg_volatility = np.mean(volatilities)

        # Normalize to 0-1 scale (assuming normal volatility around 1-2%)
        normalized_volatility = min(avg_volatility / 0.05, 1.0)  # 5% as max

        return normalized_volatility

    async def _calculate_price_movement_stress(self) -> float:
        """Calculate price movement-based stress score."""
        if not any(self.price_history.values()):
            return 0.0

        max_movement = 0.0

        for symbol in self.symbols:
            if len(self.price_history[symbol]) < 2:
                continue

            prices = [entry['price'] for entry in list(self.price_history[symbol])[-30:]]  # Last 30 entries
            if len(prices) > 1:
                price_change = abs(prices[-1] - prices[0]) / prices[0]
                max_movement = max(max_movement, price_change)

        # Normalize to 0-1 scale (5% movement as max)
        normalized_movement = min(max_movement / 0.05, 1.0)

        return normalized_movement

    async def _calculate_liquidity_stress(self) -> float:
        """Calculate liquidity-based stress score."""
        if not any(self.order_book_depth.values()):
            return 0.0

        liquidity_scores = []

        for symbol in self.symbols:
            # Check order book depth
            if self.order_book_depth[symbol]:
                recent_depth = list(self.order_book_depth[symbol])[-10:]
                avg_size = np.mean([entry['total_size'] for entry in recent_depth])

                # Check spreads
                if self.spread_history[symbol]:
                    recent_spreads = list(self.spread_history[symbol])[-10:]
                    avg_spread_bps = np.mean([entry['spread_bps'] for entry in recent_spreads])

                    # Liquidity stress based on depth and spread
                    depth_stress = max(0, (10000 - avg_size) / 10000)  # Normalize by 10K
                    spread_stress = min(avg_spread_bps / 100, 1.0)  # 100 bps as max

                    liquidity_score = (depth_stress + spread_stress) / 2
                    liquidity_scores.append(liquidity_score)

        if not liquidity_scores:
            return 0.0

        return np.mean(liquidity_scores)

    async def _calculate_volume_stress(self) -> float:
        """Calculate volume-based stress score."""
        if not any(self.volume_history.values()):
            return 0.0

        volume_scores = []

        for symbol in self.symbols:
            if len(self.volume_history[symbol]) < 10:
                continue

            recent_volumes = [entry['volume'] for entry in list(self.volume_history[symbol])[-60:]]
            avg_volume = np.mean(recent_volumes)

            # Calculate baseline volume (earlier in history)
            if len(self.volume_history[symbol]) > 120:
                baseline_volumes = [entry['volume'] for entry in list(self.volume_history[symbol])[-120:-60]]
                baseline_avg = np.mean(baseline_volumes)

                if baseline_avg > 0:
                    volume_ratio = avg_volume / baseline_avg
                    # Normalize to 0-1 scale (5x normal volume as max)
                    volume_score = min(volume_ratio / 5.0, 1.0)
                    volume_scores.append(volume_score)

        if not volume_scores:
            return 0.0

        return np.mean(volume_scores)

    async def _calculate_correlation_stress(self) -> float:
        """Calculate correlation-based stress score."""
        if len(self.symbols) < 2:
            return 0.0

        # Calculate correlations between symbols
        correlations = []

        for i, symbol1 in enumerate(self.symbols):
            for symbol2 in self.symbols[i+1:]:
                correlation = await self._calculate_symbol_correlation(symbol1, symbol2)
                if correlation is not None:
                    correlations.append(abs(correlation))

        if not correlations:
            return 0.0

        avg_correlation = np.mean(correlations)

        # Stress when correlations are low (divergence) or very high (herding)
        # Optimal range is 0.3-0.7
        if avg_correlation < 0.3:
            stress_score = (0.3 - avg_correlation) / 0.3
        elif avg_correlation > 0.8:
            stress_score = (avg_correlation - 0.8) / 0.2
        else:
            stress_score = 0.0

        return min(stress_score, 1.0)

    async def _calculate_symbol_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate correlation between two symbols."""
        if (len(self.price_history[symbol1]) < 20 or
            len(self.price_history[symbol2]) < 20):
            return None

        # Get recent price data
        prices1 = [entry['price'] for entry in list(self.price_history[symbol1])[-60:]]
        prices2 = [entry['price'] for entry in list(self.price_history[symbol2])[-60:]]

        if len(prices1) != len(prices2) or len(prices1) < 10:
            return None

        # Calculate returns
        returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1] for i in range(1, len(prices1))]
        returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1] for i in range(1, len(prices2))]

        if len(returns1) < 5:
            return None

        # Calculate correlation
        correlation_matrix = np.corrcoef(returns1, returns2)
        correlation = correlation_matrix[0, 1]

        return correlation if not np.isnan(correlation) else 0.0

    async def _calculate_system_health_stress(self) -> float:
        """Calculate system health-based stress score."""
        stress_score = 0.0

        # Check latency
        latency_stats = self.latency_tracker.get_statistics()
        if latency_stats:
            avg_latency = np.mean([stats.get('avg_ms', 0) for stats in latency_stats.values()])
            latency_stress = min(avg_latency / self.config.latency_threshold_ms, 1.0)
            stress_score += latency_stress * 0.4

        # Check error rate
        recent_errors = sum(1 for error_time in self.error_history
                           if error_time > datetime.now(timezone.utc) - timedelta(minutes=5))
        total_operations = len(self.system_metrics)

        if total_operations > 0:
            error_rate = recent_errors / max(total_operations, 1)
            error_stress = min(error_rate / self.config.error_rate_threshold, 1.0)
            stress_score += error_stress * 0.6

        return min(stress_score, 1.0)

    def _determine_stress_level(self, stress_score: float) -> StressLevel:
        """Determine stress level from stress score."""
        if stress_score >= 0.9:
            return StressLevel.CRITICAL
        elif stress_score >= 0.7:
            return StressLevel.SEVERE
        elif stress_score >= 0.5:
            return StressLevel.HIGH
        elif stress_score >= 0.3:
            return StressLevel.ELEVATED
        else:
            return StressLevel.NORMAL

    async def _check_circuit_breaker_triggers(self, stress_metrics: MarketStressMetrics):
        """Check for circuit breaker trigger conditions."""
        if self.state == CircuitBreakerState.OPEN:
            return  # Already tripped

        triggers = []

        # Check individual stress thresholds
        if stress_metrics.volatility_score > 0.8:
            triggers.append(TriggerType.VOLATILITY_SPIKE)

        if stress_metrics.price_movement_score > 0.8:
            triggers.append(TriggerType.PRICE_MOVEMENT)

        if stress_metrics.liquidity_score > 0.8:
            triggers.append(TriggerType.LIQUIDITY_CRISIS)

        if stress_metrics.system_health_score > 0.8:
            triggers.append(TriggerType.SYSTEM_STRESS)

        if stress_metrics.volume_score > 0.8:
            triggers.append(TriggerType.VOLUME_ANOMALY)

        if stress_metrics.correlation_score > 0.8:
            triggers.append(TriggerType.CORRELATION_BREAKDOWN)

        # Check overall stress level
        if stress_metrics.stress_level in [StressLevel.CRITICAL, StressLevel.SEVERE]:
            if not triggers:  # If no specific triggers, use general stress
                triggers.append(TriggerType.SYSTEM_STRESS)

        # Check trip frequency limit
        if self._check_trip_frequency_limit():
            if triggers:
                # Trip the circuit breaker
                trigger_type = triggers[0]  # Use first trigger as primary
                await self._trip_circuit_breaker(trigger_type, stress_metrics)

    async def _trip_circuit_breaker(self, trigger_type: TriggerType, stress_metrics: MarketStressMetrics):
        """Trip the circuit breaker."""
        previous_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.current_trip_time = datetime.now(timezone.utc)
        self.last_state_change = self.current_trip_time

        # Create event record
        event = CircuitBreakerEvent(
            timestamp=self.current_trip_time,
            trigger_type=trigger_type,
            previous_state=previous_state,
            new_state=self.state,
            stress_metrics=stress_metrics,
            reason=f"Circuit breaker tripped due to {trigger_type.value}",
            recovery_conditions=self._get_recovery_conditions(stress_metrics)
        )

        self.circuit_breaker_events.append(event)
        self.trip_count_per_hour.append(self.current_trip_time)

        # Log the event
        self.logger.warning(f"CIRCUIT BREAKER TRIPPED: {trigger_type.value}")
        self.logger.warning(f"Stress Level: {stress_metrics.stress_level.value}")
        self.logger.warning(f"Overall Stress Score: {stress_metrics.overall_stress_score:.2f}")
        if stress_metrics.trigger_conditions:
            self.logger.warning(f"Trigger Conditions: {', '.join(stress_metrics.trigger_conditions)}")

        # Notify subscribers
        await self._notify_state_change(event)

    async def _check_recovery_conditions(self, stress_metrics: MarketStressMetrics):
        """Check for recovery conditions to reset circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if cooldown period has passed
            if (datetime.now(timezone.utc) - self.current_trip_time).total_seconds() < self.config.cooldown_period_seconds:
                return

            # Check if stress levels have reduced
            if stress_metrics.stress_level == StressLevel.NORMAL:
                # Move to half-open state for testing
                await self._set_half_open_state(stress_metrics)

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Check if test period has passed
            if (datetime.now(timezone.utc) - self.last_state_change).total_seconds() > self.config.test_period_seconds:
                if stress_metrics.stress_level == StressLevel.NORMAL:
                    # Recovery successful - close circuit breaker
                    await self._close_circuit_breaker(stress_metrics)
                else:
                    # Recovery failed - trip again
                    await self._trip_circuit_breaker(TriggerType.SYSTEM_STRESS, stress_metrics)

    async def _set_half_open_state(self, stress_metrics: MarketStressMetrics):
        """Set circuit breaker to half-open state for testing."""
        previous_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = datetime.now(timezone.utc)

        event = CircuitBreakerEvent(
            timestamp=self.last_state_change,
            trigger_type=TriggerType.SYSTEM_STRESS,
            previous_state=previous_state,
            new_state=self.state,
            stress_metrics=stress_metrics,
            reason="Entering half-open state for testing",
            recovery_conditions=["Monitor conditions during test period"]
        )

        self.circuit_breaker_events.append(event)
        self.logger.info("Circuit breaker entering HALF-OPEN state for testing")
        await self._notify_state_change(event)

    async def _close_circuit_breaker(self, stress_metrics: MarketStressMetrics):
        """Close circuit breaker - recovery successful."""
        previous_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = datetime.now(timezone.utc)

        if self.circuit_breaker_events:
            # Calculate duration of the trip
            last_event = self.circuit_breaker_events[-1]
            duration = (self.last_state_change - last_event.timestamp).total_seconds()
            last_event.duration_seconds = duration

        event = CircuitBreakerEvent(
            timestamp=self.last_state_change,
            trigger_type=TriggerType.SYSTEM_STRESS,
            previous_state=previous_state,
            new_state=self.state,
            stress_metrics=stress_metrics,
            reason="Recovery successful - circuit breaker closed",
            recovery_conditions=[]
        )

        self.circuit_breaker_events.append(event)
        self.logger.info("Circuit breaker CLOSED - recovery successful")
        await self._notify_state_change(event)

    def _get_recovery_conditions(self, stress_metrics: MarketStressMetrics) -> List[str]:
        """Get list of conditions needed for recovery."""
        conditions = []

        if stress_metrics.volatility_score > 0.5:
            conditions.append("Volatility must return to normal levels")

        if stress_metrics.price_movement_score > 0.5:
            conditions.append("Price movements must stabilize")

        if stress_metrics.liquidity_score > 0.5:
            conditions.append("Liquidity must improve")

        if stress_metrics.system_health_score > 0.5:
            conditions.append("System health must be restored")

        conditions.append(f"Cooldown period of {self.config.cooldown_period_seconds} seconds")

        return conditions

    def _check_trip_frequency_limit(self) -> bool:
        """Check if circuit breaker has tripped too frequently."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_trips = [trip_time for trip_time in self.trip_count_per_hour if trip_time > cutoff_time]

        return len(recent_trips) < self.config.max_trips_per_hour

    async def _check_immediate_triggers(self, symbol: str, order_book: OrderBook):
        """Check for immediate trigger conditions from order book."""
        # Immediate price movement check
        if len(self.price_history[symbol]) >= 2:
            recent_prices = list(self.price_history[symbol])[-5:]
            price_change = abs(order_book.mid_price - recent_prices[0]['price']) / recent_prices[0]['price']

            if price_change > self.config.price_movement_threshold:
                # Immediate trigger for extreme price movement
                stress_metrics = await self._calculate_stress_metrics()
                await self._trip_circuit_breaker(TriggerType.PRICE_MOVEMENT, stress_metrics)

    async def _update_stress_level(self, stress_metrics: MarketStressMetrics):
        """Update current stress level and notify if changed."""
        if stress_metrics.stress_level != self.current_stress_level:
            previous_level = self.current_stress_level
            self.current_stress_level = stress_metrics.stress_level

            # Store stress history
            self.stress_history.append(stress_metrics)

            # Notify if stress level increased
            if stress_metrics.stress_level.value > previous_level.value:
                await self._notify_stress_alert(stress_metrics)

    async def _monitor_correlations(self):
        """Monitor correlations between symbols."""
        if len(self.symbols) < 2:
            return

        # Update correlation matrix
        for i, symbol1 in enumerate(self.symbols):
            if symbol1 not in self.correlation_matrix:
                self.correlation_matrix[symbol1] = {}

            for symbol2 in self.symbols:
                if symbol1 != symbol2:
                    correlation = await self._calculate_symbol_correlation(symbol1, symbol2)
                    if correlation is not None:
                        self.correlation_matrix[symbol1][symbol2] = correlation

    async def _notify_state_change(self, event: CircuitBreakerEvent):
        """Notify subscribers of state change."""
        for callback in self.state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error notifying state change callback: {e}")

    async def _notify_stress_alert(self, stress_metrics: MarketStressMetrics):
        """Notify subscribers of stress level change."""
        for callback in self.stress_alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stress_metrics)
                else:
                    callback(stress_metrics)
            except Exception as e:
                self.logger.error(f"Error notifying stress alert callback: {e}")

    def subscribe_to_state_changes(self, callback: Callable):
        """Subscribe to circuit breaker state change notifications."""
        self.state_change_callbacks.add(callback)

    def subscribe_to_stress_alerts(self, callback: Callable):
        """Subscribe to stress level change notifications."""
        self.stress_alert_callbacks.add(callback)

    def unsubscribe_from_state_changes(self, callback: Callable):
        """Unsubscribe from state change notifications."""
        self.state_change_callbacks.discard(callback)

    def unsubscribe_from_stress_alerts(self, callback: Callable):
        """Unsubscribe from stress alert notifications."""
        self.stress_alert_callbacks.discard(callback)

    async def manual_trip(self, reason: str = "Manual override"):
        """Manually trip the circuit breaker."""
        stress_metrics = await self._calculate_stress_metrics()
        await self._trip_circuit_breaker(TriggerType.MANUAL_OVERRIDE, stress_metrics)

        # Update reason
        if self.circuit_breaker_events:
            self.circuit_breaker_events[-1].reason = reason

    async def manual_reset(self):
        """Manually reset the circuit breaker."""
        stress_metrics = await self._calculate_stress_metrics()

        if self.state != CircuitBreakerState.CLOSED:
            await self._close_circuit_breaker(stress_metrics)

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            'state': self.state.value,
            'stress_level': self.current_stress_level.value,
            'current_trip_time': self.current_trip_time.isoformat() if self.current_trip_time else None,
            'last_state_change': self.last_state_change.isoformat(),
            'is_monitoring': self.is_monitoring,
            'trip_count_last_hour': len([t for t in self.trip_count_per_hour
                                       if t > datetime.now(timezone.utc) - timedelta(hours=1)]),
            'total_events': len(self.circuit_breaker_events),
            'correlation_matrix': self.correlation_matrix,
            'config': {
                'volatility_threshold': self.config.volatility_threshold,
                'price_movement_threshold': self.config.price_movement_threshold,
                'cooldown_period_seconds': self.config.cooldown_period_seconds
            }
        }

    def get_stress_metrics(self) -> Optional[MarketStressMetrics]:
        """Get current stress metrics."""
        if self.stress_history:
            return self.stress_history[-1]
        return None

    def get_event_history(self, limit: int = 50) -> List[CircuitBreakerEvent]:
        """Get circuit breaker event history."""
        return self.circuit_breaker_events[-limit:]

    def reset(self):
        """Reset circuit breaker system."""
        self.state = CircuitBreakerState.CLOSED
        self.current_trip_time = None
        self.last_state_change = datetime.now(timezone.utc)
        self.current_stress_level = StressLevel.NORMAL

        # Clear histories
        self.price_history.clear()
        self.volume_history.clear()
        self.spread_history.clear()
        self.order_book_depth.clear()
        self.error_history.clear()
        self.system_metrics.clear()
        self.stress_history.clear()
        self.circuit_breaker_events.clear()
        self.trip_count_per_hour.clear()
        self.correlation_matrix.clear()

        # Clear latency tracking
        self.latency_tracker = LatencyTracker()

        self.logger.info("Circuit breaker system reset")