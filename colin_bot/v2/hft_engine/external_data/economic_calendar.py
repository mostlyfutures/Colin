"""
Economic Calendar for High-Frequency Trading

Monitors economic events, announcements, and macroeconomic data releases
that can impact cryptocurrency markets and trading strategies.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json

from ..utils.data_structures import TradingSignal, SignalDirection, SignalStrength


class EventImpact(Enum):
    """Economic event impact levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of economic events."""
    INTEREST_RATE = "interest_rate"
    INFLATION = "inflation"
    EMPLOYMENT = "employment"
    GDP = "gdp"
    MANUFACTURING = "manufacturing"
    CONSUMER_SENTIMENT = "consumer_sentiment"
    RETAIL_SALES = "retail_sales"
    HOUSING = "housing"
    TRADE_BALANCE = "trade_balance"
    CENTRAL_BANK_SPEECH = "central_bank_speech"
    REGULATORY_ANNOUNCEMENT = "regulatory_announcement"
    GEOPOLITICAL_EVENT = "geopolitical_event"


@dataclass
class EconomicEvent:
    """Economic event data structure."""
    id: str
    title: str
    description: str
    country: str
    currency: str
    event_type: EventType
    impact: EventImpact
    timestamp: datetime
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    units: str = ""
    relevance_score: float = 0.0  # 0.0 to 1.0
    affected_symbols: Set[str] = field(default_factory=set)
    market_impact_estimate: float = 0.0  # -1.0 to 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class EconomicSignal:
    """Trading signal derived from economic events."""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    economic_events: List[EconomicEvent] = field(default_factory=list)
    reasoning: str = ""
    expected_impact_duration: timedelta = field(default=timedelta(hours=1))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EconomicCalendar:
    """
    Economic calendar monitoring system for high-frequency trading.

    Tracks economic announcements, central bank decisions, and macroeconomic
    data releases that can significantly impact cryptocurrency markets.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.api_keys = self.config.get('api_keys', {})
        self.update_interval = self.config.get('update_interval', 300.0)  # 5 minutes
        self.event_horizon_hours = self.config.get('event_horizon_hours', 168)  # 1 week

        # Economic data sources
        self.data_sources = [
            'forex_factory_api',
            'trading_economics',
            'federal_reserve',
            'ecb',
            'bank_of_england'
        ]

        # Symbol relevance mapping
        self.symbol_relevance = {
            'BTC': ['USD', 'global_economy', 'inflation', 'interest_rates'],
            'ETH': ['USD', 'global_economy', 'technology_sector'],
            'USDT': ['USD', 'banking_regulations', 'stablecoin_regulations'],
            'USDC': ['USD', 'banking_regulations', 'stablecoin_regulations'],
            'All': ['crypto_regulations', 'macro_economics', 'geopolitical_events']
        }

        # Impact multipliers
        self.impact_multipliers = {
            EventImpact.LOW: 0.1,
            EventImpact.MEDIUM: 0.3,
            EventImpact.HIGH: 0.6,
            EventImpact.CRITICAL: 1.0
        }

        # Data storage
        self.economic_events: deque = deque(maxlen=1000)
        self.economic_signals: deque = deque(maxlen=500)
        self.processed_events: Set[str] = set()

        # Performance tracking
        self.events_processed = 0
        self.signals_generated = 0

        # Subscribers
        self.event_subscribers: Set[callable] = set()
        self.signal_subscribers: Set[callable] = set()

    async def start_monitoring(self):
        """Start economic calendar monitoring."""
        self.logger.info("Starting economic calendar monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop economic calendar monitoring."""
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Economic calendar monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for economic events."""
        while True:
            try:
                # Fetch economic events
                await self._fetch_economic_events()

                # Process upcoming events
                await self._process_upcoming_events()

                # Check for recent announcements
                await self._check_recent_announcements()

                # Generate trading signals
                await self._generate_trading_signals()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in economic calendar monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _fetch_economic_events(self):
        """Fetch economic events from various sources."""
        for source in self.data_sources:
            try:
                if source == 'forex_factory_api':
                    await self._fetch_forex_factory_events()
                elif source == 'trading_economics':
                    await self._fetch_trading_economics_events()
                elif source == 'federal_reserve':
                    await self._fetch_fed_events()
            except Exception as e:
                self.logger.error(f"Error fetching events from {source}: {e}")

    async def _fetch_forex_factory_events(self):
        """Fetch events from Forex Factory API."""
        # Mock implementation - replace with actual API call
        mock_events = [
            {
                'title': 'FOMC Interest Rate Decision',
                'description': 'Federal Reserve announces interest rate decision',
                'country': 'US',
                'currency': 'USD',
                'event_type': 'interest_rate',
                'impact': 'high',
                'timestamp': datetime.now(timezone.utc) + timedelta(hours=2),
                'forecast': 5.25,
                'previous': 5.00
            },
            {
                'title': 'Consumer Price Index (CPI)',
                'description': 'Monthly CPI data release',
                'country': 'US',
                'currency': 'USD',
                'event_type': 'inflation',
                'impact': 'high',
                'timestamp': datetime.now(timezone.utc) + timedelta(hours=6),
                'forecast': 3.2,
                'previous': 3.0
            }
        ]

        for event_data in mock_events:
            await self._process_economic_event_data(event_data, 'forex_factory')

    async def _fetch_trading_economics_events(self):
        """Fetch events from Trading Economics API."""
        # Mock implementation
        mock_events = [
            {
                'title': 'ECB Interest Rate Decision',
                'description': 'European Central Bank announces interest rate decision',
                'country': 'EU',
                'currency': 'EUR',
                'event_type': 'interest_rate',
                'impact': 'high',
                'timestamp': datetime.now(timezone.utc) + timedelta(days=1),
                'forecast': 4.00,
                'previous': 3.75
            }
        ]

        for event_data in mock_events:
            await self._process_economic_event_data(event_data, 'trading_economics')

    async def _fetch_fed_events(self):
        """Fetch Federal Reserve events."""
        # Mock implementation
        mock_events = [
            {
                'title': 'Fed Chair Speech',
                'description': 'Federal Reserve Chair delivers speech on monetary policy',
                'country': 'US',
                'currency': 'USD',
                'event_type': 'central_bank_speech',
                'impact': 'medium',
                'timestamp': datetime.now(timezone.utc) + timedelta(hours=4)
            }
        ]

        for event_data in mock_events:
            await self._process_economic_event_data(event_data, 'federal_reserve')

    async def _process_economic_event_data(self, event_data: Dict, source: str):
        """Process economic event data from API."""
        try:
            event_id = f"{source}_{hash(event_data.get('title', ''))}_{event_data.get('timestamp', datetime.now())}"

            if event_id in self.processed_events:
                return

            economic_event = EconomicEvent(
                id=event_id,
                title=event_data.get('title', ''),
                description=event_data.get('description', ''),
                country=event_data.get('country', ''),
                currency=event_data.get('currency', ''),
                event_type=EventType(event_data.get('event_type', 'interest_rate')),
                impact=EventImpact(event_data.get('impact', 'medium')),
                timestamp=event_data.get('timestamp', datetime.now(timezone.utc)),
                actual_value=event_data.get('actual'),
                forecast_value=event_data.get('forecast'),
                previous_value=event_data.get('previous'),
                units=event_data.get('units', '%')
            )

            # Calculate relevance and affected symbols
            economic_event.relevance_score = self._calculate_relevance(economic_event)
            economic_event.affected_symbols = self._get_affected_symbols(economic_event)
            economic_event.market_impact_estimate = self._estimate_market_impact(economic_event)

            self.economic_events.append(economic_event)
            self.processed_events.add(event_id)
            self.events_processed += 1

            # Notify event subscribers
            await self._notify_event_subscribers(economic_event)

        except Exception as e:
            self.logger.error(f"Error processing economic event: {e}")

    def _calculate_relevance(self, event: EconomicEvent) -> float:
        """Calculate relevance score for economic event."""
        base_relevance = self.impact_multipliers.get(event.impact, 0.3)

        # Time-based relevance (events happening sooner are more relevant)
        hours_until = (event.timestamp - datetime.now(timezone.utc)).total_seconds() / 3600
        if hours_until < 0:  # Past event
            time_factor = 0.3
        elif hours_until < 1:
            time_factor = 1.0
        elif hours_until < 6:
            time_factor = 0.8
        elif hours_until < 24:
            time_factor = 0.6
        elif hours_until < 168:  # 1 week
            time_factor = 0.4
        else:
            time_factor = 0.2

        # Currency relevance for crypto
        if event.currency == 'USD':
            currency_factor = 1.0
        elif event.currency in ['EUR', 'GBP', 'JPY']:
            currency_factor = 0.6
        else:
            currency_factor = 0.3

        # Event type relevance
        high_relevance_types = {
            EventType.INTEREST_RATE,
            EventType.INFLATION,
            EventType.REGULATORY_ANNOUNCEMENT,
            EventType.GEOPOLITICAL_EVENT
        }

        type_factor = 1.2 if event.event_type in high_relevance_types else 0.8

        return min(base_relevance * time_factor * currency_factor * type_factor, 1.0)

    def _get_affected_symbols(self, event: EconomicEvent) -> Set[str]:
        """Get cryptocurrency symbols affected by economic event."""
        affected_symbols = set()

        # Direct USD relevance
        if event.currency == 'USD':
            affected_symbols.update(['BTC', 'ETH', 'USDT', 'USDC'])

        # Global economic relevance
        if event.event_type in [EventType.INTEREST_RATE, EventType.INFLATION, EventType.GDP]:
            affected_symbols.add('BTC')  # Bitcoin as digital gold
            affected_symbols.add('ETH')  # Ethereum as risk asset

        # Regulatory relevance
        if event.event_type == EventType.REGULATORY_ANNOUNCEMENT:
            if 'crypto' in event.title.lower() or 'digital' in event.title.lower():
                affected_symbols.update(['BTC', 'ETH', 'All'])
            elif 'stablecoin' in event.title.lower():
                affected_symbols.update(['USDT', 'USDC'])

        # Central bank speeches can affect all crypto
        if event.event_type == EventType.CENTRAL_BANK_SPEECH:
            affected_symbols.add('All')

        return affected_symbols

    def _estimate_market_impact(self, event: EconomicEvent) -> float:
        """Estimate market impact direction (-1 to 1)."""
        if event.actual_value is None or event.forecast_value is None:
            return 0.0

        # Calculate surprise factor
        if event.forecast_value != 0:
            surprise = (event.actual_value - event.forecast_value) / abs(event.forecast_value)
        else:
            surprise = 0.0

        # Adjust based on event type and expected market reaction
        impact_direction = 0.0

        if event.event_type == EventType.INTEREST_RATE:
            # Higher rates are generally negative for risk assets
            impact_direction = -surprise * 0.7
        elif event.event_type == EventType.INFLATION:
            # Higher inflation can be negative (rate hike concerns) or positive (hedge demand)
            impact_direction = -surprise * 0.3
        elif event.event_type == EventType.EMPLOYMENT:
            # Strong employment can be positive or negative depending on context
            impact_direction = surprise * 0.4
        elif event.event_type == EventType.GDP:
            # Strong GDP is generally positive for risk assets
            impact_direction = surprise * 0.5

        return max(-1.0, min(1.0, impact_direction))

    async def _process_upcoming_events(self):
        """Process upcoming economic events."""
        cutoff_time = datetime.now(timezone.utc) + timedelta(hours=self.event_horizon_hours)
        upcoming_events = [
            event for event in self.economic_events
            if datetime.now(timezone.utc) < event.timestamp < cutoff_time
            and event.relevance_score > 0.3
        ]

        # Sort by relevance and timing
        upcoming_events.sort(key=lambda e: (e.relevance_score, e.timestamp))

        # Process high-impact upcoming events
        for event in upcoming_events[:10]:  # Top 10 events
            if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                await self._generate_upcoming_signal(event)

    async def _check_recent_announcements(self):
        """Check for recent economic announcements."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=2)
        recent_events = [
            event for event in self.economic_events
            if event.timestamp > cutoff_time
            and event.actual_value is not None
        ]

        for event in recent_events:
            if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                await self._generate_announcement_signal(event)

    async def _generate_upcoming_signal(self, event: EconomicEvent):
        """Generate trading signal for upcoming economic event."""
        try:
            # Base direction on expectations
            direction = SignalDirection.HOLD  # Default to hold for upcoming events

            if event.event_type == EventType.INTEREST_RATE:
                if event.forecast_value and event.previous_value:
                    if event.forecast_value > event.previous_value:
                        direction = SignalDirection.SHORT  # Potential negative impact
                    else:
                        direction = SignalDirection.LONG   # Potential positive impact
            elif event.event_type == EventType.INFLATION:
                if event.forecast_value and event.previous_value:
                    if event.forecast_value > event.previous_value:
                        direction = SignalDirection.SHORT
                    else:
                        direction = SignalDirection.LONG

            # Determine strength based on impact and relevance
            if event.impact == EventImpact.CRITICAL:
                strength = SignalStrength.STRONG
            elif event.impact == EventImpact.HIGH:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK

            # Calculate confidence
            confidence = event.relevance_score * 0.7  # Reduce confidence for predictions

            # Generate reasoning
            reasoning = f"Upcoming {event.title} ({event.impact.value} impact) scheduled for {event.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"

            # Create signal for each affected symbol
            for symbol in event.affected_symbols:
                economic_signal = EconomicSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    economic_events=[event],
                    reasoning=reasoning,
                    expected_impact_duration=timedelta(hours=4)
                )

                self.economic_signals.append(economic_signal)
                await self._notify_signal_subscribers(economic_signal)

        except Exception as e:
            self.logger.error(f"Error generating upcoming signal for {event.id}: {e}")

    async def _generate_announcement_signal(self, event: EconomicEvent):
        """Generate trading signal for economic announcement."""
        try:
            # Direction based on market impact estimate
            if event.market_impact_estimate > 0.2:
                direction = SignalDirection.LONG
                strength = SignalStrength.STRONG if event.market_impact_estimate > 0.5 else SignalStrength.MODERATE
            elif event.market_impact_estimate < -0.2:
                direction = SignalDirection.SHORT
                strength = SignalStrength.STRONG if event.market_impact_estimate < -0.5 else SignalStrength.MODERATE
            else:
                return  # No clear signal

            # Calculate confidence
            confidence = event.relevance_score * abs(event.market_impact_estimate)
            confidence = min(confidence, 0.9)

            # Generate reasoning
            reasoning_parts = [
                f"{event.title} announcement",
                f"Actual: {event.actual_value}{event.units}",
                f"Forecast: {event.forecast_value}{event.units}",
                f"Surprise: {(event.actual_value - event.forecast_value):.2f}{event.units}"
            ]
            reasoning = "; ".join(reasoning_parts)

            # Create signal for each affected symbol
            for symbol in event.affected_symbols:
                economic_signal = EconomicSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    economic_events=[event],
                    reasoning=reasoning,
                    expected_impact_duration=timedelta(hours=6)
                )

                self.economic_signals.append(economic_signal)
                self.signals_generated += 1
                await self._notify_signal_subscribers(economic_signal)

        except Exception as e:
            self.logger.error(f"Error generating announcement signal for {event.id}: {e}")

    async def _generate_trading_signals(self):
        """Generate aggregated trading signals from multiple events."""
        # Group recent signals by symbol
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_signals = [
            signal for signal in self.economic_signals
            if signal.timestamp > cutoff_time
        ]

        signals_by_symbol = {}
        for signal in recent_signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)

        # Generate aggregated signals
        for symbol, signals in signals_by_symbol.items():
            if len(signals) >= 2:  # Minimum events for aggregation
                await self._aggregate_signals(symbol, signals)

    async def _aggregate_signals(self, symbol: str, signals: List[EconomicSignal]):
        """Aggregate multiple signals for a symbol."""
        try:
            # Calculate weighted direction
            total_weight = 0
            weighted_direction = 0

            for signal in signals:
                weight = signal.confidence * (1.5 if signal.strength == SignalStrength.STRONG else 1.0)
                direction_value = 1 if signal.direction == SignalDirection.LONG else -1 if signal.direction == SignalDirection.SHORT else 0

                total_weight += weight
                weighted_direction += direction_value * weight

            if total_weight == 0:
                return

            avg_direction = weighted_direction / total_weight

            # Determine final direction and strength
            if avg_direction > 0.3:
                final_direction = SignalDirection.LONG
                final_strength = SignalStrength.STRONG if avg_direction > 0.6 else SignalStrength.MODERATE
            elif avg_direction < -0.3:
                final_direction = SignalDirection.SHORT
                final_strength = SignalStrength.STRONG if avg_direction < -0.6 else SignalStrength.MODERATE
            else:
                return  # No clear aggregated signal

            # Calculate aggregated confidence
            final_confidence = min(total_weight / len(signals), 0.9)

            # Combine all economic events
            all_events = []
            for signal in signals:
                all_events.extend(signal.economic_events)

            # Generate reasoning
            reasoning = f"Aggregated signal from {len(signals)} economic events over last 24 hours"

            # Create aggregated signal
            aggregated_signal = EconomicSignal(
                symbol=symbol,
                direction=final_direction,
                strength=final_strength,
                confidence=final_confidence,
                economic_events=list(set(all_events)),  # Remove duplicates
                reasoning=reasoning,
                expected_impact_duration=timedelta(hours=12)
            )

            self.economic_signals.append(aggregated_signal)
            await self._notify_signal_subscribers(aggregated_signal)

        except Exception as e:
            self.logger.error(f"Error aggregating signals for {symbol}: {e}")

    async def _notify_event_subscribers(self, event: EconomicEvent):
        """Notify subscribers of new economic events."""
        for callback in self.event_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error notifying event subscriber: {e}")

    async def _notify_signal_subscribers(self, signal: EconomicSignal):
        """Notify subscribers of new economic signals."""
        for callback in self.signal_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                self.logger.error(f"Error notifying signal subscriber: {e}")

    def subscribe_to_events(self, callback: callable):
        """Subscribe to economic event notifications."""
        self.event_subscribers.add(callback)

    def subscribe_to_signals(self, callback: callable):
        """Subscribe to economic-based trading signals."""
        self.signal_subscribers.add(callback)

    def unsubscribe_from_events(self, callback: callable):
        """Unsubscribe from economic event notifications."""
        self.event_subscribers.discard(callback)

    def unsubscribe_from_signals(self, callback: callable):
        """Unsubscribe from economic-based trading signals."""
        self.signal_subscribers.discard(callback)

    def get_upcoming_events(self, hours: int = 24) -> List[EconomicEvent]:
        """Get upcoming economic events."""
        cutoff_time = datetime.now(timezone.utc) + timedelta(hours=hours)
        return [
            event for event in self.economic_events
            if datetime.now(timezone.utc) < event.timestamp < cutoff_time
            and event.relevance_score > 0.3
        ]

    def get_recent_signals(self, symbol: str = None, hours: int = 24) -> List[EconomicSignal]:
        """Get recent economic-based trading signals."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_signals = [
            signal for signal in self.economic_signals
            if signal.timestamp > cutoff_time
        ]

        if symbol:
            recent_signals = [s for s in recent_signals if s.symbol == symbol]

        return recent_signals

    def get_statistics(self) -> Dict:
        """Get economic calendar statistics."""
        return {
            'events_processed': self.events_processed,
            'signals_generated': self.signals_generated,
            'upcoming_events_24h': len(self.get_upcoming_events(24)),
            'recent_signals_24h': len(self.get_recent_signals(hours=24)),
            'active_subscribers_events': len(self.event_subscribers),
            'active_subscribers_signals': len(self.signal_subscribers),
            'total_events_stored': len(self.economic_events),
            'total_signals_stored': len(self.economic_signals)
        }

    def reset(self):
        """Reset economic calendar state."""
        self.economic_events.clear()
        self.economic_signals.clear()
        self.processed_events.clear()
        self.events_processed = 0
        self.signals_generated = 0