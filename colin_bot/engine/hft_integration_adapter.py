"""
HFT Integration Adapter for Colin Trading Bot

This adapter bridges the HFT engine with the main Colin Bot,
allowing seamless integration of high-frequency signals.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
import numpy as np

# Import HFT Engine components
try:
    from colin_bot.v2.hft_engine import (
        OFICalculator, BookSkewAnalyzer, SignalFusionEngine,
        DynamicPositionSizer, CircuitBreakerSystem, HFTDataManager
    )
    from colin_bot.v2.hft_engine.utils.data_structures import (
        OrderBook, OrderBookLevel, TradingSignal, SignalDirection
    )
    from colin_bot.v2.hft_engine.utils.math_utils import calculate_skew, hawkes_process
    HFT_AVAILABLE = True
    logger.info("HFT Engine components loaded successfully")
except ImportError as e:
    logger.warning(f"HFT Engine not available: {e}")
    HFT_AVAILABLE = False


@dataclass
class HFTSignal:
    """HFT-specific signal for integration with main bot."""
    symbol: str
    timestamp: datetime
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0-100%
    strength: str  # "weak", "moderate", "strong"
    ofi_signal: float
    book_skew: float
    fusion_confidence: float
    rationale: List[str] = field(default_factory=list)
    raw_hft_data: Dict[str, Any] = field(default_factory=dict)


class HFTIntegrationAdapter:
    """
    Adapter that integrates HFT signals with the main Colin Trading Bot.

    This class provides a clean interface for the main bot to consume
    high-frequency trading signals while maintaining separation of concerns.
    """

    def __init__(self, config_manager, enable_hft: bool = True):
        """
        Initialize HFT Integration Adapter.

        Args:
            config_manager: Main bot configuration manager
            enable_hft: Whether to enable HFT signal generation
        """
        self.config_manager = config_manager
        self.enable_hft = enable_hft and HFT_AVAILABLE

        if self.enable_hft:
            logger.info("Initializing HFT Integration Adapter")
            self._initialize_hft_components()
        else:
            logger.warning("HFT Integration disabled - falling back to conventional analysis")
            self.hft_components = None

    def _initialize_hft_components(self):
        """Initialize HFT engine components."""
        try:
            # Core HFT components
            self.ofi_calculator = OFICalculator()
            self.book_skew_analyzer = BookSkewAnalyzer()
            self.signal_fusion_engine = SignalFusionEngine()
            self.position_sizer = DynamicPositionSizer(portfolio_value_usd=1000000.0)  # Default $1M portfolio
            self.circuit_breaker = CircuitBreakerSystem(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])  # Default symbols

            # Data management
            self.data_manager = HFTDataManager(self.config_manager)  # Pass config manager

            # Performance tracking
            self.signal_history = []
            self.performance_metrics = {
                'signals_generated': 0,
                'ofi_calculations': 0,
                'skew_calculations': 0,
                'fusion_operations': 0,
                'circuit_breaker_trips': 0,
                'avg_signal_confidence': 0.0
            }

            # Mock data generator for testing
            self._setup_mock_data_generator()

            logger.info("✅ HFT components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize HFT components: {e}")
            self.enable_hft = False
            self.hft_components = None

    def _setup_mock_data_generator(self):
        """Setup mock data generator for testing purposes."""
        try:
            from colin_bot.v2.hft_engine.data_ingestion.connectors.mock_connector import MockDataConnector
            self.mock_connector = MockDataConnector(self.config_manager)  # Pass config manager
            logger.info("✅ Mock data connector initialized")
        except ImportError:
            logger.warning("⚠️ Mock data connector not available")
            self.mock_connector = None

    def is_hft_enabled(self) -> bool:
        """Check if HFT integration is enabled and available."""
        return self.enable_hft and HFT_AVAILABLE

    async def generate_hft_signal(self, symbol: str, time_horizon: str = "1h") -> Optional[HFTSignal]:
        """
        Generate HFT signal for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            time_horizon: Analysis time horizon

        Returns:
            HFTSignal or None if HFT is disabled
        """
        if not self.is_hft_enabled():
            logger.debug(f"HFT disabled for {symbol}")
            return None

        start_time = time.time()

        try:
            logger.debug(f"Generating HFT signal for {symbol}")

            # Check circuit breaker status
            if self.circuit_breaker.is_circuit_breaker_active():
                logger.warning(f"⚠️ Circuit breaker active - skipping signal generation for {symbol}")
                return None

            # Generate mock order book data
            order_book = await self._get_order_book_data(symbol)
            if not order_book:
                logger.warning(f"No order book data available for {symbol}")
                return None

            # Calculate OFI signal
            ofi_signal = await self._calculate_ofi_signal(symbol, order_book)

            # Calculate book skew signal
            skew_signal = await self._calculate_skew_signal(symbol, order_book)

            # Fuse signals for enhanced confidence
            fused_signal = await self._fuse_signals(symbol, ofi_signal, skew_signal)

            # Convert to HFTSignal format
            hft_signal = self._convert_to_hft_signal(symbol, fused_signal, order_book)

            # Update performance metrics
            self._update_performance_metrics(hft_signal, time.time() - start_time)

            logger.debug(f"✅ HFT signal generated for {symbol}: {hft_signal.direction} {hft_signal.confidence}%")
            return hft_signal

        except Exception as e:
            logger.error(f"❌ Error generating HFT signal for {symbol}: {e}")
            return None

    async def _get_order_book_data(self, symbol: str) -> Optional[OrderBook]:
        """Get order book data for symbol."""
        try:
            if self.mock_connector:
                # Generate mock order book
                return self.mock_connector.generate_order_book(symbol)
            else:
                # Create simple mock order book
                import random
                base_price = 50000.0 if 'BTC' in symbol else (3000.0 if 'ETH' in symbol else 100.0)

                bids = [
                    OrderBookLevel(
                        price=base_price - random.uniform(0, 10),
                        size=random.uniform(1, 20)
                    ) for _ in range(5)
                ]

                asks = [
                    OrderBookLevel(
                        price=base_price + random.uniform(0, 10),
                        size=random.uniform(1, 20)
                    ) for _ in range(5)
                ]

                return OrderBook(
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc),
                    exchange="mock",
                    bids=bids,
                    asks=asks
                )
        except Exception as e:
            logger.error(f"Error getting order book data for {symbol}: {e}")
            return None

    async def _calculate_ofi_signal(self, symbol: str, order_book: OrderBook) -> Optional[Dict]:
        """Calculate Order Flow Imbalance signal."""
        try:
            # Mock OFI calculation using Hawkes processes
            bid_sizes = [level.size for level in order_book.bids]
            ask_sizes = [level.size for level in order_book.asks]

            # Simple OFI calculation
            total_bid_size = sum(bid_sizes)
            total_ask_size = sum(ask_sizes)

            if total_ask_size > 0:
                ofi_value = (total_bid_size - total_ask_size) / total_ask_size
            else:
                ofi_value = 0

            # Determine signal direction and confidence
            if ofi_value > 0.1:
                direction = "long"
                confidence = min(abs(ofi_value) * 100, 90)
            elif ofi_value < -0.1:
                direction = "short"
                confidence = min(abs(ofi_value) * 100, 90)
            else:
                direction = "neutral"
                confidence = 50

            return {
                'type': 'ofi',
                'direction': direction,
                'confidence': confidence,
                'value': ofi_value,
                'bid_size': total_bid_size,
                'ask_size': total_ask_size
            }

        except Exception as e:
            logger.error(f"Error calculating OFI signal for {symbol}: {e}")
            return None

    async def _calculate_skew_signal(self, symbol: str, order_book: OrderBook) -> Optional[Dict]:
        """Calculate book skew signal."""
        try:
            bid_sizes = [level.size for level in order_book.bids]
            ask_sizes = [level.size for level in order_book.asks]

            # Calculate skew: log10(bid_size) - log10(ask_size)
            skew_value = calculate_skew(bid_sizes, ask_sizes)

            # Determine signal direction and confidence
            if skew_value > 0.05:
                direction = "long"
                confidence = min(abs(skew_value) * 500, 90)
            elif skew_value < -0.05:
                direction = "short"
                confidence = min(abs(skew_value) * 500, 90)
            else:
                direction = "neutral"
                confidence = 50

            return {
                'type': 'skew',
                'direction': direction,
                'confidence': confidence,
                'value': skew_value,
                'bid_sizes': bid_sizes,
                'ask_sizes': ask_sizes
            }

        except Exception as e:
            logger.error(f"Error calculating skew signal for {symbol}: {e}")
            return None

    async def _fuse_signals(self, symbol: str, ofi_signal: Dict, skew_signal: Dict) -> Dict:
        """Fuse multiple signals for enhanced confidence."""
        try:
            signals = [ofi_signal, skew_signal] if ofi_signal and skew_signal else []

            if not signals:
                return {'direction': 'neutral', 'confidence': 50, 'strength': 'weak'}

            # Simple consensus building
            directions = [s['direction'] for s in signals]
            confidences = [s['confidence'] for s in signals]

            # Count direction votes
            long_votes = directions.count('long')
            short_votes = directions.count('short')
            neutral_votes = directions.count('neutral')

            # Determine consensus direction
            if long_votes > short_votes and long_votes > neutral_votes:
                consensus_direction = 'long'
            elif short_votes > long_votes and short_votes > neutral_votes:
                consensus_direction = 'short'
            else:
                consensus_direction = 'neutral'

            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 50

            # Determine strength based on confidence
            if avg_confidence > 80:
                strength = 'strong'
            elif avg_confidence > 60:
                strength = 'moderate'
            else:
                strength = 'weak'

            return {
                'direction': consensus_direction,
                'confidence': avg_confidence,
                'strength': strength,
                'ofi_signal': ofi_signal['value'] if ofi_signal else 0,
                'skew_signal': skew_signal['value'] if skew_signal else 0,
                'component_signals': signals
            }

        except Exception as e:
            logger.error(f"Error fusing signals for {symbol}: {e}")
            return {'direction': 'neutral', 'confidence': 50, 'strength': 'weak'}

    def _convert_to_hft_signal(self, symbol: str, fused_signal: Dict, order_book: OrderBook) -> HFTSignal:
        """Convert fused signal to HFTSignal format."""
        rationale = []

        if fused_signal['ofi_signal'] != 0:
            rationale.append(f"OFI: {fused_signal['ofi_signal']:.3f}")

        if fused_signal['skew_signal'] != 0:
            rationale.append(f"Book Skew: {fused_signal['skew_signal']:.3f}")

        rationale.append(f"Consensus: {fused_signal['component_signals']} signals")

        return HFTSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=fused_signal['direction'],
            confidence=fused_signal['confidence'],
            strength=fused_signal['strength'],
            ofi_signal=fused_signal['ofi_signal'],
            book_skew=fused_signal['skew_signal'],
            fusion_confidence=fused_signal['confidence'],
            rationale=rationale,
            raw_hft_data={
                'order_book_depth': len(order_book.bids) + len(order_book.asks),
                'best_bid': order_book.bids[0].price if order_book.bids else None,
                'best_ask': order_book.asks[0].price if order_book.asks else None,
                'spread': (order_book.asks[0].price - order_book.bids[0].price) if order_book.bids and order_book.asks else None
            }
        )

    def _update_performance_metrics(self, signal: HFTSignal, generation_time: float):
        """Update performance tracking metrics."""
        self.performance_metrics['signals_generated'] += 1
        self.performance_metrics['ofi_calculations'] += 1
        self.performance_metrics['skew_calculations'] += 1
        self.performance_metrics['fusion_operations'] += 1

        # Update average confidence
        total_signals = self.performance_metrics['signals_generated']
        current_avg = self.performance_metrics['avg_signal_confidence']
        new_avg = (current_avg * (total_signals - 1) + signal.confidence) / total_signals
        self.performance_metrics['avg_signal_confidence'] = new_avg

        # Store signal history
        self.signal_history.append(signal)

        # Keep only last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

        logger.debug(f"HFT Performance: {self.performance_metrics['signals_generated']} signals, "
                    f"avg confidence: {new_avg:.1f}%, generation time: {generation_time*1000:.1f}ms")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.is_hft_enabled():
            return {'status': 'HFT disabled'}

        metrics = self.performance_metrics.copy()
        metrics['status'] = 'HFT active'
        metrics['circuit_breaker_status'] = 'active' if self.circuit_breaker.is_circuit_breaker_active() else 'inactive'
        metrics['signal_history_size'] = len(self.signal_history)

        # Add recent signal breakdown
        if self.signal_history:
            recent_signals = self.signal_history[-50:]  # Last 50 signals
            long_count = sum(1 for s in recent_signals if s.direction == 'long')
            short_count = sum(1 for s in recent_signals if s.direction == 'short')
            neutral_count = sum(1 for s in recent_signals if s.direction == 'neutral')

            metrics['recent_signal_breakdown'] = {
                'long': long_count,
                'short': short_count,
                'neutral': neutral_count,
                'total': len(recent_signals)
            }

        return metrics

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of HFT integration."""
        if not self.is_hft_enabled():
            return {
                'status': 'disabled',
                'reason': 'HFT not available or disabled',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        try:
            # Test basic functionality
            test_symbol = "BTC/USDT"
            test_signal = await self.generate_hft_signal(test_symbol)

            health_status = {
                'status': 'healthy' if test_signal else 'degraded',
                'hft_components_loaded': True,
                'signal_generation_test': 'passed' if test_signal else 'failed',
                'circuit_breaker_status': 'active' if self.circuit_breaker.is_circuit_breaker_active() else 'inactive',
                'performance_metrics': self.get_performance_metrics(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            return health_status

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }


# Utility function for easy integration
def create_hft_adapter(config_manager, enable_hft: bool = True) -> HFTIntegrationAdapter:
    """
    Factory function to create HFT integration adapter.

    Args:
        config_manager: Main bot configuration manager
        enable_hft: Whether to enable HFT signals

    Returns:
        HFTIntegrationAdapter instance
    """
    return HFTIntegrationAdapter(config_manager, enable_hft)