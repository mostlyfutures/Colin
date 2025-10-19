"""
Signal Integration Manager

Coordinates all signal processing components and manages the signal generation pipeline.
"""

import asyncio
from typing import List, Dict, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
import logging

from ..utils.data_structures import OrderBook, Trade, TradingSignal, SignalDirection
from ..utils.performance import LatencyTracker, PerformanceMonitor
from .ofi_calculator import OFICalculator
from .book_skew_analyzer import BookSkewAnalyzer
from .liquidity_detector import LiquidityDetector
from .signal_fusion import SignalFusionEngine, FusionMethod


class SignalIntegrationManager:
    """
    Central coordinator for all signal processing components.

    Manages the flow of market data through various signal generators
    and produces enhanced fused signals for trading decisions.
    """

    def __init__(self, symbols: List[str], config: Dict = None):
        self.symbols = symbols
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Performance monitoring
        self.latency_tracker = LatencyTracker()
        self.performance_monitor = PerformanceMonitor()

        # Signal processors
        self.ofi_calculator = OFICalculator(
            window_size=self.config.get('ofi_window_size', 100),
            decay_rate=self.config.get('ofi_decay_rate', 0.5)
        )
        self.skew_analyzer = BookSkewAnalyzer(
            window_size=self.config.get('skew_window_size', 50),
            sensitivity=self.config.get('skew_sensitivity', 0.1)
        )
        self.liquidity_detector = LiquidityDetector(
            window_size=self.config.get('liquidity_window_size', 30),
            sensitivity=self.config.get('liquidity_sensitivity', 0.15)
        )
        self.fusion_engine = SignalFusionEngine(
            window_size=self.config.get('fusion_window_size', 50),
            fusion_methods=[
                FusionMethod.CONSENSUS_BUILDER,
                FusionMethod.WEIGHTED_AVERAGE,
                FusionMethod.CONFLICT_RESOLUTION
            ]
        )

        # Signal storage
        self.active_signals: Dict[str, List] = defaultdict(list)
        self.signal_history: Dict[str, List] = defaultdict(list)
        self.fused_signals: Dict[str, List] = defaultdict(list)

        # Configuration
        self.signal_generation_interval = self.config.get('signal_generation_interval', 1.0)
        self.max_signal_history = self.config.get('max_signal_history', 1000)
        self.enable_signal_fusion = self.config.get('enable_signal_fusion', True)

        # Statistics
        self.signal_counts = defaultdict(int)
        self.fusion_counts = defaultdict(int)
        self.last_signal_times = defaultdict(lambda: datetime.min)

        # Subscribers for signal updates
        self.signal_subscribers: Set[callable] = set()

    async def initialize(self):
        """Initialize signal processing components."""
        self.logger.info("Initializing Signal Integration Manager")

        # Initialize components
        await self._initialize_components()

        self.logger.info(f"Signal Integration Manager initialized for {len(self.symbols)} symbols")

    async def _initialize_components(self):
        """Initialize individual signal processing components."""
        # Components are already initialized in __init__
        pass

    def subscribe_to_signals(self, callback: callable):
        """
        Subscribe to receive signal updates.

        Args:
            callback: Function to call when new signals are generated
        """
        self.signal_subscribers.add(callback)

    def unsubscribe_from_signals(self, callback: callable):
        """
        Unsubscribe from signal updates.

        Args:
            callback: Function to remove from subscribers
        """
        self.signal_subscribers.discard(callback)

    async def process_order_book(self, order_book: OrderBook):
        """
        Process order book update through all signal processors.

        Args:
            order_book: Current order book
        """
        symbol = order_book.symbol

        # Process through OFI calculator
        try:
            self.ofi_calculator.process_order_book_update(order_book)
        except Exception as e:
            self.logger.error(f"Error processing order book in OFI calculator for {symbol}: {e}")

        # Process through skew analyzer
        try:
            self.skew_analyzer.process_order_book(order_book)
        except Exception as e:
            self.logger.error(f"Error processing order book in skew analyzer for {symbol}: {e}")

        # Process through liquidity detector
        try:
            self.liquidity_detector.process_order_book(order_book)
        except Exception as e:
            self.logger.error(f"Error processing order book in liquidity detector for {symbol}: {e}")

        # Generate signals
        await self._generate_signals_for_symbol(symbol, order_book)

    async def process_trade(self, trade: Trade):
        """
        Process trade update through signal processors.

        Args:
            trade: Current trade
        """
        symbol = trade.symbol

        # Process through OFI calculator
        try:
            self.ofi_calculator.process_trade(trade)
        except Exception as e:
            self.logger.error(f"Error processing trade in OFI calculator for {symbol}: {e}")

    async def _generate_signals_for_symbol(self, symbol: str, order_book: OrderBook):
        """
        Generate signals for a specific symbol.

        Args:
            symbol: Trading symbol
            order_book: Current order book
        """
        try:
            # Check rate limiting
            current_time = datetime.now(timezone.utc)
            time_since_last = (current_time - self.last_signal_times[symbol]).total_seconds()

            if time_since_last < self.signal_generation_interval:
                return

            # Generate individual signals
            signals = []

            # OFI signal
            ofi_signal = await self.ofi_calculator.calculate_ofi(symbol)
            if ofi_signal:
                signals.append(ofi_signal)
                self.signal_counts[f'{symbol}_ofi'] += 1

            # Skew signal
            skew_signal = await self.skew_analyzer.analyze_skew(symbol, order_book)
            if skew_signal:
                signals.append(skew_signal)
                self.signal_counts[f'{symbol}_skew'] += 1

            # Liquidity signal
            liquidity_signal = await self.liquidity_detector.detect_liquidity_signal(symbol, order_book)
            if liquidity_signal:
                signals.append(liquidity_signal)
                self.signal_counts[f'{symbol}_liquidity'] += 1

            # Store signals
            self.active_signals[symbol] = signals
            self._add_to_signal_history(symbol, signals)

            # Generate fused signal if enabled
            if self.enable_signal_fusion and len(signals) >= 2:
                fused_signal = await self.fusion_engine.fuse_signals(symbol, signals)
                if fused_signal:
                    self.fused_signals[symbol].append(fused_signal)
                    self.fusion_counts[symbol] += 1

                    # Limit history size
                    if len(self.fused_signals[symbol]) > self.max_signal_history:
                        self.fused_signals[symbol] = self.fused_signals[symbol][-self.max_signal_history:]

                    # Notify subscribers
                    await self._notify_signal_subscribers(symbol, fused_signal, signals)

            self.last_signal_times[symbol] = current_time

        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")

    def _add_to_signal_history(self, symbol: str, signals: List):
        """Add signals to history with timestamp."""
        timestamped_signals = {
            'timestamp': datetime.now(timezone.utc),
            'signals': signals
        }
        self.signal_history[symbol].append(timestamped_signals)

        # Limit history size
        if len(self.signal_history[symbol]) > self.max_signal_history:
            self.signal_history[symbol] = self.signal_history[symbol][-self.max_signal_history:]

    async def _notify_signal_subscribers(self, symbol: str, fused_signal, component_signals):
        """Notify all subscribers of new signals."""
        notification_data = {
            'symbol': symbol,
            'fused_signal': fused_signal,
            'component_signals': component_signals,
            'timestamp': datetime.now(timezone.utc)
        }

        # Notify all subscribers
        for callback in self.signal_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification_data)
                else:
                    callback(notification_data)
            except Exception as e:
                self.logger.error(f"Error notifying signal subscriber: {e}")

    async def get_latest_signals(self, symbol: str) -> Dict:
        """
        Get the latest signals for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary containing latest signals
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'active_signals': self.active_signals.get(symbol, []),
            'fused_signal': None,
            'signal_counts': {}
        }

        # Get latest fused signal
        if self.fused_signals.get(symbol):
            result['fused_signal'] = self.fused_signals[symbol][-1]

        # Get signal counts
        result['signal_counts'] = {
            'ofi': self.signal_counts.get(f'{symbol}_ofi', 0),
            'skew': self.signal_counts.get(f'{symbol}_skew', 0),
            'liquidity': self.signal_counts.get(f'{symbol}_liquidity', 0),
            'fused': self.fusion_counts.get(symbol, 0)
        }

        return result

    async def get_signal_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get signal history for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of historical entries to return

        Returns:
            List of historical signal entries
        """
        history = self.signal_history.get(symbol, [])
        return history[-limit:] if limit > 0 else history

    async def get_fusion_statistics(self) -> Dict:
        """Get fusion engine statistics."""
        return self.fusion_engine.get_fusion_statistics()

    async def get_component_statistics(self) -> Dict:
        """Get statistics from individual signal components."""
        return {
            'ofi_calculator': self.ofi_calculator.get_status(),
            'skew_analyzer': self.skew_analyzer.get_status(),
            'liquidity_detector': self.liquidity_detector.get_status(),
            'signal_counts': dict(self.signal_counts),
            'fusion_counts': dict(self.fusion_counts)
        }

    async def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the signal integration manager."""
        return {
            'latency_stats': self.latency_tracker.get_statistics(),
            'performance_metrics': self.performance_monitor.get_all_metrics(),
            'active_symbols': len(self.symbols),
            'active_subscribers': len(self.signal_subscribers),
            'total_signals_generated': sum(self.signal_counts.values()),
            'total_fusions_generated': sum(self.fusion_counts.values())
        }

    def update_configuration(self, new_config: Dict):
        """
        Update configuration for signal components.

        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)

        # Update component configurations
        if 'ofi_window_size' in new_config:
            self.ofi_calculator.window_size = new_config['ofi_window_size']
        if 'ofi_decay_rate' in new_config:
            self.ofi_calculator.decay_rate = new_config['ofi_decay_rate']
        if 'skew_window_size' in new_config:
            self.skew_analyzer.window_size = new_config['skew_window_size']
        if 'skew_sensitivity' in new_config:
            self.skew_analyzer.sensitivity = new_config['skew_sensitivity']
        if 'signal_generation_interval' in new_config:
            self.signal_generation_interval = new_config['signal_generation_interval']
        if 'enable_signal_fusion' in new_config:
            self.enable_signal_fusion = new_config['enable_signal_fusion']

        self.logger.info("Signal Integration Manager configuration updated")

    def reset_all_components(self):
        """Reset all signal processing components."""
        self.ofi_calculator.reset()
        self.skew_analyzer.reset()
        self.liquidity_detector.reset()
        self.fusion_engine.reset()

        # Clear stored signals
        self.active_signals.clear()
        self.signal_history.clear()
        self.fused_signals.clear()

        # Reset statistics
        self.signal_counts.clear()
        self.fusion_counts.clear()
        self.last_signal_times.clear()

        self.logger.info("All signal processing components reset")

    async def health_check(self) -> Dict:
        """
        Perform health check on all components.

        Returns:
            Health status of all components
        """
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }

        # Check OFI calculator
        try:
            ofi_status = self.ofi_calculator.get_status()
            health_status['components']['ofi_calculator'] = {
                'status': 'healthy',
                'details': ofi_status
            }
        except Exception as e:
            health_status['components']['ofi_calculator'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['issues'].append(f"OFI Calculator: {e}")
            health_status['overall_status'] = 'degraded'

        # Check skew analyzer
        try:
            skew_status = self.skew_analyzer.get_status()
            health_status['components']['skew_analyzer'] = {
                'status': 'healthy',
                'details': skew_status
            }
        except Exception as e:
            health_status['components']['skew_analyzer'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['issues'].append(f"Skew Analyzer: {e}")
            health_status['overall_status'] = 'degraded'

        # Check liquidity detector
        try:
            liquidity_status = self.liquidity_detector.get_status()
            health_status['components']['liquidity_detector'] = {
                'status': 'healthy',
                'details': liquidity_status
            }
        except Exception as e:
            health_status['components']['liquidity_detector'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['issues'].append(f"Liquidity Detector: {e}")
            health_status['overall_status'] = 'degraded'

        # Check fusion engine
        try:
            fusion_status = self.fusion_engine.get_fusion_statistics()
            health_status['components']['fusion_engine'] = {
                'status': 'healthy',
                'details': fusion_status
            }
        except Exception as e:
            health_status['components']['fusion_engine'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['issues'].append(f"Fusion Engine: {e}")
            health_status['overall_status'] = 'degraded'

        return health_status