"""
Order Flow Analyzer for institutional signal analysis.

This module analyzes order book imbalance, trade delta, and order flow
patterns to identify smart money activity and predictive signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..core.config import ConfigManager
from ..adapters.binance import BinanceAdapter


class OrderFlowSignal(Enum):
    """Types of order flow signals."""
    BUYING_PRESSURE = "buying_pressure"
    SELLING_PRESSURE = "selling_pressure"
    ABSORPTION = "absorption"
    STOP_RUN = "stop_run"
    LIQUIDITY_GRAB = "liquidity_grab"
    EXHAUSTION = "exhaustion"


@dataclass
class OrderFlowMetrics:
    """Order flow analysis metrics."""
    timestamp: datetime
    normalized_order_book_imbalance: float
    trade_delta: float
    volume_weighted_delta: float
    aggressive_buy_ratio: float
    aggressive_sell_ratio: float
    bid_ask_spread: float
    market_depth_ratio: float
    signal_strength: float
    signal_type: Optional[OrderFlowSignal] = None


@dataclass
class LiquidityAnalysis:
    """Liquidity analysis results."""
    total_bid_liquidity: float
    total_ask_liquidity: float
    liquidity_ratio: float
    liquidity_imbalance: float
    large_order_clusters: List[Dict[str, Any]]
    stop_loss_clusters: List[Dict[str, Any]]
    iceberg_probability: float


class OrderFlowAnalyzer:
    """Analyzes order flow data for institutional signals."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize order flow analyzer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.order_flow_config = self.config.order_flow
        self.imbalance_threshold = self.order_flow_config['order_book']['imbalance_threshold']
        self.depth_levels = self.order_flow_config['order_book']['depth_levels']
        self.lookback_minutes = self.order_flow_config['trade_delta']['lookback_minutes']
        self.volume_threshold = self.order_flow_config['trade_delta']['volume_threshold']

    async def analyze_order_book(self, symbol: str, binance_adapter: BinanceAdapter) -> Dict[str, Any]:
        """
        Analyze order book for imbalance and liquidity patterns.

        Args:
            symbol: Trading symbol
            binance_adapter: Binance adapter instance

        Returns:
            Dictionary with order book analysis
        """
        try:
            # Fetch order book data
            orderbook = await binance_adapter.get_order_book(symbol, self.depth_levels)
            bids_df = orderbook['bids']
            asks_df = orderbook['asks']

            # Calculate basic metrics
            bid_liquidity = (bids_df['price'] * bids_df['volume']).sum()
            ask_liquidity = (asks_df['price'] * asks_df['volume']).sum()
            total_liquidity = bid_liquidity + ask_liquidity

            # Normalized Order Book Imbalance (NOBI)
            nobi = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0

            # Market depth analysis
            bid_depth = self._calculate_market_depth(bids_df)
            ask_depth = self._calculate_market_depth(asks_df)
            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else 0

            # Bid-ask spread
            best_bid = bids_df['price'].iloc[0]
            best_ask = asks_df['price'].iloc[0]
            spread = (best_ask - best_bid) / best_bid

            # Identify large order clusters
            large_orders = self._identify_large_orders(bids_df, asks_df)

            # Iceberg detection
            iceberg_probability = self._detect_iceberg_orders(bids_df, asks_df)

            # Liquidity analysis
            liquidity_analysis = LiquidityAnalysis(
                total_bid_liquidity=bid_liquidity,
                total_ask_liquidity=ask_liquidity,
                liquidity_ratio=bid_liquidity / ask_liquidity if ask_liquidity > 0 else 0,
                liquidity_imbalance=abs(bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0,
                large_order_clusters=large_orders,
                stop_loss_clusters=[],  # Would need historical data
                iceberg_probability=iceberg_probability
            )

            # Determine signal
            signal_type = self._determine_order_book_signal(nobi, depth_ratio, spread, large_orders)
            signal_strength = abs(nobi)

            analysis = {
                'timestamp': datetime.now(),
                'normalized_order_book_imbalance': nobi,
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'liquidity_ratio': liquidity_analysis.liquidity_ratio,
                'market_depth_ratio': depth_ratio,
                'bid_ask_spread': spread,
                'signal_type': signal_type.value if signal_type else None,
                'signal_strength': signal_strength,
                'liquidity_analysis': liquidity_analysis,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': (best_bid + best_ask) / 2
            }

            logger.debug(f"Order book analysis for {symbol}: NOBI={nobi:.3f}, Signal={signal_type}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze order book for {symbol}: {e}")
            raise

    async def analyze_trade_delta(self, symbol: str, binance_adapter: BinanceAdapter) -> Dict[str, Any]:
        """
        Analyze trade delta and aggressive trading behavior.

        Args:
            symbol: Trading symbol
            binance_adapter: Binance adapter instance

        Returns:
            Dictionary with trade delta analysis
        """
        try:
            # Fetch recent trades
            trades_df = await binance_adapter.get_recent_trades(symbol, limit=1000)

            if trades_df.empty:
                return self._empty_trade_delta_analysis()

            # Calculate trade metrics
            total_buy_volume = trades_df[trades_df['side'] == 'buy']['amount'].sum()
            total_sell_volume = trades_df[trades_df['side'] == 'sell']['amount'].sum()
            total_volume = total_buy_volume + total_sell_volume

            # Trade delta (buy volume - sell volume)
            trade_delta = total_buy_volume - total_sell_volume
            normalized_delta = trade_delta / total_volume if total_volume > 0 else 0

            # Volume-weighted delta
            trades_df['signed_volume'] = trades_df.apply(
                lambda x: x['amount'] if x['side'] == 'buy' else -x['amount'], axis=1
            )
            volume_weighted_delta = (trades_df['signed_volume'] * trades_df['price']).sum()

            # Aggressive trading ratios
            aggressive_buy_ratio = total_buy_volume / total_volume if total_volume > 0 else 0
            aggressive_sell_ratio = total_sell_volume / total_volume if total_volume > 0 else 0

            # Time-based analysis
            trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
            recent_trades = trades_df[
                trades_df['datetime'] > datetime.now() - timedelta(minutes=self.lookback_minutes)
            ]

            if not recent_trades.empty:
                recent_buy_volume = recent_trades[recent_trades['side'] == 'buy']['amount'].sum()
                recent_sell_volume = recent_trades[recent_trades['side'] == 'sell']['amount'].sum()
                recent_delta = recent_buy_volume - recent_sell_volume
                momentum_score = recent_delta / (recent_buy_volume + recent_sell_volume) if (recent_buy_volume + recent_sell_volume) > 0 else 0
            else:
                momentum_score = 0

            # Detect trading patterns
            patterns = self._detect_trade_patterns(trades_df)

            # Determine signal
            signal_type = self._determine_trade_signal(normalized_delta, momentum_score, patterns)
            signal_strength = abs(normalized_delta)

            analysis = {
                'timestamp': datetime.now(),
                'trade_delta': trade_delta,
                'normalized_trade_delta': normalized_delta,
                'volume_weighted_delta': volume_weighted_delta,
                'aggressive_buy_ratio': aggressive_buy_ratio,
                'aggressive_sell_ratio': aggressive_sell_ratio,
                'momentum_score': momentum_score,
                'total_volume': total_volume,
                'signal_type': signal_type.value if signal_type else None,
                'signal_strength': signal_strength,
                'patterns': patterns,
                'trade_count': len(trades_df),
                'volume_threshold_met': total_volume >= self.volume_threshold
            }

            logger.debug(f"Trade delta analysis for {symbol}: Delta={normalized_delta:.3f}, Signal={signal_type}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze trade delta for {symbol}: {e}")
            raise

    async def comprehensive_analysis(self, symbol: str, binance_adapter: BinanceAdapter) -> OrderFlowMetrics:
        """
        Perform comprehensive order flow analysis.

        Args:
            symbol: Trading symbol
            binance_adapter: Binance adapter instance

        Returns:
            Complete order flow metrics
        """
        try:
            # Get both analyses
            orderbook_analysis = await self.analyze_order_book(symbol, binance_adapter)
            trade_analysis = await self.analyze_trade_delta(symbol, binance_adapter)

            # Combine signals
            combined_signal = self._combine_signals(
                orderbook_analysis.get('signal_type'),
                trade_analysis.get('signal_type'),
                orderbook_analysis.get('signal_strength', 0),
                trade_analysis.get('signal_strength', 0)
            )

            # Create comprehensive metrics
            metrics = OrderFlowMetrics(
                timestamp=datetime.now(),
                normalized_order_book_imbalance=orderbook_analysis.get('normalized_order_book_imbalance', 0),
                trade_delta=trade_analysis.get('normalized_trade_delta', 0),
                volume_weighted_delta=trade_analysis.get('volume_weighted_delta', 0),
                aggressive_buy_ratio=trade_analysis.get('aggressive_buy_ratio', 0),
                aggressive_sell_ratio=trade_analysis.get('aggressive_sell_ratio', 0),
                bid_ask_spread=orderbook_analysis.get('bid_ask_spread', 0),
                market_depth_ratio=orderbook_analysis.get('market_depth_ratio', 1),
                signal_strength=combined_signal['strength'],
                signal_type=combined_signal['type']
            )

            logger.debug(f"Comprehensive order flow analysis for {symbol}: Signal={combined_signal['type']}, Strength={combined_signal['strength']:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Failed comprehensive order flow analysis for {symbol}: {e}")
            raise

    def _calculate_market_depth(self, df: pd.DataFrame) -> float:
        """Calculate market depth metric."""
        # Weighted depth: more weight to levels closer to mid price
        weights = np.exp(-np.arange(len(df)) * 0.1)
        weighted_volume = (df['volume'] * weights).sum()
        return weighted_volume

    def _identify_large_orders(self, bids_df: pd.DataFrame, asks_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify unusually large orders in the book."""
        large_orders = []

        # Calculate volume thresholds
        bid_threshold = bids_df['volume'].quantile(0.9)
        ask_threshold = asks_df['volume'].quantile(0.9)

        # Find large bid orders
        large_bids = bids_df[bids_df['volume'] > bid_threshold]
        for _, row in large_bids.iterrows():
            large_orders.append({
                'side': 'bid',
                'price': row['price'],
                'volume': row['volume'],
                'usd_value': row['price'] * row['volume'],
                'significance': row['volume'] / bid_threshold
            })

        # Find large ask orders
        large_asks = asks_df[asks_df['volume'] > ask_threshold]
        for _, row in large_asks.iterrows():
            large_orders.append({
                'side': 'ask',
                'price': row['price'],
                'volume': row['volume'],
                'usd_value': row['price'] * row['volume'],
                'significance': row['volume'] / ask_threshold
            })

        return large_orders

    def _detect_iceberg_orders(self, bids_df: pd.DataFrame, asks_df: pd.DataFrame) -> float:
        """
        Detect probability of iceberg orders in the book.

        Iceberg orders are large orders broken into smaller pieces.
        """
        # Simple heuristic: look for multiple orders at similar price levels
        # with similar volumes
        bid_prices = bids_df['price'].values
        ask_prices = asks_df['price'].values

        # Calculate price gaps
        bid_gaps = np.diff(bid_prices)
        ask_gaps = np.diff(ask_prices)

        # Look for small gaps (indicative of iceberg orders)
        small_bid_gaps = np.sum(bid_gaps < np.median(bid_gaps) * 0.5)
        small_ask_gaps = np.sum(ask_gaps < np.median(ask_gaps) * 0.5)

        total_gaps = len(bid_gaps) + len(ask_gaps)
        small_gap_ratio = (small_bid_gaps + small_ask_gaps) / total_gaps if total_gaps > 0 else 0

        return min(small_gap_ratio, 1.0)

    def _determine_order_book_signal(
        self,
        nobi: float,
        depth_ratio: float,
        spread: float,
        large_orders: List[Dict[str, Any]]
    ) -> Optional[OrderFlowSignal]:
        """Determine signal based on order book data."""

        # Strong buying pressure
        if nobi > self.imbalance_threshold and depth_ratio > 1.2:
            return OrderFlowSignal.BUYING_PRESSURE

        # Strong selling pressure
        elif nobi < -self.imbalance_threshold and depth_ratio < 0.8:
            return OrderFlowSignal.SELLING_PRESSURE

        # Absorption (high volume but small price moves)
        elif abs(nobi) < 0.1 and len(large_orders) > 3:
            return OrderFlowSignal.ABSORPTION

        # Exhaustion (wide spread, low liquidity)
        elif spread > 0.001 and abs(nobi) < 0.2:
            return OrderFlowSignal.EXHAUSTION

        return None

    def _detect_trade_patterns(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in trade data."""
        patterns = {
            'volume_spike': False,
            'pressure_shift': False,
            'stop_run_candidate': False,
            'liquidity_grab_candidate': False
        }

        if trades_df.empty:
            return patterns

        # Calculate rolling metrics
        trades_df = trades_df.sort_values('datetime')
        trades_df['rolling_volume'] = trades_df['amount'].rolling(window=10).mean()
        trades_df['volume_ratio'] = trades_df['amount'] / trades_df['rolling_volume']

        # Volume spike detection
        if trades_df['volume_ratio'].max() > 2.0:
            patterns['volume_spike'] = True

        # Pressure shift detection
        trades_df['buy_pressure'] = (trades_df['side'] == 'buy').rolling(window=10).mean()
        if len(trades_df) > 20:
            early_pressure = trades_df['buy_pressure'].iloc[:10].mean()
            late_pressure = trades_df['buy_pressure'].iloc[-10:].mean()
            if abs(late_pressure - early_pressure) > 0.3:
                patterns['pressure_shift'] = True

        # Stop run candidates (large volume at extremes)
        price_range = trades_df['price'].max() - trades_df['price'].min()
        if price_range > 0:
            upper_extreme = trades_df['price'].quantile(0.95)
            lower_extreme = trades_df['price'].quantile(0.05)

            extreme_trades = trades_df[
                (trades_df['price'] >= upper_extreme) | (trades_df['price'] <= lower_extreme)
            ]

            if len(extreme_trades) > 0 and extreme_trades['amount'].mean() > trades_df['amount'].mean() * 1.5:
                patterns['stop_run_candidate'] = True

        return patterns

    def _determine_trade_signal(
        self,
        normalized_delta: float,
        momentum_score: float,
        patterns: Dict[str, Any]
    ) -> Optional[OrderFlowSignal]:
        """Determine signal based on trade data."""

        # Strong buying momentum
        if normalized_delta > 0.2 and momentum_score > 0.1:
            return OrderFlowSignal.BUYING_PRESSURE

        # Strong selling momentum
        elif normalized_delta < -0.2 and momentum_score < -0.1:
            return OrderFlowSignal.SELLING_PRESSURE

        # Liquidity grab (volume spike with reversal pattern)
        elif patterns['volume_spike'] and patterns['pressure_shift']:
            return OrderFlowSignal.LIQUIDITY_GRAB

        # Stop run (extreme price movement with high volume)
        elif patterns['stop_run_candidate']:
            return OrderFlowSignal.STOP_RUN

        return None

    def _combine_signals(
        self,
        orderbook_signal: Optional[str],
        trade_signal: Optional[str],
        orderbook_strength: float,
        trade_strength: float
    ) -> Dict[str, Any]:
        """Combine order book and trade signals."""

        signals = []
        if orderbook_signal:
            signals.append((orderbook_signal, orderbook_strength))
        if trade_signal:
            signals.append((trade_signal, trade_strength))

        if not signals:
            return {'type': None, 'strength': 0.0}

        # Weight trade signals more heavily
        weighted_signals = []
        for signal, strength in signals:
            if signal in ['normalized_trade_delta', 'volume_weighted_delta']:
                weighted_strength = strength * 0.6
            else:
                weighted_strength = strength * 0.4
            weighted_signals.append((signal, weighted_strength))

        # Choose strongest signal
        strongest_signal = max(weighted_signals, key=lambda x: x[1])

        # Map signal names to enum values
        signal_mapping = {
            'buying_pressure': OrderFlowSignal.BUYING_PRESSURE,
            'selling_pressure': OrderFlowSignal.SELLING_PRESSURE,
            'absorption': OrderFlowSignal.ABSORPTION,
            'stop_run': OrderFlowSignal.STOP_RUN,
            'liquidity_grab': OrderFlowSignal.LIQUIDITY_GRAB,
            'exhaustion': OrderFlowSignal.EXHAUSTION
        }

        signal_type = signal_mapping.get(strongest_signal[0])

        return {
            'type': signal_type,
            'strength': min(strongest_signal[1], 1.0),
            'components': {
                'orderbook_signal': orderbook_signal,
                'trade_signal': trade_signal,
                'orderbook_strength': orderbook_strength,
                'trade_strength': trade_strength
            }
        }

    def _empty_trade_delta_analysis(self) -> Dict[str, Any]:
        """Return empty trade delta analysis when no data available."""
        return {
            'timestamp': datetime.now(),
            'trade_delta': 0,
            'normalized_trade_delta': 0,
            'volume_weighted_delta': 0,
            'aggressive_buy_ratio': 0,
            'aggressive_sell_ratio': 0,
            'momentum_score': 0,
            'total_volume': 0,
            'signal_type': None,
            'signal_strength': 0,
            'patterns': {},
            'trade_count': 0,
            'volume_threshold_met': False
        }