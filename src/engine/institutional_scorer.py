"""
Institutional Signal Scoring Engine.

This is the main engine that aggregates all institutional signals into
a comprehensive Long/Short Confidence score with rationale generation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger

from ..core.config import ConfigManager
from ..adapters.binance import BinanceAdapter
from ..adapters.coinglass import CoinGlassAdapter
from ..structure.ict_detector import ICTDetector
from ..orderflow.analyzer import OrderFlowAnalyzer
from ..scorers.liquidity_scorer import LiquidityScorer
from ..scorers.ict_scorer import ICTScorer
from ..scorers.killzone_scorer import KillzoneScorer
from ..scorers.volume_oi_scorer import VolumeOIScorer


@dataclass
class InstitutionalSignal:
    """Represents a comprehensive institutional signal."""
    symbol: str
    timestamp: datetime
    long_confidence: float  # 0-100%
    short_confidence: float  # 0-100%
    direction: str  # "long", "short", or "neutral"
    confidence_level: str  # "high", "medium", "low"
    rationale_points: List[str]
    risk_metrics: Dict[str, Any]
    entry_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    position_size: float  # Recommended position size
    time_horizon: str  # "1h", "4h", "24h"
    institutional_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class SignalComponents:
    """Individual signal components used in scoring."""
    liquidity_score: float
    ict_score: float
    killzone_score: float
    order_flow_score: float
    volume_oi_score: float
    component_weights: Dict[str, float]


class InstitutionalScorer:
    """Main institutional signal scoring engine."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize institutional scorer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.scoring_weights = self.config.scoring.weights

        # Initialize all components
        self.binance_adapter = BinanceAdapter(config_manager)
        self.coinglass_adapter = CoinGlassAdapter(config_manager)
        self.ict_detector = ICTDetector(config_manager)
        self.orderflow_analyzer = OrderFlowAnalyzer(config_manager)

        # Initialize scorers
        self.liquidity_scorer = LiquidityScorer(config_manager)
        self.ict_scorer = ICTScorer(config_manager)
        self.killzone_scorer = KillzoneScorer(config_manager)
        self.volume_oi_scorer = VolumeOIScorer(config_manager)

    async def analyze_symbol(
        self,
        symbol: str,
        current_price: Optional[float] = None,
        time_horizon: str = "4h"
    ) -> InstitutionalSignal:
        """
        Perform comprehensive institutional analysis for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            current_price: Current price level (optional, will fetch if not provided)
            time_horizon: Analysis time horizon ("1h", "4h", "24h")

        Returns:
            Comprehensive institutional signal
        """
        try:
            logger.info(f"Starting institutional analysis for {symbol}")

            # Get current price if not provided
            if current_price is None:
                ticker = await self.binance_adapter.get_ticker(symbol)
                current_price = float(ticker['last'])

            # Collect all data concurrently
            data_tasks = [
                self._collect_market_data(symbol, time_horizon),
                self._collect_orderflow_data(symbol),
                self._collect_liquidation_data(symbol)
            ]

            market_data, orderflow_data, liquidation_data = await asyncio.gather(*data_tasks)

            # Score all components
            component_scores = await self._score_all_components(
                symbol, current_price, market_data, orderflow_data, liquidation_data
            )

            # Generate final signal
            signal = self._generate_final_signal(
                symbol, current_price, component_scores, market_data
            )

            # Add risk metrics and position sizing
            signal.risk_metrics = self._calculate_risk_metrics(
                current_price, component_scores, market_data
            )
            signal.position_size = self._calculate_position_size(
                signal.risk_metrics, signal.confidence_level
            )

            logger.info(f"Analysis complete for {symbol}: {signal.direction} "
                       f"({signal.long_confidence:.1f}% vs {signal.short_confidence:.1f}%)")

            return signal

        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")
            return self._create_error_signal(symbol, current_price, str(e))

    async def batch_analyze(
        self,
        symbols: List[str],
        time_horizon: str = "4h"
    ) -> List[InstitutionalSignal]:
        """
        Analyze multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            time_horizon: Analysis time horizon

        Returns:
            List of institutional signals
        """
        logger.info(f"Starting batch analysis for {len(symbols)} symbols")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent analyses

        async def analyze_with_semaphore(symbol: str) -> InstitutionalSignal:
            async with semaphore:
                return await self.analyze_symbol(symbol, time_horizon=time_horizon)

        # Run analyses concurrently
        signals = await asyncio.gather(
            *[analyze_with_semaphore(symbol) for symbol in symbols],
            return_exceptions=True
        )

        # Filter out exceptions and log them
        valid_signals = []
        for symbol, result in zip(symbols, signals):
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for {symbol}: {result}")
                valid_signals.append(self._create_error_signal(symbol, None, str(result)))
            else:
                valid_signals.append(result)

        logger.info(f"Batch analysis complete: {len(valid_signals)} signals generated")
        return valid_signals

    async def _collect_market_data(
        self,
        symbol: str,
        time_horizon: str
    ) -> Dict[str, Any]:
        """Collect market data from Binance."""
        try:
            # Determine timeframe and limit based on time horizon
            if time_horizon == "1h":
                timeframe = "5m"
                limit = 100
            elif time_horizon == "4h":
                timeframe = "15m"
                limit = 100
            else:  # 24h
                timeframe = "1h"
                limit = 100

            # Collect data concurrently
            ohlcv_task = self.binance_adapter.get_ohlcv(symbol, timeframe, limit)
            oi_task = self.binance_adapter.get_open_interest(symbol)
            funding_task = self.binance_adapter.get_funding_rate(symbol)

            ohlcv_df, oi_data, funding_data = await asyncio.gather(
                ohlcv_task, oi_task, funding_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(ohlcv_df, Exception):
                logger.warning(f"Failed to get OHLCV for {symbol}: {ohlcv_df}")
                ohlcv_df = pd.DataFrame()

            if isinstance(oi_data, Exception):
                logger.warning(f"Failed to get OI for {symbol}: {oi_data}")
                oi_data = {}

            if isinstance(funding_data, Exception):
                logger.warning(f"Failed to get funding for {symbol}: {funding_data}")
                funding_data = {}

            return {
                'ohlcv': ohlcv_df,
                'open_interest': oi_data,
                'funding_rate': funding_data,
                'time_horizon': time_horizon,
                'current_volume': ohlcv_df['volume'].iloc[-1] if not ohlcv_df.empty else 0,
                'avg_volume': ohlcv_df['volume'].mean() if not ohlcv_df.empty else 0,
                'volume_trend': self._calculate_volume_trend(ohlcv_df) if not ohlcv_df.empty else 0,
                'price_change': self._calculate_price_change(ohlcv_df) if not ohlcv_df.empty else 0
            }

        except Exception as e:
            logger.error(f"Failed to collect market data for {symbol}: {e}")
            return {}

    async def _collect_orderflow_data(self, symbol: str) -> Dict[str, Any]:
        """Collect order flow data from Binance."""
        try:
            # Get order book and recent trades
            orderbook_task = self.binance_adapter.get_order_book(symbol, limit=20)
            trades_task = self.binance_adapter.get_recent_trades(symbol, limit=500)

            orderbook_data, trades_data = await asyncio.gather(
                orderbook_task, trades_task, return_exceptions=True
            )

            if isinstance(orderbook_data, Exception):
                logger.warning(f"Failed to get orderbook for {symbol}: {orderbook_data}")
                orderbook_data = {}

            if isinstance(trades_data, Exception):
                logger.warning(f"Failed to get trades for {symbol}: {trades_data}")
                trades_data = pd.DataFrame()

            return {
                'orderbook': orderbook_data,
                'trades': trades_data
            }

        except Exception as e:
            logger.error(f"Failed to collect orderflow data for {symbol}: {e}")
            return {}

    async def _collect_liquidation_data(self, symbol: str) -> Dict[str, Any]:
        """Collect liquidation data from CoinGlass."""
        try:
            # Get liquidation heatmap and levels
            heatmap_task = self.coinglass_adapter.get_liquidation_heatmap(symbol)
            levels_task = self.coinglass_adapter.get_liquidation_levels(symbol)
            indicators_task = self.coinglass_adapter.get_liquidation_indicators(symbol)

            heatmap_data, levels_data, indicators_data = await asyncio.gather(
                heatmap_task, levels_task, indicators_task, return_exceptions=True
            )

            if isinstance(heatmap_data, Exception):
                logger.warning(f"Failed to get liquidation heatmap for {symbol}: {heatmap_data}")
                heatmap_data = pd.DataFrame()

            if isinstance(levels_data, Exception):
                logger.warning(f"Failed to get liquidation levels for {symbol}: {levels_data}")
                levels_data = {}

            if isinstance(indicators_data, Exception):
                logger.warning(f"Failed to get liquidation indicators for {symbol}: {indicators_data}")
                indicators_data = {}

            # Convert to common format
            density_clusters = self._convert_liquidation_data(heatmap_data, levels_data)

            return {
                'heatmap': heatmap_data,
                'levels': levels_data,
                'indicators': indicators_data,
                'density_clusters': density_clusters
            }

        except Exception as e:
            logger.error(f"Failed to collect liquidation data for {symbol}: {e}")
            return {}

    async def _score_all_components(
        self,
        symbol: str,
        current_price: float,
        market_data: Dict[str, Any],
        orderflow_data: Dict[str, Any],
        liquidation_data: Dict[str, Any]
    ) -> SignalComponents:
        """Score all institutional components."""

        # Score liquidity
        liquidity_score_obj = self.liquidity_scorer.score_liquidation_proximity(
            current_price, liquidation_data, orderflow_data.get('orderbook', {})
        )
        liquidity_score = liquidity_score_obj.overall_score

        # Score ICT structures
        ict_score_obj = self.ict_scorer.score_ict_confluence(
            market_data.get('ohlcv', pd.DataFrame()), current_price
        )
        ict_score = ict_score_obj.overall_score

        # Score killzone timing
        killzone_score_obj = self.killzone_scorer.score_killzone_timing()
        killzone_score = killzone_score_obj.overall_score

        # Score order flow
        orderflow_metrics = await self.orderflow_analyzer.综合分析(
            symbol, self.binance_adapter
        )
        order_flow_score = orderflow_metrics.signal_strength

        # Score volume/OI
        volume_oi_score_obj = self.volume_oi_scorer.score_volume_oi_signals(
            current_price, market_data, market_data.get('open_interest', {}),
            market_data.get('funding_rate')
        )
        volume_oi_score = volume_oi_score_obj.overall_score

        return SignalComponents(
            liquidity_score=liquidity_score,
            ict_score=ict_score,
            killzone_score=killzone_score,
            order_flow_score=order_flow_score,
            volume_oi_score=volume_oi_score,
            component_weights=self.scoring_weights
        )

    def _generate_final_signal(
        self,
        symbol: str,
        current_price: float,
        component_scores: SignalComponents,
        market_data: Dict[str, Any]
    ) -> InstitutionalSignal:
        """Generate the final institutional signal."""

        # Calculate weighted scores
        long_score = self._calculate_weighted_direction_score(
            component_scores, "long", current_price, market_data
        )
        short_score = self._calculate_weighted_direction_score(
            component_scores, "short", current_price, market_data
        )

        # Normalize to percentages
        total_score = long_score + short_score
        if total_score > 0:
            long_confidence = (long_score / total_score) * 100
            short_confidence = (short_score / total_score) * 100
        else:
            long_confidence = 50.0
            short_confidence = 50.0

        # Determine direction
        if long_confidence > short_confidence + 10:  # 10% threshold
            direction = "long"
        elif short_confidence > long_confidence + 10:
            direction = "short"
        else:
            direction = "neutral"

        # Determine confidence level
        max_confidence = max(long_confidence, short_confidence)
        if max_confidence >= 80:
            confidence_level = "high"
        elif max_confidence >= 60:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Generate rationale
        rationale_points = self._generate_rationale(
            component_scores, direction, current_price, market_data
        )

        # Calculate entry and stop levels
        entry_price = current_price
        stop_loss_price = self._calculate_stop_loss(
            current_price, direction, component_scores, market_data
        )
        take_profit_price = self._calculate_take_profit(
            current_price, direction, component_scores, market_data
        )

        return InstitutionalSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            long_confidence=long_confidence,
            short_confidence=short_confidence,
            direction=direction,
            confidence_level=confidence_level,
            rationale_points=rationale_points[:3],  # Top 3 rationale points
            risk_metrics={},  # Will be filled later
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            position_size=0.0,  # Will be calculated later
            time_horizon=market_data.get('time_horizon', '4h'),
            institutional_factors={
                'liquidity': component_scores.liquidity_score,
                'ict': component_scores.ict_score,
                'killzone': component_scores.killzone_score,
                'order_flow': component_scores.order_flow_score,
                'volume_oi': component_scores.volume_oi_score
            }
        )

    def _calculate_weighted_direction_score(
        self,
        component_scores: SignalComponents,
        direction: str,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate weighted directional score."""

        # Base component scores
        liquidity_bias = component_scores.liquidity_score
        ict_bias = component_scores.ict_score
        killzone_bias = component_scores.killzone_score
        order_flow_bias = component_scores.order_flow_score
        volume_oi_bias = component_scores.volume_oi_score

        # Directional adjustments based on market conditions
        if direction == "long":
            # Long signals get boosted by bullish order flow and volume
            order_flow_bias *= 1.2 if order_flow_bias > 0.5 else 0.8
            volume_oi_bias *= 1.1 if volume_oi_bias > 0.5 else 0.9
        else:
            # Short signals get boosted by bearish order flow and volume
            order_flow_bias *= 1.2 if order_flow_bias < -0.5 else 0.8
            volume_oi_bias *= 1.1 if volume_oi_bias < -0.5 else 0.9

        # Apply weights
        weighted_score = (
            liquidity_bias * component_scores.component_weights.get('liquidity_proximity', 0.25) +
            ict_bias * component_scores.component_weights.get('ict_confluence', 0.25) +
            killzone_bias * component_scores.component_weights.get('killzone_alignment', 0.15) +
            abs(order_flow_bias) * component_scores.component_weights.get('order_flow_delta', 0.20) +
            volume_oi_bias * component_scores.component_weights.get('volume_oi_confirmation', 0.15)
        )

        return max(weighted_score, 0.0)

    def _generate_rationale(
        self,
        component_scores: SignalComponents,
        direction: str,
        current_price: float,
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Generate rationale points for the signal."""

        rationale_points = []

        # Liquidity rationale
        if component_scores.liquidity_score > 0.7:
            rationale_points.append(
                "Strong liquidity confluence with untested liquidation clusters"
            )
        elif component_scores.liquidity_score > 0.4:
            rationale_points.append(
                "Moderate liquidity conditions supporting directional bias"
            )

        # ICT rationale
        if component_scores.ict_score > 0.7:
            rationale_points.append(
                "High ICT structure confluence (FVG + Order Block + BOS)"
            )
        elif component_scores.ict_score > 0.4:
            rationale_points.append(
                "Price approaching fresh institutional structure"
            )

        # Killzone rationale
        if component_scores.killzone_score > 0.8:
            rationale_points.append(
                "Optimal timing during peak institutional session (London/NY overlap)"
            )
        elif component_scores.killzone_score > 0.5:
            rationale_points.append(
                "Alignment with institutional trading session"
            )

        # Order flow rationale
        if component_scores.order_flow_score > 0.7:
            if direction == "long":
                rationale_points.append("Strong aggressive buying pressure detected")
            else:
                rationale_points.append("Strong aggressive selling pressure detected")

        # Volume/OI rationale
        if component_scores.volume_oi_score > 0.6:
            rationale_points.append("Volume and Open Interest confirming directional bias")

        # Market structure rationale
        price_change = market_data.get('price_change', 0)
        if abs(price_change) > 0.02:  # 2% move
            rationale_points.append(
                f"Strong {'bullish' if price_change > 0 else 'bearish'} momentum "
                f"({abs(price_change)*100:.1f}% move)"
            )

        return rationale_points

    def _calculate_stop_loss(
        self,
        current_price: float,
        direction: str,
        component_scores: SignalComponents,
        market_data: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate structural stop-loss level."""

        try:
            ohlcv_df = market_data.get('ohlcv')
            if ohlcv_df is None or ohlcv_df.empty:
                # Default stop loss (2% from entry)
                buffer = 0.02
                return current_price * (1 - buffer) if direction == "long" else current_price * (1 + buffer)

            # Use ICT detector to find structural stop loss
            structural_stop = self.ict_detector.get_structural_stop_loss(
                ohlcv_df, current_price, direction
            )

            if structural_stop:
                return structural_stop

            # Fallback to recent swing high/low
            if direction == "long":
                recent_low = ohlcv_df['low'].rolling(10).min().iloc[-1]
                return recent_low * 0.998  # Small buffer
            else:
                recent_high = ohlcv_df['high'].rolling(10).max().iloc[-1]
                return recent_high * 1.002  # Small buffer

        except Exception as e:
            logger.error(f"Failed to calculate stop loss: {e}")
            return None

    def _calculate_take_profit(
        self,
        current_price: float,
        direction: str,
        component_scores: SignalComponents,
        market_data: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate take profit level based on risk/reward."""

        try:
            stop_loss = self._calculate_stop_loss(current_price, direction, component_scores, market_data)
            if not stop_loss:
                return None

            # Calculate risk amount
            if direction == "long":
                risk = current_price - stop_loss
                # 2:1 risk/reward ratio
                return current_price + (risk * 2)
            else:
                risk = stop_loss - current_price
                # 2:1 risk/reward ratio
                return current_price - (risk * 2)

        except Exception as e:
            logger.error(f"Failed to calculate take profit: {e}")
            return None

    def _calculate_risk_metrics(
        self,
        current_price: float,
        component_scores: SignalComponents,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate risk metrics for the signal."""

        try:
            # Calculate volatility
            ohlcv_df = market_data.get('ohlcv')
            if ohlcv_df is not None and not ohlcv_df.empty:
                returns = ohlcv_df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24)  # Daily volatility
            else:
                volatility = 0.03  # Default 3% daily volatility

            # Calculate signal strength
            max_component_score = max(
                component_scores.liquidity_score,
                component_scores.ict_score,
                component_scores.killzone_score,
                component_scores.order_flow_score,
                component_scores.volume_oi_score
            )

            return {
                'volatility': volatility,
                'signal_strength': max_component_score,
                'risk_reward_ratio': 2.0,  # Target 2:1
                'max_drawdown_risk': volatility * 2,  # 2x volatility as max risk
                'correlation_risk': 0.5,  # Default correlation risk
                'liquidity_risk': 1.0 - component_scores.liquidity_score,
                'structural_risk': 1.0 - component_scores.ict_score
            }

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {}

    def _calculate_position_size(
        self,
        risk_metrics: Dict[str, Any],
        confidence_level: str
    ) -> float:
        """Calculate recommended position size based on risk and confidence."""

        try:
            # Base position size from config
            base_size = self.config.risk.max_position_size

            # Adjust based on confidence level
            confidence_multiplier = {
                'high': 1.0,
                'medium': 0.7,
                'low': 0.4
            }

            # Adjust based on volatility
            volatility = risk_metrics.get('volatility', 0.03)
            volatility_adjustment = min(1.0, 0.02 / volatility)  # Reduce size for high volatility

            # Calculate final position size
            position_size = (base_size *
                           confidence_multiplier.get(confidence_level, 0.5) *
                           volatility_adjustment)

            return min(position_size, base_size)

        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.01  # Default minimal size

    def _calculate_volume_trend(self, ohlcv_df: pd.DataFrame) -> float:
        """Calculate volume trend."""
        if ohlcv_df.empty or len(ohlcv_df) < 10:
            return 0.0

        recent_volume = ohlcv_df['volume'].tail(5).mean()
        older_volume = ohlcv_df['volume'].head(5).mean()

        return (recent_volume - older_volume) / older_volume if older_volume > 0 else 0.0

    def _calculate_price_change(self, ohlcv_df: pd.DataFrame) -> float:
        """Calculate price change percentage."""
        if ohlcv_df.empty or len(ohlcv_df) < 2:
            return 0.0

        start_price = ohlcv_df['close'].iloc[0]
        end_price = ohlcv_df['close'].iloc[-1]

        return (end_price - start_price) / start_price if start_price > 0 else 0.0

    def _convert_liquidation_data(
        self,
        heatmap_data: pd.DataFrame,
        levels_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert liquidation data to common format."""
        density_clusters = []

        try:
            # Process heatmap data
            if not heatmap_data.empty:
                for _, row in heatmap_data.iterrows():
                    density_clusters.append({
                        'price_level': row['price'],
                        'density_score': row['total_liquidation'],
                        'side': 'long' if row['liquidation_long'] > row['liquidation_short'] else 'short',
                        'timestamp': row.name
                    })

            # Process levels data
            for side, df in levels_data.items():
                if not df.empty:
                    for _, row in df.iterrows():
                        density_clusters.append({
                            'price_level': row['price'],
                            'density_score': row['usd_value'],
                            'side': side,
                            'timestamp': datetime.now()
                        })

        except Exception as e:
            logger.error(f"Failed to convert liquidation data: {e}")

        return density_clusters

    def _create_error_signal(
        self,
        symbol: str,
        current_price: Optional[float],
        error_message: str
    ) -> InstitutionalSignal:
        """Create error signal when analysis fails."""

        return InstitutionalSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            long_confidence=50.0,
            short_confidence=50.0,
            direction="neutral",
            confidence_level="low",
            rationale_points=[f"Analysis failed: {error_message}"],
            risk_metrics={},
            entry_price=current_price or 0.0,
            stop_loss_price=None,
            take_profit_price=None,
            position_size=0.0,
            time_horizon="4h",
            institutional_factors={}
        )