"""
Dynamic Position Sizing Framework

Implements sophisticated position sizing algorithms that adapt to market conditions,
signal confidence, portfolio risk, and volatility metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

from ..utils.data_structures import TradingSignal, SignalDirection, SignalStrength
from ..utils.math_utils import calculate_volatility, calculate_zscore, exponential_moving_average
from ..utils.performance import profile_async_hft_operation, LatencyTracker


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE_AGGRESSIVE = "adaptive_aggressive"


class RiskLevel(Enum):
    """Risk levels for position sizing."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class PositionSize:
    """Position size recommendation."""
    symbol: str
    direction: SignalDirection
    size_quantity: float
    size_value_usd: float
    risk_amount_usd: float
    confidence_adjusted_size: float
    method: SizingMethod
    risk_level: RiskLevel
    max_loss_estimate: float
    expected_return: float
    sharpe_ratio: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        return abs(self.expected_return / self.max_loss_estimate) if self.max_loss_estimate > 0 else 0.0

    @property
    def is_high_quality(self) -> bool:
        """Check if position size recommendation is high quality."""
        return (
            self.confidence_adjusted_size > 0.5 and
            self.risk_reward_ratio >= 2.0 and
            self.sharpe_ratio > 0.5
        )


@dataclass
class SizingConstraints:
    """Position sizing constraints."""
    max_position_size_usd: float = 100000.0
    max_portfolio_allocation: float = 0.20
    min_position_size_usd: float = 1000.0
    max_risk_per_trade: float = 0.02
    max_total_risk: float = 0.10
    max_leverage: float = 3.0
    position_limit_per_symbol: int = 1


@dataclass
class MarketConditions:
    """Market conditions for sizing adjustment."""
    volatility: float
    trend_strength: float
    liquidity_score: float
    spread_bps: float
    volume_ratio: float
    market_regime: str  # trending, ranging, volatile
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DynamicPositionSizer:
    """
    Advanced dynamic position sizing framework.

    Implements multiple sizing methodologies with adaptive adjustment
    based on market conditions, signal quality, and risk constraints.
    """

    def __init__(self, portfolio_value_usd: float, constraints: SizingConstraints = None):
        self.portfolio_value_usd = portfolio_value_usd
        self.constraints = constraints or SizingConstraints()
        self.logger = logging.getLogger(__name__)

        # Data storage
        self.position_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=500)
        self.market_conditions_history: deque = deque(maxlen=200)

        # Sizing parameters
        self.base_methods = [
            SizingMethod.CONFIDENCE_BASED,
            SizingMethod.VOLATILITY_TARGET,
            SizingMethod.KELLY_CRITERION
        ]
        self.default_method = SizingMethod.CONFIDENCE_BASED

        # Risk parameters
        self.volatility_lookback = 20
        self.confidence_threshold = 0.6
        self.min_sharpe_ratio = 0.3
        self.max_volatility_multiplier = 2.0

        # Adaptive parameters
        self.performance_window = 50
        self.adjustment_factor = 0.1
        self.success_rate_target = 0.65

        # Performance tracking
        self.sizing_count = 0
        self.successful_sizing = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0

    @profile_async_hft_operation("position_sizing", LatencyTracker())
    async def calculate_position_size(
        self,
        signal: TradingSignal,
        current_price: float,
        market_conditions: MarketConditions,
        current_positions: Dict[str, float] = None
    ) -> Optional[PositionSize]:
        """
        Calculate optimal position size based on signal and market conditions.

        Args:
            signal: Trading signal with direction and confidence
            current_price: Current price of the asset
            market_conditions: Current market conditions
            current_positions: Current portfolio positions

        Returns:
            Position size recommendation or None if conditions are unfavorable
        """
        try:
            # Validate inputs
            if not self._validate_signal_conditions(signal, market_conditions):
                return None

            # Calculate base position sizes using different methods
            base_sizes = {}
            for method in self.base_methods:
                size = await self._calculate_method_position_size(
                    method, signal, current_price, market_conditions
                )
                if size > 0:
                    base_sizes[method] = size

            if not base_sizes:
                self.logger.warning(f"No valid position sizes calculated for {signal.symbol}")
                return None

            # Apply portfolio constraints
            constrained_sizes = self._apply_portfolio_constraints(
                base_sizes, signal.symbol, current_positions or {}
            )

            # Select optimal size
            optimal_method, optimal_size = self._select_optimal_position_size(
                constrained_sizes, signal, market_conditions
            )

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                optimal_size, signal, current_price, market_conditions
            )

            # Create position size recommendation
            position_size = PositionSize(
                symbol=signal.symbol,
                direction=signal.direction,
                size_quantity=optimal_size / current_price,
                size_value_usd=optimal_size,
                risk_amount_usd=risk_metrics['risk_amount'],
                confidence_adjusted_size=optimal_size * signal.confidence,
                method=optimal_method,
                risk_level=self._determine_risk_level(risk_metrics, signal.confidence),
                max_loss_estimate=risk_metrics['max_loss'],
                expected_return=risk_metrics['expected_return'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                metadata={
                    'signal_strength': signal.strength.value if hasattr(signal, 'strength') else 'unknown',
                    'market_regime': market_conditions.market_regime,
                    'all_methods': list(base_sizes.keys()),
                    'constraint_adjustments': risk_metrics.get('constraint_adjustments', {})
                }
            )

            # Store position size
            self.position_history.append({
                'timestamp': datetime.now(timezone.utc),
                'symbol': signal.symbol,
                'position_size': position_size,
                'signal': signal,
                'market_conditions': market_conditions
            })

            self.sizing_count += 1

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return None

    def _validate_signal_conditions(self, signal: TradingSignal, market_conditions: MarketConditions) -> bool:
        """Validate that signal and market conditions support position sizing."""
        # Check signal confidence
        if signal.confidence < self.confidence_threshold:
            self.logger.debug(f"Signal confidence too low: {signal.confidence:.2f}")
            return False

        # Check market conditions
        if market_conditions.volatility > self.max_volatility_multiplier:
            self.logger.debug(f"Market volatility too high: {market_conditions.volatility:.2f}")
            return False

        # Check liquidity
        if market_conditions.liquidity_score < 0.3:
            self.logger.debug(f"Liquidity score too low: {market_conditions.liquidity_score:.2f}")
            return False

        # Check spread
        if market_conditions.spread_bps > 50:  # More than 50 bps spread
            self.logger.debug(f"Spread too wide: {market_conditions.spread_bps:.1f} bps")
            return False

        return True

    async def _calculate_method_position_size(
        self,
        method: SizingMethod,
        signal: TradingSignal,
        current_price: float,
        market_conditions: MarketConditions
    ) -> float:
        """Calculate position size using specific method."""
        if method == SizingMethod.FIXED_FRACTIONAL:
            return self._fixed_fractional_sizing(signal, market_conditions)
        elif method == SizingMethod.KELLY_CRITERION:
            return self._kelly_criterion_sizing(signal, market_conditions)
        elif method == SizingMethod.VOLATILITY_TARGET:
            return self._volatility_target_sizing(signal, market_conditions, current_price)
        elif method == SizingMethod.RISK_PARITY:
            return self._risk_parity_sizing(signal, market_conditions)
        elif method == SizingMethod.CONFIDENCE_BASED:
            return self._confidence_based_sizing(signal, market_conditions)
        elif method == SizingMethod.ADAPTIVE_AGGRESSIVE:
            return self._adaptive_aggressive_sizing(signal, market_conditions)
        else:
            self.logger.warning(f"Unknown sizing method: {method}")
            return 0.0

    def _fixed_fractional_sizing(self, signal: TradingSignal, market_conditions: MarketConditions) -> float:
        """Fixed fractional position sizing."""
        base_fraction = 0.02  # 2% base allocation
        confidence_multiplier = signal.confidence

        # Adjust for volatility
        volatility_adjustment = 1.0 / (1.0 + market_conditions.volatility)

        position_size = (
            self.portfolio_value_usd *
            base_fraction *
            confidence_multiplier *
            volatility_adjustment
        )

        return min(position_size, self.constraints.max_position_size_usd)

    def _kelly_criterion_sizing(self, signal: TradingSignal, market_conditions: MarketConditions) -> float:
        """Kelly criterion position sizing."""
        # Estimate win probability from confidence
        win_prob = signal.confidence

        # Estimate win/loss ratio based on signal strength
        if hasattr(signal, 'strength'):
            if signal.strength == SignalStrength.STRONG:
                win_loss_ratio = 2.0
            elif signal.strength == SignalStrength.MODERATE:
                win_loss_ratio = 1.5
            else:
                win_loss_ratio = 1.2
        else:
            win_loss_ratio = 1.5

        # Kelly fraction: f = (p * b - q) / b
        # where p = win prob, q = loss prob, b = win/loss ratio
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        # Conservative Kelly (quarter Kelly for safety)
        conservative_kelly = max(0, kelly_fraction) * 0.25

        # Adjust for market conditions
        market_adjustment = 1.0 - (market_conditions.volatility * 0.3)

        position_size = (
            self.portfolio_value_usd *
            conservative_kelly *
            market_adjustment
        )

        return min(position_size, self.constraints.max_position_size_usd)

    def _volatility_target_sizing(
        self,
        signal: TradingSignal,
        market_conditions: MarketConditions,
        current_price: float
    ) -> float:
        """Volatility target position sizing."""
        target_volatility = 0.15  # 15% annual volatility target

        # Scale to daily
        target_daily_vol = target_volatility / np.sqrt(252)

        # Current daily volatility
        current_daily_vol = market_conditions.volatility / np.sqrt(252)

        if current_daily_vol == 0:
            return self.constraints.min_position_size_usd

        # Volatility scaling factor
        vol_scale_factor = target_daily_vol / current_daily_vol

        # Base position size
        base_size = self.portfolio_value_usd * 0.1  # 10% base allocation

        # Adjust for confidence and volatility
        position_size = base_size * vol_scale_factor * signal.confidence

        return min(position_size, self.constraints.max_position_size_usd)

    def _risk_parity_sizing(self, signal: TradingSignal, market_conditions: MarketConditions) -> float:
        """Risk parity position sizing."""
        # Equal risk contribution approach
        risk_budget = 0.1  # 10% risk budget per position

        # Estimate position risk based on volatility
        position_risk = market_conditions.volatility

        # Calculate size for equal risk contribution
        if position_risk > 0:
            target_size = (self.portfolio_value_usd * risk_budget) / position_risk
        else:
            target_size = self.constraints.min_position_size_usd

        # Adjust for confidence
        adjusted_size = target_size * signal.confidence

        return min(adjusted_size, self.constraints.max_position_size_usd)

    def _confidence_based_sizing(self, signal: TradingSignal, market_conditions: MarketConditions) -> float:
        """Confidence-based position sizing."""
        # Base sizing by confidence levels
        if signal.confidence >= 0.9:
            base_size = 0.15  # 15% for very high confidence
        elif signal.confidence >= 0.8:
            base_size = 0.10  # 10% for high confidence
        elif signal.confidence >= 0.7:
            base_size = 0.07  # 7% for moderate-high confidence
        elif signal.confidence >= 0.6:
            base_size = 0.05  # 5% for moderate confidence
        else:
            base_size = 0.02  # 2% for low confidence

        # Adjust for market conditions
        market_multiplier = 1.0

        # Reduce size in high volatility
        if market_conditions.volatility > 1.0:
            market_multiplier *= 0.7

        # Reduce size in illiquid markets
        if market_conditions.liquidity_score < 0.5:
            market_multiplier *= 0.8

        # Increase size in trending markets
        if market_conditions.market_regime == 'trending':
            market_multiplier *= 1.2

        position_size = (
            self.portfolio_value_usd *
            base_size *
            market_multiplier
        )

        return min(position_size, self.constraints.max_position_size_usd)

    def _adaptive_aggressive_sizing(self, signal: TradingSignal, market_conditions: MarketConditions) -> float:
        """Adaptive aggressive position sizing."""
        # Calculate recent performance
        recent_performance = self._calculate_recent_performance()

        # Aggressiveness factor based on recent success
        if recent_performance > 0.7:
            aggressiveness = 1.5  # More aggressive when performing well
        elif recent_performance > 0.5:
            aggressiveness = 1.0  # Normal aggressiveness
        else:
            aggressiveness = 0.7  # Less aggressive when performing poorly

        # Base confidence-based sizing
        base_size = self._confidence_based_sizing(signal, market_conditions)

        # Apply aggressiveness multiplier
        position_size = base_size * aggressiveness

        return min(position_size, self.constraints.max_position_size_usd)

    def _apply_portfolio_constraints(
        self,
        base_sizes: Dict[SizingMethod, float],
        symbol: str,
        current_positions: Dict[str, float]
    ) -> Dict[SizingMethod, float]:
        """Apply portfolio constraints to position sizes."""
        constrained_sizes = {}

        # Calculate current exposure
        current_exposure = sum(abs(position) for position in current_positions.values())
        available_capacity = (
            self.portfolio_value_usd * self.constraints.max_portfolio_allocation - current_exposure
        )

        for method, size in base_sizes.items():
            constrained_size = size

            # Apply max position size constraint
            constrained_size = min(constrained_size, self.constraints.max_position_size_usd)

            # Apply portfolio allocation constraint
            constrained_size = min(constrained_size, available_capacity)

            # Apply minimum size constraint
            constrained_size = max(constrained_size, self.constraints.min_position_size_usd)

            constrained_sizes[method] = constrained_size

        return constrained_sizes

    def _select_optimal_position_size(
        self,
        constrained_sizes: Dict[SizingMethod, float],
        signal: TradingSignal,
        market_conditions: MarketConditions
    ) -> Tuple[SizingMethod, float]:
        """Select optimal position size from available options."""
        if not constrained_sizes:
            return self.default_method, 0.0

        # Score each method
        best_method = self.default_method
        best_score = -1.0
        best_size = 0.0

        for method, size in constrained_sizes.items():
            score = self._score_sizing_method(
                method, size, signal, market_conditions
            )

            if score > best_score:
                best_score = score
                best_method = method
                best_size = size

        return best_method, best_size

    def _score_sizing_method(
        self,
        method: SizingMethod,
        size: float,
        signal: TradingSignal,
        market_conditions: MarketConditions
    ) -> float:
        """Score a sizing method based on current conditions."""
        score = 0.0

        # Base score by method preference
        method_scores = {
            SizingMethod.CONFIDENCE_BASED: 0.9,
            SizingMethod.VOLATILITY_TARGET: 0.8,
            SizingMethod.KELLY_CRITERION: 0.7,
            SizingMethod.RISK_PARITY: 0.6,
            SizingMethod.ADAPTIVE_AGGRESSIVE: 0.5,
            SizingMethod.FIXED_FRACTIONAL: 0.4
        }
        score += method_scores.get(method, 0.5)

        # Adjust for signal strength
        if hasattr(signal, 'strength'):
            if signal.strength == SignalStrength.STRONG:
                score += 0.2
            elif signal.strength == SignalStrength.MODERATE:
                score += 0.1

        # Adjust for market regime
        if market_conditions.market_regime == 'trending':
            if method in [SizingMethod.CONFIDENCE_BASED, SizingMethod.ADAPTIVE_AGGRESSIVE]:
                score += 0.1
        elif market_conditions.market_regime == 'volatile':
            if method in [SizingMethod.VOLATILITY_TARGET, SizingMethod.RISK_PARITY]:
                score += 0.1

        # Adjust for recent performance
        recent_performance = self._calculate_recent_performance()
        if method == SizingMethod.ADAPTIVE_AGGRESSIVE:
            score += recent_performance * 0.2

        return score

    def _calculate_risk_metrics(
        self,
        position_size: float,
        signal: TradingSignal,
        current_price: float,
        market_conditions: MarketConditions
    ) -> Dict[str, float]:
        """Calculate risk metrics for the position."""
        # Risk amount (2% of position size as default)
        risk_amount = position_size * self.constraints.max_risk_per_trade

        # Maximum loss estimate (based on volatility)
        max_loss = position_size * market_conditions.volatility

        # Expected return (based on signal strength and confidence)
        if hasattr(signal, 'strength'):
            if signal.strength == SignalStrength.STRONG:
                return_multiplier = 0.03  # 3% expected return
            elif signal.strength == SignalStrength.MODERATE:
                return_multiplier = 0.02  # 2% expected return
            else:
                return_multiplier = 0.01  # 1% expected return
        else:
            return_multiplier = 0.02

        expected_return = position_size * return_multiplier * signal.confidence

        # Sharpe ratio
        if max_loss > 0:
            sharpe_ratio = expected_return / max_loss
        else:
            sharpe_ratio = 0.0

        return {
            'risk_amount': risk_amount,
            'max_loss': max_loss,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'constraint_adjustments': {}
        }

    def _determine_risk_level(self, risk_metrics: Dict[str, float], confidence: float) -> RiskLevel:
        """Determine risk level based on metrics."""
        # Combine confidence and risk metrics
        risk_score = confidence * 0.6 + min(risk_metrics['sharpe_ratio'], 1.0) * 0.4

        if risk_score >= 0.8:
            return RiskLevel.AGGRESSIVE
        elif risk_score >= 0.6:
            return RiskLevel.MODERATE
        elif risk_score >= 0.4:
            return RiskLevel.CONSERVATIVE
        else:
            return RiskLevel.VERY_AGGRESSIVE  # Low score, high risk approach

    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance score."""
        if len(self.performance_history) < 10:
            return 0.5  # Default to neutral

        recent_performance = list(self.performance_history)[-20:]
        successful_trades = sum(1 for p in recent_performance if p.get('success', False))
        return successful_trades / len(recent_performance)

    def update_performance(self, symbol: str, realized_pnl: float, success: bool):
        """
        Update performance tracking with actual trade results.

        Args:
            symbol: Trading symbol
            realized_pnl: Realized profit/loss in USD
            success: Whether the trade was successful
        """
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'realized_pnl': realized_pnl,
            'success': success
        })

        self.total_pnl += realized_pnl
        if success:
            self.successful_sizing += 1

        # Update drawdown
        if realized_pnl < 0:
            self.max_drawdown = max(self.max_drawdown, abs(realized_pnl))

    def get_sizing_statistics(self) -> Dict[str, any]:
        """Get position sizing statistics."""
        success_rate = (
            self.successful_sizing / self.sizing_count
            if self.sizing_count > 0 else 0.0
        )

        return {
            'total_sizing_decisions': self.sizing_count,
            'successful_sizing': self.successful_sizing,
            'success_rate': success_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'average_position_size': np.mean([
                p['position_size'].size_value_usd
                for p in self.position_history
            ]) if self.position_history else 0.0,
            'current_methods': [m.value for m in self.base_methods],
            'portfolio_value': self.portfolio_value_usd
        }

    def update_constraints(self, new_constraints: SizingConstraints):
        """Update position sizing constraints."""
        self.constraints = new_constraints
        self.logger.info("Position sizing constraints updated")

    def reset(self):
        """Reset position sizer state."""
        self.position_history.clear()
        self.performance_history.clear()
        self.market_conditions_history.clear()
        self.sizing_count = 0
        self.successful_sizing = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0