"""
Risk-Aware Output Formatter for Colin Trading Bot.

This module formats institutional signals with comprehensive risk warnings,
position sizing recommendations, and structural stop-loss levels.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

from ..engine.institutional_scorer import InstitutionalSignal


@dataclass
class RiskMetrics:
    """Enhanced risk metrics with detailed analysis."""
    volatility_warning: str
    liquidity_warning: str
    structural_risk: str
    correlation_risk: str
    recommended_position_size: float
    max_loss_potential: float
    risk_reward_ratio: float
    confidence_adjusted_size: float


@dataclass
class FormattedSignal:
    """Formatted signal with risk awareness."""
    symbol: str
    timestamp: str
    direction: str
    confidence_level: str
    long_confidence: float
    short_confidence: float

    # Entry and exit levels
    entry_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]

    # Risk management
    position_size_percent: float
    risk_metrics: RiskMetrics

    # Analysis details
    rationale_points: List[str]
    institutional_factors: Dict[str, float]
    time_horizon: str

    # Warnings and disclaimers
    risk_warnings: List[str]
    trading_disclaimers: List[str]

    # Market context
    market_phase: str
    liquidity_conditions: str
    volatility_conditions: str


class OutputFormatter:
    """Formats institutional signals with risk awareness."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output formatter.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_config = config.get('output', {})
        self.risk_config = config.get('risk', {})

    def format_signal(self, signal: InstitutionalSignal) -> FormattedSignal:
        """
        Format an institutional signal with risk awareness.

        Args:
            signal: Raw institutional signal

        Returns:
            Formatted signal with risk analysis
        """
        try:
            # Calculate enhanced risk metrics
            risk_metrics = self._calculate_enhanced_risk_metrics(signal)

            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(signal, risk_metrics)

            # Generate trading disclaimers
            trading_disclaimers = self._generate_trading_disclaimers(signal)

            # Determine market context
            market_context = self._determine_market_context(signal)

            # Format the signal
            formatted_signal = FormattedSignal(
                symbol=signal.symbol,
                timestamp=signal.timestamp.isoformat(),
                direction=signal.direction,
                confidence_level=signal.confidence_level,
                long_confidence=round(signal.long_confidence, 1),
                short_confidence=round(signal.short_confidence, 1),
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                position_size_percent=round(signal.position_size * 100, 2),
                risk_metrics=risk_metrics,
                rationale_points=signal.rationale_points,
                institutional_factors=signal.institutional_factors,
                time_horizon=signal.time_horizon,
                risk_warnings=risk_warnings,
                trading_disclaimers=trading_disclaimers,
                market_phase=market_context['phase'],
                liquidity_conditions=market_context['liquidity'],
                volatility_conditions=market_context['volatility']
            )

            return formatted_signal

        except Exception as e:
            logger.error(f"Failed to format signal: {e}")
            raise

    def format_for_output(
        self,
        formatted_signal: FormattedSignal,
        output_format: str = "readable"
    ) -> str:
        """
        Format signal for specific output format.

        Args:
            formatted_signal: Formatted signal
            output_format: Output format ("readable", "json", "csv", "tradingview")

        Returns:
            Formatted string
        """
        try:
            if output_format == "json":
                return self._format_json(formatted_signal)
            elif output_format == "csv":
                return self._format_csv(formatted_signal)
            elif output_format == "tradingview":
                return self._format_tradingview(formatted_signal)
            else:  # readable
                return self._format_readable(formatted_signal)

        except Exception as e:
            logger.error(f"Failed to format output: {e}")
            raise

    def save_to_file(
        self,
        formatted_signal: FormattedSignal,
        file_path: str,
        output_format: str = "json"
    ):
        """
        Save formatted signal to file.

        Args:
            formatted_signal: Formatted signal
            file_path: Output file path
            output_format: Output format
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "csv":
                # Append to CSV or create new
                df = self._signal_to_dataframe(formatted_signal)

                if path.exists():
                    df.to_csv(path, mode='a', header=False, index=False)
                else:
                    df.to_csv(path, index=False)
            else:
                # Save as JSON
                content = self.format_for_output(formatted_signal, output_format)
                with open(path, 'w') as f:
                    f.write(content)

            logger.info(f"Signal saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save signal to file: {e}")
            raise

    def _calculate_enhanced_risk_metrics(self, signal: InstitutionalSignal) -> RiskMetrics:
        """Calculate enhanced risk metrics for the signal."""

        try:
            base_risk = signal.risk_metrics
            volatility = base_risk.get('volatility', 0.03)
            signal_strength = base_risk.get('signal_strength', 0.5)

            # Volatility warning
            if volatility > 0.05:  # 5% daily volatility
                volatility_warning = "HIGH - Market experiencing extreme volatility"
            elif volatility > 0.03:  # 3% daily volatility
                volatility_warning = "MODERATE - Above average volatility"
            else:
                volatility_warning = "NORMAL - Market volatility within acceptable range"

            # Liquidity warning
            liquidity_risk = base_risk.get('liquidity_risk', 0.5)
            if liquidity_risk > 0.7:
                liquidity_warning = "HIGH - Low liquidity conditions detected"
            elif liquidity_risk > 0.4:
                liquidity_warning = "MODERATE - Liquidity conditions below optimal"
            else:
                liquidity_warning = "GOOD - Adequate liquidity for position size"

            # Structural risk
            structural_risk = base_risk.get('structural_risk', 0.5)
            if structural_risk > 0.6:
                structural_warning = "HIGH - Limited structural support nearby"
            elif structural_risk > 0.3:
                structural_warning = "MODERATE - Some structural levels present"
            else:
                structural_warning = "LOW - Strong structural confluence"

            # Correlation risk (simplified)
            correlation_risk = "MODERATE - Crypto assets show high correlation"
            if signal.symbol.startswith('BTC'):
                correlation_risk = "HIGH - Bitcoin typically leads market movements"

            # Calculate position sizes
            base_position = signal.position_size
            confidence_adjustment = {
                'high': 1.0,
                'medium': 0.7,
                'low': 0.4
            }.get(signal.confidence_level, 0.5)

            confidence_adjusted_size = base_position * confidence_adjustment

            # Calculate maximum loss
            if signal.stop_loss_price:
                if signal.direction == "long":
                    max_loss_pct = (signal.entry_price - signal.stop_loss_price) / signal.entry_price
                else:
                    max_loss_pct = (signal.stop_loss_price - signal.entry_price) / signal.entry_price
                max_loss_potential = max_loss_pct * confidence_adjusted_size
            else:
                max_loss_potential = base_position * 0.02  # Default 2% risk

            return RiskMetrics(
                volatility_warning=volatility_warning,
                liquidity_warning=liquidity_warning,
                structural_risk=structural_warning,
                correlation_risk=correlation_risk,
                recommended_position_size=round(base_position * 100, 2),
                max_loss_potential=round(max_loss_potential * 100, 2),
                risk_reward_ratio=base_risk.get('risk_reward_ratio', 2.0),
                confidence_adjusted_size=round(confidence_adjusted_size * 100, 2)
            )

        except Exception as e:
            logger.error(f"Failed to calculate enhanced risk metrics: {e}")
            # Return safe defaults
            return RiskMetrics(
                volatility_warning="UNKNOWN",
                liquidity_warning="UNKNOWN",
                structural_risk="UNKNOWN",
                correlation_risk="UNKNOWN",
                recommended_position_size=1.0,
                max_loss_potential=2.0,
                risk_reward_ratio=2.0,
                confidence_adjusted_size=1.0
            )

    def _generate_risk_warnings(
        self,
        signal: InstitutionalSignal,
        risk_metrics: RiskMetrics
    ) -> List[str]:
        """Generate risk warnings for the signal."""

        warnings = []

        # Volatility warnings
        if "HIGH" in risk_metrics.volatility_warning:
            warnings.append("⚠️ HIGH VOLATILITY: Market conditions are extremely volatile - consider reducing position size")

        # Liquidity warnings
        if "HIGH" in risk_metrics.liquidity_warning:
            warnings.append("⚠️ LOW LIQUIDITY: Market conditions may result in slippage - use limit orders")

        # Confidence warnings
        if signal.confidence_level == "low":
            warnings.append("⚠️ LOW CONFIDENCE: Signal has weak confirmation - consider waiting for better setup")
        elif signal.confidence_level == "medium":
            warnings.append("⚠️ MEDIUM CONFIDENCE: Signal has moderate confirmation - use caution")

        # Directional warnings
        if signal.direction == "neutral":
            warnings.append("⚠️ NEUTRAL SIGNAL: No clear directional bias - avoid trading this signal")

        # Stop loss warnings
        if not signal.stop_loss_price:
            warnings.append("⚠️ NO STOP LOSS: Unable to determine structural stop loss - manual risk management required")

        # Position size warnings
        if risk_metrics.max_loss_potential > 5.0:
            warnings.append("⚠️ HIGH RISK: Recommended position size exceeds 5% max loss - consider reducing size")

        # Time horizon warnings
        if signal.time_horizon == "1h":
            warnings.append("⚠️ SHORT TIME HORIZON: Signal is for short-term trading - requires active monitoring")

        return warnings

    def _generate_trading_disclaimers(self, signal: InstitutionalSignal) -> List[str]:
        """Generate trading disclaimers."""

        disclaimers = [
            "📊 This signal is for informational purposes only and does not constitute financial advice",
            "⚖️ Always conduct your own research before making trading decisions",
            "💰 Never risk more than you can afford to lose",
            "📈 Past performance does not guarantee future results",
            "🔄 Market conditions can change rapidly - monitor positions actively"
        ]

        # Add specific disclaimers based on conditions
        if signal.confidence_level == "high":
            disclaimers.append("⚡ High confidence signals still carry risk and can fail")

        if signal.time_horizon == "24h":
            disclaimers.append("🌍 Long-term signals require wider stop losses and patience")

        # Add institutional trading specific disclaimers
        disclaimers.extend([
            "🏦 Institutional signals are based on smart money behavior patterns",
            "🔄 Counter-trend signals carry higher risk than trend-following signals",
            "📊 Liquidity analysis is probabilistic - clusters may not trigger as expected"
        ])

        return disclaimers

    def _determine_market_context(self, signal: InstitutionalSignal) -> Dict[str, str]:
        """Determine market context from signal data."""

        # Get institutional factors
        factors = signal.institutional_factors

        # Determine market phase
        killzone_score = factors.get('killzone', 0)
        if killzone_score > 0.8:
            phase = "Peak Institutional Hours"
        elif killzone_score > 0.5:
            phase = "Institutional Session"
        else:
            phase = "After Hours/Low Activity"

        # Determine liquidity conditions
        liquidity_score = factors.get('liquidity', 0)
        if liquidity_score > 0.7:
            liquidity = "High Liquidity"
        elif liquidity_score > 0.4:
            liquidity = "Moderate Liquidity"
        else:
            liquidity = "Low Liquidity"

        # Determine volatility conditions
        order_flow_score = factors.get('order_flow', 0)
        if order_flow_score > 0.7:
            volatility = "High Volatility Expected"
        elif order_flow_score > 0.4:
            volatility = "Moderate Volatility Expected"
        else:
            volatility = "Low Volatility Expected"

        return {
            'phase': phase,
            'liquidity': liquidity,
            'volatility': volatility
        }

    def _format_readable(self, signal: FormattedSignal) -> str:
        """Format signal in human-readable format."""

        output = []
        output.append("\n" + "="*70)
        output.append(f"🏦 INSTITUTIONAL TRADING SIGNAL: {signal.symbol}")
        output.append("="*70)

        # Signal summary
        direction_emoji = {"long": "🟢", "short": "🔴", "neutral": "🟡"}.get(signal.direction, "⚪")
        confidence_emoji = {"high": "🔥", "medium": "⚡", "low": "💧"}.get(signal.confidence_level, "❓")

        output.append(f"\n{direction_emoji} DIRECTION: {signal.direction.upper()}")
        output.append(f"{confidence_emoji} CONFIDENCE: {signal.confidence_level.upper()}")
        output.append(f"📊 Long: {signal.long_confidence}% | Short: {signal.short_confidence}%")

        # Entry and risk management
        output.append(f"\n💰 ENTRY: ${signal.entry_price:.2f}")
        if signal.stop_loss_price:
            output.append(f"🛑 STOP LOSS: ${signal.stop_loss_price:.2f}")
        if signal.take_profit_price:
            output.append(f"🎯 TAKE PROFIT: ${signal.take_profit_price:.2f}")
        output.append(f"📏 POSITION SIZE: {signal.position_size_percent}% of portfolio")

        # Risk warnings
        if signal.risk_warnings:
            output.append(f"\n⚠️  RISK WARNINGS:")
            for warning in signal.risk_warnings:
                output.append(f"   {warning}")

        # Risk metrics
        output.append(f"\n📋 RISK ANALYSIS:")
        output.append(f"   Volatility: {signal.risk_metrics.volatility_warning}")
        output.append(f"   Liquidity: {signal.risk_metrics.liquidity_warning}")
        output.append(f"   Structural Risk: {signal.risk_metrics.structural_risk}")
        output.append(f"   Max Loss Potential: {signal.risk_metrics.max_loss_potential}%")
        output.append(f"   Risk/Reward Ratio: 1:{signal.risk_metrics.risk_reward_ratio}")

        # Market context
        output.append(f"\n🌍 MARKET CONTEXT:")
        output.append(f"   Phase: {signal.market_phase}")
        output.append(f"   Liquidity: {signal.liquidity_conditions}")
        output.append(f"   Volatility: {signal.volatility_conditions}")

        # Rationale
        output.append(f"\n📋 INSTITUTIONAL RATIONALE:")
        for i, point in enumerate(signal.rationale_points, 1):
            output.append(f"   {i}. {point}")

        # Factor breakdown
        output.append(f"\n🏦 FACTOR BREAKDOWN:")
        factors = signal.institutional_factors
        output.append(f"   Liquidity Analysis: {factors.get('liquidity', 0):.3f}")
        output.append(f"   ICT Structure: {factors.get('ict', 0):.3f}")
        output.append(f"   Killzone Timing: {factors.get('killzone', 0):.3f}")
        output.append(f"   Order Flow: {factors.get('order_flow', 0):.3f}")
        output.append(f"   Volume/OI: {factors.get('volume_oi', 0):.3f}")

        # Disclaimers
        output.append(f"\n📜 IMPORTANT DISCLAIMERS:")
        for disclaimer in signal.trading_disclaimers[:3]:  # Show first 3
            output.append(f"   {disclaimer}")

        output.append(f"\n⏰ Generated: {signal.timestamp}")
        output.append("="*70)

        return "\n".join(output)

    def _format_json(self, signal: FormattedSignal) -> str:
        """Format signal as JSON."""
        return json.dumps(asdict(signal), indent=2, default=str)

    def _format_csv(self, signal: FormattedSignal) -> str:
        """Format signal as CSV."""
        csv_data = {
            'symbol': signal.symbol,
            'timestamp': signal.timestamp,
            'direction': signal.direction,
            'confidence_level': signal.confidence_level,
            'long_confidence': signal.long_confidence,
            'short_confidence': signal.short_confidence,
            'entry_price': signal.entry_price,
            'stop_loss_price': signal.stop_loss_price,
            'take_profit_price': signal.take_profit_price,
            'position_size_percent': signal.position_size_percent,
            'volatility_warning': signal.risk_metrics.volatility_warning,
            'liquidity_warning': signal.risk_metrics.liquidity_warning,
            'max_loss_potential': signal.risk_metrics.max_loss_potential,
            'risk_reward_ratio': signal.risk_metrics.risk_reward_ratio,
            'market_phase': signal.market_phase,
            'rationale': '; '.join(signal.rationale_points)
        }

        # Create CSV row
        return ','.join(f'"{str(v)}"' for v in csv_data.values())

    def _format_tradingview(self, signal: FormattedSignal) -> str:
        """Format signal for TradingView Pine Script format."""

        # Create TradingView alert message
        direction_emoji = {"long": "🟢", "short": "🔴", "neutral": "🟡"}.get(signal.direction, "⚪")

        alert_message = f"""
Colin Bot Signal - {signal.symbol}
{direction_emoji} {signal.direction.upper()} | {signal.confidence_level.upper()} Confidence
Entry: ${signal.entry_price:.2f}
Stop Loss: ${signal.stop_loss_price:.2f if signal.stop_loss_price else 'N/A'}
Take Profit: ${signal.take_profit_price:.2f if signal.take_profit_price else 'N/A'}
Position Size: {signal.position_size_percent}%

Rationale: {' | '.join(signal.rationale_points[:2])}
Time Horizon: {signal.time_horizon}
        """.strip()

        return alert_message

    def _signal_to_dataframe(self, signal: FormattedSignal) -> pd.DataFrame:
        """Convert signal to pandas DataFrame for CSV export."""
        data = {
            'symbol': [signal.symbol],
            'timestamp': [signal.timestamp],
            'direction': [signal.direction],
            'confidence_level': [signal.confidence_level],
            'long_confidence': [signal.long_confidence],
            'short_confidence': [signal.short_confidence],
            'entry_price': [signal.entry_price],
            'stop_loss_price': [signal.stop_loss_price],
            'take_profit_price': [signal.take_profit_price],
            'position_size_percent': [signal.position_size_percent],
            'volatility_warning': [signal.risk_metrics.volatility_warning],
            'liquidity_warning': [signal.risk_metrics.liquidity_warning],
            'max_loss_potential': [signal.risk_metrics.max_loss_potential],
            'risk_reward_ratio': [signal.risk_metrics.risk_reward_ratio],
            'market_phase': [signal.market_phase],
            'rationale': ['; '.join(signal.rationale_points)]
        }

        return pd.DataFrame(data)