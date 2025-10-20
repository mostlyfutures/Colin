"""
HFT Signal Bridge for Colin Trading Bot

This module bridges HFT signals with the main bot's existing signal infrastructure,
ensuring seamless integration and enhanced signal quality.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import numpy as np

from .hft_integration_adapter import HFTIntegrationAdapter, HFTSignal


class HFTSignalBridge:
    """
    Bridge that converts HFT signals to the main bot's signal format
    and enhances existing signals with HFT insights.
    """

    def __init__(self, hft_adapter: HFTIntegrationAdapter):
        """
        Initialize HFT Signal Bridge.

        Args:
            hft_adapter: HFT integration adapter instance
        """
        self.hft_adapter = hft_adapter
        self.signal_weights = {
            'hft_ofi': 0.3,      # Weight for Order Flow Imbalance
            'hft_skew': 0.25,    # Weight for Book Skew
            'hft_fusion': 0.45   # Weight for fused signal
        }

        # Signal enhancement factors
        self.confidence_boost_factor = 1.15  # 15% boost for strong HFT signals
        self.accuracy_improvement = {
            'strong': 0.10,    # 10% improvement for strong signals
            'moderate': 0.05,  # 5% improvement for moderate signals
            'weak': 0.02       # 2% improvement for weak signals
        }

        logger.info("HFT Signal Bridge initialized")

    def is_available(self) -> bool:
        """Check if HFT bridge is available."""
        return self.hft_adapter.is_hft_enabled()

    async def enhance_institutional_signal(self, symbol: str, institutional_signal: Dict) -> Dict[str, Any]:
        """
        Enhance existing institutional signal with HFT insights.

        Args:
            symbol: Trading symbol
            institutional_signal: Existing signal from main bot

        Returns:
            Enhanced signal with HFT components
        """
        if not self.is_available():
            logger.debug(f"HFT not available for {symbol} - returning original signal")
            return institutional_signal

        try:
            # Generate HFT signal
            hft_signal = await self.hft_adapter.generate_hft_signal(symbol)

            if not hft_signal:
                logger.debug(f"No HFT signal generated for {symbol}")
                return institutional_signal

            # Create enhanced signal
            enhanced_signal = self._merge_signals(institutional_signal, hft_signal)

            logger.debug(f"Enhanced signal for {symbol}: "
                        f"Original confidence: {institutional_signal.get('confidence_level', 'N/A')} -> "
                        f"Enhanced: {enhanced_signal.get('confidence_level', 'N/A')}")

            return enhanced_signal

        except Exception as e:
            logger.error(f"Error enhancing signal for {symbol}: {e}")
            return institutional_signal

    def _merge_signals(self, institutional_signal: Dict, hft_signal: HFTSignal) -> Dict[str, Any]:
        """
        Merge institutional signal with HFT signal.

        Args:
            institutional_signal: Signal from main bot
            hft_signal: Signal from HFT engine

        Returns:
            Enhanced merged signal
        """
        enhanced = institutional_signal.copy()

        # Add HFT-specific information
        enhanced['hft_components'] = {
            'ofi_signal': hft_signal.ofi_signal,
            'book_skew': hft_signal.book_skew,
            'hft_confidence': hft_signal.confidence,
            'hft_strength': hft_signal.strength,
            'hft_direction': hft_signal.direction,
            'hft_rationale': hft_signal.rationale,
            'raw_hft_data': hft_signal.raw_hft_data
        }

        # Enhance confidence based on HFT signal
        enhanced_confidence = self._calculate_enhanced_confidence(institutional_signal, hft_signal)
        enhanced['confidence_level'] = enhanced_confidence['level']
        enhanced['enhanced_long_confidence'] = enhanced_confidence['long']
        enhanced['enhanced_short_confidence'] = enhanced_confidence['short']

        # Update institutional factors with HFT components
        if 'institutional_factors' not in enhanced:
            enhanced['institutional_factors'] = {}

        enhanced['institutional_factors'].update({
            'hft_ofi_strength': abs(hft_signal.ofi_signal) * 100,
            'hft_skew_strength': abs(hft_signal.book_skew) * 100,
            'hft_fusion_confidence': hft_signal.confidence,
            'hft_signal_alignment': self._calculate_signal_alignment(institutional_signal, hft_signal)
        })

        # Add enhanced rationale
        enhanced_rationale = enhanced.get('rationale_points', []).copy()
        enhanced_rationale.extend([
            f"HFT OFI Signal: {hft_signal.ofi_signal:.3f}",
            f"HFT Book Skew: {hft_signal.book_skew:.3f}",
            f"HFT Consensus: {hft_signal.direction} ({hft_signal.confidence:.1f}%)"
        ])
        enhanced['rationale_points'] = enhanced_rationale

        # Add HFT risk metrics
        if 'risk_metrics' not in enhanced:
            enhanced['risk_metrics'] = {}

        enhanced['risk_metrics'].update({
            'hft_volatility_indicator': self._calculate_hft_volatility(hft_signal),
            'hft_liquidity_score': self._calculate_liquidity_score(hft_signal),
            'hft_market_pressure': self._calculate_market_pressure(hft_signal)
        })

        # Add timestamp for enhancement
        enhanced['hft_enhanced_at'] = datetime.now(timezone.utc).isoformat()

        return enhanced

    def _calculate_enhanced_confidence(self, institutional_signal: Dict, hft_signal: HFTSignal) -> Dict[str, Any]:
        """
        Calculate enhanced confidence using both institutional and HFT signals.

        Args:
            institutional_signal: Main bot signal
            hft_signal: HFT engine signal

        Returns:
            Enhanced confidence breakdown
        """
        # Extract institutional confidences
        inst_long = institutional_signal.get('long_confidence', 50)
        inst_short = institutional_signal.get('short_confidence', 50)

        # Convert HFT signal to confidence values
        hft_long = hft_signal.confidence if hft_signal.direction == 'long' else 0
        hft_short = hft_signal.confidence if hft_signal.direction == 'short' else 0

        # Weight the signals
        weight_hft = 0.4  # 40% weight to HFT
        weight_inst = 0.6  # 60% weight to institutional

        # Calculate weighted confidences
        enhanced_long = (inst_long * weight_inst) + (hft_long * weight_hft)
        enhanced_short = (inst_short * weight_inst) + (hft_short * weight_hft)

        # Apply confidence boost for strong HFT signals
        if hft_signal.strength == 'strong' and hft_signal.confidence > 80:
            boost = self.confidence_boost_factor
            if hft_signal.direction == 'long':
                enhanced_long *= boost
            elif hft_signal.direction == 'short':
                enhanced_short *= boost

        # Ensure values stay within bounds
        enhanced_long = min(max(enhanced_long, 0), 100)
        enhanced_short = min(max(enhanced_short, 0), 100)

        # Determine confidence level
        max_confidence = max(enhanced_long, enhanced_short)
        if max_confidence >= 75:
            level = 'high'
        elif max_confidence >= 60:
            level = 'medium'
        else:
            level = 'low'

        return {
            'long': round(enhanced_long, 1),
            'short': round(enhanced_short, 1),
            'level': level
        }

    def _calculate_signal_alignment(self, institutional_signal: Dict, hft_signal: HFTSignal) -> float:
        """
        Calculate alignment between institutional and HFT signals.

        Args:
            institutional_signal: Main bot signal
            hft_signal: HFT engine signal

        Returns:
            Alignment score (0-1, where 1 is perfect alignment)
        """
        inst_direction = institutional_signal.get('direction', 'neutral')
        hft_direction = hft_signal.direction

        if inst_direction == hft_direction:
            # Perfect alignment
            return 1.0
        elif inst_direction == 'neutral' or hft_direction == 'neutral':
            # Partial alignment
            return 0.5
        else:
            # No alignment (opposite directions)
            return 0.0

    def _calculate_hft_volatility(self, hft_signal: HFTSignal) -> float:
        """Calculate volatility indicator from HFT signal."""
        # Use signal strength and OFI magnitude as volatility proxy
        ofi_magnitude = abs(hft_signal.ofi_signal)
        skew_magnitude = abs(hft_signal.book_skew)

        # Combine factors
        volatility = (ofi_magnitude + skew_magnitude) / 2
        return min(volatility * 100, 100)  # Convert to 0-100 scale

    def _calculate_liquidity_score(self, hft_signal: HFTSignal) -> float:
        """Calculate liquidity score from HFT signal."""
        # Use order book depth from raw HFT data
        raw_data = hft_signal.raw_hft_data
        depth = raw_data.get('order_book_depth', 0)

        # Normalize to 0-100 scale (assuming max depth of 20)
        liquidity_score = min(depth / 20 * 100, 100)
        return liquidity_score

    def _calculate_market_pressure(self, hft_signal: HFTSignal) -> str:
        """Calculate market pressure from HFT signal."""
        if hft_signal.strength == 'strong' and hft_signal.confidence > 85:
            return 'high'
        elif hft_signal.strength in ['strong', 'moderate'] and hft_signal.confidence > 70:
            return 'medium'
        else:
            return 'low'

    async def generate_pure_hft_signal(self, symbol: str, time_horizon: str = "1h") -> Optional[Dict[str, Any]]:
        """
        Generate a pure HFT signal in the main bot's format.

        Args:
            symbol: Trading symbol
            time_horizon: Analysis time horizon

        Returns:
            Signal in main bot format or None
        """
        if not self.is_available():
            return None

        try:
            hft_signal = await self.hft_adapter.generate_hft_signal(symbol, time_horizon)

            if not hft_signal:
                return None

            # Convert to main bot signal format
            main_bot_signal = {
                'symbol': hft_signal.symbol,
                'timestamp': hft_signal.timestamp.isoformat(),
                'direction': hft_signal.direction,
                'confidence_level': self._confidence_to_level(hft_signal.confidence),
                'long_confidence': hft_signal.confidence if hft_signal.direction == 'long' else (100 - hft_signal.confidence),
                'short_confidence': hft_signal.confidence if hft_signal.direction == 'short' else (100 - hft_signal.confidence),
                'entry_price': hft_signal.raw_hft_data.get('best_bid') if hft_signal.direction == 'long' else hft_signal.raw_hft_data.get('best_ask'),
                'stop_loss_price': None,  # HFT signals typically don't include stops
                'take_profit_price': None,  # HFT signals typically don't include targets
                'position_size': 0.1,  # Default small size for HFT
                'time_horizon': time_horizon,
                'rationale_points': hft_signal.rationale,
                'risk_metrics': {
                    'hft_volatility': self._calculate_hft_volatility(hft_signal),
                    'hft_liquidity': self._calculate_liquidity_score(hft_signal),
                    'hft_pressure': self._calculate_market_pressure(hft_signal)
                },
                'institutional_factors': {
                    'hft_ofi': hft_signal.ofi_signal,
                    'hft_skew': hft_signal.book_skew,
                    'hft_fusion_confidence': hft_signal.confidence,
                    'hft_strength_multiplier': self._strength_to_multiplier(hft_signal.strength)
                },
                'hft_components': {
                    'ofi_signal': hft_signal.ofi_signal,
                    'book_skew': hft_signal.book_skew,
                    'hft_confidence': hft_signal.confidence,
                    'hft_strength': hft_signal.strength,
                    'hft_direction': hft_signal.direction,
                    'raw_hft_data': hft_signal.raw_hft_data
                },
                'signal_source': 'hft_pure',
                'generated_at': datetime.now(timezone.utc).isoformat()
            }

            return main_bot_signal

        except Exception as e:
            logger.error(f"Error generating pure HFT signal for {symbol}: {e}")
            return None

    def _confidence_to_level(self, confidence: float) -> str:
        """Convert confidence percentage to level."""
        if confidence >= 75:
            return 'high'
        elif confidence >= 60:
            return 'medium'
        else:
            return 'low'

    def _strength_to_multiplier(self, strength: str) -> float:
        """Convert strength to multiplier."""
        multipliers = {
            'strong': 1.2,
            'moderate': 1.0,
            'weak': 0.8
        }
        return multipliers.get(strength, 1.0)

    async def batch_enhance_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance multiple signals with HFT insights.

        Args:
            signals: List of signals to enhance

        Returns:
            List of enhanced signals
        """
        if not self.is_available():
            logger.debug("HFT not available - returning original signals")
            return signals

        enhanced_signals = []

        # Process signals concurrently for efficiency
        tasks = [
            self.enhance_institutional_signal(signal['symbol'], signal)
            for signal in signals
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error enhancing signal {i}: {result}")
                    enhanced_signals.append(signals[i])  # Use original signal
                else:
                    enhanced_signals.append(result)

        except Exception as e:
            logger.error(f"Error in batch enhancement: {e}")
            return signals  # Return original signals on error

        return enhanced_signals

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current status of the HFT signal bridge."""
        return {
            'bridge_available': self.is_available(),
            'hft_adapter_status': self.hft_adapter.get_performance_metrics(),
            'signal_weights': self.signal_weights,
            'confidence_boost_factor': self.confidence_boost_factor,
            'accuracy_improvements': self.accuracy_improvement
        }