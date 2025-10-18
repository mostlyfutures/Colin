"""
ICT (Institutional Candlestick Theory) Scorer.

This module scores ICT-based signals including Fair Value Gaps, Order Blocks,
and Break of Structure confluence for institutional signal analysis.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from ..core.config import ConfigManager
from ..structure.ict_detector import (
    ICTDetector, FairValueGap, OrderBlock, BreakOfStructure
)


@dataclass
class ICTSignal:
    """Represents an ICT-based signal."""
    signal_type: str
    strength: float
    direction: str  # "long" or "short"
    confidence: float
    rationale: str
    entry_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    timeframe: str = "1h"


@dataclass
class ICTScore:
    """Comprehensive ICT scoring result."""
    overall_score: float
    long_bias: float
    short_bias: float
    signals: List[ICTSignal]
    confluence_analysis: Dict[str, Any]
    structural_levels: Dict[str, Any]
    timestamp: datetime


class ICTScorer:
    """Scores ICT-based institutional signals."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize ICT scorer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.ict_config = self.config.ict
        self.detector = ICTDetector(config_manager)

    def score_ict_confluence(
        self,
        df,
        current_price: float,
        timeframe: str = "1h"
    ) -> ICTScore:
        """
        Score ICT structure confluence around current price.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price level
            timeframe: Timeframe for analysis

        Returns:
            Comprehensive ICT score
        """
        try:
            signals = []
            long_bias = 0.0
            short_bias = 0.0

            # Detect ICT structures
            fvg_list = self.detector.detect_fair_value_gaps(df)
            ob_list = self.detector.detect_order_blocks(df)
            bos_list = self.detector.detect_break_of_structure(df)

            # Analyze Fair Value Gaps
            fvg_signals = self._analyze_fvg_signals(fvg_list, current_price)
            signals.extend(fvg_signals)

            # Analyze Order Blocks
            ob_signals = self._analyze_order_block_signals(ob_list, current_price)
            signals.extend(ob_signals)

            # Analyze Break of Structure
            bos_signals = self._analyze_bos_signals(bos_list, current_price, df)
            signals.extend(bos_signals)

            # Analyze confluence zones
            confluence_analysis = self._analyze_structure_confluence(
                fvg_list, ob_list, bos_list, current_price
            )

            # Calculate bias scores
            for signal in signals:
                if signal.direction == "long":
                    long_bias += signal.strength * signal.confidence
                else:
                    short_bias += signal.strength * signal.confidence

            # Calculate overall score
            overall_score = self._calculate_overall_ict_score(signals, confluence_analysis)

            # Normalize bias scores
            total_bias = long_bias + short_bias
            if total_bias > 0:
                long_bias = long_bias / total_bias
                short_bias = short_bias / total_bias

            # Identify structural levels
            structural_levels = self._identify_structural_levels(
                fvg_list, ob_list, bos_list, current_price
            )

            result = ICTScore(
                overall_score=min(overall_score, 1.0),
                long_bias=long_bias,
                short_bias=short_bias,
                signals=signals,
                confluence_analysis=confluence_analysis,
                structural_levels=structural_levels,
                timestamp=datetime.now()
            )

            logger.debug(f"ICT scoring complete: Overall={overall_score:.3f}, Long={long_bias:.3f}, Short={short_bias:.3f}")
            return result

        except Exception as e:
            logger.error(f"Failed to score ICT confluence: {e}")
            return self._empty_ict_score()

    def score_fvg_mitigation_potential(
        self,
        fvg_list: List[FairValueGap],
        current_price: float,
        order_flow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score potential for FVG mitigation scenarios.

        Args:
            fvg_list: List of Fair Value Gaps
            current_price: Current price level
            order_flow_data: Order flow analysis data

        Returns:
            FVG mitigation analysis
        """
        try:
            mitigation_analysis = {
                'unfilled_fvgs': [],
                'partial_fill_fvgs': [],
                'filled_fvgs': [],
                'mitigation_probability': 0.0,
                'rationale': []
            }

            for fvg in fvg_list:
                # Determine fill status
                if current_price >= fvg.top and current_price <= fvg.bottom:
                    status = "in_progress"
                elif current_price > fvg.bottom:
                    status = "filled"
                elif current_price < fvg.top:
                    status = "unfilled"
                else:
                    status = "unfilled"

                # Calculate mitigation probability
                distance_to_fvg = min(
                    abs(current_price - fvg.top),
                    abs(current_price - fvg.bottom)
                ) / current_price

                # Order flow alignment
                order_flow_bias = order_flow_data.get('normalized_order_book_imbalance', 0)

                # FVG mitigation logic
                if status == "unfilled":
                    if distance_to_fvg < 0.01:  # Within 1%
                        mitigation_prob = fvg.confidence * 0.8
                        if (fvg.top > fvg.bottom and order_flow_bias < -0.1) or \
                           (fvg.top < fvg.bottom and order_flow_bias > 0.1):
                            mitigation_prob *= 1.2  # Boost if order flow aligns

                        mitigation_analysis['unfilled_fvgs'].append({
                            'fvg': fvg,
                            'mitigation_probability': min(mitigation_prob, 1.0),
                            'distance': distance_to_fvg,
                            'order_flow_alignment': order_flow_bias
                        })

                        mitigation_analysis['rationale'].append(
                            f"Unfilled FVG at ${fvg.midline:.2f} with {mitigation_prob:.1%} mitigation probability"
                        )

            # Calculate overall mitigation probability
            if mitigation_analysis['unfilled_fvgs']:
                mitigation_analysis['mitigation_probability'] = max(
                    fvg['mitigation_probability']
                    for fvg in mitigation_analysis['unfilled_fvgs']
                )

            return mitigation_analysis

        except Exception as e:
            logger.error(f"Failed to score FVG mitigation potential: {e}")
            return {
                'unfilled_fvgs': [],
                'partial_fill_fvgs': [],
                'filled_fvgs': [],
                'mitigation_probability': 0.0,
                'rationale': []
            }

    def _analyze_fvg_signals(
        self,
        fvg_list: List[FairValueGap],
        current_price: float
    ) -> List[ICTSignal]:
        """Analyze Fair Value Gaps for trading signals."""

        signals = []

        for fvg in fvg_list:
            # Check if price is near FVG
            distance_to_top = abs(current_price - fvg.top) / current_price
            distance_to_bottom = abs(current_price - fvg.bottom) / current_price
            min_distance = min(distance_to_top, distance_to_bottom)

            if min_distance < 0.005:  # Within 0.5% of FVG
                # Determine signal direction based on FVG type and price position
                if fvg.top > fvg.bottom:  # Bullish FVG (gap up)
                    if current_price < fvg.top:  # Price below FVG (potential fill)
                        direction = "long"
                        signal_type = "bullish_fvg_fill"
                        rationale = f"Price approaching bullish FVG ${fvg.midline:.2f} for fill"
                        entry_price = current_price
                        stop_loss = fvg.top * 0.998  # Just above FVG top
                    else:  # Price above FVG (support)
                        direction = "long"
                        signal_type = "bullish_fvg_support"
                        rationale = f"Bullish FVG ${fvg.midline:.2f} acting as support"
                        entry_price = fvg.bottom
                        stop_loss = fvg.top * 0.998

                else:  # Bearish FVG (gap down)
                    if current_price > fvg.bottom:  # Price above FVG (potential fill)
                        direction = "short"
                        signal_type = "bearish_fvg_fill"
                        rationale = f"Price approaching bearish FVG ${fvg.midline:.2f} for fill"
                        entry_price = current_price
                        stop_loss = fvg.bottom * 1.002  # Just below FVG bottom
                    else:  # Price below FVG (resistance)
                        direction = "short"
                        signal_type = "bearish_fvg_resistance"
                        rationale = f"Bearish FVG ${fvg.midline:.2f} acting as resistance"
                        entry_price = fvg.top
                        stop_loss = fvg.bottom * 1.002

                # Calculate signal strength
                proximity_score = 1.0 - (min_distance / 0.005)
                strength = fvg.confidence * proximity_score

                signals.append(ICTSignal(
                    signal_type=signal_type,
                    strength=min(strength, 1.0),
                    direction=direction,
                    confidence=fvg.confidence,
                    rationale=rationale,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    timeframe=fvg.timeframe
                ))

        return signals

    def _analyze_order_block_signals(
        self,
        ob_list: List[OrderBlock],
        current_price: float
    ) -> List[ICTSignal]:
        """Analyze Order Blocks for trading signals."""

        signals = []

        for ob in ob_list:
            # Check if price is near Order Block
            if ob.candle_low <= current_price <= ob.candle_high:
                # Determine signal direction based on Order Block type
                if ob.side == "bullish":
                    direction = "long"
                    signal_type = "bullish_order_block"
                    rationale = f"Price at bullish Order Block ${ob.price_level:.2f}"
                    entry_price = current_price
                    stop_loss = ob.candle_low * 0.998  # Just below OB low
                else:
                    direction = "short"
                    signal_type = "bearish_order_block"
                    rationale = f"Price at bearish Order Block ${ob.price_level:.2f}"
                    entry_price = current_price
                    stop_loss = ob.candle_high * 1.002  # Just above OB high

                # Calculate signal strength
                strength = ob.confidence * 0.9  # OB signals are generally reliable

                signals.append(ICTSignal(
                    signal_type=signal_type,
                    strength=min(strength, 1.0),
                    direction=direction,
                    confidence=ob.confidence,
                    rationale=rationale,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    timeframe=ob.timeframe
                ))

        return signals

    def _analyze_bos_signals(
        self,
        bos_list: List[BreakOfStructure],
        current_price: float,
        df
    ) -> List[ICTSignal]:
        """Analyze Break of Structure for trading signals."""

        signals = []

        # Look for recent BOS and potential retests
        for bos in bos_list:
            # Check if this is a recent BOS (last 24 hours)
            time_since_bos = datetime.now() - bos.timestamp
            if time_since_bos.total_seconds() > 24 * 3600:
                continue

            # Check for retest opportunity
            if bos.retest_level:
                distance_to_retest = abs(current_price - bos.retest_level) / current_price

                if distance_to_retest < 0.003:  # Within 0.3% of retest level
                    if bos.side == "bullish":
                        direction = "long"
                        signal_type = "bullish_bos_retest"
                        rationale = f"Retest of bullish BOS at ${bos.retest_level:.2f}"
                        entry_price = current_price
                        stop_loss = bos.retest_level * 0.995
                    else:
                        direction = "short"
                        signal_type = "bearish_bos_retest"
                        rationale = f"Retest of bearish BOS at ${bos.retest_level:.2f}"
                        entry_price = current_price
                        stop_loss = bos.retest_level * 1.005

                    strength = bos.confidence * 0.8  # Retest signals are moderate confidence

                    signals.append(ICTSignal(
                        signal_type=signal_type,
                        strength=min(strength, 1.0),
                        direction=direction,
                        confidence=bos.confidence,
                        rationale=rationale,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss,
                        timeframe=bos.timeframe
                    ))

        return signals

    def _analyze_structure_confluence(
        self,
        fvg_list: List[FairValueGap],
        ob_list: List[OrderBlock],
        bos_list: List[BreakOfStructure],
        current_price: float
    ) -> Dict[str, Any]:
        """Analyze confluence between different ICT structures."""

        confluence_analysis = {
            'confluence_zones': [],
            'max_confluence_score': 0.0,
            'structure_density': 0.0,
            'rationale': []
        }

        try:
            # Create price levels grid for confluence analysis
            price_levels = {}

            # Add FVG levels
            for fvg in fvg_list:
                midline = fvg.midline
                if midline not in price_levels:
                    price_levels[midline] = {'fvgs': [], 'obs': [], 'bos': []}
                price_levels[midline]['fvgs'].append(fvg)

            # Add Order Block levels
            for ob in ob_list:
                level = ob.price_level
                if level not in price_levels:
                    price_levels[level] = {'fvgs': [], 'obs': [], 'bos': []}
                price_levels[level]['obs'].append(ob)

            # Add BOS levels
            for bos in bos_list:
                if bos.retest_level:
                    level = bos.retest_level
                    if level not in price_levels:
                        price_levels[level] = {'fvgs': [], 'obs': [], 'bos': []}
                    price_levels[level]['bos'].append(bos)

            # Calculate confluence scores
            for level, structures in price_levels.items():
                total_structures = len(structures['fvgs']) + len(structures['obs']) + len(structures['bos'])
                if total_structures > 1:  # Confluence requires 2+ structures
                    distance = abs(level - current_price) / current_price
                    proximity_score = max(0, 1 - distance / 0.01)  # 1% range

                    # Weight different structure types
                    confluence_score = (
                        len(structures['fvgs']) * 0.4 +
                        len(structures['obs']) * 0.4 +
                        len(structures['bos']) * 0.2
                    ) * proximity_score

                    if confluence_score > 0.3:  # Minimum confluence threshold
                        confluence_analysis['confluence_zones'].append({
                            'price_level': level,
                            'confluence_score': min(confluence_score, 1.0),
                            'structures': {
                                'fvgs': len(structures['fvgs']),
                                'obs': len(structures['obs']),
                                'bos': len(structures['bos'])
                            },
                            'distance_from_current': distance
                        })

                        confluence_analysis['rationale'].append(
                            f"ICT confluence at ${level:.2f} with {total_structures} structures"
                        )

            # Calculate max confluence score
            if confluence_analysis['confluence_zones']:
                confluence_analysis['max_confluence_score'] = max(
                    zone['confluence_score'] for zone in confluence_analysis['confluence_zones']
                )

            # Calculate structure density
            total_price_range = max(price_levels.keys()) - min(price_levels.keys()) if price_levels else 1
            total_structures = sum(len(s) for structures in price_levels.values() for s in structures.values())
            confluence_analysis['structure_density'] = total_structures / max(total_price_range * 1000, 1)

        except Exception as e:
            logger.error(f"Failed to analyze structure confluence: {e}")

        return confluence_analysis

    def _identify_structural_levels(
        self,
        fvg_list: List[FairValueGap],
        ob_list: List[OrderBlock],
        bos_list: List[BreakOfStructure],
        current_price: float
    ) -> Dict[str, Any]:
        """Identify key structural support and resistance levels."""

        levels = {
            'support_levels': [],
            'resistance_levels': [],
            'key_levels': []
        }

        try:
            # Identify support levels (below current price)
            for fvg in fvg_list:
                if fvg.top > fvg.bottom and fvg.bottom < current_price:  # Bullish FVG below
                    levels['support_levels'].append({
                        'level': fvg.bottom,
                        'type': 'bullish_fvg',
                        'strength': fvg.confidence,
                        'source': f'FVG bottom at ${fvg.bottom:.2f}'
                    })

            for ob in ob_list:
                if ob.side == "bullish" and ob.candle_low < current_price:
                    levels['support_levels'].append({
                        'level': ob.candle_low,
                        'type': 'bullish_order_block',
                        'strength': ob.confidence,
                        'source': f'Bullish OB low at ${ob.candle_low:.2f}'
                    })

            # Identify resistance levels (above current price)
            for fvg in fvg_list:
                if fvg.top < fvg.bottom and fvg.top > current_price:  # Bearish FVG above
                    levels['resistance_levels'].append({
                        'level': fvg.top,
                        'type': 'bearish_fvg',
                        'strength': fvg.confidence,
                        'source': f'FVG top at ${fvg.top:.2f}'
                    })

            for ob in ob_list:
                if ob.side == "bearish" and ob.candle_high > current_price:
                    levels['resistance_levels'].append({
                        'level': ob.candle_high,
                        'type': 'bearish_order_block',
                        'strength': ob.confidence,
                        'source': f'Bearish OB high at ${ob.candle_high:.2f}'
                    })

            # Sort by strength and proximity
            levels['support_levels'].sort(key=lambda x: (-x['strength'], x['level']))
            levels['resistance_levels'].sort(key=lambda x: (-x['strength'], x['level']))

            # Identify key levels (highest strength)
            all_levels = levels['support_levels'] + levels['resistance_levels']
            if all_levels:
                max_strength = max(level['strength'] for level in all_levels)
                levels['key_levels'] = [
                    level for level in all_levels
                    if level['strength'] >= max_strength * 0.8
                ]

        except Exception as e:
            logger.error(f"Failed to identify structural levels: {e}")

        return levels

    def _calculate_overall_ict_score(
        self,
        signals: List[ICTSignal],
        confluence_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall ICT score from signals and confluence."""

        if not signals:
            return 0.0

        # Base score from signals
        signal_scores = [signal.strength * signal.confidence for signal in signals]
        max_signal_score = max(signal_scores) if signal_scores else 0.0

        # Boost from confluence
        confluence_boost = confluence_analysis.get('max_confluence_score', 0.0) * 0.3

        # Combine scores
        overall_score = max_signal_score * 0.7 + confluence_boost

        return min(overall_score, 1.0)

    def _empty_ict_score(self) -> ICTScore:
        """Return empty ICT score when no data available."""
        return ICTScore(
            overall_score=0.0,
            long_bias=0.0,
            short_bias=0.0,
            signals=[],
            confluence_analysis={},
            structural_levels={},
            timestamp=datetime.now()
        )