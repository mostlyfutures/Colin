"""
Volume and Open Interest Scorer for institutional signal analysis.

This module scores signals based on volume patterns, Open Interest trends,
and funding rate dynamics for crypto perpetual markets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from ..core.config import ConfigManager


@dataclass
class VolumeSignal:
    """Represents a volume-based signal."""
    signal_type: str
    strength: float
    direction: str  # "long" or "short"
    confidence: float
    rationale: str
    volume_ratio: float
    price_change: float


@dataclass
class OISignal:
    """Represents an Open Interest-based signal."""
    signal_type: str
    strength: float
    direction: str
    confidence: float
    rationale: str
    oi_change_pct: float
    price_change_pct: float


@dataclass
class VolumeOIScore:
    """Comprehensive Volume and OI scoring result."""
    overall_score: float
    volume_score: float
    oi_score: float
    funding_score: float
    volume_signals: List[VolumeSignal]
    oi_signals: List[OISignal]
    trend_analysis: Dict[str, Any]
    timestamp: datetime


class VolumeOIScorer:
    """Scores volume and Open Interest based institutional signals."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize Volume/OI scorer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.scoring_config = self.config.scoring

    def score_volume_oi_signals(
        self,
        current_price: float,
        volume_data: Dict[str, Any],
        oi_data: Dict[str, Any],
        funding_data: Optional[Dict[str, Any]] = None
    ) -> VolumeOIScore:
        """
        Score comprehensive volume and OI signals.

        Args:
            current_price: Current price level
            volume_data: Volume analysis data
            oi_data: Open Interest data
            funding_data: Funding rate data (optional)

        Returns:
            Comprehensive Volume/OI score
        """
        try:
            volume_signals = []
            oi_signals = []

            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(volume_data, current_price)
            volume_signals.extend(volume_analysis['signals'])

            # Analyze Open Interest trends
            oi_analysis = self._analyze_oi_trends(oi_data, current_price)
            oi_signals.extend(oi_analysis['signals'])

            # Analyze funding rate dynamics
            funding_analysis = self._analyze_funding_dynamics(funding_data, current_price)

            # Calculate individual scores
            volume_score = self._calculate_volume_score(volume_signals)
            oi_score = self._calculate_oi_score(oi_signals)
            funding_score = funding_analysis.get('score', 0.0)

            # Analyze overall trend
            trend_analysis = self._analyze_volume_oi_trend(
                volume_analysis, oi_analysis, funding_analysis
            )

            # Calculate overall score
            overall_score = self._calculate_overall_volume_oi_score(
                volume_score, oi_score, funding_score
            )

            result = VolumeOIScore(
                overall_score=min(overall_score, 1.0),
                volume_score=volume_score,
                oi_score=oi_score,
                funding_score=funding_score,
                volume_signals=volume_signals,
                oi_signals=oi_signals,
                trend_analysis=trend_analysis,
                timestamp=datetime.now()
            )

            logger.debug(f"Volume/OI scoring complete: Overall={overall_score:.3f}, Volume={volume_score:.3f}, OI={oi_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"Failed to score Volume/OI signals: {e}")
            return self._empty_volume_oi_score()

    def score_volume_confirmation(
        self,
        current_volume: float,
        historical_volume: pd.Series,
        price_change: float,
        direction: str
    ) -> Dict[str, Any]:
        """
        Score volume confirmation for price moves.

        Args:
            current_volume: Current trading volume
            historical_volume: Historical volume data
            price_change: Price change percentage
            direction: Direction of price move ("long" or "short")

        Returns:
            Volume confirmation analysis
        """
        try:
            confirmation_analysis = {
                'volume_confirmed': False,
                'volume_ratio': 0.0,
                'volume_score': 0.0,
                'strength': 'weak',
                'rationale': []
            }

            if historical_volume.empty:
                return confirmation_analysis

            # Calculate volume metrics
            avg_volume = historical_volume.mean()
            volume_std = historical_volume.std()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            confirmation_analysis['volume_ratio'] = volume_ratio

            # Volume confirmation criteria
            strong_volume_threshold = 1.5  # 150% of average
            moderate_volume_threshold = 1.2  # 120% of average

            if volume_ratio >= strong_volume_threshold:
                confirmation_analysis['volume_confirmed'] = True
                confirmation_analysis['volume_score'] = min(volume_ratio / 3, 1.0)  # Cap at 300% avg
                confirmation_analysis['strength'] = 'strong'
                confirmation_analysis['rationale'].append(
                    f"Strong volume confirmation: {volume_ratio:.1f}x average volume"
                )
            elif volume_ratio >= moderate_volume_threshold:
                confirmation_analysis['volume_confirmed'] = True
                confirmation_analysis['volume_score'] = volume_ratio / strong_volume_threshold
                confirmation_analysis['strength'] = 'moderate'
                confirmation_analysis['rationale'].append(
                    f"Moderate volume confirmation: {volume_ratio:.1f}x average volume"
                )
            else:
                confirmation_analysis['volume_score'] = volume_ratio / moderate_volume_threshold * 0.5
                confirmation_analysis['rationale'].append(
                    f"Weak volume: {volume_ratio:.1f}x average volume"
                )

            # Check volume-price divergence
            volume_price_alignment = self._check_volume_price_alignment(
                volume_ratio, price_change, direction
            )
            confirmation_analysis.update(volume_price_alignment)

            return confirmation_analysis

        except Exception as e:
            logger.error(f"Failed to score volume confirmation: {e}")
            return {
                'volume_confirmed': False,
                'volume_ratio': 0.0,
                'volume_score': 0.0,
                'strength': 'weak',
                'rationale': []
            }

    def _analyze_volume_patterns(self, volume_data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Analyze volume patterns for signals."""

        analysis = {
            'signals': [],
            'patterns': {},
            'trend': 'neutral'
        }

        try:
            # Extract volume metrics
            current_volume = volume_data.get('current_volume', 0)
            avg_volume = volume_data.get('avg_volume', 0)
            volume_trend = volume_data.get('volume_trend', 0)
            price_volume_correlation = volume_data.get('price_volume_correlation', 0)

            if avg_volume == 0:
                return analysis

            volume_ratio = current_volume / avg_volume

            # Volume spike signal
            if volume_ratio > 2.0:  # 200% of average
                direction = "long" if price_volume_correlation > 0.3 else "short"
                strength = min(volume_ratio / 5, 1.0)  # Normalize

                analysis['signals'].append(VolumeSignal(
                    signal_type="volume_spike",
                    strength=strength,
                    direction=direction,
                    confidence=min(volume_ratio / 3, 1.0),
                    rationale=f"Volume spike: {volume_ratio:.1f}x average volume",
                    volume_ratio=volume_ratio,
                    price_change=volume_data.get('price_change', 0)
                ))

            # Volume trend signal
            if abs(volume_trend) > 0.2:  # 20% trend
                direction = "long" if volume_trend > 0 else "short"
                strength = min(abs(volume_trend) * 2, 1.0)

                analysis['signals'].append(VolumeSignal(
                    signal_type="volume_trend",
                    strength=strength,
                    direction=direction,
                    confidence=min(abs(volume_trend) * 1.5, 1.0),
                    rationale=f"Volume {'increasing' if volume_trend > 0 else 'decreasing'} trend: {volume_trend:.1%}",
                    volume_ratio=volume_ratio,
                    price_change=volume_data.get('price_change', 0)
                ))

            # Volume exhaustion signal (high volume with small price move)
            price_change = volume_data.get('price_change', 0)
            if volume_ratio > 1.5 and abs(price_change) < 0.005:  # High volume, small price move
                analysis['signals'].append(VolumeSignal(
                    signal_type="volume_exhaustion",
                    strength=volume_ratio / 3,
                    direction="neutral",
                    confidence=0.6,
                    rationale=f"Volume exhaustion: high volume ({volume_ratio:.1f}x) with small price move",
                    volume_ratio=volume_ratio,
                    price_change=price_change
                ))

            # Determine overall volume trend
            analysis['trend'] = 'bullish' if volume_trend > 0.1 else 'bearish' if volume_trend < -0.1 else 'neutral'

        except Exception as e:
            logger.error(f"Failed to analyze volume patterns: {e}")

        return analysis

    def _analyze_oi_trends(self, oi_data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Analyze Open Interest trends for signals."""

        analysis = {
            'signals': [],
            'trend': 'neutral',
            'momentum': 'stable'
        }

        try:
            # Extract OI metrics
            current_oi = oi_data.get('current_oi', 0)
            oi_change = oi_data.get('oi_change_24h', 0)
            oi_change_pct = oi_data.get('oi_change_pct', 0)
            price_change_pct = oi_data.get('price_change_pct', 0)

            if current_oi == 0:
                return analysis

            # OI divergence signal
            if abs(oi_change_pct) > 0.1:  # 10% OI change
                # Check for divergence with price
                if oi_change_pct > 0 and price_change_pct < -0.01:
                    # OI increasing but price decreasing - bearish divergence
                    strength = min(abs(oi_change_pct) * 5, 1.0)
                    analysis['signals'].append(OISignal(
                        signal_type="oi_bearish_divergence",
                        strength=strength,
                        direction="short",
                        confidence=0.7,
                        rationale=f"OI increasing {oi_change_pct:.1%} while price falling {price_change_pct:.1%}",
                        oi_change_pct=oi_change_pct,
                        price_change_pct=price_change_pct
                    ))
                    analysis['trend'] = 'bearish'

                elif oi_change_pct < 0 and price_change_pct > 0.01:
                    # OI decreasing but price increasing - bullish divergence
                    strength = min(abs(oi_change_pct) * 5, 1.0)
                    analysis['signals'].append(OISignal(
                        signal_type="oi_bullish_divergence",
                        strength=strength,
                        direction="long",
                        confidence=0.7,
                        rationale=f"OI decreasing {oi_change_pct:.1%} while price rising {price_change_pct:.1%}",
                        oi_change_pct=oi_change_pct,
                        price_change_pct=price_change_pct
                    ))
                    analysis['trend'] = 'bullish'

            # OI momentum signal
            if abs(oi_change_pct) > 0.2:  # 20% OI change
                direction = "long" if oi_change_pct > 0 else "short"
                strength = min(abs(oi_change_pct) * 2, 1.0)

                analysis['signals'].append(OISignal(
                    signal_type="oi_momentum",
                    strength=strength,
                    direction=direction,
                    confidence=min(abs(oi_change_pct) * 1.5, 1.0),
                    rationale=f"Strong OI momentum: {oi_change_pct:.1%} change",
                    oi_change_pct=oi_change_pct,
                    price_change_pct=price_change_pct
                ))

                analysis['momentum'] = 'increasing' if oi_change_pct > 0 else 'decreasing'

            # OI exhaustion signal (high OI with declining volume)
            current_volume = oi_data.get('current_volume', 0)
            avg_volume = oi_data.get('avg_volume', 0)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if current_oi > oi_data.get('avg_oi', current_oi) * 1.5 and volume_ratio < 0.8:
                analysis['signals'].append(OISignal(
                    signal_type="oi_exhaustion",
                    strength=0.6,
                    direction="neutral",
                    confidence=0.5,
                    rationale=f"High OI with declining volume - potential exhaustion",
                    oi_change_pct=oi_change_pct,
                    price_change_pct=price_change_pct
                ))

        except Exception as e:
            logger.error(f"Failed to analyze OI trends: {e}")

        return analysis

    def _analyze_funding_dynamics(self, funding_data: Optional[Dict[str, Any]], current_price: float) -> Dict[str, Any]:
        """Analyze funding rate dynamics."""

        analysis = {
            'score': 0.0,
            'signal': 'neutral',
            'rationale': []
        }

        if not funding_data:
            return analysis

        try:
            current_funding = funding_data.get('current_funding_rate', 0)
            funding_trend = funding_data.get('funding_trend', 0)
            avg_funding = funding_data.get('avg_funding_rate', 0)

            # Extreme funding rate signals
            if abs(current_funding) > 0.01:  # 1% funding rate (very high)
                if current_funding > 0:
                    # High positive funding - shorts paying longs - bearish signal
                    analysis['score'] = min(current_funding * 50, 0.8)
                    analysis['signal'] = 'bearish'
                    analysis['rationale'].append(f"High positive funding: {current_funding:.4f} - bearish pressure")
                else:
                    # High negative funding - longs paying shorts - bullish signal
                    analysis['score'] = min(abs(current_funding) * 50, 0.8)
                    analysis['signal'] = 'bullish'
                    analysis['rationale'].append(f"High negative funding: {current_funding:.4f} - bullish pressure")

            # Funding rate reversal signal
            if avg_funding != 0 and current_funding / avg_funding < -0.5:
                analysis['score'] = 0.4
                analysis['signal'] = 'reversal'
                analysis['rationale'].append(f"Funding rate reversal from {avg_funding:.4f} to {current_funding:.4f}")

            # Funding trend signal
            if abs(funding_trend) > 0.002:  # 0.2% funding change
                direction = "bullish" if funding_trend < 0 else "bearish"
                analysis['score'] = min(abs(funding_trend) * 100, 0.5)
                analysis['rationale'].append(f"Funding rate {'decreasing' if funding_trend < 0 else 'increasing'}: {funding_trend:.6f}")

        except Exception as e:
            logger.error(f"Failed to analyze funding dynamics: {e}")

        return analysis

    def _check_volume_price_alignment(
        self,
        volume_ratio: float,
        price_change: float,
        direction: str
    ) -> Dict[str, Any]:
        """Check volume and price alignment."""

        alignment = {
            'aligned': False,
            'alignment_score': 0.0,
            'divergence_type': None
        }

        try:
            # Check if volume supports price direction
            if direction == "long" and price_change > 0 and volume_ratio > 1.2:
                alignment['aligned'] = True
                alignment['alignment_score'] = min(volume_ratio * 0.3, 0.9)
            elif direction == "short" and price_change < 0 and volume_ratio > 1.2:
                alignment['aligned'] = True
                alignment['alignment_score'] = min(volume_ratio * 0.3, 0.9)
            elif direction == "long" and price_change < 0 and volume_ratio > 1.5:
                # Bullish divergence
                alignment['divergence_type'] = 'bullish'
                alignment['alignment_score'] = 0.6
            elif direction == "short" and price_change > 0 and volume_ratio > 1.5:
                # Bearish divergence
                alignment['divergence_type'] = 'bearish'
                alignment['alignment_score'] = 0.6

        except Exception as e:
            logger.error(f"Failed to check volume-price alignment: {e}")

        return alignment

    def _calculate_volume_score(self, volume_signals: List[VolumeSignal]) -> float:
        """Calculate overall volume score."""
        if not volume_signals:
            return 0.0

        # Use the strongest volume signal
        max_score = max(signal.strength * signal.confidence for signal in volume_signals)
        return min(max_score, 1.0)

    def _calculate_oi_score(self, oi_signals: List[OISignal]) -> float:
        """Calculate overall OI score."""
        if not oi_signals:
            return 0.0

        # Use the strongest OI signal
        max_score = max(signal.strength * signal.confidence for signal in oi_signals)
        return min(max_score, 1.0)

    def _analyze_volume_oi_trend(
        self,
        volume_analysis: Dict[str, Any],
        oi_analysis: Dict[str, Any],
        funding_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze overall volume-OI trend."""

        trend_analysis = {
            'overall_trend': 'neutral',
            'confidence': 0.0,
            'supporting_factors': [],
            'conflicting_factors': []
        }

        try:
            volume_trend = volume_analysis.get('trend', 'neutral')
            oi_trend = oi_analysis.get('trend', 'neutral')
            funding_signal = funding_analysis.get('signal', 'neutral')

            # Count supporting signals
            trend_votes = {'bullish': 0, 'bearish': 0, 'neutral': 0}

            if volume_trend != 'neutral':
                trend_votes[volume_trend] += 1
                trend_analysis['supporting_factors'].append(f"Volume trend: {volume_trend}")

            if oi_trend != 'neutral':
                trend_votes[oi_trend] += 1
                trend_analysis['supporting_factors'].append(f"OI trend: {oi_trend}")

            if funding_signal != 'neutral':
                if funding_signal == 'bullish':
                    trend_votes['bullish'] += 1
                    trend_analysis['supporting_factors'].append("Funding rate: bullish")
                elif funding_signal == 'bearish':
                    trend_votes['bearish'] += 1
                    trend_analysis['supporting_factors'].append("Funding rate: bearish")

            # Determine overall trend
            if trend_votes['bullish'] > trend_votes['bearish']:
                trend_analysis['overall_trend'] = 'bullish'
                trend_analysis['confidence'] = trend_votes['bullish'] / (trend_votes['bullish'] + trend_votes['bearish'])
            elif trend_votes['bearish'] > trend_votes['bullish']:
                trend_analysis['overall_trend'] = 'bearish'
                trend_analysis['confidence'] = trend_votes['bearish'] / (trend_votes['bullish'] + trend_votes['bearish'])

        except Exception as e:
            logger.error(f"Failed to analyze volume-OI trend: {e}")

        return trend_analysis

    def _calculate_overall_volume_oi_score(
        self,
        volume_score: float,
        oi_score: float,
        funding_score: float
    ) -> float:
        """Calculate overall Volume/OI score."""

        # Weight the components
        volume_weight = self.scoring_config['weights'].get('volume_oi_confirmation', 0.15)

        # Split volume_oi_confirmation between volume and OI
        volume_weight *= 0.6
        oi_weight = volume_weight * 0.67  # 40% of 15%
        funding_weight = 0.05  # Small weight for funding

        overall_score = (volume_score * volume_weight +
                        oi_score * oi_weight +
                        funding_score * funding_weight)

        return min(overall_score, 1.0)

    def _empty_volume_oi_score(self) -> VolumeOIScore:
        """Return empty Volume/OI score when no data available."""
        return VolumeOIScore(
            overall_score=0.0,
            volume_score=0.0,
            oi_score=0.0,
            funding_score=0.0,
            volume_signals=[],
            oi_signals=[],
            trend_analysis={},
            timestamp=datetime.now()
        )