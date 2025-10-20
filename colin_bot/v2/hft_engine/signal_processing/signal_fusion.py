"""
Multi-Signal Fusion Engine

Combines multiple trading signals to generate enhanced trading decisions
with improved confidence through consensus building and conflict resolution.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from collections import deque
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..utils.data_structures import (
    TradingSignal, SignalDirection, OFISignal, BookSkewSignal
)
from ..utils.math_utils import calculate_zscore, calculate_weighted_average
from ..utils.performance import profile_async_hft_operation, LatencyTracker


class FusionMethod(Enum):
    """Signal fusion methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS_BUILDER = "consensus_builder"
    CONFLICT_RESOLUTION = "conflict_resolution"
    ENSEMBLE_VOTING = "ensemble_voting"
    BAYESIAN_FUSION = "bayesian_fusion"


class SignalQuality(Enum):
    """Signal quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class SignalWeight:
    """Signal weight configuration."""
    signal_type: str
    base_weight: float
    quality_multiplier: float = 1.0
    confidence_boost: float = 0.0
    max_weight: float = 1.0
    min_weight: float = 0.1


@dataclass
class FusionSignal:
    """Fused trading signal with enhanced confidence."""
    symbol: str
    direction: SignalDirection
    confidence: float
    strength: float
    fusion_method: FusionMethod
    component_signals: List[Dict] = field(default_factory=list)
    consensus_score: float = 0.0
    conflict_score: float = 0.0
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Check if signal has high confidence."""
        return self.confidence >= 0.8

    @property
    def is_consensus_signal(self) -> bool:
        """Check if signal is based on consensus."""
        return self.consensus_score >= 0.7

    @property
    def has_conflicts(self) -> bool:
        """Check if there were conflicts in fusion."""
        return self.conflict_score > 0.3


class SignalFusionEngine:
    """
    Advanced signal fusion engine for combining multiple trading signals.

    Implements various fusion methods to enhance signal reliability
    and confidence through intelligent combination of different signal types.
    """

    def __init__(self, window_size: int = 50, fusion_methods: List[FusionMethod] = None):
        self.window_size = window_size
        self.fusion_methods = fusion_methods or [
            FusionMethod.CONSENSUS_BUILDER,
            FusionMethod.WEIGHTED_AVERAGE,
            FusionMethod.CONFLICT_RESOLUTION
        ]
        self.logger = logging.getLogger(__name__)

        # Signal storage
        self.signal_history: deque = deque(maxlen=window_size)
        self.fusion_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)

        # Signal weights configuration
        self.signal_weights = {
            'ofi': SignalWeight('ofi', 0.4, 1.2, 0.1),
            'book_skew': SignalWeight('book_skew', 0.3, 1.1, 0.05),
            'liquidity': SignalWeight('liquidity', 0.2, 1.0, 0.0),
            'technical': SignalWeight('technical', 0.1, 0.9, 0.0)
        }

        # Fusion parameters
        self.consensus_threshold = 0.6
        self.conflict_penalty = 0.2
        self.quality_threshold = 0.5
        self.min_signals_for_fusion = 2

        # Performance tracking
        self.fusion_count = 0
        self.consensus_count = 0
        self.conflict_count = 0

        # Dynamic weight adjustment
        self.performance_tracking = {}
        self.weight_adjustment_factor = 0.1

    @profile_async_hft_operation("signal_fusion", LatencyTracker())
    async def fuse_signals(self, symbol: str, signals: List) -> Optional[FusionSignal]:
        """
        Fuse multiple signals into an enhanced trading signal.

        Args:
            symbol: Trading symbol
            signals: List of input signals (OFI, BookSkew, etc.)

        Returns:
            Fused signal with enhanced confidence or None if insufficient signals
        """
        if len(signals) < self.min_signals_for_fusion:
            self.logger.debug(f"Insufficient signals for fusion: {len(signals)}")
            return None

        try:
            # Preprocess signals
            processed_signals = self._preprocess_signals(signals)
            if len(processed_signals) < self.min_signals_for_fusion:
                return None

            # Calculate signal quality
            quality_scores = self._calculate_signal_quality(processed_signals)

            # Apply fusion methods
            fusion_results = []
            for method in self.fusion_methods:
                result = await self._apply_fusion_method(method, processed_signals, quality_scores)
                if result:
                    fusion_results.append(result)

            if not fusion_results:
                return None

            # Select best fusion result
            best_signal = self._select_best_fusion_result(fusion_results)

            # Post-process and validate
            final_signal = self._post_process_signal(best_signal, processed_signals)

            # Store results
            self.signal_history.append({
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'input_signals': len(signals),
                'fusion_signal': final_signal
            })

            self.fusion_count += 1
            if final_signal.is_consensus_signal:
                self.consensus_count += 1
            if final_signal.has_conflicts:
                self.conflict_count += 1

            return final_signal

        except Exception as e:
            self.logger.error(f"Error fusing signals for {symbol}: {e}")
            return None

    def _preprocess_signals(self, signals: List) -> List[Dict]:
        """
        Preprocess input signals for fusion.

        Args:
            signals: Raw input signals

        Returns:
            List of processed signal dictionaries
        """
        processed = []

        for signal in signals:
            try:
                # Extract signal information based on type
                signal_dict = self._extract_signal_info(signal)
                if signal_dict:
                    processed.append(signal_dict)
            except Exception as e:
                self.logger.warning(f"Error processing signal {type(signal)}: {e}")

        return processed

    def _extract_signal_info(self, signal) -> Optional[Dict]:
        """Extract relevant information from different signal types."""
        if isinstance(signal, OFISignal):
            return {
                'type': 'ofi',
                'direction': signal.forecast_direction,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'value': signal.ofi_value,
                'timestamp': signal.timestamp,
                'metadata': getattr(signal, 'hawkes_intensity', {})
            }
        elif isinstance(signal, BookSkewSignal):
            return {
                'type': 'book_skew',
                'direction': signal.signal_direction,
                'confidence': signal.confidence,
                'strength': signal.strength_category,
                'value': signal.skew_value,
                'timestamp': signal.timestamp,
                'metadata': {
                    'bid_size': getattr(signal, 'bid_size', 0),
                    'ask_size': getattr(signal, 'ask_size', 0)
                }
            }
        elif isinstance(signal, TradingSignal):
            return {
                'type': 'trading',
                'direction': signal.direction,
                'confidence': signal.confidence,
                'strength': signal.strength,
                'value': getattr(signal, 'signal_value', 0.0),
                'timestamp': signal.timestamp,
                'metadata': getattr(signal, 'metadata', {})
            }
        else:
            self.logger.warning(f"Unknown signal type: {type(signal)}")
            return None

    def _calculate_signal_quality(self, signals: List[Dict]) -> Dict[str, float]:
        """
        Calculate quality scores for each signal.

        Args:
            signals: List of processed signals

        Returns:
            Dictionary mapping signal types to quality scores
        """
        quality_scores = {}

        for signal in signals:
            signal_type = signal['type']

            # Base quality from confidence
            base_quality = signal['confidence']

            # Adjust based on recency
            age_seconds = (datetime.now(timezone.utc) - signal['timestamp']).total_seconds()
            recency_factor = max(0.1, 1.0 - (age_seconds / 300.0))  # 5-minute decay

            # Adjust based on signal strength
            strength_factor = self._get_strength_multiplier(signal['strength'])

            # Final quality score
            quality = base_quality * recency_factor * strength_factor
            quality_scores[signal_type] = min(1.0, quality)

        return quality_scores

    def _get_strength_multiplier(self, strength) -> float:
        """Get quality multiplier based on signal strength."""
        if isinstance(strength, SignalStrength):
            strength_multipliers = {
                SignalStrength.STRONG: 1.2,
                SignalStrength.MODERATE: 1.0,
                SignalStrength.WEAK: 0.7
            }
            return strength_multipliers.get(strength, 1.0)
        elif isinstance(strength, str):
            strength_multipliers = {
                'strong': 1.2,
                'moderate': 1.0,
                'weak': 0.7
            }
            return strength_multipliers.get(strength.lower(), 1.0)
        else:
            return 1.0

    async def _apply_fusion_method(self, method: FusionMethod, signals: List[Dict],
                                 quality_scores: Dict[str, float]) -> Optional[FusionSignal]:
        """Apply a specific fusion method to combine signals."""
        if method == FusionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(signals, quality_scores)
        elif method == FusionMethod.CONSENSUS_BUILDER:
            return self._consensus_builder_fusion(signals, quality_scores)
        elif method == FusionMethod.CONFLICT_RESOLUTION:
            return self._conflict_resolution_fusion(signals, quality_scores)
        elif method == FusionMethod.ENSEMBLE_VOTING:
            return self._ensemble_voting_fusion(signals, quality_scores)
        elif method == FusionMethod.BAYESIAN_FUSION:
            return self._bayesian_fusion(signals, quality_scores)
        else:
            self.logger.warning(f"Unknown fusion method: {method}")
            return None

    def _weighted_average_fusion(self, signals: List[Dict], quality_scores: Dict[str, float]) -> Optional[FusionSignal]:
        """Combine signals using weighted averaging."""
        if not signals:
            return None

        # Calculate weighted direction
        direction_values = []
        weights = []

        for signal in signals:
            signal_type = signal['type']
            weight = self.signal_weights[signal_type].base_weight

            # Adjust weight based on quality
            quality_multiplier = quality_scores.get(signal_type, 0.5)
            final_weight = weight * quality_multiplier

            # Convert direction to numeric value
            direction_value = self._direction_to_numeric(signal['direction'])

            direction_values.append(direction_value)
            weights.append(final_weight)

        if not direction_values:
            return None

        # Calculate weighted average
        weighted_direction = calculate_weighted_average(direction_values, weights)
        final_direction = self._numeric_to_direction(weighted_direction)

        # Calculate confidence
        avg_confidence = np.average([s['confidence'] for s in signals], weights=weights)
        consensus_score = self._calculate_consensus_score(signals)
        conflict_score = self._calculate_conflict_score(signals)

        # Adjust confidence based on consensus/conflict
        final_confidence = avg_confidence * (1.0 + consensus_score - conflict_score)
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Determine strength
        final_strength = self._determine_strength(final_confidence, weighted_direction)

        return FusionSignal(
            symbol=signals[0].get('symbol', 'UNKNOWN'),
            direction=final_direction,
            confidence=final_confidence,
            strength=final_strength,
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            component_signals=signals,
            consensus_score=consensus_score,
            conflict_score=conflict_score,
            quality_score=np.mean(list(quality_scores.values())),
            metadata={'weights': weights, 'method': 'weighted_average'}
        )

    def _consensus_builder_fusion(self, signals: List[Dict], quality_scores: Dict[str, float]) -> Optional[FusionSignal]:
        """Build consensus from multiple signals."""
        if not signals:
            return None

        # Count directions
        direction_counts = {SignalDirection.LONG: 0, SignalDirection.SHORT: 0, SignalDirection.HOLD: 0}
        weighted_counts = {SignalDirection.LONG: 0.0, SignalDirection.SHORT: 0.0, SignalDirection.HOLD: 0.0}

        for signal in signals:
            signal_type = signal['type']
            weight = self.signal_weights[signal_type].base_weight
            quality_mult = quality_scores.get(signal_type, 0.5)
            final_weight = weight * quality_mult

            direction = signal['direction']
            direction_counts[direction] += 1
            weighted_counts[direction] += final_weight

        # Find consensus direction
        total_signals = len(signals)
        consensus_threshold_count = max(2, int(total_signals * self.consensus_threshold))

        consensus_direction = None
        max_weighted_count = 0

        for direction in SignalDirection:
            if direction_counts[direction] >= consensus_threshold_count:
                if weighted_counts[direction] > max_weighted_count:
                    max_weighted_count = weighted_counts[direction]
                    consensus_direction = direction

        if consensus_direction is None:
            # No consensus - use HOLD
            consensus_direction = SignalDirection.HOLD

        # Calculate confidence based on consensus strength
        consensus_ratio = direction_counts[consensus_direction] / total_signals
        weight_ratio = weighted_counts[consensus_direction] / sum(weighted_counts.values())

        base_confidence = np.mean([s['confidence'] for s in signals])
        consensus_bonus = consensus_ratio * weight_ratio
        final_confidence = base_confidence * (0.7 + 0.3 * consensus_bonus)

        # Calculate scores
        consensus_score = consensus_ratio
        conflict_score = 1.0 - consensus_ratio

        return FusionSignal(
            symbol=signals[0].get('symbol', 'UNKNOWN'),
            direction=consensus_direction,
            confidence=min(1.0, final_confidence),
            strength=self._determine_strength(final_confidence, consensus_ratio),
            fusion_method=FusionMethod.CONSENSUS_BUILDER,
            component_signals=signals,
            consensus_score=consensus_score,
            conflict_score=conflict_score,
            quality_score=np.mean(list(quality_scores.values())),
            metadata={
                'consensus_ratio': consensus_ratio,
                'direction_counts': {k.value: v for k, v in direction_counts.items()},
                'method': 'consensus_builder'
            }
        )

    def _conflict_resolution_fusion(self, signals: List[Dict], quality_scores: Dict[str, float]) -> Optional[FusionSignal]:
        """Resolve conflicts between conflicting signals."""
        if not signals:
            return None

        # Separate signals by direction
        long_signals = [s for s in signals if s['direction'] == SignalDirection.LONG]
        short_signals = [s for s in signals if s['direction'] == SignalDirection.SHORT]
        hold_signals = [s for s in signals if s['direction'] == SignalDirection.HOLD]

        # Calculate weighted scores for each direction
        def calculate_direction_score(direction_signals):
            if not direction_signals:
                return 0.0

            total_weight = 0.0
            weighted_confidence = 0.0

            for signal in direction_signals:
                signal_type = signal['type']
                weight = self.signal_weights[signal_type].base_weight
                quality_mult = quality_scores.get(signal_type, 0.5)
                final_weight = weight * quality_mult

                total_weight += final_weight
                weighted_confidence += final_weight * signal['confidence']

            return weighted_confidence / total_weight if total_weight > 0 else 0.0

        long_score = calculate_direction_score(long_signals)
        short_score = calculate_direction_score(short_signals)
        hold_score = calculate_direction_score(hold_signals)

        # Determine final direction
        max_score = max(long_score, short_score, hold_score)

        if max_score == long_score:
            final_direction = SignalDirection.LONG
        elif max_score == short_score:
            final_direction = SignalDirection.SHORT
        else:
            final_direction = SignalDirection.HOLD

        # Calculate conflict level
        signal_types = set(s['type'] for s in signals)
        conflict_level = len(signal_types) / max(1, len(signals))

        # Adjust confidence based on conflict
        base_confidence = max(long_score, short_score, hold_score)
        conflict_penalty = self.conflict_penalty * conflict_level
        final_confidence = base_confidence * (1.0 - conflict_penalty)

        # Calculate consensus and conflict scores
        consensus_score = 1.0 - conflict_level
        conflict_score = conflict_level

        return FusionSignal(
            symbol=signals[0].get('symbol', 'UNKNOWN'),
            direction=final_direction,
            confidence=max(0.0, min(1.0, final_confidence)),
            strength=self._determine_strength(final_confidence, max_score),
            fusion_method=FusionMethod.CONFLICT_RESOLUTION,
            component_signals=signals,
            consensus_score=consensus_score,
            conflict_score=conflict_score,
            quality_score=np.mean(list(quality_scores.values())),
            metadata={
                'direction_scores': {
                    'long': long_score,
                    'short': short_score,
                    'hold': hold_score
                },
                'conflict_level': conflict_level,
                'method': 'conflict_resolution'
            }
        )

    def _ensemble_voting_fusion(self, signals: List[Dict], quality_scores: Dict[str, float]) -> Optional[FusionSignal]:
        """Combine signals using ensemble voting approach."""
        # Similar to consensus builder but with weighted voting
        return self._consensus_builder_fusion(signals, quality_scores)

    def _bayesian_fusion(self, signals: List[Dict], quality_scores: Dict[str, float]) -> Optional[FusionSignal]:
        """Combine signals using Bayesian fusion."""
        # Simplified Bayesian fusion - would be more complex in production
        return self._weighted_average_fusion(signals, quality_scores)

    def _direction_to_numeric(self, direction: SignalDirection) -> float:
        """Convert signal direction to numeric value."""
        direction_map = {
            SignalDirection.LONG: 1.0,
            SignalDirection.SHORT: -1.0,
            SignalDirection.HOLD: 0.0
        }
        return direction_map.get(direction, 0.0)

    def _numeric_to_direction(self, value: float) -> SignalDirection:
        """Convert numeric value to signal direction."""
        if value > 0.3:
            return SignalDirection.LONG
        elif value < -0.3:
            return SignalDirection.SHORT
        else:
            return SignalDirection.HOLD

    def _calculate_consensus_score(self, signals: List[Dict]) -> float:
        """Calculate consensus score from signals."""
        if len(signals) < 2:
            return 1.0

        directions = [s['direction'] for s in signals]
        main_direction = max(set(directions), key=directions.count)
        consensus_count = directions.count(main_direction)

        return consensus_count / len(signals)

    def _calculate_conflict_score(self, signals: List[Dict]) -> float:
        """Calculate conflict score from signals."""
        if len(signals) < 2:
            return 0.0

        directions = [s['direction'] for s in signals]
        unique_directions = len(set(directions))

        # Normalize by max possible conflicts
        return (unique_directions - 1) / 2.0

    def _determine_strength(self, confidence: float, signal_magnitude: float) -> float:
        """Determine signal strength from confidence and magnitude."""
        combined_strength = (confidence + abs(signal_magnitude)) / 2.0

        if combined_strength >= 0.8:
            return SignalStrength.STRONG
        elif combined_strength >= 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _select_best_fusion_result(self, results: List[FusionSignal]) -> FusionSignal:
        """Select the best fusion result from multiple methods."""
        if not results:
            raise ValueError("No fusion results to select from")

        if len(results) == 1:
            return results[0]

        # Score each result
        best_result = None
        best_score = -1.0

        for result in results:
            # Composite score: confidence + consensus_bonus - conflict_penalty
            score = (
                result.confidence * 0.5 +
                result.consensus_score * 0.3 +
                result.quality_score * 0.2 -
                result.conflict_score * 0.2
            )

            if score > best_score:
                best_score = score
                best_result = result

        return best_result

    def _post_process_signal(self, signal: FusionSignal, input_signals: List[Dict]) -> FusionSignal:
        """Post-process and validate the fused signal."""
        # Validate confidence bounds
        signal.confidence = max(0.0, min(1.0, signal.confidence))

        # Add metadata about input signals
        signal.metadata['input_signal_count'] = len(input_signals)
        signal.metadata['input_signal_types'] = list(set(s['type'] for s in input_signals))
        signal.metadata['fusion_timestamp'] = datetime.now(timezone.utc).isoformat()

        return signal

    def update_signal_weights(self, performance_data: Dict[str, float]):
        """
        Update signal weights based on performance feedback.

        Args:
            performance_data: Dictionary of signal types to performance metrics
        """
        for signal_type, performance in performance_data.items():
            if signal_type in self.signal_weights:
                current_weight = self.signal_weights[signal_type].base_weight

                # Adjust weight based on performance (0.0 to 1.0)
                adjustment = (performance - 0.5) * self.weight_adjustment_factor
                new_weight = current_weight + adjustment

                # Ensure weight stays in bounds
                new_weight = max(
                    self.signal_weights[signal_type].min_weight,
                    min(self.signal_weights[signal_type].max_weight, new_weight)
                )

                self.signal_weights[signal_type].base_weight = new_weight

                self.logger.info(f"Updated {signal_type} weight: {current_weight:.3f} -> {new_weight:.3f}")

    def get_fusion_statistics(self) -> Dict[str, any]:
        """Get fusion engine statistics."""
        return {
            'total_fusions': self.fusion_count,
            'consensus_signals': self.consensus_count,
            'conflict_signals': self.conflict_count,
            'consensus_rate': self.consensus_count / max(1, self.fusion_count),
            'conflict_rate': self.conflict_count / max(1, self.fusion_count),
            'signal_history_length': len(self.signal_history),
            'fusion_history_length': len(self.fusion_history),
            'active_methods': [m.value for m in self.fusion_methods],
            'current_weights': {
                name: weight.base_weight
                for name, weight in self.signal_weights.items()
            }
        }

    def get_recent_performance(self, lookback_minutes: int = 30) -> Dict[str, float]:
        """Get recent fusion performance metrics."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

        recent_fusions = [
            entry for entry in self.fusion_history
            if entry.get('timestamp', datetime.min) >= cutoff_time
        ]

        if not recent_fusions:
            return {'accuracy': 0.0, 'avg_confidence': 0.0, 'fusion_count': 0}

        # Calculate metrics
        total_confidence = sum(f.get('confidence', 0) for f in recent_fusions)
        avg_confidence = total_confidence / len(recent_fusions)

        # Accuracy would need actual outcome data - simplified here
        accuracy = min(0.75, avg_confidence * 0.9)  # Placeholder

        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'fusion_count': len(recent_fusions),
            'fusions_per_minute': len(recent_fusions) / lookback_minutes
        }

    def reset(self):
        """Reset fusion engine state."""
        self.signal_history.clear()
        self.fusion_history.clear()
        self.confidence_history.clear()
        self.fusion_count = 0
        self.consensus_count = 0
        self.conflict_count = 0

    def set_fusion_parameters(self, **kwargs):
        """Update fusion engine parameters."""
        if 'consensus_threshold' in kwargs:
            self.consensus_threshold = kwargs['consensus_threshold']
        if 'conflict_penalty' in kwargs:
            self.conflict_penalty = kwargs['conflict_penalty']
        if 'quality_threshold' in kwargs:
            self.quality_threshold = kwargs['quality_threshold']
        if 'min_signals_for_fusion' in kwargs:
            self.min_signals_for_fusion = kwargs['min_signals_for_fusion']
        if 'fusion_methods' in kwargs:
            self.fusion_methods = kwargs['fusion_methods']