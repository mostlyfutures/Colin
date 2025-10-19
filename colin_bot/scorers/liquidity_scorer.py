"""
Liquidity Scorer for institutional signal analysis.

This module scores liquidity-based signals including proximity to liquidation
clusters, stop-hunt zones, and smart money target areas.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from ..core.config import ConfigManager


@dataclass
class LiquiditySignal:
    """Represents a liquidity-based signal."""
    signal_type: str
    strength: float
    direction: str  # "long" or "short"
    proximity_score: float
    density_score: float
    rationale: str
    target_price: float
    stop_loss_price: Optional[float] = None


@dataclass
class LiquidityScore:
    """Comprehensive liquidity scoring result."""
    overall_score: float
    long_bias: float
    short_bias: float
    signals: List[LiquiditySignal]
    nearest_liquidation_cluster: Optional[Dict[str, Any]]
    liquidity_gap_analysis: Dict[str, Any]
    timestamp: datetime


class LiquidityScorer:
    """Scores liquidity-based institutional signals."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize liquidity scorer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.liquidation_config = self.config.liquidations
        self.proximity_threshold = self.liquidation_config['proximity_threshold']
        self.min_density_threshold = self.liquidation_config['heatmap']['min_density_threshold']

    def score_liquidation_proximity(
        self,
        current_price: float,
        liquidation_data: Dict[str, Any],
        order_book_data: Dict[str, Any]
    ) -> LiquidityScore:
        """
        Score signals based on proximity to liquidation clusters.

        Args:
            current_price: Current price level
            liquidation_data: Liquidation heatmap and cluster data
            order_book_data: Order book analysis data

        Returns:
            Comprehensive liquidity score
        """
        try:
            signals = []
            long_bias = 0.0
            short_bias = 0.0

            # Analyze liquidation density clusters
            density_clusters = liquidation_data.get('density_clusters', [])

            # Find nearest and most significant clusters
            nearest_clusters = self._find_nearest_clusters(current_price, density_clusters)

            for cluster in nearest_clusters:
                signal = self._create_liquidation_signal(current_price, cluster, order_book_data)
                if signal:
                    signals.append(signal)

                    # Update bias scores
                    if signal.direction == "long":
                        long_bias += signal.strength * signal.proximity_score
                    else:
                        short_bias += signal.strength * signal.proximity_score

            # Analyze liquidity gaps
            liquidity_gaps = self._analyze_liquidity_gaps(current_price, order_book_data)

            # Check for stop-hunt setups
            stop_hunt_signals = self._identify_stop_hunt_setups(
                current_price, liquidation_data, order_book_data
            )
            signals.extend(stop_hunt_signals)

            # Calculate overall scores
            overall_score = self._calculate_overall_liquidity_score(signals)

            # Normalize bias scores
            total_bias = long_bias + short_bias
            if total_bias > 0:
                long_bias = long_bias / total_bias
                short_bias = short_bias / total_bias

            # Find nearest cluster for reference
            nearest_cluster = nearest_clusters[0] if nearest_clusters else None

            result = LiquidityScore(
                overall_score=min(overall_score, 1.0),
                long_bias=long_bias,
                short_bias=short_bias,
                signals=signals,
                nearest_liquidation_cluster=nearest_cluster,
                liquidity_gap_analysis=liquidity_gaps,
                timestamp=datetime.now()
            )

            logger.debug(f"Liquidity scoring complete: Overall={overall_score:.3f}, Long={long_bias:.3f}, Short={short_bias:.3f}")
            return result

        except Exception as e:
            logger.error(f"Failed to score liquidity proximity: {e}")
            return self._empty_liquidity_score()

    def score_liquidity_grab_potential(
        self,
        current_price: float,
        liquidation_data: Dict[str, Any],
        ict_structures: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score potential for liquidity grab scenarios.

        Args:
            current_price: Current price level
            liquidation_data: Liquidation data
            ict_structures: ICT structure analysis

        Returns:
            Liquidity grab potential analysis
        """
        try:
            grab_potential = {
                'bullish_grab_score': 0.0,
                'bearish_grab_score': 0.0,
                'target_zones': [],
                'rationale': []
            }

            # Identify untested liquidity zones
            untested_zones = self._find_untested_liquidity_zones(
                current_price, liquidation_data, ict_structures
            )

            for zone in untested_zones:
                # Calculate grab potential
                if zone['type'] == 'long_liquidation':
                    grab_score = self._calculate_bullish_grab_score(zone, current_price, ict_structures)
                    if grab_score > 0.3:
                        grab_potential['bullish_grab_score'] = max(
                            grab_potential['bullish_grab_score'], grab_score
                        )
                        grab_potential['target_zones'].append({
                            'type': 'bullish_liquidity_grab',
                            'target_price': zone['price_level'],
                            'potential_score': grab_score,
                            'liquidity_size': zone['density_score']
                        })
                        grab_potential['rationale'].append(
                            f"Untested long liquidation cluster at ${zone['price_level']:.2f}"
                        )

                elif zone['type'] == 'short_liquidation':
                    grab_score = self._calculate_bearish_grab_score(zone, current_price, ict_structures)
                    if grab_score > 0.3:
                        grab_potential['bearish_grab_score'] = max(
                            grab_potential['bearish_grab_score'], grab_score
                        )
                        grab_potential['target_zones'].append({
                            'type': 'bearish_liquidity_grab',
                            'target_price': zone['price_level'],
                            'potential_score': grab_score,
                            'liquidity_size': zone['density_score']
                        })
                        grab_potential['rationale'].append(
                            f"Untested short liquidation cluster at ${zone['price_level']:.2f}"
                        )

            return grab_potential

        except Exception as e:
            logger.error(f"Failed to score liquidity grab potential: {e}")
            return {
                'bullish_grab_score': 0.0,
                'bearish_grab_score': 0.0,
                'target_zones': [],
                'rationale': []
            }

    def _find_nearest_clusters(
        self,
        current_price: float,
        density_clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find nearest liquidation clusters to current price."""
        if not density_clusters:
            return []

        # Sort clusters by distance from current price
        clusters_with_distance = []
        for cluster in density_clusters:
            distance = abs(cluster['price_level'] - current_price) / current_price
            clusters_with_distance.append({
                **cluster,
                'distance': distance
            })

        # Filter by proximity threshold and sort
        nearby_clusters = [
            c for c in clusters_with_distance
            if c['distance'] <= self.proximity_threshold
        ]

        # Sort by distance then by density
        nearby_clusters.sort(key=lambda x: (x['distance'], -x['density_score']))

        return nearby_clusters[:5]  # Return top 5 nearest clusters

    def _create_liquidation_signal(
        self,
        current_price: float,
        cluster: Dict[str, Any],
        order_book_data: Dict[str, Any]
    ) -> Optional[LiquiditySignal]:
        """Create a liquidity signal from a cluster."""

        # Skip clusters that are too far away
        if cluster['distance'] > self.proximity_threshold:
            return None

        # Calculate proximity score (closer = higher score)
        proximity_score = 1.0 - (cluster['distance'] / self.proximity_threshold)

        # Calculate density score based on cluster size
        density_score = min(cluster['density_score'] / self.min_density_threshold, 1.0)

        # Determine signal direction
        if cluster['side'] == 'long':
            # Long liquidations below price - bullish signal (stop hunt)
            if cluster['price_level'] < current_price:
                direction = "short"  # Expect price to drop to hunt longs
                signal_type = "long_liquidation_hunt"
                rationale = f"Long liquidation cluster ${cluster['price_level']:.2f} below current price"
            else:
                # Long liquidations above price - bearish signal
                direction = "long"
                signal_type = "long_liquidation_resistance"
                rationale = f"Long liquidation cluster ${cluster['price_level']:.2f} above current price"
        else:  # short liquidations
            # Short liquidations above price - bearish signal (stop hunt)
            if cluster['price_level'] > current_price:
                direction = "long"  # Expect price to rise to hunt shorts
                signal_type = "short_liquidation_hunt"
                rationale = f"Short liquidation cluster ${cluster['price_level']:.2f} above current price"
            else:
                # Short liquidations below price - bullish signal
                direction = "short"
                signal_type = "short_liquidation_support"
                rationale = f"Short liquidation cluster ${cluster['price_level']:.2f} below current price"

        # Calculate overall signal strength
        strength = (proximity_score + density_score) / 2

        # Adjust strength based on order book imbalance
        order_flow_bias = order_book_data.get('normalized_order_book_imbalance', 0)
        if (direction == "long" and order_flow_bias > 0.1) or \
           (direction == "short" and order_flow_bias < -0.1):
            strength *= 1.2  # Boost if order flow aligns

        return LiquiditySignal(
            signal_type=signal_type,
            strength=min(strength, 1.0),
            direction=direction,
            proximity_score=proximity_score,
            density_score=density_score,
            rationale=rationale,
            target_price=cluster['price_level'],
            stop_loss_price=self._calculate_liquidity_stop_loss(
                current_price, cluster['price_level'], direction
            )
        )

    def _analyze_liquidity_gaps(
        self,
        current_price: float,
        order_book_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze liquidity gaps in the order book."""

        try:
            # Calculate liquidity gaps
            bid_liquidity = order_book_data.get('bid_liquidity', 0)
            ask_liquidity = order_book_data.get('ask_liquidity', 0)
            total_liquidity = bid_liquidity + ask_liquidity

            if total_liquidity == 0:
                return {
                    'has_liquidity_gap': False,
                    'gap_size': 0,
                    'gap_direction': None,
                    'significance': 0
                }

            # Identify significant imbalance
            liquidity_ratio = bid_liquidity / ask_liquidity if ask_liquidity > 0 else float('inf')

            # Determine if there's a significant liquidity gap
            gap_threshold = 2.0  # 2:1 imbalance
            has_gap = liquidity_ratio > gap_threshold or liquidity_ratio < 1/gap_threshold

            if has_gap:
                if liquidity_ratio > gap_threshold:
                    gap_direction = "bearish"  # More bid liquidity, potential downward move
                    gap_size = (liquidity_ratio - gap_threshold) / gap_threshold
                else:
                    gap_direction = "bullish"  # More ask liquidity, potential upward move
                    gap_size = (1/liquidity_ratio - gap_threshold) / gap_threshold
            else:
                gap_direction = None
                gap_size = 0

            return {
                'has_liquidity_gap': has_gap,
                'gap_size': min(gap_size, 1.0),
                'gap_direction': gap_direction,
                'significance': min(gap_size * 0.7, 1.0),  # Liquidity gaps are significant
                'liquidity_ratio': liquidity_ratio
            }

        except Exception as e:
            logger.error(f"Failed to analyze liquidity gaps: {e}")
            return {
                'has_liquidity_gap': False,
                'gap_size': 0,
                'gap_direction': None,
                'significance': 0
            }

    def _identify_stop_hunt_setups(
        self,
        current_price: float,
        liquidation_data: Dict[str, Any],
        order_book_data: Dict[str, Any]
    ) -> List[LiquiditySignal]:
        """Identify potential stop-hunt setups."""

        stop_hunt_signals = []

        try:
            # Look for thin liquidity zones near liquidation clusters
            density_clusters = liquidation_data.get('density_clusters', [])

            for cluster in density_clusters:
                # Check if cluster is within striking distance
                distance = abs(cluster['price_level'] - current_price) / current_price

                if distance <= self.proximity_threshold * 1.5:  # Slightly wider range for stops
                    # Check if order book shows thin liquidity at cluster level
                    order_flow = order_book_data.get('normalized_order_book_imbalance', 0)

                    # Stop hunt likely when:
                    # 1. Large liquidation cluster nearby
                    # 2. Order book shows imbalance favoring the hunt direction
                    # 3. Current price is moving towards cluster

                    if cluster['density_score'] > self.min_density_threshold:
                        signal_strength = min(cluster['density_score'] / (self.min_density_threshold * 2), 1.0)

                        # Determine hunt direction
                        if cluster['side'] == 'long' and cluster['price_level'] < current_price:
                            # Hunt longs below
                            direction = "short"
                            signal_type = "bullish_stop_hunt"
                            rationale = f"Thin liquidity above long liquidation cluster at ${cluster['price_level']:.2f}"
                        elif cluster['side'] == 'short' and cluster['price_level'] > current_price:
                            # Hunt shorts above
                            direction = "long"
                            signal_type = "bearish_stop_hunt"
                            rationale = f"Thin liquidity below short liquidation cluster at ${cluster['price_level']:.2f}"
                        else:
                            continue

                        stop_hunt_signals.append(LiquiditySignal(
                            signal_type=signal_type,
                            strength=signal_strength * 0.8,  # Slightly lower confidence
                            direction=direction,
                            proximity_score=1.0 - (distance / (self.proximity_threshold * 1.5)),
                            density_score=min(cluster['density_score'] / self.min_density_threshold, 1.0),
                            rationale=rationale,
                            target_price=cluster['price_level'],
                            stop_loss_price=self._calculate_liquidity_stop_loss(
                                current_price, cluster['price_level'], direction
                            )
                        ))

            return stop_hunt_signals

        except Exception as e:
            logger.error(f"Failed to identify stop-hunt setups: {e}")
            return []

    def _find_untested_liquidity_zones(
        self,
        current_price: float,
        liquidation_data: Dict[str, Any],
        ict_structures: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find untested liquidity zones for potential grabs."""

        untested_zones = []

        try:
            # Get recent price action to determine tested levels
            recent_highs = ict_structures.get('recent_highs', [])
            recent_lows = ict_structures.get('recent_lows', [])

            # Get liquidation clusters
            density_clusters = liquidation_data.get('density_clusters', [])

            for cluster in density_clusters:
                # Check if cluster level has been tested recently
                cluster_price = cluster['price_level']

                # Simple check: if cluster is far from recent highs/lows, it's untested
                min_distance_to_recent = float('inf')

                for high in recent_highs:
                    distance = abs(cluster_price - high) / high
                    min_distance_to_recent = min(min_distance_to_recent, distance)

                for low in recent_lows:
                    distance = abs(cluster_price - low) / low
                    min_distance_to_recent = min(min_distance_to_recent, distance)

                # If no recent price action within 0.5%, consider untested
                if min_distance_to_recent > 0.005:
                    untested_zones.append({
                        'price_level': cluster_price,
                        'type': f"{cluster['side']}_liquidation",
                        'density_score': cluster['density_score'],
                        'untested_factor': min_distance_to_recent,
                        'distance_from_current': abs(cluster_price - current_price) / current_price
                    })

            # Sort by untested factor and density
            untested_zones.sort(key=lambda x: (x['untested_factor'], -x['density_score']))

            return untested_zones[:5]  # Return top 5 untested zones

        except Exception as e:
            logger.error(f"Failed to find untested liquidity zones: {e}")
            return []

    def _calculate_bullish_grab_score(
        self,
        zone: Dict[str, Any],
        current_price: float,
        ict_structures: Dict[str, Any]
    ) -> float:
        """Calculate bullish liquidity grab score."""

        score = 0.0

        # Base score from liquidity size
        score += min(zone['density_score'] / (self.min_density_threshold * 2), 0.5)

        # Boost if zone is untested
        score += min(zone['untested_factor'] * 20, 0.3)

        # Boost if current price is above zone (downward grab potential)
        if current_price > zone['price_level']:
            score += 0.2

        return min(score, 1.0)

    def _calculate_bearish_grab_score(
        self,
        zone: Dict[str, Any],
        current_price: float,
        ict_structures: Dict[str, Any]
    ) -> float:
        """Calculate bearish liquidity grab score."""

        score = 0.0

        # Base score from liquidity size
        score += min(zone['density_score'] / (self.min_density_threshold * 2), 0.5)

        # Boost if zone is untested
        score += min(zone['untested_factor'] * 20, 0.3)

        # Boost if current price is below zone (upward grab potential)
        if current_price < zone['price_level']:
            score += 0.2

        return min(score, 1.0)

    def _calculate_liquidity_stop_loss(
        self,
        current_price: float,
        target_price: float,
        direction: str
    ) -> float:
        """Calculate stop loss level for liquidity-based trade."""

        # Place stop beyond the liquidity target
        buffer_percentage = 0.002  # 0.2% buffer

        if direction == "long":
            # Going long, stop loss below target
            return target_price * (1 - buffer_percentage)
        else:
            # Going short, stop loss above target
            return target_price * (1 + buffer_percentage)

    def _calculate_overall_liquidity_score(self, signals: List[LiquiditySignal]) -> float:
        """Calculate overall liquidity score from all signals."""

        if not signals:
            return 0.0

        # Weight signals by strength and proximity
        weighted_scores = []
        for signal in signals:
            weighted_score = signal.strength * signal.proximity_score
            weighted_scores.append(weighted_score)

        # Return maximum weighted score (strongest signal)
        return max(weighted_scores) if weighted_scores else 0.0

    def _empty_liquidity_score(self) -> LiquidityScore:
        """Return empty liquidity score when no data available."""
        return LiquidityScore(
            overall_score=0.0,
            long_bias=0.0,
            short_bias=0.0,
            signals=[],
            nearest_liquidation_cluster=None,
            liquidity_gap_analysis={},
            timestamp=datetime.now()
        )