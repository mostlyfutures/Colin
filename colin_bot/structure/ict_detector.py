"""
ICT (Institutional Candlestick Theory) Structure Detector.

This module implements algorithmic detection of ICT concepts including
Fair Value Gaps (FVGs), Order Blocks (OBs), and Break of Structure (BOS).
These are key institutional trading concepts for liquidity analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..core.config import ConfigManager


class StructureType(Enum):
    """Types of ICT structures."""
    FAIR_VALUE_GAP = "fair_value_gap"
    ORDER_BLOCK = "order_block"
    BREAK_OF_STRUCTURE = "break_of_structure"
    MARKET_STRUCTURE_SHIFT = "market_structure_shift"


@dataclass
class ICTStructure:
    """Represents an ICT structure."""
    type: StructureType
    timestamp: datetime
    price_level: float
    confidence: float
    details: Dict[str, Any]
    is_valid: bool = True
    timeframe: str = "1h"


@dataclass
class FairValueGap(ICTStructure):
    """Fair Value Gap structure."""
    top: float
    bottom: float
    midline: float
    size: float
    fill_status: str = "unfilled"  # unfilled, partially_filled, filled


@dataclass
class OrderBlock(ICTStructure):
    """Order Block structure."""
    candle_high: float
    candle_low: float
    candle_close: float
    candle_volume: float
    side: str = "bullish"  # bullish, bearish
    is_fresh: bool = True


@dataclass
class BreakOfStructure(ICTStructure):
    """Break of Structure structure."""
    broken_level: float
    break_candle_high: float
    break_candle_low: float
    side: str = "bullish"  # bullish (higher high), bearish (lower low)
    retest_level: Optional[float] = None


class ICTDetector:
    """Detects ICT structures in price data."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize ICT detector.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.ict_config = self.config.ict

    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame,
        min_gap_size: Optional[float] = None,
        lookback_periods: Optional[int] = None
    ) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in OHLCV data.

        A Fair Value Gap occurs when three consecutive candles create a gap
        between the high of candle 0 and low of candle 2, with candle 1 being
        the strong impulsive candle.

        Args:
            df: DataFrame with OHLCV data
            min_gap_size: Minimum gap size as percentage
            lookback_periods: Number of periods to look back

        Returns:
            List of detected Fair Value Gaps
        """
        if len(df) < 3:
            return []

        min_gap_size = min_gap_size or self.ict_config['fair_value_gap']['min_gap_size']
        lookback_periods = lookback_periods or self.ict_config['fair_value_gap']['lookback_periods']

        fvg_list = []

        # Look for FVGs in recent data
        for i in range(len(df) - 2, max(len(df) - lookback_periods - 2, 0), -1):
            candle_0 = df.iloc[i]
            candle_1 = df.iloc[i + 1]  # Impulsive candle
            candle_2 = df.iloc[i + 2]

            # Bullish FVG: candle_0.high < candle_2.low
            if candle_0['high'] < candle_2['low']:
                gap_top = candle_0['high']
                gap_bottom = candle_2['low']
                gap_size = (gap_bottom - gap_top) / candle_1['close']

                if gap_size >= min_gap_size:
                    fvg = FairValueGap(
                        type=StructureType.FAIR_VALUE_GAP,
                        timestamp=candle_1.name,
                        price_level=candle_1['close'],
                        confidence=self._calculate_fvg_confidence(gap_size, candle_1),
                        details={
                            'impulsive_candle_index': i + 1,
                            'volume_ratio': candle_1['volume'] / df['volume'].mean(),
                            'price_change': (candle_1['close'] - candle_1['open']) / candle_1['open']
                        },
                        top=gap_top,
                        bottom=gap_bottom,
                        midline=(gap_top + gap_bottom) / 2,
                        size=gap_size,
                        timeframe=self._get_timeframe_from_df(df)
                    )
                    fvg_list.append(fvg)

            # Bearish FVG: candle_0.low > candle_2.high
            elif candle_0['low'] > candle_2['high']:
                gap_top = candle_2['high']
                gap_bottom = candle_0['low']
                gap_size = (gap_bottom - gap_top) / candle_1['close']

                if gap_size >= min_gap_size:
                    fvg = FairValueGap(
                        type=StructureType.FAIR_VALUE_GAP,
                        timestamp=candle_1.name,
                        price_level=candle_1['close'],
                        confidence=self._calculate_fvg_confidence(gap_size, candle_1),
                        details={
                            'impulsive_candle_index': i + 1,
                            'volume_ratio': candle_1['volume'] / df['volume'].mean(),
                            'price_change': (candle_1['close'] - candle_1['open']) / candle_1['open']
                        },
                        top=gap_top,
                        bottom=gap_bottom,
                        midline=(gap_top + gap_bottom) / 2,
                        size=gap_size,
                        timeframe=self._get_timeframe_from_df(df)
                    )
                    fvg_list.append(fvg)

        logger.debug(f"Detected {len(fvg_list)} Fair Value Gaps")
        return fvg_list

    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        min_candle_size: Optional[float] = None,
        lookback_periods: Optional[int] = None
    ) -> List[OrderBlock]:
        """
        Detect Order Blocks in OHLCV data.

        An Order Block is the last opposing candle before a strong impulsive move.
        It represents institutional order placement zones.

        Args:
            df: DataFrame with OHLCV data
            min_candle_size: Minimum candle size as percentage
            lookback_periods: Number of periods to look back

        Returns:
            List of detected Order Blocks
        """
        if len(df) < 5:
            return []

        min_candle_size = min_candle_size or self.ict_config['order_block']['min_candle_size']
        lookback_periods = lookback_periods or self.ict_config['order_block']['lookback_periods']

        ob_list = []

        # Look for Order Blocks in recent data
        for i in range(len(df) - 3, max(len(df) - lookback_periods - 3, 0), -1):
            # Check for bullish setup
            potential_ob = self._check_bullish_order_block(df, i, min_candle_size)
            if potential_ob:
                ob_list.append(potential_ob)

            # Check for bearish setup
            potential_ob = self._check_bearish_order_block(df, i, min_candle_size)
            if potential_ob:
                ob_list.append(potential_ob)

        logger.debug(f"Detected {len(ob_list)} Order Blocks")
        return ob_list

    def detect_break_of_structure(
        self,
        df: pd.DataFrame,
        lookback_periods: Optional[int] = None
    ) -> List[BreakOfStructure]:
        """
        Detect Break of Structure in OHLCV data.

        BOS occurs when price breaks a previous swing high (bullish) or
        swing low (bearish), indicating a change in market structure.

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Number of periods to look back

        Returns:
            List of detected Break of Structure points
        """
        if len(df) < 10:
            return []

        lookback_periods = lookback_periods or self.ict_config['break_of_structure']['lookback_periods']

        bos_list = []

        # Find swing highs and lows
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        # Check for bullish BOS (break of swing high)
        for i in range(min(lookback_periods, len(swing_highs) - 1)):
            if self._check_bullish_bos(df, swing_highs, i):
                bos = self._create_bullish_bos(df, swing_highs, i)
                if bos:
                    bos_list.append(bos)

        # Check for bearish BOS (break of swing low)
        for i in range(min(lookback_periods, len(swing_lows) - 1)):
            if self._check_bearish_bos(df, swing_lows, i):
                bos = self._create_bearish_bos(df, swing_lows, i)
                if bos:
                    bos_list.append(bos)

        logger.debug(f"Detected {len(bos_list)} Break of Structure points")
        return bos_list

    def analyze_ict_confluence(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Analyze ICT structure confluence around current price.

        Args:
            df: DataFrame with OHLCV data
            current_price: Current price level

        Returns:
            Dictionary with confluence analysis
        """
        # Detect all structures
        fvg_list = self.detect_fair_value_gaps(df)
        ob_list = self.detect_order_blocks(df)
        bos_list = self.detect_break_of_structure(df)

        # Analyze confluence zones
        confluence_zones = []

        # Check FVGs near current price
        nearby_fvgs = [
            fvg for fvg in fvg_list
            if fvg.bottom <= current_price <= fvg.top
        ]

        # Check Order Blocks near current price
        nearby_obs = [
            ob for ob in ob_list
            if ob.candle_low <= current_price <= ob.candle_high
        ]

        # Calculate confluence score
        confluence_score = 0
        rationale_points = []

        if nearby_fvgs:
            confluence_score += 0.4
            rationale_points.append(f"Price within {len(nearby_fvgs)} Fair Value Gap(s)")

        if nearby_obs:
            confluence_score += 0.4
            rationale_points.append(f"Price at Order Block level")

        if bos_list:
            recent_bos = [bos for bos in bos_list if (datetime.now() - bos.timestamp).total_seconds() < 3600 * 24]
            if recent_bos:
                confluence_score += 0.2
                rationale_points.append("Recent Break of Structure confirmed")

        return {
            'confluence_score': min(confluence_score, 1.0),
            'rationale_points': rationale_points,
            'nearby_structures': {
                'fair_value_gaps': len(nearby_fvgs),
                'order_blocks': len(nearby_obs),
                'break_of_structure': len(bos_list)
            },
            'total_structures': {
                'fair_value_gaps': len(fvg_list),
                'order_blocks': len(ob_list),
                'break_of_structure': len(bos_list)
            }
        }

    def get_structural_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        trade_direction: str  # "long" or "short"
    ) -> Optional[float]:
        """
        Calculate structural stop-loss level based on ICT structures.

        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price level
            trade_direction: Direction of trade ("long" or "short")

        Returns:
            Structural stop-loss price level
        """
        # Detect relevant structures
        fvg_list = self.detect_fair_value_gaps(df)
        ob_list = self.detect_order_blocks(df)

        if trade_direction.lower() == "long":
            # For longs, place stop below recent structure
            relevant_levels = []

            # Add FVG bottoms below entry
            for fvg in fvg_list:
                if fvg.bottom < entry_price:
                    relevant_levels.append(fvg.bottom)

            # Add Order Block lows below entry
            for ob in ob_list:
                if ob.candle_low < entry_price:
                    relevant_levels.append(ob.candle_low)

            # Return the highest level below entry (tightest stop)
            if relevant_levels:
                return max(relevant_levels)

        elif trade_direction.lower() == "short":
            # For shorts, place stop above recent structure
            relevant_levels = []

            # Add FVG tops above entry
            for fvg in fvg_list:
                if fvg.top > entry_price:
                    relevant_levels.append(fvg.top)

            # Add Order Block highs above entry
            for ob in ob_list:
                if ob.candle_high > entry_price:
                    relevant_levels.append(ob.candle_high)

            # Return the lowest level above entry (tightest stop)
            if relevant_levels:
                return min(relevant_levels)

        return None

    def _calculate_fvg_confidence(self, gap_size: float, impulsive_candle: pd.Series) -> float:
        """Calculate confidence score for a Fair Value Gap."""
        # Base confidence from gap size
        confidence = min(gap_size * 100, 0.7)  # Max 0.7 from size alone

        # Boost from strong impulsive candle
        candle_body_ratio = abs(impulsive_candle['close'] - impulsive_candle['open']) / impulsive_candle['open']
        confidence += min(candle_body_ratio * 50, 0.2)

        # Boost from high volume
        volume_bonus = min(impulsive_candle['volume'] / 1000000, 0.1)  # Simple volume bonus
        confidence += volume_bonus

        return min(confidence, 1.0)

    def _check_bullish_order_block(self, df: pd.DataFrame, i: int, min_candle_size: float) -> Optional[OrderBlock]:
        """Check for bullish Order Block at index i."""
        # Look for strong down candle followed by strong up move
        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]
        next_next_candle = df.iloc[i + 2]

        # Current candle should be bearish (down)
        if candle['close'] >= candle['open']:
            return None

        # Next candles should show strong bullish rejection
        if not (next_candle['close'] > next_candle['open'] and
                next_next_candle['close'] > next_next_candle['open']):
            return None

        # Check if current candle is significant
        candle_size = abs(candle['close'] - candle['open']) / candle['open']
        if candle_size < min_candle_size:
            return None

        # Calculate confidence
        confidence = min(candle_size * 100, 0.6)
        confidence += min(next_candle['volume'] / df['volume'].mean(), 0.2)
        confidence += min(next_next_candle['volume'] / df['volume'].mean(), 0.2)

        return OrderBlock(
            type=StructureType.ORDER_BLOCK,
            timestamp=candle.name,
            price_level=candle['close'],
            confidence=min(confidence, 1.0),
            details={
                'candle_index': i,
                'candle_size': candle_size,
                'next_candle_size': abs(next_candle['close'] - next_candle['open']) / next_candle['open'],
                'volume_ratio': candle['volume'] / df['volume'].mean()
            },
            candle_high=candle['high'],
            candle_low=candle['low'],
            candle_close=candle['close'],
            candle_volume=candle['volume'],
            side="bullish",
            is_fresh=True,
            timeframe=self._get_timeframe_from_df(df)
        )

    def _check_bearish_order_block(self, df: pd.DataFrame, i: int, min_candle_size: float) -> Optional[OrderBlock]:
        """Check for bearish Order Block at index i."""
        # Look for strong up candle followed by strong down move
        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]
        next_next_candle = df.iloc[i + 2]

        # Current candle should be bullish (up)
        if candle['close'] <= candle['open']:
            return None

        # Next candles should show strong bearish rejection
        if not (next_candle['close'] < next_candle['open'] and
                next_next_candle['close'] < next_next_candle['open']):
            return None

        # Check if current candle is significant
        candle_size = abs(candle['close'] - candle['open']) / candle['open']
        if candle_size < min_candle_size:
            return None

        # Calculate confidence
        confidence = min(candle_size * 100, 0.6)
        confidence += min(next_candle['volume'] / df['volume'].mean(), 0.2)
        confidence += min(next_next_candle['volume'] / df['volume'].mean(), 0.2)

        return OrderBlock(
            type=StructureType.ORDER_BLOCK,
            timestamp=candle.name,
            price_level=candle['close'],
            confidence=min(confidence, 1.0),
            details={
                'candle_index': i,
                'candle_size': candle_size,
                'next_candle_size': abs(next_candle['close'] - next_candle['open']) / next_candle['open'],
                'volume_ratio': candle['volume'] / df['volume'].mean()
            },
            candle_high=candle['high'],
            candle_low=candle['low'],
            candle_close=candle['close'],
            candle_volume=candle['volume'],
            side="bearish",
            is_fresh=True,
            timeframe=self._get_timeframe_from_df(df)
        )

    def _find_swing_highs(self, df: pd.DataFrame, window: int = 3) -> List[Tuple[datetime, float]]:
        """Find swing high points in price data."""
        swing_highs = []
        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            is_swing_high = True

            # Check if this is the highest in the window
            for j in range(i - window, i + window + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append((df.iloc[i].name, current_high))

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame, window: int = 3) -> List[Tuple[datetime, float]]:
        """Find swing low points in price data."""
        swing_lows = []
        for i in range(window, len(df) - window):
            current_low = df.iloc[i]['low']
            is_swing_low = True

            # Check if this is the lowest in the window
            for j in range(i - window, i + window + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append((df.iloc[i].name, current_low))

        return swing_lows

    def _check_bullish_bos(self, df: pd.DataFrame, swing_highs: List[Tuple[datetime, float]], i: int) -> bool:
        """Check for bullish Break of Structure."""
        if i + 1 >= len(swing_highs):
            return False

        previous_swing_high = swing_highs[i][1]
        current_price = df.iloc[-1]['high']

        return current_price > previous_swing_high

    def _check_bearish_bos(self, df: pd.DataFrame, swing_lows: List[Tuple[datetime, float]], i: int) -> bool:
        """Check for bearish Break of Structure."""
        if i + 1 >= len(swing_lows):
            return False

        previous_swing_low = swing_lows[i][1]
        current_price = df.iloc[-1]['low']

        return current_price < previous_swing_low

    def _create_bullish_bos(self, df: pd.DataFrame, swing_highs: List[Tuple[datetime, float]], i: int) -> Optional[BreakOfStructure]:
        """Create bullish Break of Structure object."""
        if i + 1 >= len(swing_highs):
            return None

        broken_level = swing_highs[i][1]
        break_time = swing_highs[i][0]

        # Find the break candle
        break_candle_idx = df.index.get_loc(break_time) + 1
        if break_candle_idx >= len(df):
            return None

        break_candle = df.iloc[break_candle_idx]

        return BreakOfStructure(
            type=StructureType.BREAK_OF_STRUCTURE,
            timestamp=break_candle.name,
            price_level=break_candle['close'],
            confidence=0.7,  # BOS is generally high confidence
            details={
                'broken_level': broken_level,
                'break_strength': (break_candle['high'] - broken_level) / broken_level
            },
            broken_level=broken_level,
            break_candle_high=break_candle['high'],
            break_candle_low=break_candle['low'],
            side="bullish",
            retest_level=broken_level,  # Potential retest level
            timeframe=self._get_timeframe_from_df(df)
        )

    def _create_bearish_bos(self, df: pd.DataFrame, swing_lows: List[Tuple[datetime, float]], i: int) -> Optional[BreakOfStructure]:
        """Create bearish Break of Structure object."""
        if i + 1 >= len(swing_lows):
            return None

        broken_level = swing_lows[i][1]
        break_time = swing_lows[i][0]

        # Find the break candle
        break_candle_idx = df.index.get_loc(break_time) + 1
        if break_candle_idx >= len(df):
            return None

        break_candle = df.iloc[break_candle_idx]

        return BreakOfStructure(
            type=StructureType.BREAK_OF_STRUCTURE,
            timestamp=break_candle.name,
            price_level=break_candle['close'],
            confidence=0.7,  # BOS is generally high confidence
            details={
                'broken_level': broken_level,
                'break_strength': (broken_level - break_candle['low']) / broken_level
            },
            broken_level=broken_level,
            break_candle_high=break_candle['high'],
            break_candle_low=break_candle['low'],
            side="bearish",
            retest_level=broken_level,  # Potential retest level
            timeframe=self._get_timeframe_from_df(df)
        )

    def _get_timeframe_from_df(self, df: pd.DataFrame) -> str:
        """Estimate timeframe from DataFrame index frequency."""
        if len(df) < 2:
            return "1h"

        time_diff = df.index[1] - df.index[0]
        total_minutes = time_diff.total_seconds() / 60

        if total_minutes <= 5:
            return "5m"
        elif total_minutes <= 15:
            return "15m"
        elif total_minutes <= 60:
            return "1h"
        elif total_minutes <= 240:
            return "4h"
        else:
            return "1d"