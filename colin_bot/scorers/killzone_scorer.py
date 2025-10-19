"""
Killzone Scorer for institutional timing analysis.

This module scores signals based on trading session timing, institutional
activity patterns, and optimal entry windows.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from ..core.config import ConfigManager
from ..utils.sessions import SessionAnalyzer, SessionType


@dataclass
class KillzoneSignal:
    """Represents a killzone-based timing signal."""
    session_type: SessionType
    strength: float
    timing_score: float
    liquidity_expectation: str
    volatility_expectation: str
    rationale: str
    optimal_entry: bool
    time_to_session_end: Optional[timedelta] = None


@dataclass
class KillzoneScore:
    """Comprehensive killzone scoring result."""
    overall_score: float
    session_multiplier: float
    timing_bonus: float
    signals: List[KillzoneSignal]
    session_analysis: Dict[str, Any]
    optimal_windows: List[Dict[str, Any]]
    timestamp: datetime


class KillzoneScorer:
    """Scores killzone-based institutional timing signals."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize killzone scorer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.session_analyzer = SessionAnalyzer(config_manager)

    def score_killzone_timing(
        self,
        current_time: Optional[datetime] = None
    ) -> KillzoneScore:
        """
        Score signals based on current session timing.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Comprehensive killzone score
        """
        try:
            if current_time is None:
                current_time = datetime.now()

            signals = []

            # Get session analysis
            session_analysis = self.session_analyzer.get_killzone_analysis(current_time)

            # Analyze current session
            current_session_score = self._analyze_current_session(session_analysis)
            if current_session_score:
                signals.append(current_session_score)

            # Analyze upcoming sessions
            upcoming_signals = self._analyze_upcoming_sessions(current_time)
            signals.extend(upcoming_signals)

            # Calculate session multiplier
            session_multiplier = session_analysis.get('session_score', 0.1)

            # Calculate timing bonus
            timing_bonus = self._calculate_timing_bonus(session_analysis, current_time)

            # Identify optimal windows
            optimal_windows = self._identify_optimal_windows(current_time)

            # Calculate overall score
            overall_score = self._calculate_overall_killzone_score(
                session_multiplier, timing_bonus, signals
            )

            result = KillzoneScore(
                overall_score=min(overall_score, 1.0),
                session_multiplier=session_multiplier,
                timing_bonus=timing_bonus,
                signals=signals,
                session_analysis=session_analysis,
                optimal_windows=optimal_windows,
                timestamp=current_time
            )

            logger.debug(f"Killzone scoring complete: Overall={overall_score:.3f}, Session={session_multiplier:.3f}")
            return result

        except Exception as e:
            logger.error(f"Failed to score killzone timing: {e}")
            return self._empty_killzone_score()

    def score_institutional_alignment(
        self,
        current_time: Optional[datetime] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Score alignment with institutional activity patterns.

        Args:
            current_time: Current time
            market_data: Market data for validation

        Returns:
            Institutional alignment analysis
        """
        try:
            if current_time is None:
                current_time = datetime.now()

            alignment_analysis = {
                'institutional_activity_score': 0.0,
                'smart_money_probability': 0.0,
                'alignment_factors': [],
                'rationale': []
            }

            # Get session status
            session_status = self.session_analyzer.get_session_status(current_time)

            # Base score from active sessions
            if session_status.is_active:
                active_weights = session_status.session_weights.values()
                alignment_analysis['institutional_activity_score'] = max(active_weights) if active_weights else 0.1

                # Boost for multiple active sessions (overlap)
                if len(session_status.active_sessions) > 1:
                    alignment_analysis['institutional_activity_score'] *= 1.2
                    alignment_analysis['alignment_factors'].append("Multiple session overlap")

                # Analyze specific session characteristics
                for session_type in session_status.active_sessions:
                    session_alignment = self._analyze_session_characteristics(session_type, current_time)
                    alignment_analysis['alignment_factors'].extend(session_alignment['factors'])
                    alignment_analysis['rationale'].extend(session_alignment['rationale'])

            # Time-based factors
            time_factors = self._analyze_time_characteristics(current_time)
            alignment_analysis['alignment_factors'].extend(time_factors['factors'])
            alignment_analysis['rationale'].extend(time_factors['rationale'])

            # Calculate smart money probability
            alignment_analysis['smart_money_probability'] = self._calculate_smart_money_probability(
                session_status, alignment_analysis['alignment_factors']
            )

            # Normalize scores
            alignment_analysis['institutional_activity_score'] = min(
                alignment_analysis['institutional_activity_score'], 1.0
            )
            alignment_analysis['smart_money_probability'] = min(
                alignment_analysis['smart_money_probability'], 1.0
            )

            return alignment_analysis

        except Exception as e:
            logger.error(f"Failed to score institutional alignment: {e}")
            return {
                'institutional_activity_score': 0.0,
                'smart_money_probability': 0.0,
                'alignment_factors': [],
                'rationale': []
            }

    def _analyze_current_session(self, session_analysis: Dict[str, Any]) -> Optional[KillzoneSignal]:
        """Analyze the current active session."""

        if not session_analysis.get('is_institutional_hours'):
            return None

        session_name = session_analysis.get('primary_session')
        if not session_name:
            return None

        # Map session name to enum
        session_mapping = {
            'asian': SessionType.ASIAN,
            'london': SessionType.LONDON,
            'new_york': SessionType.NEW_YORK,
            'london_ny_overlap': SessionType.LONDON_NY_OVERLAP
        }

        session_type = session_mapping.get(session_name)
        if not session_type:
            return None

        # Calculate timing score
        session_score = session_analysis.get('session_score', 0.1)
        timing_score = session_score

        # Check if optimal entry time
        optimal_entry = self.session_analyzer.is_optimal_entry_time()

        # Calculate strength based on session type and timing
        base_strength = session_score
        if optimal_entry:
            base_strength *= 1.3

        strength = min(base_strength, 1.0)

        return KillzoneSignal(
            session_type=session_type,
            strength=strength,
            timing_score=timing_score,
            liquidity_expectation=session_analysis.get('liquidity_expectation', 'Medium'),
            volatility_expectation=session_analysis.get('volatility_expectation', 'Medium'),
            rationale=f"Active {session_name.replace('_', ' ').title()} session with {strength:.1%} strength",
            optimal_entry=optimal_entry,
            time_to_session_end=self._parse_time_delta(session_analysis.get('time_to_current_end'))
        )

    def _analyze_upcoming_sessions(self, current_time: datetime) -> List[KillzoneSignal]:
        """Analyze upcoming sessions for preparation."""

        signals = []
        session_status = self.session_analyzer.get_session_status(current_time)

        if session_status.time_to_next:
            # Look ahead to next session
            time_to_next = session_status.time_to_next

            # If next session starts within 30 minutes, prepare for it
            if time_to_next.total_seconds() < 1800:  # 30 minutes
                session_schedule = self.session_analyzer.get_session_schedule(current_time)

                for session in session_schedule:
                    if session.get('is_active'):
                        continue

                    session_start = datetime.fromisoformat(session['session'])
                    time_until_start = session_start - current_time

                    if 0 < time_until_start.total_seconds() < 3600:  # Within 1 hour
                        session_name = session['session']
                        weight = session['weight']

                        # Map session name to enum
                        session_mapping = {
                            'asian': SessionType.ASIAN,
                            'london': SessionType.LONDON,
                            'new_york': SessionType.NEW_YORK,
                            'london_ny_overlap': SessionType.LONDON_NY_OVERLAP
                        }

                        session_type = session_mapping.get(session_name)
                        if session_type:
                            # Preparation signal (lower strength)
                            prep_strength = weight * 0.5  # Preparation is half strength

                            signals.append(KillzoneSignal(
                                session_type=session_type,
                                strength=prep_strength,
                                timing_score=0.5,  # Neutral timing for preparation
                                liquidity_expectation="Unknown",
                                volatility_expectation="Unknown",
                                rationale=f"Upcoming {session_name.replace('_', ' ').title()} session preparation",
                                optimal_entry=False,
                                time_to_session_end=time_until_start
                            ))

        return signals

    def _calculate_timing_bonus(
        self,
        session_analysis: Dict[str, Any],
        current_time: datetime
    ) -> float:
        """Calculate timing bonus based on specific time characteristics."""

        timing_bonus = 0.0

        try:
            # Session-specific timing bonuses
            primary_session = session_analysis.get('primary_session')

            if primary_session == 'london_ny_overlap':
                # Peak institutional hours
                timing_bonus = 0.3
            elif primary_session in ['london', 'new_york']:
                # First 2 hours get bonus
                current_hour = current_time.hour
                if (primary_session == 'london' and 7 <= current_hour < 9) or \
                   (primary_session == 'new_york' and 12 <= current_hour < 14):
                    timing_bonus = 0.2
                else:
                    timing_bonus = 0.1
            elif primary_session == 'asian':
                # Asian session gets smaller bonus
                timing_bonus = 0.05

            # Day of week adjustment
            weekday = current_time.weekday()
            if weekday == 0:  # Monday
                timing_bonus *= 0.9  # Lower activity
            elif weekday == 4:  # Friday
                timing_bonus *= 0.8  # Lower activity

            # Market open/close timing
            hour = current_time.hour
            if hour in [0, 1, 22, 23]:  # Low activity hours
                timing_bonus *= 0.5

        except Exception as e:
            logger.error(f"Failed to calculate timing bonus: {e}")

        return max(timing_bonus, 0.0)

    def _identify_optimal_windows(self, current_time: datetime) -> List[Dict[str, Any]]:
        """Identify optimal trading windows for the next 24 hours."""

        optimal_windows = []

        try:
            # Get session schedule for today and tomorrow
            session_schedule = self.session_analyzer.get_session_schedule(current_time)

            for session in session_schedule:
                if not session.get('weight', 0) >= 1.0:  # Only consider weighted sessions
                    continue

                session_start = datetime.fromisoformat(session['session'])
                session_end = datetime.fromisoformat(session['end'])

                # Check if session is in the next 24 hours
                if 0 < (session_start - current_time).total_seconds() < 86400:
                    # Calculate optimal entry window (first 2 hours of session)
                    optimal_start = session_start
                    optimal_end = min(session_start + timedelta(hours=2), session_end)

                    optimal_windows.append({
                        'session_type': session['session'],
                        'start_time': optimal_start.isoformat(),
                        'end_time': optimal_end.isoformat(),
                        'weight': session['weight'],
                        'description': session['description'],
                        'expected_liquidity': self._estimate_session_liquidity(session['session']),
                        'expected_volatility': self._estimate_session_volatility(session['session']),
                        'time_until_start': (session_start - current_time).total_seconds()
                    })

            # Sort by time until start
            optimal_windows.sort(key=lambda x: x['time_until_start'])

        except Exception as e:
            logger.error(f"Failed to identify optimal windows: {e}")

        return optimal_windows

    def _analyze_session_characteristics(
        self,
        session_type: SessionType,
        current_time: datetime
    ) -> Dict[str, List[str]]:
        """Analyze specific session characteristics for institutional alignment."""

        analysis = {
            'factors': [],
            'rationale': []
        }

        try:
            if session_type == SessionType.LONDON_NY_OVERLAP:
                analysis['factors'].extend([
                    "peak_liquidity",
                    "high_institutional_participation",
                    "optimal_volatility"
                ])
                analysis['rationale'].extend([
                    "London/NY overlap provides highest liquidity",
                    "Both European and US institutions active",
                    "Optimal conditions for large orders"
                ])

            elif session_type == SessionType.LONDON:
                analysis['factors'].extend([
                    "european_institutional_flow",
                    "moderate_to_high_liquidity",
                    "market_structure_definition"
                ])
                analysis['rationale'].extend([
                    "European institutional desks active",
                    "Good liquidity for medium-sized orders",
                    "Tends to define daily market structure"
                ])

            elif session_type == SessionType.NEW_YORK:
                analysis['factors'].extend([
                    "us_institutional_flow",
                    "high_volatility_potential",
                    "trend_continuation"
                ])
                analysis['rationale'].extend([
                    "US institutional desks active",
                    "Higher volatility potential",
                    "Often continues London session trends"
                ])

            elif session_type == SessionType.ASIAN:
                analysis['factors'].extend([
                    "asian_institutional_flow",
                    "lower_liquidity",
                    "range_bound_tendencies"
                ])
                analysis['rationale'].extend([
                    "Asian institutional participation",
                    "Lower liquidity than Western sessions",
                    "Tends to be more range-bound"
                ])

        except Exception as e:
            logger.error(f"Failed to analyze session characteristics: {e}")

        return analysis

    def _analyze_time_characteristics(self, current_time: datetime) -> Dict[str, List[str]]:
        """Analyze time-based characteristics."""

        analysis = {
            'factors': [],
            'rationale': []
        }

        try:
            hour = current_time.hour
            weekday = current_time.weekday()

            # Hour-based characteristics
            if 12 <= hour < 16:  # NY session hours
                analysis['factors'].append("us_market_hours")
                analysis['rationale'].append("US market hours - higher volatility expected")
            elif 7 <= hour < 16:  # London session hours
                analysis['factors'].append("european_market_hours")
                analysis['rationale'].append("European market hours - good liquidity")

            # Day-based characteristics
            if weekday == 0:  # Monday
                analysis['factors'].append("weekend_gap_risk")
                analysis['rationale'].append("Monday - potential weekend gap effects")
            elif weekday == 4:  # Friday
                analysis['factors'].append("weekend_positioning")
                analysis['rationale'].append("Friday - weekend positioning may affect prices")

            # Month-based characteristics
            day_of_month = current_time.day
            if day_of_month <= 3:  # Beginning of month
                analysis['factors'].append("month_start_flow")
                analysis['rationale'].append("Month start - new capital allocations")
            elif day_of_month >= 28:  # End of month
                analysis['factors'].append("month_end_rebalancing")
                analysis['rationale'].append("Month end - portfolio rebalancing")

        except Exception as e:
            logger.error(f"Failed to analyze time characteristics: {e}")

        return analysis

    def _calculate_smart_money_probability(
        self,
        session_status: Any,
        alignment_factors: List[str]
    ) -> float:
        """Calculate probability of smart money activity."""

        try:
            base_probability = 0.1  # Base probability

            # Session-based probability
            if session_status.is_active:
                active_weights = session_status.session_weights.values()
                base_probability = max(active_weights) if active_weights else 0.1

            # Factor-based adjustments
            factor_adjustments = {
                "peak_liquidity": 0.3,
                "high_institutional_participation": 0.25,
                "optimal_volatility": 0.2,
                "multiple_session_overlap": 0.2,
                "us_market_hours": 0.15,
                "european_market_hours": 0.15,
                "market_structure_definition": 0.1
            }

            for factor in alignment_factors:
                if factor in factor_adjustments:
                    base_probability += factor_adjustments[factor]

            return min(base_probability, 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate smart money probability: {e}")
            return 0.1

    def _estimate_session_liquidity(self, session_name: str) -> str:
        """Estimate liquidity conditions for a session."""
        liquidity_mapping = {
            'asian': 'Low',
            'london': 'High',
            'new_york': 'High',
            'london_ny_overlap': 'Very High'
        }
        return liquidity_mapping.get(session_name, 'Medium')

    def _estimate_session_volatility(self, session_name: str) -> str:
        """Estimate volatility conditions for a session."""
        volatility_mapping = {
            'asian': 'Low',
            'london': 'Medium',
            'new_york': 'High',
            'london_ny_overlap': 'High'
        }
        return volatility_mapping.get(session_name, 'Medium')

    def _parse_time_delta(self, time_str: Optional[str]) -> Optional[timedelta]:
        """Parse time delta from string."""
        if not time_str:
            return None

        try:
            # Simple parsing - expects format like "0:30:00"
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        except:
            pass

        return None

    def _calculate_overall_killzone_score(
        self,
        session_multiplier: float,
        timing_bonus: float,
        signals: List[KillzoneSignal]
    ) -> float:
        """Calculate overall killzone score."""

        # Base score from session multiplier
        base_score = session_multiplier

        # Add timing bonus
        base_score += timing_bonus

        # Boost from active signals
        if signals:
            max_signal_strength = max(signal.strength for signal in signals)
            base_score += max_signal_strength * 0.2

        return min(base_score, 1.0)

    def _empty_killzone_score(self) -> KillzoneScore:
        """Return empty killzone score when no data available."""
        return KillzoneScore(
            overall_score=0.1,  # Minimum score outside sessions
            session_multiplier=0.1,
            timing_bonus=0.0,
            signals=[],
            session_analysis={},
            optimal_windows=[],
            timestamp=datetime.now()
        )