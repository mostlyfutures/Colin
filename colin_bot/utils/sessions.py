"""
Session utilities for killzone timing analysis.

This module provides tools for identifying institutional trading sessions
and calculating session-based confidence adjustments for signal scoring.
"""

import pytz
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.config import ConfigManager


class SessionType(Enum):
    """Trading session types."""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"


@dataclass
class SessionInfo:
    """Information about a trading session."""
    name: str
    start_time: time
    end_time: time
    weight: float
    timezone: str = "UTC"
    description: str = ""


@dataclass
class SessionStatus:
    """Current session status."""
    is_active: bool
    active_sessions: List[SessionType]
    session_weights: Dict[str, float]
    time_to_next: Optional[timedelta] = None
    time_to_end: Optional[timedelta] = None
    current_session: Optional[SessionType] = None


class SessionAnalyzer:
    """Analyzes trading sessions and killzones for institutional timing."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize session analyzer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.utc = pytz.UTC
        self.sessions = self._load_sessions()

    def _load_sessions(self) -> Dict[SessionType, SessionInfo]:
        """
        Load session configurations.

        Returns:
            Dictionary of session information
        """
        sessions = {}

        # Load from config
        for session_name, session_config in self.config.sessions.items():
            start_time = datetime.strptime(session_config.start, "%H:%M").time()
            end_time = datetime.strptime(session_config.end, "%H:%M").time()

            session_info = SessionInfo(
                name=session_name,
                start_time=start_time,
                end_time=end_time,
                weight=session_config.weight,
                timezone="UTC",
                description=self._get_session_description(session_name)
            )

            # Map to enum
            if session_name == "asian":
                sessions[SessionType.ASIAN] = session_info
            elif session_name == "london":
                sessions[SessionType.LONDON] = session_info
            elif session_name == "new_york":
                sessions[SessionType.NEW_YORK] = session_info
            elif session_name == "london_ny_overlap":
                sessions[SessionType.LONDON_NY_OVERLAP] = session_info

        return sessions

    def _get_session_description(self, session_name: str) -> str:
        """
        Get description for a session.

        Args:
            session_name: Name of the session

        Returns:
            Session description
        """
        descriptions = {
            "asian": "Asian session - Lower liquidity, focus on JPY/KRW flows",
            "london": "London session - High liquidity, European institutional flow",
            "new_york": "New York session - US institutional flow, high volatility",
            "london_ny_overlap": "London/NY overlap - Highest liquidity and volatility"
        }
        return descriptions.get(session_name, "")

    def get_current_time(self) -> datetime:
        """
        Get current UTC time.

        Returns:
            Current UTC datetime
        """
        return datetime.now(self.utc)

    def is_time_in_session(self, dt: datetime, session: SessionInfo) -> bool:
        """
        Check if a datetime is within a session.

        Args:
            dt: Datetime to check (should be timezone-aware)
            session: Session information

        Returns:
            True if datetime is in session
        """
        # Convert to UTC
        if dt.tzinfo is None:
            dt = self.utc.localize(dt)
        else:
            dt = dt.astimezone(self.utc)

        current_time = dt.time()

        # Handle sessions that cross midnight
        if session.start_time > session.end_time:
            # Session crosses midnight (e.g., 22:00 to 06:00)
            return current_time >= session.start_time or current_time < session.end_time
        else:
            # Normal session
            return session.start_time <= current_time < session.end_time

    def get_active_sessions(self, dt: Optional[datetime] = None) -> List[SessionType]:
        """
        Get currently active sessions.

        Args:
            dt: Datetime to check (defaults to current time)

        Returns:
            List of active session types
        """
        if dt is None:
            dt = self.get_current_time()

        active_sessions = []
        for session_type, session_info in self.sessions.items():
            if self.is_time_in_session(dt, session_info):
                active_sessions.append(session_type)

        return active_sessions

    def get_session_status(self, dt: Optional[datetime] = None) -> SessionStatus:
        """
        Get comprehensive session status.

        Args:
            dt: Datetime to analyze (defaults to current time)

        Returns:
            Session status information
        """
        if dt is None:
            dt = self.get_current_time()

        active_sessions = self.get_active_sessions(dt)
        session_weights = {}

        # Calculate weights for active sessions
        for session_type in active_sessions:
            session_info = self.sessions[session_type]
            session_weights[session_type.value] = session_info.weight

        # Determine current session (highest weight)
        current_session = None
        if active_sessions:
            current_session = max(
                active_sessions,
                key=lambda x: self.sessions[x].weight
            )

        # Calculate time to next session and end of current session
        time_to_next = None
        time_to_end = None

        if active_sessions:
            # Find when current sessions end
            end_times = [
                datetime.combine(dt.date(), self.sessions[session].end_time)
                for session in active_sessions
            ]

            # Adjust for sessions that cross midnight
            for i, end_time in enumerate(end_times):
                if end_time.time() < dt.time():
                    end_time += timedelta(days=1)

            time_to_end = min(end_times) - dt

        # Find next session start
        next_sessions = []
        for session_type, session_info in self.sessions.items():
            if session_type not in active_sessions:
                start_time = datetime.combine(dt.date(), session_info.start_time)
                if start_time.time() < dt.time():
                    start_time += timedelta(days=1)
                next_sessions.append(start_time)

        if next_sessions:
            time_to_next = min(next_sessions) - dt

        return SessionStatus(
            is_active=len(active_sessions) > 0,
            active_sessions=active_sessions,
            session_weights=session_weights,
            time_to_next=time_to_next,
            time_to_end=time_to_end,
            current_session=current_session
        )

    def calculate_session_score(self, dt: Optional[datetime] = None) -> float:
        """
        Calculate session-based confidence score.

        Args:
            dt: Datetime to score (defaults to current time)

        Returns:
            Session score (0.0 to 1.0)
        """
        status = self.get_session_status(dt)

        if not status.is_active:
            return 0.1  # Low score outside sessions

        # Use highest weight session
        max_weight = max(status.session_weights.values()) if status.session_weights else 0.1

        # Normalize to 0-1 range (assuming weights from 1.0 to 1.5)
        normalized_score = min(max_weight / 1.5, 1.0)

        return normalized_score

    def get_killzone_analysis(self, dt: Optional[datetime] = None) -> Dict[str, any]:
        """
        Get comprehensive killzone analysis.

        Args:
            dt: Datetime to analyze (defaults to current time)

        Returns:
            Dictionary with killzone analysis
        """
        if dt is None:
            dt = self.get_current_time()

        status = self.get_session_status(dt)

        analysis = {
            'current_time_utc': dt.isoformat(),
            'is_institutional_hours': status.is_active,
            'active_sessions': [s.value for s in status.active_sessions],
            'primary_session': status.current_session.value if status.current_session else None,
            'session_weights': status.session_weights,
            'session_score': self.calculate_session_score(dt),
            'time_to_next_session': str(status.time_to_next) if status.time_to_next else None,
            'time_to_current_end': str(status.time_to_end) if status.time_to_end else None,
            'market_phase': self._determine_market_phase(status),
            'liquidity_expectation': self._estimate_liquidity(status),
            'volatility_expectation': self._estimate_volatility(status)
        }

        return analysis

    def _determine_market_phase(self, status: SessionStatus) -> str:
        """
        Determine market phase based on sessions.

        Args:
            status: Session status

        Returns:
            Market phase description
        """
        if not status.is_active:
            return "After Hours"

        active_names = [s.value for s in status.active_sessions]

        if SessionType.LONDON_NY_OVERLAP.value in active_names:
            return "Peak Institutional"
        elif SessionType.LONDON.value in active_names or SessionType.NEW_YORK.value in active_names:
            return "Institutional Hours"
        elif SessionType.ASIAN.value in active_names:
            return "Asian Session"
        else:
            return "Mixed Sessions"

    def _estimate_liquidity(self, status: SessionStatus) -> str:
        """
        Estimate liquidity conditions.

        Args:
            status: Session status

        Returns:
            Liquidity estimate
        """
        if not status.is_active:
            return "Very Low"

        if SessionType.LONDON_NY_OVERLAP in status.active_sessions:
            return "Very High"
        elif SessionType.LONDON in status.active_sessions or SessionType.NEW_YORK in status.active_sessions:
            return "High"
        elif SessionType.ASIAN in status.active_sessions:
            return "Medium"
        else:
            return "Low"

    def _estimate_volatility(self, status: SessionStatus) -> str:
        """
        Estimate volatility conditions.

        Args:
            status: Session status

        Returns:
            Volatility estimate
        """
        if not status.is_active:
            return "Very Low"

        # Overlap sessions typically have highest volatility
        if len(status.active_sessions) > 1:
            return "High"
        elif SessionType.NEW_YORK in status.active_sessions:
            return "Medium-High"
        elif SessionType.LONDON in status.active_sessions:
            return "Medium"
        elif SessionType.ASIAN in status.active_sessions:
            return "Low-Medium"
        else:
            return "Low"

    def is_optimal_entry_time(self, dt: Optional[datetime] = None) -> bool:
        """
        Determine if current time is optimal for entries.

        Args:
            dt: Datetime to check (defaults to current time)

        Returns:
            True if optimal entry time
        """
        status = self.get_session_status(dt)

        # Optimal times: London/NY overlap or first 2 hours of major sessions
        if SessionType.LONDON_NY_OVERLAP in status.active_sessions:
            return True

        # First 2 hours of London session (07:00-09:00 UTC)
        if SessionType.LONDON in status.active_sessions:
            current_time = dt.time() if dt else self.get_current_time().time()
            if time(7, 0) <= current_time < time(9, 0):
                return True

        # First 2 hours of NY session (12:00-14:00 UTC)
        if SessionType.NEW_YORK in status.active_sessions:
            current_time = dt.time() if dt else self.get_current_time().time()
            if time(12, 0) <= current_time < time(14, 0):
                return True

        return False

    def get_session_schedule(self, date: Optional[datetime] = None) -> List[Dict[str, any]]:
        """
        Get session schedule for a specific date.

        Args:
            date: Date to get schedule for (defaults to today)

        Returns:
            List of scheduled sessions
        """
        if date is None:
            date = self.get_current_time()

        schedule = []

        for session_type, session_info in self.sessions.items():
            start_dt = datetime.combine(date.date(), session_info.start_time)
            end_dt = datetime.combine(date.date(), session_info.end_time)

            # Handle sessions crossing midnight
            if session_info.start_time > session_info.end_time:
                end_dt += timedelta(days=1)

            schedule.append({
                'session': session_type.value,
                'start': start_dt.isoformat(),
                'end': end_dt.isoformat(),
                'weight': session_info.weight,
                'description': session_info.description,
                'is_active': self.is_time_in_session(self.get_current_time(), session_info)
            })

        return sorted(schedule, key=lambda x: x['start'])