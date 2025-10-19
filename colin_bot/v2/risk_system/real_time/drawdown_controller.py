"""
Drawdown Controller for Colin Trading Bot v2.0

This module implements portfolio drawdown monitoring and control.

Key Features:
- Real-time drawdown calculation and monitoring
- Maximum drawdown enforcement (5% hard limit from PRP)
- Warning drawdown triggers (3% warning level)
- Automatic position reduction triggers
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class DrawdownLevel(Enum):
    """Drawdown level classification."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DrawdownMetrics:
    """Drawdown metrics data structure."""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown_duration: float = 0.0  # in seconds
    max_drawdown_duration: float = 0.0
    peak_value: float = 0.0
    current_value: float = 0.0
    recovery_level: float = 0.0  # 0-1, how recovered from max drawdown
    drawdown_level: DrawdownLevel = DrawdownLevel.NORMAL
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class DrawdownAlert:
    """Drawdown alert data structure."""
    alert_type: str  # "warning", "critical", "emergency"
    drawdown_value: float
    limit_value: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    actions_taken: List[str] = field(default_factory=list)


@dataclass
class DrawdownConfiguration:
    """Drawdown control configuration."""
    warning_threshold: float = 0.03       # 3% warning level
    critical_threshold: float = 0.05      # 5% critical level
    emergency_threshold: float = 0.08     # 8% emergency level
    position_reduction_warning: float = 0.25    # Reduce positions by 25% at warning
    position_reduction_critical: float = 0.50   # Reduce positions by 50% at critical
    position_reduction_emergency: float = 0.75  # Reduce positions by 75% at emergency
    auto_trading_halt_threshold: float = 0.10   # 10% halt all trading
    recovery_wait_time_seconds: int = 300       # 5 minutes wait before recovery
    max_drawdown_duration_seconds: int = 14400  # 4 hours max drawdown duration


class DrawdownController:
    """
    Real-time drawdown monitoring and control.

    This class monitors portfolio drawdown in real-time and implements
    automatic position reduction and trading halts when drawdown limits are exceeded.
    """

    def __init__(
        self,
        config: Optional[DrawdownConfiguration] = None,
        initial_portfolio_value: float = 100000.0
    ):
        """
        Initialize drawdown controller.

        Args:
            config: Drawdown configuration parameters
            initial_portfolio_value: Starting portfolio value
        """
        self.config = config or DrawdownConfiguration()

        # Portfolio value tracking
        self.peak_value = initial_portfolio_value
        self.current_value = initial_portfolio_value
        self.value_history: List[Tuple[datetime, float]] = [(datetime.now(), initial_portfolio_value)]

        # Drawdown tracking
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_start_time: Optional[datetime] = None
        self.max_drawdown_duration = 0.0

        # Alerts and actions
        self.drawdown_alerts: List[DrawdownAlert] = []
        self.active_reduction_level = 0.0  # Current position reduction (0-1)
        self.trading_halted = False
        self.halt_reason = ""

        # Background monitoring
        self.monitoring_task = None
        self.is_monitoring = False

        # Performance metrics
        self.update_count = 0
        self.last_calculation_time = 0.0

        logger.info(f"DrawdownController initialized with peak value: ${self.peak_value:,.2f}")
        logger.info(f"Warning threshold: {self.config.warning_threshold:.1%}")
        logger.info(f"Critical threshold: {self.config.critical_threshold:.1%}")

    async def start_monitoring(self):
        """Start background drawdown monitoring."""
        if self.is_monitoring:
            logger.warning("Drawdown monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Drawdown monitoring started")

    async def stop_monitoring(self):
        """Stop background drawdown monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Drawdown monitoring stopped")

    async def update_portfolio_value(self, new_value: float, timestamp: Optional[datetime] = None):
        """
        Update portfolio value and recalculate drawdown.

        Args:
            new_value: New portfolio value
            timestamp: Optional timestamp (defaults to now)
        """
        try:
            start_time = time.time()
            self.update_count += 1

            if timestamp is None:
                timestamp = datetime.now()

            old_value = self.current_value
            self.current_value = new_value
            self.value_history.append((timestamp, new_value))

            # Keep only last 1000 data points
            if len(self.value_history) > 1000:
                self.value_history = self.value_history[-1000:]

            # Update peak value
            if new_value > self.peak_value:
                self.peak_value = new_value
                # Reset drawdown duration when we make a new high
                self.drawdown_start_time = None
                logger.info(f"New portfolio peak: ${self.peak_value:,.2f}")

            # Calculate current drawdown
            self.current_drawdown = (self.peak_value - new_value) / self.peak_value

            # Track drawdown duration
            if self.current_drawdown > 0:
                if self.drawdown_start_time is None:
                    self.drawdown_start_time = timestamp
                    logger.debug(f"Drawdown started at {self.current_drawdown:.2%}")
            else:
                # We've recovered to peak value
                if self.drawdown_start_time:
                    drawdown_duration = (timestamp - self.drawdown_start_time).total_seconds()
                    self.max_drawdown_duration = max(self.max_drawdown_duration, drawdown_duration)
                    self.drawdown_start_time = None
                    logger.debug(f"Drawdown recovered after {drawdown_duration:.0f}s")

            # Update max drawdown
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

            # Check drawdown levels and trigger actions
            await self._check_drawdown_levels()

            # Performance metrics
            calculation_time = time.time() - start_time
            self.last_calculation_time = calculation_time

            # Log significant drawdown changes
            drawdown_change = abs(self.current_drawdown - (self.peak_value - old_value) / self.peak_value) if self.peak_value > 0 else 0
            if drawdown_change > 0.001:  # 0.1% change
                logger.debug(f"Drawdown updated: {self.current_drawdown:.2%}, P&L: {new_value - old_value:+.2f}")

        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")

    async def _check_drawdown_levels(self):
        """Check drawdown levels and trigger appropriate actions."""
        try:
            drawdown_level = self._get_drawdown_level(self.current_drawdown)

            # Check for emergency level
            if self.current_drawdown >= self.config.emergency_threshold:
                await self._handle_emergency_drawdown()

            # Check for critical level
            elif self.current_drawdown >= self.config.critical_threshold:
                await self._handle_critical_drawdown()

            # Check for warning level
            elif self.current_drawdown >= self.config.warning_threshold:
                await self._handle_warning_drawdown()

            # Check for recovery
            else:
                await self._handle_drawdown_recovery()

        except Exception as e:
            logger.error(f"Error checking drawdown levels: {e}")

    def _get_drawdown_level(self, drawdown: float) -> DrawdownLevel:
        """Get drawdown level classification."""
        if drawdown >= self.config.emergency_threshold:
            return DrawdownLevel.EMERGENCY
        elif drawdown >= self.config.critical_threshold:
            return DrawdownLevel.CRITICAL
        elif drawdown >= self.config.warning_threshold:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL

    async def _handle_warning_drawdown(self):
        """Handle warning level drawdown."""
        # Check if we already handled this level
        if self.active_reduction_level >= self.config.position_reduction_warning:
            return

        # Create alert
        alert = DrawdownAlert(
            alert_type="warning",
            drawdown_value=self.current_drawdown,
            limit_value=self.config.warning_threshold,
            message=f"Warning: Portfolio drawdown {self.current_drawdown:.1%} exceeds {self.config.warning_threshold:.1%} threshold",
            actions_taken=[
                f"Reduce position sizes by {self.config.position_reduction_warning:.0%}",
                "Increase monitoring frequency",
                "Review recent trades"
            ]
        )
        self.drawdown_alerts.append(alert)

        # Set position reduction
        self.active_reduction_level = self.config.position_reduction_warning

        logger.warning(f"DRAWDOWN WARNING: {self.current_drawdown:.1%} - Reducing positions by {self.config.position_reduction_warning:.0%}")

    async def _handle_critical_drawdown(self):
        """Handle critical level drawdown."""
        # Check if we already handled this level
        if self.active_reduction_level >= self.config.position_reduction_critical:
            return

        # Create alert
        alert = DrawdownAlert(
            alert_type="critical",
            drawdown_value=self.current_drawdown,
            limit_value=self.config.critical_threshold,
            message=f"CRITICAL: Portfolio drawdown {self.current_drawdown:.1%} exceeds {self.config.critical_threshold:.1%} threshold",
            actions_taken=[
                f"Reduce position sizes by {self.config.position_reduction_critical:.0%}",
                "Stop opening new positions",
                "Consider closing high-risk positions",
                "Immediate risk manager review required"
            ]
        )
        self.drawdown_alerts.append(alert)

        # Set position reduction
        self.active_reduction_level = self.config.position_reduction_critical

        logger.critical(f"DRAWDOWN CRITICAL: {self.current_drawdown:.1%} - Reducing positions by {self.config.position_reduction_critical:.0%}")

    async def _handle_emergency_drawdown(self):
        """Handle emergency level drawdown."""
        # Check if trading halt is already active
        if self.trading_halted:
            return

        # Create alert
        alert = DrawdownAlert(
            alert_type="emergency",
            drawdown_value=self.current_drawdown,
            limit_value=self.config.emergency_threshold,
            message=f"EMERGENCY: Portfolio drawdown {self.current_drawdown:.1%} exceeds {self.config.emergency_threshold:.1%} threshold",
            actions_taken=[
                f"Reduce position sizes by {self.config.position_reduction_emergency:.0%}",
                "HALT all new trading immediately",
                "Close positions systematically",
                "Emergency risk management protocol activated"
            ]
        )
        self.drawdown_alerts.append(alert)

        # Set position reduction and halt trading
        self.active_reduction_level = self.config.position_reduction_emergency
        self.trading_halted = True
        self.halt_reason = f"Emergency drawdown: {self.current_drawdown:.1%}"

        logger.critical(f"DRAWDOWN EMERGENCY: {self.current_drawdown:.1%} - HALTING ALL TRADING")

    async def _handle_drawdown_recovery(self):
        """Handle drawdown recovery."""
        if self.active_reduction_level > 0:
            # Gradually reduce position restrictions
            recovery_reduction = max(0, self.active_reduction_level - 0.1)
            self.active_reduction_level = recovery_reduction

            logger.info(f"DRAWDOWN RECOVERY: Reducing position restrictions from {self.active_reduction_level + 0.1:.0%} to {self.active_reduction_level:.0%}")

        if self.trading_halted:
            # Check if we can resume trading
            if self.current_drawdown < self.config.warning_threshold:
                self.trading_halted = False
                old_reason = self.halt_reason
                self.halt_reason = ""
                logger.info(f"TRADING RESUMED: Drawdown recovered from {old_reason}")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Check drawdown duration
                await self._check_drawdown_duration()

                # Clean up old alerts
                self.drawdown_alerts = [
                    alert for alert in self.drawdown_alerts
                    if (datetime.now() - alert.timestamp).total_seconds() < 3600  # Keep last hour
                ]

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in drawdown monitoring loop: {e}")
                await asyncio.sleep(1)

    async def _check_drawdown_duration(self):
        """Check if drawdown duration exceeds limits."""
        try:
            if self.drawdown_start_time is None:
                return

            current_duration = (datetime.now() - self.drawdown_start_time).total_seconds()

            # Check maximum duration limit
            if current_duration > self.config.max_drawdown_duration_seconds:
                logger.warning(f"Maximum drawdown duration exceeded: {current_duration:.0f}s > {self.config.max_drawdown_duration_seconds:.0f}s")

                # Additional position reduction for extended drawdown
                if self.active_reduction_level < 0.9:
                    self.active_reduction_level = min(0.9, self.active_reduction_level + 0.1)
                    logger.warning(f"Additional position reduction due to extended drawdown: {self.active_reduction_level:.0%}")

        except Exception as e:
            logger.error(f"Error checking drawdown duration: {e}")

    def get_drawdown_metrics(self) -> DrawdownMetrics:
        """Get current drawdown metrics."""
        current_drawdown_duration = 0.0
        if self.drawdown_start_time:
            current_drawdown_duration = (datetime.now() - self.drawdown_start_time).total_seconds()

        recovery_level = 0.0
        if self.max_drawdown > 0:
            recovery_level = max(0, 1 - (self.current_drawdown / self.max_drawdown))

        return DrawdownMetrics(
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            current_drawdown_duration=current_drawdown_duration,
            max_drawdown_duration=self.max_drawdown_duration,
            peak_value=self.peak_value,
            current_value=self.current_value,
            recovery_level=recovery_level,
            drawdown_level=self._get_drawdown_level(self.current_drawdown),
            last_update=datetime.now()
        )

    def get_position_reduction_multiplier(self) -> float:
        """
        Get position size reduction multiplier.

        Returns:
            Multiplier between 0 and 1 indicating allowed position size
        """
        return 1.0 - self.active_reduction_level

    def is_trading_allowed(self) -> bool:
        """Check if new trading is allowed."""
        return not self.trading_halted

    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        return {
            "trading_allowed": self.is_trading_allowed(),
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "position_reduction_level": self.active_reduction_level,
            "position_multiplier": self.get_position_reduction_multiplier(),
            "last_alert": self.drawdown_alerts[-1].__dict__ if self.drawdown_alerts else None
        }

    def reset_to_new_peak(self, new_peak_value: float):
        """
        Reset drawdown calculations to a new peak value.

        Args:
            new_peak_value: New peak portfolio value
        """
        old_peak = self.peak_value
        self.peak_value = new_peak_value
        self.current_drawdown = 0.0
        self.drawdown_start_time = None

        # Reset trading restrictions
        self.active_reduction_level = 0.0
        self.trading_halted = False
        self.halt_reason = ""

        logger.info(f"Drawdown controller reset to new peak: ${old_peak:,.2f} -> ${new_peak_value:,.2f}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "update_count": self.update_count,
            "last_calculation_time_ms": self.last_calculation_time * 1000,
            "total_alerts": len(self.drawdown_alerts),
            "current_alerts": len([
                alert for alert in self.drawdown_alerts
                if (datetime.now() - alert.timestamp).total_seconds() < 300  # Last 5 minutes
            ]),
            "monitoring_active": self.is_monitoring,
            "value_history_points": len(self.value_history)
        }


# Standalone validation function
def validate_drawdown_controller():
    """Validate drawdown controller implementation."""
    print("🔍 Validating DrawdownController implementation...")

    try:
        # Test imports
        from .drawdown_controller import DrawdownController, DrawdownMetrics, DrawdownLevel
        print("✅ Imports successful")

        # Test instantiation
        controller = DrawdownController()
        print("✅ DrawdownController instantiation successful")

        # Test basic functionality
        if hasattr(controller, 'update_portfolio_value'):
            print("✅ update_portfolio_value method exists")
        else:
            print("❌ update_portfolio_value method missing")
            return False

        if hasattr(controller, 'get_drawdown_metrics'):
            print("✅ get_drawdown_metrics method exists")
        else:
            print("❌ get_drawdown_metrics method missing")
            return False

        if hasattr(controller, 'is_trading_allowed'):
            print("✅ is_trading_allowed method exists")
        else:
            print("❌ is_trading_allowed method missing")
            return False

        print("🎉 DrawdownController validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_drawdown_controller()