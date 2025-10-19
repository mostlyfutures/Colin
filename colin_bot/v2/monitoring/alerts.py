"""
Alert Management for Colin Trading Bot v2.0

This module implements alerting system.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Alert management system."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[str, List[Callable]] = {}
        self.alert_count = 0

    async def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Dict[str, Any] = None
    ) -> Alert:
        """Create and process new alert."""
        self.alert_count += 1
        alert_id = f"alert_{self.alert_count}_{int(datetime.now().timestamp())}"

        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )

        self.alerts.append(alert)
        await self._process_alert(alert)

        return alert

    async def _process_alert(self, alert: Alert):
        """Process alert through handlers."""
        logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

        # Call registered handlers
        severity_handlers = self.alert_handlers.get(alert.severity.value, [])
        for handler in severity_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def register_handler(self, severity: AlertSeverity, handler: Callable):
        """Register alert handler for severity level."""
        if severity.value not in self.alert_handlers:
            self.alert_handlers[severity.value] = []
        self.alert_handlers[severity.value].append(handler)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in active_alerts if alert.severity == severity
            ])

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "acknowledged_alerts": len([a for a in active_alerts if a.acknowledged]),
            "severity_breakdown": severity_counts,
            "recent_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)[:10]
            ]
        }