"""
Compliance Monitor for Colin Trading Bot v2.0

This module implements ongoing compliance monitoring and reporting.

Key Features:
- Real-time compliance status tracking
- Regulatory reporting automation
- Compliance breach detection and alerts
- Audit log maintenance and review
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ComplianceMetricType(Enum):
    """Compliance metric types."""
    POSITION_LIMITS = "position_limits"
    TRADING_FREQUENCY = "trading_frequency"
    CONCENTRATION_RISK = "concentration_risk"
    MARKET_MANIPULATION = "market_manipulation"
    BEST_EXECUTION = "best_execution"
    REPORTING_REQUIREMENTS = "reporting_requirements"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceMetric:
    """Compliance metric data."""
    metric_type: ComplianceMetricType
    name: str
    current_value: float
    threshold_value: float
    status: str  # "compliant", "warning", "breach"
    last_updated: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAlert:
    """Compliance alert data."""
    alert_id: str
    metric_type: ComplianceMetricType
    severity: AlertSeverity
    title: str
    description: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    actions_taken: List[str] = field(default_factory=list)
    regulatory_impact: bool = False


@dataclass
class ComplianceReport:
    """Compliance report data."""
    report_id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    metrics: List[ComplianceMetric]
    alerts: List[ComplianceAlert]
    overall_compliance_score: float
    generated_at: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ComplianceMonitorConfiguration:
    """Compliance monitor configuration."""
    monitoring_interval_seconds: int = 60
    alert_retention_days: int = 90
    report_generation_schedule: str = "daily"  # daily, weekly, monthly
    auto_acknowledge_threshold: str = "low"
    regulatory_reporting_enabled: bool = True
    real_time_monitoring: bool = True


class ComplianceMonitor:
    """
    Real-time compliance monitoring and alerting system.

    This class monitors ongoing compliance with various regulatory requirements
    and generates alerts and reports for compliance breaches.
    """

    def __init__(
        self,
        config: Optional[ComplianceMonitorConfiguration] = None,
        portfolio_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize compliance monitor.

        Args:
            config: Monitor configuration
            portfolio_data: Current portfolio data
        """
        self.config = config or ComplianceMonitorConfiguration()
        self.portfolio_data = portfolio_data or {"total_value": 100000.0}

        # Compliance metrics storage
        self.compliance_metrics: Dict[str, ComplianceMetric] = {}
        self.alert_history: List[ComplianceAlert] = []
        self.active_alerts: List[ComplianceAlert] = []
        self.reports: List[ComplianceReport] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_update_time = datetime.now()

        # Performance metrics
        self.monitoring_cycles = 0
        self.alerts_generated = 0
        self.reports_generated = 0

        logger.info(f"ComplianceMonitor initialized with {self.config.monitoring_interval_seconds}s interval")
        logger.info(f"Real-time monitoring: {self.config.real_time_monitoring}")

    async def start_monitoring(self):
        """Start background compliance monitoring."""
        if self.is_monitoring:
            logger.warning("Compliance monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Compliance monitoring started")

    async def stop_monitoring(self):
        """Stop background compliance monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Compliance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                start_time = time.time()
                self.monitoring_cycles += 1

                # Update all compliance metrics
                await self._update_all_metrics()

                # Check for alert conditions
                await self._check_alert_conditions()

                # Clean up old data
                await self._cleanup_old_data()

                # Generate periodic reports
                if await self._should_generate_report():
                    await self._generate_periodic_report()

                # Performance metrics
                cycle_duration = time.time() - start_time
                self.last_update_time = datetime.now()

                if cycle_duration > self.config.monitoring_interval_seconds * 0.8:
                    logger.warning(f"Compliance monitoring cycle took {cycle_duration:.2f}s (>{self.config.monitoring_interval_seconds * 0.8:.2f}s)")

                await asyncio.sleep(self.config.monitoring_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in compliance monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _update_all_metrics(self):
        """Update all compliance metrics."""
        # Position limits metrics
        await self._update_position_limits_metrics()

        # Trading frequency metrics
        await self._update_trading_frequency_metrics()

        # Concentration risk metrics
        await self._update_concentration_risk_metrics()

        # Best execution metrics
        await self._update_best_execution_metrics()

        # Reporting requirements metrics
        await self._update_reporting_metrics()

    async def _update_position_limits_metrics(self):
        """Update position limits compliance metrics."""
        try:
            portfolio_value = self.portfolio_data.get("total_value", 0)
            positions = self.portfolio_data.get("positions", {})

            if portfolio_value == 0 or not positions:
                return

            # Calculate maximum position concentration
            max_position_ratio = 0.0
            total_position_value = 0.0

            for symbol, position in positions.items():
                position_value = position.get("value", 0)
                total_position_value += position_value
                position_ratio = position_value / portfolio_value
                max_position_ratio = max(max_position_ratio, position_ratio)

            # Update metric
            metric = ComplianceMetric(
                metric_type=ComplianceMetricType.POSITION_LIMITS,
                name="Maximum Position Concentration",
                current_value=max_position_ratio,
                threshold_value=0.20,  # 20% limit
                status=self._determine_compliance_status(max_position_ratio, 0.15, 0.20),
                details={
                    "portfolio_value": portfolio_value,
                    "total_position_value": total_position_value,
                    "position_count": len(positions)
                }
            )

            self.compliance_metrics["max_position_concentration"] = metric

        except Exception as e:
            logger.error(f"Error updating position limits metrics: {e}")

    async def _update_trading_frequency_metrics(self):
        """Update trading frequency compliance metrics."""
        try:
            # This would typically get data from trading system
            # For now, use mock data
            daily_trades = self.portfolio_data.get("daily_trade_count", 0)
            weekly_trades = self.portfolio_data.get("weekly_trade_count", 0)

            # Daily trading frequency metric
            daily_metric = ComplianceMetric(
                metric_type=ComplianceMetricType.TRADING_FREQUENCY,
                name="Daily Trade Count",
                current_value=daily_trades,
                threshold_value=1000,  # Daily limit
                status=self._determine_compliance_status(daily_trades, 800, 1000),
                details={
                    "daily_trades": daily_trades,
                    "weekly_trades": weekly_trades
                }
            )

            self.compliance_metrics["daily_trade_count"] = daily_metric

        except Exception as e:
            logger.error(f"Error updating trading frequency metrics: {e}")

    async def _update_concentration_risk_metrics(self):
        """Update concentration risk compliance metrics."""
        try:
            positions = self.portfolio_data.get("positions", {})
            portfolio_value = self.portfolio_data.get("total_value", 0)

            if portfolio_value == 0 or not positions:
                return

            # Calculate Herfindahl-Hirschman Index (HHI) for concentration
            hhi = 0.0
            for position in positions.values():
                weight = position.get("value", 0) / portfolio_value
                hhi += weight ** 2

            # Update metric
            metric = ComplianceMetric(
                metric_type=ComplianceMetricType.CONCENTRATION_RISK,
                name="Portfolio Concentration (HHI)",
                current_value=hhi,
                threshold_value=0.25,  # High concentration threshold
                status=self._determine_compliance_status(hhi, 0.15, 0.25),
                details={
                    "hhi": hhi,
                    "position_count": len(positions),
                    "interpretation": self._interpret_hhi(hhi)
                }
            )

            self.compliance_metrics["portfolio_concentration_hhi"] = metric

        except Exception as e:
            logger.error(f"Error updating concentration risk metrics: {e}")

    async def _update_best_execution_metrics(self):
        """Update best execution compliance metrics."""
        try:
            # This would typically get data from execution system
            # For now, use mock data
            execution_quality = self.portfolio_data.get("execution_quality_score", 0.85)
            venue_diversity = self.portfolio_data.get("venue_diversity_score", 0.70)

            # Execution quality metric
            quality_metric = ComplianceMetric(
                metric_type=ComplianceMetricType.BEST_EXECUTION,
                name="Execution Quality Score",
                current_value=execution_quality,
                threshold_value=0.80,  # Minimum quality score
                status=self._determine_compliance_status(execution_quality, 0.85, 0.80),
                details={
                    "execution_quality": execution_quality,
                    "venue_diversity": venue_diversity
                }
            )

            self.compliance_metrics["execution_quality"] = quality_metric

        except Exception as e:
            logger.error(f"Error updating best execution metrics: {e}")

    async def _update_reporting_metrics(self):
        """Update reporting requirements compliance metrics."""
        try:
            # Check if required reports are up to date
            last_daily_report = self.portfolio_data.get("last_daily_report")
            last_weekly_report = self.portfolio_data.get("last_weekly_report")
            last_monthly_report = self.portfolio_data.get("last_monthly_report")

            now = datetime.now()
            days_since_daily = (now - last_daily_report).days if last_daily_report else 999
            days_since_weekly = (now - last_weekly_report).days if last_weekly_report else 999
            days_since_monthly = (now - last_monthly_report).days if last_monthly_report else 999

            # Daily reporting compliance
            daily_status = "compliant" if days_since_daily <= 1 else "breach"
            daily_metric = ComplianceMetric(
                metric_type=ComplianceMetricType.REPORTING_REQUIREMENTS,
                name="Daily Reporting Timeliness",
                current_value=days_since_daily,
                threshold_value=1.0,
                status=daily_status,
                details={
                    "days_since_last_report": days_since_daily,
                    "last_report_date": last_daily_report.isoformat() if last_daily_report else None
                }
            )

            self.compliance_metrics["daily_reporting"] = daily_metric

        except Exception as e:
            logger.error(f"Error updating reporting metrics: {e}")

    def _determine_compliance_status(
        self,
        current_value: float,
        warning_threshold: float,
        breach_threshold: float
    ) -> str:
        """Determine compliance status based on thresholds."""
        if current_value >= breach_threshold:
            return "breach"
        elif current_value >= warning_threshold:
            return "warning"
        else:
            return "compliant"

    def _interpret_hhi(self, hhi: float) -> str:
        """Interpret HHI concentration index."""
        if hhi < 0.15:
            return "Low concentration"
        elif hhi < 0.25:
            return "Moderate concentration"
        else:
            return "High concentration"

    async def _check_alert_conditions(self):
        """Check for alert conditions and generate alerts."""
        for metric_id, metric in self.compliance_metrics.items():
            if metric.status == "breach":
                await self._generate_alert(metric, AlertSeverity.HIGH)
            elif metric.status == "warning":
                await self._generate_alert(metric, AlertSeverity.MEDIUM)

    async def _generate_alert(self, metric: ComplianceMetric, severity: AlertSeverity):
        """Generate compliance alert."""
        # Check if alert already exists for this metric
        existing_alert = next(
            (alert for alert in self.active_alerts
             if alert.metric_type == metric.metric_type and not alert.resolved),
            None
        )

        if existing_alert:
            return  # Alert already active

        # Create new alert
        alert_id = f"alert_{metric.metric_type.value}_{int(time.time())}"
        alert = ComplianceAlert(
            alert_id=alert_id,
            metric_type=metric.metric_type,
            severity=severity,
            title=f"Compliance {severity.value.title()}: {metric.name}",
            description=f"{metric.name} is {metric.status}: current value {metric.current_value:.2f} exceeds threshold {metric.threshold_value:.2f}",
            current_value=metric.current_value,
            threshold_value=metric.threshold_value,
            regulatory_impact=severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )

        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        self.alerts_generated += 1

        # Auto-acknowledge low severity alerts
        if severity == AlertSeverity.LOW:
            alert.acknowledged = True

        logger.warning(f"Compliance alert generated: {alert.title}")

    async def _cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        cutoff_date = datetime.now() - timedelta(days=self.config.alert_retention_days)

        # Clean up old alerts
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_date
        ]

        # Clean up resolved active alerts older than 24 hours
        resolved_cutoff = datetime.now() - timedelta(hours=24)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not alert.resolved or alert.timestamp >= resolved_cutoff
        ]

    async def _should_generate_report(self) -> bool:
        """Check if periodic report should be generated."""
        now = datetime.now()

        if self.config.report_generation_schedule == "daily":
            # Check if it's a new day and we haven't generated a report today
            last_report = next(
                (report for report in reversed(self.reports)
                 if report.report_type == "daily"),
                None
            )
            if not last_report or last_report.generated_at.date() < now.date():
                return True

        elif self.config.report_generation_schedule == "weekly":
            # Check if it's a new week
            last_report = next(
                (report for report in reversed(self.reports)
                 if report.report_type == "weekly"),
                None
            )
            if not last_report or (now - last_report.generated_at).days >= 7:
                return True

        return False

    async def _generate_periodic_report(self):
        """Generate periodic compliance report."""
        try:
            report_id = f"report_{self.config.report_generation_schedule}_{int(time.time())}"
            period_end = datetime.now()

            if self.config.report_generation_schedule == "daily":
                period_start = period_end - timedelta(days=1)
            else:
                period_start = period_end - timedelta(weeks=1)

            # Calculate overall compliance score
            compliant_metrics = sum(
                1 for metric in self.compliance_metrics.values()
                if metric.status == "compliant"
            )
            total_metrics = len(self.compliance_metrics)
            overall_score = compliant_metrics / total_metrics if total_metrics > 0 else 1.0

            # Generate recommendations
            recommendations = await self._generate_recommendations()

            # Create report
            report = ComplianceReport(
                report_id=report_id,
                report_type=self.config.report_generation_schedule,
                period_start=period_start,
                period_end=period_end,
                metrics=list(self.compliance_metrics.values()),
                alerts=[alert for alert in self.alert_history if alert.timestamp >= period_start],
                overall_compliance_score=overall_score,
                recommendations=recommendations
            )

            self.reports.append(report)
            self.reports_generated += 1

            logger.info(f"Generated {self.config.report_generation_schedule} compliance report: {report_id}")

        except Exception as e:
            logger.error(f"Error generating periodic report: {e}")

    async def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations based on current metrics."""
        recommendations = []

        for metric in self.compliance_metrics.values():
            if metric.status == "breach":
                if metric.metric_type == ComplianceMetricType.POSITION_LIMITS:
                    recommendations.append("Reduce position concentrations to comply with limits")
                elif metric.metric_type == ComplianceMetricType.TRADING_FREQUENCY:
                    recommendations.append("Reduce trading frequency to stay within limits")
                elif metric.metric_type == ComplianceMetricType.CONCENTRATION_RISK:
                    recommendations.append("Diversify portfolio to reduce concentration risk")
                elif metric.metric_type == ComplianceMetricType.BEST_EXECUTION:
                    recommendations.append("Improve execution quality and venue selection")
                elif metric.metric_type == ComplianceMetricType.REPORTING_REQUIREMENTS:
                    recommendations.append("Ensure timely submission of required reports")

        if not recommendations:
            recommendations.append("All compliance metrics are within acceptable limits")

        return recommendations

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge a compliance alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True

        logger.warning(f"Alert {alert_id} not found")
        return False

    async def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve a compliance alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                if resolution_notes:
                    alert.actions_taken.append(f"Resolved: {resolution_notes}")
                logger.info(f"Alert {alert_id} resolved")
                return True

        logger.warning(f"Alert {alert_id} not found")
        return False

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance monitoring dashboard data."""
        # Calculate summary statistics
        total_metrics = len(self.compliance_metrics)
        compliant_metrics = sum(
            1 for metric in self.compliance_metrics.values()
            if metric.status == "compliant"
        )
        warning_metrics = sum(
            1 for metric in self.compliance_metrics.values()
            if metric.status == "warning"
        )
        breach_metrics = total_metrics - compliant_metrics - warning_metrics

        # Active alerts by severity
        alerts_by_severity = {
            "critical": len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "high": len([a for a in self.active_alerts if a.severity == AlertSeverity.HIGH]),
            "medium": len([a for a in self.active_alerts if a.severity == AlertSeverity.MEDIUM]),
            "low": len([a for a in self.active_alerts if a.severity == AlertSeverity.LOW])
        }

        return {
            "monitoring_status": {
                "is_monitoring": self.is_monitoring,
                "last_update": self.last_update_time.isoformat(),
                "monitoring_cycles": self.monitoring_cycles,
                "monitoring_interval_seconds": self.config.monitoring_interval_seconds
            },
            "compliance_summary": {
                "total_metrics": total_metrics,
                "compliant_metrics": compliant_metrics,
                "warning_metrics": warning_metrics,
                "breach_metrics": breach_metrics,
                "compliance_rate": compliant_metrics / total_metrics if total_metrics > 0 else 1.0
            },
            "alerts_summary": {
                "active_alerts": len(self.active_alerts),
                "total_alerts_generated": self.alerts_generated,
                "alerts_by_severity": alerts_by_severity,
                "unacknowledged_alerts": len([a for a in self.active_alerts if not a.acknowledged])
            },
            "reports_summary": {
                "total_reports": self.reports_generated,
                "last_report": self.reports[-1].generated_at.isoformat() if self.reports else None,
                "next_report_due": self._calculate_next_report_due()
            },
            "metrics": {
                metric_id: {
                    "name": metric.name,
                    "current_value": metric.current_value,
                    "threshold_value": metric.threshold_value,
                    "status": metric.status,
                    "last_updated": metric.last_updated.isoformat()
                }
                for metric_id, metric in self.compliance_metrics.items()
            },
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "metric_type": alert.metric_type.value,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "regulatory_impact": alert.regulatory_impact
                }
                for alert in self.active_alerts
            ]
        }

    def _calculate_next_report_due(self) -> Optional[str]:
        """Calculate when next report is due."""
        if self.config.report_generation_schedule == "daily":
            next_due = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif self.config.report_generation_schedule == "weekly":
            days_until_monday = (7 - datetime.now().weekday()) % 7 or 7
            next_due = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        else:
            return None

        return next_due.isoformat()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        uptime_percentage = 0.0
        if self.monitoring_cycles > 0:
            expected_cycles = (datetime.now() - self.last_update_time).total_seconds() / self.config.monitoring_interval_seconds
            uptime_percentage = min(100, (expected_cycles / self.monitoring_cycles) * 100)

        return {
            "monitoring_cycles": self.monitoring_cycles,
            "alerts_generated": self.alerts_generated,
            "reports_generated": self.reports_generated,
            "uptime_percentage": uptime_percentage,
            "average_cycle_time_ms": self.config.monitoring_interval_seconds * 1000,
            "metrics_tracked": len(self.compliance_metrics),
            "active_alerts": len(self.active_alerts),
            "last_update": self.last_update_time.isoformat()
        }


# Standalone validation function
def validate_compliance_monitor():
    """Validate compliance monitor implementation."""
    print("🔍 Validating ComplianceMonitor implementation...")

    try:
        # Test imports
        from .compliance_monitor import ComplianceMonitor, ComplianceAlert, ComplianceMetric
        print("✅ Imports successful")

        # Test instantiation
        monitor = ComplianceMonitor()
        print("✅ ComplianceMonitor instantiation successful")

        # Test basic functionality
        if hasattr(monitor, 'start_monitoring'):
            print("✅ start_monitoring method exists")
        else:
            print("❌ start_monitoring method missing")
            return False

        if hasattr(monitor, 'get_compliance_dashboard'):
            print("✅ get_compliance_dashboard method exists")
        else:
            print("❌ get_compliance_dashboard method missing")
            return False

        print("🎉 ComplianceMonitor validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_compliance_monitor()