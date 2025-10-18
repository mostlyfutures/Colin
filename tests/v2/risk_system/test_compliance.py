"""
Tests for Compliance Engine

This module contains comprehensive tests for the compliance engine components.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.v2.risk_system.compliance.pre_trade_check import (
    PreTradeChecker, ComplianceResult, ComplianceStatus, RegulatoryRegime,
    ComplianceConfiguration, ComplianceRule
)
from src.v2.risk_system.compliance.compliance_monitor import (
    ComplianceMonitor, ComplianceAlert, ComplianceMetric, ComplianceMetricType,
    ComplianceMonitorConfiguration, AlertSeverity
)
from src.v2.execution_engine.smart_routing.router import Order, OrderSide, OrderType


class TestPreTradeChecker:
    """Test cases for Pre-Trade Compliance Checker."""

    @pytest.fixture
    def compliance_checker(self):
        """Create a test compliance checker."""
        config = ComplianceConfiguration(
            max_position_size_portfolio=0.015,  # 1.5% of portfolio
            max_position_size_symbol=0.18,       # 18% single symbol
            max_daily_trades=500,
            max_order_size_usd=500000.0,
            min_order_size_usd=50.0,
            restricted_symbols=["PENNY", "RISKY"],
            regulatory_regime=RegulatoryRegime.SEC_FINRA
        )
        portfolio_data = {
            "total_value": 1000000.0,  # $1M portfolio
            "positions": {
                "BTC": {"value": 150000.0, "total_exposure": 150000.0},
                "ETH": {"value": 100000.0, "total_exposure": 100000.0}
            }
        }
        return PreTradeChecker(config=config, portfolio_data=portfolio_data)

    @pytest.fixture
    def compliant_order(self):
        """Create a compliant order for testing."""
        return Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=150.0,
            client_order_id="compliant_order_001"
        )

    @pytest.fixture
    def large_order(self):
        """Create a large order that should trigger limits."""
        return Order(
            symbol="TSLA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5000.0,
            price=200.0,
            client_order_id="large_order_001"
        )

    @pytest.fixture
    def restricted_order(self):
        """Create an order with restricted symbol."""
        return Order(
            symbol="PENNY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,
            price=0.10,
            client_order_id="restricted_order_001"
        )

    def test_compliance_checker_initialization(self, compliance_checker):
        """Test compliance checker initialization."""
        assert compliance_checker.config.regulatory_regime == RegulatoryRegime.SEC_FINRA
        assert compliance_checker.config.max_position_size_portfolio == 0.015
        assert len(compliance_checker.compliance_rules) > 0
        assert compliance_checker.portfolio_data["total_value"] == 1000000.0

    def test_rule_initialization(self, compliance_checker):
        """Test compliance rules initialization."""
        rule_types = {rule.rule_id for rule in compliance_checker.compliance_rules}

        assert "POS_001" in rule_types  # Portfolio position limit
        assert "POS_002" in rule_types  # Single symbol concentration
        assert "ORD_001" in rule_types  # Maximum order size
        assert "GEN_001" in rule_types  # Order validity

        # Check SEC/FINRA rules are present
        sec_rules = [
            rule for rule in compliance_checker.compliance_rules
            if rule.regulatory_regime == RegulatoryRegime.SEC_FINRA
        ]
        assert len(sec_rules) > 0

    @pytest.mark.asyncio
    async def test_compliant_order_approval(self, compliance_checker, compliant_order):
        """Test that compliant orders are approved."""
        current_positions = {
            "AAPL": {"value": 50000.0, "total_exposure": 50000.0}
        }

        result = await compliance_checker.check_compliance(
            compliant_order, current_positions
        )

        assert isinstance(result, ComplianceResult)
        assert result.compliant is True
        assert result.status == ComplianceStatus.COMPLIANT
        assert result.rules_failed == 0
        assert result.rules_passed > 0
        assert len(result.failed_rules) == 0

    @pytest.mark.asyncio
    async def test_large_order_rejection(self, compliance_checker, large_order):
        """Test that oversized orders are rejected."""
        result = await compliance_checker.check_compliance(large_order)

        assert result.compliant is False
        assert result.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.REQUIRES_REVIEW]
        assert result.rules_failed > 0
        assert len(result.failed_rules) > 0

        # Check that the right rule failed
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        assert "ORD_001" in failed_rule_ids  # Maximum order size rule

    @pytest.mark.asyncio
    async def test_restricted_symbol_rejection(self, compliance_checker, restricted_order):
        """Test that restricted symbols are rejected."""
        result = await compliance_checker.check_compliance(restricted_order)

        assert result.compliant is False
        assert result.rules_failed > 0

        # Check that the right rule failed
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        assert "SYM_001" in failed_rule_ids  # Restricted symbols rule

    @pytest.mark.asyncio
    async def test_position_limit_enforcement(self, compliance_checker):
        """Test position limit enforcement."""
        # Order that would exceed portfolio position limit
        large_portfolio_order = Order(
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000.0,
            price=300.0,  # $300,000 order - would exceed 1.5% of $1M portfolio
            client_order_id="large_portfolio_order_001"
        )

        result = await compliance_checker.check_compliance(large_portfolio_order)

        assert result.compliant is False
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        assert "POS_001" in failed_rule_ids

    @pytest.mark.asyncio
    async def test_symbol_concentration_limit(self, compliance_checker):
        """Test single symbol concentration limit."""
        # Order that would exceed single symbol limit (18% of $1M = $180,000)
        # Current BTC exposure is $150,000, so additional $40,000 should be okay
        # but $50,000 should exceed limit
        concentration_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.5,  # 2.5 BTC at $50,000 = $125,000, would exceed limit
            price=50000.0,
            client_order_id="concentration_order_001"
        )

        current_positions = {
            "BTC": {"value": 150000.0, "total_exposure": 150000.0},
            "ETH": {"value": 100000.0, "total_exposure": 100000.0}
        }

        result = await compliance_checker.check_compliance(concentration_order, current_positions)

        assert result.compliant is False
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        assert "POS_002" in failed_rule_ids

    @pytest.mark.asyncio
    async def test_minimum_order_size(self, compliance_checker):
        """Test minimum order size enforcement."""
        tiny_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=150.0,  # $15 total - below $50 minimum
            client_order_id="tiny_order_001"
        )

        result = await compliance_checker.check_compliance(tiny_order)

        assert result.compliant is False
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        assert "ORD_002" in failed_rule_ids  # Minimum order size rule

    @pytest.mark.asyncio
    async def test_order_validity_checks(self, compliance_checker):
        """Test basic order validity checks."""
        # Invalid symbol
        invalid_symbol_order = Order(
            symbol="",  # Empty symbol
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            price=100.0,
            client_order_id="invalid_symbol_order_001"
        )

        result = await compliance_checker.check_compliance(invalid_symbol_order)

        assert result.compliant is False
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        assert "GEN_001" in failed_rule_ids

        # Invalid quantity
        invalid_quantity_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,  # Zero quantity
            price=100.0,
            client_order_id="invalid_quantity_order_001"
        )

        result = await compliance_checker.check_compliance(invalid_quantity_order)

        assert result.compliant is False

        # Invalid limit price
        invalid_price_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=0,  # Zero price for limit order
            client_order_id="invalid_price_order_001"
        )

        result = await compliance_checker.check_compliance(invalid_price_order)

        assert result.compliant is False

    @pytest.mark.asyncio
    async def test_mifid_ii_regulatory_rules(self):
        """Test MiFID II specific regulatory rules."""
        config = ComplianceConfiguration(
            regulatory_regime=RegulatoryRegime.MIFID_II
        )
        checker = PreTradeChecker(config=config)

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            price=150.0,
            client_order_id="mifid_order_001"
        )

        # Without market data, MiFID II rules should fail
        result = await checker.check_compliance(order)

        # Should have MiFID II rule failures due to missing market data
        failed_rule_ids = {rule["rule_id"] for rule in result.failed_rules}
        mifid_rules = [rule_id for rule_id in failed_rule_ids if rule_id.startswith("MIFID_")]
        assert len(mifid_rules) > 0

    def test_audit_trail_functionality(self, compliance_checker, compliant_order):
        """Test audit trail functionality."""
        # Initial audit trail should be empty
        assert len(compliance_checker.audit_trail) == 0

        # Run compliance check
        result = asyncio.run(compliance_checker.check_compliance(compliant_order))

        # Should have audit trail entry
        assert len(compliance_checker.audit_trail) == 1
        assert compliance_checker.audit_trail[0].order_id == compliant_order.client_order_id
        assert compliance_checker.audit_trail[0].compliant == result.compliant

    def test_compliance_summary(self, compliance_checker, compliant_order):
        """Test compliance summary generation."""
        # Run some compliance checks
        asyncio.run(compliance_checker.check_compliance(compliant_order))

        summary = compliance_checker.get_compliance_summary()

        assert "total_checks" in summary
        assert "compliance_rate" in summary
        assert "average_check_time_ms" in summary
        assert "active_rules" in summary
        assert "regulatory_regime" in summary
        assert summary["regulatory_regime"] == "sec_finra"

    @pytest.mark.asyncio
    async def test_performance_sub_5ms_target(self, compliance_checker, compliant_order):
        """Test that compliance check meets sub-5ms performance target."""
        start_time = time.time()

        result = await compliance_checker.check_compliance(compliant_order)

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Should complete well under 5ms for simple compliance check
        assert duration_ms < 5.0, f"Compliance check took {duration_ms:.2f}ms, expected < 5ms"

    def test_custom_rule_addition(self, compliance_checker):
        """Test adding custom compliance rules."""
        initial_rule_count = len(compliance_checker.compliance_rules)

        custom_rule = ComplianceRule(
            rule_id="CUSTOM_001",
            name="Custom Test Rule",
            description="Test rule for validation",
            regulatory_regime=RegulatoryRegime.SEC_FINRA,
            parameters={"test_param": "test_value"}
        )

        compliance_checker.add_custom_rule(custom_rule)

        assert len(compliance_checker.compliance_rules) == initial_rule_count + 1
        assert custom_rule in compliance_checker.compliance_rules

    def test_rule_status_update(self, compliance_checker):
        """Test updating rule status."""
        # Find a rule to update
        rule_to_update = compliance_checker.compliance_rules[0]
        original_status = rule_to_update.is_active

        # Update status
        compliance_checker.update_rule_status(rule_to_update.rule_id, not original_status)

        # Check that status was updated
        updated_rule = next(
            rule for rule in compliance_checker.compliance_rules
            if rule.rule_id == rule_to_update.rule_id
        )
        assert updated_rule.is_active != original_status

    def test_audit_trail_filtering(self, compliance_checker):
        """Test audit trail filtering functionality."""
        # Create some audit trail entries
        now = datetime.now()
        old_time = now - timedelta(days=10)

        # Manually add some audit entries for testing
        old_result = ComplianceResult(
            order_id="old_order",
            compliant=True,
            status=ComplianceStatus.COMPLIANT,
            rules_checked=5,
            rules_passed=5,
            rules_failed=0,
            failed_rules=[]
        )
        old_result.check_timestamp = old_time

        new_result = ComplianceResult(
            order_id="new_order",
            compliant=True,
            status=ComplianceStatus.COMPLIANT,
            rules_checked=5,
            rules_passed=5,
            rules_failed=0,
            failed_rules=[]
        )
        new_result.check_timestamp = now

        compliance_checker.audit_trail.extend([old_result, new_result])

        # Test date filtering
        recent_trail = compliance_checker.get_audit_trail(start_date=now - timedelta(days=1))
        assert len(recent_trail) == 1
        assert recent_trail[0].order_id == "new_order"

        # Test order ID filtering
        specific_trail = compliance_checker.get_audit_trail(order_id="old_order")
        assert len(specific_trail) == 1
        assert specific_trail[0].order_id == "old_order"

    @pytest.mark.asyncio
    async def test_error_handling(self, compliance_checker):
        """Test error handling in compliance checks."""
        # Test with None order
        result = await compliance_checker.check_compliance(None)

        assert result.compliant is False
        assert result.status == ComplianceStatus.ERROR
        assert len(result.failed_rules) == 1
        assert result.failed_rules[0]["rule_id"] == "SYS_ERROR"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_checks(self, compliance_checker, compliant_order):
        """Test multiple concurrent compliance checks."""
        orders = [
            Order(
                symbol=f"SYM{i:03d}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10.0,
                price=100.0,
                client_order_id=f"concurrent_order_{i:03d}"
            )
            for i in range(10)
        ]

        # Run compliance checks concurrently
        tasks = [
            compliance_checker.check_compliance(order)
            for order in orders
        ]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 10
        assert all(isinstance(result, ComplianceResult) for result in results)

        # Check that audit trail has all entries
        assert len(compliance_checker.audit_trail) == 10


class TestComplianceMonitor:
    """Test cases for Compliance Monitor."""

    @pytest.fixture
    def compliance_monitor(self):
        """Create a test compliance monitor."""
        config = ComplianceMonitorConfiguration(
            monitoring_interval_seconds=1,  # Short interval for testing
            alert_retention_days=30,
            report_generation_schedule="daily"
        )
        portfolio_data = {
            "total_value": 1000000.0,
            "positions": {
                "BTC": {"value": 200000.0},
                "ETH": {"value": 150000.0},
                "AAPL": {"value": 300000.0}
            }
        }
        return ComplianceMonitor(config=config, portfolio_data=portfolio_data)

    def test_compliance_monitor_initialization(self, compliance_monitor):
        """Test compliance monitor initialization."""
        assert compliance_monitor.config.monitoring_interval_seconds == 1
        assert compliance_monitor.config.report_generation_schedule == "daily"
        assert compliance_monitor.portfolio_data["total_value"] == 1000000.0
        assert compliance_monitor.is_monitoring is False

    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, compliance_monitor):
        """Test monitoring start and stop functionality."""
        # Start monitoring
        await compliance_monitor.start_monitoring()
        assert compliance_monitor.is_monitoring is True

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop monitoring
        await compliance_monitor.stop_monitoring()
        assert compliance_monitor.is_monitoring is False

    @pytest.mark.asyncio
    async def test_metrics_update(self, compliance_monitor):
        """Test compliance metrics updating."""
        # Update portfolio data to trigger metric updates
        compliance_monitor.portfolio_data = {
            "total_value": 1200000.0,
            "positions": {
                "BTC": {"value": 400000.0},  # High concentration
                "ETH": {"value": 200000.0},
                "AAPL": {"value": 100000.0}
            },
            "daily_trade_count": 50,
            "execution_quality_score": 0.78
        }

        # Manually trigger metrics update
        await compliance_monitor._update_all_metrics()

        # Check that metrics were updated
        assert len(compliance_monitor.compliance_metrics) > 0

        # Check specific metrics
        if "max_position_concentration" in compliance_monitor.compliance_metrics:
            metric = compliance_monitor.compliance_metrics["max_position_concentration"]
            assert metric.current_value > 0

    @pytest.mark.asyncio
    async def test_position_limits_monitoring(self, compliance_monitor):
        """Test position limits compliance monitoring."""
        # Set up high concentration scenario
        compliance_monitor.portfolio_data = {
            "total_value": 1000000.0,
            "positions": {
                "BTC": {"value": 250000.0}  # 25% concentration - should trigger warning
            }
        }

        # Update metrics
        await compliance_monitor._update_position_limits_metrics()

        # Check metric
        if "max_position_concentration" in compliance_monitor.compliance_metrics:
            metric = compliance_monitor.compliance_metrics["max_position_concentration"]
            assert metric.current_value == 0.25
            assert metric.status in ["warning", "breach"]

    @pytest.mark.asyncio
    async def test_alert_generation(self, compliance_monitor):
        """Test compliance alert generation."""
        # Create a breach scenario
        compliance_monitor.portfolio_data = {
            "total_value": 1000000.0,
            "positions": {
                "BTC": {"value": 300000.0}  # 30% concentration - should breach 20% limit
            }
        }

        # Update metrics
        await compliance_monitor._update_position_limits_metrics()

        # Check for alert conditions
        await compliance_monitor._check_alert_conditions()

        # Should have generated alerts
        assert len(compliance_monitor.active_alerts) > 0

        alert = compliance_monitor.active_alerts[0]
        assert isinstance(alert, ComplianceAlert)
        assert alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        assert alert.acknowledged is False
        assert alert.resolved is False

    @pytest.mark.asyncio
    async def test_alert_acknowledgement(self, compliance_monitor):
        """Test alert acknowledgement functionality."""
        # Generate an alert
        await compliance_monitor._generate_alert(
            metric=ComplianceMetric(
                metric_type=ComplianceMetricType.POSITION_LIMITS,
                name="Test Metric",
                current_value=0.8,
                threshold_value=0.2,
                status="breach"
            ),
            severity=AlertSeverity.HIGH
        )

        alert_id = compliance_monitor.active_alerts[0].alert_id

        # Acknowledge alert
        result = await compliance_monitor.acknowledge_alert(alert_id)

        assert result is True
        assert compliance_monitor.active_alerts[0].acknowledged is True

    @pytest.mark.asyncio
    async def test_alert_resolution(self, compliance_monitor):
        """Test alert resolution functionality."""
        # Generate an alert
        await compliance_monitor._generate_alert(
            metric=ComplianceMetric(
                metric_type=ComplianceMetricType.POSITION_LIMITS,
                name="Test Metric",
                current_value=0.8,
                threshold_value=0.2,
                status="breach"
            ),
            severity=AlertSeverity.HIGH
        )

        alert_id = compliance_monitor.active_alerts[0].alert_id

        # Resolve alert
        result = await compliance_monitor.resolve_alert(alert_id, "Issue resolved")

        assert result is True
        assert compliance_monitor.active_alerts[0].resolved is True
        assert "Issue resolved" in compliance_monitor.active_alerts[0].actions_taken

    def test_compliance_dashboard(self, compliance_monitor):
        """Test compliance dashboard data generation."""
        dashboard = compliance_monitor.get_compliance_dashboard()

        assert "monitoring_status" in dashboard
        assert "compliance_summary" in dashboard
        assert "alerts_summary" in dashboard
        assert "reports_summary" in dashboard
        assert "metrics" in dashboard
        assert "active_alerts" in dashboard

        # Check monitoring status
        assert dashboard["monitoring_status"]["is_monitoring"] == compliance_monitor.is_monitoring
        assert "monitoring_cycles" in dashboard["monitoring_status"]

        # Check compliance summary
        compliance_summary = dashboard["compliance_summary"]
        assert "total_metrics" in compliance_summary
        assert "compliance_rate" in compliance_summary

        # Check alerts summary
        alerts_summary = dashboard["alerts_summary"]
        assert "active_alerts" in alerts_summary
        assert "alerts_by_severity" in alerts_summary

    def test_performance_metrics(self, compliance_monitor):
        """Test performance metrics collection."""
        # Simulate some monitoring activity
        compliance_monitor.monitoring_cycles = 100
        compliance_monitor.alerts_generated = 5
        compliance_monitor.reports_generated = 2

        metrics = compliance_monitor.get_performance_metrics()

        assert "monitoring_cycles" in metrics
        assert "alerts_generated" in metrics
        assert "reports_generated" in metrics
        assert "uptime_percentage" in metrics
        assert "metrics_tracked" in metrics

        assert metrics["monitoring_cycles"] == 100
        assert metrics["alerts_generated"] == 5
        assert metrics["reports_generated"] == 2

    @pytest.mark.asyncio
    async def test_report_generation(self, compliance_monitor):
        """Test periodic report generation."""
        # Set up some metrics and alerts
        compliance_monitor.compliance_metrics = {
            "test_metric": ComplianceMetric(
                metric_type=ComplianceMetricType.POSITION_LIMITS,
                name="Test Metric",
                current_value=0.1,
                threshold_value=0.2,
                status="compliant"
            )
        }

        # Generate an alert
        await compliance_monitor._generate_alert(
            compliance_monitor.compliance_metrics["test_metric"],
            AlertSeverity.MEDIUM
        )

        # Generate report
        await compliance_monitor._generate_periodic_report()

        # Check report was generated
        assert len(compliance_monitor.reports) > 0
        assert compliance_monitor.reports_generated > 0

        report = compliance_monitor.reports[-1]
        assert report.report_type == compliance_monitor.config.report_generation_schedule
        assert len(report.metrics) > 0
        assert report.overall_compliance_score >= 0

    @pytest.mark.asyncio
    async def test_background_monitoring_loop(self, compliance_monitor):
        """Test background monitoring loop functionality."""
        # Start monitoring
        await compliance_monitor.start_monitoring()

        # Update portfolio data to trigger changes
        compliance_monitor.portfolio_data["daily_trade_count"] = 100

        # Let monitoring run for a few cycles
        await asyncio.sleep(0.1)

        # Stop monitoring
        await compliance_monitor.stop_monitoring()

        # Should have completed some monitoring cycles
        assert compliance_monitor.monitoring_cycles > 0

        # Should have updated last update time
        assert compliance_monitor.last_update_time > (datetime.now() - timedelta(seconds=10))

    def test_metric_status_determination(self, compliance_monitor):
        """Test metric status determination logic."""
        # Test compliant status
        status = compliance_monitor._determine_compliance_status(0.1, 0.15, 0.2)
        assert status == "compliant"

        # Test warning status
        status = compliance_monitor._determine_compliance_status(0.175, 0.15, 0.2)
        assert status == "warning"

        # Test breach status
        status = compliance_monitor._determine_compliance_status(0.25, 0.15, 0.2)
        assert status == "breach"

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, compliance_monitor):
        """Test cleanup of old data."""
        # Add some old alerts
        old_timestamp = datetime.now() - timedelta(days=100)
        old_alert = ComplianceAlert(
            alert_id="old_alert",
            metric_type=ComplianceMetricType.POSITION_LIMITS,
            severity=AlertSeverity.LOW,
            title="Old Alert",
            description="Old alert test",
            current_value=0.1,
            threshold_value=0.2
        )
        old_alert.timestamp = old_timestamp

        recent_alert = ComplianceAlert(
            alert_id="recent_alert",
            metric_type=ComplianceMetricType.POSITION_LIMITS,
            severity=AlertSeverity.LOW,
            title="Recent Alert",
            description="Recent alert test",
            current_value=0.1,
            threshold_value=0.2
        )

        compliance_monitor.alert_history.extend([old_alert, recent_alert])

        # Run cleanup
        await compliance_monitor._cleanup_old_data()

        # Old alert should be removed, recent alert should remain
        assert len(compliance_monitor.alert_history) == 1
        assert compliance_monitor.alert_history[0].alert_id == "recent_alert"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])