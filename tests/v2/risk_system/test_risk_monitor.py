"""
Tests for Real-Time Risk Monitoring System

This module contains comprehensive tests for the real-time risk monitoring components.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

from src.v2.risk_system.real_time.risk_monitor import (
    RealTimeRiskController, RiskDecision, RiskLevel, RiskLimits
)
from src.v2.execution_engine.smart_routing.router import Order, OrderSide, OrderType


class TestRealTimeRiskController:
    """Test cases for RealTimeRiskController."""

    @pytest.fixture
    def risk_controller(self):
        """Create a test risk controller."""
        limits = RiskLimits(
            max_position_size_usd=50000.0,
            max_portfolio_exposure=0.15,
            max_leverage=2.5,
            max_correlation_exposure=0.65,
            max_drawdown_hard=0.04,
            max_drawdown_warning=0.025,
            var_limit_95_1d=0.015,
            var_limit_99_5d=0.045
        )
        return RealTimeRiskController(risk_limits=limits)

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        return Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=50000.0,
            client_order_id="test_order_001"
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample current positions."""
        return {
            "BTC/USDT": {
                "quantity": 2.0,
                "value_usd": 100000.0,
                "side": "long"
            },
            "ETH/USDT": {
                "quantity": 10.0,
                "value_usd": 30000.0,
                "side": "long"
            }
        }

    @pytest.fixture
    def sample_portfolio_metrics(self):
        """Create sample portfolio metrics."""
        return {
            "total_value": 200000.0,
            "current_drawdown": 0.02,
            "portfolio_var_95_1d": 0.015,
            "portfolio_var_99_5d": 0.04,
            "correlation_matrix": {
                "BTC/USDT": {"ETH/USDT": 0.6}
            }
        }

    @pytest.mark.asyncio
    async def test_validate_trade_approval(self, risk_controller, sample_order, sample_positions, sample_portfolio_metrics):
        """Test that valid trades are approved."""
        decision = await risk_controller.validate_trade(
            sample_order, sample_positions, sample_portfolio_metrics
        )

        assert decision.approved is True
        assert decision.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert 0 <= decision.risk_score < 50
        assert len(decision.required_modifications) == 0

    @pytest.mark.asyncio
    async def test_validate_trade_rejection_position_size(self, risk_controller, sample_positions, sample_portfolio_metrics):
        """Test that oversized trades are rejected."""
        large_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,  # Large position
            price=50000.0,
            client_order_id="large_order_001"
        )

        decision = await risk_controller.validate_trade(
            large_order, sample_positions, sample_portfolio_metrics
        )

        assert decision.approved is False
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.risk_score >= 60
        assert "exceeds maximum" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_validate_trade_portfolio_exposure_limit(self, risk_controller, sample_positions, sample_portfolio_metrics):
        """Test portfolio exposure limit enforcement."""
        # Create high exposure portfolio
        high_exposure_positions = {
            "BTC/USDT": {
                "quantity": 8.0,
                "value_usd": 400000.0,
                "side": "long"
            }
        }

        high_exposure_metrics = {
            "total_value": 500000.0,
            "current_drawdown": 0.01,
            "portfolio_var_95_1d": 0.01,
            "portfolio_var_99_5d": 0.025
        }

        additional_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0,
            price=50000.0,
            client_order_id="additional_order_001"
        )

        decision = await risk_controller.validate_trade(
            additional_order, high_exposure_positions, high_exposure_metrics
        )

        assert decision.approved is False
        assert "exposure" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_validate_trade_drawdown_warning(self, risk_controller, sample_order, sample_positions):
        """Test drawdown warning levels."""
        high_drawdown_metrics = {
            "total_value": 200000.0,
            "current_drawdown": 0.035,  # Above warning threshold
            "portfolio_var_95_1d": 0.02,
            "portfolio_var_99_5d": 0.05
        }

        decision = await risk_controller.validate_trade(
            sample_order, sample_positions, high_drawdown_metrics
        )

        # Should be approved but with warnings
        assert decision.approved is True
        assert len(decision.warnings) > 0
        assert any("drawdown" in warning.lower() for warning in decision.warnings)

    @pytest.mark.asyncio
    async def test_validate_trade_drawdown_circuit_breaker(self, risk_controller, sample_order, sample_positions):
        """Test circuit breaker activation on critical drawdown."""
        critical_drawdown_metrics = {
            "total_value": 200000.0,
            "current_drawdown": 0.045,  # Near hard limit
            "portfolio_var_95_1d": 0.03,
            "portfolio_var_99_5d": 0.07
        }

        decision = await risk_controller.validate_trade(
            sample_order, sample_positions, critical_drawdown_metrics
        )

        assert decision.approved is False
        assert decision.risk_level == RiskLevel.CRITICAL
        assert "circuit breaker" in decision.reasoning.lower()
        assert risk_controller.circuit_breaker_active is True

    @pytest.mark.asyncio
    async def test_validate_trade_correlation_limit(self, risk_controller, sample_order, sample_positions):
        """Test correlation limit enforcement."""
        high_correlation_metrics = {
            "total_value": 200000.0,
            "current_drawdown": 0.01,
            "portfolio_var_95_1d": 0.015,
            "portfolio_var_99_5d": 0.04,
            "correlation_matrix": {
                "BTC/USDT": {"ETH/USDT": 0.8}  # Above limit
            }
        }

        decision = await risk_controller.validate_trade(
            sample_order, sample_positions, high_correlation_metrics
        )

        assert decision.approved is False
        assert "correlation" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_validate_trade_var_limits(self, risk_controller, sample_order, sample_positions):
        """Test VaR limit enforcement."""
        high_var_metrics = {
            "total_value": 200000.0,
            "current_drawdown": 0.01,
            "portfolio_var_95_1d": 0.025,  # Above limit
            "portfolio_var_99_5d": 0.06,
            "correlation_matrix": {
                "BTC/USDT": {"ETH/USDT": 0.5}
            }
        }

        decision = await risk_controller.validate_trade(
            sample_order, sample_positions, high_var_metrics
        )

        assert decision.approved is False
        assert "var" in decision.reasoning.lower()

    def test_risk_level_determination(self, risk_controller):
        """Test risk level determination logic."""
        # Test low risk
        assert risk_controller._determine_risk_level(10, []) == RiskLevel.LOW

        # Test medium risk
        assert risk_controller._determine_risk_level(35, ["warning"]) == RiskLevel.MEDIUM

        # Test high risk
        assert risk_controller._determine_risk_level(65, ["warning1", "warning2"]) == RiskLevel.HIGH

        # Test critical risk
        assert risk_controller._determine_risk_level(85, ["warning1", "warning2", "warning3"]) == RiskLevel.CRITICAL

    def test_circuit_breaker_operations(self, risk_controller):
        """Test circuit breaker functionality."""
        # Test triggering circuit breaker
        risk_controller._trigger_circuit_breaker("Test reason")
        assert risk_controller.circuit_breaker_active is True
        assert risk_controller.circuit_breaker_reason == "Test reason"

        # Test resetting circuit breaker
        risk_controller.reset_circuit_breaker()
        assert risk_controller.circuit_breaker_active is False
        assert risk_controller.circuit_breaker_reason == ""

    def test_risk_metrics_collection(self, risk_controller, sample_order, sample_positions, sample_portfolio_metrics):
        """Test risk metrics collection."""
        # Simulate some decisions
        for _ in range(10):
            risk_controller.validation_count += 1
            risk_controller.validation_latency_ms.append(5.0)

        metrics = risk_controller.get_risk_metrics()

        assert metrics["validation_count"] == 10
        assert metrics["average_latency_ms"] == 5.0
        assert "circuit_breaker_active" in metrics
        assert "current_drawdown" in metrics
        assert "risk_score_distribution" in metrics
        assert "risk_level_distribution" in metrics

    @pytest.mark.asyncio
    async def test_performance_sub_5ms_target(self, risk_controller, sample_order, sample_positions, sample_portfolio_metrics):
        """Test that validation meets sub-5ms performance target."""
        start_time = time.time()

        await risk_controller.validate_trade(
            sample_order, sample_positions, sample_portfolio_metrics
        )

        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Should complete well under 5ms for simple validation
        assert duration_ms < 5.0, f"Validation took {duration_ms:.2f}ms, expected < 5ms"

    @pytest.mark.asyncio
    async def test_error_handling(self, risk_controller, sample_order):
        """Test error handling in risk validation."""
        # Test with None positions (should handle gracefully)
        decision = await risk_controller.validate_trade(sample_order, None, {})

        assert decision.approved is False
        assert decision.risk_level == RiskLevel.CRITICAL
        assert "error" in decision.reasoning.lower()

    def test_decision_recording(self, risk_controller):
        """Test that decisions are properly recorded."""
        decision = RiskDecision(
            approved=True,
            risk_level=RiskLevel.LOW,
            reasoning="Test decision",
            risk_score=10.0
        )

        risk_controller._record_decision(decision)

        assert len(risk_controller.risk_decisions_history) == 1
        assert risk_controller.risk_decisions_history[0]["approved"] is True
        assert risk_controller.risk_decisions_history[0]["risk_score"] == 10.0

    @pytest.mark.asyncio
    async def test_multiple_validations(self, risk_controller, sample_positions, sample_portfolio_metrics):
        """Test multiple consecutive validations."""
        decisions = []

        for i in range(5):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
                price=50000.0,
                client_order_id=f"test_order_{i:03d}"
            )

            decision = await risk_controller.validate_trade(
                order, sample_positions, sample_portfolio_metrics
            )
            decisions.append(decision)

        # All should be approved (small orders)
        approved_count = sum(1 for d in decisions if d.approved)
        assert approved_count == 5

        # Check rejection rate calculation
        assert risk_controller.rejection_rate == 0.0

    def test_configuration_validation(self, risk_controller):
        """Test configuration parameter validation."""
        limits = risk_controller.risk_limits

        # Test valid configuration
        assert limits.max_position_size_usd > 0
        assert 0 < limits.max_portfolio_exposure <= 1
        assert limits.max_leverage > 0
        assert 0 < limits.max_drawdown_hard <= 1

    def test_edge_cases(self, risk_controller):
        """Test edge cases and boundary conditions."""
        # Test zero portfolio value
        metrics_zero_portfolio = {
            "total_value": 0,
            "current_drawdown": 0
        }

        # Should handle gracefully without division by zero
        assert risk_controller._validate_portfolio_exposure(
            Mock(), metrics_zero_portfolio
        )["approved"] is True

        # Test empty correlation matrix
        metrics_empty_correlation = {
            "total_value": 100000,
            "current_drawdown": 0,
            "correlation_matrix": {}
        }

        # Should handle empty correlation matrix
        result = risk_controller._validate_correlation(
            Mock(symbol="BTC/USDT"), metrics_empty_correlation
        )
        assert result["approved"] is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])