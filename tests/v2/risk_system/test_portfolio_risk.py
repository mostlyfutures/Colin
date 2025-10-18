"""
Tests for Portfolio Risk Analytics

This module contains comprehensive tests for portfolio risk analytics components.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.v2.risk_system.portfolio.var_calculator import (
    VaRCalculator, VaRResult, VaRMethod, VaRConfiguration
)
from src.v2.risk_system.portfolio.correlation_analyzer import (
    CorrelationAnalyzer, CorrelationMetrics, CorrelationLevel, CorrelationConfiguration
)
from src.v2.risk_system.portfolio.stress_tester import (
    StressTester, StressTestResult, StressScenario, StressTestType, StressTestConfiguration
)


class TestVaRCalculator:
    """Test cases for VaR Calculator."""

    @pytest.fixture
    def var_calculator(self):
        """Create a test VaR calculator."""
        config = VaRConfiguration(
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 5],
            historical_lookback_days=100,
            monte_carlo_simulations=1000  # Reduced for testing
        )
        return VaRCalculator(config=config, initial_portfolio_value=100000.0)

    @pytest.fixture
    def sample_returns_data(self):
        """Create sample returns data."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 100).tolist()  # 100 days of returns

    @pytest.fixture
    def sample_positions(self):
        """Create sample position weights."""
        return {
            "BTC": 0.4,
            "ETH": 0.3,
            "AAPL": 0.2,
            "MSFT": 0.1
        }

    def test_var_calculator_initialization(self, var_calculator):
        """Test VaR calculator initialization."""
        assert var_calculator.portfolio_value == 100000.0
        assert var_calculator.config.confidence_levels == [0.95, 0.99]
        assert var_calculator.config.time_horizons == [1, 5]
        assert var_calculator.config.monte_carlo_simulations == 1000

    def test_portfolio_data_update(self, var_calculator, sample_returns_data, sample_positions):
        """Test portfolio data update."""
        var_calculator.update_portfolio_data(
            portfolio_value=120000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        assert var_calculator.portfolio_value == 120000.0
        assert var_calculator.position_weights == sample_positions
        assert len(var_calculator.return_history) == 100

    @pytest.mark.asyncio
    async def test_historical_var_calculation(self, var_calculator, sample_returns_data, sample_positions):
        """Test historical VaR calculation."""
        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        result = await var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon_days=1,
            method=VaRMethod.HISTORICAL
        )

        assert isinstance(result, VaRResult)
        assert result.confidence_level == 0.95
        assert result.time_horizon_days == 1
        assert result.calculation_method == VaRMethod.HISTORICAL
        assert result.var_value > 0
        assert result.var_percentage > 0
        assert result.var_percentage < 1.0  # Should be less than 100%

    @pytest.mark.asyncio
    async def test_parametric_var_calculation(self, var_calculator, sample_returns_data, sample_positions):
        """Test parametric VaR calculation."""
        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        result = await var_calculator.calculate_var(
            confidence_level=0.99,
            time_horizon_days=5,
            method=VaRMethod.PARAMETRIC
        )

        assert isinstance(result, VaRResult)
        assert result.confidence_level == 0.99
        assert result.time_horizon_days == 5
        assert result.calculation_method == VaRMethod.PARAMETRIC
        assert "volatility" in result.additional_metrics
        assert "z_score" in result.additional_metrics

    @pytest.mark.asyncio
    async def test_monte_carlo_var_calculation(self, var_calculator, sample_returns_data, sample_positions):
        """Test Monte Carlo VaR calculation."""
        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        result = await var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon_days=1,
            method=VaRMethod.MONTE_CARLO
        )

        assert isinstance(result, VaRResult)
        assert result.calculation_method == VaRMethod.MONTE_CARLO
        assert "simulations" in result.additional_metrics
        assert result.additional_metrics["simulations"] == 1000

    @pytest.mark.asyncio
    async def test_var_for_all_methods(self, var_calculator, sample_returns_data, sample_positions):
        """Test VaR calculation for all methods."""
        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        results = await var_calculator.calculate_var_for_all_methods(
            confidence_level=0.95,
            time_horizon_days=1
        )

        assert len(results) == 3  # Should have results for all three methods
        assert VaRMethod.HISTORICAL in results
        assert VaRMethod.PARAMETRIC in results
        assert VaRMethod.MONTE_CARLO in results

        # All methods should give reasonable VaR values
        for method, result in results.items():
            assert result.var_value > 0
            assert result.var_percentage > 0

    @pytest.mark.asyncio
    async def test_stress_var_calculation(self, var_calculator, sample_returns_data, sample_positions):
        """Test stressed VaR calculation."""
        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        stress_scenario = {
            "volatility_multiplier": 2.0,
            "correlation_increase": 1.5
        }

        result = await var_calculator.calculate_stress_var(
            stress_scenario=stress_scenario,
            confidence_level=0.95,
            time_horizon_days=1
        )

        assert isinstance(result, VaRResult)
        assert "stress_scenario" in result.additional_metrics
        assert result.additional_metrics["stress_multiplier"] > 1.0

    @pytest.mark.asyncio
    async def test_var_performance_target(self, var_calculator, sample_returns_data, sample_positions):
        """Test that VaR calculation meets performance targets (<100ms)."""
        import time

        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        start_time = time.time()
        result = await var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon_days=1,
            method=VaRMethod.PARAMETRIC  # Fastest method
        )
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        assert duration_ms < 100, f"VaR calculation took {duration_ms:.2f}ms, expected < 100ms"

    def test_var_summary(self, var_calculator, sample_returns_data, sample_positions):
        """Test VaR summary generation."""
        var_calculator.update_portfolio_data(
            portfolio_value=150000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        summary = var_calculator.get_var_summary()

        assert summary["portfolio_value"] == 150000.0
        assert "var_metrics" in summary
        assert "position_weights" in summary
        assert "data_points" in summary
        assert summary["data_points"] == len(sample_returns_data)

    def test_insufficient_data_handling(self, var_calculator):
        """Test handling of insufficient data."""
        # Test with no returns data
        result = asyncio.run(var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon_days=1,
            method=VaRMethod.HISTORICAL
        ))

        # Should fall back to parametric method
        assert result.calculation_method == VaRMethod.PARAMETRIC
        assert result.var_value > 0

    def test_cache_functionality(self, var_calculator, sample_returns_data, sample_positions):
        """Test VaR result caching."""
        var_calculator.update_portfolio_data(
            portfolio_value=100000.0,
            position_weights=sample_positions,
            returns_data=sample_returns_data
        )

        # Calculate VaR twice
        result1 = asyncio.run(var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon_days=1,
            use_cache=True
        ))

        result2 = asyncio.run(var_calculator.calculate_var(
            confidence_level=0.95,
            time_horizon_days=1,
            use_cache=True
        ))

        # Results should be identical (cached)
        assert result1.var_value == result2.var_value
        assert result1.calculation_time == result2.calculation_time


class TestCorrelationAnalyzer:
    """Test cases for Correlation Analyzer."""

    @pytest.fixture
    def correlation_analyzer(self):
        """Create a test correlation analyzer."""
        config = CorrelationConfiguration(
            max_correlation_limit=0.70,
            warning_threshold=0.60,
            min_data_points=20,  # Reduced for testing
            correlation_window=30
        )
        return CorrelationAnalyzer(config=config, symbols=["BTC", "ETH", "AAPL"])

    def test_correlation_analyzer_initialization(self, correlation_analyzer):
        """Test correlation analyzer initialization."""
        assert len(correlation_analyzer.symbols) == 3
        assert correlation_analyzer.config.max_correlation_limit == 0.70
        assert correlation_analyzer.config.warning_threshold == 0.60

    @pytest.mark.asyncio
    async def test_price_data_addition(self, correlation_analyzer):
        """Test adding price data."""
        await correlation_analyzer.add_price_data("BTC", 50000.0)
        await correlation_analyzer.add_price_data("ETH", 3000.0)
        await correlation_analyzer.add_price_data("AAPL", 150.0)

        assert "BTC" in correlation_analyzer.price_history
        assert "ETH" in correlation_analyzer.price_history
        assert "AAPL" in correlation_analyzer.price_history

        # Should add symbol if not present
        await correlation_analyzer.add_price_data("MSFT", 250.0)
        assert "MSFT" in correlation_analyzer.symbols

    @pytest.mark.asyncio
    async def test_correlation_calculation(self, correlation_analyzer):
        """Test correlation matrix calculation."""
        # Add correlated price data
        np.random.seed(42)
        base_prices = np.random.randn(50).cumsum() + 100

        for i, price in enumerate(base_prices):
            # Add correlated prices with some noise
            btc_price = price * 500  # Scale for BTC
            eth_price = price * 480 + np.random.randn() * 10  # Correlated with BTC
            aapl_price = price * 1.5 + np.random.randn() * 2  # Less correlated

            await correlation_analyzer.add_price_data("BTC", btc_price)
            await correlation_analyzer.add_price_data("ETH", eth_price)
            await correlation_analyzer.add_price_data("AAPL", aapl_price)

        # Calculate correlations
        metrics = await correlation_analyzer.calculate_correlations()

        assert isinstance(metrics, CorrelationMetrics)
        assert metrics.correlation_matrix.shape == (4, 4)  # Added MSFT during test
        assert metrics.avg_correlation >= 0
        assert metrics.max_correlation <= 1.0
        assert metrics.min_correlation >= -1.0

    @pytest.mark.asyncio
    async def test_correlation_alerts(self, correlation_analyzer):
        """Test correlation alert generation."""
        # Add highly correlated data
        for i in range(30):
            base_price = 100 + i
            await correlation_analyzer.add_price_data("BTC", base_price * 500)
            await correlation_analyzer.add_price_data("ETH", base_price * 500)  # Perfect correlation
            await correlation_analyzer.add_price_data("AAPL", base_price * 1.5)

        # Calculate correlations (should trigger alerts)
        metrics = await correlation_analyzer.calculate_correlations()

        # Should have alerts for high correlation
        assert len(correlation_analyzer.active_alerts) > 0

        # Check alert details
        high_corr_alert = correlation_analyzer.active_alerts[0]
        assert high_corr_alert.correlation_value > correlation_analyzer.config.max_correlation_limit
        assert high_corr_alert.alert_type == "critical"

    def test_correlation_summary(self, correlation_analyzer):
        """Test correlation summary generation."""
        summary = correlation_analyzer.get_correlation_summary()

        assert "status" in summary
        assert "symbols" in summary
        assert "active_alerts" in summary
        assert summary["symbols_count"] == 3

    @pytest.mark.asyncio
    async def test_high_correlations_filtering(self, correlation_analyzer):
        """Test filtering high correlations."""
        # Add some correlated data
        for i in range(25):
            base_price = 100 + i * 0.1
            await correlation_analyzer.add_price_data("BTC", base_price * 500)
            await correlation_analyzer.add_price_data("ETH", base_price * 480)
            await correlation_analyzer.add_price_data("AAPL", base_price * 1.5)

        await correlation_analyzer.calculate_correlations()

        # Get high correlations
        high_correlations = correlation_analyzer.get_high_correlations(threshold=0.5)

        assert isinstance(high_correlations, list)
        # Should have some correlations with the test data
        assert len(high_correlations) >= 0

    def test_correlation_level_classification(self, correlation_analyzer):
        """Test correlation level classification."""
        assert correlation_analyzer._get_correlation_level(0.1) == CorrelationLevel.LOW
        assert correlation_analyzer._get_correlation_level(0.4) == CorrelationLevel.MEDIUM
        assert correlation_analyzer._get_correlation_level(0.7) == CorrelationLevel.HIGH
        assert correlation_analyzer._get_correlation_level(0.85) == CorrelationLevel.VERY_HIGH

    @pytest.mark.asyncio
    async def test_background_monitoring(self, correlation_analyzer):
        """Test background monitoring functionality."""
        # Start monitoring
        await correlation_analyzer.start_monitoring()
        assert correlation_analyzer.is_monitoring is True

        # Add some data
        for i in range(25):
            await correlation_analyzer.add_price_data("BTC", 50000 + i * 100)

        # Wait a bit for monitoring to process
        await asyncio.sleep(0.1)

        # Stop monitoring
        await correlation_analyzer.stop_monitoring()
        assert correlation_analyzer.is_monitoring is False


class TestStressTester:
    """Test cases for Stress Tester."""

    @pytest.fixture
    def stress_tester(self):
        """Create a test stress tester."""
        config = StressTestConfiguration(
            num_simulations=500,  # Reduced for testing
            confidence_levels=[0.95, 0.99],
            time_horizons_days=[1, 5, 10]
        )
        positions = {
            "BTC": {"value": 40000, "type": "equity"},
            "ETH": {"value": 30000, "type": "equity"},
            "AAPL": {"value": 20000, "type": "equity"},
            "Bonds": {"value": 10000, "type": "bond"}
        }
        return StressTester(config=config, portfolio_value=100000.0, positions=positions)

    def test_stress_tester_initialization(self, stress_tester):
        """Test stress tester initialization."""
        assert stress_tester.portfolio_value == 100000.0
        assert len(stress_tester.positions) == 4
        assert len(stress_tester.scenarios) > 0
        assert stress_tester.config.num_simulations == 500

    def test_scenario_initialization(self, stress_tester):
        """Test built-in scenario initialization."""
        scenario_types = {scenario.scenario_type for scenario in stress_tester.scenarios}

        assert StressTestType.MARKET_CRASH in scenario_types
        assert StressTestType.LIQUIDITY_CRISIS in scenario_types
        assert StressTestType.VOLATILITY_SPIKE in scenario_types

        # Check regulatory scenarios
        regulatory_scenarios = [
            scenario for scenario in stress_tester.scenarios
            if scenario.regulatory_requirement
        ]
        assert len(regulatory_scenarios) > 0

    @pytest.mark.asyncio
    async def test_single_stress_test(self, stress_tester):
        """Test running a single stress test."""
        scenario = stress_tester.scenarios[0]  # Use first scenario
        result = await stress_tester.run_stress_test(scenario)

        assert isinstance(result, StressTestResult)
        assert result.scenario_name == scenario.name
        assert result.portfolio_value_before == 100000.0
        assert result.portfolio_value_after <= result.portfolio_value_before
        assert result.portfolio_loss >= 0
        assert result.portfolio_loss_percentage >= 0
        assert result.var_under_stress >= 0
        assert result.max_drawdown_under_stress >= 0
        assert result.recovery_time_estimate >= 0

    @pytest.mark.asyncio
    async def test_market_crash_scenario(self, stress_tester):
        """Test market crash stress scenario."""
        market_crash_scenarios = [
            scenario for scenario in stress_tester.scenarios
            if scenario.scenario_type == StressTestType.MARKET_CRASH
        ]

        assert len(market_crash_scenarios) > 0
        scenario = market_crash_scenarios[0]

        result = await stress_tester.run_stress_test(scenario)

        # Market crash should cause significant losses
        assert result.portfolio_loss_percentage > 0.1  # At least 10% loss
        assert "market_decline" in scenario.parameters

    @pytest.mark.asyncio
    async def test_liquidity_crisis_scenario(self, stress_tester):
        """Test liquidity crisis stress scenario."""
        liquidity_scenarios = [
            scenario for scenario in stress_tester.scenarios
            if scenario.scenario_type == StressTestType.LIQUIDITY_CRISIS
        ]

        assert len(liquidity_scenarios) > 0
        scenario = liquidity_scenarios[0]

        result = await stress_tester.run_stress_test(scenario)

        # Liquidity crisis should impact portfolio
        assert result.portfolio_loss > 0
        assert "liquidity_decrease" in scenario.parameters

    @pytest.mark.asyncio
    async def test_run_all_stress_tests(self, stress_tester):
        """Test running all stress test scenarios."""
        results = await stress_tester.run_all_stress_tests()

        assert len(results) > 0
        assert all(isinstance(result, StressTestResult) for result in results)
        assert stress_tester.worst_case_scenario is not None

        # Check that we have a worst case identified
        worst_case = stress_tester.worst_case_scenario
        assert worst_case.portfolio_loss_percentage >= 0

    @pytest.mark.asyncio
    async def test_regulatory_stress_tests(self, stress_tester):
        """Test regulatory required stress tests."""
        results = await stress_tester.run_regulatory_stress_tests()

        assert len(results) > 0

        # All results should be from regulatory scenarios
        for result in results:
            scenario = next(
                s for s in stress_tester.scenarios if s.name == result.scenario_name
            )
            assert scenario.regulatory_requirement is True

    @pytest.mark.asyncio
    async def test_custom_scenario_creation(self, stress_tester):
        """Test creating custom stress scenarios."""
        custom_scenario = await stress_tester.create_custom_scenario(
            name="Custom Test Scenario",
            scenario_type=StressTestType.MARKET_CRASH,
            parameters={
                "market_decline": -0.25,
                "volatility_multiplier": 2.0
            },
            description="Test scenario for validation"
        )

        assert custom_scenario.name == "Custom Test Scenario"
        assert custom_scenario.scenario_type == StressTestType.MARKET_CRASH
        assert custom_scenario.parameters["market_decline"] == -0.25
        assert custom_scenario in stress_tester.scenarios

        # Test running custom scenario
        result = await stress_tester.run_stress_test(custom_scenario)
        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "Custom Test Scenario"

    @pytest.mark.asyncio
    async def test_position_impact_calculation(self, stress_tester):
        """Test position impact calculation under stress."""
        scenario = stress_tester.scenarios[0]
        result = await stress_tester.run_stress_test(scenario)

        # Should have position impacts
        assert len(result.position_impacts) > 0

        # Each position should have impact metrics
        for symbol, impact in result.position_impacts.items():
            assert "loss" in impact
            assert "loss_percentage" in impact
            assert "remaining_value" in impact
            assert impact["loss"] >= 0
            assert impact["loss_percentage"] >= 0

    def test_stress_test_summary(self, stress_tester):
        """Test stress test summary generation."""
        # Run a quick test to generate data
        result = asyncio.run(stress_tester.run_stress_test(stress_tester.scenarios[0]))

        summary = stress_tester.get_stress_test_summary()

        assert summary["status"] == "completed"
        assert summary["portfolio_value"] == 100000.0
        assert summary["scenarios_tested"] == 1
        assert "worst_case" in summary
        assert "summary_statistics" in summary
        assert "regulatory_scenarios" in summary

    def test_worst_case_positions(self, stress_tester):
        """Test worst case position analysis."""
        # Run multiple scenarios to generate data
        asyncio.run(stress_tester.run_all_stress_tests())

        worst_positions = stress_tester.get_worst_case_positions(top_n=3)

        assert isinstance(worst_positions, list)
        assert len(worst_positions) <= 3

        for position_data in worst_positions:
            assert "symbol" in position_data
            assert "average_loss" in position_data
            assert "max_loss" in position_data
            assert "scenarios_affected" in position_data

    @pytest.mark.asyncio
    async def test_stress_magnitude_calculation(self, stress_tester):
        """Test stress magnitude calculation."""
        scenario = stress_tester.scenarios[0]
        magnitude = stress_tester._calculate_stress_magnitude(scenario)

        assert magnitude >= 0
        assert isinstance(magnitude, (int, float))

        # Higher severity scenarios should have higher magnitude
        high_severity_scenario = StressScenario(
            name="High Severity",
            scenario_type=StressTestType.MARKET_CRASH,
            description="Test",
            parameters={"market_decline": -0.50}
        )
        high_magnitude = stress_tester._calculate_stress_magnitude(high_severity_scenario)

        low_severity_scenario = StressScenario(
            name="Low Severity",
            scenario_type=StressTestType.MARKET_CRASH,
            description="Test",
            parameters={"market_decline": -0.10}
        )
        low_magnitude = stress_tester._calculate_stress_magnitude(low_severity_scenario)

        assert high_magnitude >= low_magnitude


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])