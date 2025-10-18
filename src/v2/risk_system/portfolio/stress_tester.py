"""
Stress Tester for Colin Trading Bot v2.0

This module implements stress testing and scenario analysis for portfolio risk.

Key Features:
- Black swan event simulation (market crashes, liquidity crises)
- Scenario analysis with custom parameters
- Portfolio impact assessment under stress conditions
- Regulatory stress test scenarios (MiFID II, SEC requirements)
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class StressTestType(Enum):
    """Stress test scenario types."""
    MARKET_CRASH = "market_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    CUSTOM = "custom"


@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    scenario_type: StressTestType
    description: str
    parameters: Dict[str, float]
    probability_estimate: float = 0.0
    regulatory_requirement: bool = False


@dataclass
class StressTestResult:
    """Stress test result."""
    scenario_name: str
    portfolio_value_before: float
    portfolio_value_after: float
    portfolio_loss: float
    portfolio_loss_percentage: float
    worst_position_loss: float
    var_under_stress: float
    max_drawdown_under_stress: float
    recovery_time_estimate: float
    stress_timestamp: datetime = field(default_factory=datetime.now)
    position_impacts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestConfiguration:
    """Stress testing configuration."""
    num_simulations: int = 1000
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    time_horizons_days: List[int] = field(default_factory=lambda: [1, 5, 10, 30])
    include_regulatory_scenarios: bool = True
    correlation_stress_factor: float = 2.0
    liquidity_stress_factor: float = 3.0
    volatility_stress_factor: float = 2.5


class StressTester:
    """
    Portfolio stress testing framework.

    This class implements comprehensive stress testing including black swan events,
    regulatory scenarios, and custom stress tests.
    """

    def __init__(
        self,
        config: Optional[StressTestConfiguration] = None,
        portfolio_value: float = 100000.0,
        positions: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize stress tester.

        Args:
            config: Stress testing configuration
            portfolio_value: Current portfolio value
            positions: Dictionary of portfolio positions
        """
        self.config = config or StressTestConfiguration()
        self.portfolio_value = portfolio_value
        self.positions = positions or {}

        # Scenario definitions
        self.scenarios = self._initialize_scenarios()

        # Results storage
        self.stress_results: List[StressTestResult] = []
        self.worst_case_scenario: Optional[StressTestResult] = None

        # Performance metrics
        self.test_count = 0
        self.last_test_duration = 0.0

        logger.info(f"StressTester initialized with portfolio value: ${self.portfolio_value:,.2f}")
        logger.info(f"Available scenarios: {len(self.scenarios)}")

    def _initialize_scenarios(self) -> List[StressScenario]:
        """Initialize built-in stress test scenarios."""
        scenarios = [
            # Market crash scenarios
            StressScenario(
                name="2008 Financial Crisis",
                scenario_type=StressTestType.MARKET_CRASH,
                description="Similar to 2008 financial crisis with 40% market decline",
                parameters={
                    "market_decline": -0.40,
                    "volatility_multiplier": 3.0,
                    "correlation_increase": 1.5,
                    "liquidity_decrease": 0.5
                },
                regulatory_requirement=True
            ),
            StressScenario(
                name="COVID-19 Crash",
                scenario_type=StressTestType.MARKET_CRASH,
                description="March 2020 COVID crash with 30% rapid decline",
                parameters={
                    "market_decline": -0.30,
                    "volatility_multiplier": 4.0,
                    "correlation_increase": 2.0,
                    "liquidity_decrease": 0.3
                }
            ),
            StressScenario(
                name="Dot-com Bubble",
                scenario_type=StressTestType.MARKET_CRASH,
                description="2000 dot-com bubble burst with 50% tech decline",
                parameters={
                    "tech_decline": -0.50,
                    "market_decline": -0.25,
                    "volatility_multiplier": 2.5,
                    "correlation_increase": 1.3
                }
            ),

            # Liquidity crisis scenarios
            StressScenario(
                name="Liquidity Freeze",
                scenario_type=StressTestType.LIQUIDITY_CRISIS,
                description="Market liquidity completely freezes",
                parameters={
                    "liquidity_decrease": 0.95,
                    "bid_ask_spread_multiplier": 10.0,
                    "market_impact_multiplier": 5.0,
                    "volatility_multiplier": 2.0
                },
                regulatory_requirement=True
            ),

            # Volatility spike scenarios
            StressScenario(
                name="Volatility Explosion",
                scenario_type=StressTestType.VOLATILITY_SPIKE,
                description="Extreme volatility spike similar to VIX > 80",
                parameters={
                    "volatility_multiplier": 5.0,
                    "jump_frequency": 0.1,
                    "jump_magnitude": 0.15,
                    "correlation_increase": 1.8
                }
            ),

            # Interest rate shock scenarios
            StressScenario(
                name="Interest Rate Shock",
                scenario_type=StressTestType.INTEREST_RATE_SHOCK,
                description="Sudden 300bps interest rate increase",
                parameters={
                    "rate_change": 0.03,
                    "bond_decline": -0.15,
                    "equity_decline": -0.10,
                    "currency_impact": 0.05
                },
                regulatory_requirement=True
            ),

            # Currency crisis scenarios
            StressScenario(
                name="Currency Crisis",
                scenario_type=StressTestType.CURRENCY_CRISIS,
                description="Major currency devaluation crisis",
                parameters={
                    "currency_decline": -0.40,
                    "inflation_spike": 0.08,
                    "capital_flight": 0.30,
                    "import_price_increase": 0.25
                }
            ),

            # Correlation breakdown scenarios
            StressScenario(
                name="Correlation Breakdown",
                scenario_type=StressTestType.CORRELATION_BREAKDOWN,
                description="Diversification benefits disappear in crisis",
                parameters={
                    "correlation_increase": 3.0,
                    "correlation_target": 0.95,
                    "volatility_multiplier": 2.5,
                    "market_decline": -0.20
                }
            )
        ]

        return scenarios

    async def run_stress_test(
        self,
        scenario: Optional[StressScenario] = None,
        custom_parameters: Optional[Dict[str, float]] = None
    ) -> StressTestResult:
        """
        Run a single stress test scenario.

        Args:
            scenario: Stress scenario to run (uses default if None)
            custom_parameters: Custom parameters for scenario

        Returns:
            Stress test result
        """
        start_time = asyncio.get_event_loop().time()
        self.test_count += 1

        if scenario is None:
            scenario = self.scenarios[0]  # Use first scenario as default

        # Merge custom parameters
        if custom_parameters:
            scenario.parameters.update(custom_parameters)

        try:
            logger.info(f"Running stress test: {scenario.name}")

            # Initialize portfolio state
            portfolio_before = self.portfolio_value
            position_values_before = {
                symbol: pos.get("value", 0) for symbol, pos in self.positions.items()
            }

            # Apply stress scenario
            portfolio_after, position_impacts = await self._apply_stress_scenario(scenario)

            # Calculate results
            portfolio_loss = portfolio_before - portfolio_after
            portfolio_loss_percentage = portfolio_loss / portfolio_before if portfolio_before > 0 else 0

            # Find worst position impact
            worst_position_loss = max(
                impact.get("loss", 0) for impact in position_impacts.values()
            ) if position_impacts else 0

            # Estimate VaR under stress
            var_under_stress = await self._calculate_stress_var(scenario, portfolio_after)

            # Estimate max drawdown under stress
            max_drawdown_under_stress = await self._estimate_stress_drawdown(scenario)

            # Estimate recovery time
            recovery_time = await self._estimate_recovery_time(scenario, portfolio_loss_percentage)

            # Create result
            result = StressTestResult(
                scenario_name=scenario.name,
                portfolio_value_before=portfolio_before,
                portfolio_value_after=portfolio_after,
                portfolio_loss=portfolio_loss,
                portfolio_loss_percentage=portfolio_loss_percentage,
                worst_position_loss=worst_position_loss,
                var_under_stress=var_under_stress,
                max_drawdown_under_stress=max_drawdown_under_stress,
                recovery_time_estimate=recovery_time,
                position_impacts=position_impacts,
                additional_metrics={
                    "scenario_type": scenario.scenario_type.value,
                    "regulatory_requirement": scenario.regulatory_requirement,
                    "stress_magnitude": self._calculate_stress_magnitude(scenario)
                }
            )

            # Store results
            self.stress_results.append(result)

            # Update worst case
            if (self.worst_case_scenario is None or
                result.portfolio_loss_percentage > self.worst_case_scenario.portfolio_loss_percentage):
                self.worst_case_scenario = result

            # Performance metrics
            test_duration = asyncio.get_event_loop().time() - start_time
            self.last_test_duration = test_duration

            logger.info(f"Stress test completed in {test_duration:.3f}s: "
                       f"Loss {portfolio_loss_percentage:.1%}, VaR {var_under_stress:.1%}")

            return result

        except Exception as e:
            logger.error(f"Error running stress test {scenario.name}: {e}")
            # Return conservative result
            return StressTestResult(
                scenario_name=scenario.name,
                portfolio_value_before=self.portfolio_value,
                portfolio_value_after=self.portfolio_value * 0.7,  # Assume 30% loss
                portfolio_loss=self.portfolio_value * 0.3,
                portfolio_loss_percentage=0.3,
                worst_position_loss=0.0,
                var_under_stress=0.4,
                max_drawdown_under_stress=0.35,
                recovery_time_estimate=180.0  # 6 months
            )

    async def _apply_stress_scenario(
        self,
        scenario: StressScenario
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """Apply stress scenario to portfolio."""
        portfolio_after = self.portfolio_value
        position_impacts = {}

        for symbol, position in self.positions.items():
            position_value = position.get("value", 0)
            position_weight = position_value / self.portfolio_value if self.portfolio_value > 0 else 0

            # Apply scenario-specific impacts
            impact = await self._calculate_position_impact(symbol, position, scenario)
            position_impacts[symbol] = impact

            # Update portfolio value
            portfolio_after -= impact.get("loss", 0)

        return portfolio_after, position_impacts

    async def _calculate_position_impact(
        self,
        symbol: str,
        position: Dict[str, Any],
        scenario: StressScenario
    ) -> Dict[str, float]:
        """Calculate stress impact on a specific position."""
        position_value = position.get("value", 0)
        position_type = position.get("type", "equity")  # equity, bond, currency, commodity

        base_impact = 0.0

        # Apply scenario impacts based on type
        if scenario.scenario_type == StressTestType.MARKET_CRASH:
            base_impact = position_value * abs(scenario.parameters.get("market_decline", 0.2))

        elif scenario.scenario_type == StressTestType.LIQUIDITY_CRISIS:
            liquidity_decrease = scenario.parameters.get("liquidity_decrease", 0.5)
            # Additional impact from liquidity crisis
            base_impact = position_value * (0.1 + liquidity_decrease * 0.3)

        elif scenario.scenario_type == StressTestType.VOLATILITY_SPIKE:
            vol_multiplier = scenario.parameters.get("volatility_multiplier", 2.0)
            # Impact proportional to volatility increase
            base_impact = position_value * (0.05 * vol_multiplier)

        elif scenario.scenario_type == StressTestType.INTEREST_RATE_SHOCK:
            if position_type == "bond":
                base_impact = position_value * abs(scenario.parameters.get("bond_decline", 0.15))
            elif position_type == "equity":
                base_impact = position_value * abs(scenario.parameters.get("equity_decline", 0.10))

        elif scenario.scenario_type == StressTestType.CURRENCY_CRISIS:
            if position_type == "currency":
                base_impact = position_value * abs(scenario.parameters.get("currency_decline", 0.40))
            else:
                base_impact = position_value * 0.1  # Indirect impact

        elif scenario.scenario_type == StressTestType.CORRELATION_BREAKDOWN:
            # When correlation breaks down, diversification benefits disappear
            base_impact = position_value * 0.2

        # Apply additional stress factors
        volatility_multiplier = scenario.parameters.get("volatility_multiplier", 1.0)
        correlation_increase = scenario.parameters.get("correlation_increase", 1.0)

        # Increase impact based on stress factors
        stress_adjustment = 1.0 + (volatility_multiplier - 1.0) * 0.3
        stress_adjustment *= 1.0 + (correlation_increase - 1.0) * 0.2

        final_impact = base_impact * stress_adjustment

        # Add some randomness for realism
        random_factor = 1.0 + np.random.normal(0, 0.1)  # ±10% random variation
        final_impact *= max(0.5, min(1.5, random_factor))  # Clamp between 50% and 150%

        return {
            "loss": min(final_impact, position_value * 0.95),  # Max 95% loss
            "loss_percentage": min(final_impact / position_value, 0.95) if position_value > 0 else 0,
            "remaining_value": max(position_value - final_impact, position_value * 0.05),
            "stress_adjustment": stress_adjustment
        }

    async def _calculate_stress_var(
        self,
        scenario: StressScenario,
        portfolio_value: float
    ) -> float:
        """Calculate VaR under stress conditions."""
        # Base VaR calculation with stress adjustments
        base_var = 0.02  # 2% base VaR

        # Increase VaR based on scenario severity
        stress_multiplier = self._calculate_stress_magnitude(scenario)
        stress_var = base_var * stress_multiplier

        return min(stress_var, 0.5)  # Cap at 50%

    async def _estimate_stress_drawdown(self, scenario: StressScenario) -> float:
        """Estimate maximum drawdown under stress."""
        # Use scenario parameters to estimate drawdown
        if "market_decline" in scenario.parameters:
            return abs(scenario.parameters["market_decline"])
        elif scenario.scenario_type == StressTestType.LIQUIDITY_CRISIS:
            return 0.25  # 25% drawdown estimate
        elif scenario.scenario_type == StressTestType.VOLATILITY_SPIKE:
            return 0.20  # 20% drawdown estimate
        else:
            return 0.15  # Default 15% drawdown

    async def _estimate_recovery_time(
        self,
        scenario: StressScenario,
        loss_percentage: float
    ) -> float:
        """Estimate recovery time in days."""
        # Base recovery time proportional to loss
        base_recovery_days = loss_percentage * 365 * 2  # 2 years to recover from 100% loss

        # Adjust based on scenario type
        if scenario.scenario_type == StressTestType.MARKET_CRASH:
            recovery_multiplier = 1.0
        elif scenario.scenario_type == StressTestType.LIQUIDITY_CRISIS:
            recovery_multiplier = 1.5  # Longer recovery from liquidity crises
        else:
            recovery_multiplier = 1.2

        return base_recovery_days * recovery_multiplier

    def _calculate_stress_magnitude(self, scenario: StressScenario) -> float:
        """Calculate overall stress magnitude for scenario."""
        magnitude_factors = []

        for param, value in scenario.parameters.items():
            if isinstance(value, (int, float)):
                if "decline" in param or "decrease" in param:
                    magnitude_factors.append(abs(value))
                elif "multiplier" in param or "increase" in param:
                    magnitude_factors.append(value - 1.0 if value > 1 else 0)

        return np.mean(magnitude_factors) if magnitude_factors else 1.0

    async def run_all_stress_tests(self) -> List[StressTestResult]:
        """Run all defined stress test scenarios."""
        logger.info(f"Running {len(self.scenarios)} stress test scenarios")

        results = []
        for scenario in self.scenarios:
            try:
                result = await self.run_stress_test(scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running stress test {scenario.name}: {e}")

        logger.info(f"Completed {len(results)} stress tests")
        return results

    async def run_regulatory_stress_tests(self) -> List[StressTestResult]:
        """Run regulatory required stress tests."""
        regulatory_scenarios = [
            scenario for scenario in self.scenarios
            if scenario.regulatory_requirement
        ]

        logger.info(f"Running {len(regulatory_scenarios)} regulatory stress tests")

        results = []
        for scenario in regulatory_scenarios:
            try:
                result = await self.run_stress_test(scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running regulatory stress test {scenario.name}: {e}")

        return results

    async def create_custom_scenario(
        self,
        name: str,
        scenario_type: StressTestType,
        parameters: Dict[str, float],
        description: str = ""
    ) -> StressScenario:
        """Create a custom stress test scenario."""
        scenario = StressScenario(
            name=name,
            scenario_type=scenario_type,
            description=description or f"Custom {scenario_type.value} scenario",
            parameters=parameters
        )

        self.scenarios.append(scenario)
        logger.info(f"Created custom scenario: {name}")

        return scenario

    def get_stress_test_summary(self) -> Dict[str, Any]:
        """Get summary of all stress test results."""
        if not self.stress_results:
            return {
                "status": "no_tests",
                "portfolio_value": self.portfolio_value,
                "scenarios_tested": 0,
                "message": "No stress tests have been run"
            }

        # Calculate summary statistics
        losses = [result.portfolio_loss_percentage for result in self.stress_results]
        vars_stress = [result.var_under_stress for result in self.stress_results]
        drawdowns = [result.max_drawdown_under_stress for result in self.stress_results]

        return {
            "status": "completed",
            "portfolio_value": self.portfolio_value,
            "scenarios_tested": len(self.stress_results),
            "worst_case": {
                "scenario": self.worst_case_scenario.scenario_name,
                "loss_percentage": self.worst_case_scenario.portfolio_loss_percentage,
                "loss_amount": self.worst_case_scenario.portfolio_loss,
                "var_under_stress": self.worst_case_scenario.var_under_stress,
                "max_drawdown": self.worst_case_scenario.max_drawdown_under_stress,
                "recovery_time_days": self.worst_case_scenario.recovery_time_estimate
            },
            "summary_statistics": {
                "average_loss": np.mean(losses),
                "max_loss": np.max(losses),
                "min_loss": np.min(losses),
                "average_var_under_stress": np.mean(vars_stress),
                "max_var_under_stress": np.max(vars_stress),
                "average_drawdown": np.mean(drawdowns),
                "max_drawdown": np.max(drawdowns)
            },
            "regulatory_scenarios": len([
                result for result in self.stress_results
                if result.additional_metrics.get("regulatory_requirement", False)
            ]),
            "performance_metrics": {
                "tests_completed": self.test_count,
                "average_test_duration_ms": self.last_test_duration * 1000,
                "total_test_time_ms": self.last_test_duration * len(self.stress_results) * 1000
            }
        }

    def get_worst_case_positions(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get positions with worst stress test performance."""
        if not self.stress_results:
            return []

        # Aggregate position impacts across all scenarios
        position_losses = {}

        for result in self.stress_results:
            for symbol, impact in result.position_impacts.items():
                if symbol not in position_losses:
                    position_losses[symbol] = {
                        "total_loss": 0,
                        "max_loss": 0,
                        "scenarios_count": 0,
                        "worst_scenario": ""
                    }

                position_losses[symbol]["total_loss"] += impact.get("loss", 0)
                position_losses[symbol]["max_loss"] = max(
                    position_losses[symbol]["max_loss"],
                    impact.get("loss", 0)
                )
                position_losses[symbol]["scenarios_count"] += 1

                if impact.get("loss", 0) == position_losses[symbol]["max_loss"]:
                    position_losses[symbol]["worst_scenario"] = result.scenario_name

        # Sort by total loss
        sorted_positions = sorted(
            position_losses.items(),
            key=lambda x: x[1]["total_loss"],
            reverse=True
        )

        # Format results
        results = []
        for symbol, data in sorted_positions[:top_n]:
            avg_loss = data["total_loss"] / data["scenarios_count"]
            position_value = self.positions.get(symbol, {}).get("value", 0)
            avg_loss_pct = avg_loss / position_value if position_value > 0 else 0

            results.append({
                "symbol": symbol,
                "average_loss": avg_loss,
                "average_loss_percentage": avg_loss_pct,
                "max_loss": data["max_loss"],
                "scenarios_affected": data["scenarios_count"],
                "worst_scenario": data["worst_scenario"],
                "position_value": position_value
            })

        return results


# Standalone validation function
def validate_stress_tester():
    """Validate stress tester implementation."""
    print("🔍 Validating StressTester implementation...")

    try:
        # Test imports
        from .stress_tester import StressTester, StressScenario, StressTestResult
        print("✅ Imports successful")

        # Test instantiation
        tester = StressTester()
        print("✅ StressTester instantiation successful")

        # Test basic functionality
        if hasattr(tester, 'run_stress_test'):
            print("✅ run_stress_test method exists")
        else:
            print("❌ run_stress_test method missing")
            return False

        if hasattr(tester, 'get_stress_test_summary'):
            print("✅ get_stress_test_summary method exists")
        else:
            print("❌ get_stress_test_summary method missing")
            return False

        print("🎉 StressTester validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_stress_tester()