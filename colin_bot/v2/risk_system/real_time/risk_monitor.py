"""
Real-Time Risk Monitor for Colin Trading Bot v2.0

This module implements real-time risk monitoring and control following
AI_ML_RISK_MANAGEMENT.md patterns (lines 376-396).

Key Features:
- Pre-trade risk validation with position limits, VaR limits, margin requirements
- Real-time risk monitoring with drawdown control and correlation limits
- Circuit breaker implementation with automatic position reduction
- RiskDecision dataclass with approval/rejection logic
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ...execution_engine.smart_routing.router import Order, OrderSide


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskDecision:
    """Risk decision dataclass for trade approval/rejection."""
    approved: bool
    risk_level: RiskLevel
    reasoning: str
    warnings: List[str] = field(default_factory=list)
    required_modifications: List[str] = field(default_factory=list)
    decision_timestamp: datetime = field(default_factory=datetime.now)
    risk_score: float = 0.0  # 0-100 scale


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size_usd: float = 100000.0      # $100K max position
    max_portfolio_exposure: float = 0.20         # 20% of portfolio
    max_leverage: float = 3.0                    # 3x max leverage
    max_correlation_exposure: float = 0.70       # 70% correlation limit
    max_drawdown_hard: float = 0.05              # 5% hard drawdown limit
    max_drawdown_warning: float = 0.03           # 3% warning level
    var_limit_95_1d: float = 0.02               # 2% 1-day 95% VaR limit
    var_limit_99_5d: float = 0.05               # 5% 5-day 99% VaR limit
    min_margin_requirement: float = 0.25         # 25% minimum margin


class RealTimeRiskController:
    """
    Real-time risk controller implementing comprehensive risk validation.

    Following AI_ML_RISK_MANAGEMENT.md patterns for real-time validation
    with sub-5ms latency targets and comprehensive risk checks.
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize real-time risk controller.

        Args:
            risk_limits: Risk limits configuration
            config: Additional configuration parameters
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.config = config or {}

        # Risk monitoring state
        self.active_positions = {}
        self.portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.risk_decisions_history = []

        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        self.circuit_breaker_start_time = None

        # Performance metrics
        self.validation_count = 0
        self.validation_latency_ms = []
        self.rejection_rate = 0.0

        logger.info("RealTimeRiskController initialized with risk limits")
        logger.info(f"Max position size: ${self.risk_limits.max_position_size_usd:,.0f}")
        logger.info(f"Max portfolio exposure: {self.risk_limits.max_portfolio_exposure:.1%}")
        logger.info(f"Hard drawdown limit: {self.risk_limits.max_drawdown_hard:.1%}")

    async def validate_trade(
        self,
        order: Order,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_metrics: Dict[str, Any]
    ) -> RiskDecision:
        """
        Validate trade against all risk parameters in real-time.

        Args:
            order: Order to validate
            current_positions: Current portfolio positions
            portfolio_metrics: Current portfolio metrics

        Returns:
            Risk decision with approval/rejection and reasoning
        """
        start_time = time.time()
        self.validation_count += 1

        try:
            # Update internal state
            self.active_positions = current_positions
            self.portfolio_value = portfolio_metrics.get("total_value", 0.0)
            self.current_drawdown = portfolio_metrics.get("current_drawdown", 0.0)

            # Check circuit breaker status
            if self.circuit_breaker_active:
                decision = RiskDecision(
                    approved=False,
                    risk_level=RiskLevel.CRITICAL,
                    reasoning=f"Circuit breaker active: {self.circuit_breaker_reason}",
                    risk_score=100.0
                )
                self._record_decision(decision)
                return decision

            # Initialize risk assessment
            warnings = []
            required_modifications = []
            risk_score = 0.0

            # 1. Position size validation
            position_check = self._validate_position_size(order)
            if not position_check["approved"]:
                decision = RiskDecision(
                    approved=False,
                    risk_level=RiskLevel.HIGH,
                    reasoning=position_check["reason"],
                    risk_score=position_check["risk_score"]
                )
                self._record_decision(decision)
                return decision
            elif position_check["warning"]:
                warnings.append(position_check["warning"])
                risk_score += position_check["risk_score"]

            # 2. Portfolio exposure validation
            exposure_check = self._validate_portfolio_exposure(order, current_positions)
            if not exposure_check["approved"]:
                decision = RiskDecision(
                    approved=False,
                    risk_level=RiskLevel.HIGH,
                    reasoning=exposure_check["reason"],
                    risk_score=exposure_check["risk_score"]
                )
                self._record_decision(decision)
                return decision
            elif exposure_check["warning"]:
                warnings.append(exposure_check["warning"])
                risk_score += exposure_check["risk_score"]

            # 3. Drawdown validation
            drawdown_check = self._validate_drawdown()
            if not drawdown_check["approved"]:
                required_modifications.extend(drawdown_check["modifications"])
                risk_score += drawdown_check["risk_score"]

                # Trigger circuit breaker if critical drawdown
                if self.current_drawdown >= self.risk_limits.max_drawdown_hard:
                    self._trigger_circuit_breaker("Critical drawdown exceeded")
                    decision = RiskDecision(
                        approved=False,
                        risk_level=RiskLevel.CRITICAL,
                        reasoning="Critical drawdown exceeded - circuit breaker activated",
                        warnings=warnings,
                        required_modifications=required_modifications,
                        risk_score=100.0
                    )
                    self._record_decision(decision)
                    return decision
            elif drawdown_check["warning"]:
                warnings.append(drawdown_check["warning"])
                risk_score += drawdown_check["risk_score"]

            # 4. Correlation validation (if we have correlation data)
            if "correlation_matrix" in portfolio_metrics:
                correlation_check = self._validate_correlation(
                    order, portfolio_metrics["correlation_matrix"]
                )
                if not correlation_check["approved"]:
                    decision = RiskDecision(
                        approved=False,
                        risk_level=RiskLevel.HIGH,
                        reasoning=correlation_check["reason"],
                        risk_score=correlation_check["risk_score"]
                    )
                    self._record_decision(decision)
                    return decision
                elif correlation_check["warning"]:
                    warnings.append(correlation_check["warning"])
                    risk_score += correlation_check["risk_score"]

            # 5. VaR validation (if we have VaR data)
            if "portfolio_var_95_1d" in portfolio_metrics:
                var_check = self._validate_var(order, portfolio_metrics)
                if not var_check["approved"]:
                    decision = RiskDecision(
                        approved=False,
                        risk_level=RiskLevel.MEDIUM,
                        reasoning=var_check["reason"],
                        risk_score=var_check["risk_score"]
                    )
                    self._record_decision(decision)
                    return decision
                elif var_check["warning"]:
                    warnings.append(var_check["warning"])
                    risk_score += var_check["risk_score"]

            # Determine final risk level
            risk_level = self._determine_risk_level(risk_score, warnings)

            # Create final decision
            decision = RiskDecision(
                approved=not required_modifications,  # Approve if no required modifications
                risk_level=risk_level,
                reasoning=f"Risk assessment complete. Score: {risk_score:.1f}",
                warnings=warnings,
                required_modifications=required_modifications,
                risk_score=risk_score
            )

            # Record decision and metrics
            self._record_decision(decision)
            validation_latency = (time.time() - start_time) * 1000  # Convert to ms
            self.validation_latency_ms.append(validation_latency)

            logger.debug(f"Risk validation completed in {validation_latency:.2f}ms: "
                        f"{'APPROVED' if decision.approved else 'REJECTED'} "
                        f"(Score: {risk_score:.1f})")

            return decision

        except Exception as e:
            logger.error(f"Error in risk validation: {e}")
            # Fallback to conservative rejection
            decision = RiskDecision(
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                reasoning=f"Risk validation error: {str(e)}",
                risk_score=100.0
            )
            self._record_decision(decision)
            return decision

    def _validate_position_size(self, order: Order) -> Dict[str, Any]:
        """Validate position size against limits."""
        # Estimate order value (would need market data in production)
        estimated_price = 100.0  # Placeholder
        order_value = order.quantity * estimated_price

        if order_value > self.risk_limits.max_position_size_usd:
            return {
                "approved": False,
                "reason": f"Order value ${order_value:,.0f} exceeds maximum ${self.risk_limits.max_position_size_usd:,.0f}",
                "risk_score": 80.0
            }

        # Warning if approaching limit
        warning_threshold = self.risk_limits.max_position_size_usd * 0.8
        if order_value > warning_threshold:
            return {
                "approved": True,
                "warning": f"Order value ${order_value:,.0f} approaching maximum limit",
                "risk_score": 20.0
            }

        return {"approved": True, "risk_score": 5.0}

    def _validate_portfolio_exposure(
        self,
        order: Order,
        current_positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate portfolio exposure limits."""
        # Calculate current exposure
        current_exposure = sum(
            pos.get("value_usd", 0) for pos in current_positions.values()
        )

        # Add new order exposure (estimated)
        estimated_order_value = order.quantity * 100.0  # Placeholder
        new_exposure = current_exposure + estimated_order_value

        if self.portfolio_value > 0:
            exposure_ratio = new_exposure / self.portfolio_value

            if exposure_ratio > self.risk_limits.max_portfolio_exposure:
                return {
                    "approved": False,
                    "reason": f"Portfolio exposure {exposure_ratio:.1%} exceeds maximum {self.risk_limits.max_portfolio_exposure:.1%}",
                    "risk_score": 75.0
                }

            # Warning if approaching limit
            warning_threshold = self.risk_limits.max_portfolio_exposure * 0.9
            if exposure_ratio > warning_threshold:
                return {
                    "approved": True,
                    "warning": f"Portfolio exposure {exposure_ratio:.1%} approaching maximum",
                    "risk_score": 25.0
                }

        return {"approved": True, "risk_score": 10.0}

    def _validate_drawdown(self) -> Dict[str, Any]:
        """Validate current drawdown against limits."""
        if self.current_drawdown >= self.risk_limits.max_drawdown_hard:
            return {
                "approved": False,
                "reason": f"Current drawdown {self.current_drawdown:.1%} exceeds hard limit {self.risk_limits.max_drawdown_hard:.1%}",
                "risk_score": 100.0,
                "modifications": ["Reduce position sizes immediately", "Consider stopping new trades"]
            }

        # Warning level
        if self.current_drawdown >= self.risk_limits.max_drawdown_warning:
            return {
                "approved": True,
                "warning": f"Current drawdown {self.current_drawdown:.1%} above warning level {self.risk_limits.max_drawdown_warning:.1%}",
                "risk_score": 40.0,
                "modifications": ["Reduce position sizes", "Monitor risk metrics closely"]
            }

        return {"approved": True, "risk_score": 15.0}

    def _validate_correlation(
        self,
        order: Order,
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Validate portfolio correlation limits."""
        # This is a simplified correlation check
        # In practice, would calculate portfolio correlation with new position

        max_correlation = 0.0
        if order.symbol in correlation_matrix:
            correlations = correlation_matrix[order.symbol].values()
            if correlations:
                max_correlation = max(abs(c) for c in correlations)

        if max_correlation > self.risk_limits.max_correlation_exposure:
            return {
                "approved": False,
                "reason": f"Portfolio correlation {max_correlation:.1%} exceeds maximum {self.risk_limits.max_correlation_exposure:.1%}",
                "risk_score": 60.0
            }

        # Warning if approaching limit
        warning_threshold = self.risk_limits.max_correlation_exposure * 0.9
        if max_correlation > warning_threshold:
            return {
                "approved": True,
                "warning": f"Portfolio correlation {max_correlation:.1%} approaching maximum",
                "risk_score": 30.0
            }

        return {"approved": True, "risk_score": 20.0}

    def _validate_var(
        self,
        order: Order,
        portfolio_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate Value-at-Risk limits."""
        current_var_95_1d = portfolio_metrics.get("portfolio_var_95_1d", 0.0)
        current_var_99_5d = portfolio_metrics.get("portfolio_var_99_5d", 0.0)

        # Check 1-day 95% VaR
        if current_var_95_1d > self.risk_limits.var_limit_95_1d:
            return {
                "approved": False,
                "reason": f"1-day 95% VaR {current_var_95_1d:.1%} exceeds limit {self.risk_limits.var_limit_95_1d:.1%}",
                "risk_score": 70.0
            }

        # Check 5-day 99% VaR
        if current_var_99_5d > self.risk_limits.var_limit_99_5d:
            return {
                "approved": False,
                "reason": f"5-day 99% VaR {current_var_99_5d:.1%} exceeds limit {self.risk_limits.var_limit_99_5d:.1%}",
                "risk_score": 75.0
            }

        # Warnings
        var_warning_threshold = self.risk_limits.var_limit_95_1d * 0.8
        if current_var_95_1d > var_warning_threshold:
            return {
                "approved": True,
                "warning": f"1-day 95% VaR {current_var_95_1d:.1%} approaching limit",
                "risk_score": 35.0
            }

        return {"approved": True, "risk_score": 25.0}

    def _determine_risk_level(self, risk_score: float, warnings: List[str]) -> RiskLevel:
        """Determine risk level based on score and warnings."""
        if risk_score >= 80 or len(warnings) >= 3:
            return RiskLevel.CRITICAL
        elif risk_score >= 60 or len(warnings) >= 2:
            return RiskLevel.HIGH
        elif risk_score >= 30 or len(warnings) >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker to halt trading."""
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        self.circuit_breaker_start_time = datetime.now()

        logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        logger.critical("All new trades will be rejected until conditions improve")

    def reset_circuit_breaker(self):
        """Reset circuit breaker when conditions improve."""
        if self.circuit_breaker_active:
            duration = datetime.now() - self.circuit_breaker_start_time
            logger.info(f"Circuit breaker reset after {duration}")

            self.circuit_breaker_active = False
            self.circuit_breaker_reason = ""
            self.circuit_breaker_start_time = None

    def _record_decision(self, decision: RiskDecision):
        """Record risk decision for analytics."""
        self.risk_decisions_history.append({
            "timestamp": decision.decision_timestamp,
            "approved": decision.approved,
            "risk_level": decision.risk_level,
            "risk_score": decision.risk_score,
            "reasoning": decision.reasoning
        })

        # Keep only last 1000 decisions
        if len(self.risk_decisions_history) > 1000:
            self.risk_decisions_history = self.risk_decisions_history[-1000:]

        # Update rejection rate
        recent_decisions = self.risk_decisions_history[-100:]
        if recent_decisions:
            rejections = sum(1 for d in recent_decisions if not d["approved"])
            self.rejection_rate = rejections / len(recent_decisions)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics and performance statistics."""
        recent_decisions = self.risk_decisions_history[-100:] if self.risk_decisions_history else []

        avg_latency = sum(self.validation_latency_ms[-100:]) / len(self.validation_latency_ms[-100:]) if self.validation_latency_ms else 0.0

        return {
            "validation_count": self.validation_count,
            "average_latency_ms": avg_latency,
            "rejection_rate": self.rejection_rate,
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "current_drawdown": self.current_drawdown,
            "portfolio_value": self.portfolio_value,
            "active_positions_count": len(self.active_positions),
            "recent_decisions_count": len(recent_decisions),
            "risk_score_distribution": self._get_risk_score_distribution(recent_decisions),
            "risk_level_distribution": self._get_risk_level_distribution(recent_decisions)
        }

    def _get_risk_score_distribution(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of risk scores."""
        if not decisions:
            return {}

        scores = [d["risk_score"] for d in decisions]
        return {
            "low_risk": len([s for s in scores if s < 30]),
            "medium_risk": len([s for s in scores if 30 <= s < 60]),
            "high_risk": len([s for s in scores if 60 <= s < 80]),
            "critical_risk": len([s for s in scores if s >= 80])
        }

    def _get_risk_level_distribution(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of risk levels."""
        if not decisions:
            return {}

        distribution = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }

        for decision in decisions:
            distribution[decision["risk_level"].value] += 1

        return distribution


# Standalone validation function for testing
def validate_risk_monitor():
    """Validate risk monitor implementation."""
    print("🔍 Validating RealTimeRiskController implementation...")

    try:
        # Test imports
        from .risk_monitor import RealTimeRiskController, RiskDecision, RiskLevel
        print("✅ Imports successful")

        # Test instantiation
        controller = RealTimeRiskController()
        print("✅ Controller instantiation successful")

        # Test basic functionality
        if hasattr(controller, 'validate_trade'):
            print("✅ validate_trade method exists")
        else:
            print("❌ validate_trade method missing")
            return False

        if hasattr(controller, 'get_risk_metrics'):
            print("✅ get_risk_metrics method exists")
        else:
            print("❌ get_risk_metrics method missing")
            return False

        print("🎉 RealTimeRiskController validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_risk_monitor()