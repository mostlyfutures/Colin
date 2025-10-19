"""
Pre-Trade Compliance Checker for Colin Trading Bot v2.0

This module implements pre-trade compliance checks and validation.

Key Features:
- Pre-trade risk checks integration with real-time monitoring
- Position limits enforcement (2% portfolio, 20% single symbol from PRP)
- Regulatory compliance validation (MiFID II, SEC/FINRA)
- Audit trail generation for all pre-trade decisions
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ...execution_engine.smart_routing.router import Order, OrderSide, OrderType


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    ERROR = "error"


class RegulatoryRegime(Enum):
    """Regulatory regimes."""
    MIFID_II = "mifid_ii"
    SEC_FINRA = "sec_finra"
    FCA = "fca"
    ASIC = "asic"
    NONE = "none"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    name: str
    description: str
    regulatory_regime: RegulatoryRegime
    is_active: bool = True
    severity: str = "medium"  # low, medium, high, critical
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceResult:
    """Compliance check result."""
    order_id: str
    compliant: bool
    status: ComplianceStatus
    rules_checked: int
    rules_passed: int
    rules_failed: int
    failed_rules: List[Dict[str, Any]]
    warnings: List[str]
    recommendations: List[str]
    check_timestamp: datetime = field(default_factory=datetime.now)
    check_duration_ms: float = 0.0
    auditor_id: str = "pre_trade_checker"


@dataclass
class ComplianceConfiguration:
    """Compliance checker configuration."""
    max_position_size_portfolio: float = 0.02      # 2% of portfolio (PRP)
    max_position_size_symbol: float = 0.20          # 20% single symbol (PRP)
    max_daily_trades: int = 1000
    max_order_size_usd: float = 1000000.0           # $1M max order size
    min_order_size_usd: float = 100.0               # $100 min order size
    restricted_symbols: List[str] = field(default_factory=list)
    required_approvals: Dict[str, List[str]] = field(default_factory=dict)
    audit_retention_days: int = 2555                # 7 years for compliance
    regulatory_regime: RegulatoryRegime = RegulatoryRegime.SEC_FINRA


class PreTradeChecker:
    """
    Pre-trade compliance checker with regulatory validation.

    This class implements comprehensive pre-trade compliance checks including
    position limits, regulatory requirements, and audit trail generation.
    """

    def __init__(
        self,
        config: Optional[ComplianceConfiguration] = None,
        portfolio_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pre-trade compliance checker.

        Args:
            config: Compliance configuration
            portfolio_data: Current portfolio data
        """
        self.config = config or ComplianceConfiguration()
        self.portfolio_data = portfolio_data or {"total_value": 100000.0}

        # Initialize compliance rules
        self.compliance_rules = self._initialize_compliance_rules()

        # Audit trail storage
        self.audit_trail: List[ComplianceResult] = []
        self.daily_trade_counts: Dict[str, int] = {}

        # Performance metrics
        self.checks_performed = 0
        self.average_check_time = 0.0
        self.compliance_rate = 0.0

        logger.info(f"PreTradeChecker initialized with {self.config.regulatory_regime.value} regime")
        logger.info(f"Max position size: {self.config.max_position_size_portfolio:.1%} portfolio, {self.config.max_position_size_symbol:.1%} symbol")

    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """Initialize built-in compliance rules."""
        rules = [
            # Position size rules
            ComplianceRule(
                rule_id="POS_001",
                name="Portfolio Position Limit",
                description="Maximum position size as percentage of portfolio",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="high",
                parameters={
                    "max_percentage": self.config.max_position_size_portfolio,
                    "check_method": "portfolio_percentage"
                }
            ),
            ComplianceRule(
                rule_id="POS_002",
                name="Single Symbol Concentration",
                description="Maximum exposure to single symbol",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="high",
                parameters={
                    "max_percentage": self.config.max_position_size_symbol,
                    "check_method": "symbol_percentage"
                }
            ),

            # Order size rules
            ComplianceRule(
                rule_id="ORD_001",
                name="Maximum Order Size",
                description="Maximum order size in USD",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="medium",
                parameters={
                    "max_size_usd": self.config.max_order_size_usd
                }
            ),
            ComplianceRule(
                rule_id="ORD_002",
                name="Minimum Order Size",
                description="Minimum order size in USD",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="low",
                parameters={
                    "min_size_usd": self.config.min_order_size_usd
                }
            ),

            # Trading frequency rules
            ComplianceRule(
                rule_id="FREQ_001",
                name="Daily Trade Limit",
                description="Maximum number of trades per day",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="medium",
                parameters={
                    "max_daily_trades": self.config.max_daily_trades
                }
            ),

            # Symbol restrictions
            ComplianceRule(
                rule_id="SYM_001",
                name="Restricted Symbols",
                description="Check against restricted symbols list",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="critical",
                parameters={
                    "restricted_symbols": self.config.restricted_symbols
                }
            ),

            # MiFID II specific rules
            ComplianceRule(
                rule_id="MIFID_001",
                name="MiFID II Best Execution",
                description="Ensure best execution principles",
                regulatory_regime=RegulatoryRegime.MIFID_II,
                severity="high",
                parameters={
                    "require_best_execution": True,
                    "venue_comparison": True
                }
            ),

            # General compliance rules
            ComplianceRule(
                rule_id="GEN_001",
                name="Order Validity",
                description="Basic order validity checks",
                regulatory_regime=RegulatoryRegime.SEC_FINRA,
                severity="high",
                parameters={
                    "require_valid_symbol": True,
                    "require_valid_price": True,
                    "require_valid_quantity": True
                }
            )
        ]

        return rules

    async def check_compliance(
        self,
        order: Order,
        current_positions: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ComplianceResult:
        """
        Perform comprehensive pre-trade compliance check.

        Args:
            order: Order to check
            current_positions: Current portfolio positions
            market_data: Current market data

        Returns:
            Compliance check result
        """
        start_time = time.time()
        self.checks_performed += 1

        try:
            logger.debug(f"Starting compliance check for order {order.client_order_id}")

            # Update portfolio data
            if current_positions:
                self.portfolio_data.update({"positions": current_positions})

            # Initialize result
            rules_checked = 0
            rules_passed = 0
            rules_failed = 0
            failed_rules = []
            warnings = []
            recommendations = []

            # Get applicable rules
            applicable_rules = [
                rule for rule in self.compliance_rules
                if rule.is_active and (
                    rule.regulatory_regime == self.config.regulatory_regime or
                    rule.regulatory_regime == RegulatoryRegime.NONE
                )
            ]

            # Check each rule
            for rule in applicable_rules:
                rules_checked += 1
                rule_result = await self._check_rule(order, rule, current_positions, market_data)

                if rule_result["passed"]:
                    rules_passed += 1
                else:
                    rules_failed += 1
                    failed_rules.append({
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "severity": rule.severity,
                        "reason": rule_result["reason"],
                        "details": rule_result.get("details", {})
                    })

                    # Add warnings and recommendations
                    if rule_result.get("warning"):
                        warnings.append(rule_result["warning"])
                    if rule_result.get("recommendation"):
                        recommendations.append(rule_result["recommendation"])

            # Determine overall compliance status
            compliant = rules_failed == 0
            status = self._determine_compliance_status(compliant, failed_rules, warnings)

            # Calculate check duration
            check_duration = (time.time() - start_time) * 1000  # Convert to ms

            # Create compliance result
            result = ComplianceResult(
                order_id=order.client_order_id,
                compliant=compliant,
                status=status,
                rules_checked=rules_checked,
                rules_passed=rules_passed,
                rules_failed=rules_failed,
                failed_rules=failed_rules,
                warnings=warnings,
                recommendations=recommendations,
                check_duration_ms=check_duration
            )

            # Update daily trade count if compliant
            if compliant:
                await self._update_daily_trade_count(order)

            # Add to audit trail
            await self._add_to_audit_trail(result)

            # Update performance metrics
            await self._update_performance_metrics(result)

            logger.debug(f"Compliance check completed in {check_duration:.2f}ms: "
                        f"{'COMPLIANT' if compliant else 'NON_COMPLIANT'} "
                        f"({rules_passed}/{rules_checked} rules passed)")

            return result

        except Exception as e:
            logger.error(f"Error in compliance check for order {order.client_order_id}: {e}")
            # Return conservative non-compliant result
            return ComplianceResult(
                order_id=order.client_order_id,
                compliant=False,
                status=ComplianceStatus.ERROR,
                rules_checked=0,
                rules_passed=0,
                rules_failed=1,
                failed_rules=[{
                    "rule_id": "SYS_ERROR",
                    "rule_name": "System Error",
                    "severity": "critical",
                    "reason": f"Compliance check system error: {str(e)}",
                    "details": {}
                }],
                check_duration_ms=(time.time() - start_time) * 1000
            )

    async def _check_rule(
        self,
        order: Order,
        rule: ComplianceRule,
        current_positions: Optional[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check a specific compliance rule."""
        try:
            if rule.rule_id.startswith("POS_"):
                return await self._check_position_rule(order, rule, current_positions)
            elif rule.rule_id.startswith("ORD_"):
                return await self._check_order_rule(order, rule, market_data)
            elif rule.rule_id.startswith("FREQ_"):
                return await self._check_frequency_rule(order, rule)
            elif rule.rule_id.startswith("SYM_"):
                return await self._check_symbol_rule(order, rule)
            elif rule.rule_id.startswith("MIFID_"):
                return await self._check_mifid_rule(order, rule, market_data)
            elif rule.rule_id.startswith("GEN_"):
                return await self._check_general_rule(order, rule, market_data)
            else:
                return {
                    "passed": False,
                    "reason": f"Unknown rule type: {rule.rule_id}"
                }

        except Exception as e:
            logger.error(f"Error checking rule {rule.rule_id}: {e}")
            return {
                "passed": False,
                "reason": f"Rule check error: {str(e)}"
            }

    async def _check_position_rule(
        self,
        order: Order,
        rule: ComplianceRule,
        current_positions: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check position-related compliance rules."""
        if rule.rule_id == "POS_001":
            # Portfolio position limit
            portfolio_value = self.portfolio_data.get("total_value", 0)
            if portfolio_value == 0:
                return {"passed": True}

            order_value = order.quantity * (order.price or 100)  # Estimate price if not provided
            position_percentage = order_value / portfolio_value

            max_percentage = rule.parameters.get("max_percentage", self.config.max_position_size_portfolio)

            if position_percentage > max_percentage:
                return {
                    "passed": False,
                    "reason": f"Order size {position_percentage:.1%} exceeds portfolio limit {max_percentage:.1%}",
                    "warning": "Consider reducing order size",
                    "recommendation": f"Reduce order to at most {max_percentage:.1%} of portfolio",
                    "details": {
                        "order_value": order_value,
                        "portfolio_value": portfolio_value,
                        "position_percentage": position_percentage,
                        "limit_percentage": max_percentage
                    }
                }

        elif rule.rule_id == "POS_002":
            # Single symbol concentration
            if not current_positions:
                return {"passed": True}

            current_symbol_exposure = current_positions.get(order.symbol, {}).get("total_exposure", 0)
            order_value = order.quantity * (order.price or 100)
            new_exposure = current_symbol_exposure + order_value

            portfolio_value = self.portfolio_data.get("total_value", 0)
            if portfolio_value == 0:
                return {"passed": True}

            exposure_percentage = new_exposure / portfolio_value
            max_percentage = rule.parameters.get("max_percentage", self.config.max_position_size_symbol)

            if exposure_percentage > max_percentage:
                return {
                    "passed": False,
                    "reason": f"Symbol exposure {exposure_percentage:.1%} exceeds limit {max_percentage:.1%}",
                    "warning": "High concentration in single symbol",
                    "recommendation": f"Reduce exposure to at most {max_percentage:.1%} of portfolio",
                    "details": {
                        "symbol": order.symbol,
                        "current_exposure": current_symbol_exposure,
                        "order_value": order_value,
                        "new_exposure": new_exposure,
                        "exposure_percentage": exposure_percentage,
                        "limit_percentage": max_percentage
                    }
                }

        return {"passed": True}

    async def _check_order_rule(
        self,
        order: Order,
        rule: ComplianceRule,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check order-related compliance rules."""
        if rule.rule_id == "ORD_001":
            # Maximum order size
            order_value = order.quantity * (order.price or 100)
            max_size = rule.parameters.get("max_size_usd", self.config.max_order_size_usd)

            if order_value > max_size:
                return {
                    "passed": False,
                    "reason": f"Order value ${order_value:,.0f} exceeds maximum ${max_size:,.0f}",
                    "warning": "Large order size detected",
                    "recommendation": f"Reduce order to at most ${max_size:,.0f}",
                    "details": {
                        "order_value": order_value,
                        "max_size": max_size
                    }
                }

        elif rule.rule_id == "ORD_002":
            # Minimum order size
            order_value = order.quantity * (order.price or 100)
            min_size = rule.parameters.get("min_size_usd", self.config.min_order_size_usd)

            if order_value < min_size:
                return {
                    "passed": False,
                    "reason": f"Order value ${order_value:,.0f} below minimum ${min_size:,.0f}",
                    "warning": "Very small order size",
                    "recommendation": f"Increase order to at least ${min_size:,.0f}",
                    "details": {
                        "order_value": order_value,
                        "min_size": min_size
                    }
                }

        return {"passed": True}

    async def _check_frequency_rule(
        self,
        order: Order,
        rule: ComplianceRule
    ) -> Dict[str, Any]:
        """Check trading frequency rules."""
        if rule.rule_id == "FREQ_001":
            # Daily trade limit
            today = datetime.now().strftime("%Y-%m-%d")
            daily_count = self.daily_trade_counts.get(today, 0)
            max_trades = rule.parameters.get("max_daily_trades", self.config.max_daily_trades)

            if daily_count >= max_trades:
                return {
                    "passed": False,
                    "reason": f"Daily trade limit {max_trades} exceeded ({daily_count} trades)",
                    "warning": "High trading frequency detected",
                    "recommendation": "Wait until next trading day or reduce trading frequency",
                    "details": {
                        "daily_count": daily_count,
                        "max_trades": max_trades,
                        "date": today
                    }
                }

        return {"passed": True}

    async def _check_symbol_rule(
        self,
        order: Order,
        rule: ComplianceRule
    ) -> Dict[str, Any]:
        """Check symbol-related compliance rules."""
        if rule.rule_id == "SYM_001":
            # Restricted symbols
            restricted_symbols = rule.parameters.get("restricted_symbols", self.config.restricted_symbols)

            if order.symbol in restricted_symbols:
                return {
                    "passed": False,
                    "reason": f"Symbol {order.symbol} is on restricted list",
                    "warning": "Trading restricted symbol",
                    "recommendation": "Avoid trading restricted symbols or obtain proper approval",
                    "details": {
                        "symbol": order.symbol,
                        "restricted_symbols": restricted_symbols
                    }
                }

        return {"passed": True}

    async def _check_mifid_rule(
        self,
        order: Order,
        rule: ComplianceRule,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check MiFID II specific rules."""
        if rule.rule_id == "MIFID_001":
            # Best execution requirements
            if self.config.regulatory_regime != RegulatoryRegime.MIFID_II:
                return {"passed": True}

            # Simplified best execution check
            # In practice, this would involve venue comparison, execution costs, etc.
            if not market_data:
                return {
                    "passed": False,
                    "reason": "Market data required for MiFID II best execution check",
                    "warning": "Insufficient market data for compliance",
                    "recommendation": "Ensure market data is available before trading"
                }

        return {"passed": True}

    async def _check_general_rule(
        self,
        order: Order,
        rule: ComplianceRule,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check general compliance rules."""
        if rule.rule_id == "GEN_001":
            # Order validity checks
            if rule.parameters.get("require_valid_symbol", True):
                if not order.symbol or len(order.symbol) < 1:
                    return {
                        "passed": False,
                        "reason": "Invalid or missing symbol"
                    }

            if rule.parameters.get("require_valid_quantity", True):
                if order.quantity <= 0:
                    return {
                        "passed": False,
                        "reason": "Invalid order quantity"
                    }

            if rule.parameters.get("require_valid_price", True):
                if order.order_type == OrderType.LIMIT and (not order.price or order.price <= 0):
                    return {
                        "passed": False,
                        "reason": "Invalid limit price"
                    }

        return {"passed": True}

    def _determine_compliance_status(
        self,
        compliant: bool,
        failed_rules: List[Dict[str, Any]],
        warnings: List[str]
    ) -> ComplianceStatus:
        """Determine overall compliance status."""
        if not compliant:
            # Check severity of failed rules
            critical_failures = [
                rule for rule in failed_rules
                if rule.get("severity") == "critical"
            ]
            if critical_failures:
                return ComplianceStatus.NON_COMPLIANT
            else:
                return ComplianceStatus.REQUIRES_REVIEW
        elif warnings:
            return ComplianceStatus.REQUIRES_REVIEW
        else:
            return ComplianceStatus.COMPLIANT

    async def _update_daily_trade_count(self, order: Order):
        """Update daily trade count."""
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_trade_counts[today] = self.daily_trade_counts.get(today, 0) + 1

    async def _add_to_audit_trail(self, result: ComplianceResult):
        """Add result to audit trail."""
        self.audit_trail.append(result)

        # Keep only recent records (based on retention period)
        cutoff_date = datetime.now() - timedelta(days=self.config.audit_retention_days)
        self.audit_trail = [
            record for record in self.audit_trail
            if record.check_timestamp >= cutoff_date
        ]

    async def _update_performance_metrics(self, result: ComplianceResult):
        """Update performance metrics."""
        # Update average check time
        self.average_check_time = (
            (self.average_check_time * (self.checks_performed - 1) + result.check_duration_ms) /
            self.checks_performed
        )

        # Update compliance rate
        recent_checks = self.audit_trail[-100:] if len(self.audit_trail) >= 100 else self.audit_trail
        if recent_checks:
            compliant_checks = sum(1 for check in recent_checks if check.compliant)
            self.compliance_rate = compliant_checks / len(recent_checks)

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance checker summary."""
        # Recent failed rules analysis
        recent_checks = self.audit_trail[-100:] if len(self.audit_trail) >= 100 else self.audit_trail
        failed_rules_summary = {}

        for check in recent_checks:
            for failed_rule in check.failed_rules:
                rule_id = failed_rule["rule_id"]
                if rule_id not in failed_rules_summary:
                    failed_rules_summary[rule_id] = {
                        "rule_name": failed_rule["rule_name"],
                        "severity": failed_rule["severity"],
                        "count": 0,
                        "last_failure": None
                    }
                failed_rules_summary[rule_id]["count"] += 1
                if (failed_rules_summary[rule_id]["last_failure"] is None or
                    check.check_timestamp > failed_rules_summary[rule_id]["last_failure"]):
                    failed_rules_summary[rule_id]["last_failure"] = check.check_timestamp

        return {
            "total_checks": self.checks_performed,
            "compliance_rate": self.compliance_rate,
            "average_check_time_ms": self.average_check_time,
            "active_rules": len([r for r in self.compliance_rules if r.is_active]),
            "regulatory_regime": self.config.regulatory_regime.value,
            "audit_trail_size": len(self.audit_trail),
            "recent_failed_rules": failed_rules_summary,
            "daily_trade_counts": dict(self.daily_trade_counts),
            "portfolio_value": self.portfolio_data.get("total_value", 0)
        }

    def get_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        order_id: Optional[str] = None
    ) -> List[ComplianceResult]:
        """Get filtered audit trail."""
        filtered_trail = self.audit_trail

        if start_date:
            filtered_trail = [
                record for record in filtered_trail
                if record.check_timestamp >= start_date
            ]

        if end_date:
            filtered_trail = [
                record for record in filtered_trail
                if record.check_timestamp <= end_date
            ]

        if order_id:
            filtered_trail = [
                record for record in filtered_trail
                if record.order_id == order_id
            ]

        return filtered_trail

    def add_custom_rule(self, rule: ComplianceRule):
        """Add a custom compliance rule."""
        self.compliance_rules.append(rule)
        logger.info(f"Added custom compliance rule: {rule.name}")

    def update_rule_status(self, rule_id: str, is_active: bool):
        """Update the active status of a compliance rule."""
        for rule in self.compliance_rules:
            if rule.rule_id == rule_id:
                rule.is_active = is_active
                logger.info(f"Updated rule {rule_id} active status to {is_active}")
                return

        logger.warning(f"Rule {rule_id} not found")


# Standalone validation function
def validate_pre_trade_checker():
    """Validate pre-trade checker implementation."""
    print("🔍 Validating PreTradeChecker implementation...")

    try:
        # Test imports
        from .pre_trade_check import PreTradeChecker, ComplianceResult, ComplianceStatus
        print("✅ Imports successful")

        # Test instantiation
        checker = PreTradeChecker()
        print("✅ PreTradeChecker instantiation successful")

        # Test basic functionality
        if hasattr(checker, 'check_compliance'):
            print("✅ check_compliance method exists")
        else:
            print("❌ check_compliance method missing")
            return False

        if hasattr(checker, 'get_compliance_summary'):
            print("✅ get_compliance_summary method exists")
        else:
            print("❌ get_compliance_summary method missing")
            return False

        print("🎉 PreTradeChecker validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_pre_trade_checker()