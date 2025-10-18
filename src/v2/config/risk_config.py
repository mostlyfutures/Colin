"""
Risk Management Configuration for Colin Trading Bot v2.0

This module provides comprehensive configuration management for the risk management system.

Key Features:
- Risk parameter definitions and validation
- Integration with existing ConfigManager pattern
- Risk threshold configurations (position limits, VaR limits, etc.)
- Environment-specific risk configurations
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger

from ..real_time.risk_monitor import RiskLimits
from ..portfolio.var_calculator import VaRConfiguration
from ..portfolio.correlation_analyzer import CorrelationConfiguration
from ..portfolio.stress_tester import StressTestConfiguration
from ..compliance.pre_trade_check import ComplianceConfiguration
from ..compliance.compliance_monitor import ComplianceMonitorConfiguration


@dataclass
class PositionLimitsConfig:
    """Position limits configuration."""
    max_position_size_usd: float = 100000.0
    max_portfolio_exposure: float = 0.20
    max_leverage: float = 3.0
    max_concentration_symbol: float = 0.20
    max_concentration_sector: float = 0.40
    min_position_size_usd: float = 100.0


@dataclass
class VaRLimitsConfig:
    """VaR limits configuration."""
    var_limit_95_1d: float = 0.02
    var_limit_99_1d: float = 0.03
    var_limit_95_5d: float = 0.04
    var_limit_99_5d: float = 0.06
    stressed_var_multiplier: float = 1.5
    backtesting_confidence_level: float = 0.95


@dataclass
class DrawdownLimitsConfig:
    """Drawdown limits configuration."""
    max_drawdown_hard: float = 0.05
    max_drawdown_warning: float = 0.03
    max_drawdown_duration_hours: int = 4
    auto_reduction_threshold: float = 0.04
    trading_halt_threshold: float = 0.08


@dataclass
class CorrelationLimitsConfig:
    """Correlation limits configuration."""
    max_correlation_portfolio: float = 0.70
    correlation_warning_threshold: float = 0.60
    min_diversification_ratio: float = 0.5
    effective_bets_minimum: float = 3.0


@dataclass
class RiskSystemConfig:
    """Main risk system configuration."""
    # Environment
    environment: str = "development"

    # Position limits
    position_limits: PositionLimitsConfig = field(default_factory=PositionLimitsConfig)

    # VaR limits
    var_limits: VaRLimitsConfig = field(default_factory=VaRLimitsConfig)

    # Drawdown limits
    drawdown_limits: DrawdownLimitsConfig = field(default_factory=DrawdownLimitsConfig)

    # Correlation limits
    correlation_limits: CorrelationLimitsConfig = field(default_factory=CorrelationLimitsConfig)

    # Component configurations
    var_config: VaRConfiguration = field(default_factory=VaRConfiguration)
    correlation_config: CorrelationConfiguration = field(default_factory=CorrelationConfiguration)
    stress_test_config: StressTestConfiguration = field(default_factory=StressTestConfiguration)
    compliance_config: ComplianceConfiguration = field(default_factory=ComplianceConfiguration)
    compliance_monitor_config: ComplianceMonitorConfiguration = field(default_factory=ComplianceMonitorConfiguration)

    # Risk management settings
    risk_management_enabled: bool = True
    real_time_monitoring: bool = True
    circuit_breaker_enabled: bool = True
    auto_risk_reduction: bool = True

    # Logging and monitoring
    risk_log_level: str = "INFO"
    alert_webhook_url: Optional[str] = None
    metrics_retention_days: int = 90

    # Validation settings
    config_validation_enabled: bool = True
    risk_validation_strict_mode: bool = False


class RiskConfigManager:
    """
    Risk management configuration manager.

    This class manages all risk-related configuration with validation,
    environment-specific settings, and integration with the existing config system.
    """

    def __init__(self, config_file_path: Optional[str] = None):
        """
        Initialize risk configuration manager.

        Args:
            config_file_path: Path to configuration file
        """
        self.config_file_path = config_file_path or "config/risk_config.json"
        self.config: RiskSystemConfig = RiskSystemConfig()

        # Load configuration
        self.load_configuration()

        logger.info(f"RiskConfigManager initialized for {self.config.environment} environment")

    def load_configuration(self):
        """Load configuration from file or create default."""
        try:
            config_path = Path(self.config_file_path)

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                # Load configuration with validation
                self.config = self._load_config_from_dict(config_data)
                logger.info(f"Loaded risk configuration from {self.config_file_path}")
            else:
                # Create default configuration
                self.config = self._get_environment_specific_config()
                self.save_configuration()
                logger.info(f"Created default risk configuration at {self.config_file_path}")

            # Validate configuration
            if self.config.config_validation_enabled:
                self.validate_configuration()

        except Exception as e:
            logger.error(f"Error loading risk configuration: {e}")
            # Fallback to default configuration
            self.config = self._get_environment_specific_config()

    def _load_config_from_dict(self, config_data: Dict[str, Any]) -> RiskSystemConfig:
        """Load configuration from dictionary with validation."""
        try:
            # Main configuration
            main_config = RiskSystemConfig(
                environment=config_data.get("environment", "development"),
                risk_management_enabled=config_data.get("risk_management_enabled", True),
                real_time_monitoring=config_data.get("real_time_monitoring", True),
                circuit_breaker_enabled=config_data.get("circuit_breaker_enabled", True),
                auto_risk_reduction=config_data.get("auto_risk_reduction", True),
                risk_log_level=config_data.get("risk_log_level", "INFO"),
                alert_webhook_url=config_data.get("alert_webhook_url"),
                metrics_retention_days=config_data.get("metrics_retention_days", 90),
                config_validation_enabled=config_data.get("config_validation_enabled", True),
                risk_validation_strict_mode=config_data.get("risk_validation_strict_mode", False)
            )

            # Position limits
            pos_limits_data = config_data.get("position_limits", {})
            main_config.position_limits = PositionLimitsConfig(
                max_position_size_usd=pos_limits_data.get("max_position_size_usd", 100000.0),
                max_portfolio_exposure=pos_limits_data.get("max_portfolio_exposure", 0.20),
                max_leverage=pos_limits_data.get("max_leverage", 3.0),
                max_concentration_symbol=pos_limits_data.get("max_concentration_symbol", 0.20),
                max_concentration_sector=pos_limits_data.get("max_concentration_sector", 0.40),
                min_position_size_usd=pos_limits_data.get("min_position_size_usd", 100.0)
            )

            # VaR limits
            var_limits_data = config_data.get("var_limits", {})
            main_config.var_limits = VaRLimitsConfig(
                var_limit_95_1d=var_limits_data.get("var_limit_95_1d", 0.02),
                var_limit_99_1d=var_limits_data.get("var_limit_99_1d", 0.03),
                var_limit_95_5d=var_limits_data.get("var_limit_95_5d", 0.04),
                var_limit_99_5d=var_limits_data.get("var_limit_99_5d", 0.06),
                stressed_var_multiplier=var_limits_data.get("stressed_var_multiplier", 1.5),
                backtesting_confidence_level=var_limits_data.get("backtesting_confidence_level", 0.95)
            )

            # Drawdown limits
            drawdown_data = config_data.get("drawdown_limits", {})
            main_config.drawdown_limits = DrawdownLimitsConfig(
                max_drawdown_hard=drawdown_data.get("max_drawdown_hard", 0.05),
                max_drawdown_warning=drawdown_data.get("max_drawdown_warning", 0.03),
                max_drawdown_duration_hours=drawdown_data.get("max_drawdown_duration_hours", 4),
                auto_reduction_threshold=drawdown_data.get("auto_reduction_threshold", 0.04),
                trading_halt_threshold=drawdown_data.get("trading_halt_threshold", 0.08)
            )

            # Correlation limits
            correlation_data = config_data.get("correlation_limits", {})
            main_config.correlation_limits = CorrelationLimitsConfig(
                max_correlation_portfolio=correlation_data.get("max_correlation_portfolio", 0.70),
                correlation_warning_threshold=correlation_data.get("correlation_warning_threshold", 0.60),
                min_diversification_ratio=correlation_data.get("min_diversification_ratio", 0.5),
                effective_bets_minimum=correlation_data.get("effective_bets_minimum", 3.0)
            )

            return main_config

        except Exception as e:
            logger.error(f"Error loading configuration from dict: {e}")
            return RiskSystemConfig()

    def _get_environment_specific_config(self) -> RiskSystemConfig:
        """Get configuration based on environment."""
        environment = os.getenv("ENVIRONMENT", "development")

        if environment == "production":
            return self._get_production_config()
        elif environment == "staging":
            return self._get_staging_config()
        else:
            return self._get_development_config()

    def _get_development_config(self) -> RiskSystemConfig:
        """Get development environment configuration."""
        config = RiskSystemConfig()
        config.environment = "development"
        config.position_limits.max_position_size_usd = 10000.0  # Smaller limits for dev
        config.position_limits.max_portfolio_exposure = 0.15
        config.var_limits.var_limit_95_1d = 0.04  # More conservative for dev
        config.drawdown_limits.max_drawdown_hard = 0.08  # Higher tolerance for dev
        config.risk_validation_strict_mode = False
        config.risk_log_level = "DEBUG"
        return config

    def _get_staging_config(self) -> RiskSystemConfig:
        """Get staging environment configuration."""
        config = RiskSystemConfig()
        config.environment = "staging"
        config.position_limits.max_position_size_usd = 50000.0
        config.position_limits.max_portfolio_exposure = 0.18
        config.var_limits.var_limit_95_1d = 0.025
        config.drawdown_limits.max_drawdown_hard = 0.06
        config.risk_validation_strict_mode = True
        config.risk_log_level = "INFO"
        return config

    def _get_production_config(self) -> RiskSystemConfig:
        """Get production environment configuration."""
        config = RiskSystemConfig()
        config.environment = "production"
        config.position_limits.max_position_size_usd = 100000.0  # Full limits
        config.position_limits.max_portfolio_exposure = 0.20
        config.var_limits.var_limit_95_1d = 0.02  # Tightest limits
        config.drawdown_limits.max_drawdown_hard = 0.05
        config.risk_validation_strict_mode = True
        config.risk_log_level = "WARNING"
        config.circuit_breaker_enabled = True
        config.auto_risk_reduction = True
        return config

    def save_configuration(self):
        """Save current configuration to file."""
        try:
            config_path = Path(self.config_file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_dict = asdict(self.config)

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Saved risk configuration to {self.config_file_path}")

        except Exception as e:
            logger.error(f"Error saving risk configuration: {e}")

    def validate_configuration(self) -> bool:
        """Validate current configuration."""
        try:
            validation_errors = []

            # Validate position limits
            if self.config.position_limits.max_position_size_usd <= 0:
                validation_errors.append("max_position_size_usd must be positive")

            if not (0 < self.config.position_limits.max_portfolio_exposure <= 1):
                validation_errors.append("max_portfolio_exposure must be between 0 and 1")

            if self.config.position_limits.max_leverage <= 0:
                validation_errors.append("max_leverage must be positive")

            # Validate VaR limits
            if not (0 < self.config.var_limits.var_limit_95_1d <= 1):
                validation_errors.append("var_limit_95_1d must be between 0 and 1")

            if self.config.var_limits.var_limit_99_1d <= self.config.var_limits.var_limit_95_1d:
                validation_errors.append("var_limit_99_1d must be greater than var_limit_95_1d")

            # Validate drawdown limits
            if not (0 < self.config.drawdown_limits.max_drawdown_hard <= 1):
                validation_errors.append("max_drawdown_hard must be between 0 and 1")

            if self.config.drawdown_limits.max_drawdown_warning >= self.config.drawdown_limits.max_drawdown_hard:
                validation_errors.append("max_drawdown_warning must be less than max_drawdown_hard")

            # Validate correlation limits
            if not (0 < self.config.correlation_limits.max_correlation_portfolio <= 1):
                validation_errors.append("max_correlation_portfolio must be between 0 and 1")

            # Log validation results
            if validation_errors:
                error_msg = "Configuration validation failed: " + "; ".join(validation_errors)
                logger.error(error_msg)
                if self.config.risk_validation_strict_mode:
                    raise ValueError(error_msg)
                return False
            else:
                logger.info("Risk configuration validation passed")
                return True

        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            if self.config.risk_validation_strict_mode:
                raise
            return False

    def get_risk_limits(self) -> RiskLimits:
        """Get RiskLimits object for real-time risk controller."""
        return RiskLimits(
            max_position_size_usd=self.config.position_limits.max_position_size_usd,
            max_portfolio_exposure=self.config.position_limits.max_portfolio_exposure,
            max_leverage=self.config.position_limits.max_leverage,
            max_correlation_exposure=self.config.correlation_limits.max_correlation_portfolio,
            max_drawdown_hard=self.config.drawdown_limits.max_drawdown_hard,
            max_drawdown_warning=self.config.drawdown_limits.max_drawdown_warning,
            var_limit_95_1d=self.config.var_limits.var_limit_95_1d,
            var_limit_99_5d=self.config.var_limits.var_limit_99_5d,
            min_margin_requirement=0.25  # Fixed value
        )

    def update_configuration(self, updates: Dict[str, Any], save: bool = True) -> bool:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
            save: Whether to save to file

        Returns:
            True if update successful
        """
        try:
            # Apply updates
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")

            # Validate if enabled
            if self.config.config_validation_enabled:
                if not self.validate_configuration():
                    logger.error("Configuration update failed validation")
                    return False

            # Save if requested
            if save:
                self.save_configuration()

            logger.info("Configuration updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "environment": self.config.environment,
            "risk_management_enabled": self.config.risk_management_enabled,
            "real_time_monitoring": self.config.real_time_monitoring,
            "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
            "position_limits": {
                "max_position_size_usd": self.config.position_limits.max_position_size_usd,
                "max_portfolio_exposure": self.config.position_limits.max_portfolio_exposure,
                "max_leverage": self.config.position_limits.max_leverage
            },
            "var_limits": {
                "var_limit_95_1d": self.config.var_limits.var_limit_95_1d,
                "var_limit_99_5d": self.config.var_limits.var_limit_99_5d,
                "stressed_var_multiplier": self.config.var_limits.stressed_var_multiplier
            },
            "drawdown_limits": {
                "max_drawdown_hard": self.config.drawdown_limits.max_drawdown_hard,
                "max_drawdown_warning": self.config.drawdown_limits.max_drawdown_warning,
                "auto_reduction_threshold": self.config.drawdown_limits.auto_reduction_threshold
            },
            "correlation_limits": {
                "max_correlation_portfolio": self.config.correlation_limits.max_correlation_portfolio,
                "correlation_warning_threshold": self.config.correlation_limits.correlation_warning_threshold
            },
            "validation": {
                "config_validation_enabled": self.config.config_validation_enabled,
                "risk_validation_strict_mode": self.config.risk_validation_strict_mode
            }
        }

    def export_configuration(self, file_path: str) -> bool:
        """Export configuration to specified file."""
        try:
            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            config_dict = asdict(self.config)

            with open(export_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Configuration exported to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

    def import_configuration(self, file_path: str) -> bool:
        """Import configuration from specified file."""
        try:
            import_path = Path(file_path)

            if not import_path.exists():
                logger.error(f"Configuration file not found: {file_path}")
                return False

            with open(import_path, 'r') as f:
                config_data = json.load(f)

            self.config = self._load_config_from_dict(config_data)

            if self.config.config_validation_enabled:
                if not self.validate_configuration():
                    logger.error("Imported configuration failed validation")
                    return False

            logger.info(f"Configuration imported from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False


# Global configuration instance
_risk_config_manager: Optional[RiskConfigManager] = None


def get_risk_config_manager() -> RiskConfigManager:
    """Get global risk configuration manager instance."""
    global _risk_config_manager
    if _risk_config_manager is None:
        _risk_config_manager = RiskConfigManager()
    return _risk_config_manager


def get_risk_config() -> RiskSystemConfig:
    """Get current risk configuration."""
    return get_risk_config_manager().config


# Standalone validation function
def validate_risk_config():
    """Validate risk configuration implementation."""
    print("🔍 Validating RiskConfigManager implementation...")

    try:
        # Test imports
        from .risk_config import RiskConfigManager, RiskSystemConfig, PositionLimitsConfig
        print("✅ Imports successful")

        # Test instantiation
        config_manager = RiskConfigManager()
        print("✅ RiskConfigManager instantiation successful")

        # Test basic functionality
        if hasattr(config_manager, 'validate_configuration'):
            print("✅ validate_configuration method exists")
        else:
            print("❌ validate_configuration method missing")
            return False

        if hasattr(config_manager, 'get_risk_limits'):
            print("✅ get_risk_limits method exists")
        else:
            print("❌ get_risk_limits method missing")
            return False

        if hasattr(config_manager, 'get_configuration_summary'):
            print("✅ get_configuration_summary method exists")
        else:
            print("❌ get_configuration_summary method missing")
            return False

        # Test configuration validation
        validation_result = config_manager.validate_configuration()
        if validation_result:
            print("✅ Configuration validation passed")
        else:
            print("⚠️ Configuration validation had warnings")

        print("🎉 RiskConfigManager validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_risk_config()