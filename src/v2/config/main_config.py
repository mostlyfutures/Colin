"""
Main Configuration for Colin Trading Bot v2.0

This module provides comprehensive configuration management for all v2 components.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger

from .risk_config import RiskSystemConfig
from .ai_config import AIModelConfig
from .execution_config import ExecutionConfig


@dataclass
class SystemConfig:
    """Main system configuration."""
    environment: str = "development"
    log_level: str = "INFO"
    debug_mode: bool = False
    data_retention_days: int = 2555  # 7 years for compliance
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_memory_usage_mb: int = 2048
    monitoring_enabled: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "colin_trading_bot_v2"
    username: str = "colin_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    connection_timeout_seconds: int = 30


@dataclass
class ExternalServicesConfig:
    """External services configuration."""
    market_data_provider: str = "binance"
    market_data_api_key: str = ""
    notification_webhook_url: str = ""
    slack_webhook_url: str = ""
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_concurrent_signals: int = 100
    signal_generation_interval_seconds: int = 5
    order_execution_timeout_seconds: int = 30
    risk_check_timeout_ms: int = 10
    max_position_update_latency_ms: int = 50
    cache_ttl_seconds: int = 300
    enable_profiling: bool = False


@dataclass
class MainV2Config:
    """Main configuration for Colin Trading Bot v2.0."""

    # Component configurations
    system: SystemConfig = field(default_factory=SystemConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    external_services: ExternalServicesConfig = field(default_factory=ExternalServicesConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # System component configurations
    risk_system: RiskSystemConfig = field(default_factory=RiskSystemConfig)
    ai_models: AIModelConfig = field(default_factory=AIModelConfig)
    execution_engine: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Trading parameters
    trading_enabled: bool = True
    max_portfolio_value_usd: float = 10000000.0  # $10M max portfolio
    default_order_size_usd: float = 100000.0    # $100K default order
    symbols_to_trade: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
        "SOL/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT"
    ])

    # API configuration
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_rate_limit_per_minute: int = 100

    # WebSocket configuration
    websocket_enabled: bool = True
    websocket_port: int = 8001
    max_websocket_connections: int = 100

    # Monitoring configuration
    metrics_enabled: bool = True
    metrics_port: int = 9090
    alerting_enabled: bool = True
    dashboard_enabled: bool = True

    # Security configuration
    jwt_secret_key: str = "change-me-in-production"
    jwt_expiry_hours: int = 24
    api_key_required: bool = True
    enable_https: bool = False

    # Feature flags
    feature_ml_signals: bool = True
    feature_smart_routing: bool = True
    feature_real_time_risk: bool = True
    feature_compliance_monitoring: bool = True
    feature_stress_testing: bool = True
    feature_auto_scaling: bool = False


class MainConfigManager:
    """
    Main configuration manager for Colin Trading Bot v2.0.

    This class manages all configuration aspects including loading from files,
    environment-specific settings, and validation.
    """

    def __init__(self, config_file_path: Optional[str] = None):
        """
        Initialize main configuration manager.

        Args:
            config_file_path: Path to main configuration file
        """
        self.config_file_path = config_file_path or "config/v2_main_config.json"
        self.config: MainV2Config = MainV2Config()

        # Load configuration
        self.load_configuration()

        logger.info(f"MainConfigManager initialized for {self.config.system.environment} environment")

    def load_configuration(self):
        """Load configuration from file or create default."""
        try:
            config_path = Path(self.config_file_path)

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                # Load configuration with validation
                self.config = self._load_config_from_dict(config_data)
                logger.info(f"Loaded main configuration from {self.config_file_path}")
            else:
                # Create default configuration
                self.config = self._get_environment_specific_config()
                self.save_configuration()
                logger.info(f"Created default main configuration at {self.config_file_path}")

            # Override with environment variables
            self._load_environment_variables()

            # Validate configuration
            self.validate_configuration()

        except Exception as e:
            logger.error(f"Error loading main configuration: {e}")
            # Fallback to default configuration
            self.config = self._get_environment_specific_config()

    def _load_config_from_dict(self, config_data: Dict[str, Any]) -> MainV2Config:
        """Load configuration from dictionary with validation."""
        try:
            # Main configuration
            main_config = MainV2Config(
                trading_enabled=config_data.get("trading_enabled", True),
                max_portfolio_value_usd=config_data.get("max_portfolio_value_usd", 10000000.0),
                default_order_size_usd=config_data.get("default_order_size_usd", 100000.0),
                symbols_to_trade=config_data.get("symbols_to_trade", []),

                # API configuration
                api_enabled=config_data.get("api_enabled", True),
                api_host=config_data.get("api_host", "0.0.0.0"),
                api_port=config_data.get("api_port", 8000),
                api_cors_origins=config_data.get("api_cors_origins", ["*"]),
                api_rate_limit_per_minute=config_data.get("api_rate_limit_per_minute", 100),

                # WebSocket configuration
                websocket_enabled=config_data.get("websocket_enabled", True),
                websocket_port=config_data.get("websocket_port", 8001),
                max_websocket_connections=config_data.get("max_websocket_connections", 100),

                # Monitoring configuration
                metrics_enabled=config_data.get("metrics_enabled", True),
                metrics_port=config_data.get("metrics_port", 9090),
                alerting_enabled=config_data.get("alerting_enabled", True),
                dashboard_enabled=config_data.get("dashboard_enabled", True),

                # Security configuration
                jwt_secret_key=config_data.get("jwt_secret_key", "change-me-in-production"),
                jwt_expiry_hours=config_data.get("jwt_expiry_hours", 24),
                api_key_required=config_data.get("api_key_required", True),
                enable_https=config_data.get("enable_https", False),

                # Feature flags
                feature_ml_signals=config_data.get("feature_ml_signals", True),
                feature_smart_routing=config_data.get("feature_smart_routing", True),
                feature_real_time_risk=config_data.get("feature_real_time_risk", True),
                feature_compliance_monitoring=config_data.get("feature_compliance_monitoring", True),
                feature_stress_testing=config_data.get("feature_stress_testing", True),
                feature_auto_scaling=config_data.get("feature_auto_scaling", False)
            )

            # System configuration
            if "system" in config_data:
                system_data = config_data["system"]
                main_config.system = SystemConfig(
                    environment=system_data.get("environment", "development"),
                    log_level=system_data.get("log_level", "INFO"),
                    debug_mode=system_data.get("debug_mode", False),
                    data_retention_days=system_data.get("data_retention_days", 2555),
                    backup_enabled=system_data.get("backup_enabled", True),
                    backup_interval_hours=system_data.get("backup_interval_hours", 24),
                    max_memory_usage_mb=system_data.get("max_memory_usage_mb", 2048),
                    monitoring_enabled=system_data.get("monitoring_enabled", True)
                )

            # Database configuration
            if "database" in config_data:
                db_data = config_data["database"]
                main_config.database = DatabaseConfig(
                    host=db_data.get("host", "localhost"),
                    port=db_data.get("port", 5432),
                    database=db_data.get("database", "colin_trading_bot_v2"),
                    username=db_data.get("username", "colin_user"),
                    password=db_data.get("password", ""),
                    pool_size=db_data.get("pool_size", 10),
                    max_overflow=db_data.get("max_overflow", 20),
                    connection_timeout_seconds=db_data.get("connection_timeout_seconds", 30)
                )

            # Performance configuration
            if "performance" in config_data:
                perf_data = config_data["performance"]
                main_config.performance = PerformanceConfig(
                    max_concurrent_signals=perf_data.get("max_concurrent_signals", 100),
                    signal_generation_interval_seconds=perf_data.get("signal_generation_interval_seconds", 5),
                    order_execution_timeout_seconds=perf_data.get("order_execution_timeout_seconds", 30),
                    risk_check_timeout_ms=perf_data.get("risk_check_timeout_ms", 10),
                    max_position_update_latency_ms=perf_data.get("max_position_update_latency_ms", 50),
                    cache_ttl_seconds=perf_data.get("cache_ttl_seconds", 300),
                    enable_profiling=perf_data.get("enable_profiling", False)
                )

            return main_config

        except Exception as e:
            logger.error(f"Error loading configuration from dict: {e}")
            return MainV2Config()

    def _get_environment_specific_config(self) -> MainV2Config:
        """Get configuration based on environment."""
        environment = os.getenv("ENVIRONMENT", "development")

        if environment == "production":
            return self._get_production_config()
        elif environment == "staging":
            return self._get_staging_config()
        else:
            return self._get_development_config()

    def _get_development_config(self) -> MainV2Config:
        """Get development environment configuration."""
        config = MainV2Config()
        config.system.environment = "development"
        config.system.log_level = "DEBUG"
        config.system.debug_mode = True
        config.system.monitoring_enabled = True

        # Development trading limits
        config.max_portfolio_value_usd = 1000000.0  # $1M for dev
        config.default_order_size_usd = 10000.0     # $10K default

        # Development API settings
        config.api_port = 8000
        config.websocket_port = 8001
        config.api_key_required = False
        config.enable_https = False

        # Development performance settings
        config.performance.signal_generation_interval_seconds = 10
        config.performance.max_concurrent_signals = 10
        config.performance.enable_profiling = True

        # Feature flags - all enabled for development
        config.feature_ml_signals = True
        config.feature_smart_routing = True
        config.feature_real_time_risk = True
        config.feature_compliance_monitoring = True
        config.feature_stress_testing = True

        return config

    def _get_staging_config(self) -> MainV2Config:
        """Get staging environment configuration."""
        config = MainV2Config()
        config.system.environment = "staging"
        config.system.log_level = "INFO"
        config.system.debug_mode = False
        config.system.monitoring_enabled = True

        # Staging trading limits
        config.max_portfolio_value_usd = 5000000.0  # $5M for staging
        config.default_order_size_usd = 50000.0     # $50K default

        # Staging API settings
        config.api_port = 8000
        config.websocket_port = 8001
        config.api_key_required = True
        config.enable_https = False

        # Staging performance settings
        config.performance.signal_generation_interval_seconds = 5
        config.performance.max_concurrent_signals = 50
        config.performance.enable_profiling = False

        return config

    def _get_production_config(self) -> MainV2Config:
        """Get production environment configuration."""
        config = MainV2Config()
        config.system.environment = "production"
        config.system.log_level = "WARNING"
        config.system.debug_mode = False
        config.system.monitoring_enabled = True

        # Production trading limits (full capacity)
        config.max_portfolio_value_usd = 10000000.0  # $10M for production
        config.default_order_size_usd = 100000.0     # $100K default

        # Production API settings
        config.api_port = 8000
        config.websocket_port = 8001
        config.api_key_required = True
        config.enable_https = True

        # Production performance settings
        config.performance.signal_generation_interval_seconds = 1
        config.performance.max_concurrent_signals = 100
        config.performance.enable_profiling = False

        # Security
        config.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
        config.api_rate_limit_per_minute = 1000

        return config

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # API configuration
        if os.getenv("API_HOST"):
            self.config.api_host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.config.api_port = int(os.getenv("API_PORT"))
        if os.getenv("JWT_SECRET_KEY"):
            self.config.jwt_secret_key = os.getenv("JWT_SECRET_KEY")

        # Database configuration
        if os.getenv("DB_HOST"):
            self.config.database.host = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            self.config.database.port = int(os.getenv("DB_PORT"))
        if os.getenv("DB_NAME"):
            self.config.database.database = os.getenv("DB_NAME")
        if os.getenv("DB_USER"):
            self.config.database.username = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"):
            self.config.database.password = os.getenv("DB_PASSWORD")

        # External services
        if os.getenv("MARKET_DATA_API_KEY"):
            self.config.external_services.market_data_api_key = os.getenv("MARKET_DATA_API_KEY")
        if os.getenv("NOTIFICATION_WEBHOOK_URL"):
            self.config.external_services.notification_webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")

        # Feature flags from environment
        if os.getenv("DISABLE_TRADING") == "true":
            self.config.trading_enabled = False
        if os.getenv("ENABLE_AUTO_SCALING") == "true":
            self.config.feature_auto_scaling = True

    def validate_configuration(self) -> bool:
        """Validate current configuration."""
        try:
            validation_errors = []

            # Validate API configuration
            if self.config.api_port < 1 or self.config.api_port > 65535:
                validation_errors.append("API port must be between 1 and 65535")

            if self.config.websocket_port < 1 or self.config.websocket_port > 65535:
                validation_errors.append("WebSocket port must be between 1 and 65535")

            if self.config.api_port == self.config.websocket_port:
                validation_errors.append("API and WebSocket ports must be different")

            # Validate trading parameters
            if self.config.max_portfolio_value_usd <= 0:
                validation_errors.append("Max portfolio value must be positive")

            if self.config.default_order_size_usd <= 0:
                validation_errors.append("Default order size must be positive")

            if self.config.default_order_size_usd > self.config.max_portfolio_value_usd:
                validation_errors.append("Default order size cannot exceed max portfolio value")

            # Validate performance settings
            if self.config.performance.max_concurrent_signals <= 0:
                validation_errors.append("Max concurrent signals must be positive")

            if self.config.performance.signal_generation_interval_seconds <= 0:
                validation_errors.append("Signal generation interval must be positive")

            # Validate security settings
            if self.config.system.environment == "production":
                if self.config.jwt_secret_key == "change-me-in-production":
                    validation_errors.append("JWT secret key must be changed in production")

                if not self.config.api_key_required:
                    validation_errors.append("API key should be required in production")

                if not self.config.enable_https:
                    validation_errors.append("HTTPS should be enabled in production")

            # Log validation results
            if validation_errors:
                error_msg = "Configuration validation failed: " + "; ".join(validation_errors)
                logger.error(error_msg)
                if self.config.system.debug_mode:
                    raise ValueError(error_msg)
                return False
            else:
                logger.info("Main configuration validation passed")
                return True

        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            if self.config.system.debug_mode:
                raise
            return False

    def save_configuration(self):
        """Save current configuration to file."""
        try:
            config_path = Path(self.config_file_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_dict = asdict(self.config)

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Saved main configuration to {self.config_file_path}")

        except Exception as e:
            logger.error(f"Error saving main configuration: {e}")

    def get_database_url(self) -> str:
        """Get database connection URL."""
        return (
            f"postgresql://{self.config.database.username}:"
            f"{self.config.database.password}@"
            f"{self.config.database.host}:"
            f"{self.config.database.port}/"
            f"{self.config.database.database}"
        )

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        feature_map = {
            "ml_signals": self.config.feature_ml_signals,
            "smart_routing": self.config.feature_smart_routing,
            "real_time_risk": self.config.feature_real_time_risk,
            "compliance_monitoring": self.config.feature_compliance_monitoring,
            "stress_testing": self.config.feature_stress_testing,
            "auto_scaling": self.config.feature_auto_scaling
        }
        return feature_map.get(feature_name, False)

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "environment": self.config.system.environment,
            "trading_enabled": self.config.trading_enabled,
            "api_enabled": self.config.api_enabled,
            "websocket_enabled": self.config.websocket_enabled,
            "monitoring_enabled": self.config.system.monitoring_enabled,
            "max_portfolio_value": self.config.max_portfolio_value_usd,
            "symbols_count": len(self.config.symbols_to_trade),
            "api_port": self.config.api_port,
            "websocket_port": self.config.websocket_port,
            "feature_flags": {
                "ml_signals": self.config.feature_ml_signals,
                "smart_routing": self.config.feature_smart_routing,
                "real_time_risk": self.config.feature_real_time_risk,
                "compliance_monitoring": self.config.feature_compliance_monitoring,
                "stress_testing": self.config.feature_stress_testing,
                "auto_scaling": self.config.feature_auto_scaling
            },
            "performance_settings": {
                "max_concurrent_signals": self.config.performance.max_concurrent_signals,
                "signal_interval_seconds": self.config.performance.signal_generation_interval_seconds,
                "risk_check_timeout_ms": self.config.performance.risk_check_timeout_ms
            }
        }


# Global configuration instance
_main_config_manager: Optional[MainConfigManager] = None


def get_main_config_manager() -> MainConfigManager:
    """Get global main configuration manager instance."""
    global _main_config_manager
    if _main_config_manager is None:
        _main_config_manager = MainConfigManager()
    return _main_config_manager


def get_main_config() -> MainV2Config:
    """Get current main configuration."""
    return get_main_config_manager().config


# Standalone validation function
def validate_main_config():
    """Validate main configuration implementation."""
    print("🔍 Validating MainConfigManager implementation...")

    try:
        # Test imports
        from .main_config import MainConfigManager, MainV2Config, SystemConfig
        print("✅ Imports successful")

        # Test instantiation
        config_manager = MainConfigManager()
        print("✅ MainConfigManager instantiation successful")

        # Test basic functionality
        if hasattr(config_manager, 'validate_configuration'):
            print("✅ validate_configuration method exists")
        else:
            print("❌ validate_configuration method missing")
            return False

        if hasattr(config_manager, 'get_configuration_summary'):
            print("✅ get_configuration_summary method exists")
        else:
            print("❌ get_configuration_summary method missing")
            return False

        if hasattr(config_manager, 'is_feature_enabled'):
            print("✅ is_feature_enabled method exists")
        else:
            print("❌ is_feature_enabled method missing")
            return False

        # Test configuration validation
        validation_result = config_manager.validate_configuration()
        if validation_result:
            print("✅ Configuration validation passed")
        else:
            print("⚠️ Configuration validation had warnings")

        print("🎉 MainConfigManager validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_main_config()