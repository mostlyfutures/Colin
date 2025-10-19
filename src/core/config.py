"""
Configuration management for Colin Trading Bot.

Handles loading and validating configuration from YAML files.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SessionConfig(BaseModel):
    """Session configuration."""
    start: str
    end: str
    weight: float = Field(gt=0, description="Session weight multiplier")


class APIConfig(BaseModel):
    """API configuration."""
    base_url: str
    rate_limit: Optional[int] = None
    testnet: bool = False


class ICTConfig(BaseModel):
    """ICT (Institutional Candlestick Theory) configuration."""
    fair_value_gap: Dict[str, Any]
    order_block: Dict[str, Any]
    break_of_structure: Dict[str, Any]


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    weights: Dict[str, float] = Field(description="Signal weights")
    thresholds: Dict[str, int] = Field(description="Confidence thresholds")


class OrderFlowConfig(BaseModel):
    """Order flow analysis configuration."""
    order_book: Dict[str, Any]
    trade_delta: Dict[str, Any]


class LiquidationConfig(BaseModel):
    """Liquidation analysis configuration."""
    heatmap: Dict[str, Any]
    proximity_threshold: float


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size: float = Field(gt=0, le=1)
    stop_loss_buffer: float = Field(gt=0)
    volatility_threshold: float = Field(gt=0)


class OutputConfig(BaseModel):
    """Output configuration."""
    format: str = Field(pattern="^(json|text|both)$")
    include_rationale: bool = True
    max_rationale_points: int = Field(gt=0, le=10)
    include_stop_loss: bool = True
    include_volatility_warning: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file: Optional[str] = None
    max_size: Optional[str] = None
    backup_count: Optional[int] = None


class DevelopmentConfig(BaseModel):
    """Development configuration."""
    test_mode: bool = False
    mock_api_responses: bool = False
    save_intermediate_data: bool = False


class Config(BaseModel):
    """Main configuration model."""
    symbols: list[str]
    apis: Dict[str, APIConfig]
    sessions: Dict[str, SessionConfig]
    ict: ICTConfig
    scoring: ScoringConfig
    order_flow: OrderFlowConfig
    liquidations: LiquidationConfig
    risk: RiskConfig
    output: OutputConfig
    logging: LoggingConfig
    development: DevelopmentConfig


class ConfigManager:
    """Manages configuration loading and access."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. Defaults to config.yaml
        """
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config.yaml")

        self.config_path = Path(config_path)
        self._config: Optional[Config] = None

    def load_config(self) -> Config:
        """
        Load configuration from file.

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            self._config = Config(**config_data)
            return self._config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config: {e}")

    def _substitute_env_vars(self, data: Any) -> Any:
        """
        Recursively substitute environment variables in config data.

        Args:
            data: Configuration data

        Returns:
            Data with environment variables substituted
        """
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            env_var = data[2:-1]
            default_value = None

            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)

            return os.getenv(env_var, default_value)
        return data

    @property
    def config(self) -> Config:
        """Get loaded configuration."""
        if self._config is None:
            self.load_config()
        return self._config

    def get_api_key(self, api_name: str) -> Optional[str]:
        """
        Get API key from environment variables.

        Args:
            api_name: Name of the API (e.g., 'binance')

        Returns:
            API key if found, None otherwise
        """
        return os.getenv(f"{api_name.upper()}_API_KEY")

    def get_api_secret(self, api_name: str) -> Optional[str]:
        """
        Get API secret from environment variables.

        Args:
            api_name: Name of the API (e.g., 'binance')

        Returns:
            API secret if found, None otherwise
        """
        return os.getenv(f"{api_name.upper()}_API_SECRET")


# Global configuration instance
config_manager = ConfigManager()