"""
Bot integration layer for the Colin Trading Bot CLI.

This module provides integration with the main Colin Trading Bot system,
including async wrappers, context managers, and bot state management.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, AsyncContextManager
from contextlib import asynccontextmanager
from datetime import datetime
import json

from ..utils.error_handler import BotError, ConfigurationError, with_error_handling


class BotNotInstalledError(BotError):
    """Raised when the main bot system is not available."""
    pass


class BotIntegrationError(BotError):
    """Raised when bot integration fails."""
    pass


class ColinBotWrapper:
    """
    Async wrapper for the Colin Trading Bot system.

    This class provides a clean interface between the CLI and the main bot
    system, handling initialization, state management, and async operations.
    """

    def __init__(self, mode: str = "development", config_path: Optional[str] = None):
        self.mode = mode
        self.config_path = config_path
        self.bot_instance = None
        self.initialized = False
        self.running = False

        # Add project root to path for imports
        self._setup_python_path()

    def _setup_python_path(self):
        """Add the project root to Python path for imports."""
        project_root = Path(__file__).parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    async def initialize(self) -> bool:
        """
        Initialize the bot system.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Check if bot system is available
            self._check_bot_availability()

            # Load configuration
            config = await self._load_configuration()

            # Initialize bot components
            await self._initialize_components(config)

            self.initialized = True
            return True

        except Exception as e:
            raise BotIntegrationError(f"Failed to initialize bot: {str(e)}")

    def _check_bot_availability(self):
        """Check if the main bot system is available for import."""
        try:
            # Try to import main bot modules
            import sys
            project_root = Path(__file__).parent.parent.parent.parent

            # Check for v2 system
            v2_path = project_root / "src" / "v2"
            if v2_path.exists():
                sys.path.insert(0, str(v2_path))
                # Try importing main components
                try:
                    from main import ColinTradingBot  # type: ignore
                    self.bot_module_path = "src.v2"
                    return
                except ImportError:
                    pass

            # Check for colin_bot package
            colin_bot_path = project_root / "colin_bot" / "v2"
            if colin_bot_path.exists():
                sys.path.insert(0, str(colin_bot_path))
                try:
                    from main import ColinTradingBot  # type: ignore
                    self.bot_module_path = "colin_bot.v2"
                    return
                except ImportError:
                    pass

            raise BotNotInstalledError(
                "Colin Trading Bot system not found. Please ensure the bot is properly installed."
            )

        except ImportError as e:
            raise BotNotInstalledError(f"Cannot import bot system: {str(e)}")

    async def _load_configuration(self) -> Dict[str, Any]:
        """Load bot configuration."""
        config_file = self.config_path or f"config/{self.mode}.yaml"

        if not os.path.exists(config_file):
            # Create default configuration
            config = self._get_default_config()
        else:
            # Load existing configuration
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
            except ImportError:
                # Fallback to JSON if yaml not available
                with open(config_file.replace('.yaml', '.json'), 'r') as f:
                    config = json.load(f)

        # Merge with defaults
        default_config = self._get_default_config()
        return {**default_config, **config}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default bot configuration."""
        return {
            'system': {
                'environment': self.mode,
                'log_level': 'INFO',
                'debug_mode': self.mode == 'development',
            },
            'trading': {
                'enabled': self.mode != 'development',
                'simulation_mode': self.mode != 'production',
                'max_portfolio_value_usd': 100000.0,
            },
            'market_data': {
                'primary_source': 'coingecko',
                'fallback_sources': ['kraken', 'cryptocompare'],
                'cache_ttl_seconds': 300,
            },
            'risk_management': {
                'max_position_size_usd': 10000.0,
                'max_portfolio_exposure': 0.20,
                'max_leverage': 3.0,
            },
            'api': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 8000,
                'require_auth': False,
            }
        }

    async def _initialize_components(self, config: Dict[str, Any]):
        """Initialize bot components based on configuration."""
        # Mock initialization for demonstration
        # In real implementation, this would initialize actual bot components
        component_tasks = []

        # Initialize AI Engine
        if config.get('ai_engine', {}).get('enabled', True):
            component_tasks.append(self._init_ai_engine(config))

        # Initialize Execution Engine
        if config.get('execution_engine', {}).get('enabled', True):
            component_tasks.append(self._init_execution_engine(config))

        # Initialize Risk Management
        if config.get('risk_management', {}).get('enabled', True):
            component_tasks.append(self._init_risk_management(config))

        # Initialize Market Data
        if config.get('market_data', {}).get('enabled', True):
            component_tasks.append(self._init_market_data(config))

        # Wait for all components to initialize
        if component_tasks:
            await asyncio.gather(*component_tasks, return_exceptions=True)

    async def _init_ai_engine(self, config: Dict[str, Any]):
        """Initialize AI Engine component."""
        # Mock initialization
        await asyncio.sleep(0.1)
        return True

    async def _init_execution_engine(self, config: Dict[str, Any]):
        """Initialize Execution Engine component."""
        # Mock initialization
        await asyncio.sleep(0.1)
        return True

    async def _init_risk_management(self, config: Dict[str, Any]):
        """Initialize Risk Management component."""
        # Mock initialization
        await asyncio.sleep(0.1)
        return True

    async def _init_market_data(self, config: Dict[str, Any]):
        """Initialize Market Data component."""
        # Mock initialization
        await asyncio.sleep(0.1)
        return True

    async def start(self) -> bool:
        """Start the bot system."""
        if not self.initialized:
            await self.initialize()

        try:
            # Start bot components
            await self._start_components()
            self.running = True
            return True

        except Exception as e:
            raise BotIntegrationError(f"Failed to start bot: {str(e)}")

    async def _start_components(self):
        """Start all bot components."""
        # Mock component startup
        start_tasks = [
            self._start_ai_engine(),
            self._start_execution_engine(),
            self._start_risk_management(),
            self._start_market_data(),
        ]

        await asyncio.gather(*start_tasks, return_exceptions=True)

    async def _start_ai_engine(self):
        """Start AI Engine component."""
        await asyncio.sleep(0.1)
        return True

    async def _start_execution_engine(self):
        """Start Execution Engine component."""
        await asyncio.sleep(0.1)
        return True

    async def _start_risk_management(self):
        """Start Risk Management component."""
        await asyncio.sleep(0.1)
        return True

    async def _start_market_data(self):
        """Start Market Data component."""
        await asyncio.sleep(0.1)
        return True

    async def stop(self):
        """Stop the bot system."""
        if self.running:
            try:
                # Stop bot components
                await self._stop_components()
                self.running = False
            except Exception as e:
                raise BotIntegrationError(f"Failed to stop bot: {str(e)}")

    async def _stop_components(self):
        """Stop all bot components."""
        # Mock component shutdown
        stop_tasks = [
            self._stop_ai_engine(),
            self._stop_execution_engine(),
            self._stop_risk_management(),
            self._stop_market_data(),
        ]

        await asyncio.gather(*stop_tasks, return_exceptions=True)

    async def _stop_ai_engine(self):
        """Stop AI Engine component."""
        await asyncio.sleep(0.1)
        return True

    async def _stop_execution_engine(self):
        """Stop Execution Engine component."""
        await asyncio.sleep(0.1)
        return True

    async def _stop_risk_management(self):
        """Stop Risk Management component."""
        await asyncio.sleep(0.1)
        return True

    async def _stop_market_data(self):
        """Stop Market Data component."""
        await asyncio.sleep(0.1)
        return True

    async def generate_signals(self, symbols: List[str], timeframe: str = "1h") -> List[Dict[str, Any]]:
        """Generate trading signals for specified symbols."""
        if not self.running:
            raise BotIntegrationError("Bot is not running")

        # Mock signal generation
        signals = []
        for symbol in symbols:
            signal = {
                'symbol': symbol,
                'direction': 'BUY' if hash(symbol) % 2 else 'HOLD',
                'strength': 0.3 + (hash(symbol) % 70) / 100,
                'confidence': 0.5 + (hash(symbol) % 40) / 100,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'price': 3980.50 if 'ETH' in symbol else 67200.00,
            }
            signals.append(signal)

        return signals

    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        """Create a new order."""
        if not self.running:
            raise BotIntegrationError("Bot is not running")

        # Mock order creation
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount,
            'price': price or (3980.50 if 'ETH' in symbol else 67200.00),
            'status': 'filled' if order_type == 'market' else 'open',
            'created_at': datetime.now(),
            'filled_amount': amount if order_type == 'market' else 0,
        }

        return order

    async def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio information."""
        if not self.running:
            raise BotIntegrationError("Bot is not running")

        # Mock portfolio data
        portfolio = {
            'total_value_usd': 124580.50,
            'available_balance_usd': 45230.00,
            'unrealized_pnl': 2340.25,
            'pnl_percent': 1.92,
            'positions': [
                {
                    'symbol': 'ETH/USDT',
                    'side': 'long',
                    'size': 2.5,
                    'entry_price': 3850.00,
                    'current_price': 3980.50,
                    'unrealized_pnl': 326.25,
                }
            ],
        }

        return portfolio

    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market data for specified symbols."""
        if not self.running:
            raise BotIntegrationError("Bot is not running")

        # Mock market data
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = {
                'price': 3980.50 if 'ETH' in symbol else 67200.00,
                'change_24h': 2.3 if 'ETH' in symbol else 1.8,
                'volume_24h': 1250000000 if 'ETH' in symbol else 28000000000,
                'high_24h': 4050.00 if 'ETH' in symbol else 68500.00,
                'low_24h': 3850.00 if 'ETH' in symbol else 66000.00,
            }

        return market_data

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        status = {
            'initialized': self.initialized,
            'running': self.running,
            'mode': self.mode,
            'uptime': datetime.now().isoformat() if self.initialized else None,
            'components': {
                'ai_engine': 'healthy' if self.running else 'stopped',
                'execution_engine': 'healthy' if self.running else 'stopped',
                'risk_management': 'healthy' if self.running else 'stopped',
                'market_data': 'healthy' if self.running else 'stopped',
            },
            'performance': {
                'cpu_usage': '23%',
                'memory_usage': '4.7GB',
                'response_time': '45ms',
            }
        }

        return status


@asynccontextmanager
async def bot_session(mode: str = "development", config_path: Optional[str] = None):
    """
    Context manager for bot sessions.

    Usage:
        async with bot_session("development") as bot:
            signals = await bot.generate_signals(["ETH/USDT"])
            print(signals)
    """
    bot = ColinBotWrapper(mode=mode, config_path=config_path)

    try:
        await bot.initialize()
        await bot.start()
        yield bot
    finally:
        await bot.stop()


class BotManager:
    """
    Manager class for handling multiple bot instances and sessions.

    This provides a higher-level interface for managing bot lifecycle,
    including multiple instances and session tracking.
    """

    def __init__(self):
        self.active_sessions: Dict[str, ColinBotWrapper] = {}
        self.default_mode = "development"

    async def create_session(self, session_id: str, mode: str = None,
                           config_path: Optional[str] = None) -> ColinBotWrapper:
        """Create a new bot session."""
        if session_id in self.active_sessions:
            raise BotIntegrationError(f"Session '{session_id}' already exists")

        mode = mode or self.default_mode
        bot = ColinBotWrapper(mode=mode, config_path=config_path)

        await bot.initialize()
        await bot.start()

        self.active_sessions[session_id] = bot
        return bot

    async def get_session(self, session_id: str) -> Optional[ColinBotWrapper]:
        """Get an existing bot session."""
        return self.active_sessions.get(session_id)

    async def close_session(self, session_id: str):
        """Close a bot session."""
        if session_id in self.active_sessions:
            await self.active_sessions[session_id].stop()
            del self.active_sessions[session_id]

    async def close_all_sessions(self):
        """Close all active sessions."""
        for session_id in list(self.active_sessions.keys()):
            await self.close_session(session_id)

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.active_sessions.keys())

    async def run_in_session(self, session_id: str, coro):
        """Run a coroutine within a specific bot session."""
        bot = await self.get_session(session_id)
        if not bot:
            raise BotIntegrationError(f"Session '{session_id}' not found")

        return await coro(bot)


# Global bot manager instance
_bot_manager = BotManager()


def get_bot_manager() -> BotManager:
    """Get the global bot manager instance."""
    return _bot_manager


# Convenience functions
async def quick_signal_generation(symbols: List[str], timeframe: str = "1h") -> List[Dict[str, Any]]:
    """Quick signal generation without session management."""
    async with bot_session("quick_session") as bot:
        return await bot.generate_signals(symbols, timeframe)


async def quick_order_create(symbol: str, side: str, amount: float) -> Dict[str, Any]:
    """Quick order creation without session management."""
    async with bot_session("quick_order_session") as bot:
        return await bot.create_order(symbol, side, amount)


async def quick_portfolio_check() -> Dict[str, Any]:
    """Quick portfolio check without session management."""
    async with bot_session("quick_portfolio_session") as bot:
        return await bot.get_portfolio()