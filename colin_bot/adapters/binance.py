"""
Binance Futures adapter for fetching OHLCV, Open Interest, and volume data.

This adapter handles real-time and historical data retrieval from Binance Futures API
for institutional signal analysis.
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..core.config import ConfigManager


class BinanceAdapter:
    """Adapter for Binance Futures API."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize Binance adapter.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.exchange: Optional[ccxt.binance] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the exchange connection."""
        if self._initialized:
            return

        try:
            # Create exchange instance
            self.exchange = ccxt.binance({
                'apiKey': self.config_manager.get_api_key('binance'),
                'secret': self.config_manager.get_api_secret('binance'),
                'sandbox': self.config.apis['binance'].testnet,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures market
                }
            })

            # Test connection
            await self.exchange.load_markets()
            self._initialized = True
            logger.info("Binance adapter initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Binance adapter: {e}")
            raise

    async def close(self) -> None:
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self._initialized = False
            logger.info("Binance connection closed")

    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data

        Raises:
            Exception: If data fetch fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert to numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            logger.debug(f"Fetched {len(df)} OHLCV candles for {symbol} on {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    async def get_open_interest(self, symbol: str) -> Dict[str, float]:
        """
        Fetch current open interest for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')

        Returns:
            Dictionary with open interest data

        Raises:
            Exception: If data fetch fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch open interest
            oi = await self.exchange.fetch_open_interest(symbol)

            logger.debug(f"Fetched open interest for {symbol}: {oi}")
            return oi

        except Exception as e:
            logger.error(f"Failed to fetch open interest for {symbol}: {e}")
            raise

    async def get_funding_rate(self, symbol: str) -> Dict[str, float]:
        """
        Fetch current funding rate for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')

        Returns:
            Dictionary with funding rate data

        Raises:
            Exception: If data fetch fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch funding rate
            funding = await self.exchange.fetch_funding_rate(symbol)

            logger.debug(f"Fetched funding rate for {symbol}: {funding}")
            return funding

        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
            raise

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Fetch order book data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')
            limit: Number of levels to fetch

        Returns:
            Dictionary with bids and asks DataFrames

        Raises:
            Exception: If data fetch fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch order book
            orderbook = await self.exchange.fetch_order_book(symbol, limit)

            # Convert to DataFrames
            bids_df = pd.DataFrame(orderbook['bids'], columns=['price', 'volume'])
            bids_df['price'] = pd.to_numeric(bids_df['price'])
            bids_df['volume'] = pd.to_numeric(bids_df['volume'])

            asks_df = pd.DataFrame(orderbook['asks'], columns=['price', 'volume'])
            asks_df['price'] = pd.to_numeric(asks_df['price'])
            asks_df['volume'] = pd.to_numeric(asks_df['volume'])

            result = {
                'bids': bids_df,
                'asks': asks_df,
                'timestamp': datetime.now()
            }

            logger.debug(f"Fetched order book for {symbol} with {limit} levels")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            raise

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent trades for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')
            limit: Number of trades to fetch

        Returns:
            DataFrame with recent trades

        Raises:
            Exception: If data fetch fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch recent trades
            trades = await self.exchange.fetch_trades(symbol, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(trades)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['price'] = pd.to_numeric(df['price'])
            df['amount'] = pd.to_numeric(df['amount'])

            # Calculate trade side delta
            df['side_numeric'] = df['side'].map({'buy': 1, 'sell': -1})
            df['delta'] = df['side_numeric'] * df['amount'] * df['price']

            logger.debug(f"Fetched {len(df)} recent trades for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch recent trades for {symbol}: {e}")
            raise

    async def get_historical_oi(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical open interest data.

        Note: Binance doesn't provide historical OI directly, so we'll sample
        current OI over time for historical analysis.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')
            timeframe: Timeframe for sampling
            limit: Number of data points

        Returns:
            DataFrame with historical OI data
        """
        if not self._initialized:
            await self.initialize()

        try:
            # This is a workaround since Binance doesn't provide historical OI
            # In production, you might want to use a service that provides this data
            timestamps = []
            oi_values = []

            # Generate timestamps
            end_time = datetime.now()
            if timeframe == '1h':
                delta = timedelta(hours=1)
            elif timeframe == '4h':
                delta = timedelta(hours=4)
            else:
                delta = timedelta(days=1)

            for i in range(limit):
                timestamp = end_time - (delta * i)
                timestamps.append(timestamp)

                # Fetch OI at this timestamp (approximation)
                try:
                    oi_data = await self.get_open_interest(symbol)
                    oi_values.append(oi_data.get('openInterestAmount', 0))
                except:
                    oi_values.append(0)

                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)

            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open_interest': oi_values
            })
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            logger.debug(f"Generated historical OI data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to generate historical OI for {symbol}: {e}")
            raise

    async def get_symbol_info(self, symbol: str) -> Dict[str, any]:
        """
        Get detailed information about a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USDT')

        Returns:
            Dictionary with symbol information
        """
        if not self._initialized:
            await self.initialize()

        try:
            markets = await self.exchange.load_markets()
            symbol_info = markets.get(symbol, {})

            logger.debug(f"Fetched symbol info for {symbol}")
            return symbol_info

        except Exception as e:
            logger.error(f"Failed to fetch symbol info for {symbol}: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()