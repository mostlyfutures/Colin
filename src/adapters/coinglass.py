"""
CoinGlass adapter for fetching liquidation heatmap data.

This adapter retrieves liquidation data from CoinGlass API to identify
stop-hunt zones and liquidity clusters for institutional signal analysis.
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger

from ..core.config import ConfigManager


class CoinGlassAdapter:
    """Adapter for CoinGlass liquidation data API."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize CoinGlass adapter.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.base_url = self.config.apis['coinglass'].base_url
        self.rate_limit = self.config.apis['coinglass'].rate_limit
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0

    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("CoinGlass adapter initialized")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("CoinGlass connection closed")

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make HTTP request to CoinGlass API with rate limiting.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            Exception: If request fails
        """
        if not self.session:
            await self.initialize()

        # Rate limiting
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        min_interval = 60 / self.rate_limit if self.rate_limit else 0

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                self._last_request_time = asyncio.get_event_loop().time()
                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Request to CoinGlass API failed: {e}")
            raise

    async def get_liquidation_heatmap(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Fetch liquidation heatmap data.

        Args:
            symbol: Trading symbol (e.g., 'ETH')
            timeframe: Timeframe (1h, 4h, 24h)
            limit: Number of data points

        Returns:
            DataFrame with liquidation heatmap data

        Raises:
            Exception: If data fetch fails
        """
        try:
            endpoint = "/liquidation/heatmap/model3"
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit
            }

            data = await self._make_request(endpoint, params)

            # Extract liquidation levels
            liquidations = []
            for item in data.get('data', []):
                liquidations.append({
                    'timestamp': pd.to_datetime(item.get('createTime'), unit='ms'),
                    'price': float(item.get('price', 0)),
                    'liquidation_long': float(item.get('longAmount', 0)),
                    'liquidation_short': float(item.get('shortAmount', 0)),
                    'total_liquidation': float(item.get('amount', 0)),
                    'level': item.get('level', 'medium')
                })

            df = pd.DataFrame(liquidations)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

            logger.debug(f"Fetched {len(df)} liquidation data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch liquidation heatmap for {symbol}: {e}")
            raise

    async def get_liquidation_levels(
        self,
        symbol: str,
        side: str = "both"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get current liquidation levels for long and short positions.

        Args:
            symbol: Trading symbol (e.g., 'ETH')
            side: Side to fetch ('long', 'short', 'both')

        Returns:
            Dictionary with long and short liquidation DataFrames

        Raises:
            Exception: If data fetch fails
        """
        try:
            endpoint = "/v1/liquidation"
            params = {
                "symbol": symbol,
                "side": side
            }

            data = await self._make_request(endpoint, params)

            long_levels = []
            short_levels = []

            for item in data.get('data', []):
                level_data = {
                    'price': float(item.get('price', 0)),
                    'amount': float(item.get('amount', 0)),
                    'usd_value': float(item.get('usdValue', 0)),
                    'percentage': float(item.get('percentage', 0))
                }

                if item.get('side') == 'long':
                    long_levels.append(level_data)
                else:
                    short_levels.append(level_data)

            result = {}
            if long_levels:
                result['long'] = pd.DataFrame(long_levels).sort_values('price', ascending=False)
            if short_levels:
                result['short'] = pd.DataFrame(short_levels).sort_values('price')

            logger.debug(f"Fetched liquidation levels for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch liquidation levels for {symbol}: {e}")
            raise

    async def get_liquidation_history(
        self,
        symbol: str,
        hours: int = 24
    ) -> pd.DataFrame:
        """
        Fetch historical liquidation data.

        Args:
            symbol: Trading symbol (e.g., 'ETH')
            hours: Number of hours of history to fetch

        Returns:
            DataFrame with historical liquidations

        Raises:
            Exception: If data fetch fails
        """
        try:
            endpoint = "/v1/history"
            params = {
                "symbol": symbol,
                "hours": hours
            }

            data = await self._make_request(endpoint, params)

            liquidations = []
            for item in data.get('data', []):
                liquidations.append({
                    'timestamp': pd.to_datetime(item.get('time'), unit='ms'),
                    'side': item.get('side', ''),
                    'price': float(item.get('price', 0)),
                    'amount': float(item.get('amount', 0)),
                    'usd_value': float(item.get('usdValue', 0))
                })

            df = pd.DataFrame(liquidations)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

            logger.debug(f"Fetched {len(df)} historical liquidations for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch liquidation history for {symbol}: {e}")
            raise

    async def analyze_liquidation_density(
        self,
        symbol: str,
        price_range: float = 0.02,
        bins: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze liquidation density around current price.

        Args:
            symbol: Trading symbol (e.g., 'ETH')
            price_range: Price range percentage (0.02 = 2%)
            bins: Number of price bins

        Returns:
            Dictionary with density analysis results

        Raises:
            Exception: If analysis fails
        """
        try:
            # Get current liquidation levels
            levels = await self.get_liquidation_levels(symbol)

            # Get current price (simplified - in production you'd get this from Binance)
            current_price = 0
            if 'long' in levels and not levels['long'].empty:
                current_price = levels['long']['price'].iloc[0]
            elif 'short' in levels and not levels['short'].empty:
                current_price = levels['short']['price'].iloc[0]

            if current_price == 0:
                logger.warning(f"Could not determine current price for {symbol}")
                return {'density_clusters': []}

            # Calculate price range
            lower_bound = current_price * (1 - price_range)
            upper_bound = current_price * (1 + price_range)

            # Analyze density clusters
            density_clusters = []

            for side, df in levels.items():
                if df.empty:
                    continue

                # Filter levels within range
                mask = (df['price'] >= lower_bound) & (df['price'] <= upper_bound)
                filtered_df = df[mask].copy()

                if filtered_df.empty:
                    continue

                # Create price bins
                filtered_df['price_bin'] = pd.cut(
                    filtered_df['price'],
                    bins=bins,
                    labels=False
                )

                # Aggregate by bins
                binned = filtered_df.groupby('price_bin').agg({
                    'amount': 'sum',
                    'usd_value': 'sum',
                    'price': 'mean'
                }).reset_index()

                # Find high density clusters
                threshold = binned['usd_value'].quantile(0.8)
                clusters = binned[binned['usd_value'] >= threshold]

                for _, cluster in clusters.iterrows():
                    density_clusters.append({
                        'side': side,
                        'price_level': cluster['price'],
                        'density_score': cluster['usd_value'],
                        'total_amount': cluster['amount'],
                        'distance_from_current': abs(cluster['price'] - current_price) / current_price
                    })

            # Sort by density score
            density_clusters.sort(key=lambda x: x['density_score'], reverse=True)

            result = {
                'current_price': current_price,
                'price_range': price_range,
                'density_clusters': density_clusters[:10],  # Top 10 clusters
                'timestamp': datetime.now()
            }

            logger.debug(f"Analyzed liquidation density for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze liquidation density for {symbol}: {e}")
            raise

    async def get_liquidation_indicators(
        self,
        symbol: str
    ) -> Dict[str, float]:
        """
        Calculate liquidation-based indicators.

        Args:
            symbol: Trading symbol (e.g., 'ETH')

        Returns:
            Dictionary with liquidation indicators

        Raises:
            Exception: If calculation fails
        """
        try:
            # Get liquidation density analysis
            density = await self.analyze_liquidation_density(symbol)

            if not density['density_clusters']:
                return {
                    'long_liquidation_pressure': 0.0,
                    'short_liquidation_pressure': 0.0,
                    'liquidation_imbalance': 0.0,
                    'nearest_liquidation_distance': 1.0
                }

            # Calculate indicators
            long_pressure = sum(
                c['density_score'] for c in density['density_clusters']
                if c['side'] == 'long'
            )

            short_pressure = sum(
                c['density_score'] for c in density['density_clusters']
                if c['side'] == 'short'
            )

            total_pressure = long_pressure + short_pressure
            imbalance = (long_pressure - short_pressure) / total_pressure if total_pressure > 0 else 0

            # Find nearest liquidation cluster
            nearest_distance = min(
                c['distance_from_current']
                for c in density['density_clusters']
            ) if density['density_clusters'] else 1.0

            indicators = {
                'long_liquidation_pressure': long_pressure,
                'short_liquidation_pressure': short_pressure,
                'liquidation_imbalance': imbalance,
                'nearest_liquidation_distance': nearest_distance,
                'timestamp': datetime.now().timestamp()
            }

            logger.debug(f"Calculated liquidation indicators for {symbol}")
            return indicators

        except Exception as e:
            logger.error(f"Failed to calculate liquidation indicators for {symbol}: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()