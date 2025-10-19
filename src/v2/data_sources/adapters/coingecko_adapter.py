"""
CoinGecko Adapter

Adapter for CoinGecko free cryptocurrency API.
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from .base_adapter import BaseAdapter
from ..models import StandardMarketData, DataQuality, DataSource
from ..config import DataSourceConfig


class CoinGeckoAdapter(BaseAdapter):
    """Adapter for CoinGecko API."""

    # Symbol mapping for common cryptocurrencies
    SYMBOL_MAPPING = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "XRP": "ripple",
        "SOL": "solana",
        "DOT": "polkadot",
        "DOGE": "dogecoin",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        "UNI": "uniswap",
        "LTC": "litecoin",
        "ATOM": "cosmos",
        "XLM": "stellar",
        "ETC": "ethereum-classic",
        "FIL": "filecoin",
        "TRX": "tron",
        "XMR": "monero",
        "AAVE": "aave"
    }

    def __init__(self, config: DataSourceConfig):
        """
        Initialize CoinGecko adapter.

        Args:
            config: Data source configuration
        """
        super().__init__(config, DataSource.COINGECKO)
        logger.info("CoinGecko adapter initialized")

    async def get_market_data(self, symbol: str) -> StandardMarketData:
        """
        Get market data for a symbol from CoinGecko.

        Args:
            symbol: Trading symbol (e.g., 'ETH', 'BTC')

        Returns:
            Standardized market data

        Raises:
            Exception: If data fetch fails
        """
        try:
            # Map symbol to CoinGecko ID
            coin_id = self._map_symbol_to_id(symbol)
            if not coin_id:
                raise ValueError(f"Unsupported symbol: {symbol}")

            # Fetch market data
            endpoint = "/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_last_updated_at": "true",
                "include_market_cap": "true",
                "include_24hr_high": "true",
                "include_24hr_low": "true"
            }

            data = await self._make_request(endpoint, params)

            if not data or coin_id not in data:
                raise ValueError(f"No data returned for symbol: {symbol}")

            coin_data = data[coin_id]
            timestamp = datetime.now()

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(coin_data)

            # Determine data quality
            data_quality = self._determine_data_quality(coin_data, confidence)

            # Create standardized market data
            market_data = StandardMarketData(
                symbol=symbol,
                price=float(coin_data.get("usd", 0)),
                volume_24h=float(coin_data.get("usd_24h_vol", 0)),
                change_24h=float(coin_data.get("usd_24h_change", 0)),
                change_pct_24h=float(coin_data.get("usd_24h_change", 0)),
                high_24h=float(coin_data.get("usd_24h_high", 0)) if coin_data.get("usd_24h_high") else None,
                low_24h=float(coin_data.get("usd_24h_low", 0)) if coin_data.get("usd_24h_low") else None,
                timestamp=timestamp,
                source=self.source_type,
                confidence=confidence,
                market_cap=float(coin_data.get("usd_market_cap", 0)) if coin_data.get("usd_market_cap") else None,
                data_quality=data_quality,
                raw_data=coin_data
            )

            logger.debug(f"Fetched market data for {symbol} from CoinGecko: ${market_data.price}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol} from CoinGecko: {e}")
            raise

    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Returns:
            List of supported symbol names
        """
        return list(self.SYMBOL_MAPPING.keys())

    async def get_detailed_market_data(self, symbol: str) -> StandardMarketData:
        """
        Get detailed market data including additional metrics.

        Args:
            symbol: Trading symbol

        Returns:
            Detailed market data
        """
        try:
            # Get basic market data first
            market_data = await self.get_market_data(symbol)

            # Fetch additional data
            coin_id = self._map_symbol_to_id(symbol)
            if not coin_id:
                return market_data

            endpoint = f"/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }

            data = await self._make_request(endpoint, params)

            if "market_data" in data:
                market_detail = data["market_data"]

                # Update with additional data
                if "ath" in market_detail and "usd" in market_detail["ath"]:
                    market_data.ath = float(market_detail["ath"]["usd"])

                if "atl" in market_detail and "usd" in market_detail["atl"]:
                    market_data.atl = float(market_detail["atl"]["usd"])

                if "circulating_supply" in market_detail:
                    market_data.circulating_supply = float(market_detail["circulating_supply"])

                if "total_supply" in market_detail:
                    market_data.total_supply = float(market_detail["total_supply"])

                # Update raw data with detailed info
                market_data.raw_data = {
                    **market_data.raw_data,
                    "detailed_data": market_detail
                }

            return market_data

        except Exception as e:
            logger.warning(f"Failed to fetch detailed market data for {symbol}: {e}")
            # Return basic data if detailed fetch fails
            return await self.get_market_data(symbol)

    def _map_symbol_to_id(self, symbol: str) -> Optional[str]:
        """
        Map trading symbol to CoinGecko coin ID.

        Args:
            symbol: Trading symbol

        Returns:
            CoinGecko coin ID or None if not found
        """
        symbol = symbol.upper()
        return self.SYMBOL_MAPPING.get(symbol)

    def _calculate_confidence(self, coin_data: Dict) -> float:
        """
        Calculate confidence score based on data completeness.

        Args:
            coin_data: Raw data from CoinGecko

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 7.0

        # Basic price data
        if coin_data.get("usd"):
            score += 1.0

        # Volume data
        if coin_data.get("usd_24h_vol"):
            score += 1.0

        # 24h change
        if coin_data.get("usd_24h_change") is not None:
            score += 1.0

        # Market cap
        if coin_data.get("usd_market_cap"):
            score += 1.0

        # 24h high/low
        if coin_data.get("usd_24h_high") and coin_data.get("usd_24h_low"):
            score += 1.0

        # Last updated timestamp
        if coin_data.get("last_updated_at"):
            score += 1.0

        # Recent update (within last 10 minutes)
        if coin_data.get("last_updated_at"):
            import time
            now = int(time.time())
            last_update = coin_data.get("last_updated_at", 0)
            if now - last_update <= 600:  # 10 minutes
                score += 1.0

        return min(score / max_score, 1.0)

    def _determine_data_quality(self, coin_data: Dict, confidence: float) -> DataQuality:
        """
        Determine data quality based on completeness and recency.

        Args:
            coin_data: Raw data from CoinGecko
            confidence: Confidence score

        Returns:
            Data quality rating
        """
        if confidence >= 0.9:
            return DataQuality.EXCELLENT
        elif confidence >= 0.7:
            return DataQuality.GOOD
        elif confidence >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    async def health_check(self) -> bool:
        """
        Perform health check using CoinGecko ping endpoint.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to ping the API
            await self._make_request("/ping")

            # Try to fetch Bitcoin data as additional check
            await self.get_market_data("BTC")

            return True
        except Exception as e:
            logger.warning(f"CoinGecko health check failed: {e}")
            return False