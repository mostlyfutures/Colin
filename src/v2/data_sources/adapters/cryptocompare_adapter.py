"""
CryptoCompare Adapter

Adapter for CryptoCompare API.
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from .base_adapter import BaseAdapter
from ..models import StandardMarketData, DataQuality, DataSource
from ..config import DataSourceConfig


class CryptoCompareAdapter(BaseAdapter):
    """Adapter for CryptoCompare API."""

    # Symbol mapping for common cryptocurrencies
    SYMBOL_MAPPING = {
        "BTC": "BTC",
        "ETH": "ETH",
        "BNB": "BNB",
        "ADA": "ADA",
        "XRP": "XRP",
        "SOL": "SOL",
        "DOT": "DOT",
        "DOGE": "DOGE",
        "AVAX": "AVAX",
        "MATIC": "MATIC",
        "LINK": "LINK",
        "LTC": "LTC",
        "ATOM": "ATOM",
        "XLM": "XLM",
        "ETC": "ETC",
        "FIL": "FIL",
        "TRX": "TRX",
        "XMR": "XMR",
        "AAVE": "AAVE"
    }

    def __init__(self, config: DataSourceConfig):
        """
        Initialize CryptoCompare adapter.

        Args:
            config: Data source configuration
        """
        super().__init__(config, DataSource.CRYPTOCOMPARE)
        logger.info("CryptoCompare adapter initialized")

    async def get_market_data(self, symbol: str) -> StandardMarketData:
        """
        Get market data for a symbol from CryptoCompare.

        Args:
            symbol: Trading symbol (e.g., 'ETH', 'BTC')

        Returns:
            Standardized market data

        Raises:
            Exception: If data fetch fails
        """
        try:
            # Map symbol to CryptoCompare format
            cc_symbol = self._map_symbol(symbol)
            if not cc_symbol:
                raise ValueError(f"Unsupported symbol: {symbol}")

            # Fetch multi-price data
            endpoint = "/pricemultifull"
            params = {
                "fsyms": cc_symbol,
                "tsyms": "USD",
                "e": "CCCAGG"  # Aggregate data from all exchanges
            }

            data = await self._make_request(endpoint, params)

            if not data or "RAW" not in data or cc_symbol not in data["RAW"]:
                raise ValueError(f"No data returned for symbol: {symbol}")

            raw_data = data["RAW"][cc_symbol]["USD"]
            timestamp = datetime.now()

            # Extract market data from CryptoCompare response
            price = float(raw_data.get("PRICE", 0))
            volume_24h = float(raw_data.get("VOLUME24HOUR", 0))
            change_24h = float(raw_data.get("CHANGE24HOUR", 0))
            change_pct_24h = float(raw_data.get("CHANGEPCT24HOUR", 0))
            high_24h = float(raw_data.get("HIGH24HOUR", 0))
            low_24h = float(raw_data.get("LOW24HOUR", 0))
            market_cap = float(raw_data.get("MKTCAP", 0)) if raw_data.get("MKTCAP") else None

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(raw_data)

            # Determine data quality
            data_quality = self._determine_data_quality(raw_data, confidence)

            # Create standardized market data
            market_data = StandardMarketData(
                symbol=symbol,
                price=price,
                volume_24h=volume_24h,
                change_24h=change_24h,
                change_pct_24h=change_pct_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                timestamp=timestamp,
                source=self.source_type,
                confidence=confidence,
                market_cap=market_cap,
                data_quality=data_quality,
                raw_data=raw_data
            )

            logger.debug(f"Fetched market data for {symbol} from CryptoCompare: ${market_data.price}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol} from CryptoCompare: {e}")
            raise

    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Returns:
            List of supported symbol names
        """
        return list(self.SYMBOL_MAPPING.keys())

    async def get_historical_data(self, symbol: str, hours: int = 24) -> List[Dict]:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Trading symbol
            hours: Number of hours of historical data

        Returns:
            List of historical data points
        """
        try:
            cc_symbol = self._map_symbol(symbol)
            if not cc_symbol:
                raise ValueError(f"Unsupported symbol: {symbol}")

            endpoint = "/v2/histohour"
            params = {
                "fsym": cc_symbol,
                "tsym": "USD",
                "limit": hours,
                "e": "CCCAGG"
            }

            data = await self._make_request(endpoint, params)

            if "Data" in data and "Data" in data["Data"]:
                return data["Data"]["Data"]
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return []

    async def get_social_stats(self, symbol: str) -> Dict:
        """
        Get social statistics for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Social statistics data
        """
        try:
            cc_symbol = self._map_symbol(symbol)
            if not cc_symbol:
                raise ValueError(f"Unsupported symbol: {symbol}")

            endpoint = "/social/stats"
            params = {
                "fsym": cc_symbol
            }

            data = await self._make_request(endpoint, params)

            if "Data" in data:
                return data["Data"]
            else:
                return {}

        except Exception as e:
            logger.warning(f"Failed to fetch social stats for {symbol}: {e}")
            return {}

    def _map_symbol(self, symbol: str) -> Optional[str]:
        """
        Map trading symbol to CryptoCompare format.

        Args:
            symbol: Trading symbol

        Returns:
            CryptoCompare symbol or None if not found
        """
        symbol = symbol.upper()
        return self.SYMBOL_MAPPING.get(symbol)

    def _calculate_confidence(self, raw_data: Dict) -> float:
        """
        Calculate confidence score based on data completeness.

        Args:
            raw_data: Raw data from CryptoCompare

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 8.0

        # Basic price data
        if raw_data.get("PRICE") and float(raw_data["PRICE"]) > 0:
            score += 1.0

        # Volume data
        if raw_data.get("VOLUME24HOUR") and float(raw_data["VOLUME24HOUR"]) > 0:
            score += 1.0

        # 24h change
        if raw_data.get("CHANGE24HOUR") is not None:
            score += 1.0

        # High/Low data
        if (raw_data.get("HIGH24HOUR") and raw_data.get("LOW24HOUR") and
            float(raw_data["HIGH24HOUR"]) > 0 and float(raw_data["LOW24HOUR"]) > 0):
            score += 1.0

        # Market cap
        if raw_data.get("MKTCAP") and float(raw_data["MKTCAP"]) > 0:
            score += 1.0

        # Supply data
        if raw_data.get("SUPPLY") and float(raw_data["SUPPLY"]) > 0:
            score += 1.0

        # Last update timestamp
        if raw_data.get("LASTUPDATE"):
            score += 1.0

        # Last market (exchange)
        if raw_data.get("LASTMARKET"):
            score += 1.0

        return min(score / max_score, 1.0)

    def _determine_data_quality(self, raw_data: Dict, confidence: float) -> DataQuality:
        """
        Determine data quality based on completeness and other factors.

        Args:
            raw_data: Raw data from CryptoCompare
            confidence: Confidence score

        Returns:
            Data quality rating
        """
        # Additional quality factors specific to CryptoCompare
        has_volume_from_exchanges = (
            raw_data.get("VOLUME24HOURTO") and
            float(raw_data.get("VOLUME24HOURTO", 0)) > 0
        )

        has_multiple_markets = (
            raw_data.get("MARKET") and
            raw_data.get("MARKET") == "CCCAGG"
        )

        # Adjust confidence based on additional factors
        final_confidence = confidence
        if has_volume_from_exchanges:
            final_confidence = min(final_confidence + 0.05, 1.0)
        if has_multiple_markets:
            final_confidence = min(final_confidence + 0.05, 1.0)

        if final_confidence >= 0.9:
            return DataQuality.EXCELLENT
        elif final_confidence >= 0.7:
            return DataQuality.GOOD
        elif final_confidence >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    async def health_check(self) -> bool:
        """
        Perform health check using CryptoCompare price endpoint.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get BTC price as health check
            endpoint = "/price"
            params = {
                "fsym": "BTC",
                "tsyms": "USD"
            }

            await self._make_request(endpoint, params)

            # Try to fetch full market data for additional check
            await self.get_market_data("BTC")

            return True
        except Exception as e:
            logger.warning(f"CryptoCompare health check failed: {e}")
            return False