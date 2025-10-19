"""
Kraken Adapter

Adapter for Kraken public cryptocurrency API.
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from .base_adapter import BaseAdapter
from ..models import StandardMarketData, DataQuality, DataSource
from ..config import DataSourceConfig


class KrakenAdapter(BaseAdapter):
    """Adapter for Kraken API."""

    # Symbol mapping for common cryptocurrencies (Kraken uses different naming)
    SYMBOL_MAPPING = {
        "BTC": "XBTUSD",  # Kraken uses XBT for Bitcoin
        "ETH": "ETHUSD",
        "BNB": "BNBUSD",
        "ADA": "ADAUSD",
        "XRP": "XRPUSD",
        "SOL": "SOLUSD",
        "DOT": "DOTUSD",
        "DOGE": "XDGUSD",  # Kraken uses XDG for Dogecoin
        "AVAX": "AVAXUSD",
        "MATIC": "MATICUSD",
        "LINK": "LINKUSD",
        "LTC": "LTCUSD",
        "ATOM": "ATOMUSD",
        "XLM": "XLMD",  # Note: Different format for XLM
        "ETC": "ETCUSD",
        "FIL": "FILUSD",
        "TRX": "TRXUSD",
        "XMR": "XMRUSD",
        "AAVE": "AAVEUSD"
    }

    def __init__(self, config: DataSourceConfig):
        """
        Initialize Kraken adapter.

        Args:
            config: Data source configuration
        """
        super().__init__(config, DataSource.KRAKEN)
        logger.info("Kraken adapter initialized")

    async def get_market_data(self, symbol: str) -> StandardMarketData:
        """
        Get market data for a symbol from Kraken.

        Args:
            symbol: Trading symbol (e.g., 'ETH', 'BTC')

        Returns:
            Standardized market data

        Raises:
            Exception: If data fetch fails
        """
        try:
            # Map symbol to Kraken pair
            kraken_pair = self._map_symbol_to_pair(symbol)
            if not kraken_pair:
                raise ValueError(f"Unsupported symbol: {symbol}")

            # Fetch ticker data
            endpoint = "/Ticker"
            params = {"pair": kraken_pair}

            data = await self._make_request(endpoint, params)

            if not data or "result" not in data or not data["result"]:
                raise ValueError(f"No data returned for symbol: {symbol}")

            # Extract ticker data (Kraken returns the pair name as the key)
            result_data = data["result"]
            ticker_key = list(result_data.keys())[0]
            ticker_data = result_data[ticker_key]

            timestamp = datetime.now()

            # Parse Kraken ticker data
            # Kraken format: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v', 'x', 'y', 'z']
            # Where:
            # a = ask price, b = ask size, c = bid price, d = bid size
            # e = last trade price, f = last trade size
            # g = 24h low, h = 24h high
            # i = 24h volume weighted average price
            # j = number of trades in last 24h
            # k = 24h volume, l = 24h high, m = 24h low
            # n = opening price, o = todays high, p = todays low
            # q = vwap, r = vwap, v = volume, x = volume
            # y = y, z = z

            price = float(ticker_data['c'][0]) if ticker_data.get('c') else 0
            volume_24h = float(ticker_data['v'][1]) if ticker_data.get('v') and len(ticker_data['v']) > 1 else 0
            high_24h = float(ticker_data['h'][1]) if ticker_data.get('h') and len(ticker_data['h']) > 1 else 0
            low_24h = float(ticker_data['l'][1]) if ticker_data.get('l') and len(ticker_data['l']) > 1 else 0
            vwap_24h = float(ticker_data['p'][1]) if ticker_data.get('p') and len(ticker_data['p']) > 1 else 0

            # Calculate 24h change (need to compute from available data)
            change_24h = 0
            change_pct_24h = 0
            if price > 0 and vwap_24h > 0:
                change_24h = price - vwap_24h
                change_pct_24h = (change_24h / vwap_24h) * 100 if vwap_24h > 0 else 0

            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(ticker_data)

            # Determine data quality
            data_quality = self._determine_data_quality(ticker_data, confidence)

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
                data_quality=data_quality,
                raw_data=ticker_data
            )

            logger.debug(f"Fetched market data for {symbol} from Kraken: ${market_data.price}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol} from Kraken: {e}")
            raise

    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Returns:
            List of supported symbol names
        """
        return list(self.SYMBOL_MAPPING.keys())

    async def get_asset_pairs(self) -> Dict[str, Dict]:
        """
        Get all available asset pairs from Kraken.

        Returns:
            Dictionary of asset pairs information
        """
        try:
            endpoint = "/AssetPairs"
            data = await self._make_request(endpoint)

            if "result" in data:
                return data["result"]
            else:
                return {}

        except Exception as e:
            logger.error(f"Failed to fetch asset pairs from Kraken: {e}")
            return {}

    def _map_symbol_to_pair(self, symbol: str) -> Optional[str]:
        """
        Map trading symbol to Kraken pair name.

        Args:
            symbol: Trading symbol

        Returns:
            Kraken pair name or None if not found
        """
        symbol = symbol.upper()
        return self.SYMBOL_MAPPING.get(symbol)

    def _calculate_confidence(self, ticker_data: Dict) -> float:
        """
        Calculate confidence score based on data completeness.

        Args:
            ticker_data: Raw ticker data from Kraken

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 6.0

        # Last trade price
        if ticker_data.get('c') and ticker_data['c'] and float(ticker_data['c'][0]) > 0:
            score += 1.0

        # Volume data
        if ticker_data.get('v') and len(ticker_data['v']) > 1 and float(ticker_data['v'][1]) > 0:
            score += 1.0

        # High/Low data
        if (ticker_data.get('h') and len(ticker_data['h']) > 1 and
            ticker_data.get('l') and len(ticker_data['l']) > 1 and
            float(ticker_data['h'][1]) > 0 and float(ticker_data['l'][1]) > 0):
            score += 1.0

        # Bid/Ask data (indicates liquidity)
        if (ticker_data.get('a') and ticker_data['a'] and
            ticker_data.get('b') and ticker_data['b'] and
            float(ticker_data['a'][0]) > 0 and float(ticker_data['b'][0]) > 0):
            score += 1.0

        # VWAP data
        if ticker_data.get('p') and len(ticker_data['p']) > 1 and float(ticker_data['p'][1]) > 0:
            score += 1.0

        # Trade count
        if ticker_data.get('j') and int(ticker_data['j']) > 0:
            score += 1.0

        return min(score / max_score, 1.0)

    def _determine_data_quality(self, ticker_data: Dict, confidence: float) -> DataQuality:
        """
        Determine data quality based on completeness and other factors.

        Args:
            ticker_data: Raw ticker data from Kraken
            confidence: Confidence score

        Returns:
            Data quality rating
        """
        if confidence >= 0.85:
            return DataQuality.EXCELLENT
        elif confidence >= 0.65:
            return DataQuality.GOOD
        elif confidence >= 0.45:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

    async def health_check(self) -> bool:
        """
        Perform health check using Kraken server time endpoint.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get server time
            await self._make_request("/Time")

            # Try to fetch Bitcoin ticker as additional check
            await self.get_market_data("BTC")

            return True
        except Exception as e:
            logger.warning(f"Kraken health check failed: {e}")
            return False