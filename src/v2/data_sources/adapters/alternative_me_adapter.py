"""
Alternative.me Adapter

Adapter for Alternative.me API providing Fear & Greed Index and other sentiment data.
"""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from .base_adapter import BaseAdapter
from ..models import SentimentData, StandardMarketData, DataQuality, DataSource
from ..config import DataSourceConfig


class AlternativeMeAdapter(BaseAdapter):
    """Adapter for Alternative.me API."""

    def __init__(self, config: DataSourceConfig):
        """
        Initialize Alternative.me adapter.

        Args:
            config: Data source configuration
        """
        super().__init__(config, DataSource.ALTERNATIVE_ME)
        logger.info("Alternative.me adapter initialized")

    async def get_fear_and_greed_index(self) -> SentimentData:
        """
        Get current Fear & Greed Index.

        Returns:
            Fear & Greed sentiment data

        Raises:
            Exception: If data fetch fails
        """
        try:
            endpoint = "/fng/"
            data = await self._make_request(endpoint)

            if not data or "data" not in data or not data["data"]:
                raise ValueError("No Fear & Greed data returned")

            # Get the most recent data point
            fng_data = data["data"][0]

            sentiment = SentimentData(
                value=int(fng_data["value"]),
                value_classification=fng_data["value_classification"],
                timestamp=datetime.fromtimestamp(int(fng_data["timestamp"])),
                time_until_update=fng_data.get("time_until_update")
            )

            logger.debug(f"Fetched Fear & Greed Index: {sentiment.value} ({sentiment.value_classification})")
            return sentiment

        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            raise

    async def get_historical_fear_and_greed(self, limit: int = 365) -> List[Dict]:
        """
        Get historical Fear & Greed Index data.

        Args:
            limit: Number of data points to fetch (max ~365 days)

        Returns:
            List of historical sentiment data points
        """
        try:
            endpoint = "/fng/"
            params = {"limit": limit}

            data = await self._make_request(endpoint, params)

            if "data" in data:
                return data["data"]
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to fetch historical Fear & Greed data: {e}")
            return []

    async def get_market_data(self, symbol: str) -> StandardMarketData:
        """
        Alternative.me doesn't provide market data for specific symbols.
        This method returns a dummy implementation for interface compatibility.

        Args:
            symbol: Trading symbol (ignored)

        Returns:
            Empty market data with low confidence

        Raises:
            Exception: Always raises as this source doesn't provide market data
        """
        raise NotImplementedError("Alternative.me only provides sentiment data, not market data for specific symbols")

    async def get_supported_symbols(self) -> List[str]:
        """
        Alternative.me doesn't support specific symbol data.

        Returns:
            Empty list
        """
        return []

    async def get_crypto_fear_and_greed(self) -> SentimentData:
        """
        Get Fear & Greed Index specifically for cryptocurrency market.

        Returns:
            Crypto-specific Fear & Greed sentiment data
        """
        try:
            endpoint = "/fng/"
            params = {"limit": 1}

            data = await self._make_request(endpoint, params)

            if not data or "data" not in data or not data["data"]:
                raise ValueError("No crypto Fear & Greed data returned")

            fng_data = data["data"][0]

            sentiment = SentimentData(
                value=int(fng_data["value"]),
                value_classification=fng_data["value_classification"],
                timestamp=datetime.fromtimestamp(int(fng_data["timestamp"])),
                time_until_update=fng_data.get("time_until_update")
            )

            logger.debug(f"Fetched Crypto Fear & Greed Index: {sentiment.value} ({sentiment.value_classification})")
            return sentiment

        except Exception as e:
            logger.error(f"Failed to fetch crypto Fear & Greed Index: {e}")
            raise

    def get_sentiment_interpretation(self, value: int) -> Dict[str, str]:
        """
        Get interpretation of Fear & Greed value.

        Args:
            value: Fear & Greed index value (0-100)

        Returns:
            Dictionary with interpretation details
        """
        if value <= 20:
            return {
                "sentiment": "Extreme Fear",
                "interpretation": "Investors are very worried. This could be a buying opportunity as markets often rebound from extreme fear levels.",
                "trading_implication": "Consider contrarian positions, potential bottom signal"
            }
        elif value <= 40:
            return {
                "sentiment": "Fear",
                "interpretation": "Market shows fear but not extreme. Some investors are cautious.",
                "trading_implication": "Watch for reversal patterns, risk-off sentiment"
            }
        elif value <= 60:
            return {
                "sentiment": "Neutral",
                "interpretation": "Market sentiment is balanced between fear and greed.",
                "trading_implication": "Market in equilibrium, follow technical indicators"
            }
        elif value <= 80:
            return {
                "sentiment": "Greed",
                "interpretation": "Investors are becoming optimistic and taking on more risk.",
                "trading_implication": "Markets may be overbought, consider profit-taking"
            }
        else:
            return {
                "sentiment": "Extreme Greed",
                "interpretation": "Investors are very optimistic and potentially taking excessive risks. Markets may be due for a correction.",
                "trading_implication": "Strong warning signal, consider reducing exposure"
            }

    async def health_check(self) -> bool:
        """
        Perform health check using Fear & Greed endpoint.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to fetch Fear & Greed data
            await self.get_fear_and_greed_index()
            return True
        except Exception as e:
            logger.warning(f"Alternative.me health check failed: {e}")
            return False

    def calculate_sentiment_score(self, sentiment_data: SentimentData) -> float:
        """
        Calculate normalized sentiment score from Fear & Greed data.

        Args:
            sentiment_data: Fear & Greed sentiment data

        Returns:
            Normalized sentiment score (0.0 to 1.0)
        """
        # Convert 0-100 scale to 0.0-1.0 scale
        return sentiment_data.value / 100.0

    def get_trading_signals_from_sentiment(self, sentiment_data: SentimentData) -> Dict[str, str]:
        """
        Generate trading signals based on sentiment data.

        Args:
            sentiment_data: Fear & Greed sentiment data

        Returns:
            Dictionary with trading signals
        """
        value = sentiment_data.value
        interpretation = self.get_sentiment_interpretation(value)

        signals = {}

        if value <= 25:
            # Extreme fear - potential buying opportunity
            signals["overall_signal"] = "BULLISH"
            signals["strength"] = "MODERATE"
            signals["reasoning"] = f"Extreme Fear ({value}) often coincides with market bottoms"
        elif value <= 40:
            # Fear - cautious bullish
            signals["overall_signal"] = "NEUTRAL_TO_BULLISH"
            signals["strength"] = "WEAK"
            signals["reasoning"] = f"Fear ({value}) suggests oversold conditions"
        elif value <= 60:
            # Neutral - follow technicals
            signals["overall_signal"] = "NEUTRAL"
            signals["strength"] = "WEAK"
            signals["reasoning"] = f"Neutral sentiment ({value}) - follow price action"
        elif value <= 75:
            # Greed - caution
            signals["overall_signal"] = "NEUTRAL_TO_BEARISH"
            signals["strength"] = "WEAK"
            signals["reasoning"] = f"Greed ({value}) suggests potential overbought conditions"
        else:
            # Extreme greed - potential bearish signal
            signals["overall_signal"] = "BEARISH"
            signals["strength"] = "MODERATE"
            signals["reasoning"] = f"Extreme Greed ({value}) often precedes market corrections"

        signals.update({
            "sentiment_value": str(value),
            "sentiment_classification": sentiment_data.value_classification,
            "interpretation": interpretation["sentiment"],
            "trading_implication": interpretation["trading_implication"]
        })

        return signals