"""
HFT Data Ingestion Layer

Real-time market data ingestion and processing for high-frequency trading.
"""

from .market_data_manager import HFTDataManager
from .connectors.mock_connector import MockDataConnector

__all__ = [
    "HFTDataManager",
    "MockDataConnector"
]