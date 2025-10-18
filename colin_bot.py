#!/usr/bin/env python3
"""
Colin Trading Bot CLI Script.

Usage:
    python colin_bot.py ETHUSDT BTCUSDT
    python colin_bot.py --continuous --interval 30 ETHUSDT
    python colin_bot.py --format json --output results.json ETHUSDT
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import main

if __name__ == "__main__":
    asyncio.run(main())