# Colin Trading Bot - Institutional-Grade Signal Scoring Bot

A sophisticated crypto perpetuals trading signal analysis system that incorporates institutional-grade market structure and order flow principles from the ICT (Institutional Candlestick Theory) framework and market microstructure research.

## 🏦 Overview

Colin Trading Bot generates real-time **Long/Short Confidence percentages** (0-100%) for crypto perpetuals by synthesizing:

- **Liquidity Dynamics**: Proximity to liquidation clusters and stop-hunt zones
- **ICT Market Structure**: Fair Value Gaps (FVGs), Order Blocks (OBs), Break of Structure (BOS)
- **Session Timing**: Killzone filters for Asian, London, and NY sessions
- **Order Flow Analysis**: Book imbalance and aggressive trade delta
- **Volume & Open Interest**: Confirmation from Binance Futures data

The output includes human-readable rationale based on smart money behavior patterns, not just generic technical indicators.

## ✨ Key Features

### 🔍 Institutional Signal Analysis
- **Liquidity Proximity Scoring**: Identifies proximity to high-density liquidation zones
- **ICT Structure Detection**: Algorithmic detection of FVGs, Order Blocks, and BOS
- **Killzone Timing**: Optimal entry windows during institutional session overlaps
- **Order Flow Analytics**: Real-time order book imbalance and trade delta analysis
- **Volume/OI Confirmation**: Correlates signals with volume and open interest trends

### 📊 Risk Management
- **Structural Stop-Loss Levels**: Based on ICT structures, not arbitrary percentages
- **Position Sizing**: Confidence-adjusted position sizing with volatility considerations
- **Risk/Reward Analysis**: Calculates optimal take-profit levels with 2:1 ratio targeting
- **Comprehensive Warnings**: Volatility, liquidity, and structural risk alerts

### 🌐 Multi-Data Source Integration
- **Binance Futures**: Real-time OHLCV, Open Interest, Volume, and Order Book data
- **CoinGlass API**: Liquidation heatmap and density cluster analysis
- **Session Analysis**: UTC-based institutional trading session timing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for Binance Futures (optional for demo mode)
- CoinGlass API access (optional for demo mode)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Colin_TradingBot
```

2. **Create virtual environment**
```bash
python -m venv venv_linux
source venv_linux/bin/activate  # On Linux/macOS
# or
venv_linux\Scripts\activate  # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys** (optional)
```bash
# Create .env file
echo "BINANCE_API_KEY=your_binance_api_key" >> .env
echo "BINANCE_API_SECRET=your_binance_api_secret" >> .env
```

### Basic Usage

#### Single Symbol Analysis
```bash
python colin_bot.py ETHUSDT
```

#### Multiple Symbols
```bash
python colin_bot.py ETHUSDT BTCUSDT SOLUSDT
```

#### Continuous Analysis
```bash
python colin_bot.py --continuous --interval 30 ETHUSDT BTCUSDT
```

#### Save Results to File
```bash
python colin_bot.py --format json --output results.json ETHUSDT
```

#### Custom Time Horizon
```bash
python colin_bot.py --time-horizon 1h ETHUSDT
```

## 📋 Output Format

### Human-Readable Output
```
🏦 INSTITUTIONAL TRADING SIGNAL: ETHUSDT
======================================================================

🟢 DIRECTION: LONG
🔥 CONFIDENCE: HIGH
📊 Long Confidence: 78.5% | Short Confidence: 21.5%

💰 ENTRY: $2,045.32
🛑 STOP LOSS: $2,032.15
🎯 TAKE PROFIT: $2,071.49
📏 POSITION SIZE: 1.5% of portfolio

📋 INSTITUTIONAL RATIONALE:
   1. Strong liquidity confluence with untested liquidation clusters
   2. Price approaching fresh bullish Order Block during London session
   3. Order flow showing aggressive buying pressure (NOBI: 0.73)

🏦 FACTOR BREAKDOWN:
   Liquidity Analysis: 0.825
   ICT Structure: 0.690
   Killzone Timing: 0.800
   Order Flow: 0.730
   Volume/OI: 0.545
```

### JSON Output
```json
{
  "symbol": "ETHUSDT",
  "timestamp": "2024-10-18T15:30:00",
  "direction": "long",
  "confidence_level": "high",
  "long_confidence": 78.5,
  "short_confidence": 21.5,
  "entry_price": 2045.32,
  "stop_loss_price": 2032.15,
  "take_profit_price": 2071.49,
  "position_size_percent": 1.5,
  "rationale": [
    "Strong liquidity confluence with untested liquidation clusters",
    "Price approaching fresh bullish Order Block during London session",
    "Order flow showing aggressive buying pressure (NOBI: 0.73)"
  ],
  "institutional_factors": {
    "liquidity": 0.825,
    "ict": 0.690,
    "killzone": 0.800,
    "order_flow": 0.730,
    "volume_oi": 0.545
  }
}
```

## ⚙️ Configuration

### Main Config File (`config.yaml`)

```yaml
# Trading Symbols
symbols:
  - "ETHUSDT"
  - "BTCUSDT"

# Session Configuration (UTC)
sessions:
  asian:
    start: "00:00"
    end: "09:00"
    weight: 1.0
  london:
    start: "07:00"
    end: "16:00"
    weight: 1.2
  new_york:
    start: "12:00"
    end: "22:00"
    weight: 1.2
  london_ny_overlap:
    start: "12:00"
    end: "16:00"
    weight: 1.5  # Highest conviction window

# Scoring Weights
scoring:
  weights:
    liquidity_proximity: 0.25
    ict_confluence: 0.25
    killzone_alignment: 0.15
    order_flow_delta: 0.20
    volume_oi_confirmation: 0.15

# Risk Management
risk:
  max_position_size: 0.02  # 2% max position size
  stop_loss_buffer: 0.002  # 0.2% buffer beyond structure
  volatility_threshold: 0.03  # 3% volatility threshold
```

### Environment Variables

```bash
# API Keys (optional - demo mode works without them)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional: Custom config path
COLIN_BOT_CONFIG=/path/to/custom/config.yaml
```

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

## 📊 Signal Components

### Liquidity Analysis
- **Liquidation Heatmap**: Identifies high-density liquidation zones
- **Stop-Hunt Detection**: Flags potential stop-loss hunting scenarios
- **Liquidity Grab Analysis**: Detects untested liquidity targets

### ICT Structure Detection
- **Fair Value Gaps (FVGs)**: 3-candle imbalance patterns
- **Order Blocks (OBs)**: Last opposing candle before strong displacement
- **Break of Structure (BOS)**: Confirmed trend changes with retest levels

### Session Timing
- **Asian Session**: Lower liquidity, focus on JPY/KRW flows
- **London Session**: High liquidity, European institutional flow
- **NY Session**: US institutional flow, higher volatility
- **London/NY Overlap**: Peak liquidity (12:00-16:00 UTC)

### Order Flow Analysis
- **Normalized Order Book Imbalance (NOBI)**: Bid/ask liquidity imbalance
- **Trade Delta**: Aggressive buying vs selling pressure
- **Market Depth**: Liquidity distribution across price levels

## 🎯 confidence Levels

- **HIGH (80-100%)**: Multiple institutional factors aligned, optimal entry conditions
- **MEDIUM (60-79%)**: Good signal with moderate confirmation
- **LOW (40-59%)**: Weak signal, limited institutional alignment
- **NEUTRAL**: No clear directional bias, avoid trading

## ⚠️ Risk Warnings

The bot provides comprehensive risk warnings:
- **Volatility Warnings**: High market volatility alerts
- **Liquidity Warnings**: Low liquidity condition alerts
- **Structural Risk**: Limited support/resistance warnings
- **Position Size Alerts**: Over-leveraging warnings

## 📈 Performance Monitoring

### Signal Accuracy
- Target: >90% FVG/OB detection accuracy vs manual labeling
- Target: Pearson r > 0.3 correlation with forward returns

### Risk Metrics
- Maximum drawdown tracking
- Risk/reward ratio monitoring
- Position size effectiveness

## 🔄 Development Mode

Enable development mode for testing:
```yaml
development:
  test_mode: true
  mock_api_responses: true
  save_intermediate_data: false
```

## 🛠️ Architecture

```
src/
├── core/           # Configuration management
├── adapters/       # Data adapters (Binance, CoinGlass)
├── structure/      # ICT structure detection
├── orderflow/      # Order flow analysis
├── scorers/        # Institutional factor scorers
├── engine/         # Main scoring engine
├── output/         # Risk-aware formatting
├── utils/          # Session utilities
└── main.py         # Application entry point
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚡ Disclaimer

**IMPORTANT**: This software is for educational and informational purposes only. It does not constitute financial advice. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.

The developers of this software are not responsible for any financial losses incurred while using this trading bot. Use at your own risk.

## 📞 Support

For questions, bug reports, or feature requests:
- Create an issue on GitHub
- Check the documentation in the `/docs` folder
- Review the test files for usage examples

---

**Built with ❤️ for institutional-grade crypto analysis**