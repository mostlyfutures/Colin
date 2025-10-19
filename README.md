# Colin Trading Bot 🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange.svg)](CHANGELOG.md)

A sophisticated AI-powered cryptocurrency trading system that has evolved from a signal-only bot into a comprehensive institutional trading platform.

## 🚀 **Features**

### **Version 2.0 - Institutional Trading Platform**
- 🤖 **AI-Driven Signals**: LSTM, Transformer, and Ensemble models with >65% accuracy
- ⚡ **Smart Execution**: VWAP/TWAP algorithms with sub-50ms latency
- 🛡️ **Real-Time Risk Management**: Circuit breakers, position limits, VaR calculations
- 🌐 **Multi-Exchange Connectivity**: Binance, Bybit, OKX, Deribit integration
- 📊 **Comprehensive Monitoring**: Real-time metrics, alerts, and dashboard
- 🔐 **Regulatory Compliance**: Pre-trade checks, audit trails, 7-year data retention
- 📡 **Multi-Source Market Data**: Free API integration with intelligent failover

### **Version 1.0 - Legacy Signal Scoring**
- 📈 **Technical Analysis**: ICT concepts, kill zones, liquidity analysis
- 🔍 **Market Structure**: Smart money concepts, order flow analysis
- 📊 **Volume Analysis**: Volume-Open Interest correlation
- 🎯 **Signal Scoring**: Institutional-grade signal generation

## 🏗️ **Architecture**

```
colin_bot/
├── v2/                    # Current version (institutional platform)
│   ├── ai_engine/         # AI/ML components
│   ├── execution_engine/  # Order execution & routing
│   ├── risk_system/       # Risk management & compliance
│   ├── data_sources/      # Multi-source market data
│   ├── api_gateway/       # REST & WebSocket APIs
│   └── monitoring/        # System monitoring
├── v1/                    # Legacy version (signal scoring)
│   ├── scorers/           # Technical analysis scorers
│   ├── structure/         # Market structure detection
│   └── adapters/          # Exchange adapters
└── shared/                # Common utilities
```

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- PostgreSQL (for production)
- Redis (for caching)

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd colin-trading-bot

# Install dependencies
pip install -r requirements_v2.txt

# Set up configuration
cp config/development.yaml.example config/development.yaml
# Edit your configuration file

# Run the system
python -m colin_bot.v2.main --mode development
```

### **Multi-Source Market Data Demo**

```bash
# Test the new multi-source data system
python tools/analysis/demo_real_api.py

# Analyze Ethereum with multiple sources
python tools/analysis/analyze_ethereum_multi_source.py --sources 3
```

## 📖 **Documentation**

- [📚 Getting Started Guide](docs/GETTING_STARTED.md)
- [🏗️ Architecture Overview](docs/ARCHITECTURE.md)
- [⚙️ Configuration Guide](docs/v2/CONFIGURATION.md)
- [🔧 API Reference](docs/v2/API_REFERENCE.md)
- [🚀 Deployment Guide](docs/DEPLOYMENT.md)
- [🐛 Troubleshooting](docs/TROUBLESHOOTING.md)

## 🧪 **Testing**

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/v2/data_sources/ -v
python -m pytest tests/v2/risk_system/ -v

# Run with coverage
python -m pytest tests/ --cov=colin_bot --cov-report=html
```

### **Validation Scripts**

```bash
# Validate implementation
python tools/validation/validate_implementation.py

# Validate specific components
python tools/validation/validate_phase1.py  # AI Engine
python tools/validation/validate_phase2.py  # Execution Engine
python tools/validation/validate_phase3.py  # Risk System
python tools/validation/validate_phase4.py  # Integration & Monitoring
```

## 📊 **Performance Metrics**

### **Version 2.0 Targets**
- ✅ **Signal Accuracy**: >65% directional accuracy
- ✅ **Execution Latency**: <50ms end-to-end execution
- ✅ **Risk Validation**: <5ms per risk check
- ✅ **System Uptime**: >99.9% availability
- ✅ **Symbol Capacity**: 100+ simultaneous symbols
- ✅ **API Response**: <100ms market data retrieval

### **Recent Live Performance**
```
Ethereum Analysis (Oct 19, 2025):
- Current Price: $3,988.03
- Fear & Greed Index: 29 (Fear)
- Data Sources: CoinGecko + Alternative.me
- API Latency: ~0.12s
- Signal: BULLISH - Consider buying opportunities
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Database
export DB_HOST=localhost
export DB_NAME=colin_trading_bot_v2
export DB_USER=colin_user
export DB_PASSWORD=your_password

# API Keys (optional)
export COINGECKO_API_KEY=your_key
export BINANCE_API_KEY=your_binance_key

# Features
export ENVIRONMENT=development
export DISABLE_TRADING=false
```

### **Main Config File** (`config/development.yaml`)
```yaml
system:
  environment: "development"
  log_level: "INFO"
  monitoring_enabled: true

trading:
  enabled: true
  max_portfolio_value_usd: 1000000.0
  default_order_size_usd: 10000.0

market_data:
  primary_source: "coingecko"
  fallback_sources: ["kraken", "cryptocompare"]
  cache_ttl_seconds: 300
```

## 🌐 **API Access**

### **REST API**
```bash
# Get trading signals
curl -X POST "http://localhost:8000/api/v2/signals/generate" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["ETH/USDT", "BTC/USDT"]}'

# Get portfolio status
curl -X GET "http://localhost:8000/api/v2/portfolio"

# System health check
curl -X GET "http://localhost:8000/api/v2/health"
```

### **WebSocket Streaming**
```javascript
// Connect to real-time data
const ws = new WebSocket('ws://localhost:8001/ws/signals');
ws.onmessage = (event) => {
    const signal = JSON.parse(event.data);
    console.log('New signal:', signal);
};
```

## 🛡️ **Security & Compliance**

- **API Authentication**: Secure API key management
- **Rate Limiting**: Protection against API abuse
- **Audit Trail**: Complete logging of all trading activities
- **Data Encryption**: Sensitive data encrypted at rest
- **Regulatory Compliance**: Pre-trade checks, position limits
- **Risk Controls**: Circuit breakers, automatic position reduction

## 📈 **Market Data Sources**

### **Free Sources (No API Key Required)**
- **CoinGecko**: Primary free source with excellent reliability
- **Kraken**: Exchange data with real-time ticker information
- **CryptoCompare**: Comprehensive market data with good free tier
- **Alternative.me**: Fear & Greed sentiment index

### **Intelligent Failover**
- Automatic source switching on failures
- Circuit breaker patterns for failing sources
- Data cross-referencing for validation
- Confidence scoring for data quality

## 🤖 **AI/ML Features**

### **Prediction Models**
- **LSTM Networks**: Time series price prediction
- **Transformer Models**: Multi-timeframe analysis
- **Ensemble Methods**: Combining multiple approaches

### **Feature Engineering**
- **Technical Indicators**: 50+ technical analysis features
- **Order Book Analysis**: Liquidity and flow analysis
- **Alternative Data**: Sentiment and on-chain metrics

## 🚀 **Development**

### **Code Structure**
- Modular architecture with clear separation of concerns
- Async/await patterns for non-blocking operations
- Comprehensive error handling and logging
- Type hints throughout the codebase
- 90%+ test coverage

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run validation scripts
6. Submit a pull request

## 📞 **Support**

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎯 **Roadmap**

### **Current (v2.0)**
- ✅ Multi-source market data system
- ✅ AI-powered signal generation
- ✅ Real-time risk management
- ✅ REST & WebSocket APIs
- ✅ Comprehensive monitoring

### **Upcoming (v2.1)**
- 🔄 Advanced portfolio optimization
- 🔄 Machine learning model auto-retraining
- 🔄 Enhanced sentiment analysis
- 🔄 Mobile trading app

### **Future (v3.0)**
- 📋 Decentralized exchange integration
- 📋 Cross-chain arbitrage
- 📋 Social trading features
- 📋 Advanced analytics dashboard

---

**Built with ❤️ for institutional cryptocurrency trading**

*Last Updated: October 19, 2025*