# 🚀 Getting Started with Colin Trading Bot

Welcome to the Colin Trading Bot! This guide will help you get up and running with both the v1 legacy signal scorer and the v2 institutional trading platform.

## 📋 **Prerequisites**

### **System Requirements**
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 10GB free space
- **Network**: Stable internet connection

### **Software Dependencies**
- **PostgreSQL**: For production database (optional for development)
- **Redis**: For caching (optional for development)
- **Git**: For version control

## 🔧 **Installation**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd colin-trading-bot
```

### **2. Set Up Python Environment**

```bash
# Create virtual environment
python -m venv venv_colin_bot

# Activate virtual environment
# On macOS/Linux:
source venv_colin_bot/bin/activate
# On Windows:
venv_colin_bot\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### **3. Install Dependencies**

```bash
# For basic functionality
pip install -r requirements.txt

# For v2 features (recommended)
pip install -r requirements_v2.txt
```

### **4. Set Up Configuration**

```bash
# Copy example configuration
cp config/development.yaml.example config/development.yaml

# Create environment file
touch .env
```

Edit your configuration file:
```yaml
# config/development.yaml
system:
  environment: "development"
  log_level: "INFO"

trading:
  enabled: false  # Start with trading disabled

market_data:
  primary_source: "coingecko"
  cache_ttl_seconds: 300
```

## 🎯 **Quick Start Guide**

### **Option 1: Test Multi-Source Market Data (Easiest)**

```bash
# Test the new market data system
python tools/analysis/demo_real_api.py

# This will show live Ethereum and Bitcoin prices with sentiment analysis
```

**Expected Output:**
```
🚀 REAL Multi-Source Market Data Demo
💰 ETHEREUM (ETH) MARKET ANALYSIS
   Current Price: $3,988.03
   Fear & Greed Index: 29 (Fear)
   Overall Recommendation: BULLISH
```

### **Option 2: Run V1 Legacy Signal Scoring**

```bash
# Run the original signal analysis
python tools/analysis/colin_bot.py

# Analyze specific assets
python tools/analysis/colin_bot.py --symbol ETH/USDT --timeframe 4h
```

### **Option 3: Start V2 Institutional Platform**

```bash
# Start in development mode (simulation)
python -m colin_bot.v2.main --mode development

# Start with API server
python -m colin_bot.v2.api_gateway.rest_api --host 0.0.0.0 --port 8000
```

## 🔍 **Testing Your Installation**

### **1. Run Validation Scripts**

```bash
# Test basic functionality
python tools/validation/validate_implementation.py

# Test market data specifically
python tools/validation/validate_multi_source_data.py

# Test individual components
python tools/validation/validate_phase1.py  # AI Engine
python tools/validation/validate_phase3.py  # Risk System
```

### **2. Run Unit Tests**

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/v2/data_sources/ -v
python -m pytest tests/v2/risk_system/ -v
```

### **3. Test API Endpoints**

```bash
# Start API server (in another terminal)
python -m colin_bot.v2.api_gateway.rest_api

# Test health endpoint
curl -X GET "http://localhost:8000/api/v2/health"

# Test market data
curl -X GET "http://localhost:8000/api/v2/market-data/ETH"
```

## 📊 **Configuration Guide**

### **Environment Variables (.env)**
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=colin_trading_bot_v2
DB_USER=colin_user
DB_PASSWORD=your_password

# API Keys (Optional - Free sources work without keys)
COINGECKO_API_KEY=your_key_if_you_have_one
CRYPTOCOMPARE_API_KEY=your_key_if_you_have_one

# System Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DISABLE_TRADING=true  # Safety first!
```

### **Configuration Files**

#### **Development Config** (`config/development.yaml`)
```yaml
system:
  environment: "development"
  log_level: "DEBUG"
  monitoring_enabled: true

trading:
  enabled: false  # Start disabled for safety
  max_portfolio_value_usd: 100000.0
  default_order_size_usd: 1000.0

market_data:
  primary_source: "coingecko"
  fallback_sources: ["kraken", "cryptocompare"]
  cache_ttl_seconds: 300
  max_concurrent_requests: 5

api:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  require_auth: false  # Disable for development
```

#### **Production Config** (`config/production.yaml`)
```yaml
system:
  environment: "production"
  log_level: "WARNING"
  monitoring_enabled: true

trading:
  enabled: true
  max_portfolio_value_usd: 10000000.0
  default_order_size_usd: 100000.0

security:
  require_auth: true
  jwt_secret_key: "your-secret-key"
  enable_https: true
```

## 🛠️ **Development Workflow**

### **1. Make Changes**
- Edit source files in `colin_bot/`
- Add tests in `tests/`
- Update documentation

### **2. Test Changes**
```bash
# Run validation
python tools/validation/validate_implementation.py

# Run tests
python -m pytest tests/ -v

# Check specific functionality
python tools/analysis/demo_real_api.py
```

### **3. Debug Issues**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m colin_bot.v2.main --mode development --verbose
```

## 📈 **Common Use Cases**

### **1. Market Analysis**
```bash
# Quick market check
python tools/analysis/demo_real_api.py

# Detailed Ethereum analysis
python tools/analysis/analyze_ethereum_multi_source.py --sources 3 --verbose
```

### **2. Signal Generation**
```bash
# Generate trading signals
python -m colin_bot.v2.main --mode development --generate-signals

# Specific symbols
python -m colin_bot.v2.main --symbols ETH,BTC,ADA
```

### **3. Risk Management**
```bash
# Test risk system
python tools/validation/validate_phase3.py

# Monitor positions
python -m colin_bot.v2.risk_system.real_time.position_monitor
```

### **4. API Development**
```bash
# Start REST API
python -m colin_bot.v2.api_gateway.rest_api

# Start WebSocket API
python -m colin_bot.v2.api_gateway.websocket_api

# Test API
curl -X POST "http://localhost:8000/api/v2/signals/generate" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["ETH/USDT"]}'
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Ensure you're in the project directory
cd /path/to/colin-trading-bot

# Activate virtual environment
source venv_colin_bot/bin/activate

# Install dependencies
pip install -r requirements_v2.txt
```

#### **Database Connection Issues**
```bash
# Check PostgreSQL is running
pg_ctl status

# Test connection
python -c "import psycopg2; print('PostgreSQL available')"

# Use SQLite for development (set in config)
```

#### **API Rate Limits**
```bash
# Free APIs have rate limits
# Increase cache TTL in config
# Add API keys if you have them
```

#### **Permission Errors**
```bash
# Check file permissions
ls -la config/

# Fix permissions if needed
chmod 600 config/development.yaml
chmod 600 .env
```

### **Getting Help**

1. **Check logs**: Look at console output for error messages
2. **Run validation**: `python tools/validation/validate_implementation.py`
3. **Check configuration**: Ensure all required fields are set
4. **Review documentation**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
5. **Create issue**: [GitHub Issues](https://github.com/your-repo/issues)

## 📚 **Next Steps**

1. **Read Architecture Overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Explore API Documentation**: [v2/API_REFERENCE.md](v2/API_REFERENCE.md)
3. **Review Configuration Options**: [v2/CONFIGURATION.md](v2/CONFIGURATION.md)
4. **Check Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)

## 🎉 **Success!**

You're now ready to use the Colin Trading Bot! Here are some suggestions for what to do next:

- 🧪 **Run the demo**: `python tools/analysis/demo_real_api.py`
- 📊 **Explore market data**: Check different cryptocurrencies
- 🔧 **Configure settings**: Adjust the configuration to your needs
- 📖 **Read documentation**: Learn about advanced features
- 🚀 **Start developing**: Build your own trading strategies

Happy Trading! 🚀