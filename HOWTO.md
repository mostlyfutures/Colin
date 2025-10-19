
  Here are the different ways you can run the multi-source analysis:

  1. Basic Live Analysis (using my demo):

  cd "/Users/gdove/Desktop/DEEPs_Colin_TradingBot copy"
  python demo_real_api.py

  2. Full Multi-Source Analysis (if you have dependencies):

  # First, install required dependencies if needed:
  pip install aiohttp pandas numpy

  # Run the full analysis:
  python analyze_ethereum_multi_source.py --sources 3 --verbose

  3. Custom Analysis Script:

  Create your own script using the core components:

  import asyncio
  from src.v2.data_sources import MarketDataManager, get_market_data_config

  async def analyze_crypto():
      config = get_market_data_config()
      manager = MarketDataManager(config)
      await manager.initialize()

      # Get Ethereum data from multiple sources
      eth_data = await manager.get_market_data("ETH", max_sources=3)
      print(f"ETH Price: ${eth_data.consensus_price:.2f}")
      print(f"Sources: {len(eth_data.price_sources)}")

      # Get sentiment data
      sentiment = await manager.get_sentiment_data()
      if sentiment:
          print(f"Fear & Greed: {sentiment.value} ({sentiment.value_classification})")

      await manager.close()

  # Run the analysis
  asyncio.run(analyze_crypto())

  4. Test Individual Data Sources:

  # Test only CoinGecko
  from src.v2.data_sources.adapters.coingecko_adapter import CoinGeckoAdapter
  from src.v2.data_sources.config import DataSourceConfig

  config = DataSourceConfig(
      name="CoinGecko",
      base_url="https://api.coingecko.com/api/v3"
  )

  adapter = CoinGeckoAdapter(config)
  await adapter.initialize()
  data = await adapter.get_market_data("ETH")
  print(f"CoinGecko ETH Price: ${data.price:.2f}")