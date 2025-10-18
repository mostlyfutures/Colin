# Implementation Plan: Institutional-Grade Signal Scoring Bot for Crypto Perpetuals

## Overview
This plan details the development of a **signal scoring bot** that synthesizes institutional-grade market structure and order flow principles—specifically from the ICT framework and market microstructure research—into a real-time “Long Confidence %” (0–100%) for crypto perpetuals. Unlike generic technical bots, this system is grounded in **liquidity dynamics, order book imbalance, killzones, and liquidation mechanics** as validated by empirical research on leveraged crypto derivatives. The output includes a human-readable rationale based on **smart money behavior**, not just indicator confluence.

## Requirements Summary
- Generate a real-time **Long/Short Confidence Score** for ETH/BTC perpetuals
- Incorporate **liquidation heatmap data** to identify stop-hunt zones
- Detect **ICT structural elements**: Fair Value Gaps (FVGs), Order Blocks (OBs), Break of Structure (BOS)
- Apply **Killzone session filters** (Asian, London, NY in UTC)
- Analyze **order book imbalance** and **order flow delta** as predictive signals
- Track **Open Interest (OI)** and **volume trends** from Binance Futures
- Output top 3 institutional rationale points (e.g., “Price approaching untested bullish OB during London killzone”)
- Modular design to support future auto-execution

## Research Findings

### Best Practices (from Knowledge Base)
- **Liquidity is the core driver**: “Liquidity, in an ICT context, refers to any price level where a concentration of stop orders exists… Identifying these liquidity pockets allows a trader to anticipate potential reversals” [[p.8]].
- **Order book imbalance is predictive**: “A strongly negative value… indicates overwhelming sell pressure, which historically has correlated with lower future price returns” [[p.3]].
- **Killzones matter**: “Entering a trade during a London or New York session open… aligns one with the primary market movers” [[p.8]].
- **Stop-loss placement must be structural**: “For long positions, the stop is typically placed just below a validated Order Block or the low of a Bullish FVG reaction” [[p.12]].
- **Leverage amplifies liquidation risk**: “Forced closures can create a domino effect… Trading against these clusters can be profitable” [[p.11]].

### Reference Implementations & APIs
- **CoinGlass** provides liquidation heatmap via `/api/futures/liquidation/heatmap/model3` .
- **Bookmap Connect API (L0)** allows feeding raw order book data for custom imbalance/delta analysis [[14],[16]].
- **Binance Futures** exposes real-time OI via `GET /fapi/v1/openInterest` .

### Technology Decisions
- **Python** for rapid prototyping and rich data science stack
- **ICT concepts operationalized algorithmically**:
  - FVG = 3-candle pattern with non-overlapping wicks/body
  - OB = last opposing candle before strong displacement
  - BOS = break of prior swing high/low
- **Order flow metrics**: Use normalized order book imbalance (NOBI) and trade-side delta from Level 2 data
- **Session logic**: UTC-based killzones (Asian: 00:00–09:00, London: 07:00–16:00, NY: 12:00–22:00)

## Implementation Tasks

### Phase 1: Foundation
1. **Project Setup & Config Management**
   - Description: Initialize repo with config for symbols, sessions, API keys, and scoring weights
   - Files: `config.yaml`, `requirements.txt`, `src/core/`
   - Effort: 2h

2. **Data Adapters for Institutional Signals**
   - Description: Build adapters for CoinGlass (liquidations), Binance (OI/volume/OHLCV), and session time
   - Files: `src/adapters/coinglass.py`, `src/adapters/binance.py`, `src/utils/sessions.py`
   - Effort: 6h

### Phase 2: Core Signal Engine
3. **ICT Structure Detector**
   - Description: Algorithmically detect FVGs, OBs, and BOS on 1H/4H timeframes using candle logic
   - Files: `src/structure/ict_detector.py`
   - Dependencies: Binance adapter
   - Effort: 8h

4. **Order Flow Analyzer**
   - Description: Compute order book imbalance and aggressive trade delta from Binance L2 data (or Bookmap L0 if available)
   - Files: `src/orderflow/analyzer.py`
   - Effort: 10h

5. **Normalized Scorers for Institutional Factors**
   - Description: Build scorers for:
     - Liquidity proximity (to liquidation clusters)
     - ICT confluence (price near fresh OB + FVG)
     - Killzone alignment
     - OI/volume confirmation
     - Order flow delta
   - Files: `src/scorers/`
   - Effort: 12h

6. **Composite Scoring Engine**
   - Description: Weighted aggregation into Long/Short Confidence %; generate top-3 rationale using template strings
   - Files: `src/engine/institutional_scorer.py`
   - Effort: 6h

### Phase 3: Integration & Validation
7. **Risk-Aware Output Formatter**
   - Description: Include volatility warning and structural stop-loss level in output
   - Files: `src/output/formatter.py`
   - Effort: 3h

8. **Backtesting & Edge Validation**
   - Description: Test if high-confidence signals correlate with positive returns over 1h/4h; validate ICT logic
   - Files: `tests/backtest_signal_correlation.py`
   - Effort: 10h

## Codebase Integration Points

### New Files to Create
- `src/structure/ict_detector.py` – Detect FVGs, OBs, BOS algorithmically
- `src/orderflow/analyzer.py` – Compute NOBI and delta from L2 data
- `src/scorers/liquidity_scorer.py` – Proximity to CoinGlass liquidation zones
- `src/scorers/killzone_scorer.py` – Session-based confidence boost
- `src/engine/institutional_scorer.py` – Final aggregation and rationale

### Existing Patterns to Follow
- **Algorithmic ICT**: Translate subjective concepts into rule-based detection (e.g., FVG = candle[0].high < candle[1].low and candle[2].high < candle[1].low)
- **Killzone Filtering**: Only score trades during high-institutional-activity windows [[p.8]]
- **Liquidity-First Logic**: Prioritize liquidation heatmap and stop-hunt zones over generic support/resistance [[p.11]]

## Technical Design

### Data Flow
1. Fetch OHLCV, OI, volume from Binance
2. Pull liquidation heatmap from CoinGlass
3. Determine active killzone (UTC session)
4. Detect ICT structures (FVG, OB) on 4H/1H
5. Calculate order flow metrics from L2 data
6. Normalize and score each institutional factor
7. Aggregate into confidence % + rationale

### Key Metrics
- **Liquidity Proximity Score**: Distance to nearest high-density liquidation zone
- **ICT Confluence Score**: Price within fresh OB + FVG zone
- **Killzone Boost**: +10–20% if in London/NY overlap
- **Order Flow Delta**: Aggressive buy/sell imbalance near structure

## Dependencies and Libraries
- `ccxt` – Binance Futures API
- `requests` – CoinGlass API
- `pandas` – Time series analysis
- `pytz` – Session time logic
- Optional: `bookmap-api` (if using L0 feed)

## Testing Strategy
- Unit tests for FVG/OB detection logic
- Mock API responses for liquidation/OI data
- Backtest: Does 80%+ confidence predict positive 4h returns?
- Edge cases: No liquidations, flat OI, session overlaps

## Success Criteria
- [ ] Detects FVGs and OBs with >90% accuracy vs manual labeling
- [ ] Confidence score correlates with forward returns (Pearson r > 0.3)
- [ ] Rationale explains institutional context (e.g., “liquidity grab below swing low”)
- [ ] Handles all API failures gracefully
- [ ] Outputs structural stop-loss level for risk context

## Notes and Considerations
- **Bookmap L0 is premium**—start with Binance L2 as proxy for order flow 
- **Killzone timing is critical**: London/NY overlap (12:00–16:00 UTC) is highest conviction [[p.8]]
- **Never trade against liquidation clusters**: Use them as targets, not entries [[p.11]]
- **Future enhancement**: Add funding rate filter (extreme rates = reversal signal) [[p.13]]

---
*This plan is ready for execution with `/execute-plan`*