# Stock-Analysis

Mean-Reversion Strategy Analyzer

- Bollinger Bands (BB) reversion
- RSI overbought/oversold
- Rolling VWAP reversion (20D by default)
- Pair Trading (stat-arb) vs optional peer (default: SPY)

Usage:
    python MeanReversion.py --symbol KO --pair PEP --period 1y --interval 1d

Outputs:
- Console summary of each strategy's latest signal
- Overall aggregated recommendation
- CSV report with indicators & signals: ./report_<SYMBOL>.csv
- Stock snapshot: month high/low, current price, EPS, P/E, P/B

Testing strategies on a simulated stock trading environment which handles stocks from the LSE and NYSE.

- Example trades:
    - trader.place_order('AAPL', 'SELL', 5)
    - trader.place_order('LLOY.L', 'BUY', 100)
    - trader.display_portfolio()
    - trader.display_trade_history()

