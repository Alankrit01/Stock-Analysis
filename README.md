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

## Screenshots
<img width="1120" height="800" alt="Screenshot 2025-09-07 at 21 52 42" src="https://github.com/user-attachments/assets/beb3f788-90d1-4c90-b274-4ded9f92b236" />
<img width="1154" height="595" alt="Screenshot 2025-09-07 at 21 52 49" src="https://github.com/user-attachments/assets/24217f15-9c07-4257-8d77-1e0437bf949f" />
<img width="1166" height="624" alt="Screenshot 2025-09-07 at 21 53 02" src="https://github.com/user-attachments/assets/fe399afe-5baf-4be9-8790-f26e3ff00940" />
<img width="1153" height="676" alt="Screenshot 2025-09-07 at 21 53 09" src="https://github.com/user-attachments/assets/8feab063-418a-4d30-aa87-926dae50fdb8" />





