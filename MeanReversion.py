"""
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

Dependencies: pandas, numpy, yfinance, statsmodels
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import yfinance as yf
try:
    import statsmodels.api as sm
except Exception:
    sm = None

# ------------------------ Utility & Data ------------------------ #

def load_price_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Load OHLCV for a symbol from yfinance."""
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {symbol} from yfinance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return data[cols].dropna()


# ------------------------ Indicators ------------------------ #

# Rule of thumb: Buy if Z ≤ -2, Sell if Z ≥ +2.
def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    close = df["Close"]
    sma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    z = (close - sma) / (std.replace(0, np.nan))
    out = pd.DataFrame({"BB_Mid": sma, "BB_Upper": upper, "BB_Lower": lower, "BB_Z": z})
    return df.join(out)

# Rule of thumb: Buy if RSI < 30 (oversold), Sell if RSI > 70 (overbought).
def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    close = df["Close"].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return df.join(pd.DataFrame({"RSI": rsi_val}))

# Rule of thumb: Buy if price < VWAP −2%, Sell if > VWAP +2%.
def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"].astype(float)
    vwap = pv.rolling(window).sum() / df["Volume"].rolling(window).sum()
    return df.join(pd.DataFrame({"VWAP": vwap}))


# ------------------------ Pair Trading Helpers ------------------------ #
# Rule of thumb: Buy KO/Sell PEP if z ≤ -2 (KO is cheap), Sell KO/Buy PEP if z ≥ +2 (KO is expensive)
def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    if sm is None:
        return 1.0
    x_const = sm.add_constant(x.values)
    model = sm.OLS(y.values, x_const, missing='drop')
    res = model.fit()
    beta = res.params[1] if len(res.params) > 1 else 1.0
    return float(beta)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.rolling(60).mean()
    sigma = series.rolling(60).std(ddof=0)
    return (series - mu) / sigma.replace(0, np.nan)


def pair_signal(y_df: pd.DataFrame, x_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[float]]:
    merged = pd.DataFrame({
        'Y': y_df["Close"].astype(float),
        'X': x_df["Close"].astype(float)
    }).dropna()
    if merged.empty:
        return merged, None
    beta = hedge_ratio(merged['Y'], merged['X'])
    spread = merged['Y'] - beta * merged['X']
    z = zscore(spread)
    out = merged.join(pd.DataFrame({'Spread': spread, 'Pair_Z': z}))
    latest_z = float(out['Pair_Z'].iloc[-1]) if not np.isnan(out['Pair_Z'].iloc[-1]) else None
    return out, latest_z


# ------------------------ Signal Logic ------------------------ #

@dataclass
class Signal:
    name: str
    score: float
    details: Dict[str, float]
    text: str


def bb_signal(df: pd.DataFrame, z_buy: float = -2.0, z_sell: float = 2.0) -> Signal:
    row = df.iloc[-1]
    z = float(row.get("BB_Z", np.nan))
    text = "No signal"
    score = 0.0
    if np.isfinite(z):
        if z <= z_buy:
            text = f"BB mean-reversion BUY: z={z:.2f}"
            score = 1.0
        elif z >= z_sell:
            text = f"BB mean-reversion SELL: z={z:.2f}"
            score = -1.0
        else:
            text = f"BB neutral: z={z:.2f}"
    return Signal(name="BollingerBands", score=score, details={"BB_Z": z}, text=text)


def rsi_signal(df: pd.DataFrame, low: int = 30, high: int = 70) -> Signal:
    r = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else np.nan
    text = "No signal"
    score = 0.0
    if np.isfinite(r):
        if r <= low:
            text = f"RSI BUY: RSI={r:.1f}"
            score = 1.0
        elif r >= high:
            text = f"RSI SELL: RSI={r:.1f}"
            score = -1.0
        else:
            text = f"RSI neutral: RSI={r:.1f}"
    return Signal(name="RSI", score=score, details={"RSI": r}, text=text)


def vwap_signal(df: pd.DataFrame, pct_band: float = 0.02) -> Signal:
    price = float(df["Close"].iloc[-1])
    vwap = float(df["VWAP"].iloc[-1]) if np.isfinite(df["VWAP"].iloc[-1]) else np.nan
    text = "No signal"
    score = 0.0
    dev = np.nan
    if np.isfinite(vwap):
        dev = (price - vwap) / vwap
        if dev <= -pct_band:
            text = f"VWAP BUY: {dev*100:.1f}% below VWAP"
            score = 1.0
        elif dev >= pct_band:
            text = f"VWAP SELL: {dev*100:.1f}% above VWAP"
            score = -1.0
        else:
            text = f"VWAP neutral: {dev*100:.1f}%"
    return Signal(name="VWAP", score=score, details={"VWAP": vwap, "VWAP_Deviation": dev}, text=text)


def pair_trading_signal(target: str, target_df: pd.DataFrame, pair_symbol: str, pair_df: pd.DataFrame,
                        open_z: float = 2.0) -> Signal:
    out, latest_z = pair_signal(target_df, pair_df)
    text = "No signal"
    score = 0.0
    if latest_z is None or not np.isfinite(latest_z):
        return Signal(name="PairTrading", score=0.0, details={"Pair_Z": np.nan}, text="Pair signal unavailable")

    if latest_z <= -open_z:
        score = 1.0
        text = f"PAIR BUY: z={latest_z:.2f}"
    elif latest_z >= open_z:
        score = -1.0
        text = f"PAIR SELL: z={latest_z:.2f}"
    else:
        text = f"PAIR neutral: z={latest_z:.2f}"
    return Signal(name=f"Pair({pair_symbol})", score=score, details={"Pair_Z": latest_z}, text=text)


# ------------------------ Aggregation ------------------------ #

def aggregate_signals(signals: Dict[str, Signal]) -> Tuple[str, float, str]:
    scores = np.array([s.score for s in signals.values() if np.isfinite(s.score)])
    if len(scores) == 0:
        return "HOLD", 0.0, "No valid signals"
    avg = float(np.clip(scores.mean(), -1.0, 1.0))
    if avg > 0.2:
        rec = "BUY"
    elif avg < -0.2:
        rec = "SELL"
    else:
        rec = "HOLD"
    rationale = "; ".join([s.text for s in signals.values()])
    return rec, avg, rationale


# ------------------------ Runner ------------------------ #

def run(symbol: str, period: str = "1y", interval: str = "1d", pair: Optional[str] = None,
        bb_window: int = 20, bb_std: float = 2.0,
        rsi_period: int = 14,
        vwap_window: int = 20,
        vwap_pct_band: float = 0.02) -> Dict[str, Signal]:

    df = load_price_data(symbol, period=period, interval=interval)
    df = bollinger_bands(df, window=bb_window, num_std=bb_std)
    df = rsi(df, period=rsi_period)
    df = rolling_vwap(df, window=vwap_window)

    signals = {
        "BollingerBands": bb_signal(df),
        "RSI": rsi_signal(df),
        "VWAP": vwap_signal(df, pct_band=vwap_pct_band)
    }

    if pair is None:
        pair = "SPY"
    pair_df = load_price_data(pair, period=period, interval=interval)
    signals[f"Pair({pair})"] = pair_trading_signal(symbol, df, pair, pair_df)

    return signals, df


# ------------------------ Stock Snapshot ------------------------ #

def stock_snapshot(symbol: str, df: pd.DataFrame) -> Dict[str, float]:
    ticker = yf.Ticker(symbol)
    info = ticker.info

    last_price = float(df["Close"].iloc[-1])
    month_high = df["High"].tail(30).max()
    month_low = df["Low"].tail(30).min()

    return {
        "Current Price": last_price,
        "1M High": month_high,
        "1M Low": month_low,
        "EPS": info.get("trailingEps", np.nan),
        "P/E Ratio": info.get("trailingPE", np.nan),
        "P/B Ratio": info.get("priceToBook", np.nan)
    }


# ------------------------ Main ------------------------ #

def main():
    p = argparse.ArgumentParser(description="Mean-Reversion Strategy Analyzer")
    p.add_argument("--symbol", required=True, help="Ticker symbol, e.g., AAPL")
    p.add_argument("--pair", default=None, help="Optional pair symbol (default SPY)")
    p.add_argument("--period", default="1y", help="yfinance period, e.g., 6mo, 1y, 2y")
    p.add_argument("--interval", default="1d", help="yfinance interval, e.g., 1d, 1h, 15m")
    args = p.parse_args()

    try:
        signals, df = run(symbol=args.symbol, period=args.period, interval=args.interval, pair=args.pair)
        rec, score, rationale = aggregate_signals(signals)

        print("\n=== Mean-Reversion Signals ===")
        for name, s in signals.items():
            print(f"- {name}: {s.text} (score={s.score:+.2f})")

        print("\n=== Overall Recommendation ===")
        print(f"{args.symbol.upper()}: {rec} (avg score={score:+.2f})")
        print("Rationale:", rationale)

        snap = stock_snapshot(args.symbol, df)
        print("\n=== Stock Snapshot ===")
        for k, v in snap.items():
            print(f"{k}: {v}")

        # Save report
        rows = []
        for name, s in signals.items():
            row = {"Strategy": name, "Score": s.score, "Details": s.details, "Text": s.text}
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"report_{args.symbol.upper()}.csv", index=False)
        print(f"\nSaved CSV report to: report_{args.symbol.upper()}.csv")

    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()