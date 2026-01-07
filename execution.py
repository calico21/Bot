# execution.py

import os
import requests
from datetime import datetime, timedelta
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from strategy import MonthlyFortressStrategy

# --- CONFIG ---
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PAPER_MODE = True

trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

def get_price_history_alpaca(tickers, days=400):
    """Gets historical data using the IEX Feed (Free)."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df.reset_index()
    pivot = df.pivot(index="timestamp", columns="symbol", values="close")
    pivot.index = pivot.index.tz_convert(None) 
    return pivot

def execute_monthly_rebalance():
    print("--- 🚀 STARTING MONTHLY REBALANCE ---")
    send_telegram("🚀 **Monthly Fortress Bot**\nAnalyzing market data...")
    
    strat = MonthlyFortressStrategy()
    
    # 1. Get Universe Data
    all_tickers = list(set(strat.risk_assets + strat.safe_assets + [strat.market_filter, strat.bond_benchmark]))
    try:
        prices = get_price_history_alpaca(all_tickers)
    except Exception as e:
        send_telegram(f"❌ Data Error: {e}")
        return

    if prices.empty: return

    # 2. Get Signal
    today = prices.index[-1]
    target_portfolio = strat.get_signal(prices, today)
    target_dict = {t: w for t, w in target_portfolio}
    
    # 3. Execution Setup
    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    trade_log = []
    
    # Use the LAST KNOWN CLOSE as the trade price (Fail-safe)
    last_known_prices = prices.iloc[-1].to_dict()

    # 4. Sell First
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    for p in positions:
        if p.symbol not in target_dict:
            trade_client.close_position(p.symbol)
            trade_log.append(f"🔴 Sold {p.symbol}")
            
    # 5. Buy New Positions
    target_equity = float(trade_client.get_account().equity) * 0.93 # 7% Buffer for slippage/price moves
    
    for symbol, weight in target_dict.items():
        # Fallback price logic
        price = last_known_prices.get(symbol)
        if not price: 
            print(f"Skipping {symbol} (No Price)")
            continue
            
        target_val = target_equity * weight
        target_qty = int(target_val / price)
        current_qty = current_holdings.get(symbol, 0)
        delta_qty = target_qty - current_qty
        
        if delta_qty > 0:
            print(f"Buying {delta_qty} {symbol}...")
            try:
                trade_client.submit_order(MarketOrderRequest(symbol=symbol, qty=delta_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                trade_log.append(f"🟢 Bought {delta_qty} {symbol}")
            except Exception as e:
                print(f"Error {symbol}: {e}")

        elif delta_qty < 0:
            sell_qty = abs(delta_qty)
            print(f"Trimming {sell_qty} {symbol}...")
            try:
                trade_client.submit_order(MarketOrderRequest(symbol=symbol, qty=sell_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                trade_log.append(f"📉 Trimmed {sell_qty} {symbol}")
            except Exception as e:
                print(f"Error {symbol}: {e}")

    # 6. Report
    report = f"✅ **Rebalance Complete**\n💰 Equity: ${equity:,.2f}\n\n"
    report += "\n".join(trade_log) if trade_log else "No trades needed."
    report += "\n\n**Target:**\n" + "\n".join([f"• {s}: {w:.1%}" for s, w in target_dict.items()])
    
    send_telegram(report)
    print("--- COMPLETE ---")

if __name__ == "__main__":
    execute_monthly_rebalance()