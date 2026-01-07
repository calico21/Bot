# execution.py

import os
import requests
from datetime import datetime, timedelta
import pandas as pd
# from dotenv import load_dotenv # Not needed for GitHub Actions
# load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
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
    """Sends a message to your Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram secrets missing. Skipping message.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

def get_latest_prices(tickers):
    """Fetches the latest price for a list of tickers."""
    if not tickers: return {}
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=5),
        end=datetime.now()
    )
    bars = data_client.get_stock_bars(request_params)
    latest_prices = {}
    for symbol in tickers:
        try:
            latest_prices[symbol] = bars[symbol][-1].close
        except:
            print(f"Warning: Could not get price for {symbol}")
    return latest_prices

def get_price_history_alpaca(tickers, days=400):
    """Gets historical data formatted for the Strategy."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df.reset_index()
    pivot = df.pivot(index="timestamp", columns="symbol", values="close")
    pivot.index = pivot.index.tz_convert(None) # Remove timezone for compatibility
    return pivot

def execute_monthly_rebalance():
    print("--- 🚀 STARTING MONTHLY REBALANCE ---")
    send_telegram("🚀 **Monthly Fortress Bot**\nAnalyzing market data...")
    
    strat = MonthlyFortressStrategy()
    
    # 1. Get Universe Data
    all_tickers = list(set(strat.risk_assets + strat.safe_assets + [strat.market_filter, strat.bond_benchmark]))
    prices = get_price_history_alpaca(all_tickers)
    
    # 2. Get Signal
    today = prices.index[-1]
    target_portfolio = strat.get_signal(prices, today)
    target_dict = {t: w for t, w in target_portfolio}
    
    print(f"Target: {target_dict}")
    
    # 3. Get Account Info
    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    
    trade_log = []

    # 4. Sell First (Clear old positions)
    for p in positions:
        if p.symbol not in target_dict:
            print(f"Selling {p.symbol}...")
            trade_client.close_position(p.symbol)
            trade_log.append(f"🔴 Sold {p.symbol}")
            
    # 5. Buy New Positions
    # Re-check buying power after sales
    buying_power = float(trade_client.get_account().equity) * 0.98 # 2% Cash Buffer
    latest_prices = get_latest_prices(list(target_dict.keys()))
    
    for symbol, weight in target_dict.items():
        if symbol not in latest_prices: continue
        
        target_value = buying_power * weight
        price = latest_prices[symbol]
        qty = int(target_value / price)
        
        if qty > 0:
            print(f"Buying {qty} shares of {symbol}...")
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            try:
                trade_client.submit_order(req)
                trade_log.append(f"🟢 Bought {symbol} ({weight:.0%})")
            except Exception as e:
                print(f"Error buying {symbol}: {e}")

    # 6. Send Report
    report = f"✅ **Rebalance Complete**\n"
    report += f"💰 Equity: ${equity:,.2f}\n\n"
    if trade_log:
        report += "\n".join(trade_log)
    else:
        report += "No trades needed."
        
    report += "\n\n**Current Portfolio:**\n"
    for s, w in target_dict.items():
        report += f"• {s}: {w:.1%}\n"
        
    send_telegram(report)
    print("--- REBALANCE COMPLETE ---")

if __name__ == "__main__":
    try:
        execute_monthly_rebalance()
    except Exception as e:
        send_telegram(f"❌ **CRITICAL ERROR**\nBot crashed:\n{str(e)}")
        raise e