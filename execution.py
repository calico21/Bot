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
from alpaca.data.enums import DataFeed  # <--- NEW IMPORT

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
    """Fetches the latest price using the IEX Feed (Free Real-Time)."""
    if not tickers: return {}
    
    # FIX: Added feed=DataFeed.IEX to avoid 403 Forbidden Error
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=5),
        end=datetime.now(),
        feed=DataFeed.IEX 
    )
    
    try:
        bars = data_client.get_stock_bars(request_params)
        latest_prices = {}
        for symbol in tickers:
            if symbol in bars:
                latest_prices[symbol] = bars[symbol][-1].close
        return latest_prices
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return {}

def get_price_history_alpaca(tickers, days=400):
    """Gets historical data using the IEX Feed (Free)."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    
    # FIX: Added feed=DataFeed.IEX to avoid 403 Forbidden Error
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
    pivot.index = pivot.index.tz_convert(None) # Remove timezone for compatibility
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
        print(f"Data Error: {e}")
        send_telegram(f"❌ Data Error: {e}")
        return

    # 2. Get Signal
    if prices.empty:
        print("Error: No price data returned.")
        return

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
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    
    for p in positions:
        if p.symbol not in target_dict:
            print(f"Selling {p.symbol}...")
            trade_client.close_position(p.symbol)
            trade_log.append(f"🔴 Sold {p.symbol}")
            
    # 5. Buy New Positions
    # Re-check buying power after sales
    # Use 95% of equity to be safe against price fluctuations during execution
    target_equity = float(trade_client.get_account().equity) * 0.95
    latest_prices = get_latest_prices(list(target_dict.keys()))
    
    for symbol, weight in target_dict.items():
        if symbol not in latest_prices: 
            print(f"Skipping {symbol} (No price data)")
            continue
        
        price = latest_prices[symbol]
        target_val = target_equity * weight
        target_qty = int(target_val / price)
        
        # Check if we already hold it
        current_qty = current_holdings.get(symbol, 0)
        
        # Calculate difference
        delta_qty = target_qty - current_qty
        
        if delta_qty > 0:
            print(f"Buying {delta_qty} shares of {symbol}...")
            req = MarketOrderRequest(
                symbol=symbol,
                qty=delta_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            try:
                trade_client.submit_order(req)
                trade_log.append(f"🟢 Bought {delta_qty} {symbol} ({weight:.0%})")
            except Exception as e:
                print(f"Error buying {symbol}: {e}")
                
        elif delta_qty < 0:
            # Need to sell some shares to rebalance
            sell_qty = abs(delta_qty)
            print(f"Trimming {sell_qty} shares of {symbol}...")
            req = MarketOrderRequest(
                symbol=symbol,
                qty=sell_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            try:
                trade_client.submit_order(req)
                trade_log.append(f"📉 Trimmed {sell_qty} {symbol}")
            except Exception as e:
                print(f"Error trimming {symbol}: {e}")

    # 6. Send Report
    report = f"✅ **Rebalance Complete**\n"
    report += f"💰 Equity: ${equity:,.2f}\n\n"
    if trade_log:
        report += "\n".join(trade_log)
    else:
        report += "No trades needed (Portfolio is balanced)."
        
    report += "\n\n**Current Target:**\n"
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