# live_bot.py

import os
import argparse
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Import Consolidated Logic
from strategy_core import MonthlyFortressStrategy, is_rebalance_day

# --- CONFIGURATION ---
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PAPER_MODE = True  # Set to False for real money

LOG_FILE = "trade_history.csv"

# Initialize Alpaca
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)

# --- TELEGRAM ALERTS ---
def send_telegram(message):
    """Sends a notification to your phone."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

# --- MAIN EXECUTION ---
def execute_rebalance(force_trade=False):
    print("--- 🚀 STARTING REBALANCE CHECK ---")
    
    # 1. Check Schedule (unless forced)
    if force_trade:
        print("⚠️ FORCE MODE: Skipping Calendar Check.")
        send_telegram("⚠️ **Force Trade Activated**")
    else:
        if not is_rebalance_day(API_KEY, SECRET_KEY, PAPER_MODE):
            return # Exit silently if not rebalance day

    send_telegram("🚀 **Fortress Bot Activated**\nAnalyzing Market...")

    # 2. Run Strategy
    strategy = MonthlyFortressStrategy()
    tickers = list(set(strategy.risk_assets + strategy.safe_assets + ['SPY']))
    
    print("📊 Downloading Live Data...")
    try:
        data = yf.download(tickers, period="2y", progress=False)
        # Handle MultiIndex if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
    except Exception as e:
        print(f"❌ Data Error: {e}")
        send_telegram(f"❌ Critical Data Error: {e}")
        return

    # Get Signal
    today = pd.Timestamp.today()
    target_portfolio = strategy.get_signal(data, today)
    target_dict = {t: w for t, w in target_portfolio}
    
    print(f"🎯 Target Allocation: {target_dict}")

    # 3. Get Current Account Status
    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    
    trade_log = []
    
    # 4. EXECUTE TRADES
    
    # A. SELL First (to free up cash)
    for p in positions:
        if p.symbol not in target_dict:
            try:
                trade_client.close_position(p.symbol)
                trade_log.append(f"🔴 Sold All {p.symbol}")
                print(f"🔴 Sold All {p.symbol}")
            except Exception as e:
                print(f"❌ Error selling {p.symbol}: {e}")

    # B. BUY/TRIM
    # We take 95% of equity to leave a cash buffer for fees/slippage
    target_equity = equity * 0.95
    
    # Fetch live prices for sizing
    live_prices = {}
    for t in target_dict.keys():
        try:
            # Quick price check using Alpaca or fallback
            trade = trade_client.get_latest_trade(t)
            live_prices[t] = float(trade.price)
        except:
            print(f"⚠️ Could not get Alpaca price for {t}")

    for symbol, weight in target_dict.items():
        if symbol not in live_prices: continue
        
        price = live_prices[symbol]
        target_val = target_equity * weight
        
        # Skip tiny trades (<$50)
        if target_val < 50: continue 
        
        # --- FIXED FRACTIONAL LOGIC ---
        # Calculate exact quantity (e.g., 1.654 shares)
        target_qty = target_val / price
        
        # Alpaca requires only 9 decimal places max
        target_qty = round(target_qty, 9)

        current_qty = current_holdings.get(symbol, 0)
        delta_qty = target_qty - current_qty

        # Apply a small threshold so we don't trade tiny adjustments (e.g. 0.001 shares)
        if abs(delta_qty * price) < 10: continue 

        try:
            if delta_qty > 0:
                # Use Notional (Dollar Amount) for Buys if possible, or Qty for precise fractional
                # Ideally, use 'qty' but ensure your account has 'Fractional Trading' enabled in Alpaca Dashboard.
                trade_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol, 
                        qty=delta_qty,  # Now a float, not int
                        side=OrderSide.BUY, 
                        time_in_force=TimeInForce.DAY
                    )
                )
                trade_log.append(f"🟢 Bought {delta_qty:.4f} {symbol}")
                print(f"🟢 Bought {delta_qty:.4f} {symbol}")
                
            elif delta_qty < 0:
                trade_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol, 
                        qty=abs(delta_qty), 
                        side=OrderSide.SELL, 
                        time_in_force=TimeInForce.DAY
                    )
                )
                trade_log.append(f"📉 Trimmed {abs(delta_qty):.4f} {symbol}")
                print(f"📉 Trimmed {abs(delta_qty):.4f} {symbol}")
                
        except Exception as e:
            print(f"❌ Order Error {symbol}: {e}")

    # 5. Reporting
    report = f"✅ **Rebalance Complete**\n💰 Equity: ${equity:,.2f}\n"
    if trade_log: report += "\n".join(trade_log[:10]) 
    else: report += "\n(Portfolio matches target)"
    
    send_telegram(report)
    
    # Log to file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        for trade in trade_log: f.write(f"{timestamp},{trade}\n")
        
    print("--- EXECUTION COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help="Ignore schedule and trade NOW")
    args = parser.parse_args()

    execute_rebalance(force_trade=args.force)