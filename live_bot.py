# live_bot.py

import os
import argparse
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Import Consolidated Logic (Removed is_rebalance_day to avoid conflict)
from strategy_core import MonthlyFortressStrategy

# --- CONFIGURATION ---
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PAPER_MODE = True  # Set to False for real money

LOG_FILE = "trade_history.csv"
PERF_FILE = "live_performance.csv"
DASHBOARD_IMG = "dashboard.png"
DNA_FILE = "winner_dna.json"

# Initialize Alpaca
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)

# --- TELEGRAM ALERTS ---
def send_telegram_message(message):
    """Sends a text notification."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Telegram Text Error: {e}")

def send_telegram_photo(photo_path, caption=""):
    """Sends an image with caption."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            requests.post(
                url, 
                data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}, 
                files={'photo': photo}
            )
    except Exception as e:
        print(f"❌ Telegram Photo Error: {e}")

# --- 🧬 DNA LOADER (NEW) ---
def load_fixed_dna(file_path=DNA_FILE):
    """Loads your already-optimized parameters from the JSON file."""
    if not os.path.exists(file_path):
        print(f"⚠️ {file_path} not found. Using default strategy params.")
        return None
        
    try:
        with open(file_path, 'r') as f:
            dna = json.load(f)
            # Support both 'params' (standard) and 'best_params' (Optuna) keys
            params = dna.get('params', dna.get('best_params', {}))
            print(f"🧬 DNA Loaded: Using Optimized Settings (Lev: {params.get('max_lev', 'N/A')})")
            return params
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None

# --- 📅 SCHEDULER LOGIC (NEW) ---
def is_rebalance_day(client):
    """
    Returns True ONLY on the Last Friday of the Month,
    and ONLY if the market is actually open.
    """
    today = datetime.now()
    
    # 1. Check if it is Friday (Friday = 4)
    if today.weekday() != 4:
        return False

    # 2. Check if it is the LAST Friday
    # Logic: If next week is a new month, then today is the last specific weekday of this month.
    next_week = today + timedelta(days=7)
    if next_week.month == today.month:
        return False  # There is another Friday left in this month

    # 3. Check if Market is Open (Handles holidays)
    try:
        clock = client.get_clock()
        if not clock.is_open:
            print("⚠️ It is the Last Friday, but the Market is CLOSED.")
            return False
    except Exception as e:
        print(f"⚠️ Clock Error: {e}")
        return False

    return True

# --- 🛡️ CIRCUIT BREAKER ---
def run_circuit_breaker(max_loss_percent=-0.045):
    """
    Checks if today's P&L is worse than max_loss_percent.
    If so, LIQUIDATES ALL POSITIONS to Cash.
    """
    try:
        acct = trade_client.get_account()
        equity = float(acct.equity)
        last_equity = float(acct.last_equity)
        
        # Calculate daily return
        daily_return = (equity - last_equity) / last_equity
        
        if daily_return < max_loss_percent:
            msg = f"🚨 **CIRCUIT BREAKER TRIGGERED** 🚨\n\n📉 Daily Loss: {daily_return:.2%}\n🛑 Liquidating Portfolio to Cash!"
            print(msg)
            send_telegram_message(msg)
            
            # LIQUIDATE EVERYTHING
            trade_client.close_all_positions(cancel_orders=True)
            send_telegram_message("✅ **Emergency Liquidation Complete.** Sleeping until manual reset.")
            return True # Triggered
            
    except Exception as e:
        print(f"⚠️ Circuit Breaker Error: {e}")
        
    return False # Safe

# --- TRACKER MODE (EVENING REPORT) ---
def run_tracker():
    """Generates daily stats, updates CSV, creates chart, and sends report."""
    print("--- 🌙 RUNNING DAILY TRACKER ---")
    try:
        acct = trade_client.get_account()
        equity = float(acct.equity)
        last_equity = float(acct.last_equity)
        day_change_usd = equity - last_equity
        day_change_pct = (day_change_usd / last_equity) * 100 if last_equity else 0
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        new_row = pd.DataFrame([{"Date": today_str, "Equity": equity}])
        
        if os.path.exists(PERF_FILE):
            df = pd.read_csv(PERF_FILE)
            df = df[df["Date"] != today_str]
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
            
        df.to_csv(PERF_FILE, index=False)
        
        # Charting
        plt.figure(figsize=(10, 5))
        plot_df = df.copy()
        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
        plot_df = plot_df.sort_values('Date')
        
        plt.plot(plot_df['Date'], plot_df['Equity'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        plt.fill_between(plot_df['Date'], plot_df['Equity'], alpha=0.1, color='#1f77b4')
        plt.title(f"Portfolio Value: ${equity:,.2f}", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(DASHBOARD_IMG)
        plt.close()

        positions = trade_client.get_all_positions()
        header_emoji = "🚀" if day_change_pct > 0 else "🔻"
        report = f"{header_emoji} **Daily Market Wrap**\n\n"
        report += f"💰 **Equity:** ${equity:,.2f}\n"
        report += f"📊 **Day Change:** {day_change_pct:+.2f}% (${day_change_usd:+.2f})\n\n"
        report += "**Current Holdings:**\n"

        if not positions:
            report += "_(100% Cash)_"
        else:
            for p in positions:
                symbol = p.symbol
                market_val = float(p.market_value)
                total_pl_pct = float(p.unrealized_plpc) * 100
                total_icon = "🟢" if total_pl_pct >= 0 else "🔻"
                report += f"• **{symbol}** | 💵 ${market_val:,.0f} | {total_icon} {total_pl_pct:+.1f}%\n"

        send_telegram_photo(DASHBOARD_IMG, report)
        print("✅ Daily report sent.")
        
    except Exception as e:
        print(f"❌ Tracker Error: {e}")
        send_telegram_message(f"❌ Tracker Failed: {e}")

# --- REBALANCE MODE (TRADING) ---
def execute_rebalance(force_trade=False):
    print("--- 🚀 STARTING REBALANCE CHECK ---")
    
    # CHECK 1: Is it the right day?
    if not force_trade:
        if not is_rebalance_day(trade_client):
            print("💤 Not the Last Friday or Market is Closed. Sleeping.")
            return 

    send_telegram_message("🚀 **Fortress Bot Activated**\nAnalyzing Market...")

    # Load DNA & Initialize Strategy
    dna_params = load_fixed_dna()
    strategy = MonthlyFortressStrategy()
    
    # Inject Optimized Parameters if available
    if dna_params:
        strategy.MAX_PORTFOLIO_LEVERAGE = dna_params.get('max_lev', strategy.MAX_PORTFOLIO_LEVERAGE)
        strategy.CRASH_THRESHOLD = dna_params.get('crash_thresh', strategy.CRASH_THRESHOLD)
        # Add other parameter injections here if your strategy class supports them
    
    tickers = list(set(strategy.risk_assets + strategy.safe_assets + ['SPY']))
    
    print("📊 Downloading Live Data (5y History)...")
    try:
        # FIX: Increased to 5y to ensure 300-day SMA has enough data
        data = yf.download(tickers, period="5y", progress=False)
        
        # Handle yfinance MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
            
        # FIX: Forward Fill to prevent NaNs from killing the signals
        data = data.ffill()
        
    except Exception as e:
        print(f"❌ Data Error: {e}")
        send_telegram_message(f"❌ Critical Data Error: {e}")
        return

    # FIX: Use the last available market date (avoids timestamp mismatch)
    last_market_date = data.index[-1]
    print(f"📅 Analyzing data for date: {last_market_date.date()}")
    
    target_portfolio = strategy.get_signal(data, last_market_date)
    target_dict = {t: w for t, w in target_portfolio}
    
    print(f"🎯 Target Allocation: {target_dict}")

    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    
    trade_log = []
    
    # SELL First (Clear space)
    for p in positions:
        if p.symbol not in target_dict:
            try:
                trade_client.close_position(p.symbol)
                trade_log.append(f"🔴 Sold All {p.symbol}")
            except Exception as e:
                print(f"❌ Error selling {p.symbol}: {e}")

    # BUY/TRIM (Delta Execution)
    target_equity = equity * 0.95 # Leave 5% buffer for slippage
    live_prices = {}
    
    # Get prices for target assets
    for t in target_dict.keys():
        try:
            trade = trade_client.get_latest_trade(t)
            live_prices[t] = float(trade.price)
        except:
            print(f"⚠️ No live price found for {t}")

    # Execute Orders
    for symbol, weight in target_dict.items():
        if symbol not in live_prices: continue
        price = live_prices[symbol]
        target_val = target_equity * weight
        
        # Skip tiny allocations
        if target_val < 50: continue 
        
        target_qty = round(target_val / price, 4) # Rounding to 4 decimal places
        current_qty = current_holdings.get(symbol, 0)
        delta_qty = target_qty - current_qty

        # Skip insignificant moves (save fees/noise)
        if abs(delta_qty * price) < 10: continue 

        try:
            if delta_qty > 0:
                trade_client.submit_order(MarketOrderRequest(symbol=symbol, qty=delta_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                trade_log.append(f"🟢 Bought {delta_qty:.4f} {symbol}")
            elif delta_qty < 0:
                # We use abs() because sell order quantity must be positive
                trade_client.submit_order(MarketOrderRequest(symbol=symbol, qty=abs(delta_qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                trade_log.append(f"📉 Trimmed {abs(delta_qty):.4f} {symbol}")
        except Exception as e:
            print(f"❌ Order Error {symbol}: {e}")

    report = f"✅ **Rebalance Complete**\n💰 Equity: ${equity:,.2f}\n"
    if trade_log: report += "\n".join(trade_log[:10]) 
    else: report += "\n(Portfolio matches target)"
    send_telegram_message(report)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        for trade in trade_log: f.write(f"{timestamp},{trade}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help="Ignore schedule and trade NOW")
    parser.add_argument('--track', action='store_true', help="Run Daily Tracker (No Trading)")
    args = parser.parse_args()

    # MODE 1: Daily Tracker (Runs once in the evening)
    if args.track:
        run_tracker()

    # MODE 2: Normal Trading Loop (Runs every 5 mins or daily via GitHub)
    else:
        # STEP 1: SAFETY CHECK (Circuit Breaker)
        crash_triggered = run_circuit_breaker(max_loss_percent=-0.045)

        if crash_triggered:
            print("🛑 Execution Halted due to Circuit Breaker.")
            exit() # Stop script here. Do NOT rebalance.

        # STEP 2: STRATEGY CHECK
        execute_rebalance(force_trade=args.force)