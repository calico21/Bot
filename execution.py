import os
import sys
import argparse
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 1. Import Strategy
from strategy import MonthlyFortressStrategy

# 2. Import Logic
from execution_logic import is_rebalance_day

# --- CONFIG ---
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PAPER_MODE = True

# FILES
LOG_FILE = "trade_history.csv"
PERF_FILE = "live_performance.csv"
DASHBOARD_IMG = "live_dashboard.png"

# CLIENTS
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)

# --- TELEGRAM FUNCTIONS ---
def send_telegram(message):
    """Sends a text message to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram Token/Chat ID missing.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

def send_telegram_photo(image_path):
    """Sends an image to Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(image_path, 'rb') as photo:
            payload = {'chat_id': TELEGRAM_CHAT_ID}
            files = {'photo': photo}
            requests.post(url, data=payload, files=files)
            print("✅ Dashboard sent to Telegram.")
    except Exception as e:
        print(f"❌ Failed to send photo: {e}")

# --- PERFORMANCE TRACKING ---
def get_performance_stats(current_equity):
    if not os.path.exists(PERF_FILE): return 0.0, 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(PERF_FILE)
        if df.empty: return 0.0, 0.0, 0.0, 0.0
        
        start_equity = float(df.iloc[0]['Equity'])
        total_chg = current_equity - start_equity
        total_pct = (total_chg / start_equity) * 100 if start_equity else 0.0
        
        if len(df) > 1:
            last_close = float(df.iloc[-2]['Equity'])
        else:
            last_close = start_equity
            
        daily_chg = current_equity - last_close
        daily_pct = (daily_chg / last_close) * 100 if last_close else 0.0
        
        return daily_chg, daily_pct, total_chg, total_pct
    except: return 0.0, 0.0, 0.0, 0.0

def update_performance_tracker(current_equity):
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{"Date": today_str, "Equity": float(current_equity)}])
    
    if os.path.exists(PERF_FILE):
        df = pd.read_csv(PERF_FILE)
        if not df.empty and df.iloc[-1]['Date'] == today_str:
            df.iloc[-1, df.columns.get_loc("Equity")] = float(current_equity)
        else:
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    
    df.to_csv(PERF_FILE, index=False)
    
    if len(df) < 2: return None
    
    # Generate Chart
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
    
    plt.switch_backend('Agg') 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(df.index, df['Equity'], color='#00ff00', linewidth=2)
    ax1.set_title("Live Portfolio Performance", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Equity ($)")
    
    ax2.fill_between(df.index, df['Drawdown'], 0, color='#ff0000', alpha=0.3)
    ax2.plot(df.index, df['Drawdown'], color='#ff0000', linewidth=1)
    ax2.set_title("Drawdown Risk (%)", fontsize=10)
    ax2.set_ylabel("DD %")
    ax2.grid(True, alpha=0.3)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(DASHBOARD_IMG)
    plt.close()
    return DASHBOARD_IMG

# --- MODE 1: TRACKER (DAILY) ---
def run_tracker_only():
    print("--- 📊 RUNNING DAILY TRACKER ---")
    acct = trade_client.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    
    print(f"💰 Equity: ${equity:,.2f}")
    d_chg, d_pct, t_chg, t_pct = get_performance_stats(equity)
    img_path = update_performance_tracker(equity)
    
    d_icon = "🚀" if d_chg >= 0 else "🔻"
    t_icon = "🏆" if t_chg >= 0 else "📉"
    
    msg = (f"📊 **Daily Update**\n"
           f"💰 Equity: ${equity:,.2f}\n"
           f"💵 Cash: ${cash:,.2f}\n\n"
           f"{d_icon} **Day:** ${d_chg:+.2f} ({d_pct:+.2f}%)\n"
           f"{t_icon} **Total:** ${t_chg:+.2f} ({t_pct:+.2f}%)")
           
    send_telegram(msg)
    if img_path and os.path.exists(img_path):
        send_telegram_photo(img_path)
    print("--- TRACKING COMPLETE ---")

# --- MODE 2: REBALANCE (MONTHLY) ---
def execute_monthly_rebalance(force_trade=False):
    print("--- 🚀 STARTING REBALANCE CHECK ---")
    
    # 1. SMART CALENDAR CHECK
    # Logic: If force_trade is TRUE, we skip the calendar check.
    if force_trade:
        print("⚠️ FORCE MODE ENABLED: Skipping Calendar Check.")
        send_telegram("⚠️ **Force Trade Activated**\nManual Override: Executing Strategy Immediately.")
    else:
        if not is_rebalance_day(API_KEY, SECRET_KEY, paper=PAPER_MODE):
            print("💤 Not a rebalance day.")
            # We removed the spammy 'Sleeping' message here. It will just be silent.
            return
        else:
            print("✅ Today is the First Trading Day! Executing Strategy.")
            send_telegram("🚀 **Fortress Bot Activated**\nMarket Open. Analyzing Regime...")

    # 2. STRATEGY EXECUTION
    strategy = MonthlyFortressStrategy()
    tickers = list(set(strategy.risk_assets + strategy.safe_assets + ['SPY', 'IEF']))
    
    print("📊 Downloading Market Data...")
    data = yf.download(tickers, period="2y", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
        
    today = pd.Timestamp.today()
    target_portfolio = strategy.get_signal(data, today)
    target_dict = {t: w for t, w in target_portfolio}
    print(f"🎯 Target Allocation: {target_dict}")

    # 3. EXECUTE TRADES
    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    
    trade_log = []
    
    live_prices = {}
    for t in target_dict.keys():
        try:
            trade = trade_client.get_latest_trade(t)
            live_prices[t] = float(trade.price)
        except:
            print(f"⚠️ Could not get price for {t}")

    # A. Sell
    for p in positions:
        if p.symbol not in target_dict:
            try:
                trade_client.close_position(p.symbol)
                trade_log.append(f"🔴 Sold All {p.symbol}")
            except Exception as e:
                print(f"❌ Error selling {p.symbol}: {e}")

    # B. Buy
    target_equity = equity * 0.95
    for symbol, weight in target_dict.items():
        if symbol not in live_prices: continue
        
        price = live_prices[symbol]
        target_val = target_equity * weight
        if target_val < 50: continue 
        
        target_qty = int(target_val / price)
        current_qty = current_holdings.get(symbol, 0)
        delta_qty = target_qty - current_qty
        
        try:
            if delta_qty > 0:
                trade_client.submit_order(
                    MarketOrderRequest(symbol=symbol, qty=delta_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                )
                trade_log.append(f"🟢 Bought {delta_qty} {symbol}")
            elif delta_qty < 0:
                trade_client.submit_order(
                    MarketOrderRequest(symbol=symbol, qty=abs(delta_qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                )
                trade_log.append(f"📉 Trimmed {abs(delta_qty)} {symbol}")
        except Exception as e:
            print(f"❌ Order Error {symbol}: {e}")

    # 4. REPORTING
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if trade_log:
        with open(LOG_FILE, "a") as f:
            for trade in trade_log: f.write(f"{timestamp},{trade}\n")

    update_performance_tracker(equity)
    
    report = f"✅ **Rebalance Complete**\n💰 Equity: ${equity:,.2f}\n"
    if trade_log: report += "\n".join(trade_log[:10]) 
    else: report += "\n(Portfolio matches target)"
    
    send_telegram(report)
    print("--- COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', action='store_true', help='Run in tracker mode (no trading)')
    parser.add_argument('--force', action='store_true', help='Force trade ignoring calendar')
    args = parser.parse_args()

    if args.track:
        run_tracker_only()
    else:
        # Pass the force flag to the function
        execute_monthly_rebalance(force_trade=args.force)
