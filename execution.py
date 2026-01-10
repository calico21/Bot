import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from decimal import Decimal

# Alpaca Libraries
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Import Strategy
from strategy import MonthlyFortressStrategy, RSI2MeanReversionStrategy, CompositeStrategy

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

trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# --- TELEGRAM FUNCTIONS ---
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

def send_telegram_photo(image_path):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(image_path, 'rb') as photo:
            payload = {'chat_id': TELEGRAM_CHAT_ID}
            files = {'photo': photo}
            requests.post(url, data=payload, files=files)
    except Exception as e:
        print(f"Failed to send photo: {e}")

# --- DATA FUNCTIONS ---
def get_price_history_alpaca(tickers, days=800):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(req)
        if bars.df.empty: return pd.DataFrame()
        
        df = bars.df.reset_index()
        pivot = df.pivot(index="timestamp", columns="symbol", values="close")
        pivot.index = pivot.index.tz_convert(None) 
        return pivot
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return pd.DataFrame()

# --- TRACKER & DASHBOARD GENERATOR ---
def update_performance_tracker(current_equity):
    """
    1. Appends today's equity to live_performance.csv
    2. Generates a PNG dashboard with Equity Curve + Drawdown
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Update CSV
    new_row = pd.DataFrame([{"Date": today_str, "Equity": float(current_equity)}])
    
    if os.path.exists(PERF_FILE):
        df = pd.read_csv(PERF_FILE)
        # Avoid duplicate entries for the same day (overwrite last)
        if df.iloc[-1]['Date'] == today_str:
            df.iloc[-1, df.columns.get_loc("Equity")] = float(current_equity)
        else:
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
        
    df.to_csv(PERF_FILE, index=False)
    
    # 2. Generate Graph
    if len(df) < 2: return # Need at least 2 points to graph
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Calculate Drawdown
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
    
    # Plotting
    plt.switch_backend('Agg') # Server-safe backend (no GUI needed)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Equity Curve
    ax1.plot(df.index, df['Equity'], color='#0052cc', linewidth=2)
    ax1.set_title("Live Portfolio Performance", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Equity ($)")
    
    # Drawdown Area
    ax2.fill_between(df.index, df['Drawdown'], 0, color='#d62728', alpha=0.3)
    ax2.plot(df.index, df['Drawdown'], color='#d62728', linewidth=1)
    ax2.set_title("Drawdown Risk (%)", fontsize=10)
    ax2.set_ylabel("DD %")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    # Format Dates
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(DASHBOARD_IMG)
    plt.close()
    
    return DASHBOARD_IMG

# --- MAIN EXECUTION ---
def execute_monthly_rebalance():
    print("--- 🚀 STARTING MONTHLY REBALANCE ---")
    send_telegram("🚀 **Fortress Centurion Bot**\nAnalyzing market data...")
    
    core = MonthlyFortressStrategy()
    sat = RSI2MeanReversionStrategy()
    strat = CompositeStrategy(main_strat=core, sat_strat=sat, main_weight=0.9)
    
    # 1. Get Data
    all_tickers = list(set(strat.risk_assets + strat.safe_assets + [strat.market_filter, strat.bond_benchmark]))
    try:
        prices = get_price_history_alpaca(all_tickers)
    except Exception as e:
        send_telegram(f"❌ Data Error: {e}")
        return

    if prices.empty: 
        print("No data found.")
        return

    # 2. Signal
    today = prices.index[-1]
    print(f"Signal Date: {today}")
    
    target_portfolio = strat.get_signal(prices, today)
    target_dict = {t: w for t, w in target_portfolio}
    
    # 3. Execution
    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    trade_log = []
    
    last_known_prices = prices.iloc[-1].to_dict()

    # Sell
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    for p in positions:
        if p.symbol not in target_dict or target_dict[p.symbol] <= 0:
            print(f"Closing {p.symbol}...")
            try:
                trade_client.close_position(p.symbol)
                trade_log.append(f"🔴 Sold All {p.symbol}")
            except Exception as e:
                print(f"Error closing {p.symbol}: {e}")
            
    # Buy
    target_equity = float(trade_client.get_account().equity) * 0.95
    for symbol, weight in target_dict.items():
        price = last_known_prices.get(symbol)
        if not price: continue
            
        target_val = target_equity * weight
        if target_val < 500: continue
        
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

    # 4. REPORTING & TRACKING
    
    # A) Save Trade Log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if trade_log:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                for trade in trade_log:
                    f.write(f"{timestamp},{trade}\n")
            print(f"✅ Trade log saved to {LOG_FILE}")
        except: pass

    # B) Update Tracker & Generate Graph
    print("📊 Updating Performance Tracker...")
    img_path = update_performance_tracker(equity)

    # C) Send Telegram
    report = f"✅ **Rebalance Complete**\n💰 Equity: ${equity:,.2f}\n"
    if trade_log:
        report += "\n".join(trade_log)
    else:
        report += "\n(No trades needed, portfolio matches target)"

    report += "\n\n**Current Targets:**\n" + "\n".join([f"• {s}: {w:.1%}" for s, w in sorted(target_dict.items(), key=lambda x: x[1], reverse=True)])
    
    if len(report) > 4000:
        send_telegram(report[:4000])
        send_telegram(report[4000:])
    else:
        send_telegram(report)
        
    # Send the Graph if it exists
    if img_path and os.path.exists(img_path):
        send_telegram_photo(img_path)

    print("--- COMPLETE ---")

if __name__ == "__main__":
    execute_monthly_rebalance()