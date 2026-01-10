import os
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from decimal import Decimal

# Alpaca Libraries
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
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

# --- PERFORMANCE STATS ENGINE ---
def get_performance_stats(current_equity):
    if not os.path.exists(PERF_FILE): return 0.0, 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(PERF_FILE)
        if df.empty: return 0.0, 0.0, 0.0, 0.0
        
        start_equity = float(df.iloc[0]['Equity'])
        total_chg = current_equity - start_equity
        total_pct = (total_chg / start_equity) * 100 if start_equity else 0.0
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        if df.iloc[-1]['Date'] == today_str and len(df) > 1:
            last_close = float(df.iloc[-2]['Equity'])
        else:
            last_close = float(df.iloc[-1]['Equity'])
            
        daily_chg = current_equity - last_close
        daily_pct = (daily_chg / last_close) * 100 if last_close else 0.0
        
        return daily_chg, daily_pct, total_chg, total_pct
    except: return 0.0, 0.0, 0.0, 0.0

def update_performance_tracker(current_equity):
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{"Date": today_str, "Equity": float(current_equity)}])
    
    if os.path.exists(PERF_FILE):
        df = pd.read_csv(PERF_FILE)
        if df.iloc[-1]['Date'] == today_str:
            df.iloc[-1, df.columns.get_loc("Equity")] = float(current_equity)
        else:
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(PERF_FILE, index=False)
    
    if len(df) < 2: return None
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
    
    plt.switch_backend('Agg') 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(df.index, df['Equity'], color='#0052cc', linewidth=2)
    ax1.set_title("Live Portfolio Performance", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Equity ($)")
    
    ax2.fill_between(df.index, df['Drawdown'], 0, color='#d62728', alpha=0.3)
    ax2.plot(df.index, df['Drawdown'], color='#d62728', linewidth=1)
    ax2.set_title("Drawdown Risk (%)", fontsize=10)
    ax2.set_ylabel("DD %")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(DASHBOARD_IMG)
    plt.close()
    return DASHBOARD_IMG

# --- MODE 1: TRACK ONLY (DAILY) ---
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

# --- MODE 2: FULL REBALANCE (MONTHLY) ---
def execute_monthly_rebalance():
    print("--- 🚀 CHECKING MARKET WINDOW ---")
    
    # 1. SAFETY CHECK: IS MARKET OPEN & SAFE?
    try:
        clock = trade_client.get_clock()
        if not clock.is_open:
            print("❌ Market is closed. Skipping.")
            return

        calendar = trade_client.get_calendar(start=clock.timestamp.date(), end=clock.timestamp.date())[0]
        now_utc = clock.timestamp
        open_utc = calendar.open.replace(tzinfo=timezone.utc) if calendar.open.tzinfo is None else calendar.open.astimezone(timezone.utc)

        time_since_open = (now_utc - open_utc).total_seconds() / 60
        print(f"🕒 Time since open: {time_since_open:.0f} minutes")

        if time_since_open < 45:
            print("⚠️ Market just opened (< 45 mins). Too Volatile. Exiting.")
            return # Exit silently. Will retry next hour.
            
    except Exception as e:
        print(f"⚠️ Clock Check Failed: {e}. Proceeding with caution.")

    # 2. CHECK FOR TRADES TODAY (The 100% Fix)
    # We ask Alpaca: "Did we submit any orders today?"
    try:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        # Filter for ALL orders created after midnight UTC today
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=today_start, limit=5)
        recent_orders = trade_client.get_orders(filter=req)
        
        if len(recent_orders) > 0:
            print(f"✅ Found {len(recent_orders)} orders from today. Already rebalanced. Exiting.")
            return
            
    except Exception as e:
        print(f"⚠️ History Check Failed: {e}")

    # 3. EXECUTION
    print("--- ✅ STARTING REBALANCE EXECUTION ---")
    send_telegram("🚀 **Fortress Bot**\nMarket Safe. Analyzing Data...")
    
    core = MonthlyFortressStrategy()
    sat = RSI2MeanReversionStrategy()
    strat = CompositeStrategy(main_strat=core, sat_strat=sat, main_weight=0.9)
    
    all_tickers = list(set(strat.risk_assets + strat.safe_assets + [strat.market_filter, strat.bond_benchmark]))
    try: prices = get_price_history_alpaca(all_tickers)
    except: return
    if prices.empty: return

    today = prices.index[-1]
    target_portfolio = strat.get_signal(prices, today)
    target_dict = {t: w for t, w in target_portfolio}
    
    acct = trade_client.get_account()
    equity = float(acct.equity)
    positions = trade_client.get_all_positions()
    trade_log = []
    last_known_prices = prices.iloc[-1].to_dict()

    # Sell
    current_holdings = {p.symbol: float(p.qty) for p in positions}
    for p in positions:
        if p.symbol not in target_dict or target_dict[p.symbol] <= 0:
            try:
                trade_client.close_position(p.symbol)
                trade_log.append(f"🔴 Sold All {p.symbol}")
            except: pass
            
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
        
        try:
            if delta_qty > 0:
                trade_client.submit_order(MarketOrderRequest(symbol=symbol, qty=delta_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                trade_log.append(f"🟢 Bought {delta_qty} {symbol}")
            elif delta_qty < 0:
                trade_client.submit_order(MarketOrderRequest(symbol=symbol, qty=abs(delta_qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                trade_log.append(f"📉 Trimmed {abs(delta_qty)} {symbol}")
        except: pass

    # 4. REPORTING
    
    # Save Trade Log (Only persists if you download artifacts later, but useful for logs)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if trade_log:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                for trade in trade_log: f.write(f"{timestamp},{trade}\n")
        except: pass

    d_chg, d_pct, t_chg, t_pct = get_performance_stats(equity)
    img_path = update_performance_tracker(equity)

    report = f"✅ **Rebalance Complete**\n💰 Equity: ${equity:,.2f}\n"
    if trade_log: report += "\n".join(trade_log)
    else: report += "\n(No trades needed, portfolio matches target)"

    report += "\n\n**Targets:**\n" + "\n".join([f"• {s}: {w:.1%}" for s, w in sorted(target_dict.items(), key=lambda x: x[1], reverse=True)])
    t_icon = "🏆" if t_chg >= 0 else "📉"
    report += f"\n\n{t_icon} **Total Return:** {t_pct:+.2f}%"

    if len(report) > 4000:
        send_telegram(report[:4000])
        send_telegram(report[4000:])
    else:
        send_telegram(report)
        
    if img_path and os.path.exists(img_path): send_telegram_photo(img_path)
    print("--- COMPLETE ---")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--track":
        run_tracker_only()
    else:
        execute_monthly_rebalance()