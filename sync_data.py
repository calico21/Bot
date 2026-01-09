import yfinance as yf
import pandas as pd
from quant_db_manager import MarketDB
from strategy import RISK_ASSETS, SAFE_ASSETS, MARKET_FILTER, BOND_BENCHMARK

def sync_market_data():
    print("🔄 SYNC: Starting Global Market Data Update (Full History)...")
    
    # 1. Dynamic Ticker List
    tickers = list(set(RISK_ASSETS + SAFE_ASSETS + [MARKET_FILTER, BOND_BENCHMARK, 'VIXY']))
    
    print(f"📊 Tracking {len(tickers)} Assets...")

    # 2. Bulk Download (FROM 2000)
    # FIXED: Changed period="5y" to start="2000-01-01"
    try:
        data = yf.download(tickers, start="2000-01-01", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"❌ Download Error: {e}")
        return

    # 3. Process & Save to DB
    db = MarketDB()
    formatted_data = {}
    
    if len(tickers) == 1:
        ticker = tickers[0]
        formatted_data[ticker] = data['Close']
    else:
        for ticker in tickers:
            try:
                if ticker in data.columns.levels[0]:
                    series = data[ticker]['Close'].dropna()
                    if not series.empty:
                        formatted_data[ticker] = series
            except Exception:
                pass

    if formatted_data:
        df_final = pd.DataFrame(formatted_data)
        db.save_data(df_final)
        print(f"✅ Success: Database updated with {len(formatted_data)} assets (from 2000).")
    else:
        print("⚠️ Warning: No data found.")

    db.close()

if __name__ == "__main__":
    sync_market_data()