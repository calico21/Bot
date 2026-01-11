# data_manager.py

import pandas as pd
import os
import yfinance as yf
import warnings

# Import configuration from the Core file we just made
from strategy_core import RISK_ASSETS, SAFE_ASSETS, MARKET_FILTER, BOND_BENCHMARK

DB_FILE = "market_data.parquet"

class MarketDB:
    """
    Manages the local Parquet database for market data.
    """
    def __init__(self):
        self.file_path = DB_FILE

    def save_data(self, df: pd.DataFrame):
        """Saves a dataframe to a parquet file."""
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Save
        df.to_parquet(self.file_path)
        print(f"✅ Data saved to {self.file_path} ({len(df)} rows)")

    def load_data(self, tickers=None):
        """Loads data, optionally filtering by columns (tickers)."""
        if not os.path.exists(self.file_path):
            print("⚠️ DB empty. Run this script directly to sync data: 'python data_manager.py'")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.file_path)
            
            # Filter for requested tickers if provided
            if tickers:
                # Find which tickers are actually in the file
                valid_tickers = [t for t in tickers if t in df.columns]
                
                # Warn if some are missing
                missing = [t for t in tickers if t not in df.columns]
                if missing:
                    print(f"⚠️ Warning: Missing data for {missing}")
                    
                df = df[valid_tickers]
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading DB: {e}")
            return pd.DataFrame()

    def close(self):
        # Placeholder for compatibility if we switch to SQL later
        pass

def sync_market_data():
    """
    Downloads full history for all assets defined in strategy_core.py
    """
    print("🔄 SYNC: Starting Global Market Data Update (Full History)...")
    
    # 1. Dynamic Ticker List from Strategy Core
    # We add VIXY just in case, though not strictly used in the logic provided
    tickers = list(set(RISK_ASSETS + SAFE_ASSETS + [MARKET_FILTER, BOND_BENCHMARK, 'VIXY']))
    
    print(f"📊 Tracking {len(tickers)} Assets...")

    # 2. Bulk Download (FROM 2000)
    try:
        # Threads enabled for speed
        data = yf.download(tickers, start="2000-01-01", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"❌ Download Error: {e}")
        return

    # 3. Process & Save to DB
    db = MarketDB()
    formatted_data = {}
    
    # Handle single ticker vs multi-ticker structure from yfinance
    if len(tickers) == 1:
        ticker = tickers[0]
        # If it's a Series or DataFrame, normalize it
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
             formatted_data[ticker] = data['Close']
        else:
             formatted_data[ticker] = data
    else:
        for ticker in tickers:
            try:
                # yfinance returns a MultiIndex (Ticker, OHLCV)
                if ticker in data.columns.levels[0]:
                    series = data[ticker]['Close'].dropna()
                    if not series.empty:
                        formatted_data[ticker] = series
            except Exception:
                pass

    if formatted_data:
        df_final = pd.DataFrame(formatted_data)
        
        # Ensure timezone naive for Parquet compatibility
        if df_final.index.tz is not None:
            df_final.index = df_final.index.tz_localize(None)
            
        db.save_data(df_final)
        print(f"✅ Success: Database updated with {len(formatted_data)} assets (from 2000).")
    else:
        print("⚠️ Warning: No data found.")

    db.close()

if __name__ == "__main__":
    # If run directly, perform the sync
    sync_market_data()