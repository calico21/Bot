import pandas as pd
import os

DB_FILE = "market_data.parquet"

class MarketDB:
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
            print("⚠️ DB empty. Run sync_data.py first.")
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
        pass