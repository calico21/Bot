# setup_data.py

from quant_db_manager import QuantDBManager

def initialize_database():
    print("--- 📥 INITIALIZING MARKET DATA ---")
    
    # 1. Connect to DB
    db = QuantDBManager("market_data.db")
    
    # 2. Define all tickers used in the Strategy
    # We must include EVERYTHING: Risk assets, Safe assets, and Benchmarks
    all_tickers = [
        # Risk Assets (Attack)
        'XLK', 'SMH', 'XLF', 'XLV', 'XLI', 'XLY', 'QQQ', 'XLE',
        
        # Safe Assets (Defense)
        'IEF', 'SHV', 'GLD', 'DBC',
        
        # Benchmarks / Filters
        'SPY' 
    ]
    
    # 3. Download and Save
    print(f"Downloading full history for: {all_tickers}")
    db.update_market_data(all_tickers)
    
    print("\n✅ Database populated successfully!")
    db.close()

if __name__ == "__main__":
    initialize_database()