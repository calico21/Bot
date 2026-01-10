# execution_logic.py
from alpaca.trading.client import TradingClient
from datetime import date, timedelta
import pandas as pd

def is_rebalance_day(api_key, secret_key, paper=True):
    """
    Determines if TODAY is the first trading day of the month.
    """
    trading_client = TradingClient(api_key, secret_key, paper=paper)
    
    # 1. Check if Market is Open Today
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("❌ Market is CLOSED today.")
        return False
    
    # 2. Get all trading days for this month
    today = date.today()
    start_date = today.replace(day=1)
    
    # Get calendar for the whole month
    calendar = trading_client.get_calendar(
        start=start_date,
        end=today.replace(day=28) + timedelta(days=4) # Rough end of month
    )
    
    # Extract just the dates
    trading_days = [day.date for day in calendar]
    
    # 3. Check if today is the FIRST one in the list
    first_trading_day = trading_days[0]
    
    if today == first_trading_day:
        print(f"✅ Today ({today}) is the First Trading Day of the month. EXECUTE.")
        return True
    else:
        print(f"💤 Today ({today}) is a trading day, but not the first. (First was {first_trading_day})")
        return False