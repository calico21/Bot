# execution.py

import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from strategy import MonthlyFortressStrategy

load_dotenv()
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
PAPER_MODE = True

trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_price_history_alpaca(tickers, days=400) -> pd.DataFrame:
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start_dt,
        end=end_dt,
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df.reset_index()
    pivot = df.pivot(index="timestamp", columns="symbol", values="close")
    pivot = pivot.sort_index()
    return pivot


def get_latest_prices(tickers):
    from alpaca.data.requests import StockLatestQuoteRequest
    quotes = data_client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=tickers)
    )
    out = {}
    for t in tickers:
        q = quotes[t]
        price = q.ask_price or q.bid_price or q.midpoint
        out[t] = float(price)
    return out


def run_monthly_rebalance():
    strat = MonthlyFortressStrategy()
    universe = list(set(strat.risk_assets + strat.safe_assets + [strat.market_filter, strat.bond_benchmark]))

    prices = get_price_history_alpaca(universe, days=500)
    today = prices.index[-1]

    target_portfolio = strat.get_signal(prices, today)  # list[(ticker, weight)]
    target_dict = {t: w for t, w in target_portfolio}

    print(f"Target portfolio for {today.date()}: {target_dict}")

    # Get account info
    acct = trade_client.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)

    # Get current positions
    positions = trade_client.get_all_positions()
    current = {p.symbol: float(p.market_value) / equity for p in positions}

    print(f"Current portfolio: {current}")

    # Simple rebalance: close everything not in target
    for p in positions:
        if p.symbol not in target_dict:
            print(f"Closing position: {p.symbol}")
            trade_client.close_position(p.symbol)

    # Get latest prices for sizing
    latest_prices = get_latest_prices(list(target_dict.keys()))
    buying_power = float(trade_client.get_account().cash)

    # Allocate based on target weights and available cash
    for symbol, w in target_dict.items():
        alloc_value = equity * w
        price = latest_prices[symbol]
        qty = int(alloc_value / price)
        if qty <= 0:
            continue

        print(f"Placing market buy: {symbol} x {qty}")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trade_client.submit_order(order)


if __name__ == "__main__":
    run_monthly_rebalance()