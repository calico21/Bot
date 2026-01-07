# backtest.py

import pandas as pd
import numpy as np
from quant_db_manager import MarketDB
from strategy import MonthlyFortressStrategy
from reports import (
    print_performance_report,
    print_annual_returns_table
)

START_CAPITAL = 10_000


def backtest_monthly_fortress():
    print("Leyendo desde Base de Datos Local...")

    db = MarketDB()
    strat = MonthlyFortressStrategy()

    tickers = list(set(
        strat.risk_assets +
        strat.safe_assets +
        [strat.market_filter, strat.bond_benchmark]
    ))

    prices = db.load_data(tickers)
    db.close()

    prices = prices.dropna(subset=[strat.market_filter])

    monthly_dates = prices.resample("ME").last().index

    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []

    current_portfolio = {}
    last_prices = {}

    trade_log = []

    for date in monthly_dates:
        if date not in prices.index:
            continue

        target_portfolio = strat.get_signal(prices, date)
        target_dict = {t: w for t, w in target_portfolio}

        # Apply PnL
        if current_portfolio and last_prices:
            port_ret = 0.0
            for t, w in current_portfolio.items():
                if t not in prices.columns:
                    continue
                price_today = prices[t].loc[date]
                price_then = last_prices.get(t)
                if price_then is None or pd.isna(price_today) or pd.isna(price_then):
                    continue
                asset_ret = price_today / price_then
                port_ret += w * (asset_ret - 1.0)
            equity *= (1.0 + port_ret)

        trade_log.append({
            "date": date,
            "portfolio": target_dict,
            "equity": equity
        })

        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}

        equity_curve.append(equity)
        equity_dates.append(date)

    equity_curve = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))

    # Save trade log
    pd.DataFrame(trade_log).to_csv("trades_log.csv", index=False)
    print("Trade log saved to: trades_log.csv")

    # Benchmark SPY
    spy = prices[strat.market_filter].loc[equity_curve.index]
    spy = spy / spy.iloc[0] * START_CAPITAL

    # Performance report
    print_performance_report(equity_curve, spy, name="Monthly Fortress")

    # Annual returns table
    print_annual_returns_table(equity_curve, spy,
                               strategy_name="Monthly Fortress",
                               benchmark_name="SPY")

    # Live equity curve plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Strategy", linewidth=2)
    plt.plot(spy, label="SPY", linewidth=2, alpha=0.7)
    plt.title("Equity Curve", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return equity_curve, spy


if __name__ == "__main__":
    backtest_monthly_fortress()