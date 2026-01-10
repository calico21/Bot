# backtest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quant_db_manager import MarketDB

from strategy import MonthlyFortressStrategy, RSI2MeanReversionStrategy, CompositeStrategy, CarryEngine

from reports import (
    print_performance_report,
    print_annual_returns_table
)

START_CAPITAL = 100_000

def backtest_monthly_fortress():
    print("Reading from Local Database...")

    db = MarketDB()

    # ==========================================
    # STRATEGY INITIALIZATION
    # ==========================================
    core_strat = MonthlyFortressStrategy()
    satellite_strat = RSI2MeanReversionStrategy(
        ticker='SPY',
        rsi_period=2,
        rsi_entry=10,
        trend_ma=200,
        exit_ma=5
    )

    carry_engine = CarryEngine(top_n=5)

    strat = CompositeStrategy(
        main_strat=core_strat,
        sat_strat=satellite_strat,
        carry_engine=carry_engine,
        main_weight=0.70,
        sat_weight=0.10,
        carry_weight=0.20
)


    # ==========================================
    # LOAD DATA
    # ==========================================
    tickers = list(set(
        strat.risk_assets +
        strat.safe_assets +
        [strat.market_filter, strat.bond_benchmark]
    ))

    prices = db.load_data(tickers)
    db.close()

    prices = prices.dropna(subset=[strat.market_filter])
    monthly_dates = prices.resample("ME").last().index

    # ==========================================
    # BACKTEST STATE
    # ==========================================
    equity = START_CAPITAL
    equity_peak = equity
    equity_curve = []
    equity_dates = []

    current_portfolio = {}
    prev_weights = {}          # NEW
    last_prices = {}
    trade_log = []

    # ==========================================
    # BACKTEST LOOP
    # ==========================================
    for date in monthly_dates:
        if date not in prices.index:
            continue

        # --- 1. Compute current drawdown ---
        current_dd = (equity / equity_peak) - 1.0

        # --- 2. Get strategy signal (v15 requires DD + prev weights) ---
        target_portfolio = strat.get_signal(
            prices,
            date,
            current_dd=current_dd,
            prev_weights=prev_weights
        )
        target_dict = {t: w for t, w in target_portfolio}

        # --- 3. Apply PnL from previous month ---
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
            equity_peak = max(equity_peak, equity)

        # --- 4. Record trade ---
        trade_log.append({
            "date": date,
            "portfolio": target_dict,
            "equity": equity
        })

        # --- 5. Roll forward ---
        prev_weights = current_portfolio.copy() if current_portfolio else {}
        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}

        equity_curve.append(equity)
        equity_dates.append(date)

    # ==========================================
    # REPORTING
    # ==========================================
    equity_curve = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))

    pd.DataFrame(trade_log).to_csv("trades_log.csv", index=False)
    print("Trade log saved to: trades_log.csv")

    if strat.market_filter in prices.columns:
        spy = prices[strat.market_filter].loc[equity_curve.index]
        spy = spy / spy.iloc[0] * START_CAPITAL
    else:
        print("Warning: Benchmark ticker not found in prices.")
        spy = pd.Series(START_CAPITAL, index=equity_curve.index)

    print_performance_report(equity_curve, spy, name="Fortress + RSI Blend")
    print_annual_returns_table(
        equity_curve, spy,
        strategy_name="Fortress + RSI",
        benchmark_name="SPY"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Composite Strategy (80/20)", linewidth=2)
    plt.plot(spy, label="SPY Benchmark", linewidth=2, alpha=0.7, color='grey')
    plt.title("Equity Curve: Fortress + RSI Mean Reversion", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return equity_curve, spy


if __name__ == "__main__":
    backtest_monthly_fortress()
