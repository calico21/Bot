# research.py

import itertools
import pandas as pd
import numpy as np

from quant_db_manager import MarketDB
from strategy import FortressSubStrategy, MonthlyFortressStrategy
from reports import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    print_performance_report,
)

START_CAPITAL = 10_000


# ============================================================
# Backtest for a single sub-strategy (not the ensemble)
# ============================================================

def run_substrategy_backtest(prices: pd.DataFrame, substrat: FortressSubStrategy) -> pd.Series:
    monthly_dates = prices.resample("ME").last().index

    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []

    current_portfolio = {}
    last_prices = {}

    for date in monthly_dates:
        if date not in prices.index:
            continue

        target_portfolio = substrat.get_signal(prices, date)
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

        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}

        equity_curve.append(equity)
        equity_dates.append(date)

    return pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))


# ============================================================
# Walk-Forward Optimization (WFO)
# ============================================================

def walk_forward_optimization():
    # Load data
    db = MarketDB()
    base = MonthlyFortressStrategy()
    tickers = list(set(
        base.risk_assets +
        base.safe_assets +
        [base.market_filter, base.bond_benchmark]
    ))
    prices = db.load_data(tickers)
    db.close()

    prices = prices.dropna(subset=[base.market_filter])

    # Full grid of sub-strategies
    momentum_windows = [63, 84, 126, 189]
    vol_windows = [21, 63, 126]
    top_ns = [1, 2]  # Option B

    substrategies = []
    for m, v, t in itertools.product(momentum_windows, vol_windows, top_ns):
        name = f"m{m}_v{v}_top{t}"
        substrategies.append(
            FortressSubStrategy(
                name=name,
                momentum_window=m,
                vol_window=v,
                top_n=t,
            )
        )

    # WFO config
    train_years = 5
    test_years = 2

    start_date = prices.index[0]
    final_date = prices.index[-1]

    wf_segments = []

    current_start = start_date
    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if train_end >= final_date:
            break

        train_prices = prices.loc[current_start:train_end]
        test_prices = prices.loc[train_end:test_end]

        if len(train_prices) < 252 * 3:
            break

        print(f"\nWFO Window {current_start.date()} -> {train_end.date()} train,"
              f" {train_end.date()} -> {test_end.date()} test")

        # Evaluate all sub-strategies on training window
        results = []
        for s in substrategies:
            eq = run_substrategy_backtest(train_prices, s)
            cagr = calculate_cagr(eq)
            results.append((s, cagr))

        # Select all positive-CAGR sub-strategies
        selected = [s for s, c in results if c > 0]

        print(f"  Selected {len(selected)} sub-strategies with positive CAGR")

        if not selected:
            print("  No positive sub-strategies — using all slow ones as fallback")
            selected = [s for s in substrategies if "189" in s.name]

        # Build ensemble for test window
        test_equity = run_ensemble_test(test_prices, selected)
        wf_segments.append(test_equity)

        current_start = test_end

    if not wf_segments:
        print("No WFO segments produced.")
        return

    wf_equity = pd.concat(wf_segments).sort_index()
    wf_equity.name = "WFO_equity"

    print("\n=== WFO OUT-OF-SAMPLE PERFORMANCE ===")
    print_performance_report(wf_equity, None, name="Monthly Fortress (WFO OOS)")

    return wf_equity


# ============================================================
# Ensemble test for selected sub-strategies
# ============================================================

def run_ensemble_test(prices: pd.DataFrame, substrategies):
    monthly_dates = prices.resample("ME").last().index

    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []

    current_portfolio = {}
    last_prices = {}

    for date in monthly_dates:
        if date not in prices.index:
            continue

        # Blend signals from all selected sub-strategies
        agg = {}
        for s in substrategies:
            port = s.get_signal(prices, date)
            for t, w in port:
                agg[t] = agg.get(t, 0) + w

        # Normalize
        total = sum(agg.values())
        if total <= 0:
            target_dict = {"SHV": 1.0}
        else:
            target_dict = {t: w / total for t, w in agg.items()}

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

        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}

        equity_curve.append(equity)
        equity_dates.append(date)

    return pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))


# ============================================================
# Monte Carlo
# ============================================================

def monte_carlo_from_equity(equity_curve: pd.Series, n_sims=1000, seed=42):
    if seed is not None:
        np.random.seed(seed)

    rets = equity_curve.pct_change().dropna().values
    n = len(rets)

    cagr_list = []
    mdd_list = []
    sharpe_list = []

    for _ in range(n_sims):
        sim_rets = np.random.choice(rets, size=n, replace=True)
        sim_equity = [equity_curve.iloc[0]]
        for r in sim_rets:
            sim_equity.append(sim_equity[-1] * (1 + r))
        sim_series = pd.Series(sim_equity, index=equity_curve.index[:len(sim_equity)])

        cagr_list.append(calculate_cagr(sim_series))
        mdd_list.append(calculate_max_drawdown(sim_series))
        sharpe_list.append(calculate_sharpe(sim_series.pct_change()))

    print("\n=== MONTE CARLO SUMMARY ===")
    print(f"CAGR median:       {np.median(cagr_list):.2%}")
    print(f"CAGR 10–90%:       {np.percentile(cagr_list, 10):.2%} – {np.percentile(cagr_list, 90):.2%}")
    print(f"MaxDD median:      {np.median(mdd_list):.2%}")
    print(f"MaxDD 10–90%:      {np.percentile(mdd_list, 10):.2%} – {np.percentile(mdd_list, 90):.2%}")
    print(f"Sharpe median:     {np.nanmedian(sharpe_list):.2f}")
    print(f"Sharpe 10–90%:     {np.nanpercentile(sharpe_list, 10):.2f} – {np.nanpercentile(sharpe_list, 90):.2f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    from backtest import backtest_monthly_fortress

    print("\n=== BASELINE BACKTEST ===")
    base_equity, _ = backtest_monthly_fortress()

    print("\n=== WFO OUT-OF-SAMPLE ===")
    wf_equity = walk_forward_optimization()

    print("\n=== MONTE CARLO ON BASELINE EQUITY ===")
    monte_carlo_from_equity(base_equity, n_sims=1000)