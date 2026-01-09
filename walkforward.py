# walkforward.py

import itertools
import pandas as pd
import numpy as np

import strategy as strat_mod
from strategy import MonthlyFortressStrategy
from quant_db_manager import MarketDB
from reports import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
)
from research import run_backtest


# ---------------------------------------------------------
# 1. Load full price data once
# ---------------------------------------------------------
def load_full_data():
    db = MarketDB()
    base_strat = MonthlyFortressStrategy()

    tickers = list(set(
        base_strat.risk_assets +
        base_strat.safe_assets +
        [base_strat.market_filter, base_strat.bond_benchmark]
    ))

    prices = db.load_data(tickers).bfill()
    db.close()

    try:
        prices = prices.loc["2000-01-01":]
    except Exception:
        pass

    return prices


# ---------------------------------------------------------
# 2. Helper: create a strategy with given params
# ---------------------------------------------------------
def make_strategy(ANCHOR_WEIGHT, CRASH_THRESHOLD, TARGET_VOL):
    strat_mod.ANCHOR_WEIGHT = ANCHOR_WEIGHT
    strat_mod.CRASH_THRESHOLD = CRASH_THRESHOLD
    strat_mod.TARGET_VOL = TARGET_VOL

    strat_mod.MOMENTUM_WINDOWS = (63, 126, 189)
    strat_mod.TOP_N = 1
    strat_mod.MAX_LEVERAGE = 1.0

    return MonthlyFortressStrategy()


# ---------------------------------------------------------
# 3. Mini-sweep on a given train window (FAST)
# ---------------------------------------------------------
def sweep_on_window(prices_train):

    anchor_grid = [0.20, 0.25]
    crash_grid  = [-0.07, -0.10]
    tvol_grid   = [0.15]

    rows = []

    for (ANCHOR_WEIGHT, CRASH_THRESHOLD, TARGET_VOL) in itertools.product(
        anchor_grid, crash_grid, tvol_grid):

        print(f"[WFO] Testing params: aw={ANCHOR_WEIGHT}, ct={CRASH_THRESHOLD}, tv={TARGET_VOL}")

        strat = make_strategy(ANCHOR_WEIGHT, CRASH_THRESHOLD, TARGET_VOL)

        # === DAILY PRICES, MONTHLY SIGNALS ===
        equity_train, _, _ = run_backtest(prices_train, strat)

        rets_train = equity_train.pct_change(fill_method=None).dropna()

        sharpe = calculate_sharpe(rets_train)
        sortino = calculate_sortino(rets_train)
        cagr = calculate_cagr(equity_train)
        maxdd = calculate_max_drawdown(equity_train)

        rows.append({
            "ANCHOR_WEIGHT": ANCHOR_WEIGHT,
            "CRASH_THRESHOLD": CRASH_THRESHOLD,
            "TARGET_VOL": TARGET_VOL,
            "sharpe": sharpe,
            "sortino": sortino,
            "cagr": cagr,
            "maxdd": maxdd,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["sharpe", "maxdd"], ascending=[False, False]).reset_index(drop=True)

    best = df.iloc[0]
    print("\n[WFO] Best params on train window:")
    print(best)

    return best


# ---------------------------------------------------------
# 4. Rolling walk-forward engine (FAST + CORRECT)
# ---------------------------------------------------------
def run_walkforward(train_years=10, test_years=2):

    prices = load_full_data()
    all_dates = prices.index

    train_days = int(train_years * 252)
    test_days = int(test_years * 252)

    oos_segments = []
    current_start_idx = 0
    oos_last_equity = None  # track last OOS equity level

    while True:
        train_start = current_start_idx
        train_end = train_start + train_days
        test_end = train_end + test_days

        if test_end >= len(all_dates):
            break

        train_idx = all_dates[train_start:train_end]
        test_idx = all_dates[train_end:test_end]

        prices_train = prices.loc[train_idx]
        prices_test = prices.loc[test_idx]

        print("\n" + "="*60)
        print(f"[WFO] Train: {train_idx[0].date()} → {train_idx[-1].date()}")
        print(f"[WFO] Test:  {test_idx[0].date()} → {test_idx[-1].date()}")

        # 1) Sweep on train window
        best = sweep_on_window(prices_train)

        # 2) Build strategy with best params
        strat = make_strategy(
            best["ANCHOR_WEIGHT"],
            best["CRASH_THRESHOLD"],
            best["TARGET_VOL"],
        )

        # 3) Run backtest on TEST window (starts at 100k internally)
        equity_test, _, _ = run_backtest(prices_test, strat)

        # 4) Chain segments: rescale so each test window starts at last OOS equity
        if oos_last_equity is None:
            # first segment: keep as is
            chained = equity_test.copy()
        else:
            # scale so first value matches previous last equity
            factor = oos_last_equity / equity_test.iloc[0]
            chained = equity_test * factor

        oos_last_equity = chained.iloc[-1]
        oos_segments.append(chained)

        current_start_idx += test_days

    # Stitch OOS segments
    oos_equity = pd.concat(oos_segments)
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="first")]

    return oos_equity



# ---------------------------------------------------------
# 5. Compare OOS vs SPY
# ---------------------------------------------------------
def run_walkforward_with_report():

    oos_equity = run_walkforward()

    db = MarketDB()
    spy = db.load_data(["SPY"])["SPY"].bfill()
    db.close()

    # Align SPY to the same frequency as the strategy (monthly)
    spy = spy.loc[oos_equity.index.min():oos_equity.index.max()]
    spy_equity = (spy / spy.iloc[0]) * oos_equity.iloc[0]

    # Resample SPY to month-end to match oos_equity
    spy_equity_m = spy_equity.resample("ME").last()
    spy_equity_m = spy_equity_m.loc[oos_equity.index]  # align indices

    strat_rets = oos_equity.pct_change(fill_method=None).dropna()
    spy_rets = spy_equity_m.pct_change(fill_method=None).dropna()


    print("\n" + "="*60)
    print("WALK-FORWARD OUT-OF-SAMPLE PERFORMANCE")
    print("="*60)

    print("\n--- Strategy OOS ---")
    print(f"CAGR:          {calculate_cagr(oos_equity):.2%}")
    print(f"Max Drawdown:  {calculate_max_drawdown(oos_equity):.2%}")
    print(f"Sharpe:        {calculate_sharpe(strat_rets):.2f}")
    print(f"Sortino:       {calculate_sortino(strat_rets):.2f}")

    print("\n--- SPY (Same Period) ---")
    print(f"CAGR:          {calculate_cagr(spy_equity_m):.2%}")
    print(f"Max Drawdown:  {calculate_max_drawdown(spy_equity_m):.2%}")
    print(f"Sharpe:        {calculate_sharpe(spy_rets):.2f}")
    print(f"Sortino:       {calculate_sortino(spy_rets):.2f}")

    oos_equity.to_csv("walkforward_oos_equity.csv")
    print("\nSaved OOS equity curve to walkforward_oos_equity.csv")


if __name__ == "__main__":
    run_walkforward_with_report()
