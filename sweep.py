# sweep.py

import itertools
import pandas as pd
import strategy as strat_mod
from strategy import MonthlyFortressStrategy
from research import run_full_strategy_backtest
from reports import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
)

def make_strategy(ANCHOR_WEIGHT, CRASH_THRESHOLD, TARGET_VOL,
                  momentum_set, top_n, MAX_LEVERAGE):

    # Override global parameters in strategy.py
    strat_mod.ANCHOR_WEIGHT = ANCHOR_WEIGHT
    strat_mod.CRASH_THRESHOLD = CRASH_THRESHOLD
    strat_mod.TARGET_VOL = TARGET_VOL
    strat_mod.MAX_LEVERAGE = MAX_LEVERAGE

    # Override momentum windows + top_n inside the strategy class
    strat_mod.MOMENTUM_WINDOWS = momentum_set
    strat_mod.TOP_N = top_n

    return MonthlyFortressStrategy()

def run_param_sweep():

    anchor_grid = [0.15, 0.20, 0.25]
    crash_grid  = [-0.07, -0.10]
    tvol_grid   = [0.15]

    momentum_grid = [
        (42, 84, 126),
        (63, 126, 189),
        (84, 168, 252)
    ]

    topn_grid = [1, 2, 3]

    maxlev_grid = [1.0, 1.25, 1.5]

    rows = []

    for (ANCHOR_WEIGHT, CRASH_THRESHOLD, TARGET_VOL,
         momentum_set, top_n, MAX_LEVERAGE) in itertools.product(
            anchor_grid, crash_grid, tvol_grid,
            momentum_grid, topn_grid, maxlev_grid):

        strat = make_strategy(
            ANCHOR_WEIGHT, CRASH_THRESHOLD, TARGET_VOL,
            momentum_set, top_n, MAX_LEVERAGE
        )

        equity, turnover, _, _ = run_full_strategy_backtest(strat_override=strat)
        rets = equity.pct_change().dropna()

        row = {
            "ANCHOR_WEIGHT": ANCHOR_WEIGHT,
            "CRASH_THRESHOLD": CRASH_THRESHOLD,
            "TARGET_VOL": TARGET_VOL,
            "momentum_set": momentum_set,
            "top_n": top_n,
            "MAX_LEVERAGE": MAX_LEVERAGE,
            "cagr": calculate_cagr(equity),
            "maxdd": calculate_max_drawdown(equity),
            "sharpe": calculate_sharpe(rets),
            "sortino": calculate_sortino(rets),
            "avg_turnover": turnover.mean(),
            "max_turnover": turnover.max(),
        }

        rows.append(row)

        print(
            f"aw={ANCHOR_WEIGHT}, ct={CRASH_THRESHOLD}, tv={TARGET_VOL}, "
            f"mom={momentum_set}, topN={top_n}, lev={MAX_LEVERAGE} "
            f"→ Sharpe {row['sharpe']:.2f}, MaxDD {row['maxdd']:.2%}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("param_sweep_expanded.csv", index=False)
    print("\nSaved expanded sweep to param_sweep_expanded.csv")
    return df


if __name__ == "__main__":
    run_param_sweep()
