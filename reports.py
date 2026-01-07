# reports.py

import numpy as np
import pandas as pd


# ============================
# Performance Metrics
# ============================

def calculate_cagr(equity_curve: pd.Series) -> float:
    if equity_curve is None or len(equity_curve) < 2:
        return np.nan
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return np.nan
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1
    return dd.min()


def calculate_sharpe(returns: pd.Series, risk_free_rate=0.0) -> float:
    """
    Sharpe ratio for MONTHLY returns.
    Annualization uses sqrt(12).
    """
    if returns is None or returns.empty:
        return np.nan

    excess = returns - risk_free_rate / 12
    if excess.std() == 0:
        return np.nan

    return np.sqrt(12) * excess.mean() / excess.std()


def calculate_sortino(returns: pd.Series, risk_free_rate=0.0) -> float:
    """
    Sortino ratio for MONTHLY returns.
    Annualization uses sqrt(12).
    """
    if returns is None or returns.empty:
        return np.nan

    excess = returns - risk_free_rate / 12
    downside = excess[excess < 0]

    if downside.std() == 0:
        return np.nan

    return np.sqrt(12) * excess.mean() / downside.std()


# ============================
# Annual Returns Table
# ============================

def print_annual_returns_table(strategy_curve: pd.Series,
                               benchmark_curve: pd.Series,
                               strategy_name="Strategy",
                               benchmark_name="SPY"):
    """
    Prints a side-by-side table of annual returns for strategy vs benchmark.
    """

    strat_yearly = strategy_curve.resample("Y").last()
    bench_yearly = benchmark_curve.resample("Y").last()

    strat_ret = strat_yearly.pct_change().dropna()
    bench_ret = bench_yearly.pct_change().dropna()

    df = pd.DataFrame({
        strategy_name: strat_ret,
        benchmark_name: bench_ret
    })

    df.index = df.index.year

    print("\n=== ANNUAL RETURNS (Side-by-Side) ===")
    print(df.to_string(float_format=lambda x: f"{x:.2%}"))


# ============================
# Full Performance Report
# ============================

def print_performance_report(equity_curve: pd.Series,
                             benchmark_curve: pd.Series = None,
                             name="Strategy"):
    if equity_curve is None or equity_curve.empty:
        print("No equity curve to report.")
        return

    rets = equity_curve.pct_change().dropna()

    print("\n=== PERFORMANCE REPORT ===")
    print(f"Strategy: {name}")
    print(f"Period: {equity_curve.index[0].date()} → {equity_curve.index[-1].date()}")

    print("\n--- Strategy Metrics ---")
    print(f"CAGR:           {calculate_cagr(equity_curve):.2%}")
    print(f"Max Drawdown:   {calculate_max_drawdown(equity_curve):.2%}")
    print(f"Sharpe Ratio:   {calculate_sharpe(rets):.2f}")
    print(f"Sortino Ratio:  {calculate_sortino(rets):.2f}")
    print(f"Final Equity:   {equity_curve.iloc[-1]:,.2f}")

    if benchmark_curve is not None:
        bench_rets = benchmark_curve.pct_change().dropna()

        print("\n--- Benchmark (SPY) ---")
        print(f"CAGR:           {calculate_cagr(benchmark_curve):.2%}")
        print(f"Max Drawdown:   {calculate_max_drawdown(benchmark_curve):.2%}")
        print(f"Sharpe Ratio:   {calculate_sharpe(bench_rets):.2f}")
        print(f"Sortino Ratio:  {calculate_sortino(bench_rets):.2f}")
def print_annual_returns_table(strategy_curve: pd.Series,
                               benchmark_curve: pd.Series,
                               strategy_name="Strategy",
                               benchmark_name="SPY"):
    """
    Prints a side-by-side table of annual returns for strategy vs benchmark.
    """

    strat_yearly = strategy_curve.resample("Y").last()
    bench_yearly = benchmark_curve.resample("Y").last()

    strat_ret = strat_yearly.pct_change().dropna()
    bench_ret = bench_yearly.pct_change().dropna()

    df = pd.DataFrame({
        strategy_name: strat_ret,
        benchmark_name: bench_ret
    })

    df.index = df.index.year

    print("\n=== ANNUAL RETURNS (Side-by-Side) ===")
    print(df.to_string(float_format=lambda x: f"{x:.2%}"))