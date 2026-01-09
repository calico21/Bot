# research.py

import itertools
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from quant_db_manager import MarketDB
from strategy import MonthlyFortressStrategy
from reports import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    print_performance_report,
)

# --- CONFIG ---
START_CAPITAL = 100_000
SLIPPAGE_PER_TURNOVER = 0.001  # 10 bps per 100% turnover

# ============================================================
# Helper: Run the MAIN Strategy (The Baseline)
# ============================================================
def run_full_strategy_backtest(strat_override=None):
    db = MarketDB()
    strat = strat_override or MonthlyFortressStrategy()
    
    tickers = list(set(
        strat.risk_assets + 
        strat.safe_assets + 
        [strat.market_filter, strat.bond_benchmark]
    ))
    
    prices = db.load_data(tickers).bfill()
    db.close()

    try:
        prices = prices.loc["2000-01-01":]
    except Exception:
        pass

    monthly_dates = prices.resample("ME").last().index
    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []
    
    current_portfolio = {}  # weights
    last_prices = {}
    turnover_series = []

    for date in monthly_dates:
        if date not in prices.index:
            continue
        
        # Get target weights from strategy
        target_list = strat.get_signal(prices, date)
        target_dict = {t: w for t, w in target_list}

        # Apply portfolio return from last month to this date
        if current_portfolio and last_prices:
            port_ret = 0.0
            for t, w in current_portfolio.items():
                if t in prices.columns:
                    price_today = prices[t].loc[date]
                    price_then = last_prices.get(t)
                    if price_then and not pd.isna(price_today):
                        asset_ret = price_today / price_then - 1.0
                        port_ret += w * asset_ret
            equity *= (1.0 + port_ret)

        # Delta-based rebalancing: compute turnover between old and new weights
        if current_portfolio:
            all_tickers = set(current_portfolio.keys()) | set(target_dict.keys())
            gross_turnover = 0.0
            for t in all_tickers:
                w_old = current_portfolio.get(t, 0.0)
                w_new = target_dict.get(t, 0.0)
                gross_turnover += abs(w_new - w_old)
            # Standard definition: turnover = 0.5 * sum |w_new - w_old|
            turnover = 0.5 * gross_turnover
        else:
            turnover = 0.0

        # Apply slippage cost proportional to turnover
        if turnover > 0:
            equity *= (1.0 - SLIPPAGE_PER_TURNOVER * turnover)

        turnover_series.append(turnover)

        # Update portfolio to new target weights
        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}
        
        equity_curve.append(equity)
        equity_dates.append(date)

    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))
    turnover_series = pd.Series(turnover_series, index=pd.DatetimeIndex(equity_dates))

    # Optional: print basic turnover stats
    if not turnover_series.empty:
        avg_turnover = turnover_series.mean()
        max_turnover = turnover_series.max()
        print(f"\nTurnover stats (monthly): avg={avg_turnover:.3f}, max={max_turnover:.3f}")

    return equity_series, turnover_series, prices, strat


# ============================================================
# PLOTTING & ANALYSIS
# ============================================================

def plot_monte_carlo(equity_curve: pd.Series, n_sims=1000):
    rets = equity_curve.pct_change().dropna().values
    n = len(rets)
    
    plt.figure(figsize=(10, 6))
    final_values = []
    
    print(f"Running {n_sims} Monte Carlo simulations...")
    for i in range(n_sims):
        sim_rets = np.random.choice(rets, size=n, replace=True)
        sim_equity = [equity_curve.iloc[0]]
        for r in sim_rets:
            sim_equity.append(sim_equity[-1] * (1 + r))
        
        final_values.append(sim_equity[-1])
        if i < 100:  # Plot 100 lines only
            plt.plot(equity_curve.index[:len(sim_equity)], sim_equity, color='gray', alpha=0.05)

    plt.plot(equity_curve.index, equity_curve, color='#FF5733', linewidth=2, label='Actual Strategy')
    plt.title(f"Monte Carlo Stress Test ({n_sims} Sims)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_3_monte_carlo.png")
    plt.show()

def plot_advanced_dashboard(base_equity):
    # Get SPY
    db = MarketDB()
    spy_price = db.load_data(['SPY'])['SPY']
    db.close()
    
    start_date = base_equity.index[0]
    spy_curve = spy_price.loc[start_date:]
    spy_curve = spy_curve / spy_curve.iloc[0] * START_CAPITAL

    strat_rets = base_equity.pct_change().dropna()
    rolling_sharpe = strat_rets.rolling(12).mean() / strat_rets.rolling(12).std() * np.sqrt(12)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Subplot 1: Rolling Sharpe
    ax1.plot(rolling_sharpe, color='purple', label='12-Month Rolling Sharpe')
    ax1.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(0.0, color='black', linewidth=1)
    ax1.set_title("Stability Check: Rolling Sharpe Ratio (12-Month)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Heatmap
    m_rets = strat_rets.to_frame(name='ret')
    m_rets['Year'] = m_rets.index.year
    m_rets['Month'] = m_rets.index.month
    heatmap = m_rets.pivot(index='Year', columns='Month', values='ret')
    
    im = ax2.imshow(heatmap, cmap='RdYlGn', aspect='auto', interpolation='nearest', vmin=-0.1, vmax=0.1)
    for i in range(len(heatmap)):
        for j in range(12):
            val = heatmap.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.05 else 'black'
                ax2.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=8)

    ax2.set_title("Monthly Returns Heatmap")
    ax2.set_yticks(np.arange(len(heatmap)))
    ax2.set_yticklabels(heatmap.index)
    ax2.set_xticks(np.arange(12))
    ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fig.colorbar(im, ax=ax2, label='Monthly Return')

    # Subplot 3: Annual Bar Chart
    strat_y = base_equity.resample('YE').last().pct_change()
    spy_y = spy_curve.resample('YE').last().pct_change()
    
    common_years = sorted(list(set(strat_y.index.year) & set(spy_y.index.year)))
    s_vals = [strat_y[strat_y.index.year == y].iloc[0] for y in common_years]
    b_vals = [spy_y[spy_y.index.year == y].iloc[0] for y in common_years]
    x = np.arange(len(common_years))
    
    ax3.bar(x - 0.175, s_vals, 0.35, label='Strategy', color='#1f77b4')
    ax3.bar(x + 0.175, b_vals, 0.35, label='SPY', color='gray', alpha=0.5)
    ax3.set_title('Annual Returns Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(common_years, rotation=45)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("chart_2_dashboard.png")
    plt.show()

def plot_comparisons(base_equity):
    db = MarketDB()
    spy_price = db.load_data(['SPY'])['SPY']
    db.close()
    
    start_date = base_equity.index[0]
    spy_curve = spy_price.loc[start_date:]
    spy_curve = spy_curve / spy_curve.iloc[0] * START_CAPITAL
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(base_equity, label='Baseline Strategy', color='blue', linewidth=2)
    plt.plot(spy_curve, label='S&P 500 (SPY)', color='black', alpha=0.6)
    
    plt.title(f"Strategy Performance vs Benchmark (Start: ${START_CAPITAL:,.0f})", fontsize=14)
    plt.ylabel("Equity ($)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    def get_dd(curve): return (curve / curve.cummax() - 1) * 100
    
    plt.plot(get_dd(base_equity), label='Baseline DD', color='red', alpha=0.6, linewidth=1)
    plt.fill_between(base_equity.index, get_dd(base_equity), 0, color='red', alpha=0.1)
    plt.plot(get_dd(spy_curve), label='SPY DD', color='gray', alpha=0.4, linewidth=1)
    
    plt.title("Drawdown Risk (%)")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_1_performance_comparison.png")
    plt.show()

def plot_turnover(turnover_series):
    plt.figure(figsize=(12, 4))
    plt.plot(turnover_series, color='darkorange', linewidth=1.5)
    plt.title("Monthly Portfolio Turnover", fontsize=12)
    plt.ylabel("Turnover (0–1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_4_turnover.png")
    plt.show()

def plot_slippage_cost(turnover_series, equity_curve, slippage_per_turnover):
    slippage_cost = (turnover_series * slippage_per_turnover).fillna(0)
    cumulative_cost = (1 - slippage_cost).cumprod()

    plt.figure(figsize=(12, 4))
    plt.plot(cumulative_cost, color='red', linewidth=1.5)
    plt.title("Cumulative Slippage Impact", fontsize=12)
    plt.ylabel("Multiplier")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_5_slippage.png")
    plt.show()

def plot_regime_timeline(prices, strat, equity_curve):
    regimes = []
    for date in equity_curve.index:
        regimes.append(strat.detect_regime(prices, date))

    regime_series = pd.Series(regimes, index=equity_curve.index)

    colors = {
        "bull": "green",
        "bear": "red",
        "high_vol": "orange",
        "crash": "black",
        "unknown": "gray"
    }

    plt.figure(figsize=(12, 4))
    for reg, col in colors.items():
        mask = (regime_series == reg)
        plt.fill_between(regime_series.index, 0, 1, where=mask,
                         color=col, alpha=0.15, transform=plt.gca().get_xaxis_transform())

    plt.plot(equity_curve.index, equity_curve / equity_curve.iloc[0],
             color='blue', linewidth=1.5)

    plt.title("Regime Timeline Overlay", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_6_regime_timeline.png")
    plt.show()

def plot_rolling_metrics(equity_curve):
    rets = equity_curve.pct_change().dropna()

    rolling_cagr = (1 + rets).rolling(36).apply(lambda x: np.prod(x)**(12/36) - 1)
    rolling_vol = rets.rolling(36).std() * np.sqrt(12)

    def rolling_dd(series, window=36):
        dd = []
        for i in range(window, len(series)):
            window_curve = series.iloc[i-window:i]
            dd.append((window_curve / window_curve.cummax() - 1).min())
        return pd.Series(dd, index=series.index[window:])

    rolling_maxdd = rolling_dd(equity_curve)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].plot(rolling_cagr, color='blue')
    axs[0].set_title("Rolling 3-Year CAGR")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(rolling_maxdd, color='red')
    axs[1].set_title("Rolling 3-Year Max Drawdown")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(rolling_vol, color='purple')
    axs[2].set_title("Rolling 3-Year Volatility")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chart_7_rolling_metrics.png")
    plt.show()


if __name__ == "__main__":
    print(f"\n=== BASELINE BACKTEST (Capital: ${START_CAPITAL:,.0f}) ===")
    base_equity, turnover_series, prices, strat = run_full_strategy_backtest()
    print_performance_report(base_equity, None, name="Baseline")

    print("\n📊 Generating Clean Dashboards...")
    plot_comparisons(base_equity)
    plot_advanced_dashboard(base_equity)
    plot_turnover(turnover_series)
    plot_slippage_cost(turnover_series, base_equity, SLIPPAGE_PER_TURNOVER)
    plot_regime_timeline(prices, strat, base_equity)
    plot_rolling_metrics(base_equity)
    plot_monte_carlo(base_equity, n_sims=10000)
