# research.py

import itertools
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

# ============================================================
# Helper: Run the MAIN Strategy (The Baseline)
# ============================================================
def run_full_strategy_backtest() -> pd.Series:
    db = MarketDB()
    strat = MonthlyFortressStrategy()
    
    tickers = list(set(
        strat.risk_assets + 
        strat.safe_assets + 
        [strat.market_filter, strat.bond_benchmark]
    ))
    
    prices = db.load_data(tickers)
    # Justo después de db.load_data(tickers)
    prices = prices.fillna(method='bfill') # Rellena hacia atrás con el primer precio disponible
    db.close()

    try: prices = prices.loc["2000-01-01":]
    except: pass

    monthly_dates = prices.resample("ME").last().index
    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []
    
    current_portfolio = {}
    last_prices = {}

    for date in monthly_dates:
        if date not in prices.index: continue
        
        target_list = strat.get_signal(prices, date) 
        target_dict = {t: w for t, w in target_list}

        if current_portfolio and last_prices:
            port_ret = 0.0
            for t, w in current_portfolio.items():
                if t in prices.columns:
                    price_today = prices[t].loc[date]
                    price_then = last_prices.get(t)
                    if price_then and not pd.isna(price_today):
                        asset_ret = price_today / price_then
                        port_ret += w * (asset_ret - 1.0)
            equity *= (1.0 + port_ret)

        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}
        
        equity_curve.append(equity)
        equity_dates.append(date)

    return pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))

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
        if i < 100: # Plot 100 lines only
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

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"\n=== BASELINE BACKTEST (Capital: ${START_CAPITAL:,.0f}) ===")
    base_equity = run_full_strategy_backtest()
    print_performance_report(base_equity, None, name="Baseline")

    print("\n📊 Generating Clean Dashboards...")
    plot_comparisons(base_equity)
    plot_advanced_dashboard(base_equity)
    
    if base_equity is not None and not base_equity.empty:
        plot_monte_carlo(base_equity, n_sims=10000)