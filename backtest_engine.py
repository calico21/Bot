# backtest_engine.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf
import warnings

# Import from our new modules
from data_manager import MarketDB
from strategy_core import MonthlyFortressStrategy

# Silence pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
START_CAPITAL = 100_000
CIRCUIT_BREAKER_THRESHOLD = -0.045 # -4.5% Daily Loss Limit
OUTPUT_DIR = "reports"         

# Slippage Configuration (Basis Points)
SLIPPAGE_BPS = {
    'liquid': 0.0005,  # 5 bps for SPY, IEF, etc.
    'standard': 0.0010, # 10 bps for Sector ETFs
    'stress': 0.0025    # 25 bps during high vol
}

# Create reports folder if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================
# 1. Performance Metrics
# ============================

def calculate_cagr(equity_curve):
    if equity_curve is None or len(equity_curve) < 2: return np.nan
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0 or start <= 0: return np.nan
    return (end / start) ** (1 / years) - 1

def calculate_max_drawdown(equity_curve):
    if equity_curve is None or equity_curve.empty: return np.nan
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1
    return dd.min()

def calculate_sharpe(returns, risk_free_rate=0.0):
    if returns.empty: return np.nan
    excess = returns - risk_free_rate / 12
    if excess.std() == 0: return np.nan
    return np.sqrt(12) * excess.mean() / excess.std()

def calculate_sortino(returns, risk_free_rate=0.0):
    if returns.empty: return np.nan
    excess = returns - risk_free_rate / 12
    downside = excess[excess < 0]
    if downside.std() == 0: return np.nan
    return np.sqrt(12) * excess.mean() / downside.std()
    
def calculate_calmar(cagr, max_dd):
    if abs(max_dd) < 0.0001: return 0.0
    return cagr / abs(max_dd)

def print_performance_report(equity_curve, benchmark_curve=None, name="Strategy"):
    print(f"\n=== PERFORMANCE REPORT: {name} ===")
    print(f"Period: {equity_curve.index[0].date()} -> {equity_curve.index[-1].date()}")
    
    cagr = calculate_cagr(equity_curve)
    dd = calculate_max_drawdown(equity_curve)
    rets = equity_curve.pct_change().dropna()
    sharpe = calculate_sharpe(rets)
    sortino = calculate_sortino(rets)
    calmar = calculate_calmar(cagr, dd)
    
    print(f"CAGR:           {cagr:.2%}")
    print(f"Max Drawdown:   {dd:.2%}")
    print(f"Sharpe Ratio:   {sharpe:.2f}")
    print(f"Sortino Ratio:  {sortino:.2f}")
    print(f"Calmar Ratio:   {calmar:.2f}")
    print(f"Final Equity:   ${equity_curve.iloc[-1]:,.2f}")
    
    # Win Rate Analytics
    monthly_wins = (rets > 0).sum()
    total_months = len(rets)
    win_rate = monthly_wins / total_months if total_months > 0 else 0
    print(f"Win Rate:       {win_rate:.1%}")

    if benchmark_curve is not None:
        print(f"\n--- Benchmark (SPY) ---")
        print(f"CAGR:           {calculate_cagr(benchmark_curve):.2%}")
        print(f"Max Drawdown:   {calculate_max_drawdown(benchmark_curve):.2%}")

# ============================
# 2. Core Backtest Engine
# ============================

def run_full_strategy_backtest(strat_override=None):
    """
    Runs the simulation using the Strategy Class and Data Manager.
    INCLUDES: Daily Circuit Breaker, Dynamic Slippage, and Regime Logging.
    """
    db = MarketDB()
    strat = strat_override or MonthlyFortressStrategy()
    
    # 1. Identify required tickers
    tickers = list(set(
        strat.risk_assets + 
        strat.safe_assets + 
        strat.satellite_assets + 
        [strat.market_filter, strat.bond_benchmark]
    ))
    
    # 2. Load Data
    prices = db.load_data(tickers)
    db.close()

    # 3. Auto-Download Missing Data
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"⚠️ Missing data for {missing}. Downloading from Yahoo...")
        try:
            new_data = yf.download(missing, start="2000-01-01", progress=False)
            clean_new_data = pd.DataFrame()
            if 'Adj Close' in new_data:
                data_slice = new_data['Adj Close']
            elif 'Close' in new_data:
                data_slice = new_data['Close']
            else:
                data_slice = new_data

            if isinstance(data_slice, pd.Series):
                 clean_new_data[missing[0]] = data_slice
            else:
                 clean_new_data = data_slice

            if not clean_new_data.empty:
                clean_new_data.index = pd.to_datetime(clean_new_data.index).tz_localize(None)
                if not prices.empty:
                    prices.index = pd.to_datetime(prices.index).tz_localize(None)
                prices = prices.combine_first(clean_new_data)
        except Exception as e:
            print(f"❌ Failed download: {e}")

    # 4. Prepare Data
    prices = prices.bfill().ffill()
    try:
        prices = prices.loc["2000-01-01":]
    except:
        pass

    monthly_dates = prices.resample("M").last().index

    # 5. Execution Loop
    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []
    
    current_portfolio = {}
    last_prices = {}
    turnover_series = []
    circuit_breaker_events = 0
    
    # Track the previous month's date for daily slicing
    prev_date = None

    for date in monthly_dates:
        if date not in prices.index: continue
        
        # --- A. CALCULATE RETURNS (With Circuit Breaker) ---
        port_ret = 0.0
        circuit_triggered = False

        if current_portfolio and prev_date:
            # 1. VIRTUAL CIRCUIT BREAKER CHECK
            # We scan daily returns within the month to see if we breached the loss limit
            daily_slice = prices.loc[prev_date:date]
            
            if not daily_slice.empty and len(daily_slice) > 1:
                # Reconstruct daily portfolio value
                daily_vals = pd.Series(0.0, index=daily_slice.index)
                start_prices_slice = daily_slice.iloc[0]
                
                valid_assets = [t for t in current_portfolio if t in daily_slice.columns]
                
                # Base Value = 1.0
                for t in valid_assets:
                    w = current_portfolio[t]
                    if start_prices_slice[t] > 0:
                        daily_vals += (daily_slice[t] / start_prices_slice[t]) * w
                
                # Daily % Change of the WHOLE portfolio
                daily_pct_change = daily_vals.pct_change().dropna()
                
                # Check for crash
                for d, ret in daily_pct_change.items():
                    if ret < CIRCUIT_BREAKER_THRESHOLD: 
                        # TRIGGERED!
                        circuit_triggered = True
                        circuit_breaker_events += 1
                        
                        # Return is calculated up to the crash day close
                        port_ret = (daily_vals.loc[d] / daily_vals.iloc[0]) - 1.0
                        break 
            
            # 2. If NO Crash, use standard monthly return
            if not circuit_triggered:
                for t, w in current_portfolio.items():
                    if t in prices.columns:
                        price_today = prices[t].loc[date]
                        price_then = last_prices.get(t)
                        if price_then and not pd.isna(price_today):
                            asset_ret = price_today / price_then - 1.0
                            port_ret += w * asset_ret
        
        # Apply Return
        equity *= (1.0 + port_ret)

        # --- B. GENERATE NEW SIGNAL ---
        target_list = strat.get_signal(prices, date)
        target_dict = {t: w for t, w in target_list}

        # --- C. TURNOVER & DYNAMIC SLIPPAGE ---
        all_tickers = set(current_portfolio.keys()) | set(target_dict.keys())
        gross_turnover = 0.0
        slippage_cost_total = 0.0
        
        # Determine Volatility Regime for Slippage
        # Simple proxy: if last month SPY moved > 5%, market is stressed
        spy_ret = prices['SPY'].loc[prev_date:date].pct_change().std() * np.sqrt(21) if prev_date else 0
        is_stress = spy_ret > 0.02 # approx 30% annualized vol
        
        for t in all_tickers:
            # Change in weight
            w_old = current_portfolio.get(t, 0.0)
            w_new = target_dict.get(t, 0.0)
            trade_size = abs(w_new - w_old)
            gross_turnover += trade_size
            
            # Slippage Rate
            if is_stress:
                rate = SLIPPAGE_BPS['stress']
            elif t in ['SPY', 'IEF', 'SHV', 'GLD', 'QQQ']:
                rate = SLIPPAGE_BPS['liquid']
            else:
                rate = SLIPPAGE_BPS['standard']
                
            slippage_cost_total += trade_size * rate
        
        turnover = 0.5 * gross_turnover
        
        if turnover > 0:
            equity *= (1.0 - slippage_cost_total)

        turnover_series.append(turnover)

        # Update State
        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}
        prev_date = date 
        
        equity_curve.append(equity)
        equity_dates.append(date)

    if circuit_breaker_events > 0:
        print(f"⚡ Circuit Breaker triggered {circuit_breaker_events} times during backtest.")

    return pd.Series(equity_curve, index=equity_dates), pd.Series(turnover_series, index=equity_dates), prices, strat

# ============================
# 3. Institutional Grade Plotting
# ============================

def plot_comparisons(base_equity):
    print("  • Generating Performance Chart...")
    db = MarketDB()
    try:
        spy_price = db.load_data(['SPY'])['SPY']
    except: 
        spy_price = yf.download("SPY", start=base_equity.index[0], progress=False)['Close']
    db.close()
    
    start_date = base_equity.index[0]
    spy_curve = spy_price.loc[start_date:]
    # Rebase
    spy_curve = spy_curve / spy_curve.iloc[0] * START_CAPITAL
    
    # Align dates
    common_idx = base_equity.index.intersection(spy_curve.index)
    base_equity = base_equity.loc[common_idx]
    spy_curve = spy_curve.loc[common_idx]

    plt.figure(figsize=(12, 10))
    
    # Plot 1: Equity
    plt.subplot(2, 1, 1)
    plt.plot(base_equity, label='Fortress Strategy', color='#0052cc', linewidth=2)
    plt.plot(spy_curve, label='S&P 500 (SPY)', color='gray', alpha=0.7, linewidth=1.5)
    plt.title(f"Strategy vs Benchmark (Log Scale)", fontsize=14, fontweight='bold')
    plt.ylabel("Equity ($)")
    plt.yscale('log')
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    
    # Plot 2: Drawdown
    plt.subplot(2, 1, 2)
    def get_dd(curve): return (curve / curve.cummax() - 1) * 100
    
    dd_strat = get_dd(base_equity)
    dd_spy = get_dd(spy_curve)
    plt.plot(dd_strat, label='Strategy DD', color='#d62728', linewidth=1)
    plt.fill_between(dd_strat.index, dd_strat, 0, color='#d62728', alpha=0.15)
    plt.plot(dd_spy, label='SPY DD', color='gray', alpha=0.4, linewidth=1)
    plt.title("Drawdown Profile", fontsize=12, fontweight='bold')
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_1_performance_comparison.png"), dpi=300)
    plt.close()

def plot_advanced_dashboard(base_equity):
    print("  • Generating Dashboard...")
    
    db = MarketDB()
    spy_price = db.load_data(['SPY'])['SPY']
    db.close()
    
    strat_rets = base_equity.pct_change().dropna()
    
    fig = plt.figure(figsize=(14, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1])

    # 1. Underwater
    ax1 = fig.add_subplot(gs[0])
    dd_strat = (base_equity / base_equity.cummax() - 1) * 100
    ax1.fill_between(dd_strat.index, dd_strat, 0, color='#d62728', alpha=0.3)
    ax1.plot(dd_strat.index, dd_strat, color='#d62728', linewidth=1)
    ax1.axhline(-20, color='black', linestyle=':', alpha=0.5)
    ax1.set_title("Underwater Plot (Depth & Duration)", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Heatmap
    ax2 = fig.add_subplot(gs[1])
    m_rets = strat_rets.to_frame(name='ret')
    m_rets['Year'] = m_rets.index.year
    m_rets['Month'] = m_rets.index.month
    heatmap = m_rets.pivot(index='Year', columns='Month', values='ret')
    cmap = LinearSegmentedColormap.from_list('rg', ["#d73027", "#ffffff", "#1a9850"], N=256)
    sns.heatmap(heatmap, annot=True, fmt=".1%", cmap=cmap, center=0, cbar=False, ax=ax2)
    ax2.set_title("Monthly Returns Heatmap", fontweight='bold')
    ax2.set_ylabel("")

    # 3. Annual Returns
    ax3 = fig.add_subplot(gs[2])
    strat_y = base_equity.resample('Y').last().pct_change()
    spy_y = spy_price.resample('Y').last().pct_change()
    
    common = sorted(list(set(strat_y.index.year) & set(spy_y.index.year)))
    
    s_vals = [strat_y[strat_y.index.year == y].iloc[0] if not strat_y[strat_y.index.year == y].empty else 0 for y in common]
    b_vals = [spy_y[spy_y.index.year == y].iloc[0] if not spy_y[spy_y.index.year == y].empty else 0 for y in common]
    x = np.arange(len(common))
    
    ax3.bar(x - 0.175, s_vals, 0.35, label='Strategy', color='#1f77b4')
    ax3.bar(x + 0.175, b_vals, 0.35, label='S&P 500', color='#bdc3c7')
    ax3.set_xticks(x)
    ax3.set_xticklabels(common, rotation=45)
    ax3.legend()
    ax3.set_title("Annual Returns", fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_2_dashboard.png"), dpi=300)
    plt.close()

def plot_monte_carlo(equity_curve, n_sims=5000):
    print(f"  • Running Monte Carlo ({n_sims} sims)...")
    rets = equity_curve.pct_change().dropna().values
    n_days = len(rets)
    
    sim_paths = []
    for _ in range(n_sims):
        random_rets = np.random.choice(rets, size=n_days, replace=True)
        path = np.cumprod(1 + random_rets) * equity_curve.iloc[0]
        sim_paths.append(path)
    
    sim_array = np.array(sim_paths)
    p95 = np.percentile(sim_array, 95, axis=0)
    p50 = np.percentile(sim_array, 50, axis=0)
    p05 = np.percentile(sim_array, 5, axis=0)
    
    plt.figure(figsize=(12, 7))
    plt.fill_between(equity_curve.index[1:], p05, p95, color='gray', alpha=0.15, label='90% CI')
    plt.plot(equity_curve.index[1:], p50, color='black', linestyle='--', label='Median Sim')
    plt.plot(equity_curve.index, equity_curve, color='#d62728', linewidth=2, label='Actual')
    plt.yscale('log')
    plt.title(f"Monte Carlo Stress Test (Re-shuffled History)", fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_3_monte_carlo.png"), dpi=300)
    plt.close()

def plot_distribution_analysis(equity_curve):
    print("  • Generating Distribution Analysis...")
    monthly_rets = equity_curve.resample('M').last().pct_change(fill_method=None).dropna() * 100
    
    plt.figure(figsize=(12, 6))
    sns.histplot(monthly_rets, kde=True, bins=45, color='#0052cc', alpha=0.6)
    plt.axvline(0, color='black', linestyle='--')
    
    # Stats
    mean_ret = monthly_rets.mean()
    skew = monthly_rets.skew()
    plt.text(monthly_rets.max()*0.7, plt.gca().get_ylim()[1]*0.8, 
             f"Mean: {mean_ret:.2f}%\nSkew: {skew:.2f}", 
             bbox=dict(facecolor='white', alpha=0.8))
             
    plt.title("Monthly Return Distribution", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_4_distribution.png"), dpi=300)
    plt.close()

def plot_regime_timeline(prices, strat, equity_curve):
    print("  • Generating Regime Timeline...")
    regimes = []
    # Only calculate regime for dates where we have enough history
    # We re-instantiate MarketState to tap into its logic
    from strategy_core import MarketState 
    
    dates_to_plot = equity_curve.index[10:] # Skip first few to ensure data
    
    for d in dates_to_plot:
        ms = MarketState(prices, d, strat.market_filter, strat.bond_benchmark)
        regimes.append(ms.regime)
        
    regime_series = pd.Series(regimes, index=dates_to_plot)
    
    colors = {"bull": "#d9f0a3", "bear": "#fcbba1", "high_vol": "#fecc5c", "crash": "#cb181d", "neutral": "#f0f0f0"}
    
    plt.figure(figsize=(12, 6))
    
    # Plot background regimes
    for reg, col in colors.items():
        mask = (regime_series == reg)
        if mask.sum() > 0:
            plt.fill_between(regime_series.index, 0, 1, where=mask, color=col, alpha=0.5, transform=plt.gca().get_xaxis_transform(), label=reg.upper())
        
    # Plot normalized equity
    norm_eq = (equity_curve / equity_curve.max())
    norm_eq = norm_eq.loc[dates_to_plot]
    plt.plot(norm_eq.index, norm_eq, color='#0052cc', label="Equity (Norm)", linewidth=2)
    
    plt.yticks([])
    
    # Deduplicate Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', ncol=6)
    
    plt.title("Regime Timeline Analysis", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_6_regime_timeline.png"), dpi=300)
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"\n=== BASELINE BACKTEST (Capital: ${START_CAPITAL:,.0f}) ===")
    base_equity, turnover_series, prices, strat = run_full_strategy_backtest()
    print_performance_report(base_equity, None, name="Baseline")

    print(f"\n📊 Generating Reports into '{OUTPUT_DIR}/' folder...")
    plot_comparisons(base_equity)         
    plot_advanced_dashboard(base_equity)  
    plot_monte_carlo(base_equity, n_sims=5000) 
    plot_distribution_analysis(base_equity) 
    plot_regime_timeline(prices, strat, base_equity)
    
    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder.")