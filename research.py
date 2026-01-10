# research.py

import itertools
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf  # Required for auto-downloading missing data

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
    
    # 1. Identify all required tickers
    tickers = list(set(
        strat.risk_assets + 
        strat.safe_assets + 
        strat.satellite_assets + 
        [strat.market_filter, strat.bond_benchmark]
    ))
    
    # 2. Load available data from DB
    prices = db.load_data(tickers)
    db.close()

    # 3. AUTO-FIX: Download missing tickers from Yahoo Finance
    missing_tickers = [t for t in tickers if t not in prices.columns]
    if missing_tickers:
        print(f"\n⚠️ Missing data for {missing_tickers}. Downloading from Yahoo Finance...")
        try:
            # Download missing data
            new_data = yf.download(missing_tickers, start="2000-01-01", progress=False)
            
            # Handle Yahoo Finance data structure (MultiIndex or Single Index)
            clean_new_data = pd.DataFrame()
            
            if 'Adj Close' in new_data:
                # If MultiIndex (Price, Ticker) or just (Adj Close)
                data_slice = new_data['Adj Close']
                
                if isinstance(data_slice, pd.Series): # Single ticker result
                    clean_new_data[missing_tickers[0]] = data_slice
                else: # DataFrame result
                    clean_new_data = data_slice
            
            elif 'Close' in new_data:
                # Fallback to Close if Adj Close is missing
                data_slice = new_data['Close']
                if isinstance(data_slice, pd.Series):
                    clean_new_data[missing_tickers[0]] = data_slice
                else:
                    clean_new_data = data_slice

            # Merge new data into prices
            if not clean_new_data.empty:
                # Ensure timezone-naive indices for compatibility
                clean_new_data.index = pd.to_datetime(clean_new_data.index).tz_localize(None)
                if not prices.empty:
                    prices.index = pd.to_datetime(prices.index).tz_localize(None)
                
                # Combine (prefer DB data, fill with Yahoo data)
                prices = prices.combine_first(clean_new_data)
                print(f"✅ Successfully added data for: {list(clean_new_data.columns)}")
            else:
                print("❌ Download returned no usable data.")

        except Exception as e:
            print(f"❌ Failed to download missing data: {e}")

    # 4. Final Cleanup
    prices = prices.bfill().ffill() # Fill gaps
    
    try:
        prices = prices.loc["2000-01-01":]
    except Exception:
        pass

    # 5. Run Backtest Loop
    monthly_dates = prices.resample("ME").last().index
    equity = START_CAPITAL
    equity_curve = []
    equity_dates = []
    
    current_portfolio = {}
    last_prices = {}
    turnover_series = []

    for date in monthly_dates:
        if date not in prices.index:
            continue
        
        # Get target weights
        target_list = strat.get_signal(prices, date)
        target_dict = {t: w for t, w in target_list}

        # Calculate Portfolio Return
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

        # Calculate Turnover & Slippage
        if current_portfolio:
            all_tickers = set(current_portfolio.keys()) | set(target_dict.keys())
            gross_turnover = 0.0
            for t in all_tickers:
                w_old = current_portfolio.get(t, 0.0)
                w_new = target_dict.get(t, 0.0)
                gross_turnover += abs(w_new - w_old)
            turnover = 0.5 * gross_turnover
        else:
            turnover = 0.0

        if turnover > 0:
            equity *= (1.0 - SLIPPAGE_PER_TURNOVER * turnover)

        turnover_series.append(turnover)

        # Update Portfolio
        current_portfolio = target_dict
        last_prices = {t: prices[t].loc[date] for t in current_portfolio.keys()}
        
        equity_curve.append(equity)
        equity_dates.append(date)

    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_dates))
    turnover_series = pd.Series(turnover_series, index=pd.DatetimeIndex(equity_dates))

    if not turnover_series.empty:
        avg_turnover = turnover_series.mean()
        max_turnover = turnover_series.max()
        print(f"\nTurnover stats (monthly): avg={avg_turnover:.3f}, max={max_turnover:.3f}")

    return equity_series, turnover_series, prices, strat


# ============================================================
# INSTITUTIONAL GRADE PLOTTING (FINAL & FIXED)
# ============================================================

# --- CHART 1: PERFORMANCE VS BENCHMARK ---
def plot_comparisons(base_equity):
    db = MarketDB()
    spy_price = db.load_data(['SPY'])['SPY']
    db.close()
    
    start_date = base_equity.index[0]
    spy_curve = spy_price.loc[start_date:]
    spy_curve = spy_curve / spy_curve.iloc[0] * START_CAPITAL
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Equity
    plt.subplot(2, 1, 1)
    plt.plot(base_equity, label='Baseline Strategy', color='#0052cc', linewidth=2)
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
    plt.savefig("chart_1_performance_comparison.png", dpi=300)
    plt.show()

# --- CHART 2: THE DASHBOARD (Heatmap + Underwater + Annual) ---
def plot_advanced_dashboard(base_equity):
    db = MarketDB()
    spy_price = db.load_data(['SPY'])['SPY']
    db.close()
    
    start_date = base_equity.index[0]
    spy_curve = spy_price.loc[start_date:]
    spy_curve = spy_curve / spy_curve.iloc[0] * START_CAPITAL

    strat_rets = base_equity.pct_change().dropna()
    
    fig = plt.figure(figsize=(14, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1])

    # --- 1. UNDERWATER PLOT ---
    ax1 = fig.add_subplot(gs[0])
    def get_dd(curve): return (curve / curve.cummax() - 1) * 100
    dd_strat = get_dd(base_equity)
    
    ax1.fill_between(dd_strat.index, dd_strat, 0, color='#d62728', alpha=0.3)
    ax1.plot(dd_strat.index, dd_strat, color='#d62728', linewidth=1)
    ax1.axhline(-20, color='black', linestyle=':', alpha=0.5, label='-20% Threshold')
    
    ax1.set_title("Underwater Plot (Drawdown Depth & Duration)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Drawdown %")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # --- 2. MONTHLY HEATMAP ---
    ax2 = fig.add_subplot(gs[1])
    m_rets = strat_rets.to_frame(name='ret')
    m_rets['Year'] = m_rets.index.year
    m_rets['Month'] = m_rets.index.month
    heatmap = m_rets.pivot(index='Year', columns='Month', values='ret')
    
    cmap = LinearSegmentedColormap.from_list('rg', ["#d73027", "#ffffff", "#1a9850"], N=256)
    
    sns.heatmap(heatmap, annot=True, fmt=".1%", cmap=cmap, center=0, 
                cbar=False, linewidths=1, linecolor='#f0f0f0', ax=ax2,
                annot_kws={"size": 10, "weight": "bold"}) 
    
    ax2.set_title("Monthly Returns Heatmap", fontweight='bold', fontsize=12)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax2.tick_params(axis='y', rotation=0)

    # --- 3. ANNUAL RETURNS BAR CHART ---
    ax3 = fig.add_subplot(gs[2])
    strat_y = base_equity.resample('YE').last().pct_change()
    spy_y = spy_curve.resample('YE').last().pct_change()
    
    common_years = sorted(list(set(strat_y.index.year) & set(spy_y.index.year)))
    s_vals = [strat_y[strat_y.index.year == y].iloc[0] for y in common_years]
    b_vals = [spy_y[spy_y.index.year == y].iloc[0] for y in common_years]
    x = np.arange(len(common_years))
    
    ax3.bar(x - 0.175, s_vals, 0.35, label='Strategy', color='#1f77b4', edgecolor='black', linewidth=0.5, zorder=3)
    ax3.bar(x + 0.175, b_vals, 0.35, label='S&P 500', color='#bdc3c7', edgecolor='black', linewidth=0.5, zorder=3)
    
    for i, v in enumerate(s_vals):
        offset = 0.02 if v >= 0 else -0.04
        ax3.text(i - 0.175, v + offset, f"{v:.0%}", ha='center', fontsize=9, fontweight='bold', color='black')

    ax3.set_title('Annual Returns Comparison', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(common_years, rotation=45, ha='right')
    ax3.legend(loc='upper left')
    ax3.grid(True, axis='y', alpha=0.3, zorder=0)
    ax3.axhline(0, color='black', linewidth=1)
    
    plt.savefig("chart_2_dashboard.png", dpi=300)
    plt.show()

# --- CHART 3: MONTE CARLO (Cone) ---
def plot_monte_carlo(equity_curve: pd.Series, n_sims=1000):
    rets = equity_curve.pct_change().dropna().values
    n_days = len(rets)
    
    print(f"Running {n_sims} Monte Carlo simulations...")
    sim_paths = []
    for _ in range(n_sims):
        random_rets = np.random.choice(rets, size=n_days, replace=True)
        path = np.cumprod(1 + random_rets) * equity_curve.iloc[0]
        sim_paths.append(path)
    
    sim_array = np.array(sim_paths)
    
    p95 = np.percentile(sim_array, 95, axis=0)
    p50 = np.percentile(sim_array, 50, axis=0)
    p05 = np.percentile(sim_array, 5, axis=0)
    
    dates = equity_curve.index[1:]
    
    plt.figure(figsize=(12, 7))
    plt.fill_between(dates, p05, p95, color='gray', alpha=0.15, label='90% Conf. Interval')
    plt.plot(dates, p50, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Median Simulation')
    plt.plot(equity_curve.index, equity_curve, color='#d62728', linewidth=2, label='Actual Strategy')
    
    plt.title(f"Monte Carlo Stress Test ({n_sims} Sims)", fontsize=14, fontweight='bold')
    plt.ylabel("Equity ($) - Log Scale")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("chart_3_monte_carlo.png", dpi=300)
    plt.show()

# --- CHART 4: RETURN DISTRIBUTION (Log Scale Fixed) ---
def plot_distribution_analysis(equity_curve):
    """
    Shows the histogram of monthly returns vs Normal Distribution.
    Uses Log Scale on Y-Axis to handle the 'Cash Spike' at 0%.
    """
    # Fix FutureWarning by adding fill_method=None
    monthly_rets = equity_curve.resample('ME').last().pct_change(fill_method=None).dropna() * 100
    
    plt.figure(figsize=(12, 6))
    
    # VISIBILITY FIX: Reduced bins to 45 so bars have width. 
    sns.histplot(monthly_rets, kde=True, bins=45, color='#0052cc', alpha=0.6, 
                 label='Strategy Returns', log_scale=(False, True), 
                 edgecolor='black', linewidth=1.2)
    
    # Statistics
    mean_ret = monthly_rets.mean()
    sigma = monthly_rets.std()
    skew = monthly_rets.skew()
    kurt = monthly_rets.kurtosis()
    
    stats_text = f"Mean: {mean_ret:.2f}%\nStdDev: {sigma:.2f}%\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Monthly Return Distribution (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Monthly Return (%)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.3, which="both") 
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_4_distribution.png", dpi=300)
    plt.show()

# --- CHART 5: YEARLY EFFICIENCY (Fixed Crash) ---
def plot_yearly_efficiency(equity_curve):
    """
    Scatter plot of every year's performance.
    FIXED: Properly handles integer vs datetime indexing.
    """
    # 1. Get Yearly Returns (Index is Datetime)
    yearly_resample = equity_curve.resample('YE').last()
    yearly_rets = yearly_resample.pct_change(fill_method=None).dropna()
    
    # 2. Get Yearly Volatility (Index is ALREADY Integers due to groupby)
    daily_rets = equity_curve.pct_change(fill_method=None).dropna()
    yearly_vol = daily_rets.groupby(daily_rets.index.year).std() * np.sqrt(252)
    
    # 3. Force Yearly Returns to Integer Index to match Volatility
    yearly_rets.index = yearly_rets.index.year
    
    # 4. Find Common Years (Intersection)
    common_years = sorted(list(set(yearly_rets.index) & set(yearly_vol.index)))
    
    # 5. Select Data safely
    y_vals = yearly_rets.loc[common_years] * 100
    x_vals = yearly_vol.loc[common_years] * 100
    
    plt.figure(figsize=(10, 8))
    
    # Scatter Plot
    plt.scatter(x_vals, y_vals, c=y_vals, cmap='RdYlGn', s=150, edgecolors='black', alpha=0.8)
    
    # Add Labels
    for year, x, y in zip(common_years, x_vals, y_vals):
        plt.text(x, y + (max(y_vals)*0.04), str(year), fontsize=9, ha='center', fontweight='bold')
        
    # Add Quadrant Lines
    plt.axhline(y_vals.mean(), color='gray', linestyle='--', alpha=0.5, label='Avg Return')
    plt.axvline(x_vals.mean(), color='gray', linestyle='--', alpha=0.5, label='Avg Vol')
    
    plt.title("Yearly Risk vs. Reward Efficiency", fontsize=14, fontweight='bold')
    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annual Return (%)")
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Annual Return %')
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_5_efficiency.png", dpi=300)
    plt.show()

# --- CHART 6: REGIME TIMELINE ---
def plot_regime_timeline(prices, strat, equity_curve):
    regimes = []
    for date in equity_curve.index:
        regimes.append(strat.detect_regime(prices, date))

    regime_series = pd.Series(regimes, index=equity_curve.index)

    colors = {
        "bull": "#d9f0a3",      
        "bear": "#fcbba1",      
        "high_vol": "#fecc5c",  
        "crash": "#cb181d",     
        "unknown": "#f0f0f0"    
    }

    plt.figure(figsize=(12, 5))
    for reg, col in colors.items():
        mask = (regime_series == reg)
        plt.fill_between(regime_series.index, 0, 1, where=mask,
                         color=col, alpha=0.5, transform=plt.gca().get_xaxis_transform(), 
                         label=reg.upper())

    norm_equity = (equity_curve / equity_curve.max()) * 0.8 + 0.1
    plt.plot(equity_curve.index, norm_equity, color='#0052cc', linewidth=2, label="Equity Curve")

    plt.title("Market Regime Detection Timeline", fontsize=14, fontweight='bold')
    plt.yticks([])
    plt.legend(loc='upper left', ncol=6, frameon=True)
    plt.tight_layout()
    plt.savefig("chart_6_regime_timeline.png", dpi=300)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"\n=== BASELINE BACKTEST (Capital: ${START_CAPITAL:,.0f}) ===")
    base_equity, turnover_series, prices, strat = run_full_strategy_backtest()
    print_performance_report(base_equity, None, name="Baseline")

    print("\n📊 Generating Clean Dashboards...")
    plot_comparisons(base_equity)         
    plot_advanced_dashboard(base_equity)  
    plot_monte_carlo(base_equity, n_sims=10000) 
    plot_distribution_analysis(base_equity) 
    plot_yearly_efficiency(base_equity)     
    plot_regime_timeline(prices, strat, base_equity)