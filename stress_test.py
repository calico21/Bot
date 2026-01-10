# stress_test.py

import numpy as np
import pandas as pd
import warnings

# --- SILENCE PANDAS FUTURE WARNINGS ---
# This suppresses the "fill_method='pad'" warnings from strategy.py
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from quant_db_manager import MarketDB
from strategy import MonthlyFortressStrategy

# --- CONFIGURATION ---
SIMULATIONS = 100000   # How many "Alternative Universes" to run
YEARS = 20           # Length of simulation
START_CAPITAL = 10000

def get_strategy_returns():
    """
    Runs the backtest once to extract the strategy's daily return profile (The DNA).
    """
    print("--- Extracting Strategy DNA (Daily Returns) ---")
    
    # 1. Initialize DB
    db = MarketDB()
    strat = MonthlyFortressStrategy()
    
    # 2. Load Data
    all_tickers = list(set(
        strat.risk_assets + 
        strat.safe_assets + 
        [strat.market_filter, strat.bond_benchmark]
    ))
    
    prices = db.load_data(all_tickers)
    db.close()
    
    # Filter 2005+
    try: prices = prices.loc["2005-01-01":]
    except: pass
    
    # 3. Run simplified loop to get daily equity
    dates = prices.index
    cash = START_CAPITAL
    holdings = {}
    
    equity_curve = []
    
    # Identify rebalance dates (Month End)
    monthly_dates = prices.resample('ME').last().index
    valid_dates = set(dates)
    clean_rebalance_dates = set()
    for d in monthly_dates:
        loc = dates.searchsorted(d)
        if loc > 0: clean_rebalance_dates.add(dates[loc-1])

    total_steps = len(dates)
    
    for i, d in enumerate(dates):
        # Update Value
        day_val = cash
        for t, qty in holdings.items():
            if t in prices.columns:
                px = prices.at[d, t]
                if not pd.isna(px): day_val += qty * px
        
        equity_curve.append(day_val)
        
        # Rebalance
        if d in clean_rebalance_dates:
            # Print progress periodically to show it's not frozen
            if i % 500 == 0:
                print(f"Processing date: {d.date()}...")
                
            target = dict(strat.get_signal(prices, d))
            holdings = {}
            cash = day_val
            for t, w in target.items():
                if t in prices.columns:
                    px = prices.at[d, t]
                    if not pd.isna(px) and px > 0:
                        qty = (day_val * w) / px
                        holdings[t] = qty
                        cash -= (day_val * w)
                        
    equity_series = pd.Series(equity_curve, index=dates)
    daily_returns = equity_series.pct_change().dropna()
    return daily_returns

def run_monte_carlo(daily_returns):
    print(f"\n--- Running {SIMULATIONS} Simulations of {YEARS} Years ---")
    
    results_cagr = []
    results_maxdd = []
    ruin_count = 0
    
    # Trading days in N years
    n_days = 252 * YEARS
    
    for i in range(SIMULATIONS):
        # 1. Shuffle the returns (Randomize the order of days)
        sim_rets = np.random.choice(daily_returns, size=n_days, replace=True)
        
        # 2. Build Equity Curve
        sim_curve = np.cumprod(1 + sim_rets) * START_CAPITAL
        
        # 3. Calculate Metrics
        final_val = sim_curve[-1]
        
        # Avoid division by zero
        if final_val <= 0:
            cagr = -1.0
        else:
            cagr = (final_val / START_CAPITAL) ** (1/YEARS) - 1
        
        # Max Drawdown
        running_max = np.maximum.accumulate(sim_curve)
        drawdown = (sim_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        results_cagr.append(cagr)
        results_maxdd.append(max_dd)
        
        # Risk of Ruin: Did we lose more than 70%?
        if max_dd < -0.70:
            ruin_count += 1
            
    # --- REPORTING ---
    print("\n" + "="*40)
    print(" 🎲 MONTE CARLO STRESS TEST RESULTS")
    print("="*40)
    print(f"Based on {SIMULATIONS} random timelines:")
    
    print(f"\n--- GROWTH (CAGR) ---")
    print(f"Worst Case (1%):    {np.percentile(results_cagr, 1):.2%}")
    print(f"Bad Case (10%):     {np.percentile(results_cagr, 10):.2%}")
    print(f"Median Case (50%):  {np.median(results_cagr):.2%}")
    print(f"Best Case (90%):    {np.percentile(results_cagr, 90):.2%}")
    
    print(f"\n--- RISK (Max Drawdown) ---")
    print(f"Median Drawdown:    {np.median(results_maxdd):.2%}")
    print(f"Worst Drawdown:     {np.min(results_maxdd):.2%}")
    
    print(f"\n--- SURVIVAL ---")
    prob_ruin = (ruin_count / SIMULATIONS) * 100
    print(f"Risk of Ruin (>70% Loss): {prob_ruin:.2f}%")
    
    if prob_ruin < 1.0:
        print("✅ VERDICT: Strategy is ROBUST.")
    elif prob_ruin < 5.0:
        print("⚠️ VERDICT: Strategy is RISKY.")
    else:
        print("❌ VERDICT: Strategy is FRAGILE.")

if __name__ == "__main__":
    dna = get_strategy_returns()
    run_monte_carlo(dna)