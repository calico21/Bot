# optimization.py

import optuna
import pandas as pd
import numpy as np
import strategy
import research
import logging

# Mute standard logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """
    Survival-First Optimization Function.
    Prioritizes low Drawdown and Sortino Ratio over raw CAGR.
    """
    
    # --- 1. DEFINE SAFER SEARCH SPACE ---
    # We restrict the AI from even trying dangerous values.
    
    # Leverage: Cap at 2.15 (Anything higher risks ruin)
    strategy.MAX_PORTFOLIO_LEVERAGE = trial.suggest_float('max_lev', 1.6, 2.15)
    
    # Concentration: Cap at 55%
    strategy.MAX_SINGLE_ASSET_EXPOSURE = trial.suggest_float('max_single', 0.35, 0.55)
    
    # Crash Anchor: Force it to be aggressive (70% - 100% Cash)
    strategy.ANCHOR_WEIGHT_CRASH = trial.suggest_float('anchor_crash', 0.70, 1.00)
    
    # Trend Speed: Allow it to find the best fit
    strategy.LOOKBACK_SMA_TREND = trial.suggest_int('sma_trend', 180, 240, step=10)
    
    # Volatility Targets (Conservative)
    strategy.TARGET_VOL_BULL = trial.suggest_float('vol_bull', 0.15, 0.22)
    strategy.TARGET_VOL_BEAR = trial.suggest_float('vol_bear', 0.05, 0.10)

    # --- 2. RUN BACKTEST ---
    try:
        # Run the standard backtest using the monkey-patched settings
        equity, _, _, _ = research.run_full_strategy_backtest()
    except Exception:
        return -99999

    # --- 3. CALCULATE RISK METRICS ---
    
    # Max Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    
    # CAGR
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (365.0 / days) - 1
    
    # Sortino Ratio (The best measure of risk-adjusted return)
    # Fixed: added fill_method=None to silence warning
    daily_rets = equity.pct_change(fill_method=None).dropna()
    downside_rets = daily_rets[daily_rets < 0]
    sortino = np.sqrt(252) * daily_rets.mean() / (downside_rets.std() + 1e-9)

    # --- 4. THE "SURVIVAL" SCORING FUNCTION ---
    
    # RULE 1: The Death Penalty
    # If the strategy ever loses more than 35% in the past, it fails immediately.
    if max_dd < -0.35: 
        return -1000 + (max_dd * 100) # Returns e.g. -1040

    # RULE 2: Tail Risk Penalty
    # If the worst 1% of months are disastrous, penalize.
    # Fixed: added fill_method=None to silence warning
    monthly_rets = equity.resample('ME').last().pct_change(fill_method=None).dropna()
    worst_month = monthly_rets.min()
    if worst_month < -0.15: # If it lost >15% in a single month
        return -500

    # RULE 3: Score based on Sortino (Stability) rather than just CAGR
    # A high Sortino means the line goes up smoothly without jagged crashes.
    score = (sortino * 100) + (cagr * 50) + (max_dd * 50) 
    
    # This formula rewards Profit (CAGR) but subtracts points for Drawdown (max_dd is negative)
    
    return score

if __name__ == "__main__":
    print("\n🛡️ STARTING ROBUSTNESS OPTIMIZATION...")
    print("Goal: maximize profit with 0% Risk of Ruin constraint.")
    print("Trials: 50 (Fast Mode)")
    print("-" * 60)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("\n" + "="*60)
    print("✅ SAFE OPTIMIZATION COMPLETE")
    print("="*60)
    
    print(f"Best Safe Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  • {key}: {value}")
    
    print("\n⚠️ ACTION: Copy these parameters into strategy.py to fix the Risk of Ruin.")