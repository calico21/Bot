# research_lab.py

import argparse
import optuna
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import seaborn as sns
import warnings
import math

# Import our unified modules
import strategy_core as strategy
import backtest_engine as research

# Mute logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

STATE_FILE = "winner_dna.json"
DB_FILE = "sqlite:///optimization.db"
OUTPUT_DIR = "reports" 

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 1. ANALYTICAL UTILITIES (The "Antifragile" Upgrade)
# ==============================================================================

def calculate_smart_score(equity_curve, turnover_series=None):
    """
    Upgraded Objective Function: 'Antifragile Score'
    Rewards: Growth, Smoothness
    Penalties: Deep Drawdowns (Ulcer), Stagnation, High Turnover
    """
    # FIX: Changed from 252 (Days) to 12 (Months) to handle monthly backtest data
    if len(equity_curve) < 52: return -999.0
    
    # 1. Base Metrics
    cagr = research.calculate_cagr(equity_curve)
    
    # 2. Risk Metric: Ulcer Index
    roll_max = equity_curve.cummax()
    drawdowns = (equity_curve / roll_max - 1.0) * 100
    ulcer_index = np.sqrt((drawdowns**2).mean())
    
    if ulcer_index == 0: ulcer_index = 0.1 # Avoid div/0
    
    # 3. Base Utility (Risk-Adjusted Return)
    score = (cagr * 100) / ulcer_index
    
    # 4. Critical Penalties
    max_dd = drawdowns.min() / 100.0
    
    # A. Catastrophe Penalty (Exponential decay if DD > 25%)
    if max_dd < -0.25:
        score *= 0.5
    if max_dd < -0.40:
        score = -10.0 # Instant fail
        
    # B. Turnover Penalty (If available)
    if turnover_series is not None:
        avg_monthly_turnover = turnover_series.mean()
        # If turnover > 80% per month, penalize score
        if avg_monthly_turnover > 0.80:
            score *= 0.85
            
    # C. Stagnation Penalty (Time underwater)
    deep_pain_time = (drawdowns < -10).mean()
    if deep_pain_time > 0.50:
        score -= 2.0

    if cagr < 0: return max_dd # If losing money, score is just the drawdown
    
    return score

def block_bootstrap_sampling(returns, n_sims=2000, block_size=63, years=20):
    """Robust Bootstrap for Stress Testing"""
    # Assuming daily returns (~252 trading days)
    n_periods_target = 12 * years 
    n_blocks = int(n_periods_target / block_size)
    
    dd_results = []
    cagr_results = []
    
    ret_values = returns.values
    data_len = len(ret_values)
    
    if data_len < block_size: 
        return [-1.0] * n_sims, [0.0] * n_sims

    for _ in range(n_sims):
        sim_path = []
        starts = np.random.randint(0, data_len - block_size, size=n_blocks)
        for start in starts:
            block = ret_values[start : start + block_size]
            sim_path.extend(block)
        
        sim_rets = np.array(sim_path)
        equity = np.cumprod(1 + sim_rets)
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        dd_results.append(dd.min())
        
        # CAGR
        final_multiple = equity[-1]
        cagr = (final_multiple ** (1 / years)) - 1
        cagr_results.append(cagr)

    return dd_results, cagr_results

# ==============================================================================
# 2. PERSISTENCE LAYER
# ==============================================================================

def load_previous_winner():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            print(f"🧬 LOADED DNA from {STATE_FILE}")
            print(f"   Prev Score: {data.get('score', 0):.4f}")
            return data['params']
        except:
            pass
    return None

def save_new_winner(study, best_trial):
    if best_trial.value == -999: return

    data = {
        'score': best_trial.value,
        'params': best_trial.params,
        'cagr': f"{best_trial.user_attrs.get('test_cagr', 0):.2%}",
        'dd': f"{best_trial.user_attrs.get('test_dd', 0):.2%}"
    }
    
    with open(STATE_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n💾 NEW CHAMPION SAVED to {STATE_FILE}")

# ============================
# 3. OPTIMIZATION LOOP
# ============================
def run_optimization(trials=1000):
    print(f"\n🛡️ STARTING 'ANTIFRAGILE' OPTIMIZATION ({trials} trials)...")
    print("Objective: Maximize Return/Ulcer Ratio while Minimizing Turnover")
    
    # Initial run to get dates
    base_eq, _, _, _ = research.run_full_strategy_backtest()
    split_idx = int(len(base_eq) * 0.75) # 75% Train / 25% Test
    split_date = base_eq.index[split_idx]
    
    print(f"📅 Split Date: {split_date.date()}")
    print("-" * 140)
    print(f"{'ITER':<5} | {'SCORE':<8} | {'CAGR':<8} | {'MaxDD':<8} | {'Sharpe':<6} | {'Lev':<4} | {'TopN':<4} | {'Crash':<5} | {'Status'}")
    print("-" * 140)

    anchor = load_previous_winner()

    def objective(trial):
        # --- PARAMETER MAPPING (Matches New Strategy Core) ---
        if anchor and trial.number < (trials * 0.4): # Use anchor for first 40%
            # Narrow Search around Winner
            lev_c = anchor.get('max_lev', 2.0)
            strategy.MAX_PORTFOLIO_LEVERAGE = trial.suggest_float('max_lev', max(1.0, lev_c - 0.25), min(2.5, lev_c + 0.25))
            
            top_c = anchor.get('top_n', 6)
            strategy.TOP_N_ASSETS = trial.suggest_int('top_n', max(3, top_c - 1), min(8, top_c + 1))
            
            crash_c = anchor.get('crash_thresh', -0.12)
            strategy.CRASH_THRESHOLD = trial.suggest_float('crash_thresh', max(-0.20, crash_c - 0.05), min(-0.05, crash_c + 0.05))
            
            vol_bull = anchor.get('vol_bull', 0.18)
            strategy.TARGET_VOL_BULL = trial.suggest_float('vol_bull', max(0.10, vol_bull - 0.05), min(0.30, vol_bull + 0.05))

            # New Adaptive Params
            strategy.LOOKBACK_SMA_TREND = trial.suggest_int('sma_trend', 150, 250, step=10)
            strategy.LOOKBACK_SMA_FAST = trial.suggest_int('sma_fast', 30, 80, step=10)
            
            w1 = trial.suggest_int('mom_w1', 40, 80, step=5)
            w2 = trial.suggest_int('mom_w2', 100, 160, step=10)
            w3 = trial.suggest_int('mom_w3', 180, 250, step=10)
            strategy.MOMENTUM_WINDOW_1 = w1
            strategy.MOMENTUM_WINDOW_2 = w2
            strategy.MOMENTUM_WINDOW_3 = w3

        else:
            # Wide Search (Exploration)
            strategy.MAX_PORTFOLIO_LEVERAGE = trial.suggest_float('max_lev', 1.0, 2.2)
            strategy.TOP_N_ASSETS = trial.suggest_int('top_n', 3, 10)
            strategy.CRASH_THRESHOLD = trial.suggest_float('crash_thresh', -0.20, -0.05)
            strategy.TARGET_VOL_BULL = trial.suggest_float('vol_bull', 0.12, 0.35)
            strategy.LOOKBACK_SMA_TREND = trial.suggest_int('sma_trend', 120, 300, step=20)
            strategy.LOOKBACK_SMA_FAST = trial.suggest_int('sma_fast', 20, 100, step=10)
            
            w1 = trial.suggest_int('mom_w1', 20, 90, step=10)
            w2 = trial.suggest_int('mom_w2', 80, 180, step=20)
            w3 = trial.suggest_int('mom_w3', 150, 300, step=30)
            strategy.MOMENTUM_WINDOW_1 = w1
            strategy.MOMENTUM_WINDOW_2 = w2
            strategy.MOMENTUM_WINDOW_3 = w3
            
        # Satellite Params
        strategy.SATELLITE_ALLOCATION = trial.suggest_float('sat_alloc', 0.0, 0.25)
        strategy.SATELLITE_RSI_ENTRY = trial.suggest_int('sat_rsi', 15, 40, step=5)

        try:
            # RUN BACKTEST
            equity, turnover, _, _ = research.run_full_strategy_backtest()
            
            # SPLIT DATA
            train_eq = equity.loc[:split_date]
            test_eq = equity.loc[split_date:]
            train_to = turnover.loc[:split_date]
            test_to = turnover.loc[split_date:]
            
            # CALCULATE SCORES
            score_train = calculate_smart_score(train_eq, train_to)
            score_test = calculate_smart_score(test_eq, test_to)
            
            # Weighted Score (Favor Test Consistency)
            # If train is great but test fails, score is heavily penalized
            if score_test < 0:
                final_score = score_test # Fail
            else:
                final_score = (0.4 * score_train) + (0.6 * score_test)
            
            # Reporting Metrics
            test_cagr = research.calculate_cagr(test_eq)
            test_dd = research.calculate_max_drawdown(test_eq)
            test_sharpe = research.calculate_sharpe(test_eq.pct_change().dropna())
            
            status = "PASS" if test_cagr > 0.15 and test_dd > -0.30 else "WEAK"

            print(f"{trial.number:<5} | {final_score:8.4f} | {test_cagr:8.2%} | {test_dd:8.2%} | {test_sharpe:6.2f} | {strategy.MAX_PORTFOLIO_LEVERAGE:.2f} | {strategy.TOP_N_ASSETS:<4} | {strategy.CRASH_THRESHOLD:.2f} | {status}")
            
            trial.set_user_attr("test_cagr", test_cagr)
            trial.set_user_attr("test_dd", test_dd)
            trial.set_user_attr("test_sharpe", test_sharpe)
            
            return final_score
            
        except Exception:
            return -9999.0

    study = optuna.create_study(
        direction='maximize', 
        study_name="fortress_study_v2", 
        storage=DB_FILE,
        load_if_exists=True
    )
    
    if anchor:
        print("💉 Injecting DNA from previous winner...")
        study.enqueue_trial(anchor)
    
    study.optimize(objective, n_trials=trials)
    save_new_winner(study, study.best_trial)
    
    print("\n🏆 OPTIMIZATION COMPLETE")
    bt = study.best_trial
    print(f"Best Score: {bt.value:.4f}")
    
    # Auto-Run Analysis
    run_visualization()
    run_stress_test(study.best_trial.params)

# ============================
# 4. TOOLS: VISUALIZATION
# ============================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def run_visualization():
    print("\n🎨 Generating Visualization Suite...")
    try:
        study = optuna.load_study(study_name="fortress_study_v2", storage=DB_FILE)
    except:
        print("❌ Could not load DB.")
        return

    df = study.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    
    # --- 1. Pareto Frontier (Risk vs Return) ---
    if 'user_attrs_test_cagr' in df.columns and 'user_attrs_test_dd' in df.columns:
        plt.figure(figsize=(10, 8))
        
        # Invert DD for plotting (so top-right is best)
        x = df['user_attrs_test_dd'] * 100
        y = df['user_attrs_test_cagr'] * 100
        c = df['value'] # Score
        
        sc = plt.scatter(x, y, c=c, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(sc, label='Objective Score')
        
        plt.xlabel("Max Drawdown (%)")
        plt.ylabel("CAGR (%)")
        plt.title("Pareto Frontier: Risk vs Reward Exploration")
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='black', lw=1)
        
        # Highlight Best
        best_idx = df['value'].idxmax()
        plt.scatter(x[best_idx], y[best_idx], color='red', s=150, edgecolors='black', label='Champion')
        plt.legend()
        
        plt.savefig(os.path.join(OUTPUT_DIR, "chart_9_pareto_frontier.png"), dpi=300)
        plt.close()
        print("   • Pareto Frontier saved.")

    # --- 2. 3D Radar (Parameters) ---
    # (Same as before but updated params)
    params = ['params_max_lev', 'params_top_n', 'params_vol_bull', 'params_mom_w2', 'params_crash_thresh']
    valid_params = [p for p in params if p in df.columns]
    
    if len(valid_params) >= 3:
        subset = df[valid_params].copy()
        for col in subset.columns:
            subset[col] = normalize(subset[col])
        
        subset['score'] = df['value']
        subset = subset.sort_values('score').tail(50) # Top 50
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        N = len(valid_params)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.concatenate((angles, [angles[0]])) 

        polys = []
        z_heights = []
        colors = []

        for i, row in subset.iterrows():
            values = row[valid_params].values
            values = np.concatenate((values, [values[0]]))
            x = values * np.cos(angles)
            y = values * np.sin(angles)
            z = row['score']
            polys.append(list(zip(x, y)))
            z_heights.append(z)
            colors.append(plt.cm.plasma((z - subset['score'].min()) / (subset['score'].max() - subset['score'].min())))

        poly = PolyCollection(polys, facecolors=colors, edgecolors='grey', alpha=0.3)
        ax.add_collection3d(poly, zs=z_heights, zdir='z')

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_zlim(subset['score'].min(), subset['score'].max())
        ax.set_title("3D Parameter Convergence")
        ax.axis('off')
        
        plt.savefig(os.path.join(OUTPUT_DIR, "chart_8_optimization_radar.png"), dpi=300)
        plt.close()
        print("   • 3D Radar saved.")

# ============================
# 5. TOOL: WALK-FORWARD VALIDATION (NEW)
# ============================
def run_walk_forward_analysis():
    """
    Simulates rolling 3-year windows to test regime consistency.
    """
    print("\n🚶 RUNNING WALK-FORWARD ANALYSIS...")
    
    # Load DNA
    params = load_previous_winner()
    if not params:
        print("❌ No winner_dna.json found. Optimize first.")
        return

    # Inject Params
    mapping = {
        'max_lev': 'MAX_PORTFOLIO_LEVERAGE',
        'top_n': 'TOP_N_ASSETS',
        'crash_thresh': 'CRASH_THRESHOLD',
        'vol_bull': 'TARGET_VOL_BULL',
        'sma_trend': 'LOOKBACK_SMA_TREND',
        'sma_fast': 'LOOKBACK_SMA_FAST',
        'mom_w1': 'MOMENTUM_WINDOW_1',
        'mom_w2': 'MOMENTUM_WINDOW_2',
        'mom_w3': 'MOMENTUM_WINDOW_3',
        'sat_alloc': 'SATELLITE_ALLOCATION',
        'sat_rsi': 'SATELLITE_RSI_ENTRY'
    }
    for k, v in params.items():
        if k in mapping: setattr(strategy, mapping[k], v)

    # Run Full Backtest
    equity, _, _, _ = research.run_full_strategy_backtest()
    
    # Rolling 3-Year Windows (approx 756 trading days)
    window_size = 756
    step_size = 126 # Step 6 months
    
    results = []
    
    for start_idx in range(0, len(equity) - window_size, step_size):
        end_idx = start_idx + window_size
        subset = equity.iloc[start_idx : end_idx]
        
        start_date = subset.index[0]
        end_date = subset.index[-1]
        
        cagr = research.calculate_cagr(subset)
        dd = research.calculate_max_drawdown(subset)
        
        results.append({
            'Start': start_date,
            'End': end_date,
            'CAGR': cagr,
            'MaxDD': dd,
            'Score': cagr / abs(dd) if dd != 0 else 0
        })
        
    res_df = pd.DataFrame(results)
    
    print("\n--- WALK FORWARD RESULTS (Rolling 3-Year) ---")
    print(res_df[['Start', 'End', 'CAGR', 'MaxDD']].tail(10))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['Start'], res_df['CAGR'], label='Rolling CAGR', color='green')
    plt.plot(res_df['Start'], res_df['MaxDD'], label='Rolling MaxDD', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Walk-Forward Stability Check (Rolling 3-Year)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_10_walk_forward.png"), dpi=300)
    plt.close()
    print("   • Walk-Forward chart saved.")

# ============================
# 6. TOOL: STRESS TEST
# ============================
def run_stress_test(params=None):
    print(f"\n🎲 Running BLOCK BOOTSTRAP Stress Test...")
    
    # Inject Params if provided
    if params:
        mapping = {
            'max_lev': 'MAX_PORTFOLIO_LEVERAGE',
            'top_n': 'TOP_N_ASSETS',
            'crash_thresh': 'CRASH_THRESHOLD',
            'vol_bull': 'TARGET_VOL_BULL',
            'sma_trend': 'LOOKBACK_SMA_TREND',
            'sma_fast': 'LOOKBACK_SMA_FAST',
            'mom_w1': 'MOMENTUM_WINDOW_1',
            'mom_w2': 'MOMENTUM_WINDOW_2',
            'mom_w3': 'MOMENTUM_WINDOW_3'
        }
        for k, v in params.items():
            if k in mapping: setattr(strategy, mapping[k], v)

    equity, _, _, _ = research.run_full_strategy_backtest()
    daily_rets = equity.pct_change().dropna()
    
    dd_results, cagr_results = block_bootstrap_sampling(daily_rets, n_sims=2000, block_size=63, years=20)
    
    median_dd = np.median(dd_results)
    worst_1_percent = np.percentile(dd_results, 1)
    
    print("=" * 40)
    print(" 🎲 STRESS TEST RESULTS")
    print("=" * 40)
    print(f"Median CAGR:        {np.median(cagr_results):.2%}")
    print(f"Median Drawdown:    {median_dd:.2%}")
    print(f"Worst 1% Drawdown:  {worst_1_percent:.2%}")
    print("-" * 40)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=dd_results, y=cagr_results, alpha=0.3, color="blue", s=10)
    plt.xlabel("Max Drawdown")
    plt.ylabel("CAGR")
    plt.title("Stress Test: Risk vs Reward Cloud")
    plt.axvline(median_dd, color='red', linestyle='--', label='Median DD')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "chart_7_stress_test.png"), dpi=300)
    plt.close()

# ============================
# MAIN ENTRY POINT
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['optimize', 'visualize', 'stress', 'walkforward'])
    args = parser.parse_args()
    
    if args.mode == 'optimize': run_optimization()
    elif args.mode == 'visualize': run_visualization()
    elif args.mode == 'stress': run_stress_test()
    elif args.mode == 'walkforward': run_walk_forward_analysis()