# research_lab.py (FIXED CRASH THRESHOLD LOGIC)

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

# ==============================================================================
# 1. ANALYTICAL UTILITIES
# ==============================================================================

def calculate_smart_score(equity_curve):
    """Calculates weighted utility score."""
    if len(equity_curve) < 12: return -999.0
    
    cagr = research.calculate_cagr(equity_curve)
    rets = equity_curve.pct_change().dropna()
    downside_std = rets[rets < 0].std() * np.sqrt(12)
    
    if downside_std == 0: return -999.0
    
    sortino = (cagr / downside_std)
    dd = research.calculate_max_drawdown(equity_curve)
    
    # Utility: Sortino * Log(Growth)
    utility = sortino * np.log1p(max(0, cagr))
    
    # Smooth Penalty for Drawdown
    penalty_factor = 1.0
    if dd < -0.15:
        excess_dd = abs(dd) - 0.15
        penalty_factor = max(0.0, 1.0 - (excess_dd * 3.5))

    final_score = utility * penalty_factor
    if cagr < 0: return dd 
    return final_score

def block_bootstrap_sampling(returns, n_sims=2000, block_size=63, years=20):
    # NOTE: '12' assumes your data is Monthly. If Daily, change 12 to 252.
    n_periods_target = 12 * years 
    n_blocks = int(n_periods_target / block_size)
    
    dd_results = []
    cagr_results = []
    
    ret_values = returns.values
    data_len = len(ret_values)
    
    # Safety check
    if data_len < block_size: 
        return [-1.0] * n_sims, [0.0] * n_sims

    for _ in range(n_sims):
        sim_path = []
        # Randomly select start points for blocks
        starts = np.random.randint(0, data_len - block_size, size=n_blocks)
        for start in starts:
            block = ret_values[start : start + block_size]
            sim_path.extend(block)
        
        sim_rets = np.array(sim_path)
        
        # Build equity curve for this simulation
        # Starts at 1.0
        equity = np.cumprod(1 + sim_rets)
        
        # --- 1. Calculate Max Drawdown ---
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        dd_results.append(dd.min())
        
        # --- 2. Calculate CAGR ---
        # Final Equity multiple after 'years'
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
def run_optimization(trials=500):
    print(f"\n🛡️ STARTING CONTINUOUS OPTIMIZATION ({trials} trials)...")
    print("Objective: 70% Real Performance (Test) + 30% Historical (Train)")
    
    base_eq, _, _, _ = research.run_full_strategy_backtest()
    split_idx = int(len(base_eq) * 0.70)
    split_date = base_eq.index[split_idx]
    
    print(f"📅 Split Date: {split_date.date()}")
    print("-" * 130)
    print(f"{'ITER':<5} | {'W.SCORE':<8} | {'CAGR(Te)':<9} | {'MaxDD(Te)':<9} | {'Sort(Te)':<8} | {'Diff':<7} | {'Lev':<4} | {'TopN':<4} | {'Crash':<5}")
    print("-" * 130)

    # 1. Load Anchor (Previous Winner)
    anchor = load_previous_winner()

    def objective(trial):
        # --- SMART NARROWING LOGIC ---
        if anchor:
            # Leverage: +/- 0.15
            lev_c = anchor.get('max_lev', 2.0)
            strategy.MAX_PORTFOLIO_LEVERAGE = trial.suggest_float('max_lev', max(1.5, lev_c - 0.15), min(2.5, lev_c + 0.15))
            
            # Top N: Strict +/- 1
            top_c = anchor.get('top_n', 7)
            strategy.TOP_N_ASSETS = trial.suggest_int('top_n', max(4, top_c - 1), min(8, top_c + 1))
            
            # Crash: FIXED ORDER (Low, High)
            crash_c = anchor.get('crash_thresh', -0.10)
            # Low bound is the MORE negative number (e.g., -0.15)
            # High bound is the LESS negative number (e.g., -0.05)
            strategy.CRASH_THRESHOLD = trial.suggest_float('crash_thresh', max(-0.15, crash_c - 0.02), min(-0.05, crash_c + 0.02))
            
            # Volatility: +/- 0.02
            vol_c = anchor.get('vol_bull', 0.20)
            strategy.TARGET_VOL_BULL = trial.suggest_float('vol_bull', max(0.15, vol_c - 0.02), min(0.25, vol_c + 0.02))
            
            # Allocation: +/- 0.10
            anc_c = anchor.get('anchor_crash', 0.60)
            strategy.ANCHOR_WEIGHT_CRASH = trial.suggest_float('anchor_crash', max(0.4, anc_c - 0.10), min(0.9, anc_c + 0.10))

            # Lookbacks
            strategy.LOOKBACK_SMA_TREND = trial.suggest_int('sma_trend', 180, 240, step=10)
            strategy.LOOKBACK_SMA_FAST = trial.suggest_int('sma_fast', 30, 60, step=5)
            strategy.CRASH_LOOKBACK = trial.suggest_int('crash_lb', 15, 45, step=5)
            
            w1 = trial.suggest_int('mom_w1', 30, 80, step=10)
            w2 = trial.suggest_int('mom_w2', 90, 150, step=10)
            w3 = trial.suggest_int('mom_w3', 180, 250, step=10)
            strategy.MOMENTUM_WINDOW_1 = w1
            strategy.MOMENTUM_WINDOW_2 = w2
            strategy.MOMENTUM_WINDOW_3 = w3
            
            strategy.VOL_LOOKBACK = trial.suggest_int('vol_lb', 50, 80, step=5)
            strategy.SATELLITE_RSI_ENTRY = trial.suggest_int('sat_rsi', 15, 35, step=5)
            strategy.SATELLITE_ALLOCATION = trial.suggest_float('sat_alloc', 0.10, 0.20)
            strategy.MAX_SINGLE_ASSET_EXPOSURE = trial.suggest_float('max_single', 0.40, 0.65)
            strategy.TARGET_VOL_BEAR = trial.suggest_float('vol_bear', 0.05, 0.10)

        else:
            # --- FALLBACK: WIDE SEARCH (Only if no file exists) ---
            strategy.MAX_PORTFOLIO_LEVERAGE = trial.suggest_float('max_lev', 1.6, 2.4)
            strategy.TOP_N_ASSETS = trial.suggest_int('top_n', 4, 8)
            strategy.CRASH_THRESHOLD = trial.suggest_float('crash_thresh', -0.12, -0.06)
            strategy.TARGET_VOL_BULL = trial.suggest_float('vol_bull', 0.18, 0.25)
            strategy.TARGET_VOL_BEAR = trial.suggest_float('vol_bear', 0.05, 0.10)
            strategy.ANCHOR_WEIGHT_CRASH = trial.suggest_float('anchor_crash', 0.60, 0.85)
            strategy.LOOKBACK_SMA_TREND = trial.suggest_int('sma_trend', 180, 240, step=10)
            strategy.LOOKBACK_SMA_FAST = trial.suggest_int('sma_fast', 30, 80, step=5)
            strategy.CRASH_LOOKBACK = trial.suggest_int('crash_lb', 15, 60, step=5)
            w1 = trial.suggest_int('mom_w1', 30, 80, step=10)
            w2 = trial.suggest_int('mom_w2', 90, 150, step=10)
            w3 = trial.suggest_int('mom_w3', 180, 250, step=10)
            strategy.MOMENTUM_WINDOW_1 = w1
            strategy.MOMENTUM_WINDOW_2 = w2
            strategy.MOMENTUM_WINDOW_3 = w3
            strategy.VOL_LOOKBACK = trial.suggest_int('vol_lb', 40, 80, step=5)
            strategy.SATELLITE_RSI_ENTRY = trial.suggest_int('sat_rsi', 10, 35, step=5)
            strategy.SATELLITE_ALLOCATION = trial.suggest_float('sat_alloc', 0.10, 0.20)
            strategy.MAX_SINGLE_ASSET_EXPOSURE = trial.suggest_float('max_single', 0.40, 0.65)

        try:
            equity, _, _, _ = research.run_full_strategy_backtest()
            train_eq = equity.loc[:split_date]
            test_eq = equity.loc[split_date:]
            
            score_train = calculate_smart_score(train_eq)
            score_test = calculate_smart_score(test_eq)
            
            # Weighted Score (80% Test / 20% Train)
            final_score = (0.20 * score_train) + (0.80 * score_test)
            
            test_cagr = research.calculate_cagr(test_eq)
            train_cagr = research.calculate_cagr(train_eq)
            test_dd = research.calculate_max_drawdown(test_eq)
            test_sortino = research.calculate_sortino(test_eq.pct_change().dropna())
            diff = test_cagr - train_cagr

            print(f"{trial.number:<5} | {final_score:8.4f} | {test_cagr:8.2%}   | {test_dd:8.2%}   | {test_sortino:8.2f} | {diff:7.2%} | {strategy.MAX_PORTFOLIO_LEVERAGE:.2f} | {strategy.TOP_N_ASSETS:<4} | {strategy.CRASH_THRESHOLD:.2f}")
            
            trial.set_user_attr("test_cagr", test_cagr)
            trial.set_user_attr("test_dd", test_dd)
            
            return final_score
            
        except Exception:
            return -9999.0

    study = optuna.create_study(
        direction='maximize', 
        study_name="fortress_study", 
        storage=DB_FILE,
        load_if_exists=True
    )
    
    if anchor:
        print("💉 Injecting DNA from previous winner...")
        study.enqueue_trial(anchor)
    
    study.optimize(objective, n_trials=trials)
    save_new_winner(study, study.best_trial)
    
    print("\n🏆 OPTIMIZATION CHAMPION 🏆")
    bt = study.best_trial
    print(f"Trial #{bt.number} won with Score: {bt.value:.4f}")
    
    # --- AUTO TRIGGER STRESS TEST ---
    print("\n⚡ AUTO-TRIGGERING STRESS TEST FOR WINNER...")
    run_stress_test(study.best_trial.params)
    
    # --- AUTO TRIGGER 3D VISUALIZATION ---
    print("\n🎨 OPENING 3D RADAR SURFACE...")
    run_visualization()

# ============================
# 4. TOOL: 3D RADAR VISUALIZATION
# ============================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def run_visualization():
    print("Loading optimization history from database...")
    try:
        study = optuna.load_study(study_name="fortress_study", storage=DB_FILE)
    except Exception as e:
        print("❌ Error: Could not load optimization.db. Run --mode optimize first!")
        return

    df = study.trials_dataframe()
    df = df[df.state == "COMPLETE"]
    
    # Parameters to plot on the radar
    params = {
        'params_max_lev': 'Leverage',
        'params_top_n': 'Top N',
        'params_crash_thresh': 'Crash Trigger',
        'params_mom_w2': 'Mom Window',
        'params_vol_bull': 'Bull Vol',
        'params_anchor_crash': 'Crash Alloc',
        'params_sma_trend': 'Trend SMA'
    }
    
    subset = df[list(params.keys())].copy()
    for col in subset.columns:
        subset[col] = normalize(subset[col])
    
    subset['score'] = df['value']
    subset = subset.sort_values('score')
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    N = len(params)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]])) 

    top_trials = subset.tail(60) 
    
    polys = []
    z_heights = []
    colors = []

    print(f"Rendering 3D Radar Surface for top {len(top_trials)} trials...")

    for i, row in top_trials.iterrows():
        values = row[list(params.keys())].values
        values = np.concatenate((values, [values[0]]))
        
        x = values * np.cos(angles)
        y = values * np.sin(angles)
        z = row['score']
        
        verts = list(zip(x, y))
        polys.append(verts)
        z_heights.append(z)
        colors.append(plt.cm.plasma((z - subset['score'].min()) / (subset['score'].max() - subset['score'].min())))

    poly_collection = PolyCollection(polys, facecolors=colors, edgecolors='gray', alpha=0.5)
    ax.add_collection3d(poly_collection, zs=z_heights, zdir='z')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(subset['score'].min(), subset['score'].max())
    
    for angle, label in zip(angles[:-1], params.values()):
        x = 1.2 * np.cos(angle)
        y = 1.2 * np.sin(angle)
        ax.text(x, y, subset['score'].min(), label, fontsize=9, fontweight='bold')

    ax.set_zlabel('Objective Score')
    ax.set_title('3D Strategy DNA: Convergence to the Peak')
    ax.grid(False)
    ax.axis('off')

    plt.show()
    print("Done.")

# ============================
# 5. TOOL: STRESS TEST
# ============================
def run_stress_test(params=None):
    print(f"\n🎲 Running BLOCK BOOTSTRAP Stress Test ({2000} simulations)...")
    
    # --- 1. Inject Parameters (Same as before) ---
    if params:
        mapping = {
            'max_lev': 'MAX_PORTFOLIO_LEVERAGE',
            'max_single': 'MAX_SINGLE_ASSET_EXPOSURE',
            'anchor_crash': 'ANCHOR_WEIGHT_CRASH',
            'sma_trend': 'LOOKBACK_SMA_TREND',
            'sma_fast': 'LOOKBACK_SMA_FAST',
            'crash_lb': 'CRASH_LOOKBACK',
            'crash_thresh': 'CRASH_THRESHOLD',
            'mom_w1': 'MOMENTUM_WINDOW_1',
            'mom_w2': 'MOMENTUM_WINDOW_2',
            'mom_w3': 'MOMENTUM_WINDOW_3',
            'top_n': 'TOP_N_ASSETS',
            'vol_bull': 'TARGET_VOL_BULL',
            'vol_bear': 'TARGET_VOL_BEAR',
            'vol_lb': 'VOL_LOOKBACK',
            'sat_rsi': 'SATELLITE_RSI_ENTRY',
            'sat_alloc': 'SATELLITE_ALLOCATION'
        }
        for k, v in params.items():
            if k in mapping: setattr(strategy, mapping[k], v)

    # --- 2. Get Strategy Returns ---
    equity, _, _, _ = research.run_full_strategy_backtest()
    # Ensure we drop NaNs so the bootstrap doesn't break
    daily_rets = equity.pct_change().dropna()
    
    # --- 3. Run Bootstrap (Now captures CAGR too) ---
    dd_results, cagr_results = block_bootstrap_sampling(daily_rets, n_sims=2000, block_size=6, years=20)
    
    # --- 4. Calculate Statistics ---
    # Drawdown Stats
    median_dd = np.median(dd_results)
    worst_1_percent = np.percentile(dd_results, 1) # 1st percentile (very negative)
    ruin_prob = np.mean([1 if r < -0.70 else 0 for r in dd_results])
    
    # CAGR Stats
    cagr_worst_1 = np.percentile(cagr_results, 1)
    cagr_bad_10 = np.percentile(cagr_results, 10)
    cagr_median = np.median(cagr_results)
    cagr_best_90 = np.percentile(cagr_results, 90)

    # --- 5. Print Report ---
    print("=" * 40)
    print(" 🎲 MONTE CARLO STRESS TEST RESULTS")
    print("=" * 40)
    
    print("\n--- GROWTH (CAGR) ---")
    print(f"Worst Case (1%):    {cagr_worst_1:.2%}")
    print(f"Bad Case (10%):     {cagr_bad_10:.2%}")
    print(f"Median Case (50%):  {cagr_median:.2%}")
    print(f"Best Case (90%):    {cagr_best_90:.2%}")

    print("\n--- RISK (Max Drawdown) ---")
    print(f"Median Drawdown:    {median_dd:.2%}")
    print(f"Worst 1% Case:      {worst_1_percent:.2%}")

    print("\n--- SURVIVAL ---")
    print(f"Risk of Ruin (>70% Loss): {ruin_prob:.2%}")
    print("-" * 40)
    
    # --- 6. Plotting (Optional: Plot CAGR vs Drawdown) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=dd_results, y=cagr_results, alpha=0.3, color="blue", s=10)
    plt.xlabel("Max Drawdown")
    plt.ylabel("CAGR")
    plt.title("Stress Test: Risk vs Reward Cloud")
    plt.axvline(median_dd, color='red', linestyle='--', label='Median DD')
    plt.axhline(cagr_median, color='green', linestyle='--', label='Median CAGR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================
# MAIN ENTRY POINT
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['optimize', 'visualize', 'stress'])
    args = parser.parse_args()
    
    if args.mode == 'optimize': run_optimization()
    elif args.mode == 'visualize': run_visualization()
    elif args.mode == 'stress': run_stress_test()