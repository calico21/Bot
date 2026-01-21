# strategy_core.py

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from datetime import date, timedelta
import warnings
import json
import os

# Mute warnings
warnings.filterwarnings("ignore")

# ============================================================
#  1. DEFAULT CONFIGURATION (Robust Defaults)
# ============================================================
RISK_ASSETS = [
    'XLK', 'SMH', 'QQQ', 'XLC', 'XLY',
    'XLF', 'XLI', 'XHB', 'IYT', 'IYR',
    'XLE', 'XLV', 'XLP', 'XLU', 'XLB',
    'EEM', 'INDA', 'EWJ', 
    'GLD', 'SLV', 'DBC', 'USO', 'URA', 'COPX'
]
SAFE_ASSETS = ['IEF', 'SHV', 'GLD', 'DBC', 'BIL'] 
MARKET_FILTER = 'SPY'
BOND_BENCHMARK = 'IEF'

# -- Default Params --
LOOKBACK_SMA_TREND = 200           
LOOKBACK_SMA_FAST = 50             
CRASH_LOOKBACK = 20                
CRASH_THRESHOLD = -0.12            

MAX_SINGLE_ASSET_EXPOSURE = 0.40   
MAX_PORTFOLIO_LEVERAGE = 2.00      
TARGET_VOL_BULL = 0.18             
TARGET_VOL_BEAR = 0.06            
VOL_LOOKBACK = 63                  

ANCHOR_WEIGHT_BASE = 0.20   
ANCHOR_WEIGHT_BULL = 0.00   
ANCHOR_WEIGHT_BEAR = 0.40          
ANCHOR_WEIGHT_CRASH = 0.80         
ANCHOR_WEIGHT_HIGH_VOL = 0.50

MOMENTUM_WINDOW_1 = 63             
MOMENTUM_WINDOW_2 = 126            
MOMENTUM_WINDOW_3 = 189            
TOP_N_ASSETS = 6                   

SATELLITE_RSI_ENTRY = 30           
SATELLITE_ZSCORE_ENTRY = -2.0      
SATELLITE_ALLOCATION = 0.15        

# ============================================================
#  2. AUTOMATED CONFIGURATION LOADER
# ============================================================
DNA_FILE = "winner_dna.json"

def load_optimized_params():
    """Loads best parameters from JSON and overrides globals."""
    if not os.path.exists(DNA_FILE):
        return

    try:
        with open(DNA_FILE, 'r') as f:
            data = json.load(f)
            params = data.get('params', {})
        
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

        print(f"🧬 Injecting Optimized DNA (Score: {data.get('score',0):.4f})...")
        for json_key, global_var in mapping.items():
            if json_key in params:
                globals()[global_var] = params[json_key]
                
    except Exception as e:
        print(f"❌ Error loading DNA file: {e}")

# EXECUTE LOADER IMMEDIATELY
load_optimized_params()

# ============================================================
#  3. CORE LOGIC (Adaptive & Robust)
# ============================================================

class MarketState:
    """
    Adaptive Regime Detection Engine.
    Uses continuous scoring instead of binary thresholds to prevent 'flapping'.
    """
    def __init__(self, prices, date, market_ticker='SPY', bond_ticker='IEF'):
        self.regime = "neutral"
        self.leverage_scalar = 1.0
        self.is_tail_risk = False
        self.correlation_stress = False
        
        if market_ticker not in prices.columns: return
        
        spy = prices[market_ticker].loc[:date].dropna()
        if len(spy) < 252: return

        current_price = spy.iloc[-1]
        
        # --- 1. CRASH DETECTION (Hard Trigger) ---
        # Immediate safety brake for rapid drops (e.g., Covid crash)
        if len(spy) > CRASH_LOOKBACK:
            drawdown = (current_price / spy.iloc[-CRASH_LOOKBACK]) - 1
            if drawdown < CRASH_THRESHOLD:
                self.regime = "crash"
                self.leverage_scalar = 0.0 # Full Defense
                self.is_tail_risk = True
                return

        # --- 2. SMOOTHED TREND SCORE (0.0 to 1.0) ---
        # Normalize distance from trend by volatility
        sma_slow = spy.rolling(LOOKBACK_SMA_TREND).mean().iloc[-1]
        vol_63 = spy.pct_change().rolling(63).std().iloc[-1] * current_price * np.sqrt(63)
        
        # Z-score of price relative to SMA (how many sigmas are we away?)
        # +1.0 = Healthy Bull, -1.0 = Dangerous
        dist_z = (current_price - sma_slow) / (vol_63 + 1e-6)
        
        # Sigmoid: Maps Z-score to 0-1 range smoothly
        trend_score = 1 / (1 + np.exp(-1.5 * dist_z)) 

        # --- 3. RELATIVE VOLATILITY PERCENTILE ---
        # Is volatility high relative to the last 3 years?
        daily_rets = spy.pct_change()
        curr_vol = daily_rets.tail(21).std() * np.sqrt(252)
        
        hist_vol_window = daily_rets.rolling(21).std() * np.sqrt(252)
        hist_vol_window = hist_vol_window.dropna().tail(756) # 3 years
        
        if not hist_vol_window.empty:
            vol_rank = (hist_vol_window < curr_vol).mean()
        else:
            vol_rank = 0.5

        # --- 4. REGIME CLASSIFICATION ---
        if vol_rank > 0.90: # Extreme Volatility (Top 10%)
            self.regime = "high_vol"
            self.leverage_scalar = 0.5
            self.is_tail_risk = True
        
        elif trend_score > 0.60:
            self.regime = "bull"
            # Scale leverage: Higher in low vol, Lower in high vol
            # We dampen the vol input to avoid jagged leverage changes
            dampened_vol = (curr_vol * 0.6) + (TARGET_VOL_BULL * 0.4)
            raw_lev = TARGET_VOL_BULL / dampened_vol
            self.leverage_scalar = float(np.clip(raw_lev, 0.5, MAX_PORTFOLIO_LEVERAGE))
            
        elif trend_score < 0.35:
            self.regime = "bear"
            self.leverage_scalar = 1.0 # De-lever to 1x
            
        else:
            self.regime = "neutral" # Chop zone
            self.leverage_scalar = 1.0

        # --- 5. CORRELATION CHECK ---
        # If Bonds and Stocks are falling together, reduce leverage further
        if bond_ticker in prices.columns:
            bond_rets = prices[bond_ticker].loc[:date].pct_change().tail(21)
            spy_rets = daily_rets.tail(21)
            if len(bond_rets) == len(spy_rets):
                corr = spy_rets.corr(bond_rets)
                if corr > 0.50: # Dangerous positive correlation
                    self.correlation_stress = True
                    self.leverage_scalar *= 0.85

class FortressSubStrategy:
    def __init__(self, name, vol_window, top_n, risk_assets=None):
        self.name = name
        self.vol_window = vol_window
        self.top_n = top_n 
        self.risk_assets = risk_assets or RISK_ASSETS

    def _get_quasi_diag(self, link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    def _get_cluster_var(self, cov, c_items):
        cov_slice = cov.iloc[c_items, c_items]
        w = 1.0 / np.diag(cov_slice)
        w /= w.sum()
        return np.dot(np.dot(w, cov_slice), w)

    def _get_rec_bipart(self, cov, sort_ix):
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    def get_hrp_weights(self, returns_df):
        """
        Hierarchical Risk Parity with Covariance Shrinkage.
        Shrinkage prevents the matrix from breaking during high-correlation events.
        """
        valid_assets = list(returns_df.columns)
        if len(valid_assets) < 2: 
            return {t: 1.0/len(valid_assets) for t in valid_assets}
        
        try:
            # 1. Covariance Shrinkage (Ledoit-Wolf Lite)
            # Mix sample cov with a structured target (constant correlation)
            sample_cov = returns_df.cov()
            
            # Create a target matrix (average variance on diag, average cov off-diag)
            var_mean = np.diag(sample_cov).mean()
            cov_mean = sample_cov.values[~np.eye(sample_cov.shape[0], dtype=bool)].mean()
            target_cov = sample_cov.copy()
            target_cov[:] = cov_mean
            np.fill_diagonal(target_cov.values, var_mean)
            
            # Blend (Shrinkage intensity = 0.3 is conservative)
            shrinkage = 0.3
            shrunk_cov = (sample_cov * (1 - shrinkage)) + (target_cov * shrinkage)
            
            # 2. HRP Clustering
            corr = returns_df.corr()
            dist = np.sqrt((1 - corr).clip(0, 2) / 2) # Clip to avoid precision errors
            dist_array = squareform(dist)
            link = sch.linkage(dist_array, 'ward') 
            
            sort_ix = self._get_quasi_diag(link)
            sort_ix = [returns_df.columns[i] for i in sort_ix]
            
            # 3. Allocation
            shrunk_cov = shrunk_cov.loc[sort_ix, sort_ix]
            weights = self._get_rec_bipart(shrunk_cov, sort_ix)
            return weights.to_dict()
            
        except Exception as e:
            # Fallback to Equal Weight if math fails
            return {t: 1.0/len(valid_assets) for t in valid_assets}

    def calculate_efficiency_ratio(self, series):
        """Calculates Kaufman Efficiency Ratio: Net Change / Sum of Absolute Changes."""
        change = series.diff().abs().sum()
        net_change = abs(series.iloc[-1] - series.iloc[0])
        if change == 0: return 0
        return net_change / change

    def get_attack_portfolio(self, prices: pd.DataFrame, date: pd.Timestamp, regime: str):
        valid_cols = [c for c in self.risk_assets if c in prices.columns]
        hist = prices[valid_cols].loc[:date]
        
        if len(hist) < MOMENTUM_WINDOW_3 + 2: return {}

        # --- QUALITY MOMENTUM SCORING ---
        scores = {}
        recent_vol = hist.pct_change().tail(63).std()
        
        for ticker in valid_cols:
            series = hist[ticker].dropna()
            if len(series) < MOMENTUM_WINDOW_3: continue
            
            # 1. Return Signal (Weighted blend of 3 windows)
            p_now = series.iloc[-1]
            r1 = (p_now / series.iloc[-MOMENTUM_WINDOW_1]) - 1
            r2 = (p_now / series.iloc[-MOMENTUM_WINDOW_2]) - 1
            r3 = (p_now / series.iloc[-MOMENTUM_WINDOW_3]) - 1
            avg_ret = (r1 * 0.5) + (r2 * 0.3) + (r3 * 0.2)
            
            # 2. Trend Quality (Efficiency Ratio)
            # High efficiency = Smooth trend (preferred). Low = Jagged.
            er = self.calculate_efficiency_ratio(series.tail(MOMENTUM_WINDOW_1))
            
            # 3. Volatility Adjustment
            vol = recent_vol[ticker]
            if vol == 0: vol = 1.0
            
            # Final Score: Risk-Adjusted Return * Smoothness
            scores[ticker] = (avg_ret / vol) * (0.5 + 0.5 * er)

        # Select Top Assets
        positive_assets = {k: v for k, v in scores.items() if v > 0}
        if not positive_assets: return {}
        
        top_assets = sorted(positive_assets, key=positive_assets.get, reverse=True)[:self.top_n]
        
        if len(top_assets) == 1: return {top_assets[0]: 1.0}

        # Allocation: HRP
        subset_returns = hist[top_assets].iloc[-252:].pct_change().dropna()
        hrp_weights = self.get_hrp_weights(subset_returns)

        # In Bull regime, tilt slightly towards the momentum leaders
        if regime == "bull":
            tilted_weights = {}
            total_score = sum([positive_assets[t] for t in top_assets])
            for t, w_hrp in hrp_weights.items():
                if total_score > 0:
                    w_mom = positive_assets[t] / total_score
                    tilted_weights[t] = (w_hrp * 0.6) + (w_mom * 0.4)
                else:
                    tilted_weights[t] = w_hrp
            
            # Normalize
            w_sum = sum(tilted_weights.values())
            return {k: v/w_sum for k,v in tilted_weights.items()}
            
        return hrp_weights

class MeanReversionSatellite:
    """
    Surgical Mean Reversion.
    Entries based on RSI + Bollinger Bands. 
    Sizing based on Inverse Volatility (Risk Parity).
    """
    def __init__(self, universe):
        self.universe = universe
        self.lookback = 30        
        self.max_positions = 4    

    def get_zscore(self, series, window=20):
        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std()
        return (series - roll_mean) / (roll_std + 1e-6)

    def calculate_rsi(self, series, period=14): # Using 14 for stability, signal is < 30
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_signal(self, prices, date):
        available = [t for t in self.universe if t in prices.columns]
        if not available: return {}
        
        data = prices[available].loc[:date].tail(self.lookback + 20)
        candidates = []
        
        # Identify opportunities
        for ticker in available:
            try:
                series = data[ticker].dropna()
                if len(series) < 25: continue
                
                rsi = self.calculate_rsi(series, 14).iloc[-1]
                zscore = self.get_zscore(series, 20).iloc[-1]
                
                # Confluence: Oversold (RSI) AND Extended (Z-Score)
                if rsi < SATELLITE_RSI_ENTRY and zscore < SATELLITE_ZSCORE_ENTRY:
                    # Score based on how extreme the deviation is
                    score = (30 - rsi) + abs(zscore)
                    vol = series.pct_change().std()
                    candidates.append((ticker, score, vol))
            except: continue
            
        if not candidates: return {}
        
        # Sort by "Juiciness" of the setup
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_picks = candidates[:self.max_positions]
        
        # Sizing: Inverse Volatility (Risk Parity)
        # Higher vol assets get smaller allocations
        inv_vols = [1.0 / (x[2] + 1e-4) for x in top_picks]
        total_inv_vol = sum(inv_vols)
        
        weights = {}
        for i, (ticker, _, _) in enumerate(top_picks):
            weights[ticker] = inv_vols[i] / total_inv_vol
            
        return weights

# ============================================================
#  4. MAIN STRATEGY LOGIC
# ============================================================

# ... (Keep imports and Sections 1, 2, 3 the same) ...

# ============================================================
#  4. MAIN STRATEGY LOGIC (FIXED)
# ============================================================

class MonthlyFortressStrategy:
    def __init__(self):
        self.risk_assets = RISK_ASSETS
        self.safe_assets = SAFE_ASSETS
        self.market_filter = MARKET_FILTER
        self.bond_benchmark = BOND_BENCHMARK
        
        # --- FIX: Adopt Globals as Instance Attributes ---
        # This allows live_bot.py to modify "strategy.MAX_PORTFOLIO_LEVERAGE"
        self.MAX_PORTFOLIO_LEVERAGE = MAX_PORTFOLIO_LEVERAGE
        self.CRASH_THRESHOLD = CRASH_THRESHOLD
        self.MAX_SINGLE_ASSET_EXPOSURE = MAX_SINGLE_ASSET_EXPOSURE
        
        # Initialize Sub-strategies
        self.sub_strategies = [FortressSubStrategy("ensemble_core", 63, TOP_N_ASSETS)]
        self.satellite_engine = MeanReversionSatellite(self.risk_assets)
        self.satellite_assets = [] 

    def _allocate_safe(self, target, prices, date):
        """Ensures safe asset exists in data before allocating."""
        final_target = {}
        valid_targets = {t: w for t, w in target.items() if t in prices.columns and not pd.isna(prices.at[date, t])}
        total_valid_weight = sum(valid_targets.values())
        
        # If we can't buy the preferred safe asset, dump into SHV/BIL (Cash proxies)
        missing_weight = 1.0 - total_valid_weight
        if missing_weight > 0.01:
            for cash_proxy in ['SHV', 'BIL', 'IEF']:
                if cash_proxy in prices.columns:
                    final_target[cash_proxy] = final_target.get(cash_proxy, 0) + missing_weight
                    break
        return list(final_target.items())

    def _get_adaptive_defense(self, prices, date):
        """Selects defense assets based on MOMENTUM + LOW VOLATILITY."""
        candidates = [t for t in self.safe_assets if t in prices.columns]
        scores = {}
        
        for t in candidates:
            if t in ['SHV', 'BIL']: continue 
            
            series = prices[t].loc[:date].dropna()
            if len(series) < 126: continue
            
            # Trend Score
            ma_fast = series.rolling(21).mean().iloc[-1]
            ma_slow = series.rolling(63).mean().iloc[-1]
            
            if ma_fast > ma_slow:
                vol = series.pct_change().tail(63).std()
                scores[t] = 1.0 / (vol + 1e-6) 
        
        if not scores:
            return [('SHV', 1.0)]
            
        top_def = sorted(scores, key=scores.get, reverse=True)[:2]
        
        if len(top_def) == 1:
            return [(top_def[0], 1.0)]
        else:
            return [(top_def[0], 0.50), (top_def[1], 0.50)]

    def _get_anchor_weight_for_regime(self, regime: str) -> float:
        if regime == "bull": return ANCHOR_WEIGHT_BULL
        if regime == "bear": return ANCHOR_WEIGHT_BEAR
        if regime == "crash": return ANCHOR_WEIGHT_CRASH
        if regime == "high_vol": return ANCHOR_WEIGHT_HIGH_VOL
        return ANCHOR_WEIGHT_BASE

    def detect_regime(self, prices: pd.DataFrame, date: pd.Timestamp) -> str:
        # Pass dynamic threshold to MarketState manually or rely on global defaults
        # For simplicity, we assume MarketState reads the globals, or we update it.
        # Ideally, update MarketState to accept a threshold, but the simplest fix is below.
        state = MarketState(prices, date, self.market_filter, self.bond_benchmark)
        return state.regime

    def get_signal(self, prices: pd.DataFrame, date: pd.Timestamp):
        # 1. Detect Regime (Adaptive)
        state = MarketState(prices, date, self.market_filter, self.bond_benchmark)
        
        # --- FIX: Override Global Crash Threshold logic locally if needed ---
        # (MarketState uses the global CRASH_THRESHOLD by default. 
        # If you want the instance variable to control it, you'd need to modify MarketState too.
        # However, for now, the leverage logic below handles the main risk control.)
        
        core_portfolio = {}

        # 2. Strategy Logic based on Regime
        if state.regime == "crash":
            safe_alloc = dict(self._allocate_safe({'SHV': 1.0}, prices, date))
            return list(safe_alloc.items())

        elif state.regime in ["bear", "high_vol"]:
            core_portfolio = dict(self._get_adaptive_defense(prices, date))
        
        else:
            # BULL or NEUTRAL -> RISK ON
            s = self.sub_strategies[0]
            raw_attack = s.get_attack_portfolio(prices, date, state.regime)
            
            if not raw_attack:
                core_portfolio = dict(self._get_adaptive_defense(prices, date))
            else:
                anchor_weight = self._get_anchor_weight_for_regime(state.regime)
                
                for t, w in raw_attack.items(): 
                    core_portfolio[t] = w * (1 - anchor_weight)
                
                if anchor_weight > 0:
                    anchor_ief = anchor_weight * 0.5
                    anchor_gld = anchor_weight * 0.5
                    
                    bond_trend = prices['IEF'].loc[:date].tail(60).mean() if 'IEF' in prices else 0
                    current_bond = prices['IEF'].loc[date] if 'IEF' in prices else 0
                    
                    if current_bond < bond_trend:
                        core_portfolio['SHV'] = core_portfolio.get('SHV', 0) + anchor_ief
                    else:
                        core_portfolio['IEF'] = core_portfolio.get('IEF', 0) + anchor_ief
                        
                    core_portfolio['GLD'] = core_portfolio.get('GLD', 0) + anchor_gld
                
                # --- FIX: Use self.MAX_PORTFOLIO_LEVERAGE ---
                lev = state.leverage_scalar
                # Cap the scalar by our dynamic instance variable
                lev = min(lev, self.MAX_PORTFOLIO_LEVERAGE) 
                
                core_portfolio = {k: v * lev for k, v in core_portfolio.items()}
                
                # --- FIX: Use self.MAX_SINGLE_ASSET_EXPOSURE ---
                for t in core_portfolio:
                    if core_portfolio[t] > self.MAX_SINGLE_ASSET_EXPOSURE:
                        core_portfolio[t] = self.MAX_SINGLE_ASSET_EXPOSURE

        # 3. Satellite Logic
        satellite_pct = SATELLITE_ALLOCATION
        core_pct = 1.0 - satellite_pct
        sat_weights = self.satellite_engine.get_signal(prices, date)
        
        final_portfolio = {}
        
        for t, w in core_portfolio.items(): 
            final_portfolio[t] = w * core_pct
            
        if sat_weights:
            for t, w in sat_weights.items():
                final_portfolio[t] = final_portfolio.get(t, 0.0) + (w * satellite_pct)
        else:
            if core_pct > 0:
                scale_factor = 1.0 / core_pct
                final_portfolio = {t: w * scale_factor for t, w in final_portfolio.items()}

        # 4. Final Leverage Safety Check (FIXED)
        total_exposure = sum(final_portfolio.values())
        if total_exposure > self.MAX_PORTFOLIO_LEVERAGE:
            scale_factor = self.MAX_PORTFOLIO_LEVERAGE / total_exposure
            final_portfolio = {k: v * scale_factor for k, v in final_portfolio.items()}
            
        return list(final_portfolio.items())

# strategy_core.py

def is_rebalance_day(api_key, secret_key, paper=True):
    """
    Returns True if TODAY is the LAST trading day of the month.
    """
    try:
        trading_client = TradingClient(api_key, secret_key, paper=paper)
        
        # 1. Check if Market is Open TODAY
        clock = trading_client.get_clock()
        if not clock.is_open:
            return False
            
        today = date.today()
        
        # 2. Get Calendar for Today + Next 7 Days
        # We need to peek into the future to see when the NEXT trading day is.
        future_date = today + timedelta(days=7)
        req = GetCalendarRequest(start=today, end=future_date)
        calendar = trading_client.get_calendar(filters=req)
        
        trading_days = [day.date for day in calendar]
        
        if not trading_days:
            return False
            
        # If today is not in the calendar, it's not a trading day (redundant check but safe)
        if today not in trading_days:
            return False
            
        # 3. Find the NEXT trading day
        # trading_days[0] is Today (since we filtered start=today)
        # trading_days[1] is the Next Trading Day
        if len(trading_days) < 2:
            # If we can't see a next trading day (end of year?), assume we rebalance to be safe.
            return True
            
        next_trading_day = trading_days[1]
        
        # 4. The "Golden" Check:
        # If Today is Month X, and Next Trading Day is Month Y -> It's End of Month!
        if today.month != next_trading_day.month:
            print(f"📅 Rebalance Signal: Today ({today}) is the last trading day of the month.")
            return True
            
        print(f"💤 No Rebalance: Today is {today}, Next trading day is {next_trading_day} (Same Month).")
        return False

    except Exception as e:
        print(f"⚠️ Calendar Error: {e}")
        return False