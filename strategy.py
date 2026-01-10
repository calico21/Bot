# strategy.py

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================
# Global Configuration (AI-OPTIMIZED / AGGRESSIVE)
# ============================

# ============================
# Global Configuration (FINAL GOLD STANDARD)
# ============================

RISK_ASSETS = [
    'XLK', 'SMH', 'QQQ', 'XLC', 'XLY',
    'XLF', 'XLI', 'XHB', 'IYT', 'IYR',
    'XLE', 'XLV', 'XLP', 'XLU', 'XLB',
    'EEM', 'INDA', 'EWJ', 
    'GLD', 'SLV', 'DBC', 'USO'
]

SAFE_ASSETS = ['IEF', 'SHV', 'GLD', 'DBC'] 
MARKET_FILTER = 'SPY'
BOND_BENCHMARK = 'IEF'

# Regime Settings
LOOKBACK_SMA_TREND = 220           # Stability > Speed
LOOKBACK_SMA_FAST = 50
CRASH_LOOKBACK = 21
CRASH_THRESHOLD = -0.07

# --- FORTRESS GOLD SETTINGS ---
VOL_LOOKBACK = 63

# Safety Caps (Optimized for 18% CAGR + 0% Ruin)
MAX_SINGLE_ASSET_EXPOSURE = 0.50   # 50% Cap (Standard Institutional Limit)
MAX_PORTFOLIO_LEVERAGE = 2.00      # 2.0x (The Sweet Spot)

# Dynamic Volatility Targeting
TARGET_VOL_BULL = 0.20             
TARGET_VOL_BEAR = 0.08             

# Tail Risk
TAIL_VOL_CEILING = 0.30        
TAIL_VOL_ACCELERATION = 1.35   

# Leverage Bands
MAX_CALC_LEVERAGE = 2.0            
MIN_LEVERAGE = 0.5       

# Anchor Weights
ANCHOR_WEIGHT_BASE = 0.25   
ANCHOR_WEIGHT_BULL = 0.00   
ANCHOR_WEIGHT_BEAR = 0.35          
ANCHOR_WEIGHT_CRASH = 0.90         # 90% Cash (Allows 10% participation in rebounds)
ANCHOR_WEIGHT_HIGH_VOL = 0.40

# ============================
# 0. Helper: Market State Analysis
# ============================

class MarketState:
    """
    Centralizes market analysis. 
    LOGIC: Defaults to 'Bull' in the Gray Zone for maximum CAGR.
    """
    def __init__(self, prices, date, market_ticker='SPY', bond_ticker='IEF'):
        self.regime = "unknown"
        self.leverage_scalar = 1.0
        self.is_tail_risk = False
        self.correlation_stress = False
        
        if market_ticker not in prices.columns:
            return

        spy = prices[market_ticker].loc[:date].dropna()
        if len(spy) < LOOKBACK_SMA_TREND:
            return

        # --- 1. Pre-calculate Metrics ---
        current_price = spy.iloc[-1]
        
        # Volatility
        rets_spy = spy.pct_change()
        vol_21 = rets_spy.tail(21).std() * np.sqrt(252)
        vol_63 = rets_spy.tail(63).std() * np.sqrt(252)
        
        # Trend SMAs
        sma50 = spy.rolling(LOOKBACK_SMA_FAST).mean().iloc[-1]
        sma200 = spy.rolling(LOOKBACK_SMA_TREND).mean().iloc[-1]
        
        # Crash Detection
        prior_crash_price = spy.iloc[-CRASH_LOOKBACK]
        crash_drawdown = (current_price / prior_crash_price) - 1

        # --- 2. Determine Regime ---
        if crash_drawdown < CRASH_THRESHOLD:
            self.regime = "crash"
        elif vol_21 > 0.20:
            self.regime = "high_vol"
        elif current_price > sma50:
            self.regime = "bull"      # Clear Uptrend
        elif current_price < sma200:
            self.regime = "bear"      # Clear Downtrend
        else:
            # Gray Zone = Bullish (Aggressive)
            self.regime = "bull"      

        # --- 3. Tail Risk & Correlation Checks ---
        is_vol_accelerating = vol_21 > (vol_63 * TAIL_VOL_ACCELERATION)
        is_vol_critical = vol_21 > TAIL_VOL_CEILING
        is_downtrend = current_price < sma200
        
        # Correlation Panic
        if bond_ticker in prices.columns:
            ief = prices[bond_ticker].loc[:date].dropna()
            if len(ief) > 63:
                rets_ief = ief.pct_change().tail(63)
                aligned = pd.concat([rets_spy.tail(63), rets_ief], axis=1).dropna()
                if len(aligned) > 20:
                    corr = aligned.iloc[:,0].corr(aligned.iloc[:,1])
                    if corr > 0.40 and current_price < sma50:
                         self.correlation_stress = True

        if (is_downtrend and is_vol_accelerating) or is_vol_critical:
            self.is_tail_risk = True

        # --- 4. Calculate Leverage ---
        if self.regime in ["crash", "high_vol"] or vol_21 > 0.30:
            self.leverage_scalar = 1.0
        elif vol_21 > 0:
            target_vol = TARGET_VOL_BULL if self.regime == "bull" else TARGET_VOL_BEAR
            base_lev = target_vol / vol_21
            cap = MAX_CALC_LEVERAGE if self.regime == "bull" else 1.2
            self.leverage_scalar = float(np.clip(base_lev, MIN_LEVERAGE, cap))


# ============================
# 1. Sub-Strategy: Momentum Engine
# ============================

class FortressSubStrategy:
    def __init__(self, name, vol_window, top_n, risk_assets=None):
        self.name = name
        self.vol_window = vol_window
        self.top_n = top_n 
        self.risk_assets = risk_assets or RISK_ASSETS

    # --- HRP Utilities ---
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
        valid_assets = list(returns_df.columns)
        if len(valid_assets) < 2: 
            return {t: 1.0/len(valid_assets) for t in valid_assets}

        try:
            cov = returns_df.cov()
            corr = returns_df.corr()
            dist = np.sqrt((1 - corr) / 2)
            dist_array = squareform(dist)
            link = sch.linkage(dist_array, 'ward') 
            sort_ix = self._get_quasi_diag(link)
            sort_ix = [returns_df.columns[i] for i in sort_ix]
            cov = cov.loc[sort_ix, sort_ix]
            weights = self._get_rec_bipart(cov, sort_ix)
            return weights.to_dict()
        except:
            return {t: 1.0/len(valid_assets) for t in valid_assets}

    def get_attack_portfolio(self, prices: pd.DataFrame, date: pd.Timestamp, regime: str):
        # 1. Filter Data
        valid_cols = [c for c in self.risk_assets if c in prices.columns]
        hist = prices[valid_cols].loc[:date]
        
        # 2. Vectorized Momentum
        lookbacks = [63, 126, 189]
        momentum_scores = pd.Series(0.0, index=valid_cols)
        valid_lb_counts = pd.Series(0.0, index=valid_cols)

        recent_returns = hist.pct_change().tail(22).dropna()
        if len(recent_returns) < 21: return {}
        
        vol_21 = recent_returns.std()
        current_prices = hist.iloc[-1]
        
        for lb in lookbacks:
            if len(hist) > lb:
                past_prices = hist.iloc[-(lb+1)]
                moms = (current_prices / past_prices) - 1.0
                safe_vols = vol_21.replace(0, np.inf)
                scores = moms / safe_vols
                momentum_scores += scores
                valid_lb_counts += 1

        final_scores = momentum_scores / valid_lb_counts.replace(0, 1)
        final_scores = final_scores[valid_lb_counts > 0]

        # 3. Select Top Assets
        positive_assets = final_scores[final_scores > 0]
        if positive_assets.empty:
            return {}

        candidate_count = max(self.top_n, 5)
        top_assets = positive_assets.nlargest(candidate_count).index.tolist()

        if len(top_assets) == 1:
            return {top_assets[0]: 1.0}

        # 4. HRP Allocation
        subset_returns = hist[top_assets].iloc[-252:].pct_change().dropna()
        hrp_weights = self.get_hrp_weights(subset_returns)

        # 5. Bull Market Tilt
        if regime == "bull":
            tilted_weights = {}
            total_score_sum = final_scores[top_assets].sum()
            for t, w_hrp in hrp_weights.items():
                if total_score_sum > 0:
                    w_mom = final_scores[t] / total_score_sum
                    tilted_weights[t] = (w_hrp * 0.5) + (w_mom * 0.5)
                else:
                    tilted_weights[t] = w_hrp
            w_sum = sum(tilted_weights.values())
            return {k: v/w_sum for k,v in tilted_weights.items()} if w_sum > 0 else hrp_weights
            
        return hrp_weights


# ============================
# 2. Mean Reversion Satellite (Z-Score + RSI)
# ============================

class MeanReversionSatellite:
    """
    The 'Precision' Satellite.
    Logic: RSI(2) + Z-Score Mean Reversion.
    Universe: Uses EXISTING Risk Assets (No new tickers).
    """
    def __init__(self, universe):
        self.universe = universe  # Use the Main Strategy's RISK_ASSETS
        self.lookback = 22        
        self.max_positions = 3    

    def get_zscore(self, series, window=20):
        roll_mean = series.rolling(window).mean()
        roll_std = series.rolling(window).std()
        zscore = (series - roll_mean) / roll_std
        return zscore

    def calculate_rsi(self, series, period=2):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_signal(self, prices, date):
        available = [t for t in self.universe if t in prices.columns]
        if not available:
            return {}

        data = prices[available].loc[:date].tail(self.lookback + 5)
        if len(data) < self.lookback:
            return {}

        candidates = {}

        for ticker in available:
            try:
                series = data[ticker]
                rsi = self.calculate_rsi(series, 2).iloc[-1]
                zscore = self.get_zscore(series, 20).iloc[-1]
                
                # STRICT ENTRY: Only buy deep crashes
                if rsi < 10 and zscore < -2.5:
                    candidates[ticker] = rsi + zscore 
            except:
                continue

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])
        top_picks = sorted_candidates[:self.max_positions]
        
        weights = {}
        if top_picks:
            weight_each = 1.0 / len(top_picks)
            for ticker, score in top_picks:
                weights[ticker] = weight_each
        
        return weights


# ============================
# 3. Main Strategy (Integrated)
# ============================

class MonthlyFortressStrategy:
    def __init__(self):
        self.risk_assets = RISK_ASSETS
        self.safe_assets = SAFE_ASSETS
        self.market_filter = MARKET_FILTER
        self.bond_benchmark = BOND_BENCHMARK
        
        self.sub_strategies = [FortressSubStrategy("ensemble_core", 63, 5)]
        self.satellite_engine = MeanReversionSatellite(self.risk_assets)
        self.satellite_assets = [] 

    def _is_asset_healthy(self, ticker, prices, date):
        if ticker not in prices.columns: return False
        series = prices[ticker].loc[:date].dropna()
        if len(series) < 200: return True 
        sma200 = series.rolling(200).mean().iloc[-1]
        return series.iloc[-1] > sma200

    def _allocate_safe(self, target, prices, date):
        final_target = {}
        valid_targets = {t: w for t, w in target.items() if t in prices.columns and not pd.isna(prices.at[date, t])}
        total_valid_weight = sum(valid_targets.values())
        missing_weight = 1.0 - total_valid_weight
        final_target = valid_targets.copy()

        if missing_weight > 0.01:
            if 'SHV' in prices.columns: 
                final_target['SHV'] = final_target.get('SHV', 0) + missing_weight
            elif 'IEF' in prices.columns:
                 final_target['IEF'] = final_target.get('IEF', 0) + missing_weight

        return list(final_target.items())

    def _get_adaptive_defense(self, prices, date):
        candidates = [t for t in self.safe_assets if t in prices.columns]
        scores = {}
        for t in candidates:
            if t == 'SHV': 
                scores[t] = 0.0 
                continue
            series = prices[t].loc[:date].dropna()
            if len(series) < 63: continue
            
            p_now = series.iloc[-1]
            r1 = (p_now / series.iloc[-21]) - 1 if len(series) > 21 else 0
            r3 = (p_now / series.iloc[-63]) - 1 if len(series) > 63 else 0
            r6 = (p_now / series.iloc[-126]) - 1 if len(series) > 126 else 0
            scores[t] = (r1 + r3 + r6) / 3.0

        valid_assets = {k: v for k, v in scores.items() if v > 0 or k == 'SHV'}
        top_assets = sorted(valid_assets, key=valid_assets.get, reverse=True)[:2]
        
        if not top_assets or (top_assets[0] != 'SHV' and scores[top_assets[0]] < 0.0):
            return [('SHV', 1.0)]
            
        if len(top_assets) == 1:
            return [(top_assets[0], 1.0)]
        else:
            return [(top_assets[0], 0.60), (top_assets[1], 0.40)]

    def _get_defense_portfolio(self, prices, date, regime):
        return self._get_adaptive_defense(prices, date)

    def _get_anchor_weight_for_regime(self, regime: str) -> float:
        if regime == "bull": return ANCHOR_WEIGHT_BULL
        if regime == "bear": return ANCHOR_WEIGHT_BEAR
        if regime == "crash": return ANCHOR_WEIGHT_CRASH
        if regime == "high_vol": return ANCHOR_WEIGHT_HIGH_VOL
        return ANCHOR_WEIGHT_BASE

    def detect_regime(self, prices: pd.DataFrame, date: pd.Timestamp) -> str:
        state = MarketState(prices, date, self.market_filter, self.bond_benchmark)
        return state.regime

    def get_signal(self, prices: pd.DataFrame, date: pd.Timestamp):
        state = MarketState(prices, date, self.market_filter, self.bond_benchmark)
        
        core_portfolio = {}

        if state.is_tail_risk:
            safe_alloc = dict(self._allocate_safe({'SHV': 1.0}, prices, date))
            return list(safe_alloc.items())

        elif state.regime in ["crash", "bear", "high_vol", "unknown"]:
            core_portfolio = dict(self._get_adaptive_defense(prices, date))
        
        else:
            s = self.sub_strategies[0]
            raw_attack = s.get_attack_portfolio(prices, date, state.regime)
            
            if not raw_attack:
                core_portfolio = dict(self._get_defense_portfolio(prices, date, "bear"))
            else:
                anchor_weight = self._get_anchor_weight_for_regime(state.regime)
                
                for t, w in raw_attack.items():
                    core_portfolio[t] = w * (1 - anchor_weight)
                
                if anchor_weight > 0:
                    anchor_ief = anchor_weight * 0.5
                    anchor_gld = anchor_weight * 0.5
                    if not self._is_asset_healthy('IEF', prices, date):
                        core_portfolio['SHV'] = core_portfolio.get('SHV', 0) + anchor_ief
                    else:
                        core_portfolio['IEF'] = core_portfolio.get('IEF', 0) + anchor_ief
                    core_portfolio['GLD'] = core_portfolio.get('GLD', 0) + anchor_gld
                
                lev = state.leverage_scalar
                if state.correlation_stress:
                     lev = min(lev, 1.0) 

                core_portfolio = {k: v * lev for k, v in core_portfolio.items()}
                
                for t in core_portfolio:
                    if core_portfolio[t] > MAX_SINGLE_ASSET_EXPOSURE:
                        core_portfolio[t] = MAX_SINGLE_ASSET_EXPOSURE

        satellite_pct = 0.15
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

        total_exposure = sum(final_portfolio.values())
        if total_exposure > MAX_PORTFOLIO_LEVERAGE:
            scale_factor = MAX_PORTFOLIO_LEVERAGE / total_exposure
            final_portfolio = {k: v * scale_factor for k, v in final_portfolio.items()}
            
        return list(final_portfolio.items())