# strategy.py

import pandas as pd
import numpy as np

# ============================
# Global Configuration
# ============================

# Risk Assets (The "Attack" Universe)
# Added 'XLE' (Energy) to catch inflation rallies where Tech fails
RISK_ASSETS = ['XLK', 'SMH', 'XLF', 'XLV', 'XLI', 'XLY', 'QQQ', 'XLE']

# Safe Assets (The "Defense" Universe)
# IEF: Bonds, SHV: Cash, GLD: Gold, DBC: Commodities
SAFE_ASSETS = ['IEF', 'SHV', 'GLD', 'DBC']

MARKET_FILTER = 'SPY'
BOND_BENCHMARK = 'IEF'

# Regime Settings
LOOKBACK_SMA_TREND = 200
LOOKBACK_SMA_FAST = 50
CRASH_LOOKBACK = 21
CRASH_THRESHOLD = -0.07

# Volatility Settings
VOL_LOOKBACK = 63
TARGET_VOL = 0.15
MAX_LEVERAGE = 1.5
MIN_LEVERAGE = 0.5

# Portfolio Construction
ANCHOR_WEIGHT = 0.15

# ============================
# Sub-Strategy: Momentum Engine
# ============================

class FortressSubStrategy:
    def __init__(self, name, momentum_window, vol_window, top_n, risk_assets=None):
        self.name = name
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.top_n = top_n
        self.risk_assets = risk_assets or RISK_ASSETS

    def get_attack_portfolio(self, prices: pd.DataFrame, date: pd.Timestamp):
        window = prices.loc[:date]
        moms = {}
        for t in self.risk_assets:
            if t not in window.columns: continue
            series = window[t].dropna()
            if len(series) < self.momentum_window: continue
            mom = series.iloc[-1] / series.iloc[-self.momentum_window] - 1
            moms[t] = mom

        # Only buy assets with positive momentum
        moms = {k: v for k, v in moms.items() if v > 0}
        if not moms: return {}

        top_assets = sorted(moms, key=moms.get, reverse=True)[:self.top_n]
        vols = {}
        for t in top_assets:
            series = window[t].dropna()
            if len(series) < self.vol_window: continue
            vols[t] = series.pct_change().tail(self.vol_window).std()

        inv_vols = {t: 1.0/v for t, v in vols.items() if v > 0}
        if not inv_vols: return {t: 1.0/len(top_assets) for t in top_assets}

        total = sum(inv_vols.values())
        return {t: w/total for t, w in inv_vols.items()}

# ============================
# Main Strategy
# ============================

class MonthlyFortressStrategy:
    def __init__(self):
        self.risk_assets = RISK_ASSETS
        self.safe_assets = SAFE_ASSETS
        self.market_filter = MARKET_FILTER
        self.bond_benchmark = BOND_BENCHMARK

        self.sub_strategies = [
            FortressSubStrategy("fast_1",   63,  21, 1),
            FortressSubStrategy("fast_2",   63,  21, 2),
            FortressSubStrategy("med_1",    126, 63, 1),
            FortressSubStrategy("med_2",    126, 63, 2),
            FortressSubStrategy("slow_1",   189, 63, 1),
            FortressSubStrategy("slow_2",   189, 63, 2),
        ]

    # --- Helper: Check if Asset is Healthy ---
    def _is_asset_healthy(self, ticker, prices, date):
        """Returns True if asset is above its 10-month (200d) SMA."""
        if ticker not in prices.columns: return False
        series = prices[ticker].loc[:date].dropna()
        if len(series) < 200: return True # Assume healthy if not enough data
        
        sma200 = series.rolling(200).mean().iloc[-1]
        return series.iloc[-1] > sma200

    def detect_regime(self, prices: pd.DataFrame, date: pd.Timestamp) -> str:
        if self.market_filter not in prices.columns: return "unknown"
        spy = prices[self.market_filter].loc[:date].dropna()
        if len(spy) < LOOKBACK_SMA_TREND: return "unknown"

        current = spy.iloc[-1]
        
        # 1. CRASH CHECK
        prior = spy.iloc[-CRASH_LOOKBACK]
        if (current / prior) - 1 < CRASH_THRESHOLD: return "crash"

        # 2. VOL CHECK
        vol = spy.pct_change().tail(21).std() * np.sqrt(252)
        if vol > 0.22: return "high_vol"

        # 3. RECOVERY CHECK (Bullish if Price > SMA50)
        sma50 = spy.rolling(LOOKBACK_SMA_FAST).mean().iloc[-1]
        if current > sma50: return "bull"

        # 4. TREND CHECK (Bearish if Price < SMA200)
        sma200 = spy.rolling(LOOKBACK_SMA_TREND).mean().iloc[-1]
        if current < sma200: return "bear"

        return "bull"

    def _allocate_safe(self, target, prices, date):
        """
        Allocates to safe assets, but applies the 'Bond Guard'.
        If IEF (Bonds) is selected but is in a downtrend, swap it for SHV (Cash).
        """
        final_target = {}
        
        # 1. Apply Bond Guard
        if 'IEF' in target:
            # If Bonds are unhealthy (Prices falling / Yields rising), move allocation to Cash
            if not self._is_asset_healthy('IEF', prices, date):
                weight = target.pop('IEF')
                target['SHV'] = target.get('SHV', 0) + weight

        # 2. Distribute weights
        missing_weight = 0.0
        for t, w in target.items():
            if t in prices.columns and not pd.isna(prices.at[date, t]):
                final_target[t] = w
            else:
                missing_weight += w
        
        # 3. Fill missing data holes
        if missing_weight > 0:
            if 'SHV' in final_target: final_target['SHV'] += missing_weight
            elif 'GLD' in final_target: final_target['GLD'] += missing_weight
            elif 'IEF' in final_target: final_target['IEF'] += missing_weight
        
        return list(final_target.items())

    def _get_defense_portfolio(self, prices, date, regime):
        # Crash: Cash is King
        if regime == "crash":
            return self._allocate_safe({'SHV': 0.90, 'IEF': 0.10}, prices, date)
        
        # High Vol: Hunker down
        if regime == "high_vol":
            return self._allocate_safe({'SHV': 0.60, 'IEF': 0.20, 'GLD': 0.20}, prices, date)
            
        # Bear Market: Diversified Defense
        if regime == "bear":
            # IEF (Bonds) + GLD (Gold) + DBC (Commodities) + SHV (Cash)
            # If IEF is crashing (2022), _allocate_safe will auto-swap it to SHV
            return self._allocate_safe({'IEF': 0.30, 'GLD': 0.25, 'DBC': 0.20, 'SHV': 0.25}, prices, date)
        
        return self._allocate_safe({'SHV': 1.0}, prices, date)

    def _get_attack_portfolio(self, prices, date, regime):
        agg = {}
        active = self.sub_strategies
        for s in active:
            p = s.get_attack_portfolio(prices, date)
            if not p: continue
            w_strat = 1.0 / len(active)
            for t, w in p.items():
                agg[t] = agg.get(t, 0) + (w * w_strat)
        
        total = sum(agg.values())
        if total == 0: return {}
        return {k: v/total for k, v in agg.items()}

    def _get_leverage(self, prices, date):
        spy = prices[self.market_filter].loc[:date].dropna()
        if len(spy) < 63: return 1.0
        
        vol = spy.pct_change().tail(21).std() * np.sqrt(252)
        if vol == 0: return 1.0
        
        base = TARGET_VOL / vol
        
        sma200 = spy.rolling(200).mean().iloc[-1]
        high210 = spy.rolling(210).max().iloc[-1]
        current = spy.iloc[-1]
        
        is_turbo = (current > sma200) and (current >= high210 * 0.98)
        cap = MAX_LEVERAGE if is_turbo else 1.0
        
        return float(np.clip(base, MIN_LEVERAGE, cap))

    def get_signal(self, prices: pd.DataFrame, date: pd.Timestamp):
        regime = self.detect_regime(prices, date)

        if regime in ["crash", "bear", "high_vol", "unknown"]:
            return self._get_defense_portfolio(prices, date, regime)

        raw_attack = self._get_attack_portfolio(prices, date, regime)
        if not raw_attack:
            return self._get_defense_portfolio(prices, date, "bear")

        # Apply Anchor
        attack = {k: v * (1 - ANCHOR_WEIGHT) for k, v in raw_attack.items()}
        
        # SMART ANCHOR: Apply "Bond Guard" to the anchor too
        anchor_ief_weight = ANCHOR_WEIGHT * 0.5
        anchor_gld_weight = ANCHOR_WEIGHT * 0.5
        
        # If IEF is bad, move IEF anchor to SHV
        if not self._is_asset_healthy('IEF', prices, date):
             anchor_cash = anchor_ief_weight
             anchor_ief_weight = 0
        else:
             anchor_cash = 0

        # Add Anchor Assets
        if 'IEF' in prices.columns: attack['IEF'] = attack.get('IEF', 0) + anchor_ief_weight
        if 'GLD' in prices.columns: attack['GLD'] = attack.get('GLD', 0) + anchor_gld_weight
        
        # Add displaced anchor cash
        if anchor_cash > 0:
            attack['SHV'] = attack.get('SHV', 0) + anchor_cash

        # Apply Leverage
        lev = self._get_leverage(prices, date)
        final = {k: v * lev for k, v in attack.items()}

        return list(final.items())