# strategy.py

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================
# Global Configuration
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

LOOKBACK_SMA_TREND = 200
LOOKBACK_SMA_FAST = 50
CRASH_LOOKBACK = 21
CRASH_THRESHOLD = -0.07

VOL_LOOKBACK = 63
MAX_SINGLE_ASSET_EXPOSURE = 0.50
MAX_PORTFOLIO_LEVERAGE = 2.00

TARGET_VOL_BULL = 0.20
TARGET_VOL_BEAR = 0.12

TAIL_VOL_CEILING = 0.30
TAIL_VOL_ACCELERATION = 1.35

MAX_CALC_LEVERAGE = 2.4
MIN_LEVERAGE = 0.5

ANCHOR_WEIGHT_BASE = 0.25
ANCHOR_WEIGHT_BULL = 0.00
ANCHOR_WEIGHT_BEAR = 0.30
ANCHOR_WEIGHT_CRASH = 0.60
ANCHOR_WEIGHT_HIGH_VOL = 0.40

# Turnover control
HYSTERESIS_THRESHOLD = 0.05   # 5% change required to rebalance
MIN_WEIGHT_THRESHOLD = 0.03   # Drop positions below 3%
SMOOTHING_ALPHA = 0.5         # Blend old/new weights: new = α*new + (1-α)*old


# ============================
# 1. Sub-Strategy: Momentum + HRP Engine
# ============================

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
            c_items = [i[j:k] for i in c_items
                       for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                       if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        return w

    def get_hrp_weights(self, prices, assets):
        subset = prices[assets].iloc[-504:].pct_change().dropna()
        subset = subset.loc[:, subset.std() > 1e-6]
        valid_assets = list(subset.columns)
        if len(valid_assets) < 2:
            return {t: 1.0 / len(assets) for t in assets}
        try:
            cov = subset.cov()
            corr = subset.corr()
            dist = np.sqrt((1 - corr) / 2)
            dist_array = squareform(dist)
            link = sch.linkage(dist_array, 'single')
            sort_ix = self._get_quasi_diag(link)
            sort_ix = [subset.columns[i] for i in sort_ix]
            cov = cov.loc[sort_ix, sort_ix]
            weights = self._get_rec_bipart(cov, sort_ix)
            return weights.to_dict()
        except Exception:
            return {t: 1.0 / len(assets) for t in assets}

    def get_attack_portfolio(self, prices: pd.DataFrame, date: pd.Timestamp, regime: str):
        window = prices.loc[:date]
        lookbacks = [63, 126, 189, 252]
        scores = {}

        for t in self.risk_assets:
            if t not in window.columns:
                continue
            series = window[t].dropna()
            asset_score = 0.0
            valid_lookbacks = 0
            for lb in lookbacks:
                if len(series) > lb:
                    mom = series.iloc[-1] / series.iloc[-lb] - 1
                    vol = series.pct_change().tail(21).std()
                    if vol and vol > 0:
                        asset_score += (mom / vol)
                        valid_lookbacks += 1
            if valid_lookbacks > 0:
                scores[t] = asset_score / valid_lookbacks

        positive_assets = {k: v for k, v in scores.items() if v > 0}
        if not positive_assets:
            return {}

        candidate_count = max(self.top_n, 5)
        top_assets = sorted(positive_assets, key=positive_assets.get, reverse=True)[:candidate_count]

        if len(top_assets) == 1:
            return {top_assets[0]: 1.0}

        hrp_weights = self.get_hrp_weights(window, top_assets)

        if regime == "bull":
            tilted_weights = {}
            total_score = sum([scores[t] for t in top_assets if t in scores])
            for t, w_hrp in hrp_weights.items():
                if t in scores and total_score > 0:
                    w_mom = scores[t] / total_score
                    tilted_weights[t] = 0.5 * w_hrp + 0.5 * w_mom
                else:
                    tilted_weights[t] = w_hrp
            w_sum = sum(tilted_weights.values())
            if w_sum > 0:
                tilted_weights = {k: v / w_sum for k, v in tilted_weights.items()}
            return tilted_weights

        return hrp_weights


# ============================
# 2. Main Strategy: Fortress v15 (Drawdown + Turnover Aware)
# ============================

class MonthlyFortressStrategy:
    def __init__(self):
        self.risk_assets = RISK_ASSETS
        self.safe_assets = SAFE_ASSETS
        self.market_filter = MARKET_FILTER
        self.bond_benchmark = BOND_BENCHMARK
        self.sub_strategies = [FortressSubStrategy("ensemble_core", VOL_LOOKBACK, 5)]

    # ---------- Health & Tail-Risk ----------

    def _is_asset_healthy(self, ticker, prices, date):
        if ticker not in prices.columns:
            return False
        series = prices[ticker].loc[:date].dropna()
        if len(series) < 200:
            return True
        sma200 = series.rolling(200).mean().iloc[-1]
        return series.iloc[-1] > sma200

    def _check_tail_risk(self, prices, date):
        if 'SPY' not in prices or 'IEF' not in prices:
            return False
        spy = prices['SPY'].loc[:date].dropna()
        ief = prices['IEF'].loc[:date].dropna()
        if len(spy) < 63:
            return False

        vol_21 = spy.pct_change().tail(21).std() * np.sqrt(252)
        vol_63 = spy.pct_change().tail(63).std() * np.sqrt(252)

        is_vol_accelerating = vol_21 > (vol_63 * TAIL_VOL_ACCELERATION)
        is_vol_critical = vol_21 > TAIL_VOL_CEILING

        sma200 = spy.rolling(200).mean().iloc[-1]
        price = spy.iloc[-1]
        is_downtrend = price < sma200

        spy_rets = spy.pct_change().tail(63)
        ief_rets = ief.pct_change().tail(63)
        is_corr_broken = spy_rets.corr(ief_rets) > 0.25

        if is_downtrend and is_vol_accelerating:
            return True
        if is_downtrend and is_corr_broken:
            return True
        if is_vol_critical:
            return True
        return False

    # ---------- Regime Detection ----------

    def detect_regime(self, prices, date: pd.Timestamp) -> str:
        if self.market_filter not in prices.columns:
            return "unknown"
        spy = prices[self.market_filter].loc[:date].dropna()
        if len(spy) < LOOKBACK_SMA_TREND:
            return "unknown"

        current = spy.iloc[-1]
        prior = spy.iloc[-CRASH_LOOKBACK]

        if (current / prior) - 1 < CRASH_THRESHOLD:
            return "crash"

        vol = spy.pct_change().tail(21).std() * np.sqrt(252)
        if vol > 0.20:
            return "high_vol"

        sma50 = spy.rolling(LOOKBACK_SMA_FAST).mean().iloc[-1]
        if current > sma50:
            return "bull"

        sma200 = spy.rolling(LOOKBACK_SMA_TREND).mean().iloc[-1]
        if current < sma200:
            return "bear"

        return "bull"

    # ---------- Safe Allocation ----------

    def _allocate_safe(self, target, prices, date):
        final_target = {}
        missing_weight = 0.0
        for t, w in target.items():
            if t in prices.columns and not pd.isna(prices.at[date, t]):
                final_target[t] = w
            else:
                missing_weight += w

        if missing_weight > 0:
            if 'SHV' in final_target:
                final_target['SHV'] += missing_weight
            elif 'GLD' in final_target:
                final_target['GLD'] += missing_weight
            elif 'IEF' in final_target:
                final_target['IEF'] += missing_weight

        return list(final_target.items())

    def _get_defense_portfolio(self, prices, date, regime):
        if regime == "crash":
            return self._allocate_safe({'SHV': 1.0}, prices, date)
        if regime == "high_vol":
            return self._allocate_safe({'SHV': 0.50, 'IEF': 0.25, 'GLD': 0.25}, prices, date)
        if regime == "bear":
            return self._allocate_safe({'IEF': 0.40, 'GLD': 0.20, 'SHV': 0.40}, prices, date)
        return self._allocate_safe({'SHV': 1.0}, prices, date)

    # ---------- Attack Portfolio ----------

    def _get_attack_portfolio(self, prices, date, regime):
        s = self.sub_strategies[0]
        return s.get_attack_portfolio(prices, date, regime)

    # ---------- Anchor & Leverage Helpers ----------

    def _get_anchor_weight_for_regime(self, regime: str) -> float:
        if regime == "bull":
            return ANCHOR_WEIGHT_BULL
        if regime == "bear":
            return ANCHOR_WEIGHT_BEAR
        if regime == "crash":
            return ANCHOR_WEIGHT_CRASH
        if regime == "high_vol":
            return ANCHOR_WEIGHT_HIGH_VOL
        return ANCHOR_WEIGHT_BASE

    def _dd_anchor_adjustment(self, anchor_weight, current_dd):
        if current_dd is None:
            return anchor_weight
        if current_dd < -0.40:
            return 1.0
        if current_dd < -0.30:
            anchor_weight += 0.20
        elif current_dd < -0.20:
            anchor_weight += 0.10
        elif current_dd < -0.10:
            anchor_weight += 0.05
        return min(anchor_weight, 1.0)

    def _dd_leverage_adjustment(self, current_dd):
        if current_dd is None:
            return 1.0
        if current_dd < -0.40:
            return 0.20
        if current_dd < -0.30:
            return 0.40
        if current_dd < -0.20:
            return 0.60
        if current_dd < -0.10:
            return 0.85
        return 1.0

    def _get_leverage(self, prices, date, regime: str, current_dd):
        spy = prices[self.market_filter].loc[:date].dropna()
        if len(spy) < 63:
            return 1.0
        vol = spy.pct_change().tail(21).std() * np.sqrt(252)
        if vol == 0:
            return 1.0
        if vol > 0.30:
            base = 1.0
        else:
            target_vol = TARGET_VOL_BULL if regime == "bull" else TARGET_VOL_BEAR
            base = target_vol / vol
        cap = MAX_CALC_LEVERAGE if regime == "bull" else 1.2
        lev = float(np.clip(base, MIN_LEVERAGE, cap))
        lev *= self._dd_leverage_adjustment(current_dd)
        return min(lev, MAX_PORTFOLIO_LEVERAGE)

    # ---------- Turnover-Aware Smoothing ----------

    def _apply_turnover_controls(self, new_weights: dict, prev_weights: dict | None):
        if prev_weights is None or len(prev_weights) == 0:
            # First period: just normalize and apply min threshold
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: v / total for k, v in new_weights.items()}
            return {k: v for k, v in new_weights.items() if v >= MIN_WEIGHT_THRESHOLD}

        # 1) Hysteresis: only change weights if |new - old| > threshold
        adjusted = {}
        all_tickers = set(new_weights.keys()) | set(prev_weights.keys())
        for t in all_tickers:
            old_w = prev_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            if abs(new_w - old_w) < HYSTERESIS_THRESHOLD:
                # Keep old weight
                adj_w = old_w
            else:
                # Smooth transition
                adj_w = SMOOTHING_ALPHA * new_w + (1 - SMOOTHING_ALPHA) * old_w
            adjusted[t] = adj_w

        # 2) Drop tiny weights
        adjusted = {k: v for k, v in adjusted.items() if v >= MIN_WEIGHT_THRESHOLD}

        # 3) Renormalize to preserve total exposure scale
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    # ---------- Main Signal ----------

    def get_signal(self, prices: pd.DataFrame, date: pd.Timestamp,
                   current_dd: float | None = None,
                   prev_weights: dict | None = None):
        if self._check_tail_risk(prices, date):
            safe = dict(self._allocate_safe({'SHV': 1.0}, prices, date))
            safe = self._apply_turnover_controls(safe, prev_weights)
            return list(safe.items())

        regime = self.detect_regime(prices, date)

        if regime in ["crash", "bear", "high_vol", "unknown"]:
            safe = dict(self._get_defense_portfolio(prices, date, regime))
            safe = self._apply_turnover_controls(safe, prev_weights)
            return list(safe.items())

        raw_attack = self._get_attack_portfolio(prices, date, regime)
        if not raw_attack:
            safe = dict(self._get_defense_portfolio(prices, date, "bear"))
            safe = self._apply_turnover_controls(safe, prev_weights)
            return list(safe.items())

        anchor_weight = self._get_anchor_weight_for_regime(regime)
        anchor_weight = self._dd_anchor_adjustment(anchor_weight, current_dd)

        portfolio = {}
        for t, w in raw_attack.items():
            portfolio[t] = w * (1 - anchor_weight)

        anchor_ief = anchor_weight * 0.5
        anchor_gld = anchor_weight * 0.5
        anchor_cash = 0.0

        if not self._is_asset_healthy('IEF', prices, date):
            anchor_cash += anchor_ief
            anchor_ief = 0.0

        if 'IEF' in prices.columns:
            portfolio['IEF'] = portfolio.get('IEF', 0.0) + anchor_ief
        if 'GLD' in prices.columns:
            portfolio['GLD'] = portfolio.get('GLD', 0.0) + anchor_gld
        if anchor_cash > 0:
            portfolio['SHV'] = portfolio.get('SHV', 0.0) + anchor_cash

        # Turnover-aware smoothing BEFORE leverage
        portfolio = self._apply_turnover_controls(portfolio, prev_weights)

        lev = self._get_leverage(prices, date, regime, current_dd)
        final_portfolio = {k: v * lev for k, v in portfolio.items()}

        for t in list(final_portfolio.keys()):
            if final_portfolio[t] > MAX_SINGLE_ASSET_EXPOSURE:
                final_portfolio[t] = MAX_SINGLE_ASSET_EXPOSURE

        total_exposure = sum(final_portfolio.values())
        if total_exposure > MAX_PORTFOLIO_LEVERAGE:
            scale_factor = MAX_PORTFOLIO_LEVERAGE / total_exposure
            final_portfolio = {k: v * scale_factor for k, v in final_portfolio.items()}

        return list(final_portfolio.items())


# ============================
# 3. Satellite Strategy: RSI-2 Mean Reversion
# ============================

class RSI2MeanReversionStrategy:
    def __init__(self, ticker='SPY', rsi_period=2, rsi_entry=10,
                 trend_ma=200, exit_ma=5, allocation=0.20):
        self.ticker = ticker
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.trend_ma = trend_ma
        self.exit_ma = exit_ma
        self.allocation = allocation
        self.in_position = False

    def _calc_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def get_signal(self, prices: pd.DataFrame, date: pd.Timestamp,
                   current_dd: float | None = None,
                   prev_weights: dict | None = None):
        if self.ticker not in prices.columns:
            return []
        history = prices[self.ticker].loc[:date]
        if len(history) < self.trend_ma + 5:
            return []

        current_price = history.iloc[-1]
        sma_trend = history.rolling(self.trend_ma).mean().iloc[-1]
        sma_exit = history.rolling(self.exit_ma).mean().iloc[-1]
        rsi_series = self._calc_rsi(history, self.rsi_period)
        current_rsi = rsi_series.iloc[-1]

        if self.in_position and current_price > sma_exit:
            self.in_position = False
            return []

        if not self.in_position:
            if current_price > sma_trend and current_rsi < self.rsi_entry:
                self.in_position = True

        if self.in_position:
            return [(self.ticker, 1.0)]
        return []


# ============================
# 4. Composite Strategy: Fortress + RSI-2
# ============================

class CompositeStrategy:
    def __init__(self, main_strat, sat_strat, main_weight=0.9):
        self.main = main_strat
        self.satellite = sat_strat
        self.main_weight = main_weight
        self.sat_weight = 1.0 - main_weight

        sat_assets = [sat_strat.ticker] if hasattr(sat_strat, 'ticker') else []
        self.risk_assets = list(set(main_strat.risk_assets + sat_assets))
        self.safe_assets = main_strat.safe_assets
        self.market_filter = main_strat.market_filter
        self.bond_benchmark = main_strat.bond_benchmark

    def get_signal(self, prices: pd.DataFrame, date: pd.Timestamp,
               current_dd: float | None = None,
               prev_weights: dict | None = None):

        sig_main = dict(self.main.get_signal(prices, date, current_dd, prev_weights))
        sig_sat = dict(self.satellite.get_signal(prices, date, current_dd, prev_weights))
        final_portfolio = {}

        for ticker, w in sig_main.items():
            final_portfolio[ticker] = final_portfolio.get(ticker, 0.0) + w * self.main_weight

        for ticker, w in sig_sat.items():
            current_w = final_portfolio.get(ticker, 0.0)
            final_portfolio[ticker] = current_w + (w * self.sat_weight)

        return list(final_portfolio.items())
