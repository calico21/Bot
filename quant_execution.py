import alpaca_trade_api as tradeapi
import time
import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from quant_db_manager import MarketDB

# ==========================================
# 🔐 GESTIÓN DE CREDENCIALES
# ==========================================
load_dotenv()
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not API_KEY or not SECRET_KEY:
    print("❌ ERROR: Faltan las claves API en el archivo .env o en GitHub Secrets.")
    exit()

# ==========================================
# 🧠 ESTRATEGIA (GOLDEN CONFIG v240)
# ==========================================
class AlphaHunterStrategy:
    def __init__(self):
        self.tickers = [
            'SPY', 'QQQ', 'DIA',            
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 
            'TLT', 'GLD', 'VNQ',            
            'SMH', 'IGV', 'SOXX'
        ]
        self.top_n = 2 
        self.db = MarketDB()
        
        # Configuración Maestra
        self.lookback = 240         
        self.target_vol = 0.40      
        self.max_leverage = 2.0     
        self.vol_window = 20        

    def get_data(self):
        prices = self.db.load_data(self.tickers)
        if prices.empty or len(prices.columns) < len(self.tickers):
            print("⚡ Sincronizando datos...")
            self.db.sync_data(self.tickers)
            prices = self.db.load_data(self.tickers)
        return prices

    def calculate_signals(self):
        prices = self.get_data()
        if prices.empty: return [], 0.0, 1.0

        # A. Ranking (240 días)
        recent_data = prices.iloc[-self.lookback:]
        total_return = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1
        ranking = total_return.drop('SPY', errors='ignore')
        top_assets = ranking.sort_values(ascending=False).head(self.top_n).index.tolist()
        
        # B. Volatilidad
        recent_prices_short = prices.iloc[-self.vol_window:][top_assets]
        returns_short = recent_prices_short.pct_change().dropna()
        
        if returns_short.empty: avg_vol = 0.01 
        else: avg_vol = returns_short.mean(axis=1).std() * np.sqrt(252)
        
        if avg_vol < 0.01: avg_vol = 0.01 
        
        leverage_factor = self.target_vol / avg_vol
        final_leverage = min(leverage_factor, self.max_leverage)
        weight_per_asset = final_leverage / self.top_n
        
        return top_assets, weight_per_asset, final_leverage

# ==========================================
# 🤖 MOTOR DE EJECUCIÓN (CON REPORTING AVANZADO)
# ==========================================
def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

class ExecutionEngine:
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
        self.strategy = AlphaHunterStrategy()

    def get_daily_stats(self):
        """Calcula Ganancias/Pérdidas del día"""
        try:
            account = self.api.get_account()
            equity = float(account.equity)
            last_equity = float(account.last_equity) # Valor al cierre de ayer
            
            pnl_amount = equity - last_equity
            pnl_pct = (pnl_amount / last_equity) * 100
            
            return equity, pnl_amount, pnl_pct
        except:
            return 0.0, 0.0, 0.0

    def rebalance(self):
        print("\n=== 🦁 INICIANDO QUANTBOT (REPORTING MODE) ===")
        
        # 1. INFORME DE ESTADO (NUEVO)
        equity, daily_pnl, daily_pct = self.get_daily_stats()
        
        # Icono dinámico según ganancias o pérdidas
        icon = "🟢" if daily_pnl >= 0 else "🔴"
        
        report = (
            f"📅 *Informe Diario QuantBot*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Capital:* ${equity:,.2f}\n"
            f"{icon} *P&L Hoy:* ${daily_pnl:,.2f} ({daily_pct:+.2f}%)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ *Analizando mercado...*"
        )
        send_telegram(report)

        # 2. CÁLCULO DE ESTRATEGIA
        target_assets, target_weight, lev = self.strategy.calculate_signals()
        
        if not target_assets:
            send_telegram("⚠️ Error: No hay datos suficientes para operar.")
            return

        print(f"🎯 Objetivos: {target_assets} (Lev: x{lev:.2f})")
        
        # 3. DATOS DE CUENTA
        account = self.api.get_account()
        buying_power = float(account.buying_power)
        
        target_val_per_asset = equity * target_weight
        
        # Ajuste de seguridad
        total_needed = target_val_per_asset * len(target_assets)
        if total_needed > buying_power:
            target_val_per_asset = (buying_power * 0.95) / len(target_assets)

        try:
            positions = self.api.list_positions()
            current_positions = {p.symbol: int(p.qty) for p in positions}
        except: current_positions = {}
        
        # 4. EJECUCIÓN (Ventas primero)
        actions_log = ""
        
        for symbol, qty in current_positions.items():
            if symbol not in target_assets:
                try:
                    self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                    msg = f"👋 Venta: {symbol}"
                    print(msg)
                    actions_log += f"{msg}\n"
                except: pass
        
        time.sleep(2)
        
        # 5. EJECUCIÓN (Compras)
        for symbol in target_assets:
            try:
                price = float(self.api.get_latest_trade(symbol).price)
                if price <= 0: continue
                
                target_qty = int(target_val_per_asset / price)
                current_qty = current_positions.get(symbol, 0)
                diff = target_qty - current_qty
                
                if diff > 0:
                    self.api.submit_order(symbol=symbol, qty=diff, side='buy', type='market', time_in_force='day')
                    msg = f"🚀 Compra: {symbol} (+{diff})"
                    print(msg)
                    actions_log += f"{msg}\n"
                elif diff < 0:
                    sell_diff = abs(diff)
                    self.api.submit_order(symbol=symbol, qty=sell_diff, side='sell', type='market', time_in_force='day')
                    msg = f"📉 Ajuste: {symbol} (-{sell_diff})"
                    print(msg)
                    actions_log += f"{msg}\n"
                else:
                    # Si no hay cambios, lo registramos para saber que está vigilando
                    pass # Mantenemos posición
                    
            except Exception as e:
                print(f"❌ Error {symbol}: {e}")

        # 6. RESUMEN FINAL A TELEGRAM
        if actions_log:
            send_telegram(f"⚡ *Actividad Ejecutada:*\n{actions_log}")
        else:
            send_telegram(f"😴 *Sin cambios:* Mantenemos posiciones\n({', '.join(target_assets)})")

        print("✅ Ejecución completada.")

if __name__ == "__main__":
    bot = ExecutionEngine()
    bot.rebalance()