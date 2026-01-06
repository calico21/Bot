import alpaca_trade_api as tradeapi
import time
import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv  # <--- NUEVO: Para leer el archivo .env en tu PC
from quant_db_manager import MarketDB

# ==========================================
# 🔐 GESTIÓN DE CREDENCIALES (NIVEL PRO)
# ==========================================
# 1. Carga las claves del archivo .env (solo si estás en tu PC)
load_dotenv()

# 2. Lee las variables. Si no existen (ni en .env ni en GitHub), da error.
# FÍJATE: ¡Aquí ya no hay claves escritas!
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Verificación de seguridad antes de arrancar
if not API_KEY or not SECRET_KEY:
    print("❌ ERROR CRÍTICO: No se han encontrado las API KEYS.")
    print("Asegúrate de tener el archivo .env en tu PC o los Secrets configurados en GitHub.")
    exit()

# ==========================================
# 🧠 ESTRATEGIA: QUANTBOT 'YEARLY CYCLE' (Golden Config v240)
# ==========================================
class AlphaHunterStrategy:
    def __init__(self):
        # UNIVERSO VALIDADO (Alpha Hunter)
        self.tickers = [
            'SPY', 'QQQ', 'DIA',            
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 
            'TLT', 'GLD', 'VNQ',            
            'SMH', 'IGV', 'SOXX'
        ]
        self.top_n = 2 
        self.db = MarketDB()
        
        # --- ⚙️ LA CONFIGURACIÓN MAESTRA (Heatmap 240/0.40) ---
        self.lookback = 240         
        self.target_vol = 0.40      
        self.max_leverage = 2.0     
        self.vol_window = 20        

    def get_data(self):
        prices = self.db.load_data(self.tickers)
        if prices.empty or len(prices.columns) < len(self.tickers):
            print("⚡ Sincronizando universo de datos...")
            self.db.sync_data(self.tickers)
            prices = self.db.load_data(self.tickers)
        return prices

    def calculate_signals(self):
        prices = self.get_data()
        if prices.empty: return [], 0.0, 1.0

        # --- RANKING ---
        recent_data = prices.iloc[-self.lookback:]
        total_return = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1
        ranking = total_return.drop('SPY', errors='ignore')
        top_assets = ranking.sort_values(ascending=False).head(self.top_n).index.tolist()
        
        # --- VOLATILIDAD ---
        recent_prices_short = prices.iloc[-self.vol_window:][top_assets]
        returns_short = recent_prices_short.pct_change().dropna()
        
        if returns_short.empty:
            avg_vol = 0.01 
        else:
            avg_vol = returns_short.mean(axis=1).std() * np.sqrt(252)
        
        if avg_vol < 0.01: avg_vol = 0.01 
        
        leverage_factor = self.target_vol / avg_vol
        final_leverage = min(leverage_factor, self.max_leverage)
        weight_per_asset = final_leverage / self.top_n
        
        return top_assets, weight_per_asset, final_leverage

# ==========================================
# 🤖 MOTOR DE EJECUCIÓN
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
        
    def rebalance(self):
        print("\n=== 🦁 INICIANDO QUANTBOT (GOLDEN CONFIG v240) ===")
        send_telegram("🦁 *QuantBot v240*: Iniciando secuencia segura...")
        
        target_assets, target_weight, lev = self.strategy.calculate_signals()
        
        if not target_assets:
            print("❌ Datos insuficientes.")
            return

        print(f"🎯 Objetivos: {target_assets}")
        
        try:
            account = self.api.get_account()
        except:
            print("❌ Error de credenciales Alpaca.")
            return

        equity = float(account.equity)
        buying_power = float(account.buying_power)
        print(f"💰 Capital: ${equity:,.2f}")
        
        target_val_per_asset = equity * target_weight
        
        total_needed = target_val_per_asset * len(target_assets)
        if total_needed > buying_power:
            print("⚠️ Ajustando a Buying Power.")
            target_val_per_asset = (buying_power * 0.95) / len(target_assets)

        try:
            positions = self.api.list_positions()
            current_positions = {p.symbol: int(p.qty) for p in positions}
        except: current_positions = {}
        
        # VENTAS
        for symbol, qty in current_positions.items():
            if symbol not in target_assets:
                print(f"🔻 VENDIENDO: {symbol}")
                try:
                    self.api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                    send_telegram(f"👋 *Venta*: {symbol}")
                except: pass
        
        time.sleep(2)
        
        # COMPRAS
        for symbol in target_assets:
            try:
                price = float(self.api.get_latest_trade(symbol).price)
                target_qty = int(target_val_per_asset / price)
                current_qty = current_positions.get(symbol, 0)
                diff = target_qty - current_qty
                
                if diff > 0:
                    print(f"🚀 COMPRANDO: {symbol} x {diff}")
                    self.api.submit_order(symbol=symbol, qty=diff, side='buy', type='market', time_in_force='day')
                    send_telegram(f"🚀 *Compra*: {symbol}")
                elif diff < 0:
                    sell_diff = abs(diff)
                    print(f"📉 REAJUSTE: {symbol} x {sell_diff}")
                    self.api.submit_order(symbol=symbol, qty=sell_diff, side='sell', type='market', time_in_force='day')
            except Exception as e:
                print(f"❌ Error {symbol}: {e}")

        print("✅ Ejecución finalizada.")

if __name__ == "__main__":
    bot = ExecutionEngine()
    bot.rebalance()