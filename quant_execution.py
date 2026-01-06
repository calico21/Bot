import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- LIBRERÍAS NUEVAS (SDK v2) ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 🔐 CONFIGURACIÓN
# ==========================================
load_dotenv()
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PAPER_MODE = True # Cambiar a False para dinero real

if not API_KEY or not SECRET_KEY:
    print("❌ ERROR: Faltan credenciales.")
    exit()

# Clientes de Alpaca (Trading y Datos separados)
trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_MODE)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ==========================================
# 🧠 ESTRATEGIA: MONTHLY FORTRESS (ROTACIÓN MENSUAL)
# ==========================================
class MonthlyStrategy:
    def __init__(self):
        # UNIVERSO DE "ATAQUE" (Sectores Ofensivos)
        self.risk_assets = ['XLK', 'SMH', 'XLF', 'XLV', 'XLI', 'XLY']
        # ACTIVOS DE "DEFENSA" (Refugio)
        self.safe_assets = ['IEF', 'SHV'] # Bonos 7-10 años y Cash corto plazo
        # FILTRO DE MERCADO
        self.market_filter = 'SPY'
        
        self.lookback_days = 200  # Para la media móvil de seguridad
        self.momentum_window = 126 # 6 Meses para ranking de fuerza

    def get_data_alpaca(self, tickers, days):
        """Descarga datos oficiales desde Alpaca"""
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days * 2) # Margen de seguridad
        
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt
        )
        bars = data_client.get_stock_bars(req)
        return bars.df # Devuelve DataFrame directo

    def get_signals(self):
        print("📊 Analizando Régimen de Mercado...")
        
        # 1. FILTRO DE RÉGIMEN (¿Estamos en Crisis?)
        spy_data = self.get_data_alpaca([self.market_filter], 300)
        if spy_data.empty: return [], "ERROR_DATA"
        
        # Calcular Media Móvil 200 días
        spy_closes = spy_data[spy_data['symbol'] == self.market_filter]['close']
        sma_200 = spy_closes.rolling(window=200).mean().iloc[-1]
        current_price = spy_closes.iloc[-1]
        
        market_is_bullish = current_price > sma_200
        
        if not market_is_bullish:
            print(f"🐻 MERCADO BAJISTA DETECTADO (SPY ${current_price:.2f} < SMA200 ${sma_200:.2f})")
            print("🛡️ MODO DEFENSA: Comprando Bonos.")
            return ['IEF'], "DEFENSE"
        
        print(f"🐂 MERCADO ALCISTA CONFIRMADO (SPY > SMA200)")
        
        # 2. RANKING DE MOMENTUM (¿Qué sectores lideran?)
        risk_data = self.get_data_alpaca(self.risk_assets, self.momentum_window + 20)
        scores = {}
        
        for ticker in self.risk_assets:
            try:
                df = risk_data[risk_data['symbol'] == ticker]['close']
                if len(df) < self.momentum_window: continue
                
                # Momentum de 6 meses (Retorno)
                # Fórmula: Precio Hoy / Precio hace 126 días - 1
                ret = (df.iloc[-1] / df.iloc[-self.momentum_window]) - 1
                scores[ticker] = ret
            except: continue
            
        # Elegir el TOP 1 (El más fuerte) o TOP 2
        # La "Ruta Inteligente" suele concentrar en el mejor. Vamos con el TOP 1 para máxima potencia.
        top_asset = sorted(scores, key=scores.get, reverse=True)[:1]
        
        print(f"🚀 LÍDER DEL MERCADO: {top_asset} (Ret: {scores[top_asset[0]]:.2%})")
        return top_asset, "ATTACK"

# ==========================================
# 🤖 MOTOR DE EJECUCIÓN (LIMIT ORDERS)
# ==========================================
def send_telegram(msg):
    if not TELEGRAM_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

def run_bot():
    print(f"\n=== 🏯 MONTHLY FORTRESS EJECUTANDO ({datetime.now()}) ===")
    strategy = MonthlyStrategy()
    
    # 1. OBTENER SEÑAL
    try:
        target_assets, mode = strategy.get_signals()
    except Exception as e:
        print(f"❌ Error calculando señales: {e}")
        return

    # 2. ESTADO DE CUENTA
    acct = trade_client.get_account()
    equity = float(acct.equity)
    buying_power = float(acct.buying_power)
    
    msg = f"📅 *Monthly Rebalance*\n💰 Equity: ${equity:,.0f}\nModo: {'🟢 ATAQUE' if mode == 'ATTACK' else '🔴 DEFENSA'}\nObjetivo: {target_assets}"
    send_telegram(msg)
    
    # 3. GESTIÓN DE POSICIONES
    # Primero obtenemos posiciones actuales
    current_positions = trade_client.get_all_positions()
    current_tickers = [p.symbol for p in current_positions]
    
    # A. VENDER LO QUE YA NO QUEREMOS
    for p in current_positions:
        if p.symbol not in target_assets:
            print(f"🔻 CERRANDO: {p.symbol}")
            trade_client.close_position(p.symbol) # Venta a mercado para cerrar rápido
            send_telegram(f"👋 Cerrando {p.symbol}")
    
    # Esperar a que se libere el cash
    if current_tickers: time.sleep(5) 
    
    # B. COMPRAR EL NUEVO LÍDER
    # Usamos el 95% del equity para dejar margen
    target_val = float(trade_client.get_account().cash) * 0.95
    
    for symbol in target_assets:
        try:
            # Obtener precio actual para calcular límite
            quote = data_client.get_stock_latest_quote(StockBarsRequest(symbol_or_symbols=[symbol]))
            current_price = quote[symbol].ask_price
            
            # PROTECCIÓN: Limit Order un 0.5% arriba (Asegura entrada pero evita slippage infinito)
            limit_price = round(current_price * 1.005, 2)
            qty = int(target_val / limit_price)
            
            if qty > 0:
                print(f"🚀 ORDEN LIMIT: {symbol} x {qty} @ ${limit_price}")
                
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
                
                trade_client.submit_order(req)
                send_telegram(f"🚀 *Compra*: {symbol}\nQty: {qty}\nLimit: ${limit_price}")
            
        except Exception as e:
            print(f"❌ Error comprando {symbol}: {e}")

if __name__ == "__main__":
    run_bot()