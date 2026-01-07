import sqlite3
import pandas as pd
import yfinance as yf
import datetime

class MarketDB:
    def __init__(self, db_name="market_data.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.initialize_db()

    def initialize_db(self):
        """Crea la tabla si no existe."""
        cursor = self.conn.cursor()
        # Creamos una tabla simple y eficiente: Fecha, Ticker, Precio
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                date TIMESTAMP,
                ticker TEXT,
                adj_close REAL,
                PRIMARY KEY (date, ticker)
            )
        ''')
        self.conn.commit()

    def sync_data(self, tickers, start_date="2000-01-01"):
        """
        Descarga datos de Yahoo y actualiza la Base de Datos Local.
        """
        print(f"--- Sincronizando BD para {len(tickers)} activos ---")
        
        try:
            # 1. Descarga Masiva
            # Usamos threads=True para máxima velocidad
            data = yf.download(tickers, start=start_date, progress=False, group_by='ticker', auto_adjust=False, threads=True)
            
            # 2. Transformación de Datos (El paso clave)
            # Yahoo devuelve una matriz ancha (Columnas por ticker). SQL quiere formato largo (Filas).
            
            prices_list = []
            
            for t in tickers:
                try:
                    # Extraer serie de precios
                    if isinstance(data.columns, pd.MultiIndex):
                        if t in data:
                            series = data[t]['Adj Close'] if 'Adj Close' in data[t] else data[t]['Close']
                        else:
                            # Intento de acceso alternativo
                            series = data.xs(t, axis=1, level=0)['Adj Close']
                    else:
                        series = data['Adj Close']
                    
                    # Limpiar y preparar
                    df_t = series.to_frame(name='adj_close')
                    df_t['ticker'] = t
                    df_t.reset_index(inplace=True) # La fecha pasa a ser columna 'Date'
                    
                    # Normalizar nombres
                    df_t.rename(columns={'Date': 'date'}, inplace=True)
                    df_t = df_t[['date', 'ticker', 'adj_close']].dropna()
                    
                    prices_list.append(df_t)
                    
                except Exception as e:
                    print(f"Advertencia procesando {t}: {e}")
            
            if not prices_list:
                print("No se pudieron procesar datos.")
                return

            # Unir todo en un solo DataFrame gigante
            full_df = pd.concat(prices_list)
            
            print(f"Guardando {len(full_df)} registros en SQL...")
            
            # 3. Guardar en SQLite
            # 'replace' es un poco bruto (borra y escribe), pero para empezar es lo más seguro 
            # para evitar duplicados sin lógica compleja de "upsert".
            full_df.to_sql('prices', self.conn, if_exists='replace', index=False)
            
            # Crear índice para velocidad de lectura futura
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON prices (ticker)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON prices (date)")
            
            print(">>> Base de Datos Actualizada con Éxito.")
            
        except Exception as e:
            print(f"Error Crítico en Sync: {e}")

    def load_data(self, tickers=None):
        """
        Lee datos DESDE EL DISCO (SQL), no de internet.
        Devuelve el formato que le gusta a tu estrategia (Pivot Table).
        """
        print("Leyendo desde Base de Datos Local...")
        
        query = "SELECT date, ticker, adj_close FROM prices"
        if tickers:
            # Filtrar solo tickers específicos si se pide
            tickers_str = "','" .join(tickers)
            query += f" WHERE ticker IN ('{tickers_str}')"
            
        # Leer SQL directo a Pandas
        df = pd.read_sql(query, self.conn, parse_dates=['date'])
        
        if df.empty:
            print("BD vacía. Ejecuta sync_data() primero.")
            return pd.DataFrame()
            
        # Pivotar para volver al formato que usan nuestros scripts anteriores
        # Índice: Fecha, Columnas: Tickers
        pivot_df = df.pivot(index='date', columns='ticker', values='adj_close')
        pivot_df.sort_index(inplace=True)
        
        return pivot_df

    def close(self):
        self.conn.close()

# Bloque de ejecución manual para probar
if __name__ == "__main__":
    # Universo de Sectores + Bonos + Oro
    MY_UNIVERSE = [
    'XLK', 'SMH', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
    'SPY', 'TLT', 'GLD', 'SHV', 'IEF'
    ]
    db = MarketDB()
    # 1. Descargar y Guardar (Solo se hace una vez al día/semana)
    db.sync_data(MY_UNIVERSE)
    
    # 2. Probar lectura
    df = db.load_data()
    print("\nVista previa de datos cargados localmente:")
    print(df.tail())
    db.close()