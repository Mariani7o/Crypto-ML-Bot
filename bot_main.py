"""⚙️ CELDA 1 – Librerias"""

import requests

import pandas as pd

import numpy as np

import time

import pytz

import ta

import os

import joblib

from datetime import datetime, timezone, timedelta

from sklearn.linear_model import LogisticRegression # Modelo ML

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler # ¡NUEVA! Necesaria para normalizar

from ta.momentum import RSIIndicator

from ta.volatility import BollingerBands

from ta.trend import MACD, ADXIndicator



"""⚙️ CELDA 2 – Configuración de Binance"""



# >> USAR LA URL DE ALTA DISPONIBILIDAD PARA MEJOR TASA DE LÍMITE <<

BINANCE_BASE = "https://data-api.binance.vision"

SYMBOL = "ETHUSDT"

INTERVAL = "5m"



# Eliminamos get_latest_klines ya que no se usa en el bucle.

# Mantenemos las variables globales.



"""📊 CELDA 3 – Descarga de datos intradía"""



# CELDA 3 - Funciones de Data Fetching, Cálculo de Indicadores y Preparación ML



# Parámetros del Target de Machine Learning (Se mantienen)

FUTURE_CANDLES = 5    # Mirar 5 velas al futuro (ej. 5 minutos)

PROFIT_THRESHOLD = 0.003 # 0.3% de movimiento para considerar éxito



# ----------------------------------------------------

# A. FUNCIÓN PARA OBTENER DATOS DE BINANCE (get_intraday_data)

# ----------------------------------------------------

def get_intraday_data(symbol, interval, samples):

    """Obtiene el historial de Klines de Binance."""



    # URL de la API de Klines usando la URL de alta disponibilidad

    base_url = "https://data-api.binance.vision/api/v3/klines"



    # Parámetros para la solicitud

    params = {

        'symbol': symbol,

        'interval': interval,

        'limit': samples

    }



    print(f"Buscando {samples} velas de {symbol} en {interval}...")



    try:

        response = requests.get(base_url, params=params)

        response.raise_for_status() # Lanza una excepción para errores HTTP (como el 451)

        data = response.json()

    except requests.exceptions.RequestException as e:

        print(f"Error al obtener datos de Binance: {e}")

        return pd.DataFrame()



    # Convertir a DataFrame de Pandas

    df = pd.DataFrame(data, columns=[

        'timestamp', 'open', 'high', 'low', 'close', 'volume',

        'close_time', 'quote_asset_volume', 'number_of_trades',

        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'

    ])



    # Limpieza y conversión de tipos

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df.set_index('timestamp')

    df['price'] = df['close'].astype(float) # Usaremos 'price' para el análisis

    df['volume'] = df['volume'].astype(float)

    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)



    # Llamada a la nueva función modular

    df = calculate_indicators(df)



    return df



# ----------------------------------------------------

# B. FUNCIÓN PARA CÁLCULO DE INDICADORES (calculate_indicators) - ¡NUEVA!

# ----------------------------------------------------

def calculate_indicators(df):

    """Calcula todos los indicadores técnicos necesarios y limpia NaNs."""

    df['RSI'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()

    df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()



    bb = ta.volatility.BollingerBands(df['price'], window=20, window_dev=2)

    df['BB_UPPER'] = bb.bollinger_hband()

    df['BB_LOWER'] = bb.bollinger_lband()



    df['EMA_20'] = df['price'].ewm(span=20, adjust=False).mean()

    df['EMA_50'] = df['price'].ewm(span=50, adjust=False).mean()

    df['EMA_200'] = df['price'].ewm(span=200, adjust=False).mean()



    macd = ta.trend.MACD(df['price'])

    df['MACD_HIST'] = macd.macd_diff()



    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)

    df['ADX'] = adx.adx()

    df['ADX_DI_POS'] = adx.adx_pos()

    df['ADX_DI_NEG'] = adx.adx_neg()



    # Limpiamos las primeras filas con NaN generados por los indicadores

    df.dropna(inplace=True)



    # Aseguramos que las columnas necesarias sean float después de los cálculos

    df = df.astype({'RSI': float, 'ATR': float, 'BB_UPPER': float, 'BB_LOWER': float, 'EMA_20': float, 'EMA_50': float, 'EMA_200': float, 'MACD_HIST': float, 'ADX': float, 'ADX_DI_POS': float, 'ADX_DI_NEG': float})



    return df





# ----------------------------------------------------

# C. FUNCIÓN PARA PREPARAR DATOS PARA ML (prepare_data_for_ml) - ¡MEJORADA!

# ----------------------------------------------------

def prepare_data_for_ml(df, scaler=None):

    """

    Crea las variables de entrada (features) y la variable objetivo (target)

    para entrenar o predecir el modelo de Machine Learning.



    Retorna X normalizado, Y_Long, Y_Short y el objeto scaler (si se entrena).

    """

    feature_columns = [

        'RSI', 'ATR',

        'BB_UPPER', 'BB_LOWER', 'EMA_20', 'EMA_50', 'EMA_200',

        'MACD_HIST', 'ADX', 'ADX_DI_POS', 'ADX_DI_NEG'

    ]

    features = df[feature_columns].copy()



    # --- Estandarización/Normalización ---

    if scaler is None:

        # Modo ENTRENAMIENTO: Creamos y ajustamos un nuevo escalador.

        scaler = StandardScaler()

        # Ajustamos y transformamos en un solo paso

        X_scaled = scaler.fit_transform(features)

        X = pd.DataFrame(X_scaled, index=features.index, columns=feature_columns)

        is_training = True

    else:

        # Modo PREDICCIÓN: Usamos el escalador ya entrenado.

        X_scaled = scaler.transform(features)

        X = pd.DataFrame(X_scaled, index=features.index, columns=feature_columns)

        is_training = False





    # 2. Crear la variable Target (Y): ¿Ganó la operación en el futuro?



    # Calculamos el precio máximo y mínimo en las siguientes N velas

    future_max_prices = df['price'].rolling(window=FUTURE_CANDLES).max().shift(-FUTURE_CANDLES)

    future_min_prices = df['price'].rolling(window=FUTURE_CANDLES).min().shift(-FUTURE_CANDLES)



    # Target LONG: ¿El precio futuro subirá lo suficiente?

    target_long = (future_max_prices / df['price'] - 1) >= PROFIT_THRESHOLD

    X['Target_Long'] = target_long.astype(int).shift(FUTURE_CANDLES)



    # Target SHORT: ¿El precio futuro caerá lo suficiente?

    target_short = (1 - future_min_prices / df['price']) >= PROFIT_THRESHOLD

    X['Target_Short'] = target_short.astype(int).shift(FUTURE_CANDLES)





    # Limpiar NaN generados por la ventana de rolling y shift

    X.dropna(inplace=True)



    # Separar X (features) e Y (targets)

    if is_training:

        Y_Long = X['Target_Long']

        Y_Short = X['Target_Short']

        X = X.drop(columns=['Target_Long', 'Target_Short'])

        # Devolvemos el scaler para guardarlo

        return X, Y_Long, Y_Short, scaler

    else:

        # En predicción, solo necesitamos X_predict_final

        X_predict_final = X.drop(columns=['Target_Long', 'Target_Short'])

        # Devolvemos solo la última fila normalizada

        return X_predict_final.iloc[-1].to_frame().T, None, None, scaler



"""🤖 CELDA 4 – Lógica de señales detalladas"""



# Generador de señales con LÓGICA MACHINE LEARNING Y MENSAJES



# Variables globales para el modelo ML (se entrenarán una sola vez)

model_long = None

model_short = None

prev_signal = None

scaler = None # Objeto StandardScaler para normalización

# RUTAS DE PERSISTENCIA (Usaremos el directorio /tmp, que es accesible y grabable en Render)

MODEL_DIR = '/tmp/'

MODEL_LONG_FILE = MODEL_DIR + 'model_long.pkl'

MODEL_SHORT_FILE = MODEL_DIR + 'model_short.pkl'

SCALER_FILE = MODEL_DIR + 'scaler.pkl'





# Variables de riesgo

MIN_ATR = 2.0  # Suavizado para evitar descarte por ATR muy bajo

INERTIA_ATR_MULTIPLIER = 0.5

MIN_PROFIT_USD = 15.0



SL_MULTIPLIER = 3.5

TP_MULTIPLIER = 5.0



PROB_LONG_MIN = 0.65 # Requerir 65% de probabilidad

PROB_SHORT_MIN = 0.70 # Requerir 70% de probabilidad (más estricto para cortos)



MIN_CONFIDENCE_FOR_SIGNAL = 80 # Solo se envía LONG/SHORT si la confianza >= 80%

INERTIA_CONFIDENCE = 50





def generate_signal(df):

    # Aseguramos que podemos leer y escribir en estas variables globales

    global prev_signal, model_long, model_short, scaler



    # -----------------------------------------------\n

    # 0. VERIFICACIÓN DE DATOS

    # -----------------------------------------------



    if df.empty:

        return "WAIT", "Datos insuficientes (DataFrame vacío).", "⚪", "Baja"



    if "RSI" not in df.columns:

        return "WAIT", "Error interno: Columna RSI faltante. ¿Se ejecutó Celda 3?", "🔴", "RIESGO"



    # Usaremos 200 velas para entrenamiento, así que necesitamos al menos 200 + FUTURE_CANDLES

    if len(df) < 205 or df["RSI"].isnull().all():

        return "WAIT", "Datos insuficientes (necesitamos al menos 205 velas con indicadores válidos).", "⚪", "Baja"



    # -----------------------------------------------\n

    # 0. PERSISTENCIA Y ENTRENAMIENTO DEL MODELO ML

    # -----------------------------------------------

    

    is_model_loaded = False

    

    # Intenta cargar modelos y scaler del disco (si existen)

    if model_long is None:

        try:

            model_long = joblib.load(MODEL_LONG_FILE)

            model_short = joblib.load(MODEL_SHORT_FILE)

            scaler = joblib.load(SCALER_FILE)

            is_model_loaded = True

            print("💾 Modelos y Scaler cargados desde /tmp.")

        except:

            pass # Si falla (no existe el archivo), model_long/short seguirán siendo None



    if model_long is None or model_short is None:

        # Se requiere entrenamiento si la carga falló

        print("🚨 Reentrenamiento ML iniciado...")

        

        # Obtenemos datos de entrenamiento, incluyendo el nuevo scaler

        X_train, Y_Long, Y_Short, new_scaler = prepare_data_for_ml(df.iloc[-205:], scaler=None)



        scaler = new_scaler # Asignamos el nuevo scaler



        if X_train.empty:

            print("⚠️ Error al preparar datos de entrenamiento ML.")

            return "WAIT", "Error al preparar datos de entrenamiento ML.", "🔴", "RIESGO"





        model_long = LogisticRegression(solver='liblinear', random_state=42)

        model_long.fit(X_train, Y_Long)



        model_short = LogisticRegression(solver='liblinear', random_state=42)

        model_short.fit(X_train, Y_Short)



        # >> GUARDAR LOS MODELOS Y SCALER AL DISCO

        joblib.dump(model_long, MODEL_LONG_FILE)

        joblib.dump(model_short, MODEL_SHORT_FILE)

        joblib.dump(scaler, SCALER_FILE)

        

        long_acc = accuracy_score(Y_Long, model_long.predict(X_train))

        short_acc = accuracy_score(Y_Short, model_short.predict(X_train))

        

        print(f"🧠 Modelos ML reentrenados y GUARDADOS en /tmp. Precisión LONG: {long_acc*100:.2f}%.")

        return "WAIT", "Modelos ML en entrenamiento inicial. Esperar siguiente ciclo.", "⚪", "BAJA"



    # -----------------------------------------------\n

    # 1. TOMA DE DATOS E INDICADORES

    # -----------------------------------------------

    current_data = df.iloc[-1]

    last_price = current_data["price"]

    rsi = current_data["RSI"]

    bb_upper = current_data["BB_UPPER"]

    bb_lower = current_data["BB_LOWER"]

    ema_20 = current_data["EMA_20"]

    ema_50 = current_data["EMA_50"]

    atr = current_data["ATR"]

    ema_200 = current_data["EMA_200"]

    macd_hist = current_data["MACD_HIST"]

    adx = current_data["ADX"]



    # -----------------------------------------------\n

    # 2. DEFINICIÓN DE SEÑAL Y PROBABILIDAD (ML)

    # -----------------------------------------------



    final_signal = "WAIT"

    confidence_score = 0



    # Usamos el scaler GLOBAL (ya cargado) para NORMALIZAR los datos de predicción

    # prepare_data_for_ml retorna X_predict_final (una sola fila normalizada) en modo predicción

    # Usamos df.tail(20) para el contexto de normalización/predicción

    X_predict_final, _, _, _ = prepare_data_for_ml(df.tail(20), scaler) 



    if X_predict_final.empty:

        return "WAIT", "Error al preparar datos para predicción ML (DataFrame vacío).", "🔴", "RIESGO"



    # Predicción de probabilidades

    prob_long = model_long.predict_proba(X_predict_final)[0][1]

    prob_short = model_short.predict_proba(X_predict_final)[0][1]



    confidence_score_long = int(prob_long * 100)

    confidence_score_short = int(prob_short * 100)



    # --- Determinar la condición técnica más relevante y ZONAS DE RIESGO ---

    is_oversold_area = rsi < 35 or last_price <= bb_lower

    is_overbought_area = rsi > 65 or last_price >= bb_upper



    base_comment = "Predicción basada en patrón de volatilidad y momentum."

    if is_oversold_area:

        base_comment = "Cerca de Soporte/Sobreventa. Buscar rebote."

    elif is_overbought_area:

        base_comment = "Cerca de Resistencia/Sobrecompra. Buscar corrección."

    elif ema_20 > ema_50 and macd_hist > 0:

        base_comment = "Continuación Alcista (EMAs alineadas y MACD positivo)."

    elif ema_50 > ema_20 and macd_hist < 0:

        base_comment = "Continuación Bajista (EMAs alineadas y MACD negativo)."





    # Lógica de ML con Filtro de Zona

    if prob_long > PROB_LONG_MIN and prob_long > prob_short:

        # Descartar LONG si está en zona de Sobrecompra/Resistencia

        if is_overbought_area:

             final_signal = "WAIT"

             comment = f"ML LONG descartado: En zona de Resistencia/Sobrecompra ({rsi:.1f})."

        else:

             final_signal = "LONG"

             confidence_score = confidence_score_long

             comment = f"ML LONG. {base_comment} Probabilidad de éxito ({prob_long*100:.1f}%)"



    elif prob_short > PROB_SHORT_MIN and prob_short > prob_long:

        # Descartar SHORT si está en zona de Sobreventa/Soporte

        if is_oversold_area:

            final_signal = "WAIT"

            comment = f"ML SHORT descartado: En zona de Soporte/Sobreventa ({rsi:.1f})."

        else:

            final_signal = "SHORT"

            confidence_score = confidence_score_short

            comment = f"ML SHORT. {base_comment} Probabilidad de éxito ({prob_short*100:.1f}%)"



    else:

        final_signal = "WAIT"



        if abs(prob_long - prob_short) < 0.05:

            wait_reason = "Probabilidades de subida y bajada están muy parejas."

        elif prob_long > prob_short and prob_long < PROB_LONG_MIN:

            wait_reason = f"Predicción alcista es débil (solo {prob_long*100:.1f}%)."

        elif prob_short > prob_long and prob_short < PROB_SHORT_MIN:

            wait_reason = f"Predicción bajista es débil (solo {prob_short*100:.1f}%)."

        else:

            wait_reason = "Probabilidad de movimiento decisivo es insuficiente."



        comment = f"Mercado indefinido: {wait_reason}"

        confidence_score = max(confidence_score_long, confidence_score_short)





    # -----------------------------------------------\n

    # 3. FILTRO FINAL Y NIVEL DE CONFIANZA

    # -----------------------------------------------

    # ... (El resto del código de la Celda 4 es el mismo) ...

    # (Ya que solo se modificó la lógica de persistencia/entrenamiento)

    

    if confidence_score >= 80:

        conf_text = "ALTA"

        conf_emoji = "🔥"

    elif confidence_score >= 60:

        conf_text = "MEDIA"

        conf_emoji = "🟡"

    else:

        conf_text = "BAJA"

        conf_emoji = "⚪" 



    # --- FILTRO CLAVE: EXIGIR CONFIANZA ALTA (80%) ---

    if final_signal != "WAIT" and confidence_score < MIN_CONFIDENCE_FOR_SIGNAL:

        comment = f"ML descartado: Confianza ({confidence_score}%) por debajo del umbral ALTA ({MIN_CONFIDENCE_FOR_SIGNAL}%)."

        final_signal = "WAIT"

        conf_text = "BAJA"

        conf_emoji = "⚪"



    # --- FILTROS DE RIESGO ---

    is_lateral = abs(df["price"].iloc[-5] - last_price) < atr * INERTIA_ATR_MULTIPLIER



    if final_signal != "WAIT":

        if atr < MIN_ATR or is_lateral:

            final_signal = "WAIT"

            comment = f"ML descartado: Volatilidad muy baja (ATR {atr:.2f}) o lateralidad extrema."

            conf_text = "RIESGO"

            conf_emoji = "🔴"

            prev_signal = None

        else:

             prev_signal = final_signal

    else:

        # Si la señal es WAIT, verificamos inercia

        if prev_signal in ["LONG", "SHORT"]:

            last_signal_price = df["price"].iloc[-2]

            if abs(last_price - last_signal_price) < atr * INERTIA_ATR_MULTIPLIER:

                final_signal = prev_signal

                comment = f"Mercado en consolidación. Mantenemos señal {prev_signal} por inercia."

                conf_text = "MEDIA"

                conf_emoji = "🟡"

                confidence_score = INERTIA_CONFIDENCE

            else:

                 prev_signal = None



    # -----------------------------------------------\n

    # 4. CONSTRUCCIÓN DEL MENSAJE FINAL UNIFICADO (Con Emojis y SL/TP)

    # -----------------------------------------------



    stop_loss_val = 0.0

    take_profit_val = 0.0

    signal_color_emoji = "⚪"



    # EMOJIS DE CONTEXTO

    ema_trend = ema_20 > ema_50

    if ema_trend: trend_label = "Tendencia ⬆️ Alcista"

    else: trend_label = "Tendencia ⬇️ Bajista"



    if last_price > ema_200: ema200_label = f"Largo Plazo 🟢"

    else: ema200_label = f"Largo Plazo 🔴"



    macd_label = f"MACD Hist: {macd_hist:.4f} ({'Alcista' if macd_hist > 0 else 'Bajista'})"

    adx_label = f"ADX: {adx:.1f} ({'Fuerte' if adx > 25 else 'Débil/Lateral'})"



    # Importante: Necesitas 'pytz' y 'datetime' importados en la Celda 1

    ba_time = datetime.now(pytz.timezone("America/Argentina/Buenos_Aires")).strftime("%H:%M:%S")



    # --- CÁLCULO DE SL/TP (Solo si hay señal) ---

    if final_signal == "LONG":

        stop_loss_val = last_price - (atr * SL_MULTIPLIER)

        take_profit_val = last_price + (atr * TP_MULTIPLIER)

        min_tp_price = last_price + MIN_PROFIT_USD

        take_profit_val = max(take_profit_val, min_tp_price)

        signal_color_emoji = "🟢"

    elif final_signal == "SHORT":

        stop_loss_val = last_price + (atr * SL_MULTIPLIER)

        take_profit_val = last_price - (atr * TP_MULTIPLIER)

        max_tp_price = last_price - MIN_PROFIT_USD

        take_profit_val = min(take_profit_val, max_tp_price)

        signal_color_emoji = "🔴"



    # Clasificación ATR

    if atr > 10.0: atr_desc = "ALTA"

    elif atr < 5.0: atr_desc = "BAJA"

    else: atr_desc = "NORMAL"



    if "volume" in df.columns:

        vol_usd = df.iloc[-1]["volume"] * last_price

        vol_status = f"(${vol_usd/1e6:.1f}M)"

    else:

        vol_status = "(Volumen no disponible)"



    # --- DEFINIR CONTEXTO TÉCNICO UNIFICADO ---

    contexto_tecnico = (

        f"**Contexto Técnico:**\n"

        f"  - 💰 Precio: ${last_price:,.2f}\n"

        f"  - 📊 RSI(14): {rsi:.1f}\n"

        f"  - 📈 Volumen: {vol_status}\n"

        f"  - ⚡ Volatilidad (ATR): {atr:.4f} ({atr_desc})\n"

        f"  - {trend_label} (EMA 20/50)\n"

        f"  - {ema200_label} (EMA 200): {ema_200:,.2f}\n"

        f"  - {macd_label}\n"

        f"  - {adx_label}\n"

    )



    # --- CONSTRUCCIÓN DEL MENSAJE FINAL ---



    if final_signal != "WAIT":

        # MENSAJE LONG/SHORT

        if last_price >= bb_upper: zone_text = "resistencia"

        elif last_price <= bb_lower: zone_text = "soporte"

        elif ema_trend: zone_text = "tendencia alcista"

        else: zone_text = "tendencia bajista"



        message_content = (

            f"**SEÑAL: {final_signal}** ({SYMBOL}) | Zona: {zone_text}\n\n"

            f"{signal_color_emoji} **Comentario:** {comment.strip()}\n"

            f"**Confianza (ML):** {conf_text} ({confidence_score}%)\n\n"

            f"🔍 **Parámetros de Orden:**\n"

            f"  - 💡 Señal: {final_signal}\n"

            f"  - 📉 Stop Loss: {stop_loss_val:,.2f}\n"

            f"  - 📈 Take Profit: {take_profit_val:,.2f}\n\n"

            f"{contexto_tecnico}"

            f"⏰ Hora Buenos Aires: {ba_time}"

        )

    else:

        # MENSAJE WAIT

        message_content = (

            f"🛑 **SEÑAL: {final_signal}** ({SYMBOL})\n\n"

            f"{conf_emoji} **Comentario:** {comment.strip()}\n"

            f"**Confianza (ML):** {conf_text} ({confidence_score}%)\n\n"

            f"{contexto_tecnico}"

            f"⏰ Hora Buenos Aires: {ba_time}"

        )



    return final_signal, message_content, conf_emoji, conf_text

"""💬 CELDA 5 – Integración con Telegram"""



# CELDA 5 - Envío de alertas a Telegram



TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

CHAT_ID = os.environ.get("CHAT_ID")



def send_telegram_message(token, chat_id, message):

    """Envía mensaje al bot de Telegram"""

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:

        r = requests.post(url, data=payload)

        r.raise_for_status()

        print("✅ Mensaje enviado a Telegram.")

    except Exception as e:

        print("❌ Error enviando mensaje a Telegram:", e)



"""🔁 CELDA 6 – Monitoreo automático y envío de señales"""



# Requerimiento mínimo de velas para EMA 200/ADX es 200. Usamos 250 por seguridad.

SAMPLES = 250

# Tiempo de espera en segundos. Si INTERVAL es "5m", esperar 300 segundos.

interval_sec = 300

REMINDER_MINUTES = 30 # Recordatorio si no hay cambios en 30 minutos



# Nota: SYMBOL, INTERVAL, etc. son variables globales de la Celda 2.





def initialize_models(df):

    global model_long, model_short, scaler

    # ... (El código de inicialización que ya te envié está aquí, se omite por brevedad) ...

    # ... (El código de inicialización es el mismo) ...

    

    # 🛑 Esto debe ser el código completo de la función initialize_models que ya tienes

    

    # --------------------------------------------------

    # Esta es la parte crítica:

    if len(df) < 205:

        print("⚠️ DATOS INSUFICIENTES PARA ENTRENAMIENTO. Requeridos 205.")

        return False

    

    try:

        X_train, Y_Long, Y_Short, new_scaler = prepare_data_for_ml(df.iloc[-205:], scaler=None)



        model_long = LogisticRegression(solver='liblinear', random_state=42)

        model_long.fit(X_train, Y_Long)



        model_short = LogisticRegression(solver='liblinear', random_state=42)

        model_short.fit(X_train, Y_Short)

        

        scaler = new_scaler

        joblib.dump(model_long, MODEL_LONG_FILE)

        joblib.dump(model_short, MODEL_SHORT_FILE)

        joblib.dump(scaler, SCALER_FILE)

        

        print("✅ INICIALIZACIÓN COMPLETA. Modelos guardados en /tmp.")

        return True

        

    except Exception as e:

        print(f"❌ ERROR FATAL EN EL ENTRENAMIENTO INICIAL: {e}")

        return False

    

    # ... (Fin de la función initialize_models) ...

    # --------------------------------------------------



# >> FUNCIÓN DE EJECUCIÓN (run_bot)

def run_bot():

    global model_long, model_short, scaler



    # >> CORRECCIÓN CLAVE: Inicialización LOCAL de las variables de estado

    last_signal = "INIT"

    last_msg_time = datetime.now(pytz.timezone("America/Argentina/Buenos_Aires"))

    

    print("--------------------------------------------------")

    print("🚀 INICIO DEL PROCESO DE TRADING ML")



    # 1. PASO DE INICIALIZACIÓN Y CARGA DE MODELOS

    try:

        model_long = joblib.load(MODEL_LONG_FILE)

        model_short = joblib.load(MODEL_SHORT_FILE)

        scaler = joblib.load(SCALER_FILE)

        print("✅ Modelos ML cargados desde /tmp (Estado guardado).")

    except:

        print("⚠️ Modelos no encontrados. Iniciando ENTRENAMIENTO OBLIGATORIO.")

        

        df_initial = get_intraday_data(symbol=SYMBOL, interval=INTERVAL, samples=250)

        

        if not initialize_models(df_initial):

            print("❌ FALLO AL INICIALIZAR. Forzando el cierre del Worker.")

            sys.exit(1)





    # 2. BUCLE PRINCIPAL (while True) - SOLO PREDICCIÓN

    while True:

        # LLAMADA A LA API DE BINANCE

        df = get_intraday_data(symbol=SYMBOL, interval=INTERVAL, samples=250)



        if df.empty:

            print("⚠️ No se pudieron obtener datos. Reintentando...")

            time.sleep(30)

            continue



        # Generar la señal

        final_signal, message, conf_emoji, conf_text = generate_signal(df)

        ba_time = datetime.now(pytz.timezone("America/Argentina/Buenos_Aires"))



        # 3. Lógica de Envío y Recordatorio

        send_message = False



        # >> CORRECCIÓN: Uso de la variable 'last_signal' LOCAL

        if final_signal != last_signal: 

            send_message = True

        elif (ba_time - last_msg_time).total_seconds() >= REMINDER_MINUTES * 60:

            if final_signal == "WAIT":

                send_message = True



        if send_message:

            last_signal = final_signal

            last_msg_time = ba_time # Actualiza la variable LOCAL

            print(f"📡 Enviando señal a Telegram: {final_signal} | Confianza: {conf_text}")

            send_telegram_message(TELEGRAM_TOKEN, CHAT_ID, message)

        else:

            time_to_next = REMINDER_MINUTES * 60 - (ba_time - last_msg_time).total_seconds()

            mins = time_to_next / 60

            print(f"📭 Señal {final_signal} sin cambios. Próximo recordatorio en {mins:.1f} min.")



        time.sleep(interval_sec)





# >> LLAMADA FINAL AL SCRIPT

if __name__ == "__main__":

    run_bot()
