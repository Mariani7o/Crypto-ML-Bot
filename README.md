# 🤖 Crypto ML Trading Bot (Alta Confianza)

Signals for trading ETHUSDT.

## 🌟 Resumen y Objetivo

Este bot de trading algorítmico está diseñado para el mercado de criptomonedas (Binance) enfocándose en la **calidad (precisión) sobre la cantidad**. El objetivo es identificar movimientos de precio de alta probabilidad, filtrando la mayoría del ruido del mercado lateral y de baja volatilidad.

| Característica | Detalle |
| :--- | :--- |
| **Asset Principal** | `ETHUSDT` (Configurable) |
| **Intervalo de Operación** | **5 minutos (`5m`)** |
| **Modelo Central** | Regresión Logística (Entrenamiento dinámico) |
| **Filtro Clave** | **Confianza ML ≥ 80% (ALTA)** |
| **Riesgo/Recompensa** | Estricta relación R:R favorable (SL/TP dinámico por ATR). |
| **Despliegue** | **Render.com** (Worker 24/7). |

---

## ⚙️ Estrategia de Alta Precisión

El bot utiliza una combinación de indicadores técnicos y un modelo de Machine Learning (ML) con filtros muy restrictivos para asegurar la calidad de la señal.

### 1. Lógica del Modelo ML

El modelo predice movimientos significativos de **$0.3\%$ o más** en las siguientes 5 velas (25 minutos).

### 2. Filtros de Señal (Alta Confianza)

Solo se emite una señal **LONG** o **SHORT** si se cumplen simultáneamente:

1.  **Puntaje ML Alto:** La confianza de la predicción debe ser **80% o superior**.
2.  **Validación Técnica:** La señal no debe estar en una zona extrema (ej. No LONG en sobrecompra/resistencia).
3.  **Volatilidad Suficiente:** Se requiere un nivel mínimo de **ATR** para asegurar que la operación sea viable.

### 3. Gestión de Riesgo Dinámica (SL/TP)

Los niveles de Stop Loss (SL) y Take Profit (TP) se ajustan automáticamente a la volatilidad del momento, utilizando el **Average True Range (ATR)**.

---

## 🛠️ Despliegue y Configuración

El bot está configurado para correr continuamente en la nube. La ejecución se basa en los archivos `bot_main.py` y `requirements.txt`.

### Archivos Clave

| Archivo | Función |
| :--- | :--- |
| `bot_main.py` | Contiene toda la lógica de obtención de datos, indicadores, el entrenamiento del modelo y el bucle de ejecución 24/7. |
| `requirements.txt` | Lista de librerías Python necesarias para la instalación en el servidor (pip). |

### 🔒 Variables de Entorno (Seguridad)

Para mantener la seguridad, el bot lee sus claves de Telegram directamente desde el entorno del servidor (Render). **Estas claves deben configurarse en Render, no en el código.**

| Variable | Propósito |
| :--- | :--- |
| `TELEGRAM_TOKEN` | Token de su Bot de Telegram. |
| `CHAT_ID` | ID de su chat o canal de Telegram para recibir alertas. |

---
## 🤝 Contribuciones y Desarrollo

Este proyecto fue diseñado como una solución de trading personal. Se fomenta la experimentación y el desarrollo continuo.

Siéntase libre de **bifurcar (fork)** este repositorio para:

* Modificar la lógica de riesgo (ajustar SL/TP).
* Cambiar los umbrales de confianza (ej. mover el filtro ALTA del 80%).
* Experimentar con diferentes modelos de Machine Learning para optimizar el rendimiento y la precisión.
