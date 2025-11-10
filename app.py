# app.py
# Tablero de Digitalización de Planta - Streamlit
# Integración InfluxDB + Tablero industrial estático (simulado) + método predictivo simple
#
# Requisitos: streamlit, pandas, numpy, influxdb-client, plotly, altair

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import altair as alt
import plotly.express as px
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi
from typing import Tuple, Optional

st.set_page_config(
    page_title="📊 Tablero — Digitalización de Planta",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# ---------- CONFIG ------------
# ------------------------------
# (Credenciales provistas por el usuario)
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "JcKXoXE30JQvV9Ggb4-zv6sQc0Zh6B6Haz5eMRW0FrJEduG2KcFJN9-7RoYvVORcFgtrHR-Q_ly-52pD7IC6JQ=="
INFLUXDB_ORG = "0925ccf91ab36478"
INFLUXDB_BUCKET = "EXTREME_MANUFACTURING"

# ------------------------------
# ------- UTILS / HELPERS -------
# ------------------------------
def conectar_influx(url: str, token: str, org: str) -> Optional[InfluxDBClient]:
    """Intenta crear un cliente de InfluxDB. Devuelve None si falla."""
    try:
        client = InfluxDBClient(url=url, token=token, org=org, timeout=20_000)
        # Simple ping: list buckets or call ready property
        _ = client.health()
        return client
    except Exception as e:
        st.warning(f"No se pudo conectar a InfluxDB: {e}")
        return None

def consulta_influx(query_api: QueryApi, flux_query: str, org: str) -> pd.DataFrame:
    """Ejecuta una consulta Flux y regresa DataFrame consolidado."""
    try:
        res = query_api.query_data_frame(org=org, query=flux_query)
        # query_data_frame puede devolver lista de DF
        if isinstance(res, list):
            if len(res) == 0:
                return pd.DataFrame()
            df = pd.concat(res, ignore_index=True)
        else:
            df = res
        return df
    except Exception as e:
        st.error(f"Error en consulta InfluxDB: {e}")
        return pd.DataFrame()

@st.cache_data
def generar_datos_industriales_simulados() -> pd.DataFrame:
    """Genera un DataFrame simulado (usa tu tablero estático como base)."""
    np.random.seed(42)
    fechas = pd.date_range(start=datetime.now() - timedelta(days=3), periods=24*6*3, freq='30min')
    datos = {
        'Fecha': fechas,
        'temperatura': 250 + np.random.normal(0, 10, len(fechas)),      # Temperatura_Reactor_1
        'presion': 15 + np.random.normal(0, 2, len(fechas)),           # Presion_Sistema
        'flujo_entrada': 100 + np.random.normal(0, 5, len(fechas)),
        'nivel_tanque': 75 + np.random.normal(0, 8, len(fechas)),
        'consumo_energia': 450 + np.random.normal(0, 25, len(fechas)),
        'ph_proceso': 7.2 + np.random.normal(0, 0.3, len(fechas)),
        'vibration_motor': 0.5 + np.random.normal(0, 0.1, len(fechas)),
        'eficiencia': 85 + np.random.normal(0, 5, len(fechas))
    }
    df = pd.DataFrame(datos)
    return df

def preparar_df_influx(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza un DataFrame retornado por query_data_frame de Influx
    y lo deja con columnas: Tiempo, Variable, Valor
    o, si ya viene en formato pivot, intenta pivotear.
    """
    if df_raw.empty:
        return df_raw

    # Casos comunes: Influx devuelve columnas ['_time','_field','_value'] o ya pivoted
    if set(['_time', '_field', '_value']).issubset(df_raw.columns):
        df = df_raw[['_time', '_field', '_value']].rename(
            columns={'_time': 'Tiempo', '_field': 'Variable', '_value': 'Valor'})
        df['Tiempo'] = pd.to_datetime(df['Tiempo'])
        return df
    else:
        # Try to pivot time series where each field is a column
        # If there's a time column
        time_cols = [c for c in df_raw.columns if 'time' in c.lower() or '_time' in c.lower()]
        if len(time_cols) > 0:
            tcol = time_cols[0]
            df_raw = df_raw.rename(columns={tcol: 'Tiempo'})
            df_raw['Tiempo'] = pd.to_datetime(df_raw['Tiempo'])
            # if many numeric columns, melt them
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                df_melt = df_raw.melt(id_vars=['Tiempo'], value_vars=numeric_cols,
                                      var_name='Variable', value_name='Valor')
                return df_melt
        # fallback: return original
        return df_raw

def calcular_metricas_rapidas(df: pd.DataFrame, variable: str) -> Tuple[float, float, float, Optional[pd.Timestamp]]:
    """Calcula el último valor, promedio, min, max para una variable en df (DataFrame con Tiempo, Variable, Valor o columnas pivot)."""
    if df.empty:
        return (np.nan, np.nan, np.nan, None)

    if set(['Tiempo', 'Variable', 'Valor']).issubset(df.columns):
        sub = df[df['Variable'] == variable].sort_values('Tiempo')
        if sub.empty:
            return (np.nan, np.nan, np.nan, None)
        ultimo = sub['Valor'].iloc[-1]
        promedio = sub['Valor'].mean()
        minimo = sub['Valor'].min()
        maximo = sub['Valor'].max()
        tiempo_ultimo = sub['Tiempo'].iloc[-1]
        return (ultimo, promedio, minimo, tiempo_ultimo)
    else:
        # Pivot style
        if variable not in df.columns:
            return (np.nan, np.nan, np.nan, None)
        serie = df[variable].dropna().astype(float)
        if serie.empty:
            return (np.nan, np.nan, np.nan, None)
        return (serie.iloc[-1], serie.mean(), serie.min(), df['Tiempo'].iloc[-1] if 'Tiempo' in df.columns else None)

def generar_pronostico_promedio_movil(serie: pd.Series, window: int, pasos_futuros: int) -> pd.Series:
    """Pronóstico simple: extendemos el promedio móvil como valor constante para los pasos futuros."""
    rol = serie.rolling(window=window, min_periods=1).mean()
    ultimo = rol.dropna().iloc[-1]
    fut = pd.Series([ultimo] * pasos_futuros, 
                    index=[serie.index[-1] + i * (serie.index[1] - serie.index[0]) for i in range(1, pasos_futuros+1)])
    return fut

def generar_pronostico_ewm(serie: pd.Series, alpha: float, pasos_futuros: int) -> pd.Series:
    """Suavizado exponencial simple: usamos el último valor suavizado como pronóstico repetido."""
    ewm = serie.ewm(alpha=alpha, adjust=False).mean()
    ultimo = ewm.iloc[-1]
    # calcular intervalo de tiempo medio
    if len(serie.index) >= 2 and isinstance(serie.index, pd.DatetimeIndex):
        delta = serie.index[1] - serie.index[0]
    else:
        delta = pd.Timedelta(minutes=30)
    future_index = [serie.index[-1] + (i+1) * delta for i in range(pasos_futuros)]
    fut = pd.Series([ultimo] * pasos_futuros, index=future_index)
    return fut

# ------------------------------
# ---------- LAYOUT ------------
# ------------------------------
# Top header (usar parte de CSS del tablero estático)
st.markdown(
    """
    <style>
    .main-header { font-size: 2.0rem; font-weight: bold; color: #1f77b4; text-align: left; }
    .kpi-card { background: white; padding: 0.6rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);}
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="main-header">🏭 Tablero — Digitalización de Planta (Temperatura / Humedad / Vibración)</div>', unsafe_allow_html=True)
st.write("Proyecto final — EAFIT · Digitalización de Plantas Productivas")

# Sidebar: opciones generales
st.sidebar.header("⚙️ Controles")
use_simulated = st.sidebar.checkbox("Usar datos simulados (sin InfluxDB)", value=False)
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# Conexión Influx (si aplica)
influx_client = None
query_api = None
if not use_simulated:
    influx_client = conectar_influx(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG)
    if influx_client:
        query_api = influx_client.query_api()

# Elección sensor / measurement
sensor_option = st.sidebar.selectbox("Seleccionar fuente:", ["DHT22 (temp/hum)", "MPU6050 (vibración)", "Simulado (tablero)"])

# Rango de fecha (si usamos Influx, permitimos seleccionar por días; si simulado, por fechas disponibles)
if use_simulated or sensor_option == "Simulado (tablero)":
    df_sim = generar_datos_industriales_simulados()
    fecha_min = df_sim['Fecha'].min().date()
    fecha_max = df_sim['Fecha'].max().date()
    fecha_inicio = st.sidebar.date_input("Fecha inicio", value=fecha_min, min_value=fecha_min, max_value=fecha_max)
    fecha_fin = st.sidebar.date_input("Fecha fin", value=fecha_max, min_value=fecha_min, max_value=fecha_max)
    # Convertir a datetime range inclusive
    start_dt = datetime.combine(fecha_inicio, datetime.min.time())
    end_dt = datetime.combine(fecha_fin, datetime.max.time())
else:
    # Para Influx: seleccionar rango relativo (en días)
    dias = st.sidebar.slider("Rango (días hacia atrás)", min_value=1, max_value=30, value=7)
    start_dt = f"-{dias}d"
    end_dt = "now()"

# Selector de variables específicas (para Influx se listan por measurement)
variables_seleccionadas = None  # definiremos luego según el DF cargado

# ------------------------------
# -------- CARGA DATOS ---------
# ------------------------------
df_final = pd.DataFrame()

if use_simulated or sensor_option == "Simulado (tablero)":
    df_final = df_sim.copy()
    # renombrar columnas a formato más amigable
    df_final = df_final.rename(columns={'Fecha': 'Tiempo'})
    # convertir Tiempo a datetime index para modelos
    df_final['Tiempo'] = pd.to_datetime(df_final['Tiempo'])
    df_final = df_final.set_index('Tiempo')
    # llenar missing
else:
    # Construir consulta Flux según sensor_option
    if query_api is None:
        st.warning("No hay conexión a InfluxDB: activa 'Usar datos simulados' o revisa credenciales.")
        # fallback a simulados
        df_final = generar_datos_industriales_simulados().rename(columns={'Fecha': 'Tiempo'})
        df_final['Tiempo'] = pd.to_datetime(df_final['Tiempo'])
        df_final = df_final.set_index('Tiempo')
        st.info("Usando datos simulados como fallback.")
    else:
        if sensor_option == "DHT22 (temp/hum)":
            measurement = "studio-dht22"
            fields_filter = 'r._field == "humedad" or r._field == "temperatura" or r._field == "sensacion_termica"'
        else:
            measurement = "mpu6050"
            fields_filter = ('r._field == "accel_x" or r._field == "accel_y" or r._field == "accel_z" or '
                             'r._field == "gyro_x" or r._field == "gyro_y" or r._field == "gyro_z" or '
                             'r._field == "temperature"')

        flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {start_dt}, stop: {end_dt})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> filter(fn: (r) => {fields_filter})
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        raw = consulta_influx(query_api, flux_query, INFLUXDB_ORG)
        if raw.empty:
            st.warning("La consulta a InfluxDB no devolvió datos para el rango solicitado.")
            df_final = generar_datos_industriales_simulados().rename(columns={'Fecha': 'Tiempo'})
            df_final['Tiempo'] = pd.to_datetime(df_final['Tiempo'])
            df_final = df_final.set_index('Tiempo')
            st.info("Usando datos simulados como fallback.")
        else:
            # preparar DataFrame: si ya está pivoted, usar columnas numéricas y tiempo
            df_prep = preparar_df_influx(raw)
            # Si viene en formato Tiempo,Variable,Valor -> pivot para facilitar uso
            if set(['Tiempo', 'Variable', 'Valor']).issubset(df_prep.columns):
                df_pivot = df_prep.pivot(index='Tiempo', columns='Variable', values='Valor').sort_index()
                df_pivot.index = pd.to_datetime(df_pivot.index)
                df_final = df_pivot
            else:
                # si ya tiene columnas por campo y un time column
                if 'Tiempo' in df_prep.columns:
                    df_prep['Tiempo'] = pd.to_datetime(df_prep['Tiempo'])
                    df_final = df_prep.set_index('Tiempo').sort_index()
                else:
                    # fallback
                    df_final = df_prep

# A partir de df_final con índice de tiempo, definimos variables disponibles
if df_final.empty:
    st.error("No hay datos disponibles.")
    st.stop()

variables_disponibles = [c for c in df_final.columns if pd.api.types.is_numeric_dtype(df_final[c])]
if not variables_disponibles:
    st.error("No hay columnas numéricas en los datos.")
    st.stop()

variables_seleccionadas = st.sidebar.multiselect("Variables a mostrar", variables_disponibles, default=variables_disponibles[:4])

# ------------------------------------------------
# --------- DASHBOARD PRINCIPAL (visual) ---------
# ------------------------------------------------
st.markdown("## 📈 Visualizaciones y Métricas")

# Top metrics: mostrar por variable seleccionada (hasta 4 en fila)
top_cols = st.columns(min(4, len(variables_seleccionadas) or 1))
for i, var in enumerate(variables_seleccionadas[:4]):
    serie = df_final[var].dropna()
    if serie.empty:
        top_cols[i].metric(label=var, value="N/A")
        continue
    ultimo, promedio, minimo, tiempo_ultimo = calcular_metricas_rapidas(
        df_final.reset_index().melt(id_vars='Tiempo', value_vars=[var], var_name='Variable', value_name='Valor'),
        var
    )
    # si calcular_metricas_rapidas no funciona para pivot style:
    if np.isnan(ultimo):
        ultimo = serie.iloc[-1]
        promedio = serie.mean()
        minimo = serie.min()
        maximo = serie.max()
    else:
        maximo = df_final[var].max()

    top_cols[i].metric(label=var, value=f"{ultimo:.2f}", delta=f"avg {promedio:.2f}")

# Gráficas por variable seleccionada (plotly)
for var in variables_seleccionadas:
    st.markdown(f"### {var.replace('_',' ').title()}")
    df_plot = df_final[[var]].dropna().reset_index()
    if df_plot.empty:
        st.info("No hay datos para esta variable.")
        continue

    # line + rolling average overlay
    fig = px.line(df_plot, x=df_plot.columns[0], y=var, title=f"{var}", template="plotly_white")
    # add rolling mean as an extra trace
    window_default = min(12, max(1, int(len(df_plot)/10)))
    df_plot['rolling_mean'] = df_plot[var].rolling(window=window_default, min_periods=1).mean()
    fig.add_scatter(x=df_plot[df_plot.columns[0]], y=df_plot['rolling_mean'],
                    mode='lines', name=f'Rolling mean ({window_default})', line=dict(dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# ---- PREDICTIVE / FORECAST ----
# ------------------------------
st.markdown("## 🤖 Análisis Predictivo (sencillo)")

predict_var = st.selectbox("Variable para predecir", options=variables_seleccionadas or variables_disponibles)
pasos_futuros = st.number_input("Pasos futuros a predecir (horizonte)", min_value=1, max_value=48, value=6, step=1)
metodo = st.radio("Método predictivo", options=["Promedio móvil", "Suavizado exponencial (EWM)"])

# Parámetros
if metodo == "Promedio móvil":
    window = st.slider("Ventana promedio móvil (n puntos)", min_value=1, max_value=48, value=6)
else:
    alpha = st.slider("Alpha (0-1) para EWM", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# Preparar serie con índice datetime
serie = df_final[predict_var].dropna()
if serie.empty:
    st.info("No hay datos numéricos suficientes para la variable seleccionada.")
else:
    # Asegurar DatetimeIndex
    if not isinstance(serie.index, pd.DatetimeIndex):
        # try to convert
        try:
            serie.index = pd.to_datetime(serie.index)
        except Exception:
            # make a simple range
            serie.index = pd.date_range(start=datetime.now() - timedelta(hours=len(serie)), periods=len(serie), freq='30min')
    # ejecutar método
    if metodo == "Promedio móvil":
        # generar pronóstico simple: tomar rolling mean y replicar
        fut_series = generar_pronostico_promedio_movil(serie, window=window, pasos_futuros=pasos_futuros)
    else:
        fut_series = generar_pronostico_ewm(serie, alpha=alpha, pasos_futuros=pasos_futuros)

    # visualizar serie histórica + pronóstico
    df_hist = serie.reset_index().rename(columns={serie.index.name or 'index': 'Tiempo', serie.name: 'Valor'})
    df_hist.columns = ['Tiempo', 'Valor']
    df_fore = fut_series.reset_index()
    df_fore.columns = ['Tiempo', 'Valor']

    # Combine and flag
    df_hist['Tipo'] = 'Histórico'
    df_fore['Tipo'] = 'Pronóstico'
    df_comb = pd.concat([df_hist, df_fore], ignore_index=True)



    # Show recent numeric summary
    st.markdown("### Resumen del pronóstico")
    st.write(df_fore.set_index('Tiempo').round(3))



# ------------------------------
# --------- ALARMAS -------------
# ------------------------------
st.markdown("## 🚨 Sistema de alarmas (simple)")
# Definir umbrales demo (puedes adaptar a tus rangos reales)
umbrales = {
    'temperatura': {'min': 230, 'max': 270},
    'presion': {'min': 10, 'max': 20},
    'flujo_entrada': {'min': 80, 'max': 120},
    'nivel_tanque': {'min': 40, 'max': 100},
    'vibration_motor': {'min': 0, 'max': 1.0},
    'eficiencia': {'min': 70, 'max': 100}
}

alarmas = []
ultimo_ts = None
for var in variables_seleccionadas:
    if var in df_final.columns:
        s = df_final[var].dropna()
        if s.empty:
            continue
        ultimo = s.iloc[-1]
        ultimo_ts = s.index[-1]
        if var in umbrales:
            u = umbrales[var]
            if not (u['min'] <= ultimo <= u['max']):
                gravedad = "Alta" if (ultimo < u['min'] * 0.9 or ultimo > u['max'] * 1.1) else "Media"
                accion = "Inspección inmediata" if gravedad == "Alta" else "Monitoreo"
                alarmas.append({
                    'Variable': var,
                    'Valor': f"{ultimo:.2f}",
                    'Gravedad': gravedad,
                    'Acción': accion
                })

if alarmas:
    st.table(pd.DataFrame(alarmas))
else:
    st.success("✅ No se detectaron alarmas con los umbrales actuales")

# ------------------------------
# --------- FOOTER --------------
# ------------------------------
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("🔧 **Sistema de Monitoreo Industrial**")
with c2:
    st.markdown(f"🕒 Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with c3:
    st.markdown(f"📡 Estado conexión: {'✅ Conectado a InfluxDB' if (influx_client and not use_simulated) else '⚠️ Simulado / sin conexión'}")

# Auto refresh
if auto_refresh:
    time.sleep(30)
    st.experimental_rerun()
