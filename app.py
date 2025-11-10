import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient
import plotly.express as px

# --- Configuración de conexión ---
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "JcKXoXE30JQvV9Ggb4-zv6sQc0Zh6B6Haz5eMRW0FrJEduG2KcFJN9-7RoYvVORcFgtrHR-Q_ly-52pD7IC6JQ=="
INFLUXDB_ORG = "0925ccf91ab36478"
INFLUXDB_BUCKET = "EXTREME_MANUFACTURING"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

# --- Interfaz de usuario ---
st.title("📊 Tablero de Digitalización de Planta")
st.write("Visualización de datos desde InfluxDB en tiempo real.")

sensor = st.selectbox("Selecciona el sensor:", ["DHT22", "MPU6050"])
start = st.slider(
    "Selecciona el rango de tiempo de inicio (start):",
    min_value=1, max_value=15, value=15
)

stop = st.slider(
    "Selecciona el rango de tiempo de finalización (stop):",
    min_value=5, max_value=15, value=9
)

# --- Consulta dinámica ---
if sensor == "DHT22":
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -{start}d, stop: -{stop}d)
        |> filter(fn: (r) => r._measurement == "studio-dht22")
        |> filter(fn: (r) => r._field == "humedad" or r._field == "temperatura" or r._field == "sensacion_termica")
    '''
else:
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -{start}d, stop: -{stop}d)
        |> filter(fn: (r) => r._measurement == "mpu6050")
        |> filter(fn: (r) =>
            r._field == "accel_x" or r._field == "accel_y" or r._field == "accel_z" or
            r._field == "gyro_x" or r._field == "gyro_y" or r._field == "gyro_z" or
            r._field == "temperature")
    '''

# --- Cargar datos ---
try:
    df = query_api.query_data_frame(org=INFLUXDB_ORG, query=query)
    if isinstance(df, list):
        df = pd.concat(df)
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# --- Limpieza de datos ---
if df.empty:
    st.warning("⚠️ No se encontraron datos para el rango seleccionado.")
    st.stop()

df = df[["_time", "_field", "_value"]]
df = df.rename(columns={"_time": "Tiempo", "_field": "Variable", "_value": "Valor"})
df["Tiempo"] = pd.to_datetime(df["Tiempo"])

# --- Gráficos ---
st.subheader("📈 Visualización de variables")

for var in df["Variable"].unique():
    sub_df = df[df["Variable"] == var]
    fig = px.line(sub_df, x="Tiempo", y="Valor", title=f"{var}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(df.describe())
