import streamlit as st
import joblib
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from PIL import Image

# Cargar y mostrar el icono
icon_path = "iconoDS.png"  # Ajusta la ruta si es necesario
try:
    icon = Image.open(icon_path)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(icon, width=100)  # Ajusta el tamaño si lo deseas
    with col2:
        st.title("ValueTrak")
except FileNotFoundError:
    st.title("ValueTrak")  # Si no se encuentra la imagen, solo muestra el título

def load_model(ruta_activo):
    """Carga un modelo Prophet desde una ruta especificada."""
    return joblib.load(ruta_activo)

def graficar(train, pre, accion, dias):
    """Genera gráficos para datos históricos, reales y predicciones futuras."""
    train['ds'] = pd.to_datetime(train['ds'])
    pre['ds'] = pd.to_datetime(pre['ds'])

    # Separar datos históricos y futuros
    fecha_maxima = train['ds'].max()
    future_start_date = fecha_maxima + timedelta(days=1)

    historical = pre[pre['ds'] <= fecha_maxima]
    future = pre[pre['ds'] >= future_start_date]

    fig, ax = plt.subplots(figsize=(15, 6))

    # Predicciones históricas
    ax.plot(historical['ds'], historical['yhat'], color='blue', label='Predicción Histórica')

    # Predicciones futuras
    ax.plot(future['ds'], future['yhat'], color='red', label='Predicción Futura')

    # Datos reales
    ax.scatter(train['ds'], train['y'], color='black', label='Datos Reales', s=10)

    # Configuración del gráfico
    ax.set_title(f"Predicción para {accion}", fontsize=14)
    ax.set_xlabel('Fecha', fontsize=14)
    ax.set_ylabel('Valor', fontsize=14)
    ax.legend()

    st.pyplot(fig)

# Interfaz de usuario
st.write("Selecciona la cantidad de días y activo que quieras visualizar")
dias = st.slider("Número de días", 1, 30)
accion = st.selectbox('Activo', ['VTI', 'ORO', 'BTC'])

if st.button("Predecir"):
    # Diccionario para asociar cada activo con sus datos y modelo
    activos = {
        "VTI": ("train_VTI.csv", "vti_Prophet_default.pkl"),
        "ORO": ("train_oro.csv", "oro_Prophet_default.pkl"),
        "BTC": ("train_btc.csv", "btc_Prophet_default.pkl"),
    }

    # Cargar los datos y el modelo del activo seleccionado
    train_path, model_path = activos[accion]
    try:
        train = pd.read_csv(train_path)

        model = load_model(model_path)

        # Generar el DataFrame de fechas futuras
        future = model.make_future_dataframe(periods=dias)

        # Hacer predicciones
        prediction = model.predict(future)

        # Mostrar fechas de predicción
        fecha_maxima = train['ds'].max()
        fecha_corte = pd.to_datetime(fecha_maxima) + timedelta(days=dias)
        st.write(f"La predicción incluye fechas desde {fecha_maxima} hasta {fecha_corte}.")

        # Generar el gráfico
        graficar(train, prediction, accion, dias)

    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}")
    except ValueError as e:
        st.error(f"Error en los datos: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
