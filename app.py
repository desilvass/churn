import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
st.set_page_config(
    page_title="Predicción de Churn",
    layout="wide",
    page_icon="🤖"
)

# ===============================
# CARGA DE MODELOS (ROBUSTA)
# ===============================
@st.cache_resource
def cargar_modelos():
    modelos = {}

    rutas = {
        "Regresión Logística": "models/logistic.pkl",
        "Gradient Boosting": "models/gradient_boosting.pkl",
    }

    for nombre, ruta in rutas.items():
        if not os.path.exists(ruta):
            st.warning(f"⚠️ Archivo no encontrado: {ruta}")
            continue

        try:
            modelos[nombre] = joblib.load(ruta)
        except Exception:
            st.warning(
                f"⚠️ No se pudo cargar {nombre} "
                f"(incompatibilidad de versiones)."
            )

    if not modelos:
        st.error("❌ No se pudo cargar ningún modelo.")
        st.stop()

    return modelos


# ===============================
# CARGA DE DATOS
# ===============================
@st.cache_data
def cargar_datos():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ===============================
# EDA
# ===============================
def seccion_eda():
    st.header("📊 Análisis Exploratorio de Datos (EDA)")

    df = cargar_datos()

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    st.subheader("Distribución de Churn")
    churn_counts = df["Churn"].value_counts()
    st.bar_chart(churn_counts)

    st.subheader("Resumen estadístico")
    st.write(df.describe())


# ===============================
# PREPROCESAMIENTO SIMPLE
# ===============================
def preparar_input(datos):
    df = pd.DataFrame([datos])

    # Conversión básica
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # One-hot encoding
    df = pd.get_dummies(df)

    return df


# ===============================
# PREDICCIÓN INDIVIDUAL
# ===============================
def seccion_prediccion():
    st.header("🤖 Predicción Individual de Churn")

    modelos = cargar_modelos()

    modelo_seleccionado = st.selectbox(
        "Seleccione el modelo",
        list(modelos.keys())
    )

    st.subheader("Ingrese los datos del cliente")

    genero = st.selectbox("Género", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    pareja = st.selectbox("Tiene pareja", ["Yes", "No"])
    dependientes = st.selectbox("Tiene dependientes", ["Yes", "No"])
    tenure = st.slider("Meses de permanencia", 0, 72, 12)
    monthly = st.number_input("Cargo mensual", 0.0, 200.0, 70.0)
    total = st.number_input("Total facturado", 0.0, 10000.0, 1000.0)

    if st.button("🔮 Predecir Churn"):
        datos = {
            "gender": genero,
            "SeniorCitizen": senior,
            "Partner": pareja,
            "Dependents": dependientes,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        X = preparar_input(datos)
        modelo = modelos[modelo_seleccionado]

        # Alinear columnas si el modelo fue entrenado con más features
        if hasattr(modelo, "feature_names_in_"):
            for col in modelo.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[modelo.feature_names_in_]

        pred = modelo.predict(X)[0]

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0][1]
            st.info(f"📈 Probabilidad de Churn: {proba:.2%}")

        if pred == 1 or pred == "Yes":
            st.error("❌ El cliente probablemente abandonará (Churn)")
        else:
            st.success("✅ El cliente probablemente NO abandonará")


# ===============================
# NAVEGACIÓN PRINCIPAL
# ===============================
def main():
    st.sidebar.title("🧭 Navegación")
    seccion = st.sidebar.radio(
        "Seleccione sección:",
        ["EDA", "Predicción"]
    )

    if seccion == "EDA":
        seccion_eda()
    else:
        seccion_prediccion()


if __name__ == "__main__":
    main()
