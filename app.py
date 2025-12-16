import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIGURACIÓN =====================
st.set_page_config(
    page_title="Sistema Predicción Churn + EDA",
    page_icon="📊",
    layout="wide"
)

MODELOS_PKL = {
    "Regresión Logística": "models/logistic.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "Árbol de Decisión": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
}

VARIABLES_RELEVANTES = [
    'Contract', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'InternetService', 'OnlineSecurity', 'TechSupport',
    'PaymentMethod', 'PaperlessBilling', 'SeniorCitizen'
]

TODAS_VARIABLES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# ===================== CARGA MODELOS =====================
@st.cache_resource
def cargar_modelos():
    modelos = {}
    for nombre, ruta in MODELOS_PKL.items():
        modelos[nombre] = joblib.load(ruta)
    return modelos

# ===================== CARGA DATOS =====================
@st.cache_data
def cargar_datos():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

# ===================== EDA =====================
def seccion_eda(df):
    st.header("📊 Análisis Exploratorio de Datos")
    st.dataframe(df.head(), use_container_width=True)

    if "Churn" in df.columns:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        st.metric("Tasa de Churn", f"{churn_rate:.2f}%")

    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ===================== PREDICCIÓN =====================
def seccion_prediccion(df, modelos):
    st.header("🔮 Predicción Individual de Churn")

    usar_todas = st.radio(
        "Variables a usar:",
        ["Variables Relevantes", "Todas las Variables"]
    ) == "Todas las Variables"

    variables = TODAS_VARIABLES if usar_todas else VARIABLES_RELEVANTES
    variables = [v for v in variables if v in df.columns]

    datos_cliente = {}
    col1, col2 = st.columns(2)

    for i, var in enumerate(variables):
        with col1 if i % 2 == 0 else col2:
            if df[var].dtype == "object":
                datos_cliente[var] = st.selectbox(var, df[var].unique())
            else:
                datos_cliente[var] = st.number_input(
                    var, value=float(df[var].median())
                )

    modelos_seleccionados = st.multiselect(
        "Modelos a evaluar:",
        list(modelos.keys()),
        default=list(modelos.keys())
    )

    if st.button("🚀 Predecir"):
        cliente_df = pd.DataFrame([datos_cliente])
        resultados = {}

        for nombre in modelos_seleccionados:
            modelo = modelos[nombre]

            for col in modelo.feature_names_in_:
                if col not in cliente_df.columns:
                    cliente_df[col] = df[col].mode()[0]

            cliente_df = cliente_df[modelo.feature_names_in_]

            proba = modelo.predict_proba(cliente_df)[0, 1]
            pred = "CHURN" if proba >= 0.5 else "NO CHURN"

            resultados[nombre] = proba

            st.metric(
                nombre,
                pred,
                f"{proba:.1%}"
            )

        st.markdown("---")
        st.subheader("📊 Consenso")
        st.metric(
            "Probabilidad promedio",
            f"{np.mean(list(resultados.values())):.1%}"
        )

# ===================== MAIN =====================
def main():
    st.title("📱 Sistema de Predicción de Churn")

    df = cargar_datos()
    modelos = cargar_modelos()

    seccion = st.sidebar.radio(
        "🧭 Navegación",
        ["EDA", "Predicción"]
    )

    if seccion == "EDA":
        seccion_eda(df)
    else:
        seccion_prediccion(df, modelos)

if __name__ == "__main__":
    main()
