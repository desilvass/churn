#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =====================================
# IMPORTS
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# =====================================
# CONFIGURACIÓN
# =====================================
st.set_page_config(
    page_title="Sistema Predicción Churn + EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CARGA DE MODELOS
# =====================================
@st.cache_resource
def cargar_modelos():
    modelos = {
        "Regresión Logística": joblib.load("models/logistic.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "Gradient Boosting": joblib.load("models/gradient_boosting.pkl"),
    }
    return modelos

# =====================================
# CARGA DE DATOS
# =====================================
@st.cache_data
def cargar_datos():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================
# SECCIÓN EDA
# =====================================
def seccion_eda(df):

    st.header("📊 ANÁLISIS EXPLORATORIO DE DATOS")

    tab1, tab2, tab3 = st.tabs(
        ["📋 Vista General", "📈 Distribuciones", "🎯 Churn"]
    )

    # ---- TAB 1
    with tab1:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total registros", len(df))
            with col2:
                churn_rate = (df["Churn"] == "Yes").mean() * 100
                st.metric("Tasa churn", f"{churn_rate:.1f}%")

            buffer = StringIO()
            df.info(buf=buffer)
            st.text_area(
                "Información del Dataset",
                buffer.getvalue(),
                height=230,
                key="eda_info"
            )

    # ---- TAB 2
    with tab2:
        with st.container():
            variable = st.selectbox(
                "Selecciona variable",
                [c for c in df.columns if c != "customerID"],
                key="eda_variable"
            )

            fig, ax = plt.subplots(figsize=(9,5))
            if df[variable].dtype != "object":
                df[variable].hist(ax=ax, bins=30)
            else:
                df[variable].value_counts().head(10).plot.bar(ax=ax)

            ax.set_title(f"Distribución de {variable}")
            st.pyplot(fig, clear_figure=True)

    # ---- TAB 3
    with tab3:
        with st.container():
            var = st.selectbox(
                "Analizar churn por",
                [c for c in df.columns if c not in ["customerID", "Churn"]],
                key="eda_churn_var"
            )

            ct = pd.crosstab(df[var], df["Churn"], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(10,5))
            ct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Porcentaje (%)")
            ax.set_title(f"Tasa de Churn por {var}")
            st.pyplot(fig, clear_figure=True)

# =====================================
# SECCIÓN PREDICCIÓN
# =====================================
def seccion_prediccion():

    st.header("🤖 Predicción Individual de Churn")

    modelos = cargar_modelos()

    datos = {}
    col1, col2 = st.columns(2)

    with col1:
        datos["Contract"] = st.selectbox(
            "Contrato",
            ["Month-to-month", "One year", "Two year"],
            key="pred_contract"
        )
        datos["tenure"] = st.number_input(
            "Antigüedad (meses)",
            0, 100, 12,
            key="pred_tenure"
        )
        datos["MonthlyCharges"] = st.number_input(
            "Cargos mensuales ($)",
            0.0, 200.0, 50.0,
            key="pred_charges"
        )

    with col2:
        datos["PaymentMethod"] = st.selectbox(
            "Método de pago",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            key="pred_payment"
        )
        datos["OnlineSecurity"] = st.selectbox(
            "Seguridad online",
            ["Yes", "No"],
            key="pred_security"
        )
        datos["TechSupport"] = st.selectbox(
            "Soporte técnico",
            ["Yes", "No"],
            key="pred_support"
        )

    if st.button("🔮 PREDECIR CHURN", key="pred_button"):

        X = pd.DataFrame([datos])

        resultados = {}
        for nombre, modelo in modelos.items():
            prob = modelo.predict_proba(X)[0][1]
            resultados[nombre] = prob

        prob_media = np.mean(list(resultados.values()))
        riesgo = int(prob_media * 100)

        st.markdown("---")
        st.subheader("📊 Resultados por modelo")

        cols = st.columns(len(resultados))
        for col, (nombre, prob) in zip(cols, resultados.items()):
            with col:
                st.metric(nombre, f"{prob:.1%}")

        st.markdown("---")
        st.metric("Riesgo promedio de churn", f"{riesgo}/100")

        if riesgo > 70:
            st.error("🚨 Alto riesgo de churn")
        elif riesgo > 50:
            st.warning("⚠️ Riesgo medio de churn")
        else:
            st.success("✅ Riesgo bajo de churn")

# =====================================
# MAIN
# =====================================
def main():

    df = cargar_datos()

    st.sidebar.title("🧭 Navegación")
    seccion = st.sidebar.radio(
        "Seleccione sección:",
        ["EDA", "Predicción"],
        key="sidebar_nav"
    )

    if seccion == "EDA":
        seccion_eda(df)
    else:
        seccion_prediccion()

if __name__ == "__main__":
    main()

