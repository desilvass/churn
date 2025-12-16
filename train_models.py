#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# =============================
# CARGAR DATA
# =============================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

# =============================
# PREPROCESAMIENTO
# =============================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# =============================
# MODELOS
# =============================
modelos = {
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "gradient_boosting": GradientBoostingClassifier(),
    "decision_tree": DecisionTreeClassifier(class_weight="balanced"),
    "knn": KNeighborsClassifier()
}

os.makedirs("models", exist_ok=True)

# =============================
# ENTRENAR Y GUARDAR
# =============================
for nombre, modelo in modelos.items():
    print(f"Entrenando {nombre}...")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", modelo)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)

    joblib.dump(pipe, f"models/{nombre}.pkl")
    print(f"✔ models/{nombre}.pkl guardado")

print("🎉 Todos los modelos fueron generados correctamente")

