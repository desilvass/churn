#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# =============================
# 1. Cargar datos
# =============================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# =============================
# 2. Columnas
# =============================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# =============================
# 3. Modelos
# =============================
modelos = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

# =============================
# 4. Entrenamiento
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("models", exist_ok=True)

for nombre, modelo in modelos.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", modelo)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(pipe, f"models/{nombre}.pkl")

    print(f"✅ {nombre}.pkl guardado | Accuracy: {acc:.3f}")

print("🎉 TODOS LOS MODELOS FUERON GUARDADOS")

