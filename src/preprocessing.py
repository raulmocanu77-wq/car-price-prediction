import pandas as pd
import numpy as np

# Incarcam datasetul cu caracteristicile noi
df = pd.read_csv("data/cars_featured.csv")
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import joblib
import os


# Separam caracteristicile de variabila tinta
X = df.drop("priceUSD", axis=1)
y = df["priceUSD"]

print("Dimensiunea lui X:", X.shape)
print("Dimensiunea lui y:", y.shape)

from sklearn.model_selection import train_test_split

# Impartim datele in set de antrenare si set de testare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dimensiunea X_train:", X_train.shape)
print("Dimensiunea X_test:", X_test.shape)
print("Dimensiunea y_train:", y_train.shape)
print("Dimensiunea y_test:", y_test.shape)

# Identificam coloanele categorice
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

# Identificam coloanele numerice
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("Coloane categorice:", categorical_columns)
print("Numar coloane categorice:", len(categorical_columns))

print("Coloane numerice:", numerical_columns)
print("Numar coloane numerice:", len(numerical_columns))

# Construim preprocesorul
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)
# Aplicam preprocesorul pe datele de antrenare
X_train_processed = preprocessor.fit_transform(X_train)

# Aplicam aceleasi transformari pe datele de testare
X_test_processed = preprocessor.transform(X_test)

print("Dimensiunea datelor de antrenare dupa preprocessing:", X_train_processed.shape)
print("Dimensiunea datelor de testare dupa preprocessing:", X_test_processed.shape)

# Salvam preprocesorul
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("Preprocesorul a fost salvat cu succes!")