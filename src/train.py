import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Incarcam datasetul
df = pd.read_csv("data/cars_featured.csv")

# Separam caracteristicile de variabila tinta
X = df.drop("priceUSD", axis=1)
y = df["priceUSD"]

# Impartim datele in antrenare si testare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identificam coloanele categorice si numerice
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Construim preprocesorul
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Aplicam preprocesarea
X_train_processed = preprocessor.fit_transform(X_train)

# Cream modelul Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# Antrenam modelul
model.fit(X_train_processed, y_train)

# Salvam modelul si preprocesorul
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.joblib")
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("Modelul Random Forest a fost antrenat si salvat cu succes!")
