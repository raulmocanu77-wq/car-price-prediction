import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Incarcam datasetul cu caracteristici noi
df = pd.read_csv("data/cars_featured.csv")

# Separam caracteristicile de variabila tinta
X = df.drop("priceUSD", axis=1)
y = df["priceUSD"]

# Impartim datele in antrenare si testare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Incarcam preprocesorul si modelul antrenat
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/random_forest_model.joblib")

print("Datele, preprocesorul si modelul au fost incarcate cu succes!")

# Aplicam preprocesorul pe datele de testare
X_test_processed = preprocessor.transform(X_test)

# Generam predictiile
y_pred = model.predict(X_test_processed)

print("Predictiile au fost generate cu succes!")

# Calculam metricile de evaluare
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Afisam rezultatele
print("\nRezultatele modelului Random Forest:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Salvam rezultatele evaluarii
results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})

results.to_csv("models/evaluation_results.csv", index=False)

print("\nRezultatele au fost salvate in models/evaluation_results.csv")
