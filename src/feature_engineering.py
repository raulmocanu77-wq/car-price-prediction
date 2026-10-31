import pandas as pd
import numpy as np

# Incarcam datasetul curatat
df = pd.read_csv("data/cars_cleaned.csv")

# Calculam varsta masinii
df["car_age"] = 2026 - df["year"]

# Calculam kilometrajul mediu pe an
df["mileage_per_year"] = df["mileage(kilometers)"] / df["car_age"]

print("Masini cu varsta zero sau negativa:", (df["car_age"] <= 0).sum())

# Transformam volumul motorului din cm3 in litri
df["engine_volume_liters"] = df["volume(cm3)"] / 1000

# Identificam masinile fabricate incepand cu 2015
df["is_newer_car"] = (df["year"] >= 2015).astype(int)

# Identificam masinile cu kilometraj ridicat
df["is_high_mileage"] = (
    df["mileage(kilometers)"] >= 200000
).astype(int)

# Combinam marca si modelul masinii
df["brand_model"] = (
    df["make"].astype(str) + "_" + df["model"].astype(str)
)

# Salvam datasetul cu noile caracteristici
df.to_csv("data/cars_featured.csv", index=False)

print("Datasetul cu noile caracteristici a fost salvat cu succes!")