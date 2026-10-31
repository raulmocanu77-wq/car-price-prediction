import pandas as pd
import numpy as np

df = pd.read_csv("data/IMLP6_TASK_04-cars.csv")

# Verificam cate randuri duplicate exista
duplicates = df.duplicated().sum()
print("Numar de duplicate:", duplicates)

# Eliminam randurile duplicate
df = df.drop_duplicates()

print("Dimensiunea datelor dupa eliminarea duplicatelor:", df.shape)

# Verificam valorile lipsa
print("Valori lipsa inainte de completare:")
print(df.isnull().sum())

# Completam valorile lipsa din volume cu mediana
df["volume(cm3)"] = df["volume(cm3)"].fillna(
    df["volume(cm3)"].median()
)

# Completam valorile lipsa din drive_unit si segment
df["drive_unit"] = df["drive_unit"].fillna("unknown")
df["segment"] = df["segment"].fillna("unknown")

print("Valori lipsa dupa completare:")
print(df.isnull().sum())

# Tratam valorile suspecte din kilometraj
df.loc[df["mileage(kilometers)"] >= 1000000, "mileage(kilometers)"] = np.nan

# Completam valorile lipsa cu mediana
df["mileage(kilometers)"] = df["mileage(kilometers)"].fillna(
    df["mileage(kilometers)"].median()
)

# Tratam valorile suspecte din volumul motorului
df.loc[df["volume(cm3)"] >= 10000, "volume(cm3)"] = np.nan

# Completam valorile lipsa cu mediana
df["volume(cm3)"] = df["volume(cm3)"].fillna(
    df["volume(cm3)"].median()
)

# Verificam daca mai exista valori lipsa
print("Verificare finala:")
print(df.isnull().sum())

print("Dimensiunea finala a datasetului:", df.shape)

# Salvam datasetul curatat
df.to_csv("data/cars_cleaned.csv", index=False)

print("Datasetul curatat a fost salvat cu succes!")

