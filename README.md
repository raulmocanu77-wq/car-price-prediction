# Car Price Prediction

## Project Description

This project uses Machine Learning to predict the prices of used cars based on different vehicle characteristics.

The main goal is to analyze car data, clean and preprocess the dataset, train regression models, and evaluate their performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## Project Structure

```text
car-price-prediction/
│
├── data/
│   ├── IMLP6_TASK_04-cars.csv
│   ├── cars_cleaned.csv
│   └── cars_featured.csv
│
├── models/
│   ├── preprocessor.joblib
│   ├── random_forest_model.joblib
│   ├── evaluation_results.csv
│   └── model_comparison.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_preprocessing.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
└── README.md


## Pasul 17 — Explicăm cum se rulează proiectul

Sub structura proiectului, adaugă:

```markdown
## How to Run the Project

1. Clone the repository.
2. Install the required libraries.
3. Run the scripts in the following order:

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py


**Important:** ordinea contează, deoarece fiecare etapă folosește fișierele generate de etapa anterioară.

Salvează cu `Ctrl + S` și spune-mi când ai terminat. După aceea adăugăm rezultatele modelului și verificăm dacă toate informațiile din README corespund proiectului tău.

## Model Performance

The project uses a Random Forest Regressor to predict used car prices.

The model was evaluated using the following metrics:

| Metric | Result |
|---|---:|
| MAE | 1173.44 |
| MSE | 8354654.70 |
| RMSE | 2890.44 |
| R² Score | 0.8755 |

The model achieved an R² score of approximately 0.876, indicating that it explains around 87.6% of the variance in car prices on the test dataset.

