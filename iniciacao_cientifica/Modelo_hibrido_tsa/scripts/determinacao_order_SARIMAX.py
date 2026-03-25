# Nesse arquivo, se procura determinar os parametros de modelagem (p, d ,q) (P, D, Q , S)
# do modelo SARIMAX
# Usaremos a biblioteca pmdarima para automatizar essa determinação sem usar box-jenkins.

import pmdarima as pm
import pandas as pd

# importação + tratamento dos dados
file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"
dataset = pd.read_csv(file_path, encoding="latin1", sep=";", skiprows=8, decimal=",")
dataset = dataset.fillna({"RADIACAO GLOBAL (Kj/m²)": 0.0})

marca_treino = int(len(dataset) * 0.1)
print(f"Numero de linhas usadas para o auto_arima: {marca_treino}")

modelo_auto = pm.auto_arima(
    dataset["RADIACAO GLOBAL (Kj/m²)"].head(marca_treino),
    seasonal=True,
    m=24,
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
)

print(modelo_auto.summary())
