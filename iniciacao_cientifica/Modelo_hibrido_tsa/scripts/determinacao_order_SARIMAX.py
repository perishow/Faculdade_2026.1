# Nesse arquivo, se procura determinar os parametros de modelagem (p, d ,q) (P, D, Q , S)
# do modelo SARIMAX
# Usaremos a biblioteca pmdarima para automatizar essa determinação sem usar box-jenkins.

import pmdarima as pm
import pandas as pd

file_path_3 = "../datasets/2025/INMET_NE_PE_A328_SURUBIM_01-01-2025_A_31-12-2025.CSV"
dataset = pd.read_csv(file_path_3, encoding="latin1", sep=";", skiprows=8, decimal=",")


dataset = dataset.fillna({"RADIACAO GLOBAL (Kj/m²)": 0.0})


modelo_auto = pm.auto_arima(
    dataset["RADIACAO GLOBAL (Kj/m²)"],
    seasonal=True,
    m=24,
    setpwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
)

print(modelo_auto.summary())
