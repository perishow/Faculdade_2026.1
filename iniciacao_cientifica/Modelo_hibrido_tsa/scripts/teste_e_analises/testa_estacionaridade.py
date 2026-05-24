#
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import adfuller

file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"
dataset = pd.read_csv(file_path, encoding="latin1", sep=";", skiprows=8, decimal=",")

dataset = dataset.fillna({"RADIACAO GLOBAL (Kj/m²)": 0.0})

sm.graphics.tsa.plot_acf(dataset["RADIACAO GLOBAL (Kj/m²)"], lags=50)
plt.savefig("grafico_acf.png", bbox_inches="tight")
print("Gráfico salvo como grafico_acf.png")

dick_fuller = adfuller(dataset["RADIACAO GLOBAL (Kj/m²)"])
if dick_fuller[1] < 0.05:
    print("A serie é estacionaria!")
    print(f"{dick_fuller[1]} < 0.05")
else:
    print("A serie não é estacinoaria :(")


"""
    Ao final do experimento, determinou-se que a série é sim estacionária:

    p-value = 6.327256571812126e-17
    como o p-value é menor que 0.05, se descarta a hipotese H0 (nula), 
    validando a hipotese H1 (alternativa: serie é sim estacionaria)
"""
