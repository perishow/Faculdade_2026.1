import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"
dataset = pd.read_csv(file_path, encoding="latin1", sep=";", skiprows=8, decimal=",")
dataset = dataset.fillna({"RADIACAO GLOBAL (Kj/m²)": 0.0})

treino = dataset["RADIACAO GLOBAL (Kj/m²)"].iloc[0:365]
teste = dataset["RADIACAO GLOBAL (Kj/m²)"].iloc[365:437]

model = SARIMAX(treino, order=(2, 0, 0), seasonal_order=(1, 0, 1, 24))
model = model.fit()

predictions = model.forecast(steps=72)

mse = mean_squared_error(teste, predictions)

print(f"{mse:.2f}")


def plotar_comparacao(teste, predictions):
    # Define o tamanho do gráfico para ficar bem visível
    plt.figure(figsize=(12, 6))

    # Plota a linha dos valores reais (Teste)
    plt.plot(
        teste.index, teste, label="Valores Reais (Teste)", color="blue", marker="o"
    )

    # Plota a linha das previsões sobrepostas
    plt.plot(
        predictions.index,
        predictions,
        label="Previsões (SARIMAX)",
        color="orange",
        linestyle="--",
        marker="x",
    )

    # Adiciona títulos e rótulos aos eixos
    plt.title("Comparação: Valores Reais vs Previsões de Radiação Global", fontsize=14)
    plt.xlabel("Índice (Horas/Períodos)", fontsize=12)
    plt.ylabel("Radiação Global (Kj/m²)", fontsize=12)

    # Ativa a legenda e adiciona uma grade de fundo para facilitar a leitura
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.7)

    # Garante que o gráfico seja ajustado perfeitamente na janela
    plt.tight_layout()

    # Exibe o gráfico na tela
    plt.show()


# Para gerar o gráfico, basta chamar a função passando as suas variáveis:
plotar_comparacao(teste, predictions)
