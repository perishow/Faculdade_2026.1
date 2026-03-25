from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

file_path = "./previsoes/previsoes_SARIMA_1.csv"
dataset = pd.read_csv(file_path)

mse = mean_squared_error(dataset["Valor_real"], dataset["Previsao_SARIMAX"])

print(f"MSE: {mse}")


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


residuo = dataset["Valor_real"] - dataset["Previsao_SARIMAX"]

plt.plot(residuo)
plt.show()
