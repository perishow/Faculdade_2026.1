import matplotlib.pyplot as plt
import pandas as pd


def plotar_radiacao(df):
    """
    Função para plotar a série temporal de Radiação Global.
    O dataset precisa ter um índice temporal (DatetimeIndex).
    """
    # Define o tamanho do gráfico (largura, altura)
    plt.figure(figsize=(14, 6))

    # Plota os dados. O Matplotlib é inteligente e já usa o índice como eixo X!
    plt.plot(
        df.index,
        df["RADIACAO GLOBAL (Kj/m²)"],
        color="darkorange",
        linewidth=1.5,
        label="Radiação Global",
    )

    # Adiciona título e nomes aos eixos
    plt.title("Variação da Radiação Solar Global", fontsize=15, fontweight="bold")
    plt.xlabel("Data e Hora", fontsize=12)
    plt.ylabel("Radiação (Kj/m²)", fontsize=12)

    # Adiciona uma grade de fundo para facilitar a leitura
    plt.grid(True, linestyle="--", alpha=0.6)

    # Adiciona a legenda
    plt.legend()

    # Rotaciona os textos das datas no eixo X para não se sobreporem
    plt.xticks(rotation=45)

    # Ajusta o layout para garantir que nada fique cortado na imagem final
    plt.tight_layout()

    # Mostra o gráfico na tela
    plt.show()


def plotar_comparacao(df_real, df_previsto):
    # Prepara a figura
    _, ax = plt.subplots(figsize=(14, 6))

    # 1. Plota o PRIMEIRO dataset (Dados Reais)
    ax.plot(
        df_real.index,
        df_real["RADIACAO GLOBAL (Kj/m²)"],
        color="darkorange",
        label="Dados Reais",
        linewidth=2,
    )

    # 2. Plota o SEGUNDO dataset (Previsões) na mesma tela
    # Dica: Usar uma linha tracejada ('--') e um pouco de transparência (alpha)
    # ajuda muito a ver os dados reais que estão por trás da previsão.
    ax.plot(
        df_previsto.index,
        df_previsto["Previsao"],
        color="blue",
        label="Previsão SARIMA",
        linewidth=2,
        linestyle="--",
        alpha=0.8,
    )

    # Textos e Formatação
    ax.set_title(
        "Comparação: Radiação Real vs Previsão SARIMA", fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Radiação (Kj/m²)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Chama a legenda (isso usa os 'labels' definidos lá no .plot)
    # É fundamental quando sobrepomos dados para sabermos quem é quem!
    ax.legend(loc="upper right")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


file_path = "./resultados_previsoes.csv"
file2_path = "./resultados_previsoes_1.csv"
dataset = pd.read_csv(file_path)
dataset2 = pd.read_csv(file2_path)

plotar_comparacao(dataset, dataset2)
