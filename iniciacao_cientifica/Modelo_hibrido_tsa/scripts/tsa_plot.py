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


def plotar_comparacao_series(
    y_real: pd.Series,
    y_previsto: pd.Series,
    titulo="Comparação: Valor Real vs Previsões",
    save_path=None,
    show=True,
):
    """
    Recebe duas pd.Series (valor real e previsões) e plota um gráfico de linha comparativo.

    Parâmetros:
    - y_real (pd.Series): A série temporal com os valores reais.
    - y_previsto (pd.Series): A série temporal com os valores preditos pelo modelo.
    - titulo (str): O título que aparecerá no topo do gráfico.
    """

    # Define o tamanho da figura (largura, altura)
    plt.figure(figsize=(12, 6))

    # Plota os valores reais (linha contínua azul)
    plt.plot(
        y_real.index,  # type: ignore
        y_real.values,  # type: ignore
        label="Valor Real",
        color="#1f77b4",
        linewidth=2,
    )

    # Plota as previsões (linha tracejada laranja/vermelha)
    plt.plot(
        y_previsto.index,
        y_previsto.values,  # type: ignore
        label="Previsão",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )

    # Adiciona rótulos e título
    plt.title(titulo, fontsize=16, pad=15)
    plt.xlabel("Tempo", fontsize=12)
    plt.ylabel("Valores", fontsize=12)

    # Configura a legenda e a grade (grid)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)

    # Rotaciona as datas no eixo X para evitar sobreposição (se o index for datetime)
    plt.gcf().autofmt_xdate()

    # Ajusta o layout para não cortar nenhuma informação
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico salvo com sucesso em: {save_path}")
    if show:
        # Exibe o gráfico
        plt.show()


def plotar_comparacao_series_sequencial(inicio, fim, fatia, save_path_base):
    contador = 0
    for i in range(inicio, fim, fatia):
        contador += 1
        inicio = i
        fim = inicio + fatia
        print(f"{inicio} --> {fim}")

        valores_reais = dataset["Valor_real"].iloc[inicio:fim]
        previsoes = dataset["Previsao_SARIMAX"].iloc[inicio:fim]

        save_path = f"{save_path_base}/plotagem_comparativa_SARIMA_{contador}"
        try:
            plotar_comparacao_series(
                y_real=valores_reais,
                y_previsto=previsoes,
                save_path=save_path,
                show=False,
            )
        except Exception as e:
            print(f"{contador} - Erro de plotagem")
            print(e)


file_path = "./previsoes/previsoes_SARIMA_3.csv"
save_path = "./plotagens/plotagem_comparativa_SARIMA"
dataset = pd.read_csv(file_path)

inicio = 0
fim = 3000
fatia = 300

print(len(dataset))

plotar_comparacao_series_sequencial(
    inicio=inicio, fim=fim, fatia=fatia, save_path_base=save_path
)
