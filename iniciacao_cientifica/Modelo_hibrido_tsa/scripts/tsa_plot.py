import matplotlib.pyplot as plt
import pandas as pd


def plotar_serie(
    serie: pd.Series,
    titulo="Meu Gráfico",
    cor="#1f77b4",
    tipo="line",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Recebe uma Pandas Series e plota usando Matplotlib.

    Parâmetros:
    - serie: pd.Series
    - titulo: str (Título do gráfico)
    - cor: str (Cor da linha ou das barras)
    - tipo: str ('line' para linha, 'bar' para barras)
    """
    # Criando a figura e os eixos
    plt.figure(figsize=(10, 6))

    # Plotando de acordo com o tipo escolhido
    if tipo == "line":
        plt.plot(serie.index, serie.values, linestyle="-", color=cor)  # type: ignore
    elif tipo == "bar":
        plt.bar(serie.index, serie.values, color=cor)  # type: ignore

    # Customização
    plt.title(titulo, fontsize=14, fontweight="bold")
    plt.xlabel(serie.index.name if serie.index.name else "Índice")  # type:ignore
    plt.ylabel("Valores")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)  # Rotaciona os nomes do eixo X se forem longos

    # Ajusta o layout para não cortar informações
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico salvo com sucesso em: {save_path}")
    if show:
        # Exibe o gráfico
        plt.show()


def plotar_comparacao_series(
    y_real: pd.Series,
    y_previsto: pd.Series,
    titulo="Comparação: Valor Real vs Previsões",
    save_path: str | None = None,
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


def plotar_serie_sequencial(inicio, fim, fatia, save_path_base, formato: str = ".png"):
    contador = 0
    for i in range(inicio, fim, fatia):
        contador += 1
        inicio = i
        fim = inicio + fatia
        print(f"{inicio} --> {fim}")

        residuos = dataset["Residuo"].iloc[inicio:fim]

        save_path = f"{save_path_base}/plotagem_residuo_{contador}{formato}"
        try:
            plotar_serie(residuos, titulo="Residuos", save_path=save_path, show=False)
        except Exception as e:
            print(f"{contador} - erro de plotagem")
            print(e)


def plotar_comparacao_series_sequencial(
    inicio: int, fim: int, fatia: int, save_path_base: str
):
    contador = 0
    for i in range(inicio, fim, fatia):
        contador += 1
        inicio = i
        fim = inicio + fatia
        print(f"{inicio} --> {fim}")

        valores_reais = dataset["valor_real"].iloc[inicio:fim]
        previsoes = dataset["previsao_sarimax"].iloc[inicio:fim]

        save_path = f"{save_path_base}/plotagem_comparativa_sarima_{contador}.pdf"
        try:
            plotar_comparacao_series(
                y_real=valores_reais,
                y_previsto=previsoes,
                save_path=save_path,
                show=False,
            )
        except Exception as e:
            print(f"{contador} - erro de plotagem")
            print(e)


if __name__ == "__main__":
    file_path = "./previsoes/previsoes_SARIMA_3.csv"
    save_path = "./plotagens/plotagem_residuo_SARIMA"
    dataset = pd.read_csv(file_path)

    inicio = 0
    fim = 3000
    fatia = 300

    print(len(dataset))

    residuo = dataset["Residuo"]

    plotar_serie_sequencial(
        inicio=inicio, fim=fim, fatia=fatia, save_path_base=save_path
    )
