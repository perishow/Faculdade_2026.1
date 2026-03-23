import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import warnings

# Ignora os avisos chatos do statsmodels que poluem o terminal durante o loop
warnings.filterwarnings("ignore")

# 1. Carrega os dados
data = pd.read_csv("sales.csv")

# Garante que estamos pegando apenas os 100 primeiros itens e convertendo para uma lista/array
serie_completa = data["Units Sold"].values[:100]

tamanho_janela = 50
previsoes = []
valores_reais = []

print("Iniciando o treinamento online do ARIMA...")
inicio = time.perf_counter()

# 2. O Loop da Janela Deslizante
# Começa no índice 50 e vai até o 99 (totalizando os 100 itens)
for i in range(tamanho_janela, len(serie_completa)):
    # Fatiamos a série: pega do índice atual menos 50, até o índice atual
    # Ex: no primeiro passo, pega do [0:50]. No segundo, do [1:51], etc.
    janela_treino = serie_completa[i - tamanho_janela : i]

    # Treina o modelo apenas com os 50 itens da janela atual
    # Dica: coloquei order=(1,1,1) para o modelo tentar capturar tendências
    modelo = sm.tsa.ARIMA(janela_treino, order=(3, 1, 3))
    modelo_ajustado = modelo.fit()

    # Faz a previsão de 1 passo à frente e pega o valor numérico exato ([0])
    previsao_1_passo = modelo_ajustado.forecast(steps=1)[0]

    # Guarda a previsão e o valor que realmente aconteceu para compararmos depois
    previsoes.append(previsao_1_passo)
    valores_reais.append(serie_completa[i])

    print(
        f"Passo {i}: Previsão = {previsao_1_passo:.2f} | Realidade = {serie_completa[i]:.2f}"
    )

fim = time.perf_counter()

# 3. Visualizando o resultado do seu modelo Online
plt.figure(figsize=(10, 5))

# Plotamos a partir do item 50 para alinhar o gráfico
eixo_x = range(tamanho_janela, len(serie_completa))

plt.plot(
    eixo_x, valores_reais, label="Valores Reais", color="blue", marker="o", markersize=4
)
plt.plot(
    eixo_x, previsoes, label="Previsões ARIMA (1 passo)", color="red", linestyle="--"
)

print(f"\nO loop rodou {len(previsoes)} vezes e levou {fim - inicio:.4f} segundos.")

df_resultados = pd.DataFrame(
    {
        "Indice_Tempo": range(tamanho_janela, len(serie_completa)),
        "Valor_real": valores_reais,
        "Previsao_ARIMA": previsoes,
    }
)

df_resultados.to_csv("resultados_previsoes.csv", index=False)
print("Resultados salvos em CSV com sucesso!")


# Visulização
plt.title("Validação Walk-Forward: Modelo ARIMA Online")
plt.xlabel("Índice do Tempo")
plt.ylabel("Unidades Vendidas")
plt.legend()
plt.show()
