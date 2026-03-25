import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import time
import warnings

# Ignora os avisos chatos do statsmodels que poluem o terminal durante o loop
warnings.filterwarnings("ignore")

# 1. Carrega os dados
file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"
dataset = pd.read_csv(file_path, encoding="latin1", sep=";", skiprows=8, decimal=",")
dataset = dataset.fillna({"RADIACAO GLOBAL (Kj/m²)": 0.0})

# --- CORREÇÃO 1: Extrair a série completa em formato de array ---
serie_completa = dataset["RADIACAO GLOBAL (Kj/m²)"].values

tamanho_janela = int(len(serie_completa) * 0.6)
previsoes = []
valores_reais = []

# --- CORREÇÃO 2: Limite de passos para não travar seu PC por dias ---
passos_para_prever = int(len(serie_completa) * 0.4)

fim_loop = tamanho_janela + passos_para_prever

print(f"Tamanho da janela de treino: {tamanho_janela} horas.")
print(
    f"Iniciando previsão passo a passo para as próximas {passos_para_prever} horas..."
)
inicio = time.perf_counter()

# 2. O Loop da Janela Deslizante
for i in range(tamanho_janela, fim_loop):
    # Fatiamos a série (ex: [0:tamanho_janela], depois [1:tamanho_janela+1], etc.)
    janela_treino = serie_completa[i - tamanho_janela : i]

    # --- CORREÇÃO 3: Parâmetros do seu auto_arima ---
    modelo = SARIMAX(janela_treino, order=(1, 0, 0), seasonal_order=(2, 0, 0, 24))

    # disp=False impede que o SARIMAX imprima relatórios gigantescos a cada passo
    modelo_ajustado = modelo.fit(disp=False)  # type: ignore

    # Faz a previsão de 1 passo à frente e pega o valor numérico exato
    previsao_1_passo = modelo_ajustado.forecast(steps=1)[0]  # type: ignore

    previsoes.append(previsao_1_passo)
    valores_reais.append(serie_completa[i])

    print(
        f"Passo {i}: Previsto = {previsao_1_passo:.2f} | Real = {serie_completa[i]:.2f}"
    )

fim = time.perf_counter()

# 3. Visualizando e Salvando os Resultados
print(f"\nO loop rodou {len(previsoes)} vezes e levou {fim - inicio:.4f} segundos.")

# Cálculo dos resíduos (Erro = Real - Previsto)
residuos = [real - pred for real, pred in zip(valores_reais, previsoes)]

df_resultados = pd.DataFrame(
    {
        "Indice_Tempo": range(tamanho_janela, fim_loop),
        "Valor_real": valores_reais,
        "Previsao_SARIMAX": previsoes,
        "Residuo": residuos,  # Nova coluna adicionada aqui
    }
)

output_path = "./previsoes/previsoes_SARIMA_2.csv"
df_resultados.to_csv(output_path, index=False)
print("Resultados salvos em CSV com sucesso (incluindo coluna de resíduos)!")


# Visualização
plt.figure(figsize=(12, 6))
eixo_x = range(tamanho_janela, fim_loop)

plt.plot(
    eixo_x, valores_reais, label="Valores Reais", color="blue", marker="o", markersize=4
)
plt.plot(
    eixo_x, previsoes, label="Previsões SARIMAX (1 passo)", color="red", linestyle="--"
)

# --- CORREÇÃO 4: Nomes dos eixos ---
plt.title("Validação Walk-Forward: Modelo SARIMAX Online - Radiação Solar")
plt.xlabel("Índice do Tempo (Horas)")
plt.ylabel("Radiação Global (Kj/m²)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.show()
