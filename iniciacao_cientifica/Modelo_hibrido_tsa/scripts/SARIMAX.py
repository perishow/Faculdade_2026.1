import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
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

# ... (código anterior)

print(f"Tamanho da janela de treino: {tamanho_janela} horas.")
print(
    f"Iniciando previsão passo a passo para as próximas {passos_para_prever} horas..."
)
inicio = time.perf_counter()

# Variável para guardar os parâmetros da iteração anterior (Warm Start)
parametros_iniciais = None

# 2. O Loop da Janela Deslizante
for i in range(tamanho_janela, fim_loop):
    janela_treino = serie_completa[i - tamanho_janela : i]

    # --- CORREÇÃO: Desativar enforce_stationarity e enforce_invertibility ---
    modelo = SARIMAX(
        janela_treino,
        order=(1, 0, 0),
        seasonal_order=(2, 0, 0, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    try:
        # --- CORREÇÃO: Usar Warm Start ---
        if parametros_iniciais is not None:
            modelo_ajustado = modelo.fit(start_params=parametros_iniciais, disp=False)
        else:
            modelo_ajustado = modelo.fit(disp=False)

        # Salva os parâmetros para ajudar na previsão da PRÓXIMA janela
        parametros_iniciais = modelo_ajustado.params  # type: ignore

        # Faz a previsão
        previsao_1_passo = modelo_ajustado.forecast(steps=1)[0]  # type: ignore

        # Opcional: Evitar previsões negativas de radiação solar
        previsao_1_passo = max(0.0, previsao_1_passo)

    except np.linalg.LinAlgError:
        # --- CORREÇÃO: Cinto de segurança para erros de matriz ---
        print(
            f"\n[!] Erro de decomposição LU no passo {i}. Usando previsão ingênua (0.0)."
        )
        previsao_1_passo = (
            0.0  # Como deu erro geralmente na madrugada (vide seu log), 0 é seguro.
        )
        parametros_iniciais = None  # Reseta os parâmetros para a próxima rodada

    previsoes.append(previsao_1_passo)
    valores_reais.append(serie_completa[i])

    print(
        f"Passo {i}: Previsto = {previsao_1_passo:.2f} | Real = {serie_completa[i]:.2f}"
    )

fim = time.perf_counter()

# ... (resto do código continua igual)

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

output_path = "./previsoes/previsoes_SARIMA_3.csv"
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
