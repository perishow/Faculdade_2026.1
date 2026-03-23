import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.metrics import mean_squared_error

# Ignora os avisos chatos do statsmodels
warnings.filterwarnings("ignore")

# 1. Carrega os dados
file_path = "../datasets_tratados/sales.csv"
data = pd.read_csv(file_path)

# Garante que estamos pegando apenas os 100 primeiros itens
serie_completa = data["Units Sold"].values[:100]

tamanho_janela = 50

print("Iniciando o treinamento online do ARIMA com Grid Search...")
inicio = time.perf_counter()

# Configurações iniciais do Grid Search
best_order = None
best_mse = float("inf")  # Começa com infinito
best_previsoes = None

# Os valores reais são os mesmos para todos os testes, extraímos apenas uma vez
valores_reais = serie_completa[tamanho_janela:]

# Usamos 'p' e 'q' para os parâmetros do ARIMA para evitar conflito
for p in range(4):
    for d in range(2):
        for q in range(4):
            # Limpamos as previsões a cada nova combinação do Grid Search
            previsoes_atuais = []

            print(f"Testando ordem ARIMA({p}, {d}, {q})...")

            try:
                # 2. O Loop da Janela Deslizante (usamos 't' para o tempo)
                for t in range(tamanho_janela, len(serie_completa)):
                    janela_treino = serie_completa[t - tamanho_janela : t]

                    # Aplica o p e q dinâmicos do Grid Search
                    modelo = sm.tsa.ARIMA(
                        janela_treino,
                        order=(p, d, q),
                    )
                    modelo_ajustado = modelo.fit()

                    # Previsão de 1 passo
                    previsao_1_passo = modelo_ajustado.forecast(steps=1)[0]
                    previsoes_atuais.append(previsao_1_passo)

                # Calcula o MSE APENAS quando a janela deslizante da ordem atual termina
                mse_atual = mean_squared_error(valores_reais, previsoes_atuais)
                print(f"   -> Concluído! MSE = {mse_atual:.4f}")

                # Verifica se é o melhor modelo até agora
                if mse_atual < best_mse:
                    best_mse = mse_atual
                    best_order = (p, d, q)
                    best_previsoes = previsoes_atuais

            except Exception as e:
                # O statsmodels pode falhar ao ajustar certas ordens se os dados não ajudarem
                print(
                    f"   -> Erro ao ajustar ARIMA({p}, 1, {q}): ignorando esta combinação."
                )

fim = time.perf_counter()

# --- RESULTADOS FINAIS ---
print("\n" + "=" * 45)
print(f"Tempo total do Grid Search: {fim - inicio:.2f} segundos.")
print(f"🏆 MELHOR ORDEM ENCONTRADA: ARIMA{best_order}")
print(f"📉 MELHOR MSE: {best_mse:.4f}")
print("=" * 45 + "\n")

# 3. Exportando o melhor modelo para CSV
eixo_x = range(tamanho_janela, len(serie_completa))

df_resultados = pd.DataFrame(
    {
        "Indice_Tempo": eixo_x,
        "Valor_real": valores_reais,
        "Previsao_ARIMA": best_previsoes,
    }
)
df_resultados.to_csv("resultados_previsoes.csv", index=False)
print("Resultados do melhor modelo salvos em 'resultados_previsoes.csv' com sucesso!")


# 4. Visualizando o resultado do MELHOR modelo Online
plt.figure(figsize=(10, 5))

plt.plot(
    eixo_x, valores_reais, label="Valores Reais", color="blue", marker="o", markersize=4
)
# Plota apenas as melhores previsões, indicando a ordem na legenda
plt.plot(
    eixo_x,
    best_previsoes,
    label=f"Melhor Previsão ARIMA {best_order}",
    color="red",
    linestyle="--",
)

plt.title(
    f"Validação Walk-Forward: Melhor Modelo ARIMA {best_order} (MSE: {best_mse:.2f})"
)
plt.xlabel("Índice do Tempo")
plt.ylabel("Unidades Vendidas")
plt.legend()
plt.show()
