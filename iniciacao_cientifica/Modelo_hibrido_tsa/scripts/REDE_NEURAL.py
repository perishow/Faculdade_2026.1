import pandas as pd
import numpy as np
from tensorflow.keras import layers, Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler  # <-- Necessário para redes neuronais

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 0. Importação dos dados
file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"
file_path_residuo = "./previsoes/previsoes_SARIMA_3.csv"

dataset = pd.read_csv(file_path, encoding="latin1", sep=";", skiprows=8, decimal=",")
dataset_residuo = pd.read_csv(file_path_residuo)

# Alinhamento dos resíduos com o dataset
tamanho_inicio = int(len(dataset) * 0.6)
tamanho_fim = tamanho_inicio + len(dataset_residuo)
dataset = dataset.iloc[tamanho_inicio:tamanho_fim]

dataset = dataset.reset_index(drop=True)
dataset_residuo = dataset_residuo.reset_index(drop=True)

# Definição das features relevantes
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "VENTO, VELOCIDADE HORARIA (m/s)",
]

# ==========================================
# 1. TRATAMENTO DE NaNs (A MÁGICA ACONTECE AQUI)
# ==========================================
# Interpolação linear para Temperatura e Vento (suaviza os "buracos")
dataset["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"] = dataset[
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)"
].interpolate(method="linear")
dataset["VENTO, VELOCIDADE HORARIA (m/s)"] = dataset[
    "VENTO, VELOCIDADE HORARIA (m/s)"
].interpolate(method="linear")

# Para a chuva, se não registou, assumimos 0
dataset["PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"] = dataset[
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)"
].fillna(0)

# Cinto de segurança: preenche qualquer NaN restante (ex: na primeira linha) com o valor seguinte ou 0
dataset[features] = dataset[features].ffill().bfill().fillna(0)

# ==========================================
# 2. EXTRAÇÃO E NORMALIZAÇÃO DOS DADOS
# ==========================================
# O Keras precisa de arrays NumPy (.values) e não de DataFrames
X = dataset[features].values
y = dataset_residuo["Residuo"].values  # Selecionamos apenas a coluna do erro

marca_treino = int(len(X) * 0.8)

x_treino_bruto = X[:marca_treino]
y_treino_bruto = y[:marca_treino]

x_teste_bruto = X[marca_treino:]
y_teste_bruto = y[marca_treino:]

# Inicialização dos Scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# O fit_transform aprende os padrões e normaliza os dados de treino
x_treino = scaler_x.fit_transform(x_treino_bruto)
x_teste = scaler_x.transform(x_teste_bruto)

# Reshape exigido pelo StandardScaler para o 'y', seguido de flatten para voltar ao normal para o Keras
y_treino = scaler_y.fit_transform(y_treino_bruto.reshape(-1, 1)).flatten()
y_teste = scaler_y.transform(y_teste_bruto.reshape(-1, 1)).flatten()


# ==========================================
# 3. CRIAÇÃO DO MODELO E GRID SEARCH
# ==========================================
def criar_modelo(hidden_neurons=4):
    model = Sequential(
        [
            layers.Input(shape=(3,)),
            layers.Dense(hidden_neurons, activation="relu"),
            layers.Dense(1),  # Saída para regressão
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# Criar o wrapper do Keras para o Scikit-Learn
model_wrapper = KerasRegressor(model=criar_modelo, epochs=50, verbose=0)

# Definir o dicionário de busca (Grid)
param_grid = {"model__hidden_neurons": [32, 64, 128]}

# Configurar o GridSearchCV
grid = GridSearchCV(
    estimator=model_wrapper,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=2,
)

# Executar a busca
print("Iniciando Grid Search...")
grid_result = grid.fit(x_treino, y_treino)

# Exibir os melhores resultados
print(f"\nMelhor configuração: {grid_result.best_params_}")
print(f"Melhor MSE (em escala normalizada): {abs(grid_result.best_score_):.4f}")

# ==========================================
# 4. AVALIAÇÃO E VISUALIZAÇÃO NO CONJUNTO DE TESTE
# ==========================================
print("\n--- Avaliando o melhor modelo no conjunto de TESTE ---")

# O GridSearchCV guarda automaticamente o modelo com a melhor configuração
melhor_modelo = grid_result.best_estimator_

# Fazendo as previsões com os dados de teste (normalizados)
previsoes_normalizadas = melhor_modelo.predict(x_teste)

# Desfazendo a normalização para ver os valores na escala real (Kj/m²)
# O reshape é necessário porque o scaler exige formato 2D, e depois o flatten volta para 1D
previsoes_reais = scaler_y.inverse_transform(
    previsoes_normalizadas.reshape(-1, 1)
).flatten()
y_teste_real = scaler_y.inverse_transform(y_teste.reshape(-1, 1)).flatten()

# Calculando os erros na escala real do negócio
mse_teste = mean_squared_error(y_teste_real, previsoes_reais)
mae_teste = mean_absolute_error(y_teste_real, previsoes_reais)

print(f"MSE no Teste: {mse_teste:.2f}")
print(f"MAE no Teste: {mae_teste:.2f} Kj/m²")

# ==========================================
# 5. PLOTANDO OS RESULTADOS
# ==========================================
plt.figure(figsize=(14, 6))

# Dica: Plotar o teste inteiro pode gerar um borrão ininteligível.
# Vamos plotar, por exemplo, as primeiras 200 horas de teste para enxergar o comportamento.
horas_para_plotar = 200
eixo_x = range(horas_para_plotar)

plt.plot(
    eixo_x,
    y_teste_real[:horas_para_plotar],
    label="Resíduo Real do SARIMA",
    color="blue",
    alpha=0.6,
    marker="o",
    markersize=4,
)
plt.plot(
    eixo_x,
    previsoes_reais[:horas_para_plotar],
    label="Previsão da Rede Neural",
    color="red",
    linestyle="--",
    linewidth=2,
)

plt.title("Comparação: Resíduo Real vs Previsão da Rede Neural (Conjunto de Teste)")
plt.xlabel("Tempo (Horas)")
plt.ylabel("Erro da Radiação (Kj/m²)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()

# Salva a imagem (opcional) e mostra na tela
plt.savefig("grafico_rede_neural_teste.png")
plt.show()
