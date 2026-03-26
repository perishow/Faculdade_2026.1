import pandas as pd
import numpy as np
from tensorflow.keras import layers, Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler  # <-- Importação do StandardScaler

# 0. Importação dos dados
file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"
file_path_residuo = "./previsoes/previsoes_SARIMA_2.csv"

dataset = pd.read_csv(file_path, encoding="latin1", sep=";", skiprows=8, decimal=",")
dataset_residuo = pd.read_csv(file_path_residuo)

# Alinhamento dos resíduos com o dataset
tamanho_inicio = int(len(dataset) * 0.6)
tamanho_fim = tamanho_inicio + len(dataset_residuo)
dataset = dataset.iloc[tamanho_inicio:tamanho_fim]

dataset = dataset.reset_index(drop=True)
dataset_residuo = dataset_residuo.reset_index(drop=True)

# Seleção das features relevantes e tratamento de nulos
features = [
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)",
    "TEMPERATURA DO AR - BULBO SECO, HORÁRIA (°C)",
    "VENTO, VELOCIDADE HORARIA (m/s)",
]

# Preenchemos eventuais valores nulos (NaN) para não quebrar a rede neural
dataset = dataset[features].ffill().bfill()

# Extraímos apenas os valores numéricos (arrays) e a coluna alvo específica
X = dataset.values
y = dataset_residuo["Residuo"].values

# Divisão Treino/Teste ANTES da normalização
marca_treino = int(len(X) * 0.8)

x_treino_bruto = X[:marca_treino]
y_treino_bruto = y[:marca_treino]

x_teste_bruto = X[marca_treino:]
y_teste_bruto = y[marca_treino:]

# --- NORMALIZAÇÃO COM STANDARDSCALER ---
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# O fit_transform aprende os padrões do treino e já aplica a normalização
x_treino = scaler_x.fit_transform(x_treino_bruto)
# O transform APENAS aplica a normalização no teste (usando as métricas do treino)
x_teste = scaler_x.transform(x_teste_bruto)

# O StandardScaler exige que o array tenha formato 2D, por isso o reshape(-1, 1).
# Depois usamos flatten() para voltar ao formato 1D exigido pelo Keras
y_treino = scaler_y.fit_transform(y_treino_bruto.reshape(-1, 1)).flatten()
y_teste = scaler_y.transform(y_teste_bruto.reshape(-1, 1)).flatten()
# ---------------------------------------


# 1. Função que cria o modelo (necessária para o wrapper)
def criar_modelo(hidden_neurons=4):
    model = Sequential(
        [
            # input_shape=(3,) pois você tem 3 neurônios de entrada
            layers.Dense(hidden_neurons, activation="relu", input_shape=(3,)),
            layers.Dense(1),  # Saída para regressão
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# 2. Criar o wrapper do Keras para o Scikit-Learn
# Passamos fixo epochs=20 como você pediu
model_wrapper = KerasRegressor(model=criar_modelo, epochs=20, verbose=0)

# 3. Definir o dicionário de busca (Grid)
# Vamos testar, por exemplo: 2, 4, 8, 16 e 32 neurônios
param_grid = {"model__hidden_neurons": [2, 4, 8, 16, 32]}

# 4. Configurar o GridSearchCV
# cv=3 faz uma validação cruzada de 3 dobras
grid = GridSearchCV(
    estimator=model_wrapper,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
)

# 5. Executar a busca (Agora usando os dados normalizados)
print("Iniciando Grid Search...")
grid_result = grid.fit(x_treino, y_treino)

# 6. Exibir os melhores resultados
print(f"Melhor configuração: {grid_result.best_params_}")
print(f"Melhor MSE (em escala normalizada): {abs(grid_result.best_score_):.4f}")
