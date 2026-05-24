from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# 1 - Import dos dados:
file_path: str = "../previsoes/previsoes_SARIMA_3.csv"
serie_residuo: pd.Series = pd.read_csv(file_path)["Residuo"]  # type:ignore

# 2 - Divisão + Normalização:
marca_treino = 0.6
marca_vali = 0.8

# 2.1 - Divisão
tamanho_treino = int(len(serie_residuo) * marca_treino)
tamanho_vali = int(len(serie_residuo) * marca_vali)

conjunto_treino = serie_residuo.iloc[0:tamanho_treino]
conjunto_vali = serie_residuo.iloc[tamanho_treino:tamanho_vali]
conjunto_teste = serie_residuo.iloc[tamanho_vali:]

# 2.2 - Normalização:
scaler = MinMaxScaler()

df_treino = conjunto_treino.to_frame()
df_vali = conjunto_vali.to_frame()
df_teste = conjunto_teste.to_frame()

treino_normalizado = scaler.fit_transform(df_treino)
vali_normalizada = scaler.transform(df_vali)
teste_normalizada = scaler.transform(df_teste)

# 3 - Inclusão dos lags e da Volatilidade no dataframe
n_lags = 24
step_ahead = 1


def criar_lags_com_shift(dados_normalizados, n_lags):
    # 1. Transforma o array numpy do scaler de volta para DataFrame
    df = pd.DataFrame(dados_normalizados, columns=["Alvo_T0"])

    # 2. Cria as colunas com os lags usando um loop rápido e o .shift()
    for i in range(1, n_lags + 1):
        df[f"Lag_{i}"] = df["Alvo_T0"].shift(i)

    # 3. NOVIDADE: Adiciona a feature de volatilidade (Desvio Padrão da Janela)
    # O .shift(1) garante que a janela olhe apenas para o passado, ignorando o "Alvo_T0"
    df["Volatilidade_Lags"] = df["Alvo_T0"].shift(1).rolling(window=n_lags).std()

    # 4. Remove as primeiras linhas que ficaram com NaN devido ao shift e ao rolling
    df.dropna(inplace=True)

    # 5. Separa quem é X (as entradas/lags + volatilidade) e quem é y (o alvo)
    y = df["Alvo_T0"].values
    X = df.drop(columns=["Alvo_T0"]).values

    return X, y


X_treino, y_treino = criar_lags_com_shift(treino_normalizado, n_lags)
X_vali, y_vali = criar_lags_com_shift(vali_normalizada, n_lags)
X_teste, y_teste = criar_lags_com_shift(teste_normalizada, n_lags)


# 4 - Treinamento + validação


# Instanciando o modelo baseline
def treina_rn(n_neuronios: int, X_treino, y_treino) -> MLPRegressor:
    modelo_rn = MLPRegressor(
        hidden_layer_sizes=(n_neuronios,),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=804,
    )
    print(f"Treinando a rede neural: ({n_neuronios},)")
    modelo_rn.fit(X_treino, y_treino)
    print(f"Treinamento concluído em {modelo_rn.n_iter_} épocas.")
    print("---" * 10)
    return modelo_rn


def treina_rf(profundidade: int, X_treino, y_treino) -> RandomForestRegressor:
    modelo_rf = RandomForestRegressor(
        n_estimators=300,  # Cria 100 árvores de decisão
        max_depth=profundidade,  # Profundidade máxima de cada árvore
        random_state=804,  # Reprodutibilidade
        n_jobs=-1,  # Usa todos os núcleos do seu processador para treinar rápido
    )
    print(f"Treinando Random Forest com profundidade máxima: {profundidade}")
    modelo_rf.fit(X_treino, y_treino)
    return modelo_rf


# ==========================================
# 1. BUSCA PELO MELHOR RANDOM FOREST
# ==========================================
print("--- Iniciando a Batalha de Modelos ---")
print("1/2: Treinando e buscando os melhores parâmetros para o Random Forest...")

grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [1.0, "sqrt", "log2"],
}

rf_base = RandomForestRegressor(random_state=804, n_jobs=-1)
busca_rf = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=grid_rf,
    n_iter=30,
    scoring="neg_mean_squared_error",
    cv=3,
    random_state=804,
    n_jobs=-1,
)

busca_rf.fit(X_treino, y_treino)
melhor_rf = busca_rf.best_estimator_

# Avalia o RF na Validação
pred_vali_rf = melhor_rf.predict(X_vali)
mse_rf = mean_squared_error(y_vali, pred_vali_rf)
print(f"Concluído! RMSE do melhor Random Forest: {mse_rf**0.5:.4f}\n")


# ==========================================
# 2. BUSCA PELO MELHOR GRADIENT BOOSTING
# ==========================================
print("2/2: Treinando e buscando os melhores parâmetros para o Gradient Boosting...")

grid_gb = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", "log2"],
}

gb_base = GradientBoostingRegressor(random_state=804)
busca_gb = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=grid_gb,
    n_iter=30,
    scoring="neg_mean_squared_error",
    cv=3,
    random_state=804,
    n_jobs=-1,
)

busca_gb.fit(X_treino, y_treino)
melhor_gb = busca_gb.best_estimator_

# Avalia o GB na Validação
pred_vali_gb = melhor_gb.predict(X_vali)
mse_gb = mean_squared_error(y_vali, pred_vali_gb)
print(f"Concluído! RMSE do melhor Gradient Boosting: {mse_gb**0.5:.4f}\n")


# ==========================================
# 3. COMPARAÇÃO E SELEÇÃO DO GRANDE CAMPEÃO
# ==========================================
print("==========================================")
print("           RESULTADO DA BATALHA           ")
print("==========================================")

if mse_rf < mse_gb:
    best_model_geral = melhor_rf
    nome_vencedor = "Random Forest"
    melhor_mse = mse_rf
    melhor_mae = mean_absolute_error(y_vali, pred_vali_rf)
    hiperparametros = busca_rf.best_params_
else:
    best_model_geral = melhor_gb
    nome_vencedor = "Gradient Boosting"
    melhor_mse = mse_gb
    melhor_mae = mean_absolute_error(y_vali, pred_vali_gb)
    hiperparametros = busca_gb.best_params_

print(f"O Modelo Vencedor é: ** {nome_vencedor.upper()} **")
print(f"MAE (Erro Médio Absoluto): {melhor_mae:.4f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {melhor_mse**0.5:.4f}")
print("\nHiperparâmetros Vencedores:")
for param, valor in hiperparametros.items():
    print(f" - {param}: {valor}")
print("==========================================\n")


# ==========================================
# 4. AVALIAÇÃO VISUAL NO CONJUNTO DE TESTE
# ==========================================
def plotar_previsoes(
    y_real, y_previsto, nome_do_modelo, titulo="Valores Reais vs Previsões"
):
    """
    Plota as séries real e prevista sobrepostas para comparação visual.
    """
    plt.figure(figsize=(14, 6))

    plt.plot(y_real, label="Valores Reais", color="blue", alpha=0.7, linewidth=1.5)

    # A legenda agora usa a variável 'nome_do_modelo'
    plt.plot(
        y_previsto,
        label=f"Previsões ({nome_do_modelo})",
        color="red",
        alpha=0.7,
        linewidth=1.5,
    )

    plt.title(titulo, fontsize=14)
    plt.xlabel("Tempo (Amostras)", fontsize=12)
    plt.ylabel("Valor do Resíduo", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# Gerando previsões para o teste usando APENAS o modelo que ganhou a batalha
prev_teste_norm = best_model_geral.predict(X_teste)

# Plotando as 200 primeiras amostras
plotar_previsoes(
    y_teste[0:200],
    prev_teste_norm[0:200],
    nome_do_modelo=nome_vencedor,
    titulo=f"Desempenho no Teste: {nome_vencedor}",
)
