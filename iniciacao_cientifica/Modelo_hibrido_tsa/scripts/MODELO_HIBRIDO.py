import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ==========================================
# 1 - Import dos dados
# ==========================================
file_path: str = "./previsoes/previsoes_SARIMA_3.csv"
# Substitua o read_csv de simulação pelo seu read_csv real com os parâmetros adequados se necessário
dataframe = pd.read_csv(file_path)

# ==========================================
# 2 - Divisão e Normalização
# ==========================================
marca_treino = 0.6
marca_vali = 0.8

tamanho_treino = int(len(dataframe) * marca_treino)
tamanho_vali = int(len(dataframe) * marca_vali)

conjunto_treino = dataframe.iloc[0:tamanho_treino]
conjunto_vali = dataframe.iloc[tamanho_treino:tamanho_vali]
conjunto_teste = dataframe.iloc[tamanho_vali:]

# Normalização de todo o dataframe
scaler = MinMaxScaler()
treino_norm = scaler.fit_transform(conjunto_treino)
vali_norm = scaler.transform(conjunto_vali)
teste_norm = scaler.transform(conjunto_teste)

# ==========================================
# 3 - CRIAÇÃO DOS LAGS HÍBRIDOS (24 horas)
# ==========================================
IDX_REAL = 1
IDX_SARIMAX = 2
IDX_RESIDUO = 3

def criar_lags_hibridos(dados_array, n_lags=24):
    X, y, sarimax_base, real_base = [], [], [], []
    for i in range(n_lags, len(dados_array) - 1):
        # Features (X): Últimos 24 erros do SARIMAX
        X.append(dados_array[i-n_lags+1 : i+1, IDX_RESIDUO])
        
        # Alvo (y): O Erro da PRÓXIMA hora
        y.append(dados_array[i + 1, IDX_RESIDUO])
        
        # Referências de t+1 para a reconstrução
        sarimax_base.append(dados_array[i + 1, IDX_SARIMAX])
        real_base.append(dados_array[i + 1, IDX_REAL])
        
    return np.array(X), np.array(y), np.array(sarimax_base), np.array(real_base)

n_lags = 24
print(f"Preparando matrizes temporais com {n_lags} lags...")

# Aqui estamos juntando treino e validação para o GridSearch fazer o Split Interno
X_treino, y_treino, _, _ = criar_lags_hibridos(treino_norm, n_lags)
X_teste, y_teste, sarimax_teste, real_teste = criar_lags_hibridos(teste_norm, n_lags)

# ==========================================
# 4 - CONFIGURAÇÃO DOS 3 MODELOS
# ==========================================
# Dicionário contendo os modelos base e as grades de hiperparâmetros
modelos = {
    'MLP (Rede Neural)': (
        MLPRegressor(random_state=42, early_stopping=True, max_iter=500),
        {'hidden_layer_sizes': [(16,), (16, 8)], 'activation': ['relu', 'tanh'], 'learning_rate_init': [0.001, 0.01]}
    ),
    'Gradient Boosting': (
        GradientBoostingRegressor(random_state=42),
        {'n_estimators': [50, 100], 'max_depth': [3, 4], 'learning_rate': [0.05, 0.1]}
    ),
    'SVR': (
        SVR(),
        {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.05], 'kernel': ['rbf']}
    )
}

tscv = TimeSeriesSplit(n_splits=3)
resultados_previsoes = {}
metricas_modelos = {}

print("\nIniciando treinamento comparativo... (Isso pode levar alguns minutos)")

# ==========================================
# 5 - TREINAMENTO E RECONSTRUÇÃO NA ESCALA ORIGINAL
# ==========================================
print("\nIniciando treinamento comparativo... (Isso pode levar alguns minutos)")

# Função auxiliar para reverter a normalização de uma coluna específica
def reverter_escala(valores_normalizados, indice_coluna, scaler, total_colunas=4):
    dummy = np.zeros((len(valores_normalizados), total_colunas))
    dummy[:, indice_coluna] = valores_normalizados
    inverso = scaler.inverse_transform(dummy)
    return inverso[:, indice_coluna]

# Desnormalizando as bases de teste para a escala real (PASSANDO O SCALER AQUI)
real_teste_orig = reverter_escala(real_teste, IDX_REAL, scaler)
sarimax_teste_orig = reverter_escala(sarimax_teste, IDX_SARIMAX, scaler)

# Avaliação do modelo Base (SARIMAX Puro) na escala real
mse_base = mean_squared_error(real_teste_orig, sarimax_teste_orig)
mae_base = mean_absolute_error(real_teste_orig, sarimax_teste_orig)

resultados_previsoes_orig = {}

for nome, (modelo_base, param_grid) in modelos.items():
    print(f"\n> Treinando e Otimizando: {nome}...")
    
    grid = GridSearchCV(modelo_base, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_treino, y_treino)
    melhor_modelo = grid.best_estimator_
    
    # Previsão do resíduo (ainda normalizado)
    residuo_previsto_norm = melhor_modelo.predict(X_teste)
    
    # 1. Revertendo o resíduo previsto para a escala real (PASSANDO O SCALER AQUI TAMBÉM)
    residuo_previsto_orig = reverter_escala(residuo_previsto_norm, IDX_RESIDUO, scaler)
    
    # 2. Soma Híbrida CORRETA (na escala original)
    previsao_hibrida_orig = sarimax_teste_orig + residuo_previsto_orig
    resultados_previsoes_orig[nome] = previsao_hibrida_orig
    
    # Calcula as métricas na escala real
    mse_hibrido = mean_squared_error(real_teste_orig, previsao_hibrida_orig)
    mae_hibrido = mean_absolute_error(real_teste_orig, previsao_hibrida_orig)
    melhoria = ((mse_base - mse_hibrido) / mse_base) * 100
    
    metricas_modelos[nome] = {
        'MSE': mse_hibrido,
        'MAE': mae_hibrido,
        'Melhoria': melhoria,
    }

# ==========================================
# 6 - RELATÓRIO FINAL NO TERMINAL (Escala Real)
# ==========================================
print("\n" + "="*60)
print(" RELATÓRIO DE DESEMPENHO NO TESTE (ESCALA ORIGINAL)")
print("="*60)
print(f"{'SARIMAX PURO (Base)':<25} | MSE: {mse_base:.2f} | MAE: {mae_base:.2f}")
print("-" * 60)

for nome, mets in metricas_modelos.items():
    sinal = "+" if mets['Melhoria'] > 0 else ""
    print(f"{nome:<25} | MSE: {mets['MSE']:.2f} | MAE: {mets['MAE']:.2f} | Melhoria: {sinal}{mets['Melhoria']:.2f}%")

print("="*60)

# ==========================================
# 7 - VISUALIZAÇÃO GRÁFICA COMPARATIVA
# ==========================================
def plotar_comparacao_total(real, sarimax, prevs_dict, amostras=150):
    plt.figure(figsize=(16, 8))
    
    # Linhas Base
    plt.plot(real[:amostras], label='Valor Real (Target)', color='black', linewidth=2, marker='.')
    plt.plot(sarimax[:amostras], label='SARIMAX Puro', color='gray', linestyle=':', linewidth=2, alpha=0.8)
    
    # Cores e Estilos
    estilos = [
        ('MLP (Rede Neural)', 'blue', '--'),
        ('Gradient Boosting', 'green', '-.'),
        ('SVR', 'red', '--')
    ]
    
    # Plotando os Híbridos (agora sem o deslocamento!)
    for nome, cor, estilo in estilos:
        prev = prevs_dict[nome]
        plt.plot(prev[:amostras], label=f'Híbrido: {nome}', color=cor, linestyle=estilo, alpha=0.7)
    
    plt.title(f'Comparação de Modelos Híbridos na Escala Real (Primeiras {amostras} horas)', fontsize=14)
    plt.xlabel('Tempo (Horas)', fontsize=12)
    plt.ylabel('Valor Original (Desnormalizado)', fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    nome_arquivo = 'grafico_comparativo_modelos.png'
    plt.savefig(nome_arquivo, dpi=300)
    print(f"\nGráfico salvo com sucesso como '{nome_arquivo}'. O deslocamento Y foi corrigido!")

# Agora passamos os dados revertidos para a função plotar
plotar_comparacao_total(real_teste_orig, sarimax_teste_orig, resultados_previsoes_orig)
