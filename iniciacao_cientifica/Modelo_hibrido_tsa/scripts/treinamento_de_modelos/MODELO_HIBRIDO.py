import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from fpdf import FPDF

def gerar_pdf_relatorio(metricas, mse_base, mae_base, imagem_grafico, nome_arquivo_pdf="../relatorios/Relatorio_Modelos_Hibridos.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Título
    pdf.cell(0, 10, "Relatório de Desempenho: Modelos Híbridos vs SARIMA", ln=True, align='C')
    pdf.ln(10)
    
    # Resumo do Modelo Base
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Desempenho do Modelo Base (Baseline)", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Modelo: SARIMA Puro", ln=True)
    pdf.cell(0, 8, f"MSE: {mse_base:.4f}", ln=True)
    pdf.cell(0, 8, f"MAE: {mae_base:.4f}", ln=True)
    pdf.ln(5)
    
    # Tabela de Comparação
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Comparação de Modelos Híbridos", ln=True)
    
    # Cabeçalho da Tabela
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 10, "Modelo", 1, 0, 'C', True)
    pdf.cell(40, 10, "MSE", 1, 0, 'C', True)
    pdf.cell(40, 10, "MAE", 1, 0, 'C', True)
    pdf.cell(40, 10, "Melhoria (%)", 1, 1, 'C', True)
    
    # Dados da Tabela
    pdf.set_font("Arial", "", 10)
    for nome, mets in metricas.items():
        pdf.cell(50, 10, nome, 1)
        pdf.cell(40, 10, f"{mets['MSE']:.4f}", 1, 0, 'C')
        pdf.cell(40, 10, f"{mets['MAE']:.4f}", 1, 0, 'C')
        sinal = "+" if mets['Melhoria'] > 0 else ""
        pdf.cell(40, 10, f"{sinal}{mets['Melhoria']:.2f}%", 1, 1, 'C')
        
    pdf.ln(10)
    
    # Inserir o Gráfico
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Visualização dos Resultados", ln=True)
    # Ajusta a imagem para caber na largura da página (A4 tem ~210mm)
    pdf.image(imagem_grafico, x=10, y=None, w=190)
    
    # Guardar o arquivo
    pdf.output(nome_arquivo_pdf)
    print(f"\nRelatório PDF gerado com sucesso: {nome_arquivo_pdf}")

# ==========================================
# 1 - Import dos dados enriquecidos
# ==========================================
# Certifique-se de que o caminho aponta para o arquivo correto criado no passo anterior
file_path = "../previsoes/previsoes_enriquecidas_SARIMA.csv"
dataframe = pd.read_csv(file_path)

# Mapeamento das colunas (Verifique se os nomes batem com o seu CSV)
# 0: Indice_Tempo, 1: Valor_real, 2: Previsao_SARIMA, 3: Residuo
# 4: RADIACAO, 5: TEMP_AR, 6: TEMP_MAX, 7: TEMP_MIN
COL_REAL = 1
COL_SARIMA = 2
COL_RESIDUO = 3
# As colunas de 4 em diante são as nossas novas "Features Exógenas"

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

scaler = MinMaxScaler()
treino_norm = scaler.fit_transform(conjunto_treino)
vali_norm = scaler.transform(conjunto_vali)
teste_norm = scaler.transform(conjunto_teste)

# ==========================================
# 3 - CRIAÇÃO DOS LAGS + FEATURES EXÓGENAS
# ==========================================
def criar_matrizes_hibridas_enriquecidas(dados_array, n_lags=24):
    X, y, sarimax_base, real_base = [], [], [], []
    
    for i in range(n_lags, len(dados_array)):
        # 1. Pegamos os lags do resíduo (autocorrelação do erro)
        lags_erro = dados_array[i-n_lags : i, COL_RESIDUO]
        
        # 2. Pegamos as features exógenas do momento ATUAL (t)
        # (Radiação e Temperaturas nas colunas 4, 5, 6, 7)
        features_exo = dados_array[i, 4:] 
        
        # Concatenamos: [erro_t-24, ..., erro_t-1, rad_t, temp_t...]
        input_total = np.concatenate([lags_erro, features_exo])
        
        X.append(input_total)
        y.append(dados_array[i, COL_RESIDUO])
        sarimax_base.append(dados_array[i, COL_SARIMA])
        real_base.append(dados_array[i, COL_REAL])
        
    return np.array(X), np.array(y), np.array(sarimax_base), np.array(real_base)

n_lags = 24
X_treino, y_treino, _, _ = criar_matrizes_hibridas_enriquecidas(treino_norm, n_lags)
X_teste, y_teste, sarimax_teste, real_teste = criar_matrizes_hibridas_enriquecidas(teste_norm, n_lags)

# ==========================================
# 4 - CONFIGURAÇÃO E TREINAMENTO
# ==========================================
# (O dicionário de modelos e o loop de treinamento permanecem quase iguais)
modelos = {
    'MLP (Rede Neural)': (
        MLPRegressor(random_state=42, early_stopping=True, max_iter=500),
        {'hidden_layer_sizes': [(32,), (32, 16)], 'activation': ['relu', 'tanh']}
    ),
    'Gradient Boosting': (
        GradientBoostingRegressor(random_state=42),
        {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.1]}
    ),
    'SVR': (
        SVR(),
        {'C': [1, 10], 'kernel': ['rbf']}
    )
}

# Função auxiliar para reverter escala (ajustada para o novo número de colunas)
def reverter_escala_dinamica(valores, indice, scaler, dataframe):
    total_cols = dataframe.shape[1]
    dummy = np.zeros((len(valores), total_cols))
    dummy[:, indice] = valores
    return scaler.inverse_transform(dummy)[:, indice]

real_teste_orig = reverter_escala_dinamica(real_teste, COL_REAL, scaler, dataframe)
sarimax_teste_orig = reverter_escala_dinamica(sarimax_teste, COL_SARIMA, scaler, dataframe)

mse_base = mean_squared_error(real_teste_orig, sarimax_teste_orig)
mae_base = mean_absolute_error(real_teste_orig, sarimax_teste_orig)

resultados_previsoes_orig = {}
metricas_modelos = {}

tscv = TimeSeriesSplit(n_splits=3)

for nome, (mod, params) in modelos.items():
    print(f"Otimizando {nome} com features enriquecidas...")
    grid = GridSearchCV(mod, params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_treino, y_treino)
    
    residuo_prev_norm = grid.best_estimator_.predict(X_teste)
    residuo_prev_orig = reverter_escala_dinamica(residuo_prev_norm, COL_RESIDUO, scaler, dataframe)
    
    previsao_hibrida = sarimax_teste_orig + residuo_prev_orig
    resultados_previsoes_orig[nome] = previsao_hibrida
    
    mse_h = mean_squared_error(real_teste_orig, previsao_hibrida)
    mae_h = mean_absolute_error(real_teste_orig, previsao_hibrida)
    melhoria = ((mse_base - mse_h) / mse_base) * 100
    
    metricas_modelos[nome] = {'MSE': mse_h, 'MAE': mae_h, 'Melhoria': melhoria}

# ==========================================
# 5 - RELATÓRIO E PDF (Aproveitando suas funções)
# ==========================================
# (Aqui você chama as funções plotar_comparacao_total e gerar_pdf_relatorio 
# que você já definiu no seu código original)

gerar_pdf_relatorio(
    metricas=metricas_modelos, 
    mse_base=mse_base, 
    mae_base=mae_base, 
    imagem_grafico='../plotagens/plotagem_comparativa_hibrido/grafico_comparativo_modelos.png'
)
print(f"{'SARIMA PURO (Base)':<25} | MSE: {mse_base:.2f} | MAE: {mae_base:.2f}")
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
    plt.plot(sarimax[:amostras], label='SARIMA Puro', color='gray', linestyle=':', linewidth=2, alpha=0.8)
    
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
    
    nome_arquivo = '../plotagens/plotagem_comparativa_hibrido/grafico_comparativo_modelos.png'
    plt.savefig(nome_arquivo, dpi=300)
    print(f"\nGráfico salvo com sucesso como '{nome_arquivo}'. O deslocamento Y foi corrigido!")


plotar_comparacao_total(real_teste_orig, sarimax_teste_orig, resultados_previsoes_orig)
# Agora passamos os dados revertidos para a função plotar
plotar_comparacao_total(real_teste_orig, sarimax_teste_orig, resultados_previsoes_orig)

# ... (seu código anterior até o plotar_comparacao_total)

# Chamar a função de plotar (que já salva o PNG)
plotar_comparacao_total(real_teste_orig, sarimax_teste_orig, resultados_previsoes_orig)

# Gerar o PDF agora
gerar_pdf_relatorio(
    metricas=metricas_modelos, 
    mse_base=mse_base, 
    mae_base=mae_base, 
    imagem_grafico='../plotagens/plotagem_comparativa_hibrido/grafico_comparativo_modelos.png'
)
