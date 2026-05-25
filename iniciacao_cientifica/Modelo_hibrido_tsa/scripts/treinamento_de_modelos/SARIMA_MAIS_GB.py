import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Importações atualizadas para fpdf2
from fpdf import FPDF
from fpdf.enums import XPos, YPos


# ==========================================
# FUNÇÕES DE APOIO
# ==========================================
def gerar_pdf_relatorio(
    metricas,
    mse_base,
    mae_base,
    imagem_grafico,
    nome_arquivo_pdf="../relatorios/Relatorio_Melhor_Gradient_Boosting.pdf",
):
    os.makedirs(os.path.dirname(nome_arquivo_pdf), exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    # Aviso corrigido: Uso da fonte 'Helvetica' nativa em vez de 'Arial'
    pdf.set_font("Helvetica", "B", 16)

    # Aviso corrigido: Uso de new_x e new_y em vez de ln=True
    pdf.cell(
        0,
        10,
        "Relatorio de Desempenho: Hibrido Gradient Boosting",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
    )
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        10,
        "1. Desempenho do Modelo Base (Baseline)",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Modelo: SARIMA Puro", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"MSE: {mse_base:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"MAE: {mae_base:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        10,
        "2. Resultados do Modelo Hibrido Otimizado",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    # Cabeçalho da Tabela
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("Helvetica", "B", 10)
    # Aviso corrigido: border=1, new_x=XPos.RIGHT, new_y=YPos.TOP substituem 1, 0
    pdf.cell(
        50,
        10,
        "Modelo",
        border=1,
        new_x=XPos.RIGHT,
        new_y=YPos.TOP,
        align="C",
        fill=True,
    )
    pdf.cell(
        40, 10, "MSE", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True
    )
    pdf.cell(
        40, 10, "MAE", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C", fill=True
    )
    pdf.cell(
        40,
        10,
        "Melhoria (%)",
        border=1,
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        align="C",
        fill=True,
    )

    # Dados da Tabela
    pdf.set_font("Helvetica", "", 10)
    for nome, mets in metricas.items():
        pdf.cell(50, 10, nome, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(
            40,
            10,
            f"{mets['MSE']:.4f}",
            border=1,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C",
        )
        pdf.cell(
            40,
            10,
            f"{mets['MAE']:.4f}",
            border=1,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C",
        )
        sinal = "+" if mets["Melhoria"] > 0 else ""
        pdf.cell(
            40,
            10,
            f"{sinal}{mets['Melhoria']:.2f}%",
            border=1,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="C",
        )

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(
        0,
        10,
        "Melhores Hiperparametros Encontrados:",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 10)
    for k, v in metricas["Gradient Boosting"]["Best_Params"].items():
        pdf.cell(0, 8, f"- {k}: {v}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)

    # Inserir o Gráfico
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0, 10, "3. Visualizacao dos Resultados", new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )
    pdf.image(imagem_grafico, x=10, y=None, w=190)

    pdf.output(nome_arquivo_pdf)
    print(f"\n📄 Relatório PDF gerado com sucesso: {nome_arquivo_pdf}")


def plotar_comparacao_total(real, sarimax, prevs_dict, amostras=150):
    plt.figure(figsize=(16, 8))

    plt.plot(
        real[:amostras],
        label="Valor Real (Target)",
        color="black",
        linewidth=2,
        marker=".",
    )
    plt.plot(
        sarimax[:amostras],
        label="SARIMA Puro",
        color="gray",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
    )

    nome = "Gradient Boosting"
    prev = prevs_dict[nome]
    plt.plot(
        prev[:amostras],
        label=f"Híbrido: {nome} (Otimizado)",
        color="green",
        linestyle="-.",
        alpha=0.9,
        linewidth=2,
    )

    plt.title(
        f"Comparação do Modelo Híbrido na Escala Real (Primeiras {amostras} horas)",
        fontsize=14,
    )
    plt.xlabel("Tempo (Horas)", fontsize=12)
    plt.ylabel("Valor Original (Desnormalizado)", fontsize=12)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    nome_arquivo = (
        "../plotagens/plotagem_comparativa_hibrido/grafico_comparativo_gb_otimizado.png"
    )
    os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
    plt.savefig(nome_arquivo, dpi=300)
    print(f"📈 Gráfico salvo com sucesso como '{nome_arquivo}'.")
    return nome_arquivo


# ==========================================
# 1 - Import dos dados enriquecidos
# ==========================================
file_path = "../previsoes/previsoes_enriquecidas_SARIMA.csv"
dataframe = pd.read_csv(file_path)

COL_REAL = 1
COL_SARIMA = 2
COL_RESIDUO = 3

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
        lags_erro = dados_array[i - n_lags : i, COL_RESIDUO]
        features_exo = dados_array[i, 4:]
        input_total = np.concatenate([lags_erro, features_exo])

        X.append(input_total)
        y.append(dados_array[i, COL_RESIDUO])
        sarimax_base.append(dados_array[i, COL_SARIMA])
        real_base.append(dados_array[i, COL_REAL])

    return np.array(X), np.array(y), np.array(sarimax_base), np.array(real_base)


n_lags = 24
treino_vali_norm = np.vstack((treino_norm, vali_norm))
X_treino_cv, y_treino_cv, _, _ = criar_matrizes_hibridas_enriquecidas(
    treino_vali_norm, n_lags
)
X_teste, y_teste, sarimax_teste, real_teste = criar_matrizes_hibridas_enriquecidas(
    teste_norm, n_lags
)

# ==========================================
# 4 - CONFIGURAÇÃO E TREINAMENTO
# ==========================================
modelo = {
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200, 300, 400, 500, 700, 1000],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.1, 0.01, 0.001],
        },
    ),
}


def reverter_escala_dinamica(valores, indice, scaler, dataframe):
    total_cols = dataframe.shape[1]
    dummy = np.zeros((len(valores), total_cols))
    dummy[:, indice] = valores
    return scaler.inverse_transform(dummy)[:, indice]


real_teste_orig = reverter_escala_dinamica(real_teste, COL_REAL, scaler, dataframe)
sarimax_teste_orig = reverter_escala_dinamica(
    sarimax_teste, COL_SARIMA, scaler, dataframe
)

mse_base = mean_squared_error(real_teste_orig, sarimax_teste_orig)
mae_base = mean_absolute_error(real_teste_orig, sarimax_teste_orig)

resultados_previsoes_orig = {}
metricas_modelos = {}

tscv = TimeSeriesSplit(n_splits=3)

print("\n" + "=" * 50)
print("🚀 INICIANDO TREINAMENTO DOS MODELOS")
print("=" * 50)
print(f"📊 Baseline SARIMA -> MSE: {mse_base:.4f} | MAE: {mae_base:.4f}\n")

for nome, (mod, params) in modelo.items():
    print(f"⏳ [{nome}] Iniciando otimização na grade de parâmetros...")

    grid = GridSearchCV(
        mod, params, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
    )
    grid.fit(X_treino_cv, y_treino_cv)

    print(f"✅ [{nome}] Treinamento finalizado!")
    print(f"🏆 Melhores parâmetros: {grid.best_params_}")

    residuo_prev_norm = grid.best_estimator_.predict(X_teste)
    residuo_prev_orig = reverter_escala_dinamica(
        residuo_prev_norm, COL_RESIDUO, scaler, dataframe
    )

    previsao_hibrida = sarimax_teste_orig + residuo_prev_orig
    resultados_previsoes_orig[nome] = previsao_hibrida

    mse_h = mean_squared_error(real_teste_orig, previsao_hibrida)
    mae_h = mean_absolute_error(real_teste_orig, previsao_hibrida)
    melhoria = ((mse_base - mse_h) / mse_base) * 100

    metricas_modelos[nome] = {
        "MSE": mse_h,
        "MAE": mae_h,
        "Melhoria": melhoria,
        "Best_Params": grid.best_params_,
    }

    # Log de resultados do modelo recém-treinado
    sinal = "+" if melhoria > 0 else ""
    print(
        f"📉 Resultados: MSE: {mse_h:.4f} | MAE: {mae_h:.4f} | Melhoria: {sinal}{melhoria:.2f}%\n"
    )

# ==========================================
# 5 - RELATÓRIO E PDF
# ==========================================
print("=" * 50)
print("🛠 GERANDO ARTEFATOS FINAIS")
print("=" * 50)

# Gerar gráfico e pegar o caminho
caminho_grafico = plotar_comparacao_total(
    real_teste_orig, sarimax_teste_orig, resultados_previsoes_orig
)

# Gerar PDF sem os deprecation warnings
gerar_pdf_relatorio(
    metricas=metricas_modelos,
    mse_base=mse_base,
    mae_base=mae_base,
    imagem_grafico=caminho_grafico,
)
