import pandas as pd

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


if len(dataset) == len(dataset_residuo):
    print("tudo certo\n")
else:
    print("fudeu\n")

# --- CORREÇÃO: Limpando a coluna de Hora UTC ---
# Substitui ' UTC' por nada e converte a string resultante ('0000', '1200') para int (0, 1200)
dataset["Hora UTC"] = dataset["Hora UTC"].str.replace(" UTC", "").astype(int)

correlacoes = {}

for feature in dataset.columns:
    if feature not in ["Data", "HORÁRIO (mm)"]:
        # Correção feita aqui: Adicionado o ")" no final da string da coluna
        pearson = dataset[feature].corr(dataset_residuo["Residuo"])  # type: ignore

        correlacoes[feature] = pearson


correlacoes_ordenadas = sorted(
    correlacoes.items(), key=lambda x: abs(x[1]), reverse=True
)
for feature, pearson in correlacoes_ordenadas:
    print(f"{pearson:.4} : {feature}")

"""
    Ao final desse estudo, foi revelado as 3 features mais relevantes:
    -0.1114 : PRECIPITAÇÃO TOTAL, HORÁRIO (mm)
    0.1012 : TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)
    0.09942 : VENTO, VELOCIDADE HORARIA (m/s)

    Apesar de não muito correlacionado, utilizaremos esses 3 para o treinamento da rede neural.
"""
