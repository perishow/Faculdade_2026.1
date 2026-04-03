import pandas as pd

# Esse script checa a relação entre índice do csv de previsão e os horários on csv original

# 1. Carregando os datasets
previsoes = pd.read_csv("./previsoes/previsoes_SARIMA_3.csv")
file_path = "../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV"

dataset_original = pd.read_csv(
    file_path, encoding="latin1", sep=";", skiprows=8, decimal=","
)

# 2. Pegando o índice DIRETAMENTE (sem subtrair nada)
indice_inicial = previsoes["Indice_Tempo"].iloc[0]
indice_final = previsoes["Indice_Tempo"].iloc[-1]

# 3. Buscando a Data e Hora
data_inicial = dataset_original.loc[indice_inicial, "Data"]
hora_inicial = dataset_original.loc[indice_inicial, "Hora UTC"]

data_final = dataset_original.loc[indice_final, "Data"]
hora_final = dataset_original.loc[indice_final, "Hora UTC"]

# 4. Exibindo os resultados
print(
    f"Previsão Inicia em: Índice {indice_inicial} -> {data_inicial} às {hora_inicial}"
)
print(f"Previsão Termina em: Índice {indice_final} -> {data_final} às {hora_final}")
