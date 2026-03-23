from sklearn.metrics import mean_squared_error
import pandas as pd

file_path_2 = "./resultados_previsoes.csv"
file_path_3 = "./resultados_previsoes_1.csv"

previsoes_1 = pd.read_csv(file_path_2)
previsoes_2 = pd.read_csv(file_path_3)

mse1 = mean_squared_error(previsoes_1["Valor_real"], previsoes_1["Previsao_ARIMA"])

mse2 = mean_squared_error(previsoes_2["Valor_real"], previsoes_2["Previsao_ARIMA"])

print(f"mse1: {mse1}\nmse2: {mse2}")
