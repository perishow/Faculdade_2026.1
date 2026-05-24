import pandas as pd

# ==========================================
# 1. Carregar e limpar os dados do INMET
# ==========================================
caminho_inmet = '../datasets/2025/INMET_CO_DF_A001_BRASILIA_01-01-2025_A_30-11-2025.CSV'

# Use encoding='ISO-8859-1' se estiver lendo o arquivo original ou 'UTF-8' se for o convertido
df_inmet = pd.read_csv(
    caminho_inmet,
    sep=';',
    skiprows=8,
    decimal=',',
    na_values=[''],
    encoding='ISO-8859-1' 
)

# Remove a coluna vazia do final gerada pelo ';'
df_inmet = df_inmet.loc[:, ~df_inmet.columns.str.contains('^Unnamed')]

# ==========================================
# 2. Selecionar as variáveis correlacionadas
# ==========================================
# Transformar o índice em uma coluna explícita para podermos cruzar os dados
df_inmet = df_inmet.reset_index(names='Indice_Tempo')

features_selecionadas = [
    "Indice_Tempo",  # A chave primária para juntar os dados!
    "RADIACAO GLOBAL (Kj/m²)",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)"
]

df_features = df_inmet[features_selecionadas].copy()

# ==========================================
# 3. Carregar as previsões do SARIMAX
# ==========================================
caminho_previsoes = 'previsoes/previsoes_SARIMA_3.csv'
df_previsoes = pd.read_csv(caminho_previsoes)

# ==========================================
# 4. Fazer o Merge (Cruzamento dos Dados)
# ==========================================
# Une os dataframes usando a coluna 'Indice_Tempo'
# how='left' garante que manteremos apenas as linhas que existem nas previsões
df_enriquecido = pd.merge(df_previsoes, df_features, on='Indice_Tempo', how='left')

# (OPCIONAL/RECOMENDADO) Tratar valores nulos da Radiação Global na madrugada
# O pandas vai importar dados em branco como NaN. Podemos preencher buracos pequenos com interpolação
# e os restantes com 0 (já que não há radiação à noite)
df_enriquecido['RADIACAO GLOBAL (Kj/m²)'] = df_enriquecido['RADIACAO GLOBAL (Kj/m²)'].fillna(0)

# Interpola buracos nas temperaturas caso existam falhas nos sensores do INMET
for col in features_selecionadas[2:]:
    df_enriquecido[col] = df_enriquecido[col].interpolate(method='linear')

# ==========================================
# 5. Salvar o novo CSV
# ==========================================
# 1. Remove qualquer vírgula dos nomes das colunas para evitar conflitos
df_enriquecido.columns = df_enriquecido.columns.str.replace(',', '', regex=False)

# 2. Salvar o arquivo no padrão internacional (ideal para ML)
nome_arquivo_saida = 'previsoes/previsoes_enriquecidas_SARIMA.csv'
df_enriquecido.to_csv(nome_arquivo_saida, index=False)

print(f"Sucesso! Arquivo '{nome_arquivo_saida}' criado.")
print("\nPrimeiras linhas do novo arquivo:")
print(df_enriquecido.head())
