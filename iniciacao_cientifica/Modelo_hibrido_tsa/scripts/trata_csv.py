# Esse arquivo pega um CSV não muito bem formatado do INMET e formata para ficar melhor de usar.
# Os arquivos tratados são armazenados no diretório "/datasets_tratados"

import sys
import pandas as pd


def tratar_csv(file_path: str, export_name: str):
    # 1. Usando decimal="," o Pandas já converte as vírgulas para ponto automaticamente!
    dataset = pd.read_csv(
        file_path, encoding="latin1", sep=";", skiprows=8, decimal=","
    )
    dataset = dataset[["Data", "Hora UTC", "RADIACAO GLOBAL (Kj/m²)"]]

    # 2. Preenchendo os NaNs
    dataset = dataset.fillna({"RADIACAO GLOBAL (Kj/m²)": 0.0})

    # 3. Juntando Data e Hora (com o # type: ignore para o Pyright ignorar o .str aqui)
    dataset["DataHora"] = pd.to_datetime(
        dataset["Data"] + " " + dataset["Hora UTC"].astype(str).str.replace(" UTC", "")  # type: ignore
    )

    # 4. Definindo como índice e limpando a tabela
    dataset = dataset.set_index("DataHora")
    dataset = dataset.drop(columns=["Data", "Hora UTC"])

    # 5. Exportação do dataset tratado:
    caminho_exportacao = f"../datasets_tratados/{export_name}.CSV"
    dataset.to_csv(caminho_exportacao)
    print(f"Dataset exportado em: {caminho_exportacao}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Erro! Uso correto: python3 trata_csv.py <caminho_do_arquivo> <nome de exportacao>"
        )
        sys.exit(1)

    caminho_arquivo = sys.argv[1]
    nome_exportacao = sys.argv[2]

    tratar_csv(caminho_arquivo, nome_exportacao)
