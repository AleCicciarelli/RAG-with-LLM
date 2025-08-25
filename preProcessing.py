import pandas as pd
import os
import json

def preprocess_csv_with_pandas(csv_path, table_name):
    df = pd.read_csv(csv_path)
    documents = []

    for idx, row in df.iterrows():
        # Costruzione di una frase semantica
        values_str = ", ".join(f"{col} = {row[col]}" for col in df.columns)
        text = f"Table {table_name}, Row {idx}: {values_str}."

        # Documento da indicizzare + metadati
        doc = {
            "id": f"{table_name}_{idx}",
            "text": text,
            "metadata": {
                "table": table_name,
                "row_id": idx,
                "columns": df.columns.tolist(),
                "raw": row.to_dict()
            }
        }
        documents.append(doc)

    return documents

# Esempio: processa tutti i CSV nella cartella
def process_all_csvs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            table_name = filename.replace(".csv", "")
            path = os.path.join(folder_path, filename)
            docs = preprocess_csv_with_pandas(path, table_name)
            all_docs.extend(docs)
    return all_docs

# Salvataggio
docs = process_all_csvs("csv_data")
with open("semantic_documents.json", "w") as f:
    json.dump(docs, f, indent=2)
